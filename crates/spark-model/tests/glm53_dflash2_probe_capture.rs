// SPDX-License-Identifier: AGPL-3.0-only
//! RED: actual owned readback used by the diagnostic observer, no CUDA calls.

#[path = "../src/model/glm53/dflash2_probe_capture.rs"]
#[allow(dead_code)]
mod capture;
#[path = "../src/model/glm53/dflash2_probe_contract.rs"]
#[allow(dead_code)]
mod dflash2_probe_contract;
#[path = "../src/model/glm53/owned_verify_readback.rs"]
#[allow(dead_code)]
mod owned_verify_readback;
use anyhow::{Result, bail};
use capture::{ProbeCapture, ProbeIo};
use dflash2_probe_contract::{ProbeLayout, ProbeStage};

fn layout() -> ProbeLayout {
    ProbeLayout::new(2, 32, 16, 16, 32, 4, 99).unwrap()
}
#[derive(Default)]
struct Deferred {
    pointer: Option<(*mut u8, usize)>,
    events: Vec<(u64, u64)>,
    fail_copy: bool,
    fail_fence: bool,
    panic_copy: bool,
    payload: Vec<u8>,
}
impl ProbeIo for Deferred {
    fn copy(&mut self, source: u64, dst: &mut [u8], stream: u64) -> Result<()> {
        self.events.push((source, stream));
        self.pointer = Some((dst.as_mut_ptr(), dst.len()));
        if self.panic_copy {
            panic!("copy submitted before panic");
        }
        if self.fail_copy {
            bail!("copy submitted before error");
        }
        Ok(())
    }
    fn drain(&mut self, stream: u64) -> Result<()> {
        self.events.push((0, stream));
        if self.fail_fence {
            bail!("injected completion failure");
        }
        if let Some((pointer, bytes)) = self.pointer.take() {
            let payload = if self.payload.is_empty() {
                vec![0; bytes]
            } else {
                assert_eq!(self.payload.len(), bytes);
                self.payload.clone()
            };
            // The production capture owns this allocation through completion.
            unsafe {
                std::ptr::copy_nonoverlapping(payload.as_ptr(), pointer, bytes);
            }
        }
        Ok(())
    }
}

#[test]
fn exact_order_extents_and_stream_are_checked_before_any_copy() {
    let mut capture = ProbeCapture::new(layout(), 17, 0).unwrap();
    let mut io = Deferred::default();
    assert!(
        capture
            .observe(ProbeStage::ValueCache(0), 4096, 32, 0, &mut io)
            .is_err()
    );
    assert!(
        capture
            .observe(ProbeStage::KeyCache(0), 4096, 30, 0, &mut io)
            .is_err()
    );
    assert!(
        capture
            .observe(ProbeStage::KeyCache(0), 4096, 32, 1, &mut io)
            .is_err()
    );
    assert!(
        capture
            .observe(ProbeStage::KeyCache(0), 0, 32, 0, &mut io)
            .is_err()
    );
    assert!(
        capture
            .observe(ProbeStage::KeyCache(0), u64::MAX - 15, 32, 0, &mut io)
            .is_err()
    );
    assert!(io.events.is_empty());
    assert!(capture.frames().is_err());
    let expected = layout();
    for &stage in expected.stages() {
        capture
            .observe(stage, 4096, expected.bytes(stage).unwrap(), 0, &mut io)
            .unwrap();
    }
    let frames = capture.frames().unwrap();
    assert_eq!(frames.len(), 9);
    for (frame, stage) in frames.iter().zip(expected.stages()) {
        assert_eq!(frame.stage, *stage);
        assert_eq!(frame.bytes, vec![0; expected.bytes(*stage).unwrap()]);
    }
    assert_eq!(io.events.len(), 18);
    assert!(!capture.pending());
    assert!(
        capture
            .observe(ProbeStage::DraftIds, 4096, 4, 0, &mut io)
            .is_err()
    );
}

#[test]
fn failed_copy_is_terminal_even_if_completion_succeeds() {
    let mut capture = ProbeCapture::new(layout(), 17, 4).unwrap();
    let mut io = Deferred {
        fail_copy: true,
        ..Deferred::default()
    };
    let error = capture
        .observe(ProbeStage::KeyCache(0), 4096, 32, 4, &mut io)
        .unwrap_err();
    assert!(format!("{error:#}").contains("copy submitted"));
    assert!(!capture.pending());
    assert!(capture.frames().is_err());
    io.fail_copy = false;
    assert!(
        capture
            .observe(ProbeStage::KeyCache(0), 4096, 32, 4, &mut io)
            .is_err()
    );
    assert_eq!(io.events, [(4096, 4), (0, 4)]);
}

#[test]
fn failed_fence_and_caught_panic_retain_host_storage_until_exact_stream_drain() {
    for panic_copy in [false, true] {
        let mut capture = ProbeCapture::new(layout(), 17, 0).unwrap();
        let mut io = Deferred {
            fail_fence: true,
            panic_copy,
            ..Deferred::default()
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            capture.observe(ProbeStage::KeyCache(0), 4096, 32, 0, &mut io)
        }));
        assert!(result.is_err() || result.unwrap().is_err());
        assert!(capture.pending());
        assert!(capture.drain(&mut io).is_err());
        io.fail_fence = false;
        capture.drain(&mut io).unwrap();
        assert!(!capture.pending());
        assert!(io.pointer.is_none());
        assert!(io.events.iter().all(|(_, stream)| *stream == 0));
        assert!(capture.frames().is_err());
    }
}

#[test]
fn nonfinite_bf16_and_out_of_vocabulary_ids_never_publish_a_complete_receipt() {
    let mut capture = ProbeCapture::new(layout(), 17, 0).unwrap();
    let mut io = Deferred {
        payload: [0x7f80u16.to_le_bytes(); 16].concat(),
        ..Deferred::default()
    };
    assert!(
        capture
            .observe(ProbeStage::KeyCache(0), 4096, 32, 0, &mut io)
            .is_err()
    );
    assert!(!capture.pending());
    assert!(capture.frames().is_err());
    let expected = layout();
    let mut capture = ProbeCapture::new(layout(), 17, 0).unwrap();
    let mut io = Deferred::default();
    for &stage in &expected.stages()[..8] {
        capture
            .observe(stage, 4096, expected.bytes(stage).unwrap(), 0, &mut io)
            .unwrap();
    }
    io.payload = 99u32.to_le_bytes().to_vec();
    assert!(
        capture
            .observe(ProbeStage::DraftIds, 4096, 4, 0, &mut io)
            .is_err()
    );
    assert!(capture.frames().is_err());
}
