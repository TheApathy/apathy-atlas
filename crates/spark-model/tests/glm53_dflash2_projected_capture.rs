// SPDX-License-Identifier: AGPL-3.0-only
//! P11 RED: new frames use the actual retained-buffer readback implementation.
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
    ProbeLayout::new(1, 32, 16, 16, 32, 4, 99)
        .unwrap()
        .with_projected_target(2, 4)
        .unwrap()
}
#[derive(Default)]
struct Deferred {
    pointer: Option<(*mut u8, usize)>,
    events: Vec<(u64, u64)>,
    fail_fence: bool,
    panic_copy: bool,
    payload: Vec<u8>,
}
impl ProbeIo for Deferred {
    fn copy(&mut self, source: u64, dst: &mut [u8], stream: u64) -> Result<()> {
        self.events.push((source, stream));
        self.pointer = Some((dst.as_mut_ptr(), dst.len()));
        if self.panic_copy {
            panic!("submitted before panic");
        }
        Ok(())
    }
    fn drain(&mut self, stream: u64) -> Result<()> {
        self.events.push((0, stream));
        if self.fail_fence {
            bail!("completion failed");
        }
        if let Some((pointer, bytes)) = self.pointer.take() {
            assert_eq!(bytes, self.payload.len());
            // Written only after completion into the production-owned Vec.
            unsafe {
                std::ptr::copy_nonoverlapping(self.payload.as_ptr(), pointer, bytes);
            }
        }
        Ok(())
    }
}

#[test]
fn before_after_are_ordered_immutable_owned_frames_not_reused_readback_views() {
    let expected = layout();
    let mut capture = ProbeCapture::new(expected.clone(), 2, 17).unwrap();
    let mut io = Deferred::default();
    assert!(
        capture
            .observe(ProbeStage::KeyCache(0), 4096, 32, 17, &mut io)
            .is_err()
    );
    assert!(io.events.is_empty());
    let before = [0x3f80u16.to_le_bytes(); 8].concat();
    let after = [0x4000u16.to_le_bytes(); 8].concat();
    for &stage in expected.stages() {
        io.payload = match stage {
            ProbeStage::ProjectedTargetBefore => before.clone(),
            ProbeStage::ProjectedTargetAfter => after.clone(),
            _ => vec![0; expected.bytes(stage).unwrap()],
        };
        capture
            .observe(stage, 4096, io.payload.len(), 17, &mut io)
            .unwrap();
    }
    let frames = capture.frames().unwrap();
    assert_eq!(frames.first().unwrap().bytes, before);
    assert_eq!(frames.last().unwrap().bytes, after);
    assert_ne!(frames.first().unwrap().bytes, frames.last().unwrap().bytes);
    assert_eq!(io.events.len(), expected.stages().len() * 2);
    assert!(io.events.iter().all(|(_, stream)| *stream == 17));
}

#[test]
fn projected_copy_failure_and_panic_keep_pending_allocation_until_matching_fence() {
    for panic_copy in [false, true] {
        let mut capture = ProbeCapture::new(layout(), 2, 31).unwrap();
        let mut io = Deferred {
            fail_fence: true,
            panic_copy,
            payload: vec![0; 16],
            ..Deferred::default()
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            capture.observe(ProbeStage::ProjectedTargetBefore, 4096, 16, 31, &mut io)
        }));
        assert!(result.is_err() || result.unwrap().is_err());
        assert!(capture.pending());
        assert!(capture.retained_frames().is_empty());
        assert!(capture.frames().is_err());
        assert!(capture.drain(&mut io).is_err());
        io.fail_fence = false;
        capture.drain(&mut io).unwrap();
        assert!(!capture.pending());
        assert!(io.pointer.is_none());
        assert!(io.events.iter().all(|(_, stream)| *stream == 31));
        assert!(
            capture
                .observe(ProbeStage::ProjectedTargetBefore, 4096, 16, 31, &mut io)
                .is_err()
        );
    }
}

#[test]
fn nonfinite_projected_input_is_retained_but_never_published_complete() {
    let mut capture = ProbeCapture::new(layout(), 2, 0).unwrap();
    let mut io = Deferred {
        payload: [0x7f80u16.to_le_bytes(); 8].concat(),
        ..Deferred::default()
    };
    assert!(
        capture
            .observe(ProbeStage::ProjectedTargetBefore, 4096, 16, 0, &mut io)
            .is_err()
    );
    assert_eq!(capture.retained_frames()[0].bytes, io.payload);
    assert!(!capture.pending());
    assert!(capture.frames().is_err());
}
