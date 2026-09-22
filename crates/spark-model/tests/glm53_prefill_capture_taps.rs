// SPDX-License-Identifier: AGPL-3.0-only
//! Production tap-enqueue bookkeeping against real destination bytes, not CUDA emulation.
// The plan/ingest sources are compiled in via #[path]; this test exercises only
// the Copies8 tap path, so the Gather128 half of the plan is unused here.
#![allow(dead_code)]
mod model {
    pub mod glm53 {
        pub use spark_model::model::glm53::GLM53_CAPTURE_LAYERS;
    }
}
mod layers {
    pub use spark_model::layers::Glm53TargetGeometry;
}
#[path = "../src/model/glm53/prefill_capture_ingest.rs"]
mod prefill_capture_ingest;
#[path = "../src/model/glm53/prefill_capture_plan.rs"]
mod prefill_capture_plan;

use anyhow::{Result, bail};
use prefill_capture_ingest::{CaptureReceipt, PrefillCaptureIo};
use prefill_capture_plan::{
    CaptureTransferMode, CopyStep, DeviceSpan, GatherStep, PrefillCapturePlan, ProjectionStep,
};

fn receipt() -> CaptureReceipt {
    let plan = PrefillCapturePlan::new(32, 17, 0, 0, 2047, 8192).unwrap();
    let bank = DeviceSpan {
        address: 0x1000_0000,
        bytes: plan.bank_bytes(),
    };
    let staging = DeviceSpan {
        address: 0x2000_0000,
        bytes: 8 * 5 * 8192,
    };
    let projected = DeviceSpan {
        address: 0x3000_0000,
        bytes: 2047 * 8192,
    };
    // Copies8 is the default (legacy per-row tap) transfer mode.
    CaptureReceipt::new(
        plan.bind(bank, staging, projected, CaptureTransferMode::Copies8)
            .unwrap(),
    )
}

struct CountIo(usize);
impl PrefillCaptureIo for CountIo {
    fn copy(&mut self, _: CopyStep, _: u64) -> Result<()> {
        self.0 += 1;
        Ok(())
    }
    fn gather(&mut self, _: GatherStep, _: u64) -> Result<()> {
        self.0 += 1;
        Ok(())
    }
    fn project_norm(&mut self, _: ProjectionStep, _: u64) -> Result<()> {
        self.0 += 1;
        Ok(())
    }
    fn synchronize(&mut self, _: u64) -> Result<()> {
        self.0 += 1;
        Ok(())
    }
}

#[test]
fn staged_header_requires_exact_rows_position_and_a_fresh_receipt() {
    let mut receipt = receipt();
    receipt.validate_staged(17, 0).unwrap();
    assert!(receipt.validate_staged(16, 0).is_err());
    assert!(receipt.validate_staged(17, 1).is_err());
    receipt
        .enqueue_tap(model::glm53::GLM53_CAPTURE_LAYERS[0], 0, |_| Ok(()))
        .unwrap();
    assert!(receipt.validate_staged(17, 0).is_err());
}

#[test]
fn all_five_real_destination_writes_precede_successful_publication() {
    let mut receipt = receipt();
    let mut bytes = vec![0xa5; 32 * 5 * 8192];
    for (slot, layer) in model::glm53::GLM53_CAPTURE_LAYERS.into_iter().enumerate() {
        receipt
            .enqueue_tap(layer, slot, |destination| {
                let offset = usize::try_from(destination.address - 0x1000_0000).unwrap();
                assert_eq!(offset, slot * 32 * 8192);
                assert_eq!(destination.bytes, 17 * 8192);
                bytes[offset..offset + destination.bytes].fill(slot as u8);
                Ok(())
            })
            .unwrap();
        assert_eq!(receipt.published_position(), None);
    }
    for slot in 0..5 {
        let base = slot * 32 * 8192;
        assert!(
            bytes[base..base + 17 * 8192]
                .iter()
                .all(|byte| *byte == slot as u8)
        );
        assert!(
            bytes[base + 17 * 8192..base + 32 * 8192]
                .iter()
                .all(|byte| *byte == 0xa5)
        );
    }
    assert_eq!(receipt.ingest(&mut CountIo(0), 17, 0, 7).unwrap(), 17);
}

#[test]
fn wrong_or_duplicate_taps_never_invoke_the_enqueue_closure() {
    let mut receipt = receipt();
    let taps = model::glm53::GLM53_CAPTURE_LAYERS;
    for (layer, slot) in [(taps[1], 1), (taps[0] + 1, 0), (taps[0], 5)] {
        assert!(
            receipt
                .enqueue_tap(layer, slot, |_| panic!("invalid tap executed"))
                .is_err()
        );
    }
    receipt.enqueue_tap(taps[0], 0, |_| Ok(())).unwrap();
    assert!(
        receipt
            .enqueue_tap(taps[0], 0, |_| panic!("duplicate tap executed"))
            .is_err()
    );
}

#[test]
fn partial_enqueue_failure_is_terminal_without_fabricating_a_completed_tap() {
    for failed_slot in 0..5 {
        let mut receipt = receipt();
        let taps = model::glm53::GLM53_CAPTURE_LAYERS;
        for (slot, &layer) in taps.iter().enumerate().take(failed_slot) {
            receipt.enqueue_tap(layer, slot, |_| Ok(())).unwrap();
        }
        let mut partial = [0u8; 2];
        assert!(
            receipt
                .enqueue_tap(taps[failed_slot], failed_slot, |_| {
                    partial.copy_from_slice(&[0x80, 0x3f]);
                    bail!("injected enqueue failure after partial write")
                })
                .is_err()
        );
        assert_eq!(partial, [0x80, 0x3f]);
        assert!(receipt.failed());
        assert!(
            receipt
                .enqueue_tap(taps[failed_slot], failed_slot, |_| panic!("retry executed"))
                .is_err()
        );
        let mut io = CountIo(0);
        assert!(receipt.ingest(&mut io, 17, 0, 7).is_err());
        assert_eq!(io.0, 0);
        assert_eq!(receipt.published_position(), None);
    }
}
