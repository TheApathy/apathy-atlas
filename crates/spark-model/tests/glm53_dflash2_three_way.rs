// SPDX-License-Identifier: AGPL-3.0-only
//! The actual diagnostic report must not relabel new arithmetic as old parity.
#[path = "../examples/glm53_dflash2_kv_parity/artifacts.rs"]
#[allow(dead_code)]
mod artifacts;
use anyhow::Result;
use spark_model::model::glm53::{ProbeCapture, ProbeIo, ProbeLayout, ProbeStage};

struct Payload(Vec<u8>);
impl ProbeIo for Payload {
    fn copy(&mut self, _: u64, dst: &mut [u8], _: u64) -> Result<()> {
        dst.copy_from_slice(&self.0);
        Ok(())
    }
    fn drain(&mut self, _: u64) -> Result<()> {
        Ok(())
    }
}
fn layout() -> ProbeLayout {
    ProbeLayout::new(1, 10 * 2048, 2, 2, 4, 4, 100).unwrap()
}
fn capture(changed: Option<ProbeStage>) -> ProbeCapture {
    let layout = layout();
    let mut capture = ProbeCapture::new(layout.clone(), 2, 0).unwrap();
    for &stage in layout.stages() {
        let size = layout.bytes(stage).unwrap();
        let mut payload = vec![0; size];
        if Some(stage) == changed {
            payload[0] = 1;
        }
        capture
            .observe(stage, 4096, size, 0, &mut Payload(payload))
            .unwrap();
    }
    capture
}

#[test]
fn stable_cache_exactness_does_not_erase_original_arithmetic_drift() {
    let original = capture(Some(ProbeStage::KeyCache(0)));
    let stable = capture(None);
    let cached = capture(None);
    let report = artifacts::compare_three(&layout(), &original, &stable, &cached).unwrap();
    assert_eq!(report["stable_cache_exact"], true);
    assert_eq!(report["original_baseline_exact"], false);
    assert_eq!(report["original_vs_stable_full"]["exact"], false);
    assert_eq!(report["original_vs_stable_cached"]["exact"], false);
    assert_eq!(report["stable_full_vs_cached"]["exact"], true);
    assert_eq!(report["quality_qualified"], false);
    assert_eq!(report["speed_qualified"], false);
}

#[test]
fn every_stage_including_ids_is_a_strict_cache_gate() {
    for &stage in layout().stages() {
        let report = artifacts::compare_three(
            &layout(),
            &capture(None),
            &capture(None),
            &capture(Some(stage)),
        )
        .unwrap();
        assert_eq!(report["stable_cache_exact"], false, "{stage:?}");
        assert_eq!(report["original_baseline_exact"], false);
    }
}

#[test]
fn incomplete_or_wrong_context_cannot_be_an_exact_receipt() {
    let incomplete = ProbeCapture::new(layout(), 2, 0).unwrap();
    assert!(
        artifacts::compare_three(&layout(), &capture(None), &incomplete, &capture(None)).is_err()
    );
    let wrong_context = ProbeCapture::new(layout(), 1, 0).unwrap();
    assert!(
        artifacts::compare_three(&layout(), &capture(None), &capture(None), &wrong_context)
            .is_err()
    );
}

#[test]
fn all_exact_still_is_not_a_quality_or_performance_certificate() {
    let report =
        artifacts::compare_three(&layout(), &capture(None), &capture(None), &capture(None))
            .unwrap();
    assert_eq!(report["stable_cache_exact"], true);
    assert_eq!(report["original_baseline_exact"], true);
    assert_eq!(report["quality_qualified"], false);
    assert_eq!(report["speed_qualified"], false);
}
