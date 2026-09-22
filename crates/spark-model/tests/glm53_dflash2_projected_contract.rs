// SPDX-License-Identifier: AGPL-3.0-only
//! P11 RED: optional input frames derive from actual admitted context geometry.
#[path = "../src/model/glm53/dflash2_probe_capture.rs"]
#[allow(dead_code)]
mod capture;
#[path = "../src/model/glm53/dflash2_probe_contract.rs"]
#[allow(dead_code)]
mod dflash2_probe_contract;
#[path = "../src/model/glm53/owned_verify_readback.rs"]
#[allow(dead_code)]
mod owned_verify_readback;
use capture::ProbeCapture;
use dflash2_probe_contract::{MAX_PROBE_BYTES, ProbeLayout, ProbeStage};

fn actual() -> ProbeLayout {
    ProbeLayout::new(
        5,
        129 * 16 * 8 * 128 * 2,
        8 * 4096 * 2,
        7 * 4096 * 2,
        7 * 154_880 * 2,
        7 * 4,
        154_880,
    )
    .unwrap()
}

#[test]
fn opt_in_wraps_unchanged_eighteen_stages_with_exact_context_one_two_bytes() {
    let base = actual();
    assert_eq!(base.stages().len(), 18);
    assert_eq!(base.total_bytes(), 44_824_092);
    assert!(base.bytes(ProbeStage::ProjectedTargetBefore).is_err());
    assert!(
        !ProbeCapture::new(base.clone(), 1, 0)
            .unwrap()
            .wants_projected_target()
    );
    for context in [1, 2] {
        let expanded = base.clone().with_projected_target(context, 4096).unwrap();
        assert_eq!(expanded.stages().len(), 20);
        assert_eq!(expanded.stages()[0], ProbeStage::ProjectedTargetBefore);
        assert_eq!(&expanded.stages()[1..19], base.stages());
        assert_eq!(expanded.stages()[19], ProbeStage::ProjectedTargetAfter);
        let bytes = context as usize * 4096 * 2;
        assert_eq!(
            expanded.bytes(ProbeStage::ProjectedTargetBefore).unwrap(),
            bytes
        );
        assert_eq!(
            expanded.bytes(ProbeStage::ProjectedTargetAfter).unwrap(),
            bytes
        );
        assert_eq!(expanded.total_bytes(), base.total_bytes() + bytes * 2);
        assert_eq!(expanded.max_stage_bytes(), base.max_stage_bytes());
        assert_eq!(
            ProbeStage::ProjectedTargetBefore.name(),
            "projected-target-before"
        );
        assert_eq!(
            ProbeStage::ProjectedTargetAfter.name(),
            "projected-target-after"
        );
        assert!(
            ProbeCapture::new(expanded.clone(), context, 7)
                .unwrap()
                .wants_projected_target()
        );
        assert!(ProbeCapture::new(expanded, context + 1, 7).is_err());
    }
}

#[test]
fn invalid_derived_extent_duplicate_opt_in_and_total_cap_fail_before_io() {
    for (context, hidden) in [
        (0, 4096),
        (1, 0),
        (1, usize::MAX),
        (u32::MAX, usize::MAX / 2),
    ] {
        assert!(actual().with_projected_target(context, hidden).is_err());
    }
    // Capturing all 2047 projected rows twice would exceed the existing cap.
    assert!(actual().with_projected_target(2047, 4096).is_err());
    assert!(
        actual()
            .with_projected_target(1, MAX_PROBE_BYTES / 2)
            .is_err()
    );
    assert!(
        actual()
            .with_projected_target(1, 4096)
            .unwrap()
            .with_projected_target(1, 4096)
            .is_err()
    );
}

#[test]
fn observer_admission_rejects_plain_or_different_projected_input_geometry() {
    let extended = actual().with_projected_target(2, 4096).unwrap();
    let capture = ProbeCapture::new(extended.clone(), 2, 9).unwrap();
    capture.admit(&extended, 2, 9).unwrap();
    assert!(capture.admit(&actual(), 2, 9).is_err());
    assert!(
        capture
            .admit(&actual().with_projected_target(2, 2048).unwrap(), 2, 9)
            .is_err()
    );
    assert!(capture.admit(&extended, 1, 9).is_err());
    assert!(capture.admit(&extended, 2, 0).is_err());
}
