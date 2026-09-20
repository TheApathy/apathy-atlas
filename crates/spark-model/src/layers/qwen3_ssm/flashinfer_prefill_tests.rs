// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::layers::ops::nvfp4_dynamic_scale::Nvfp4DynamicScalePlan;

#[test]
fn route_is_default_off_exact_and_fail_closed() {
    assert_eq!(
        ssm_flashinfer_prefill_route(false, true, true, true),
        SsmFlashinferPrefillRoute::Disabled
    );
    for (geometry, nvfp4) in [(false, true), (true, false), (false, false)] {
        assert_eq!(
            ssm_flashinfer_prefill_route(true, geometry, nvfp4, true),
            SsmFlashinferPrefillRoute::Ineligible
        );
    }
    assert_eq!(
        ssm_flashinfer_prefill_route(true, true, true, false),
        SsmFlashinferPrefillRoute::Missing
    );
    assert_eq!(
        ssm_flashinfer_prefill_route(true, true, true, true),
        SsmFlashinferPrefillRoute::Complete
    );
}

#[test]
fn ssm_route_is_bound_to_its_independent_selector() {
    let init = include_str!("init.rs");
    let helper = include_str!("trait_prefill_helper.rs");
    let loader = include_str!("../../weight_loader/qwen35_dense.rs");

    assert!(init.contains("prefill_ssm_flashinfer_enabled"));
    assert!(helper.contains("prefill_ssm_flashinfer_enabled"));
    assert!(!init.contains("prefill_proj_flashinfer_enabled"));
    assert!(!helper.contains("prefill_proj_flashinfer_enabled"));
    assert!(loader.contains("if flashinfer_attn_proj_requested"));
    assert!(loader.contains("if flashinfer_ssm_proj_requested"));
    assert!(!loader.contains("if flashinfer_proj_requested"));
}

#[test]
fn only_current_qwen38_ssm_geometry_and_frozen_tactics_are_admitted() {
    for rows in [2_079, 8_192] {
        assert!(qwen38_ssm_flashinfer_geometry(rows, 5_120, 16_384, 6_144));
    }
    for stale in [
        (2_048, 5_120, 16_384, 6_144),
        (2_079, 2_048, 12_288, 4_096),
        (2_079, 5_120, 12_288, 6_144),
        (2_079, 5_120, 16_384, 4_096),
        (8_193, 5_120, 16_384, 6_144),
    ] {
        assert!(!qwen38_ssm_flashinfer_geometry(
            stale.0, stale.1, stale.2, stale.3
        ));
    }
    assert_eq!(qwen38_ssm_flashinfer_tactics(2_079), Some((4, 2)));
    assert_eq!(qwen38_ssm_flashinfer_tactics(8_192), Some((2, 2)));
    assert_eq!(qwen38_ssm_flashinfer_tactics(2_048), None);
}

#[test]
fn shared_arena_layout_fits_both_dynamic_projection_widths() {
    let cases = [
        (2_079, 5_120, 6_018_576),
        (2_079, 6_144, 7_222_288),
        (8_192, 5_120, 23_592_976),
        (8_192, 6_144, 28_311_568),
    ];
    for (rows, cols, expected) in cases {
        let plan = Nvfp4DynamicScalePlan::qwen38_ssm_projection(rows, cols).unwrap();
        let capacity = rows as usize * QWEN38_FLASHINFER_SSM_QKVZ * 4;
        let layout = ssm_flashinfer_scratch_layout(plan, capacity).unwrap();
        assert_eq!(layout.packed_offset, 0);
        assert!(layout.scales_offset.is_multiple_of(16));
        assert!(layout.global_max_offset.is_multiple_of(16));
        assert_eq!(layout.total_bytes, expected);
        assert!(layout.total_bytes <= capacity);
        assert!(ssm_flashinfer_scratch_layout(plan, expected - 1).is_err());
    }
}

#[test]
fn all_live_prefill_paths_preflight_before_effect_and_receipt_after_success() {
    let monolithic = include_str!("trait_prefill.rs");
    let phase1 = include_str!("trait_prefill_phase1.rs");
    let phase3 = include_str!("trait_prefill_phase3.rs");
    let helper = include_str!("trait_prefill_helper.rs");

    assert!(
        monolithic.find("preflight_flashinfer_ssm_prefill").unwrap()
            < monolithic.find("ops::rms_norm_residual(").unwrap()
    );
    assert!(
        phase1.find("preflight_flashinfer_ssm_prefill").unwrap()
            < phase1.find("ops::rms_norm_residual(").unwrap()
    );
    assert!(
        phase3.find("preflight_flashinfer_ssm_prefill").unwrap()
            < phase3.find("ops::gated_rms_norm_prefill(").unwrap()
    );
    assert!(monolithic.contains("launch_flashinfer_ssm_qkvz"));
    assert!(monolithic.contains("launch_flashinfer_ssm_output"));
    assert!(phase1.contains("launch_flashinfer_ssm_qkvz"));
    assert!(phase3.contains("launch_flashinfer_ssm_output"));

    let launch = helper.find("fn launch_flashinfer_ssm_projection").unwrap();
    let receipt = helper
        .find("ATLAS_PREFILL_SSM_FLASHINFER ENGAGED family=ssm")
        .unwrap();
    assert!(launch < receipt);
    assert!(helper[launch..receipt].contains(".launch_eager("));
    assert!(helper.contains("enqueue_qwen38_ssm_projection_dynamic_scale"));
    assert!(!helper.contains("copy_d2h"));
    assert!(!helper.contains("synchronize("));
}
