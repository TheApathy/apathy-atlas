// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    Qwen4ExactSsmRowsRoute, parse_qwen4_k16_batched_verify, parse_qwen4_k16_exact,
    qwen4_exact_ssm_rows_route,
};

#[test]
fn native_k16_ssm_admission_is_exact_and_keeps_shortened_frames_on_parent_routes() {
    for value in [None, Some("0")] {
        assert!(!parse_qwen4_k16_batched_verify(value).unwrap());
    }
    assert!(parse_qwen4_k16_batched_verify(Some("1")).unwrap());
    assert!(parse_qwen4_k16_exact(Some("1")).unwrap());
    for hostile in ["", "true", "01", " 1", "1 ", "-1", "2"] {
        assert!(parse_qwen4_k16_batched_verify(Some(hostile)).is_err());
        assert!(parse_qwen4_k16_exact(Some(hostile)).is_err());
    }

    assert_eq!(
        qwen4_exact_ssm_rows_route(16, false, false, true, true, true),
        Some(Qwen4ExactSsmRowsRoute::NativeK16)
    );
    for (global, exact) in [(false, false), (false, true), (true, false)] {
        assert_eq!(
            qwen4_exact_ssm_rows_route(16, false, false, global, exact, true),
            None,
            "K16 SSM armed without both global={global} and exact={exact} selectors"
        );
    }
    for (rows, has_intermediates) in [
        (0, true),
        (1, true),
        (4, true),
        (15, true),
        (16, false),
        (17, true),
        (32, true),
    ] {
        assert_eq!(
            qwen4_exact_ssm_rows_route(rows, false, false, true, true, has_intermediates),
            None,
            "K16 route widened to rows={rows}, intermediates={has_intermediates}"
        );
    }
    assert_eq!(
        qwen4_exact_ssm_rows_route(16, true, true, false, true, true),
        None,
        "legacy K5 knobs must not arm K16"
    );

    for rows in [5, 9] {
        let legacy = Some(Qwen4ExactSsmRowsRoute::LegacyK5OrK9);
        assert_eq!(
            qwen4_exact_ssm_rows_route(rows, true, true, false, false, true),
            legacy
        );
        assert_eq!(
            qwen4_exact_ssm_rows_route(rows, true, true, true, true, true),
            legacy,
            "global K16 opt-in changed a shortened legacy frame"
        );
        assert_eq!(
            qwen4_exact_ssm_rows_route(rows, true, false, true, true, true),
            None,
            "K16 opt-in bypassed the legacy K5/K9 SSM sub-gate"
        );
    }
}

#[test]
fn native_k16_source_preserves_ordered_live_and_checkpoint_state_contract() {
    let source = include_str!("qwen4_k5_ssm.rs");
    let trait_source = include_str!("trait_decode.rs");

    assert!(source.contains("for row in 0..rows"));
    assert!(source.contains("state.conv_state"));
    assert!(source.contains("conv_intermediate.offset(row * conv_intermediate_stride)"));
    assert!(source.contains("state.h_state"));
    assert!(source.contains("h_intermediate.offset(row * h_intermediate_stride)"));
    assert!(source.contains("!state.h_is_f16"));
    assert!(source.contains("h_intermediate != state.h_state"));
    assert!(source.contains("conv_intermediate != state.conv_state"));
    assert!(source.contains("ops::conv1d_update_l2norm_f32_sequence("));
    assert!(source.contains("ops::gdn_decode_f32_sequence("));
    assert!(source.contains("route.force_sequence_kernels()"));
    let preflight = trait_source
        .find("preflight_qwen4_exact_ssm_rows(")
        .unwrap();
    let first_effect = trait_source.find("attn_hyper.prepare_decode(").unwrap();
    assert!(
        preflight < first_effect,
        "K16 preflight follows a device effect"
    );
    let forward = trait_source
        .find("self.ssm_forward_qwen4_exact_rows(")
        .unwrap();
    let receipt = trait_source
        .find("ssm_exact_m16_projection_conv_gdn_sequence_enqueued")
        .unwrap();
    assert!(
        forward < receipt,
        "K16 SSM receipt precedes exact-row enqueue"
    );
}
