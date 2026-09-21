// SPDX-License-Identifier: AGPL-3.0-only

use std::ffi::OsStr;

use super::gdn_c143_prefill::{
    GdnC143PrefillRoute, QWEN38_GDN_QKVZ_WIDTH, gdn_c143_exact_geometry, gdn_c143_prefill_route,
    parse_gdn_c143_selector,
};

#[test]
fn selector_and_route_are_strict_default_off() {
    assert!(!parse_gdn_c143_selector(None).unwrap());
    assert!(!parse_gdn_c143_selector(Some(OsStr::new("0"))).unwrap());
    assert!(parse_gdn_c143_selector(Some(OsStr::new("1"))).unwrap());
    for invalid in ["", "true", "01", "2", " 1"] {
        assert!(parse_gdn_c143_selector(Some(OsStr::new(invalid))).is_err());
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(parse_gdn_c143_selector(Some(OsStr::from_bytes(&[0xff]))).is_err());
    }
    assert_eq!(
        gdn_c143_prefill_route(false, true),
        GdnC143PrefillRoute::Disabled
    );
    assert_eq!(
        gdn_c143_prefill_route(true, false),
        GdnC143PrefillRoute::Ineligible
    );
    assert_eq!(
        gdn_c143_prefill_route(true, true),
        GdnC143PrefillRoute::Required
    );
}

#[test]
fn only_exact_qwen38_c1_eager_shapes_are_eligible() {
    for rows in [2_079, 8_192] {
        assert!(gdn_c143_exact_geometry(
            rows, 16, 48, 128, 128, 10_240, 96, 16_384
        ));
    }
    for stale in [
        (2_078, 16, 48, 128, 128, 10_240, 96, 16_384),
        (2_079, 8, 48, 128, 128, 10_240, 96, 16_384),
        (2_079, 16, 32, 128, 128, 10_240, 96, 16_384),
        (2_079, 16, 48, 64, 128, 10_240, 96, 16_384),
        (2_079, 16, 48, 128, 64, 10_240, 96, 16_384),
        (2_079, 16, 48, 128, 128, 8_192, 96, 16_384),
        (2_079, 16, 48, 128, 128, 10_240, 48, 16_384),
        (2_079, 16, 48, 128, 128, 10_240, 96, 12_288),
    ] {
        assert!(!gdn_c143_exact_geometry(
            stale.0, stale.1, stale.2, stale.3, stale.4, stale.5, stale.6, stale.7
        ));
    }
}

#[test]
fn authoritative_shared_arena_exceeds_both_exact_workspaces() {
    for (rows, required) in [(2_079usize, 130_565_952usize), (8_192, 506_462_208)] {
        let capacity = rows * QWEN38_GDN_QKVZ_WIDTH * 4;
        assert!(capacity >= required);
        assert_eq!(
            capacity,
            if rows == 2_079 {
                136_249_344
            } else {
                536_870_912
            }
        );
    }
}

#[test]
fn every_live_route_preflights_before_effect_and_receipts_follow_success() {
    let monolithic = include_str!("trait_prefill.rs");
    let phase1 = include_str!("trait_prefill_phase1.rs");
    let two_phase = include_str!("trait_prefill_gdn.rs");
    let adapter = include_str!("gdn_c143_prefill.rs");

    assert!(
        monolithic.find("preflight_gdn_c143_prefill").unwrap()
            < monolithic.find("ops::rms_norm_residual(").unwrap()
    );
    assert!(
        phase1.find("preflight_gdn_c143_prefill").unwrap()
            < phase1.find("ops::rms_norm_residual(").unwrap()
    );
    let two_phase_body = &two_phase[two_phase
        .find("pub(super) fn prefill_gdn_full_inner")
        .unwrap()..];
    assert!(
        two_phase_body.find("preflight_gdn_c143_prefill").unwrap()
            < two_phase_body.find("gdn_profile_enabled()").unwrap()
    );
    assert!(monolithic.contains("gdn_c143.launch(ctx.gpu)?"));
    assert!(two_phase.contains("gdn_c143.launch(ctx.gpu)?"));

    let lock = adapter.find(".launch_lock").unwrap();
    let launch = adapter.find("prepared.launch_eager").unwrap();
    let receipt = adapter.find("ENGAGED ATLAS_GDN_C143_PREFILL").unwrap();
    assert!(lock < launch);
    assert!(launch < receipt);
    for forbidden in [
        ".alloc(",
        ".free(",
        ".synchronize(",
        "GraphHandle",
        "capture_begin",
    ] {
        assert!(
            !adapter.contains(forbidden),
            "forbidden adapter token: {forbidden}"
        );
    }
}
