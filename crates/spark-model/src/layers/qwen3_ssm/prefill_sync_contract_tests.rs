// SPDX-License-Identifier: AGPL-3.0-only

//! Guards against reintroducing unconditional device drains into prefill.

#[test]
fn production_prefill_has_no_historical_crash_localization_barriers() {
    let monolithic = include_str!("trait_prefill.rs");
    let phase_one = include_str!("trait_prefill_phase1.rs");

    for marker in [
        "SSM prefill ENTRY",
        "SSM prefill: SYNC after rms_norm",
        "ssm phase1 ENTRY",
        "ssm phase1 L{}: SYNC after rms_norm",
        "ssm phase1: SYNC after QKVZ GEMM",
        "ssm phase1: SYNC after deinterleave",
        "ssm phase1: SYNC after BA+gates",
        "ssm phase1: SYNC after conv1d",
    ] {
        assert!(!monolithic.contains(marker), "monolithic marker: {marker}");
        assert!(!phase_one.contains(marker), "phase-one marker: {marker}");
    }
    assert!(!monolithic.contains("if k > 4096"));
    assert!(!phase_one.contains("synchronize(stream)"));
}
