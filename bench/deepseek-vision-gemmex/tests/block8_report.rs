// SPDX-License-Identifier: AGPL-3.0-only
//! Same-input numerical reporting must never invent an authoritative teacher.
#[path = "../src/metrics.rs"]
mod metrics;
#[path = "../src/block8/report.rs"]
mod report;

use report::{CandidateMode, check_native_control, compare_repeats};
use serde_json::json;

const WORDS: usize = 20 * 5632;
fn raw(word: u16) -> Vec<u8> {
    word.to_le_bytes().repeat(WORDS)
}
fn modes() -> [CandidateMode; 4] {
    [
        CandidateMode::GemmExDefault,
        CandidateMode::GemmExFull,
        CandidateMode::LtBaseline,
        CandidateMode::LtComputeTypeOnly,
    ]
}

#[test]
fn native_replay_requires_exact_bytes_before_any_candidate_comparison() {
    let native = raw(0x3f80);
    check_native_control(&native, &native).unwrap();
    let mut changed = native.clone();
    changed[0] ^= 1;
    assert!(check_native_control(&changed, &native).is_err());
    assert!(check_native_control(&native, &changed).is_err());
    assert!(check_native_control(&native[..native.len() - 2], &native).is_err());
    for word in [0x7f80, 0xff80, 0x7fc1] {
        let bad = raw(word);
        assert!(check_native_control(&bad, &bad).is_err());
    }
}

#[test]
fn exact_operator_match_is_not_teacher_encoder_or_performance_qualification() {
    let native = raw(0x3f80);
    for mode in modes() {
        let repeats = [native.clone(), native.clone()];
        let before = repeats.clone();
        let value = compare_repeats(mode, &repeats, &native).unwrap();
        assert_eq!(value["mode"], mode.label());
        assert_eq!(value["status"], "DIAGNOSTIC_COMPLETE");
        assert_eq!(
            value["comparison_kind"],
            "same-input-operator-versus-native-control"
        );
        assert_eq!(value["metrics_denominator"], "captured native block-08-fc1");
        assert_eq!(value["exact_vs_native"], true);
        assert_eq!(value["repeat_byte_equal"], true);
        assert_eq!(value["repeat_count"], 2);
        assert_eq!(value["teacher_reference_available"], false);
        assert!(value.get("teacher_reference_sha256").is_some());
        assert!(value.get("teacher_reference_payload").is_some());
        assert_eq!(value["teacher_reference_sha256"], json!(null));
        assert_eq!(value["teacher_reference_payload"], json!(null));
        assert_eq!(value["full_encoder_qualified"], false);
        assert_eq!(value["performance_qualified"], false);
        assert!(
            value.get("reference_exact").is_none(),
            "ambiguous teacher gate leaked"
        );
        assert!(
            value.get("passed").is_none(),
            "operator match promoted to a generic pass"
        );
        assert_eq!(value["metrics"]["relative_l2"], 0.0);
        assert_eq!(value["metrics"]["f32_exact_fraction"], 1.0);
        assert_eq!(value["metrics"]["shape"], json!([20, 5632]));
        assert_eq!(repeats, before);
    }
    assert_eq!(
        modes().map(|m| m.label()),
        [
            "gemmex-default",
            "gemmex-full",
            "lt-baseline",
            "lt-compute-type-only"
        ]
    );
}

#[test]
fn candidate_differences_are_retained_with_native_denominator_not_faked_teacher_failure() {
    let native = raw(0x4000); // 2
    let other = raw(0x3f80); // 1
    for mode in modes() {
        let v = compare_repeats(mode, &[other.clone(), other.clone()], &native).unwrap();
        assert_eq!(v["status"], "DIAGNOSTIC_COMPLETE");
        assert_eq!(v["exact_vs_native"], false);
        assert_eq!(v["teacher_reference_available"], false);
        assert_eq!(v["full_encoder_qualified"], false);
        assert_eq!(v["metrics"]["relative_l2"], 0.5);
        assert_eq!(v["metrics"]["max_abs_error"], 1.0);
        assert_eq!(v["metrics"]["f32_exact_fraction"], 0.0);
        assert!((v["metrics"]["cosine"].as_f64().unwrap() - 1.0).abs() < 1e-13);
        let reverse = compare_repeats(mode, &[native.clone(), native.clone()], &other).unwrap();
        assert_eq!(reverse["metrics"]["relative_l2"], 1.0);
    }
}

#[test]
fn reset_mismatch_bad_repetition_extent_nonfinite_or_zero_rows_fail_closed() {
    let native = raw(0x3f80);
    assert!(compare_repeats(CandidateMode::GemmExDefault, &[], &native).is_err());
    assert!(compare_repeats(CandidateMode::GemmExDefault, &[native.clone()], &native).is_err());
    assert!(
        compare_repeats(
            CandidateMode::GemmExDefault,
            &[native.clone(), native.clone(), native.clone()],
            &native
        )
        .is_err()
    );
    let mut changed = native.clone();
    changed[0] ^= 1;
    assert!(
        compare_repeats(
            CandidateMode::GemmExFull,
            &[native.clone(), changed],
            &native
        )
        .is_err()
    );
    for bad in [
        raw(0x7f80),
        raw(0xff80),
        raw(0x7fc1),
        vec![0; WORDS * 2],
        vec![1; WORDS * 2 - 1],
    ] {
        assert!(
            compare_repeats(
                CandidateMode::LtBaseline,
                &[bad.clone(), bad.clone()],
                &native
            )
            .is_err()
        );
        assert!(
            compare_repeats(
                CandidateMode::LtBaseline,
                &[native.clone(), native.clone()],
                &bad
            )
            .is_err()
        );
    }
    let mut zero_row = native.clone();
    zero_row[..5632 * 2].fill(0);
    assert!(
        compare_repeats(
            CandidateMode::LtComputeTypeOnly,
            &[zero_row.clone(), zero_row],
            &native
        )
        .is_err()
    );
}
