// SPDX-License-Identifier: AGPL-3.0-only
//! P11 RED: production example's raw input comparison, not a GPU oracle.
#[path = "../examples/glm53_dflash2_kv_parity/projected.rs"]
#[allow(dead_code)]
mod projected;
use projected::{InputPair, compare_inputs};

fn pair<'a>(before: &'a [u8], after: &'a [u8]) -> InputPair<'a> {
    InputPair { before, after }
}
fn row(value: u16) -> Vec<u8> {
    [value.to_le_bytes(); 4096].concat()
}

#[test]
fn actual_context_one_two_checks_pair_identity_immutability_and_prior_row() {
    let first = row(0x3f80);
    let report = compare_inputs(pair(&first, &first), pair(&first, &first), None, 8192).unwrap();
    assert_eq!(report["exact"], true);
    assert!(report["previous_prefix_exact"].is_null());
    let second = [first.clone(), row(0x4000)].concat();
    let report = compare_inputs(
        pair(&second, &second),
        pair(&second, &second),
        Some(&first),
        8192,
    )
    .unwrap();
    for key in [
        "exact",
        "pair_before_exact",
        "pair_after_exact",
        "reference_immutable",
        "candidate_immutable",
        "previous_prefix_exact",
    ] {
        assert_eq!(report[key], true, "{key}");
    }
}

#[test]
fn reference_candidate_and_old_prefix_changes_are_distinct_failed_receipts() {
    let first = row(0x3f80);
    let second = [first.clone(), row(0x4000)].concat();
    let mut changed = second.clone();
    changed[0] ^= 1; // Finite BF16 one-bit difference must not be tolerated.
    let report = compare_inputs(
        pair(&second, &second),
        pair(&second, &changed),
        Some(&first),
        8192,
    )
    .unwrap();
    assert_eq!(report["exact"], false);
    assert_eq!(report["pair_before_exact"], true);
    assert_eq!(report["pair_after_exact"], false);
    assert_eq!(report["reference_immutable"], true);
    assert_eq!(report["candidate_immutable"], false);
    let report = compare_inputs(
        pair(&second, &changed),
        pair(&second, &second),
        Some(&first),
        8192,
    )
    .unwrap();
    assert_eq!(report["reference_immutable"], false);
    assert_eq!(report["candidate_immutable"], true);
    let report = compare_inputs(
        pair(&changed, &changed),
        pair(&changed, &changed),
        Some(&first),
        8192,
    )
    .unwrap();
    assert_eq!(report["pair_before_exact"], true);
    assert_eq!(report["reference_immutable"], true);
    assert_eq!(report["previous_prefix_exact"], false);
    assert_eq!(report["exact"], false);
}

#[test]
fn malformed_extents_prior_rows_and_nonfinite_payloads_are_not_comparisons() {
    let first = row(0x3f80);
    let two = [first.clone(), first.clone()].concat();
    assert!(compare_inputs(pair(&first, &first), pair(&two, &two), None, 8192).is_err());
    assert!(compare_inputs(pair(&first, &first), pair(&first, &first), Some(&two), 8192).is_err());
    assert!(
        compare_inputs(
            pair(&first, &first),
            pair(&first, &first),
            Some(&first[..2]),
            8192
        )
        .is_err()
    );
    assert!(compare_inputs(pair(&[], &[]), pair(&[], &[]), None, 8192).is_err());
    for row_bytes in [0, 3, 8190] {
        assert!(
            compare_inputs(pair(&first, &first), pair(&first, &first), None, row_bytes).is_err()
        );
    }
    let infinite = row(0x7f80);
    assert!(
        compare_inputs(
            pair(&infinite, &infinite),
            pair(&infinite, &infinite),
            None,
            8192
        )
        .is_err()
    );
}
