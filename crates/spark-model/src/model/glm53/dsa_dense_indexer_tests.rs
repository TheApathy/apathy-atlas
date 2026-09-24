// SPDX-License-Identifier: AGPL-3.0-only
// Exercise the registered production helper and its existing geometry constants.
use super::dsa_dense_indexer as candidate;

use candidate::{Glm53DsaDenseIndexerPlan, dense_full_coverage, parse_dense_indexer_skip};

#[test]
fn strict_opt_in_missing_and_zero_preserve_existing_work() {
    assert_eq!(parse_dense_indexer_skip(None).unwrap(), false);
    assert_eq!(parse_dense_indexer_skip(Some("0")).unwrap(), false);
    assert_eq!(parse_dense_indexer_skip(Some("1")).unwrap(), true);
    for flag in [None, Some("0")] {
        let requested = parse_dense_indexer_skip(flag).unwrap();
        let plan = Glm53DsaDenseIndexerPlan::new(2000, 0, 2048, true, requested).unwrap();
        assert!(plan.dense_full_coverage());
        assert!(!plan.skip_indexer_query());
        assert_eq!(plan.indexer_launches(), 4);
    }
}

#[test]
fn malformed_flags_are_errors_not_silent_defaults_or_truthy_values() {
    for value in [
        "", "true", "false", "yes", "2", "01", "+1", " 1", "1 ", "1\n", "\0", "１",
    ] {
        assert!(
            parse_dense_indexer_skip(Some(value)).is_err(),
            "accepted {value:?}"
        );
    }
}

#[test]
fn dense_end_2051_is_inclusive_and_2052_requires_indexer() {
    for (rows, start, end, dense) in [
        (2048, 3, 2051, true),
        (2048, 4, 2052, false),
        (2, 2049, 2051, true),
        (2, 2050, 2052, false),
    ] {
        let plan = Glm53DsaDenseIndexerPlan::new(rows, start, end, true, true).unwrap();
        assert_eq!(plan.dense_full_coverage(), dense);
        assert_eq!(plan.skip_indexer_query(), dense);
        assert_eq!(plan.indexer_launches(), if dense { 0 } else { 4 });
        assert_eq!(dense_full_coverage(rows, start, end, true).unwrap(), dense);
    }
}

#[test]
fn nonzero_start_uses_absolute_end_not_just_chunk_rows() {
    let initial = Glm53DsaDenseIndexerPlan::new(128, 0, 4096, true, true).unwrap();
    let dense = Glm53DsaDenseIndexerPlan::new(128, 1923, 4096, true, true).unwrap();
    let sparse = Glm53DsaDenseIndexerPlan::new(128, 1924, 4096, true, true).unwrap();
    assert!(initial.skip_indexer_query() && dense.skip_indexer_query());
    assert!(!sparse.skip_indexer_query());
}

#[test]
fn m1_and_exact_small_m_do_not_enter_layer_major_skip() {
    for layer_major in [false, true] {
        for requested in [false, true] {
            let plan =
                Glm53DsaDenseIndexerPlan::new(1, 2050, 2051, layer_major, requested).unwrap();
            assert!(!plan.dense_full_coverage() && !plan.skip_indexer_query());
            assert_eq!(plan.indexer_launches(), 4);
        }
    }
    for rows in 2..=8 {
        let legacy = Glm53DsaDenseIndexerPlan::new(rows, 0, 2048, false, true).unwrap();
        let prompt = Glm53DsaDenseIndexerPlan::new(rows, 0, 2048, true, true).unwrap();
        assert!(!legacy.dense_full_coverage() && !legacy.skip_indexer_query());
        assert!(prompt.dense_full_coverage() && prompt.skip_indexer_query());
    }
}

#[test]
fn invalid_geometry_fails_before_a_plan_can_authorize_effects() {
    for (rows, position, capacity, layer_major) in [
        (0, 0, 2048, true),
        (8193, 0, 16384, true),
        (9, 0, 2048, false),
        (2, 0, 0, true),
        (2, 2047, 2048, true),
        (2, u32::MAX, u32::MAX, true),
        (2048, u32::MAX - 1024, u32::MAX, true),
        (2, 0, 1_048_577, true),
    ] {
        for requested in [false, true] {
            assert!(
                Glm53DsaDenseIndexerPlan::new(rows, position, capacity, layer_major, requested)
                    .is_err()
            );
        }
        assert!(dense_full_coverage(rows, position, capacity, layer_major).is_err());
    }
}

#[test]
fn opt_in_changes_only_indexer_work_not_the_existing_density_decision() {
    for layer_major in [false, true] {
        for rows in [1, 2, 4, 8] {
            for position in [0, 3, 2043, 2047, 2048, 2049, 2050, 3000] {
                let off = Glm53DsaDenseIndexerPlan::new(rows, position, 4096, layer_major, false)
                    .unwrap();
                let on =
                    Glm53DsaDenseIndexerPlan::new(rows, position, 4096, layer_major, true).unwrap();
                assert_eq!(off.dense_full_coverage(), on.dense_full_coverage());
                assert_eq!(on.skip_indexer_query(), on.dense_full_coverage());
                assert!(!off.skip_indexer_query());
            }
        }
    }
}
