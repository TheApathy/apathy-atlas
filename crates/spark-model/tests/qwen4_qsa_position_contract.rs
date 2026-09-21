// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../src/layers/qwen4_qsa/position_contract.rs"]
mod production;

use production::{prefill_index_batch_is_safe, valid_physical_query};

#[test]
fn only_the_current_physical_query_is_admitted() {
    for position in 0..8193 {
        assert!(valid_physical_query(position, position + 1));
        assert!(!valid_physical_query(position, position));
        assert!(!valid_physical_query(position, position + 2));
    }
    assert!(!valid_physical_query(0, 0));
    assert!(!valid_physical_query(usize::MAX, 0));
}

#[test]
fn unsupported_batches_and_partial_group_overwrite_fail_closed() {
    for start in [0, 1, 2, 3, 4, 12, 13, 14, 15, 16, 2047, 2048] {
        for rows in 0..=8192 {
            let admitted = rows != 0 && rows <= 4 && start % 4 + rows <= 4;
            assert_eq!(
                prefill_index_batch_is_safe(rows, start),
                admitted,
                "start={start}, rows={rows}"
            );
        }
    }
    assert!(!prefill_index_batch_is_safe(1, usize::MAX));
    assert!(!prefill_index_batch_is_safe(usize::MAX, 1));
}

#[test]
fn valid_partial_groups_remain_available_without_fallback() {
    for (start, rows) in [(0, 4), (1, 3), (2, 2), (3, 1), (12, 4), (15, 1), (2048, 4)] {
        assert!(prefill_index_batch_is_safe(rows, start));
    }
    for (start, rows) in [(1, 4), (2, 3), (3, 2), (14, 3), (15, 2), (2047, 2)] {
        assert!(!prefill_index_batch_is_safe(rows, start));
    }
}
