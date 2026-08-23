// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    compute_num_splits_for, flat_window_crosses_k1_split_boundary_for, flat_window_min_seq_len_host,
};

#[test]
fn default_workspace_regime_keeps_occupancy_policy() {
    assert_eq!(compute_num_splits_for(24, 1, 94, false), 2);
    assert_eq!(compute_num_splits_for(24, 5, 98, false), 1);
    assert_eq!(compute_num_splits_for(4, 3, 98, false), 4);
}

#[test]
fn aggressive_regime_matches_independent_k1_geometry() {
    for seq_len in [1, 94, 511, 512, 513, 1536, 32_768] {
        let k1 = compute_num_splits_for(24, 1, seq_len, true);
        for rows in [2, 3, 5, 17, 32] {
            assert_eq!(
                compute_num_splits_for(24, rows, seq_len, true),
                k1,
                "rows={rows} seq_len={seq_len}"
            );
        }
    }
}

#[test]
fn aggressive_regime_retains_sequence_and_cap_bounds() {
    assert_eq!(compute_num_splits_for(24, 5, 98, true), 2);
    assert_eq!(compute_num_splits_for(24, 5, 1536, true), 3);
    assert_eq!(compute_num_splits_for(24, 5, u32::MAX, true), 64);
    assert_eq!(compute_num_splits_for(4, 32, u32::MAX, true), 64);
}

#[test]
fn aggressive_regime_does_not_overcommit_small_head_arena() {
    assert_eq!(compute_num_splits_for(4, 32, 98, true), 1);
    assert_ne!(compute_num_splits_for(4, 32, 98, true), 12);
}

#[test]
fn qwen_flat_window_detects_sequence_split_boundaries() {
    assert_eq!(flat_window_min_seq_len_host(5, 1025), 1021);
    assert!(!flat_window_crosses_k1_split_boundary_for(
        24, 5, 1024, true
    ));
    assert!(flat_window_crosses_k1_split_boundary_for(24, 5, 1025, true));
    assert!(!flat_window_crosses_k1_split_boundary_for(
        24, 5, 1029, true
    ));
    assert!(!flat_window_crosses_k1_split_boundary_for(
        24, 5, 1025, false
    ));
    assert!(!flat_window_crosses_k1_split_boundary_for(
        4, 32, 1025, true
    ));
}
