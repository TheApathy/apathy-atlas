// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    ExactAttnOProjDispatch, exact_attn_o_proj_dispatch, should_auto_serialize_paged_split_boundary,
};
use crate::layers::ops::exact_lm_head_route_for_rows;

#[test]
fn ordinary_nvfp4_uses_exact_tiers_at_every_boundary() {
    for rows in [2, 4, 5, 8, 9, 17, 18, 32] {
        assert_eq!(
            exact_attn_o_proj_dispatch(true, exact_lm_head_route_for_rows(rows, true),),
            ExactAttnOProjDispatch::Exact,
            "rows={rows}"
        );
    }
}

#[test]
fn missing_exact_tier_fails_closed_to_k1_rows() {
    for rows in 2..=32 {
        assert_eq!(
            exact_attn_o_proj_dispatch(true, exact_lm_head_route_for_rows(rows, false),),
            ExactAttnOProjDispatch::PerRowK1,
            "rows={rows}"
        );
    }
}

#[test]
fn other_encodings_and_out_of_range_rows_keep_existing_paths() {
    assert_eq!(
        exact_attn_o_proj_dispatch(false, exact_lm_head_route_for_rows(5, true)),
        ExactAttnOProjDispatch::Existing
    );
    for rows in [0, 1, 33, 256] {
        assert_eq!(
            exact_attn_o_proj_dispatch(true, exact_lm_head_route_for_rows(rows, true),),
            ExactAttnOProjDispatch::Existing,
            "rows={rows}"
        );
    }
}

#[test]
fn split_boundary_serialization_is_flat_only() {
    assert!(should_auto_serialize_paged_split_boundary(
        2, false, false, true
    ));
    assert!(!should_auto_serialize_paged_split_boundary(
        2, false, true, true
    ));
    assert!(!should_auto_serialize_paged_split_boundary(
        2, true, false, true
    ));
    assert!(!should_auto_serialize_paged_split_boundary(
        1, false, false, true
    ));
}
