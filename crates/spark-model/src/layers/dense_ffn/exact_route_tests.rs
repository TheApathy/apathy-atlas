// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    ExactFfnDispatch, ExactFfnMaterializedAvailability, ExactFfnMaterializedRoute,
    ExactFfnPhysicalRoute, ExactFfnRowOffsets, exact_ffn_auto_kgamma_applicable,
    exact_ffn_dispatch, exact_ffn_materialized_route, exact_ffn_physical_route,
    exact_ffn_row_offsets,
};

const ALL_MATERIALIZED: ExactFfnMaterializedAvailability = ExactFfnMaterializedAvailability {
    m8_dual_silu: true,
    m8_f32_down: true,
    m17_dual_silu: true,
    m17_f32_down: true,
    m17_fused_dual_silu: true,
};

#[test]
fn exact_w4_silu_rows_use_available_tier() {
    for rows in 2..=32 {
        assert_eq!(
            exact_ffn_dispatch(rows, true, true),
            ExactFfnDispatch::Batched,
            "rows={rows}"
        );
    }
}

#[test]
fn exact_w4_silu_rows_auto_enter_kgamma_without_legacy_gate() {
    for rows in 2..=32 {
        assert!(exact_ffn_auto_kgamma_applicable(rows, true), "rows={rows}");
    }
    for rows in [0, 1, 33, 64] {
        assert!(!exact_ffn_auto_kgamma_applicable(rows, true), "rows={rows}");
    }
    assert!(!exact_ffn_auto_kgamma_applicable(4, false));
}

#[test]
fn missing_exact_tier_fails_closed_to_k1_rows() {
    for rows in 2..=32 {
        assert_eq!(
            exact_ffn_dispatch(rows, true, false),
            ExactFfnDispatch::PerRowK1,
            "rows={rows}"
        );
    }
}

#[test]
fn non_w4_silu_and_large_batches_keep_existing_route() {
    for rows in 2..=32 {
        assert_eq!(
            exact_ffn_dispatch(rows, false, true),
            ExactFfnDispatch::Existing,
            "rows={rows}"
        );
    }
    assert_eq!(
        exact_ffn_dispatch(1, true, true),
        ExactFfnDispatch::Existing
    );
    assert_eq!(
        exact_ffn_dispatch(33, true, true),
        ExactFfnDispatch::Existing
    );
}

#[test]
fn materialized_m8_requires_both_handles_and_full_fp32_scratch() {
    let rows = 5;
    let intermediate = 17_408;
    let required = rows as usize * intermediate as usize * std::mem::size_of::<f32>();

    assert_eq!(
        exact_ffn_materialized_route(rows, intermediate, ALL_MATERIALIZED, required),
        ExactFfnMaterializedRoute::Split
    );
    for handles in [
        ExactFfnMaterializedAvailability {
            m8_dual_silu: false,
            ..ALL_MATERIALIZED
        },
        ExactFfnMaterializedAvailability {
            m8_f32_down: false,
            ..ALL_MATERIALIZED
        },
    ] {
        assert_eq!(
            exact_ffn_materialized_route(rows, intermediate, handles, required),
            ExactFfnMaterializedRoute::Inline
        );
    }
    assert_eq!(
        exact_ffn_materialized_route(rows, intermediate, ALL_MATERIALIZED, required - 1),
        ExactFfnMaterializedRoute::Inline
    );
}

#[test]
fn fused_m17_is_preferred_and_requires_down_plus_scratch() {
    let rows = 17;
    let intermediate = 17_408;
    let required = rows as usize * intermediate as usize * std::mem::size_of::<f32>();

    assert_eq!(
        exact_ffn_materialized_route(rows, intermediate, ALL_MATERIALIZED, required),
        ExactFfnMaterializedRoute::FusedM17
    );
    let no_down = ExactFfnMaterializedAvailability {
        m17_f32_down: false,
        ..ALL_MATERIALIZED
    };
    assert_eq!(
        exact_ffn_materialized_route(rows, intermediate, no_down, required),
        ExactFfnMaterializedRoute::Inline
    );
    assert_eq!(
        exact_ffn_materialized_route(rows, intermediate, ALL_MATERIALIZED, required - 1),
        ExactFfnMaterializedRoute::Inline
    );
}

#[test]
fn incomplete_fused_m17_falls_back_to_split_then_inline() {
    let no_fused = ExactFfnMaterializedAvailability {
        m17_fused_dual_silu: false,
        ..ALL_MATERIALIZED
    };
    assert_eq!(
        exact_ffn_materialized_route(17, 16, no_fused, 17 * 16 * 4),
        ExactFfnMaterializedRoute::Split
    );

    let fused_without_split = ExactFfnMaterializedAvailability {
        m17_dual_silu: false,
        ..ALL_MATERIALIZED
    };
    assert_eq!(
        exact_ffn_materialized_route(17, 16, fused_without_split, 17 * 16 * 4),
        ExactFfnMaterializedRoute::FusedM17
    );

    let neither = ExactFfnMaterializedAvailability {
        m17_dual_silu: false,
        m17_fused_dual_silu: false,
        ..ALL_MATERIALIZED
    };
    assert_eq!(
        exact_ffn_materialized_route(17, 16, neither, 17 * 16 * 4),
        ExactFfnMaterializedRoute::Inline
    );
}

#[test]
fn materialized_path_is_scoped_to_m8_and_m17_rows() {
    let ample_scratch = usize::MAX;
    for rows in [4, 18, 32] {
        assert_eq!(
            exact_ffn_materialized_route(rows, 17_408, ALL_MATERIALIZED, ample_scratch),
            ExactFfnMaterializedRoute::Inline
        );
    }
}

#[test]
fn materialized_tiers_do_not_borrow_each_others_handles() {
    let m17_only = ExactFfnMaterializedAvailability {
        m8_dual_silu: false,
        m8_f32_down: false,
        ..ALL_MATERIALIZED
    };
    let m8_only = ExactFfnMaterializedAvailability {
        m17_dual_silu: false,
        m17_f32_down: false,
        m17_fused_dual_silu: false,
        ..ALL_MATERIALIZED
    };

    assert_eq!(
        exact_ffn_materialized_route(8, 16, m17_only, 1024),
        ExactFfnMaterializedRoute::Inline
    );
    assert_eq!(
        exact_ffn_materialized_route(9, 16, m8_only, 1024),
        ExactFfnMaterializedRoute::Inline
    );
}

#[test]
fn physical_m8_split_is_diagnostic_exactly_16_and_fail_closed() {
    let rows = 16;
    let intermediate = 17_408;
    let required = rows as usize * intermediate as usize * std::mem::size_of::<f32>();

    assert_eq!(
        exact_ffn_physical_route(rows, intermediate, true, true, true, required),
        ExactFfnPhysicalRoute::SplitM8x2
    );
    for (enabled, exact, materialized, scratch) in [
        (false, true, true, required),
        (true, false, true, required),
        (true, true, false, required),
        (true, true, true, required - 1),
    ] {
        assert_eq!(
            exact_ffn_physical_route(rows, intermediate, enabled, exact, materialized, scratch),
            ExactFfnPhysicalRoute::Native
        );
    }
    for other_rows in [8, 15, 17, 32] {
        assert_eq!(
            exact_ffn_physical_route(other_rows, intermediate, true, true, true, usize::MAX),
            ExactFfnPhysicalRoute::Native
        );
    }
}

#[test]
fn physical_m8_split_offsets_every_tensor_by_eight_rows() {
    let hidden = 5_120;
    let intermediate = 17_408;
    assert_eq!(
        exact_ffn_row_offsets(0, hidden, intermediate),
        Some(ExactFfnRowOffsets {
            input: 0,
            gate_up: 0,
            preactivation: 0,
            output: 0,
        })
    );
    assert_eq!(
        exact_ffn_row_offsets(8, hidden, intermediate),
        Some(ExactFfnRowOffsets {
            input: 8 * hidden as usize * 2,
            gate_up: 8 * intermediate as usize * 2,
            preactivation: 8 * intermediate as usize * 4,
            output: 8 * hidden as usize * 2,
        })
    );
}
