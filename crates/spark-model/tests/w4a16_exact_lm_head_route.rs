// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../src/layers/ops/gemv_exact_lm_head/route.rs"]
mod route;

use route::{
    ExactLmHeadRoute, ExactLmHeadTier, exact_lm_head_route_for_rows, exact_lm_head_tier_for_rows,
};

#[test]
fn tiers_cover_every_and_only_qualified_multi_row_width() {
    for (rows, tier) in [
        (2, ExactLmHeadTier::M4),
        (4, ExactLmHeadTier::M4),
        (5, ExactLmHeadTier::M8),
        (8, ExactLmHeadTier::M8),
        (9, ExactLmHeadTier::M17),
        (17, ExactLmHeadTier::M17),
        (18, ExactLmHeadTier::M32),
        (32, ExactLmHeadTier::M32),
    ] {
        assert_eq!(exact_lm_head_tier_for_rows(rows), Some(tier));
        assert!(tier.max_rows() >= rows);
        assert!(tier.symbol().ends_with(&format!("m{}", tier.max_rows())));
        // The register-tiled twin must name the SAME tier width, so a widened
        // tier can never silently launch a narrower rt2 kernel.
        assert_eq!(
            tier.symbol_rt2(),
            tier.symbol().replace("exact_m", "exact_rt2_m")
        );
        assert!(
            tier.symbol_rt2()
                .ends_with(&format!("m{}", tier.max_rows()))
        );
    }
    for rows in [0, 1, 33, u32::MAX] {
        assert_eq!(exact_lm_head_tier_for_rows(rows), None);
    }
}

#[test]
fn route_and_provenance_fail_closed_for_every_qualified_width() {
    for rows in 2..=32 {
        let tier = exact_lm_head_tier_for_rows(rows).expect("covered row");
        let exact = exact_lm_head_route_for_rows(rows, true);
        let serial = exact_lm_head_route_for_rows(rows, false);
        assert_eq!(exact, Some(ExactLmHeadRoute::Exact(tier)));
        assert_eq!(serial, Some(ExactLmHeadRoute::SerialK1(tier)));
        assert_eq!(exact.unwrap().tier().label(), tier.label());
        assert_eq!(serial.unwrap().tier().label(), tier.label());
        assert_eq!(exact.unwrap().provenance(), "exact_dynamic_m_nvfp4");
        assert_eq!(serial.unwrap().provenance(), "serial_k1_row_major_nvfp4");
    }
}

#[test]
fn missing_selected_tier_never_promotes_to_another_exact_handle() {
    for rows in 5..=8 {
        assert_eq!(
            exact_lm_head_route_for_rows(rows, false),
            Some(ExactLmHeadRoute::SerialK1(ExactLmHeadTier::M8))
        );
    }
    assert_eq!(
        exact_lm_head_route_for_rows(9, true),
        Some(ExactLmHeadRoute::Exact(ExactLmHeadTier::M17))
    );
}
