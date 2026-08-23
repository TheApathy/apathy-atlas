// SPDX-License-Identifier: AGPL-3.0-only

//! Dependency-free exact dense-FFN tier and fallback policy.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactFfnTier {
    M4,
    M8,
    M17,
    M32,
}

impl ExactFfnTier {
    pub const fn max_rows(self) -> u32 {
        match self {
            Self::M4 => 4,
            Self::M8 => 8,
            Self::M17 => 17,
            Self::M32 => 32,
        }
    }

    pub const fn dual_symbol(self) -> &'static str {
        match self {
            Self::M4 => "w4a16_gemv_dual_exact_m4",
            Self::M8 => "w4a16_gemv_dual_exact_m8",
            Self::M17 => "w4a16_gemv_dual_exact_m17",
            Self::M32 => "w4a16_gemv_dual_exact_m32",
        }
    }

    pub const fn silu_input_symbol(self) -> &'static str {
        match self {
            Self::M4 => "w4a16_gemv_silu_input_exact_m4",
            Self::M8 => "w4a16_gemv_silu_input_exact_m8",
            Self::M17 => "w4a16_gemv_silu_input_exact_m17",
            Self::M32 => "w4a16_gemv_silu_input_exact_m32",
        }
    }
}

/// Select the smallest exact tier that can hold `rows`. M=1 stays K1 GEMV.
pub const fn exact_ffn_tier_for_rows(rows: u32) -> Option<ExactFfnTier> {
    match rows {
        2..=4 => Some(ExactFfnTier::M4),
        5..=8 => Some(ExactFfnTier::M8),
        9..=17 => Some(ExactFfnTier::M17),
        18..=32 => Some(ExactFfnTier::M32),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactFfnRoute {
    Exact(ExactFfnTier),
    SerialK1(ExactFfnTier),
}

impl ExactFfnRoute {
    pub const fn tier(self) -> ExactFfnTier {
        match self {
            Self::Exact(tier) | Self::SerialK1(tier) => tier,
        }
    }

    pub const fn provenance(self) -> &'static str {
        match self {
            Self::Exact(_) => "exact_dynamic_m_dense_ffn_nvfp4",
            Self::SerialK1(_) => "serial_k1_dense_ffn_nvfp4",
        }
    }
}

/// Route exact only when both FFN stages resolve for the selected tier.
pub const fn exact_ffn_route_for_rows(
    rows: u32,
    dual_present: bool,
    silu_input_present: bool,
) -> Option<ExactFfnRoute> {
    let Some(tier) = exact_ffn_tier_for_rows(rows) else {
        return None;
    };
    Some(if dual_present && silu_input_present {
        ExactFfnRoute::Exact(tier)
    } else {
        ExactFfnRoute::SerialK1(tier)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_boundaries_select_smallest_register_family() {
        let cases = [
            (1, None),
            (2, Some(ExactFfnTier::M4)),
            (4, Some(ExactFfnTier::M4)),
            (5, Some(ExactFfnTier::M8)),
            (8, Some(ExactFfnTier::M8)),
            (9, Some(ExactFfnTier::M17)),
            (17, Some(ExactFfnTier::M17)),
            (18, Some(ExactFfnTier::M32)),
            (32, Some(ExactFfnTier::M32)),
            (33, None),
        ];

        for (rows, expected) in cases {
            assert_eq!(exact_ffn_tier_for_rows(rows), expected, "rows={rows}");
        }
    }

    #[test]
    fn symbols_bind_dual_and_down_to_the_same_tier() {
        let cases = [
            (
                ExactFfnTier::M4,
                4,
                "w4a16_gemv_dual_exact_m4",
                "w4a16_gemv_silu_input_exact_m4",
            ),
            (
                ExactFfnTier::M8,
                8,
                "w4a16_gemv_dual_exact_m8",
                "w4a16_gemv_silu_input_exact_m8",
            ),
            (
                ExactFfnTier::M17,
                17,
                "w4a16_gemv_dual_exact_m17",
                "w4a16_gemv_silu_input_exact_m17",
            ),
            (
                ExactFfnTier::M32,
                32,
                "w4a16_gemv_dual_exact_m32",
                "w4a16_gemv_silu_input_exact_m32",
            ),
        ];

        for (tier, max_rows, dual, down) in cases {
            assert_eq!(tier.max_rows(), max_rows);
            assert_eq!(tier.dual_symbol(), dual);
            assert_eq!(tier.silu_input_symbol(), down);
        }
    }

    #[test]
    fn route_falls_back_unless_both_stage_handles_exist() {
        let tier = ExactFfnTier::M8;
        let exact = exact_ffn_route_for_rows(5, true, true).unwrap();
        assert_eq!(exact, ExactFfnRoute::Exact(tier));
        assert_eq!(exact.tier(), tier);
        assert_eq!(exact.provenance(), "exact_dynamic_m_dense_ffn_nvfp4");
        assert_eq!(
            exact_ffn_route_for_rows(5, false, true),
            Some(ExactFfnRoute::SerialK1(tier))
        );
        assert_eq!(
            exact_ffn_route_for_rows(5, true, false),
            Some(ExactFfnRoute::SerialK1(tier))
        );
        let serial = exact_ffn_route_for_rows(5, false, false).unwrap();
        assert_eq!(serial, ExactFfnRoute::SerialK1(tier));
        assert_eq!(serial.tier(), tier);
        assert_eq!(serial.provenance(), "serial_k1_dense_ffn_nvfp4");
        assert_eq!(exact_ffn_route_for_rows(1, true, true), None);
    }
}
