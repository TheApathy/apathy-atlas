// SPDX-License-Identifier: AGPL-3.0-only

//! Dependency-free exact LM-head tier and arithmetic-provenance policy.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactLmHeadTier {
    M4,
    M8,
    M17,
    M32,
}

impl ExactLmHeadTier {
    pub const fn max_rows(self) -> u32 {
        match self {
            Self::M4 => 4,
            Self::M8 => 8,
            Self::M17 => 17,
            Self::M32 => 32,
        }
    }

    pub const fn symbol(self) -> &'static str {
        match self {
            Self::M4 => "w4a16_gemv_batch_logits_exact_m4",
            Self::M8 => "w4a16_gemv_batch_logits_exact_m8",
            Self::M17 => "w4a16_gemv_batch_logits_exact_m17",
            Self::M32 => "w4a16_gemv_batch_logits_exact_m32",
        }
    }

    /// Register-tiled twin of `symbol()`: T=2 adjacent output rows per lane
    /// group. Same per-output operand sequence, so it is a numerics-preserving
    /// substitution, not a different arithmetic.
    pub const fn symbol_rt2(self) -> &'static str {
        match self {
            Self::M4 => "w4a16_gemv_batch_logits_exact_rt2_m4",
            Self::M8 => "w4a16_gemv_batch_logits_exact_rt2_m8",
            Self::M17 => "w4a16_gemv_batch_logits_exact_rt2_m17",
            Self::M32 => "w4a16_gemv_batch_logits_exact_rt2_m32",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::M4 => "m4",
            Self::M8 => "m8",
            Self::M17 => "m17",
            Self::M32 => "m32",
        }
    }
}

/// Select the smallest exact tier that can hold `rows`. M=1 stays K1 GEMV.
pub const fn exact_lm_head_tier_for_rows(rows: u32) -> Option<ExactLmHeadTier> {
    match rows {
        2..=4 => Some(ExactLmHeadTier::M4),
        5..=8 => Some(ExactLmHeadTier::M8),
        9..=17 => Some(ExactLmHeadTier::M17),
        18..=32 => Some(ExactLmHeadTier::M32),
        _ => None,
    }
}

/// Qualified exact launch or independent row-major K1 fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactLmHeadRoute {
    Exact(ExactLmHeadTier),
    SerialK1(ExactLmHeadTier),
}

impl ExactLmHeadRoute {
    pub const fn tier(self) -> ExactLmHeadTier {
        match self {
            Self::Exact(tier) | Self::SerialK1(tier) => tier,
        }
    }

    pub const fn provenance(self) -> &'static str {
        match self {
            Self::Exact(_) => "exact_dynamic_m_nvfp4",
            Self::SerialK1(_) => "serial_k1_row_major_nvfp4",
        }
    }
}

/// Pure policy decision. Presence refers only to the selected tier.
pub const fn exact_lm_head_route_for_rows(
    rows: u32,
    selected_kernel_present: bool,
) -> Option<ExactLmHeadRoute> {
    let Some(tier) = exact_lm_head_tier_for_rows(rows) else {
        return None;
    };
    Some(if selected_kernel_present {
        ExactLmHeadRoute::Exact(tier)
    } else {
        ExactLmHeadRoute::SerialK1(tier)
    })
}
