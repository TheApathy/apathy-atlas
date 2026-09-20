// SPDX-License-Identifier: AGPL-3.0-only

//! Q/K/V projection + Flash Attention prefill paths.
//!
//! Wave-3 refactor split this 2619-line file into two methods-per-file
//! sub-modules. Both `paged.rs` and `cache_skip.rs` exceed the 500-LoC
//! cap because each contains a single monolithic 1000-1400 LoC method
//! (`prefill_attention_paged` / `prefill_attention_with_cache_skip`)
//! whose body interleaves 10+ sections with deep cross-section state
//! coupling. Splitting further requires extracting each section as a
//! helper method with 10-20 args — multi-day kernel-level surgery
//! beyond this wave's scope.

#[cfg(test)]
mod attention_gate_fused_tests;
mod cache_skip;
mod cache_skip_mla;
mod cache_skip_qkv;
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(super) mod flashinfer_projection;
#[cfg(test)]
mod flashinfer_projection_tests;
mod paged;
mod paged_attn;
mod paged_attn_batched;
#[cfg(test)]
mod paged_attn_br128_tests;
mod paged_attn_fp8k;
mod paged_attn_turbok;
mod paged_mla;
mod paged_oproj;
mod paged_qkv;
#[cfg(test)]
mod paged_qkv_tests;
#[cfg(test)]
mod qknorm_rope_tests;

#[cfg(any(test, all(feature = "cuda", target_os = "linux")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Qwen38FlashinferProjectionRoute {
    Disabled,
    Ineligible,
    Complete,
    Missing,
}

#[cfg(any(test, all(feature = "cuda", target_os = "linux")))]
#[allow(clippy::too_many_arguments)]
pub(super) const fn qwen38_flashinfer_projection_route(
    requested: bool,
    exact_qwen38_dense: bool,
    single_sequence: bool,
    standard_attention: bool,
    gated: bool,
    // `m_qualified` is supplied by the caller so the frozen M=2079 entry and the
    // opt-in extended range (`ATLAS_FLASHINFER_PROJ_EXTRA_M`) share one source of
    // truth with `select_qwen38_attention_projection_launch`, instead of this
    // predicate hardcoding a second copy that can drift from it.
    m_qualified: bool,
    hidden: u32,
    qg: usize,
    kv: usize,
    prepared: bool,
) -> Qwen38FlashinferProjectionRoute {
    if !requested {
        return Qwen38FlashinferProjectionRoute::Disabled;
    }
    if !exact_qwen38_dense
        || !single_sequence
        || !standard_attention
        || !gated
        || !m_qualified
        || hidden != 5_120
        || qg != 12_288
        || kv != 1_024
    {
        return Qwen38FlashinferProjectionRoute::Ineligible;
    }
    if prepared {
        Qwen38FlashinferProjectionRoute::Complete
    } else {
        Qwen38FlashinferProjectionRoute::Missing
    }
}

pub(super) fn mark_prefill_kv_dual_paged_engaged() {
    static ENGAGED: std::sync::Once = std::sync::Once::new();
    ENGAGED.call_once(|| {
        tracing::info!("ENGAGED ATLAS_PREFILL_KV_DUAL: attention_kv_paged");
    });
}

pub(super) fn mark_prefill_kv_dual_cache_skip_engaged() {
    static ENGAGED: std::sync::Once = std::sync::Once::new();
    ENGAGED.call_once(|| {
        tracing::info!("ENGAGED ATLAS_PREFILL_KV_DUAL: attention_kv_cache_skip");
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Qwen38PrefillAttnGateRoute {
    Disabled,
    Ineligible,
    Complete,
    Missing,
}

#[allow(clippy::too_many_arguments)]
pub(super) const fn qwen38_prefill_attn_gate_route(
    requested: bool,
    exact_qwen38_dense: bool,
    single_sequence: bool,
    standard_attention: bool,
    gated: bool,
    num_tokens: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    gate_stride: u32,
    has_parent_attention: bool,
    has_parent_gate: bool,
    has_fused: bool,
) -> Qwen38PrefillAttnGateRoute {
    if !requested {
        return Qwen38PrefillAttnGateRoute::Disabled;
    }
    if !exact_qwen38_dense
        || !single_sequence
        || !standard_attention
        || !gated
        || num_tokens < 64
        || num_q_heads != 24
        || num_kv_heads != 4
        || head_dim != 256
        || gate_stride != 12_288
    {
        return Qwen38PrefillAttnGateRoute::Ineligible;
    }
    if has_parent_attention && has_parent_gate && has_fused {
        Qwen38PrefillAttnGateRoute::Complete
    } else {
        Qwen38PrefillAttnGateRoute::Missing
    }
}

pub(super) fn parse_qwen38_prefill_attn_gate(
    value: Option<&str>,
) -> std::result::Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_PREFILL_ATTN_GATE_FUSED must be exactly 0 or 1"),
    }
}

pub(super) fn qwen38_prefill_attn_gate_requested() -> anyhow::Result<bool> {
    static GATE: std::sync::OnceLock<std::result::Result<bool, &'static str>> =
        std::sync::OnceLock::new();
    (*GATE.get_or_init(|| match std::env::var("ATLAS_PREFILL_ATTN_GATE_FUSED") {
        Ok(value) => parse_qwen38_prefill_attn_gate(Some(value.as_str())),
        Err(std::env::VarError::NotPresent) => parse_qwen38_prefill_attn_gate(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("ATLAS_PREFILL_ATTN_GATE_FUSED must be valid UTF-8 and exactly 0 or 1")
        }
    }))
    .map_err(anyhow::Error::msg)
}

pub(super) fn mark_qwen38_prefill_attn_gate_engaged() {
    static ENGAGED: std::sync::Once = std::sync::Once::new();
    ENGAGED.call_once(|| {
        tracing::info!("ENGAGED ATLAS_PREFILL_ATTN_GATE_FUSED: cache-skip-br64");
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Qwen38QkNormRopeRoute {
    Disabled,
    Ineligible,
    Complete,
    Missing,
}

#[allow(clippy::too_many_arguments)]
pub(super) const fn qwen38_qknorm_rope_route(
    requested: bool,
    exact_qwen38_dense: bool,
    single_sequence: bool,
    standard_attention: bool,
    gated: bool,
    q_norm: bool,
    k_norm: bool,
    no_full_norms: bool,
    no_v_norm: bool,
    interleaved_mrope: bool,
    proportional_rope: bool,
    num_tokens: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    qg_stride: u32,
    rotary_dim: u32,
    theta_valid: bool,
    parent_rope_ready: bool,
    has_kernel: bool,
) -> Qwen38QkNormRopeRoute {
    if !requested {
        return Qwen38QkNormRopeRoute::Disabled;
    }
    if !exact_qwen38_dense
        || !single_sequence
        || !standard_attention
        || !gated
        || !q_norm
        || !k_norm
        || !no_full_norms
        || !no_v_norm
        || !interleaved_mrope
        || proportional_rope
        || num_tokens == 0
        || num_q_heads != 24
        || num_kv_heads != 4
        || head_dim != 256
        || qg_stride != 12_288
        || rotary_dim != 64
        || !theta_valid
    {
        return Qwen38QkNormRopeRoute::Ineligible;
    }
    if parent_rope_ready && has_kernel {
        Qwen38QkNormRopeRoute::Complete
    } else {
        Qwen38QkNormRopeRoute::Missing
    }
}

pub(super) fn parse_qwen38_qknorm_rope(
    value: Option<&str>,
) -> std::result::Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_PREFILL_QKNORM_ROPE must be exactly 0 or 1"),
    }
}

/// Strict default-off gate for the exact Qwen3.8 C=1 prefill fusion.
pub(super) fn qwen38_qknorm_rope_requested() -> anyhow::Result<bool> {
    static GATE: std::sync::OnceLock<std::result::Result<bool, &'static str>> =
        std::sync::OnceLock::new();
    (*GATE.get_or_init(|| match std::env::var("ATLAS_PREFILL_QKNORM_ROPE") {
        Ok(value) => parse_qwen38_qknorm_rope(Some(value.as_str())),
        Err(std::env::VarError::NotPresent) => parse_qwen38_qknorm_rope(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("ATLAS_PREFILL_QKNORM_ROPE must be valid UTF-8 and exactly 0 or 1")
        }
    }))
    .map_err(anyhow::Error::msg)
}

pub(super) fn mark_qwen38_qknorm_rope_paged_engaged() {
    static ENGAGED: std::sync::Once = std::sync::Once::new();
    ENGAGED.call_once(|| {
        tracing::info!("ENGAGED ATLAS_PREFILL_QKNORM_ROPE: paged-mrope-thw");
    });
}

pub(super) fn mark_qwen38_qknorm_rope_cache_skip_engaged() {
    static ENGAGED: std::sync::Once = std::sync::Once::new();
    ENGAGED.call_once(|| {
        tracing::info!("ENGAGED ATLAS_PREFILL_QKNORM_ROPE: cache-skip-scalar");
    });
}
