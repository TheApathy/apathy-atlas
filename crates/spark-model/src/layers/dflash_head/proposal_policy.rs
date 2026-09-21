// SPDX-License-Identifier: AGPL-3.0-only

//! Immutable host policy for the DFlash forward path.

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ProposalPolicy {
    pub force_no_ctx: bool,
    pub force_ctx_used: Option<usize>,
    pub debug_dump: bool,
    pub zero_late: usize,
    pub hf_override: Option<String>,
    pub force_pattern: bool,
    pub force_noise_pattern: bool,
    pub debug_dump_full: bool,
    pub dump_min_pos: usize,
    pub fc_layernorm: bool,
    pub mask_override: Option<u32>,
    pub denoise_steps: usize,
    pub denoise_margin: f32,
    pub denoise_freeze: bool,
    pub dump_all_layers: bool,
    pub lm_head_nvfp4: bool,
    pub margin_gate: f32,
    pub adaptive_gamma: bool,
    pub tps_router: bool,
    pub adaptive_min: usize,
    pub adaptive_slack: usize,
    pub adaptive_max: usize,
    pub adaptive_probe_interval: usize,
    pub truncate_mode: bool,
    pub climbdrop_mode: bool,
    pub climbdrop_floor: usize,
    pub climbdrop_cap: Option<usize>,
    pub router_widths: Option<String>,
    pub router_probe_interval: u64,
    pub skip_decode_append: bool,
    pub pld: bool,
    pub pld_ngram: usize,
    pub retr_wide: usize,
    pub portfolio: bool,
    pub corroborate: bool,
    pub corroborate_min: usize,
    pub recycle: bool,
    pub recycle_max_accept: usize,
    pub accept_fallback: bool,
    pub fallback_thresh: usize,
    pub fallback_cooldown: usize,
    pub caterpillar: bool,
    pub branch_margin: f32,
    pub free_slots: usize,
    pub free_slots_tail: usize,
    pub branch: bool,
    pub ddtree: bool,
    pub ddtree_nonflat: bool,
    pub ddtree_chain_only: bool,
    pub prefill_pipe: bool,
}

impl ProposalPolicy {
    pub fn from_env() -> Self {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    fn from_lookup(mut get: impl FnMut(&str) -> Option<String>) -> Self {
        let force_no_ctx = is_one(get("ATLAS_DFLASH_DEBUG_CTX_OFF"));
        let force_ctx_used = parse(get("ATLAS_DFLASH_DEBUG_CTX_USED"));
        let debug_dump = is_one(get("ATLAS_DFLASH_DEBUG_DUMP"));
        let zero_late = parse(get("ATLAS_DFLASH_ZERO_LATE_LAYERS")).unwrap_or(0);
        let hf_override = get("ATLAS_DFLASH_HF_OVERRIDE");
        let force_pattern = is_one(get("ATLAS_DFLASH_DEBUG_FORCE_PATTERN"));
        let force_noise_pattern = is_one(get("ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN"));
        let debug_dump_full = is_one(get("ATLAS_DFLASH_DEBUG_DUMP_FULL"));
        let dump_min_pos = parse(get("ATLAS_DFLASH_DUMP_MIN_POS")).unwrap_or(0);
        let fc_layernorm = is_one(get("ATLAS_DFLASH_FC_LAYERNORM"));
        let mask_override = parse(get("ATLAS_DFLASH_MASK_OVERRIDE"));
        let denoise_steps = parse(get("ATLAS_DFLASH_DENOISE_STEPS"))
            .unwrap_or(1usize)
            .clamp(1, 8);
        let denoise_margin = parse(get("ATLAS_DFLASH_DENOISE_MARGIN")).unwrap_or(1.0f32);
        let denoise_freeze = !is_zero(get("ATLAS_DFLASH_DENOISE_FREEZE"));
        let dump_all_layers = is_one(get("ATLAS_DFLASH_DEBUG_DUMP_ALL_LAYERS"));
        let lm_head_nvfp4 = is_one(get("ATLAS_DFLASH_LM_HEAD_NVFP4"));
        let margin_gate = parse(get("ATLAS_DFLASH_MARGIN_GATE")).unwrap_or(0.0f32);
        let adaptive_gamma = is_one(get("ATLAS_DFLASH_ADAPTIVE_GAMMA"));
        let tps_router = is_one(get("ATLAS_DFLASH_TPS_ROUTER"));
        let adaptive_min = parse(get("ATLAS_DFLASH_ADAPTIVE_MIN")).unwrap_or(4);
        let adaptive_slack = parse(get("ATLAS_DFLASH_ADAPTIVE_SLACK")).unwrap_or(2);
        let adaptive_max = parse(get("ATLAS_DFLASH_ADAPTIVE_MAX")).unwrap_or(0);
        let adaptive_probe_interval =
            parse(get("ATLAS_DFLASH_ADAPTIVE_PROBE_INTERVAL")).unwrap_or(0);
        let truncate_mode = get("ATLAS_DFLASH_ADAPTIVE_MODE").as_deref() == Some("truncate");
        let climbdrop_mode = get("ATLAS_DFLASH_TPS_ROUTER_MODE").as_deref() == Some("climbdrop");
        let climbdrop_floor = parse(get("ATLAS_DFLASH_TPS_ROUTER_FLOOR")).unwrap_or(4);
        let climbdrop_cap = parse(get("ATLAS_DFLASH_TPS_ROUTER_MAX"));
        let router_widths = get("ATLAS_DFLASH_TPS_ROUTER_WIDTHS");
        let router_probe_interval =
            parse(get("ATLAS_DFLASH_TPS_ROUTER_PROBE_INTERVAL")).unwrap_or(16);
        let skip_decode_append = is_one(get("ATLAS_DFLASH_DEBUG_NO_DECODE_APPEND"));
        let pld = is_one(get("ATLAS_DFLASH_PLD"));
        let pld_ngram = parse(get("ATLAS_PLD_NGRAM")).unwrap_or(5);
        let retr_wide = parse(get("ATLAS_DFLASH_RETR_WIDE")).unwrap_or(0);
        let portfolio = is_one(get("ATLAS_DFLASH_PORTFOLIO"));
        let corroborate = is_one(get("ATLAS_DFLASH_SAM_CORROBORATE"));
        let corroborate_min = parse(get("ATLAS_DFLASH_SAM_CORROBORATE_MIN"))
            .unwrap_or(4usize)
            .max(1);
        let recycle = is_one(get("ATLAS_DFLASH_RECYCLE"));
        let recycle_max_accept = parse(get("ATLAS_DFLASH_RECYCLE_MAX_ACCEPT")).unwrap_or(1);
        let accept_fallback = is_one(get("ATLAS_DFLASH_ACCEPT_FALLBACK"));
        let fallback_thresh = parse(get("ATLAS_DFLASH_FALLBACK_THRESH")).unwrap_or(6);
        let fallback_cooldown = parse(get("ATLAS_DFLASH_FALLBACK_COOLDOWN"))
            .unwrap_or(8usize)
            .max(1);
        let caterpillar = is_one(get("ATLAS_DFLASH_CATERPILLAR"));
        let branch_margin = parse(get("ATLAS_DFLASH_BRANCH_MARGIN")).unwrap_or(2.0);
        let free_slots = parse(get("ATLAS_DFLASH_FREE_SLOTS")).unwrap_or(0);
        let free_slots_tail = parse(get("ATLAS_DFLASH_FREE_SLOTS_TAIL")).unwrap_or(4);
        let branch = is_one(get("ATLAS_DFLASH_BRANCH"));
        let ddtree = get("ATLAS_DFLASH_METHOD").as_deref() == Some("ddtree");
        let ddtree_nonflat = is_one(get("ATLAS_DDTREE_NONFLAT"));
        let ddtree_chain_only = is_one(get("ATLAS_DDTREE_CHAIN_ONLY"));
        let prefill_pipe = is_one(get("ATLAS_DFLASH_PREFILL_PIPE"));
        Self {
            force_no_ctx,
            force_ctx_used,
            debug_dump,
            zero_late,
            hf_override,
            force_pattern,
            force_noise_pattern,
            debug_dump_full,
            dump_min_pos,
            fc_layernorm,
            mask_override,
            denoise_steps,
            denoise_margin,
            denoise_freeze,
            dump_all_layers,
            lm_head_nvfp4,
            margin_gate,
            adaptive_gamma,
            tps_router,
            adaptive_min,
            adaptive_slack,
            adaptive_max,
            adaptive_probe_interval,
            truncate_mode,
            climbdrop_mode,
            climbdrop_floor,
            climbdrop_cap,
            router_widths,
            router_probe_interval,
            skip_decode_append,
            pld,
            pld_ngram,
            retr_wide,
            portfolio,
            corroborate,
            corroborate_min,
            recycle,
            recycle_max_accept,
            accept_fallback,
            fallback_thresh,
            fallback_cooldown,
            caterpillar,
            branch_margin,
            free_slots,
            free_slots_tail,
            branch,
            ddtree,
            ddtree_nonflat,
            ddtree_chain_only,
            prefill_pipe,
        }
    }
}

/// The pipelined W4A16 kernel is a byte-exact baseline shadow only for a
/// complete 64-row reduction stage. Keep it on the bulk prompt-ingest side of
/// the M=32 boundary; decode-sized matrices retain their tuned transposed
/// kernels even when the prefill gate is enabled.
pub(super) fn use_prefill_pipe(
    requested: bool,
    has_kernel: bool,
    m: usize,
    k: usize,
) -> Result<bool, &'static str> {
    let eligible = requested && m > 32 && k.is_multiple_of(64);
    if eligible && !has_kernel {
        return Err(
            "ATLAS_DFLASH_PREFILL_PIPE=1 requires w4a16_gemm_pipe for eligible bulk ingest",
        );
    }
    Ok(eligible)
}

/// The V3 K/V context projections have identical shapes and share their input,
/// so the bulk prefill policy requires the exact dual-output pipe as well as
/// the single-output pipe used by FC ingestion.
pub(super) fn use_prefill_dual_pipe(
    requested: bool,
    has_kernel: bool,
    m: usize,
    k: usize,
) -> Result<bool, &'static str> {
    let eligible = requested && m > 32 && k.is_multiple_of(64);
    if eligible && !has_kernel {
        return Err(
            "ATLAS_DFLASH_PREFILL_PIPE=1 requires w4a16_gemm_pipe_dual for eligible K/V bulk ingest",
        );
    }
    Ok(eligible)
}

fn parse<T: std::str::FromStr>(value: Option<String>) -> Option<T> {
    value.and_then(|raw| raw.parse().ok())
}

fn is_one(value: Option<String>) -> bool {
    value.as_deref() == Some("1")
}

fn is_zero(value: Option<String>) -> bool {
    value.as_deref() == Some("0")
}

#[cfg(test)]
mod tests {
    use super::{ProposalPolicy, use_prefill_dual_pipe, use_prefill_pipe};
    use std::collections::HashMap;

    #[test]
    fn defaults_match_the_existing_forward_contract() {
        let policy = ProposalPolicy::from_lookup(|_| None);
        assert_eq!(policy.denoise_steps, 1);
        assert_eq!(policy.denoise_margin, 1.0);
        assert!(policy.denoise_freeze);
        assert_eq!(policy.adaptive_min, 4);
        assert_eq!(policy.adaptive_slack, 2);
        assert_eq!(policy.router_probe_interval, 16);
        assert!(!policy.truncate_mode);
        assert!(!policy.climbdrop_mode);
        assert!(!policy.pld);
        assert!(!policy.portfolio);
        assert!(!policy.accept_fallback);
        assert_eq!(policy.branch_margin, 2.0);
        assert_eq!(policy.free_slots_tail, 4);
        assert!(!policy.prefill_pipe);
    }

    #[test]
    fn explicit_values_are_parsed_once_and_bounded() {
        let values = HashMap::from([
            ("ATLAS_DFLASH_DENOISE_STEPS", "99"),
            ("ATLAS_DFLASH_DENOISE_FREEZE", "0"),
            ("ATLAS_DFLASH_ADAPTIVE_MODE", "truncate"),
            ("ATLAS_DFLASH_TPS_ROUTER_MODE", "climbdrop"),
            ("ATLAS_DFLASH_TPS_ROUTER_WIDTHS", "4,8,15"),
            ("ATLAS_DFLASH_MASK_OVERRIDE", "248319"),
            ("ATLAS_DFLASH_PREFILL_PIPE", "1"),
        ]);
        let policy = ProposalPolicy::from_lookup(|name| values.get(name).map(ToString::to_string));
        assert_eq!(policy.denoise_steps, 8);
        assert!(!policy.denoise_freeze);
        assert!(policy.truncate_mode);
        assert!(policy.climbdrop_mode);
        assert_eq!(policy.router_widths.as_deref(), Some("4,8,15"));
        assert_eq!(policy.mask_override, Some(248319));
        assert!(policy.prefill_pipe);
    }

    #[test]
    fn malformed_values_retain_legacy_defaults() {
        let policy = ProposalPolicy::from_lookup(|_| Some("invalid".into()));
        assert_eq!(policy.denoise_steps, 1);
        assert_eq!(policy.margin_gate, 0.0);
        assert_eq!(policy.adaptive_max, 0);
        assert_eq!(policy.router_probe_interval, 16);
    }

    #[test]
    fn prefill_pipe_is_bulk_only_and_requires_an_exact_reduction_stage() {
        assert_eq!(use_prefill_pipe(false, true, 4096, 2048), Ok(false));
        assert_eq!(use_prefill_pipe(true, true, 32, 2048), Ok(false));
        assert_eq!(use_prefill_pipe(true, true, 33, 2048), Ok(true));
        assert_eq!(use_prefill_pipe(true, true, 4096, 2047), Ok(false));
    }

    #[test]
    fn v3_checkpoint_prefill_geometry_selects_both_pipe_routes() {
        // V3 captures five 8192-wide target layers before its FC projection,
        // then projects the 5120-wide drafter hidden state into every layer's
        // context K/V cache. The promoted 4096-token window is bulk-eligible.
        assert_eq!(use_prefill_pipe(true, true, 4096, 5 * 8192), Ok(true));
        assert_eq!(use_prefill_dual_pipe(true, true, 4096, 5120), Ok(true));
    }

    #[test]
    fn requested_bulk_prefill_fails_closed_without_the_pipe_kernel() {
        assert!(use_prefill_pipe(true, false, 4096, 2048).is_err());
        assert_eq!(use_prefill_pipe(false, false, 4096, 2048), Ok(false));
        assert_eq!(use_prefill_pipe(true, false, 32, 2048), Ok(false));
    }

    #[test]
    fn requested_bulk_kv_fails_closed_without_the_dual_pipe_kernel() {
        assert_eq!(use_prefill_dual_pipe(true, true, 4096, 2048), Ok(true));
        assert!(use_prefill_dual_pipe(true, false, 4096, 2048).is_err());
        assert_eq!(use_prefill_dual_pipe(false, false, 4096, 2048), Ok(false));
        assert_eq!(use_prefill_dual_pipe(true, false, 32, 2048), Ok(false));
        assert_eq!(use_prefill_dual_pipe(true, false, 4096, 2047), Ok(false));
    }
}
