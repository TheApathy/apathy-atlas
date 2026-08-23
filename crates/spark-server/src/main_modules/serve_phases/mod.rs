// SPDX-License-Identifier: AGPL-3.0-only

//! Helper functions extracted from `serve.rs` to keep that file under the
//! 500-LoC cap. Split across themed sub-files because the combined helper
//! body is larger than 500 LoC itself.

mod build;
mod config;
mod kernel_gate;
mod kv_cache;
mod preflight;
mod runtime;
mod sampling_audit;
mod tokenizer_runtime;
mod topology;
mod weights;

/// Resolve the construction-time speculative draft width in one unit.
///
/// DFlash owns a separate CLI width and also enables speculative target state
/// even when `--speculative` itself is absent.  Keeping this decision shared
/// prevents preflight and the runtime arena from silently sizing a DFlash-only
/// server as plain decode.
fn resolve_speculative_draft_count(
    dflash_gamma: Option<usize>,
    native_speculative: bool,
    num_drafts: usize,
) -> Option<usize> {
    dflash_gamma
        .map(|gamma| gamma.max(1))
        .or_else(|| native_speculative.then_some(num_drafts))
}

fn speculative_draft_count(args: &crate::cli::ServeArgs) -> Option<usize> {
    resolve_speculative_draft_count(
        // Before the drafter config is loaded, reserve for the largest
        // currently supported published family default (DFlash B16 -> γ15).
        // `serve` replaces this with the parsed family-specific value before
        // model construction and runtime logging.
        args.dflash.then_some(args.dflash_gamma.unwrap_or(15)),
        args.speculative || args.self_speculative || args.ngram_speculative,
        args.num_drafts,
    )
}

pub(super) use build::{
    build_high_speed_swap_config, build_model, build_prefix_cache, maybe_run_ep_worker,
    validate_head_high_speed_swap,
};
pub(super) use config::{
    apply_model_default_num_drafts, cap_vocab_size_to_tokenizer, load_model_config,
    merge_sidecar_quant_config, resolve_model_dir,
};
pub(super) use kernel_gate::check_and_exit;
pub(super) use kv_cache::{
    KvCacheConfig, PrefillBudget, resolve_kv_cache_config, resolve_prefill_budget,
};
pub(super) use preflight::{
    ReservePreflight, init_gpu_backend, init_rest_store, post_load_memory_audit, preflight_reserve,
};
pub(super) use runtime::{
    SamplingDefaults, load_eos_tokens, load_sampling_defaults, log_behavior_audit,
    log_response_store_audit, open_dump_writer, resolve_model_name, resolve_tool_call_parser,
};
pub(super) use sampling_audit::log_sampling_presets;
pub(super) use tokenizer_runtime::{TokenizerRuntime, resolve_tokenizer_runtime};
pub(super) use topology::{Topology, init_nccl_comm, resolve_topology};
pub(super) use weights::{auto_detect_weight_prefix, load_dflash_drafter, load_weight_store};

#[cfg(test)]
mod speculative_width_tests {
    use super::resolve_speculative_draft_count;

    #[test]
    fn dflash_only_uses_its_actual_draft_count() {
        assert_eq!(
            resolve_speculative_draft_count(Some(15), false, 1),
            Some(15)
        );
        assert_eq!(resolve_speculative_draft_count(Some(6), false, 3), Some(6));
    }

    #[test]
    fn dflash_width_wins_when_native_mtp_is_also_loaded() {
        assert_eq!(resolve_speculative_draft_count(Some(15), true, 1), Some(15));
    }

    #[test]
    fn native_and_plain_modes_preserve_their_existing_units() {
        assert_eq!(resolve_speculative_draft_count(None, true, 3), Some(3));
        assert_eq!(resolve_speculative_draft_count(None, false, 3), None);
    }
}
