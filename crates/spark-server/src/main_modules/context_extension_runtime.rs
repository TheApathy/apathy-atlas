// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};

use super::{ContextAdmissionReceipt, ContextExtension};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ContextRuntimeMode {
    pub(crate) speculative: bool,
    pub(crate) dflash: bool,
    pub(crate) self_speculative: bool,
    pub(crate) ngram_speculative: bool,
    pub(crate) high_speed_swap: bool,
    pub(crate) hss_cache_blocks_per_seq: u32,
    pub(crate) block_size: usize,
    pub(crate) max_batch_size: usize,
    pub(crate) max_seq_len: usize,
    pub(crate) config_capacity: usize,
}

impl Default for ContextRuntimeMode {
    fn default() -> Self {
        Self {
            speculative: false,
            dflash: false,
            self_speculative: false,
            ngram_speculative: false,
            high_speed_swap: false,
            hss_cache_blocks_per_seq: 0,
            block_size: 0,
            max_batch_size: 1,
            max_seq_len: 262_144,
            config_capacity: 262_144,
        }
    }
}

pub(crate) fn validate_context_extension_runtime(
    extension: Option<ContextExtension>,
    mode: ContextRuntimeMode,
) -> Result<ContextAdmissionReceipt> {
    ensure!(
        mode.max_seq_len > 0 && mode.max_seq_len <= mode.config_capacity,
        "effective max sequence length must fit the admitted config capacity"
    );
    let extended = match extension {
        Some(ContextExtension::Yarn {
            native_tokens,
            requested_tokens,
            original_tokens,
            ..
        }) => {
            ensure!(
                original_tokens > 0
                    && native_tokens >= original_tokens
                    && native_tokens <= mode.config_capacity,
                "YaRN receipt must bind its origin and captured checkpoint capacity"
            );
            ensure!(
                requested_tokens == mode.max_seq_len,
                "YaRN receipt differs from the effective max sequence length"
            );
            requested_tokens > original_tokens
        }
        Some(ContextExtension::Theta {
            native_tokens,
            requested_tokens,
            ..
        }) => {
            ensure!(
                native_tokens > 0 && native_tokens <= mode.config_capacity,
                "theta receipt carries an invalid checkpoint capacity"
            );
            ensure!(
                requested_tokens == mode.max_seq_len,
                "theta receipt differs from the effective max sequence length"
            );
            requested_tokens > native_tokens
        }
        None => false,
    };
    if extended {
        ensure!(
            mode.max_batch_size == 1,
            "extended context is qualified only for C1 decoding; set --max-batch-size 1"
        );
        ensure!(
            !(mode.speculative || mode.dflash || mode.self_speculative || mode.ngram_speculative),
            "extended context is qualified only for target-only decoding; disable DFlash and all speculative modes"
        );
        if mode.high_speed_swap {
            ensure!(
                mode.block_size > 0,
                "extended context with high-speed-swap requires a positive block size"
            );
            let resident_tokens = (mode.hss_cache_blocks_per_seq as usize)
                .checked_mul(mode.block_size)
                .context(
                    "extended-context high-speed-swap resident token capacity overflows usize",
                )?;
            ensure!(
                resident_tokens >= mode.max_seq_len,
                "extended context requires the complete prefill to remain HBM-resident; \
                 --high-speed-swap-cache-blocks-per-seq={} at --block-size={} covers only {} of {} tokens",
                mode.hss_cache_blocks_per_seq,
                mode.block_size,
                resident_tokens,
                mode.max_seq_len,
            );
        }
    }
    Ok(ContextAdmissionReceipt::mint(mode, extended))
}

#[cfg(test)]
mod serve_integration_tests {
    // TWO files, because `serve.rs` split into two and the guard went with
    // only one of them. The once-per-process prologue stayed in `serve.rs`;
    // everything model-dependent, this guard included, became
    // `serve_load::load_model`, which is the function a model swap re-runs.
    //
    // `SERVE_LOAD` is where the guard must BE, and `PROLOGUE` is where it must
    // NOT be. Reading only `serve.rs` would now pass vacuously against a file
    // containing no guard at all — a check that cannot fail — and reading only
    // `serve_load.rs` would miss a SECOND guard appearing in the prologue. The
    // guard block itself is byte-identical to the one this test has always
    // read; only the file around it changed.
    const SERVE_LOAD: &str = include_str!("serve_load.rs");
    const PROLOGUE: &str = include_str!("serve.rs");
    const START: &str = "    let context_extension =";
    const VALIDATOR_CALL: &str = "validate_context_extension_runtime(";
    const END: &str = "    if let Some(extension) = context_extension {";
    const PHASE3: &str = "spark_runtime::progress::phase(3, \"gpu init\");";
    const GPU_INIT: &str =
        "    let (gpu, free_mem) = serve_phases::init_gpu_backend(&args, &ptx_set)?;\n";
    const EXPECTED: &str = r#"
let context_extension=super::context_extension::apply_context_extension(
&mut config,args.max_seq_len,args.rope_theta_override,args.rope_yarn_factor,)?;
	let context_admission=super::context_extension::validate_context_extension_runtime(
	context_extension,super::context_extension::ContextRuntimeMode{
	speculative:args.speculative,dflash:args.dflash,self_speculative:args.self_speculative,
	ngram_speculative:args.ngram_speculative,high_speed_swap:args.high_speed_swap,
	hss_cache_blocks_per_seq:args.high_speed_swap_cache_blocks_per_seq,
	block_size:args.block_size,max_batch_size:args.max_batch_size,
	max_seq_len:args.max_seq_len,config_capacity:config.max_position_embeddings,},)?;
"#;

    fn compact(source: &str) -> String {
        source
            .chars()
            .filter(|c| !c.is_ascii_whitespace())
            .collect()
    }

    fn guard_is_exact_and_pre_gpu(source: &str) -> bool {
        let Some(start) = source.find(START) else {
            return false;
        };
        let Some(relative_end) = source[start..].find(END) else {
            return false;
        };
        let end = start + relative_end;
        compact(&source[start..end]) == compact(EXPECTED)
            && source.matches(START).count() == 1
            && source.matches(END).count() == 1
            && source[start..].matches(PHASE3).count() == 1
            && source[start..].matches(GPU_INIT.trim()).count() == 1
            && start < source.find(PHASE3).unwrap_or(0)
            && start < source.find(GPU_INIT.trim()).unwrap_or(0)
    }

    fn mutate_once(from: &str, to: &str) -> String {
        assert_eq!(
            SERVE_LOAD.matches(from).count(),
            1,
            "ambiguous hostile: {from}"
        );
        SERVE_LOAD.replacen(from, to, 1)
    }

    #[test]
    fn serve_context_guard_is_exact_and_pre_gpu() {
        assert!(guard_is_exact_and_pre_gpu(SERVE_LOAD));
    }

    /// Does the guard appear exactly once, in the per-model half and not the
    /// prologue?
    ///
    /// A FUNCTION over two strings rather than an assertion over the two
    /// `include_str!`s directly, so the rule can be handed inputs that must
    /// make it say false. An assertion that only ever sees the real files is
    /// an assertion nobody has watched fail, and a check that cannot fail is
    /// not evidence — see `the_duplication_check_can_actually_fail`.
    fn guard_appears_exactly_once(prologue: &str, per_model: &str) -> bool {
        prologue.matches(START).count() == 0
            && prologue.matches(VALIDATOR_CALL).count() == 0
            && per_model.matches(START).count() + prologue.matches(START).count() == 1
    }

    /// The guard exists exactly once, and not in the process prologue.
    ///
    /// `guard_is_exact_and_pre_gpu` proves the guard is correct where it
    /// lives. It cannot prove there is not a SECOND one. A context-extension
    /// guard that runs twice — once per process and again per load — is
    /// exactly what a later refactor reintroduces while "restoring" something,
    /// and the existing check would sail straight through it: the copy in
    /// `serve_load.rs` would still be exact and still be pre-GPU.
    ///
    /// Counted ACROSS both files rather than asserted as absence from one, so
    /// that a guard MOVED back into the prologue fails as loudly as a guard
    /// DUPLICATED into it. The `args`/`config` the prologue holds are the ones
    /// `load_model` is about to be handed, so a prologue copy would mutate
    /// `config.rope_*` and `max_position_embeddings` before the real guard
    /// ever validated them — the receipt would then describe a config that had
    /// already been extended once, which is the one thing this module exists
    /// to make impossible.
    #[test]
    fn the_guard_exists_exactly_once_and_never_in_the_process_prologue() {
        assert_eq!(
            PROLOGUE.matches(START).count(),
            0,
            "the context-extension guard is in serve.rs, which runs ONCE per \
             process. It belongs in serve_load::load_model, which runs per \
             model — a per-process guard extends the config of the first model \
             and of no other."
        );
        assert_eq!(
            PROLOGUE.matches(VALIDATOR_CALL).count(),
            0,
            "validate_context_extension_runtime is called from serve.rs. Its \
             receipt binds the RESOLVED per-model config, so a call from the \
             prologue mints one against a config no model was built from."
        );
        assert!(guard_appears_exactly_once(PROLOGUE, SERVE_LOAD));
    }

    /// The positive control for the check above.
    ///
    /// Written because tonight produced four separate findings about checks
    /// that could not fail, and a duplication check that has only ever been
    /// run against a tree with no duplicate in it is a fifth waiting to
    /// happen. Each case below is a way the guard actually goes wrong, and the
    /// rule must reject every one of them.
    #[test]
    fn the_duplication_check_can_actually_fail() {
        let guard = &SERVE_LOAD[SERVE_LOAD.find(START).unwrap()..][..START.len() + 64];
        let clean_prologue = "fn startup() { tracing::info!(\"banner\"); }";

        // The real shape, as a baseline: this is the only case that passes.
        assert!(guard_appears_exactly_once(clean_prologue, SERVE_LOAD));

        // DUPLICATED into the prologue — the case the original check missed
        // entirely, because the per-model copy is still exact and pre-GPU.
        assert!(
            !guard_appears_exactly_once(&format!("{clean_prologue}\n{guard}"), SERVE_LOAD),
            "a guard in BOTH halves must be rejected"
        );

        // MOVED back to the prologue: per-process, so it extends the first
        // model's config and no other model's.
        assert!(
            !guard_appears_exactly_once(&format!("{clean_prologue}\n{guard}"), clean_prologue),
            "a guard only in the prologue must be rejected"
        );

        // GONE from both — the vacuous pass this whole restructure exists to
        // prevent.
        assert!(
            !guard_appears_exactly_once(clean_prologue, clean_prologue),
            "no guard at all must be rejected, not silently accepted"
        );

        // The validator CALL hoisted to the prologue while the extension
        // itself stays put: a receipt minted against a config no model was
        // built from.
        assert!(
            !guard_appears_exactly_once(
                &format!("{clean_prologue}\nlet _ = {VALIDATOR_CALL});"),
                SERVE_LOAD
            ),
            "a validator call in the prologue must be rejected"
        );
    }

    #[test]
    fn serve_context_guard_rejects_wiring_and_receipt_hostiles() {
        for (from, to) in [
            ("speculative: args.speculative", "speculative: false"),
            ("dflash: args.dflash", "dflash: false"),
            (
                "self_speculative: args.self_speculative",
                "self_speculative: false",
            ),
            (
                "ngram_speculative: args.ngram_speculative",
                "ngram_speculative: false",
            ),
            (
                "high_speed_swap: args.high_speed_swap",
                "high_speed_swap: false",
            ),
            (
                "hss_cache_blocks_per_seq: args.high_speed_swap_cache_blocks_per_seq",
                "hss_cache_blocks_per_seq: u32::MAX",
            ),
            ("block_size: args.block_size", "block_size: 16"),
            ("max_batch_size: args.max_batch_size", "max_batch_size: 8"),
            (
                "max_seq_len: args.max_seq_len,\n            config_capacity:",
                "max_seq_len: 1_048_576,\n            config_capacity:",
            ),
            (
                "config_capacity: config.max_position_embeddings",
                "config_capacity: 1_048_576",
            ),
            (
                "validate_context_extension_runtime(\n        context_extension,",
                "validate_context_extension_runtime(\n        None,",
            ),
            (
                "validate_context_extension_runtime(\n        context_extension,",
                "validate_context_extension_runtime(\n        super::context_extension::apply_context_extension(&mut config, args.max_seq_len, args.rope_theta_override, args.rope_yarn_factor)?,",
            ),
        ] {
            assert!(!guard_is_exact_and_pre_gpu(&mutate_once(from, to)));
        }
    }

    #[test]
    fn serve_context_guard_rejects_missing_and_post_gpu_admission() {
        let validator = SERVE_LOAD
            .find(
                "    let context_admission = super::context_extension::validate_context_extension_runtime(",
            )
            .unwrap();
        let validator_end = SERVE_LOAD[validator..].find(END).unwrap() + validator;
        let missing = format!("{}{}", &SERVE_LOAD[..validator], &SERVE_LOAD[validator_end..]);
        assert!(!guard_is_exact_and_pre_gpu(&missing));

        let start = SERVE_LOAD.find(START).unwrap();
        let end = SERVE_LOAD[start..].find("    if let Some(ref qc)").unwrap() + start;
        let chunk = &SERVE_LOAD[start..end];
        let without = SERVE_LOAD.replacen(chunk, "", 1);
        let moved = without.replacen(GPU_INIT, &format!("{GPU_INIT}{chunk}"), 1);
        assert!(!guard_is_exact_and_pre_gpu(&moved));
    }
}
