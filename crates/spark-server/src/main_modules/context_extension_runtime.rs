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
    const SERVE: &str = include_str!("serve.rs");
    const START: &str = "    let context_extension =";
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
        assert_eq!(SERVE.matches(from).count(), 1, "ambiguous hostile: {from}");
        SERVE.replacen(from, to, 1)
    }

    #[test]
    fn serve_context_guard_is_exact_and_pre_gpu() {
        assert!(guard_is_exact_and_pre_gpu(SERVE));
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
        let validator = SERVE
            .find(
                "    let context_admission = super::context_extension::validate_context_extension_runtime(",
            )
            .unwrap();
        let validator_end = SERVE[validator..].find(END).unwrap() + validator;
        let missing = format!("{}{}", &SERVE[..validator], &SERVE[validator_end..]);
        assert!(!guard_is_exact_and_pre_gpu(&missing));

        let start = SERVE.find(START).unwrap();
        let end = SERVE[start..].find("    if let Some(ref qc)").unwrap() + start;
        let chunk = &SERVE[start..end];
        let without = SERVE.replacen(chunk, "", 1);
        let moved = without.replacen(GPU_INIT, &format!("{GPU_INIT}{chunk}"), 1);
        assert!(!guard_is_exact_and_pre_gpu(&moved));
    }
}
