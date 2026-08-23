// SPDX-License-Identifier: AGPL-3.0-only

//! Shared mixed-precision helpers for Qwen3.5/3.6/3.8 loaders.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::weight_map::{DenseWeight, dense, dense_auto_fp8_or_bf16};

#[inline]
fn lm_head_needs_fp8_dequant(dtype: WeightDtype) -> bool {
    dtype == WeightDtype::FP8E4M3
}

/// Load an LM head while intercepting only the FP8 mixed-precision layout.
///
/// BF16 heads and Standard-NVFP4 UInt8-packed heads retain pointer-alias
/// behavior: the latter is consumed by the existing packed LM-head path. An
/// FP8 E4M3 head, however, must be expanded to BF16 or a BF16 GEMM will read
/// twice the allocation. Unsloth's Qwen3.8 27B compressed-tensors artifact
/// uses FP32 `[vocab, 1]` per-row scales for this tensor.
pub(super) fn load_lm_head(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    for prefix in ["lm_head", "language_model.lm_head", "model.lm_head"] {
        let key = format!("{prefix}.weight");
        if !store.contains(&key) {
            continue;
        }
        let dtype = store.get(&key)?.dtype;
        return if lm_head_needs_fp8_dequant(dtype) {
            dense_auto_fp8_or_bf16(store, prefix, gpu)
        } else {
            dense(store, &key)
        };
    }

    let prefix = &config.weight_prefix;
    dense(store, &format!("{prefix}.embed_tokens.weight"))
}

#[cfg(test)]
mod tests {
    use super::lm_head_needs_fp8_dequant;
    use spark_runtime::weights::WeightDtype;

    #[test]
    fn lm_head_dequants_only_fp8() {
        assert!(lm_head_needs_fp8_dequant(WeightDtype::FP8E4M3));
        assert!(!lm_head_needs_fp8_dequant(WeightDtype::BF16));
        assert!(!lm_head_needs_fp8_dequant(WeightDtype::UInt8));
    }
}
