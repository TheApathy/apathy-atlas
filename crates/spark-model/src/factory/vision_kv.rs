// SPDX-License-Identifier: AGPL-3.0-only

//! Actual Vision-Exp KV admission. The generic BF16/Turbo decode kernels do
//! not implement V4's 576-wide cache, sink, and compressed history; its old
//! NVFP4 MLA kernel also lacks the hybrid-attention contract. FP8 alone has
//! the required V4 decoder. This guard does not change text-model admission.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::kv_cache::KvCacheDtype;

/// Validate CLI intent and the selected target's default before GPU/weight
/// initialization. No unsupported request may be silently rewritten to FP8.
pub fn validate_deepseek_vision_kv_request(
    config: &ModelConfig,
    requested: &str,
    target_default: &str,
    high_precision_layers: &str,
) -> Result<()> {
    if config.deepseek_vision.is_none() {
        return Ok(());
    }
    ensure!(
        requested == "fp8",
        "DeepSeek Vision requires --kv-cache-dtype fp8; requested '{requested}' lacks the V4 hybrid-attention decode contract"
    );
    ensure!(
        target_default.is_empty() || target_default == "fp8",
        "DeepSeek Vision requires FP8 KV, but MODEL.toml would resolve the requested fp8 to '{target_default}'; fix the kernel target default"
    );
    ensure!(
        high_precision_layers.parse::<usize>().ok() == Some(0),
        "DeepSeek Vision requires --kv-high-precision-layers 0; BF16 boundary layers do not implement V4 hybrid-attention decode (got '{high_precision_layers}')"
    );
    Ok(())
}

/// Validate resolved cache types at the serving and model-factory boundaries.
/// An empty vector means uniform `dtype`; otherwise every attention layer must
/// be present and FP8. The base dtype remains part of the cache allocation ABI.
pub fn validate_deepseek_vision_kv(
    config: &ModelConfig,
    dtype: KvCacheDtype,
    layer_dtypes: &[KvCacheDtype],
) -> Result<()> {
    if config.deepseek_vision.is_none() {
        return Ok(());
    }
    ensure!(
        dtype == KvCacheDtype::Fp8,
        "DeepSeek Vision requires FP8 KV on every attention layer; effective base dtype is {dtype}"
    );
    ensure!(
        layer_dtypes.is_empty() || layer_dtypes.len() == config.num_attention_layers(),
        "DeepSeek Vision per-layer KV dtype count must match all attention layers"
    );
    for (index, dtype) in layer_dtypes.iter().enumerate() {
        ensure!(
            *dtype == KvCacheDtype::Fp8,
            "DeepSeek Vision requires FP8 KV on every attention layer; layer {index} is {dtype}; disable BF16 boundary layers"
        );
    }
    Ok(())
}
