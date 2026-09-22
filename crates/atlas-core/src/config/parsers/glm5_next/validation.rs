// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use serde_json::Value;

use super::{required, required_bool, required_str, required_usize};
use crate::config::{Glm5NextConfig, Glm5NextIndexerType, Glm5NextMlpType, LayerType, ModelConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GlmQuantization {
    None,
    Fp8,
    Exl3,
}

fn usize_array(raw: &Value, key: &str) -> Result<Vec<usize>> {
    required(raw, key)?
        .as_array()
        .with_context(|| format!("glm5_next {key} must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_u64()
                .with_context(|| format!("glm5_next {key}[{index}] must be unsigned"))
                .and_then(|value| usize::try_from(value).context("glm5_next layer index overflow"))
        })
        .collect()
}

pub(super) fn validate_quantization_config(raw: &Value) -> Result<GlmQuantization> {
    let Some(value) = raw.get("quantization_config") else {
        return Ok(GlmQuantization::None);
    };
    let config = value
        .as_object()
        .context("glm5_next quantization_config must be an object")?;
    match config.get("quant_method").and_then(Value::as_str) {
        Some("fp8") => {
            validate_fp8_object(config, "glm5_next quantization_config")?;
            Ok(GlmQuantization::Fp8)
        }
        Some("exl3") => {
            validate_exl3_object(config)?;
            Ok(GlmQuantization::Exl3)
        }
        Some(method) => bail!("unsupported glm5_next quantization method {method}"),
        None => bail!("glm5_next quantization_config.quant_method must be a string"),
    }
}

fn validate_fp8_object(config: &serde_json::Map<String, Value>, label: &str) -> Result<()> {
    for (key, expected) in [
        ("quant_method", "fp8"),
        ("fmt", "e4m3"),
        ("activation_scheme", "dynamic"),
    ] {
        if config.get(key).and_then(Value::as_str) != Some(expected) {
            bail!("{label}.{key} must be {expected}");
        }
    }
    let block = config
        .get("weight_block_size")
        .and_then(Value::as_array)
        .with_context(|| format!("{label}.weight_block_size must be an array"))?;
    if block.len() != 2 || block.iter().any(|value| value.as_u64() != Some(128)) {
        bail!("{label}.weight_block_size must be [128, 128]");
    }
    let exclusions = config
        .get("modules_to_not_convert")
        .and_then(Value::as_array)
        .with_context(|| format!("{label}.modules_to_not_convert must be an array"))?;
    if exclusions.is_empty()
        || exclusions
            .iter()
            .any(|value| value.as_str().is_none_or(str::is_empty))
    {
        bail!("{label}.modules_to_not_convert must contain nonempty strings");
    }
    Ok(())
}

fn validate_exl3_object(config: &serde_json::Map<String, Value>) -> Result<()> {
    let label = "glm5_next quantization_config";
    for (key, expected) in [
        ("quant_method", "exl3"),
        ("version", "1.4.4"),
        ("out_scales", "always"),
        ("codebook", "mul1"),
    ] {
        if config.get(key).and_then(Value::as_str) != Some(expected) {
            bail!("{label}.{key} must be {expected}");
        }
    }
    if config.get("bits").and_then(Value::as_f64) != Some(2.05) {
        bail!("{label}.bits must be 2.05");
    }
    if config.get("head_bits").and_then(Value::as_u64) != Some(5) {
        bail!("{label}.head_bits must be 5");
    }
    let calibration = config
        .get("calibration")
        .and_then(Value::as_object)
        .with_context(|| format!("{label}.calibration must be an object"))?;
    if calibration.get("rows").and_then(Value::as_u64) != Some(250)
        || calibration.get("cols").and_then(Value::as_u64) != Some(2048)
    {
        bail!("{label}.calibration must be rows=250, cols=2048");
    }
    let original = config
        .get("original_quantization_config")
        .and_then(Value::as_object)
        .with_context(|| format!("{label}.original_quantization_config must be an object"))?;
    validate_fp8_object(original, "glm5_next EXL3 original_quantization_config")
}

pub(super) fn require_geometry(
    config: &ModelConfig,
    glm: &Glm5NextConfig,
    raw: &Value,
    text: &Value,
    linear: &Value,
    mhc: bool,
) -> Result<()> {
    let architecture = required(raw, "architectures")?
        .as_array()
        .context("glm5_next architectures must be an array")?;
    let architecture_exact = architecture.len() == 1
        && architecture[0].as_str() == Some("Glm5NextForConditionalGeneration");
    let full_layers: Vec<_> = config
        .layer_types
        .iter()
        .enumerate()
        .filter_map(|(index, kind)| (*kind == LayerType::FullAttention).then_some(index))
        .collect();
    let expected_full: Vec<_> = (3..45).step_by(4).collect();
    let expected_kda: Vec<_> = (0..45).filter(|index| index % 4 != 3).collect();
    let dense_prefix = glm
        .mlp_layer_types
        .iter()
        .take(3)
        .all(|kind| *kind == Glm5NextMlpType::Dense);
    let sparse_tail = glm
        .mlp_layer_types
        .iter()
        .skip(3)
        .all(|kind| *kind == Glm5NextMlpType::Sparse);
    let exact = architecture_exact
        && config.hidden_size == 4096
        && config.num_hidden_layers == 45
        && config.intermediate_size == 12288
        && config.vocab_size == 154880
        && config.num_attention_heads == 64
        && config.num_key_value_heads == 64
        && config.q_lora_rank == 1536
        && config.kv_lora_rank == 512
        && config.qk_nope_head_dim == 256
        && config.qk_rope_head_dim == 0
        && config.v_head_dim == 256
        && config.linear_num_key_heads == 64
        && config.linear_num_value_heads == 64
        && config.linear_key_head_dim == 128
        && config.linear_value_head_dim == 128
        && config.linear_conv_kernel_dim == 4
        && config.num_experts == 288
        && config.num_experts_per_tok == 8
        && config.moe_intermediate_size == 2048
        && config.shared_expert_intermediate_size == 2048
        && config.hc_mult == 4
        && config.hc_sinkhorn_iters == 20
        && (config.hc_eps - 1e-6).abs() < f32::EPSILON
        && config.max_position_embeddings == 1_048_576
        && config.scoring_func == "sigmoid"
        && config.norm_topk_prob
        && config.use_routing_bias
        && (config.routed_scaling_factor - 2.5).abs() < f64::EPSILON
        && config.num_mtp_modules == 1
        && !config.tie_word_embeddings
        && required_usize(text, "head_dim")? == 0
        && required_usize(text, "qk_head_dim")? == 256
        && required_usize(text, "first_k_dense_replace")? == 3
        && required_usize(text, "n_group")? == 1
        && required_usize(text, "topk_group")? == 1
        && required_usize(text, "pad_token_id")? == 154820
        && required_str(text, "hidden_act")? == "silu"
        && required_str(text, "moe_router_dtype")? == "float32"
        && !required_bool(text, "attention_bias")?
        && usize_array(linear, "kda_layers")? == expected_kda
        && usize_array(linear, "full_attn_layers")? == expected_full
        && glm.eos_token_ids == [154820, 154827, 154829]
        && glm.index_head_dim == 128
        && glm.index_n_heads == 32
        && glm.index_topk == 2048
        && glm.index_kpool == 4
        && glm.index_kpool_always_select_tail
        && glm.index_kpool_compress
        && glm.index_share_for_mtp_iteration
        && glm.indexer_rope_interleave
        && (glm.linear_lower_bound + 5.0).abs() < f32::EPSILON
        && (glm.swiglu_limit - 10.0).abs() < f32::EPSILON
        && glm.mla_use_nope
        && mhc
        && full_layers == expected_full
        && dense_prefix
        && sparse_tail
        && glm
            .indexer_types
            .iter()
            .all(|kind| *kind == Glm5NextIndexerType::Full);
    if !exact {
        bail!("unsupported glm5_next geometry; Atlas currently accepts exact GLM-5.3-Flash only");
    }
    Ok(())
}
