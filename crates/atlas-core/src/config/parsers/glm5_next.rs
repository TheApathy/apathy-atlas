// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3-Flash nested configuration parser.

use anyhow::{Context, Result, bail};
use serde_json::Value;

use super::super::{
    Glm5NextConfig, Glm5NextIndexerType, Glm5NextMlpType, LayerType, ModelConfig, finalize_config,
};
use super::parse_vision_config;

mod validation;
use validation::{GlmQuantization, require_geometry, validate_quantization_config};

fn required<'a>(raw: &'a Value, key: &str) -> Result<&'a Value> {
    raw.get(key)
        .with_context(|| format!("glm5_next text_config missing {key}"))
}

fn required_usize(raw: &Value, key: &str) -> Result<usize> {
    let value = required(raw, key)?
        .as_u64()
        .with_context(|| format!("glm5_next {key} must be an unsigned integer"))?;
    usize::try_from(value).with_context(|| format!("glm5_next {key} overflows usize"))
}

fn required_f32(raw: &Value, key: &str) -> Result<f32> {
    let value = required(raw, key)?
        .as_f64()
        .with_context(|| format!("glm5_next {key} must be a number"))?;
    let value = value as f32;
    if !value.is_finite() {
        bail!("glm5_next {key} must be finite");
    }
    Ok(value)
}

fn required_bool(raw: &Value, key: &str) -> Result<bool> {
    required(raw, key)?
        .as_bool()
        .with_context(|| format!("glm5_next {key} must be a boolean"))
}

fn required_str<'a>(raw: &'a Value, key: &str) -> Result<&'a str> {
    required(raw, key)?
        .as_str()
        .with_context(|| format!("glm5_next {key} must be a string"))
}

fn parse_string_array<T>(
    raw: &Value,
    key: &str,
    mut parse: impl FnMut(&str) -> Option<T>,
) -> Result<Vec<T>> {
    required(raw, key)?
        .as_array()
        .with_context(|| format!("glm5_next {key} must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let name = value
                .as_str()
                .with_context(|| format!("glm5_next {key}[{index}] must be a string"))?;
            parse(name)
                .with_context(|| format!("glm5_next {key}[{index}] has unsupported value {name}"))
        })
        .collect()
}

fn parse_eos_ids(text: &Value) -> Result<Vec<u32>> {
    let values = required(text, "eos_token_id")?
        .as_array()
        .context("glm5_next eos_token_id must be an array")?;
    if values.is_empty() {
        bail!("glm5_next eos_token_id must not be empty");
    }
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let id = value.as_u64().with_context(|| {
                format!("glm5_next eos_token_id[{index}] must be an unsigned integer")
            })?;
            u32::try_from(id)
                .with_context(|| format!("glm5_next eos_token_id[{index}] overflows u32"))
        })
        .collect()
}

pub(crate) fn parse_glm5_next(raw: &Value) -> Result<ModelConfig> {
    let quantization = validate_quantization_config(raw)?;
    let text = required(raw, "text_config")?;
    if required_str(text, "model_type")? != "glm5_next_text" {
        bail!("glm5_next text_config.model_type must be glm5_next_text");
    }

    let eos_token_ids = parse_eos_ids(text)?;
    let mut deserializable = text.clone();
    let object = deserializable
        .as_object_mut()
        .context("glm5_next text_config must be an object")?;
    object.insert("eos_token_id".to_string(), Value::from(eos_token_ids[0]));
    // Atlas uses a different enum spelling for the sparse-MLA layers. Parse the
    // official schedule explicitly below instead of asking serde to guess.
    object.remove("layer_types");
    let mut config: ModelConfig =
        serde_json::from_value(deserializable).context("failed to parse glm5_next text_config")?;

    config.model_type = "glm5_next".to_string();
    config.nested_config = true;
    config.weight_prefix = "model.language_model".to_string();
    config.attn_gated = false;
    config.head_dim = config
        .qk_nope_head_dim
        .checked_add(config.qk_rope_head_dim)
        .context("glm5_next attention head dimension overflow")?;
    config.partial_rotary_factor = if config.head_dim == 0 {
        0.0
    } else {
        config.qk_rope_head_dim as f64 / config.head_dim as f64
    };

    config.layer_types = parse_string_array(text, "layer_types", |name| match name {
        "linear_attention" => Some(LayerType::LinearAttention),
        "deepseek_sparse_attention" => Some(LayerType::FullAttention),
        _ => None,
    })?;
    let mlp_layer_types = parse_string_array(text, "mlp_layer_types", |name| match name {
        "dense" => Some(Glm5NextMlpType::Dense),
        "sparse" => Some(Glm5NextMlpType::Sparse),
        _ => None,
    })?;
    let indexer_types = parse_string_array(text, "indexer_types", |name| match name {
        "full" => Some(Glm5NextIndexerType::Full),
        "shared" => Some(Glm5NextIndexerType::Shared),
        _ => None,
    })?;
    if mlp_layer_types.len() != config.num_hidden_layers
        || indexer_types.len() != config.num_hidden_layers
    {
        bail!("glm5_next layer schedules must match num_hidden_layers");
    }

    config.num_experts = required_usize(text, "n_routed_experts")?;
    config.n_routed_experts = config.num_experts;
    let shared_experts = required_usize(text, "n_shared_experts")?;
    config.shared_expert_intermediate_size = shared_experts
        .checked_mul(config.moe_intermediate_size)
        .context("glm5_next shared expert width overflow")?;
    config.use_routing_bias = required_str(text, "topk_method")? == "noaux_tc";

    let linear = required(text, "linear_attn_config")?;
    config.linear_num_key_heads = required_usize(linear, "num_heads")?;
    config.linear_num_value_heads = config.linear_num_key_heads;
    config.linear_key_head_dim = required_usize(linear, "head_dim")?;
    config.linear_value_head_dim = config.linear_key_head_dim;
    config.linear_conv_kernel_dim = required_usize(linear, "short_conv_kernel_size")?;

    let nextn = required_usize(text, "num_nextn_predict_layers")?;
    config.num_mtp_modules = nextn;
    config.mtp_transformer_layers = usize::from(nextn > 0);
    config.mtp_num_hidden_layers = nextn;

    let index_topk = required_usize(text, "index_topk")?;
    let index_kpool = required_usize(text, "index_kpool")?;
    if index_kpool == 0 || !index_topk.is_multiple_of(index_kpool) {
        bail!("glm5_next index_topk must be divisible by nonzero index_kpool");
    }
    let glm = Glm5NextConfig {
        mlp_layer_types,
        indexer_types,
        eos_token_ids,
        index_head_dim: required_usize(text, "index_head_dim")?,
        index_n_heads: required_usize(text, "index_n_heads")?,
        index_topk,
        index_kpool,
        index_kpool_always_select_tail: required_bool(text, "index_kpool_always_select_tail")?,
        index_kpool_compress: required_bool(text, "index_kpool_compress")?,
        index_share_for_mtp_iteration: required_bool(text, "index_share_for_mtp_iteration")?,
        indexer_rope_interleave: required_bool(text, "indexer_rope_interleave")?,
        linear_lower_bound: required_f32(linear, "gate_lower_bound")?,
        swiglu_limit: required_f32(text, "swiglu_limit")?,
        mla_use_nope: required_bool(text, "mla_use_nope")?,
    };
    require_geometry(
        &config,
        &glm,
        raw,
        text,
        linear,
        required_bool(text, "mhc")?,
    )?;
    config.glm5_next = Some(glm);
    config.vision = parse_vision_config(raw);
    if let Some(vision) = &config.vision {
        let exact = vision.model_type == "glm5_next_vision"
            && vision.depth == 24
            && vision.hidden_size == 1024
            && vision.num_heads == 16
            && vision.patch_size == 14
            && vision.temporal_patch_size == 2
            && vision.spatial_merge_size == 2
            && vision.intermediate_size == 4096
            && vision.out_hidden_size == 4096
            && vision.in_channels == 3
            && vision.image_size == 448
            && vision.projection_intermediate_size == 10240
            && (vision.rms_norm_eps - 1e-5).abs() < f64::EPSILON
            && vision.swiglu_limit == Some(10.0)
            && vision.deepstack_visual_indexes.is_empty()
            && vision.image_pad_token_id == 154854
            && vision.video_pad_token_id == 154855;
        if !exact {
            bail!("unsupported glm5_next vision geometry");
        }
    }
    finalize_config(&mut config, raw)?;
    if quantization != GlmQuantization::None && config.quantization_config.is_none() {
        bail!("glm5_next quantization metadata was not preserved");
    }
    if let Some(quant) = &mut config.quantization_config {
        let expected = match quantization {
            GlmQuantization::None => {
                bail!("glm5_next parsed quantization metadata without a validated source")
            }
            GlmQuantization::Fp8 => "fp8",
            GlmQuantization::Exl3 => "exl3",
        };
        if quant.quant_method != expected {
            bail!(
                "glm5_next quantization method changed during parsing: expected {expected}, got {}",
                quant.quant_method,
            );
        }
        let aliases: Vec<_> = quant
            .ignore_modules
            .iter()
            .filter_map(|path| path.strip_prefix("model."))
            .map(|path| format!("model.language_model.{path}"))
            .collect();
        for alias in aliases {
            if !quant.ignore_modules.contains(&alias) {
                quant.ignore_modules.push(alias);
            }
        }
    }
    Ok(config)
}

#[cfg(test)]
#[path = "glm5_next_tests.rs"]
mod tests;
