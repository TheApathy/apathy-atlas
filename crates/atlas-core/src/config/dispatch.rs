// SPDX-License-Identifier: AGPL-3.0-only

//! Top-level model-type dispatch for [`super::parse_config`]. Split out of
//! `config.rs` for file-size budget — handles the JSON `model_type` field
//! and routes to the appropriate parser sub-module.

#![allow(unused_imports)]

use anyhow::{Context, Result};

use super::parsers::parse_qwen_rope_parameters;
use super::{
    LayerType, ModelConfig, default_conv_kernel, default_rms_eps, finalize_config,
    parse_gemma4_params, parse_glm5_next, parse_minimax_m2, parse_mistral_params, parse_quantization_config,
    parse_vision_config, validate_config,
};

pub fn parse_config(json: &str) -> Result<ModelConfig> {
    // First, probe the top-level model_type.
    let raw: serde_json::Value =
        serde_json::from_str(json).context("Invalid JSON in config.json")?;

    let top_model_type = raw
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");

    match top_model_type {
        "qwen4_exp" | "qwen3_8_flash_next" => {
            let text_config = raw
                .get("text_config")
                .context("qwen4_exp config missing text_config")?;
            if top_model_type == "qwen3_8_flash_next" {
                let nested_model_type = text_config
                    .get("model_type")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                if nested_model_type != "qwen3_8_flash_next_text" {
                    anyhow::bail!(
                        "qwen3_8_flash_next requires text_config.model_type=qwen3_8_flash_next_text, got {nested_model_type:?}"
                    );
                }
            }
            let mut config: ModelConfig = serde_json::from_value(text_config.clone())
                .context("Failed to parse qwen4_exp text_config")?;
            // Atlas implements this architecture under its canonical Qwen4
            // experimental target. Canonicalize only after the exact Mia
            // top/nested alias pair has passed the validation above.
            config.model_type = "qwen4_exp".to_string();
            config.nested_config = true;
            config.attn_gated = true;
            config.weight_prefix = "model.language_model".to_string();
            config.vision = parse_vision_config(&raw);
            config.norm_topk_prob = true;

            if let Some(rope_params) = text_config.get("rope_parameters") {
                if let Some(theta) = rope_params
                    .get("rope_theta")
                    .and_then(serde_json::Value::as_f64)
                {
                    config.rope_theta = theta;
                }
                if let Some(factor) = rope_params
                    .get("partial_rotary_factor")
                    .and_then(serde_json::Value::as_f64)
                {
                    config.partial_rotary_factor = factor;
                }
                if let Some(ms) = rope_params.get("mrope_section").and_then(|v| v.as_array())
                    && ms.len() == 3
                {
                    config.mrope_section = [
                        ms[0].as_u64().unwrap_or(0) as usize,
                        ms[1].as_u64().unwrap_or(0) as usize,
                        ms[2].as_u64().unwrap_or(0) as usize,
                    ];
                }
                config.mrope_interleaved = rope_params
                    .get("mrope_interleaved")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
            }

            validate_qwen4_exp(&config)?;
            finalize_config(&mut config, &raw)?;
            Ok(config)
        }
        "qwen3_vl_moe" | "qwen3_5_moe" | "qwen3_5" => {
            let text_config = raw
                .get("text_config")
                .context("qwen3_5_moe config missing text_config")?;
            let mut config: ModelConfig = serde_json::from_value(text_config.clone())
                .context("Failed to parse text_config")?;
            // Override model_type to the top-level one (text_config has "*_text" suffix)
            config.model_type = top_model_type.to_string();
            // Weight prefix is auto-detected from store keys in main.rs after loading
            // (different quantizers use different prefixes)
            // eos_token_id from text_config
            if config.eos_token_id == 0 {
                config.eos_token_id = text_config
                    .get("eos_token_id")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as u32;
            }
            // Vocab size can also be at top level
            if config.vocab_size == 0 {
                config.vocab_size = raw
                    .get("vocab_size")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as usize;
            }
            parse_qwen_rope_parameters(&mut config, text_config, top_model_type)?;
            // Qwen3.5 MoE unconditionally normalizes top-K expert weights
            // (hardcoded in HF's Qwen3_5MoeTopKRouter, no config toggle).
            config.norm_topk_prob = true;
            // Architecture flags
            config.nested_config = true;
            config.attn_gated = top_model_type != "qwen3_vl_moe";
            // Parse vision_config for VL models. Qwen3.6 also ships a ViT
            // tower (detected via the mrope_interleaved flag set below,
            // but we don't have that until after this block, so also
            // trigger when the raw config has a `vision_config` key).
            if top_model_type == "qwen3_vl_moe" || raw.get("vision_config").is_some() {
                config.vision = parse_vision_config(&raw);
            }
            // MRoPE detection: Qwen3.6 MoE sets mrope_interleaved + mrope_section
            // inside text_config.rope_parameters. When present on a MoE
            // variant, rewrite model_type to "qwen3_6_moe" so kernel-target
            // resolution picks the right directory (Qwen3.5-MoE and
            // Qwen3.6-MoE share hidden_size=2048 and would otherwise collide).
            // The backing weight loader stays in the qwen3_5 family — MoE
            // architecture is identical except for MRoPE layout and the
            // full-attention layer gate.
            //
            // Kbenkhaled's Qwen3.5-27B-NVFP4 is dense (top_model_type="qwen3_5",
            // no experts) but also enables MRoPE. For dense, do NOT rewrite:
            // the Qwen35 MoE weight loader would fail looking for mlp.gate.
            // The qwen3.5-27b kernel target handles MRoPE at runtime via the
            // mrope_interleaved / mrope_section flags.
            finalize_config(&mut config, &raw)?;
            Ok(config)
        }
        "nemotron_h" => {
            let mut config: ModelConfig =
                serde_json::from_str(json).context("Failed to parse nemotron_h config.json")?;
            // Map Nemotron-H field names → Atlas canonical names
            if config.num_experts == 0 && config.n_routed_experts > 0 {
                config.num_experts = config.n_routed_experts;
            }
            if config.rms_norm_eps == default_rms_eps() && config.norm_eps > 0.0 {
                config.rms_norm_eps = config.norm_eps;
            }
            if config.linear_conv_kernel_dim == default_conv_kernel() && config.conv_kernel > 0 {
                config.linear_conv_kernel_dim = config.conv_kernel;
            }
            if config.shared_expert_intermediate_size == 0
                && config.moe_shared_expert_intermediate_size > 0
            {
                config.shared_expert_intermediate_size = config.moe_shared_expert_intermediate_size;
            }
            // Architecture flags
            config.attn_gated = false;
            config.weight_prefix = "backbone".to_string();
            // Parse hybrid_override_pattern → layer_types
            if !config.hybrid_override_pattern.is_empty() && config.layer_types.is_empty() {
                config.layer_types = config
                    .hybrid_override_pattern
                    .chars()
                    .map(|c| match c {
                        'M' => LayerType::LinearAttention,
                        'E' => LayerType::Moe,
                        '*' => LayerType::FullAttention,
                        other => panic!("Unknown hybrid_override_pattern char: '{other}'"),
                    })
                    .collect();
            }
            finalize_config(&mut config, &raw)?;
            Ok(config)
        }
        "gemma4" => parse_gemma4_params(&raw),
        "glm5_next" => parse_glm5_next(&raw),
        // The earlier, flat DeepSeek-V4 family (a different release from V4.1 -- do NOT share
        // the v41 arm, see parsers/deepseek_v41.rs). Ported alongside V4.1's parser since both
        // share parsers/deepseek_v4.rs::parse_deepseek_family.
        "deepseek_v4" => super::parse_deepseek_v4(json),
        // DeepSeek-V4.1 nests its text config under `text_config`; own parser.
        "deepseek_v41" => super::parse_deepseek_v41(json),
        "minimax_m2" => parse_minimax_m2(&raw),
        _ => {
            // Flat config (qwen3_next, etc.)
            let mut config: ModelConfig =
                serde_json::from_str(json).context("Failed to parse config.json")?;
            config.attn_gated = true;
            finalize_config(&mut config, &raw)?;
            Ok(config)
        }
    }
}

fn validate_qwen4_exp(config: &ModelConfig) -> Result<()> {
    if config.hc_count <= 1 {
        anyhow::bail!("qwen4_exp requires hc_count > 1, got {}", config.hc_count);
    }
    if config.hc_lowrank == 0 {
        anyhow::bail!("qwen4_exp requires hc_lowrank > 0");
    }
    let qsa = [
        config.indexer_n_heads,
        config.indexer_kv_heads,
        config.indexer_head_dim,
        config.indexer_budget,
        config.indexer_compress_ratio,
    ];
    if qsa.contains(&0) {
        anyhow::bail!("qwen4_exp requires complete non-zero QSA indexer geometry");
    }
    if config.indexer_kv_heads != 1 {
        anyhow::bail!(
            "qwen4_exp QSA requires indexer_kv_heads=1, got {}",
            config.indexer_kv_heads
        );
    }
    if !config
        .indexer_budget
        .is_multiple_of(config.indexer_compress_ratio)
    {
        anyhow::bail!("qwen4_exp indexer_budget must be divisible by indexer_compress_ratio");
    }
    if config.ngram_size < 2 || config.heads_per_ngram == 0 {
        anyhow::bail!("qwen4_exp PLE requires ngram_size >= 2 and heads_per_ngram > 0");
    }
    let ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram;
    if config.ple_embed_dim == 0 || !config.ple_embed_dim.is_multiple_of(ngram_heads) {
        anyhow::bail!("qwen4_exp ple_embed_dim must be divisible by the n-gram head count");
    }
    if config
        .ple_layer_ids
        .iter()
        .any(|&id| id == 0 || id > config.num_hidden_layers)
    {
        anyhow::bail!("qwen4_exp ple_layer_ids must be one-indexed decoder layers");
    }
    Ok(())
}
