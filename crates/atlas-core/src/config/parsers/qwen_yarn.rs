// SPDX-License-Identifier: AGPL-3.0-only

//! Strict Qwen nested RoPE and static-YaRN parsing.

use anyhow::{Context, Result, ensure};

use super::super::ModelConfig;

pub(crate) fn parse_qwen_rope_parameters(
    config: &mut ModelConfig,
    text_config: &serde_json::Value,
    top_model_type: &str,
) -> Result<()> {
    let Some(rope) = text_config.get("rope_parameters") else {
        return Ok(());
    };
    ensure!(rope.is_object(), "Qwen rope_parameters must be an object");
    if let Some(value) = rope.get("rope_theta") {
        let theta = value
            .as_f64()
            .context("Qwen nested rope_theta must be numeric")?;
        reject_conflict(text_config, "rope_theta", theta)?;
        config.rope_theta = theta;
    }
    if let Some(value) = rope.get("partial_rotary_factor") {
        let partial = value
            .as_f64()
            .context("Qwen nested partial_rotary_factor must be numeric")?;
        reject_conflict(text_config, "partial_rotary_factor", partial)?;
        config.partial_rotary_factor = partial;
    }

    let rope_type = match rope.get("rope_type") {
        Some(value) => Some(value.as_str().context("Qwen rope_type must be a string")?),
        None => None,
    };
    ensure!(
        matches!(rope_type, None | Some("default") | Some("yarn")),
        "unsupported Qwen rope_type {rope_type:?}"
    );
    if rope_type == Some("yarn") {
        parse_yarn(config, rope)?;
    }

    if let Some(value) = rope.get("mrope_section") {
        let section = value
            .as_array()
            .context("Qwen mrope_section must be an array")?;
        ensure!(
            section.len() == 3,
            "Qwen mrope_section must have three entries"
        );
        config.mrope_section = [
            parse_usize(&section[0], "mrope_section[0]")?,
            parse_usize(&section[1], "mrope_section[1]")?,
            parse_usize(&section[2], "mrope_section[2]")?,
        ];
    }
    config.mrope_interleaved = match rope.get("mrope_interleaved") {
        Some(value) => value
            .as_bool()
            .context("Qwen mrope_interleaved must be boolean")?,
        None => false,
    };

    validate_qwen38_yarn(config, top_model_type)?;
    let is_moe = matches!(top_model_type, "qwen3_5_moe" | "qwen3_vl_moe");
    if is_moe && config.mrope_interleaved && config.mrope_section.iter().sum::<usize>() > 0 {
        config.model_type = "qwen3_6_moe".to_string();
    }
    Ok(())
}

fn reject_conflict(text: &serde_json::Value, key: &str, nested: f64) -> Result<()> {
    if let Some(value) = text.get(key) {
        let top = value
            .as_f64()
            .with_context(|| format!("Qwen top-level {key} must be numeric"))?;
        ensure!(
            (top - nested).abs() <= f64::EPSILON,
            "conflicting Qwen {key} values"
        );
    }
    Ok(())
}

fn parse_yarn(config: &mut ModelConfig, rope: &serde_json::Value) -> Result<()> {
    let factor = required_f64(rope, "factor")?;
    ensure!(
        factor.is_finite() && factor > 1.0 && factor <= f32::MAX as f64,
        "Qwen YaRN factor must be finite, representable, and > 1"
    );
    let original = rope
        .get("original_max_position_embeddings")
        .and_then(serde_json::Value::as_u64)
        .context("Qwen YaRN original_max_position_embeddings is required")?;
    let original = usize::try_from(original).context("Qwen YaRN origin is too large")?;
    ensure!(original > 0, "Qwen YaRN origin must be positive");
    let beta_fast = optional_f64(rope, "beta_fast", 32.0)?;
    let beta_slow = optional_f64(rope, "beta_slow", 1.0)?;
    ensure!(
        beta_fast.is_finite()
            && beta_slow.is_finite()
            && beta_fast <= f32::MAX as f64
            && beta_slow <= f32::MAX as f64
            && beta_fast > beta_slow
            && beta_slow > 0.0,
        "Qwen YaRN requires finite representable beta_fast > beta_slow > 0"
    );
    let derived_attention = 1.0 + 0.1 * factor.ln();
    if rope.get("attention_factor").is_some() {
        let explicit = required_f64(rope, "attention_factor")?;
        ensure!(
            explicit.is_finite() && (explicit - derived_attention).abs() <= 1e-6,
            "custom Qwen YaRN attention_factor is not supported"
        );
    }
    config.yarn_factor = factor as f32;
    config.yarn_beta_fast = beta_fast as f32;
    config.yarn_beta_slow = beta_slow as f32;
    config.yarn_original_max_position_embeddings = original;
    Ok(())
}

fn validate_qwen38_yarn(config: &ModelConfig, top_model_type: &str) -> Result<()> {
    if top_model_type != "qwen3_5"
        || config.num_experts != 0
        || config.hidden_size != 5120
        || config.yarn_factor == 0.0
    {
        return Ok(());
    }
    ensure!(
        config.yarn_factor == 4.0,
        "Qwen3.8 1M requires YaRN factor 4"
    );
    ensure!(
        config.num_hidden_layers == 64
            && config.num_attention_heads == 24
            && config.num_key_value_heads == 4
            && config.head_dim == 256,
        "Qwen3.8 1M requires L64/Q24/KV4/head_dim256"
    );
    ensure!(
        config.yarn_original_max_position_embeddings == 262_144,
        "Qwen3.8 1M requires YaRN origin 262144"
    );
    ensure!(
        config.yarn_beta_fast == 32.0 && config.yarn_beta_slow == 1.0,
        "Qwen3.8 1M requires YaRN beta_fast=32 and beta_slow=1"
    );
    ensure!(
        config.rope_theta == 10_000_000.0,
        "Qwen3.8 1M keeps theta at 10000000"
    );
    ensure!(
        config.partial_rotary_factor == 0.25,
        "Qwen3.8 1M requires partial_rotary_factor 0.25"
    );
    ensure!(
        config.mrope_interleaved && config.mrope_section == [11, 11, 10],
        "Qwen3.8 1M requires interleaved MRoPE [11,11,10]"
    );
    Ok(())
}

fn required_f64(value: &serde_json::Value, key: &str) -> Result<f64> {
    value
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .with_context(|| format!("Qwen YaRN {key} must be numeric"))
}

fn optional_f64(value: &serde_json::Value, key: &str, default: f64) -> Result<f64> {
    match value.get(key) {
        Some(_) => required_f64(value, key),
        None => Ok(default),
    }
}

fn parse_usize(value: &serde_json::Value, label: &str) -> Result<usize> {
    let value = value
        .as_u64()
        .with_context(|| format!("{label} must be u64"))?;
    usize::try_from(value).with_context(|| format!("{label} is too large"))
}
