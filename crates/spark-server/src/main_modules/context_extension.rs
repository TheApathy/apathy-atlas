// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed context extension applied before model construction.

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::ModelConfig;

#[path = "context_extension_admission.rs"]
mod admission;
#[path = "context_extension_runtime.rs"]
mod runtime;
pub(crate) use admission::ContextAdmissionReceipt;
pub(crate) use runtime::{ContextRuntimeMode, validate_context_extension_runtime};

pub(crate) const QWEN38_YARN_MAX_TOKENS: usize = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ContextExtension {
    Yarn {
        native_tokens: usize,
        requested_tokens: usize,
        factor: f32,
        original_tokens: usize,
        attention_factor: f32,
    },
    Theta {
        native_tokens: usize,
        requested_tokens: usize,
        rope_theta: f64,
    },
}

pub(crate) fn apply_context_extension(
    config: &mut ModelConfig,
    requested_tokens: usize,
    rope_theta_override: Option<f64>,
    rope_yarn_factor: Option<f32>,
) -> Result<Option<ContextExtension>> {
    ensure!(requested_tokens > 0, "--max-seq-len must be positive");
    ensure!(
        rope_theta_override.is_none() || rope_yarn_factor.is_none(),
        "--rope-theta-override and --rope-yarn-factor are mutually exclusive"
    );
    let native_tokens = config.max_position_embeddings;

    if config.yarn_factor > 0.0 {
        ensure!(
            rope_theta_override.is_none(),
            "a YaRN checkpoint cannot use --rope-theta-override"
        );
        ensure!(
            rope_yarn_factor.is_none(),
            "--rope-yarn-factor cannot override checkpoint YaRN"
        );
        validate_yarn(config, requested_tokens)?;
        if requested_tokens > native_tokens {
            config.max_position_embeddings = requested_tokens;
        }
        return Ok(Some(yarn_receipt(config, native_tokens, requested_tokens)));
    }

    if requested_tokens <= native_tokens {
        ensure!(
            rope_yarn_factor.is_none(),
            "--rope-yarn-factor is only valid when extending beyond native context"
        );
        if let Some(theta) = rope_theta_override {
            validate_theta(theta)?;
            config.rope_theta = theta;
        }
        return Ok(None);
    }
    ensure!(
        native_tokens > 0,
        "context extension requires checkpoint max_position_embeddings"
    );
    if requested_tokens > i32::MAX as usize {
        bail!(
            "requested context {requested_tokens} exceeds the i32 position ABI ceiling {}",
            i32::MAX
        );
    }

    if let Some(factor) = rope_yarn_factor {
        ensure!(
            is_qwen38_dense(config),
            "--rope-yarn-factor runtime extension is supported only for dense Qwen3.8's scaled MRoPE route"
        );
        let mut extended = config.clone();
        extended.yarn_factor = factor;
        extended.yarn_beta_fast = 32.0;
        extended.yarn_beta_slow = 1.0;
        extended.yarn_original_max_position_embeddings = native_tokens;
        validate_yarn(&extended, requested_tokens)?;
        extended.max_position_embeddings = requested_tokens;
        let receipt = yarn_receipt(&extended, native_tokens, requested_tokens);
        *config = extended;
        return Ok(Some(receipt));
    }

    if is_qwen38_dense(config) {
        bail!(
            "--max-seq-len {requested_tokens} exceeds Qwen3.8 native context {native_tokens}; \
             pass the official static YaRN setting --rope-yarn-factor 4"
        );
    }

    let theta = rope_theta_override.context(format!(
        "--max-seq-len {requested_tokens} exceeds checkpoint max_position_embeddings \
         {native_tokens}; pass an explicitly validated scaling mode"
    ))?;
    validate_theta(theta)?;
    config.rope_theta = theta;
    config.max_position_embeddings = requested_tokens;
    Ok(Some(ContextExtension::Theta {
        native_tokens,
        requested_tokens,
        rope_theta: theta,
    }))
}

fn yarn_receipt(
    config: &ModelConfig,
    native_tokens: usize,
    requested_tokens: usize,
) -> ContextExtension {
    ContextExtension::Yarn {
        native_tokens,
        requested_tokens,
        factor: config.yarn_factor,
        original_tokens: config.yarn_original_max_position_embeddings,
        attention_factor: 1.0 + 0.1 * config.yarn_factor.ln(),
    }
}

fn validate_yarn(config: &ModelConfig, requested_tokens: usize) -> Result<()> {
    let factor = config.yarn_factor;
    ensure!(
        factor.is_finite() && factor > 1.0,
        "YaRN factor must be finite and greater than 1"
    );
    let original = config.yarn_original_max_position_embeddings;
    ensure!(
        original > 0,
        "YaRN requires original_max_position_embeddings"
    );
    let scaled_capacity_f64 = original as f64 * factor as f64;
    ensure!(
        scaled_capacity_f64.is_finite() && scaled_capacity_f64 <= usize::MAX as f64,
        "YaRN capacity overflows the host token extent"
    );
    let scaled_capacity = scaled_capacity_f64.floor() as usize;
    ensure!(
        requested_tokens <= scaled_capacity,
        "requested context {requested_tokens} exceeds YaRN capacity {scaled_capacity}"
    );

    if is_qwen38_dense(config) {
        ensure!(
            (factor - 4.0).abs() <= f32::EPSILON,
            "Qwen3.8 1M requires YaRN factor 4"
        );
        ensure!(
            original == 262_144,
            "Qwen3.8 1M requires YaRN origin 262144"
        );
        ensure!(
            config.yarn_beta_fast == 32.0 && config.yarn_beta_slow == 1.0,
            "Qwen3.8 1M requires YaRN beta_fast=32 and beta_slow=1"
        );
        ensure!(
            config.rope_theta == 10_000_000.0,
            "Qwen3.8 1M keeps rope_theta at 10000000"
        );
        ensure!(
            (config.partial_rotary_factor - 0.25).abs() <= f64::EPSILON,
            "Qwen3.8 1M requires partial_rotary_factor 0.25"
        );
        ensure!(
            config.mrope_interleaved && config.mrope_section == [11, 11, 10],
            "Qwen3.8 1M requires interleaved MRoPE [11,11,10]"
        );
        ensure!(
            config.num_hidden_layers == 64
                && config.num_attention_heads == 24
                && config.num_key_value_heads == 4
                && config.head_dim == 256
                && config.rotary_dim() == 64,
            "Qwen3.8 1M requires L64/Q24/KV4/head_dim256/rotary_dim64"
        );
        ensure!(
            requested_tokens <= QWEN38_YARN_MAX_TOKENS,
            "Qwen3.8 static YaRN is qualified only through {QWEN38_YARN_MAX_TOKENS} total tokens"
        );
    }
    Ok(())
}

fn is_qwen38_dense(config: &ModelConfig) -> bool {
    config.model_type == "qwen3_5" && config.num_experts == 0 && config.hidden_size == 5120
}

fn validate_theta(theta: f64) -> Result<()> {
    if !theta.is_finite() || theta <= 1.0 {
        bail!("--rope-theta-override must be finite and greater than 1, got {theta}");
    }
    Ok(())
}

#[cfg(test)]
#[path = "context_extension_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "context_extension_regression_tests.rs"]
mod regression_tests;

#[cfg(test)]
#[path = "context_extension_source_authority_tests.rs"]
mod source_authority_tests;
