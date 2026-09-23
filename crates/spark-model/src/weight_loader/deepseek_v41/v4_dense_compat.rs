// SPDX-License-Identifier: AGPL-3.0-only

//! Embedding / final-norm / lm-head loaders, moved verbatim from
//! `weight_loader::deepseek_v4::DeepSeekV4WeightLoader` (DeepSeek-V4-Flash-0731's loader).
//!
//! V4.1's [`super::DeepSeekV41WeightLoader`] delegates only these three dense tensors to
//! the V4 loader (see that module's doc comment for why `load_layers`/`load_mtp_weights`
//! are NOT delegated: EXL3 vs CB3 expert quantization diverges there). Porting the whole
//! `deepseek_v4` module onto `integrate/all-models` would also drag in its DSpark/MTP/
//! indexer/vision_moe internals for DeepSeek-V4-Flash, a model outside the all-models
//! matrix — out of scope (team-lead, 2026-09-23). These three functions are copied as
//! plain functions instead of as a `ModelWeightLoader` impl, kept as close to the D source
//! as possible so they stay correct by provenance; the port's end gate (runG logits
//! byte-identical to dsv41/integration) is what actually proves it.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::weight_map::{DenseWeight, dense, dense_auto};

pub(crate) fn load_embedding(
    store: &WeightStore,
    _config: &ModelConfig,
    _gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    // RedHatAI re-quant uses flattened naming; try it first, then standard HF names.
    if let Ok(w) = dense(store, "embed.weight") {
        return Ok(w);
    }
    if let Ok(w) = dense(store, "model.embed_tokens.weight") {
        return Ok(w);
    }
    dense(store, "embed_tokens.weight")
        .context("DeepSeek-V4: no embedding tensor found (tried embed.weight, model.embed_tokens.weight, embed_tokens.weight)")
}

pub(crate) fn load_final_norm(
    store: &WeightStore,
    _config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    // DeepSeek-V4 ships HF-vanilla RMSNorm weights (scale = weight). Load them
    // EXACTLY; the model dispatches `rms_norm_vanilla` (see
    // `crate::ships_vanilla_norm_weights`).
    if let Ok(w) = dense_auto(store, "norm.weight", gpu) {
        return Ok(w);
    }
    if let Ok(w) = dense_auto(store, "model.norm.weight", gpu) {
        return Ok(w);
    }
    dense_auto(store, "final_norm.weight", gpu)
        .context("DeepSeek-V4: no final norm tensor found (tried norm.weight, model.norm.weight, final_norm.weight)")
}

pub(crate) fn load_lm_head(
    store: &WeightStore,
    config: &ModelConfig,
    _gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    // Try standard HF name first
    if store.contains("lm_head.weight") {
        return dense(store, "lm_head.weight");
    }
    // RedHatAI / consolidated checkpoints
    if store.contains("output.weight") {
        return dense(store, "output.weight");
    }
    if store.contains("head.weight") {
        return dense(store, "head.weight");
    }
    // Tied embeddings: either config says so, or no separate head exists
    if config.tie_word_embeddings
        || store.contains("embed.weight")
        || store.contains("model.embed_tokens.weight")
    {
        // Tied: reuse the embedding tensor. Inline the same dense lookups as
        // load_embedding (load_lm_head has no `gpu`, and they don't need it).
        if let Ok(w) = dense(store, "embed.weight") {
            return Ok(w);
        }
        if let Ok(w) = dense(store, "model.embed_tokens.weight") {
            return Ok(w);
        }
        return dense(store, "embed_tokens.weight")
            .context("DeepSeek-V4: tied lm_head — no embedding tensor found");
    }
    anyhow::bail!(
        "DeepSeek-V4: lm_head not found (tried lm_head.weight, output.weight, head.weight, and tied embeddings)"
    )
}
