// SPDX-License-Identifier: AGPL-3.0-only

//! Dense MTP head weights loader (Qwen3.5/3.6 27B-class dense models).
//!
//! Mirrors `MtpWeights` (MoE) but with a single dense MLP block instead of
//! a routed expert mixture. Used by AEON-7's NVFP4 + MTP re-quants of the
//! Qwen3.6-27B Multimodal checkpoint, which ship 15 `mtp.*` tensors with
//! the dense layout:
//!
//! ```text
//! mtp.fc.weight                                [hidden, 2*hidden]   BF16
//! mtp.pre_fc_norm_embedding.weight             [hidden]             BF16
//! mtp.pre_fc_norm_hidden.weight                [hidden]             BF16
//! mtp.layers.0.input_layernorm.weight          [hidden]             BF16
//! mtp.layers.0.self_attn.q_proj.weight         [n_q*hd*2, hidden]   BF16  (×2 for attn_output_gate)
//! mtp.layers.0.self_attn.k_proj.weight         [n_kv*hd, hidden]    BF16
//! mtp.layers.0.self_attn.v_proj.weight         [n_kv*hd, hidden]    BF16
//! mtp.layers.0.self_attn.o_proj.weight         [hidden, n_q*hd]     BF16
//! mtp.layers.0.self_attn.q_norm.weight         [head_dim]           BF16
//! mtp.layers.0.self_attn.k_norm.weight         [head_dim]           BF16
//! mtp.layers.0.post_attention_layernorm.weight [hidden]             BF16
//! mtp.layers.0.mlp.gate_proj.weight            [intermediate, hidden] BF16
//! mtp.layers.0.mlp.up_proj.weight              [intermediate, hidden] BF16
//! mtp.layers.0.mlp.down_proj.weight            [hidden, intermediate] BF16
//! mtp.norm.weight                              [hidden]             BF16
//! ```

#![allow(unused_imports)]

use anyhow::{Context, Result};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use super::{DenseWeight, dense};

/// Dense MTP head weights (single transformer block + dense MLP).
///
/// Storage is BF16 throughout — AEON's modelopt NVFP4 quantization
/// excludes `mtp.*` from the quant set (see `hf_quant_config.json`'s
/// `exclude_modules`).
pub struct MtpDenseWeights {
    /// RMSNorm on token embedding before concat: `[hidden_size]` BF16.
    pub pre_fc_norm_embedding: DenseWeight,
    /// RMSNorm on target hidden state before concat: `[hidden_size]` BF16.
    pub pre_fc_norm_hidden: DenseWeight,
    /// Concat projection: `[hidden_size, 2*hidden_size]` BF16.
    pub fc: DenseWeight,

    /// Input layernorm before attention: `[hidden_size]` BF16.
    pub input_layernorm: DenseWeight,
    /// Q projection (with `attn_output_gate` doubling output dim): BF16.
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub o_proj: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,

    /// Post-attention layernorm: `[hidden_size]` BF16.
    pub post_attn_layernorm: DenseWeight,

    /// Dense MLP — gate + up + down projections (BF16). NO MoE routing.
    pub mlp_gate: DenseWeight,
    pub mlp_up: DenseWeight,
    pub mlp_down: DenseWeight,

    /// Final output RMSNorm: `[hidden_size]` BF16.
    pub norm: DenseWeight,
}

/// Load dense MTP head weights from a WeightStore.
///
/// Returns `Ok(None)` when the store has no `mtp.fc.weight` tensor (model
/// has no MTP head). Returns an error if a key is missing once the head's
/// presence is established.
pub fn load_mtp_dense(store: &WeightStore) -> Result<Option<MtpDenseWeights>> {
    if !store.contains("mtp.fc.weight") {
        return Ok(None);
    }
    Ok(Some(MtpDenseWeights {
        pre_fc_norm_embedding: dense(store, "mtp.pre_fc_norm_embedding.weight")
            .context("mtp.pre_fc_norm_embedding")?,
        pre_fc_norm_hidden: dense(store, "mtp.pre_fc_norm_hidden.weight")
            .context("mtp.pre_fc_norm_hidden")?,
        fc: dense(store, "mtp.fc.weight").context("mtp.fc")?,
        input_layernorm: dense(store, "mtp.layers.0.input_layernorm.weight")
            .context("mtp.layers.0.input_layernorm")?,
        q_proj: dense(store, "mtp.layers.0.self_attn.q_proj.weight")
            .context("mtp.layers.0.self_attn.q_proj")?,
        k_proj: dense(store, "mtp.layers.0.self_attn.k_proj.weight")
            .context("mtp.layers.0.self_attn.k_proj")?,
        v_proj: dense(store, "mtp.layers.0.self_attn.v_proj.weight")
            .context("mtp.layers.0.self_attn.v_proj")?,
        o_proj: dense(store, "mtp.layers.0.self_attn.o_proj.weight")
            .context("mtp.layers.0.self_attn.o_proj")?,
        q_norm: dense(store, "mtp.layers.0.self_attn.q_norm.weight")
            .context("mtp.layers.0.self_attn.q_norm")?,
        k_norm: dense(store, "mtp.layers.0.self_attn.k_norm.weight")
            .context("mtp.layers.0.self_attn.k_norm")?,
        post_attn_layernorm: dense(store, "mtp.layers.0.post_attention_layernorm.weight")
            .context("mtp.layers.0.post_attention_layernorm")?,
        mlp_gate: dense(store, "mtp.layers.0.mlp.gate_proj.weight")
            .context("mtp.layers.0.mlp.gate_proj")?,
        mlp_up: dense(store, "mtp.layers.0.mlp.up_proj.weight")
            .context("mtp.layers.0.mlp.up_proj")?,
        mlp_down: dense(store, "mtp.layers.0.mlp.down_proj.weight")
            .context("mtp.layers.0.mlp.down_proj")?,
        norm: dense(store, "mtp.norm.weight").context("mtp.norm")?,
    }))
}
