// SPDX-License-Identifier: AGPL-3.0-only

//! Strict schema admission for Qwen3.8-Flash-Next native MTP sidecars.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::weights::{WeightDtype, WeightStore};

fn tensor(store: &WeightStore, name: &str, shape: &[usize], dtype: WeightDtype) -> Result<()> {
    let value = store.get(name)?;
    ensure!(
        value.shape == shape,
        "{name} shape {:?} != {shape:?}",
        value.shape
    );
    ensure!(
        value.dtype == dtype,
        "{name} dtype {:?} != {dtype:?}",
        value.dtype
    );
    Ok(())
}

fn validate_experts(store: &WeightStore, config: &ModelConfig) -> Result<()> {
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    for expert in 0..config.num_experts {
        for projection in ["gate_proj", "up_proj"] {
            let p = format!("mtp.layers.0.mlp.experts.{expert}.{projection}");
            tensor(
                store,
                &format!("{p}.weight"),
                &[inter, h / 2],
                WeightDtype::UInt8,
            )?;
            tensor(
                store,
                &format!("{p}.weight_scale"),
                &[inter, h / 16],
                WeightDtype::FP8E4M3,
            )?;
            tensor(
                store,
                &format!("{p}.weight_scale_2"),
                &[],
                WeightDtype::FP32,
            )?;
            tensor(store, &format!("{p}.input_scale"), &[], WeightDtype::FP32)?;
        }
        let p = format!("mtp.layers.0.mlp.experts.{expert}.down_proj");
        tensor(
            store,
            &format!("{p}.weight"),
            &[h, inter / 2],
            WeightDtype::UInt8,
        )?;
        tensor(
            store,
            &format!("{p}.weight_scale"),
            &[h, inter / 16],
            WeightDtype::FP8E4M3,
        )?;
        tensor(
            store,
            &format!("{p}.weight_scale_2"),
            &[],
            WeightDtype::FP32,
        )?;
        tensor(store, &format!("{p}.input_scale"), &[], WeightDtype::FP32)?;
    }

    Ok(())
}

/// Validate a sidecar containing only the per-expert native NVFP4 tensors.
///
/// Official converted targets may already contain the 29 fixed MTP tensors
/// (and two packed BF16 expert tensors). Loading only the numbered expert bank
/// avoids ambiguous duplicate replacement while still admitting the native
/// NVFP4 representation needed by Atlas's MoE kernels.
pub fn validate_qwen4_mtp_expert_store(store: &WeightStore, config: &ModelConfig) -> Result<()> {
    ensure!(
        config.is_qwen4_exp(),
        "native Qwen4 MTP requires model_type=qwen4_exp"
    );
    validate_experts(store, config)?;
    ensure!(
        store.len() == config.num_experts * 12,
        "Qwen4 MTP expert sidecar tensor count {} != {}",
        store.len(),
        config.num_experts * 12
    );
    Ok(())
}

pub fn validate_qwen4_mtp_store(store: &WeightStore, config: &ModelConfig) -> Result<()> {
    ensure!(
        config.is_qwen4_exp(),
        "native Qwen4 MTP requires model_type=qwen4_exp"
    );
    let h = config.hidden_size;
    let r = config.residual_width();
    let rank = config.hc_lowrank;
    let inter = config.moe_intermediate_size;
    let q = config.num_attention_heads * config.head_dim * 2;
    let kv = config.num_key_value_heads * config.head_dim;
    let qsa = (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;

    for (name, shape) in [
        ("mtp.fc_embedding.weight", vec![h, h]),
        ("mtp.fc_hidden.weight", vec![h, h]),
        ("mtp.pre_fc_norm_embedding.weight", vec![h]),
        ("mtp.pre_fc_norm_hidden.weight", vec![r]),
        ("mtp.hyper_connection_mixer.hc_norm.weight", vec![r]),
        (
            "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
            vec![rank, r],
        ),
        (
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
            vec![r, rank],
        ),
        ("mtp.layers.0.mlp.gate.weight", vec![config.num_experts, h]),
        (
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
            vec![inter, h],
        ),
        (
            "mtp.layers.0.mlp.shared_expert.up_proj.weight",
            vec![inter, h],
        ),
        (
            "mtp.layers.0.mlp.shared_expert.down_proj.weight",
            vec![h, inter],
        ),
        ("mtp.layers.0.mlp.shared_expert_gate.weight", vec![1, h]),
        ("mtp.layers.0.self_attn.q_proj.weight", vec![q, h]),
        ("mtp.layers.0.self_attn.k_proj.weight", vec![kv, h]),
        ("mtp.layers.0.self_attn.v_proj.weight", vec![kv, h]),
        ("mtp.layers.0.self_attn.o_proj.weight", vec![h, q / 2]),
        (
            "mtp.layers.0.self_attn.q_norm.weight",
            vec![config.head_dim],
        ),
        (
            "mtp.layers.0.self_attn.k_norm.weight",
            vec![config.head_dim],
        ),
        (
            "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
            vec![qsa, h],
        ),
        (
            "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
            vec![config.indexer_head_dim],
        ),
        (
            "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
            vec![config.indexer_head_dim],
        ),
    ] {
        tensor(store, name, &shape, WeightDtype::BF16)?;
    }

    for prefix in [
        "mtp.layers.0.attn_hyper_connection",
        "mtp.layers.0.mlp_hyper_connection",
    ] {
        tensor(
            store,
            &format!("{prefix}.hc_norm.weight"),
            &[r],
            WeightDtype::BF16,
        )?;
        tensor(
            store,
            &format!("{prefix}.input_mix_weight_down.weight"),
            &[rank, r],
            WeightDtype::BF16,
        )?;
        tensor(
            store,
            &format!("{prefix}.input_mix_weight_up.weight"),
            &[r, rank],
            WeightDtype::BF16,
        )?;
        tensor(
            store,
            &format!("{prefix}.block_inject_weight.weight"),
            &[config.hc_count, r],
            WeightDtype::BF16,
        )?;
    }

    validate_experts(store, config)?;

    let mtp_count = store
        .names()
        .filter(|name| name.starts_with("mtp."))
        .count();
    let expected = 29 + config.num_experts * 12;
    // Converted targets can retain two packed BF16 expert tensors alongside
    // the numbered NVFP4 bank. No other unrecognised MTP tensors are admitted.
    ensure!(
        mtp_count == expected || mtp_count == expected + 2,
        "Qwen4 MTP tensor count {mtp_count} is neither {expected} nor {}",
        expected + 2
    );
    Ok(())
}
