// SPDX-License-Identifier: AGPL-3.0-only

//! Strict schema admission for Qwen3.8-Flash-Next native MTP targets and sidecars.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::weights::{WeightDtype, WeightStore};

#[cfg(test)]
#[path = "qwen4_mtp_tests.rs"]
mod tests;

#[derive(Clone, Copy)]
pub(crate) struct MtpMetadata<'a> {
    pub(crate) shape: &'a [usize],
    pub(crate) dtype: WeightDtype,
}

pub(crate) trait MtpMetadataSource {
    fn metadata(&self, name: &str) -> Option<MtpMetadata<'_>>;
    fn names(&self) -> Box<dyn Iterator<Item = &str> + '_>;
}

impl MtpMetadataSource for WeightStore {
    fn metadata(&self, name: &str) -> Option<MtpMetadata<'_>> {
        self.get(name).ok().map(|value| MtpMetadata {
            shape: &value.shape,
            dtype: value.dtype,
        })
    }

    fn names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        Box::new(self.names())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Qwen4MtpExpertLayout {
    PackedBf16,
    NumberedNvfp4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PackedBf16ExpertOffsets {
    pub(crate) gate: usize,
    pub(crate) up: usize,
    pub(crate) down: usize,
}

pub(crate) fn packed_bf16_expert_offsets(
    expert: usize,
    num_experts: usize,
    inter: usize,
    hidden: usize,
) -> Result<PackedBf16ExpertOffsets> {
    ensure!(
        expert < num_experts,
        "packed expert index {expert} >= {num_experts}"
    );
    let projection_bytes = inter
        .checked_mul(hidden)
        .and_then(|value| value.checked_mul(WeightDtype::BF16.byte_size()))
        .ok_or_else(|| anyhow::anyhow!("packed BF16 expert projection byte size overflow"))?;
    let gate_up_bytes = projection_bytes
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("packed BF16 gate/up byte size overflow"))?;
    let gate = expert
        .checked_mul(gate_up_bytes)
        .ok_or_else(|| anyhow::anyhow!("packed BF16 gate offset overflow"))?;
    let up = gate
        .checked_add(projection_bytes)
        .ok_or_else(|| anyhow::anyhow!("packed BF16 up offset overflow"))?;
    let down = expert
        .checked_mul(projection_bytes)
        .ok_or_else(|| anyhow::anyhow!("packed BF16 down offset overflow"))?;
    let gate_up_total = num_experts
        .checked_mul(gate_up_bytes)
        .ok_or_else(|| anyhow::anyhow!("packed BF16 gate/up bank byte size overflow"))?;
    let down_total = num_experts
        .checked_mul(projection_bytes)
        .ok_or_else(|| anyhow::anyhow!("packed BF16 down bank byte size overflow"))?;
    ensure!(
        up.checked_add(projection_bytes)
            .is_some_and(|end| end <= gate_up_total)
            && down
                .checked_add(projection_bytes)
                .is_some_and(|end| end <= down_total),
        "packed BF16 expert slice exceeds admitted bank"
    );
    Ok(PackedBf16ExpertOffsets { gate, up, down })
}

fn tensor<S: MtpMetadataSource + ?Sized>(
    store: &S,
    name: &str,
    shape: &[usize],
    dtype: WeightDtype,
) -> Result<()> {
    let value = store
        .metadata(name)
        .ok_or_else(|| anyhow::anyhow!("Weight '{name}' not found in store"))?;
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

fn validate_experts<S: MtpMetadataSource + ?Sized>(store: &S, config: &ModelConfig) -> Result<()> {
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

fn validate_exact_flash_next_config(config: &ModelConfig) -> Result<()> {
    ensure!(
        config.is_qwen4_exp()
            && config.hidden_size == 2_560
            && config.num_hidden_layers == 48
            && matches!(config.vocab_size, 248_077 | 248_320)
            && config.num_attention_heads == 24
            && config.num_key_value_heads == 2
            && config.head_dim == 256
            && config.num_experts == 512
            && config.num_experts_per_tok == 10
            && config.moe_intermediate_size == 640
            && config.shared_expert_intermediate_size == 640
            && config.hc_count == 4
            && config.hc_lowrank == 320
            && config.mtp_num_hidden_layers == 1
            && config.indexer_n_heads == 4
            && config.indexer_kv_heads == 1
            && config.indexer_head_dim == 128,
        "native packed MTP requires the exact Qwen3.8-Flash-Next target geometry"
    );
    let quant = config
        .quantization_config
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("native packed MTP requires ModelOpt NVFP4 metadata"))?;
    ensure!(
        quant.quant_method == "modelopt" && quant.quant_algo == "NVFP4",
        "native packed MTP requires exact ModelOpt NVFP4 metadata"
    );
    Ok(())
}

fn validate_packed_experts<S: MtpMetadataSource + ?Sized>(
    store: &S,
    config: &ModelConfig,
) -> Result<()> {
    let experts = config.num_experts;
    let inter = config.moe_intermediate_size;
    let hidden = config.hidden_size;
    let gate_up_rows = inter
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("packed BF16 gate/up shape overflow"))?;
    tensor(
        store,
        "mtp.layers.0.mlp.experts.gate_up_proj",
        &[experts, gate_up_rows, hidden],
        WeightDtype::BF16,
    )?;
    tensor(
        store,
        "mtp.layers.0.mlp.experts.down_proj",
        &[experts, hidden, inter],
        WeightDtype::BF16,
    )?;
    let _ = packed_bf16_expert_offsets(experts - 1, experts, inter, hidden)?;
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
    let expected = config
        .num_experts
        .checked_mul(12)
        .ok_or_else(|| anyhow::anyhow!("Qwen4 MTP expert sidecar count overflow"))?;
    ensure!(
        store.len() == expected,
        "Qwen4 MTP expert sidecar tensor count {} != {}",
        store.len(),
        expected
    );
    Ok(())
}

fn validate_fixed_tensors<S: MtpMetadataSource + ?Sized>(
    store: &S,
    config: &ModelConfig,
) -> Result<()> {
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

    Ok(())
}

fn is_numbered_expert_name(name: &str) -> bool {
    name.strip_prefix("mtp.layers.0.mlp.experts.")
        .and_then(|suffix| suffix.split('.').next())
        .is_some_and(|expert| {
            !expert.is_empty() && expert.bytes().all(|byte| byte.is_ascii_digit())
        })
}

pub(crate) fn classify_qwen4_mtp_metadata<S: MtpMetadataSource + ?Sized>(
    store: &S,
    config: &ModelConfig,
) -> Result<Option<Qwen4MtpExpertLayout>> {
    let mut mtp_count = 0usize;
    let mut expert_marker = false;
    let mut numbered = false;
    for name in store.names() {
        if name.starts_with("mtp.") {
            mtp_count = mtp_count
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("Qwen4 MTP tensor count overflow"))?;
        }
        if name.starts_with("mtp.layers.0.mlp.experts.") {
            expert_marker = true;
            numbered |= is_numbered_expert_name(name);
        }
    }
    if mtp_count == 0 {
        return Ok(None);
    }

    validate_exact_flash_next_config(config)?;
    ensure!(
        expert_marker,
        "Qwen4 MTP tensors are present but the expert bank is missing or uses an unrecognised name"
    );
    validate_fixed_tensors(store, config)?;

    let packed_gate = store
        .metadata("mtp.layers.0.mlp.experts.gate_up_proj")
        .is_some();
    let packed_down = store
        .metadata("mtp.layers.0.mlp.experts.down_proj")
        .is_some();
    ensure!(
        packed_gate == packed_down,
        "Qwen4 MTP packed BF16 expert bank is partial"
    );
    if packed_gate {
        validate_packed_experts(store, config)?;
    }

    if numbered {
        validate_experts(store, config)?;
        let expected = 29usize
            .checked_add(
                config
                    .num_experts
                    .checked_mul(12)
                    .ok_or_else(|| anyhow::anyhow!("Qwen4 MTP expert count overflow"))?,
            )
            .and_then(|value| value.checked_add(usize::from(packed_gate) * 2))
            .ok_or_else(|| anyhow::anyhow!("Qwen4 MTP tensor count overflow"))?;
        ensure!(
            mtp_count == expected,
            "Qwen4 MTP numbered tensor count {mtp_count} != {expected}"
        );
        return Ok(Some(Qwen4MtpExpertLayout::NumberedNvfp4));
    }

    ensure!(
        packed_gate,
        "Qwen4 MTP expert bank uses an unrecognised name or layout"
    );
    ensure!(
        mtp_count == 31,
        "Qwen4 MTP packed tensor count {mtp_count} != 31"
    );
    Ok(Some(Qwen4MtpExpertLayout::PackedBf16))
}

pub(crate) fn classify_qwen4_mtp_store(
    store: &WeightStore,
    config: &ModelConfig,
) -> Result<Option<Qwen4MtpExpertLayout>> {
    classify_qwen4_mtp_metadata(store, config)
}

pub fn validate_qwen4_mtp_store(store: &WeightStore, config: &ModelConfig) -> Result<()> {
    ensure!(
        classify_qwen4_mtp_store(store, config)?.is_some(),
        "Qwen4 MTP expert bank is absent"
    );
    Ok(())
}
