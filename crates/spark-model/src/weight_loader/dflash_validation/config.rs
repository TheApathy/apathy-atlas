// SPDX-License-Identifier: AGPL-3.0-only

//! Validation of DFlash config fields that determine checkpoint shapes.

use std::collections::BTreeSet;

use anyhow::{Result, ensure};

use crate::weight_loader::dflash_loader::DflashConfig;

#[derive(Clone, Copy)]
pub(super) struct Dimensions {
    pub hidden: usize,
    pub intermediate: usize,
    pub query: usize,
    pub key_value: usize,
    pub head: usize,
    pub target_stack: usize,
}

pub(super) fn validate_config(
    config: &DflashConfig,
    require_qwen38_dflash2: bool,
) -> Result<Dimensions> {
    config.validate_supported_semantics()?;
    for (name, value) in [
        ("hidden_size", config.hidden_size),
        ("num_hidden_layers", config.num_hidden_layers),
        ("intermediate_size", config.intermediate_size),
        ("num_attention_heads", config.num_attention_heads),
        ("num_key_value_heads", config.num_key_value_heads),
        ("head_dim", config.head_dim),
        ("vocab_size", config.vocab_size),
        ("block_size", config.block_size),
    ] {
        ensure!(value > 0, "DFlash config `{name}` must be non-zero");
    }
    ensure!(
        config
            .num_attention_heads
            .is_multiple_of(config.num_key_value_heads),
        "DFlash config num_attention_heads={} is not divisible by num_key_value_heads={}",
        config.num_attention_heads,
        config.num_key_value_heads
    );
    if let Some(layer_types) = &config.layer_types {
        ensure!(
            layer_types.len() == config.num_hidden_layers,
            "DFlash config has {} layer_types but num_hidden_layers={}",
            layer_types.len(),
            config.num_hidden_layers
        );
    }

    let sub = config
        .dflash_config
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DFlash config is missing `dflash_config`"))?;
    ensure!(
        !sub.fc_layernorm,
        "DFlash checkpoint requires dflash_config.fc_layernorm=true, but Atlas does not yet \
         load or execute the per-capture pre_fc_norms weights"
    );
    ensure!(
        !sub.target_layer_ids.is_empty(),
        "DFlash config `target_layer_ids` must not be empty"
    );
    let unique_target_layers: BTreeSet<_> = sub.target_layer_ids.iter().copied().collect();
    ensure!(
        unique_target_layers.len() == sub.target_layer_ids.len(),
        "DFlash config `target_layer_ids` contains duplicates: {:?}",
        sub.target_layer_ids
    );
    if require_qwen38_dflash2 {
        validate_qwen38_dflash2_config(config, sub)?;
    }

    let checked_product = |name: &str, left: usize, right: usize| {
        left.checked_mul(right)
            .ok_or_else(|| anyhow::anyhow!("DFlash config overflow computing {name}"))
    };
    Ok(Dimensions {
        hidden: config.hidden_size,
        intermediate: config.intermediate_size,
        query: checked_product("query width", config.num_attention_heads, config.head_dim)?,
        key_value: checked_product(
            "key/value width",
            config.num_key_value_heads,
            config.head_dim,
        )?,
        head: config.head_dim,
        target_stack: checked_product(
            "target hidden stack width",
            sub.target_layer_ids.len(),
            config.hidden_size,
        )?,
    })
}

pub(super) fn declares_qwen38_dflash2(config: &DflashConfig) -> bool {
    let architecture = config
        .architectures
        .iter()
        .any(|name| name == "DFlash2DraftModel");
    let geometry = config.dflash_config.as_ref().is_some_and(|sub| {
        sub.conv_kernel_size != 0
            || sub.conv_group_size != 0
            || sub.selector_rank != 0
            || sub.selector_top_k != 0
    });
    architecture || geometry
}

fn validate_qwen38_dflash2_config(
    config: &DflashConfig,
    sub: &crate::weight_loader::dflash_loader::DflashSubConfig,
) -> Result<()> {
    ensure!(
        config.architectures.len() == 1 && config.architectures[0] == "DFlash2DraftModel",
        "official Qwen3.8 DFlash2 requires architectures=[DFlash2DraftModel]"
    );
    ensure!(
        (
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers
        ) == (5120, 17408, 5),
        "official Qwen3.8 DFlash2 requires hidden/intermediate/layers=5120/17408/5"
    );
    ensure!(
        (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim
        ) == (32, 8, 128),
        "official Qwen3.8 DFlash2 requires Q/KV/head_dim=32/8/128"
    );
    ensure!(
        config.vocab_size == 248320
            && config.draft_vocab_size.unwrap_or(config.vocab_size) == 248320,
        "official Qwen3.8 DFlash2 requires target and draft vocab 248320"
    );
    ensure!(
        sub.block_size == Some(8) && config.resolved_block_size() == 8,
        "official Qwen3.8 DFlash2 requires nested block_size=8"
    );
    ensure!(
        sub.mask_token_id == 248070 && sub.mask_token_id < config.vocab_size as u32,
        "official Qwen3.8 DFlash2 requires mask_token_id=248070 within vocab"
    );
    ensure!(
        sub.target_layer_ids == [5, 19, 33, 47, 61],
        "official Qwen3.8 DFlash2 requires ordered target taps [5,19,33,47,61]"
    );
    ensure!(
        (sub.conv_kernel_size, sub.conv_group_size) == (2, 16)
            && config.hidden_size.is_multiple_of(sub.conv_group_size),
        "official Qwen3.8 DFlash2 requires conv kernel/group=2/16 dividing hidden"
    );
    ensure!(
        (sub.selector_rank, sub.selector_top_k) == (256, 16),
        "official Qwen3.8 DFlash2 requires selector rank/top_k=256/16"
    );
    let layer_types = config
        .layer_types
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("official Qwen3.8 DFlash2 requires layer_types"))?;
    ensure!(
        layer_types.len() == 5
            && layer_types
                .iter()
                .all(|layer_type| layer_type == "sliding_attention"),
        "official Qwen3.8 DFlash2 requires five sliding_attention layers"
    );
    ensure!(
        config.sliding_window == Some(2048) && config.is_causal == Some(false),
        "official Qwen3.8 DFlash2 requires SWA=2048 and explicit noncausal attention"
    );
    let rope = config
        .rope_scaling
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("official Qwen3.8 DFlash2 requires rope_parameters"))?;
    ensure!(
        config.rope_theta.to_bits() == 10_000_000.0f32.to_bits()
            && rope.rope_type.as_deref() == Some("default")
            && rope.rope_theta.map(f32::to_bits) == Some(10_000_000.0f32.to_bits())
            && rope.factor.is_none()
            && rope.beta_fast.is_none()
            && rope.beta_slow.is_none()
            && rope.original_max_position_embeddings.is_none()
            && rope.attention_factor.is_none(),
        "official Qwen3.8 DFlash2 requires default RoPE with theta=10000000 and no scaling"
    );
    ensure!(
        config.rms_norm_eps.map(f32::to_bits) == Some(1e-6f32.to_bits())
            && config.hidden_act.as_deref() == Some("silu"),
        "official Qwen3.8 DFlash2 requires explicit rms_norm_eps=1e-6 and hidden_act=silu"
    );
    ensure!(
        sub.projector_type.is_none()
            && config.markov_rank == 0
            && !config.enable_confidence_head.unwrap_or(false)
            && !config.confidence_head_with_markov.unwrap_or(false)
            && !sub.enable_confidence_head.unwrap_or(false)
            && !sub.confidence_head_with_markov.unwrap_or(false),
        "official Qwen3.8 DFlash2 forbids DSpark/Markov/confidence extensions"
    );
    Ok(())
}
