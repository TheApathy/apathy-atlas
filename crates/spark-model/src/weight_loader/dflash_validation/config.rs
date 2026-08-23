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

pub(super) fn validate_config(config: &DflashConfig) -> Result<Dimensions> {
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
