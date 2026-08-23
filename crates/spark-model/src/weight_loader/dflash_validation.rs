// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed validation for DFlash checkpoint metadata.

use std::collections::BTreeSet;

use anyhow::{Result, bail, ensure};
use spark_runtime::weights::{WeightDtype, WeightStore};

use self::config::{Dimensions, validate_config};
use super::dflash_loader::DflashConfig;

mod config;

#[derive(Clone, Copy)]
struct TensorMetadata<'a> {
    shape: &'a [usize],
    dtype: WeightDtype,
}

trait TensorMetadataSource {
    fn metadata(&self, name: &str) -> Option<TensorMetadata<'_>>;
    fn names(&self) -> Box<dyn Iterator<Item = &str> + '_>;
}

impl TensorMetadataSource for WeightStore {
    fn metadata(&self, name: &str) -> Option<TensorMetadata<'_>> {
        self.get(name).ok().map(|tensor| TensorMetadata {
            shape: &tensor.shape,
            dtype: tensor.dtype,
        })
    }

    fn names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        Box::new(self.names())
    }
}

pub(super) fn validate_dflash_store(
    store: &WeightStore,
    config: &DflashConfig,
) -> Result<Option<&'static str>> {
    validate_dflash_metadata(store, config)
}

fn validate_dflash_metadata(
    source: &dyn TensorMetadataSource,
    config: &DflashConfig,
) -> Result<Option<&'static str>> {
    let prefix = match (
        source.metadata("fc.weight").is_some(),
        source.metadata("model.fc.weight").is_some(),
    ) {
        (true, false) => "",
        (false, true) => "model.",
        (true, true) => bail!(
            "DFlash checkpoint contains both `fc.weight` and `model.fc.weight`; \
             refusing ambiguous mixed schemas"
        ),
        (false, false) if has_dflash_signature(source) => {
            bail!("DFlash checkpoint is incomplete: required tensor `fc.weight` is missing")
        }
        (false, false) => return Ok(None),
    };

    let dimensions = validate_config(config)?;
    validate_layer_indices(source, prefix, config.num_hidden_layers)?;
    validate_required_tensors(source, prefix, config, dimensions)?;
    validate_markov_tensors(source, prefix, config)?;
    validate_confidence_tensors(source, prefix, config)?;
    Ok(Some(prefix))
}

fn has_dflash_signature(source: &dyn TensorMetadataSource) -> bool {
    // A normal target checkpoint also has `model.layers.*`; its embedding is
    // the disambiguating root because supported DFlash checkpoints omit it
    // and share the target embedding at runtime. The explicit --dflash caller
    // rejects `None`, so even an unknown future layout cannot fall back.
    let has_target_embedding = source
        .names()
        .any(|name| matches!(name, "embed_tokens.weight" | "model.embed_tokens.weight"));
    source.names().any(|name| {
        matches!(
            name,
            "hidden_norm.weight"
                | "model.hidden_norm.weight"
                | "markov_head.markov_w1.weight"
                | "model.markov_head.markov_w1.weight"
                | "markov_head.markov_w2.weight"
                | "model.markov_head.markov_w2.weight"
                | "d2t"
                | "model.d2t"
                | "draft_id_to_target_id"
                | "model.draft_id_to_target_id"
        ) || name.starts_with("layers.")
            || name.starts_with("pre_fc_norms.")
            || name.starts_with("model.pre_fc_norms.")
            || (name.starts_with("model.layers.") && !has_target_embedding)
    })
}

fn validate_layer_indices(
    source: &dyn TensorMetadataSource,
    prefix: &str,
    layer_count: usize,
) -> Result<()> {
    let layer_prefix = format!("{prefix}layers.");
    let observed: BTreeSet<usize> = source
        .names()
        .filter_map(|name| {
            name.strip_prefix(&layer_prefix)?
                .split_once('.')?
                .0
                .parse()
                .ok()
        })
        .collect();
    let expected: BTreeSet<usize> = (0..layer_count).collect();
    ensure!(
        observed == expected,
        "DFlash checkpoint layer indices {observed:?} do not match config {expected:?}"
    );
    Ok(())
}

fn require_tensor(
    source: &dyn TensorMetadataSource,
    name: &str,
    expected_shape: &[usize],
) -> Result<()> {
    let tensor = source
        .metadata(name)
        .ok_or_else(|| anyhow::anyhow!("DFlash checkpoint missing required tensor `{name}`"))?;
    ensure!(
        tensor.dtype == WeightDtype::BF16,
        "DFlash tensor `{name}` has dtype {:?}; expected BF16",
        tensor.dtype
    );
    ensure!(
        tensor.shape == expected_shape,
        "DFlash tensor `{name}` has shape {:?}; expected {:?}",
        tensor.shape,
        expected_shape
    );
    Ok(())
}

fn validate_required_tensors(
    source: &dyn TensorMetadataSource,
    prefix: &str,
    config: &DflashConfig,
    dim: Dimensions,
) -> Result<()> {
    require_tensor(
        source,
        &format!("{prefix}fc.weight"),
        &[dim.hidden, dim.target_stack],
    )?;
    require_tensor(
        source,
        &format!("{prefix}hidden_norm.weight"),
        &[dim.hidden],
    )?;
    require_tensor(source, &format!("{prefix}norm.weight"), &[dim.hidden])?;

    for layer in 0..config.num_hidden_layers {
        let lp = format!("{prefix}layers.{layer}");
        require_tensor(
            source,
            &format!("{lp}.input_layernorm.weight"),
            &[dim.hidden],
        )?;
        require_tensor(
            source,
            &format!("{lp}.post_attention_layernorm.weight"),
            &[dim.hidden],
        )?;
        require_tensor(
            source,
            &format!("{lp}.self_attn.q_proj.weight"),
            &[dim.query, dim.hidden],
        )?;
        require_tensor(
            source,
            &format!("{lp}.self_attn.k_proj.weight"),
            &[dim.key_value, dim.hidden],
        )?;
        require_tensor(
            source,
            &format!("{lp}.self_attn.v_proj.weight"),
            &[dim.key_value, dim.hidden],
        )?;
        require_tensor(
            source,
            &format!("{lp}.self_attn.o_proj.weight"),
            &[dim.hidden, dim.query],
        )?;
        require_tensor(
            source,
            &format!("{lp}.self_attn.q_norm.weight"),
            &[dim.head],
        )?;
        require_tensor(
            source,
            &format!("{lp}.self_attn.k_norm.weight"),
            &[dim.head],
        )?;
        require_tensor(
            source,
            &format!("{lp}.mlp.gate_proj.weight"),
            &[dim.intermediate, dim.hidden],
        )?;
        require_tensor(
            source,
            &format!("{lp}.mlp.up_proj.weight"),
            &[dim.intermediate, dim.hidden],
        )?;
        require_tensor(
            source,
            &format!("{lp}.mlp.down_proj.weight"),
            &[dim.hidden, dim.intermediate],
        )?;
    }
    Ok(())
}

fn validate_markov_tensors(
    source: &dyn TensorMetadataSource,
    prefix: &str,
    config: &DflashConfig,
) -> Result<()> {
    let w1 = format!("{prefix}markov_head.markov_w1.weight");
    let w2 = format!("{prefix}markov_head.markov_w2.weight");
    let present = source.metadata(&w1).is_some() || source.metadata(&w2).is_some();
    if config.markov_rank == 0 {
        ensure!(
            !present,
            "DFlash checkpoint has Markov tensors but config markov_rank=0"
        );
        return Ok(());
    }
    let shape = [config.vocab_size, config.markov_rank];
    require_tensor(source, &w1, &shape)?;
    require_tensor(source, &w2, &shape)
}

fn validate_confidence_tensors(
    source: &dyn TensorMetadataSource,
    prefix: &str,
    config: &DflashConfig,
) -> Result<()> {
    let weight = format!("{prefix}confidence_head.proj.weight");
    let bias = format!("{prefix}confidence_head.proj.bias");
    match config.confidence_head_config()? {
        Some(confidence) => {
            require_tensor(source, &weight, &[1, confidence.input_dim])?;
            require_tensor(source, &bias, &[1])
        }
        None => {
            let present = source.metadata(&weight).is_some() || source.metadata(&bias).is_some();
            ensure!(
                !present,
                "Drafter checkpoint contains `{weight}` / `{bias}`, but its config disables the \
                 confidence head; refusing to silently discard trained tensors"
            );
            Ok(())
        }
    }
}

#[cfg(test)]
#[path = "dflash_validation_tests.rs"]
mod tests;
