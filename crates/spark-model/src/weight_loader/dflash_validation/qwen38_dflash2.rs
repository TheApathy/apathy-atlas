// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeSet;

use anyhow::{Result, ensure};

use super::config::Dimensions;
use super::{TensorMetadataSource, require_tensor};
use crate::weight_loader::dflash_loader::DflashConfig;

const OFFICIAL_TENSOR_COUNT: usize = 81;

pub(super) fn has_tensor_signature(source: &dyn TensorMetadataSource) -> bool {
    source.names().any(|name| {
        let name = name.strip_prefix("model.").unwrap_or(name);
        name.starts_with("candidate_selector.")
            || (name.starts_with("layers.")
                && (name.contains(".attention_conv.") || name.contains(".mlp_conv.")))
    })
}

pub(super) fn validate_official_tensors(
    source: &dyn TensorMetadataSource,
    prefix: &str,
    config: &DflashConfig,
    dim: Dimensions,
) -> Result<()> {
    let mut expected = BTreeSet::new();
    add(&mut expected, format!("{prefix}fc.weight"))?;
    add(&mut expected, format!("{prefix}hidden_norm.weight"))?;
    add(&mut expected, format!("{prefix}norm.weight"))?;

    let groups = dim
        .hidden
        .checked_div(16)
        .ok_or_else(|| anyhow::anyhow!("official DFlash2 conv group division failed"))?;
    let projection_rows = 2usize
        .checked_mul(2)
        .and_then(|value| value.checked_mul(groups))
        .ok_or_else(|| anyhow::anyhow!("official DFlash2 conv projection width overflow"))?;

    for layer in 0..config.num_hidden_layers {
        let lp = format!("{prefix}layers.{layer}");
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            add(&mut expected, format!("{lp}.{suffix}"))?;
        }
        for stem in ["attention_conv", "mlp_conv"] {
            let base = format!("{lp}.{stem}.base_kernel");
            let projection = format!("{lp}.{stem}.kernel_projection.weight");
            add(&mut expected, base.clone())?;
            add(&mut expected, projection.clone())?;
            require_tensor(source, &base, &[2, 2, dim.hidden])?;
            require_tensor(source, &projection, &[projection_rows, dim.hidden])?;
        }
    }

    let hidden_projection = format!("{prefix}candidate_selector.hidden_projection.weight");
    let predecessor = format!("{prefix}candidate_selector.predecessor_codebook");
    let successor = format!("{prefix}candidate_selector.successor_codebook");
    for name in [&hidden_projection, &predecessor, &successor] {
        add(&mut expected, name.as_str().to_owned())?;
    }
    require_tensor(source, &hidden_projection, &[256, dim.hidden])?;
    require_tensor(source, &predecessor, &[config.vocab_size, 256])?;
    require_tensor(source, &successor, &[config.vocab_size, 256])?;

    ensure!(
        expected.len() == OFFICIAL_TENSOR_COUNT,
        "internal official DFlash2 tensor census is not {OFFICIAL_TENSOR_COUNT}"
    );
    let observed = source.names().map(str::to_owned).collect::<BTreeSet<_>>();
    ensure!(
        observed == expected,
        "official Qwen3.8 DFlash2 requires the exact {OFFICIAL_TENSOR_COUNT}-tensor census; \
         missing={:?}, extra={:?}",
        expected.difference(&observed).collect::<Vec<_>>(),
        observed.difference(&expected).collect::<Vec<_>>()
    );
    Ok(())
}

fn add(expected: &mut BTreeSet<String>, name: String) -> Result<()> {
    ensure!(
        expected.insert(name.clone()),
        "duplicate official DFlash2 tensor name `{name}`"
    );
    Ok(())
}
