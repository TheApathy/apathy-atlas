// SPDX-License-Identifier: AGPL-3.0-only
//! Borrowed native-indexer metadata admission only: no device I/O, conversion,
//! persistent state, or attention activation. Store allocation ownership stays unchanged.
//! Verified against Vision EXL3-K2 revision c171bea574201ff25530256fbd63626c7fd20f3c.
use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::weights::{WeightDtype, WeightStore};
use std::collections::HashSet;

// Official model.py block_size=128 is the weight quantization block, not KV K64.
const WEIGHT_BLOCK: usize = 128;

pub(super) fn admit_indexer_weights(store: &WeightStore, config: &ModelConfig) -> Result<usize> {
    let Some(indexer) = &config.deepseek_v4_indexer else {
        ensure!(
            !store.names().any(|name| name.contains(".attn.indexer.")),
            "DeepSeek indexer tensors exist without a typed indexer config"
        );
        return Ok(0);
    };
    indexer.validate_model(config)?;
    let q_width = indexer
        .num_heads
        .checked_mul(indexer.head_dim)
        .context("Indexer query width overflow")?;
    let compressor_width = indexer
        .head_dim
        .checked_mul(2)
        .context("Indexer overlap compressor width overflow")?;
    let hidden = config.hidden_size;
    let q_rank = config.q_lora_rank;
    let mut expected = HashSet::new();
    let mut admitted = 0;
    for (layer, ratio) in config
        .compress_ratios
        .iter()
        .take(config.num_hidden_layers)
        .enumerate()
    {
        if *ratio != 4 {
            continue;
        }
        // A native ratio-four indexer has its OWN overlap compressor. These
        // tensors must not alias names of the attention compressor by fallback.
        let specifications = [
            ("wq_b.weight", WeightDtype::FP8E4M3, vec![q_width, q_rank]),
            (
                "wq_b.scale",
                WeightDtype::FP8E8M0,
                vec![
                    q_width.div_ceil(WEIGHT_BLOCK),
                    q_rank.div_ceil(WEIGHT_BLOCK),
                ],
            ),
            (
                "weights_proj.weight",
                WeightDtype::BF16,
                vec![indexer.num_heads, hidden],
            ),
            (
                "compressor.wkv.weight",
                WeightDtype::BF16,
                vec![compressor_width, hidden],
            ),
            (
                "compressor.wgate.weight",
                WeightDtype::BF16,
                vec![compressor_width, hidden],
            ),
            (
                "compressor.norm.weight",
                WeightDtype::BF16,
                vec![indexer.head_dim],
            ),
            (
                "compressor.ape",
                WeightDtype::FP32,
                vec![*ratio, compressor_width],
            ),
        ];
        for (suffix, dtype, shape) in specifications {
            let name = format!("layers.{layer}.attn.indexer.{suffix}");
            admit_tensor(store, &name, dtype, &shape)?;
            expected.insert(name);
        }
        admitted += 1;
    }
    for name in store.names().filter(|name| name.contains(".attn.indexer.")) {
        ensure!(
            expected.contains(name),
            "Unexpected DeepSeek indexer tensor: {name}"
        );
    }
    Ok(admitted)
}

fn admit_tensor(
    store: &WeightStore,
    name: &str,
    dtype: WeightDtype,
    shape: &[usize],
) -> Result<()> {
    let tensor = store.get(name)?;
    ensure!(
        tensor.dtype == dtype,
        "Indexer tensor {name}: expected {dtype:?}, got {:?}",
        tensor.dtype
    );
    ensure!(
        tensor.shape.as_slice() == shape,
        "Indexer tensor {name}: expected shape {shape:?}, got {:?}",
        tensor.shape
    );
    let bytes = shape
        .iter()
        .try_fold(dtype.byte_size(), |bytes, dimension| {
            bytes
                .checked_mul(*dimension)
                .context("Indexer tensor byte extent overflow")
        })?;
    ensure!(
        bytes > 0 && bytes <= isize::MAX as usize,
        "Indexer tensor {name}: invalid extent"
    );
    let alignment = u64::try_from(dtype.byte_size()).context("Indexer alignment overflow")?;
    ensure!(
        tensor.ptr.0 != 0 && tensor.ptr.0 % alignment == 0,
        "Indexer tensor {name}: null or misaligned address"
    );
    tensor
        .ptr
        .0
        .checked_add(u64::try_from(bytes).context("Indexer tensor address extent overflow")?)
        .with_context(|| format!("Indexer tensor {name}: address overflow"))?;
    // WeightStore does not expose its allocation's physical length. This is
    // metadata arithmetic validation, not a proof of allocation or payload values.
    Ok(())
}
