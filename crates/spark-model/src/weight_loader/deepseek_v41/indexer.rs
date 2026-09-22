// SPDX-License-Identifier: AGPL-3.0-only

//! **V4.1 indexer weight admission.** Metadata only — no device I/O, no conversion.
//!
//! ## Why this is not `deepseek_v4::indexer`
//! That module is correct for V4-Flash-0731 and wrong for V4.1 in four independent ways,
//! each verified against the shipped index rather than read off the source:
//!
//! | | V4-Flash-0731 | V4.1-Flash-Next |
//! |---|---|---|
//! | which layers | `compress_ratios[l] == 4` | `index_source_layer_ids` = {2,8,14,20,24,28,32,36} |
//! | quant block | 128 | **32** (`wq_b.scale` is `[128, 40]` for a `[4096, 1280]` weight) |
//! | k-side tensors | none | `wk.weight`, `k_norm.weight` on `kv_source_layer_ids` only |
//! | compressor | under `attn.indexer.` | under `attn.` (a sibling, not a child) |
//!
//! V4.1's `compress_ratios` are 0/2/1 and never 4, so the V4 loop admits nothing and
//! `expected` stays empty — at which point its trailing sweep reaches
//! `layers.2.attn.indexer.wq_b.weight` and reports **"Unexpected DeepSeek indexer
//! tensor"**. The tensor is not unexpected; the admission RULE is V4-only. A confidently
//! wrong error message costs as much debugging time as a silent gap, so V4.1 gets its own
//! rule rather than a loosened shared one.
//!
//! ## The layer sets come from the STORE, not from the config
//! `index_source_layer_ids`, `kv_source_layer_ids` and `candidate_source_layer_id` are in
//! `config.json` but are **not fields on `ModelConfig`** (checked: no parser reads them).
//! Rather than widen a shared config struct that `dsv41-attention` is also working in,
//! this derives the sets from which layers actually carry the tensors, then asserts the
//! result against the checkpoint's declared values. If a future checkpoint disagrees, the
//! assertion fires here instead of the attention path quietly indexing the wrong layers.

use anyhow::{Context, Result, ensure};
use std::collections::{BTreeSet, HashSet};

use atlas_core::config::ModelConfig;
use spark_runtime::weights::{WeightDtype, WeightStore};

/// V4.1's FP8 weight-quantisation block. **32, not 128.**
///
/// Read off the artifact: `layers.2.attn.indexer.wq_b.weight` is `[4096, 1280]` and its
/// `.scale` is `[128, 40]`. 4096/128 = 32 and 1280/40 = 32. Using the V4 constant of 128
/// would expect `[32, 10]` and reject a correct checkpoint.
const WEIGHT_BLOCK: usize = 32;

/// Layers that recompute a top-k selection, from the checkpoint's
/// `text_config.index_source_layer_ids`. The other 32 layers attend with their
/// predecessor's selection.
pub const INDEX_SOURCE_LAYERS: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];
/// Layers that build compressed KV and index KEYS, from `text_config.kv_source_layer_ids`.
/// Only these carry `indexer.wk` / `indexer.k_norm`; 3-7 reuse 2's, 9-13 reuse 8's,
/// 21-39 reuse 20's.
pub const KV_SOURCE_LAYERS: [usize; 4] = [2, 8, 14, 20];

/// What admission found, for the attention lane to consume and for startup logging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexerAdmission {
    /// Layers carrying a query-side indexer, ascending.
    pub index_source_layers: Vec<usize>,
    /// Layers carrying the key-side indexer tensors, ascending.
    pub kv_source_layers: Vec<usize>,
    /// `index_n_heads * index_head_dim`, the indexer query width.
    pub query_width: usize,
}

/// Admit V4.1's indexer tensors, or fail naming the exact disagreement.
///
/// Returns the derived layer sets. **Never returns an empty success**: a V4.1 checkpoint
/// with no indexer tensors is not a valid V4.1 checkpoint, and reporting `Ok` for one
/// would hand the attention lane a model that silently attends to everything.
pub fn admit_indexer_weights(store: &WeightStore, config: &ModelConfig) -> Result<IndexerAdmission> {
    let indexer = config.deepseek_v4_indexer.as_ref().context(
        "DeepSeek-V4.1 has no typed indexer config. The checkpoint ships 32 \
         `attn.indexer.*` tensors over 8 layers, so a missing config is a parser defect, \
         not a model without an indexer.",
    )?;

    let query_width = indexer
        .num_heads
        .checked_mul(indexer.head_dim)
        .context("Indexer query width overflow")?;
    let hidden = config.hidden_size;
    let q_rank = config.q_lora_rank;
    let kv_lora = config.kv_lora_rank;

    // Derive the layer sets from what is actually present.
    let mut index_source: BTreeSet<usize> = BTreeSet::new();
    let mut kv_source: BTreeSet<usize> = BTreeSet::new();
    for name in store.names() {
        let Some(rest) = name.strip_prefix("layers.") else {
            continue;
        };
        let Some((layer_text, suffix)) = rest.split_once(".attn.indexer.") else {
            continue;
        };
        let layer: usize = layer_text
            .parse()
            .with_context(|| format!("Indexer tensor {name} has a non-numeric layer index"))?;
        ensure!(
            layer < config.num_hidden_layers,
            "Indexer tensor {name} names layer {layer}, beyond the model's {}",
            config.num_hidden_layers
        );
        index_source.insert(layer);
        if suffix.starts_with("wk.") || suffix.starts_with("k_norm.") {
            kv_source.insert(layer);
        }
    }

    ensure!(
        !index_source.is_empty(),
        "DeepSeek-V4.1: no `attn.indexer.*` tensors in the weight store. The checkpoint \
         must carry 32 of them over layers {INDEX_SOURCE_LAYERS:?}; a model without them \
         would attend without sparse selection and produce plausible, wrong output."
    );

    // Pin the derivation against the checkpoint's declared values. These constants come
    // from `text_config.index_source_layer_ids` / `kv_source_layer_ids`, which are NOT on
    // `ModelConfig` — see the module note.
    let derived_index: Vec<usize> = index_source.iter().copied().collect();
    let derived_kv: Vec<usize> = kv_source.iter().copied().collect();
    ensure!(
        derived_index == INDEX_SOURCE_LAYERS,
        "DeepSeek-V4.1 indexer layers are {derived_index:?}, but this checkpoint family \
         declares index_source_layer_ids = {INDEX_SOURCE_LAYERS:?}. The attention path \
         inherits selections along this exact chain, so a mismatch is a different model."
    );
    ensure!(
        derived_kv == KV_SOURCE_LAYERS,
        "DeepSeek-V4.1 indexer key layers are {derived_kv:?}, expected \
         kv_source_layer_ids = {KV_SOURCE_LAYERS:?}."
    );

    // Now check every tensor's dtype and shape.
    let mut expected: HashSet<String> = HashSet::new();
    for &layer in &derived_index {
        let mut specifications: Vec<(&str, WeightDtype, Vec<usize>)> = vec![
            ("wq_b.weight", WeightDtype::FP8E4M3, vec![query_width, q_rank]),
            (
                "wq_b.scale",
                WeightDtype::FP8E8M0,
                vec![query_width / WEIGHT_BLOCK, q_rank / WEIGHT_BLOCK],
            ),
            (
                "weights_proj.weight",
                WeightDtype::BF16,
                vec![indexer.num_heads, hidden],
            ),
        ];
        if derived_kv.contains(&layer) {
            specifications.push(("wk.weight", WeightDtype::BF16, vec![indexer.head_dim, kv_lora]));
            specifications.push(("k_norm.weight", WeightDtype::BF16, vec![indexer.head_dim]));
        }
        for (suffix, dtype, shape) in specifications {
            let name = format!("layers.{layer}.attn.indexer.{suffix}");
            admit_tensor(store, &name, dtype, &shape)?;
            expected.insert(name);
        }
    }

    // Nothing indexer-shaped may be left over. An unaccounted tensor means the layout
    // moved and something downstream is reading a stale name.
    for name in store.names().filter(|name| name.contains(".attn.indexer.")) {
        ensure!(
            expected.contains(name),
            "Unaccounted DeepSeek-V4.1 indexer tensor: {name}"
        );
    }

    tracing::info!(
        "DeepSeek-V4.1 indexer admitted: {} tensors over index layers {:?} (key layers {:?}), \
         query width {} = {} heads x {}, fp8 block {}",
        expected.len(),
        derived_index,
        derived_kv,
        query_width,
        indexer.num_heads,
        indexer.head_dim,
        WEIGHT_BLOCK,
    );

    Ok(IndexerAdmission {
        index_source_layers: derived_index,
        kv_source_layers: derived_kv,
        query_width,
    })
}

fn admit_tensor(store: &WeightStore, name: &str, dtype: WeightDtype, shape: &[usize]) -> Result<()> {
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
    ensure!(
        tensor.ptr.0 != 0,
        "Indexer tensor {name}: null address"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const INDEX: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K\
/model.safetensors.index.json";
    const CONFIG: &str =
        "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/config.json";

    fn indexer_keys() -> Option<Vec<String>> {
        let raw = std::fs::read_to_string(INDEX).ok()?;
        let index: serde_json::Value = serde_json::from_str(&raw).ok()?;
        Some(
            index["weight_map"]
                .as_object()?
                .keys()
                .filter(|name| name.contains(".attn.indexer."))
                .cloned()
                .collect(),
        )
    }

    /// The layer sets must match the ARTIFACT, not my expectation of it.
    #[test]
    fn the_derived_layer_sets_match_the_shipped_index() {
        let Some(keys) = indexer_keys() else {
            eprintln!("skipping: {INDEX} not present");
            return;
        };
        assert_eq!(keys.len(), 32, "V4.1 ships 32 indexer tensors");

        let mut index_source = BTreeSet::new();
        let mut kv_source = BTreeSet::new();
        for name in &keys {
            let (layer_text, suffix) = name
                .strip_prefix("layers.")
                .unwrap()
                .split_once(".attn.indexer.")
                .unwrap();
            let layer: usize = layer_text.parse().unwrap();
            index_source.insert(layer);
            if suffix.starts_with("wk.") || suffix.starts_with("k_norm.") {
                kv_source.insert(layer);
            }
        }
        assert_eq!(
            index_source.iter().copied().collect::<Vec<_>>(),
            INDEX_SOURCE_LAYERS
        );
        assert_eq!(
            kv_source.iter().copied().collect::<Vec<_>>(),
            KV_SOURCE_LAYERS
        );
        // The compressor is a SIBLING of indexer in V4.1, not a child. The V4 admission
        // looks for `attn.indexer.compressor.*`, which does not exist here.
        assert!(
            !keys.iter().any(|k| k.contains("indexer.compressor")),
            "V4.1's compressor is at attn.compressor.*, not under attn.indexer."
        );
    }

    /// The V4 rule must be shown to FAIL on this checkpoint — otherwise "V4.1 needs its
    /// own admission" is an assertion, not a finding.
    ///
    /// The V4 rule is `compress_ratios[layer] == 4`. Replaying it over the real ratios must
    /// select ZERO layers, while indexer tensors demonstrably exist. That combination is
    /// exactly what drives its trailing sweep into "Unexpected DeepSeek indexer tensor".
    #[test]
    fn the_v4_admission_rule_selects_no_layers_on_this_checkpoint() {
        let Ok(raw) = std::fs::read_to_string(CONFIG) else {
            eprintln!("skipping: {CONFIG} not present");
            return;
        };
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let ratios: Vec<i64> = json["text_config"]["compress_ratios"]
            .as_array()
            .expect("compress_ratios")
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();

        let v4_selected: Vec<usize> = ratios
            .iter()
            .take(40)
            .enumerate()
            .filter(|(_, ratio)| **ratio == 4)
            .map(|(layer, _)| layer)
            .collect();
        assert!(
            v4_selected.is_empty(),
            "the V4 rule was expected to select nothing here, but chose {v4_selected:?}"
        );
        // And the tensors it would then call "unexpected" are present.
        if let Some(keys) = indexer_keys() {
            assert!(!keys.is_empty());
        }
        // Meanwhile OUR rule selects the eight real ones.
        assert_eq!(INDEX_SOURCE_LAYERS.len(), 8);
        assert!(INDEX_SOURCE_LAYERS.iter().all(|l| ratios[*l] != 4));
    }

    /// The fp8 block is 32 here; the V4 constant of 128 would reject a correct checkpoint.
    #[test]
    fn the_weight_block_is_thirty_two_not_one_twenty_eight() {
        // From the artifact: wq_b.weight [4096, 1280], wq_b.scale [128, 40].
        assert_eq!(4096 / WEIGHT_BLOCK, 128);
        assert_eq!(1280 / WEIGHT_BLOCK, 40);
        // NEGATIVE CONTROL: the V4 block would predict a different, wrong scale shape.
        assert_ne!(4096 / 128, 128);
        assert_ne!(1280 / 128, 40);
    }
}
