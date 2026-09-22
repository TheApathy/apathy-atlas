// SPDX-License-Identifier: AGPL-3.0-only

//! `load_layers` for DeepSeek-V4.1-Flash-Next — the seam the loader used to stop at.
//!
//! Order matters and is deliberate:
//!   1. **Admit** the indexer tensors, so a layout disagreement fails before any allocation.
//!   2. **Parse** the CB3 manifest and CHECK residency against the real budget. On GB10 a
//!      GPU over-allocation takes the HOST down, so nothing is allocated until the whole
//!      plan is known to fit.
//!   3. **Upload** the pack once, shared across all 40 layers.
//!   4. **Build** the layers.
//!
//! Doing (3) before (1) would spend several minutes and 71.7 GB before discovering a name
//! mismatch; doing (3) before (2) is the over-allocation that kills the box.

use anyhow::{Context, Result, ensure};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use atlas_core::config::{ExpertPack, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, dense, dense_auto};

use super::cb3_arena::{Cb3ExpertArena, resolve_packed_keep};
use super::layer::{
    DeepSeekV41Layer, V41HcAttn, V41HyperConnections, V41SharedExpert,
};
use super::moe::{Cb3Reconstruct, expert_matrices};
use super::moe_forward::RouterF32;
use super::seams::{ENGRAM_LAYERS, MissingAttention, MissingEngram, SparseShared};

/// Where the CB3 pack lives, relative to the model directory.
pub const PACK_SUBDIR: &str = "k154-cb3";
/// Environment override for the model directory, when the store does not expose it.
pub const MODEL_DIR_ENV: &str = "ATLAS_DSV41_MODEL_DIR";

pub fn load_all_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let n = config.num_hidden_layers;
    tracing::info!(
        "DeepSeek-V4.1 load_layers: {n} layers, hidden {}, moe_inter {}, router space {}, \
         q_heads {} / kv_heads {} (MQA), q_lora {}, head_dim {}",
        config.hidden_size,
        config.moe_intermediate_size,
        config.num_experts,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.q_lora_rank,
        config.head_dim,
    );

    // (1) ADMISSION FIRST — cheap, and it fails on a layout disagreement before anything
    // is allocated. V4.1 needs its OWN rule; `deepseek_v4::indexer` is wrong here in four
    // ways and reports a misleading "Unexpected indexer tensor". See `super::indexer`.
    let admission = super::indexer::admit_indexer_weights(store, config)?;

    // (2) THE PLAN, CHECKED BEFORE IT IS SPENT.
    let model_dir = resolve_model_dir()?;
    let pack_dir = model_dir.join(PACK_SUBDIR);
    let manifest_path = pack_dir.join("manifest.json");
    let manifest = std::fs::read_to_string(&manifest_path).with_context(|| {
        format!(
            "DeepSeek-V4.1 expert pack manifest not readable at {}. The routed experts are \
             NOT in the main safetensors shards — they live in the `{PACK_SUBDIR}` pack. Set \
             {MODEL_DIR_ENV} if the model directory is elsewhere.",
            manifest_path.display()
        )
    })?;
    let packed_keep = resolve_packed_keep()?;
    let pack = ExpertPack::parse(&manifest, packed_keep)?;
    ensure!(
        pack.num_layers() == n,
        "CB3 pack covers {} layers but the config declares {n}",
        pack.num_layers()
    );

    // (3) RESIDENCY. One arena, shared by every layer.
    let arena = Arc::new(Cb3ExpertArena::load(&pack_dir, &pack, gpu)?);

    // (4) LAYERS.
    let reconstruct = Cb3Reconstruct::new(gpu)?;
    // ONE SparseShared for the whole model, cloned into every layer. Layer 0 resets it at
    // the top of each forward.
    let sparse = Arc::new(Mutex::new(SparseShared::default()));
    let matrices = expert_matrices(config)?;
    let attention_loader = super::seams::attention_loader();
    let engram_loader = super::seams::engram_loader();
    if attention_loader.is_none() {
        tracing::warn!(
            "DeepSeek-V4.1: no attention implementation registered — layers will LOAD but \
             refuse to run. OWNER: dsv41-attention."
        );
    }
    if engram_loader.is_none() {
        tracing::warn!(
            "DeepSeek-V4.1: no engram implementation registered — layers {ENGRAM_LAYERS:?} \
             will LOAD but refuse to run. OWNER: dsv41-engram."
        );
    }

    let fwd_kernels = super::ops::Dsv41Kernels::load(gpu)?;
    let stream = gpu.default_stream();
    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(n);
    for i in 0..n {
        let lp = format!("layers.{i}");
        let kv_dtype = layer_kv_dtypes.get(i).copied().unwrap_or(KvCacheDtype::Bf16);

        let input_norm = dense_auto(store, &format!("{lp}.attn_norm.weight"), gpu)?;
        let post_attn_norm = dense_auto(store, &format!("{lp}.ffn_norm.weight"), gpu)?;

        // The router stays in the reference's dtypes: weight widened to fp32, both biases F32
        // (never through `dense_auto`, which narrows F32 to bf16). See `RouterF32`.
        let router = RouterF32::load(store, i, config.hidden_size, gpu, &fwd_kernels, stream)?;

        // The shared expert is FP8 block-quantised IN THE MAIN SHARDS, not CB3. Its
        // `.scale` siblings are loaded alongside because an FP8 weight without its scale
        // is not a weight.
        let shared = format!("{lp}.ffn.shared_experts");
        let shared_expert = V41SharedExpert {
            w1: dense(store, &format!("{shared}.w1.weight"))?,
            w1_scale: dense(store, &format!("{shared}.w1.scale"))?,
            w2: dense(store, &format!("{shared}.w2.weight"))?,
            w2_scale: dense(store, &format!("{shared}.w2.scale"))?,
            w3: dense(store, &format!("{shared}.w3.weight"))?,
            w3_scale: dense(store, &format!("{shared}.w3.scale"))?,
        };

        // PER-LAYER mHC, both sides, on EVERY layer.
        //
        // V4-Flash-0731 has ONE model-level `hc_head` replicated to all layers; V4.1 has
        // `hc_attn_*` and `hc_ffn_*` per layer. Counted against the index: 43 `hc_attn_fn`
        // tensors = 40 layers + 3 MTP modules, layer 0 included.
        //
        // An earlier draft of this loader asserted layer 0 had NO attention-side mHC and
        // would have refused the real checkpoint on its first layer. That belief came from
        // a listing filtered on the substring "attn", which silently dropped `hc_attn_*`
        // along with the attention tensors it meant to exclude. The assertion below is the
        // corrected one and is checked against the artifact in this module's tests.
        let hyper_connections = V41HyperConnections {
            ffn_fn: dense(store, &format!("{lp}.hc_ffn_fn"))?,
            ffn_base: dense(store, &format!("{lp}.hc_ffn_base"))?,
            ffn_scale: dense(store, &format!("{lp}.hc_ffn_scale"))?,
            attn: Some(V41HcAttn {
                attn_fn: dense(store, &format!("{lp}.hc_attn_fn"))?,
                base: dense(store, &format!("{lp}.hc_attn_base"))?,
                scale: dense(store, &format!("{lp}.hc_attn_scale"))?,
            }),
        };

        let attention = match attention_loader {
            Some(loader) => loader.load_attention(i, store, config, gpu, kv_dtype)?,
            None => Box::new(MissingAttention { layer: i }),
        };

        // `None` for the 38 layers without an engram is the CORRECT answer, and the one
        // place in this port where an empty success is not a silent gap.
        let engram: Option<Box<dyn super::seams::Dsv41Engram>> = match engram_loader {
            Some(loader) => loader.load_engram(i, &model_dir, config, gpu)?,
            None if ENGRAM_LAYERS.contains(&i) => Some(Box::new(MissingEngram { layer: i })),
            None => None,
        };

        layers.push(Box::new(DeepSeekV41Layer {
            layer: i,
            input_norm,
            post_attn_norm,
            router,
            shared_expert,
            hyper_connections,
            arena: Arc::clone(&arena),
            reconstruct,
            expert_matrices: matrices,
            attention,
            sparse: Arc::clone(&sparse),
            engram,
        }));
    }

    tracing::info!(
        "DeepSeek-V4.1: {n} layers built. Experts resident: {:.1} GB at packed_keep={}. \
         Indexer: {} query layers, {} key layers. Forward is NOT runnable yet — attention, \
         engram and routing/combine are open.",
        arena.resident_bytes() as f64 / 1e9,
        arena.packed_keep(),
        admission.index_source_layers.len(),
        admission.kv_source_layers.len(),
    );
    Ok(layers)
}

/// Locate the model directory — the CB3 pack and the engram tables both hang off it.
///
/// The engram tables are ~95 GB each and are never loaded into the `WeightStore`, so the
/// engram seam takes this path rather than a store. See `seams::Dsv41EngramLoader`.
pub fn resolve_model_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var(MODEL_DIR_ENV) {
        return Ok(PathBuf::from(dir));
    }
    // The default is the local checkpoint. This is a development default, not a
    // deployment one: a served build should set MODEL_DIR_ENV.
    let default = PathBuf::from("/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K");
    ensure!(
        default.join(PACK_SUBDIR).is_dir(),
        "DeepSeek-V4.1 expert pack not found at {} and {MODEL_DIR_ENV} is unset",
        default.join(PACK_SUBDIR).display()
    );
    Ok(default)
}

/// Unused-import guard: `DenseWeight` is named in the struct fields this module builds.
#[allow(dead_code)]
type _DenseWeightIsUsed = DenseWeight;

#[cfg(test)]
mod tests {

    const INDEX: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K\
/model.safetensors.index.json";

    fn weight_map() -> Option<serde_json::Map<String, serde_json::Value>> {
        let raw = std::fs::read_to_string(INDEX).ok()?;
        let index: serde_json::Value = serde_json::from_str(&raw).ok()?;
        index["weight_map"].as_object().cloned()
    }

    /// Every per-layer key this loader builds must EXIST in the shipped index.
    ///
    /// Built by string formatting, so a typo is invisible until a serve. Checked against
    /// the artifact for all 40 layers rather than a sample.
    #[test]
    fn every_per_layer_key_resolves_against_the_shipped_index() {
        let Some(map) = weight_map() else {
            eprintln!("skipping: {INDEX} not present");
            return;
        };
        for i in 0..40 {
            let lp = format!("layers.{i}");
            let mut required = vec![
                format!("{lp}.attn_norm.weight"),
                format!("{lp}.ffn_norm.weight"),
                format!("{lp}.ffn.gate.weight"),
                format!("{lp}.ffn.gate.bias"),
                format!("{lp}.ffn.gate.bias_vl"),
                format!("{lp}.hc_ffn_fn"),
                format!("{lp}.hc_ffn_base"),
                format!("{lp}.hc_ffn_scale"),
            ];
            for w in ["w1", "w2", "w3"] {
                required.push(format!("{lp}.ffn.shared_experts.{w}.weight"));
                required.push(format!("{lp}.ffn.shared_experts.{w}.scale"));
            }
            if i != 0 {
                required.push(format!("{lp}.hc_attn_fn"));
                required.push(format!("{lp}.hc_attn_base"));
                required.push(format!("{lp}.hc_attn_scale"));
            }
            for key in required {
                assert!(map.contains_key(&key), "{key} is not in the shipped index");
            }
        }
    }

    /// Every layer carries BOTH mHC sides — there is no layer-0 exception.
    ///
    /// This test previously asserted the opposite and FAILED against the checkpoint, which
    /// is how the loader's wrong `hc_attn present == (layer != 0)` guard was found before
    /// it could reject layer 0 of the real model. Kept as the pin on the corrected claim.
    #[test]
    fn every_layer_carries_both_mhc_sides() {
        let Some(map) = weight_map() else {
            eprintln!("skipping: {INDEX} not present");
            return;
        };
        for i in 0..40 {
            for suffix in ["hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                           "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"] {
                assert!(
                    map.contains_key(&format!("layers.{i}.{suffix}")),
                    "layers.{i}.{suffix} must be present — there is no layer-0 exception"
                );
            }
        }
        // 43 = 40 layers + 3 MTP modules. Pins the count so a checkpoint that drops a
        // layer's mHC fails here rather than at a serve.
        let attn_fn = map.keys().filter(|k| k.ends_with(".hc_attn_fn")).count();
        assert_eq!(attn_fn, 43, "40 layers + 3 MTP modules carry hc_attn_fn");
    }

    /// The routed experts are NOT in the main shards. If they were, the pack would be
    /// redundant and `resolve_pack_dir` would be dead weight.
    #[test]
    fn routed_experts_are_absent_from_the_main_shards() {
        let Some(map) = weight_map() else {
            eprintln!("skipping: {INDEX} not present");
            return;
        };
        // Scoped to `layers.*` ON PURPOSE. The MTP modules DO ship per-expert FP8 tensors
        // in the main shards (`mtp.0.ffn.experts.0.w1.weight`, ...); an unscoped filter
        // matches those and fails. That is a real difference between the main stack and
        // the MTP stack, not noise to filter away, and MTP loading is still unimplemented.
        let routed: Vec<&String> = map
            .keys()
            .filter(|k| k.starts_with("layers."))
            .filter(|k| k.contains(".ffn.experts.") || k.contains(".ffn.routed_experts."))
            .collect();
        assert!(
            routed.is_empty(),
            "routed experts must live in the CB3 pack, not the shards; found {routed:?}"
        );
        // Shared experts, by contrast, ARE in the shards — so the filter above is not
        // simply matching nothing by accident.
        assert!(map.contains_key("layers.0.ffn.shared_experts.w1.weight"));
        // And the MTP stack genuinely has in-shard routed experts, which is why the
        // `layers.` scoping above is load-bearing rather than cosmetic.
        assert!(
            map.contains_key("mtp.0.ffn.experts.0.w1.weight"),
            "MTP routed experts live in the shards; if that changes, re-examine the scope"
        );
    }
}
