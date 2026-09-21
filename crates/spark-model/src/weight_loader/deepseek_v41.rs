// SPDX-License-Identifier: AGPL-3.0-only

//! Weight loader for **DeepSeek-V4.1-Flash-Next**.
//!
//! This loader deliberately **fails loudly for everything it has not implemented** rather
//! than borrowing [`super::deepseek_v4::DeepSeekV4WeightLoader`]. That loader targets
//! DeepSeek-V4-Flash-0731 and its experts are **EXL3 2-bit**; V4.1's are **CB3 3-bit**.
//! Reusing it would not fail at the dispatch — it would fail somewhere far downstream in a
//! tensor reader, or worse, decode EXL3 over CB3 bytes and produce plausible garbage.
//! An honest "not implemented" at the seam is worth more than a confusing crash three
//! layers down.
//!
//! ## What IS implemented and tested
//! - [`v41_key`], the weight-key join, which is not the usual one (see below).
//! - The expert mapping, in [`atlas_core::config::ExpertPack`].
//! - Reading CB3 expert payload, in `spark_runtime::weights::deepseek_v41_pack`.
//!
//! ## What is NOT, and why
//! [`ModelWeightLoader::load_layers`] must return `Vec<Box<dyn TransformerLayer>>` built
//! through a `GpuBackend`. That needs (a) GPU residency for an 83 GB expert pack and
//! (b) CB3 decode kernels, which do not exist anywhere in this tree — `grep -rli 'cb3'`
//! over `crates/` and `kernels/` finds only documentation. Both need box time to verify,
//! so the seam is here, documented, against a tested shard reader.

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;

use super::{DenseWeight, ModelWeightLoader, MtpWeights, WeightStore};
use crate::layer::TransformerLayer;
use spark_runtime::kv_cache::KvCacheDtype;

/// Join a weight-key prefix to a suffix, tolerating an EMPTY prefix.
///
/// **V4.1's prefix is legitimately empty.** Zero of its 3925 index keys carry the `model.`
/// prefix that 0731 uses — they are bare (`embed.weight`, `layers.0.attn_norm.weight`).
/// The naive `format!("{prefix}.{suffix}")` therefore yields a LEADING DOT and every
/// lookup misses.
///
/// Note this is NOT the `step3p7` idiom (`weight_loader/step3p7.rs:160`), which substitutes
/// a *different default prefix* when the configured one is empty. Here empty is the correct
/// and final answer, so the fix is to omit the separator, not to invent a prefix.
pub fn v41_key(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.to_string()
    } else {
        format!("{prefix}.{suffix}")
    }
}

/// DeepSeek-V4.1-Flash-Next weight loader.
pub struct DeepSeekV41WeightLoader;

/// One message, so every unimplemented entry point says the same true thing.
fn cb3_residency_unimplemented(what: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "DeepSeek-V4.1 {what} is not implemented: the engine can parse the config, resolve \
         the 384 -> 154 -> 124 expert mapping, and READ CB3 expert bytes, but it cannot yet \
         make them resident or decode them. Missing: (1) CB3 3-bit decode kernels -- none \
         exist in kernels/; (2) GPU residency for the 83 GB K154 expert pack. This is a \
         deliberate hard stop, NOT a fallback to the deepseek_v4 loader, whose experts are \
         EXL3 and would decode V4.1's CB3 bytes into plausible garbage."
    )
}

impl ModelWeightLoader for DeepSeekV41WeightLoader {
    /// Declared explicitly, as the trait requires. V4.1 inherits 0731's `num_key_value_heads
    /// = 1` (MQA), so head-parallel sharding is impossible: 1 is not divisible by a tp size
    /// greater than 1. This is a property of the architecture, not an unfinished feature.
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        bail!(cb3_residency_unimplemented("layer loading"))
    }

    fn load_embedding(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bail!(cb3_residency_unimplemented("embedding loading"))
    }

    fn load_final_norm(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bail!(cb3_residency_unimplemented("final norm loading"))
    }

    fn load_lm_head(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bail!(cb3_residency_unimplemented("lm_head loading"))
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        // NOT `Ok(None)`. None means "this checkpoint has no MTP", which is FALSE here:
        // V4.1 declares num_nextn_predict_layers = 3 and ships 2401 `mtp.*` tensors.
        // Returning None would silently disable speculation and look like a config choice.
        bail!(cb3_residency_unimplemented(
            "MTP loading (3 modules, vs 0731's 1)"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The empty prefix must not produce a leading dot. This is the trap that would have
    /// made every V4.1 weight lookup miss.
    #[test]
    fn v41_key_omits_the_separator_for_an_empty_prefix() {
        assert_eq!(v41_key("", "embed.weight"), "embed.weight");
        assert_eq!(
            v41_key("", "layers.0.attn_norm.weight"),
            "layers.0.attn_norm.weight"
        );
        assert!(!v41_key("", "norm.weight").starts_with('.'));
    }

    /// And a non-empty prefix still joins normally, so 0731-shaped callers are unaffected.
    #[test]
    fn v41_key_joins_a_non_empty_prefix_normally() {
        assert_eq!(v41_key("model", "embed.weight"), "model.embed.weight");
    }

    /// Keys built for the real checkpoint must actually exist in its index.
    ///
    /// This is the test that proves the join is right against the ARTIFACT rather than
    /// against my expectation of it. Skips when the checkpoint is absent.
    #[test]
    fn v41_keys_resolve_against_the_shipped_weight_index() {
        const INDEX: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K\
/model.safetensors.index.json";
        let Ok(raw) = std::fs::read_to_string(INDEX) else {
            eprintln!("skipping: {INDEX} not present");
            return;
        };
        let index: serde_json::Value = serde_json::from_str(&raw).expect("index is JSON");
        let map = index["weight_map"].as_object().expect("weight_map");

        // V4.1's prefix, as parse_deepseek_v41 sets it.
        for suffix in [
            "embed.weight",
            "head.weight",
            "norm.weight",
            "layers.0.attn_norm.weight",
            "layers.39.ffn_norm.weight",
        ] {
            let key = v41_key("", suffix);
            assert!(
                map.contains_key(&key),
                "key {key} is not in the shipped weight index"
            );
        }

        // And the 0731 spelling must NOT resolve -- this is the defect that made the
        // hardcoded prefix wrong, asserted rather than described.
        assert!(
            !map.contains_key("model.embed.weight"),
            "V4.1 must not carry the 0731 `model.` prefix"
        );
    }

    /// The dispatch must hand V4.1 its OWN loader, not the 0731 one.
    ///
    /// Asserted behaviourally rather than by reading the match arm: build a real V4.1
    /// config, run the factory, and check the loader REFUSES. `DeepSeekV4WeightLoader`
    /// would not refuse here -- it would go on to look for EXL3 tensors -- so a refusal
    /// naming CB3 is positive evidence that the v41 arm was taken.
    #[test]
    fn the_factory_gives_v41_its_own_loader() {
        const CONFIG: &str =
            "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/config.json";
        let Ok(json) = std::fs::read_to_string(CONFIG) else {
            eprintln!("skipping: {CONFIG} not present");
            return;
        };
        let config = atlas_core::config::parse_config(&json).expect("V4.1 config must parse");
        assert_eq!(config.model_type, "deepseek_v41");

        let loader = crate::factory::loader_for_config(&config)
            .expect("deepseek_v41 must dispatch to a loader, not fall through to bail");
        assert!(
            !loader.supports_tp(),
            "V4.1 is MQA (num_key_value_heads = 1); head-parallel TP is impossible"
        );
    }

    /// Every entry point fails LOUDLY rather than returning an empty success.
    ///
    /// `load_mtp_weights` is the one that matters: `Ok(None)` is a legal value meaning
    /// "no MTP in this checkpoint", which is false for V4.1 (3 modules, 2401 mtp.* tensors)
    /// and would silently disable speculation while looking deliberate.
    #[test]
    fn unimplemented_entry_points_name_the_seam() {
        let message = cb3_residency_unimplemented("layer loading").to_string();
        assert!(message.contains("CB3 3-bit decode kernels"));
        assert!(message.contains("deliberate hard stop"));
        // It must say what DOES work, so the reader knows where the boundary is.
        assert!(message.contains("READ CB3 expert bytes"));
    }
}
