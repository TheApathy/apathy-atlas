// SPDX-License-Identifier: AGPL-3.0-only
//! Construct a served [`Glm53Model`] from an already-loaded GLM-5.3 GGUF store.
//!
//! GLM-5.3-Flash cannot use [`super::build::build_model`]'s generic transformer
//! path: its trunk is 34 KDA gated-delta-net layers interleaved with 11
//! absorbed-MLA DSA layers, wrapped in mHC hyper-connections, and its weights
//! arrive as a four-shard GGUF payload rather than safetensors. The generic path
//! takes a `WeightStore`; this one takes the `GgufDeviceStore` that
//! `TargetStoreLoadPlan::load_glm53_gguf` produces.
//!
//! This is the seam that was missing: `Glm53Model` implements `Model`, and
//! `LoadedTargetStore::Glm53Gguf` exists, but nothing joined them, so the only
//! way to execute GLM was a test harness. Serving GLM through the same engine
//! as every other model starts here.

use anyhow::{Result, anyhow};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::gguf::GgufDeviceStore;

use super::{Glm53QuantProfile, Glm53TargetRuntimeWeights};
use crate::model::glm53::Glm53Model;
use crate::traits::Model;

/// Build a servable GLM-5.3-Flash model.
///
/// `max_seq_len` becomes the arena's position capacity. Note this is the
/// RUNTIME capacity, not the architectural 1,048,576 the DSA pool plan is
/// defined against — sizing the arena for the full 1M overruns a GB10's free
/// memory by ~95 MiB on the UD-IQ2_XXS checkpoint.
pub fn build_glm53_model(
    profile: Glm53QuantProfile,
    config: &ModelConfig,
    store: GgufDeviceStore,
    gpu: std::sync::Arc<dyn GpuBackend>,
    max_seq_len: usize,
) -> Result<Box<dyn Model>> {
    let positions = u32::try_from(max_seq_len)
        .map_err(|_| anyhow!("GLM-5.3 max_seq_len {max_seq_len} exceeds u32"))?;
    // Admission errors carry their own refusal text; they do not implement
    // std::error::Error, so surface Display rather than using `?`.
    let weights = Glm53TargetRuntimeWeights::new(profile, config, store)
        .map_err(|error| anyhow!("GLM-5.3 runtime weights refused: {error}"))?;
    let model = Glm53Model::new(gpu, weights, positions)?;
    Ok(Box::new(model))
}
