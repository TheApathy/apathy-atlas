// SPDX-License-Identifier: AGPL-3.0-only

//! The two SEAMS DeepSeek-V4.1 layer loading does not own: **attention** and **engram**.
//!
//! These are traits rather than `todo!()`s inside the layer for one reason: three lanes are
//! building this model at once, and a seam that is a trait can be filled by the lane that
//! owns it without that lane editing the layer, the loader, or each other's files. The
//! layer calls through these; whoever implements them drops in.
//!
//! ## The contract, stated once
//! Both take and return the same shapes the rest of the engine already uses, so an
//! implementation is a `TransformerLayer`-shaped body without the residual bookkeeping:
//! `hidden` is `[num_tokens, hidden_size]` BF16, read and written IN PLACE, and the
//! residual add is the LAYER's job, not the seam's. A seam that also added the residual
//! would double it, which is a silent accuracy failure rather than a crash.
//!
//! ## Hard stop, deliberately
//! [`MissingAttention`] and [`MissingEngram`] are what the loader installs when no lane has
//! registered. They FAIL LOUDLY on the first forward, naming the seam and its owner. They
//! are NOT no-ops: a no-op attention would let the model run and emit fluent, wrong tokens,
//! which is the failure shape this port has spent the week avoiding.

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};
use spark_runtime::weights::WeightStore;

use crate::layer::ForwardContext;

// =====================================================================================
// ATTENTION SEAM — owner: dsv41-attention
// =====================================================================================

/// One layer's V4.1 attention: MLA + cross-layer sparse selection.
///
/// Implementations own: the two RoPE tables (`freqs_c` with YaRN for layers 2-39,
/// `freqs_w` without for layers 0/1), the inverse RoPE on the attention output before the
/// o-projection, the MQA K==V aliasing, and the compressed/window gather order. None of
/// that is visible from here and none of it is this module's to decide.
pub trait Dsv41Attention: Send + Sync {
    /// Decode ONE token. `hidden` is `[1, hidden_size]` BF16, modified in place; the
    /// caller has already applied the input norm and holds the residual.
    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        hidden: DevicePtr,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()>;

    /// Prefill `num_tokens`. `hidden` is `[num_tokens, hidden_size]` BF16, in place.
    ///
    /// `kv_write_start` is the number of leading positions whose KV is already populated
    /// (prefix caching); an implementation must skip writing those, not recompute them.
    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        num_tokens: usize,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()>;
}

/// Builds one layer's attention from the weight store.
///
/// Registered ONCE via [`register_attention_loader`], before `load_layers` runs.
///
/// `layer` is the absolute layer index, which the implementation needs: V4.1's behaviour
/// is per-layer and not derivable from the config alone — `compress_ratios[layer]` is
/// 0 for layers 0-1 (window only, no indexer), 2 for 2-19 and 1 for 20-39, and only the
/// eight layers in `index_source_layer_ids` recompute a top-k at all.
pub trait Dsv41AttentionLoader: Send + Sync {
    fn load_attention(
        &self,
        layer: usize,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        kv_dtype: KvCacheDtype,
    ) -> Result<Box<dyn Dsv41Attention>>;
}

// =====================================================================================
// ENGRAM SEAM — owner: dsv41-engram
// =====================================================================================

/// The V4.1 engram lookup, present on **layers 1 and 14 only**.
///
/// The tables are ~95 GB each over a 16M n-gram vocabulary, 768M rows total. They do NOT
/// fit alongside the 71.7 GB expert arena on a 119.7 GB box, so an implementation is a
/// sparse row-gather tier over NVMe, not a resident table. The Python engine's two headline
/// numbers (1190 tok/s warm, 765 cold) are exactly this I/O floor; any port inherits it.
pub trait Dsv41Engram: Send + Sync {
    /// Gather and add the engram contribution into `hidden` `[num_tokens, hidden_size]`.
    ///
    /// `token_ids` is the host-side token stream for this call, which the gather needs
    /// because the table is keyed by n-gram over token ids, not by hidden state.
    fn apply(
        &self,
        hidden: DevicePtr,
        num_tokens: usize,
        token_ids: &[u32],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()>;
}

/// Builds a layer's engram, or `None` for the 38 layers that have none.
///
/// Returning `None` is the CORRECT answer for layers other than 1 and 14, and is the one
/// place in this port where an empty success is not a silent gap.
pub trait Dsv41EngramLoader: Send + Sync {
    fn load_engram(
        &self,
        layer: usize,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<Box<dyn Dsv41Engram>>>;
}

/// The layers that carry an engram table, from the checkpoint.
pub const ENGRAM_LAYERS: [usize; 2] = [1, 14];

// =====================================================================================
// REGISTRY
// =====================================================================================

use std::sync::RwLock;

static ATTENTION_LOADER: RwLock<Option<&'static dyn Dsv41AttentionLoader>> = RwLock::new(None);
static ENGRAM_LOADER: RwLock<Option<&'static dyn Dsv41EngramLoader>> = RwLock::new(None);

/// Install the attention implementation. Call before `load_layers`.
///
/// Takes `&'static` rather than a `Box` so the registry holds no allocation and the
/// implementation is a plain `static`. A second call REPLACES the first and is not an
/// error: tests install a stub over whatever the binary registered at startup.
pub fn register_attention_loader(loader: &'static dyn Dsv41AttentionLoader) {
    *ATTENTION_LOADER.write().expect("attention registry poisoned") = Some(loader);
}

pub fn register_engram_loader(loader: &'static dyn Dsv41EngramLoader) {
    *ENGRAM_LOADER.write().expect("engram registry poisoned") = Some(loader);
}

pub(super) fn attention_loader() -> Option<&'static dyn Dsv41AttentionLoader> {
    *ATTENTION_LOADER.read().expect("attention registry poisoned")
}

pub(super) fn engram_loader() -> Option<&'static dyn Dsv41EngramLoader> {
    *ENGRAM_LOADER.read().expect("engram registry poisoned")
}

// =====================================================================================
// HARD STOPS
// =====================================================================================

/// Installed when no attention loader is registered. Fails on first forward.
pub(super) struct MissingAttention {
    pub(super) layer: usize,
}

fn attention_stop(layer: usize, phase: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "DeepSeek-V4.1 attention is not implemented (layer {layer}, {phase}). The layer, the \
         resident CB3 expert arena and the MoE path are built and the model LOADS; what is \
         missing is the attention forward. OWNER: dsv41-attention. Implement \
         `weight_loader::deepseek_v41::seams::Dsv41Attention` + `Dsv41AttentionLoader` and \
         call `register_attention_loader` before `load_layers`. This is a deliberate hard \
         stop, NOT a no-op: a pass-through attention would produce fluent, wrong tokens and \
         no error."
    )
}

impl Dsv41Attention for MissingAttention {
    fn decode(
        &self,
        _hidden: DevicePtr,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        bail!(attention_stop(self.layer, "decode"))
    }

    fn prefill(
        &self,
        _hidden: DevicePtr,
        _num_tokens: usize,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _kv_write_start: usize,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        bail!(attention_stop(self.layer, "prefill"))
    }
}

/// Installed on layers 1 and 14 when no engram loader is registered.
pub(super) struct MissingEngram {
    pub(super) layer: usize,
}

impl Dsv41Engram for MissingEngram {
    fn apply(
        &self,
        _hidden: DevicePtr,
        _num_tokens: usize,
        _token_ids: &[u32],
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        bail!(
            "DeepSeek-V4.1 engram is not implemented (layer {}). The checkpoint ships two \
             ~95 GB tables (layers {:?}, 768M rows over a 16M n-gram vocab) which do NOT fit \
             alongside the 71.7 GB expert arena on this 119.7 GB box — an implementation is a \
             sparse NVMe row-gather tier, not a resident table. OWNER: dsv41-engram. \
             Implement `weight_loader::deepseek_v41::seams::Dsv41Engram` + \
             `Dsv41EngramLoader` and call `register_engram_loader`. Skipping the engram is \
             NOT a valid fallback: these layers' outputs would be wrong and nothing would say so.",
            self.layer,
            ENGRAM_LAYERS,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hard stops must FAIL, and say who owns the seam.
    ///
    /// This is the negative control for "the model loads": a load that succeeds because the
    /// seams silently did nothing is worse than a load that fails.
    #[test]
    fn the_missing_seams_refuse_and_name_their_owner() {
        let message = attention_stop(7, "decode").to_string();
        assert!(message.contains("dsv41-attention"), "{message}");
        assert!(message.contains("deliberate hard stop"), "{message}");
        assert!(
            message.contains("fluent, wrong tokens"),
            "the message must say why a no-op is not acceptable: {message}"
        );
    }

    /// Engram belongs to exactly two layers, and the constant must match the checkpoint.
    #[test]
    fn engram_layers_are_one_and_fourteen() {
        assert_eq!(ENGRAM_LAYERS, [1, 14]);
    }
}
