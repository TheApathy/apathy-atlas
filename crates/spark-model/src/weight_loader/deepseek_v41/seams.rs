// SPDX-License-Identifier: AGPL-3.0-only

//! The two SEAMS DeepSeek-V4.1 layer loading does not own: **attention** and **engram**.
//!
//! These are traits rather than `todo!()`s inside the layer for one reason: three lanes are
//! building this model at once, and a seam that is a trait can be filled by the lane that
//! owns it without that lane editing the layer, the loader, or each other's files.
//!
//! ## Everything here is `DevicePtr`, not `Tensor`
//! Both lanes proposed `&Tensor` signatures. **There is no `Tensor` type in this crate's
//! layer idiom** — `TransformerLayer::decode` / `prefill` take `DevicePtr` plus explicit
//! extents plus a CUDA stream, and every layer in the tree is written that way. Introducing
//! a `Tensor` would be a deliberate, tree-wide change, not something a seam should assume.
//! So the shapes below are exactly the ones the lanes specified; only the carrier differs.
//! Extents that the callee cannot derive from `ModelConfig` are passed explicitly.
//!
//! ## Hard stop, deliberately
//! [`MissingAttention`] and [`MissingEngram`] are what the loader installs when no lane has
//! registered. They FAIL LOUDLY on first use, naming the seam and its owner. They are NOT
//! no-ops: a no-op attention would let the model run and emit fluent, wrong tokens.

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::ForwardContext;

// =====================================================================================
// ATTENTION SEAM — owner: dsv41-attention
// =====================================================================================

/// Layers that recompute a top-k selection (`index_source_layer_ids`). The other 32 layers
/// attend with their predecessor's selection.
pub const INDEX_SOURCE_LAYERS: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];
/// Layers that build compressed KV and index keys (`kv_source_layer_ids`). 3-7 reuse 2's,
/// 9-13 reuse 8's, 21-39 reuse 20's.
pub const KV_SOURCE_LAYERS: [usize; 4] = [2, 8, 14, 20];
/// The ONE layer whose indexer produces the candidate block mask that prunes 24/28/32/36.
pub const CANDIDATE_SOURCE_LAYER: usize = 20;
/// `index_topk`. **`cidx` is ALWAYS exactly this wide**, padded with -1, even when fewer
/// rows are reachable. A varying N flips cuBLAS kernel choice and the ulp differences
/// change router decisions downstream. This is a numerics contract, not a buffer size.
pub const INDEX_TOPK: usize = 512;

/// Cross-layer state for ONE forward pass, threaded through all 40 layers.
///
/// ## Where this lives, and why it is here rather than on `ForwardContext`
/// The lifetime asked for is "persists across the whole 40-layer loop of one forward,
/// resets per forward". That does **not** fit [`crate::layer::LayerState`], which is
/// per-layer. The structurally right home is `ForwardContext`, which this lane does not
/// own.
///
/// So the layers share one `Arc<Mutex<SparseShared>>` and **layer 0 resets it**. That is
/// sound rather than merely convenient: layer 0 runs exactly once per forward, and it has
/// `compress_ratio == 0`, so it does no sparse work that a reset could destroy. It is
/// still a workaround — if `ForwardContext` ever gains a per-forward slot, move it there.
///
/// ## What is NOT here
/// The compressor's `pending` (one unpaired position per ratio-2 layer) carries across
/// CHUNKS, not just across layers, so it belongs in the KV cache. Putting it here would
/// silently drop it between chunks of the same sequence.
#[derive(Default)]
pub struct SparseShared {
    /// Per layer, the inherited compressed-KV rows: `[n, 512]` bf16. Written on
    /// [`KV_SOURCE_LAYERS`], read by their inheritors.
    pub ckv: Option<DevicePtr>,
    /// Rows currently in `ckv`.
    pub ckv_rows: usize,
    /// Index keys built alongside `ckv` on the same layers.
    pub index_keys: Option<DevicePtr>,
    /// Per layer, the inherited selection: `[T, 512]` i64, -1 = none. Written on
    /// [`INDEX_SOURCE_LAYERS`], read by their inheritors. ALWAYS 512 wide.
    pub topk: Option<DevicePtr>,
    /// The candidate block mask from layer 20's indexer alone, pruning 24/28/32/36.
    pub candidates: Option<DevicePtr>,
    /// `compress_ratios[layer]` of whichever layer last wrote `ckv`. Flips 2 -> 1 at 20.
    pub ratio: usize,
}

impl SparseShared {
    /// Called by layer 0 at the top of every forward. See the note on the struct.
    pub fn reset_for_new_forward(&mut self) {
        *self = Self::default();
    }
}

/// One layer's V4.1 attention.
///
/// Split into two entry points on purpose: the indexer runs on 8 layers and attention on
/// 40. Collapsing them hides exactly the asymmetry that must stay visible.
pub trait Dsv41Attention: Send + Sync {
    /// Recompute the sparse selection. Called **only** for `layer` in
    /// [`INDEX_SOURCE_LAYERS`]; every other layer inherits `shared.topk` untouched.
    ///
    /// * `x` — `[num_tokens, 5120]` bf16, post `attn_norm`.
    /// * `qr` — `[num_tokens, 1280]` bf16, the rmsnorm'd q_lora ALREADY computed by
    ///   attention. Passed in rather than recomputed so the two paths cannot drift.
    /// * `chunk_start` / `chunk_len` — position of this chunk within the sequence.
    ///
    /// Writes `shared.topk` (`[num_tokens, 512]` i64), and on
    /// [`CANDIDATE_SOURCE_LAYER`] also `shared.candidates`.
    #[allow(clippy::too_many_arguments)]
    fn sparse_index_select(
        &self,
        x: DevicePtr,
        qr: DevicePtr,
        layer: usize,
        num_tokens: usize,
        chunk_start: usize,
        chunk_len: usize,
        shared: &mut SparseShared,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()>;

    /// Attend. Called on every layer with `compress_ratio != 0` — i.e. NOT layers 0 and 1,
    /// which are window-only.
    ///
    /// * `q` — `[num_tokens, 64, 512]` bf16, already RoPE'd.
    /// * `ring` — `[ring_len, 512]` bf16. **MQA: this is K and V both.**
    ///   `num_key_value_heads == 1`, so there is no separate V tensor; the kernel does
    ///   `dot(q, k^T)` then `dot(p, k)` against this same buffer.
    /// * `wpos` — `[num_tokens, 128]` i64 absolute positions, -1 = none.
    /// * `ckv` / `cidx` — from `shared`. `cidx` is ALWAYS [`INDEX_TOPK`] wide.
    /// * `sink` — `[64]` f32 attention sink.
    /// * `scale` — `head_dim^-0.5`.
    ///
    /// Writes `out`, `[num_tokens, 64, 512]` bf16. The INVERSE RoPE before the
    /// o-projection is the caller's business, not this call's.
    #[allow(clippy::too_many_arguments)]
    fn sparse_attention(
        &self,
        q: DevicePtr,
        ring: DevicePtr,
        ring_len: usize,
        wpos: DevicePtr,
        win_lo: usize,
        ckv: Option<DevicePtr>,
        ckv_rows: usize,
        cidx: Option<DevicePtr>,
        sink: DevicePtr,
        scale: f32,
        num_tokens: usize,
        out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()>;
}

/// Builds one layer's attention from the weight store. Registered ONCE via
/// [`register_attention_loader`], before `load_layers` runs.
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

/// The layers that carry an engram table.
pub const ENGRAM_LAYERS: [usize; 2] = [1, 14];
/// Rows fetched per token per layer.
pub const ENGRAM_ROWS_PER_TOKEN: usize = 24;
/// Values per row. Each row is 256 fp8_e4m3 plus 8 ue8m0 scales on disk; the gather returns
/// them dequantised as `fp8[j] * 2^(scale[j/32] - 127)`, which is exact in bf16.
pub const ENGRAM_ROW_DIM: usize = 256;

/// The V4.1 engram row gather, on **layers 1 and 14 only**.
///
/// A PURE GATHER: `[T, 24]` i64 row ids in, `[T, 24, 256]` f32 out. Dequantisation happens
/// inside; everything downstream — the dead-head mask, the projection, the residual add —
/// is the caller's.
///
/// ## Two contracts that are easy to break
/// 1. **Do NOT fold the dead-head mask into the gather.** The reference applies
///    `rows.masked_fill(dead_heads.unsqueeze(-1), 0)` AFTER the fetch. This call returns
///    PRE-mask rows; the mask travels separately.
/// 2. **Decode reads cannot be prefetched.** The n-gram for position `p` includes `c[p]`,
///    the token just sampled, so step N's row ids are unknowable until step N-1's sampling
///    completes. This read sits on the decode critical path and cannot hide behind the
///    previous step's compute. Measured at ~1.21 ms against a ~30 ms step (~4%), so it is a
///    constraint, not a problem — but no pipeline here may assume engram is prefetchable.
///    At prefill it is fully batched and the point is moot.
///
/// The tables are ~95 GB each and do NOT fit alongside the 71.7 GB expert arena on a
/// 119.7 GB box, so an implementation is a sparse NVMe row-gather, not a resident table.
/// Its measured I/O ceiling is 19,637 tok/s at prefill and 827 tok/s at decode — cheap
/// enough that nothing in this layer design should be contorted to avoid it.
pub trait Dsv41Engram: Send + Sync {
    /// `row_ids` is `[num_tokens, 24]` i64, host-side. `out` is `[num_tokens, 24, 256]`
    /// f32 on the device.
    fn gather_rows(
        &self,
        row_ids: &[i64],
        num_tokens: usize,
        out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()>;
}

/// Builds a layer's engram, or `None` for the 38 layers that have none.
///
/// Returning `None` is the CORRECT answer off [`ENGRAM_LAYERS`], and is the one place in
/// this port where an empty success is not a silent gap.
pub trait Dsv41EngramLoader: Send + Sync {
    fn load_engram(
        &self,
        layer: usize,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<Box<dyn Dsv41Engram>>>;
}

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

/// Installed when no attention loader is registered. Fails on first use.
pub(super) struct MissingAttention {
    pub(super) layer: usize,
}

fn attention_stop(layer: usize, phase: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "DeepSeek-V4.1 attention is not implemented (layer {layer}, {phase}). The layer, the \
         resident CB3 expert arena and the per-expert MoE path are built and the model \
         LOADS; what is missing is the attention forward. OWNER: dsv41-attention. Implement \
         `weight_loader::deepseek_v41::seams::Dsv41Attention` + `Dsv41AttentionLoader` and \
         call `register_attention_loader` before `load_layers`. This is a deliberate hard \
         stop, NOT a no-op: a pass-through attention would produce fluent, wrong tokens and \
         no error."
    )
}

impl Dsv41Attention for MissingAttention {
    fn sparse_index_select(
        &self,
        _x: DevicePtr,
        _qr: DevicePtr,
        _layer: usize,
        _num_tokens: usize,
        _chunk_start: usize,
        _chunk_len: usize,
        _shared: &mut SparseShared,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        bail!(attention_stop(self.layer, "sparse_index_select"))
    }

    fn sparse_attention(
        &self,
        _q: DevicePtr,
        _ring: DevicePtr,
        _ring_len: usize,
        _wpos: DevicePtr,
        _win_lo: usize,
        _ckv: Option<DevicePtr>,
        _ckv_rows: usize,
        _cidx: Option<DevicePtr>,
        _sink: DevicePtr,
        _scale: f32,
        _num_tokens: usize,
        _out: DevicePtr,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        bail!(attention_stop(self.layer, "sparse_attention"))
    }
}

/// Installed on [`ENGRAM_LAYERS`] when no engram loader is registered.
pub(super) struct MissingEngram {
    pub(super) layer: usize,
}

impl Dsv41Engram for MissingEngram {
    fn gather_rows(
        &self,
        _row_ids: &[i64],
        _num_tokens: usize,
        _out: DevicePtr,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        bail!(
            "DeepSeek-V4.1 engram is not implemented (layer {}). The checkpoint ships two \
             ~95 GB tables (layers {:?}) which do NOT fit alongside the 71.7 GB expert arena \
             on this 119.7 GB box — an implementation is a sparse NVMe row-gather returning \
             [T, {}, {}], not a resident table. OWNER: dsv41-engram. Implement \
             `weight_loader::deepseek_v41::seams::Dsv41Engram` + `Dsv41EngramLoader` and \
             call `register_engram_loader`. Skipping the engram is NOT a valid fallback: \
             these layers' outputs would be wrong and nothing would say so.",
            self.layer,
            ENGRAM_LAYERS,
            ENGRAM_ROWS_PER_TOKEN,
            ENGRAM_ROW_DIM,
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
        let message = attention_stop(7, "sparse_attention").to_string();
        assert!(message.contains("dsv41-attention"), "{message}");
        assert!(message.contains("deliberate hard stop"), "{message}");
        assert!(
            message.contains("fluent, wrong tokens"),
            "the message must say why a no-op is not acceptable: {message}"
        );
    }

    /// The three inheritance chains are DISTINCT sets. Treating them as one is the whole
    /// reason this constant block exists.
    #[test]
    fn the_three_cross_layer_chains_are_distinct() {
        // Every kv-source layer is also an index-source layer, but not conversely.
        for layer in KV_SOURCE_LAYERS {
            assert!(INDEX_SOURCE_LAYERS.contains(&layer), "kv {layer} must index too");
        }
        assert_ne!(
            KV_SOURCE_LAYERS.len(),
            INDEX_SOURCE_LAYERS.len(),
            "if these were the same set, the two chains would be one"
        );
        // The candidate mask comes from ONE layer, not from the index-source set.
        assert!(INDEX_SOURCE_LAYERS.contains(&CANDIDATE_SOURCE_LAYER));
        assert_eq!(CANDIDATE_SOURCE_LAYER, 20);
        // Layers 0 and 1 are in none of them: ratio 0, window only.
        for layer in [0usize, 1] {
            assert!(!INDEX_SOURCE_LAYERS.contains(&layer));
            assert!(!KV_SOURCE_LAYERS.contains(&layer));
        }
    }

    /// `cidx` width is a numerics contract, and `SparseShared` must reset cleanly.
    #[test]
    fn topk_width_is_fixed_and_shared_resets() {
        assert_eq!(INDEX_TOPK, 512);
        let mut shared = SparseShared {
            ckv: Some(DevicePtr(0xdead)),
            ckv_rows: 99,
            ratio: 2,
            ..Default::default()
        };
        shared.reset_for_new_forward();
        assert!(shared.ckv.is_none());
        assert_eq!(shared.ckv_rows, 0);
        assert_eq!(shared.ratio, 0);
    }

    /// Engram belongs to exactly two layers, with the shape the engram lane specified.
    #[test]
    fn engram_layers_and_row_shape_match_the_spec() {
        assert_eq!(ENGRAM_LAYERS, [1, 14]);
        assert_eq!(ENGRAM_ROWS_PER_TOKEN, 24);
        assert_eq!(ENGRAM_ROW_DIM, 256);
    }
}
