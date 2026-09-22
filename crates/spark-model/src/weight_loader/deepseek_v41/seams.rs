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

/// Per-CHUNK sparse state, threaded through all 40 layers of one pass.
///
/// ## What is in here, and what is deliberately NOT
/// Only the things whose lifetime is ONE CHUNK: `topk` and `candidates`. Both are indexed
/// by the CURRENT chunk's tokens, so rebuilding them per chunk is correct.
///
/// **`ckv` and `ik` are NOT storage here — they are borrowed handles.** An earlier version
/// of this struct OWNED them and cleared them in `begin_pass`. That was wrong, and
/// dsv41-attention settled it from the reference with evidence I could not have derived:
///
/// - they are allocated ONCE per sequence, sized `max_seq / ratio + 1` (`Caches.__init__`),
///   never rebuilt per chunk;
/// - writes land at an ABSOLUTE offset, `c.ckv[L][j0 : j0+nj]` with `j0 = chunk_start /
///   ratio`, so a chunk fills its own slice of a sequence-long cache;
/// - `Shared` in the reference only ALIASES them — `sh.ckv, sh.ik, sh.ratio = c.ckv[L],
///   c.ik[L], r` is a pointer plus the ratio, never a copy;
/// - `Caches.rollback` says it outright: these caches are APPEND-ONLY.
///
/// And the clincher: the indexer selects ABSOLUTE positions. At chunk 2 a token may select
/// compressed row 12, which chunk 1 wrote. Clearing `ckv` per chunk makes that gather read
/// ZEROS — attention silently mixes in zero rows, output is wrong, nothing errors. Exactly
/// the failure class this port is organised against, which is why it was flagged rather
/// than guessed.
///
/// So `ckv`/`ik` live in the KV CACHE alongside the window ring and the compressor's
/// `pending` — all three are sequence-lifetime, append-only and indexed by absolute
/// position. `pending` feeds `ckv`, so they must share a lifetime; that symmetry holds.
/// [`Self::begin_pass`] therefore has nothing to special-case, and "ckv got cleared"
/// becomes unrepresentable rather than a comment someone has to remember.
///
/// ## Where this lives, and why it is NOT on `ForwardContext`
/// The layers share one `Arc<Mutex<SparseShared>>`. The objection to a shared mutable
/// singleton is that two concurrent forwards would corrupt each other. **They cannot.**
/// `spark-server`'s scheduler takes `mut model: Box<dyn Model>` by value
/// (`crates/spark-server/src/scheduler/mod.rs:206`) — the model is MOVED into one thread
/// and exactly one forward runs at a time. Batched decode batches WITHIN a forward. So the
/// singleton is sound, and a `ForwardContext` field would buy nothing while touching 34
/// construction sites across files two other lanes are editing.
///
/// The lock is taken once per LAYER, not per token — about 40 uncontended acquisitions per
/// forward. If the scheduler ever grows concurrent forwards over one model, this becomes a
/// real bug and `ForwardContext` becomes the right answer. That is the trigger to watch.
#[derive(Default)]
pub struct SparseShared {
    /// Handle to the current kv-source layer's compressed-KV cache: `[rows, 512]` bf16.
    ///
    /// BORROWED from the KV cache, never owned here. Set on [`KV_SOURCE_LAYERS`], read by
    /// their inheritors. Survives chunk boundaries because the cache does.
    pub ckv: Option<DevicePtr>,
    /// Handle to the matching index-key cache, `[rows, index_head_dim]`.
    pub ik: Option<DevicePtr>,
    /// Rows currently VALID in `ckv`/`ik` — grows with the sequence, not with the chunk.
    pub ckv_rows: usize,
    /// `compress_ratios[layer]` of the kv-source layer whose caches `ckv`/`ik` point at.
    ///
    /// **Re-read from the kv-source layer on EVERY layer; never cached across a forward.**
    /// The ratio flips 2 -> 1 at layer 20, which is itself a kv-source layer. The reference
    /// asserts `sh.ratio == w.ratio` on every layer for exactly this reason: if a future
    /// schedule moved the flip off a source boundary, a layer would inherit a cache built
    /// at the wrong ratio and nothing else would catch it.
    pub ratio: usize,
    /// Per-chunk selection: `[num_tokens, 512]` i64, -1 = none. Written on
    /// [`INDEX_SOURCE_LAYERS`], read by their inheritors. ALWAYS 512 wide. RESET per chunk.
    pub topk: Option<DevicePtr>,
    /// The candidate block mask from layer 20's indexer alone, pruning 24/28/32/36.
    /// RESET per chunk.
    pub candidates: Option<DevicePtr>,
    /// Chunk start of the pass in flight, for debugging a stale-state bug.
    pub pass_start: usize,
    /// Passes begun since construction. Lets a caller assert the reset actually ran.
    pub passes: u64,
}

impl SparseShared {
    /// Begin one pass over the 40 layers, for the chunk starting at `chunk_start`.
    ///
    /// Called by LAYER 0 and nowhere else. Layer 0 is a sound reset point for a reason, not
    /// by convenience: it runs exactly once per pass and has `compress_ratio == 0`, so it
    /// performs no sparse work a reset could destroy.
    ///
    /// Clears ONLY the per-chunk fields. `ckv`/`ik`/`ckv_rows`/`ratio` are handles into
    /// sequence-lifetime caches and are re-published by the next kv-source layer; clearing
    /// them here is the bug described on the struct.
    pub fn begin_pass(&mut self, chunk_start: usize) {
        self.topk = None;
        self.candidates = None;
        self.pass_start = chunk_start;
        self.passes += 1;
    }

    /// Publish a kv-source layer's caches for its inheritors.
    ///
    /// Takes `ratio` alongside the pointers so the two cannot drift: a handle without the
    /// ratio it was built at is how a layer ends up reading a cache at the wrong stride.
    pub fn publish_compressed(
        &mut self,
        ckv: DevicePtr,
        ik: DevicePtr,
        rows: usize,
        ratio: usize,
    ) {
        self.ckv = Some(ckv);
        self.ik = Some(ik);
        self.ckv_rows = rows;
        self.ratio = ratio;
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
    /// Takes a MODEL DIRECTORY, not a `WeightStore`.
    ///
    /// This signature originally took `&WeightStore` by symmetry with the attention loader.
    /// That was a lie about where the data comes from: at ~95 GB per table the engram
    /// tensors are never loaded into the store at all, and an implementation must parse
    /// `model.safetensors.index.json` itself to find the right shard. The old signature
    /// "worked" only because the implementation ignored the argument and took the directory
    /// through a separate registration call — which is exactly the kind of quiet divergence
    /// between a signature and reality that costs the next reader an hour.
    fn load_engram(
        &self,
        layer: usize,
        model_dir: &std::path::Path,
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

    /// `cidx` width is a numerics contract, and `begin_pass` must clear the per-chunk
    /// fields while LEAVING the sequence-lifetime handles alone.
    ///
    /// The second half is the regression test for a bug this struct actually had: clearing
    /// `ckv` per chunk makes a chunk-2 gather of a chunk-1 row read zeros, which attention
    /// mixes in silently.
    #[test]
    fn begin_pass_clears_per_chunk_state_and_preserves_the_sequence_caches() {
        assert_eq!(INDEX_TOPK, 512);
        let mut shared = SparseShared::default();
        shared.publish_compressed(DevicePtr(0xC10), DevicePtr(0x1C0), 1024, 2);
        shared.topk = Some(DevicePtr(0x700));
        shared.candidates = Some(DevicePtr(0xCA0));

        shared.begin_pass(2048);

        // Per-chunk state is gone.
        assert!(shared.topk.is_none(), "topk is per-chunk and must be cleared");
        assert!(shared.candidates.is_none(), "candidates are per-chunk");
        assert_eq!(shared.pass_start, 2048);
        assert_eq!(shared.passes, 1);

        // Sequence-lifetime handles SURVIVE. If this ever fails, a chunk-2 gather of a
        // chunk-1 compressed row reads zeros and nothing errors.
        assert_eq!(
            shared.ckv,
            Some(DevicePtr(0xC10)),
            "ckv is a handle into a sequence-lifetime cache and must survive a chunk boundary"
        );
        assert_eq!(shared.ik, Some(DevicePtr(0x1C0)), "ik survives with ckv");
        assert_eq!(shared.ckv_rows, 1024, "valid rows grow with the sequence, not the chunk");
        assert_eq!(shared.ratio, 2, "the ratio travels with the handle");

        // The pass counter keeps counting across passes.
        shared.begin_pass(4096);
        assert_eq!(shared.passes, 2);
        assert_eq!(shared.pass_start, 4096);
        assert_eq!(shared.ckv_rows, 1024);
    }

    /// The ratio flip at layer 20 lands ON a kv-source layer.
    ///
    /// That is what makes "re-read the ratio from the kv-source layer every layer" safe. If
    /// a future schedule moved the flip off a source boundary, a layer would inherit a
    /// cache built at the wrong stride and nothing else would catch it — so the property is
    /// asserted rather than assumed.
    #[test]
    fn the_ratio_flip_lands_on_a_kv_source_layer() {
        // compress_ratios: 0 for 0-1, 2 for 2-19, 1 for 20-39.
        const FLIP: usize = 20;
        assert!(
            KV_SOURCE_LAYERS.contains(&FLIP),
            "the 2 -> 1 ratio flip at layer {FLIP} must coincide with a kv-source layer"
        );
        // And it is the last one, so no later source re-publishes at the old ratio.
        assert_eq!(*KV_SOURCE_LAYERS.last().unwrap(), FLIP);
    }

    /// Engram belongs to exactly two layers, with the shape the engram lane specified.
    #[test]
    fn engram_layers_and_row_shape_match_the_spec() {
        assert_eq!(ENGRAM_LAYERS, [1, 14]);
        assert_eq!(ENGRAM_ROWS_PER_TOKEN, 24);
        assert_eq!(ENGRAM_ROW_DIM, 256);
    }
}
