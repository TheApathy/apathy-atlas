// SPDX-License-Identifier: AGPL-3.0-only

//! Model trait (SDD: single trait, multiple implementations possible).
//!
//! The Model trait defines the interface for running inference. Business
//! logic (scheduler, engine) programs against this trait, not concrete types.

use spark_runtime::gpu::DevicePtr;

use crate::layer::LayerState;
use crate::speculative::ProposerState;

/// Result of a mixed forward pass (decode + prefill in one pass).
pub struct MixedForwardResult {
    /// Logits for decode sequences: [N, vocab_size] BF16.
    /// NULL if no decode sequences.
    pub decode_logits: DevicePtr,
    /// Logits for the prefill sequence's last token: [1, vocab_size] BF16.
    /// NULL if `is_last_chunk` was false (intermediate chunk, no logits).
    pub prefill_logits: DevicePtr,
}

/// Per-stream input slice for batched prefill.
///
/// One of these per concurrent prefilling stream — `prefill_batch_chunk` and
/// `mixed_forward_batch` accept a `&mut [PrefillSlice<'_>]` and process all
/// streams' chunks in a single forward pass. See Q12 in
/// `/workspace/atlas-internal/qwen-refactor/notes.md` for the bug this
/// fixes (concurrent prefills serialized through `prefilling.first_mut()`
/// in the scheduler, causing 5× asymmetric TTFT).
pub struct PrefillSlice<'a> {
    /// Full prompt tokens for this stream.
    pub prompt_tokens: &'a [u32],
    /// Per-stream sequence state (KV blocks, SSM slot, etc.).
    pub seq: &'a mut SequenceState,
    /// Token offset into `prompt_tokens` where this chunk starts.
    pub chunk_start: usize,
    /// Number of tokens in this chunk.
    pub chunk_len: usize,
    /// Whether this is the final chunk for this stream (controls whether
    /// the model emits last-token logits for sampling).
    pub is_last_chunk: bool,
}

/// Result of a fully-batched mixed forward pass: M decode tokens + N prefill
/// chunks in one pass.
pub struct MixedBatchResult {
    /// Logits for decode lanes: [M, vocab] BF16. NULL if no decode lanes.
    pub decode_logits: DevicePtr,
    /// Logits per prefill stream — one DevicePtr per stream in the input
    /// slice, in the same order. Each entry is `[1, vocab]` BF16 when that
    /// stream's chunk was `is_last_chunk`, or NULL otherwise.
    pub prefill_logits: Vec<DevicePtr>,
}

/// Per-sequence paged attention metadata for chunked prefill.
///
/// Positions and slots remain chunk-local, but the paged block table and
/// running sequence length can persist across chunks so we only upload the
/// changed tail instead of rebuilding the full page metadata every time.
pub struct ChunkedPrefillPageMetadata {
    /// Device buffer holding the sequence block table as raw 32-bit entries.
    pub block_table: DevicePtr,
    /// Device buffer holding the running paged-prefill sequence length.
    pub seq_len: DevicePtr,
    /// Total block-table entries allocated for this prompt.
    pub block_capacity: usize,
    /// Number of block-table entries already uploaded to `block_table`.
    pub uploaded_blocks: usize,
}

/// Sequence state tracked across decode steps.
pub struct SequenceState {
    /// Token IDs generated so far (including prompt).
    pub tokens: Vec<u32>,
    /// Block table for paged KV cache (indices into PagedKvCache).
    pub block_table: Vec<u32>,
    /// Current sequence length (prompt + generated).
    pub seq_len: usize,
    /// Whether this request can reach Qwen4's sparse-attention boundary.
    ///
    /// QSA must index every prefix token when true, but maintaining its side
    /// cache is pure overhead for requests whose prompt plus decode budget
    /// remains below 2048 tokens. The scheduler sets this once from the
    /// immutable request budget before prefill starts.
    pub qwen4_qsa_required: bool,
    /// Per-layer state (EmptyLayerState for attention, SsmLayerState for SSM).
    pub layer_states: Vec<Box<dyn LayerState>>,
    /// Per-sequence state for the speculative proposer on the ACTIVE arm
    /// (None if no proposer). Every reader in the engine uses this field and
    /// none of them need to know which arm is live — see
    /// `proposer_state_alt` for how that invariant is maintained.
    pub proposer_state: Option<Box<dyn ProposerState>>,
    /// Per-sequence state for the INACTIVE proposer arm, parked.
    ///
    /// Each proposer type owns a different concrete state (the DFlash drafter
    /// keeps a ctx accumulator plus fc/K/V caches; the MTP head keeps its own
    /// paged KV block table), and readers downcast `proposer_state` to the
    /// type they expect. So an arm switch cannot simply repoint the model's
    /// proposer: the per-sequence state has to move with it.
    ///
    /// `Model::swap_proposer_state` exchanges the two fields, keeping the
    /// invariant that `proposer_state` ALWAYS belongs to the live arm. That
    /// is what lets the other ~65 `seq.proposer_state` call sites stay
    /// unchanged and arm-oblivious.
    ///
    /// `None` on every single-proposer build.
    pub proposer_state_alt: Option<Box<dyn ProposerState>>,
    /// SSM state pool slot index. Used for CUDA graph stability — all sequences
    /// at the same slot_idx use the same fixed GPU addresses.
    pub slot_idx: usize,
    /// Marconi: token position up to which SSM state is valid from a snapshot.
    /// Set on chunk 0's prefix cache lookup, read by subsequent chunks to skip
    /// computation for tokens already covered by the snapshot + KV cache.
    pub marconi_skip_to: usize,
    /// Session hash for SSM snapshot isolation. Set by the scheduler before
    /// prefill. The model uses this to tag saved snapshots and verify ownership
    /// before restoring. 0 = no session tracking (legacy behavior).
    pub session_hash: u64,
    /// Persistent paged metadata for chunked prefill, allocated lazily on the
    /// first chunk that needs paged attention.
    pub chunked_prefill_meta: Option<ChunkedPrefillPageMetadata>,
    /// Number of prompt tokens served by the prefix cache (block-aligned).
    /// Set by the model layer on the chunk-0 prefix-cache lookup; read by
    /// the scheduler to populate `usage.prompt_tokens_details.cached_tokens`.
    /// 0 when prefix caching is disabled or the prompt had no cache match.
    pub cached_prefix_tokens: usize,
    /// Original prompt token count, set at the first prefill and never
    /// mutated by decode. Used by `cache_sequence` to split seq.tokens into
    /// prompt (already inserted + ref-bumped by prefill) vs generated
    /// (needs a fresh bump so `release` in `free_sequence` leaves the
    /// cache's baseline ref intact). 0 before the first prefill.
    pub prompt_len: usize,
    /// Disk-block-ID list for `--high-speed-swap` (Phase 6.1.c).
    /// Each entry is a stable disk-side identifier that outlives HBM block
    /// recycling. `disk_block_ids` grows monotonically with the sequence
    /// and represents its **full historical block list**. IDs are
    /// layer-agnostic — the same ID indexes a slot in every layer's
    /// on-disk file. Empty when `--high-speed-swap` is disabled.
    ///
    /// **Sliding-window invariant** (Phase 6.3): in HSS mode `block_table`
    /// is the suffix `disk_block_ids[hss_window_start()..]`, so
    /// `disk_block_ids.len() == hss_window_start() + block_table.len()`.
    /// Both vectors are grown together by the alloc helper; the offload
    /// helper only fills layer K/V data (no length growth). When
    /// `block_table.len() == cap` and a new logical block is needed, the
    /// alloc helper drops `block_table[0]` (frees the physical HBM block
    /// back to the pool) but keeps `disk_block_ids[0]` — the evicted
    /// block's data lives on at that disk_id for streaming reads.
    pub disk_block_ids: Vec<u32>,
    /// Host-side ring buffer for cross-chunk MTP last-K target hidden capture.
    ///
    /// Populated when `ATLAS_MTP_LASTK_PREFILL=K` (K>0) and an MTP proposer is
    /// wired. Each prefill chunk's tail rows are D2H copied here from the
    /// shared `hidden_states()` device buffer (which gets clobbered by the
    /// next sequence's chunk or this sequence's next chunk). At the last
    /// chunk's `finalize_last`, the accumulated host rows are H2D'd back to
    /// the shared `mtp_lastk_buf` device buffer so the MTP head can replay
    /// `forward_one` against the full last-K window — not just the last
    /// chunk's contribution.
    ///
    /// Layout: `[K, hidden_size]` BF16 (or FP32 when `use_fp32_residual()`),
    /// where row 0 is the oldest captured position and row K-1 is the newest.
    /// `mtp_lastk_host_filled` tracks how many rows are currently populated
    /// (0..=K).
    ///
    /// Lazy allocation: empty Vec until the first prefill chunk for an
    /// MTP-enabled sequence. Sized once to `K * hidden_size * fp_size`.
    /// Freed naturally when `SequenceState` drops.
    ///
    /// The historical "single-chunk capture" path (mtp-lastk-v2) read directly
    /// from `hidden_states()` at finalize_last and silently truncated to the
    /// last chunk's `proc_count`, which collapsed to K_eff=83 at K=512 when
    /// `--max-prefill-tokens 1024` made the trailing chunk small. This
    /// cross-chunk ring fixes that truncation.
    pub mtp_lastk_host_buf: Vec<u8>,
    /// Number of rows currently populated in `mtp_lastk_host_buf` (0..=K).
    /// Reset to 0 on `free_sequence`. Reaches K once enough prefill chunks
    /// have been seen; subsequent chunks shift older rows out.
    pub mtp_lastk_host_filled: usize,
    /// Absolute prompt position of the LAST captured row in
    /// `mtp_lastk_host_buf` (`base_position` arg to `prefill_last_k`).
    /// 0 when no rows captured.
    pub mtp_lastk_end_abs: usize,
    /// Per-attention-layer offload progress tracker for `--high-speed-swap`
    /// (Phase 6.1.d critical fix). `disk_last_offloaded_per_layer[L]` is
    /// the number of `disk_block_ids` entries this attention layer has
    /// successfully offloaded to its on-disk file. Each layer maintains
    /// its own counter because each layer writes its own K/V independently;
    /// without per-layer tracking, only the first layer to encounter a new
    /// block would offload, leaving subsequent layers' on-disk slots
    /// uninitialised. Length equals the model's attention layer count;
    /// empty when HSS is disabled.
    pub disk_last_offloaded_per_layer: Vec<u32>,
}

impl SequenceState {
    /// Phase 6.3 sliding-window helper: the absolute logical block index
    /// of `block_table[0]`. Returns 0 when `--high-speed-swap` is off
    /// (`disk_block_ids` is empty then; `block_table` is the full history).
    /// Derived rather than stored — the invariant
    /// `disk_block_ids.len() == hss_window_start() + block_table.len()`
    /// is maintained by the alloc helper and asserted by the offload
    /// helper, so no separate field is needed.
    #[inline]
    pub fn hss_window_start(&self) -> usize {
        self.disk_block_ids
            .len()
            .saturating_sub(self.block_table.len())
    }

    /// Map an absolute logical block index → physical HBM block id.
    /// Returns `None` when the block has been evicted to disk-only
    /// (the caller should route attention through the HSS orchestrator's
    /// `attend_layer_on_stream` for that position). With HSS off,
    /// `hss_window_start()` is 0 and this is a direct lookup.
    #[inline]
    pub fn physical_block_for(&self, abs_block_idx: usize) -> Option<u32> {
        let ws = self.hss_window_start();
        if abs_block_idx < ws {
            return None;
        }
        self.block_table.get(abs_block_idx - ws).copied()
    }
}

/// Model trait for forward pass execution.
///
/// Implementations: `TransformerModel` (all architectures).
///
/// # Safety
///
/// `Send + Sync` is required by `Box<dyn Model>` usage patterns.
/// `Sync` safety: the model is exclusively accessed from the scheduler
/// thread. The `unsafe impl Sync` on `TransformerModel` documents this
/// single-thread invariant — do NOT share `&dyn Model` across threads.
mod model;
pub use model::{EP_CMD_VERIFY_KGAMMA, EP_VERIFY_KGAMMA_ABORT, Model};
