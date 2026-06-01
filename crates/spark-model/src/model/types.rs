// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Architecture-agnostic transformer model.
///
/// Composes `Vec<Box<dyn TransformerLayer>>` into a full forward pass.
/// Adding a new model only requires implementing [`TransformerLayer`]
/// for each layer type — the model loop stays unchanged.
#[allow(dead_code)]
pub struct TransformerModel {
    pub(super) config: ModelConfig,
    /// M8A: per-step DDTree parent-ids channel. verify_d.rs uploads a
    /// payload's parent_indices tensor and stashes the device pointer
    /// here BEFORE the K=γ layer loop; the SSM dispatch reads it to
    /// decide flat (gdn_wy17_k) vs tree-aware (gdn_tree_k) kernel.
    /// `None` outside DDTree verify or when payload is flat-chain.
    pub ddtree_parent_ids_dev: Mutex<Option<spark_runtime::gpu::DevicePtr>>,
    /// Length of the parent_ids tensor above (= num_tree_tokens = K verify).
    pub ddtree_num_tree_tokens: Mutex<usize>,
    /// CUDA-graph-safe persistent parent_ids buffer for the K=γ verify
    /// path. Allocated ONCE at init, sized for `dflash_kgamma * 4` bytes
    /// (= 17 i32 slots on the 27b target). Holds two state-overlays:
    ///   * Default (linear-chain): `[-1, 0, 1, ..., K-2]` — bit-equivalent
    ///     to the wy17 flat path when consumed by `gated_delta_rule_tree_wy`.
    ///     Re-stamped after every tree-mode verify by `clear_ddtree_parent_ids`.
    ///   * Tree-mode override: `set_ddtree_parent_ids` overwrites in place
    ///     with the kernel-frame mapping of the current payload.
    /// The DEVICE POINTER never changes — that's the invariant captured
    /// CUDA graphs rely on. `DevicePtr::NULL` when DFlash is disabled.
    pub ddtree_parent_ids_persistent: spark_runtime::gpu::DevicePtr,
    /// Capacity of `ddtree_parent_ids_persistent` in i32 slots (= dflash_kgamma).
    /// Zero when DFlash is disabled. Used to stamp linear-chain defaults.
    pub ddtree_parent_ids_capacity: usize,
    /// Host-side mirror of the kernel-frame parent_ids tensor. Populated by
    /// `set_ddtree_parent_ids` so the K=γ verify path can derive per-token
    /// tree depths (for depth-based RoPE positions and seq_lens) without
    /// a D2H copy. Empty when no tree payload is active (chain mode).
    ///
    /// Layout matches `ddtree_parent_ids_persistent`: index 0 is the bonus
    /// (parent=-1), index i+1 corresponds to draft i, with each entry the
    /// kernel-frame parent (`-1` = pre-tree state / bonus's parent, `k≥0`
    /// = parent kernel slot).
    pub ddtree_parent_ids_host: Mutex<Vec<i32>>,
    /// Inverse DFS permutation from the last K=γ verify (option C reorder).
    /// `dfs_inv_perm[orig_compact_idx] = dfs_slot_idx`. Empty when DFS
    /// reorder is inactive (env var off, or payload was trivially flat).
    ///
    /// Populated by `verify_d.rs` BEFORE the SSM kernel runs (because the
    /// kernel writes h_state_inter in DFS slot order). Read by
    /// `commit_verify_state_async_with_slot` to translate the original-
    /// compact `last_inter_slot` from greedy_sample_ddtree into the DFS
    /// slot where the SSM canonical state actually lives.
    pub ddtree_dfs_inv_perm: Mutex<Vec<usize>>,
    /// ATLAS_TREE_AWARE_ATTN: persistent per-row KV indirection buffer.
    /// Layout: `[dflash_kgamma rows × dflash_kgamma cols]` i32, row-major.
    /// `verify_d.rs` writes the ancestor-chain mapping for the current
    /// K=γ tree (single-shot upload) so the paged-decode attention kernel
    /// can remap tree-window positions to true ancestors.
    /// `DevicePtr::NULL` when DFlash is disabled.
    pub tree_kv_indir_persistent: spark_runtime::gpu::DevicePtr,
    /// Row stride of `tree_kv_indir_persistent`, in i32 slots
    /// (= dflash_kgamma). Zero when DFlash is disabled.
    pub tree_kv_indir_stride: usize,
    /// ATLAS_TREE_AWARE_ATTN CUDA graph fix: persistent 1×i32 device buffer
    /// holding the current tree-window base position (= `seq.seq_len`).
    /// The paged-decode-attn + tree-kv-scatter kernels read this via
    /// pointer arg so a captured graph sees the fresh value on each replay
    /// — the prior scalar arg was baked into the kernel-launch node at
    /// capture time and went stale as `seq.seq_len` advanced.
    /// `verify_d.rs` writes the current `seq.seq_len` here before each K=γ
    /// verify step. `DevicePtr::NULL` when DFlash is disabled.
    pub tree_kv_indir_base_persistent: spark_runtime::gpu::DevicePtr,
    /// Pinned-host shadow of `tree_kv_indir_base_persistent` (1×i32).
    /// `verify_d.rs` writes `seq.seq_len` into this pinned buffer then
    /// fires `cuMemcpyHtoDAsync` from it on the verify stream. Pinned
    /// host memory establishes a proper stream-ordered dependency for
    /// the H2D copy (pageable Vec sources do not, leading to a race
    /// where captured graph kernels dereference the device buffer
    /// before the upload completes → stale `kv_indir_base` → `1, 2, 2, 3`
    /// style token-duplication artifacts).
    /// `null` when DFlash is disabled.
    pub tree_kv_indir_base_host_pinned: *mut u8,
    /// ATLAS_TREE_KV_PACK: per-attention-layer packed-KV scratch pool.
    /// One entry per FullAttention layer; each is the K-pool base pointer
    /// for `[num_seqs blocks × stride tokens]` of the layer's KV dtype.
    /// `verify_d.rs` triggers a small scatter kernel that materializes the
    /// ancestor chain for each row into this scratch, and the existing
    /// `paged_decode_attn_*` kernels run over it with NULL indirection
    /// (fast BC=4 batched path). Empty when DFlash is disabled or
    /// `ATLAS_TREE_KV_PACK` is unset.
    pub tree_kv_pack_scratch_k: Vec<spark_runtime::gpu::DevicePtr>,
    /// Same layout as `tree_kv_pack_scratch_k` for V pool.
    pub tree_kv_pack_scratch_v: Vec<spark_runtime::gpu::DevicePtr>,
    /// Identity block table for the packed-KV path: `[num_seqs × 1]` i32
    /// with `bt[seq] = seq`. Allocated once.
    pub tree_kv_pack_block_table: spark_runtime::gpu::DevicePtr,
    /// `seq_lens` buffer for the packed-KV path: `[num_seqs]` i32 holding
    /// `depth[t] + 1` (= chain length for row t). Uploaded per-step by
    /// `verify_d.rs` when the packed-KV path is active.
    pub tree_kv_pack_seq_lens: spark_runtime::gpu::DevicePtr,
    /// Bytes per scratch block (one block per seq). Equals
    /// `stride * num_kv_heads * head_dim * elem_bytes` for FP8 (and
    /// `nvfp4_data_bytes + nvfp4_scale_bytes` for NVFP4). Stored in u64
    /// because the consumer kernel takes `block_stride_bytes` as u64.
    pub tree_kv_pack_block_stride_bytes: u64,
    /// NVFP4-only: data-section size in bytes per block (scales follow
    /// immediately after). Zero for FP8.
    pub tree_kv_pack_data_section_bytes: u64,
    /// FP8 scatter kernel handle (`tree_kv_scatter_fp8`). `KernelHandle(0)`
    /// when not loaded.
    pub tree_kv_pack_scatter_fp8_kernel: spark_runtime::gpu::KernelHandle,
    /// NVFP4 scatter kernel handle (`tree_kv_scatter_nvfp4`).
    pub tree_kv_pack_scatter_nvfp4_kernel: spark_runtime::gpu::KernelHandle,
    /// True when `ATLAS_TREE_KV_PACK=1` AND scratch is allocated AND
    /// at least one scatter kernel is loaded.
    pub tree_kv_pack_active: bool,
    pub(super) embed_tokens: DenseWeight,
    pub(super) final_norm: DenseWeight,
    pub(super) lm_head_weight: DenseWeight,
    pub(super) lm_head_nvfp4: Option<QuantizedWeight>,
    pub(super) layers: Vec<Box<dyn TransformerLayer>>,
    pub(super) buffers: BufferArena,
    pub(super) kv_cache: Mutex<PagedKvCache>,
    pub(super) gpu: Box<dyn GpuBackend>,
    pub(super) rms_norm_kernel: KernelHandle,
    pub(super) bf16_to_f32_kernel: KernelHandle,
    pub(super) dense_gemv_kernel: KernelHandle,
    /// FP32-output variant of dense_gemv_bf16. Used by the LM head when
    /// `use_fp32_logits` is true, so the FP32 accumulator is preserved across
    /// the BF16-storage rounding boundary that flips greedy argmax tiebreaks
    /// on Gemma-4-31B (top-1 vs top-2 = 0.125 logit gap = exact BF16 step at
    /// value 16-32 → BF16 store snaps the wrong way and starts a stop-word
    /// loop). Loaded once at model init.
    pub(super) dense_gemv_fp32out_kernel: KernelHandle,
    pub(super) w4a16_gemv_kernel: KernelHandle,
    pub(super) w4a16_gemv_logits_kernel: KernelHandle, // FP32 output for LM head
    pub(super) w4a16_gemm_kernel: KernelHandle,
    pub(super) w4a16_gemv_batch2_kernel: KernelHandle,
    /// W4A16 M=3 GEMV specialized for LM head (large N=vocab).
    /// Replaces the M=3 fallback through `w4a16_gemm` (95% wasted M-tile)
    /// when `ATLAS_LM_HEAD_BATCH3=1`. See `w4a16_gemv_batch3_logits` in
    /// `kernels/gb10/nvfp4/w4a16_gemv.cu`.
    pub(super) w4a16_gemv_batch3_logits_kernel: KernelHandle,
    pub(super) dense_gemm_kernel: KernelHandle,
    pub(super) argmax_kernel: KernelHandle,
    pub(super) argmax_logits_kernel: KernelHandle, // FP32 argmax for logits
    pub(super) batched_embed_kernel: KernelHandle,
    pub(super) fill_slots_kernel: KernelHandle,
    /// Cached CUDA graph for single-sequence decode (layer loop + norm + LM head).
    /// CUDA graph cache for n=1 decode, keyed by `seq.slot_idx`. The captured
    /// graph has SSM h_state/conv_state pointers baked in as kernel arguments,
    /// so a graph captured for slot S can ONLY be replayed for slot S — replay
    /// for any other slot reads/writes the wrong sequence's recurrent state
    /// and produces gibberish for both sequences. With concurrent users we may
    /// alternate between slots in n=1 decode (e.g. via the per-seq fresh-decode
    /// fix in scheduler::step_decode_only), so we keep one graph per slot.
    pub(super) decode_graph: Mutex<std::collections::HashMap<usize, GraphHandle>>,
    /// Cached CUDA graphs for batched decode.
    ///
    /// Two cache schemes coexist:
    /// 1. **padded_n key** (default): keyed by `(vec![], padded_n)`. Used when
    ///    `ATLAS_SSM_MULTI_SEQ_GRAPH` is OFF — graphs are NOT replayed across
    ///    batches because SSM h_state/conv_state pointers are slot-specific.
    /// 2. **slot-set key** (graph mode): keyed by `(sorted_slot_ids, padded_n)`.
    ///    Used when `ATLAS_SSM_MULTI_SEQ_GRAPH=1`. The graph bakes the per-slot
    ///    SSM pool pointers in; replay is safe as long as the same slot set is
    ///    active. When a slot is freed, all entries containing it are dropped
    ///    (`free_sequence`).
    pub(super) batch_decode_graphs: Mutex<HashMap<(Vec<usize>, usize), GraphHandle>>,
    /// Pre-allocated SSM state pool for stable GPU addresses across graph replays.
    pub(super) ssm_pool: SsmStatePool,
    /// SSM state snapshot pool for Marconi prefix caching.
    pub(super) ssm_snapshots: SsmSnapshotPool,
    /// Fixed max blocks per sequence (max_seq_len / block_size + 1).
    /// Used as constant stride in attention metadata for CUDA graph compatibility.
    pub(super) max_blocks_per_seq: u32,
    /// Permanent KV cache block for padding sequences in batched decode.
    pub(super) dummy_kv_block: u32,
    /// Profile mode: skip graphs, sync+time each layer. Set ATLAS_PROFILE=1.
    pub(super) profile: bool,
    /// One-shot profile flag for the next prefill request only. Set
    /// ATLAS_PROFILE_FIRST=1 to capture per-step timing on the first prefill
    /// after startup without disabling CUDA graphs for subsequent decodes.
    /// Consumed (atomically swapped to false) by `prefill_chunk` / `prefill`.
    pub(super) profile_first_pending: std::sync::atomic::AtomicBool,
    /// When true, decode() skips CUDA graph capture/replay. Set during
    /// per-sequence batch decode to prevent SSM state pointer baking.
    pub(super) suppress_graphs: std::sync::atomic::AtomicBool,
    /// MTP draft proposer (built from mtp_weights at init).
    pub(super) proposer: Option<Arc<dyn DraftProposer>>,
    /// Dedicated buffer for saving hidden state before MTP head runs.
    /// Size: hidden_size * 4 bytes (one FP32 vector). MTP overwrites shared
    /// buffers (norm_output etc.), so the target hidden must be saved here first.
    pub(super) mtp_hidden_save: DevicePtr,
    /// Last-K prompt-tail target hidden capture buffer for MTP prefill.
    ///
    /// Allocated when `ATLAS_MTP_LASTK_PREFILL=N` (N>0). Layout:
    /// `[K × hidden_size × fp_size]` (BF16 or FP32 depending on residual mode)
    /// — the last K target-side hidden states from the final prefill chunk are
    /// copied here so the MTP head can replay them through `forward_one` to
    /// populate its own KV cache before the first decode. Without this, MTP's
    /// self-attention sees zero prompt context at long ctx, dropping draft
    /// accept from 1.83 → 0.92 (target_seq=5085 vs mtp_seq=423 at 4K-prompt
    /// + 1K decode).
    pub(super) mtp_lastk_buf: Option<DevicePtr>,
    /// Capacity (K) of `mtp_lastk_buf`. Read once from
    /// `ATLAS_MTP_LASTK_PREFILL` at model init; 0 disables the feature.
    pub(super) mtp_lastk_capacity: usize,
    /// DFlash 5-layer hidden-state stack. Allocated only when a
    /// `BlockDiffusionDraftHead` proposer is built. Layout:
    /// `[5 × hidden_size × bf16]` shallow-to-deep at the layer indices
    /// declared by `dflash_capture_layers`. Holds the most-recently-decoded
    /// token's intermediate hiddens; the drafter consumes them via its `fc`
    /// projection on the next propose() call. None for non-DFlash runs.
    pub(super) dflash_hidden_save: Option<DevicePtr>,
    /// Layer indices to capture for DFlash. Empty when DFlash is disabled.
    /// Sourced from drafter's `dflash_config.target_layer_ids` at model build.
    pub(super) dflash_capture_layers: Vec<usize>,
    /// Cached CUDA graphs for K=2 verification, **keyed by `seq.slot_idx`**.
    /// Same rationale as `decode_graph`: the captured graph has SSM
    /// h_state/conv_state pointers baked in as kernel arguments, so replay for
    /// a different slot writes to the wrong sequence's recurrent state. With
    /// concurrent users alternating through MTP verify, a single
    /// `Option<GraphHandle>` would corrupt both slots' SSM state.
    pub(super) verify2_graph: Mutex<std::collections::HashMap<usize, GraphHandle>>,
    /// Cached CUDA graphs for K=3 verification, keyed by `seq.slot_idx`.
    pub(super) verify3_graph: Mutex<std::collections::HashMap<usize, GraphHandle>>,
    /// Cached CUDA graphs for K=4 verification, keyed by `seq.slot_idx`.
    pub(super) verify4_graph: Mutex<std::collections::HashMap<usize, GraphHandle>>,
    /// Cached CUDA graphs for DFlash K=γ verification, keyed by
    /// `(seq.slot_idx, K)`. K is `tokens.len()` (γ+1 typically). One graph
    /// per (slot, K) — different γ values coexist via the K dimension.
    // Key: (slot_idx, k, pack_active_flag). The pack_active flag forces a
    // separate captured graph for the ATLAS_TREE_KV_PACK fast path so the
    // first (pre-tree, flat-chain) verify can't bake in stale arg pointers.
    pub(super) verify_kgamma_graph: Mutex<std::collections::HashMap<(usize, usize, u32), GraphHandle>>,
    /// Prefix cache for KV block reuse across requests.
    pub(super) prefix_cache: Box<dyn spark_runtime::prefix_cache::PrefixCache>,
    /// Secondary CUDA stream for pipelining checkpoint D2D with MTP propose.
    pub(super) secondary_stream: u64,
    /// CUDA event for GPU-side inter-stream synchronization (avoids CPU-blocking sync).
    pub(super) secondary_event: u64,
    /// Communication backend for expert parallelism (EP) all-reduce.
    /// None for single-GPU (no distributed communication needed).
    pub(super) comm: Option<std::sync::Arc<dyn spark_comm::CommBackend>>,
    /// Small GPU buffer for EP token broadcast (4 bytes).
    pub(super) ep_cmd_buf: DevicePtr,
    /// Self-speculative decoding mode: draft via layer-skipping (no MTP weights needed).
    pub(super) self_speculative: bool,
    /// Last token index passed to save_hidden_for_mtp (for EP broadcast to rank 1).
    pub(super) last_mtp_hidden_idx: std::sync::atomic::AtomicUsize,
    /// Optional vision encoder for VL models (Qwen3-VL).
    pub(super) vision_encoder: Option<crate::layers::VisionEncoder>,
    /// Number of patches encoded by the last prepare_vision_embed() call.
    /// 0 means no vision embeddings pending.
    pub(super) vision_embed_patches: Mutex<usize>,
    /// Per-image `(grid_h_post_merge, grid_w_post_merge)` from the most
    /// recent prepare_vision_embed() call. Used by MRoPE prefill to
    /// assign correct (h, w) spatial position IDs to each image patch
    /// token. Empty when no images are pending.
    pub(super) vision_image_grids: Mutex<Vec<(usize, usize)>>,
    /// Single-entry vision cache: fingerprint of the (grid + pixel)
    /// data the ViT forward last ran on. When the next request's
    /// fingerprint matches, we restore the cached features into
    /// ve.buf_out from `vision_cache_buf` and skip ViT forward.
    /// 0 = no cached forward. See prepare_vision_embed_dispatch for
    /// the read/write flow.
    pub(super) vision_cache_fp: std::sync::atomic::AtomicU64,
    /// Cached `vision_image_grids` paired with `vision_cache_fp` so
    /// the cache hit can restore them without re-running the encoder.
    pub(super) vision_cache_grids: Mutex<Vec<(usize, usize)>>,
    /// Dedicated GPU buffer holding a snapshot of `ve.buf_out` from
    /// the last ViT forward. On cache hit we D2D-copy this back into
    /// `ve.buf_out` so the splice reads stable features even if other
    /// code paths have stomped on `buf_out` since the encode.
    pub(super) vision_cache_buf: Mutex<DevicePtr>,
    /// Allocated size in bytes of `vision_cache_buf`. Grows as needed
    /// when a larger image is encountered.
    pub(super) vision_cache_bytes: std::sync::atomic::AtomicUsize,
    /// Page-locked host staging for batched metadata H2D transfers.
    /// Allocated once at init via cuMemAllocHost, freed in Drop.
    ///
    /// Uses UnsafeCell (not Mutex) because TransformerModel is only accessed
    /// from the scheduler thread after construction. The Model trait requires
    /// Send+Sync for the move to the scheduler thread, but the model is never
    /// accessed from multiple threads simultaneously. A Mutex here caused a
    /// 500x EP=2 decode regression (50 tok/s → 0.1 tok/s) due to contention
    /// with the NCCL all-reduce path.
    pub(super) pinned_staging: std::cell::UnsafeCell<PinnedMetaStaging>,
    /// Save SSM snapshots every N blocks during chunked prefill.
    /// 0 = disabled (leaf-only). When > 0, intermediate checkpoints are saved
    /// at block boundaries, enabling partial prefix SSM restore.
    pub(super) ssm_checkpoint_interval: usize,
    /// Kernel handle for fused SSM state normalization (prevents state explosion
    /// during long chunked prefill — the SSM forgetting bug).
    pub(super) ssm_state_norm_kernel: KernelHandle,
    /// GPU buffer for ssm_state_clamp_norm_fused's pointer table [num_ssm_layers].
    pub(super) ssm_norm_ptrs_buf: DevicePtr,

    // ── Two-phase SSM prefill buffers ──
    // These hold GDN inputs/outputs for the full sequence, allowing the GDN
    // recurrence to run in a single kernel launch while GEMM projections are
    // processed in smaller chunks (memory-bounded).
    //
    // Allocated at model init for max_seq_len tokens. Reused across layers
    // (only one layer runs at a time) and across sequences.
    /// Packed QKV for two-phase SSM prefill: [max_seq_len, conv_dim] BF16.
    /// Layout per token: [Q(key_dim) | K(key_dim) | V(value_dim)].
    pub(super) gdn_buf_qkv: DevicePtr,
    /// Interleaved gate/beta for two-phase SSM prefill: [max_seq_len, 2*num_v_heads] FP32.
    /// Layout per token: [gate(nv) | beta(nv)].
    pub(super) gdn_buf_gate_beta: DevicePtr,
    /// Full-sequence GDN output: [max_seq_len, value_dim] BF16
    pub(super) gdn_buf_out: DevicePtr,
    /// Full-sequence Z gate (for gated RMS norm in phase 3): [max_seq_len, value_dim] BF16
    pub(super) gdn_buf_z: DevicePtr,
    /// Max sequence length these buffers were allocated for.
    pub(super) gdn_buf_max_len: usize,

    /// Logit softcapping kernel: logits = cap * tanh(logits / cap).
    /// KernelHandle(0) = disabled (no softcapping for this model).
    pub(super) logit_softcap_kernel: KernelHandle,
    /// FP32 variant of logit softcap. KernelHandle(0) when not loaded.
    /// Used when `use_fp32_logits` is true.
    pub(super) logit_softcap_fp32_kernel: KernelHandle,
    /// Whether the single-token decode LM head produces FP32 logits (rather
    /// than BF16). True when `config.use_fp32_residual()` AND the LM head is
    /// a dense BF16 weight (no NVFP4 quant). Drives:
    ///   - dense_gemv_bf16_fp32out kernel writes to `logits_fp32_buf`
    ///   - logit_softcap_fp32 kernel applied in place on the FP32 buffer
    ///   - sampler reads FP32 directly, skipping BF16→FP32 expansion
    ///
    /// Gated by config.model_type=="gemma4" via use_fp32_residual() — other
    /// models keep the BF16 path. Prefill / batched-decode lm_head still
    /// write BF16 to `buffers.logits()`; only single-token decode is FP32
    /// because the bug it fixes (greedy argmax tiebreak flip on the BF16
    /// representable boundary at value 16-32) only manifests there.
    pub(super) use_fp32_logits: bool,
    /// FP32 logits scratch [vocab_size × 4 bytes]. NULL when `use_fp32_logits`
    /// is false (no allocation).
    pub(super) logits_fp32_buf: DevicePtr,
    /// Embedding scale kernel: embeddings *= sqrt(hidden_size).
    /// KernelHandle(0) = disabled (no scaling for this model).
    pub(super) embed_scale_kernel: KernelHandle,
}

/// Pinned host memory staging buffer with reusable metadata Vecs.
pub(crate) struct PinnedMetaStaging {
    /// Page-locked host buffer (cuMemAllocHost).
    pub(super) ptr: *mut u8,
    /// Size in bytes.
    pub(super) bytes: usize,
    /// Reusable Vec<u32> for positions (avoids per-chunk heap allocation).
    pub(super) positions: Vec<u32>,
    pub(super) positions_h: Vec<u32>,
    pub(super) positions_w: Vec<u32>,
    /// Reusable Vec<i64> for slot mappings (avoids per-chunk heap allocation).
    pub(super) slots: Vec<i64>,
}

// SAFETY: TransformerModel is constructed on the main thread, then moved to
// the scheduler thread via Box<dyn Model>. After the move, ALL access
// (prefill, decode, batch_decode) happens on the single scheduler thread.
// The Model trait requires Send+Sync for the cross-thread move, but the
// Model is moved to the scheduler thread and accessed exclusively from there.
// UnsafeCell<PinnedMetaStaging> is not inherently Sync, but single-thread
// access is enforced at runtime by the scheduler architecture.
// The raw pointer in PinnedMetaStaging points to cuMemAllocHost memory which
// is process-global and valid from any thread.
unsafe impl Send for TransformerModel {}
// SAFETY: Model methods are only called from the scheduler thread. No concurrent &self access.
unsafe impl Sync for TransformerModel {}
