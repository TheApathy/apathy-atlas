// SPDX-License-Identifier: AGPL-3.0-only

//! Composable transformer layer traits (SDD).
//!
//! Decouples the generic model loop (embed -> layers -> norm -> lm_head)
//! from layer-specific logic (attention vs SSM, MoE vs dense FFN).
//! Adding a new architecture only requires implementing [`TransformerLayer`]
//! for each layer type, not duplicating the model loop.

use std::any::Any;

use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

mod transformer_layer;
pub use transformer_layer::TransformerLayer;

/// Per-layer persistent state tracked across decode steps.
///
/// Attention layers use [`EmptyLayerState`] (KV lives in `PagedKvCache`).
/// SSM layers use [`SsmLayerState`] (recurrent h_state + conv_state).
/// Custom layers can implement this trait for arbitrary state.
pub trait LayerState: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Empty state for layers that store all persistent state externally
/// (e.g., attention layers where KV is in `PagedKvCache`).
pub struct EmptyLayerState;

impl LayerState for EmptyLayerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// SSM layer state: recurrent hidden state + conv1d sliding window.
///
/// Used by Mamba, Gated Delta Net (GDN), and similar recurrent layers.
pub struct SsmLayerState {
    /// Recurrent hidden state: [num_v_heads, v_dim, k_dim] in f32.
    pub h_state: DevicePtr,
    /// Conv1d sliding window state: [d_inner, d_conv] in f32.
    pub conv_state: DevicePtr,
    /// Checkpoint buffer for h_state (allocated lazily for speculative decode).
    pub h_state_checkpoint: Option<DevicePtr>,
    /// Checkpoint buffer for conv_state (allocated lazily for speculative decode).
    pub conv_state_checkpoint: Option<DevicePtr>,
    /// Intermediate h_state snapshots during batched verification.
    /// Element i holds h_state after processing verification token i.
    /// Used by rollback_ssm_states to restore to the correct position.
    pub h_state_intermediates: Vec<DevicePtr>,
    /// Intermediate conv_state snapshots during batched verification.
    pub conv_state_intermediates: Vec<DevicePtr>,
}

impl LayerState for SsmLayerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Pre-uploaded attention metadata device pointers.
///
/// Uploaded once per decode step in the model loop, reused across all
/// 12 attention layers. Eliminates 44 redundant H2D copies per step.
///
/// For batched decode (num_seqs > 1), arrays are contiguous:
/// - positions: `[N]` u32
/// - slots: `[N]` i64
/// - seq_lens: `[N]` i32
/// - block_table: `[N * max_blocks_per_seq]` i32 (row-major)
#[derive(Clone, Copy)]
pub struct AttnMetadataDev {
    /// Position values: `[N]` u32 at this device address. For multi-modal
    /// MRoPE this is the temporal (T) stream; callers set
    /// `positions_h`/`positions_w` to distinct buffers only when the token
    /// stream contains image or video patches.
    pub positions: DevicePtr,
    /// Height (H) position stream for MRoPE-interleaved. When identical
    /// to `positions` (same pointer) the rope reduces to scalar RoPE.
    /// Default: same as `positions`.
    pub positions_h: DevicePtr,
    /// Width (W) position stream for MRoPE-interleaved. Same fallback as
    /// `positions_h`.
    pub positions_w: DevicePtr,
    /// Slot mappings: `[N]` i64 at this device address.
    pub slot: DevicePtr,
    /// Sequence lengths (+1): `[N]` i32 at this device address.
    pub seq_len: DevicePtr,
    /// Block tables: `[N * max_blocks_per_seq]` i32 at this device address.
    pub block_table: DevicePtr,
    /// Number of blocks per sequence row in block_table.
    pub max_blocks_per_seq: u32,
    /// Number of sequences in this batch (1 for single-sequence decode).
    pub num_seqs: u32,
}

/// Q12 batched-prefill device-side metadata.
///
/// The single-stream `AttnMetadataDev` collapses per-stream pointers into
/// concrete device pointers because there's only one stream. For Q12 we
/// dispatch N concurrent prefilling streams through one batched kernel,
/// and the kernel takes:
///   - stacked positions / slot tables (one big buffer with all streams'
///     data concatenated in cu_seqlens order), and
///   - per-stream pointer arrays for block_table / seq_len / h_state.
///
/// Built once per `prefill_batch_chunk_dispatch` call by
/// `stage_batched_attn_metadata`; threaded through the model-level
/// per-layer batched dispatch (`prefill_attn_batched_layer`,
/// `prefill_ssm_batched_layer`) — see `model/trait_impl/prefill_b/batch.rs`.
pub struct BatchedAttnMetadata {
    /// Stacked positions across all streams: `[total_tokens]` u32 at this
    /// address. For MRoPE interleaved this is the temporal (T) stream.
    pub positions_stacked: DevicePtr,
    /// MRoPE H position stream, stacked. Equal to `positions_stacked` when
    /// MRoPE is disabled.
    pub positions_h_stacked: DevicePtr,
    /// MRoPE W position stream, stacked. Equal to `positions_stacked` when
    /// MRoPE is disabled.
    pub positions_w_stacked: DevicePtr,
    /// Stacked slot indices for KV writes: `[total_tokens]` i64.
    pub slot_stacked: DevicePtr,
    /// Per-stream block_table pointer array: `[batch_size]` of `DevicePtr`,
    /// each element pointing to a stream's chunked-prefill block_table.
    /// Used by `prefill_attention_paged_*_batched` kernels.
    pub block_table_ptrs: DevicePtr,
    /// Per-stream seq_len pointer array: `[batch_size]` of `DevicePtr`.
    pub seq_len_ptrs: DevicePtr,
    // Note: `h_state_ptrs` is NOT cached in BatchedAttnMetadata because
    // it's per-layer (each SSM layer's SsmLayerState has its own h_state
    // allocation). `prefill_ssm_batched_layer` stages h_state_ptrs JIT
    // per-layer-call into the model's scratch buffer.
    /// Number of batched streams.
    pub batch_size: u32,
    /// Per-stream chunk_len (SAME for all streams — scheduler-enforced
    /// constraint via `can_batch_prefill_only`).
    pub chunk_len: u32,
    /// Total tokens stacked across streams: `batch_size * chunk_len`.
    pub total_tokens: u32,
    /// Maximum block_table length across the batch (kernel uses for
    /// bounds checking; per-stream block_table reads via the pointer
    /// array dereference).
    pub max_blocks_per_seq: u32,
}

/// Device pointers to full-sequence GDN input/output buffers.
///
/// Used by the two-phase SSM prefill: phase 1 writes GDN inputs here,
/// phase 2 reads them for the single-launch GDN kernel, phase 3 reads output.
///
/// Uses a **packed QKV layout** matching the conv1d output: each token occupies
/// `conv_dim` contiguous BF16 elements as `[Q(key_dim) | K(key_dim) | V(value_dim)]`.
/// This allows simple contiguous memcpy from per-chunk conv1d output buffers.
/// The GDN kernel reads Q/K/V via stride parameters (`qk_stride = conv_dim`,
/// `v_stride = conv_dim`) to index into the packed layout.
pub struct GdnPrefillBuffers {
    /// Packed Q/K/V: [total_len, conv_dim] BF16.
    /// Layout per token: [Q(key_dim) | K(key_dim) | V(value_dim)].
    pub qkv: DevicePtr,
    /// Interleaved gate/beta: [total_len, 2*num_v_heads] FP32.
    /// Layout per token: [gate(nv) | beta(nv)].
    pub gate_beta: DevicePtr,
    /// GDN recurrence output: [total_len, value_dim] BF16.
    pub output: DevicePtr,
    /// Z gate for gated RMS norm: [total_len, value_dim] BF16.
    pub z: DevicePtr,
    /// Total number of tokens across all chunks.
    pub total_len: usize,
}

/// Tree-aware attention KV indirection plumbing.
///
/// `ATLAS_TREE_AWARE_ATTN=1` + tree-mode K=γ verify: each query at compact
/// slot `t` in the tree must attend ONLY to its true ancestor chain (not
/// to sibling/cousin slots that happen to come before it in compact order).
/// The host-side builder fills `kv_indir[t * stride + j]` with the compact
/// index of the j-th ancestor of `t` (j in `[0..depth[t]+1)`). The kernel
/// remaps positions `>= kv_indir_base` via this table; positions
/// `[0..kv_indir_base)` (prior linear context) read normally.
#[derive(Clone, Copy)]
pub struct TreeAwareAttn {
    /// Device pointer to `int32_t[num_seqs * kv_indir_stride]`. Row `t`
    /// holds the compact indices of slot t's ancestors (depth[t]+1 entries,
    /// padded out — the kernel only reads up to `seq_lens[t]-kv_indir_base`).
    pub kv_indir: spark_runtime::gpu::DevicePtr,
    /// CUDA graph fix: position threshold lives in a 1×i32 device buffer.
    /// Positions `[0..*kv_indir_base_ptr)` are prior linear context (read
    /// normally); positions `[*kv_indir_base_ptr..seq_lens[t])` are the
    /// tree-window remapped via indirection. Host writes `seq.seq_len`
    /// here before each K=γ verify step so captured CUDA graphs see the
    /// fresh value on each replay instead of the stale scalar that was
    /// baked into the kernel-launch node at capture time.
    pub kv_indir_base_ptr: spark_runtime::gpu::DevicePtr,
    /// Row stride of `kv_indir`, in i32 slots (= dflash_kgamma typically).
    pub kv_indir_stride: u32,
    /// ATLAS_TREE_KV_PACK: when present, the multi-seq attention dispatch
    /// runs a tiny scatter kernel to pack the ancestor KV into a contiguous
    /// per-layer scratch pool, then calls `paged_decode_attn_*` against the
    /// scratch with NULL indirection (fast BC=4 batched path) instead of
    /// the slower indirected per-position fallback. `None` keeps the
    /// existing tree-aware single-position fallback path active.
    pub pack: Option<TreeKvPack>,
}

/// ATLAS_TREE_KV_PACK plumbing — references the per-attention-layer scratch
/// pools owned by `TransformerModel`. All pointers are stable (allocated at
/// model init); per-step `verify_d.rs` only re-uploads `seq_lens` (chain
/// length per row).
#[derive(Clone, Copy)]
pub struct TreeKvPack {
    /// Number of attention layers (= length of `scratch_k_ptrs` /
    /// `scratch_v_ptrs` arrays). Used to bounds-check `attn_layer_idx`.
    pub num_attn_layers: u32,
    /// Raw pointer to a `[num_attn_layers]` slice of K-pool `DevicePtr`
    /// values. Lifetime: pinned for the duration of the forward pass
    /// (lives on the model). The dispatcher dereferences this with
    /// `attn_layer_idx` to find the K scratch pool for the current layer.
    pub scratch_k_ptrs: *const spark_runtime::gpu::DevicePtr,
    /// Same as `scratch_k_ptrs` for V.
    pub scratch_v_ptrs: *const spark_runtime::gpu::DevicePtr,
    /// Identity block table (`[num_seqs] i32`, value seq_idx) shared by all
    /// layers. The packed scratch has `num_blocks = num_seqs` with one
    /// `stride`-wide block per seq, and the consumer kernel multiplies
    /// `seq_idx * max_blocks_per_seq=1` to land at `bt[seq] = seq`.
    pub identity_block_table: spark_runtime::gpu::DevicePtr,
    /// Per-step `seq_lens` (`[num_seqs] i32`) holding `depth[t] + 1` for
    /// the packed-KV attention call. Uploaded fresh per K=γ verify step.
    pub seq_lens: spark_runtime::gpu::DevicePtr,
    /// Per-block bytes in the scratch pool. Passed as the consumer
    /// kernel's `block_stride_bytes` (NVFP4) / `cache_stride` (FP8).
    pub block_stride_bytes: u64,
    /// NVFP4 data-section bytes per block (0 for FP8). Passed as
    /// `data_section_bytes` to the NVFP4 consumer kernel.
    pub data_section_bytes: u64,
    /// Synthetic block_size (= max_chain_len = kv_indir_stride). Passed
    /// as the consumer kernel's `block_size` argument.
    pub block_size: u32,
    /// FP8 scatter kernel handle (`KernelHandle(0)` if not loaded).
    pub scatter_fp8_kernel: spark_runtime::gpu::KernelHandle,
    /// NVFP4 scatter kernel handle (`KernelHandle(0)` if not loaded).
    pub scatter_nvfp4_kernel: spark_runtime::gpu::KernelHandle,
    /// Active KV cache geometry forwarded to the scatter kernel —
    /// the source paged cache uses `cache_block_size` not `block_size`.
    pub cache_block_size: u32,
    /// Real `max_blocks_per_seq` for the source paged cache (used by the
    /// scatter to index the caller's block_table).
    pub cache_max_blocks_per_seq: u32,
    /// CUDA graph fix: absolute position where the tree window begins
    /// (`= seq.seq_len`, same value as `TreeAwareAttn::kv_indir_base_ptr`).
    /// Stored in a 1×i32 device buffer (in practice the same buffer as
    /// `kv_indir_base_ptr` since both hold `seq.seq_len`). Forwarded to
    /// the scatter kernel so a captured graph sees the fresh value on
    /// each replay instead of the stale scalar baked in at capture time.
    pub abs_base_ptr: spark_runtime::gpu::DevicePtr,
}

// SAFETY: `TreeKvPack` holds raw pointers (`*const DevicePtr`) into a
// `Vec<DevicePtr>` owned by `TransformerModel`. The vector is allocated
// once at model init and never resized, so the pointers stay valid for
// the model's lifetime. The struct is shared across threads in the same
// way `TreeAwareAttn` is — via per-step `ForwardContext` copies. The
// `DevicePtr` values themselves are POD (`Copy`), so reading them is
// race-free.
unsafe impl Send for TreeKvPack {}
unsafe impl Sync for TreeKvPack {}

/// Shared context for a single forward pass step.
///
/// Provides access to GPU, buffers, and config without coupling
/// layer implementations to the model struct.
pub struct ForwardContext<'a> {
    /// Pre-allocated scratch buffers.
    pub buffers: &'a BufferArena,
    /// GPU backend for kernel launches and memory ops.
    pub gpu: &'a dyn GpuBackend,
    /// Model configuration (dimensions, hyperparameters).
    pub config: &'a ModelConfig,
    /// Pre-uploaded attention metadata (None if no attention layers).
    pub attn_metadata: Option<AttnMetadataDev>,
    /// Profile mode: sync+time per-operation within layers.
    pub profile: bool,
    /// Communication backend for expert parallelism (EP) all-reduce.
    /// None when running single-GPU (no distributed communication).
    pub comm: Option<&'a dyn spark_comm::CommBackend>,
    /// True when inside CUDA graph capture (between begin_capture/end_capture).
    /// MoE layers use sync all_reduce (capturable) instead of async (event-based).
    pub graph_capture: bool,
    /// M8A: DDTree parent_ids device tensor for tree-aware GDN dispatch.
    /// `Some(ptr)` when the K=γ verify path is processing a non-flat tree
    /// payload (verify_d.rs uploads it from `a.pending_tree_payload`).
    /// `None` for flat verify — GDN falls through to the fused wy_k path.
    pub ddtree_parent_ids_dev: Option<spark_runtime::gpu::DevicePtr>,
    /// ATLAS_TREE_AWARE_ATTN: optional per-row KV indirection for paged
    /// decode attention during K=γ verify. `None` for the legacy chain-mode
    /// path (every prior consumer just leaves this default).
    pub tree_aware_attn: Option<TreeAwareAttn>,
    /// Graph-safe override base address for the multi-seq SSM pointer
    /// table. When `Some(ptr)`, the SSM multi-seq decode path uses `ptr`
    /// as the `[h_state_ptrs[n] || conv_state_ptrs[n]]` array address
    /// AND SKIPS the in-layer H2D upload — the caller (e.g.
    /// `decode_batch_dispatch` under `ATLAS_SSM_MULTI_SEQ_GRAPH=1`) has
    /// already populated the table BEFORE `begin_capture`. Iteration over
    /// `decode_multi_seq` advances `ssm_layer_idx` in the caller; each
    /// layer call receives its own per-layer slice address. `None` keeps
    /// the legacy in-layer per-step H2D upload (eager / non-graphed
    /// concurrent decode path).
    pub ssm_multi_seq_ptr_table_override: Option<spark_runtime::gpu::DevicePtr>,
}

/// A single transformer layer performing the full per-layer computation.
///
/// Each layer encapsulates:
/// 1. Pre-norm -> attention/SSM -> residual add
/// 2. Post-norm -> FFN/MoE -> residual add
///
/// The generic model loop iterates `layers` without knowing whether
/// each is attention, SSM, MoE, or dense FFN.
#[cfg(test)]
mod tests;
