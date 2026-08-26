// SPDX-License-Identifier: AGPL-3.0-only

//! `TransformerLayer` trait — composable per-layer forward/decode hooks.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::{BatchedAttnMetadata, ForwardContext, GdnPrefillBuffers, LayerState};

pub trait TransformerLayer: Send + Sync {
    /// Install Qwen4-Exp gated-residual mixers after the ordinary core layer
    /// has been assembled. Other architectures fail closed by default.
    fn set_qwen4_hyperconnections(
        &mut self,
        _attn: crate::layers::Qwen4HyperConnection,
        _mlp: crate::layers::Qwen4HyperConnection,
    ) -> Result<()> {
        anyhow::bail!("layer does not support Qwen4 hyperconnections")
    }

    /// Decode one token through this layer, modifying `hidden` in-place.
    ///
    /// # Arguments
    /// * `hidden` - [1, hidden_size] BF16, read and written
    /// * `residual` - [1, hidden_size] BF16, scratch for residual stream
    /// * `state` - Per-layer state (empty for attention, SSM state for recurrent)
    /// * `kv_cache` - Paged KV cache (may be mutated for block allocation)
    /// * `seq_len` - Current sequence length (for position encoding + cache)
    /// * `block_table` - Sequence's block table (may grow if new blocks needed)
    /// * `ctx` - Shared forward context (buffers, gpu, config)
    /// * `stream` - CUDA stream handle
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        // `--high-speed-swap` disk-side IDs parallel to `block_table` (Phase
        // 6.1.c). Layer-agnostic: the same ID indexes a slot in every
        // layer's on-disk file. Empty when the feature is disabled.
        disk_block_ids: &mut Vec<u32>,
        // Per-layer offload progress (Phase 6.1.d critical fix). Layer L
        // reads/writes `disk_last_offloaded_per_layer[L]`. Each layer's
        // offload runs independently because each layer writes its own
        // K/V to a separate region of the on-disk file. Empty when HSS
        // is disabled; SSM/MoE layers ignore it.
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()>;

    /// Prefill N tokens through this layer using GEMM-batched projections.
    ///
    /// Used during prompt processing: reads weight matrices once for all N
    /// tokens (GEMM M=N) instead of N separate GEMV calls. Attention uses
    /// Flash Attention on contiguous Q/K/V. SSM/GDN recurrence remains
    /// sequential per-token.
    ///
    /// # Arguments
    /// * `hidden` - [N, hidden_size] BF16, read and written
    /// * `residual` - [N, hidden_size] BF16, scratch for residual stream
    /// * `num_tokens` - Number of tokens (N)
    /// * `state` - Per-layer state (SSM state updated sequentially)
    /// * `kv_cache` - Paged KV cache (attention layers write K/V for all N)
    /// * `seq_len_start` - Sequence position of first token (usually 0)
    /// * `block_table` - Block table for KV cache (pre-allocated for N tokens)
    /// * `ctx` - Shared forward context (buffers, gpu, config)
    /// * `stream` - CUDA stream handle
    ///
    /// Default: falls back to sequential single-token decode calls.
    ///
    /// `kv_write_start`: number of tokens whose KV cache entries are already
    /// populated (prefix caching). Attention layers skip KV writes for
    /// positions `< kv_write_start`. SSM layers ignore this (recurrent).
    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        for t in 0..num_tokens {
            let offset = t * h * 2; // BF16 = 2 bytes per element
            let h_t = hidden.offset(offset);
            let r_t = residual.offset(offset);
            self.decode(
                h_t,
                r_t,
                state,
                kv_cache,
                seq_len_start + t,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    /// Two-phase SSM prefill — Phase 1: projections and GDN input staging.
    ///
    /// Runs RMS norm, QKVZ projection, BA+gates, conv1d, and L2 norm for a
    /// chunk of `num_tokens` tokens, then copies the GDN inputs (packed QKV,
    /// gate/beta, Z) into the full-sequence `gdn_bufs` at `token_offset`.
    ///
    /// Does NOT run the GDN recurrence — that happens in `prefill_gdn_full`
    /// after all chunks have staged their inputs.
    ///
    /// Attention layers: default falls back to full `prefill` (no phasing).
    #[allow(clippy::too_many_arguments)]
    fn prefill_phase1(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Default: fall back to full prefill (attention layers, non-SSM layers)
        let _ = (gdn_bufs, token_offset);
        self.prefill(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len_start,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_write_start,
            ctx,
            stream,
        )
    }

    /// Two-phase SSM prefill — Phase 2: GDN recurrence on the full sequence.
    ///
    /// Runs the WY4-persistent GDN kernel over all `total_len` tokens in
    /// `gdn_bufs` in a single launch. The kernel reads packed QKV and
    /// gate/beta from the full-sequence buffers and writes the GDN output.
    ///
    /// Only meaningful for SSM layers. Attention layers return `Ok(())`.
    fn prefill_gdn_full(
        &self,
        _state: &mut dyn LayerState,
        _gdn_bufs: &GdnPrefillBuffers,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        Ok(()) // No-op for attention layers
    }

    /// Q12 Path B: batched attention prefill across N stacked-input streams.
    ///
    /// Runs the full attention-layer prefill (rms_norm + residual, QKV proj,
    /// RoPE, KV-write, batched attention compute, O proj, post-attn norm,
    /// FFN, final residual) over `num_tokens = batch_size * chunk_len`
    /// stacked tokens, using `batched_meta` for per-stream metadata
    /// resolution.
    ///
    /// Default impl returns Err — only `Qwen3AttentionLayer` overrides.
    /// SSM/dense layers don't override (they have their own batched paths
    /// or work without batched metadata).
    ///
    /// Caller (model-level `prefill_attn_batched_layer`) is responsible for
    /// ensuring all streams share the same chunk_len, seq_len_start
    /// (q_offset), and that the layer is not MLA / not HDIM=512 / not HSS-
    /// engaged. The override bails Err if any unsupported case is detected.
    fn prefill_inner_batched_q12(
        &self,
        _hidden_stacked: DevicePtr,
        _residual_stacked: DevicePtr,
        _num_tokens: usize,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _batched_meta: &BatchedAttnMetadata,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("prefill_inner_batched_q12: not implemented for this layer type")
    }

    /// Q12 Path B: batched GDN recurrence across N streams.
    ///
    /// Runs the same WY32 / persistent / split4 GDN kernel as
    /// `prefill_gdn_full` but with `batch_size = batch_size` and
    /// `h_state_ptrs` pointing to a device array of N per-stream h_state
    /// pointers (staged by `TransformerModel::stage_h_state_ptrs`).
    /// `gdn_bufs.qkv` / `gate_beta` / `output` are stacked across N
    /// streams contiguously: each stream's data lives at
    /// `b * chunk_len * conv_dim` (BF16) within the buffer.
    ///
    /// Default impl returns `Err` — the SSM layer override implements the
    /// actual batched dispatch using the kernel handles loaded in
    /// commit `8d07ca4`. Attention layers don't override (they don't
    /// have a GDN step).
    fn prefill_gdn_full_batched(
        &self,
        _h_state_ptrs: DevicePtr,
        _gdn_bufs: &GdnPrefillBuffers,
        _batch_size: u32,
        _chunk_len: u32,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!(
            "prefill_gdn_full_batched: layer does not implement batched GDN \
             — caller should fall back to per-stream prefill_gdn_full"
        )
    }

    /// Two-phase SSM prefill — Phase 3: post-GDN processing.
    ///
    /// Reads GDN output and Z gate from `gdn_bufs` at `token_offset`,
    /// then runs gated RMS norm, output projection, residual add, and MoE
    /// for the chunk of `num_tokens` tokens.
    ///
    /// Only meaningful for SSM layers. Attention layers return `Ok(())`.
    #[allow(clippy::too_many_arguments)]
    fn prefill_phase3(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _num_tokens: usize,
        _gdn_bufs: &GdnPrefillBuffers,
        _token_offset: usize,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        Ok(()) // No-op for attention layers
    }

    /// Returns true if this layer is an SSM layer (supports two-phase prefill).
    ///
    /// When true, the model loop can use `prefill_phase1` / `prefill_gdn_full` /
    /// `prefill_phase3` instead of the monolithic `prefill`.
    fn is_ssm_layer(&self) -> bool {
        false
    }

    /// Whether this layer preserves exact DDTree ancestor visibility when
    /// `ForwardContext::tree_aware_attn` is present.
    ///
    /// Non-attention layers return true: they do not consume the paged KV
    /// cache. Attention implementations must override this and return false
    /// for any KV dtype whose decode kernel ignores the tree indirection.
    /// `verify_d.rs` requires every layer to pass before it allows a branch
    /// commit; this deliberately fails closed as new KV formats are added.
    fn ddtree_ancestor_attention_exact(&self) -> bool {
        true
    }

    /// Per-attention-layer DDTree indirection certificate. Non-attention
    /// layers return `None`; attention layers report their cache dtype and
    /// whether the matching tree-aware decode handle is resolved.
    fn ddtree_attention_certificate(&self) -> Option<(KvCacheDtype, bool)> {
        None
    }

    /// Whether this layer can preserve DDTree's convolutional state along a
    /// non-flat parent chain. Non-SSM layers return true because they have no
    /// recurrent convolutional state; SSM implementations must fail closed
    /// when the tree re-root kernel is unavailable.
    fn ddtree_conv_state_exact(&self) -> bool {
        true
    }

    /// Allocate the transposed MoE expert weights used by the coalesced
    /// prefill GEMM kernels. Called as a post-load pass from `factory::build`
    /// after LM-head NVFP4 quantization has freed BF16 headroom, so
    /// memory-tight EP configurations (e.g. MiniMax M2.7-NVFP4 EP=2) can
    /// fit the transpose that layer-0 preflight would otherwise reject.
    ///
    /// Default: no-op (non-MoE layers, and MoE layers whose loader already
    /// called `MoeLayer::transpose_for_prefill` inline during construction).
    fn transpose_moe_for_prefill(
        &mut self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
    ) -> Result<()> {
        Ok(())
    }

    /// Like `transpose_moe_for_prefill` but only transposes the gate+up
    /// projections (skips the down projection), reducing the transpose cost
    /// from 3× to 2× per expert. Used as a memory-tight fallback by the
    /// MiniMax loader when full transpose doesn't fit.
    fn transpose_moe_gate_up_for_prefill(
        &mut self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
    ) -> Result<()> {
        Ok(())
    }

    /// Wire a shared per-prefill `down_proj` transpose scratch into this
    /// layer's MoE block. Used as a memory-tight alternative to the
    /// persistent down transpose: factory allocates one shared scratch,
    /// every MoE layer reuses it layer-by-layer during sequential
    /// prefill. No-op for non-MoE layers and MoE layers that already
    /// have a persistent transposed down.
    fn set_moe_down_transpose_scratch(
        &mut self,
        _scratch_packed: DevicePtr,
        _scratch_scale: DevicePtr,
        _packed_ptrs_t: DevicePtr,
        _scale_ptrs_t: DevicePtr,
    ) {
    }

    /// Phase 8a unified-layout MoE transpose: build persistent transposed
    /// gate/up/down for all experts and free the untransposed copies.
    /// Phased flow keeps memory budget tight enough for MiniMax M2.7 EP=2.
    /// After this call, the untransposed-layout decode kernels can no
    /// longer execute correctly — `MoeLayer::use_t_layout_for_decode()` must
    /// gate dispatch to the `_t` decode kernels. Default no-op.
    fn transpose_moe_for_prefill_unified(
        &mut self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
    ) -> Result<()> {
        Ok(())
    }

    /// Block C Path 2 hybrid-layout MoE transpose: build persistent
    /// transposed gate/up/down alongside the untransposed originals (no
    /// frees). Doubles MoE-weight memory but recovers the ~15 % decode
    /// regression of pure unified mode — decode + MTP verify dispatch
    /// keeps using the warp-reduction kernels on the originals while
    /// prefill (forward_batched) routes through transposed kernels.
    /// Caller must verify enough free memory before invocation. Default
    /// no-op for non-MoE layers.
    fn transpose_moe_for_prefill_hybrid(
        &mut self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
    ) -> Result<()> {
        Ok(())
    }

    /// Decode K tokens through this layer using GEMM-batched projections.
    ///
    /// Used for speculative decode verification: processes multiple tokens
    /// per layer with GEMM for weight-heavy projections (amortizes bandwidth)
    /// and sequential ops for stateful/recurrent components.
    ///
    /// # Arguments
    /// * `hidden` - [K, hidden_size] BF16, read and written (K tokens contiguous)
    /// * `residual` - [K, hidden_size] BF16, scratch for residual stream
    /// * `num_tokens` - Number of tokens (K)
    /// * `state` - Per-layer state
    /// * `kv_cache` - Paged KV cache
    /// * `seq_len` - Starting sequence length (before these tokens)
    /// * `block_table` - Block table for KV cache
    /// * `ctx` - Shared context
    /// * `stream` - CUDA stream
    ///
    /// Default: falls back to sequential single-token decode calls.
    #[allow(clippy::too_many_arguments)]
    fn decode_batched(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        for t in 0..num_tokens {
            let offset = (t * h * 2) as u64; // BF16 = 2 bytes per element
            let h_t = hidden.offset(offset as usize);
            let r_t = residual.offset(offset as usize);
            self.decode(
                h_t,
                r_t,
                state,
                kv_cache,
                seq_len + t,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    /// Decode N sequences through this layer in a single batched call.
    ///
    /// Each sequence contributes 1 token. The weight matrices are loaded
    /// once and applied to all N sequences (amortizing memory bandwidth).
    ///
    /// # Arguments
    /// * `hidden` - [N, hidden_size] BF16, contiguous
    /// * `residual` - [N, hidden_size] BF16, contiguous
    /// * `num_seqs` - Number of sequences (N)
    /// * `states` - N per-layer states (one per sequence)
    /// * `kv_cache` - Shared paged KV cache
    /// * `ctx` - Forward context (attn_metadata contains N-sequence metadata)
    /// * `stream` - CUDA stream
    ///
    /// Default: falls back to N sequential single-token decode calls.
    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        for i in 0..num_seqs {
            let offset = i * h * 2;
            let h_i = hidden.offset(offset);
            let r_i = residual.offset(offset);
            let mut bt = block_tables[i].clone();
            // Phase 6.1: per-seq disk_block_ids aren't threaded through this
            // default impl yet (chunked-prefill / batched-decode are Phase 6.2
            // scope). Pass empty stubs so the trait sig is satisfied; layers
            // that need disk IDs (attention) override decode_multi_seq.
            let mut stub_disk = Vec::<u32>::new();
            let mut stub_last_offloaded = Vec::<u32>::new();
            self.decode(
                h_i,
                r_i,
                states[i],
                kv_cache,
                seq_lens[i],
                &mut bt,
                &mut stub_disk,
                &mut stub_last_offloaded,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    /// CROSS-SEQ BATCHED DFLASH VERIFY (#39 v2): project Q/K/V for `num_rows`
    /// contiguous rows of `hidden` (all `c*K` rows across every sequence) in
    /// ONE weight read per Q/K/V, writing the result into `qkv_out_base` in the
    /// per-seq-strided layout that [`Self::decode_multi_seq_attn_from_qkv`]
    /// consumes. RMS-norm (`input_norm`) is applied inline. `hidden` is only
    /// read (the residual stream is untouched).
    ///
    /// Default: unimplemented (only the attention layer overrides). Returns an
    /// error so the caller falls back to the per-seq v1 path.
    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq_qkv_batched(
        &self,
        _hidden: DevicePtr,
        _num_rows: usize,
        _qkv_out_base: DevicePtr,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("decode_multi_seq_qkv_batched not implemented for this layer type")
    }

    /// CROSS-SEQ BATCHED DFLASH VERIFY (#39 v2): run attention phases 3-7 for
    /// ONE sequence's `num_seqs` (= K) rows, reading Q/K/V from `qkv_base` (a
    /// slice of the shared `qkv_output` populated by
    /// [`Self::decode_multi_seq_qkv_batched`]). Phases 1-2 are skipped.
    ///
    /// Default: unimplemented (only the attention layer overrides).
    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq_attn_from_qkv<'a, 'b: 'a>(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _num_seqs: usize,
        _states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        _ctx: &ForwardContext,
        _stream: u64,
        _qkv_base: DevicePtr,
    ) -> Result<()> {
        anyhow::bail!("decode_multi_seq_attn_from_qkv not implemented for this layer type")
    }

    /// CROSS-SEQ BATCHED DFLASH VERIFY (#39): run this layer's FFN ONCE over
    /// `total_rows` rows that were collected from every sequence's mixer output
    /// (via `ctx.ffn_defer`), then add the FFN output back into `hidden`.
    ///
    /// `ffn_input` is a contiguous `[total_rows, H]` BF16 buffer holding the
    /// post-mixer FFN input for every sequence's K rows (written by the mixer's
    /// deferred path). `hidden` is the same `[total_rows, H]` residual stream.
    /// This is the weight-amortizing step: the FFN weights are read ONCE for
    /// all `c×K` rows instead of once per sequence.
    ///
    /// Default: unimplemented (only the concrete SSM/attention layers with an
    /// FFN override it). Layers with no FFN return `Ok(())`.
    #[allow(clippy::too_many_arguments)]
    fn run_deferred_ffn(
        &self,
        _ffn_input: DevicePtr,
        _hidden: DevicePtr,
        _total_rows: usize,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("run_deferred_ffn not implemented for this layer type (#39 batched verify)")
    }

    /// Allocate per-sequence state for this layer.
    ///
    /// Called once when a new sequence is created. Returns:
    /// - `EmptyLayerState` for pure attention layers
    /// - `SsmLayerState` for SSM/recurrent layers
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>>;

    /// Piecewise-CUDA-graph hook: report whether, for a batch of `num_seqs`
    /// sequences, this layer's multi-seq decode takes the address-stable
    /// path (all per-slot device addresses indirected through fixed
    /// scratch), making it safe to include in a captured segment graph.
    ///
    /// Default `false` — a layer that returns `false` forces a segment
    /// boundary (the piecewise dispatcher runs it eagerly). Overridden by
    /// SSM layers that take the multi-seq kernel path.
    fn multiseq_graph_safe(&self, _num_seqs: usize) -> bool {
        false
    }

    /// Piecewise-CUDA-graph hook: gather-before-replay. Refresh any
    /// layer-local host→device pointer table (e.g. SSM h_state/conv_state
    /// indirection) with the current active sequences' addresses so a
    /// previously-captured segment graph replays against live state.
    ///
    /// Default no-op — layers with no per-slot indirection need nothing.
    /// Called by the piecewise dispatcher BEFORE replaying a segment that
    /// contains this layer, OUTSIDE the captured region. See
    /// `Qwen3SsmLayer::refresh_multi_seq_ptr_table`.
    #[allow(clippy::too_many_arguments)]
    fn multiseq_refresh_ptr_table<'a, 'b: 'a>(
        &self,
        _states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _num_seqs: usize,
        _gpu: &dyn GpuBackend,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// WY17 LAZY-commit hook: the `gated_delta_rule_wy17_replay` kernel handle
    /// for this layer, or a null handle (`KernelHandle(0)`) if this layer type
    /// / target doesn't provide it. The async-checkpoint commit path uses it to
    /// reconstruct a skipped (non-checkpoint) intermediate H slot on a partial
    /// accept. Default null — only SSM layers with the lazy kernel loaded
    /// return a live handle. See `Qwen3SsmLayer`.
    fn wy17_replay_kernel(&self) -> spark_runtime::gpu::KernelHandle {
        spark_runtime::gpu::KernelHandle(0)
    }

    /// GDN LAZY-commit (ExactSequence path, `ATLAS_SSM_GDN_LAZY=1`): TRUE when
    /// a `num_tokens`-wide verify on this layer dispatched the `lazyfinal`
    /// kernel (no per-token H snapshots; h_state left at step-initial). Pure
    /// function of num_tokens + process-constant env + kernel handles, so the
    /// async commit can mirror the dispatch decision exactly (graph-safe, no
    /// per-step mutable flag). Default false — non-SSM layers.
    fn gdn_seq_lazy_engaged(&self, _num_tokens: usize) -> bool {
        false
    }

    /// The FP32 replay kernel for the lazy ExactSequence commit — the nosnap
    /// sequence kernel, which re-runs the identical recurrence over the
    /// retained inputs. Null when unavailable.
    fn gdn_seq_replay_kernel(&self) -> spark_runtime::gpu::KernelHandle {
        spark_runtime::gpu::KernelHandle(0)
    }

    /// Layer-owned retention buffer for the lazy ExactSequence commit:
    /// [16 rows x qkvz_size] fp32 of this step's post-conv q/k/v(+z), with the
    /// [16 x 2*nv] fp32 gate/beta block at byte offset 16*qkvz_size*4.
    /// None until first lazy dispatch allocates it. Default None.
    fn gdn_seq_lazy_retain(&self) -> Option<spark_runtime::gpu::DevicePtr> {
        None
    }

    /// TRUE when this layer's K=γ GDN dispatch would take the tree-aware
    /// kernel branch for a ForwardContext with `ddtree_parent_ids_dev` set
    /// (i.e. `gdn_tree_k` is loaded). Used by `verify_d.rs` to decide whether
    /// the graph-safe flat-chain parent injection actually reroutes the SSM
    /// to `gated_delta_rule_tree_wy` (which leaves `h_state` stale — the
    /// commit must know; task #34). Default false — non-SSM layers.
    fn gdn_tree_kernel_loaded(&self) -> bool {
        false
    }

    /// TRUE when a `num_tokens`-wide verify on this layer runs the LAZY wy17
    /// kernel (`gated_delta_rule_wy17_lazy`, sparse intermediate H writes).
    /// The async-checkpoint commit consults this before choosing the
    /// `gated_delta_rule_wy17_replay` path: replay is only bit-exact when the
    /// lazy kernel populated the retention buffers THIS verify — a K≠17
    /// (chunked) verify under global `ATLAS_WY17_LAZY` env gates must use the
    /// plain intermediate D2D copy instead (task #34 sibling hazard).
    /// Default false — non-SSM layers. See `Qwen3SsmLayer`.
    fn wy17_lazy_engaged(&self, _num_tokens: usize) -> bool {
        false
    }
}
