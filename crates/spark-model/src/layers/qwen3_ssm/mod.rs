// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3-Next SSM (Gated Delta Net) layer implementing TransformerLayer.
//!
//! Corrected pipeline matching the HuggingFace reference implementation:
//!   1. QKVZ projection (interleaved output)
//!   2. Deinterleave QKVZ → sequential [Q | K | V | Z]
//!   3. BA projection (interleaved output)
//!   4. Compute GDN gates: gate = exp(-A * softplus(alpha + dt_bias)), beta = sigmoid(b)
//!   5. Conv1d update on [Q | K | V] concatenated (d_inner=8192)
//!   6. Split conv output → Q', K', V'
//!   7. GDN decode (Q', K', V', gate, beta) — kernel handles GQA internally
//!   8. Gated RMS norm (GDN output, Z gate)
//!   9. Output projection [value_dim → hidden_size]
//!  10. MoE FFN

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle, PinnedHostBuffer};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{
    ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::layers::{FfnComponent, Qwen4HyperConnection};
use crate::weight_map::{DenseWeight, Fp8Weight, QuantizedWeight, SsmWeights};

mod exact_flat_route;
pub(crate) mod ssm_h_fp16;
use exact_flat_route::{ExactFlatSsmRoute, contiguous_intermediate_base, exact_flat_ssm_route};

/// Qwen3-Next SSM/GDN layer (36 of 48 layers).
///
/// Supports two QKVZ projection modes:
/// - **Interleaved** (80B): `w4a16_gemv_qkvz` or GEMV + `deinterleave_qkvz`
/// - **Sequential** (3.5-35B): plain GEMV → `[Q|K|V|Z]` already in order
#[allow(dead_code)]
pub struct Qwen3SsmLayer {
    input_norm: DenseWeight,
    ssm: SsmWeights,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    qwen4_attn_hyper: Option<Qwen4HyperConnection>,
    qwen4_mlp_hyper: Option<Qwen4HyperConnection>,
    // NVFP4-quantized QKVZ weight (quarters bandwidth vs BF16)
    qkvz_nvfp4: Option<QuantizedWeight>,
    // Transposed [K/2, N] copy for coalesced w4a16_gemm reads (prefill)
    qkvz_nvfp4_t: Option<QuantizedWeight>,
    // Transposed out_proj for prefill GEMM
    out_proj_nvfp4_t: Option<QuantizedWeight>,
    // BF16 out_proj for models where SSM weights are not pre-quantized
    pub out_proj_dense: Option<DenseWeight>,
    // FP8 E4M3 checkpoint weights for native FP8 serving (w8a16_gemv LUT kernel)
    qkvz_fp8w: Option<Fp8Weight>,
    out_proj_fp8w: Option<Fp8Weight>,
    /// When true, QKVZ projection output is already sequential [Q|K|V|Z].
    /// Skips the deinterleave kernel (used by Qwen3.5 where QKV+Z are
    /// concatenated at load time rather than interleaved per-group).
    sequential_qkvz: bool,
    // Kernels — decode path (single-token GEMV)
    rms_norm_residual_k: KernelHandle,
    gated_rms_norm_k: KernelHandle,
    gated_rms_norm_f32_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    /// K=3 batched counterpart of `dense_gemv_k` for the SSM BA projection.
    /// Single launch handles all 3 tokens, eliminating ~48 μs of launch
    /// overhead per SSM layer per K=3 verify step. Gated by
    /// `ATLAS_SSM_BA_BATCHED=1`; NULL handle when the kernel isn't in the
    /// active target's PTX bundle, in which case the per-token loop runs.
    dense_gemv_batch3_k: KernelHandle,
    /// General batched (grid.y = token) counterpart of `dense_gemv_k` for
    /// the SSM BA projection at any num_tokens (DFlash K=γ=17 included).
    /// Bit-identical to the per-token loop (same kernel body per y-block).
    /// Gated by `ATLAS_SSM_BA_BATCH=1`; NULL handle → per-token loop.
    dense_gemv_batchn_k: KernelHandle,
    /// Exact multi-row BA projection with inline FP32 gate transforms.
    ba_gates_batchn_exact_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    /// Exact K1-order multi-row NVFP4 kernels shared by QKVZ and out_proj.
    /// A missing selected tier fails closed to ordinary row-major K1 GEMVs.
    w4a16_exact_projection_kernels: ops::W4a16ExactLmHeadKernels,
    /// Single-warp `w4a16_gemv_sw` — lossless, 8 outputs per 256-thread block
    /// instead of 4, no cross-warp smem round-trip. `KernelHandle(0)` on a
    /// miss falls back to the bit-identical 64-thread base GEMV.
    w4a16_gemv_sw_k: KernelHandle,
    /// `ATLAS_NO_GEMV_SW != "1"`, cached at construction.
    gemv_sw: bool,
    w8a16_gemv_k: KernelHandle,
    w4a16_gemv_qkvz_k: KernelHandle,
    /// Fused rms_norm_residual + w4a16_gemv for SSM QKVZ (sequential layout).
    /// Gated by ATLAS_FUSE_SSM_QKVZ=1 at decode dispatch time. NULL handle if
    /// the kernel module didn't load (older PTX bundles).
    fused_rms_qkvz_k: KernelHandle,
    /// K=3 batched counterpart of `fused_rms_qkvz_k` (3-token verify path).
    /// Same env gate; NULL handle when not built for the active target.
    fused_rms_qkvz_batch3_k: KernelHandle,
    deinterleave_k: KernelHandle,
    conv1d_k: KernelHandle,
    conv1d_l2norm_k: KernelHandle,
    conv1d_l2norm_f32_k: KernelHandle,
    /// Exact flat-chain multi-row FP32 conv recurrence + state snapshots.
    conv1d_l2norm_f32_sequence_k: KernelHandle,
    gdn_k: KernelHandle,
    gdn_f32_k: KernelHandle,
    /// Exact flat-chain multi-row FP32 GDN recurrence + H snapshots.
    gdn_f32_sequence_k: KernelHandle,
    /// Register-resident H variant (identical arithmetic, H cached in regs).
    gdn_f32_sequence_persistent_k: KernelHandle,
    /// Speed-probe twin with per-token snapshot writes removed (ATLAS_SSM_GDN_NOSNAP).
    /// INVALIDATES rollback — measurement only.
    gdn_f32_sequence_nosnap_k: KernelHandle,
    /// LAZY-COMMIT variant (`ATLAS_SSM_GDN_LAZY=1`): registers-resident, writes
    /// NO per-token snapshots (58.6% of the kernel at k=16), writes final H to
    /// inter[num_tokens-1], and leaves `h_state` holding the step-initial H so
    /// the commit replay (the nosnap kernel over retained inputs) can
    /// reconstruct any accepted state bit-exactly. Requires the async commit
    /// integration in `async_chkpt.rs` — never enable without it.
    gdn_f32_sequence_lazyfinal_k: KernelHandle,
    /// Lazy-commit retention: [GDN_LAZY_MAX_K x qkvz] fp32 q/k/v(+z) + the
    /// gate/beta block at GDN_LAZY_MAX_K*qkvz*4. Copied BEFORE the GDN launch
    /// (inputs are exactly what the kernel reads — no aliasing question).
    /// Lazily allocated on first engaged dispatch.
    gdn_lazy_retain: std::sync::OnceLock<DevicePtr>,
    ba_gates_k: KernelHandle,
    residual_add_k: KernelHandle,
    l2_norm_k: KernelHandle,
    residual_add_rms_norm_k: KernelHandle,
    gated_rms_norm_prefill_k: KernelHandle,
    // Kernels — batched verification path (multi-token GEMM)
    w4a16_gemm_k: KernelHandle,
    /// Byte-exact cp.async pipelined shadow of `w4a16_gemm` (see
    /// `prefill_proj_pipe_enabled`). Used by the SSM QKVZ + out_proj prefill
    /// projections at `ATLAS_PREFILL_PROJ_PIPE=1`; handle 0 falls back.
    w4a16_gemm_pipe_k: KernelHandle,
    w4a16_gemm_t_k: KernelHandle, // Transposed B layout [K/2, N] — K_STEP_T=32
    w4a16_gemm_t_k64_k: KernelHandle, // K64 variant: K_STEP_T=64, halves outer loop
    w4a16_gemm_t_m128_k: KernelHandle, // M128 variant: 2 M-chunks per CTA, halves B re-reads
    /// M16 variant: 1 CTA row × 4 warps × 32-N each (K=γ verify, M≤32).
    /// Gated by `ATLAS_TC_NVFP4_M16=1` env var. KernelHandle(0) if not compiled
    /// for this target (qwen3.6-27b NVFP4 shadow only as of 2026-05-19).
    w4a16_gemm_t_m16_k: KernelHandle,
    /// M32×N64 variant: K=γ verify small-M GEMM (single B read for M≤32 ×
    /// N_TILE=64 → full SM occupancy). Used by the SSM `out_proj`
    /// [M=17, N=5120, K=6144] at 3 < M ≤ 32 to replace the N_TILE=128
    /// `w4a16_gemm_t` (only 40 CTAs at N=5120, SM-starved). KernelHandle(0)
    /// if the kernel is not compiled for this target.
    w4a16_gemm_t_m32_n64_k: KernelHandle,
    /// Split-K variant of the m32_n64 kernel + its FP32→BF16 reduce.
    /// Used by the K=γ verify `out_proj` (ATLAS_SSM_OUT_SPLITK) and
    /// `qkvz` (ATLAS_SSM_QKVZ_SPLITK) routes: the single-slice out_proj
    /// fields only 80 CTAs on 48 SMs (floor-map measured 28% of the DRAM
    /// floor — 235µs vs 66µs); slicing K across gridDim.z into an FP32
    /// workspace reaches 84% (91µs). Mirrors the proven `ffn_down`
    /// split-K path (lossless FP32 partials, token-exact). Handle 0 when
    /// the PTX bundle lacks the kernels — routes silently stay off.
    w4a16_gemm_t_m32_n64_splitk_k: KernelHandle,
    reduce_splitk_k: KernelHandle,
    /// Lazily-allocated FP32 split-K workspace [k_splits≤8, 32, max_n].
    /// Allocated at load time (pre-graph-capture) by `alloc_ssm_splitk_ws`.
    ssm_splitk_workspace: std::sync::Mutex<Option<DevicePtr>>,
    w4a16_gemv_batch2_k: KernelHandle,
    dense_gemm_k: KernelHandle,
    gdn_prefill_k: KernelHandle,
    gdn_prefill_split_k: KernelHandle,
    gdn_prefill_split4_k: KernelHandle,
    gdn_prefill_persistent_k: KernelHandle,
    gdn_prefill_persistent_wy4_k: KernelHandle,
    /// WY32 chunked prefill: processes 32 tokens per WY iteration with H in
    /// shared memory. ~30x faster than per-token for 14k+ sequences.
    gdn_prefill_wy32_k: KernelHandle,
    // ── Q12 Phase 2b: same-chunk-len batched GDN prefill kernels ──
    // Each takes `float* const* h_state_ptrs` plus stacked QKV/gate/beta/output.
    // Used by `Qwen3SsmLayer::prefill_batched` when N≥2 streams have matching
    // chunk_len. Null on targets that don't carry the corresponding kernel.
    gdn_prefill_wy32_batched_k: KernelHandle,
    gdn_prefill_persistent_batched_k: KernelHandle,
    gdn_prefill_persistent_wy4_batched_k: KernelHandle,
    gdn_prefill_split4_batched_k: KernelHandle,
    compute_gdn_gates_k: KernelHandle,
    /// Multi-seq variants: advance `c` SSM states in ONE launch. Gated
    /// by `ATLAS_SSM_MULTI_SEQ_KERNEL=1` (additive on top of
    /// `ATLAS_SSM_MULTI_SEQ_BATCHED=1`). KernelHandle(0) if the multi-seq
    /// PTX modules aren't in the active target's bundle, in which case
    /// the trait_decode_multi_seq path falls back to the per-seq loop.
    conv1d_multi_seq_k: KernelHandle,
    conv1d_l2norm_multi_seq_k: KernelHandle,
    gdn_decode_multi_seq_k: KernelHandle,
    compute_gdn_gates_multi_seq_k: KernelHandle,
    /// FP32-output multi-seq variants. These are the production-precision
    /// kernels that the AEON-Q36-27B decode path uses when
    /// `ATLAS_SSM_MULTI_SEQ_KERNEL=1`. The BF16-output variants above
    /// (conv1d_l2norm_multi_seq_k / gdn_decode_multi_seq_k) preserve API
    /// symmetry but aren't dispatched in production because the model is
    /// calibrated for FP32 recurrent precision. KernelHandle(0) when the
    /// PTX module isn't compiled for the active target.
    conv1d_l2norm_f32_multi_seq_k: KernelHandle,
    gdn_decode_f32_multi_seq_k: KernelHandle,
    /// Multi-seq gated_rms_norm with FP32 input + per-seq strides. Lets
    /// the multi_seq decode collapse the per-seq gated_rms_norm loop into
    /// one launch, while writing into a value_dim-contig output buffer so
    /// the subsequent out_proj can fire as a batched w4a16_gemm at M=n.
    /// KernelHandle(0) when not compiled for the active target.
    gated_rms_norm_f32_multi_seq_k: KernelHandle,
    /// K=3 fused conv1d-update + L2 norm + intermediate-state save for the
    /// K=3 verify SSM forward. Replaces the per-token loop (3
    /// `conv1d_l2norm` + 3 d2d copies) with one launch. KernelHandle(0)
    /// when the PTX module isn't in the active target's bundle, in which
    /// case the per-token fallback runs.
    conv1d_l2norm_chunk3_k: KernelHandle,
    /// Scratch device buffer for per-seq state pointer arrays uploaded
    /// before each multi-seq kernel launch. Sized at init for the
    /// configured `max_batch_size`. Layout: `[h_state_ptrs[c],
    /// conv_state_ptrs[c]]` u64 each (each c × 8 bytes), so total
    /// `2 * max_c * 8` bytes. Pre-allocated to avoid per-call alloc.
    ssm_multi_seq_ptr_scratch: DevicePtr,
    /// Stable page-locked HOST staging buffer for the multi-seq ptr
    /// table H2D upload. The pre-Fix-B code used a stack array `[u64;
    /// 64]` as the H2D source — CUDA graph capture recorded that stack
    /// address, and on graph replay the GPU read invalid memory after
    /// the stack frame was gone. Backend-owned pinned storage gives a stable
    /// source address: graph capture records the pointer, replay
    /// reads the CURRENT contents (which the CPU updates fresh every
    /// step). 64 u64s = 512 bytes, well below MAX_C=32 × 2 = 64 slots.
    ///
    /// Wrapped in `Box<UnsafeCell<...>>` for interior mutability behind
    /// the `&self` decode forward. SAFETY contract: the layer is only
    /// touched on one stream at a time (decode_a2.rs serialises layer
    /// dispatch within a verify step), so concurrent mutation is
    /// impossible — even though Rust can't prove it.
    multi_seq_ptr_host: Box<std::cell::UnsafeCell<PinnedHostBuffer>>,
    /// Capacity of the pointer scratch in number of sequences. Acts as
    /// a hard cap on `num_seqs` for the multi-seq kernels (callers fall
    /// back to per-seq loop if `n > ssm_multi_seq_ptr_max`).
    ssm_multi_seq_ptr_max: usize,
    ba_gates_prefill_k: KernelHandle,
    // Kernels — prefill (multi-token sequential)
    conv1d_prefill_k: KernelHandle,
    // Kernels — fused chunk2 path (2-token verification)
    gdn_chunk2_k: KernelHandle,
    conv1d_chunk2_k: KernelHandle,
    // Kernels — fused chunk3 path (3-token verification)
    gdn_chunk3_k: KernelHandle,
    w4a16_gemv_batch3_k: KernelHandle,
    // Kernels — WY-chunkwise path (2-pass verification)
    gdn_wy2_k: KernelHandle,
    gdn_wy3_k: KernelHandle,
    gdn_wy4_k: KernelHandle,
    // ATLAS_SSM_H_FP16 twins of the three chunked-path WY kernels. The chunked
    // K>=5 route (`pick_chunk`) is what a k=13 verify actually runs — 4+4+3+2 —
    // so these three cover it. Zero when the PTX lacks the symbol; the call
    // sites turn that into a hard error rather than a silent FP32 fallback,
    // because reading an FP16 pool as floats produces fluent garbage, not a
    // crash.
    gdn_wy2_f16_k: KernelHandle,
    gdn_wy3_f16_k: KernelHandle,
    gdn_wy4_f16_k: KernelHandle,
    /// WY-Chunkwise K=17 GDN verify (DFlash γ+1). Only present in
    /// qwen3.6-35b-a3b's PTX module set; NULL handle for other targets,
    /// in which case decode_batched(K=17) falls through to the sequential
    /// per-token path.
    gdn_wy17_k: KernelHandle,
    /// V-dim-split wy17 (`gated_delta_rule_wy17_vsplit`): occupancy variant
    /// that fans the 48-head wy17 launch across `ATLAS_WY17_SPLIT` v-column
    /// bands (gridDim.z). Bit-identical output to `gdn_wy17_k`. NULL handle
    /// when the kernel isn't in the active target's PTX bundle → the K=17
    /// path uses `gdn_wy17_k` unchanged.
    gdn_wy17_vsplit_k: KernelHandle,
    /// LAZY Hi-writes wy17 (`gated_delta_rule_wy17_lazy`): takes runtime
    /// `lazy_j`; lazy_j==1 is bit-identical to `gdn_wy17_k`, lazy_j>1
    /// persists only checkpoint intermediate slots (86%-of-traffic cut).
    /// NULL handle when not in the active target's PTX bundle → dispatch
    /// uses `gdn_wy17_k` (all slots) unchanged. Gated by `ATLAS_WY17_LAZY`.
    gdn_wy17_lazy_k: KernelHandle,
    /// Combined LAZY + V-DIM SPLIT wy17 (`gated_delta_rule_wy17_lazy_vsplit`):
    /// fuses both benefits — checkpoint-gated Hi-writes (lazy_j) AND vsplit
    /// occupancy (96 CTAs on 48 SMs). Preferred over `gdn_wy17_lazy_k` when
    /// ATLAS_WY17_SPLIT is also set. NULL on targets that predate the kernel.
    gdn_wy17_lazy_vsplit_k: KernelHandle,
    /// Replay kernel (`gated_delta_rule_wy17_replay`) that reconstructs one
    /// skipped intermediate slot bit-exactly for the commit path under
    /// lazy_j>1. NULL when not compiled. Exposed for async_chkpt wiring.
    pub(crate) gdn_wy17_replay_k: KernelHandle,
    /// M8A: tree-aware GDN kernel for DDTree verify with non-flat branches.
    /// Sequential per-token loop with parent_ids state load. NULL handle
    /// when not compiled for the active target.
    pub(crate) gdn_tree_k: KernelHandle,
    /// M8A v2: tree-aware WY-fused GDN kernel. Bit-equivalent to wy17 on
    /// flat chains, supports arbitrary tree topology via ancestor walk in
    /// WY correction. Preferred over `gdn_tree_k` when present.
    pub(crate) gdn_tree_wy_k: KernelHandle,
    /// Tree-aware conv state re-root (ATLAS_DDTREE_TREE_CONV_EXACT=1).
    /// Before processing token t, copies conv_inter[parent[t]] → conv_state
    /// when parent[t] != t-1 (branch token). Makes the conv1d shift register
    /// ancestor-exact so FREE_SLOTS branch commits are byte-oracle.
    /// NULL on targets without causal_conv1d_tree_reroot compiled in.
    pub(crate) conv1d_tree_reroot_k: KernelHandle,
    // State allocation sizes (pre-computed from config)
    h_state_bytes: usize,
    conv_state_bytes: usize,
    // Pre-dequanted FP8 weights for zero-overhead prefill GEMMs
    qkvz_fp8: Option<DevicePtr>,
    out_proj_fp8: Option<DevicePtr>,
    fp8_gemm_k: KernelHandle,
    fp8_gemm_t_m128_k: KernelHandle, // M128: halves B re-reads for out_proj at ISL > 128
}

// SAFETY: `multi_seq_ptr_host` contains an `UnsafeCell<PinnedHostBuffer>` which
// makes the struct `!Sync` by default. The `TransformerLayer` trait
// requires `Send + Sync`. The architectural invariant that makes this
// safe is upheld in `decode_a2.rs`: layer dispatch is serialised within
// a verify step (the layer iteration is a single for-loop on one stream,
// not parallel), so concurrent mutation of the host ptr buffer is
// impossible. The Rust type system can't see this — manual impl with
// SAFETY note is correct.
unsafe impl Sync for Qwen3SsmLayer {}

// ── Sub-files (split for ≤500 LoC) ────────────────────────────────────────
mod debug;
mod exact_projection;
#[cfg(test)]
mod exact_projection_tests;
mod init;
mod qwen4_k5_ssm;
mod serial_diag;
mod ssm_forward;
mod trait_decode;
mod trait_decode_batched;
mod trait_decode_batched_conv_gdn;
mod trait_decode_multi_seq;
mod trait_prefill;
mod trait_prefill_gdn;
mod trait_prefill_helper;
mod trait_prefill_phase1;
mod trait_prefill_phase3;
mod trait_prefill_proj;
mod trait_prefill_recur;

// ── Phase 2 profiling helpers (ATLAS_SSM_KERNEL_PROFILE=1) ────────────────
// Accumulate per-call wall-clock ns spent in wy17 across all SSM layers.
// Reporter logs total + call count every N=5000 calls (every ~100 verify
// steps at 48 SSM layers per step). Zero overhead when disabled.
static SSM_PROFILE_NS_TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SSM_PROFILE_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn ssm_profile_record(ns: u64) {
    SSM_PROFILE_NS_TOTAL.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
    let n = SSM_PROFILE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    // Report every 480 calls (= 10 verify steps × 48 SSM layers).
    if n.is_multiple_of(480) {
        let total_ns = SSM_PROFILE_NS_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
        let mean_us = (total_ns as f64) / (n as f64) / 1_000.0;
        tracing::info!(
            "ATLAS_SSM_KERNEL_PROFILE: wy17 calls={n}, total={:.3}ms, mean={:.2}us/call",
            (total_ns as f64) / 1_000_000.0,
            mean_us
        );
    }
}

// ── TransformerLayer impl (delegates to per-file inherent _inner methods) ──
impl TransformerLayer for Qwen3SsmLayer {
    fn set_qwen4_hyperconnections(
        &mut self,
        attn: crate::layers::Qwen4HyperConnection,
        mlp: crate::layers::Qwen4HyperConnection,
    ) -> Result<()> {
        Qwen3SsmLayer::set_qwen4_hyperconnections(self, attn, mlp);
        Ok(())
    }

    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_inner(
            hidden,
            residual,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    fn decode_qwen4_batched(
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
        h_intermediate: DevicePtr,
        conv_intermediate: DevicePtr,
        h_intermediate_stride: usize,
        conv_intermediate_stride: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_qwen4_batched_inner(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            h_intermediate,
            conv_intermediate,
            h_intermediate_stride,
            conv_intermediate_stride,
            ctx,
            stream,
        )
    }

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
        self.decode_batched_inner(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    fn run_deferred_ffn(
        &self,
        ffn_input: DevicePtr,
        hidden: DevicePtr,
        total_rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.run_deferred_ffn_inner(ffn_input, hidden, total_rows, ctx, stream)
    }

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
        self.decode_multi_seq_inner(
            hidden,
            residual,
            num_seqs,
            states,
            kv_cache,
            seq_lens,
            block_tables,
            ctx,
            stream,
        )
    }

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
        kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_inner(
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

    fn is_ssm_layer(&self) -> bool {
        self.is_ssm_layer_inner()
    }

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
        self.prefill_phase1_inner(
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
            gdn_bufs,
            token_offset,
            ctx,
            stream,
        )
    }

    fn prefill_gdn_full(
        &self,
        state: &mut dyn LayerState,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_gdn_full_inner(state, gdn_bufs, ctx, stream)
    }

    fn prefill_gdn_full_batched(
        &self,
        h_state_ptrs: DevicePtr,
        gdn_bufs: &GdnPrefillBuffers,
        batch_size: u32,
        chunk_len: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_gdn_full_batched_inner(
            h_state_ptrs,
            gdn_bufs,
            batch_size,
            chunk_len,
            ctx,
            stream,
        )
    }

    fn prefill_phase3(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_phase3_inner(
            hidden,
            residual,
            num_tokens,
            gdn_bufs,
            token_offset,
            ctx,
            stream,
        )
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        self.alloc_state_inner(gpu)
    }

    fn multiseq_graph_safe(&self, num_seqs: usize) -> bool {
        self.multi_seq_kernel_path_active(num_seqs)
    }

    fn multiseq_refresh_ptr_table<'a, 'b: 'a>(
        &self,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        num_seqs: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.refresh_multi_seq_ptr_table(states, num_seqs, gpu, stream)
    }

    fn wy17_replay_kernel(&self) -> KernelHandle {
        self.gdn_wy17_replay_k
    }

    fn gdn_seq_lazy_engaged(&self, num_tokens: usize) -> bool {
        self.gdn_seq_lazy_engaged_inner(num_tokens)
    }

    fn gdn_seq_replay_kernel(&self) -> KernelHandle {
        self.gdn_f32_sequence_nosnap_k
    }

    fn gdn_seq_lazy_retain(&self) -> Option<DevicePtr> {
        self.gdn_lazy_retain.get().copied()
    }

    fn gdn_tree_kernel_loaded(&self) -> bool {
        self.gdn_tree_k.0 != 0
    }

    fn ddtree_conv_state_exact(&self) -> bool {
        self.conv1d_tree_reroot_k.0 != 0
    }

    fn wy17_lazy_engaged(&self, num_tokens: usize) -> bool {
        // MUST mirror the `use_lazy` computation in
        // `trait_decode_batched_conv_gdn.rs` (wy17 branch) exactly: the lazy
        // wy17 kernel skips non-checkpoint intermediate H writes, and the
        // commit may replay a skipped slot ONLY when that kernel actually
        // produced this verify's intermediates. Everything here is a pure
        // function of `num_tokens` + process-constant env gates + kernel
        // handles, so it is safe under CUDA-graph replay (unlike per-step
        // mutable state, which would go stale on replayed steps).
        if num_tokens != 17 || self.gdn_wy17_k.0 == 0 {
            // The lazy kernel is only dispatched from the `num_tokens == 17`
            // branch; every other K runs chunked/fused kernels that persist
            // ALL intermediate slots.
            return false;
        }
        let use_vsplit = crate::layers::wy17_split() > 0 && self.gdn_wy17_vsplit_k.0 != 0;
        let lazy_base = crate::layers::wy17_lazy() > 1
            && self.gdn_wy17_replay_k.0 != 0
            && crate::layers::wy17_lazy_commit();
        // lazy is engaged when the plain lazy kernel runs (no vsplit) OR when
        // the combined lazy_vsplit kernel covers both benefits (vsplit active).
        lazy_base
            && ((self.gdn_wy17_lazy_k.0 != 0 && !use_vsplit)
                || (use_vsplit && self.gdn_wy17_lazy_vsplit_k.0 != 0))
    }
}

/// `ATLAS_SSM_H_FP16=1` — store the GDN decode/verify h-state as FP16.
///
/// The decode scan is pure state traffic: at k=13 it moves 48 layers x 13 tokens
/// x 2 x 3.15 MB = 3.93 GB per verify, which is the entire measured 16.9 ms of
/// `ssm_gdn_fp32_seq`. Halving the footprint halves that time. Storage-only
/// narrowing — every float expression, accumulation order and gate clamp in the
/// f16 twin kernels is copied from the FP32 parent; only the h round-trip
/// rounding differs.
///
/// OFF by default: it changes generated tokens, so it needs a quality gate.
/// `ATLAS_SSM_GDN_LAZY=1` — lazy-commit GDN verify (skip per-token H
/// snapshots; commit reconstructs via FP32 replay). Bit-exact by construction;
/// dispatch + commit must agree, so both call [`Qwen3SsmLayer::gdn_seq_lazy_engaged`].
/// Retention rows for the lazy ExactSequence commit. γ+1 = 16 at the serving
/// γ=15; MTP verifies are smaller. `gdn_seq_lazy_engaged` refuses larger k.
pub const GDN_LAZY_MAX_K: usize = 16;

impl Qwen3SsmLayer {
    /// Pure mirror of the lazy ExactSequence dispatch decision — MUST equal
    /// what `trait_decode_batched.rs` does for the same `num_tokens`, because
    /// the async commit uses this to decide replay-vs-copy. Everything here is
    /// process-constant env + kernel handles + num_tokens (graph-safe).
    pub(super) fn gdn_seq_lazy_engaged_inner(&self, num_tokens: usize) -> bool {
        gdn_seq_lazy_enabled()
            && std::env::var("ATLAS_SSM_GDN_NOSNAP").ok().as_deref() != Some("1")
            && self.gdn_f32_sequence_lazyfinal_k.0 != 0
            && self.gdn_f32_sequence_nosnap_k.0 != 0
            && num_tokens <= GDN_LAZY_MAX_K
            && exact_flat_route::exact_flat_ssm_route(
                num_tokens,
                false, // tree handled separately: dispatch via ctx, commit via was_tree_mode
                self.ba_gates_batchn_exact_k.0 != 0,
                self.conv1d_l2norm_f32_sequence_k.0 != 0,
                self.gdn_f32_sequence_k.0 != 0,
                self.gated_rms_norm_f32_multi_seq_k.0 != 0,
            ) == exact_flat_route::ExactFlatSsmRoute::ExactSequence
    }
}

pub fn gdn_seq_lazy_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_SSM_GDN_LAZY").ok().as_deref() == Some("1"))
}

pub fn ssm_h_fp16_enabled() -> bool {
    std::env::var("ATLAS_SSM_H_FP16").ok().as_deref() == Some("1")
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::config::ModelConfig;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn test_ssm_state_allocation_sizes() {
        let config = ModelConfig::qwen3_next_80b_nvfp4();
        let nv = config.linear_num_value_heads; // 32
        let vd = config.linear_value_head_dim; // 128
        let nk = config.linear_num_key_heads; // 16
        let kd = config.linear_key_head_dim; // 128
        let d_conv = config.linear_conv_kernel_dim; // 4

        let h_bytes = nv * vd * kd * 4;
        assert_eq!(h_bytes, 32 * 128 * 128 * 4); // 2 MB

        // conv_dim = 2*key_dim + value_dim = 2*2048 + 4096 = 8192
        let conv_dim = nk * kd * 2 + nv * vd;
        let conv_bytes = conv_dim * d_conv * 4;
        assert_eq!(conv_bytes, 8192 * 4 * 4); // 128 KB

        // Verify allocations
        let gpu = MockGpuBackend::new();
        let h_state = gpu.alloc(h_bytes).unwrap();
        let conv_state = gpu.alloc(conv_bytes).unwrap();
        assert!(!h_state.is_null());
        assert!(!conv_state.is_null());
    }
}
