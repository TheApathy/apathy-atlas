// SPDX-License-Identifier: AGPL-3.0-only

//! MoE (Mixture of Experts) FFN component.
//!
//! Batched expert dispatch: top-K experts run in 2 fused kernel launches
//! (gate+up, silu+down) instead of 10 × 5 individual launches. Expert indices
//! and weights stay on device — zero D2H synchronization.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8ExpertWeight, MoeWeights, QuantizedWeight};

/// Widest speculative verify the dedup'd multi-row `_t` MoE covers in one
/// launch — the DDTree tree verify (6 spine rows + 2 branch).
///
/// This is the MROW of the widest compiled `_m` entry point AND the row count
/// `spark_runtime::buffers::MOE_DECODE_MAX_ROWS` sizes the split-K partial
/// buffer for; the two must move together or the dispatch's `space_ok` check
/// silently drops every verify back to the per-row loop.
pub(crate) const MOE_VERIFY_MAX_ROWS: u32 = spark_runtime::buffers::MOE_DECODE_MAX_ROWS as u32;

/// Widest verify the MROW=6 entry points cover. The m8 pair is strictly wider,
/// not better: an MROW=8 gather carries two more accumulator registers per
/// thread through the same loop, so the selectors below stay on m6 whenever the
/// row count fits it. That keeps the DSpark 6-row block verify — still the
/// default path — launch-for-launch identical to before m8 existed.
const MOE_VERIFY_M6_ROWS: u32 = 6;

#[derive(Clone, Copy)]
pub(crate) struct SplitkMPartitionHandles {
    pub(crate) gate_unique: KernelHandle,
    pub(crate) gate_duplicated: KernelHandle,
    pub(crate) down_unique: KernelHandle,
    /// `(kernel, MROW)` per multiplicity bucket. The buckets must TILE
    /// `2..=num_tokens` with no gap — the top one is open-ended (counts ≥ 5),
    /// so widening the verify means widening THAT arm's MROW, not appending a
    /// bucket.
    pub(crate) down_buckets: [(KernelHandle, u32); 3],
}

/// Device-side pointer table for one projection across all experts.
///
/// Enables GPU-side expert dispatch: the batched GEMV kernel reads
/// expert_id from device memory, then indexes these tables to find
/// the correct weight pointers — no CPU involvement needed.
pub(crate) struct ExpertPtrTable {
    /// `[num_experts]` u64 device pointers to each expert's B_packed.
    pub(crate) packed_ptrs: DevicePtr,
    /// `[num_experts]` u64 device pointers to each expert's B_scale.
    pub(crate) scale_ptrs: DevicePtr,
    /// `[num_experts]` f32 per-expert scale2 values.
    pub(crate) scale2_vals: DevicePtr,
}

/// Device-side pointer table for FP8 expert dispatch (one projection).
///
/// FP8 experts use 2 pointer arrays (weight + block_scale) instead of
/// NVFP4's 3 (packed + scale + scale2). The fused FP8 MoE kernel indexes
/// these tables by expert_id to load the correct FP8 weight matrix.
pub(crate) struct Fp8ExpertPtrTable {
    /// `[num_experts]` u64 device pointers to each expert's FP8 weight.
    pub(crate) weight_ptrs: DevicePtr,
    /// `[num_experts]` u64 device pointers to each expert's block scales.
    pub(crate) scale_ptrs: DevicePtr,
}

/// Checkpoint-native BF16 weights for a shared expert.
///
/// This is intentionally independent of routed-expert precision. Models such
/// as Laguna ship NVFP4 routed experts but explicitly exempt the shared expert
/// from quantization, so coupling these pointers to the all-BF16 routed path
/// silently changes model numerics.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Bf16SharedExpert {
    gate_proj: DenseWeight,
    up_proj: DenseWeight,
    down_proj: DenseWeight,
}

/// FP8-E4M3 row-scaled MIRROR of a BF16 shared expert
/// (ATLAS_TARGET_SHARED_FP8=1).
///
/// Same machinery as the attention mirrors (`build_attn_fp8_mirrors`): the
/// BF16 weights stay resident and authoritative (prefill and every multi-token
/// path keep reading them), while the M=1 decode GEMVs read these half-width
/// copies instead. On Laguna S-2.1 the BF16 shared expert costs 18.9 MB/layer
/// x 47 layers = 887 MB/token (~3.62 ms at the 245 GB/s wall); the mirror
/// halves that.
///
/// NOT bit-exact — this is a quantization, so it is quality-gated rather than
/// parity-gated.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Fp8SharedExpertMirror {
    pub gate_proj: crate::weight_map::Fp8DenseWeight,
    pub up_proj: crate::weight_map::Fp8DenseWeight,
    pub down_proj: crate::weight_map::Fp8DenseWeight,
}

impl Bf16SharedExpert {
    fn new(gate_proj: DenseWeight, up_proj: DenseWeight, down_proj: DenseWeight) -> Result<Self> {
        anyhow::ensure!(
            !gate_proj.weight.is_null() && !up_proj.weight.is_null() && !down_proj.weight.is_null(),
            "BF16 shared expert requires non-null gate/up/down weights"
        );
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

/// Unified expert pointer table for any quantization format.
///
/// Replaces the separate `ExpertPtrTable` (NVFP4) and `Fp8ExpertPtrTable` (FP8)
/// with a single enum. The MoE forward path matches on this to select the
/// correct fused kernel (moe_shared_expert_fused vs moe_shared_expert_fused_fp8).
#[allow(dead_code)]
pub(crate) enum ExpertPtrSet {
    /// NVFP4: 3 pointer arrays (packed_ptrs, scale_ptrs, per-expert scale2 f32).
    Nvfp4 {
        packed_ptrs: DevicePtr,
        scale_ptrs: DevicePtr,
        scale2_vals: DevicePtr,
    },
    /// FP8: 2 pointer arrays (weight_ptrs, block_scale_ptrs).
    Fp8 {
        weight_ptrs: DevicePtr,
        scale_ptrs: DevicePtr,
    },
}

/// MoE feed-forward network component.
///
/// Not a `TransformerLayer` — used as a component inside layers
/// for the FFN/MoE block after post-attention norm.
#[allow(dead_code)]
pub struct MoeLayer {
    pub weights: MoeWeights,
    /// Quant format of the ROUTED experts as landed in GPU memory. `Nvfp4`
    /// (default) = packed E2M1 + FP8-E4M3 per-16 block scales + f32 per-tensor
    /// global. Set to `Mxfp4E8m0` by the DeepSeek-V4 native-MXFP4 loader
    /// (transcode-free: E8M0 per-32 scales, no global) so the Phase-K E8M0
    /// GEMM variants dispatch on it instead of the NVFP4 kernels. Consumed at
    /// the grouped/decode GEMM call sites (assert via `WeightQuantFormat::expect`).
    // Written by the loader (Phase L); READ at the GEMM dispatch sites in Phase K.
    // Until Phase K wires the read, `deny(warnings)` would flag it never-read.
    #[allow(dead_code)]
    pub(crate) experts_scale_kind: crate::weight_map::WeightQuantFormat,
    /// Quant format of the SHARED expert (ARM-2 Phase-K RIDER A1). The native
    /// V4 ckpt is heterogeneous: routed experts `Mxfp4E8m0` but the shared
    /// expert is FP8→`Nvfp4`. Keyed off the weight tag (not `is_shared`
    /// positionality) so the dual-format decode kernel's `expect` net fires if
    /// a future ckpt ships a different shared format. Default `Nvfp4`.
    #[allow(dead_code)]
    pub(crate) shared_experts_scale_kind: crate::weight_map::WeightQuantFormat,
    // NVFP4-quantized gate weight (quarters bandwidth for routing)
    gate_nvfp4: Option<QuantizedWeight>,
    /// Pre-expert norm: applied to input AFTER routing but BEFORE expert dispatch.
    /// Gemma-4 26B: router sees raw residual, experts see pre_feedforward_layernorm_2(residual).
    pub pre_expert_norm: Option<crate::weight_map::DenseWeight>,
    pre_expert_norm_k: spark_runtime::gpu::KernelHandle,
    dense_gemv: KernelHandle,
    w4a16_gemv: KernelHandle,
    w4a16_gemm: KernelHandle,
    dense_gemm: KernelHandle,
    dense_gemm_pipelined: KernelHandle,
    /// Batched-M BF16 GEMV for the router gate at small `n`
    /// (ATLAS_MOE_GATE_GEMV=1). Zero when the kernel is absent from this
    /// target's module set — dispatch then stays on `dense_gemm`.
    dense_gemv_batchm: KernelHandle,
    /// FP32-output router GEMM + FP32-input top-K for the ATLAS_FP32_GATE path.
    /// Zero (unresolved) when the kernels are absent; dispatch falls back to BF16.
    dense_gemm_f32out: KernelHandle,
    /// FP32-in/FP32-out router GEMM for ATLAS_FP32_ROUTING (reads the FP32
    /// router_in from residual_add_rms_norm_gatef32). Zero if absent.
    dense_gemm_f32in: KernelHandle,
    moe_topk_f32: KernelHandle,
    moe_expert_gate_up_shared: KernelHandle,
    moe_expert_silu_down_shared: KernelHandle,
    moe_topk: KernelHandle,
    moe_weighted_sum_blend: KernelHandle,
    residual_add: KernelHandle,
    moe_topk_batched: KernelHandle,
    // K=2 fused MoE kernel handles
    moe_expert_gate_up_shared_batch2: KernelHandle,
    moe_expert_silu_down_shared_batch2: KernelHandle,
    // forward_k16 wide-verify batched MoE (num_tokens generalization, non-t).
    moe_expert_gate_up_shared_batchn_k: KernelHandle,
    moe_expert_silu_down_shared_batchn_k: KernelHandle,
    // batchN v2: expert-dedup gate_up (ATLAS_KN_V2=1; 0 when the target's
    // kernel set lacks it — dispatch falls back to v1).
    moe_expert_gate_up_shared_batchn_v2_k: KernelHandle,
    // v4 decoupled-silu dedup down (ATLAS_KN_V4=1; 0 when absent).
    moe_silu_precompute_batchn_k: KernelHandle,
    moe_expert_down_dedup_batchn_k: KernelHandle,
    // v5 cp.async bulk-staged gate_up + dedup down (ATLAS_KN_V5=1; bit-
    // identical to v2/v4 outputs, Laguna-shape guarded in forward_kn).
    moe_expert_gate_up_shared_batchn_v5_k: KernelHandle,
    moe_expert_down_dedup_batchn_v5_k: KernelHandle,
    // M=1 serial-decode v5 (same ATLAS_KN_V5=1 gate; bit-identical to the
    // serial moe_expert_gate_up_shared / moe_expert_silu_down_shared pair,
    // cp.async whole-slice staging; Laguna-shape guarded in forward()).
    moe_expert_gate_up_shared_v5_k: KernelHandle,
    moe_expert_silu_down_shared_v5_k: KernelHandle,
    // ── W3 Lloyd-Max (3-bit) routed-expert kernels (ATLAS_MOE_W3=1). ──
    // try_kernel: KernelHandle(0) on images that don't compile the
    // moe_fused_w3 / moe_w3a16 modules; `enable_w3` refuses to arm W3
    // unless the FULL set resolved (the graceful stay-NVFP4 gate).
    moe_expert_gate_up_shared_w3_k: KernelHandle,
    moe_expert_silu_down_shared_w3_k: KernelHandle,
    moe_expert_gate_up_shared_batchn_w3_k: KernelHandle,
    moe_expert_silu_down_shared_batchn_w3_k: KernelHandle,
    moe_expert_gate_up_shared_batchn_v2_w3_k: KernelHandle,
    moe_expert_down_dedup_batchn_w3_k: KernelHandle,
    moe_grouped_gemm_w3_k: KernelHandle,
    /// Device `[8]` f32 Lloyd-Max codebook. NULL unless W3 is armed.
    w3_lut_dev: DevicePtr,
    moe_weighted_sum_blend_batch2: KernelHandle,
    /// Fused blend + residual add (ATLAS_FUSED_ELEMWISE=1, forward_kn tail).
    /// 0 when the fused_verify_elemwise module is absent.
    moe_blend_residual_batchn_k: KernelHandle,
    w4a16_gemv_batch2: KernelHandle,
    // K=3 fused MoE kernel handles
    moe_expert_gate_up_shared_batch3: KernelHandle,
    moe_expert_silu_down_shared_batch3: KernelHandle,
    moe_weighted_sum_blend_batch3: KernelHandle,
    w4a16_gemv_batch3: KernelHandle,
    // Generic token-major NVFP4 MoE kernels. Used as an opt-in decode
    // concurrency experiment for N>=4 without grouped-GEMM sorting.
    moe_expert_gate_up_shared_token_major: KernelHandle,
    moe_expert_silu_down_shared_token_major: KernelHandle,
    moe_weighted_sum_blend_token_major: KernelHandle,
    moe_decode_atomic_c4_silu_down_accum_k: KernelHandle,
    moe_decode_atomic_c4_finalize_k: KernelHandle,
    // Sorted/grouped prefill path
    moe_sort_by_expert: KernelHandle,
    moe_sorted_gate_up: KernelHandle,
    moe_sorted_silu_down: KernelHandle,
    moe_grouped_gemm: KernelHandle,
    moe_silu_mul: KernelHandle,
    /// Activation kernel for sorted/unfused path. SiLU by default, GeGLU for Gemma-4.
    moe_act_mul: KernelHandle,
    /// When true, decode uses the sorted prefill path (avoids fused SiLU kernels).
    gelu_activation: bool,
    moe_unpermute_reduce: KernelHandle,
    moe_batched_blend: KernelHandle,
    /// Pointer tables for batched expert dispatch.
    gate_ptrs: ExpertPtrTable,
    up_ptrs: ExpertPtrTable,
    down_ptrs: ExpertPtrTable,
    /// Transposed pointer tables for coalesced prefill GEMM.
    gate_ptrs_t: Option<ExpertPtrTable>,
    up_ptrs_t: Option<ExpertPtrTable>,
    down_ptrs_t: Option<ExpertPtrTable>,
    /// CUTLASS grouped-NVFP4 swizzled SFB weight-scale tables
    /// (`ATLAS_HOLO_MOE_GROUPED_CUTLASS`). Device `[num_experts]` u64 arrays of
    /// per-expert SFB pointers, built at load by `build_cutlass_grouped_sfb` from
    /// the `gate_ptrs_t`/`up_ptrs_t` `[K/16,N]` scales (`pack_weight_sfb` swizzle).
    /// The grouped kernel reads `gate_ptrs.packed` (`[N,K/2]`) + these SFB + the
    /// real per-expert `scale2`. `None` => the CUTLASS grouped path is unavailable.
    gate_sfb_cutlass: Option<DevicePtr>,
    up_sfb_cutlass: Option<DevicePtr>,
    down_sfb_cutlass: Option<DevicePtr>,
    /// Keeps the per-expert SFB buffers + the two pointer arrays alive.
    _cutlass_sfb_owned: Vec<DevicePtr>,
    /// Lazy down_proj transpose scratch — populated at the start of each
    /// prefill call when the persistent transpose pass couldn't fit
    /// down_proj. Decode keeps using `down_ptrs` (untransposed); prefill
    /// uses `down_ptrs_t` pointing into this scratch. Shared across all
    /// MoE layers (the same scratch is overwritten layer-by-layer during
    /// the sequential forward).
    ///
    /// `down_t_scratch_packed`: contiguous `[num_experts × N × K/2]` bytes.
    /// `down_t_scratch_scale`:  contiguous `[num_experts × N × K/16]` bytes.
    /// Both `None` when the persistent transpose pass already covered
    /// down (full-fits path) or when the layer doesn't need scratch
    /// transpose (FP8 experts, etc.).
    down_t_scratch_packed: Option<DevicePtr>,
    down_t_scratch_scale: Option<DevicePtr>,
    /// Kernel handle for the batched per-expert uint8 transpose.
    moe_transpose_u8_batched_k: KernelHandle,
    // ── Phase 8a transposed-layout decode kernels (unified-layout MoE).
    // Loaded eagerly at construction. Currently NOT wired into the
    // dispatch — Phase 8a part 3/3 will route decode through these once
    // the weight loader produces transposed-only pointer tables.
    moe_expert_gate_up_shared_t_k: KernelHandle,
    moe_expert_silu_down_shared_t_k: KernelHandle,
    // ARM-2 Phase-K: native-MXFP4 (E8M0 routed / NVFP4 shared) dual-format
    // decode variants. KernelHandle(0) on models that don't ship them.
    moe_expert_gate_up_shared_t_e8m0_k: KernelHandle,
    moe_expert_silu_down_shared_t_e8m0_k: KernelHandle,
    // Split-K decode: wide (VEC=2) loads with the warps put back by splitting
    // K four ways, plus the fixed-order finalize that sums the partials. See
    // `ops::T_SPLIT_VEC`. KernelHandle(0) on targets that don't ship them; the
    // dispatch falls back to the non-split kernels above.
    moe_expert_gate_up_shared_t_splitk_k: KernelHandle,
    moe_expert_silu_down_shared_t_splitk_k: KernelHandle,
    moe_expert_gate_up_shared_t_e8m0_splitk_k: KernelHandle,
    moe_expert_silu_down_shared_t_e8m0_splitk_k: KernelHandle,
    // ATLAS_MOE_GEMV_V2 wide-load tier: `_v4s8` — 128-byte warp requests on
    // the weight stream (VEC=4 uchar4 loads) at exactly the v2s4 CTA count
    // (SPLIT=8 puts back the blocks VEC=4 gives up), plus the merged 32-bit
    // activation read (WIDE_ACT). Same fixed-order finalize; SPLIT=8 splits
    // each dot product at different points than SPLIT=4, so this tier is
    // reassociation-equivalent (not bit-equal) to v2s4 — v4s8 IS bit-equal to
    // the v2s8 witness entry, which the microtest gates on. KernelHandle(0)
    // where not compiled; the dispatch falls back to v2s4.
    moe_expert_gate_up_shared_t_splitk8_k: KernelHandle,
    moe_expert_silu_down_shared_t_splitk8_k: KernelHandle,
    moe_expert_gate_up_shared_t_e8m0_splitk8_k: KernelHandle,
    moe_expert_silu_down_shared_t_e8m0_splitk8_k: KernelHandle,
    moe_gate_up_partial_finalize_k: KernelHandle,
    moe_down_partial_finalize_k: KernelHandle,
    // Multi-row split-K decode (`_m`): the same v2s4 body, but each block
    // dedups the slots routed to its expert and carries an accumulator per
    // gathered row, so the weight bytes are read once for up to MROW rows.
    // MROW=2 backs the MTP K=2 verify, MROW=6 the DSpark block verify (5
    // proposed rows + the committed one, and every narrower γ in between —
    // an MROW=R kernel is correct for any num_tokens <= R). MROW=1 exists so
    // the microtest can prove the dedup rewrite is bit-identical to the
    // shipping single-row path before the wider variants are trusted.
    // KernelHandle(0) where not compiled — the dispatch falls back to per-row
    // `_splitk_k` launches.
    moe_expert_gate_up_shared_t_m2_k: KernelHandle,
    moe_expert_silu_down_shared_t_m2_k: KernelHandle,
    moe_expert_gate_up_shared_t_e8m0_m2_k: KernelHandle,
    moe_expert_silu_down_shared_t_e8m0_m2_k: KernelHandle,
    moe_expert_gate_up_shared_t_m6_k: KernelHandle,
    moe_expert_silu_down_shared_t_m6_k: KernelHandle,
    moe_expert_gate_up_shared_t_e8m0_m6_k: KernelHandle,
    moe_expert_silu_down_shared_t_e8m0_m6_k: KernelHandle,
    // MROW=8: the DDTree tree verify (6 spine + 2 branch) and any drafter past
    // γ=5. Selected ONLY for 7..8 rows — see `MOE_VERIFY_M6_ROWS`.
    moe_expert_gate_up_shared_t_m8_k: KernelHandle,
    moe_expert_silu_down_shared_t_m8_k: KernelHandle,
    moe_expert_gate_up_shared_t_e8m0_m8_k: KernelHandle,
    moe_expert_silu_down_shared_t_e8m0_m8_k: KernelHandle,
    // V2 wide-load tier (`ATLAS_MOE_SPLITK_V2=1`): the same MROW ladder at
    // VEC=4 / SPLIT=4 (`_m{2,6,8}v4s4`). Weight requests widen from 64 to 128
    // bytes per warp, and the gate_up side stages all gathered rows'
    // activations in dynamic smem once per k-window instead of re-issuing M
    // narrow global reads per weight byte pair. SPLIT is unchanged, so the
    // tier is BIT-IDENTICAL to the v2s4 incumbent (same split points, same
    // per-output FMA order; VEC only remaps thread→output).
    moe_expert_gate_up_shared_t_m2_v2t_k: KernelHandle,
    moe_expert_silu_down_shared_t_m2_v2t_k: KernelHandle,
    moe_expert_gate_up_shared_t_e8m0_m2_v2t_k: KernelHandle,
    moe_expert_silu_down_shared_t_e8m0_m2_v2t_k: KernelHandle,
    moe_expert_gate_up_shared_t_m6_v2t_k: KernelHandle,
    moe_expert_silu_down_shared_t_m6_v2t_k: KernelHandle,
    moe_expert_gate_up_shared_t_e8m0_m6_v2t_k: KernelHandle,
    moe_expert_silu_down_shared_t_e8m0_m6_v2t_k: KernelHandle,
    moe_expert_gate_up_shared_t_m8_v2t_k: KernelHandle,
    moe_expert_silu_down_shared_t_m8_v2t_k: KernelHandle,
    moe_expert_gate_up_shared_t_e8m0_m8_v2t_k: KernelHandle,
    moe_expert_silu_down_shared_t_e8m0_m8_v2t_k: KernelHandle,
    // Wide-verify partitions. Gate/up uses lean MROW=1 for unique groups and
    // MROW=6 (MROW=8 past six rows) for duplicated groups. Down additionally
    // buckets multiplicity as 2, 3-4, and 5-or-more so each launch reserves only
    // the dynamic shared memory its accumulator width needs. The top bucket is
    // open-ended, so widening the verify widens ITS MROW (m6c56 → m8c58) rather
    // than adding a fourth bucket: a gap in the tiling would leave the groups
    // whose count lands in it with no down arm at all.
    moe_expert_gate_up_shared_t_m1u_k: KernelHandle,
    moe_expert_silu_down_shared_t_m1u_k: KernelHandle,
    moe_expert_gate_up_shared_t_e8m0_m1u_k: KernelHandle,
    moe_expert_silu_down_shared_t_e8m0_m1u_k: KernelHandle,
    moe_expert_gate_up_shared_t_m6d_k: KernelHandle,
    moe_expert_gate_up_shared_t_e8m0_m6d_k: KernelHandle,
    moe_expert_silu_down_shared_t_m2c2_k: KernelHandle,
    moe_expert_silu_down_shared_t_e8m0_m2c2_k: KernelHandle,
    moe_expert_silu_down_shared_t_m4c34_k: KernelHandle,
    moe_expert_silu_down_shared_t_e8m0_m4c34_k: KernelHandle,
    moe_expert_silu_down_shared_t_m6c56_k: KernelHandle,
    moe_expert_silu_down_shared_t_e8m0_m6c56_k: KernelHandle,
    moe_expert_gate_up_shared_t_m8d_k: KernelHandle,
    moe_expert_gate_up_shared_t_e8m0_m8d_k: KernelHandle,
    moe_expert_silu_down_shared_t_m8c58_k: KernelHandle,
    moe_expert_silu_down_shared_t_e8m0_m8c58_k: KernelHandle,
    moe_gate_up_partial_finalize_m_k: KernelHandle,
    moe_gate_up_partial_finalize_m_act_k: KernelHandle,
    moe_down_partial_finalize_m_k: KernelHandle,
    // ── sqrtsoftplus routing (DeepSeek-V4) ──
    moe_topk_sqrtsoftplus_k: KernelHandle,
    moe_topk_sqrtsoftplus_batched_k: KernelHandle,
    /// `ATLAS_MOE_ADAPTIVE_TOPK=<threshold>` — QUALITY-AFFECTING, default OFF.
    /// Post-router in-place prune of negligible-gate-weight routed slots. Zero
    /// when the target's kernel set lacks `moe_adaptive_topk` (the feature then
    /// refuses to arm and says so once). See docs/ADAPTIVE-TOPK.md.
    moe_adaptive_topk_prune_k: KernelHandle,
    // ── hash routing (DeepSeek-V4 first `num_hash_layers` MoE layers) ──
    moe_hash_route_k: KernelHandle,
    moe_hash_route_batched_k: KernelHandle,
    /// Static `tid2eid` table [vocab_size, top_k] i64 — present ONLY for the
    /// hash-routed layers (the loader supplies it only for those). `Some`
    /// here is the SSOT that this layer routes via the static hash table
    /// instead of the learned gate's top-K.
    tid2eid_dev: Option<DevicePtr>,
    moe_expert_gate_up_shared_batch2_t_k: KernelHandle,
    moe_expert_silu_down_shared_batch2_t_k: KernelHandle,
    // Native-MXFP4 (E8M0 per-32 routed scales, NVFP4 shared) flavor of the
    // batched verify kernels. Without these the m-row speculative verify has
    // to fall back to the single-token kernel once per row, re-reading every
    // expert / the shared expert / the gate for each row — worth 1.28x
    // (learned-gate layers) to 2.01x (hash-routed layers) of the routed-expert
    // weight traffic on DeepSeek-V4-Flash, measured with ATLAS_MOE_OVERLAP=1.
    moe_expert_gate_up_shared_batch2_t_e8m0_k: KernelHandle,
    moe_expert_silu_down_shared_batch2_t_e8m0_k: KernelHandle,
    // forward_k16 wide-verify batched MoE (num_tokens generalization of batch2_t).
    moe_expert_gate_up_shared_batchn_t_k: KernelHandle,
    moe_expert_silu_down_shared_batchn_t_k: KernelHandle,
    moe_expert_gate_up_shared_batchn_t_e8m0_k: KernelHandle,
    moe_expert_silu_down_shared_batchn_t_e8m0_k: KernelHandle,
    moe_expert_gate_up_shared_batch3_t_k: KernelHandle,
    moe_expert_silu_down_shared_batch3_t_k: KernelHandle,
    moe_expert_gate_up_shared_fp8_t_k: KernelHandle,
    moe_expert_silu_down_shared_fp8_t_k: KernelHandle,
    moe_expert_gate_up_shared_fp8_batch2_t_k: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch2_t_k: KernelHandle,
    moe_expert_gate_up_shared_fp8_batch3_t_k: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch3_t_k: KernelHandle,
    /// `ATLAS_UNIFIED_MOE_LAYOUT=1` opts in to the unified-layout decode
    /// path: gate/up/down all use transposed `[K/2, N]` layout, decode
    /// dispatches to `moe_expert_*_shared_t` kernels. Default off — the
    /// dispatch falls through to the original `[N, K/2]` kernels.
    /// Resolved once at construction.
    unified_layout: bool,
    /// `ATLAS_NVFP4_GATE_UP_M128=1` opts in to the M=128 fused gate+up
    /// kernel (Block D #3, Avarok tile-shape rewrite). Halves block count
    /// at large prefill — better SM amortization on GB10's 25-SM budget.
    /// Currently only minimax-m2-229b ships the kernel; other models keep
    /// `moe_fused_gate_up_t_k64_m128 == KernelHandle(0)` and dispatch
    /// falls through to the M=64 path even when the env var is set.
    nvfp4_gate_up_m128: bool,
    /// `ATLAS_HOLO_MOE_GATEUP_FP4=1` opts the prefill fused gate_up onto the
    /// block-scaled FP4 kernel. Reads the SHARED FAST_MOE=full `gate_ptrs_t`/
    /// `up_ptrs_t` `[K/2,N]` tables (no extra MoE memory); dispatch also requires
    /// those tables present + the FP4 kernel handle != 0.
    gateup_fp4: bool,
    /// `ATLAS_HOLO_MOE_DOWN_FP4=1` — same, for the prefill down projection over
    /// the shared `down_ptrs_t` table.
    down_fp4: bool,
    /// `ATLAS_HYBRID_MOE_LAYOUT=1` opts in to the hybrid-layout path:
    /// keep BOTH original `[N, K/2]` weights (for decode + MTP verify) AND
    /// transposed `[K/2, N]` weights (for prefill). Doubles MoE-weight
    /// memory but recovers the ~15 % decode regression that pure unified
    /// layout suffers from. Resolved once at construction; mutually
    /// exclusive with `unified_layout` at the dispatch level (hybrid wins
    /// on decode paths since it preserves untransposed warp-reduction
    /// parallelism).
    hybrid_layout: bool,
    /// Transposed shared expert weights for prefill.
    shared_gate_t: Option<QuantizedWeight>,
    shared_up_t: Option<QuantizedWeight>,
    shared_down_t: Option<QuantizedWeight>,
    moe_grouped_gemm_t: KernelHandle,
    moe_grouped_gemm_t_k64: KernelHandle,
    moe_fused_gate_up_t: KernelHandle,
    moe_fused_gate_up_t_k64: KernelHandle,
    // ARM-2 Phase-K: native-MXFP4 (E8M0 per-32) prefill variants of the W4A16
    // routed-expert GEMMs. KernelHandle(0) on models that don't ship them
    // (only the deepseek-v4-flash target compiles the `_e8m0` entries).
    moe_grouped_gemm_e8m0: KernelHandle,
    moe_grouped_gemm_t_e8m0: KernelHandle,
    moe_grouped_gemm_t_k64_e8m0: KernelHandle,
    moe_fused_gate_up_t_e8m0: KernelHandle,
    moe_fused_gate_up_t_k64_e8m0: KernelHandle,
    /// M=128 variant of the K64 fused gate+up kernel (Block D #3, Avarok
    /// tile-shape rewrite). Loaded with `try_kernel` — falls back to
    /// `KernelHandle(0)` on models that don't ship the kernel; dispatch
    /// gates on `nvfp4_gate_up_m128` AND handle non-zero.
    moe_fused_gate_up_t_k64_m128: KernelHandle,
    /// FUSED FP4 (block-scaled e2m1) variant of the K64 fused gate+up kernel
    /// (`ATLAS_HOLO_MOE_GATEUP_FP4`). Same signature as `moe_fused_gate_up_t_k64`
    /// but runs one `mma.sync.kind::mxf4nvf4.scale_vec::4X.m16n8k64` per k64
    /// tile (vs 2× m16n8k32 e4m3). `try_kernel` — `KernelHandle(0)` on images
    /// lacking it; the dispatch in `forward_prefill_routed` only fires when this
    /// handle != 0, `gateup_fp4` is set, and the shared `gate_ptrs_t`/`up_ptrs_t`
    /// tables are present (FAST_MOE=full).
    moe_fused_gate_up_t_k64_fp4: KernelHandle,
    /// SMALL-M FP4 decode GEMV pair (`ATLAS_MOE_FP4_DECODE_SMALLM=1`): slot-major
    /// output-tiled GEMV kernels over the same shared `[K/2,N]` `_t` tables +
    /// per-expert `scale2`, replacing the M_TILE=64 K64 MMA kernels when
    /// `total_expanded <= fp4_decode_smallm_max` (decode: 1-2 rows/expert makes
    /// the 64-row tile 16-64x padding). `try_kernel` — 0 on images lacking them;
    /// dispatch requires handle != 0 + NVFP4 routed experts + the `_t` tables.
    moe_fused_gate_up_fp4_smallm: KernelHandle,
    moe_down_fp4_smallm: KernelHandle,
    /// Small-M threshold for the FP4 decode GEMV arm. 0 = arm disabled (the
    /// default). Set from `ATLAS_MOE_FP4_DECODE_SMALLM=1` (+ optional
    /// `ATLAS_MOE_FP4_DECODE_SMALLM_MAX`, default 96 — covers padded decode
    /// n<=8 at top_k=10 = 80 slots). Resolved once at construction so the
    /// captured-graph dispatch branch is stable per batch size.
    fp4_decode_smallm_max: u32,
    moe_fp8_grouped_gemm_t: KernelHandle,
    w4a16_gemm_t: KernelHandle,
    /// Faster shadows of `w4a16_gemm_t` for the shared-expert prefill GEMMs,
    /// selected by `ATLAS_MOE_SHARED_K64` (see `shared_prefill_arm`). All three
    /// consume the SAME NVFP4 tables as `w4a16_gemm_t` (transposed
    /// `B_packed[K/2, N]` + `B_scale[K/16, N]` E4M3 bytes + per-tensor
    /// `scale2`), so they are drop-in on `shared_gate_t`/`shared_up_t`/
    /// `shared_down_t`. `try_kernel` → 0 on model dirs that don't ship them.
    w4a16_gemm_t_v2: KernelHandle,
    w4a16_gemm_t_k64: KernelHandle,
    w4a16_gemm_t_k64_v2: KernelHandle,
    w4a16_gemm_t_m128: KernelHandle,
    bf16_to_fp8_k: KernelHandle,
    /// Pre-dequanted FP8 weights for zero-overhead prefill GEMMs.
    gate_fp8: Option<DevicePtr>,
    shared_gate_fp8: Option<DevicePtr>,
    shared_up_fp8: Option<DevicePtr>,
    shared_down_fp8: Option<DevicePtr>,
    fp8_gemm_k: KernelHandle,
    /// Secondary CUDA stream for overlapping shared expert with routed experts.
    prefill_stream: u64,
    /// Event pair for stream synchronization (input_ready, shared_done).
    event_a: u64,
    event_b: u64,
    // ── Sigmoid + correction-bias routing (DeepSeek-V3 / MiniMax-M2 style) ──
    /// Device pointer to `[num_experts]` correction bias. Populated from
    /// `MoeWeights.correction_bias` in `new()` when the loader sets it.
    /// `None` = Atlas's default softmax path. When `Some`, every top-k
    /// dispatch site branches to `moe_topk_sigmoid` with this bias arg.
    correction_bias_dev: Option<DevicePtr>,
    /// Handle to `moe_topk_sigmoid` kernel. Lazy-loaded in `new()` even
    /// when bias is `None` (harmless if kernel isn't used).
    moe_topk_sigmoid_k: KernelHandle,
    /// Batched variant for prefill / MTP-verify (one block per token).
    /// Loaded via `try_kernel` — returns KernelHandle(0) on models whose
    /// KERNEL.toml doesn't register the sigmoid kernels (e.g. Mistral).
    /// Never dispatched on those paths because `correction_bias_dev` is
    /// `None` there.
    moe_topk_sigmoid_batched_k: KernelHandle,
    // FP8 fused MoE kernels (used when experts are FP8)
    moe_expert_gate_up_shared_fp8: KernelHandle,
    moe_expert_silu_down_shared_fp8: KernelHandle,
    // FP8 batch2/3 fused MoE kernels (for MTP K=2/K=3 verify)
    moe_expert_gate_up_shared_fp8_batch2: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch2: KernelHandle,
    moe_weighted_sum_blend_fp8_batch2: KernelHandle,
    moe_expert_gate_up_shared_fp8_batch3: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch3: KernelHandle,
    moe_weighted_sum_blend_fp8_batch3: KernelHandle,
    // THE routed-expert FP8 grouped GEMM for sorted MoE prefill: grid-compaction
    // (persistent 96-CTA grid over a COMPACTED (expert, m_tile, n_tile) work-list
    // built by `moe_build_tile_worklist`). Handle may be 0 on images that don't
    // ship the kernel.
    moe_fp8_grouped_gemm_k: KernelHandle,
    // Builds the grouped-GEMM work-list (moe_build_tile_worklist, module "moe").
    // Launched on the SAME stream as the grouped GEMM (read-after-write of
    // total_tiles). Handle may be 0 on older images.
    moe_build_tile_worklist_k: KernelHandle,
    // W8A8 + FP32 epilogue MoE GEMM (vLLM-equivalent). Opt-in via
    // ATLAS_FP8_W8A8=1. Requires per-token-quanted A_fp8 + a_scale.
    moe_w8a8_grouped_gemm_k: KernelHandle,
    per_token_group_quant_fp8_k: KernelHandle,
    // Dense W8A8 (same kernel used by attention QKV/O proj) for shared-expert path.
    fp8_gemm_t_blockscaled_k: KernelHandle,
    // BF16 grouped GEMM — for FP8-source models dequanted to BF16 at load.
    // Activates the high-precision MoE path that closes the per-layer
    // 0.989 FP8 cosine ceiling. Handle may be 0 on images that don't ship
    // the kernel; dispatch site is gated on Some(bf16_*_weight_ptrs).
    moe_bf16_grouped_gemm_k: KernelHandle,
    // Fused BF16 decode kernels (mirror moe_expert_*_shared_fp8 layout).
    moe_expert_gate_up_shared_bf16_k: KernelHandle,
    moe_expert_silu_down_shared_bf16_k: KernelHandle,
    // Fused BF16 K=2 batch kernels for MTP verify (mirror the FP8 batch2 layout).
    // Handle may be 0 on images that don't ship the kernel; the K=2 BF16
    // dispatch site is gated on this being non-null and falls back to the
    // per-token batched path otherwise.
    moe_expert_gate_up_shared_bf16_batch2_k: KernelHandle,
    moe_expert_silu_down_shared_bf16_batch2_k: KernelHandle,
    w8a16_gemm_k: KernelHandle,           // for shared expert FP8 prefill
    w8a16_gemm_pipelined_k: KernelHandle, // ATLAS_W8A16_PIPELINED shared-expert variant
    // Fused gate GEMV + topK softmax (saves 1 kernel launch per layer)
    moe_gate_topk_fused_k: KernelHandle,
    // FP8 expert pointer tables (None when experts are NVFP4)
    fp8_gate_weight_ptrs: Option<Fp8ExpertPtrTable>,
    fp8_up_weight_ptrs: Option<Fp8ExpertPtrTable>,
    fp8_down_weight_ptrs: Option<Fp8ExpertPtrTable>,
    // BF16 expert pointer tables — populated by the FP8-dequant-on-load
    // path. When Some, the routed-expert dispatch in `forward_prefill_fp8`
    // routes through `moe_bf16_grouped_gemm` instead of the FP8 grouped
    // GEMM, eliminating the per-layer FP8 quantization ceiling.
    bf16_gate_weight_ptrs: Option<DevicePtr>,
    bf16_up_weight_ptrs: Option<DevicePtr>,
    bf16_down_weight_ptrs: Option<DevicePtr>,
    // Checkpoint-native BF16 shared expert. Independent of routed-expert
    // precision so mixed NVFP4-routed/BF16-shared checkpoints stay faithful.
    bf16_shared_expert: Option<Bf16SharedExpert>,
    // FP8-E4M3 row-scaled mirror of `bf16_shared_expert`, built at load time
    // under ATLAS_TARGET_SHARED_FP8=1. Consumed ONLY by the M=1 decode GEMVs
    // in `run_bf16_shared_expert`; every multi-token/prefill path keeps the
    // BF16 originals. `None` => BF16 everywhere (unchanged behaviour).
    fp8_shared_expert_mirror: Option<Fp8SharedExpertMirror>,
    // Kernel handle for the FP8 row-scaled GEMV that consumes the mirror.
    dense_gemv_fp8w_k: KernelHandle,
    // FP8 shared expert weights (None when shared expert is NVFP4)
    fp8_shared_expert: Option<Fp8ExpertWeight>,
    /// FP4 down kernel handle (`moe_w4a16_down_t_k64_fp4`). `try_kernel` =>
    /// `KernelHandle(0)` on images lacking it; the FP4-down dispatch checks this
    /// handle != 0, `down_fp4` is set, and the shared `down_ptrs_t` table is present.
    pub(crate) moe_down_t_k64_fp4: KernelHandle,
    /// `moe_permute_tokens` gather kernel — only needed by the FP4 escape-hatch
    /// (which consumes expert-sorted contiguous rows, unlike the FP8 fused
    /// kernel that gathers via `sorted_token_ids` internally). `try_kernel`
    /// (handle may be 0 on images lacking it). Now unused — the CUTLASS grouped
    /// path fuses the gather into its A-pack — kept for potential reuse.
    #[allow(dead_code)]
    pub(crate) moe_permute_tokens_k: KernelHandle,
    // Phase 2.7 Tier C — Frankenstein dispatch flag.
    // True when this layer's index is in `config.dflash_capture_layers`.
    // When the env var `ATLAS_FRANKENSTEIN_DECODE_VIA_PREFILL=1` is set,
    // `forward()` (single-token decode) will route through `forward_prefill`
    // (tensor-core grouped GEMM kernel) on this layer only, so the captured
    // hidden states use a different numerical recipe than the scalar GEMV
    // path. Used to test whether the kernel choice is the dominant cause
    // of low DFlash drafter acceptance on FP4/FP8 targets.
    pub is_dflash_capture_layer: bool,
    /// EXL3 trellis (K2/K3, 2.0/3.0 bpw) routed experts — `Some` only when the loader
    /// found `…ffn.experts.{E}.{w}.rank0.trellis` tensors (the reference tp1
    /// checkpoint, `quant_method: "exl3"`). Set post-construction by
    /// `set_exl3_experts`, which also flips `experts_scale_kind` to
    /// `Exl3Trellis` so every non-EXL3 dispatch path fails loudly. Decode
    /// M=1 dispatch lives in `exl3_decode.rs`; prefill/verify M>1 are NOT
    /// wired yet (plan §3 P1).
    pub(crate) exl3: Option<exl3_decode::Exl3MoeState>,
}

impl MoeLayer {
    /// ARM-2 Phase-K routed-expert kernel-handle select. Returns the E8M0
    /// variant when the routed experts are native MXFP4 (`Mxfp4E8m0`), else the
    /// NVFP4 handle. Panics if E8M0 is selected but the `_e8m0` kernel is
    /// absent from this target (`try_kernel` gave 0) — that means a native
    /// checkpoint reached a build that never compiled the variant, which must
    /// be loud, not silent NVFP4-on-E8M0 garbage (the straggler net).
    #[inline]
    fn e8m0_or(
        &self,
        nvfp4: spark_runtime::gpu::KernelHandle,
        e8m0: spark_runtime::gpu::KernelHandle,
        site: &str,
    ) -> spark_runtime::gpu::KernelHandle {
        if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
            assert!(
                e8m0.0 != 0,
                "ARM-2 Phase-K: routed experts tagged Mxfp4E8m0 at {site}, but the \
                 _e8m0 kernel handle is unresolved (not compiled into this target)."
            );
            e8m0
        } else {
            nvfp4
        }
    }

    /// `e8m0_or` without the assert: returns `None` when the variant this
    /// model needs isn't compiled into the target image. The split-K decode
    /// path is an optional fast path, so it declines and falls through to the
    /// single-sweep kernels rather than aborting.
    #[inline]
    fn e8m0_or_opt(
        &self,
        nvfp4: spark_runtime::gpu::KernelHandle,
        e8m0: spark_runtime::gpu::KernelHandle,
    ) -> Option<spark_runtime::gpu::KernelHandle> {
        let h = if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
            e8m0
        } else {
            nvfp4
        };
        (h.0 != 0).then_some(h)
    }

    /// Kernel pair for the two-row batched `_t` decode MoE, picked to match the
    /// routed-expert scale format. Native-MXFP4 checkpoints need the `_e8m0`
    /// entries: the NVFP4 kernels read the routed scale table as `[N, K/16]`,
    /// so an E8M0 `[N, K/32]` table walks off the end and faults (CUDA 700).
    ///
    /// Returns `(0, 0)` handles when the needed variant isn't compiled into
    /// this target — the batched path is an optimization, so the caller's
    /// `!= 0` guard falls back to two single-row dispatches instead of aborting.
    #[inline]
    pub(crate) fn batch2_t_handles(
        &self,
    ) -> (
        spark_runtime::gpu::KernelHandle,
        spark_runtime::gpu::KernelHandle,
    ) {
        let gate_up = self.e8m0_or_opt(
            self.moe_expert_gate_up_shared_batch2_t_k,
            self.moe_expert_gate_up_shared_batch2_t_e8m0_k,
        );
        let silu_down = self.e8m0_or_opt(
            self.moe_expert_silu_down_shared_batch2_t_k,
            self.moe_expert_silu_down_shared_batch2_t_e8m0_k,
        );
        match (gate_up, silu_down) {
            // Both or neither: a half-resolved pair would dispatch one stage
            // batched and the other per-row against the same buffers.
            (Some(g), Some(s)) => (g, s),
            _ => (
                spark_runtime::gpu::KernelHandle(0),
                spark_runtime::gpu::KernelHandle(0),
            ),
        }
    }

    // NOTE: no `batchn_t_handles` yet — `forward_kn` dispatches the *non*-`_t`
    // batchn kernels, so the `_t` pair (and its new `_e8m0` twin) has no call
    // site to select for. Add the accessor when the wide verify moves to `_t`.

    /// Kernel pair for the dedup'd split-K `_t` decode at the narrowest
    /// compiled MROW that covers `num_tokens`, picked to match the
    /// routed-expert scale format.
    ///
    /// Same both-or-neither contract as [`Self::batch2_t_handles`], and the
    /// same reason: a half-resolved pair would run one stage multi-row and the
    /// other per-row against buffers whose partial layouts disagree.
    ///
    /// Returns the compiled MROW alongside the handles — the down kernel sizes
    /// its dynamic shared memory by it, so the caller must pass the entry
    /// point's MROW and not `num_tokens`.
    #[inline]
    pub(crate) fn splitk_m_t_handles(
        &self,
        num_tokens: u32,
    ) -> Option<(
        spark_runtime::gpu::KernelHandle,
        spark_runtime::gpu::KernelHandle,
        u32,
    )> {
        // Narrowest first: a wider entry point is correct but carries the
        // ladder's extra arms and a wider `slots`/`s_idx`, so K=2 should keep
        // landing on m2.
        let candidates: &[(u32, KernelHandle, KernelHandle, KernelHandle, KernelHandle)] = &[
            (
                2,
                self.moe_expert_gate_up_shared_t_m2_k,
                self.moe_expert_gate_up_shared_t_e8m0_m2_k,
                self.moe_expert_silu_down_shared_t_m2_k,
                self.moe_expert_silu_down_shared_t_e8m0_m2_k,
            ),
            (
                MOE_VERIFY_M6_ROWS,
                self.moe_expert_gate_up_shared_t_m6_k,
                self.moe_expert_gate_up_shared_t_e8m0_m6_k,
                self.moe_expert_silu_down_shared_t_m6_k,
                self.moe_expert_silu_down_shared_t_e8m0_m6_k,
            ),
            (
                MOE_VERIFY_MAX_ROWS,
                self.moe_expert_gate_up_shared_t_m8_k,
                self.moe_expert_gate_up_shared_t_e8m0_m8_k,
                self.moe_expert_silu_down_shared_t_m8_k,
                self.moe_expert_silu_down_shared_t_e8m0_m8_k,
            ),
        ];
        for &(mrow, gu, gu_e8m0, sd, sd_e8m0) in candidates {
            if num_tokens > mrow {
                continue;
            }
            let gate_up = self.e8m0_or_opt(gu, gu_e8m0);
            let silu_down = self.e8m0_or_opt(sd, sd_e8m0);
            if let (Some(g), Some(s)) = (gate_up, silu_down)
                && g.0 != 0
                && s.0 != 0
            {
                return Some((g, s, mrow));
            }
        }
        None
    }

    /// [`Self::splitk_m_t_handles`] over the V2 wide-load `_v4s4` entries
    /// (`ATLAS_MOE_SPLITK_V2=1`). Same ladder, same both-or-neither contract.
    /// The gate_up entries additionally require dynamic smem for the staged
    /// activation slices — the ops wrapper sizes it off the returned MROW.
    #[inline]
    pub(crate) fn splitk_m_t_v2_handles(
        &self,
        num_tokens: u32,
    ) -> Option<(
        spark_runtime::gpu::KernelHandle,
        spark_runtime::gpu::KernelHandle,
        u32,
    )> {
        let candidates: &[(u32, KernelHandle, KernelHandle, KernelHandle, KernelHandle)] = &[
            (
                2,
                self.moe_expert_gate_up_shared_t_m2_v2t_k,
                self.moe_expert_gate_up_shared_t_e8m0_m2_v2t_k,
                self.moe_expert_silu_down_shared_t_m2_v2t_k,
                self.moe_expert_silu_down_shared_t_e8m0_m2_v2t_k,
            ),
            (
                MOE_VERIFY_M6_ROWS,
                self.moe_expert_gate_up_shared_t_m6_v2t_k,
                self.moe_expert_gate_up_shared_t_e8m0_m6_v2t_k,
                self.moe_expert_silu_down_shared_t_m6_v2t_k,
                self.moe_expert_silu_down_shared_t_e8m0_m6_v2t_k,
            ),
            (
                MOE_VERIFY_MAX_ROWS,
                self.moe_expert_gate_up_shared_t_m8_v2t_k,
                self.moe_expert_gate_up_shared_t_e8m0_m8_v2t_k,
                self.moe_expert_silu_down_shared_t_m8_v2t_k,
                self.moe_expert_silu_down_shared_t_e8m0_m8_v2t_k,
            ),
        ];
        for &(mrow, gu, gu_e8m0, sd, sd_e8m0) in candidates {
            if num_tokens > mrow {
                continue;
            }
            let gate_up = self.e8m0_or_opt(gu, gu_e8m0);
            let silu_down = self.e8m0_or_opt(sd, sd_e8m0);
            if let (Some(g), Some(s)) = (gate_up, silu_down)
                && g.0 != 0
                && s.0 != 0
            {
                return Some((g, s, mrow));
            }
        }
        None
    }

    /// Partitioned 3..=`MOE_VERIFY_MAX_ROWS`-row verify kernels. Every arm
    /// writes disjoint rows of the same partial buffer, selected by the expert
    /// group's exact multiplicity. The duplicated / count-5-or-more arms are
    /// correct for narrower inputs because their gather loop is capped by
    /// `num_tokens`, so the MROW=6 pair serves the whole 3..=6 range and the
    /// MROW=8 pair is reached only at 7..8 — where m6 would clamp the gather
    /// and silently drop rows.
    pub(crate) fn splitk_m_t_partition_handles(
        &self,
        num_tokens: u32,
    ) -> Option<SplitkMPartitionHandles> {
        if !(3..=MOE_VERIFY_MAX_ROWS).contains(&num_tokens) {
            return None;
        }
        let wide = num_tokens > MOE_VERIFY_M6_ROWS;
        if self.moe_gate_up_partial_finalize_m_act_k.0 == 0 {
            return None;
        }
        let gate_unique = self.e8m0_or_opt(
            self.moe_expert_gate_up_shared_t_m1u_k,
            self.moe_expert_gate_up_shared_t_e8m0_m1u_k,
        )?;
        let down_unique = self.e8m0_or_opt(
            self.moe_expert_silu_down_shared_t_m1u_k,
            self.moe_expert_silu_down_shared_t_e8m0_m1u_k,
        )?;
        let gate_duplicated = if wide {
            self.e8m0_or_opt(
                self.moe_expert_gate_up_shared_t_m8d_k,
                self.moe_expert_gate_up_shared_t_e8m0_m8d_k,
            )?
        } else {
            self.e8m0_or_opt(
                self.moe_expert_gate_up_shared_t_m6d_k,
                self.moe_expert_gate_up_shared_t_e8m0_m6d_k,
            )?
        };
        let down_buckets = [
            (
                self.e8m0_or_opt(
                    self.moe_expert_silu_down_shared_t_m2c2_k,
                    self.moe_expert_silu_down_shared_t_e8m0_m2c2_k,
                )?,
                2,
            ),
            (
                self.e8m0_or_opt(
                    self.moe_expert_silu_down_shared_t_m4c34_k,
                    self.moe_expert_silu_down_shared_t_e8m0_m4c34_k,
                )?,
                4,
            ),
            // Open-ended top bucket (counts ≥ 5): its MROW must cover
            // `num_tokens`, or `m_out = min(s_m, MROW)` in the kernel clamps the
            // gather and the rows past MROW never get written.
            if wide {
                (
                    self.e8m0_or_opt(
                        self.moe_expert_silu_down_shared_t_m8c58_k,
                        self.moe_expert_silu_down_shared_t_e8m0_m8c58_k,
                    )?,
                    MOE_VERIFY_MAX_ROWS,
                )
            } else {
                (
                    self.e8m0_or_opt(
                        self.moe_expert_silu_down_shared_t_m6c56_k,
                        self.moe_expert_silu_down_shared_t_e8m0_m6c56_k,
                    )?,
                    MOE_VERIFY_M6_ROWS,
                )
            },
        ];
        Some(SplitkMPartitionHandles {
            gate_unique,
            gate_duplicated,
            down_unique,
            down_buckets,
        })
    }

    /// True when the routed experts are W3 Lloyd-Max (3-bit) — every
    /// expert-weight-reading dispatch site must select a `_w3` kernel.
    #[inline]
    pub(crate) fn is_w3(&self) -> bool {
        self.experts_scale_kind == crate::weight_map::WeightQuantFormat::W3LloydMax
    }

    /// Whether the full W3 kernel set resolved for this target image.
    pub(crate) fn w3_kernels_present(&self) -> bool {
        self.moe_expert_gate_up_shared_w3_k.0 != 0
            && self.moe_expert_silu_down_shared_w3_k.0 != 0
            && self.moe_expert_gate_up_shared_batchn_w3_k.0 != 0
            && self.moe_expert_silu_down_shared_batchn_w3_k.0 != 0
            && self.moe_grouped_gemm_w3_k.0 != 0
    }

    /// Arm the W3 (3-bit Lloyd-Max) routed-expert path: the caller has
    /// already replaced `weights.experts` (and thus the pointer tables built
    /// by `MoeLayer::new`) with W3 buffers from the w3cache. Fails — so the
    /// caller can stay NVFP4 — when the `_w3` kernel set is missing.
    pub fn enable_w3(&mut self, lut_dev: DevicePtr) -> Result<()> {
        anyhow::ensure!(
            self.w3_kernels_present(),
            "W3 kernels (moe_fused_w3 / moe_w3a16 modules) not compiled into this target"
        );
        anyhow::ensure!(!lut_dev.is_null(), "W3 codebook device pointer is NULL");
        self.experts_scale_kind = crate::weight_map::WeightQuantFormat::W3LloydMax;
        self.w3_lut_dev = lut_dev;
        Ok(())
    }
}

// ── Sub-files (split for ≤500 LoC) ────────────────────────────────────────
pub(crate) mod dump;
mod exl3_decode;
mod forward;
mod forward_atomic_c4;
mod forward_batched;
mod forward_ep;
mod forward_k2;
mod forward_k3;
mod forward_km;
mod forward_kn;
mod forward_phase;
mod forward_prefill;
mod forward_prefill_bf16;
mod forward_prefill_exl3;
mod forward_prefill_fp8;
mod forward_prefill_phase;
mod forward_prefill_routed;
mod forward_token_major;
pub(crate) mod gate_hist;
mod helpers_a;
mod helpers_b;
mod helpers_c;
mod init;
#[cfg(test)]
mod mod_tests;
mod ptr_table_build;
mod route_locality;
mod union_stats;
pub(crate) use ptr_table_build::*;
