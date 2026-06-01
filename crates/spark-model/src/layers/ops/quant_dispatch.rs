// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Unified GEMV dispatch: select kernel based on weight quantization format.
///
/// Eliminates cascading if/else chains in layer forward methods. The enum
/// branch (~1 cycle) is negligible vs GPU kernel launch overhead (~5μs).
#[allow(clippy::too_many_arguments)]
pub fn quant_gemv(
    gpu: &dyn GpuBackend,
    gemv_nvfp4: KernelHandle,
    gemv_fp8: KernelHandle,
    gemv_dense: KernelHandle,
    input: DevicePtr,
    weight: &crate::weight_map::QuantWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    use crate::weight_map::QuantWeight;
    match weight {
        QuantWeight::Nvfp4(w) => w4a16_gemv(gpu, gemv_nvfp4, input, w, output, n, k, stream),
        QuantWeight::Fp8(w) => w8a16_gemv(
            gpu,
            gemv_fp8,
            input,
            w.weight,
            w.row_scale,
            output,
            n,
            k,
            stream,
        ),
        QuantWeight::Dense(w) => dense_gemv(gpu, gemv_dense, input, w, output, n, k, stream),
    }
}

/// Unified GEMM dispatch: select kernel based on weight quantization format.
///
/// For M>1 prefill projections (Q/K/V/O). Falls back to dense GEMM for BF16.
#[allow(clippy::too_many_arguments)]
pub fn quant_gemm(
    gpu: &dyn GpuBackend,
    gemm_nvfp4: KernelHandle,
    gemm_fp8: KernelHandle,
    gemm_dense: KernelHandle,
    input: DevicePtr,
    weight: &crate::weight_map::QuantWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    use crate::weight_map::QuantWeight;
    match weight {
        QuantWeight::Nvfp4(w) => w4a16_gemm(gpu, gemm_nvfp4, input, w, output, m, n, k, stream),
        QuantWeight::Fp8(w) => w8a16_gemm(
            gpu,
            gemm_fp8,
            input,
            w.weight,
            w.row_scale,
            output,
            m,
            n,
            k,
            stream,
        ),
        QuantWeight::Dense(w) => dense_gemm(gpu, gemm_dense, input, w, output, m, n, k, stream),
    }
}

/// W4A16 GEMV (M=1): C = A @ dequant(B) for single-row activations.
///
/// A: [1, K] BF16, B: NVFP4 packed, C: [1, N] BF16.
/// 4 outputs/block, 64 threads (2 warps) per output. Cross-warp smem reduction.
///
/// Kernel: `w4a16_gemv(A, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
///
/// ## Tuning sweep 2026-05-19 — keeping baseline
///
/// Profiled at 59-60% LPDDR5X bandwidth on Qwen3.6-27B M=1 small-projection
/// path (SSM qkvz_proj, attn Q/K/V/O). Four alternate block shapes were
/// benchmarked end-to-end on the K=3 verify path (mean tok/s over count /
/// essay / fruits):
///
///   variant            block_shape          mean tok/s   delta vs base
///   baseline           N=4, t=64, blk=256   20.92        —
///   v1                 N=2, t=128, blk=256  20.55        -1.8%
///   v2                 N=1, t=256, blk=256  19.69        -5.9%
///   v3                 N=8, t=32, blk=256   20.73        -0.9%
///   v4                 N=2, t=64, blk=128   20.92        ~0%
///
/// None beat baseline. K dims on these projections (~4096) are too small for
/// the inner loop to fully hide load latency; the existing (N=4, t=64) shape
/// is close to optimal for sm_121 occupancy. The variant kernels remain in
/// kernels/gb10/nvfp4/w4a16_gemv.cu as documented future-work candidates
/// (zero runtime cost when not loaded).
pub fn w4a16_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 double-GEMV (M=2): reads weights once, computes 2 outputs.
///
/// A: [2, K] BF16 contiguous, B: NVFP4 packed, C: [2, N] BF16 contiguous.
/// Same weight bandwidth as single GEMV — eliminates GEMM M=2 tile waste.
///
/// Kernel: `w4a16_gemv_batch2(A, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn w4a16_gemv_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Fused RMS Norm + Residual Save + W4A16 GEMV.
///
/// Replaces the `rms_norm_residual` + `w4a16_gemv` pair in front of the SSM
/// QKVZ projection. One CTA cooperates to compute the RMS-normalized hidden
/// state (writing both the normed and residual buffers), then each of the 4
/// outputs per CTA streams the W4A16 weights once and FMAs against the
/// smem-resident normed vector.
///
/// Dynamic shared memory:
///   `K * 2`  bytes for the normed BF16 vector
///   + `16 * 4` bytes for the E2M1 LUT
///   + `(BLOCK_SIZE / WARP_SIZE) * 4 = 32` bytes for warp partial sums
///   + `N_PER_BLOCK * 2 * 4 = 32` bytes for the GEMV cross-warp reduction
///
/// Kernel: `rms_norm_residual_w4a16_gemv(input, gamma, normed, residual,
///                                       B_packed, B_scale, scale2, C, N, K, eps)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_residual_w4a16_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gamma: &DenseWeight,
    normed_out: DevicePtr,
    residual_out: DevicePtr,
    weight: &QuantizedWeight,
    gemv_out: DevicePtr,
    n: u32,
    k: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    // K * sizeof(BF16) + LUT + warp partial sums + GEMV reduction scratch.
    let smem_bytes = k * 2 + 16 * 4 + 32 + 32;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .shared_mem(smem_bytes)
        .arg_ptr(input)
        .arg_ptr(gamma.weight)
        .arg_ptr(normed_out)
        .arg_ptr(residual_out)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(gemv_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_f32(eps)
        .launch(stream)
}

/// Fused RMS Norm + Residual Save + W4A16 batch-3 GEMV.
///
/// K=3 speculative-verify counterpart of `rms_norm_residual_w4a16_gemv`.
/// Inputs `input`, `residual`, `normed_out` are [3, K]; `gemv_out` is [3, N].
///
/// Dynamic shared memory:
///   `3 * K * 2` bytes for 3 normalized BF16 vectors
///   + 16 * 4 (LUT) + 32 (warp partials) + 32 * 3 (M=3 GEMV reduction) bytes
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_residual_w4a16_gemv_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gamma: &DenseWeight,
    normed_out: DevicePtr,
    residual_out: DevicePtr,
    weight: &QuantizedWeight,
    gemv_out: DevicePtr,
    n: u32,
    k: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    let smem_bytes = 3 * k * 2 + 16 * 4 + 32 + 32 * 3;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .shared_mem(smem_bytes)
        .arg_ptr(input)
        .arg_ptr(gamma.weight)
        .arg_ptr(normed_out)
        .arg_ptr(residual_out)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(gemv_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_f32(eps)
        .launch(stream)
}

/// W4A16 triple-GEMV (M=3): reads weights once, computes 3 outputs.
///
/// A: [3, K] BF16 contiguous, B: NVFP4 packed, C: [3, N] BF16 contiguous.
/// For K=3 speculative verification.
///
/// Kernel: `w4a16_gemv_batch3(A, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn w4a16_gemv_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 triple-GEMV (M=3) specialized for the LM head (large N=vocab).
///
/// Same M=3 algorithm as `w4a16_gemv_batch3` — reads each NVFP4 weight row
/// once and FMAs against 3 input rows — but uses `N_PER_BLOCK=8` (vs 4)
/// to halve the grid at the LM-head's huge N (≈248k for Qwen3.6-27B) and
/// 1 warp per output so there is NO cross-warp shared-memory reduce.
///
/// A: [3, K] BF16 contiguous, B: NVFP4 packed, C: [3, N] BF16 contiguous.
///
/// Kernel: `w4a16_gemv_batch3_logits(A, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/8), 1, 1)  Block: (256, 1, 1)
pub fn w4a16_gemv_batch3_logits(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMV with inline Q/Gate deinterleave on output write.
///
/// Same as `w4a16_gemv` but writes Q and Gate to deinterleaved positions,
/// eliminating the separate `deinterleave_qg` kernel (12 graph nodes saved).
///
/// Kernel: `w4a16_gemv_qg(A, B, S, s2, C, N, K, num_heads, head_dim)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .launch(stream)
}

/// W4A16 GEMV with inline QKVZ deinterleave on output write.
///
/// Same as `w4a16_gemv` but writes to deinterleaved output locations,
/// eliminating the separate `deinterleave_qkvz` kernel.
///
/// Kernel: `w4a16_gemv_qkvz(A, B, S, s2, C, N, K, ng, kd, vpg, vd)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qkvz(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_groups: u32,
    head_k_dim: u32,
    vheads_per_group: u32,
    head_v_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_groups)
        .arg_u32(head_k_dim)
        .arg_u32(vheads_per_group)
        .arg_u32(head_v_dim)
        .launch(stream)
}

/// Q+Gate GEMV for 2 tokens with inline deinterleave.
///
/// Reads the Q+Gate weight matrix once, produces 2 deinterleaved output
/// vectors (Q|Gate for each token). Replaces 2× `w4a16_gemv_qg` calls.
///
/// Kernel: `w4a16_gemv_qg_batch2(A, B, S, s2, C, N, K, num_heads, head_dim)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
/// Input A: [2, K], Output C: [2, N] deinterleaved [Q|G] per token.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .launch(stream)
}

/// W4A16 GEMV batch3 with inline Q/Gate deinterleave.
///
/// Reads the Q+Gate weight matrix once, produces 3 deinterleaved output
/// vectors (Q|Gate for each token). For K=3 speculative verification.
///
/// Kernel: `w4a16_gemv_qg_batch3(A, B, S, s2, C, N, K, num_heads, head_dim)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
/// Input A: [3, K], Output C: [3, N] deinterleaved [Q|G] per token.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .launch(stream)
}

/// Dual-projection GEMV for 3 tokens (K+V or any 2 weight matrices).
///
/// Reads each weight matrix once, produces 3 output vectors per projection.
/// `blockIdx.z` selects projection 0 or 1.
///
/// Kernel: `w4a16_gemv_dual_batch3(A, B0, S0, s2_0, C0, B1, S1, s2_1, C1, N, K)`
/// Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
/// Input A: [3, K], Output C0: [3, N], C1: [3, N].
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight0: &QuantizedWeight,
    output0: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight0.weight)
        .arg_ptr(weight0.weight_scale)
        .arg_f32(weight0.weight_scale_2)
        .arg_ptr(output0)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Tuned dual-projection GEMV for 3 tokens — gate+up fused into one CTA.
///
/// Geometry: 8 outputs per CTA (4 gate + 4 up, dispatched by warp), 64
/// threads per output (2 warps), 512 threads/block. Grid drops to
/// (ceil(N/4), 1, 1) — half the CTAs of [`w4a16_gemv_dual_batch3`] which
/// used z=2 to fan out gate/up. Each CTA reads the 3-token activation
/// vector once and reuses it across both projections via L1, halving the
/// L2 A-read traffic. Inner loop processes 16 K-values per iter (2× uint4
/// acts + 8-byte weight load + 1 FP8 scale) so the K-loop trip count and
/// scale-fetch frequency are both halved vs the baseline's K8 stride.
/// Output buffers byte-equivalent to the baseline.
///
/// Kernel: `w4a16_gemv_dual_batch3_tuned(A, B0, S0, s2_0, C0, B1, S1, s2_1, C1, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (512, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_batch3_tuned(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight0: &QuantizedWeight,
    output0: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([512, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight0.weight)
        .arg_ptr(weight0.weight_scale)
        .arg_f32(weight0.weight_scale_2)
        .arg_ptr(output0)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Strided W4A16 GEMV batch3 + inline Q/Gate deinterleave.
///
/// Like [`w4a16_gemv_qg_batch3`] but writes token i to
/// `output + i * out_stride_bf16` (in BF16 elements) so the K=3 verify
/// path can scatter directly into the interleaved `qkv_buf` layout
/// without a follow-up d2d copy.
///
/// Kernel: `w4a16_gemv_qg_batch3_strided(A, B, S, s2, C, N, K, num_heads, head_dim, out_stride)`
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg_batch3_strided(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    out_stride_bf16: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(out_stride_bf16)
        .launch(stream)
}

/// Strided dual-projection GEMV for 3 tokens.
///
/// Like [`w4a16_gemv_dual_batch3`] but writes token i to
/// `output{0,1} + i * out_stride_bf16` (in BF16 elements). Both
/// projections share the same per-token stride.
///
/// Kernel: `w4a16_gemv_dual_batch3_strided(A, B0, S0, s2_0, C0, B1, S1, s2_1, C1, N, K, out_stride)`
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_batch3_strided(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight0: &QuantizedWeight,
    output0: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    n: u32,
    k: u32,
    out_stride_bf16: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight0.weight)
        .arg_ptr(weight0.weight_scale)
        .arg_f32(weight0.weight_scale_2)
        .arg_ptr(output0)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride_bf16)
        .launch(stream)
}

/// Dual-projection GEMV for 2 tokens (K+V or any 2 weight matrices).
///
/// Reads each weight matrix once, produces 2 output vectors per projection.
/// `blockIdx.z` selects projection 0 or 1.
///
/// Kernel: `w4a16_gemv_dual_batch2(A, B0, S0, s2_0, C0, B1, S1, s2_1, C1, N, K)`
/// Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
/// Input A: [2, K], Output C0: [2, N], C1: [2, N].
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight0: &QuantizedWeight,
    output0: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight0.weight)
        .arg_ptr(weight0.weight_scale)
        .arg_f32(weight0.weight_scale_2)
        .arg_ptr(output0)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

// ── Position embeddings ────────────────────────────────────────────
