// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Dense BF16 GEMM: C = A @ B^T.
///
/// A: [M, K] row-major (activations)
/// B: [N, K] row-major (weights, HuggingFace layout)
/// C: [M, N] row-major (output)
///
/// Kernel: `dense_gemm_bf16(A, B, C, M, N, K)`
/// Grid: (ceil(N/16), ceil(M/16), 1)  Block: (16, 16, 1)
/// Tensor-core BF16 GEMM: m16n8k16 MMA for 3-5x speedup over scalar.
/// Grid: (ceil(N/64), ceil(M/16), 1), Block: (128, 1, 1)
pub fn dense_gemm_tc(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), div_ceil(m, 16), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Split-K GEMM: partial products over K_splits chunks, then reduce.
/// Uses FP32 workspace of size K_splits * M * N * 4 bytes.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_splitk(
    gpu: &dyn GpuBackend,
    partial_kernel: KernelHandle,
    reduce_kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    workspace: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    k_splits: u32,
    stream: u64,
) -> Result<()> {
    // Phase 1: partial products
    KernelLaunch::new(gpu, partial_kernel)
        .grid([div_ceil(n, 16), div_ceil(m, 16), k_splits])
        .block([16, 16, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(workspace)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(k_splits)
        .launch(stream)?;
    // Phase 2: reduce and write BF16
    KernelLaunch::new(gpu, reduce_kernel)
        .grid([div_ceil(n, 256), m, 1])
        .block([256, 1, 1])
        .arg_ptr(workspace)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k_splits)
        .launch(stream)
}

pub fn dense_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 16), div_ceil(m, 16), 1])
        .block([16, 16, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMM: C = A @ dequant(B).
///
/// A: [M, K] BF16 activations
/// B: NVFP4 packed weights (E2M1 + FP8 scales + FP32 per-tensor scale)
/// C: [M, N] BF16 output
///
/// Kernel: `w4a16_gemm(A, B_packed, B_scale, scale2, C, M, N, K)`
/// Grid: (ceil(N/64), ceil(M/64), 1)  Block: (128, 1, 1)
pub fn w4a16_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMM with M_TILE=16: small-M specialization for K=γ verify
/// (MTP K=3 → M=3, DFlash γ=16 → M=16-17).
///
/// At small M, the parent `w4a16_gemm_n128` (M_TILE=64) discards 75-95%
/// of accumulator writes via `if (r < M)` guards. This launcher targets
/// `w4a16_gemm_t_m16` which redesigns warp partitioning so all 4 warps
/// process the SAME 16 rows but different N sub-tiles (32 each).
///
/// Grid: (ceil(N/128), ceil(M/16), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_n128_m16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 16), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMM with M_TILE=16 + N_TILE=64: K=3 MTP verify variant.
///
/// Same FP8 MMA pipeline as `w4a16_gemm_n128_m16` but with the N tile
/// halved (64 cols/CTA instead of 128). At intermediate=17408 this
/// doubles the CTA count to ~272 CTAs/projection — ~2.5 CTAs/SM on
/// GB10's 110 SMs vs the N=128 variant's ~1.2 CTAs/SM. Targets the
/// K=3 verify path on dense Qwen3.6-27B where the N=128 grid is too
/// small to hide cp.async/dequant latency.
///
/// Grid: (ceil(N/64), ceil(M/16), 1)  Block: (64, 1, 1) = 2 warps
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_n64_m16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), div_ceil(m, 16), 1])
        .block([64, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMM with N_TILE=128: same kernel signature, wider N tile.
///
/// Grid: (ceil(N/128), ceil(M/64), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMM v3: MiniMax-only shadow with K_STEP=64 (was 32 in v2).
/// Halves K-iteration count; doubles per-iter MMA count. 1 CTA/SM
/// (was 3 for v2) due to larger SMEM footprint.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_n128_m128_v3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMM v2: MiniMax-only shadow of `w4a16_gemm_n128_m128`.
///
/// Same CTA tile (M=128, N=128, K_STEP=32) but:
///   - blockDim 256 (8 warps) instead of 128 (4 warps)
///   - 3-stage cp.async pipeline instead of 2-stage
///   - Chunk 0 (rows 0-63) and chunk 1 (rows 64-127) MMAs run in parallel
///     across warps 0-3 and 4-7 instead of being serialized.
///
/// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (256, 1, 1)
/// SMEM: ~42.6 KB → 2 CTAs/SM (vs 3 for v1), but 2× warps/CTA.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_n128_m128_v2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMM: C = A @ B with 2-M-chunk CTA (M_TILE2=128).
///
/// Halves weight re-reads vs `w4a16_gemm_n128` for large M (ISL > 128):
/// each CTA covers 128 rows of A, loading B once for both 64-row halves.
/// ~2× speedup on qkvz (K=2048, N=12288) at ISL=1016.
///
/// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (128, 1, 1)
/// SMEM: ~29.8 KB → 3 blocks/SM (vs 5 for m64 at ~19.6 KB).
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_n128_m128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Pre-dequanted FP8 GEMM (prefill): C = A @ B_fp8.
///
/// A: [M, K] BF16, B_fp8: [N, K] FP8 E4M3 (pre-dequanted from NVFP4), C: [M, N] BF16.
/// Eliminates runtime NVFP4→FP8 dequant — only LOAD + FP8 MMA per K step.
///
/// Grid: (ceil(N/128), ceil(M/64), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    b_fp8: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Pre-dequant NVFP4 → FP8 E4M3.  One-time conversion at model load.
///
/// Reads B_packed[N, K/2] + B_scale[N, K/GROUP_SIZE] + scale2 → B_fp8[N, K].
///
/// Grid: (ceil(N*K/2 / 256), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn predequant_nvfp4_to_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    b_packed: DevicePtr,
    b_scale: DevicePtr,
    scale2: f32,
    b_fp8: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let total = n * k / 2;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(b_packed)
        .arg_ptr(b_scale)
        .arg_f32(scale2)
        .arg_ptr(b_fp8)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Convert BF16 activations to FP8 E4M3 for FP8×FP8 GEMM.
///
/// Grid: (ceil(total_elements/2 / 256), 1, 1)  Block: (256, 1, 1)
pub fn bf16_to_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    total_elements: u32,
    stream: u64,
) -> Result<()> {
    let threads_needed = total_elements / 2;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(threads_needed, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(total_elements)
        .launch(stream)
}

/// NVFP4 global absmax scan: scans BF16 [N, K] input, writes per-tensor
/// `max(|x|)` into `global_max` (FP32 scalar).
///
/// Caller must zero-initialize `global_max` BEFORE launch (the kernel
/// reduces via `atomicMax`). Kernel: `nvfp4_global_absmax`.
///
/// Grid: (min(total/256, 1024), 1, 1)  Block: (256, 1, 1)
pub fn nvfp4_global_absmax(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    global_max: DevicePtr,
    total_elements: u32,
    stream: u64,
) -> Result<()> {
    // Match `quantize_to_nvfp4` (loaders_fp8.rs) grid math.
    let grid = (total_elements / 256).clamp(1, 1024);
    KernelLaunch::new(gpu, kernel)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(global_max)
        .arg_u32(total_elements)
        .launch(stream)
}

/// Per-row NVFP4 quantization: takes BF16 [N, K], emits packed E2M1
/// nibbles `[N, K/2]` + FP8 E4M3 per-group scales `[N, K/16]`.
///
/// `scale2` is the per-tensor second-level scale: `global_max / (6.0 * 448.0)`.
///
/// Grid: (N, 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn quantize_bf16_to_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    packed_out: DevicePtr,
    scale_out: DevicePtr,
    scale2: f32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(packed_out)
        .arg_ptr(scale_out)
        .arg_f32(scale2)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Native NVFP4×NVFP4 (W4A4) tensor-core GEMM.
///
/// Both A and B are pre-quantized to NVFP4 (E2M1 nibbles + FP8 E4M3 per-
/// group scales + one FP32 per-tensor scale each). The kernel issues
/// native `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32`
/// instructions on SM120/SM121 (GB10).
///
/// A: [M, K/2] packed nibbles + [M, K/16] FP8 scales (row-major)
/// B: [N, K/2] packed nibbles + [N, K/16] FP8 scales (HuggingFace layout)
/// C: [M, N] BF16 output (row-major)
///
/// `scale2_ab = A_scale2 * B_scale2` (product of per-tensor second-level
/// scales).
///
/// Kernel: `nvfp4_nvfp4_gemm_t_m64`
/// Grid: (ceil(N/128), ceil(M/64), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_nvfp4_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    b_packed: DevicePtr,
    b_scale: DevicePtr,
    scale2_ab: f32,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(b_packed)
        .arg_ptr(b_scale)
        .arg_f32(scale2_ab)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Split-K variant of [`nvfp4_nvfp4_gemm`]: partitions the K-axis across
/// `k_splits` CTAs (along Z), writes FP32 partials to `c_partial`, then
/// the companion reduce kernel sums + applies `scale2_ab` + casts BF16.
///
/// Wins at small K*N (< 25M) where the non-split kernel underutilizes
/// SMs (per cutlass-widen bench data: 2.75x vs W4A16 at K=N=2048,
/// 2.17x at K=N=4096; regresses at K*N > 25M because of GMEM bandwidth
/// pressure during reduce).
///
/// Scratch sizing: `c_partial` must hold `k_splits * M * N * 4` bytes (FP32).
/// Constraint: K must be divisible by `(K_STEP=64) * k_splits` — the kernel
/// gracefully clamps if not, but allocations and tile math assume even.
///
/// Kernel: `nvfp4_nvfp4_gemm_t_m64_splitk` + `nvfp4_splitk_reduce`
/// Grid (phase 1): (ceil(N/128), ceil(M/64), k_splits)  Block: (128, 1, 1)
/// Grid (phase 2): (ceil(N/256), M, 1)                  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_nvfp4_gemm_splitk(
    gpu: &dyn GpuBackend,
    partial_kernel: KernelHandle,
    reduce_kernel: KernelHandle,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    b_packed: DevicePtr,
    b_scale: DevicePtr,
    scale2_ab: f32,
    c_partial: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    k_splits: u32,
    stream: u64,
) -> Result<()> {
    // Phase 1: partial products. Scale2 deferred to reduce kernel so
    // partials stay at native MMA magnitudes.
    KernelLaunch::new(gpu, partial_kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 64), k_splits])
        .block([128, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(b_packed)
        .arg_ptr(b_scale)
        .arg_ptr(c_partial)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(k_splits)
        .launch(stream)?;
    // Phase 2: reduce across K_SPLITS partials + apply scale2_ab + BF16 cast.
    KernelLaunch::new(gpu, reduce_kernel)
        .grid([div_ceil(n, 256), m, 1])
        .block([256, 1, 1])
        .arg_ptr(c_partial)
        .arg_ptr(output)
        .arg_f32(scale2_ab)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k_splits)
        .launch(stream)
}
