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
#[track_caller]
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
    super::log_gemm_shape("quant_gemv", 1, n, k);
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
#[track_caller]
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
    super::log_gemm_shape("quant_gemm", m, n, k);
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
#[track_caller]
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
    super::log_gemm_shape("w4a16_gemv", 1, n, k);
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

/// W4A16 GEMV over a BLOCK-DIAGONAL weight in ONE launch (the V4 MLA wo_a
/// shape): `n_total = o_groups * rows_per_group` output rows, row r reading
/// activation segment `input[(r / rows_per_group) * k ..]`. Bit-identical per
/// row to per-group `w4a16_gemv` launches; replaces 8 serial ~2.3 MB launches
/// with one 19 MB stream (153 -> 194 GB/s measured at the wo_a shape).
///
/// Kernel: `w4a16_gemv_grouped(A, B_packed, B_scale, scale2, C, N, K, rpg)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n_total: u32,
    k: u32,
    rows_per_group: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_total, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n_total)
        .arg_u32(k)
        .arg_u32(rows_per_group)
        .launch(stream)
}

/// Batched (M<=8) `w4a16_gemv_grouped`: per-row math byte-identical to
/// single-row `w4a16_gemv` (weights read once per chunk for all M rows), so
/// it delivers the ATLAS_OPROJ_EXACT verify numerics at batch speed. Strided:
/// input row i at `input + i*lda`, output row i at `output + i*ldc`
/// (elements). `rows_per_group = n_total` degenerates to a batched PLAIN
/// GEMV with single-row K order (the wo_b case).
///
/// Kernel: `w4a16_gemv_grouped_batchm(A, B, S, s2, C, M, N, K, lda, ldc, rpg)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_grouped_batchm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n_total: u32,
    k: u32,
    lda: u32,
    ldc: u32,
    rows_per_group: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_total, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n_total)
        .arg_u32(k)
        .arg_u32(lda)
        .arg_u32(ldc)
        .arg_u32(rows_per_group)
        .launch(stream)
}

/// W4A16 double-GEMV (M=2): reads weights once, computes 2 outputs.
///
/// A: [2, K] BF16 contiguous, B: NVFP4 packed, C: [2, N] BF16 contiguous.
/// Same weight bandwidth as single GEMV — eliminates GEMM M=2 tile waste.
///
/// Kernel: `w4a16_gemv_batch2(A, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[track_caller]
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
    super::log_gemm_shape("w4a16_gemv_batch2", 1, n, k);
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

/// W4A16 triple-GEMV (M=3): reads weights once, computes 3 outputs.
///
/// A: [3, K] BF16 contiguous, B: NVFP4 packed, C: [3, N] BF16 contiguous.
/// For K=3 speculative verification.
///
/// Kernel: `w4a16_gemv_batch3(A, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[track_caller]
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
    super::log_gemm_shape("w4a16_gemv_batch3", 1, n, k);
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

/// W4A16 batched GEMV (M<=MAX_M) — the NVFP4 sibling of `w8a16_gemv_batch4/16`.
///
/// Reads the NVFP4 weight matrix ONCE and computes `m` outputs (one per seq),
/// amortizing the weight read across the batch. `kernel` is `w4a16_gemv_batch4`
/// (M<=4) or `w4a16_gemv_batch16` (M<=16). A:`[m,K]` BF16, C:`[m,N]` BF16.
///
/// Kernel: `w4a16_gemv_batch4/16(A, B_packed, B_scale, scale2, C, M, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[track_caller]
pub fn w4a16_gemv_batchm(
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
    super::log_gemm_shape("w4a16_gemv_batchm", m, n, k);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
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

/// W4A16 batched GEMV (M<=4) with explicit A/C row strides (elements) — the
/// NVFP4 sibling of `w8a16_gemv_batch4_ld`. For batched GEMVs over a ROW
/// SLICE of a wider activation matrix (V4-Flash block-diagonal `wo_a` in the
/// batched verify). Weight/scale pointers may be group sub-views.
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn w4a16_gemv_batch4_ld(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight_packed: DevicePtr,
    weight_scale: DevicePtr,
    weight_scale_2: f32,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldc: u32,
    stream: u64,
) -> Result<()> {
    super::log_gemm_shape("w4a16_gemv_batch4_ld", m, n, k);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight_packed)
        .arg_ptr(weight_scale)
        .arg_f32(weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(lda)
        .arg_u32(ldc)
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
#[track_caller]
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
    super::log_gemm_shape("w4a16_gemv_qg", 1, n, k);
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
#[track_caller]
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
    super::log_gemm_shape("w4a16_gemv_qkvz", 1, n, k);
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
#[track_caller]
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
    super::log_gemm_shape("w4a16_gemv_qg_batch2", 1, n, k);
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
#[track_caller]
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
    super::log_gemm_shape("w4a16_gemv_qg_batch3", 1, n, k);
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
#[track_caller]
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
    super::log_gemm_shape("w4a16_gemv_dual_batch3", 1, n, k);
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

/// Dual-projection GEMV for 2 tokens (K+V or any 2 weight matrices).
///
/// Reads each weight matrix once, produces 2 output vectors per projection.
/// `blockIdx.z` selects projection 0 or 1.
///
/// Kernel: `w4a16_gemv_dual_batch2(A, B0, S0, s2_0, C0, B1, S1, s2_1, C1, N, K)`
/// Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
/// Input A: [2, K], Output C0: [2, N], C1: [2, N].
#[allow(clippy::too_many_arguments)]
#[track_caller]
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
    super::log_gemm_shape("w4a16_gemv_dual_batch2", 1, n, k);
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
