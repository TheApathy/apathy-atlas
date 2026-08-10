// SPDX-License-Identifier: AGPL-3.0-only

//! FP8-weight dual-GEMV (batch=2) dispatch.
//!
//! `dense_gemv_fp8w_batch2` computes two output rows from one pass over the
//! FP8 weight matrix — the batch=2 sibling of `dense_gemv_fp8w`. It halves
//! FP8 weight bandwidth vs two M=1 GEMV launches and is bit-identical to
//! running `dense_gemv_fp8w` twice (per-token reduction order unchanged).
//! Used by the K=2 MTP verify path where the two verify positions share
//! weights but have distinct activations (lm_head, attention Q/K/V/O, SSM
//! out_proj).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::Fp8DenseWeight;

/// FP8-weight dual-GEMV. `input` is `[2, K]` BF16, `output` is `[2, N]` BF16.
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[track_caller]
pub fn dense_gemv_fp8w_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    super::log_gemm_shape("dense_gemv_fp8w_batch2", 1, n, k);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Block-scaled FP8 batched GEMV (M<=4). `input` is `[M, K]` BF16, `output` is
/// `[M, N]` BF16; `weight`/`block_scale` are the raw `w8a16_gemv` pointers (2D
/// block-scaled FP8). One pass over the FP8 weight serves all M rows — the M=4
/// sibling of `w8a16_gemv`, replacing `w8a16_gemm_pipelined` for n<=4 batched
/// decode (which pads M to a 128-row MMA tile). Bit-identical per-row to
/// `w8a16_gemv`. Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn w8a16_gemv_batch4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    super::log_gemm_shape("w8a16_gemv_batch4", m, n, k);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Block-scaled FP8 batched GEMV (M<=4) with explicit A/C row strides
/// (elements). For batched GEMVs over a ROW SLICE of a wider activation
/// matrix — e.g. the V4-Flash block-diagonal `wo_a`, where group g reads a
/// `group_in`-wide column slice of the `[M, nq*hd]` attention output
/// (`lda = nq*hd`) and writes an `o_lora`-wide slice of the `[M, latent]`
/// output (`ldc = latent`). Bit-identical per row to `w8a16_gemv` on the
/// same slice. Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn w8a16_gemv_batch4_ld(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldc: u32,
    stream: u64,
) -> Result<()> {
    super::log_gemm_shape("w8a16_gemv_batch4_ld", m, n, k);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(lda)
        .arg_u32(ldc)
        .launch(stream)
}

/// Block-scaled FP8 batched GEMV (M<=8) whose per-row accumulation is
/// BYTE-IDENTICAL to single-row `w8a16_gemv` (sequential adds, no pair-sum
/// fusion — see `w8a16_gemv_batchm_exact` in the kernel). The verify Phase-A
/// projections under `ATLAS_VERIFY_EXACT_GEMV=1`. Strided A/C.
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn w8a16_gemv_batchm_exact(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldc: u32,
    stream: u64,
) -> Result<()> {
    super::log_gemm_shape("w8a16_gemv_batchm_exact", m, n, k);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(lda)
        .arg_u32(ldc)
        .launch(stream)
}

/// Block-scaled FP8 dual-GEMV (batch=2). `input` is `[2, K]` BF16, `output` is
/// `[2, N]` BF16; `weight`/`block_scale` are the raw `w8a16_gemv` pointers.
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn w8a16_gemv_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    super::log_gemm_shape("w8a16_gemv_batch2", 1, n, k);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
