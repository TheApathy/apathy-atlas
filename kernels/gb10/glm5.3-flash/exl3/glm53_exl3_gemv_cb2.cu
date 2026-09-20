// SPDX-License-Identifier: AGPL-3.0-only
// Device implementation: ExLlamaV3 MIT, pinned and verified by build.rs.

#include <cuda_fp16.h>
#include <util.h>
#include <util.cuh>
#include <quant/exl3_gemv_kernel.cuh>

// Decode verifier specializations: FP16 output, mul1/cb2, rows 2..=8.
// Upstream selects narrow for N <= 8192 and wide for larger outputs.
template __global__ void exl3_gemv_kernel<2, false, 2, 1, 0, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemv_kernel<2, false, 2, 1, 1, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemv_kernel<3, false, 2, 1, 0, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemv_kernel<3, false, 2, 1, 1, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemv_kernel<4, false, 2, 1, 0, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemv_kernel<4, false, 2, 1, 1, false>(EXL3_GEMM_ARGS);
