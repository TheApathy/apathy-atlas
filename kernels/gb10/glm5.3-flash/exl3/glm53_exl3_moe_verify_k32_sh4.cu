// SPDX-License-Identifier: AGPL-3.0-only
// Diagnostic pipeline variant of the exact staged K32 verifier. Arithmetic,
// output boundaries, grids, locks and ownership match the SH3 baseline; only
// the pinned donor's fourth asynchronous shared-memory stage is restored.
#include <cuda_fp16.h>
#include <stdint.h>
#include <util.h>
#include <util.cuh>
#include <ptx.cuh>
#include <quant/exl3_kernel_map.cuh>
#include <quant/hadamard_inner.cuh>
#include <quant/exl3_gemm_inner.cuh>

extern "C" __global__ __launch_bounds__(512)
void atlas_glm53_exl3_verify_gate_up_k32_sh4(
    const half* __restrict__ state_g,
    const half* __restrict__ state_u,
    half* __restrict__ intermediate_g,
    half* __restrict__ intermediate_u,
    const uint16_t** __restrict__ gate_trellis,
    const uint16_t** __restrict__ up_trellis,
    const uint32_t* __restrict__ chunk_expert,
    const uint32_t* __restrict__ chunk_start,
    const uint32_t* __restrict__ chunk_rows,
    const uint32_t* __restrict__ chunk_count,
    int hidden_dim,
    int intermediate_dim,
    int lock_stride,
    int* __restrict__ locks)
{
    const uint32_t chunk = blockIdx.y;
    if (chunk >= *chunk_count) return;
    const uint32_t projection = blockIdx.z;
    const uint32_t expert = chunk_expert[chunk];
    const uint32_t start = chunk_start[chunk];
    const int rows = static_cast<int>(chunk_rows[chunk]);
    const half* input = (projection == 0 ? state_g : state_u) +
                        static_cast<uint64_t>(start) * hidden_dim;
    half* output = (projection == 0 ? intermediate_g : intermediate_u) +
                   static_cast<uint64_t>(start) * intermediate_dim;
    const uint16_t* trellis = (projection == 0 ? gate_trellis : up_trellis)[expert];
    int* chunk_locks = locks +
        (static_cast<uint64_t>(chunk) * 2 + projection) * lock_stride;
    exl3_gemm_kernel_inner<2, false, 2, 16, 32, 256, 4, 3, false>(
        input, trellis, output, rows, hidden_dim, intermediate_dim, chunk_locks, nullptr);
}

extern "C" __global__ __launch_bounds__(512)
void atlas_glm53_exl3_verify_down_k32_sh4(
    const half* __restrict__ intermediate,
    half* __restrict__ state,
    const uint16_t** __restrict__ down_trellis,
    const uint32_t* __restrict__ chunk_expert,
    const uint32_t* __restrict__ chunk_start,
    const uint32_t* __restrict__ chunk_rows,
    const uint32_t* __restrict__ chunk_count,
    int lock_stride,
    int* __restrict__ locks)
{
    const uint32_t chunk = blockIdx.y;
    if (chunk >= *chunk_count) return;
    const uint32_t expert = chunk_expert[chunk];
    const uint32_t start = chunk_start[chunk];
    const int rows = static_cast<int>(chunk_rows[chunk]);
    int* chunk_locks = locks + static_cast<uint64_t>(chunk) * lock_stride;
    exl3_gemm_kernel_inner<2, false, 2, 16, 32, 256, 4, 3, false>(
        intermediate + static_cast<uint64_t>(start) * 2048,
        down_trellis[expert],
        state + static_cast<uint64_t>(start) * 4096,
        rows, 2048, 4096, chunk_locks, nullptr);
}
