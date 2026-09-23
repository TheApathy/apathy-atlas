// SPDX-License-Identifier: AGPL-3.0-only
// Scheduling glue only. Arithmetic: unchanged pinned ExLlamaV3 MIT inner.
// Unlike K16 staging, K32 retains both sub-K accumulator chains and their
// ordered final addition. Outputs here are RAW FP16, before output Hadamard.
#include <cuda_fp16.h>
#include <stdint.h>
#include <util.h>
#include <util.cuh>
#include <ptx.cuh>
#include <quant/exl3_kernel_map.cuh>
#include <quant/hadamard_inner.cuh>
#include <quant/exl3_gemm_inner.cuh>

// Caller preflights rows2..8, P=8*rows, scratch/descriptor extents and fixed
// grids. Existing private chunk builder bounds every start/span before these
// launches. One CTA owns a complete K column: retain pinned locks/reset,
// but there are no cross-CTA partial sums or cooperative group barriers.
// Dynamic shared bytes: 3*(2*16*32 + 2*(2*16*256/16*2)) + 4*(4*256*4)
// = 25,600. No gather/activation/scatter state is live across this pipeline.
extern "C" __global__ __launch_bounds__(512)
void atlas_glm53_exl3_verify_gate_up_k32(
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
    exl3_gemm_kernel_inner<2, false, 2, 16, 32, 256, 3, 3, false>(
        input, trellis, output, rows, hidden_dim, intermediate_dim, chunk_locks, nullptr);
}

extern "C" __global__ __launch_bounds__(512)
void atlas_glm53_exl3_verify_down_k32(
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
    exl3_gemm_kernel_inner<2, false, 2, 16, 32, 256, 3, 3, false>(
        intermediate + static_cast<uint64_t>(start) * 2048,
        down_trellis[expert],
        state + static_cast<uint64_t>(start) * 4096,
        rows, 2048, 4096, chunk_locks, nullptr);
}
