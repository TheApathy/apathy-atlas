// SPDX-License-Identifier: AGPL-3.0-only
// Device implementation: ExLlamaV3 MIT, pinned and verified by build.rs.

#include <cuda_fp16.h>
#include <stdint.h>

#include <util.h>
#include <util.cuh>
#include <ptx.cuh>
#include <quant/exl3_kernel_map.cuh>
#include <quant/hadamard_inner.cuh>

#ifndef ATLAS_GLM53_STAGED_TILE_K
#define ATLAS_GLM53_STAGED_TILE_K 16
#endif
#ifndef ATLAS_GLM53_STAGED_TILE_N
#define ATLAS_GLM53_STAGED_TILE_N 256
#endif
#ifndef ATLAS_GLM53_STAGED_SH_STAGES
#define ATLAS_GLM53_STAGED_SH_STAGES 3
#endif
#ifndef ATLAS_GLM53_STAGED_FRAG_STAGES
#define ATLAS_GLM53_STAGED_FRAG_STAGES 3
#endif
#ifndef ATLAS_GLM53_STAGED_MIN_BLOCKS
#define ATLAS_GLM53_STAGED_MIN_BLOCKS 2
#endif
// A chunk launch uses exactly one CTA per complete N256 output tile, so there
// are no partial-K writers to serialize. Removing the donor's split-K lock
// calls also removes their otherwise unused local polling slot.
#define barrier_acquire(lock, lock_i) ((void)(lock), (void)(lock_i))
#define barrier_release(lock, lock_d, last) \
    ((void)(lock), (void)(lock_d), (void)(last))
#include <quant/exl3_gemm_inner.cuh>
#undef barrier_acquire
#undef barrier_release

extern "C" __global__ void atlas_glm53_exl3_build_chunks(
    const int64_t* __restrict__ expert_count,
    uint32_t* __restrict__ pair_expert,
    uint32_t* __restrict__ chunk_expert,
    uint32_t* __restrict__ chunk_start,
    uint32_t* __restrict__ chunk_rows,
    uint32_t* __restrict__ chunk_count,
    uint32_t* __restrict__ status,
    uint32_t experts,
    uint32_t max_chunks)
{
    __shared__ uint32_t expert_start[289];
    const uint32_t thread = threadIdx.x;
    if (thread == 0)
    {
        uint32_t pair_offset = 0;
        uint32_t chunks = 0;
        for (uint32_t expert = 0; expert < experts; ++expert)
        {
            expert_start[expert] = pair_offset;
            const uint32_t rows = static_cast<uint32_t>(expert_count[expert]);
            for (uint32_t offset = 0; offset < rows; offset += 16)
            {
                if (chunks >= max_chunks)
                {
                    atomicExch(status, 2U);
                    *chunk_count = 0;
                    return;
                }
                chunk_expert[chunks] = expert;
                chunk_start[chunks] = pair_offset + offset;
                chunk_rows[chunks] = min(16U, rows - offset);
                ++chunks;
            }
            pair_offset += rows;
        }
        expert_start[experts] = pair_offset;
        *chunk_count = chunks;
    }
    __syncthreads();

    if (thread < experts)
    {
        for (uint32_t pair = expert_start[thread];
             pair < expert_start[thread + 1];
             ++pair)
            pair_expert[pair] = thread;
    }
}

extern "C" __global__ void atlas_glm53_exl3_staged_gather(
    const half* __restrict__ hidden_state,
    half* __restrict__ state_g,
    half* __restrict__ state_u,
    const half** __restrict__ gate_suh,
    const half** __restrict__ up_suh,
    const int64_t* __restrict__ token_sorted,
    const uint32_t* __restrict__ pair_expert,
    uint32_t pairs)
{
    const uint32_t pair = blockIdx.x;
    if (pair >= pairs) return;
    const uint32_t expert = pair_expert[pair];
    const uint32_t token = static_cast<uint32_t>(token_sorted[pair]);
    const uint32_t warp = threadIdx.x / 32;
    for (uint32_t segment = warp; segment < 4096 / 128; segment += blockDim.x / 32)
    {
        const half* input = hidden_state + static_cast<uint64_t>(token) * 4096 + segment * 128;
        had_hf_r_128_inner<true, false>(
            input,
            state_g + static_cast<uint64_t>(pair) * 4096 + segment * 128,
            gate_suh[expert] + segment * 128,
            0.088388347648f);
        had_hf_r_128_inner<true, false>(
            input,
            state_u + static_cast<uint64_t>(pair) * 4096 + segment * 128,
            up_suh[expert] + segment * 128,
            0.088388347648f);
    }
}

// One independent expert-major route chunk per grid.y and projection per
// grid.z. Keeping this entry GEMM-only prevents gather/activation/scatter
// state from remaining live across the tensor-core pipeline.
extern "C" __global__ __launch_bounds__(256, ATLAS_GLM53_STAGED_MIN_BLOCKS)
void atlas_glm53_exl3_staged_gate_up_k16(
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
    if (gridDim.x != static_cast<uint32_t>(intermediate_dim / ATLAS_GLM53_STAGED_TILE_N)) return;

    const uint32_t projection = blockIdx.z;
    const uint32_t expert = chunk_expert[chunk];
    const uint32_t start = chunk_start[chunk];
    const int rows = static_cast<int>(chunk_rows[chunk]);
    const half* input = (projection == 0 ? state_g : state_u) +
                        static_cast<uint64_t>(start) * hidden_dim;
    half* output = (projection == 0 ? intermediate_g : intermediate_u) +
                   static_cast<uint64_t>(start) * intermediate_dim;
    const uint16_t* trellis =
        (projection == 0 ? gate_trellis : up_trellis)[expert];
    int* chunk_locks = locks +
        (static_cast<uint64_t>(chunk) * 2 + projection) * lock_stride;

    exl3_gemm_kernel_inner<
        2,
        false,
        2,
        16,
        ATLAS_GLM53_STAGED_TILE_K,
        ATLAS_GLM53_STAGED_TILE_N,
        ATLAS_GLM53_STAGED_SH_STAGES,
        ATLAS_GLM53_STAGED_FRAG_STAGES,
        false>(
        input,
        trellis,
        output,
        rows,
        hidden_dim,
        intermediate_dim,
        chunk_locks,
        nullptr);
}

extern "C" __global__ void atlas_glm53_exl3_staged_activate(
    half* __restrict__ intermediate_g,
    const half* __restrict__ intermediate_u,
    const half** __restrict__ gate_svh,
    const half** __restrict__ up_svh,
    const half** __restrict__ down_suh,
    const uint32_t* __restrict__ pair_expert,
    uint32_t pairs)
{
    const uint32_t pair = blockIdx.x;
    if (pair >= pairs) return;
    const uint32_t expert = pair_expert[pair];
    const uint32_t warp = threadIdx.x / 32;
    for (uint32_t segment = warp; segment < 2048 / 128; segment += blockDim.x / 32)
    {
        const uint64_t offset = static_cast<uint64_t>(pair) * 2048 + segment * 128;
        had_hf_r_128_guad_inner(
            intermediate_g + offset,
            intermediate_u + offset,
            intermediate_g + offset,
            gate_svh[expert] + segment * 128,
            up_svh[expert] + segment * 128,
            down_suh[expert] + segment * 128,
            0.088388347648f,
            10.0f,
            0);
    }
}

extern "C" __global__ __launch_bounds__(256, ATLAS_GLM53_STAGED_MIN_BLOCKS)
void atlas_glm53_exl3_staged_down_k16(
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
    if (gridDim.x != 4096 / ATLAS_GLM53_STAGED_TILE_N) return;

    const uint32_t expert = chunk_expert[chunk];
    const uint32_t start = chunk_start[chunk];
    const int rows = static_cast<int>(chunk_rows[chunk]);
    int* chunk_locks = locks + static_cast<uint64_t>(chunk) * lock_stride;
    exl3_gemm_kernel_inner<
        2,
        false,
        2,
        16,
        ATLAS_GLM53_STAGED_TILE_K,
        ATLAS_GLM53_STAGED_TILE_N,
        ATLAS_GLM53_STAGED_SH_STAGES,
        ATLAS_GLM53_STAGED_FRAG_STAGES,
        false>(
        intermediate + static_cast<uint64_t>(start) * 2048,
        down_trellis[expert],
        state + static_cast<uint64_t>(start) * 4096,
        rows,
        2048,
        4096,
        chunk_locks,
        nullptr);
}

extern "C" __global__ void atlas_glm53_exl3_staged_scatter(
    const half* __restrict__ state,
    float* __restrict__ output_state,
    const half** __restrict__ down_svh,
    const int64_t* __restrict__ token_sorted,
    const half* __restrict__ weight_sorted,
    const uint32_t* __restrict__ pair_expert,
    uint32_t pairs)
{
    const uint32_t pair = blockIdx.x;
    if (pair >= pairs) return;
    const uint32_t expert = pair_expert[pair];
    const uint32_t token = static_cast<uint32_t>(token_sorted[pair]);
    const float weight = __half2float(weight_sorted[pair]);
    const uint32_t warp = threadIdx.x / 32;
    for (uint32_t segment = warp; segment < 4096 / 128; segment += blockDim.x / 32)
    {
        had_hf_r_128_d_inner(
            state + static_cast<uint64_t>(pair) * 4096 + segment * 128,
            output_state + static_cast<uint64_t>(token) * 4096 + segment * 128,
            down_svh[expert] + segment * 128,
            0.088388347648f * weight);
    }
}
