// SPDX-License-Identifier: AGPL-3.0-only
// Atlas routing wrappers over the pinned ExLlamaV3 MIT Hadamard operations.
// Gate/up and down GEMMs use the unchanged n256_f1 module, not this file.

#include <cuda_fp16.h>
#include <stdint.h>
#include <util.h>
#include <util.cuh>
#include <ptx.cuh>
#include <quant/exl3_kernel_map.cuh>
#include <quant/hadamard_inner.cuh>
#include "glm53_exl3_moe_private_hadamard.cuh"

extern "C" __global__ void atlas_glm53_exl3_build_chunks_private(
    const int64_t* __restrict__ expert_count,
    uint32_t* __restrict__ pair_expert,
    uint32_t* __restrict__ chunk_expert,
    uint32_t* __restrict__ chunk_start,
    uint32_t* __restrict__ chunk_rows,
    uint32_t* __restrict__ chunk_count,
    uint32_t* __restrict__ status,
    uint32_t experts,
    uint32_t max_chunks,
    uint32_t pairs)
{
    __shared__ uint32_t expert_start[289];
    __shared__ uint32_t ready;
    const uint32_t thread = threadIdx.x;
    if (thread == 0)
    {
        ready = 0;
        *chunk_count = 0;
        if (*status == 0)
        {
            bool valid = experts == 288 && pairs > 0 && pairs <= 2048 * 8 &&
                         pairs % 8 == 0 && max_chunks > 0 && max_chunks <= 1312;
            uint32_t total = 0;
            uint32_t chunks = 0;
            if (valid)
            {
                for (uint32_t expert = 0; expert < experts; ++expert)
                {
                    expert_start[expert] = total;
                    const int64_t count = expert_count[expert];
                    if (count < 0 || static_cast<uint64_t>(count) > pairs - total)
                    {
                        valid = false;
                        break;
                    }
                    const uint32_t rows = static_cast<uint32_t>(count);
                    const uint32_t needed = (rows + 15) / 16;
                    if (needed > max_chunks - chunks)
                    {
                        valid = false;
                        break;
                    }
                    total += rows;
                    chunks += needed;
                }
            }
            if (valid && total == pairs)
            {
                expert_start[experts] = total;
                uint32_t chunk = 0;
                for (uint32_t expert = 0; expert < experts; ++expert)
                {
                    const uint32_t end = expert_start[expert + 1];
                    for (uint32_t start = expert_start[expert]; start < end; start += 16)
                    {
                        chunk_expert[chunk] = expert;
                        chunk_start[chunk] = start;
                        chunk_rows[chunk] = min(16U, end - start);
                        ++chunk;
                    }
                }
                *chunk_count = chunks;
                ready = 1;
            }
            else
            {
                atomicExch(status, 2U);
            }
        }
    }
    // Every thread reaches this barrier on success and failure. In particular,
    // a capacity error cannot strand the remaining threads at __syncthreads.
    __syncthreads();
    if (!ready) return;
    if (thread < experts)
    {
        for (uint32_t pair = expert_start[thread]; pair < expert_start[thread + 1]; ++pair)
            pair_expert[pair] = thread;
    }
}

struct PrivatePair
{
    uint32_t expert;
    uint32_t source_pair;
    uint32_t ready;
};

template<bool WithSource>
inline __device__ void atlas_glm53_private_pair(
    uint32_t pair,
    uint32_t pairs,
    const uint32_t* __restrict__ pair_expert,
    const int64_t* __restrict__ token_sorted,
    uint32_t* __restrict__ status,
    PrivatePair& descriptor)
{
    // Another CTA may report an invalid descriptor concurrently. Latch once
    // per CTA so no partial warp exits before a FULL_MASK Hadamard shuffle.
    if (threadIdx.x == 0)
    {
        descriptor.ready = 0;
        if (*status == 0 && pair < pairs)
        {
            const uint32_t expert = pair_expert[pair];
            const int64_t source = WithSource ? token_sorted[pair] : static_cast<int64_t>(pair);
            if (expert < 288 && source >= 0 && static_cast<uint64_t>(source) < pairs)
            {
                descriptor.expert = expert;
                descriptor.source_pair = static_cast<uint32_t>(source);
                descriptor.ready = 1;
            }
            else
            {
                atomicExch(status, 3U);
            }
        }
    }
    __syncthreads();
}

extern "C" __global__ void atlas_glm53_exl3_staged_gather_private(
    const half* __restrict__ hidden_state,
    half* __restrict__ state_g,
    half* __restrict__ state_u,
    const half** __restrict__ gate_suh,
    const half** __restrict__ up_suh,
    const int64_t* __restrict__ token_sorted,
    const uint32_t* __restrict__ pair_expert,
    uint32_t pairs,
    uint32_t* __restrict__ status)
{
    const uint32_t pair = blockIdx.x;
    __shared__ PrivatePair descriptor;
    atlas_glm53_private_pair<true>(pair, pairs, pair_expert, token_sorted, status, descriptor);
    if (!descriptor.ready) return;
    const uint32_t expert = descriptor.expert;
    const uint32_t token = descriptor.source_pair / 8;
    const uint32_t warp = threadIdx.x / 32;
    for (uint32_t segment = warp; segment < 4096 / 128; segment += blockDim.x / 32)
    {
        const half* input = hidden_state + static_cast<uint64_t>(token) * 4096 + segment * 128;
        had_hf_r_128_inner<true, false>(input,
            state_g + static_cast<uint64_t>(pair) * 4096 + segment * 128,
            gate_suh[expert] + segment * 128, 0.088388347648f);
        had_hf_r_128_inner<true, false>(input,
            state_u + static_cast<uint64_t>(pair) * 4096 + segment * 128,
            up_suh[expert] + segment * 128, 0.088388347648f);
    }
}

extern "C" __global__ void atlas_glm53_exl3_staged_activate_private(
    half* __restrict__ intermediate_g,
    const half* __restrict__ intermediate_u,
    const half** __restrict__ gate_svh,
    const half** __restrict__ up_svh,
    const half** __restrict__ down_suh,
    const uint32_t* __restrict__ pair_expert,
    uint32_t pairs,
    uint32_t* __restrict__ status)
{
    const uint32_t pair = blockIdx.x;
    __shared__ PrivatePair descriptor;
    atlas_glm53_private_pair<false>(pair, pairs, pair_expert, nullptr, status, descriptor);
    if (!descriptor.ready) return;
    const uint32_t expert = descriptor.expert;
    const uint32_t warp = threadIdx.x / 32;
    for (uint32_t segment = warp; segment < 2048 / 128; segment += blockDim.x / 32)
    {
        const uint64_t offset = static_cast<uint64_t>(pair) * 2048 + segment * 128;
        had_hf_r_128_guad_inner(intermediate_g + offset, intermediate_u + offset,
            intermediate_g + offset, gate_svh[expert] + segment * 128,
            up_svh[expert] + segment * 128, down_suh[expert] + segment * 128,
            0.088388347648f, 10.0f, 0);
    }
}

extern "C" __global__ void atlas_glm53_exl3_staged_scatter_private(
    const half* __restrict__ state,
    float* __restrict__ output_state,
    const half** __restrict__ down_svh,
    const int64_t* __restrict__ token_sorted,
    const half* __restrict__ weight_sorted,
    const uint32_t* __restrict__ pair_expert,
    uint32_t pairs,
    uint32_t* __restrict__ status)
{
    const uint32_t pair = blockIdx.x;
    __shared__ PrivatePair descriptor;
    atlas_glm53_private_pair<true>(pair, pairs, pair_expert, token_sorted, status, descriptor);
    if (!descriptor.ready) return;
    const uint32_t expert = descriptor.expert;
    const uint32_t source_pair = descriptor.source_pair;
    const float weight = __half2float(weight_sorted[pair]);
    const uint32_t warp = threadIdx.x / 32;
    for (uint32_t segment = warp; segment < 4096 / 128; segment += blockDim.x / 32)
    {
        atlas_glm53_had_hf_r_128_d_private(
            state + static_cast<uint64_t>(pair) * 4096 + segment * 128,
            output_state + static_cast<uint64_t>(source_pair) * 4096 + segment * 128,
            down_svh[expert] + segment * 128, 0.088388347648f * weight);
    }
}
