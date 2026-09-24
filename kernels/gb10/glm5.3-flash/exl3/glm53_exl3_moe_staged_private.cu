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
            bool valid = experts == 288 && pairs > 0 && pairs <= 8192 * 8 &&
                         pairs % 8 == 0 && max_chunks > 0 && max_chunks <= 4384;
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

inline __device__ void atlas_glm53_private_hadamard_values(
    const half* __restrict__ input_ptr,
    const half* __restrict__ post_scale,
    float r_scale,
    float* shared,
    float& output0,
    float& output1,
    float& output2,
    float& output3)
{
    const uint32_t lane = threadIdx.x & 31;
    half4 v = reinterpret_cast<const half4*>(input_ptr)[lane];
    const float v0 = __half2float(__low2half(v.x));
    const float v1 = __half2float(__high2half(v.x));
    const float v2 = __half2float(__low2half(v.y));
    const float v3 = __half2float(__high2half(v.y));
    const float s0 = v0 + v1;
    const float d0 = v0 - v1;
    const float s1 = v2 + v3;
    const float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    h0 *= r_scale;
    h1 *= r_scale;
    h2 *= r_scale;
    h3 *= r_scale;

    const half4 scales = reinterpret_cast<const half4*>(post_scale)[lane];
    h0 *= __low2float(scales.x);
    h1 *= __high2float(scales.x);
    h2 *= __low2float(scales.y);
    h3 *= __high2float(scales.y);
    shared[lane * 4 + 0] = h0;
    shared[lane * 4 + 1] = h1;
    shared[lane * 4 + 2] = h2;
    shared[lane * 4 + 3] = h3;
    __syncwarp();
    output0 = shared[0 + lane];
    output1 = shared[32 + lane];
    output2 = shared[64 + lane];
    output3 = shared[96 + lane];
    __syncwarp();
}

// Read expert-major rows through the inverse stable permutation and accumulate
// slots in the exact source order. This writes the final routed tensor directly
// and avoids materializing the rows*8*hidden route-private F32 tensor.
extern "C" __global__ void atlas_glm53_exl3_staged_combine_private(
    const half* __restrict__ state,
    float* __restrict__ output_state,
    const half** __restrict__ down_svh,
    const half* __restrict__ weight_sorted,
    const uint32_t* __restrict__ pair_expert,
    const uint32_t* __restrict__ source_to_sorted,
    uint32_t* __restrict__ status,
    uint32_t rows,
    uint32_t pairs)
{
    const uint32_t token = blockIdx.x;
    const uint32_t segment = blockIdx.y;
    const uint32_t lane = threadIdx.x & 31;
    if (rows < 1024 || rows > 8192 || pairs != rows * 8 ||
        gridDim.x != rows || gridDim.y != 32 || blockDim.x != 32)
    {
        if (lane == 0) atomicExch(status, 4U);
        return;
    }
    if (*status != 0) return;

    extern __shared__ float shared[];
    __shared__ uint32_t selected_pair;
    __shared__ uint32_t selected_expert;
    __shared__ uint32_t selected_ready;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (uint32_t slot = 0; slot < 8; ++slot)
    {
        const uint32_t source_pair = token * 8 + slot;
        if (lane == 0)
        {
            selected_ready = 0;
            const uint32_t pair = source_to_sorted[source_pair];
            const uint32_t expert = pair < pairs ? pair_expert[pair] : 288;
            if (pair < pairs && expert < 288)
            {
                selected_pair = pair;
                selected_expert = expert;
                selected_ready = 1;
            }
            else
            {
                atomicExch(status, 5U);
            }
        }
        __syncwarp();
        if (!selected_ready) return;
        const uint32_t pair = selected_pair;
        const uint32_t expert = selected_expert;
        float value0;
        float value1;
        float value2;
        float value3;
        atlas_glm53_private_hadamard_values(
            state + static_cast<uint64_t>(pair) * 4096 + segment * 128,
            down_svh[expert] + segment * 128,
            0.088388347648f * __half2float(weight_sorted[pair]),
            shared,
            value0,
            value1,
            value2,
            value3);
        sum0 += value0;
        sum1 += value1;
        sum2 += value2;
        sum3 += value3;
    }
    const uint64_t output = static_cast<uint64_t>(token) * 4096 + segment * 128;
    output_state[output + 0 + lane] = sum0;
    output_state[output + 32 + lane] = sum1;
    output_state[output + 64 + lane] = sum2;
    output_state[output + 96 + lane] = sum3;
}
