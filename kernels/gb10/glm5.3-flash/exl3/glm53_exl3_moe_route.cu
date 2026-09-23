// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// The fused ExLlamaV3 kernel consumes expert-major routes. One block performs
// a stable tiled pack: count experts once, prefix their destinations, then
// carry per-expert offsets across source-ordered 320-pair tiles. This preserves
// the exact K8 ordering without the old O(pairs^2) global rank scan and scales
// to layer-major prompt chunks through M2048.
extern "C" __global__ void atlas_glm53_exl3_pack_routes(
    const uint32_t* __restrict__ route_ids,
    const float* __restrict__ route_weights,
    int64_t* __restrict__ expert_count,
    int64_t* __restrict__ token_sorted,
    half* __restrict__ weight_sorted,
    uint32_t* __restrict__ status,
    uint32_t tokens,
    uint32_t experts,
    uint32_t top_k)
{
    if (blockIdx.x != 0) return;

    const uint32_t thread = threadIdx.x;
    const uint32_t pairs = tokens * top_k;
    __shared__ uint32_t tile_experts[320];
    __shared__ uint32_t expert_offsets[289];
    if (thread == 0) *status = 0;
    if (thread <= experts)
    {
        expert_count[thread] = 0;
        expert_offsets[thread] = 0;
    }
    __syncthreads();

    for (uint32_t source = thread; source < pairs; source += blockDim.x)
    {
        const uint32_t selected = route_ids[source];
        if (selected >= experts)
            atomicExch(status, 1U);
        else
            atomicAdd(reinterpret_cast<unsigned long long*>(
                          expert_count + selected), 1ULL);
    }
    __syncthreads();

    if (thread == 0)
    {
        uint32_t offset = 0;
        for (uint32_t expert = 0; expert < experts; ++expert)
        {
            expert_offsets[expert] = offset;
            offset += static_cast<uint32_t>(expert_count[expert]);
        }
        expert_offsets[experts] = offset;
    }
    __syncthreads();

    for (uint32_t tile = 0; tile < pairs; tile += blockDim.x)
    {
        const uint32_t source = tile + thread;
        const uint32_t selected = source < pairs ? route_ids[source] : experts;
        tile_experts[thread] = selected < experts ? selected : experts;
        __syncthreads();

        if (source < pairs && selected < experts)
        {
            uint32_t local_rank = 0;
            for (uint32_t prior = 0; prior < thread; ++prior)
                local_rank += tile_experts[prior] == selected;
            const uint32_t destination = expert_offsets[selected] + local_rank;
            token_sorted[destination] = static_cast<int64_t>(source / top_k);
            weight_sorted[destination] = __float2half_rn(route_weights[source]);
        }
        __syncthreads();

        if (thread < experts)
        {
            uint32_t tile_count = 0;
            const uint32_t remaining = pairs - tile;
            const uint32_t active = remaining < blockDim.x ? remaining : blockDim.x;
            for (uint32_t lane = 0; lane < active; ++lane)
                tile_count += tile_experts[lane] == thread;
            expert_offsets[thread] += tile_count;
        }
        __syncthreads();
    }
}

// Private pack preserves the original flat source pair. The route-private MoE
// derives the token for gather while retaining slot identity for its output.
extern "C" __global__ void atlas_glm53_exl3_pack_routes_private(
    const uint32_t* __restrict__ route_ids,
    const float* __restrict__ route_weights,
    int64_t* __restrict__ expert_count,
    int64_t* __restrict__ token_sorted,
    half* __restrict__ weight_sorted,
    uint32_t* __restrict__ status,
    uint32_t tokens,
    uint32_t experts,
    uint32_t top_k)
{
    if (blockIdx.x != 0) return;

    const uint32_t thread = threadIdx.x;
    const uint32_t pairs = tokens * top_k;
    __shared__ uint32_t tile_experts[320];
    __shared__ uint32_t expert_offsets[289];
    if (thread == 0) *status = 0;
    if (thread <= experts)
    {
        expert_count[thread] = 0;
        expert_offsets[thread] = 0;
    }
    __syncthreads();

    for (uint32_t source = thread; source < pairs; source += blockDim.x)
    {
        const uint32_t selected = route_ids[source];
        if (selected >= experts)
            atomicExch(status, 1U);
        else
            atomicAdd(reinterpret_cast<unsigned long long*>(
                          expert_count + selected), 1ULL);
    }
    __syncthreads();

    if (thread == 0)
    {
        uint32_t offset = 0;
        for (uint32_t expert = 0; expert < experts; ++expert)
        {
            expert_offsets[expert] = offset;
            offset += static_cast<uint32_t>(expert_count[expert]);
        }
        expert_offsets[experts] = offset;
    }
    __syncthreads();

    for (uint32_t tile = 0; tile < pairs; tile += blockDim.x)
    {
        const uint32_t source = tile + thread;
        const uint32_t selected = source < pairs ? route_ids[source] : experts;
        tile_experts[thread] = selected < experts ? selected : experts;
        __syncthreads();

        if (source < pairs && selected < experts)
        {
            uint32_t local_rank = 0;
            for (uint32_t prior = 0; prior < thread; ++prior)
                local_rank += tile_experts[prior] == selected;
            const uint32_t destination = expert_offsets[selected] + local_rank;
            token_sorted[destination] = static_cast<int64_t>(source);
            weight_sorted[destination] = __float2half_rn(route_weights[source]);
        }
        __syncthreads();

        if (thread < experts)
        {
            uint32_t tile_count = 0;
            const uint32_t remaining = pairs - tile;
            const uint32_t active = remaining < blockDim.x ? remaining : blockDim.x;
            for (uint32_t lane = 0; lane < active; ++lane)
                tile_count += tile_experts[lane] == thread;
            expert_offsets[thread] += tile_count;
        }
        __syncthreads();
    }
}

// Fold private route rows in slot order and add the shared expert in one pass.
extern "C" __global__ void atlas_glm53_exl3_combine_private_shared(
    const float* __restrict__ route_private,
    const __nv_bfloat16* __restrict__ shared,
    __nv_bfloat16* __restrict__ output,
    const uint32_t* __restrict__ status,
    uint32_t elements)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    if (*status != 0)
    {
        output[index] = __float2bfloat16(__int_as_float(0x7fc00000));
        return;
    }
    const uint32_t token = index / 4096;
    const uint32_t column = index % 4096;
    float sum = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 8; ++slot)
    {
        const uint32_t pair = token * 8 + slot;
        sum += route_private[pair * 4096 + column];
    }
    output[index] = __float2bfloat16(sum + __bfloat162float(shared[index]));
}

extern "C" __global__ void atlas_glm53_exl3_combine_shared(
    const float* __restrict__ routed,
    const __nv_bfloat16* __restrict__ shared,
    __nv_bfloat16* __restrict__ output,
    const uint32_t* __restrict__ status,
    uint32_t elements)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    if (*status != 0)
    {
        output[index] = __float2bfloat16(__int_as_float(0x7fc00000));
        return;
    }
    output[index] = __float2bfloat16(routed[index] + __bfloat162float(shared[index]));
}
