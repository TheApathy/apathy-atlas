// SPDX-License-Identifier: AGPL-3.0-only
// Exact GLM-5.3 clamped SwiGLU and ordered BF16 expert reduction.

#include <cuda_bf16.h>
#include <math.h>

#define GLM53_HIDDEN 4096U
#define GLM53_EXPERTS 288U
#define GLM53_TOP_K 8U
#define GLM53_THREADS 256U

extern "C" __global__ void __launch_bounds__(GLM53_THREADS, 1)
atlas_glm53_clamped_swiglu(
        const __nv_bfloat16 * __restrict__ gate,
        const __nv_bfloat16 * __restrict__ up,
        __nv_bfloat16 * __restrict__ output,
        unsigned int rows, unsigned int width) {
    if (rows == 0U || (width != 2048U && width != 12288U)) {
        return;
    }
    const unsigned long long values =
        (unsigned long long) rows * (unsigned long long) width;
    unsigned long long index =
        (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long) gridDim.x * blockDim.x;
    for (; index < values; index += stride) {
        const float gate_value = fminf(__bfloat162float(gate[index]), 10.0f);
        const float up_value = fminf(fmaxf(__bfloat162float(up[index]), -10.0f), 10.0f);
        const __nv_bfloat16 silu = __float2bfloat16_rn(
            gate_value / (1.0f + expf(-gate_value)));
        const __nv_bfloat16 clamped_up = __float2bfloat16_rn(up_value);
        output[index] = __float2bfloat16_rn(
            __bfloat162float(silu) * __bfloat162float(clamped_up));
    }
}

extern "C" __global__ void __launch_bounds__(GLM53_THREADS, 1)
atlas_glm53_ordered_expert_reduce(
        const __nv_bfloat16 * __restrict__ routed,
        const unsigned int * __restrict__ indices,
        const float * __restrict__ weights,
        const __nv_bfloat16 * __restrict__ shared,
        __nv_bfloat16 * __restrict__ output,
        unsigned int tokens, unsigned int hidden,
        unsigned int experts, unsigned int top_k) {
    if (tokens == 0U || hidden != GLM53_HIDDEN ||
        experts != GLM53_EXPERTS || top_k != GLM53_TOP_K) {
        return;
    }
    const unsigned int token = blockIdx.x;
    if (token >= tokens) {
        return;
    }
    __shared__ unsigned int ordered_slots[GLM53_TOP_K];
    __shared__ float ordered_weights[GLM53_TOP_K];
    __shared__ unsigned int valid;
    if (threadIdx.x == 0U) {
        valid = 1U;
        const unsigned long long route_base =
            (unsigned long long) token * GLM53_TOP_K;
        #pragma unroll
        for (unsigned int slot = 0; slot < GLM53_TOP_K; ++slot) {
            ordered_slots[slot] = slot;
        }
        #pragma unroll
        for (unsigned int at = 1; at < GLM53_TOP_K; ++at) {
            const unsigned int slot = ordered_slots[at];
            const unsigned int expert = indices[route_base + slot];
            unsigned int insert = at;
            while (insert > 0U) {
                const unsigned int previous = ordered_slots[insert - 1U];
                const unsigned int previous_expert = indices[route_base + previous];
                if (previous_expert < expert ||
                    (previous_expert == expert && previous < slot)) {
                    break;
                }
                ordered_slots[insert] = previous;
                --insert;
            }
            ordered_slots[insert] = slot;
        }
        #pragma unroll
        for (unsigned int order = 0; order < GLM53_TOP_K; ++order) {
            const unsigned int slot = ordered_slots[order];
            const unsigned int expert = indices[route_base + slot];
            if (expert >= GLM53_EXPERTS ||
                (order > 0U && expert == indices[
                    route_base + ordered_slots[order - 1U]])) {
                valid = 0U;
            }
            ordered_weights[order] = weights[route_base + slot];
        }
    }
    __syncthreads();
    if (valid == 0U) {
        return;
    }

    const unsigned long long hidden_base =
        (unsigned long long) token * GLM53_HIDDEN;
    const unsigned long long routed_base =
        (unsigned long long) token * GLM53_TOP_K * GLM53_HIDDEN;
    for (unsigned int column = threadIdx.x; column < GLM53_HIDDEN;
         column += GLM53_THREADS) {
        __nv_bfloat16 sum = __float2bfloat16_rn(0.0f);
        #pragma unroll
        for (unsigned int order = 0; order < GLM53_TOP_K; ++order) {
            const unsigned int slot = ordered_slots[order];
            const __nv_bfloat16 term = __float2bfloat16_rn(
                __bfloat162float(routed[
                    routed_base + (unsigned long long) slot * GLM53_HIDDEN + column]) *
                ordered_weights[order]);
            sum = __float2bfloat16_rn(
                __bfloat162float(sum) + __bfloat162float(term));
        }
        output[hidden_base + column] = __float2bfloat16_rn(
            __bfloat162float(sum) +
            __bfloat162float(shared[hidden_base + column]));
    }
}
