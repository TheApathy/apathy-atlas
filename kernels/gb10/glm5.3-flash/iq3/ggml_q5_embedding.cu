// SPDX-License-Identifier: AGPL-3.0-only
// Direct Q5_K token-row gather for the pinned GLM-5.3 embedding table.
// Scale/min, high-bit, and nibble chronology follows ggml-org/llama.cpp
// dequantize_row_q5_K at ca3d5a3e10d53f7ea672cb9b6178faca3e2807bc.

#include <cuda_bf16.h>
#include "../../qwen3.6-27b/nvfp4/q4k_vendor/mmq.cuh"

static_assert(QK_K == 256, "GLM Q5_K embedding requires QK_K=256");
static_assert(sizeof(block_q5_K) == 176, "GLM Q5_K embedding requires 176-byte Q5_K blocks");

__device__ __forceinline__ void atlas_q5_scale_min(
        int group,
        const unsigned char * __restrict__ packed,
        unsigned char *scale,
        unsigned char *minimum) {
    if (group < 4) {
        *scale = packed[group] & 63;
        *minimum = packed[group + 4] & 63;
    } else {
        *scale = (packed[group + 4] & 0x0f) | ((packed[group - 4] >> 6) << 4);
        *minimum = (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4);
    }
}

extern "C" __global__ void __launch_bounds__(256, 1)
atlas_q5_k_embedding_gather_bf16(
        const block_q5_K * __restrict__ source,
        const unsigned int * __restrict__ token_ids,
        __nv_bfloat16 * __restrict__ destination,
        unsigned int tokens,
        unsigned int vocab,
        unsigned int hidden) {
    if (tokens == 0 || vocab != 154880u || hidden != 4096u || hidden % QK_K != 0) {
        return;
    }
    const unsigned int output_row = blockIdx.x;
    if (output_row >= tokens) {
        return;
    }
    const unsigned int token = token_ids[output_row];
    __nv_bfloat16 *output = destination + (unsigned long long) output_row * hidden;
    if (token >= vocab) {
        for (unsigned int column = threadIdx.x; column < hidden; column += blockDim.x) {
            output[column] = __float2bfloat16_rn(0.0f);
        }
        return;
    }

    const unsigned long long blocks_per_row = hidden / QK_K;
    for (unsigned int column = threadIdx.x; column < hidden; column += blockDim.x) {
        const unsigned int within = column % QK_K;
        const unsigned int chunk = within / 64u;
        const unsigned int half = (within / 32u) & 1u;
        const unsigned int lane = within & 31u;
        const unsigned int group = 2u * chunk + half;
        const unsigned long long block_index =
            (unsigned long long) token * blocks_per_row + column / QK_K;
        const block_q5_K *block = source + block_index;

        unsigned char scale;
        unsigned char minimum;
        atlas_q5_scale_min((int) group, block->scales, &scale, &minimum);
        const unsigned char packed_quant = block->qs[chunk * 32u + lane];
        const unsigned int nibble = half == 0u ? packed_quant & 0x0fu : packed_quant >> 4;
        const unsigned int high_mask = (1u << half) << (2u * chunk);
        const unsigned int quant = nibble + ((block->qh[lane] & high_mask) != 0u ? 16u : 0u);

        const float d = __half2float(block->data.d);
        const float dmin = __half2float(block->data.dmin);
        const float scaled = d * (float) scale;
        const float offset = dmin * (float) minimum;
        const float value = scaled * (float) quant - offset;
        output[column] = __float2bfloat16_rn(value);
    }
}
