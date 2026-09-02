// SPDX-License-Identifier: AGPL-3.0-only
// Direct Q4_K token-row gather for the UD-IQ2_XXS GLM-5.3 embedding table.
//
// UD-Q2_K_XL holds `token_embd.weight` at Q5_K, which `ggml_q5_embedding`
// covers. UD-IQ2_XXS drops it to Q4_K, so the same gather is needed one quant
// lower or the profile has no way to embed a token at all.
//
// Scale/min chronology and nibble order follow ggml-org/llama.cpp
// dequantize_row_q4_K. Q4_K is Q5_K without the `qh` high-bit plane: the
// 6-bit packed scale/min layout is byte-identical (`get_scale_min_k4`), the
// chunk/half/lane walk is the same, and the only difference is that the
// quantized value is the bare nibble rather than the nibble plus 16.

#include <cuda_bf16.h>
#include "../../qwen3.6-27b/nvfp4/q4k_vendor/mmq.cuh"

static_assert(QK_K == 256, "GLM Q4_K embedding requires QK_K=256");
static_assert(sizeof(block_q4_K) == 144, "GLM Q4_K embedding requires 144-byte Q4_K blocks");

// Identical packing to the Q5_K variant; kept local so neither file's
// chronology can drift into the other.
__device__ __forceinline__ void atlas_q4_scale_min(
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
atlas_q4_k_embedding_gather_bf16(
        const block_q4_K * __restrict__ source,
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
    // An out-of-range id yields zeros rather than reading past the table.
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
        const block_q4_K *block = source + block_index;

        unsigned char scale;
        unsigned char minimum;
        atlas_q4_scale_min((int) group, block->scales, &scale, &minimum);
        const unsigned char packed_quant = block->qs[chunk * 32u + lane];
        const unsigned int quant = half == 0u ? packed_quant & 0x0fu : packed_quant >> 4;

        const float d = __half2float(block->dm.x);
        const float dmin = __half2float(block->dm.y);
        const float value = d * (float) scale * (float) quant - dmin * (float) minimum;
        output[column] = __float2bfloat16_rn(value);
    }
}
