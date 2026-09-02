// SPDX-License-Identifier: AGPL-3.0-only
// GLM-5.3 exact online DSA kpool4 compressor.

#include <cuda_bf16.h>
#include <math_constants.h>

#define GLM53_DSA_INDEX_DIM 128U
#define GLM53_DSA_KPOOL 4U
#define GLM53_DSA_TAIL 3U
#define GLM53_DSA_MAX_TOKENS 65520U
#define GLM53_DSA_MAX_POSITIONS 1048576U

__device__ __forceinline__ unsigned char glm53_dsa_source_valid(
        const unsigned char *input_validity,
        const unsigned char *tail_validity,
        unsigned int batch_index, unsigned int tokens,
        unsigned int initial_tail, unsigned int combined_index) {
    if (combined_index < initial_tail) {
        const unsigned long long index =
            (unsigned long long) batch_index * GLM53_DSA_TAIL + combined_index;
        return tail_validity[index];
    }
    const unsigned int token = combined_index - initial_tail;
    const unsigned long long index =
        (unsigned long long) batch_index * tokens + token;
    return input_validity[index];
}

__device__ __forceinline__ __nv_bfloat16 glm53_dsa_source_vector(
        const __nv_bfloat16 *input,
        const __nv_bfloat16 *tail,
        unsigned int batch_index, unsigned int tokens,
        unsigned int initial_tail, unsigned int combined_index,
        unsigned int channel) {
    if (combined_index < initial_tail) {
        const unsigned long long index =
            ((unsigned long long) batch_index * GLM53_DSA_TAIL + combined_index) *
                GLM53_DSA_INDEX_DIM + channel;
        return tail[index];
    }
    const unsigned int token = combined_index - initial_tail;
    const unsigned long long index =
        ((unsigned long long) batch_index * tokens + token) *
            GLM53_DSA_INDEX_DIM + channel;
    return input[index];
}

extern "C" __global__ void __launch_bounds__(GLM53_DSA_INDEX_DIM, 1)
atlas_glm53_dsa_pool_k4(
        const __nv_bfloat16 * __restrict__ input_keys,
        const __nv_bfloat16 * __restrict__ input_gates,
        const unsigned char * __restrict__ input_validity,
        const __nv_bfloat16 * __restrict__ prior_tail_keys,
        const __nv_bfloat16 * __restrict__ prior_tail_gates,
        const unsigned char * __restrict__ prior_tail_validity,
        const float * __restrict__ ape,
        __nv_bfloat16 * __restrict__ output_pool_keys,
        unsigned char * __restrict__ output_pool_validity,
        __nv_bfloat16 * __restrict__ output_tail_keys,
        __nv_bfloat16 * __restrict__ output_tail_gates,
        unsigned char * __restrict__ output_tail_validity,
        unsigned int batch, unsigned int tokens, unsigned int start_position,
        unsigned int initial_tail, unsigned int complete_pools,
        unsigned int final_tail) {
    if (batch == 0U || tokens == 0U || tokens > GLM53_DSA_MAX_TOKENS ||
        blockDim.x != GLM53_DSA_INDEX_DIM ||
        start_position >= GLM53_DSA_MAX_POSITIONS ||
        (unsigned long long) start_position + tokens > GLM53_DSA_MAX_POSITIONS ||
        initial_tail != start_position % GLM53_DSA_KPOOL ||
        complete_pools != (initial_tail + tokens) / GLM53_DSA_KPOOL ||
        final_tail != (initial_tail + tokens) % GLM53_DSA_KPOOL ||
        (unsigned long long) gridDim.x !=
            (unsigned long long) batch * (complete_pools + 1U)) {
        return;
    }
    const unsigned int channel = threadIdx.x;
    const unsigned long long pool_blocks =
        (unsigned long long) batch * complete_pools;
    const unsigned long long block = blockIdx.x;
    if (block < pool_blocks) {
        const unsigned int batch_index = (unsigned int) (block / complete_pools);
        const unsigned int pool = (unsigned int) (block % complete_pools);
        const unsigned int first = pool * GLM53_DSA_KPOOL;
        float logits[GLM53_DSA_KPOOL];
        float exponentials[GLM53_DSA_KPOOL];
        unsigned char valid[GLM53_DSA_KPOOL];
        float maximum = -CUDART_INF_F;
        bool all_valid = true;
        #pragma unroll
        for (unsigned int slot = 0U; slot < GLM53_DSA_KPOOL; ++slot) {
            const unsigned int combined = first + slot;
            valid[slot] = glm53_dsa_source_valid(
                input_validity, prior_tail_validity, batch_index, tokens,
                initial_tail, combined);
            all_valid = all_valid && valid[slot] != 0U;
            if (valid[slot] != 0U) {
                const float gate = __bfloat162float(glm53_dsa_source_vector(
                    input_gates, prior_tail_gates, batch_index, tokens,
                    initial_tail, combined, channel));
                logits[slot] = gate + ape[(unsigned long long) slot *
                    GLM53_DSA_INDEX_DIM + channel];
                maximum = fmaxf(maximum, logits[slot]);
            } else {
                logits[slot] = -CUDART_INF_F;
            }
        }
        float denominator = 0.0f;
        #pragma unroll
        for (unsigned int slot = 0U; slot < GLM53_DSA_KPOOL; ++slot) {
            exponentials[slot] = valid[slot] != 0U
                ? expf(logits[slot] - maximum) : 0.0f;
            denominator += exponentials[slot];
        }
        float accumulated = 0.0f;
        #pragma unroll
        for (unsigned int slot = 0U; slot < GLM53_DSA_KPOOL; ++slot) {
            const float probability = denominator == 0.0f
                ? 0.0f : exponentials[slot] / denominator;
            const __nv_bfloat16 probability_bf16 = __float2bfloat16_rn(probability);
            const __nv_bfloat16 key = glm53_dsa_source_vector(
                input_keys, prior_tail_keys, batch_index, tokens,
                initial_tail, first + slot, channel);
            const __nv_bfloat16 product = __float2bfloat16_rn(
                __bfloat162float(probability_bf16) * __bfloat162float(key));
            accumulated = __fadd_rn(accumulated, __bfloat162float(product));
        }
        const unsigned long long output =
            (block * GLM53_DSA_INDEX_DIM) + channel;
        output_pool_keys[output] = __float2bfloat16_rn(accumulated);
        if (channel == 0U) {
            output_pool_validity[block] = all_valid ? 1U : 0U;
        }
        return;
    }

    const unsigned int batch_index = (unsigned int) (block - pool_blocks);
    if (batch_index >= batch) {
        return;
    }
    const unsigned int tail_first = complete_pools * GLM53_DSA_KPOOL;
    #pragma unroll
    for (unsigned int slot = 0U; slot < GLM53_DSA_TAIL; ++slot) {
        const bool present = slot < final_tail;
        const unsigned int combined = tail_first + slot;
        const unsigned long long output =
            ((unsigned long long) batch_index * GLM53_DSA_TAIL + slot) *
                GLM53_DSA_INDEX_DIM + channel;
        output_tail_keys[output] = present
            ? glm53_dsa_source_vector(
                input_keys, prior_tail_keys, batch_index, tokens,
                initial_tail, combined, channel)
            : __float2bfloat16_rn(0.0f);
        output_tail_gates[output] = present
            ? glm53_dsa_source_vector(
                input_gates, prior_tail_gates, batch_index, tokens,
                initial_tail, combined, channel)
            : __float2bfloat16_rn(0.0f);
        if (channel == 0U) {
            const unsigned long long valid_output =
                (unsigned long long) batch_index * GLM53_DSA_TAIL + slot;
            output_tail_validity[valid_output] = present
                ? (glm53_dsa_source_valid(
                    input_validity, prior_tail_validity, batch_index, tokens,
                    initial_tail, combined) != 0U ? 1U : 0U)
                : 0U;
        }
    }
}
