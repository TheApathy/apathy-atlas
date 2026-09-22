// SPDX-License-Identifier: AGPL-3.0-only
// Correctness-first full-vocabulary top-16 for GLM-5.3 DFlash2.

#include <cuda_bf16.h>
#include <float.h>
#include <math.h>

#define GLM53_DF2_VOCAB 154880U
#define GLM53_DF2_TOP_K 16U
#define GLM53_DF2_MAX_TOKENS 8U
#define GLM53_DF2_TOPK_THREADS 256U

__device__ __forceinline__ bool glm53_dflash2_topk_better(
        float candidate_value, unsigned int candidate_token,
        float current_value, unsigned int current_token) {
    return candidate_token != 0xffffffffU &&
        (current_token == 0xffffffffU ||
         candidate_value > current_value ||
         (candidate_value == current_value &&
          candidate_token < current_token));
}

extern "C" __global__ void __launch_bounds__(GLM53_DF2_TOPK_THREADS, 1)
atlas_glm53_dflash2_topk_bf16(
        const __nv_bfloat16 * __restrict__ logits,
        unsigned int * __restrict__ candidates,
        float * __restrict__ unary,
        unsigned int * __restrict__ status,
        unsigned int batch,
        unsigned int tokens,
        unsigned int vocab,
        unsigned int top_k) {
    if (batch == 0U || tokens == 0U || tokens > GLM53_DF2_MAX_TOKENS ||
        vocab != GLM53_DF2_VOCAB || top_k != GLM53_DF2_TOP_K ||
        blockIdx.x >= (unsigned long long) batch * tokens) {
        return;
    }

    __shared__ float reduction_values[GLM53_DF2_TOPK_THREADS];
    __shared__ unsigned int reduction_tokens[GLM53_DF2_TOPK_THREADS];
    __shared__ float selected_values[GLM53_DF2_TOP_K];
    __shared__ unsigned int selected_tokens[GLM53_DF2_TOP_K];
    __shared__ unsigned int invalid;

    const unsigned int lane = threadIdx.x;
    const unsigned int row = blockIdx.x;
    const unsigned long long row_base =
        (unsigned long long) row * GLM53_DF2_VOCAB;
    if (lane == 0U) {
        invalid = 0U;
    }
    __syncthreads();
    for (unsigned int token = lane; token < GLM53_DF2_VOCAB;
         token += GLM53_DF2_TOPK_THREADS) {
        if (!isfinite(__bfloat162float(logits[row_base + token]))) {
            atomicOr(&invalid, 1U);
        }
    }
    __syncthreads();
    if (invalid != 0U) {
        if (lane == 0U) {
            status[row] = 1U;
        }
        return;
    }

    // Sixteen deterministic passes are intentionally correctness-first. Each
    // pass excludes every prior winner and reduces by value then lower token.
    #pragma unroll
    for (unsigned int pass = 0U; pass < GLM53_DF2_TOP_K; ++pass) {
        float local_value = -FLT_MAX;
        unsigned int local_token = 0xffffffffU;
        for (unsigned int token = lane; token < GLM53_DF2_VOCAB;
             token += GLM53_DF2_TOPK_THREADS) {
            bool already_selected = false;
            #pragma unroll
            for (unsigned int prior = 0U; prior < pass; ++prior) {
                if (selected_tokens[prior] == token) {
                    already_selected = true;
                }
            }
            if (already_selected) {
                continue;
            }
            const float value = __bfloat162float(logits[row_base + token]);
            if (glm53_dflash2_topk_better(
                    value, token, local_value, local_token)) {
                local_value = value;
                local_token = token;
            }
        }
        reduction_values[lane] = local_value;
        reduction_tokens[lane] = local_token;
        __syncthreads();

        #pragma unroll
        for (unsigned int stride = GLM53_DF2_TOPK_THREADS / 2U;
             stride > 0U; stride >>= 1U) {
            if (lane < stride && glm53_dflash2_topk_better(
                    reduction_values[lane + stride],
                    reduction_tokens[lane + stride],
                    reduction_values[lane],
                    reduction_tokens[lane])) {
                reduction_values[lane] = reduction_values[lane + stride];
                reduction_tokens[lane] = reduction_tokens[lane + stride];
            }
            __syncthreads();
        }
        if (lane == 0U) {
            selected_values[pass] = reduction_values[0];
            selected_tokens[pass] = reduction_tokens[0];
        }
        __syncthreads();
    }

    if (lane == 0U) {
        const unsigned long long output_base =
            (unsigned long long) row * GLM53_DF2_TOP_K;
        #pragma unroll
        for (unsigned int pass = 0U; pass < GLM53_DF2_TOP_K; ++pass) {
            candidates[output_base + pass] = selected_tokens[pass];
            unary[output_base + pass] = selected_values[pass];
        }
        status[row] = 0U;
    }
}
