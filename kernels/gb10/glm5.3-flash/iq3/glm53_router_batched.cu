// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <math.h>

#define GLM53_HIDDEN 4096U
#define GLM53_EXPERTS 288U
#define GLM53_THREADS 256U

template<unsigned int TOKENS_PER_BLOCK>
__device__ void glm53_router_logits_token_reuse(
        const __nv_bfloat16 * __restrict__ input,
        const float * __restrict__ router,
        float * __restrict__ logits,
        unsigned int tokens, unsigned int hidden, unsigned int experts) {
    if (tokens == 0U || hidden != GLM53_HIDDEN || experts != GLM53_EXPERTS ||
        blockDim.x != GLM53_THREADS) return;
    const unsigned int tile = blockIdx.x / GLM53_EXPERTS;
    const unsigned int expert = blockIdx.x % GLM53_EXPERTS;
    const unsigned int first_token = tile * TOKENS_PER_BLOCK;
    if (first_token >= tokens) return;
    const unsigned long long router_base =
        (unsigned long long) expert * GLM53_HIDDEN;

    float sums[TOKENS_PER_BLOCK];
    #pragma unroll
    for (unsigned int row = 0U; row < TOKENS_PER_BLOCK; ++row) sums[row] = 0.0f;
    for (unsigned int k = threadIdx.x; k < GLM53_HIDDEN; k += GLM53_THREADS) {
        const float weight = router[router_base + k];
        #pragma unroll
        for (unsigned int row = 0U; row < TOKENS_PER_BLOCK; ++row) {
            const unsigned int token = first_token + row;
            if (token < tokens) {
                sums[row] = fmaf(
                    __bfloat162float(input[(unsigned long long) token * GLM53_HIDDEN + k]),
                    weight,
                    sums[row]);
            }
        }
    }

    __shared__ float partial[TOKENS_PER_BLOCK][GLM53_THREADS];
    #pragma unroll
    for (unsigned int row = 0U; row < TOKENS_PER_BLOCK; ++row) {
        partial[row][threadIdx.x] = sums[row];
    }
    __syncthreads();
    #pragma unroll
    for (unsigned int stride = GLM53_THREADS / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            #pragma unroll
            for (unsigned int row = 0U; row < TOKENS_PER_BLOCK; ++row) {
                partial[row][threadIdx.x] += partial[row][threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        #pragma unroll
        for (unsigned int row = 0U; row < TOKENS_PER_BLOCK; ++row) {
            const unsigned int token = first_token + row;
            if (token < tokens) {
                logits[(unsigned long long) token * GLM53_EXPERTS + expert] = partial[row][0];
            }
        }
    }
}

#define GLM53_ROUTER_BATCHED(NAME, TOKENS) \
extern "C" __global__ __launch_bounds__(GLM53_THREADS, 2) void NAME( \
        const __nv_bfloat16 * input, const float * router, float * logits, \
        unsigned int tokens, unsigned int hidden, unsigned int experts) { \
    glm53_router_logits_token_reuse<TOKENS>(input, router, logits, tokens, hidden, experts); \
}

GLM53_ROUTER_BATCHED(atlas_glm53_router_logits_t4, 4)
GLM53_ROUTER_BATCHED(atlas_glm53_router_logits_t32, 32)

#undef GLM53_ROUTER_BATCHED
