// SPDX-License-Identifier: AGPL-3.0-only
// GLM-5.3 exact BF16 pooled-key score primitive; selection/top-k is separate.

#include <cuda_bf16.h>
#include <float.h>

#define GLM53_DSA_HEADS 32U
#define GLM53_DSA_INDEX_DIM 128U
#define GLM53_DSA_MAX_QUERIES 8U
#define GLM53_DSA_MAX_POOLS 262144U
#define GLM53_DSA_MAX_GRID_YZ 65535ULL
#define GLM53_DSA_INV_SQRT_DIM 0.08838834764831844055f
#define GLM53_DSA_INV_SQRT_HEADS 0.17677669529663688110f

extern "C" __global__ void __launch_bounds__(GLM53_DSA_INDEX_DIM, 1)
atlas_glm53_dsa_score_bf16(
        const __nv_bfloat16 * __restrict__ queries,
        const __nv_bfloat16 * __restrict__ raw_head_weights,
        const __nv_bfloat16 * __restrict__ pool_keys,
        const unsigned char * __restrict__ pool_validity,
        float * __restrict__ output_scores,
        unsigned int batch, unsigned int query_count,
        unsigned int pool_count) {
    const unsigned long long rows =
        (unsigned long long) batch * query_count;
    const unsigned long long expected_y =
        rows < GLM53_DSA_MAX_GRID_YZ ? rows : GLM53_DSA_MAX_GRID_YZ;
    const unsigned long long expected_z = expected_y == 0ULL ? 0ULL :
        (rows + expected_y - 1ULL) / expected_y;
    if (batch == 0U || query_count == 0U ||
        query_count > GLM53_DSA_MAX_QUERIES || pool_count == 0U ||
        pool_count > GLM53_DSA_MAX_POOLS ||
        blockDim.x != GLM53_DSA_INDEX_DIM || gridDim.x != pool_count ||
        (unsigned long long) gridDim.y != expected_y ||
        (unsigned long long) gridDim.z != expected_z ||
        expected_z > GLM53_DSA_MAX_GRID_YZ) {
        return;
    }
    const unsigned int channel = threadIdx.x;
    const unsigned long long row =
        (unsigned long long) blockIdx.y +
        (unsigned long long) gridDim.y * blockIdx.z;
    if (row >= rows) {
        return;
    }
    const unsigned long long pool = blockIdx.x;
    const unsigned long long batch_index = row / query_count;
    const unsigned long long pool_row = batch_index * pool_count + pool;
    const unsigned long long output = row * pool_count + pool;
    if (pool_validity[pool_row] == 0U) {
        if (channel == 0U) {
            output_scores[output] = -FLT_MAX;
        }
        return;
    }

    __shared__ float partial[GLM53_DSA_INDEX_DIM];
    float score = 0.0f;
    #pragma unroll 1
    for (unsigned int head = 0U; head < GLM53_DSA_HEADS; ++head) {
        const unsigned long long query_index =
            (row * GLM53_DSA_HEADS + head) * GLM53_DSA_INDEX_DIM + channel;
        const unsigned long long key_index =
            pool_row * GLM53_DSA_INDEX_DIM + channel;
        partial[channel] = __fmul_rn(
            __bfloat162float(queries[query_index]),
            __bfloat162float(pool_keys[key_index]));
        __syncthreads();
        #pragma unroll
        for (unsigned int stride = GLM53_DSA_INDEX_DIM / 2U;
             stride != 0U; stride >>= 1U) {
            if (channel < stride) {
                partial[channel] = __fadd_rn(
                    partial[channel], partial[channel + stride]);
            }
            __syncthreads();
        }
        if (channel == 0U) {
            float head_score = __fmul_rn(
                partial[0], GLM53_DSA_INV_SQRT_DIM);
            head_score = head_score < 0.0f ? 0.0f : head_score;
            const float weight = __fmul_rn(
                __bfloat162float(raw_head_weights[
                    row * GLM53_DSA_HEADS + head]),
                GLM53_DSA_INV_SQRT_HEADS);
            score = __fadd_rn(score, __fmul_rn(weight, head_score));
        }
        __syncthreads();
    }
    if (channel == 0U) {
        output_scores[output] = score;
    }
}
