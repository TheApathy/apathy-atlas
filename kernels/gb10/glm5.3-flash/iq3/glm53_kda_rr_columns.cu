// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <math.h>

#define GLM53_KDA_HEADS 64U
#define GLM53_KDA_DIM 128U
#define GLM53_KDA_RR_WARPS 4U

template<unsigned int COLUMNS_PER_WARP>
__device__ void glm53_kda_rr_columns(
        float * __restrict__ state,
        const __nv_bfloat16 * __restrict__ query,
        const __nv_bfloat16 * __restrict__ key,
        const __nv_bfloat16 * __restrict__ value,
        const float * __restrict__ log_decay,
        const __nv_bfloat16 * __restrict__ beta,
        __nv_bfloat16 * __restrict__ output,
        unsigned int batch, unsigned int tokens, unsigned int heads,
        unsigned int key_dim, unsigned int value_dim, float l2_epsilon) {
    if (batch == 0U || tokens <= 8U || heads != GLM53_KDA_HEADS ||
        key_dim != GLM53_KDA_DIM || value_dim != GLM53_KDA_DIM ||
        l2_epsilon != 1.0e-6f || blockDim.x != 128U) return;
    const unsigned int group = blockIdx.x;
    if (group >= batch * GLM53_KDA_HEADS) return;
    const unsigned int warp = threadIdx.x >> 5U;
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int first_column =
        (blockIdx.y * GLM53_KDA_RR_WARPS + warp) * COLUMNS_PER_WARP;
    if (first_column >= GLM53_KDA_DIM) return;
    const unsigned int sequence = group / GLM53_KDA_HEADS;
    const unsigned int head = group % GLM53_KDA_HEADS;
    const unsigned int r0 = lane;
    const unsigned int r1 = lane + 32U;
    const unsigned int r2 = lane + 64U;
    const unsigned int r3 = lane + 96U;
    const unsigned long long state_base =
        (unsigned long long) group * GLM53_KDA_DIM * GLM53_KDA_DIM;

    float s0[COLUMNS_PER_WARP];
    float s1[COLUMNS_PER_WARP];
    float s2[COLUMNS_PER_WARP];
    float s3[COLUMNS_PER_WARP];
    #pragma unroll
    for (unsigned int c = 0U; c < COLUMNS_PER_WARP; ++c) {
        const unsigned int column = first_column + c;
        if (column < GLM53_KDA_DIM) {
            s0[c] = state[state_base + r0 * GLM53_KDA_DIM + column];
            s1[c] = state[state_base + r1 * GLM53_KDA_DIM + column];
            s2[c] = state[state_base + r2 * GLM53_KDA_DIM + column];
            s3[c] = state[state_base + r3 * GLM53_KDA_DIM + column];
        }
    }

    for (unsigned int token = 0U; token < tokens; ++token) {
        const unsigned long long token_group =
            ((unsigned long long) sequence * tokens + token) * GLM53_KDA_HEADS + head;
        const unsigned long long vector_base = token_group * GLM53_KDA_DIM;
        float q0 = __bfloat162float(query[vector_base + r0]);
        float q1 = __bfloat162float(query[vector_base + r1]);
        float q2 = __bfloat162float(query[vector_base + r2]);
        float q3 = __bfloat162float(query[vector_base + r3]);
        float k0 = __bfloat162float(key[vector_base + r0]);
        float k1 = __bfloat162float(key[vector_base + r1]);
        float k2 = __bfloat162float(key[vector_base + r2]);
        float k3 = __bfloat162float(key[vector_base + r3]);
        float q_squares = (q0 * q0 + q2 * q2) + (q1 * q1 + q3 * q3);
        float k_squares = (k0 * k0 + k2 * k2) + (k1 * k1 + k3 * k3);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            q_squares += __shfl_down_sync(0xffffffffU, q_squares, offset);
            k_squares += __shfl_down_sync(0xffffffffU, k_squares, offset);
        }
        const float q_norm = sqrtf(
            __shfl_sync(0xffffffffU, q_squares, 0) + l2_epsilon);
        const float k_norm = sqrtf(
            __shfl_sync(0xffffffffU, k_squares, 0) + l2_epsilon);
        const float q_scale = (1.0f / sqrtf(128.0f)) / q_norm;
        q0 *= q_scale;
        q1 *= q_scale;
        q2 *= q_scale;
        q3 *= q_scale;
        k0 /= k_norm;
        k1 /= k_norm;
        k2 /= k_norm;
        k3 /= k_norm;
        const float d0 = expf(log_decay[vector_base + r0]);
        const float d1 = expf(log_decay[vector_base + r1]);
        const float d2 = expf(log_decay[vector_base + r2]);
        const float d3 = expf(log_decay[vector_base + r3]);
        const float beta_value = __bfloat162float(beta[token_group]);

        #pragma unroll
        for (unsigned int c = 0U; c < COLUMNS_PER_WARP; ++c) {
            const unsigned int column = first_column + c;
            if (column >= GLM53_KDA_DIM) continue;
            s0[c] *= d0;
            s1[c] *= d1;
            s2[c] *= d2;
            s3[c] *= d3;
            float memory = s0[c] * k0 + s1[c] * k1 + s2[c] * k2 + s3[c] * k3;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                memory += __shfl_down_sync(0xffffffffU, memory, offset);
            }
            memory = __shfl_sync(0xffffffffU, memory, 0);
            const float delta =
                (__bfloat162float(value[vector_base + column]) - memory) * beta_value;
            s0[c] += k0 * delta;
            s1[c] += k1 * delta;
            s2[c] += k2 * delta;
            s3[c] += k3 * delta;
            float result = s0[c] * q0 + s1[c] * q1 + s2[c] * q2 + s3[c] * q3;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                result += __shfl_down_sync(0xffffffffU, result, offset);
            }
            if (lane == 0U) {
                output[vector_base + column] = __float2bfloat16_rn(result);
            }
        }
    }

    #pragma unroll
    for (unsigned int c = 0U; c < COLUMNS_PER_WARP; ++c) {
        const unsigned int column = first_column + c;
        if (column < GLM53_KDA_DIM) {
            state[state_base + r0 * GLM53_KDA_DIM + column] = s0[c];
            state[state_base + r1 * GLM53_KDA_DIM + column] = s1[c];
            state[state_base + r2 * GLM53_KDA_DIM + column] = s2[c];
            state[state_base + r3 * GLM53_KDA_DIM + column] = s3[c];
        }
    }
}

#define GLM53_KDA_RR_KERNEL(NAME, COLUMNS) \
extern "C" __global__ __launch_bounds__(128, 4) void NAME( \
        float * state, const __nv_bfloat16 * query, const __nv_bfloat16 * key, \
        const __nv_bfloat16 * value, const float * log_decay, \
        const __nv_bfloat16 * beta, __nv_bfloat16 * output, \
        unsigned int batch, unsigned int tokens, unsigned int heads, \
        unsigned int key_dim, unsigned int value_dim, float l2_epsilon) { \
    glm53_kda_rr_columns<COLUMNS>(state, query, key, value, log_decay, beta, output, \
        batch, tokens, heads, key_dim, value_dim, l2_epsilon); \
}

GLM53_KDA_RR_KERNEL(atlas_glm53_kda_prefill_rr_c8, 8)

#undef GLM53_KDA_RR_KERNEL
