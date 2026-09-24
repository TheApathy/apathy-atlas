// SPDX-License-Identifier: AGPL-3.0-only
#include "../iq3/glm53_kda_rr_columns.cu"

// Software-pipelined variant (private tree): the next token's q/k/v/decay/beta
// are loaded into registers one iteration ahead so the sequential recurrence
// does not sit on a dependent global-load chain every step. Arithmetic order
// per element is unchanged, so results are bit-identical to `rr_c8`.
template<unsigned int COLUMNS_PER_WARP, bool PREFETCH_VALUES>
__device__ void glm53_kda_rr_columns_prefetch(
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

    // Prefetch registers for token t+1 (raw, un-normalized loads only).
    __nv_bfloat16 nq0, nq1, nq2, nq3, nk0, nk1, nk2, nk3, nbeta;
    float nl0, nl1, nl2, nl3;
    __nv_bfloat16 nv[COLUMNS_PER_WARP];
    auto load_token = [&] (unsigned int token) {
        const unsigned long long token_group =
            ((unsigned long long) sequence * tokens + token) * GLM53_KDA_HEADS + head;
        const unsigned long long vector_base = token_group * GLM53_KDA_DIM;
        nq0 = query[vector_base + r0]; nq1 = query[vector_base + r1];
        nq2 = query[vector_base + r2]; nq3 = query[vector_base + r3];
        nk0 = key[vector_base + r0]; nk1 = key[vector_base + r1];
        nk2 = key[vector_base + r2]; nk3 = key[vector_base + r3];
        nl0 = log_decay[vector_base + r0]; nl1 = log_decay[vector_base + r1];
        nl2 = log_decay[vector_base + r2]; nl3 = log_decay[vector_base + r3];
        nbeta = beta[token_group];
        if constexpr (PREFETCH_VALUES) {
            #pragma unroll
            for (unsigned int c = 0U; c < COLUMNS_PER_WARP; ++c) {
                const unsigned int column = first_column + c;
                nv[c] = column < GLM53_KDA_DIM ? value[vector_base + column] : __float2bfloat16(0.0f);
            }
        }
    };
    load_token(0U);

    for (unsigned int token = 0U; token < tokens; ++token) {
        const unsigned long long token_group =
            ((unsigned long long) sequence * tokens + token) * GLM53_KDA_HEADS + head;
        const unsigned long long vector_base = token_group * GLM53_KDA_DIM;
        float q0 = __bfloat162float(nq0);
        float q1 = __bfloat162float(nq1);
        float q2 = __bfloat162float(nq2);
        float q3 = __bfloat162float(nq3);
        float k0 = __bfloat162float(nk0);
        float k1 = __bfloat162float(nk1);
        float k2 = __bfloat162float(nk2);
        float k3 = __bfloat162float(nk3);
        const float l0 = nl0, l1 = nl1, l2 = nl2, l3 = nl3;
        const float beta_value = __bfloat162float(nbeta);
        float v[COLUMNS_PER_WARP];
        #pragma unroll
        for (unsigned int c = 0U; c < COLUMNS_PER_WARP; ++c) {
            if constexpr (PREFETCH_VALUES) {
                v[c] = __bfloat162float(nv[c]);
            } else {
                const unsigned int column = first_column + c;
                v[c] = column < GLM53_KDA_DIM ? __bfloat162float(value[vector_base + column]) : 0.0f;
            }
        }
        if (token + 1U < tokens) load_token(token + 1U);

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
        const float d0 = expf(l0);
        const float d1 = expf(l1);
        const float d2 = expf(l2);
        const float d3 = expf(l3);

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
            const float delta = (v[c] - memory) * beta_value;
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

extern "C" __global__ __launch_bounds__(128, 6) void atlas_glm53_kda_prefill_rr_c8_pf(
        float * state, const __nv_bfloat16 * query, const __nv_bfloat16 * key,
        const __nv_bfloat16 * value, const float * log_decay,
        const __nv_bfloat16 * beta, __nv_bfloat16 * output,
        unsigned int batch, unsigned int tokens, unsigned int heads,
        unsigned int key_dim, unsigned int value_dim, float l2_epsilon) {
    glm53_kda_rr_columns_prefetch<8U, true>(state, query, key, value, log_decay, beta, output,
        batch, tokens, heads, key_dim, value_dim, l2_epsilon);
}

// Reduced prefetch (q/k/decay/beta only): fewer live registers, single wave.
extern "C" __global__ __launch_bounds__(128, 6) void atlas_glm53_kda_prefill_rr_c8_pf2(
        float * state, const __nv_bfloat16 * query, const __nv_bfloat16 * key,
        const __nv_bfloat16 * value, const float * log_decay,
        const __nv_bfloat16 * beta, __nv_bfloat16 * output,
        unsigned int batch, unsigned int tokens, unsigned int heads,
        unsigned int key_dim, unsigned int value_dim, float l2_epsilon) {
    glm53_kda_rr_columns_prefetch<8U, false>(state, query, key, value, log_decay, beta, output,
        batch, tokens, heads, key_dim, value_dim, l2_epsilon);
}

// Narrower column groups per warp (bit-identical per column; more warps per head).
#define GLM53_KDA_RR_KERNEL2(NAME, COLUMNS) \
extern "C" __global__ __launch_bounds__(128, 4) void NAME( \
        float * state, const __nv_bfloat16 * query, const __nv_bfloat16 * key, \
        const __nv_bfloat16 * value, const float * log_decay, \
        const __nv_bfloat16 * beta, __nv_bfloat16 * output, \
        unsigned int batch, unsigned int tokens, unsigned int heads, \
        unsigned int key_dim, unsigned int value_dim, float l2_epsilon) { \
    glm53_kda_rr_columns<COLUMNS>(state, query, key, value, log_decay, beta, output, \
        batch, tokens, heads, key_dim, value_dim, l2_epsilon); \
}
GLM53_KDA_RR_KERNEL2(atlas_glm53_kda_prefill_rr_c4, 4)
GLM53_KDA_RR_KERNEL2(atlas_glm53_kda_prefill_rr_c2, 2)
#undef GLM53_KDA_RR_KERNEL2

// Shared-memory staged variant: the block's four warps share one head, so the
// q/k/log-decay/beta rows of GLM53_KDA_RR_STAGE_TOKENS tokens (plus the block's
// 32 value columns) are fetched with cp.async one tile ahead into a double
// buffer, taking the global-load latency off the sequential recurrence without
// the register pressure of the in-register prefetch. Every arithmetic
// expression below is `glm53_kda_rr_columns` verbatim with its global loads
// replaced by the staged copies, so results are bit-identical to `rr_c8`.
#define GLM53_KDA_RR_STAGE_TOKENS 6U
#define GLM53_KDA_RR_BLOCK_COLUMNS (GLM53_KDA_RR_WARPS * 8U)

struct __align__(16) Glm53KdaRrStage {
    __nv_bfloat16 query[GLM53_KDA_RR_STAGE_TOKENS][GLM53_KDA_DIM];
    __nv_bfloat16 key[GLM53_KDA_RR_STAGE_TOKENS][GLM53_KDA_DIM];
    float log_decay[GLM53_KDA_RR_STAGE_TOKENS][GLM53_KDA_DIM];
    __nv_bfloat16 value[GLM53_KDA_RR_STAGE_TOKENS][GLM53_KDA_RR_BLOCK_COLUMNS];
    unsigned int beta_word[GLM53_KDA_RR_STAGE_TOKENS];
};

__device__ __forceinline__ void glm53_kda_rr_cp(void * shared, const void * global, unsigned int bytes) {
    const unsigned address = (unsigned)__cvta_generic_to_shared(shared);
    if (bytes == 16U) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address), "l"(global));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(address), "l"(global));
    }
}

__device__ __forceinline__ void glm53_kda_rr_stage(
        Glm53KdaRrStage * stage, const __nv_bfloat16 * query, const __nv_bfloat16 * key,
        const __nv_bfloat16 * value, const float * log_decay, const __nv_bfloat16 * beta,
        unsigned int sequence, unsigned int tokens, unsigned int head,
        unsigned int first_token, unsigned int count, unsigned int block_column,
        unsigned int thread) {
    // Per token: 16 + 16 + 32 + 4 sixteen-byte chunks, then one beta word.
    const unsigned int chunks = 68U;
    for (unsigned int index = thread; index < count * chunks; index += 128U) {
        const unsigned int t = index / chunks;
        const unsigned int chunk = index % chunks;
        const unsigned long long token_group =
            ((unsigned long long) sequence * tokens + first_token + t) * GLM53_KDA_HEADS + head;
        const unsigned long long vector_base = token_group * GLM53_KDA_DIM;
        if (chunk < 16U) {
            glm53_kda_rr_cp(&stage->query[t][chunk * 8U], query + vector_base + chunk * 8U, 16U);
        } else if (chunk < 32U) {
            const unsigned int c = chunk - 16U;
            glm53_kda_rr_cp(&stage->key[t][c * 8U], key + vector_base + c * 8U, 16U);
        } else if (chunk < 64U) {
            const unsigned int c = chunk - 32U;
            glm53_kda_rr_cp(&stage->log_decay[t][c * 4U], log_decay + vector_base + c * 4U, 16U);
        } else {
            const unsigned int c = chunk - 64U;
            glm53_kda_rr_cp(&stage->value[t][c * 8U],
                            value + vector_base + block_column + c * 8U, 16U);
        }
    }
    if (thread < count) {
        const unsigned long long token_group =
            ((unsigned long long) sequence * tokens + first_token + thread) * GLM53_KDA_HEADS + head;
        glm53_kda_rr_cp(&stage->beta_word[thread],
                        (const unsigned int *)(beta + (token_group & ~1ULL)), 4U);
    }
    asm volatile("cp.async.commit_group;");
}

extern "C" __global__ __launch_bounds__(128, 6) void atlas_glm53_kda_prefill_rr_c8_st(
        float * __restrict__ state,
        const __nv_bfloat16 * __restrict__ query,
        const __nv_bfloat16 * __restrict__ key,
        const __nv_bfloat16 * __restrict__ value,
        const float * __restrict__ log_decay,
        const __nv_bfloat16 * __restrict__ beta,
        __nv_bfloat16 * __restrict__ output,
        unsigned int batch, unsigned int tokens, unsigned int heads,
        unsigned int key_dim, unsigned int value_dim, float l2_epsilon) {
    constexpr unsigned int COLUMNS_PER_WARP = 8U;
    if (batch == 0U || tokens <= 8U || heads != GLM53_KDA_HEADS ||
        key_dim != GLM53_KDA_DIM || value_dim != GLM53_KDA_DIM ||
        l2_epsilon != 1.0e-6f || blockDim.x != 128U) return;
    const unsigned int group = blockIdx.x;
    if (group >= batch * GLM53_KDA_HEADS) return;
    const unsigned int warp = threadIdx.x >> 5U;
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int block_column = blockIdx.y * GLM53_KDA_RR_BLOCK_COLUMNS;
    const unsigned int first_column =
        (blockIdx.y * GLM53_KDA_RR_WARPS + warp) * COLUMNS_PER_WARP;
    if (block_column >= GLM53_KDA_DIM) return;
    const unsigned int sequence = group / GLM53_KDA_HEADS;
    const unsigned int head = group % GLM53_KDA_HEADS;
    const unsigned int r0 = lane;
    const unsigned int r1 = lane + 32U;
    const unsigned int r2 = lane + 64U;
    const unsigned int r3 = lane + 96U;
    const unsigned long long state_base =
        (unsigned long long) group * GLM53_KDA_DIM * GLM53_KDA_DIM;
    __shared__ Glm53KdaRrStage stages[2];

    float s0[COLUMNS_PER_WARP];
    float s1[COLUMNS_PER_WARP];
    float s2[COLUMNS_PER_WARP];
    float s3[COLUMNS_PER_WARP];
    #pragma unroll
    for (unsigned int c = 0U; c < COLUMNS_PER_WARP; ++c) {
        const unsigned int column = first_column + c;
        s0[c] = state[state_base + r0 * GLM53_KDA_DIM + column];
        s1[c] = state[state_base + r1 * GLM53_KDA_DIM + column];
        s2[c] = state[state_base + r2 * GLM53_KDA_DIM + column];
        s3[c] = state[state_base + r3 * GLM53_KDA_DIM + column];
    }

    const unsigned int tiles =
        (tokens + GLM53_KDA_RR_STAGE_TOKENS - 1U) / GLM53_KDA_RR_STAGE_TOKENS;
    glm53_kda_rr_stage(&stages[0], query, key, value, log_decay, beta, sequence, tokens,
                       head, 0U, min(tokens, GLM53_KDA_RR_STAGE_TOKENS), block_column,
                       threadIdx.x);
    for (unsigned int tile = 0U; tile < tiles; ++tile) {
        const unsigned int first_token = tile * GLM53_KDA_RR_STAGE_TOKENS;
        const unsigned int count = min(tokens - first_token, GLM53_KDA_RR_STAGE_TOKENS);
        if (tile + 1U < tiles) {
            const unsigned int next = first_token + GLM53_KDA_RR_STAGE_TOKENS;
            glm53_kda_rr_stage(&stages[(tile + 1U) & 1U], query, key, value, log_decay, beta,
                               sequence, tokens, head, next,
                               min(tokens - next, GLM53_KDA_RR_STAGE_TOKENS), block_column,
                               threadIdx.x);
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        const Glm53KdaRrStage & staged = stages[tile & 1U];
        for (unsigned int t = 0U; t < count; ++t) {
            const unsigned int token = first_token + t;
            const unsigned long long token_group =
                ((unsigned long long) sequence * tokens + token) * GLM53_KDA_HEADS + head;
            const unsigned long long vector_base = token_group * GLM53_KDA_DIM;
            float q0 = __bfloat162float(staged.query[t][r0]);
            float q1 = __bfloat162float(staged.query[t][r1]);
            float q2 = __bfloat162float(staged.query[t][r2]);
            float q3 = __bfloat162float(staged.query[t][r3]);
            float k0 = __bfloat162float(staged.key[t][r0]);
            float k1 = __bfloat162float(staged.key[t][r1]);
            float k2 = __bfloat162float(staged.key[t][r2]);
            float k3 = __bfloat162float(staged.key[t][r3]);
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
            const float d0 = expf(staged.log_decay[t][r0]);
            const float d1 = expf(staged.log_decay[t][r1]);
            const float d2 = expf(staged.log_decay[t][r2]);
            const float d3 = expf(staged.log_decay[t][r3]);
            const unsigned int beta_word = staged.beta_word[t];
            const unsigned short beta_bits = (token_group & 1ULL)
                ? (unsigned short)(beta_word >> 16U) : (unsigned short)(beta_word & 0xffffU);
            const float beta_value = __bfloat162float(__ushort_as_bfloat16(beta_bits));

            #pragma unroll
            for (unsigned int c = 0U; c < COLUMNS_PER_WARP; ++c) {
                const unsigned int column = first_column + c;
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
                    (__bfloat162float(staged.value[t][column - block_column]) - memory) *
                    beta_value;
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
        __syncthreads();
    }

    #pragma unroll
    for (unsigned int c = 0U; c < COLUMNS_PER_WARP; ++c) {
        const unsigned int column = first_column + c;
        state[state_base + r0 * GLM53_KDA_DIM + column] = s0[c];
        state[state_base + r1 * GLM53_KDA_DIM + column] = s1[c];
        state[state_base + r2 * GLM53_KDA_DIM + column] = s2[c];
        state[state_base + r3 * GLM53_KDA_DIM + column] = s3[c];
    }
}
