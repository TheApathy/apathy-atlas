// SPDX-License-Identifier: AGPL-3.0-only
// GLM-5.3 KDA strict-F32 gated RMSNorm output boundary.

#include <cuda_bf16.h>
#include <math.h>

#define GLM53_KDA_HEADS 64U
#define GLM53_KDA_HEAD_DIM 128U
#define GLM53_KDA_QKV_DIM (GLM53_KDA_HEADS * GLM53_KDA_HEAD_DIM)
#define GLM53_ELEMENT_THREADS 256U

// One block owns one logit row. The tie comparison is explicit because the
// usual lane-first reduction does not preserve the CPU oracle's first-index
// rule when equal maxima fall in different 1024-stride lanes. The selected
// value is published with the index so the host can reject NaN/Inf-only rows
// after copying a bounded 64-byte receipt rather than all 2.48 MiB of logits.
extern "C" __global__ void __launch_bounds__(1024, 1)
atlas_glm53_argmax_bf16_rows(
        const __nv_bfloat16 * __restrict__ logits,
        unsigned int * __restrict__ out_indices,
        float * __restrict__ out_values,
        unsigned int rows, unsigned int columns) {
    if (rows == 0U || rows > 8U || columns != 154880U ||
        blockDim.x != 1024U || blockIdx.x >= rows) {
        return;
    }
    const unsigned int row = blockIdx.x;
    const unsigned long long base = (unsigned long long) row * columns;
    float best = -INFINITY;
    unsigned int best_index = 0U;
    for (unsigned int index = threadIdx.x; index < columns; index += blockDim.x) {
        const float candidate = __bfloat162float(logits[base + index]);
        if (candidate > best || (candidate == best && index < best_index)) {
            best = candidate;
            best_index = index;
        }
    }
    __shared__ float shared_values[1024];
    __shared__ unsigned int shared_indices[1024];
    shared_values[threadIdx.x] = best;
    shared_indices[threadIdx.x] = best_index;
    __syncthreads();
    for (unsigned int stride = 512U; stride != 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            const float candidate = shared_values[threadIdx.x + stride];
            const unsigned int candidate_index = shared_indices[threadIdx.x + stride];
            best = shared_values[threadIdx.x];
            best_index = shared_indices[threadIdx.x];
            if (candidate > best ||
                (candidate == best && candidate_index < best_index)) {
                shared_values[threadIdx.x] = candidate;
                shared_indices[threadIdx.x] = candidate_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        out_indices[row] = shared_indices[0];
        out_values[row] = shared_values[0];
    }
}

// EXL3's combined projection is row-major [tokens, 3, qkv_dim]. The conv and
// recurrence contracts use three dense [tokens, qkv_dim] planes, so split the
// row-major result explicitly for a multi-token verifier chunk.
extern "C" __global__ void __launch_bounds__(GLM53_ELEMENT_THREADS, 1)
atlas_glm53_kda_split_qkv(
        const __nv_bfloat16 * __restrict__ combined,
        __nv_bfloat16 * __restrict__ query,
        __nv_bfloat16 * __restrict__ key,
        __nv_bfloat16 * __restrict__ value,
        unsigned int tokens) {
    if (tokens == 0U || tokens > 2048U) return;
    const unsigned long long values =
        (unsigned long long) tokens * GLM53_KDA_QKV_DIM;
    unsigned long long index =
        (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long) gridDim.x * blockDim.x;
    for (; index < values; index += stride) {
        const unsigned long long token = index / GLM53_KDA_QKV_DIM;
        const unsigned long long column = index % GLM53_KDA_QKV_DIM;
        const unsigned long long base = token * 3ULL * GLM53_KDA_QKV_DIM + column;
        query[index] = combined[base];
        key[index] = combined[base + GLM53_KDA_QKV_DIM];
        value[index] = combined[base + 2ULL * GLM53_KDA_QKV_DIM];
    }
}

extern "C" __global__ void __launch_bounds__(GLM53_ELEMENT_THREADS, 1)
atlas_glm53_kda_forget_gate(
        const __nv_bfloat16 * __restrict__ projected,
        const float * __restrict__ dt_bias,
        const float * __restrict__ a_log,
        float * __restrict__ output,
        unsigned int tokens, unsigned int heads,
        unsigned int head_dim, float lower_bound) {
    if (tokens == 0U || heads != GLM53_KDA_HEADS ||
        head_dim != GLM53_KDA_HEAD_DIM || lower_bound != -5.0f) {
        return;
    }
    const unsigned long long values =
        (unsigned long long) tokens * GLM53_KDA_QKV_DIM;
    unsigned long long index =
        (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long) gridDim.x * blockDim.x;
    for (; index < values; index += stride) {
        const unsigned int column = (unsigned int) (index % GLM53_KDA_QKV_DIM);
        const unsigned int head = column / GLM53_KDA_HEAD_DIM;
        const float g = __bfloat162float(projected[index]) + dt_bias[column];
        // `ssm_a` holds -exp(A_log), folded at conversion time (the
        // kimi-linear/kimi-k3 convention the checkpoint follows), so
        // exp(A_log) == -ssm_a. Calling expf() here exponentiates a value that
        // was already exponentiated: with ssm_a averaging about -4.94 that is
        // exp(-4.94)=0.007 in place of 4.94, a ~700x error that pins every
        // sigmoid near 0.5 and flattens the output distribution.
        const float scaled = -a_log[head] * g;
        const float sigmoid = 1.0f / (1.0f + expf(-scaled));
        output[index] = lower_bound * sigmoid;
    }
}

extern "C" __global__ void __launch_bounds__(GLM53_ELEMENT_THREADS, 1)
atlas_glm53_kda_beta_sigmoid(
        const __nv_bfloat16 * __restrict__ projected,
        __nv_bfloat16 * __restrict__ output,
        unsigned int tokens, unsigned int heads) {
    if (tokens == 0U || heads != GLM53_KDA_HEADS) {
        return;
    }
    const unsigned long long values =
        (unsigned long long) tokens * GLM53_KDA_HEADS;
    unsigned long long index =
        (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long) gridDim.x * blockDim.x;
    for (; index < values; index += stride) {
        const float value = __bfloat162float(projected[index]);
        output[index] = __float2bfloat16_rn(
            1.0f / (1.0f + expf(-value)));
    }
}

extern "C" __global__ void __launch_bounds__(GLM53_KDA_HEAD_DIM, 1)
atlas_glm53_kda_decode(
        float * __restrict__ state,
        const __nv_bfloat16 * __restrict__ query,
        const __nv_bfloat16 * __restrict__ key,
        const __nv_bfloat16 * __restrict__ value,
        const float * __restrict__ log_decay,
        const __nv_bfloat16 * __restrict__ beta,
        __nv_bfloat16 * __restrict__ output,
        unsigned int batch, unsigned int heads,
        unsigned int key_dim, unsigned int value_dim, float l2_epsilon) {
    if (batch == 0U || heads != GLM53_KDA_HEADS ||
        key_dim != GLM53_KDA_HEAD_DIM || value_dim != GLM53_KDA_HEAD_DIM ||
        l2_epsilon != 1.0e-6f) {
        return;
    }
    const unsigned int group = blockIdx.x;
    if (group >= batch * GLM53_KDA_HEADS) {
        return;
    }
    const unsigned int column = threadIdx.x;
    const unsigned long long vector_base =
        (unsigned long long) group * GLM53_KDA_HEAD_DIM;
    const unsigned long long state_base =
        (unsigned long long) group * GLM53_KDA_HEAD_DIM * GLM53_KDA_HEAD_DIM;

    __shared__ float q_values[GLM53_KDA_HEAD_DIM];
    __shared__ float k_values[GLM53_KDA_HEAD_DIM];
    __shared__ float decay[GLM53_KDA_HEAD_DIM];
    __shared__ float q_squares[GLM53_KDA_HEAD_DIM];
    __shared__ float k_squares[GLM53_KDA_HEAD_DIM];
    __shared__ float q_norm;
    __shared__ float k_norm;

    const float q = __bfloat162float(query[vector_base + column]);
    const float k = __bfloat162float(key[vector_base + column]);
    q_values[column] = q;
    k_values[column] = k;
    decay[column] = expf(log_decay[vector_base + column]);
    q_squares[column] = q * q;
    k_squares[column] = k * k;
    __syncthreads();
    for (unsigned int stride = GLM53_KDA_HEAD_DIM / 2U;
         stride > 0U; stride >>= 1U) {
        if (column < stride) {
            q_squares[column] += q_squares[column + stride];
            k_squares[column] += k_squares[column + stride];
        }
        __syncthreads();
    }
    if (column == 0U) {
        q_norm = sqrtf(q_squares[0] + l2_epsilon);
        k_norm = sqrtf(k_squares[0] + l2_epsilon);
    }
    __syncthreads();
    q_values[column] =
        (q_values[column] / q_norm) * (1.0f / sqrtf(128.0f));
    k_values[column] = k_values[column] / k_norm;
    __syncthreads();

    float memory = 0.0f;
    #pragma unroll 4
    for (unsigned int row = 0; row < GLM53_KDA_HEAD_DIM; row += 4U) {
        #pragma unroll
        for (unsigned int inner = 0; inner < 4U; ++inner) {
            const unsigned int at = row + inner;
            const unsigned long long index =
                state_base + (unsigned long long) at * GLM53_KDA_HEAD_DIM + column;
            const float decayed = state[index] * decay[at];
            state[index] = decayed;
            memory += decayed * k_values[at];
        }
    }
    const float beta_value = __bfloat162float(beta[group]);
    const float delta =
        (__bfloat162float(value[vector_base + column]) - memory) * beta_value;
    float result = 0.0f;
    #pragma unroll 4
    for (unsigned int row = 0; row < GLM53_KDA_HEAD_DIM; row += 4U) {
        #pragma unroll
        for (unsigned int inner = 0; inner < 4U; ++inner) {
            const unsigned int at = row + inner;
            const unsigned long long index =
                state_base + (unsigned long long) at * GLM53_KDA_HEAD_DIM + column;
            const float updated = state[index] + k_values[at] * delta;
            state[index] = updated;
            result += updated * q_values[at];
        }
    }
    output[vector_base + column] = __float2bfloat16_rn(result);
}

extern "C" __global__ void __launch_bounds__(GLM53_KDA_HEAD_DIM, 1)
atlas_glm53_kda_gated_rms_norm(
        const __nv_bfloat16 * __restrict__ input,
        const float * __restrict__ weight,
        const __nv_bfloat16 * __restrict__ gate,
        __nv_bfloat16 * __restrict__ output,
        unsigned int tokens, unsigned int heads,
        unsigned int head_dim, float epsilon) {
    if (tokens == 0U || heads != GLM53_KDA_HEADS ||
        head_dim != GLM53_KDA_HEAD_DIM || epsilon != 1.0e-5f) {
        return;
    }
    const unsigned int group = blockIdx.x;
    if (group >= tokens * GLM53_KDA_HEADS) {
        return;
    }
    const unsigned int column = threadIdx.x;
    const unsigned long long index =
        (unsigned long long) group * GLM53_KDA_HEAD_DIM + column;
    const float value = __bfloat162float(input[index]);

    __shared__ float squares[GLM53_KDA_HEAD_DIM];
    __shared__ float inverse_rms;
    squares[column] = value * value;
    __syncthreads();
    for (unsigned int stride = GLM53_KDA_HEAD_DIM / 2U;
         stride > 0U; stride >>= 1U) {
        if (column < stride) {
            squares[column] += squares[column + stride];
        }
        __syncthreads();
    }
    if (column == 0U) {
        inverse_rms = rsqrtf(
            squares[0] / (float) GLM53_KDA_HEAD_DIM + epsilon);
    }
    __syncthreads();

    const float normalized = value * inverse_rms;
    const float weighted = weight[column] * normalized;
    const float gate_value = __bfloat162float(gate[index]);
    const float sigmoid_gate = 1.0f / (1.0f + expf(-gate_value));
    output[index] = __float2bfloat16_rn(weighted * sigmoid_gate);
}

extern "C" __global__ void __launch_bounds__(GLM53_KDA_HEAD_DIM, 1)
atlas_glm53_kda_prefill(
        float * __restrict__ state,
        const __nv_bfloat16 * __restrict__ query,
        const __nv_bfloat16 * __restrict__ key,
        const __nv_bfloat16 * __restrict__ value,
        const float * __restrict__ log_decay,
        const __nv_bfloat16 * __restrict__ beta,
        __nv_bfloat16 * __restrict__ output,
        unsigned int batch, unsigned int tokens, unsigned int heads,
        unsigned int key_dim, unsigned int value_dim, float l2_epsilon) {
    if (batch == 0U || tokens == 0U || heads != GLM53_KDA_HEADS ||
        key_dim != GLM53_KDA_HEAD_DIM || value_dim != GLM53_KDA_HEAD_DIM ||
        l2_epsilon != 1.0e-6f) {
        return;
    }
    const unsigned int group = blockIdx.x;
    if (group >= batch * GLM53_KDA_HEADS) {
        return;
    }
    const unsigned int sequence = group / GLM53_KDA_HEADS;
    const unsigned int head = group % GLM53_KDA_HEADS;
    const unsigned int column = threadIdx.x;
    const unsigned long long state_base =
        (unsigned long long) group * GLM53_KDA_HEAD_DIM * GLM53_KDA_HEAD_DIM;

    __shared__ float q_values[GLM53_KDA_HEAD_DIM];
    __shared__ float k_values[GLM53_KDA_HEAD_DIM];
    __shared__ float decay[GLM53_KDA_HEAD_DIM];
    __shared__ float q_squares[GLM53_KDA_HEAD_DIM];
    __shared__ float k_squares[GLM53_KDA_HEAD_DIM];
    __shared__ float q_norm;
    __shared__ float k_norm;

    for (unsigned int token = 0U; token < tokens; ++token) {
        const unsigned long long token_group =
            ((unsigned long long) sequence * tokens + token) *
                GLM53_KDA_HEADS + head;
        const unsigned long long vector_base =
            token_group * GLM53_KDA_HEAD_DIM;
        const float q = __bfloat162float(query[vector_base + column]);
        const float k = __bfloat162float(key[vector_base + column]);
        q_values[column] = q;
        k_values[column] = k;
        decay[column] = expf(log_decay[vector_base + column]);
        q_squares[column] = q * q;
        k_squares[column] = k * k;
        __syncthreads();
        for (unsigned int stride = GLM53_KDA_HEAD_DIM / 2U;
             stride > 0U; stride >>= 1U) {
            if (column < stride) {
                q_squares[column] += q_squares[column + stride];
                k_squares[column] += k_squares[column + stride];
            }
            __syncthreads();
        }
        if (column == 0U) {
            q_norm = sqrtf(q_squares[0] + l2_epsilon);
            k_norm = sqrtf(k_squares[0] + l2_epsilon);
        }
        __syncthreads();
        q_values[column] =
            (q_values[column] / q_norm) * (1.0f / sqrtf(128.0f));
        k_values[column] = k_values[column] / k_norm;
        __syncthreads();

        float memory = 0.0f;
        #pragma unroll 4
        for (unsigned int row = 0; row < GLM53_KDA_HEAD_DIM; row += 4U) {
            #pragma unroll
            for (unsigned int inner = 0; inner < 4U; ++inner) {
                const unsigned int at = row + inner;
                const unsigned long long index = state_base +
                    (unsigned long long) at * GLM53_KDA_HEAD_DIM + column;
                const float decayed = state[index] * decay[at];
                state[index] = decayed;
                memory += decayed * k_values[at];
            }
        }
        const float beta_value = __bfloat162float(beta[token_group]);
        const float delta =
            (__bfloat162float(value[vector_base + column]) - memory) * beta_value;
        float result = 0.0f;
        #pragma unroll 4
        for (unsigned int row = 0; row < GLM53_KDA_HEAD_DIM; row += 4U) {
            #pragma unroll
            for (unsigned int inner = 0; inner < 4U; ++inner) {
                const unsigned int at = row + inner;
                const unsigned long long index = state_base +
                    (unsigned long long) at * GLM53_KDA_HEAD_DIM + column;
                const float updated = state[index] + k_values[at] * delta;
                state[index] = updated;
                result += updated * q_values[at];
            }
        }
        output[vector_base + column] = __float2bfloat16_rn(result);
        __syncthreads();
    }
}

// Large-M register-resident recurrence. One warp owns one value column; each
// lane holds four key rows for that column across the complete token loop.
// This removes all per-token state traffic and block barriers while retaining
// vector-valued decay and the original q/k normalization tree. The memory and
// output dot products use a warp tree rather than the scalar row order, so this
// is admitted only by the explicit layer-major prompt path.
#define GLM53_KDA_RR_WARPS 4U
extern "C" __global__ void __launch_bounds__(128, 4)
atlas_glm53_kda_prefill_register_resident(
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
        key_dim != GLM53_KDA_HEAD_DIM || value_dim != GLM53_KDA_HEAD_DIM ||
        l2_epsilon != 1.0e-6f || blockDim.x != 128U) {
        return;
    }
    const unsigned int group = blockIdx.x;
    if (group >= batch * GLM53_KDA_HEADS) {
        return;
    }
    const unsigned int warp = threadIdx.x >> 5U;
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int column = blockIdx.y * GLM53_KDA_RR_WARPS + warp;
    if (column >= GLM53_KDA_HEAD_DIM) {
        return;
    }
    const unsigned int sequence = group / GLM53_KDA_HEADS;
    const unsigned int head = group % GLM53_KDA_HEADS;
    const unsigned int r0 = lane;
    const unsigned int r1 = lane + 32U;
    const unsigned int r2 = lane + 64U;
    const unsigned int r3 = lane + 96U;
    const unsigned long long state_base =
        (unsigned long long) group * GLM53_KDA_HEAD_DIM * GLM53_KDA_HEAD_DIM;

    float s0 = state[state_base + r0 * GLM53_KDA_HEAD_DIM + column];
    float s1 = state[state_base + r1 * GLM53_KDA_HEAD_DIM + column];
    float s2 = state[state_base + r2 * GLM53_KDA_HEAD_DIM + column];
    float s3 = state[state_base + r3 * GLM53_KDA_HEAD_DIM + column];

    for (unsigned int token = 0U; token < tokens; ++token) {
        const unsigned long long token_group =
            ((unsigned long long) sequence * tokens + token) *
                GLM53_KDA_HEADS + head;
        const unsigned long long vector_base =
            token_group * GLM53_KDA_HEAD_DIM;

        float q0 = __bfloat162float(query[vector_base + r0]);
        float q1 = __bfloat162float(query[vector_base + r1]);
        float q2 = __bfloat162float(query[vector_base + r2]);
        float q3 = __bfloat162float(query[vector_base + r3]);
        float k0 = __bfloat162float(key[vector_base + r0]);
        float k1 = __bfloat162float(key[vector_base + r1]);
        float k2 = __bfloat162float(key[vector_base + r2]);
        float k3 = __bfloat162float(key[vector_base + r3]);

        // Match the original 128-thread tree: stride64 pairs r0/r2 and r1/r3,
        // stride32 joins those pairs, then the warp completes strides16..1.
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
        s0 *= d0;
        s1 *= d1;
        s2 *= d2;
        s3 *= d3;

        float memory = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            memory += __shfl_down_sync(0xffffffffU, memory, offset);
        }
        memory = __shfl_sync(0xffffffffU, memory, 0);
        const float beta_value = __bfloat162float(beta[token_group]);
        const float delta =
            (__bfloat162float(value[vector_base + column]) - memory) * beta_value;
        s0 += k0 * delta;
        s1 += k1 * delta;
        s2 += k2 * delta;
        s3 += k3 * delta;

        float result = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            result += __shfl_down_sync(0xffffffffU, result, offset);
        }
        if (lane == 0U) {
            output[vector_base + column] = __float2bfloat16_rn(result);
        }
    }

    state[state_base + r0 * GLM53_KDA_HEAD_DIM + column] = s0;
    state[state_base + r1 * GLM53_KDA_HEAD_DIM + column] = s1;
    state[state_base + r2 * GLM53_KDA_HEAD_DIM + column] = s2;
    state[state_base + r3 * GLM53_KDA_HEAD_DIM + column] = s3;
}
