// SPDX-License-Identifier: AGPL-3.0-only
// GLM-5.3 KDA strict-F32 gated RMSNorm output boundary.

#include <cuda_bf16.h>
#include <math.h>

#define GLM53_KDA_HEADS 64U
#define GLM53_KDA_HEAD_DIM 128U
#define GLM53_KDA_QKV_DIM (GLM53_KDA_HEADS * GLM53_KDA_HEAD_DIM)
#define GLM53_ELEMENT_THREADS 256U

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
