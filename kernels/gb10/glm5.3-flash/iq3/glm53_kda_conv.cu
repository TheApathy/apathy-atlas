// SPDX-License-Identifier: AGPL-3.0-only
// GLM-5.3 three-stream F32-state/F32-weight causal convolution transaction.

#include <cuda_bf16.h>
#include <stdint.h>

#define GLM53_KDA_CONV_STREAMS 3U
#define GLM53_KDA_CONV_CHANNELS 8192U
#define GLM53_KDA_CONV_KERNEL 4U
#define GLM53_KDA_CONV_MAX_QUERIES 65520U
#define GLM53_KDA_CONV_MAX_POSITIONS 1048576U
#define GLM53_KDA_CONV_MAX_BATCH 65535U
#define GLM53_KDA_CONV_THREADS 256U

extern "C" __global__ void __launch_bounds__(GLM53_KDA_CONV_THREADS, 1)
atlas_glm53_kda_conv_f32_stage(
        const __nv_bfloat16 * __restrict__ q_input,
        const __nv_bfloat16 * __restrict__ k_input,
        const __nv_bfloat16 * __restrict__ v_input,
        const float * __restrict__ q_weight,
        const float * __restrict__ k_weight,
        const float * __restrict__ v_weight,
        const float * __restrict__ persistent_state,
        float * __restrict__ staged_state,
        __nv_bfloat16 * __restrict__ q_output,
        __nv_bfloat16 * __restrict__ k_output,
        __nv_bfloat16 * __restrict__ v_output,
        unsigned int batch, unsigned int query_count) {
    if (q_input == nullptr || k_input == nullptr || v_input == nullptr ||
        q_weight == nullptr || k_weight == nullptr || v_weight == nullptr ||
        persistent_state == nullptr || staged_state == nullptr ||
        q_output == nullptr || k_output == nullptr || v_output == nullptr ||
        batch == 0U || batch > GLM53_KDA_CONV_MAX_BATCH ||
        query_count == 0U || query_count > GLM53_KDA_CONV_MAX_QUERIES ||
        blockDim.x != GLM53_KDA_CONV_THREADS || blockDim.y != 1U ||
        blockDim.z != 1U || gridDim.x != 32U || gridDim.y != batch ||
        gridDim.z != GLM53_KDA_CONV_STREAMS) {
        return;
    }

    const unsigned int stream_index = blockIdx.z;
    const unsigned int batch_index = blockIdx.y;
    const unsigned int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= GLM53_KDA_CONV_CHANNELS) {
        return;
    }
    const __nv_bfloat16 *input = stream_index == 0U ? q_input :
        (stream_index == 1U ? k_input : v_input);
    const float *weight = stream_index == 0U ? q_weight :
        (stream_index == 1U ? k_weight : v_weight);
    __nv_bfloat16 *output = stream_index == 0U ? q_output :
        (stream_index == 1U ? k_output : v_output);
    const unsigned long long state_base =
        (((unsigned long long)batch_index * GLM53_KDA_CONV_STREAMS +
          stream_index) * GLM53_KDA_CONV_CHANNELS + channel) *
        GLM53_KDA_CONV_KERNEL;
    float s0 = persistent_state[state_base + 0ULL];
    float s1 = persistent_state[state_base + 1ULL];
    float s2 = persistent_state[state_base + 2ULL];
    float s3 = persistent_state[state_base + 3ULL];
    const unsigned long long weight_base =
        (unsigned long long)channel * GLM53_KDA_CONV_KERNEL;
    const float w0 = weight[weight_base + 0ULL];
    const float w1 = weight[weight_base + 1ULL];
    const float w2 = weight[weight_base + 2ULL];
    const float w3 = weight[weight_base + 3ULL];

    for (unsigned int token = 0U; token < query_count; ++token) {
        const unsigned long long token_index =
            ((unsigned long long)batch_index * query_count + token) *
                GLM53_KDA_CONV_CHANNELS + channel;
        const float newest = (float)input[token_index];
        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = newest;
        const float correlation =
            s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3;
        const float activated = correlation /
            (1.0f + __expf(-correlation));
        output[token_index] = __float2bfloat16(activated);
    }
    staged_state[state_base + 0ULL] = s0;
    staged_state[state_base + 1ULL] = s1;
    staged_state[state_base + 2ULL] = s2;
    staged_state[state_base + 3ULL] = s3;
}

extern "C" __global__ void __launch_bounds__(GLM53_KDA_CONV_THREADS, 1)
atlas_glm53_kda_conv_finalize(
        unsigned int * __restrict__ published_ends,
        unsigned long long * __restrict__ published_nonces,
        const unsigned int * __restrict__ logical_lengths,
        unsigned int batch, unsigned int query_count,
        unsigned int capacity, unsigned int start_position,
        unsigned int end_position, unsigned long long transaction_nonce) {
    const unsigned long long computed_end =
        (unsigned long long)start_position + query_count;
    if (published_ends == nullptr || published_nonces == nullptr ||
        logical_lengths == nullptr || batch == 0U ||
        batch > GLM53_KDA_CONV_MAX_BATCH || query_count == 0U ||
        query_count > GLM53_KDA_CONV_MAX_QUERIES || capacity == 0U ||
        capacity > GLM53_KDA_CONV_MAX_POSITIONS || transaction_nonce == 0ULL ||
        computed_end != end_position || computed_end > capacity ||
        blockDim.x != GLM53_KDA_CONV_THREADS || blockDim.y != 1U ||
        blockDim.z != 1U || gridDim.x != 1U || gridDim.y != 1U ||
        gridDim.z != 1U) {
        return;
    }
    const unsigned int lane = threadIdx.x;
    __shared__ unsigned int invalid_length;
    if (lane == 0U) {
        invalid_length = 0U;
    }
    __syncthreads();
    for (unsigned long long b = lane; b < batch; b += GLM53_KDA_CONV_THREADS) {
        if (logical_lengths[b] != start_position) {
            atomicExch(&invalid_length, 1U);
        }
    }
    __syncthreads();
    if (invalid_length != 0U) {
        return;
    }
    for (unsigned long long b = lane; b < batch; b += GLM53_KDA_CONV_THREADS) {
        published_ends[b] = end_position;
    }
    __syncthreads();
    __threadfence();
    __syncthreads();
    for (unsigned long long b = lane; b < batch; b += GLM53_KDA_CONV_THREADS) {
        atomicExch(published_nonces + b, transaction_nonce);
    }
}

extern "C" __global__ void __launch_bounds__(GLM53_KDA_CONV_THREADS, 1)
atlas_glm53_kda_conv_commit(
        const __nv_bfloat16 * __restrict__ q_input,
        const __nv_bfloat16 * __restrict__ k_input,
        const __nv_bfloat16 * __restrict__ v_input,
        const float * __restrict__ staged_state,
        float * __restrict__ persistent_state,
        unsigned int * __restrict__ published_ends,
        unsigned long long * __restrict__ published_nonces,
        unsigned int * __restrict__ logical_lengths,
        unsigned int batch, unsigned int query_count,
        unsigned int accepted_count,
        unsigned int capacity, unsigned int start_position,
        unsigned int end_position, unsigned long long transaction_nonce) {
    const unsigned long long computed_end =
        (unsigned long long)start_position + query_count;
    if (q_input == nullptr || k_input == nullptr || v_input == nullptr ||
        staged_state == nullptr || persistent_state == nullptr ||
        published_ends == nullptr || published_nonces == nullptr ||
        logical_lengths == nullptr || batch == 0U ||
        batch > GLM53_KDA_CONV_MAX_BATCH || query_count == 0U ||
        query_count > GLM53_KDA_CONV_MAX_QUERIES ||
        accepted_count > query_count || capacity == 0U ||
        capacity > GLM53_KDA_CONV_MAX_POSITIONS || transaction_nonce == 0ULL ||
        computed_end != end_position || computed_end > capacity ||
        blockDim.x != GLM53_KDA_CONV_THREADS || blockDim.y != 1U ||
        blockDim.z != 1U || gridDim.x != 1U || gridDim.y != 1U ||
        gridDim.z != 1U) {
        return;
    }
    const unsigned int lane = threadIdx.x;
    __shared__ unsigned int invalid_transaction;
    if (lane == 0U) {
        invalid_transaction = 0U;
    }
    __syncthreads();
    for (unsigned long long b = lane; b < batch; b += GLM53_KDA_CONV_THREADS) {
        if (published_ends[b] != end_position ||
            published_nonces[b] != transaction_nonce ||
            logical_lengths[b] != start_position) {
            atomicExch(&invalid_transaction, 1U);
        }
    }
    __syncthreads();
    if (invalid_transaction != 0U) {
        return;
    }

    const unsigned long long state_elements =
        (unsigned long long)batch * GLM53_KDA_CONV_STREAMS *
        GLM53_KDA_CONV_CHANNELS * GLM53_KDA_CONV_KERNEL;
    if (accepted_count == query_count) {
        for (unsigned long long element = lane; element < state_elements;
             element += GLM53_KDA_CONV_THREADS) {
            persistent_state[element] = staged_state[element];
        }
    } else if (accepted_count != 0U) {
        const unsigned long long state_channels =
            state_elements / GLM53_KDA_CONV_KERNEL;
        for (unsigned long long item = lane; item < state_channels;
             item += GLM53_KDA_CONV_THREADS) {
            const unsigned int channel = item % GLM53_KDA_CONV_CHANNELS;
            const unsigned long long stream_batch =
                item / GLM53_KDA_CONV_CHANNELS;
            const unsigned int stream_index =
                stream_batch % GLM53_KDA_CONV_STREAMS;
            const unsigned long long batch_index =
                stream_batch / GLM53_KDA_CONV_STREAMS;
            const __nv_bfloat16 *input = stream_index == 0U ? q_input :
                (stream_index == 1U ? k_input : v_input);
            const unsigned long long state_base =
                item * GLM53_KDA_CONV_KERNEL;
            float s0 = persistent_state[state_base + 0ULL];
            float s1 = persistent_state[state_base + 1ULL];
            float s2 = persistent_state[state_base + 2ULL];
            float s3 = persistent_state[state_base + 3ULL];
            for (unsigned int token = 0U; token < accepted_count; ++token) {
                const unsigned long long input_index =
                    (batch_index * query_count + token) *
                        GLM53_KDA_CONV_CHANNELS + channel;
                const float newest = (float)input[input_index];
                s0 = s1;
                s1 = s2;
                s2 = s3;
                s3 = newest;
            }
            persistent_state[state_base + 0ULL] = s0;
            persistent_state[state_base + 1ULL] = s1;
            persistent_state[state_base + 2ULL] = s2;
            persistent_state[state_base + 3ULL] = s3;
        }
    }
    __syncthreads();
    __threadfence();
    __syncthreads();
    for (unsigned long long b = lane; b < batch; b += GLM53_KDA_CONV_THREADS) {
        logical_lengths[b] = start_position + accepted_count;
    }
    __syncthreads();
    __threadfence();
    __syncthreads();
    for (unsigned long long b = lane; b < batch; b += GLM53_KDA_CONV_THREADS) {
        published_ends[b] = 0U;
    }
    __syncthreads();
    __threadfence();
    __syncthreads();
    for (unsigned long long b = lane; b < batch; b += GLM53_KDA_CONV_THREADS) {
        atomicExch(published_nonces + b, 0ULL);
    }
}
