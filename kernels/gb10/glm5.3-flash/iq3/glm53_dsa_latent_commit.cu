// SPDX-License-Identifier: AGPL-3.0-only
// Globally prevalidated commit of a GLM-5.3 DSA BF16 latent overlay.

#include <cuda_bf16.h>
#include <stdint.h>

#define GLM53_DSA_COMMIT_LATENT 512U
#define GLM53_DSA_COMMIT_MAX_QUERIES 8U
#define GLM53_DSA_COMMIT_MAX_POSITIONS 1048576U
#define GLM53_DSA_COMMIT_THREADS 256U

extern "C" __global__ void __launch_bounds__(GLM53_DSA_COMMIT_THREADS, 1)
atlas_glm53_dsa_latent_commit_bf16(
        const __nv_bfloat16 * __restrict__ transaction_overlay,
        __nv_bfloat16 * __restrict__ persistent_cache,
        unsigned int * __restrict__ published_ends,
        unsigned long long * __restrict__ published_nonces,
        unsigned int * __restrict__ logical_lengths,
        unsigned int batch, unsigned int query_count,
        unsigned int accepted_count, unsigned int capacity,
        unsigned int start_position, unsigned int end_position,
        unsigned long long transaction_nonce) {
    const unsigned long long computed_end =
        (unsigned long long)start_position + query_count;
    if (transaction_overlay == nullptr || persistent_cache == nullptr ||
        published_ends == nullptr || published_nonces == nullptr ||
        logical_lengths == nullptr || batch == 0U || query_count == 0U ||
        query_count > GLM53_DSA_COMMIT_MAX_QUERIES ||
        accepted_count > query_count || capacity == 0U ||
        capacity > GLM53_DSA_COMMIT_MAX_POSITIONS ||
        transaction_nonce == 0ULL || computed_end != end_position ||
        computed_end > capacity || blockDim.x != GLM53_DSA_COMMIT_THREADS ||
        gridDim.x != 1U || gridDim.y != 1U || gridDim.z != 1U) {
        return;
    }

    const unsigned int lane = threadIdx.x;
    __shared__ unsigned int invalid_transaction;
    if (lane == 0U) {
        invalid_transaction = 0U;
    }
    __syncthreads();
    for (unsigned long long batch_index = lane; batch_index < batch;
         batch_index += GLM53_DSA_COMMIT_THREADS) {
        if (published_ends[batch_index] != end_position ||
            published_nonces[batch_index] != transaction_nonce ||
            logical_lengths[batch_index] != start_position) {
            atomicExch(&invalid_transaction, 1U);
        }
    }
    __syncthreads();
    if (invalid_transaction != 0U) {
        return;
    }

    const unsigned long long accepted_elements =
        (unsigned long long)batch * accepted_count * GLM53_DSA_COMMIT_LATENT;
    if (accepted_count != 0U) {
        for (unsigned long long element = lane; element < accepted_elements;
             element += GLM53_DSA_COMMIT_THREADS) {
            const unsigned long long accepted_row =
                element / GLM53_DSA_COMMIT_LATENT;
            const unsigned int channel = element % GLM53_DSA_COMMIT_LATENT;
            const unsigned long long batch_index = accepted_row / accepted_count;
            const unsigned long long query_index =
                accepted_row - batch_index * accepted_count;
            const unsigned long long overlay_index =
                (batch_index * query_count + query_index) *
                    GLM53_DSA_COMMIT_LATENT + channel;
            const unsigned long long cache_index =
                (batch_index * capacity + start_position + query_index) *
                    GLM53_DSA_COMMIT_LATENT + channel;
            persistent_cache[cache_index] = transaction_overlay[overlay_index];
        }
    }
    __threadfence();
    __syncthreads();

    const unsigned int committed_length = start_position + accepted_count;
    for (unsigned long long batch_index = lane; batch_index < batch;
         batch_index += GLM53_DSA_COMMIT_THREADS) {
        logical_lengths[batch_index] = committed_length;
    }
    __threadfence();
    __syncthreads();
    for (unsigned long long batch_index = lane; batch_index < batch;
         batch_index += GLM53_DSA_COMMIT_THREADS) {
        published_ends[batch_index] = 0U;
    }
    __threadfence();
    __syncthreads();
    for (unsigned long long batch_index = lane; batch_index < batch;
         batch_index += GLM53_DSA_COMMIT_THREADS) {
        atomicExch(published_nonces + batch_index, 0ULL);
    }
}
