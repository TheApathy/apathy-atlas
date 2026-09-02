// SPDX-License-Identifier: AGPL-3.0-only
// Transaction-overlay staging for GLM-5.3 DSA rank-512 BF16 latents.

#include <cuda_bf16.h>
#include <stdint.h>

#define GLM53_DSA_APPEND_LATENT 512U
#define GLM53_DSA_APPEND_MAX_QUERIES 8U
#define GLM53_DSA_APPEND_MAX_POSITIONS 1048576U
#define GLM53_DSA_APPEND_THREADS 256U
#define GLM53_DSA_APPEND_MAX_GRID_YZ 65535ULL

extern "C" __global__ void __launch_bounds__(GLM53_DSA_APPEND_THREADS, 1)
atlas_glm53_dsa_latent_append_bf16_stage(
        const __nv_bfloat16 * __restrict__ source,
        __nv_bfloat16 * __restrict__ transaction_overlay,
        unsigned int * __restrict__ published_ends,
        unsigned long long * __restrict__ published_nonces,
        unsigned int batch, unsigned int query_count,
        unsigned int capacity, unsigned int start_position,
        unsigned int end_position, unsigned long long transaction_nonce) {
    const unsigned long long expected_y =
        batch < GLM53_DSA_APPEND_MAX_GRID_YZ
            ? batch : GLM53_DSA_APPEND_MAX_GRID_YZ;
    const unsigned long long expected_z = expected_y == 0ULL ? 0ULL :
        ((unsigned long long)batch + expected_y - 1ULL) / expected_y;
    const unsigned long long computed_end =
        (unsigned long long)start_position + query_count;
    if (source == nullptr || transaction_overlay == nullptr ||
        published_ends == nullptr || published_nonces == nullptr ||
        batch == 0U || query_count == 0U ||
        query_count > GLM53_DSA_APPEND_MAX_QUERIES ||
        capacity == 0U || capacity > GLM53_DSA_APPEND_MAX_POSITIONS ||
        transaction_nonce == 0ULL || computed_end != end_position ||
        computed_end > capacity || blockDim.x != GLM53_DSA_APPEND_THREADS ||
        gridDim.x != 1U || (unsigned long long)gridDim.y != expected_y ||
        (unsigned long long)gridDim.z != expected_z ||
        expected_z > GLM53_DSA_APPEND_MAX_GRID_YZ) {
        return;
    }

    const unsigned long long batch_index =
        (unsigned long long)blockIdx.y +
        (unsigned long long)gridDim.y * blockIdx.z;
    if (batch_index >= batch) {
        return;
    }
    const unsigned int lane = threadIdx.x;
    const unsigned long long elements =
        (unsigned long long)query_count * GLM53_DSA_APPEND_LATENT;
    const unsigned long long base = batch_index * elements;
    for (unsigned long long element = lane; element < elements;
         element += GLM53_DSA_APPEND_THREADS) {
        transaction_overlay[base + element] = source[base + element];
    }
    __syncthreads();

    if (lane == 0U) {
        // Publish metadata only after this batch's entire overlay is durable.
        published_ends[batch_index] = end_position;
        __threadfence();
        atomicExch(published_nonces + batch_index, transaction_nonce);
    }
}
