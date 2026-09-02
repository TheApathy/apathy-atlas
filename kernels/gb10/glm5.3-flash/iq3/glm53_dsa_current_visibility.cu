// SPDX-License-Identifier: AGPL-3.0-only
// GLM-5.3 B1/T1/all-11 DSA transaction-local current visibility.

#include <cuda_bf16.h>
#include <stdint.h>

#define GLM53_DSA_VIS_LAYERS 11U
#define GLM53_DSA_VIS_LATENT 512U
#define GLM53_DSA_VIS_INDEX 128U
#define GLM53_DSA_VIS_KPOOL 4U
#define GLM53_DSA_VIS_MAX_POSITIONS 1048576U
#define GLM53_DSA_VIS_THREADS 256U

extern "C" __global__ void __launch_bounds__(GLM53_DSA_VIS_THREADS, 1)
atlas_glm53_dsa_current_visibility_bf16(
        const __nv_bfloat16 * __restrict__ latent_overlay,
        const __nv_bfloat16 * __restrict__ pool_overlay,
        const unsigned char * __restrict__ pool_overlay_validity,
        const unsigned int * __restrict__ staged_ends,
        const unsigned long long * __restrict__ staged_generations,
        const unsigned long long * __restrict__ staged_nonces,
        const unsigned int * __restrict__ staged_statuses,
        __nv_bfloat16 * __restrict__ persistent_latent,
        __nv_bfloat16 * __restrict__ persistent_pool,
        unsigned char * __restrict__ persistent_pool_validity,
        unsigned int * __restrict__ ready_ends,
        unsigned long long * __restrict__ ready_generations,
        unsigned int * __restrict__ ready_statuses,
        unsigned long long * __restrict__ ready_nonces,
        unsigned int capacity, unsigned int pool_capacity,
        unsigned int logical_position, unsigned int end_position,
        unsigned long long generation, unsigned long long transaction_nonce) {
    const unsigned long long computed_end =
        (unsigned long long)logical_position + 1ULL;
    const unsigned int computed_pool_capacity =
        capacity / GLM53_DSA_VIS_KPOOL;
    const bool writes_pool =
        logical_position % GLM53_DSA_VIS_KPOOL == GLM53_DSA_VIS_KPOOL - 1U;
    if (latent_overlay == nullptr || staged_ends == nullptr ||
        staged_generations == nullptr || staged_nonces == nullptr ||
        staged_statuses == nullptr || persistent_latent == nullptr ||
        (pool_capacity == 0U &&
         (persistent_pool != nullptr || persistent_pool_validity != nullptr)) ||
        (pool_capacity != 0U &&
         (persistent_pool == nullptr || persistent_pool_validity == nullptr)) ||
        ready_ends == nullptr || ready_generations == nullptr ||
        ready_statuses == nullptr || ready_nonces == nullptr ||
        (writes_pool && (pool_overlay == nullptr ||
                         pool_overlay_validity == nullptr)) ||
        capacity == 0U || capacity > GLM53_DSA_VIS_MAX_POSITIONS ||
        logical_position >= capacity || computed_end != end_position ||
        pool_capacity != computed_pool_capacity || generation == 0ULL ||
        transaction_nonce == 0ULL || blockDim.x != GLM53_DSA_VIS_THREADS ||
        blockDim.y != 1U || blockDim.z != 1U || gridDim.x != 1U ||
        gridDim.y != 1U || gridDim.z != 1U) {
        return;
    }

    const unsigned int lane = threadIdx.x;
    __shared__ unsigned int invalid_transaction;
    if (lane == 0U) {
        invalid_transaction = 0U;
    }
    __syncthreads();
    for (unsigned int layer = lane; layer < GLM53_DSA_VIS_LAYERS;
         layer += GLM53_DSA_VIS_THREADS) {
        if (staged_ends[layer] != end_position ||
            staged_generations[layer] != generation ||
            staged_nonces[layer] != transaction_nonce ||
            staged_statuses[layer] != 0U ||
            (writes_pool && pool_overlay_validity[layer] != 1U)) {
            atomicExch(&invalid_transaction, 1U);
        }
    }
    __syncthreads();
    if (invalid_transaction != 0U) {
        return;
    }

    const unsigned long long latent_items =
        (unsigned long long)GLM53_DSA_VIS_LAYERS * GLM53_DSA_VIS_LATENT;
    for (unsigned long long item = lane; item < latent_items;
         item += GLM53_DSA_VIS_THREADS) {
        const unsigned long long layer = item / GLM53_DSA_VIS_LATENT;
        const unsigned long long feature = item % GLM53_DSA_VIS_LATENT;
        const unsigned long long destination =
            (layer * capacity + logical_position) * GLM53_DSA_VIS_LATENT + feature;
        persistent_latent[destination] = latent_overlay[item];
    }
    if (writes_pool) {
        const unsigned long long pool_items =
            (unsigned long long)GLM53_DSA_VIS_LAYERS * GLM53_DSA_VIS_INDEX;
        const unsigned long long pool_row =
            logical_position / GLM53_DSA_VIS_KPOOL;
        for (unsigned long long item = lane; item < pool_items;
             item += GLM53_DSA_VIS_THREADS) {
            const unsigned long long layer = item / GLM53_DSA_VIS_INDEX;
            const unsigned long long feature = item % GLM53_DSA_VIS_INDEX;
            const unsigned long long destination =
                (layer * pool_capacity + pool_row) * GLM53_DSA_VIS_INDEX + feature;
            persistent_pool[destination] = pool_overlay[item];
        }
        for (unsigned int layer = lane; layer < GLM53_DSA_VIS_LAYERS;
             layer += GLM53_DSA_VIS_THREADS) {
            persistent_pool_validity[
                (unsigned long long)layer * pool_capacity + pool_row] = 1U;
        }
    }
    __syncthreads();
    __threadfence_system();
    __syncthreads();
    for (unsigned int layer = lane; layer < GLM53_DSA_VIS_LAYERS;
         layer += GLM53_DSA_VIS_THREADS) {
        ready_ends[layer] = end_position;
        ready_generations[layer] = generation;
        ready_statuses[layer] = 0U;
    }
    __syncthreads();
    __threadfence_system();
    __syncthreads();
    for (unsigned int layer = lane; layer < GLM53_DSA_VIS_LAYERS;
         layer += GLM53_DSA_VIS_THREADS) {
        atomicExch(ready_nonces + layer, transaction_nonce);
    }
}
