// SPDX-License-Identifier: AGPL-3.0-only
// Globally prevalidated all-layer GLM-5.3 DSA index-tail commit.

#include <cuda_bf16.h>
#include <stdint.h>

#define GLM53_DSA_COMMIT_LAYERS 11U
#define GLM53_DSA_COMMIT_INDEX_DIM 128U
#define GLM53_DSA_COMMIT_TAIL 3U
#define GLM53_DSA_COMMIT_KPOOL 4U
#define GLM53_DSA_COMMIT_CAPACITY 1048576U
#define GLM53_DSA_COMMIT_THREADS 256U

extern "C" __global__ void __launch_bounds__(GLM53_DSA_COMMIT_THREADS, 1)
atlas_glm53_dsa_index_commit_all11(
        const __nv_bfloat16 * __restrict__ staged_tail_keys,
        const __nv_bfloat16 * __restrict__ staged_tail_gates,
        const unsigned char * __restrict__ staged_tail_validity,
        __nv_bfloat16 * __restrict__ persistent_tail_keys,
        __nv_bfloat16 * __restrict__ persistent_tail_gates,
        unsigned char * __restrict__ persistent_tail_validity,
        const unsigned char * __restrict__ persistent_pool_validity,
        const unsigned int * __restrict__ logical_lengths,
        const unsigned long long * __restrict__ owner_generations,
        unsigned int * __restrict__ latent_ends,
        unsigned long long * __restrict__ latent_nonces,
        unsigned int * __restrict__ index_ends,
        unsigned long long * __restrict__ index_nonces,
        unsigned int * __restrict__ visibility_ends,
        unsigned long long * __restrict__ visibility_nonces,
        const unsigned int * __restrict__ visibility_status,
        unsigned int accepted, unsigned int capacity,
        unsigned int start_position, unsigned int end_position,
        unsigned long long owner_generation,
        unsigned long long transaction_nonce) {
    if (staged_tail_keys == nullptr || staged_tail_gates == nullptr ||
        staged_tail_validity == nullptr || persistent_tail_keys == nullptr ||
        persistent_tail_gates == nullptr || persistent_tail_validity == nullptr ||
        persistent_pool_validity == nullptr || logical_lengths == nullptr ||
        owner_generations == nullptr || latent_ends == nullptr ||
        latent_nonces == nullptr || index_ends == nullptr ||
        index_nonces == nullptr || visibility_ends == nullptr ||
        visibility_nonces == nullptr || visibility_status == nullptr ||
        accepted > 1U || capacity != GLM53_DSA_COMMIT_CAPACITY ||
        start_position >= capacity ||
        (unsigned long long)start_position + 1ULL != end_position ||
        owner_generation == 0ULL || transaction_nonce == 0ULL ||
        blockDim.x != GLM53_DSA_COMMIT_THREADS || blockDim.y != 1U ||
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
    for (unsigned int layer = lane; layer < GLM53_DSA_COMMIT_LAYERS;
         layer += GLM53_DSA_COMMIT_THREADS) {
        const bool bad_receipt =
            logical_lengths[layer] != start_position ||
            owner_generations[layer] != owner_generation ||
            latent_ends[layer] != end_position ||
            latent_nonces[layer] != transaction_nonce ||
            index_ends[layer] != end_position ||
            index_nonces[layer] != transaction_nonce ||
            visibility_ends[layer] != end_position ||
            visibility_nonces[layer] != transaction_nonce ||
            visibility_status[layer] != 0U;
        if (bad_receipt) {
            atomicExch(&invalid_transaction, 1U);
        }
        if (accepted == 1U && start_position % GLM53_DSA_COMMIT_KPOOL == 3U) {
            const unsigned long long pool =
                (unsigned long long)layer *
                    (capacity / GLM53_DSA_COMMIT_KPOOL) +
                start_position / GLM53_DSA_COMMIT_KPOOL;
            if (persistent_pool_validity[pool] != 1U) {
                atomicExch(&invalid_transaction, 1U);
            }
        }
    }
    if (accepted == 1U) {
        const unsigned int final_tail = end_position % GLM53_DSA_COMMIT_KPOOL;
        const unsigned int validity_count =
            GLM53_DSA_COMMIT_LAYERS * GLM53_DSA_COMMIT_TAIL;
        for (unsigned int item = lane; item < validity_count;
             item += GLM53_DSA_COMMIT_THREADS) {
            const unsigned int slot = item % GLM53_DSA_COMMIT_TAIL;
            const unsigned char expected = slot < final_tail ? 1U : 0U;
            if (staged_tail_validity[item] != expected) {
                atomicExch(&invalid_transaction, 1U);
            }
        }
    }
    __syncthreads();
    if (invalid_transaction != 0U) {
        return;
    }

    if (accepted == 1U) {
        const unsigned int tail_elements = GLM53_DSA_COMMIT_LAYERS *
            GLM53_DSA_COMMIT_TAIL * GLM53_DSA_COMMIT_INDEX_DIM;
        for (unsigned int element = lane; element < tail_elements;
             element += GLM53_DSA_COMMIT_THREADS) {
            persistent_tail_keys[element] = staged_tail_keys[element];
            persistent_tail_gates[element] = staged_tail_gates[element];
        }
        const unsigned int validity_count =
            GLM53_DSA_COMMIT_LAYERS * GLM53_DSA_COMMIT_TAIL;
        for (unsigned int item = lane; item < validity_count;
             item += GLM53_DSA_COMMIT_THREADS) {
            persistent_tail_validity[item] = staged_tail_validity[item];
        }
    }
    __threadfence_system();
    __syncthreads();

    for (unsigned int layer = lane; layer < GLM53_DSA_COMMIT_LAYERS;
         layer += GLM53_DSA_COMMIT_THREADS) {
        index_ends[layer] = 0U;
        visibility_ends[layer] = 0U;
        if (accepted == 0U) {
            latent_ends[layer] = 0U;
        }
    }
    __threadfence_system();
    __syncthreads();
    for (unsigned int layer = lane; layer < GLM53_DSA_COMMIT_LAYERS;
         layer += GLM53_DSA_COMMIT_THREADS) {
        atomicExch(index_nonces + layer, 0ULL);
        atomicExch(visibility_nonces + layer, 0ULL);
        if (accepted == 0U) {
            atomicExch(latent_nonces + layer, 0ULL);
        }
    }
}
