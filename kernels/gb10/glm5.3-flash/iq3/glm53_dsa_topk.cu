// SPDX-License-Identifier: AGPL-3.0-only
// GLM-5.3 DSA deterministic kpool4 selector and raw-token expander.

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define GLM53_DSA_INDEX_TOPK 2048U
#define GLM53_DSA_KPOOL 4U
#define GLM53_DSA_SELECTED_POOLS 512U
#define GLM53_DSA_OUTPUT_WIDTH 2051U
#define GLM53_DSA_MAX_QUERIES 8192U
#define GLM53_DSA_MAX_POOLS 262144U
#define GLM53_DSA_MAX_POSITIONS 1048576U
#define GLM53_DSA_THREADS 256U
#define GLM53_DSA_NETWORK 1024U
#define GLM53_DSA_MAX_GRID_YZ 65535ULL
#define GLM53_DSA_INVALID_POOL 0xffffffffU

// Stream-ordered replacement for four tiny synchronous H2D metadata copies.
// One launch prepares the fixed K<=2048 query geometry consumed by top-k and
// selected attention. Keeping this on device also makes the path capturable.
extern "C" __global__ void atlas_glm53_dsa_prepare_metadata(
        unsigned int * __restrict__ sequence_length,
        unsigned int * __restrict__ query_positions,
        unsigned char * __restrict__ query_validity,
        unsigned char * __restrict__ tail_validity,
        unsigned int position, unsigned int rows) {
    const unsigned int lane = threadIdx.x;
    if (blockIdx.x != 0U || rows == 0U || rows > GLM53_DSA_MAX_QUERIES)
        return;
    if (lane == 0U)
        sequence_length[0] = position + rows;
    for (unsigned int row = lane; row < rows; row += blockDim.x) {
        query_positions[row] = position + row;
        query_validity[row] = 1U;
    }
    if (rows > 1U && lane < GLM53_DSA_KPOOL - 1U)
        tail_validity[lane] = 1U;
}

__device__ __forceinline__ bool glm53_dsa_better(
        float left_score, unsigned int left_pool,
        float right_score, unsigned int right_pool) {
    if (left_pool == GLM53_DSA_INVALID_POOL ||
        right_pool == GLM53_DSA_INVALID_POOL) {
        return left_pool != GLM53_DSA_INVALID_POOL;
    }
    const bool left_nan = isnan(left_score);
    const bool right_nan = isnan(right_score);
    if (left_nan != right_nan) {
        return left_nan;
    }
    if (left_score > right_score) {
        return true;
    }
    if (left_score < right_score) {
        return false;
    }
    return left_pool < right_pool;
}

extern "C" __global__ void __launch_bounds__(GLM53_DSA_THREADS, 1)
atlas_glm53_dsa_topk_k4(
        const float * __restrict__ scores,
        const unsigned char * __restrict__ pool_validity,
        const unsigned int * __restrict__ sequence_lengths,
        const unsigned int * __restrict__ query_positions,
        const unsigned char * __restrict__ query_validity,
        const unsigned char * __restrict__ tail_validity,
        int * __restrict__ output_indices,
        unsigned int batch, unsigned int query_count,
        unsigned int pool_count, unsigned int kv_capacity) {
    const unsigned long long rows =
        (unsigned long long) batch * query_count;
    const unsigned long long expected_y =
        rows < GLM53_DSA_MAX_GRID_YZ ? rows : GLM53_DSA_MAX_GRID_YZ;
    const unsigned long long expected_z = expected_y == 0ULL ? 0ULL :
        (rows + expected_y - 1ULL) / expected_y;
    if (scores == nullptr || pool_validity == nullptr ||
        sequence_lengths == nullptr || query_positions == nullptr ||
        query_validity == nullptr || tail_validity == nullptr ||
        output_indices == nullptr ||
        batch == 0U || query_count == 0U ||
        query_count > GLM53_DSA_MAX_QUERIES || pool_count == 0U ||
        pool_count > GLM53_DSA_MAX_POOLS || kv_capacity == 0U ||
        kv_capacity > GLM53_DSA_MAX_POSITIONS ||
        pool_count > (kv_capacity + GLM53_DSA_KPOOL - 1U) / GLM53_DSA_KPOOL ||
        blockDim.x != GLM53_DSA_THREADS || gridDim.x != 1U ||
        (unsigned long long) gridDim.y != expected_y ||
        (unsigned long long) gridDim.z != expected_z ||
        expected_z > GLM53_DSA_MAX_GRID_YZ) {
        return;
    }

    const unsigned long long row =
        (unsigned long long) blockIdx.y +
        (unsigned long long) gridDim.y * blockIdx.z;
    if (row >= rows) {
        return;
    }
    const unsigned int lane = threadIdx.x;
    const unsigned long long output_base =
        row * GLM53_DSA_OUTPUT_WIDTH;
    for (unsigned int index = lane; index < GLM53_DSA_OUTPUT_WIDTH;
         index += GLM53_DSA_THREADS) {
        output_indices[output_base + index] = -1;
    }

    __shared__ float selected_scores[GLM53_DSA_NETWORK];
    __shared__ unsigned int selected_pools[GLM53_DSA_NETWORK];
    #pragma unroll
    for (unsigned int item = lane; item < GLM53_DSA_NETWORK;
         item += GLM53_DSA_THREADS) {
        selected_scores[item] = 0.0f;
        selected_pools[item] = GLM53_DSA_INVALID_POOL;
    }
    __syncthreads();
    if (query_validity[row] == 0U) {
        return;
    }

    const unsigned long long batch_index = row / query_count;
    const unsigned int sequence_length = sequence_lengths[batch_index];
    const unsigned int query_position = query_positions[row];
    if (sequence_length == 0U || sequence_length > kv_capacity ||
        query_position >= sequence_length) {
        return;
    }
    const unsigned int complete_pools = sequence_length / GLM53_DSA_KPOOL;
    if (complete_pools > pool_count) {
        return;
    }

    for (unsigned int chunk = 0U; chunk < pool_count;
         chunk += GLM53_DSA_SELECTED_POOLS) {
        #pragma unroll
        for (unsigned int item = lane + GLM53_DSA_SELECTED_POOLS;
             item < GLM53_DSA_NETWORK; item += GLM53_DSA_THREADS) {
            const unsigned int pool =
                chunk + item - GLM53_DSA_SELECTED_POOLS;
            unsigned int admitted = GLM53_DSA_INVALID_POOL;
            float score = 0.0f;
            if (pool < pool_count && pool < complete_pools) {
                const unsigned long long pool_row =
                    batch_index * pool_count + pool;
                const unsigned int pool_end =
                    pool * GLM53_DSA_KPOOL + GLM53_DSA_KPOOL - 1U;
                const bool visible = pool_end <= query_position;
                if (pool_validity[pool_row] != 0U && visible) {
                    admitted = pool;
                    score = scores[row * pool_count + pool];
                }
            }
            selected_scores[item] = score;
            selected_pools[item] = admitted;
        }
        __syncthreads();

        for (unsigned int width = 2U; width <= GLM53_DSA_NETWORK;
             width <<= 1U) {
            for (unsigned int stride = width >> 1U; stride != 0U;
                 stride >>= 1U) {
                #pragma unroll
                for (unsigned int item = lane; item < GLM53_DSA_NETWORK;
                     item += GLM53_DSA_THREADS) {
                    const unsigned int partner = item ^ stride;
                    if (partner > item) {
                        const bool descending = (item & width) == 0U;
                        const bool partner_better = glm53_dsa_better(
                            selected_scores[partner], selected_pools[partner],
                            selected_scores[item], selected_pools[item]);
                        const bool item_better = glm53_dsa_better(
                            selected_scores[item], selected_pools[item],
                            selected_scores[partner], selected_pools[partner]);
                        if ((descending && partner_better) ||
                            (!descending && item_better)) {
                            const float held_score = selected_scores[item];
                            const unsigned int held_pool = selected_pools[item];
                            selected_scores[item] = selected_scores[partner];
                            selected_pools[item] = selected_pools[partner];
                            selected_scores[partner] = held_score;
                            selected_pools[partner] = held_pool;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    for (unsigned int output = lane; output < GLM53_DSA_INDEX_TOPK;
         output += GLM53_DSA_THREADS) {
        const unsigned int rank = output / GLM53_DSA_KPOOL;
        const unsigned int offset = output % GLM53_DSA_KPOOL;
        const unsigned int pool = selected_pools[rank];
        if (pool != GLM53_DSA_INVALID_POOL) {
            output_indices[output_base + output] =
                (int) (pool * GLM53_DSA_KPOOL + offset);
        }
    }

    if (lane == 0U) {
        unsigned int selected_count = 0U;
        while (selected_count < GLM53_DSA_SELECTED_POOLS &&
               selected_pools[selected_count] != GLM53_DSA_INVALID_POOL) {
            ++selected_count;
        }
        const unsigned int tail_output = selected_count * GLM53_DSA_KPOOL;
        const unsigned int visible_count = query_position + 1U;
        const unsigned int tail_count = visible_count % GLM53_DSA_KPOOL;
        const unsigned int tail_start = visible_count - tail_count;
        #pragma unroll
        for (unsigned int offset = 0U; offset < GLM53_DSA_KPOOL - 1U;
             ++offset) {
            const unsigned int index = tail_start + offset;
            const unsigned int pool = index / GLM53_DSA_KPOOL;
            const unsigned int slot = index % GLM53_DSA_KPOOL;
            const bool valid = pool < complete_pools
                ? pool_validity[batch_index * pool_count + pool] != 0U
                : (slot < GLM53_DSA_KPOOL - 1U &&
                    tail_validity[batch_index * (GLM53_DSA_KPOOL - 1U) + slot] != 0U);
            if (offset < tail_count && index < sequence_length && valid) {
                output_indices[output_base + tail_output + offset] =
                    (int) index;
            }
        }
    }
}
