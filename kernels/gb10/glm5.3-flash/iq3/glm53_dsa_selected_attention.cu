// SPDX-License-Identifier: AGPL-3.0-only
// BF16 GLM-5.3 DSA attention over selected rank-512 latent rows.

#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

#define GLM53_DSA_HEADS 64U
#define GLM53_DSA_LATENT 512U
#define GLM53_DSA_SELECTED 2051U
#define GLM53_DSA_SORT_WIDTH 4096U
#define GLM53_DSA_MAX_QUERIES 8U
#define GLM53_DSA_MAX_POSITIONS 1048576U
#define GLM53_DSA_THREADS 256U
#define GLM53_DSA_MAX_GRID_YZ 65535ULL
#define GLM53_DSA_INVALID 0xffffffffU
#define GLM53_DSA_INV_SQRT_QK 0.0625f

extern "C" __global__ void __launch_bounds__(GLM53_DSA_THREADS, 1)
atlas_glm53_dsa_selected_attention_bf16(
        const __nv_bfloat16 * __restrict__ absorbed_query,
        const __nv_bfloat16 * __restrict__ latent_cache,
        const int * __restrict__ selected_indices,
        const unsigned int * __restrict__ sequence_lengths,
        const unsigned int * __restrict__ query_positions,
        const unsigned char * __restrict__ query_validity,
        __nv_bfloat16 * __restrict__ output_weighted_latent,
        unsigned int batch, unsigned int query_count,
        unsigned int kv_capacity) {
    const unsigned long long rows =
        (unsigned long long)batch * query_count;
    const unsigned long long expected_y =
        rows < GLM53_DSA_MAX_GRID_YZ ? rows : GLM53_DSA_MAX_GRID_YZ;
    const unsigned long long expected_z = expected_y == 0ULL ? 0ULL :
        (rows + expected_y - 1ULL) / expected_y;
    if (absorbed_query == nullptr || latent_cache == nullptr ||
        selected_indices == nullptr || sequence_lengths == nullptr ||
        query_positions == nullptr || query_validity == nullptr ||
        output_weighted_latent == nullptr || batch == 0U ||
        query_count == 0U || query_count > GLM53_DSA_MAX_QUERIES ||
        kv_capacity == 0U || kv_capacity > GLM53_DSA_MAX_POSITIONS ||
        blockDim.x != GLM53_DSA_THREADS || gridDim.x != GLM53_DSA_HEADS ||
        (unsigned long long)gridDim.y != expected_y ||
        (unsigned long long)gridDim.z != expected_z ||
        expected_z > GLM53_DSA_MAX_GRID_YZ) {
        return;
    }

    const unsigned long long row =
        (unsigned long long)blockIdx.y +
        (unsigned long long)gridDim.y * blockIdx.z;
    if (row >= rows) {
        return;
    }
    const unsigned int lane = threadIdx.x;
    const unsigned int head = blockIdx.x;
    const unsigned long long output_base =
        (row * GLM53_DSA_HEADS + head) * GLM53_DSA_LATENT;
    output_weighted_latent[output_base + lane] =
        __float2bfloat16_rn(0.0f);
    output_weighted_latent[output_base + lane + GLM53_DSA_THREADS] =
        __float2bfloat16_rn(0.0f);

    const unsigned long long batch_index = row / query_count;
    const unsigned int sequence_length = sequence_lengths[batch_index];
    const unsigned int query_position = query_positions[row];
    if (query_validity[row] == 0U || sequence_length == 0U ||
        sequence_length > kv_capacity || query_position >= sequence_length) {
        return;
    }

    __shared__ unsigned int ordered[GLM53_DSA_SORT_WIDTH];
    __shared__ float exponentials[GLM53_DSA_SELECTED];
    __shared__ float partial[GLM53_DSA_THREADS];
    __shared__ unsigned int unique_count;
    for (unsigned int slot = lane; slot < GLM53_DSA_SORT_WIDTH;
         slot += GLM53_DSA_THREADS) {
        unsigned int admitted = GLM53_DSA_INVALID;
        if (slot < GLM53_DSA_SELECTED) {
            const int raw = selected_indices[
                row * GLM53_DSA_SELECTED + slot];
            if (raw >= 0) {
                const unsigned int candidate = (unsigned int)raw;
                if (candidate < sequence_length &&
                    candidate <= query_position &&
                    candidate < kv_capacity) {
                    admitted = candidate;
                }
            }
        }
        ordered[slot] = admitted;
    }
    __syncthreads();

    for (unsigned int width = 2U; width <= GLM53_DSA_SORT_WIDTH;
         width <<= 1U) {
        for (unsigned int stride = width >> 1U; stride != 0U;
             stride >>= 1U) {
            for (unsigned int left = lane; left < GLM53_DSA_SORT_WIDTH;
                 left += GLM53_DSA_THREADS) {
                const unsigned int right = left ^ stride;
                if (right > left) {
                    const bool ascending = (left & width) == 0U;
                    const bool swap = ascending
                        ? ordered[left] > ordered[right]
                        : ordered[left] < ordered[right];
                    if (swap) {
                        const unsigned int held = ordered[left];
                        ordered[left] = ordered[right];
                        ordered[right] = held;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (lane == 0U) {
        unsigned int count = 0U;
        unsigned int previous = GLM53_DSA_INVALID;
        for (unsigned int slot = 0U; slot < GLM53_DSA_SORT_WIDTH; ++slot) {
            const unsigned int candidate = ordered[slot];
            if (candidate == GLM53_DSA_INVALID) {
                break;
            }
            if (count == 0U || candidate != previous) {
                ordered[count++] = candidate;
                previous = candidate;
            }
        }
        unique_count = count;
    }
    __syncthreads();
    if (unique_count == 0U) {
        return;
    }

    const unsigned long long query_base =
        (row * GLM53_DSA_HEADS + head) * GLM53_DSA_LATENT;
    for (unsigned int item = 0U; item < unique_count; ++item) {
        const unsigned long long latent_base =
            (batch_index * (unsigned long long)kv_capacity + ordered[item]) *
            GLM53_DSA_LATENT;
        float local = __fmul_rn(
            __bfloat162float(absorbed_query[query_base + lane]),
            __bfloat162float(latent_cache[latent_base + lane]));
        local = __fadd_rn(local, __fmul_rn(
            __bfloat162float(absorbed_query[
                query_base + lane + GLM53_DSA_THREADS]),
            __bfloat162float(latent_cache[
                latent_base + lane + GLM53_DSA_THREADS])));
        partial[lane] = local;
        __syncthreads();
        for (unsigned int stride = GLM53_DSA_THREADS / 2U;
             stride != 0U; stride >>= 1U) {
            if (lane < stride) {
                partial[lane] = __fadd_rn(
                    partial[lane], partial[lane + stride]);
            }
            __syncthreads();
        }
        if (lane == 0U) {
            exponentials[item] = __fmul_rn(
                partial[0], GLM53_DSA_INV_SQRT_QK);
        }
        __syncthreads();
    }

    if (lane == 0U) {
        float maximum = -FLT_MAX;
        for (unsigned int item = 0U; item < unique_count; ++item) {
            if (exponentials[item] > maximum) {
                maximum = exponentials[item];
            }
        }
        float denominator = 0.0f;
        for (unsigned int item = 0U; item < unique_count; ++item) {
            exponentials[item] = expf(
                __fsub_rn(exponentials[item], maximum));
            denominator = __fadd_rn(denominator, exponentials[item]);
        }
        partial[0] = denominator;
    }
    __syncthreads();

    const float denominator = partial[0];
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    for (unsigned int item = 0U; item < unique_count; ++item) {
        const float probability = __bfloat162float(
            __float2bfloat16_rn(exponentials[item] / denominator));
        const unsigned long long latent_base =
            (batch_index * (unsigned long long)kv_capacity + ordered[item]) *
            GLM53_DSA_LATENT;
        const float first_value = __bfloat162float(
            latent_cache[latent_base + lane]);
        const float second_value = __bfloat162float(
            latent_cache[latent_base + lane + GLM53_DSA_THREADS]);
        first_sum = __fadd_rn(
            first_sum, __fmul_rn(probability, first_value));
        second_sum = __fadd_rn(
            second_sum, __fmul_rn(probability, second_value));
    }
    output_weighted_latent[output_base + lane] =
        __float2bfloat16_rn(first_sum);
    output_weighted_latent[output_base + lane + GLM53_DSA_THREADS] =
        __float2bfloat16_rn(second_sum);
}
