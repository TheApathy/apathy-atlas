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
#define GLM53_DSA_MAX_QUERIES 8192U
#define GLM53_DSA_MAX_POSITIONS 1048576U
#define GLM53_DSA_THREADS 256U
#define GLM53_DSA_MAX_GRID_YZ 65535ULL
#define GLM53_DSA_INVALID 0xffffffffU
#define GLM53_DSA_INV_SQRT_QK 0.0625f

extern "C" __global__ void __launch_bounds__(GLM53_DSA_THREADS, 1)
atlas_glm53_dsa_transpose_heads_bf16(
        const __nv_bfloat16 * __restrict__ input,
        __nv_bfloat16 * __restrict__ output,
        unsigned int rows, unsigned int width,
        unsigned int to_head_major) {
    if (input == nullptr || output == nullptr || rows == 0U ||
        rows > GLM53_DSA_MAX_QUERIES ||
        (width != 256U && width != 512U) || to_head_major > 1U) return;
    const unsigned long long values =
        (unsigned long long)rows * GLM53_DSA_HEADS * width;
    unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * blockDim.x;
    for (; index < values; index += stride) {
        const unsigned long long column = index % width;
        const unsigned long long pair = index / width;
        const unsigned long long row = pair / GLM53_DSA_HEADS;
        const unsigned long long head = pair % GLM53_DSA_HEADS;
        const unsigned long long head_major =
            (head * rows + row) * width + column;
        if (to_head_major != 0U) output[head_major] = input[index];
        else output[index] = input[head_major];
    }
}

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

// Before the selector becomes sparse, its top-512 pools cover the entire
// causal prefix (2048 pooled tokens plus the three raw-tail slots). Reuse the
// shipping GB10 HDIM=512 FlashAttention compute for that exact dense region:
// latent K and V are the same contiguous rank-512 cache, shared by all 64
// query heads. The common kernel tiles 32 query rows and 32 latent rows, so K/V
// are reused from shared memory instead of reread once per row and head.
#define LOAD_KV_TILE_512(cache, bt, smem_ptr, kv_s, kv_l, kvh, t, stride) \
    do { \
        (void)(bt); \
        (void)(kvh); \
        const unsigned int _cpr = HDIM_512 / 8U; \
        for (unsigned int _i = (t); _i < TILE_CHUNKS_512; _i += (stride)) { \
            const unsigned int _row = _i / _cpr; \
            const unsigned int _col = (_i % _cpr) * 8U; \
            const unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                const void* _gm = (const void*)((cache) + \
                    (unsigned long long)_pos * HDIM_512 + _col); \
                atlas_cp16(&(smem_ptr)[_row * HDIM_512 + _col], _gm); \
            } else { \
                *((uint4*)&(smem_ptr)[_row * HDIM_512 + _col]) = \
                    make_uint4(0U, 0U, 0U, 0U); \
            } \
        } \
    } while (0)

#define KERNEL_NAME atlas_glm53_dsa_dense_causal_bf16
#define K_CACHE_TYPE const __nv_bfloat16* __restrict__
#define V_CACHE_TYPE const __nv_bfloat16* __restrict__
#define KERNEL_EXTRA_PARAMS , const float inv_sqrt_d
#define KERNEL_PREAMBLE /* contiguous one-KV-head latent cache */

// This tree's common/prefill_paged_compute*.cuh predate upstream's portable
// cp.async helpers; GLM's tile-load macros call atlas_cp16, so define it here.
#ifndef GLM53_ATLAS_CP16_DEFINED
#define GLM53_ATLAS_CP16_DEFINED
__device__ __forceinline__ void atlas_cp16(void* smem_dst, const void* gmem_src) {
    unsigned _s = __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(_s), "l"(gmem_src));
}
#endif
#include "../../common/prefill_paged_compute_512.cuh"
