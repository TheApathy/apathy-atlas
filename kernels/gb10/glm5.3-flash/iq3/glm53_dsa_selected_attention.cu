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

// Row-shared rewrite of atlas_glm53_dsa_selected_attention_bf16 for prompt
// chunks, bit-identical by construction. The selected set is per ROW, not per
// head, so one CTA canonicalizes it once and serves all 64 heads in groups of
// GLM53_DSA_ROWS_GROUP (the reference re-sorts it 64 times and walks every item
// with a 256-thread tree reduction and nine barriers). Every rounding step of
// the reference is reproduced in the same order:
//  * score: partial[l] = q[l]*k[l] + q[l+256]*k[l+256] (no FMA), reduced by the
//    same halving tree (strides 128..1). A warp holds partial[j + 32r] in lane
//    j, register r, so strides 128/64/32 are in-register and 16..1 are
//    shuffles pairing exactly the reference's operands; then * 1/16.
//  * softmax: max is order-free; exponentials use the same expf; the
//    denominator is the reference's serial ascending-item sum.
//  * probability = bf16(exp / denominator); each output column accumulates
//    fadd(sum, p*v) serially in ascending item order, as the reference does.
#define GLM53_DSA_ROWS_THREADS 512U
#define GLM53_DSA_ROWS_GROUP 8U
#define GLM53_DSA_ROWS_SHARED \
    (GLM53_DSA_SORT_WIDTH * 4U + GLM53_DSA_ROWS_GROUP * GLM53_DSA_SELECTED * 4U + \
     GLM53_DSA_ROWS_GROUP * 4U)

extern "C" __global__ void __launch_bounds__(GLM53_DSA_ROWS_THREADS, 1)
atlas_glm53_dsa_selected_attention_rows_bf16(
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
    if (absorbed_query == nullptr || latent_cache == nullptr ||
        selected_indices == nullptr || sequence_lengths == nullptr ||
        query_positions == nullptr || query_validity == nullptr ||
        output_weighted_latent == nullptr || batch == 0U ||
        query_count == 0U || query_count > GLM53_DSA_MAX_QUERIES ||
        kv_capacity == 0U || kv_capacity > GLM53_DSA_MAX_POSITIONS ||
        blockDim.x != GLM53_DSA_ROWS_THREADS ||
        (unsigned long long)gridDim.x != rows || gridDim.y != 1U ||
        gridDim.z != 1U) {
        return;
    }
    extern __shared__ unsigned char rows_shared[];
    unsigned int * ordered = (unsigned int *)rows_shared;
    float * scores = (float *)(ordered + GLM53_DSA_SORT_WIDTH);
    float * denominators = scores + GLM53_DSA_ROWS_GROUP * GLM53_DSA_SELECTED;
    __shared__ unsigned int unique_count;

    const unsigned long long row = blockIdx.x;
    const unsigned int thread = threadIdx.x;
    const unsigned int lane = thread & 31U;
    const unsigned int warp = thread >> 5U;
    const unsigned long long row_base = row * GLM53_DSA_HEADS * GLM53_DSA_LATENT;

    const unsigned long long batch_index = row / query_count;
    const unsigned int sequence_length = sequence_lengths[batch_index];
    const unsigned int query_position = query_positions[row];
    if (query_validity[row] == 0U || sequence_length == 0U ||
        sequence_length > kv_capacity || query_position >= sequence_length) {
        for (unsigned int i = thread; i < GLM53_DSA_HEADS * GLM53_DSA_LATENT;
             i += GLM53_DSA_ROWS_THREADS) {
            output_weighted_latent[row_base + i] = __float2bfloat16_rn(0.0f);
        }
        return;
    }

    for (unsigned int slot = thread; slot < GLM53_DSA_SORT_WIDTH;
         slot += GLM53_DSA_ROWS_THREADS) {
        unsigned int admitted = GLM53_DSA_INVALID;
        if (slot < GLM53_DSA_SELECTED) {
            const int raw = selected_indices[row * GLM53_DSA_SELECTED + slot];
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
    for (unsigned int width = 2U; width <= GLM53_DSA_SORT_WIDTH; width <<= 1U) {
        for (unsigned int stride = width >> 1U; stride != 0U; stride >>= 1U) {
            for (unsigned int left = thread; left < GLM53_DSA_SORT_WIDTH;
                 left += GLM53_DSA_ROWS_THREADS) {
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
    if (thread == 0U) {
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
    const unsigned int count = unique_count;
    if (count == 0U) {
        for (unsigned int i = thread; i < GLM53_DSA_HEADS * GLM53_DSA_LATENT;
             i += GLM53_DSA_ROWS_THREADS) {
            output_weighted_latent[row_base + i] = __float2bfloat16_rn(0.0f);
        }
        return;
    }
    const __nv_bfloat16 * cache =
        latent_cache + batch_index * (unsigned long long)kv_capacity * GLM53_DSA_LATENT;

    // Scores are stored item-major, [item][GROUP], so the weighted-latent
    // pass reads a whole group's probabilities with two 16-byte loads.
    for (unsigned int group = 0U; group < GLM53_DSA_HEADS; group += GLM53_DSA_ROWS_GROUP) {
        // Scores: two warps per head, alternating items; four items in flight
        // per warp so the L2 latency of one key row overlaps the next three.
        {
            const unsigned int local_head = warp % GLM53_DSA_ROWS_GROUP;
            const unsigned int parity = warp / GLM53_DSA_ROWS_GROUP;
            const __nv_bfloat16 * query = absorbed_query + row_base +
                (unsigned long long)(group + local_head) * GLM53_DSA_LATENT;
            float q_low[8];
            float q_high[8];
            #pragma unroll
            for (unsigned int r = 0U; r < 8U; ++r) {
                q_low[r] = __bfloat162float(query[lane + 32U * r]);
                q_high[r] = __bfloat162float(query[lane + 32U * r + 256U]);
            }
            for (unsigned int base = parity; base < count; base += 8U) {
                __nv_bfloat16 k_low[4][8];
                __nv_bfloat16 k_high[4][8];
                #pragma unroll
                for (unsigned int u = 0U; u < 4U; ++u) {
                    const unsigned int item = base + 2U * u;
                    if (item < count) {
                        const __nv_bfloat16 * key = cache +
                            (unsigned long long)ordered[item] * GLM53_DSA_LATENT;
                        #pragma unroll
                        for (unsigned int r = 0U; r < 8U; ++r) {
                            k_low[u][r] = key[lane + 32U * r];
                            k_high[u][r] = key[lane + 32U * r + 256U];
                        }
                    }
                }
                #pragma unroll
                for (unsigned int u = 0U; u < 4U; ++u) {
                    const unsigned int item = base + 2U * u;
                    if (item >= count) break;
                    float a[8];
                    #pragma unroll
                    for (unsigned int r = 0U; r < 8U; ++r) {
                        a[r] = __fadd_rn(
                            __fmul_rn(q_low[r], __bfloat162float(k_low[u][r])),
                            __fmul_rn(q_high[r], __bfloat162float(k_high[u][r])));
                    }
                    #pragma unroll
                    for (unsigned int r = 0U; r < 4U; ++r) a[r] = __fadd_rn(a[r], a[r + 4U]);
                    #pragma unroll
                    for (unsigned int r = 0U; r < 2U; ++r) a[r] = __fadd_rn(a[r], a[r + 2U]);
                    float sum = __fadd_rn(a[0], a[1]);
                    #pragma unroll
                    for (unsigned int offset = 16U; offset != 0U; offset >>= 1U) {
                        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, offset));
                    }
                    if (lane == 0U) {
                        scores[item * GLM53_DSA_ROWS_GROUP + local_head] =
                            __fmul_rn(sum, GLM53_DSA_INV_SQRT_QK);
                    }
                }
            }
        }
        __syncthreads();
        // Maximum (order-free) per head.
        if (warp < GLM53_DSA_ROWS_GROUP) {
            float maximum = -FLT_MAX;
            for (unsigned int item = lane; item < count; item += 32U) {
                const float value = scores[item * GLM53_DSA_ROWS_GROUP + warp];
                if (value > maximum) maximum = value;
            }
            #pragma unroll
            for (unsigned int offset = 16U; offset != 0U; offset >>= 1U) {
                maximum = fmaxf(maximum,
                                __shfl_xor_sync(0xffffffffU, maximum, offset));
            }
            if (lane == 0U) denominators[warp] = maximum;
        }
        __syncthreads();
        for (unsigned int index = thread; index < GLM53_DSA_ROWS_GROUP * count;
             index += GLM53_DSA_ROWS_THREADS) {
            scores[index] = expf(__fsub_rn(
                scores[index], denominators[index % GLM53_DSA_ROWS_GROUP]));
        }
        __syncthreads();
        if (thread < GLM53_DSA_ROWS_GROUP) {
            float denominator = 0.0f;
            #pragma unroll 8
            for (unsigned int item = 0U; item < count; ++item) {
                denominator = __fadd_rn(
                    denominator, scores[item * GLM53_DSA_ROWS_GROUP + thread]);
            }
            denominators[thread] = denominator;
        }
        __syncthreads();
        for (unsigned int index = thread; index < GLM53_DSA_ROWS_GROUP * count;
             index += GLM53_DSA_ROWS_THREADS) {
            scores[index] = __bfloat162float(__float2bfloat16_rn(
                scores[index] / denominators[index % GLM53_DSA_ROWS_GROUP]));
        }
        __syncthreads();
        // Weighted latent: thread owns one column for every head of the group,
        // four items' values in flight, accumulation still strictly in order.
        {
            const unsigned int column = thread;
            float sums[GLM53_DSA_ROWS_GROUP];
            #pragma unroll
            for (unsigned int h = 0U; h < GLM53_DSA_ROWS_GROUP; ++h) sums[h] = 0.0f;
            unsigned int item = 0U;
            for (; item + 4U <= count; item += 4U) {
                float value[4];
                #pragma unroll
                for (unsigned int u = 0U; u < 4U; ++u) {
                    value[u] = __bfloat162float(cache[
                        (unsigned long long)ordered[item + u] * GLM53_DSA_LATENT + column]);
                }
                #pragma unroll
                for (unsigned int u = 0U; u < 4U; ++u) {
                    const float4 p0 = *(const float4 *)(scores + (item + u) * GLM53_DSA_ROWS_GROUP);
                    const float4 p1 = *(const float4 *)(scores + (item + u) * GLM53_DSA_ROWS_GROUP + 4U);
                    sums[0] = __fadd_rn(sums[0], __fmul_rn(p0.x, value[u]));
                    sums[1] = __fadd_rn(sums[1], __fmul_rn(p0.y, value[u]));
                    sums[2] = __fadd_rn(sums[2], __fmul_rn(p0.z, value[u]));
                    sums[3] = __fadd_rn(sums[3], __fmul_rn(p0.w, value[u]));
                    sums[4] = __fadd_rn(sums[4], __fmul_rn(p1.x, value[u]));
                    sums[5] = __fadd_rn(sums[5], __fmul_rn(p1.y, value[u]));
                    sums[6] = __fadd_rn(sums[6], __fmul_rn(p1.z, value[u]));
                    sums[7] = __fadd_rn(sums[7], __fmul_rn(p1.w, value[u]));
                }
            }
            for (; item < count; ++item) {
                const float value = __bfloat162float(cache[
                    (unsigned long long)ordered[item] * GLM53_DSA_LATENT + column]);
                #pragma unroll
                for (unsigned int h = 0U; h < GLM53_DSA_ROWS_GROUP; ++h) {
                    sums[h] = __fadd_rn(sums[h], __fmul_rn(
                        scores[item * GLM53_DSA_ROWS_GROUP + h], value));
                }
            }
            #pragma unroll
            for (unsigned int h = 0U; h < GLM53_DSA_ROWS_GROUP; ++h) {
                output_weighted_latent[row_base +
                    (unsigned long long)(group + h) * GLM53_DSA_LATENT + column] =
                    __float2bfloat16_rn(sums[h]);
            }
        }
        __syncthreads();
    }
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
