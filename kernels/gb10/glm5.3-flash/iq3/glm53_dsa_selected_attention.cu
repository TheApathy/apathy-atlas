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
// with a 256-thread tree reduction and nine barriers). Selected latent rows are
// staged through shared memory in double-buffered cp.async tiles. Every
// rounding step of the reference is reproduced in the same order:
//  * score: partial[l] = q[l]*k[l] + q[l+256]*k[l+256] (no FMA), reduced by the
//    same halving tree (strides 128..1). A warp scores one item for four heads
//    and holds partial[j + 32r] in lane j, register r, so strides 128/64/32 are
//    in-register; strides 16..1 are an xor butterfly that also splits the four
//    heads across lane halves, so every addition pairs the reference's
//    operands (a + b == b + a exactly) with 6 shuffles per 4 dots, not 20.
//    The score is then * 1/16.
//  * softmax: max is order-free; exponentials use the same expf; the
//    denominator is the reference's serial ascending-item sum.
//  * probability = bf16(exp / denominator); each output column accumulates
//    fadd(sum, p*v) serially in ascending item order, as the reference does.
#define GLM53_DSA_ROWS_THREADS 512U
#define GLM53_DSA_ROWS_GROUP 8U
#define GLM53_DSA_ROWS_TILE 8U
// [item][GROUP] f32 scores | canonical item list | sort network / 2 KV tiles
#define GLM53_DSA_ROWS_SCORE_BYTES (GLM53_DSA_SELECTED * GLM53_DSA_ROWS_GROUP * 4U)
#define GLM53_DSA_ROWS_ORDER_BYTES 8224U
#define GLM53_DSA_ROWS_TILE_BYTES (GLM53_DSA_ROWS_TILE * GLM53_DSA_LATENT * 2U)
#define GLM53_DSA_ROWS_SCRATCH_BYTES \
    (2U * GLM53_DSA_ROWS_TILE_BYTES > GLM53_DSA_SORT_WIDTH * 4U ? \
     2U * GLM53_DSA_ROWS_TILE_BYTES : GLM53_DSA_SORT_WIDTH * 4U)
#define GLM53_DSA_ROWS_SHARED \
    (GLM53_DSA_ROWS_SCORE_BYTES + GLM53_DSA_ROWS_ORDER_BYTES + GLM53_DSA_ROWS_SCRATCH_BYTES)

__device__ __forceinline__ void glm53_dsa_rows_cp16(void * shared, const void * global) {
    const unsigned address = (unsigned)__cvta_generic_to_shared(shared);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address), "l"(global));
}

// Stage items [first, first + n) of the canonical list into one tile.
__device__ __forceinline__ void glm53_dsa_rows_stage(
        __nv_bfloat16 * tile, const __nv_bfloat16 * cache,
        const unsigned int * ordered, unsigned int first, unsigned int n,
        unsigned int thread) {
    const unsigned int chunks_per_item = GLM53_DSA_LATENT * 2U / 16U;
    for (unsigned int chunk = thread; chunk < n * chunks_per_item;
         chunk += GLM53_DSA_ROWS_THREADS) {
        const unsigned int item = chunk / chunks_per_item;
        const unsigned int column = (chunk % chunks_per_item) * 8U;
        glm53_dsa_rows_cp16(
            tile + item * GLM53_DSA_LATENT + column,
            cache + (unsigned long long)ordered[first + item] * GLM53_DSA_LATENT + column);
    }
    asm volatile("cp.async.commit_group;");
}

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
    extern __shared__ __align__(16) unsigned char rows_shared[];
    float * scores = (float *)rows_shared;
    unsigned int * ordered =
        (unsigned int *)(rows_shared + GLM53_DSA_ROWS_SCORE_BYTES);
    unsigned int * network = (unsigned int *)(rows_shared + GLM53_DSA_ROWS_SCORE_BYTES +
                                              GLM53_DSA_ROWS_ORDER_BYTES);
    __nv_bfloat16 * tiles = (__nv_bfloat16 *)network;
    __shared__ float denominators[GLM53_DSA_ROWS_GROUP];
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
        network[slot] = admitted;
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
                        ? network[left] > network[right]
                        : network[left] < network[right];
                    if (swap) {
                        const unsigned int held = network[left];
                        network[left] = network[right];
                        network[right] = held;
                    }
                }
            }
            __syncthreads();
        }
    }
    // Sorted ascending with INVALID last: an entry survives dedupe when it is
    // valid and differs from its predecessor. Its output slot is the number of
    // survivors before it, a block-wide exclusive prefix count.
    {
        __shared__ unsigned int warp_counts[GLM53_DSA_ROWS_THREADS / 32U];
        const unsigned int per_thread = GLM53_DSA_SORT_WIDTH / GLM53_DSA_ROWS_THREADS;
        unsigned int keep[GLM53_DSA_SORT_WIDTH / GLM53_DSA_ROWS_THREADS];
        unsigned int value[GLM53_DSA_SORT_WIDTH / GLM53_DSA_ROWS_THREADS];
        unsigned int local = 0U;
        #pragma unroll
        for (unsigned int e = 0U; e < per_thread; ++e) {
            const unsigned int slot = thread * per_thread + e;
            value[e] = network[slot];
            keep[e] = value[e] != GLM53_DSA_INVALID &&
                (slot == 0U || network[slot - 1U] != value[e]) ? 1U : 0U;
            local += keep[e];
        }
        unsigned int inclusive = local;
        #pragma unroll
        for (unsigned int offset = 1U; offset < 32U; offset <<= 1U) {
            const unsigned int other = __shfl_up_sync(0xffffffffU, inclusive, offset);
            if (lane >= offset) inclusive += other;
        }
        if (lane == 31U) warp_counts[warp] = inclusive;
        __syncthreads();
        unsigned int before = inclusive - local;
        for (unsigned int w = 0U; w < warp; ++w) before += warp_counts[w];
        #pragma unroll
        for (unsigned int e = 0U; e < per_thread; ++e) {
            if (keep[e] != 0U) ordered[before++] = value[e];
        }
        if (thread == GLM53_DSA_ROWS_THREADS - 1U) unique_count = before;
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
    const unsigned int tile_count = (count + GLM53_DSA_ROWS_TILE - 1U) / GLM53_DSA_ROWS_TILE;

    for (unsigned int group = 0U; group < GLM53_DSA_HEADS; group += GLM53_DSA_ROWS_GROUP) {
        // Scores: warp w scores heads 4*(w&1)..+3 of the group for tile item
        // w>>1; one staged key row feeds four dots.
        {
            const unsigned int quad = warp & 1U;
            const unsigned int slot = warp >> 1U;
            float q_low[4][8];
            float q_high[4][8];
            #pragma unroll
            for (unsigned int h = 0U; h < 4U; ++h) {
                const __nv_bfloat16 * query = absorbed_query + row_base +
                    (unsigned long long)(group + quad * 4U + h) * GLM53_DSA_LATENT;
                #pragma unroll
                for (unsigned int r = 0U; r < 8U; ++r) {
                    q_low[h][r] = __bfloat162float(query[lane + 32U * r]);
                    q_high[h][r] = __bfloat162float(query[lane + 32U * r + 256U]);
                }
            }
            const bool upper16 = (lane & 16U) != 0U;
            const bool upper8 = (lane & 8U) != 0U;
            glm53_dsa_rows_stage(tiles, cache, ordered, 0U,
                                 min(count, GLM53_DSA_ROWS_TILE), thread);
            for (unsigned int t = 0U; t < tile_count; ++t) {
                const unsigned int first = t * GLM53_DSA_ROWS_TILE;
                const unsigned int n = min(count - first, GLM53_DSA_ROWS_TILE);
                if (t + 1U < tile_count) {
                    const unsigned int next = first + GLM53_DSA_ROWS_TILE;
                    glm53_dsa_rows_stage(
                        tiles + ((t + 1U) & 1U) * GLM53_DSA_ROWS_TILE * GLM53_DSA_LATENT,
                        cache, ordered, next, min(count - next, GLM53_DSA_ROWS_TILE),
                        thread);
                    asm volatile("cp.async.wait_group 1;");
                } else {
                    asm volatile("cp.async.wait_group 0;");
                }
                __syncthreads();
                if (slot < n) {
                    const __nv_bfloat16 * key = tiles +
                        (t & 1U) * GLM53_DSA_ROWS_TILE * GLM53_DSA_LATENT +
                        slot * GLM53_DSA_LATENT;
                    float k_low[8];
                    float k_high[8];
                    #pragma unroll
                    for (unsigned int r = 0U; r < 8U; ++r) {
                        k_low[r] = __bfloat162float(key[lane + 32U * r]);
                        k_high[r] = __bfloat162float(key[lane + 32U * r + 256U]);
                    }
                    float v[4];
                    #pragma unroll
                    for (unsigned int h = 0U; h < 4U; ++h) {
                        float a[8];
                        #pragma unroll
                        for (unsigned int r = 0U; r < 8U; ++r) {
                            a[r] = __fadd_rn(__fmul_rn(q_low[h][r], k_low[r]),
                                             __fmul_rn(q_high[h][r], k_high[r]));
                        }
                        #pragma unroll
                        for (unsigned int r = 0U; r < 4U; ++r) a[r] = __fadd_rn(a[r], a[r + 4U]);
                        #pragma unroll
                        for (unsigned int r = 0U; r < 2U; ++r) a[r] = __fadd_rn(a[r], a[r + 2U]);
                        v[h] = __fadd_rn(a[0], a[1]);
                    }
                    // Stride 16: the lower lane half keeps heads 0,1 and the
                    // upper half heads 2,3; each receives its partner's value.
                    const float send0 = upper16 ? v[0] : v[2];
                    const float send1 = upper16 ? v[1] : v[3];
                    const float got0 = __shfl_xor_sync(0xffffffffU, send0, 16U);
                    const float got1 = __shfl_xor_sync(0xffffffffU, send1, 16U);
                    const float w0 = __fadd_rn(upper16 ? v[2] : v[0], got0);
                    const float w1 = __fadd_rn(upper16 ? v[3] : v[1], got1);
                    // Stride 8: split the remaining pair the same way.
                    const float got = __shfl_xor_sync(0xffffffffU, upper8 ? w0 : w1, 8U);
                    float x = __fadd_rn(upper8 ? w1 : w0, got);
                    x = __fadd_rn(x, __shfl_xor_sync(0xffffffffU, x, 4U));
                    x = __fadd_rn(x, __shfl_xor_sync(0xffffffffU, x, 2U));
                    x = __fadd_rn(x, __shfl_xor_sync(0xffffffffU, x, 1U));
                    // Lanes 0/8/16/24 hold heads 0/1/2/3 at tree index 0.
                    if ((lane & 7U) == 0U) {
                        scores[(first + slot) * GLM53_DSA_ROWS_GROUP + quad * 4U + (lane >> 3U)] =
                            __fmul_rn(x, GLM53_DSA_INV_SQRT_QK);
                    }
                }
                __syncthreads();
            }
        }
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
        // Prefetch the first value tile while the softmax runs.
        glm53_dsa_rows_stage(tiles, cache, ordered, 0U,
                             min(count, GLM53_DSA_ROWS_TILE), thread);
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
        // Weighted latent: thread owns one column for every head of the group;
        // accumulation stays strictly in ascending item order.
        {
            const unsigned int column = thread;
            float sums[GLM53_DSA_ROWS_GROUP];
            #pragma unroll
            for (unsigned int h = 0U; h < GLM53_DSA_ROWS_GROUP; ++h) sums[h] = 0.0f;
            for (unsigned int t = 0U; t < tile_count; ++t) {
                const unsigned int first = t * GLM53_DSA_ROWS_TILE;
                const unsigned int n = min(count - first, GLM53_DSA_ROWS_TILE);
                if (t + 1U < tile_count) {
                    const unsigned int next = first + GLM53_DSA_ROWS_TILE;
                    glm53_dsa_rows_stage(
                        tiles + ((t + 1U) & 1U) * GLM53_DSA_ROWS_TILE * GLM53_DSA_LATENT,
                        cache, ordered, next, min(count - next, GLM53_DSA_ROWS_TILE),
                        thread);
                    asm volatile("cp.async.wait_group 1;");
                } else {
                    asm volatile("cp.async.wait_group 0;");
                }
                __syncthreads();
                const __nv_bfloat16 * tile =
                    tiles + (t & 1U) * GLM53_DSA_ROWS_TILE * GLM53_DSA_LATENT;
                for (unsigned int local = 0U; local < n; ++local) {
                    const float value =
                        __bfloat162float(tile[local * GLM53_DSA_LATENT + column]);
                    const float * p = scores + (first + local) * GLM53_DSA_ROWS_GROUP;
                    const float4 p0 = *(const float4 *)p;
                    const float4 p1 = *(const float4 *)(p + 4U);
                    sums[0] = __fadd_rn(sums[0], __fmul_rn(p0.x, value));
                    sums[1] = __fadd_rn(sums[1], __fmul_rn(p0.y, value));
                    sums[2] = __fadd_rn(sums[2], __fmul_rn(p0.z, value));
                    sums[3] = __fadd_rn(sums[3], __fmul_rn(p0.w, value));
                    sums[4] = __fadd_rn(sums[4], __fmul_rn(p1.x, value));
                    sums[5] = __fadd_rn(sums[5], __fmul_rn(p1.y, value));
                    sums[6] = __fadd_rn(sums[6], __fmul_rn(p1.z, value));
                    sums[7] = __fadd_rn(sums[7], __fmul_rn(p1.w, value));
                }
                __syncthreads();
            }
            #pragma unroll
            for (unsigned int h = 0U; h < GLM53_DSA_ROWS_GROUP; ++h) {
                output_weighted_latent[row_base +
                    (unsigned long long)(group + h) * GLM53_DSA_LATENT + column] =
                    __float2bfloat16_rn(sums[h]);
            }
        }
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
