// SPDX-License-Identifier: AGPL-3.0-only

// Compile-only DeepSeek-V4 MLA decode experiment. One 256-thread CTA owns
// eight Q heads; each warp owns one head while the CTA stages each FP8 KV row
// once. This file is intentionally outside common/ and the serving registry.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define MLA_HG_HEADS 8
#define MLA_HG_ROWS 4
#define MLA_HG_DIM 512
#define MLA_HG_CACHE_DIM 576
#define MLA_HG_BLOCK_SIZE 16
#define MLA_HG_WINDOW 128
#define COMP_BLOCK_DIM 512

static_assert(MLA_HG_HEADS * 32 == 256, "one warp must own each head");
static_assert(MLA_HG_DIM == 512 && MLA_HG_CACHE_DIM == 576,
              "experiment is DeepSeek-V4 MLA only");

__device__ __forceinline__ float mla_hg_fp8_to_f32(unsigned char value) {
    return __half2float(__nv_cvt_fp8_to_halfraw(
        (__nv_fp8_storage_t)value, __NV_E4M3));
}

template <bool KV_ALIAS>
__device__ void mla_paged_decode_fp8_heads8_impl(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int q_head_dim,
    const unsigned int kv_cache_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const float k_scale,
    const float v_scale,
    const unsigned long long cache_stride_bytes,
    const unsigned int sliding_window,
    const float* __restrict__ sinks,
    const unsigned char* __restrict__ comp_pool,
    const unsigned int* __restrict__ comp_block_count_ptr,
    const unsigned int comp_ratio) {
    // Every predicate is block-uniform and precedes the first barrier/store.
    if (Q == nullptr || K_cache == nullptr || O == nullptr ||
        block_tables == nullptr || seq_lens == nullptr ||
        (!KV_ALIAS && V_cache == nullptr) || max_blocks_per_seq == 0 ||
        num_q_heads != 64 || num_kv_heads != 1 ||
        q_head_dim != 512 || kv_cache_dim != 576 ||
        block_size != 16 || sliding_window != 128 ||
        gridDim.x != 8 || gridDim.z != 1 ||
        blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1) {
        return;
    }
    if constexpr (KV_ALIAS) {
        // The alias entry reuses decoded K values as V only under the same exact
        // scale proof required by the production host path.
        if (__float_as_uint(k_scale) != __float_as_uint(v_scale)) return;
    }

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane_id = tid & 31;
    const unsigned int q_head = blockIdx.x * MLA_HG_HEADS + warp_id;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;
    if ((unsigned long long)seq_len > (unsigned long long)max_blocks_per_seq * MLA_HG_BLOCK_SIZE)
        return;

    const unsigned long long qo_row_base =
        (unsigned long long)seq_idx * num_q_heads * MLA_HG_DIM;
    const __nv_bfloat16* q_ptr =
        Q + qo_row_base + (unsigned long long)q_head * MLA_HG_DIM + lane_id * 16;
    float q[16];
#pragma unroll
    for (unsigned int i = 0; i < 16; ++i) q[i] = __bfloat162float(q_ptr[i]);

    float maximum = -__int_as_float(0x7f800000);
    float denominator = 0.0f;
    float output[16];
#pragma unroll
    for (unsigned int i = 0; i < 16; ++i) output[i] = 0.0f;

    __shared__ float kv_tile[MLA_HG_ROWS][MLA_HG_DIM];
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    const unsigned int kv_start = seq_len > MLA_HG_WINDOW
        ? seq_len - MLA_HG_WINDOW : 0;

    for (unsigned int tile = kv_start; tile < seq_len; tile += MLA_HG_ROWS) {
        const unsigned int rows = min((unsigned int)MLA_HG_ROWS, seq_len - tile);
        for (unsigned int linear = tid;
             linear < MLA_HG_ROWS * MLA_HG_DIM; linear += blockDim.x) {
            const unsigned int row = linear / MLA_HG_DIM;
            const unsigned int dim = linear % MLA_HG_DIM;
            float value = 0.0f;
            if (row < rows) {
                const unsigned int position = tile + row;
                const unsigned int logical_block = position / MLA_HG_BLOCK_SIZE;
                const unsigned int block_offset = position % MLA_HG_BLOCK_SIZE;
                const unsigned int physical_block =
                    (unsigned int)my_block_table[logical_block];
                const unsigned char* cache_row = K_cache +
                    (unsigned long long)physical_block * cache_stride_bytes +
                    (unsigned long long)block_offset * MLA_HG_CACHE_DIM;
                const unsigned int source_dim = dim < 448 ? dim : 512 + dim - 448;
                value = mla_hg_fp8_to_f32(cache_row[source_dim]) * k_scale;
            }
            kv_tile[row][dim] = value;
        }
        __syncthreads();

        float scores[MLA_HG_ROWS];
#pragma unroll
        for (unsigned int row = 0; row < MLA_HG_ROWS; ++row) {
            scores[row] = -__int_as_float(0x7f800000);
            if (row < rows) {
                float dot = 0.0f;
#pragma unroll
                for (unsigned int i = 0; i < 16; ++i) {
                    dot += q[i] * kv_tile[row][lane_id * 16 + i];
                }
#pragma unroll
                for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                    dot += __shfl_xor_sync(0xffffffffu, dot, offset);
                }
                scores[row] = dot * inv_sqrt_d;
            }
        }
        float tile_maximum = maximum;
#pragma unroll
        for (unsigned int row = 0; row < MLA_HG_ROWS; ++row) {
            if (row < rows) tile_maximum = fmaxf(tile_maximum, scores[row]);
        }
        const float old_scale = __expf(maximum - tile_maximum);
        denominator *= old_scale;
#pragma unroll
        for (unsigned int i = 0; i < 16; ++i) output[i] *= old_scale;
        float factors[MLA_HG_ROWS];
#pragma unroll
        for (unsigned int row = 0; row < MLA_HG_ROWS; ++row) {
            factors[row] = row < rows ? __expf(scores[row] - tile_maximum) : 0.0f;
            denominator += factors[row];
        }
        maximum = tile_maximum;
        __syncthreads();

        if constexpr (!KV_ALIAS) {
            for (unsigned int linear = tid;
                 linear < MLA_HG_ROWS * MLA_HG_DIM; linear += blockDim.x) {
                const unsigned int row = linear / MLA_HG_DIM;
                const unsigned int dim = linear % MLA_HG_DIM;
                float value = 0.0f;
                if (row < rows) {
                    const unsigned int position = tile + row;
                    const unsigned int logical_block = position / MLA_HG_BLOCK_SIZE;
                    const unsigned int block_offset = position % MLA_HG_BLOCK_SIZE;
                    const unsigned int physical_block =
                        (unsigned int)my_block_table[logical_block];
                    const unsigned char* cache_row = V_cache +
                        (unsigned long long)physical_block * cache_stride_bytes +
                        (unsigned long long)block_offset * MLA_HG_CACHE_DIM;
                    const unsigned int source_dim = dim < 448 ? dim : 512 + dim - 448;
                    value = mla_hg_fp8_to_f32(cache_row[source_dim]) * v_scale;
                }
                kv_tile[row][dim] = value;
            }
            __syncthreads();
        }

#pragma unroll
        for (unsigned int row = 0; row < MLA_HG_ROWS; ++row) {
            if (row < rows) {
#pragma unroll
                for (unsigned int i = 0; i < 16; ++i) {
                    output[i] += factors[row] * kv_tile[row][lane_id * 16 + i];
                }
            }
        }
        __syncthreads();
    }

    const unsigned int comp_block_count = comp_block_count_ptr
        ? __ldg(comp_block_count_ptr) : 0;
    unsigned int comp_visible = comp_block_count;
    if (comp_ratio != 0) {
        comp_visible = seq_len / comp_ratio;
        if (comp_visible > comp_block_count) comp_visible = comp_block_count;
    }
    if (comp_pool != nullptr) {
        for (unsigned int tile = 0; tile < comp_visible; tile += MLA_HG_ROWS) {
            const unsigned int rows =
                min((unsigned int)MLA_HG_ROWS, comp_visible - tile);
            for (unsigned int linear = tid;
                 linear < MLA_HG_ROWS * MLA_HG_DIM; linear += blockDim.x) {
                const unsigned int row = linear / MLA_HG_DIM;
                const unsigned int dim = linear % MLA_HG_DIM;
                float value = 0.0f;
                if (row < rows) {
                    const unsigned char* cache_row = comp_pool +
                        (unsigned long long)(tile + row) * COMP_BLOCK_DIM;
                    value = mla_hg_fp8_to_f32(cache_row[dim]) * k_scale;
                }
                kv_tile[row][dim] = value;
            }
            __syncthreads();

            float scores[MLA_HG_ROWS];
#pragma unroll
            for (unsigned int row = 0; row < MLA_HG_ROWS; ++row) {
                scores[row] = -__int_as_float(0x7f800000);
                if (row < rows) {
                    float dot = 0.0f;
#pragma unroll
                    for (unsigned int i = 0; i < 16; ++i) {
                        dot += q[i] * kv_tile[row][lane_id * 16 + i];
                    }
#pragma unroll
                    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                        dot += __shfl_xor_sync(0xffffffffu, dot, offset);
                    }
                    scores[row] = dot * inv_sqrt_d;
                }
            }
            float tile_maximum = maximum;
#pragma unroll
            for (unsigned int row = 0; row < MLA_HG_ROWS; ++row) {
                if (row < rows) tile_maximum = fmaxf(tile_maximum, scores[row]);
            }
            const float old_scale = __expf(maximum - tile_maximum);
            denominator *= old_scale;
#pragma unroll
            for (unsigned int i = 0; i < 16; ++i) output[i] *= old_scale;
            float factors[MLA_HG_ROWS];
#pragma unroll
            for (unsigned int row = 0; row < MLA_HG_ROWS; ++row) {
                factors[row] = row < rows ? __expf(scores[row] - tile_maximum) : 0.0f;
                denominator += factors[row];
            }
            maximum = tile_maximum;
            __syncthreads();

            if constexpr (!KV_ALIAS) {
                for (unsigned int linear = tid;
                     linear < MLA_HG_ROWS * MLA_HG_DIM; linear += blockDim.x) {
                    const unsigned int row = linear / MLA_HG_DIM;
                    const unsigned int dim = linear % MLA_HG_DIM;
                    float value = 0.0f;
                    if (row < rows) {
                        const unsigned char* cache_row = comp_pool +
                            (unsigned long long)(tile + row) * COMP_BLOCK_DIM;
                        value = mla_hg_fp8_to_f32(cache_row[dim]) * v_scale;
                    }
                    kv_tile[row][dim] = value;
                }
                __syncthreads();
            }

#pragma unroll
            for (unsigned int row = 0; row < MLA_HG_ROWS; ++row) {
                if (row < rows) {
#pragma unroll
                    for (unsigned int i = 0; i < 16; ++i) {
                        output[i] += factors[row] * kv_tile[row][lane_id * 16 + i];
                    }
                }
            }
            __syncthreads();
        }
    }

    if (sinks != nullptr) denominator += __expf(sinks[q_head] - maximum);
    const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    unsigned int* out = reinterpret_cast<unsigned int*>(
        O + qo_row_base + (unsigned long long)q_head * MLA_HG_DIM + lane_id * 16);
#pragma unroll
    for (unsigned int i = 0; i < 8; ++i) {
        const unsigned int lo = (unsigned int)__bfloat16_as_ushort(
            __float2bfloat16(output[2 * i] * inverse));
        const unsigned int hi = (unsigned int)__bfloat16_as_ushort(
            __float2bfloat16(output[2 * i + 1] * inverse));
        out[i] = lo | (hi << 16);
    }
}

#define MLA_PAGED_DECODE_FP8_HEADS8_PARAMS                                    \
    const __nv_bfloat16* __restrict__ Q,                                      \
    const unsigned char* __restrict__ K_cache,                                \
    const unsigned char* __restrict__ V_cache,                                \
    __nv_bfloat16* __restrict__ O, const int* __restrict__ block_tables,      \
    const int* __restrict__ seq_lens, const unsigned int max_blocks_per_seq,  \
    const unsigned int num_q_heads, const unsigned int num_kv_heads,          \
    const unsigned int q_head_dim, const unsigned int kv_cache_dim,           \
    const unsigned int block_size, const float inv_sqrt_d,                    \
    const float k_scale, const float v_scale,                                 \
    const unsigned long long cache_stride_bytes,                              \
    const unsigned int sliding_window, const float* __restrict__ sinks,       \
    const unsigned char* __restrict__ comp_pool,                              \
    const unsigned int* __restrict__ comp_block_count_ptr,                    \
    const unsigned int comp_ratio

#define MLA_PAGED_DECODE_FP8_HEADS8_ARGS                                      \
    Q, K_cache, V_cache, O, block_tables, seq_lens, max_blocks_per_seq,       \
    num_q_heads, num_kv_heads, q_head_dim, kv_cache_dim, block_size,          \
    inv_sqrt_d, k_scale, v_scale, cache_stride_bytes, sliding_window, sinks,  \
    comp_pool, comp_block_count_ptr, comp_ratio

extern "C" __global__ __launch_bounds__(256, 1)
void mla_paged_decode_fp8_heads8(MLA_PAGED_DECODE_FP8_HEADS8_PARAMS) {
    mla_paged_decode_fp8_heads8_impl<false>(MLA_PAGED_DECODE_FP8_HEADS8_ARGS);
}

extern "C" __global__ __launch_bounds__(256, 1)
void mla_paged_decode_fp8_heads8_kvalias(MLA_PAGED_DECODE_FP8_HEADS8_PARAMS) {
    mla_paged_decode_fp8_heads8_impl<true>(MLA_PAGED_DECODE_FP8_HEADS8_ARGS);
}
