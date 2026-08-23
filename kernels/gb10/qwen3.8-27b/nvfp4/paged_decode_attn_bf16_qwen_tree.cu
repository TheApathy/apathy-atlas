// SPDX-License-Identifier: AGPL-3.0-only

// Qwen-only BF16 paged decode with DDTree ancestor indirection.
// This is deliberately a separate ABI from common/paged_decode_attn.cu:
// Gemma-4 owns the shared BF16 kernel's trailing sliding_window argument,
// while Qwen tree verify needs three graph-stable indirection arguments.
// Grid: (num_q_heads, num_seqs, 1), block: (256, 1, 1).
#include <cuda_bf16.h>

#define WARP_SIZE 32
#define NUM_WARPS 8
#define HDIM 256
#define VEC_BF16 (HDIM / WARP_SIZE)
#define VEC_U32 (VEC_BF16 / 2)
#define BC 4
__device__ __forceinline__ void unpack_bf16_pair(
    unsigned int packed, float& lo, float& hi
) {
    lo = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xffff)));
    hi = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}
extern "C" __global__ void paged_decode_attn_bf16_qwen_tree(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ out,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const int* __restrict__ kv_indirection,
    const int* __restrict__ kv_indir_base_ptr,
    const unsigned int kv_indir_stride
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;

    if (q_head >= num_q_heads || head_dim != HDIM || kv_indirection == nullptr
        || kv_indir_base_ptr == nullptr || kv_indir_stride == 0) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    const unsigned int base =
        (unsigned int)(*((const volatile int*)kv_indir_base_ptr));
    if (seq_len == 0 || base > seq_len || seq_len - base > kv_indir_stride) return;

    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);
    const unsigned int vec_offset = lane * VEC_BF16;
    const unsigned long long token_stride =
        (unsigned long long)num_kv_heads * head_dim;
    const unsigned long long page_stride =
        (unsigned long long)block_size * token_stride;
    const int* block_table = block_tables + seq_idx * max_blocks_per_seq;
    const int* indir = kv_indirection +
        (unsigned long long)seq_idx * kv_indir_stride;

    const unsigned int* q32 = (const unsigned int*)(q
        + (unsigned long long)seq_idx * q_stride
        + (unsigned long long)q_head * head_dim + vec_offset);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; ++i)
        unpack_bf16_pair(q32[i], q_reg[2 * i], q_reg[2 * i + 1]);

    const unsigned int chunk = (seq_len + NUM_WARPS - 1) / NUM_WARPS;
    const unsigned int raw_start = warp * chunk;
    const unsigned int start = raw_start < seq_len ? raw_start : seq_len;
    const unsigned int raw_end = start + chunk;
    const unsigned int end = raw_end < seq_len ? raw_end : seq_len;
    float max_score = -1.0e30f;
    float denom = 0.0f;
    float accum[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; ++i) accum[i] = 0.0f;

    unsigned int logical_pos = start;
    while (logical_pos < end) {
        // Preserve the shared BF16 kernel's BC=4 fast path over the long,
        // ordinary prefix. Never let a batch cross the tree boundary or a
        // physical cache block; the short indirect tree window falls through
        // to the one-position path below.
        if (logical_pos < base) {
            const unsigned int block_offset = logical_pos % block_size;
            unsigned int available = block_size - block_offset;
            const unsigned int to_end = end - logical_pos;
            const unsigned int to_tree = base - logical_pos;
            if (available > to_end) available = to_end;
            if (available > to_tree) available = to_tree;
            if (available >= BC) {
                const unsigned int physical_block =
                    (unsigned int)block_table[logical_pos / block_size];
                const unsigned long long cache_offset =
                    (unsigned long long)physical_block * page_stride
                    + (unsigned long long)block_offset * token_stride
                    + (unsigned long long)kv_head * head_dim + vec_offset;
                unsigned int packed_k[BC][VEC_U32];
                unsigned int packed_v[BC][VEC_U32];
                #pragma unroll
                for (int b = 0; b < BC; ++b) {
                    const unsigned int* k32 = (const unsigned int*)(
                        k_cache + cache_offset + (unsigned long long)b * token_stride);
                    const unsigned int* v32 = (const unsigned int*)(
                        v_cache + cache_offset + (unsigned long long)b * token_stride);
                    #pragma unroll
                    for (int i = 0; i < VEC_U32; ++i) {
                        packed_k[b][i] = k32[i];
                        packed_v[b][i] = v32[i];
                    }
                }

                float scores[BC];
                #pragma unroll
                for (int b = 0; b < BC; ++b) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VEC_U32; ++i) {
                        float k0, k1;
                        unpack_bf16_pair(packed_k[b][i], k0, k1);
                        dot += q_reg[2 * i] * k0 + q_reg[2 * i + 1] * k1;
                    }
                    #pragma unroll
                    for (int delta = WARP_SIZE / 2; delta > 0; delta >>= 1)
                        dot += __shfl_xor_sync(0xffffffff, dot, delta);
                    scores[b] = dot * inv_sqrt_d;
                }

                float next_max = max_score;
                #pragma unroll
                for (int b = 0; b < BC; ++b) next_max = fmaxf(next_max, scores[b]);
                const float old_scale = __expf(max_score - next_max);
                denom *= old_scale;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; ++i) accum[i] *= old_scale;
                #pragma unroll
                for (int b = 0; b < BC; ++b) {
                    const float weight = __expf(scores[b] - next_max);
                    denom += weight;
                    #pragma unroll
                    for (int i = 0; i < VEC_U32; ++i) {
                        float v0, v1;
                        unpack_bf16_pair(packed_v[b][i], v0, v1);
                        accum[2 * i] += weight * v0;
                        accum[2 * i + 1] += weight * v1;
                    }
                }
                max_score = next_max;
                logical_pos += BC;
                continue;
            }
        }

        unsigned int actual_pos = logical_pos;
        if (logical_pos >= base) {
            actual_pos = base + (unsigned int)indir[logical_pos - base];
        }
        const unsigned int logical_block = actual_pos / block_size;
        const unsigned int block_offset = actual_pos % block_size;
        const unsigned int physical_block = (unsigned int)block_table[logical_block];
        const unsigned long long cache_offset =
            (unsigned long long)physical_block * page_stride
            + (unsigned long long)block_offset * token_stride
            + (unsigned long long)kv_head * head_dim + vec_offset;
        const unsigned int* k32 = (const unsigned int*)(k_cache + cache_offset);

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_U32; ++i) {
            float k0, k1;
            unpack_bf16_pair(k32[i], k0, k1);
            dot += q_reg[2 * i] * k0 + q_reg[2 * i + 1] * k1;
        }
        #pragma unroll
        for (int delta = WARP_SIZE / 2; delta > 0; delta >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, delta);

        const float score = dot * inv_sqrt_d;
        const float next_max = fmaxf(max_score, score);
        const float old_scale = __expf(max_score - next_max);
        const float new_scale = __expf(score - next_max);
        denom = denom * old_scale + new_scale;

        const unsigned int* v32 = (const unsigned int*)(v_cache + cache_offset);
        #pragma unroll
        for (int i = 0; i < VEC_U32; ++i) {
            float v0, v1;
            unpack_bf16_pair(v32[i], v0, v1);
            accum[2 * i] = accum[2 * i] * old_scale + new_scale * v0;
            accum[2 * i + 1] = accum[2 * i + 1] * old_scale + new_scale * v1;
        }
        max_score = next_max;
        ++logical_pos;
    }

    __shared__ float warp_max[NUM_WARPS];
    __shared__ float warp_denom[NUM_WARPS];
    __shared__ float warp_accum[NUM_WARPS][HDIM];
    if (lane == 0) {
        warp_max[warp] = max_score;
        warp_denom[warp] = denom;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; ++i)
        warp_accum[warp][vec_offset + i] = accum[i];
    __syncthreads();

    #pragma unroll
    for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp < (unsigned int)stride) {
            const unsigned int other = warp + stride;
            const float other_denom = warp_denom[other];
            if (other_denom > 0.0f) {
                const float merged_max = fmaxf(warp_max[warp], warp_max[other]);
                const float self_scale = __expf(warp_max[warp] - merged_max);
                const float other_scale = __expf(warp_max[other] - merged_max);
                warp_denom[warp] =
                    warp_denom[warp] * self_scale + other_denom * other_scale;
                warp_max[warp] = merged_max;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; ++i) {
                    const unsigned int d = vec_offset + i;
                    warp_accum[warp][d] = warp_accum[warp][d] * self_scale
                        + warp_accum[other][d] * other_scale;
                }
            }
        }
        __syncthreads();
    }

    if (warp == 0) {
        const float inv_denom = warp_denom[0] > 0.0f ? 1.0f / warp_denom[0] : 0.0f;
        unsigned int* out32 = (unsigned int*)(out
            + (unsigned long long)seq_idx * num_q_heads * head_dim
            + (unsigned long long)q_head * head_dim + vec_offset);
        #pragma unroll
        for (int i = 0; i < VEC_U32; ++i) {
            const unsigned int lo = (unsigned int)__bfloat16_as_ushort(
                __float2bfloat16(warp_accum[0][vec_offset + 2 * i] * inv_denom));
            const unsigned int hi = (unsigned int)__bfloat16_as_ushort(
                __float2bfloat16(warp_accum[0][vec_offset + 2 * i + 1] * inv_denom));
            out32[i] = lo | (hi << 16);
        }
    }
}
