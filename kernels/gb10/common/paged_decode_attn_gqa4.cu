// SPDX-License-Identifier: AGPL-3.0-only
// GQA-fused twin of paged_decode_attn: one CTA per (group of HPC=4 q heads, row).
// Every per-head operation, operand order, batching (BC), warp split and
// reduction tree is identical to paged_decode_attn; only the K/V loads are
// shared across the HPC heads, so results are bit-identical. Grid: (num_q_heads/4, num_seqs).
#include <cuda_bf16.h>

#define WARP_SIZE 32
#ifndef HDIM
#define HDIM 256
#endif
#define VEC_BF16 (HDIM / WARP_SIZE)
#define VEC_U32  (HDIM / (WARP_SIZE * 2))
#define NUM_WARPS 8
#define BC 4            // KV positions batched per loop iteration

__device__ __forceinline__ void unpack2_pd(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

// Helper: compute pointer to K or V for a given position in paged cache
__device__ __forceinline__ const __nv_bfloat16* paged_kv_ptr(
    const __nv_bfloat16* __restrict__ cache,   // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_table,       // [max_blocks_per_seq]
    unsigned int pos,
    unsigned int block_size,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int kv_head
) {
    unsigned int logical_block = pos / block_size;
    unsigned int block_offset = pos % block_size;
    unsigned int physical_block = (unsigned int)block_table[logical_block];
    unsigned long long page_stride = (unsigned long long)block_size * num_kv_heads * head_dim;
    return cache + (unsigned long long)physical_block * page_stride
                 + (unsigned long long)block_offset * num_kv_heads * head_dim
                 + (unsigned long long)kv_head * head_dim;
}

#define HPC 4  // q heads per CTA (all share one kv head)
extern "C" __global__ void __launch_bounds__(256, 1) paged_decode_attn_gqa4(
    const __nv_bfloat16* __restrict__ Q,          // [num_seqs, num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ K_cache,    // [num_blocks, block_size, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ V_cache,    // [num_blocks, block_size, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ O,                // [num_seqs, num_q_heads, head_dim]
    const int* __restrict__ block_tables,         // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,             // [num_seqs]
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,              // query.stride(0) in elements
    const unsigned int sliding_window         // 0 = full attention; >0 = only attend to last `sliding_window` KV positions (Gemma-4 hybrid attn)
) {
    const unsigned int q_head0 = blockIdx.x * HPC;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head0 + HPC > num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Sliding-window start position. For Gemma-4 sliding layers with
    // window=1024, we mask out KV positions older than seq_len - 1024.
    // When sliding_window == 0 (full attention) or seq_len fits inside
    // the window, window_start = 0 (no masking).
    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head0 / gqa_ratio;
    if ((q_head0 + HPC - 1) / gqa_ratio != kv_head) return;  // all HPC heads must share the kv head
    const unsigned int vec_offset = lane_id * VEC_BF16;

    // Block table for this sequence
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q into registers (strided: Q may be a non-contiguous QKV split view)
    float q_reg[HPC][VEC_BF16];
    #pragma unroll
    for (int h = 0; h < HPC; h++) {
        const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                           + (unsigned long long)(q_head0 + h) * head_dim + vec_offset);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            unpack2_pd(q32[i], q_reg[h][2*i], q_reg[h][2*i+1]);
        }
    }

    // Each warp handles a chunk of the KV sequence. Split across the
    // ATTENDED range [window_start, seq_len) rather than the raw [0, seq_len)
    // so warps aren't wasted on positions masked out by the sliding window.
    const unsigned int attended = seq_len - window_start;
    unsigned int chunk_size = (attended + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = window_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    // Online softmax state
    float m[HPC], l[HPC];
    float o_reg[HPC][VEC_BF16];
    #pragma unroll
    for (int h = 0; h < HPC; h++) {
        m[h] = -1e30f; l[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) o_reg[h][i] = 0.0f;
    }

    // === Main loop: process positions with batched KV loading ===
    // We can batch BC=4 positions when they're in the same physical block.
    // At block boundaries, fall back to single-position processing.
    unsigned int pos = my_start;

    while (pos < my_end) {
        // Check how many consecutive positions share the same physical block
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = remaining_in_block < remaining_total ? remaining_in_block : remaining_total;

        // Get physical block pointer base
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        unsigned long long page_stride = (unsigned long long)block_size * num_kv_heads * head_dim;
        unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;
        const __nv_bfloat16* k_block_base = K_cache + (unsigned long long)physical_block * page_stride
                                                     + (unsigned long long)block_offset * head_stride_kv
                                                     + (unsigned long long)kv_head * head_dim;
        const __nv_bfloat16* v_block_base = V_cache + (unsigned long long)physical_block * page_stride
                                                     + (unsigned long long)block_offset * head_stride_kv
                                                     + (unsigned long long)kv_head * head_dim;

        // Process in batches of BC within this physical block
        unsigned int processed = 0;
        unsigned int aligned_count = (batch_count / BC) * BC;

        // Batched path: BC=4 positions at a time (contiguous in memory)
        for (; processed < aligned_count; processed += BC) {
            // Load BC K vectors
            unsigned int k_packed[BC][VEC_U32];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* k32 = (const unsigned int*)(k_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset);
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++)
                    k_packed[b][i] = k32[i];
            }

            // Compute BC dot products for every head (same order per head)
            float scores[HPC][BC];
            #pragma unroll
            for (int h = 0; h < HPC; h++) {
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VEC_U32; i++) {
                        float k0, k1;
                        unpack2_pd(k_packed[b][i], k0, k1);
                        dot += q_reg[h][2*i] * k0 + q_reg[h][2*i+1] * k1;
                    }
                    #pragma unroll
                    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                        dot += __shfl_xor_sync(0xffffffff, dot, offset);
                    scores[h][b] = dot * inv_sqrt_d;
                }
            }

            // Prefetch V
            unsigned int v_packed[BC][VEC_U32];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* v32 = (const unsigned int*)(v_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset);
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++)
                    v_packed[b][i] = v32[i];
            }

            #pragma unroll
            for (int h = 0; h < HPC; h++) {
                // Batched softmax
                float m_new = m[h];
                #pragma unroll
                for (int b = 0; b < BC; b++)
                    m_new = fmaxf(m_new, scores[h][b]);

                float exp_old = __expf(m[h] - m_new);
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++)
                    o_reg[h][i] *= exp_old;
                l[h] *= exp_old;

                float exp_factors[BC];
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    exp_factors[b] = __expf(scores[h][b] - m_new);
                    l[h] += exp_factors[b];
                }
                m[h] = m_new;

                // V accumulate
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    float ef = exp_factors[b];
                    #pragma unroll
                    for (int i = 0; i < VEC_U32; i++) {
                        float v0, v1;
                        unpack2_pd(v_packed[b][i], v0, v1);
                        o_reg[h][2*i]   += ef * v0;
                        o_reg[h][2*i+1] += ef * v1;
                    }
                }
            }
        }

        // Remainder: single positions
        for (; processed < batch_count; processed++) {
            const unsigned int* k32 = (const unsigned int*)(k_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset);
            unsigned int k1p[VEC_U32], v1p[VEC_U32];
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) k1p[i] = k32[i];
            const unsigned int* v32 = (const unsigned int*)(v_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) v1p[i] = v32[i];
            #pragma unroll
            for (int h = 0; h < HPC; h++) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) {
                    float k0, k1;
                    unpack2_pd(k1p[i], k0, k1);
                    dot += q_reg[h][2*i] * k0 + q_reg[h][2*i+1] * k1;
                }
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);

                float score = dot * inv_sqrt_d;
                float m_new = fmaxf(m[h], score);
                float exp_old = __expf(m[h] - m_new);
                float exp_new = __expf(score - m_new);
                l[h] = l[h] * exp_old + exp_new;

                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) {
                    float v0, v1;
                    unpack2_pd(v1p[i], v0, v1);
                    o_reg[h][2*i]   = o_reg[h][2*i]   * exp_old + exp_new * v0;
                    o_reg[h][2*i+1] = o_reg[h][2*i+1] * exp_old + exp_new * v1;
                }
                m[h] = m_new;
            }
        }

        pos += batch_count;
    }

    // === Tree-based inter-warp reduction (per head, same tree as the base kernel) ===
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    #pragma unroll 1
    for (int h = 0; h < HPC; h++) {
        if (h > 0) __syncthreads();
        if (lane_id == 0) {
            smem_m[warp_id] = m[h];
            smem_l[warp_id] = l[h];
        }
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) {
            smem_o[warp_id][vec_offset + i] = o_reg[h][i];
        }
        __syncthreads();

        #pragma unroll
        for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
            if (warp_id < (unsigned int)stride) {
                unsigned int other = warp_id + stride;
                float lw = smem_l[other];
                if (lw > 0.0f) {
                    float mw = smem_m[other];
                    float my_m = smem_m[warp_id];
                    float my_l = smem_l[warp_id];
                    float m_new = fmaxf(my_m, mw);
                    float scale_me = __expf(my_m - m_new);
                    float scale_w = __expf(mw - m_new);
                    smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                    smem_m[warp_id] = m_new;
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++) {
                        smem_o[warp_id][vec_offset + i] =
                            smem_o[warp_id][vec_offset + i] * scale_me +
                            smem_o[other][vec_offset + i] * scale_w;
                    }
                }
            }
            __syncthreads();
        }

        if (warp_id == 0) {
            float final_l = smem_l[0];
            float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
            unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                                  + (unsigned long long)(q_head0 + h) * head_dim + vec_offset);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                float v0 = smem_o[0][vec_offset + 2*i]     * inv_l;
                float v1 = smem_o[0][vec_offset + 2*i + 1] * inv_l;
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
                o32[i] = lo | (hi << 16);
            }
        }
    }
}
