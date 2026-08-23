// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_bf16.h>
#include <math.h>

#define WARP_SIZE 32
#define NUM_WARPS 8
#define HDIM 256
#define VEC_BF16 (HDIM / WARP_SIZE)
#define VEC_U32 (VEC_BF16 / 2)
#define BC 2

__device__ __forceinline__ void unpack2_bf16_kg(
    unsigned int packed, float& a, float& b) {
    a = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xffff)));
    b = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

extern "C" __global__ void paged_decode_attn_kgamma_bf16_shared_kv(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned long long cache_stride,
    const unsigned int num_qtile
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int global_warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int group = global_warp / NUM_WARPS;
    const unsigned int warp_id = global_warp % NUM_WARPS;
    const unsigned int query = blockIdx.y * 2 + group;

    if (q_head >= num_q_heads || num_qtile < 8 || num_qtile > 32 || head_dim != HDIM) return;
    const bool active = query < num_qtile;
    const unsigned int seq_len = active ? (unsigned int)seq_lens[query] : 0u;
    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset = lane_id * VEC_BF16;
    const unsigned long long token_stride = (unsigned long long)num_kv_heads * HDIM;
    const int* table = block_tables;

    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)(active ? query : 0u) * q_stride
        + (unsigned long long)q_head * HDIM + vec_offset);
    float qr[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) unpack2_bf16_kg(q32[i], qr[2*i], qr[2*i+1]);

    unsigned int q0_len = (blockIdx.y * 2 < num_qtile) ? (unsigned int)seq_lens[blockIdx.y * 2] : 0u;
    unsigned int q1_len = (blockIdx.y * 2 + 1 < num_qtile) ? (unsigned int)seq_lens[blockIdx.y * 2 + 1] : 0u;
    unsigned int max_len = max(q0_len, q1_len);

    unsigned int chunk = (max_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int begin = min(warp_id * chunk, max_len);
    unsigned int end = min(begin + chunk, max_len);
    float m = -1e30f, l = 0.0f, out[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) out[i] = 0.0f;

    __shared__ unsigned int smem_kp[NUM_WARPS][WARP_SIZE][BC][VEC_U32];
    __shared__ unsigned int smem_vp[NUM_WARPS][WARP_SIZE][BC][VEC_U32];

    unsigned int pos = begin;
    while (pos < end) {
        unsigned int logical = pos / block_size;
        unsigned int off = pos % block_size;
        unsigned int count = min(block_size - off, end - pos);
        unsigned int physical = (unsigned int)table[logical];
        const __nv_bfloat16* kb = K_cache + (unsigned long long)physical * cache_stride
            + (unsigned long long)off * token_stride + (unsigned long long)kv_head * HDIM;
        const __nv_bfloat16* vb = V_cache + (unsigned long long)physical * cache_stride
            + (unsigned long long)off * token_stride + (unsigned long long)kv_head * HDIM;
        unsigned int done = 0, aligned = (count / BC) * BC;
        for (; done < aligned; done += BC) {
            if (group == 0) {
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    const unsigned int* k = (const unsigned int*)(kb + (unsigned long long)(done + b) * token_stride + vec_offset);
                    const unsigned int* v = (const unsigned int*)(vb + (unsigned long long)(done + b) * token_stride + vec_offset);
                    #pragma unroll
                    for (int i = 0; i < VEC_U32; i++) {
                        smem_kp[warp_id][lane_id][b][i] = k[i];
                        smem_vp[warp_id][lane_id][b][i] = v[i];
                    }
                }
            }
            __syncthreads();

            if (active && pos + done < seq_len) {
                float scores[BC];
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VEC_U32; i++) {
                        float k0, k1;
                        unpack2_bf16_kg(smem_kp[warp_id][lane_id][b][i], k0, k1);
                        dot += qr[2*i] * k0 + qr[2*i+1] * k1;
                    }
                    #pragma unroll
                    for (int x = WARP_SIZE / 2; x > 0; x >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, x);
                    scores[b] = dot * inv_sqrt_d;
                }
                float mn = m;
                #pragma unroll
                for (int b = 0; b < BC; b++) mn = fmaxf(mn, scores[b]);
                float eo = __expf(m - mn);
                l *= eo;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) out[i] *= eo;
                float ef[BC];
                #pragma unroll
                for (int b = 0; b < BC; b++) { ef[b] = __expf(scores[b] - mn); l += ef[b]; }
                m = mn;
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    #pragma unroll
                    for (int i = 0; i < VEC_U32; i++) {
                        float v0, v1;
                        unpack2_bf16_kg(smem_vp[warp_id][lane_id][b][i], v0, v1);
                        out[2*i]   += ef[b] * v0;
                        out[2*i+1] += ef[b] * v1;
                    }
                }
            }
            __syncthreads();
        }
        for (; done < count; done++) {
            if (group == 0) {
                const unsigned int* k = (const unsigned int*)(kb + (unsigned long long)done * token_stride + vec_offset);
                const unsigned int* v = (const unsigned int*)(vb + (unsigned long long)done * token_stride + vec_offset);
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) {
                    smem_kp[warp_id][lane_id][0][i] = k[i];
                    smem_vp[warp_id][lane_id][0][i] = v[i];
                }
            }
            __syncthreads();

            if (active && pos + done < seq_len) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) {
                    float k0, k1;
                    unpack2_bf16_kg(smem_kp[warp_id][lane_id][0][i], k0, k1);
                    dot += qr[2*i] * k0 + qr[2*i+1] * k1;
                }
                #pragma unroll
                for (int x = WARP_SIZE / 2; x > 0; x >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, x);
                float score = dot * inv_sqrt_d, mn = fmaxf(m, score), eo = __expf(m - mn), en = __expf(score - mn);
                l = l * eo + en;
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) {
                    float v0, v1;
                    unpack2_bf16_kg(smem_vp[warp_id][lane_id][0][i], v0, v1);
                    out[2*i]   = out[2*i] * eo + en * v0;
                    out[2*i+1] = out[2*i+1] * eo + en * v1;
                }
                m = mn;
            }
            __syncthreads();
        }
        pos += count;
    }

    __shared__ float sm[2][NUM_WARPS], sl[2][NUM_WARPS], so[2][NUM_WARPS][HDIM];
    if (lane_id == 0) { sm[group][warp_id] = m; sl[group][warp_id] = l; }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) so[group][warp_id][vec_offset + i] = out[i];
    __syncthreads();

    #pragma unroll
    for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = sl[group][other];
            if (lw > 0.0f) {
                float mw = sm[group][other], my_m = sm[group][warp_id], my_l = sl[group][warp_id];
                float mn = fmaxf(my_m, mw), sa = __expf(my_m - mn), sb = __expf(mw - mn);
                sl[group][warp_id] = my_l * sa + lw * sb; sm[group][warp_id] = mn;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) so[group][warp_id][vec_offset + i] =
                    so[group][warp_id][vec_offset + i] * sa + so[group][other][vec_offset + i] * sb;
            }
        }
        __syncthreads();
    }

    if (active && warp_id == 0) {
        float final_l = sl[group][0];
        float il = final_l > 0.0f ? 1.0f / final_l : 0.0f;
        unsigned int* o = (unsigned int*)(O + (unsigned long long)query * num_q_heads * HDIM + (unsigned long long)q_head * HDIM + vec_offset);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            unsigned int lo = __bfloat16_as_ushort(__float2bfloat16(so[group][0][vec_offset + 2*i] * il));
            unsigned int hi = __bfloat16_as_ushort(__float2bfloat16(so[group][0][vec_offset + 2*i + 1] * il));
            o[i] = lo | (hi << 16);
        }
    }
}
