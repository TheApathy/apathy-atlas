// SPDX-License-Identifier: AGPL-3.0-only

// Exact shadow of gated_delta_rule_prefill_wy32_gatecache (v2): same arithmetic,
// same reduction trees and the same statement order per output; adds FP32
// transposed K/Q chunk copies (vector loads) and batches 8 independent pair
// reductions per step so their shuffle chains overlap. Needs +36 KB of dynamic smem.
//
// The parent WY32 kernel computes every gate prefix/product independently in
// all 128 threads even though those coefficients are identical across V
// columns. This shadow computes each coefficient once, in the same
// left-to-right FP32 multiplication order, and stores it in the unused
// diagonal/upper triangle of smem_kd. The lower triangle continues to hold
// k_i^T k_j. H, K, Q, WY correction, state update, output accumulation, and
// BF16 conversion remain statement-for-statement equivalent to the parent.

#include <cuda_bf16.h>

#define K_DIM 128
#define V_DIM 128
#define C 32
#define WARP_COUNT 4
#define KD_PAIR_COUNT 496
#define KT_STRIDE 36

static_assert(KD_PAIR_COUNT == C * (C - 1) / 2, "WY32 strict-lower pair count drift");

__device__ __forceinline__ float gatecache_v2_warp_reduce(float val) {
    for (int offset = 16; offset >= 1; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_wy32_gatecache_v2(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const float inv_sqrt_d = rsqrtf((float)k_dim);

    extern __shared__ char smem_raw[];
    float* H_smem = (float*)smem_raw;
    __nv_bfloat16* smem_k = (__nv_bfloat16*)(smem_raw + K_DIM * V_DIM * 4);
    __nv_bfloat16* smem_q = smem_k + C * K_DIM;
    float* smem_warp = (float*)(smem_q + C * K_DIM);
    float* smem_kd = smem_warp + 4;
    float* smem_g = smem_kd + C * C;
    float* smem_bt = smem_g + C;
    // Four parent-order warp partials for every strict-lower K-dot pair.
    // The host reserves KD_PAIR_COUNT * WARP_COUNT FP32 values after smem_bt.
    float* smem_dot_partials = smem_bt + C;
    // One invariant packed (i,j) entry per strict-lower pair. Building this
    // once avoids making every thread rescan all 496 triangular coordinates
    // after each chunk's partial-dot barrier.
    unsigned short* smem_pair_ij = (unsigned short*)
        (smem_dot_partials + KD_PAIR_COUNT * WARP_COUNT);
    // Exact-value FP32 transposed copies of this chunk's K and Q ([j][t], row
    // stride KT_STRIDE floats, 16-byte aligned) so the j-major loops read 32
    // consecutive t values with vector loads instead of 32 BF16 loads + cvt.
    // BF16 -> FP32 conversion is exact, so every product below sees the same
    // FP32 operand the parent computed with (float)smem_k[...].
    float* smem_kT = (float*)(((unsigned long long)(smem_pair_ij + KD_PAIR_COUNT) + 15ull) & ~15ull);
    float* smem_qT = smem_kT + K_DIM * KT_STRIDE;

    float* H_global = h_state
        + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += V_DIM) {
        H_smem[i] = H_global[i];
    }
    for (unsigned int pair = tid; pair < KD_PAIR_COUNT; pair += V_DIM) {
        unsigned int i = 1;
        unsigned int row_start = 0;
        while (pair >= row_start + i) {
            row_start += i;
            i++;
        }
        const unsigned int j = pair - row_start;
        smem_pair_ij[pair] = (unsigned short)((i << 8) | j);
    }
    __syncthreads();

    const unsigned int wy_end = (seq_len / C) * C;
    for (unsigned int chunk_start = 0; chunk_start < wy_end; chunk_start += C) {
        for (unsigned int idx = tid; idx < C * K_DIM; idx += V_DIM) {
            const unsigned int tok = idx / K_DIM;
            const unsigned int dim = idx % K_DIM;
            const unsigned long long off =
                (unsigned long long)(chunk_start + tok) * qk_stride + kh * k_dim + dim;
            const __nv_bfloat16 kb = key[off];
            const __nv_bfloat16 qb = query[off];
            smem_k[tok * K_DIM + dim] = kb;
            smem_q[tok * K_DIM + dim] = qb;
            smem_kT[dim * KT_STRIDE + tok] = (float)kb;
            smem_qT[dim * KT_STRIDE + tok] = (float)qb;
        }
        if (tid < C) {
            smem_g[tid] = gate[(unsigned long long)(chunk_start + tid) * gb_stride + vh];
            smem_bt[tid] = beta[(unsigned long long)(chunk_start + tid) * gb_stride + vh];
        }

        // Compute every strict-lower K-dot pair without a CTA rendezvous per
        // pair or before the dot phase. Each lane reads only the 32 K values
        // it just wrote at its own dimension; no cross-thread K visibility is
        // needed until the warp-local products have reached registers. Each
        // warp retains the parent's exact shuffle tree, and lane zero stores
        // all four parent-order partials into disjoint locations.
        const unsigned int warp_id = tid / 32;
        const unsigned int lane_id = tid % 32;
        // Same per-pair arithmetic and the same shfl_down tree as the parent
        // (offsets 16,8,4,2,1 on lane products k_i[tid]*k_j[tid]); the only
        // change is that PAIRS_PER_STEP independent pairs are in flight at once
        // so their shuffle chains overlap instead of serialising 496 times.
        // Pair -> (i,j) comes from the invariant packed map built above; the
        // map and the parent's (i,j) loop enumerate pairs in the same order.
        {
            #define PAIRS_PER_STEP 8
            for (int p0 = 0; p0 < KD_PAIR_COUNT; p0 += PAIRS_PER_STEP) {
                float part[PAIRS_PER_STEP];
                #pragma unroll
                for (int q = 0; q < PAIRS_PER_STEP; q++) {
                    const int pair = p0 + q;
                    const unsigned int packed = smem_pair_ij[pair];
                    const unsigned int i = packed >> 8;
                    const unsigned int j = packed & 0xFF;
                    part[q] = (float)smem_k[i * K_DIM + tid] * (float)smem_k[j * K_DIM + tid];
                }
                #pragma unroll
                for (int offset = 16; offset >= 1; offset >>= 1) {
                    #pragma unroll
                    for (int q = 0; q < PAIRS_PER_STEP; q++)
                        part[q] += __shfl_down_sync(0xFFFFFFFF, part[q], offset);
                }
                if (lane_id == 0) {
                    #pragma unroll
                    for (int q = 0; q < PAIRS_PER_STEP; q++)
                        smem_dot_partials[(p0 + q) * WARP_COUNT + warp_id] = part[q];
                }
            }
            #undef PAIRS_PER_STEP
        }
        // Publishes Q/K/gate/beta as well as all dot partials. The input loads
        // and dot products need only thread-local program order before here.
        __syncthreads();

        // Preserve the parent's cross-warp association exactly:
        // ((warp0 + warp1) + warp2) + warp3. Distributed CTA threads only
        // parallelize independent pairs; they do not alter any dot reduction.
        // The one-time packed map gives each thread only 3--4 assigned pairs.
        for (unsigned int pair = tid; pair < KD_PAIR_COUNT; pair += V_DIM) {
            const unsigned int packed = smem_pair_ij[pair];
            const unsigned int i = packed >> 8;
            const unsigned int j = packed & 0xFF;
            const float* partial = &smem_dot_partials[pair * WARP_COUNT];
            smem_kd[i * C + j] =
                partial[0] + partial[1] + partial[2] + partial[3];
        }

        // Cache the exact products used below with rolling left-to-right
        // scans. Diagonal [t,t] stores product(g[0..t)); upper [s,t]
        // stores product(g[s+1..t)). Carrying each prefix forward retains
        // the parent's initialization and multiplication order while avoiding
        // overlapping from-scratch recomputation.
        if (tid == 0) {
            float g_prefix = 1.0f;
            for (unsigned int t = 0; t < C; t++) {
                smem_kd[t * C + t] = g_prefix;
                if (t + 1 < C) g_prefix *= smem_g[t];
            }
        }
        if (tid < C) {
            float g_between = 1.0f;
            for (unsigned int t = tid + 1; t < C; t++) {
                smem_kd[tid * C + t] = g_between;
                if (t + 1 < C) g_between *= smem_g[t];
            }
        }
        // One visibility barrier publishes both final K-dots and cached gate
        // coefficients. Together with the partial barrier above, this replaces
        // the parent's three full-CTA barriers for each of 496 pairs.
        __syncthreads();

        // Gate/beta into registers once per chunk (values identical to the
        // parent's per-use smem reads).
        float g_r[C], bt_r[C];
        #pragma unroll
        for (int t = 0; t < C; t++) { g_r[t] = smem_g[t]; bt_r[t] = smem_bt[t]; }

        float hk_prev[C];
        #pragma unroll
        for (int t = 0; t < C; t++) hk_prev[t] = 0.0f;
        #pragma unroll 4
        for (int j = 0; j < K_DIM; j++) {
            const float h_j = H_smem[j * V_DIM + tid];
            const float4* kr = (const float4*)(smem_kT + j * KT_STRIDE);
            #pragma unroll
            for (int t4 = 0; t4 < C / 4; t4++) {
                const float4 kv = kr[t4];
                hk_prev[4 * t4 + 0] += h_j * kv.x;
                hk_prev[4 * t4 + 1] += h_j * kv.y;
                hk_prev[4 * t4 + 2] += h_j * kv.z;
                hk_prev[4 * t4 + 3] += h_j * kv.w;
            }
        }

        float v_new_arr[C];
        for (int t = 0; t < C; t++) {
            const float v_t = (float)value[
                (unsigned long long)(chunk_start + t) * v_stride + vh * v_dim + tid
            ];
            float hk_corr = smem_kd[t * C + t] * hk_prev[t];
            for (int s = 0; s < t; s++) {
                hk_corr += smem_kd[s * C + t] * smem_kd[t * C + s] * v_new_arr[s];
            }
            v_new_arr[t] = (v_t - g_r[t] * hk_corr) * bt_r[t];
        }

        float o_out[C];
        #pragma unroll
        for (int t = 0; t < C; t++) o_out[t] = 0.0f;
        #pragma unroll 2
        for (int j = 0; j < K_DIM; j++) {
            float h_j = H_smem[j * V_DIM + tid];
            const float4* kr = (const float4*)(smem_kT + j * KT_STRIDE);
            const float4* qr = (const float4*)(smem_qT + j * KT_STRIDE);
            #pragma unroll
            for (int t4 = 0; t4 < C / 4; t4++) {
                const float4 kv = kr[t4];
                const float4 qv = qr[t4];
                h_j = g_r[4*t4+0] * h_j + kv.x * v_new_arr[4*t4+0];
                o_out[4*t4+0] += h_j * qv.x;
                h_j = g_r[4*t4+1] * h_j + kv.y * v_new_arr[4*t4+1];
                o_out[4*t4+1] += h_j * qv.y;
                h_j = g_r[4*t4+2] * h_j + kv.z * v_new_arr[4*t4+2];
                o_out[4*t4+2] += h_j * qv.z;
                h_j = g_r[4*t4+3] * h_j + kv.w * v_new_arr[4*t4+3];
                o_out[4*t4+3] += h_j * qv.w;
            }
            H_smem[j * V_DIM + tid] = h_j;
        }

        for (int t = 0; t < C; t++) {
            const unsigned long long out_off =
                (unsigned long long)(chunk_start + t) * num_v_heads * v_dim
                + vh * v_dim + tid;
            output[out_off] = __float2bfloat16(o_out[t] * inv_sqrt_d);
        }
        __syncthreads();
    }

    // Remainder path is identical to the parent and does not use WY products.
    for (unsigned int t = wy_end; t < seq_len; t++) {
        if (tid < K_DIM) {
            const unsigned long long qk_off =
                (unsigned long long)t * qk_stride + kh * k_dim;
            smem_k[tid] = key[qk_off + tid];
            smem_q[tid] = query[qk_off + tid];
        }
        __syncthreads();

        const float v_i = (float)value[
            (unsigned long long)t * v_stride + vh * v_dim + tid
        ];
        const float g_t = gate[(unsigned long long)t * gb_stride + vh];
        const float bt_t = beta[(unsigned long long)t * gb_stride + vh];

        float hk = 0.0f;
        for (int j = 0; j < K_DIM; j += 4) {
            hk += H_smem[(j+0)*V_DIM+tid]*(float)smem_k[j]
                + H_smem[(j+1)*V_DIM+tid]*(float)smem_k[j+1]
                + H_smem[(j+2)*V_DIM+tid]*(float)smem_k[j+2]
                + H_smem[(j+3)*V_DIM+tid]*(float)smem_k[j+3];
        }
        const float vn = (v_i - g_t * hk) * bt_t;

        float q_dot = 0.0f;
        for (int j = 0; j < K_DIM; j += 4) {
            const float h0 = g_t*H_smem[(j+0)*V_DIM+tid] + (float)smem_k[j]*vn;
            const float h1 = g_t*H_smem[(j+1)*V_DIM+tid] + (float)smem_k[j+1]*vn;
            const float h2 = g_t*H_smem[(j+2)*V_DIM+tid] + (float)smem_k[j+2]*vn;
            const float h3 = g_t*H_smem[(j+3)*V_DIM+tid] + (float)smem_k[j+3]*vn;
            H_smem[(j+0)*V_DIM+tid]=h0; H_smem[(j+1)*V_DIM+tid]=h1;
            H_smem[(j+2)*V_DIM+tid]=h2; H_smem[(j+3)*V_DIM+tid]=h3;
            q_dot += h0*(float)smem_q[j] + h1*(float)smem_q[j+1]
                + h2*(float)smem_q[j+2] + h3*(float)smem_q[j+3];
        }
        const unsigned long long out_off =
            (unsigned long long)t * num_v_heads * v_dim + vh * v_dim + tid;
        output[out_off] = __float2bfloat16(q_dot * inv_sqrt_d);
        __syncthreads();
    }

    #pragma unroll 4
    for (unsigned int i = tid; i < K_DIM * V_DIM; i += V_DIM) {
        H_global[i] = H_smem[i];
    }
}
