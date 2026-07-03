// SPDX-License-Identifier: AGPL-3.0-only

// Atlas WY-Chunkwise Gated Delta Rule — K=17 verify, V-DIM SPLIT variant.
//
// Occupancy play for the DFlash γ+1 verify path. The baseline
// gated_delta_rule_wy17 launches grid=(num_v_heads=48, batch=1) = 48 CTAs
// of 128 threads on GB10's 48-SM class — exactly 1 CTA/SM, so each SM runs
// only 4 warps (12.5% of the 32-warp cap) with no second block to hide the
// long PASS-1/PASS-2 H-state streaming latency. This kernel splits the
// v_dim columns across gridDim.z, multiplying the CTA count (48 → 48*V_SPLIT)
// so each SM hosts V_SPLIT resident blocks and can overlap memory stalls.
//
// Correctness: each v_dim column of the state H[k_dim, v_dim] evolves
// INDEPENDENTLY — hk[t] = H·k_t, vn[t], and the state update H_new = g·H +
// k·vn are all computed per-(tid = v-column). The only cross-column shared
// work is kd_flat (the K*(K-1)/2 = 136 inter-token k-dot products, which
// depend only on k, not on v_dim); each split recomputes it locally over
// its own 128 threads. That recompute is the price of the extra occupancy
// (136 block-reductions), traded against hiding the H-state DRAM latency of
// the two k_dim=128 streaming passes. Output/H/Hi writes are disjoint per
// column band ⇒ no races, and the produced values are BIT-IDENTICAL to
// wy17 (same FP32 math, same reduction order within a column).
//
// Grid: (num_v_heads, batch, v_split)   Block: (128, 1, 1)
// Each CTA (vh, b, zsplit) owns v columns [zsplit*v_chunk, (zsplit+1)*v_chunk).

#include <cuda_bf16.h>
#include "../../common/gdn_reduce.cuh"
#define BLOCK_SIZE 128
#define K_TOKENS 17

extern "C" __global__ void gated_delta_rule_wy17_vsplit(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ h_state_inter_base,
    unsigned int inter_stride_floats,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int v_split          // number of v_dim column bands (gridDim.z)
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int zsplit = blockIdx.z;
    if (vh >= num_v_heads || b >= batch_size || zsplit >= v_split) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;

    // This CTA's v-column band: [v_off, v_off + v_chunk).
    const unsigned int v_chunk = (v_dim + v_split - 1) / v_split;
    const unsigned int v_off = zsplit * v_chunk;
    if (v_off >= v_dim) return;
    // vcol = absolute v-column this thread owns (or >= v_dim if past the band/tail).
    const unsigned int vcol = v_off + tid;
    const bool active = (tid < v_chunk) && (vcol < v_dim);

    float* H = h_state + ((b * num_v_heads + vh) * hv);
    float* Hi_base = h_state_inter_base + ((b * num_v_heads + vh) * hv);

    __shared__ float sk[K_TOKENS][128];
    __shared__ float sq[K_TOKENS][128];
    __shared__ float sg[K_TOKENS];
    __shared__ float sbt[K_TOKENS];
    __shared__ float smem_warp[4];

    if (tid < k_dim) {
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            const __nv_bfloat16* q_t = query + (b * K_TOKENS + t) * qk_stride + kh * k_dim;
            const __nv_bfloat16* k_t = key   + (b * K_TOKENS + t) * qk_stride + kh * k_dim;
            sq[t][tid] = (float)q_t[tid];
            sk[t][tid] = (float)k_t[tid];
        }
    }
    if (tid < K_TOKENS) {
        float g_raw = gate[(b * K_TOKENS + tid) * gb_stride + vh];
        sg[tid] = fminf(fmaxf(g_raw, 0.0f), 1.0f);
        sbt[tid] = beta[(b * K_TOKENS + tid) * gb_stride + vh];
    }
    __syncthreads();

    // ── kd_flat: K*(K-1)/2 = 136 inter-token k-dots (recomputed per split) ──
    __shared__ float kd_flat[K_TOKENS * (K_TOKENS - 1) / 2];
    #pragma unroll
    for (int t = 1; t < K_TOKENS; t++) {
        #pragma unroll
        for (int s = 0; s < t; s++) {
            float p = (tid < k_dim) ? sk[t][tid] * sk[s][tid] : 0.0f;
            float r = atlas_block_reduce_sum(p, smem_warp, tid);
            if (tid == 0) {
                kd_flat[t * (t - 1) / 2 + s] = r;
            }
            __syncthreads();
        }
    }

    if (active) {
        // Load v[K] for this thread's v-column (vcol).
        float vi[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            const __nv_bfloat16* v_t = value + (b * K_TOKENS + t) * v_stride + vh * v_dim;
            vi[t] = (float)v_t[vcol];
        }

        // PASS 1: hk[t] = H · k_t (H column `vcol`).
        float hk[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) hk[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + vcol];
            float h1 = H[(j + 1) * v_dim + vcol];
            float h2 = H[(j + 2) * v_dim + vcol];
            float h3 = H[(j + 3) * v_dim + vcol];
            #pragma unroll
            for (int t = 0; t < K_TOKENS; t++) {
                hk[t] += h0 * sk[t][j + 0] + h1 * sk[t][j + 1]
                       + h2 * sk[t][j + 2] + h3 * sk[t][j + 3];
            }
        }

        // WY correction (sequential over K tokens), per-column.
        float vn[K_TOKENS];
        vn[0] = (vi[0] - sg[0] * hk[0]) * sbt[0];
        for (int t = 1; t < K_TOKENS; t++) {
            float lead_prod = 1.0f;
            for (int u = 0; u < t; u++) lead_prod *= sg[u];
            float corrected = lead_prod * hk[t];
            for (int s = 0; s < t; s++) {
                float gprod = 1.0f;
                for (int u = s + 1; u < t; u++) gprod *= sg[u];
                corrected += gprod * kd_flat[t * (t - 1) / 2 + s] * vn[s];
            }
            vn[t] = (vi[t] - sg[t] * corrected) * sbt[t];
        }

        // PASS 2: apply K state updates, write intermediates + final H + qd.
        float qd[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) qd[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + vcol];
            float h1 = H[(j + 1) * v_dim + vcol];
            float h2 = H[(j + 2) * v_dim + vcol];
            float h3 = H[(j + 3) * v_dim + vcol];

            #pragma unroll
            for (int t = 0; t < K_TOKENS; t++) {
                h0 = sg[t] * h0 + sk[t][j + 0] * vn[t];
                h1 = sg[t] * h1 + sk[t][j + 1] * vn[t];
                h2 = sg[t] * h2 + sk[t][j + 2] * vn[t];
                h3 = sg[t] * h3 + sk[t][j + 3] * vn[t];
                if (t < K_TOKENS - 1) {
                    float* Hi_t = Hi_base + t * inter_stride_floats;
                    Hi_t[(j + 0) * v_dim + vcol] = h0;
                    Hi_t[(j + 1) * v_dim + vcol] = h1;
                    Hi_t[(j + 2) * v_dim + vcol] = h2;
                    Hi_t[(j + 3) * v_dim + vcol] = h3;
                } else {
                    H[(j + 0) * v_dim + vcol] = h0;
                    H[(j + 1) * v_dim + vcol] = h1;
                    H[(j + 2) * v_dim + vcol] = h2;
                    H[(j + 3) * v_dim + vcol] = h3;
                }
                qd[t] += h0 * sq[t][j + 0] + h1 * sq[t][j + 1]
                       + h2 * sq[t][j + 2] + h3 * sq[t][j + 3];
            }
        }

        float s = rsqrtf((float)k_dim);
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            output[((b * K_TOKENS + t) * num_v_heads + vh) * v_dim + vcol] =
                __float2bfloat16(qd[t] * s);
        }
    }
}
