// SPDX-License-Identifier: AGPL-3.0-only

// Atlas tree-aware Gated Delta Rule — DDTree M8A kernel (sequential).
//
// Verifies up to T = num_tokens spec tokens whose state-load dependency
// follows a tree topology, not the linear chain assumed by gdn_wy{2,3,17}.
// Each token i loads its INPUT state H_in from h_state (when parent_ids[i]
// == -1) or h_state_inter[parent_ids[i]] (an earlier token's OUTPUT state),
// then writes its own OUTPUT state to h_state_inter[i] for descendants.
//
// Algorithm (per token i, per (vh, b) block):
//   parent = parent_ids[i];
//   H_in   = (parent < 0) ? h_state : h_state_inter[parent];
//   hk_i   = sum_k H_in[k][tid] * k_i[k];                   // single pass
//   v_new  = (v_i[tid] - g_i * hk_i) * beta_i;
//   H_out[k][tid] = g_i * H_in[k][tid] + k_i[k] * v_new;    // write
//   qd_i  += sum_k H_out[k][tid] * q_i[k];
//   output[i][tid] = qd_i * rsqrt(k_dim);
//   h_state_inter[i][k][tid] = H_out[k][tid];
//
// Memory traffic: 2 * T passes over H per (vh, b), vs 2 passes for wy3.
// Not a perf win on its own — the throughput unlock is the verifier being
// able to follow non-flat tree branches at higher accept rates.
//
// Grid: (num_v_heads, batch, 1)   Block: (128, 1, 1)

#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#define BLOCK_SIZE 128

extern "C" __global__ void gated_delta_rule_tree(
    float* __restrict__ h_state,                // [batch, vh, k_dim, v_dim] — read-only here, NOT updated
    const __nv_bfloat16* __restrict__ query,    // [batch * T, qk_stride] strided per-token
    const __nv_bfloat16* __restrict__ key,      // same layout as query
    const __nv_bfloat16* __restrict__ value,    // [batch * T, v_stride]
    const float* __restrict__ gate,             // [batch * T, gb_stride]
    const float* __restrict__ beta,             // [batch * T, gb_stride]
    const int* __restrict__ parent_ids,         // [T] — i32, -1 for root, else < i
    __nv_bfloat16* __restrict__ output,         // [batch * T, num_v_heads, v_dim]
    // Intermediates pool: [T, batch, vh, k_dim, v_dim] FP32 (matches wy17).
    // Per-token slot at  h_state_inter + t * inter_stride_floats + (b*nv+vh)*hv.
    // Per-(b,vh) base at h_state_inter + (b*nv+vh)*hv.
    float* __restrict__ h_state_inter,
    unsigned int inter_stride_floats,           // stride between token slots (floats)
    unsigned int num_tokens,                    // T
    unsigned int batch_size,
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
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;

    // h_state base for this (b, vh). h_state stays as the immutable pre-tree
    // starting state — caller commits the chosen accepted slot post-verify
    // by copying from h_state_inter[<chosen index>] back to h_state.
    float* H_root = h_state + ((b * num_v_heads + vh) * hv);

    // wy17-compatible layout: per-token slot at
    //   h_state_inter + t * inter_stride_floats + (b * nv + vh) * hv
    // i.e. token-major pool with per-(b,vh) sub-region inside each token slice.
    float* H_inter_per_vh_offset = h_state_inter + ((b * num_v_heads + vh) * hv);

    // Shared per-token scratch for k, q (one token at a time — keep smem small
    // since we're processing T sequentially). 128 == k_dim.
    __shared__ float sk[128], sq[128];
    __shared__ float smem_warp[4];
    __shared__ float kd_dummy; (void)kd_dummy;

    // Sequential per-token loop.
    for (unsigned int t = 0; t < num_tokens; ++t) {
        // Per-token pointers.
        const __nv_bfloat16* qt = query + (b * num_tokens + t) * qk_stride + kh * k_dim;
        const __nv_bfloat16* kt = key   + (b * num_tokens + t) * qk_stride + kh * k_dim;
        const __nv_bfloat16* vt = value + (b * num_tokens + t) * v_stride  + vh * v_dim;
        // Gate clamp matches gated_delta_rule_wy / wy3 for bit-equivalent
        // outputs across kernels.
        const float gt  = fminf(fmaxf(gate[(b * num_tokens + t) * gb_stride + vh], 0.0f), 1.0f);
        const float btt = beta[(b * num_tokens + t) * gb_stride + vh];

        // Choose source state: root or parent's intermediate.
        const int parent = parent_ids[t];
        float* H_in;
        if (parent < 0) {
            H_in = H_root;
        } else {
            // Parent must be < t (DAG invariant — enforced host-side).
            H_in = h_state_inter
                 + ((unsigned int)parent) * inter_stride_floats
                 + ((b * num_v_heads + vh) * hv);
        }
        float* H_out = h_state_inter
                     + t * inter_stride_floats
                     + ((b * num_v_heads + vh) * hv);
        (void)H_inter_per_vh_offset; // legacy var name preserved for diff clarity

        // Cache k, q into shared (one token).
        if (tid < k_dim) {
            sk[tid] = (float)kt[tid];
            sq[tid] = (float)qt[tid];
        }
        __syncthreads();

        if (tid < v_dim) {
            float vi = (float)vt[tid];

            // ── PASS 1: read H_in, compute hk = H_in^T @ k ──
            float hk = 0.0f;
            #pragma unroll 4
            for (unsigned int j = 0; j < k_dim; j += 4) {
                float h0 = H_in[(j+0) * v_dim + tid];
                float h1 = H_in[(j+1) * v_dim + tid];
                float h2 = H_in[(j+2) * v_dim + tid];
                float h3 = H_in[(j+3) * v_dim + tid];
                hk += h0*sk[j] + h1*sk[j+1] + h2*sk[j+2] + h3*sk[j+3];
            }

            float v_new = (vi - gt * hk) * btt;

            // ── PASS 2: write H_out = gt*H_in + k ⊗ v_new, compute qd ──
            float qd = 0.0f;
            #pragma unroll 4
            for (unsigned int j = 0; j < k_dim; j += 4) {
                float h0 = H_in[(j+0) * v_dim + tid];
                float h1 = H_in[(j+1) * v_dim + tid];
                float h2 = H_in[(j+2) * v_dim + tid];
                float h3 = H_in[(j+3) * v_dim + tid];

                h0 = gt * h0 + sk[j  ] * v_new;
                h1 = gt * h1 + sk[j+1] * v_new;
                h2 = gt * h2 + sk[j+2] * v_new;
                h3 = gt * h3 + sk[j+3] * v_new;

                H_out[(j+0) * v_dim + tid] = h0;
                H_out[(j+1) * v_dim + tid] = h1;
                H_out[(j+2) * v_dim + tid] = h2;
                H_out[(j+3) * v_dim + tid] = h3;

                qd += h0 * sq[j]   + h1 * sq[j+1]
                    + h2 * sq[j+2] + h3 * sq[j+3];
            }

            float s = rsqrtf((float)k_dim);
            output[(b * num_tokens + t) * num_v_heads * v_dim + vh * v_dim + tid] =
                __float2bfloat16(qd * s);
        }
        __syncthreads();
    }
}
