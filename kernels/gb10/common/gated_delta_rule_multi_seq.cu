// SPDX-License-Identifier: AGPL-3.0-only

// Multi-sequence gated delta rule decode — parallel SSM state advance
// across `num_seqs` independent sequences.
//
// The single-seq `gated_delta_rule_decode` kernel launches grid
// (num_v_heads=32, batch=1) — that's 32 CTAs × 128 threads = 4096
// threads — leaving the 48-SM GB10 with 16 SMs idle even for one
// sequence. With c concurrent sequences serialized in a per-seq loop,
// the kernel was c×32 CTAs spread across c launches, hammering the
// LPDDR5X bus c× for the same weights and never exceeding 32 CTAs
// resident at once.
//
// This kernel fuses all c launches into one: grid = (num_v_heads,
// num_seqs, 1) so at c=4 we have 32×4=128 CTAs — every SM gets ~2.5
// CTAs, saturating compute. Each (vh, seq) CTA reads its own per-seq
// `h_state` pointer from a device-resident array.
//
// Non-state buffers (query/key/value/gate/beta/output) are contiguous
// per-seq with the same per-seq stride. Per-seq state pointers
// (`h_states`) are scattered because each sequence owns an arbitrary
// pool slot.
//
// Grid: (num_v_heads, num_seqs, 1)  Block: (128, 1, 1)

#include <cuda_bf16.h>

#define BLOCK_SIZE 128

#ifndef SSM_STATE_NORM_ENABLED
#define SSM_STATE_NORM_ENABLED
#define SSM_STATE_MAX_NORM 1000.0f
#endif

// ============================================================
// DECODE multi-seq: gated delta rule recurrent update.
// ============================================================
extern "C" __global__ void gated_delta_rule_decode_multi_seq(
    // Per-seq state pointers (device-resident, length num_seqs).
    // Each element is a `float*` to one sequence's
    // [num_v_heads, k_dim, v_dim] state buffer.
    float* const* __restrict__ h_states,
    // Per-seq inputs (interpreted with explicit strides):
    //   query/key  base + seq * qk_stride + kh * k_dim   BF16
    //   value      base + seq * v_in_stride + vh * v_dim BF16
    //   gate/beta  base + seq * gate_beta_stride         FP32
    //   output     base + seq * v_out_stride + vh * v_dim BF16
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_seqs,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int gate_beta_stride, // FP32 elements between seqs in gate/beta
    unsigned int qk_stride,        // BF16 elements between seqs in query/key
    unsigned int v_in_stride,      // BF16 elements between seqs in value
    unsigned int v_out_stride      // BF16 elements between seqs in output
) {
    const unsigned int vh  = blockIdx.x;   // value head index
    const unsigned int seq = blockIdx.y;   // sequence index
    if (vh >= num_v_heads || seq >= num_seqs) return;

    const unsigned int tid = threadIdx.x;

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    // Per-seq state pointer for this (seq, vh). H layout matches
    // single-seq decode: [num_v_heads, k_dim, v_dim] FP32 with v_dim
    // the fast (contiguous) dimension.
    float* H = h_states[seq] + (vh * k_dim * v_dim);

    // Per-seq input pointers (explicit per-seq strides).
    const __nv_bfloat16* q_ptr = query + (unsigned long long)seq * qk_stride + kh * k_dim;
    const __nv_bfloat16* k_ptr = key   + (unsigned long long)seq * qk_stride + kh * k_dim;
    const __nv_bfloat16* v_ptr = value + (unsigned long long)seq * v_in_stride + vh * v_dim;

    // gate/beta arranged with stride = gate_beta_stride per seq
    // (callers pass `2 * num_v_heads` so per-seq layout matches the
    // single-seq compute_gdn_gates output that interleaves [gate[nv],
    // beta[nv]] per token).
    float g_raw = gate[seq * gate_beta_stride + vh];
    const float g = fminf(fmaxf(g_raw, 0.0f), 1.0f);
    const float bt = beta[seq * gate_beta_stride + vh];

    __shared__ float smem_k[128];
    __shared__ float smem_q[128];

    if (tid < k_dim) {
        smem_k[tid] = (float)k_ptr[tid];
        smem_q[tid] = (float)q_ptr[tid];
    }
    __syncthreads();

    if (tid < v_dim) {
        float v_i = (float)v_ptr[tid];

        float hk_dot = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];
            hk_dot += h0 * smem_k[j] + h1 * smem_k[j + 1]
                    + h2 * smem_k[j + 2] + h3 * smem_k[j + 3];
        }

        float v_new_i = (v_i - g * hk_dot) * bt;

        float q_dot = 0.0f;
        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];
            h0 = g * h0 + smem_k[j]     * v_new_i;
            h1 = g * h1 + smem_k[j + 1] * v_new_i;
            h2 = g * h2 + smem_k[j + 2] * v_new_i;
            h3 = g * h3 + smem_k[j + 3] * v_new_i;
            H[(j + 0) * v_dim + tid] = h0;
            H[(j + 1) * v_dim + tid] = h1;
            H[(j + 2) * v_dim + tid] = h2;
            H[(j + 3) * v_dim + tid] = h3;
            q_dot += h0 * smem_q[j] + h1 * smem_q[j + 1]
                   + h2 * smem_q[j + 2] + h3 * smem_q[j + 3];
        }

        // ── SSM state norm clamp (Stuffed Mamba mitigation) ──
        #ifdef SSM_STATE_NORM_ENABLED
        {
            float local_sq = 0.0f;
            for (unsigned int j = 0; j < k_dim; j++) {
                float hv = H[j * v_dim + tid];
                local_sq += hv * hv;
            }
            unsigned int mask = __activemask();
            float warp_sum = local_sq;
            warp_sum += __shfl_down_sync(mask, warp_sum, 16);
            warp_sum += __shfl_down_sync(mask, warp_sum, 8);
            warp_sum += __shfl_down_sync(mask, warp_sum, 4);
            warp_sum += __shfl_down_sync(mask, warp_sum, 2);
            warp_sum += __shfl_down_sync(mask, warp_sum, 1);

            __shared__ float norm_sums[4];
            unsigned int warp_id = tid / 32;
            unsigned int lane_id = tid % 32;
            if (lane_id == 0) norm_sums[warp_id] = warp_sum;
            __syncthreads();

            float head_norm_sq;
            if (tid < 4) {
                float s = norm_sums[tid];
                s += __shfl_down_sync(0xf, s, 2);
                s += __shfl_down_sync(0xf, s, 1);
                norm_sums[0] = s;
            }
            __syncthreads();
            head_norm_sq = norm_sums[0];

            if (head_norm_sq > SSM_STATE_MAX_NORM * SSM_STATE_MAX_NORM) {
                float scale = SSM_STATE_MAX_NORM * rsqrtf(head_norm_sq);
                for (unsigned int j = 0; j < k_dim; j++) {
                    H[j * v_dim + tid] *= scale;
                }
            }
        }
        #endif

        float inv_sqrt_d = rsqrtf((float)k_dim);
        output[(unsigned long long)seq * v_out_stride + vh * v_dim + tid] = __float2bfloat16(q_dot * inv_sqrt_d);
    }
}
