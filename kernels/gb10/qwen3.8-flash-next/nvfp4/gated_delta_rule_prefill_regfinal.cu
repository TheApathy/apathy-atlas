// SPDX-License-Identifier: AGPL-3.0-only
//
// Prefill twin of `gated_delta_rule_decode_f32_sequence_nosnap`, derived
// verbatim from the in-tree `_persistent` kernel (register-resident H,
// identical arithmetic order) with the per-token snapshot block removed,
// static shared memory, and the final H written back to `h_state`.
// Verified bit-exact against the nosnap kernel on random data
// (scratchpad/gdntest: 0/393216 output and 0/786432 state mismatches).
// Same argument list as the nosnap kernel; h_state_inter/inter_stride unused.
// Grid: (num_v_heads, 1, 1)  Block: (128, 1, 1)
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#define K_DIM 128
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_f32_sequence_regfinal(
    float* __restrict__ h_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ h_state_inter,
    unsigned int num_tokens,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gate_beta_stride,
    unsigned int output_stride,
    unsigned int inter_stride
) {
    const unsigned int vh = blockIdx.x;
    if (vh >= num_v_heads) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    float* H = h_state + (unsigned long long)vh * k_dim * v_dim;
    __shared__ float smem[2 * K_DIM];
    float* smem_k = smem;
    float* smem_q = smem + K_DIM;

    // Each thread owns column tid of the k_dim x v_dim state; load once.
    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H[(unsigned long long)j * v_dim + tid];
    }

    for (unsigned int token = 0; token < num_tokens; token++) {
        const float* q_ptr = query
            + (unsigned long long)token * qk_stride + kh * k_dim;
        const float* k_ptr = key
            + (unsigned long long)token * qk_stride + kh * k_dim;
        const float* v_ptr = value
            + (unsigned long long)token * v_stride + vh * v_dim;
        const float g = fminf(fmaxf(
            gate[(unsigned long long)token * gate_beta_stride + vh], 0.0f), 1.0f);
        const float bt =
            beta[(unsigned long long)token * gate_beta_stride + vh];
        if (tid < k_dim) {
            smem_k[tid] = k_ptr[tid];
            smem_q[tid] = q_ptr[tid];
        }
        __syncthreads();
        if (tid < v_dim) {
            float v_i = v_ptr[tid];
            // Pass 1: hk_dot = H^T . k (single accumulator, unroll 4)
            float hk_dot = 0.0f;
            #pragma unroll 4
            for (unsigned int j = 0; j < K_DIM; j += 4) {
                hk_dot += H_reg[j] * smem_k[j]
                        + H_reg[j + 1] * smem_k[j + 1]
                        + H_reg[j + 2] * smem_k[j + 2]
                        + H_reg[j + 3] * smem_k[j + 3];
            }
            float v_new_i = (v_i - g * hk_dot) * bt;
            // Pass 2: H <- g*H + k*v_new; q_dot = H_new^T . q
            float q_dot = 0.0f;
            #pragma unroll 4
            for (unsigned int j = 0; j < K_DIM; j += 4) {
                float h0 = g * H_reg[j] + smem_k[j] * v_new_i;
                float h1 = g * H_reg[j + 1] + smem_k[j + 1] * v_new_i;
                float h2 = g * H_reg[j + 2] + smem_k[j + 2] * v_new_i;
                float h3 = g * H_reg[j + 3] + smem_k[j + 3] * v_new_i;
                H_reg[j] = h0;
                H_reg[j + 1] = h1;
                H_reg[j + 2] = h2;
                H_reg[j + 3] = h3;
                q_dot += h0 * smem_q[j]
                       + h1 * smem_q[j + 1]
                       + h2 * smem_q[j + 2]
                       + h3 * smem_q[j + 3];
            }
            float inv_sqrt_d = rsqrtf((float)k_dim);
            output[(unsigned long long)token * output_stride + vh * v_dim + tid] =
                q_dot * inv_sqrt_d;
            // Exact-state snapshot (commit contract): write post-token H.
                    }
        __syncthreads();
    }

    // Final H writeback.
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H[(unsigned long long)j * v_dim + tid] = H_reg[j];
    }
}

// Prefetching twin (same arithmetic; loads for token+1 issued before the compute).
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_f32_sequence_regfinal_pf(
    float* __restrict__ h_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ h_state_inter,
    unsigned int num_tokens,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gate_beta_stride,
    unsigned int output_stride,
    unsigned int inter_stride
) {
    const unsigned int vh = blockIdx.x;
    if (vh >= num_v_heads) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    float* H = h_state + (unsigned long long)vh * k_dim * v_dim;
    __shared__ float smem[2 * K_DIM];
    float* smem_k = smem;
    float* smem_q = smem + K_DIM;

    // Each thread owns column tid of the k_dim x v_dim state; load once.
    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H_reg[j] = H[(unsigned long long)j * v_dim + tid];
    }

    // Software prefetch: the next token's q/k/v/gate/beta are loaded into
    // registers while the current token is computed. Arithmetic unchanged.
    float nq = 0.0f, nk = 0.0f, nv = 0.0f, ng = 0.0f, nb = 0.0f;
    if (num_tokens > 0) {
        if (tid < k_dim) { nq = query[kh * k_dim + tid]; nk = key[kh * k_dim + tid]; }
        if (tid < v_dim) nv = value[vh * v_dim + tid];
        ng = gate[vh]; nb = beta[vh];
    }
    for (unsigned int token = 0; token < num_tokens; token++) {
        const float g = fminf(fmaxf(ng, 0.0f), 1.0f);
        const float bt = nb;
        const float v_i = nv;
        if (tid < k_dim) {
            smem_k[tid] = nk;
            smem_q[tid] = nq;
        }
        __syncthreads();
        if (token + 1 < num_tokens) {
            const unsigned long long t1 = token + 1;
            if (tid < k_dim) {
                nq = query[t1 * qk_stride + kh * k_dim + tid];
                nk = key[t1 * qk_stride + kh * k_dim + tid];
            }
            if (tid < v_dim) nv = value[t1 * v_stride + vh * v_dim + tid];
            ng = gate[t1 * gate_beta_stride + vh];
            nb = beta[t1 * gate_beta_stride + vh];
        }
        if (tid < v_dim) {
            // Pass 1: hk_dot = H^T . k (single accumulator, unroll 4)
            float hk_dot = 0.0f;
            #pragma unroll 4
            for (unsigned int j = 0; j < K_DIM; j += 4) {
                hk_dot += H_reg[j] * smem_k[j]
                        + H_reg[j + 1] * smem_k[j + 1]
                        + H_reg[j + 2] * smem_k[j + 2]
                        + H_reg[j + 3] * smem_k[j + 3];
            }
            float v_new_i = (v_i - g * hk_dot) * bt;
            // Pass 2: H <- g*H + k*v_new; q_dot = H_new^T . q
            float q_dot = 0.0f;
            #pragma unroll 4
            for (unsigned int j = 0; j < K_DIM; j += 4) {
                float h0 = g * H_reg[j] + smem_k[j] * v_new_i;
                float h1 = g * H_reg[j + 1] + smem_k[j + 1] * v_new_i;
                float h2 = g * H_reg[j + 2] + smem_k[j + 2] * v_new_i;
                float h3 = g * H_reg[j + 3] + smem_k[j + 3] * v_new_i;
                H_reg[j] = h0;
                H_reg[j + 1] = h1;
                H_reg[j + 2] = h2;
                H_reg[j + 3] = h3;
                q_dot += h0 * smem_q[j]
                       + h1 * smem_q[j + 1]
                       + h2 * smem_q[j + 2]
                       + h3 * smem_q[j + 3];
            }
            float inv_sqrt_d = rsqrtf((float)k_dim);
            output[(unsigned long long)token * output_stride + vh * v_dim + tid] =
                q_dot * inv_sqrt_d;
            // Exact-state snapshot (commit contract): write post-token H.
                    }
        __syncthreads();
    }

    // Final H writeback.
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) {
        H[(unsigned long long)j * v_dim + tid] = H_reg[j];
    }
}
