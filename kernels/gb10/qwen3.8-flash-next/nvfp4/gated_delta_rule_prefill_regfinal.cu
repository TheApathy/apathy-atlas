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

// ---------------------------------------------------------------------------
// Faster twins of `_regfinal` (selector ATLAS_QWEN4_PREFILL_GDN_LAZYFINAL=3/4/5).
// The `_regfinal` kernels above keep H_reg[128] in a 512-byte LOCAL-memory stack
// frame (ptxas -v: "512 bytes stack frame"): `#pragma unroll 4` over a 32-step loop
// leaves dynamic indices, so every token re-reads and re-writes H through L1.
// The twins below execute the SAME per-thread FP32 operation sequence (same
// operands, same order, --fmad=false keeps every mul/add separate) with:
//   _full   : both K loops fully unrolled so H lives in registers; the pass-2 k
//             reads are volatile so ptxas re-loads k instead of pinning 128 more
//             registers (0 B stack, 0 spill).
//   _chunk* : _full plus the token stream staged CH tokens at a time into shared
//             memory with double-buffered cp.async, so the per-token loop has no
//             global-load latency and no barriers (2 barriers per CH tokens).
// Bit-exactness vs `_regfinal` (outputs and final H) is checked by
// fnp-bench/gdn harness on T in {1,5,17,64,2047,2048}; requires k_dim == v_dim == 128.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp4(float* s, const float* g) {
    unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa), "l"(g));
}
__device__ __forceinline__ void cp16(float* s, const float* g) {
    unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(g));
}
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_f32_sequence_regfinal_full(
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

    #pragma unroll 1
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
            #pragma unroll
            for (unsigned int j = 0; j < K_DIM; j += 4) {
                hk_dot += H_reg[j] * smem_k[j]
                        + H_reg[j + 1] * smem_k[j + 1]
                        + H_reg[j + 2] * smem_k[j + 2]
                        + H_reg[j + 3] * smem_k[j + 3];
            }
            float v_new_i = (v_i - g * hk_dot) * bt;
            // Pass 2: H <- g*H + k*v_new; q_dot = H_new^T . q
            float q_dot = 0.0f;
            #pragma unroll
            for (unsigned int j = 0; j < K_DIM; j += 4) {
                float h0 = g * H_reg[j] + ((volatile float*)smem_k)[j] * v_new_i;
                float h1 = g * H_reg[j + 1] + ((volatile float*)smem_k)[j + 1] * v_new_i;
                float h2 = g * H_reg[j + 2] + ((volatile float*)smem_k)[j + 2] * v_new_i;
                float h3 = g * H_reg[j + 3] + ((volatile float*)smem_k)[j + 3] * v_new_i;
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


extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_f32_sequence_regfinal_chunk15(
    float* __restrict__ h_state, const float* __restrict__ query, const float* __restrict__ key,
    const float* __restrict__ value, const float* __restrict__ gate, const float* __restrict__ beta,
    float* __restrict__ output, float* __restrict__ h_state_inter, unsigned int num_tokens,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim, unsigned int v_dim,
    unsigned int qk_stride, unsigned int v_stride, unsigned int gate_beta_stride,
    unsigned int output_stride, unsigned int inter_stride
) {
    const unsigned int vh = blockIdx.x;
    if (vh >= num_v_heads) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    float* H = h_state + (unsigned long long)vh * k_dim * v_dim;
    // [buf][token][K] for k and q, [buf][token][V] for v, [buf][token] for g, beta
    __shared__ __align__(16) float sk[2][15][K_DIM];
    __shared__ __align__(16) float sq[2][15][K_DIM];
    __shared__ __align__(16) float sv[2][15][K_DIM];
    __shared__ float sg[2][15];
    __shared__ float sb[2][15];

    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) H_reg[j] = H[(unsigned long long)j * v_dim + tid];

    auto stage = [&](unsigned buf, unsigned t0) {
        // 128 threads: each copies 4 floats (16 B) of k, q, v for each token in the chunk.
        const unsigned n = min((unsigned)15, num_tokens - t0);
        const unsigned part = tid & 31, row0 = tid >> 5;  // 32 x 16B = 128 floats per row
        for (unsigned r = row0; r < n; r += 4) {
            const unsigned long long t = t0 + r;
            cp16(&sk[buf][r][part * 4], key + t * qk_stride + kh * k_dim + part * 4);
            cp16(&sq[buf][r][part * 4], query + t * qk_stride + kh * k_dim + part * 4);
            cp16(&sv[buf][r][part * 4], value + t * v_stride + vh * v_dim + part * 4);
        }
        if (tid < n) {
            cp4(&sg[buf][tid], gate + (unsigned long long)(t0 + tid) * gate_beta_stride + vh);
            cp4(&sb[buf][tid], beta + (unsigned long long)(t0 + tid) * gate_beta_stride + vh);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    const float inv_sqrt_d = rsqrtf((float)k_dim);
    if (num_tokens > 0) stage(0, 0);
    unsigned buf = 0;
    #pragma unroll 1
    for (unsigned int t0 = 0; t0 < num_tokens; t0 += 15, buf ^= 1) {
        if (t0 + 15 < num_tokens) {
            stage(buf ^ 1, t0 + 15);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const unsigned n = min((unsigned)15, num_tokens - t0);
        #pragma unroll 1
        for (unsigned r = 0; r < n; r++) {
            const float* smem_k = sk[buf][r];
            const float* smem_q = sq[buf][r];
            const float g = fminf(fmaxf(sg[buf][r], 0.0f), 1.0f);
            const float bt = sb[buf][r];
            float v_i = sv[buf][r][tid];
            float hk_dot = 0.0f;
            #pragma unroll
            for (unsigned int j = 0; j < K_DIM; j += 4) {
                hk_dot += H_reg[j] * smem_k[j]
                        + H_reg[j + 1] * smem_k[j + 1]
                        + H_reg[j + 2] * smem_k[j + 2]
                        + H_reg[j + 3] * smem_k[j + 3];
            }
            float v_new_i = (v_i - g * hk_dot) * bt;
            const volatile float* vk = smem_k;
            float q_dot = 0.0f;
            #pragma unroll
            for (unsigned int j = 0; j < K_DIM; j += 4) {
                float h0 = g * H_reg[j] + vk[j] * v_new_i;
                float h1 = g * H_reg[j + 1] + vk[j + 1] * v_new_i;
                float h2 = g * H_reg[j + 2] + vk[j + 2] * v_new_i;
                float h3 = g * H_reg[j + 3] + vk[j + 3] * v_new_i;
                H_reg[j] = h0; H_reg[j + 1] = h1; H_reg[j + 2] = h2; H_reg[j + 3] = h3;
                q_dot += h0 * smem_q[j] + h1 * smem_q[j + 1] + h2 * smem_q[j + 2] + h3 * smem_q[j + 3];
            }
            output[(unsigned long long)(t0 + r) * output_stride + vh * v_dim + tid] = q_dot * inv_sqrt_d;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) H[(unsigned long long)j * v_dim + tid] = H_reg[j];
}

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_prefill_f32_sequence_regfinal_chunk8(
    float* __restrict__ h_state, const float* __restrict__ query, const float* __restrict__ key,
    const float* __restrict__ value, const float* __restrict__ gate, const float* __restrict__ beta,
    float* __restrict__ output, float* __restrict__ h_state_inter, unsigned int num_tokens,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim, unsigned int v_dim,
    unsigned int qk_stride, unsigned int v_stride, unsigned int gate_beta_stride,
    unsigned int output_stride, unsigned int inter_stride
) {
    const unsigned int vh = blockIdx.x;
    if (vh >= num_v_heads) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    float* H = h_state + (unsigned long long)vh * k_dim * v_dim;
    // [buf][token][K] for k and q, [buf][token][V] for v, [buf][token] for g, beta
    __shared__ __align__(16) float sk[2][8][K_DIM];
    __shared__ __align__(16) float sq[2][8][K_DIM];
    __shared__ __align__(16) float sv[2][8][K_DIM];
    __shared__ float sg[2][8];
    __shared__ float sb[2][8];

    float H_reg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) H_reg[j] = H[(unsigned long long)j * v_dim + tid];

    auto stage = [&](unsigned buf, unsigned t0) {
        // 128 threads: each copies 4 floats (16 B) of k, q, v for each token in the chunk.
        const unsigned n = min((unsigned)8, num_tokens - t0);
        const unsigned part = tid & 31, row0 = tid >> 5;  // 32 x 16B = 128 floats per row
        for (unsigned r = row0; r < n; r += 4) {
            const unsigned long long t = t0 + r;
            cp16(&sk[buf][r][part * 4], key + t * qk_stride + kh * k_dim + part * 4);
            cp16(&sq[buf][r][part * 4], query + t * qk_stride + kh * k_dim + part * 4);
            cp16(&sv[buf][r][part * 4], value + t * v_stride + vh * v_dim + part * 4);
        }
        if (tid < n) {
            cp4(&sg[buf][tid], gate + (unsigned long long)(t0 + tid) * gate_beta_stride + vh);
            cp4(&sb[buf][tid], beta + (unsigned long long)(t0 + tid) * gate_beta_stride + vh);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    const float inv_sqrt_d = rsqrtf((float)k_dim);
    if (num_tokens > 0) stage(0, 0);
    unsigned buf = 0;
    #pragma unroll 1
    for (unsigned int t0 = 0; t0 < num_tokens; t0 += 8, buf ^= 1) {
        if (t0 + 8 < num_tokens) {
            stage(buf ^ 1, t0 + 8);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const unsigned n = min((unsigned)8, num_tokens - t0);
        #pragma unroll 1
        for (unsigned r = 0; r < n; r++) {
            const float* smem_k = sk[buf][r];
            const float* smem_q = sq[buf][r];
            const float g = fminf(fmaxf(sg[buf][r], 0.0f), 1.0f);
            const float bt = sb[buf][r];
            float v_i = sv[buf][r][tid];
            float hk_dot = 0.0f;
            #pragma unroll
            for (unsigned int j = 0; j < K_DIM; j += 4) {
                hk_dot += H_reg[j] * smem_k[j]
                        + H_reg[j + 1] * smem_k[j + 1]
                        + H_reg[j + 2] * smem_k[j + 2]
                        + H_reg[j + 3] * smem_k[j + 3];
            }
            float v_new_i = (v_i - g * hk_dot) * bt;
            const volatile float* vk = smem_k;
            float q_dot = 0.0f;
            #pragma unroll
            for (unsigned int j = 0; j < K_DIM; j += 4) {
                float h0 = g * H_reg[j] + vk[j] * v_new_i;
                float h1 = g * H_reg[j + 1] + vk[j + 1] * v_new_i;
                float h2 = g * H_reg[j + 2] + vk[j + 2] * v_new_i;
                float h3 = g * H_reg[j + 3] + vk[j + 3] * v_new_i;
                H_reg[j] = h0; H_reg[j + 1] = h1; H_reg[j + 2] = h2; H_reg[j + 3] = h3;
                q_dot += h0 * smem_q[j] + h1 * smem_q[j + 1] + h2 * smem_q[j + 2] + h3 * smem_q[j + 3];
            }
            output[(unsigned long long)(t0 + r) * output_stride + vh * v_dim + tid] = q_dot * inv_sqrt_d;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) H[(unsigned long long)j * v_dim + tid] = H_reg[j];
}
