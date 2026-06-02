// SPDX-License-Identifier: AGPL-3.0-only

// Argmax over BF16 logits — single-block tree reduction.
//
// Finds the index of the maximum BF16 value in an array of `n` elements.
// Writes a single u32 token ID to `out`.
//
// Grid: (1, 1, 1)  Block: (1024, 1, 1)
// For vocab_size ≤ ~200K, a single block with 1024 threads is sufficient
// (each thread handles ceil(n/1024) elements).

#include <cuda_bf16.h>

extern "C" __global__ void argmax_bf16(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out,
    unsigned int n
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;

    // Phase 1: each thread finds its local max
    float local_max = -1e30f;
    unsigned int local_idx = 0;

    for (unsigned int i = tid; i < n; i += stride) {
        float v = __bfloat162float(logits[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Phase 2: tree reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    // Phase 3: thread 0 writes result
    if (tid == 0) {
        out[0] = s_idx[0];
    }
}

// Argmax over FP32 logits — used when LM head outputs FP32 for sampling quality.
extern "C" __global__ void argmax_fp32(
    const float* __restrict__ logits,
    unsigned int* __restrict__ out,
    unsigned int n
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;

    float local_max = -1e30f;
    unsigned int local_idx = 0;
    for (unsigned int i = tid; i < n; i += stride) {
        float v = logits[i];
        if (v > local_max) { local_max = v; local_idx = i; }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_val[tid + s] > s_val[tid]) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) out[0] = s_idx[0];
}

// Top-K over BF16 logits, batched across multiple rows.
//
// For each row in [0, num_rows), finds the top-K maximum values in a vector
// of `vocab` BF16 logits and writes:
//   - top_indices[row, 0..K]  : u32 token IDs sorted by score descending
//   - top_logits[row, 0..K]   : f32 logit values (NOT log-probs; caller can
//                                normalize externally if needed)
//
// Algorithm: K-pass argmax with masking. K is small (≤ 16 typical), so this
// is O(K * vocab / threads) which beats heap-based per-thread top-K for
// the M4B v2 DDTree case (K=8, vocab=248K).
//
// Grid: (num_rows, 1, 1)  Block: (1024, 1, 1)
// Each block handles one row independently.
//
// Compile-time bound: MAX_TOP_K caps shared-mem usage; if K > MAX_TOP_K the
// kernel silently truncates (caller already validates K).

#define MAX_TOP_K 16

extern "C" __global__ void topk_bf16(
    const __nv_bfloat16* __restrict__ logits,  // [num_rows, vocab] BF16
    unsigned int* __restrict__ top_indices,    // [num_rows, k] u32
    float* __restrict__ top_logits,            // [num_rows, k] f32
    unsigned int num_rows,
    unsigned int vocab,
    unsigned int k
) {
    const unsigned int row = blockIdx.x;
    if (row >= num_rows) return;
    if (k == 0 || k > MAX_TOP_K) return;

    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];
    __shared__ unsigned int s_taken[MAX_TOP_K]; // already-selected indices

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;
    const __nv_bfloat16* row_logits = logits + (size_t)row * vocab;

    // K-pass argmax. Each pass excludes previously selected indices.
    for (unsigned int pass = 0; pass < k; ++pass) {
        // Phase 1: per-thread local max, skipping already-selected indices.
        float local_max = -1e30f;
        unsigned int local_idx = 0;

        for (unsigned int i = tid; i < vocab; i += stride) {
            bool skip = false;
            // Linear scan of already-selected indices — for K ≤ 16 this is
            // negligible vs the vocab loop body.
            for (unsigned int p = 0; p < pass; ++p) {
                if (s_taken[p] == i) { skip = true; break; }
            }
            if (skip) continue;
            float v = __bfloat162float(row_logits[i]);
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }

        s_val[tid] = local_max;
        s_idx[tid] = local_idx;
        __syncthreads();

        // Phase 2: tree reduction
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (s_val[tid + s] > s_val[tid]) {
                    s_val[tid] = s_val[tid + s];
                    s_idx[tid] = s_idx[tid + s];
                }
            }
            __syncthreads();
        }

        // Phase 3: thread 0 records the winner.
        if (tid == 0) {
            s_taken[pass] = s_idx[0];
            top_indices[(size_t)row * k + pass] = s_idx[0];
            top_logits[(size_t)row * k + pass] = s_val[0];
        }
        __syncthreads();
    }
}
