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

    // Phase 2: tree reduction in shared memory. Tie-break on the LOWER array index, not on
    // which side of the comparison a thread happens to land: `s_val[tid+s] > s_val[tid]` alone
    // keeps whichever slot the tree structure put at `tid`, which is thread-position-dependent,
    // not index-dependent -- a tie between global indices 3 and 1026 at blockDim 1024 returned
    // 1026 (thread 2's slot beat thread 3's slot at the final reduction level), which is
    // neither first-max nor last-max. First-max requires comparing the index on every tie.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            bool other_wins = (s_val[tid + s] > s_val[tid])
                || (s_val[tid + s] == s_val[tid] && s_idx[tid + s] < s_idx[tid]);
            if (other_wins) {
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

// Per-row TOP-2 over BF16 logits — block-fork tree cliff detection (doc 16).
//
// One block per row. For each row of `rows` x `n` logits, writes
// out[row*4+0]=idx1, out[row*4+1]=bits(val1), out[row*4+2]=idx2,
// out[row*4+3]=bits(val2)  (values as f32 bit patterns; host reinterprets).
// idx2/val2 are the runner-up EXCLUDING idx1. Margin = val1 - val2 on host.
//
// Grid: (rows, 1, 1)  Block: (1024, 1, 1)
extern "C" __global__ void top2_bf16_rows(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out,
    unsigned int n
) {
    __shared__ float s_v1[1024];
    __shared__ unsigned int s_i1[1024];
    __shared__ float s_v2[1024];
    __shared__ unsigned int s_i2[1024];

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;
    const __nv_bfloat16* row = logits + (unsigned long long)blockIdx.x * n;

    float v1 = -1e30f, v2 = -1e30f;
    unsigned int i1 = 0, i2 = 0;
    for (unsigned int i = tid; i < n; i += stride) {
        const float v = __bfloat162float(row[i]);
        if (v > v1) {
            v2 = v1; i2 = i1;
            v1 = v; i1 = i;
        } else if (v > v2) {
            v2 = v; i2 = i;
        }
    }
    s_v1[tid] = v1; s_i1[tid] = i1;
    s_v2[tid] = v2; s_i2[tid] = i2;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // Merge two (top1, top2) pairs into one, first-max on ties: pick the top 2 of the
            // 4 candidates by (value desc, index asc) explicitly, rather than a value-only `>`
            // cascade whose tie behaviour depends on which side of the tree a thread landed on
            // (the same bug as the plain argmax reduction above). `a`'s and `b`'s index sets
            // are always disjoint (different threads/partitions), so no candidate needs
            // excluding by identity, only by rank.
            const float cv[4] = {s_v1[tid], s_v1[tid + s], s_v2[tid], s_v2[tid + s]};
            const unsigned int ci[4] = {s_i1[tid], s_i1[tid + s], s_i2[tid], s_i2[tid + s]};
            int best = 0;
            for (int k = 1; k < 4; k++) {
                if (cv[k] > cv[best] || (cv[k] == cv[best] && ci[k] < ci[best])) best = k;
            }
            int second = -1;
            for (int k = 0; k < 4; k++) {
                if (k == best) continue;
                if (second < 0 || cv[k] > cv[second] || (cv[k] == cv[second] && ci[k] < ci[second])) second = k;
            }
            s_v1[tid] = cv[best]; s_i1[tid] = ci[best];
            s_v2[tid] = cv[second]; s_i2[tid] = ci[second];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x * 4 + 0] = s_i1[0];
        out[blockIdx.x * 4 + 1] = __float_as_uint(s_v1[0]);
        out[blockIdx.x * 4 + 2] = s_i2[0];
        out[blockIdx.x * 4 + 3] = __float_as_uint(s_v2[0]);
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

    // Same first-max tie-break as argmax_bf16 above: value-only `>` picks whichever slot the
    // reduction tree put at `tid`, not the lower array index.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            bool other_wins = (s_val[tid + s] > s_val[tid])
                || (s_val[tid + s] == s_val[tid] && s_idx[tid + s] < s_idx[tid]);
            if (other_wins) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[0] = s_idx[0];
}
