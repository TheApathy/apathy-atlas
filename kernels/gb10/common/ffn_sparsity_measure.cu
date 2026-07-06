// SPDX-License-Identifier: AGPL-3.0-only

// Atlas FFN Activation-Sparsity Measurement (feasibility harness for
// SPARSITY-DRAFTED SELF-SPECULATION / TEAL-style column skipping).
//
// PURPOSE (go/no-go gate — NOT a production kernel):
//   For a BF16 activation vector x[K] (a single decode row), count how many
//   |x[j]| fall below thresholds expressed as fractions of the ROW MAX
//   |x|. This is the exact predicate a column-sparse GEMV would use to skip
//   reading weight column j (`if |x[j]| < tau: skip B[:, j]`). The measured
//   below-threshold fraction is the *upper bound* on the weight-byte savings
//   for that matmul at that threshold.
//
//   Two call sites per FFN layer:
//     1. gate/up input  = `normed2` (residual stream, K = hidden = 5120)
//     2. down input      = silu(gate)*up (K = intermediate = 17408)
//
// OUTPUT:
//   `hist_out` is an array of `NUM_THRESH` u32 counters (one per threshold),
//   ATOMICALLY ACCUMULATED across all invocations for a given layer/site so
//   the host can average over many decode steps. Thresholds are the fixed
//   set {0.5%, 1%, 2%, 5%} of the per-row max-abs. `count_out[0]` accumulates
//   the number of rows measured and `count_out[1]` accumulates K (so the host
//   can compute fraction = hist / (rows_seen_scaled) without re-deriving K).
//
//   Layout of hist_out (length NUM_THRESH = 4, u32):
//     hist_out[0] += #(|x[j]| < 0.005 * rowmax)
//     hist_out[1] += #(|x[j]| < 0.010 * rowmax)
//     hist_out[2] += #(|x[j]| < 0.020 * rowmax)
//     hist_out[3] += #(|x[j]| < 0.050 * rowmax)
//   count_out[0] += 1 (rows), count_out[1] += K (elements)
//
// One CTA measures one row. Two-pass within the CTA:
//   Pass 1: block-reduce max(|x|).
//   Pass 2: each thread counts its strided slice into 4 thresholds; block
//           reduce; thread 0 atomicAdds into the global histogram.
//
// This deliberately does NOT touch weights and adds ~2 vector passes over the
// activation only (K * 2 bytes) — negligible vs the 89 MB/proj weight read it
// is evaluating, so it can run inline during a real decode with env-gating.

#include <cuda_bf16.h>

#define SPARSITY_BLOCK 256
#define SPARSITY_NUM_THRESH 4

__device__ __constant__ float SPARSITY_TAU[SPARSITY_NUM_THRESH] = {
    0.005f, 0.010f, 0.020f, 0.050f
};

// input:     [1, K] BF16 activation for ONE decode row.
// hist_out:  [NUM_THRESH] u32 global counters (atomically accumulated).
// count_out: [2] u32 — [0]=rows seen, [1]=elements seen (K accumulated).
extern "C" __global__ void ffn_sparsity_measure(
    const __nv_bfloat16* __restrict__ input,
    unsigned int* __restrict__ hist_out,
    unsigned int* __restrict__ count_out,
    unsigned int K
) {
    const unsigned int tid = threadIdx.x;

    __shared__ float s_max;
    __shared__ float s_warpmax[SPARSITY_BLOCK / 32];
    __shared__ unsigned int s_hist[SPARSITY_NUM_THRESH];

    if (tid < SPARSITY_NUM_THRESH) s_hist[tid] = 0u;

    // ── Pass 1: max(|x|) over the row ──
    float local_max = 0.0f;
    for (unsigned int j = tid; j < K; j += SPARSITY_BLOCK) {
        float v = fabsf(__bfloat162float(input[j]));
        local_max = fmaxf(local_max, v);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, off));
    }
    const unsigned int wid = tid / 32;
    const unsigned int wlane = tid % 32;
    if (wlane == 0) s_warpmax[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        float v = (wlane < (SPARSITY_BLOCK / 32)) ? s_warpmax[wlane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, off));
        }
        if (wlane == 0) s_max = v;
    }
    __syncthreads();

    const float rowmax = s_max;
    if (rowmax <= 0.0f) {
        // Degenerate row (all-zero). Count as fully sparse and bail.
        if (tid == 0) {
            for (int t = 0; t < SPARSITY_NUM_THRESH; t++)
                atomicAdd(&hist_out[t], K);
            atomicAdd(&count_out[0], 1u);
            atomicAdd(&count_out[1], K);
        }
        return;
    }

    // Precompute per-threshold absolute cutoffs.
    float cut[SPARSITY_NUM_THRESH];
    #pragma unroll
    for (int t = 0; t < SPARSITY_NUM_THRESH; t++) cut[t] = SPARSITY_TAU[t] * rowmax;

    // ── Pass 2: count below-threshold per thread → block hist ──
    unsigned int local_cnt[SPARSITY_NUM_THRESH] = {0u, 0u, 0u, 0u};
    for (unsigned int j = tid; j < K; j += SPARSITY_BLOCK) {
        float a = fabsf(__bfloat162float(input[j]));
        #pragma unroll
        for (int t = 0; t < SPARSITY_NUM_THRESH; t++) {
            if (a < cut[t]) local_cnt[t]++;
        }
    }
    #pragma unroll
    for (int t = 0; t < SPARSITY_NUM_THRESH; t++) {
        unsigned int c = local_cnt[t];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            c += __shfl_down_sync(0xFFFFFFFF, c, off);
        if (wlane == 0) atomicAdd(&s_hist[t], c);
    }
    __syncthreads();

    if (tid < SPARSITY_NUM_THRESH) atomicAdd(&hist_out[tid], s_hist[tid]);
    if (tid == 0) {
        atomicAdd(&count_out[0], 1u);
        atomicAdd(&count_out[1], K);
    }
}
