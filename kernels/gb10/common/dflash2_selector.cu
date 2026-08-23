// SPDX-License-Identifier: AGPL-3.0-only
//
// DFlash2 `CandidateSelector` greedy walk, ported from
// Avarok-Cybersecurity/atlas PR #648 (`dflash2_selector_walk`).
//
// Upstream launches 512 threads (16 warps, one warp per candidate) and takes
// the predecessor implicitly from `tokens[0]`. This tree's Rust op predates
// that and passes `anchor_id` explicitly with a 16-thread block — one thread
// per candidate — which `examples/dflash2_conv_selector_microtest.rs` already
// checks against a CPU reference. Ported to that ABI so the microtest gates it.
//
// Reference: `CandidateSelector.select` (z-lab/dflash `dflash/model.py`),
// greedy (T=0) branch. Per position:
//   scores[k] = unary[k] + Σ_r (pred[prev][r] * hidden[t][r]) * succ[cand_k][r]
//   index     = argmax_k scores
//   prev      = candidates[index]
// The walk is inherently SEQUENTIAL across positions: position t's predecessor
// is position t-1's choice, which is why this is one block, not a grid.

#include <cuda_bf16.h>

#define DF2_TOPK 16

extern "C" __global__ void dflash2_selector_walk(
    const float* __restrict__ unary,            // [gamma, 16]
    const unsigned int* __restrict__ candidates,// [gamma, 16]
    const __nv_bfloat16* __restrict__ hidden,   // [gamma, rank]
    const __nv_bfloat16* __restrict__ pred,     // [vocab, rank]
    const __nv_bfloat16* __restrict__ succ,     // [vocab, rank]
    unsigned int* __restrict__ path,            // [gamma] out
    unsigned int anchor_id,
    int gamma,
    int rank
) {
    __shared__ float s_score[DF2_TOPK];
    __shared__ unsigned int s_prev;

    const unsigned int k = threadIdx.x;   // one thread per candidate

    if (k == 0) s_prev = anchor_id;
    __syncthreads();

    for (int t = 0; t < gamma; ++t) {
        const __nv_bfloat16* pr = pred + (size_t)s_prev * (size_t)rank;
        const __nv_bfloat16* hr = hidden + (size_t)t * (size_t)rank;

        float acc = 0.0f;
        if (k < DF2_TOPK) {
            const unsigned int cand = candidates[(size_t)t * DF2_TOPK + k];
            const __nv_bfloat16* sr = succ + (size_t)cand * (size_t)rank;
            // gate[r] = pred[prev][r] * hidden[t][r]; score contribution is
            // gate[r] * succ[cand][r]. Recomputed per thread rather than
            // staged in shared: rank is 256, and a shared gate would need a
            // second barrier inside the sequential loop for no bandwidth win.
            for (int r = 0; r < rank; ++r) {
                acc += __bfloat162float(pr[r]) * __bfloat162float(hr[r])
                     * __bfloat162float(sr[r]);
            }
            s_score[k] = unary[(size_t)t * DF2_TOPK + k] + acc;
        }
        __syncthreads();

        if (k == 0) {
            float best = s_score[0];
            int best_k = 0;
            #pragma unroll
            for (int j = 1; j < DF2_TOPK; ++j) {
                if (s_score[j] > best) { best = s_score[j]; best_k = j; }
            }
            const unsigned int chosen =
                candidates[(size_t)t * DF2_TOPK + best_k];
            path[t] = chosen;
            s_prev = chosen;
        }
        __syncthreads();
    }
}
