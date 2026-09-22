// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1 routed-MoE epilogues that round to bf16 at the SAME points as the
// serving engine's prefill path (`tools/fp4_moe.py`, `_moe_up_kernel` / `_moe_down_kernel`):
//
//   h   = bf16( silu(min(g, L)) * clamp(u, -L, L) * route_w )   <- the ONLY bf16 rounding
//   out = bf16( sum_k  down_k )                                  <- fp32 over the six picks
//
// The generic `moe_silu_mul` + `moe_unpermute_reduce_indexed` pair rounds gate, up, act
// and every expert's down output to bf16 as well, which measured rel_l2 ~3.8e-3 against
// the engine — the same size as the oracle comparator's 1% control, so indistinguishable
// from noise there. These take fp32 GEMM outputs instead.
//
// Grid: (ceil(rows * inter / 256), 1, 1), Block: (256, 1, 1) for swiglu;
//       (num_tokens, 1, 1), Block: (256, 1, 1) for the unpermute.

#include <cuda_bf16.h>

// gate/up: [rows, inter] fp32 (expert-major permuted rows). row_weight: [rows] — the
// route weight of the (token, pick) that permuted row belongs to.
extern "C" __global__ void dsv41_swiglu_weighted(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ row_weight,
    __nv_bfloat16* __restrict__ h,
    unsigned int rows,
    unsigned int inter,
    float limit
) {
    const unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (idx >= (unsigned long long)rows * inter) return;
    const unsigned int row = (unsigned int)(idx / inter);

    const float g = fminf(gate[idx], limit);
    const float u = fminf(fmaxf(up[idx], -limit), limit);
    const float sig = 1.0f / (1.0f + expf(-g));
    // Same association as the reference: ((g * sig) * u) * w.
    h[idx] = __float2bfloat16(g * sig * u * row_weight[row]);
}

// expert_out: [total_expanded, hidden] fp32, ALREADY weighted (the weight went into h).
// token_to_perm: [num_tokens, topk] -> permuted row. Summed k = 0..topk-1 in order.
extern "C" __global__ void dsv41_unpermute_sum_f32(
    const float* __restrict__ expert_out,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ token_to_perm,
    unsigned int hidden,
    unsigned int num_tokens,
    unsigned int topk
) {
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    for (unsigned int c = threadIdx.x; c < hidden; c += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < topk; ++k) {
            const long long row = token_to_perm[token * topk + k];
            acc += expert_out[row * hidden + c];
        }
        output[(unsigned long long)token * hidden + c] = __float2bfloat16(acc);
    }
}
