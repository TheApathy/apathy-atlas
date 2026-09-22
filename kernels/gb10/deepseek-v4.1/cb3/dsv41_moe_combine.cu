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

// Router top-k on the device: one block of 128 threads per token.
//
//   score  = sqrt(softplus(logit))            (softplus: x > 20 ? x : log1p(exp(x)), fp32)
//   biased = score + (image_row ? bias_vl : bias), or -inf where the expert is NOT resident
//   top-k by biased, ties -> LOWER expert id   (what torch.topk yields here; the host router
//                                               replayed the engine exactly with this rule)
//   w_j    = score_j / (sum_{picks, in pick order} score + 1e-20) * route_scale
//
// Replaces a [T, 384] D2H plus a host sort that measured 6.5 ms per layer at T=2048.
namespace {
constexpr int ROUTE_THREADS = 128;
constexpr int ROUTE_EXPERTS = 384;
constexpr int ROUTE_PER_THREAD = ROUTE_EXPERTS / ROUTE_THREADS;
constexpr int ROUTE_MAX_K = 8;

// (value, id) argmax with the lower id winning ties.
__device__ __forceinline__ bool route_better(float v, int id, float bv, int bid) {
    return v > bv || (v == bv && id < bid);
}
}  // namespace

extern "C" __global__ void __launch_bounds__(ROUTE_THREADS) dsv41_route_topk(
    const float* __restrict__ logits,         // [T, 384] fp32 router GEMM output
    const float* __restrict__ bias,           // [384]
    const float* __restrict__ bias_vl,        // [384]
    const unsigned char* __restrict__ image,  // [T] 1 = image row
    const unsigned char* __restrict__ resident, // [384] 1 = routable
    int* __restrict__ out_idx,                // [T, k]
    float* __restrict__ out_w,                // [T, k]
    unsigned int k,
    float route_scale
) {
    __shared__ float s_val[ROUTE_THREADS / 32];
    __shared__ int s_id[ROUTE_THREADS / 32];
    __shared__ int s_pick[ROUTE_MAX_K];
    __shared__ float s_score[ROUTE_EXPERTS];

    const unsigned int token = blockIdx.x;
    const float* row = logits + (unsigned long long)token * ROUTE_EXPERTS;
    const float* b = image[token] ? bias_vl : bias;

    float biased[ROUTE_PER_THREAD];
#pragma unroll
    for (int j = 0; j < ROUTE_PER_THREAD; ++j) {
        const int e = threadIdx.x + j * ROUTE_THREADS;
        const float x = row[e];
        const float sp = x > 20.0f ? x : log1pf(expf(x));
        const float score = sqrtf(sp);
        s_score[e] = score;
        biased[j] = resident[e] ? score + b[e] : -INFINITY;
    }

    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (unsigned int pick = 0; pick < k; ++pick) {
        float bv = -INFINITY;
        int bid = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < ROUTE_PER_THREAD; ++j) {
            const int e = threadIdx.x + j * ROUTE_THREADS;
            if (route_better(biased[j], e, bv, bid)) { bv = biased[j]; bid = e; }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const int oid = __shfl_down_sync(0xffffffffu, bid, off);
            if (route_better(ov, oid, bv, bid)) { bv = ov; bid = oid; }
        }
        if (lane == 0) { s_val[warp] = bv; s_id[warp] = bid; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float fv = s_val[0];
            int fid = s_id[0];
            for (int w = 1; w < ROUTE_THREADS / 32; ++w)
                if (route_better(s_val[w], s_id[w], fv, fid)) { fv = s_val[w]; fid = s_id[w]; }
            s_pick[pick] = fid;
        }
        __syncthreads();
        const int chosen = s_pick[pick];
#pragma unroll
        for (int j = 0; j < ROUTE_PER_THREAD; ++j)
            if (threadIdx.x + j * ROUTE_THREADS == chosen) biased[j] = -INFINITY;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (unsigned int p = 0; p < k; ++p) sum += s_score[s_pick[p]];
        const float den = sum + 1e-20f;
        for (unsigned int p = 0; p < k; ++p) {
            out_idx[token * k + p] = s_pick[p];
            out_w[token * k + p] = s_score[s_pick[p]] / den * route_scale;
        }
    }
}
