// SPDX-License-Identifier: AGPL-3.0-only
//
// DSpark block-drafter kernels (docs/dspark_port.md).
//
// The drafter's attention is unlike anything the target runs: 5 query rows
// (1 committed + 4 noise), each attending bidirectionally over ALL 5 block
// rows plus a 128-entry sliding window of `main_kv` rows, MQA (one shared
// 512-d KV row per position), with a per-head attention-sink logit joining
// the softmax denominator only. Sizes are tiny (5×64 heads × ≤133 keys), so
// these kernels optimize for CORRECTNESS against the official
// inference/model.py reference, not bandwidth — the propose forward is
// dominated by the MoE and lm_head reads, not attention.
//
// Rope convention (reference `precompute_freqs_cis` / `apply_rotary_emb`):
// plain θ=10000, NO YaRN (the drafter is pure sliding-window attention),
// INTERLEAVED pairs — the last `rope_dim` dims of each head are viewed as
// [rope_dim/2, 2] (re, im) and multiplied by e^{i·pos·freq_j},
// freq_j = θ^(-2j/rope_dim). `inverse` multiplies by the conjugate (the MLA
// output de-rotation).

#include <cuda_bf16.h>

// ── dspark_rope ──
// x [rows, heads, head_dim] BF16; rotates x[..., head_dim-rope_dim:] in
// place. Row r uses position `pos_base + r * pos_stride` (stride 0 = all
// rows at pos_base, e.g. the single main_kv row; stride 1 = the block rows).
// freqs [max_pos, rope_dim/2, 2] F32 = (cos, sin) precomputed host-side.
// Grid: (rows, heads)  Block: (rope_dim/2, 1, 1) — one thread per pair.
extern "C" __global__ void dspark_rope(
    __nv_bfloat16* __restrict__ x,
    const float* __restrict__ freqs,
    const unsigned int heads,
    const unsigned int head_dim,
    const unsigned int rope_dim,
    const unsigned int pos_base,
    const unsigned int pos_stride,
    const unsigned int inverse
) {
    const unsigned int r = blockIdx.x;
    const unsigned int hh = blockIdx.y;
    const unsigned int j = threadIdx.x;              // pair index
    if (hh >= heads || j >= rope_dim / 2) return;
    const unsigned int pos = pos_base + r * pos_stride;
    const float c = freqs[((size_t)pos * (rope_dim / 2) + j) * 2 + 0];
    float s = freqs[((size_t)pos * (rope_dim / 2) + j) * 2 + 1];
    if (inverse) s = -s;
    __nv_bfloat16* p = x + ((size_t)r * heads + hh) * head_dim
                         + (head_dim - rope_dim) + 2 * j;
    const float re = __bfloat162float(p[0]);
    const float im = __bfloat162float(p[1]);
    p[0] = __float2bfloat16(re * c - im * s);
    p[1] = __float2bfloat16(re * s + im * c);
}

// ── dspark_attn ──
// q       [rows, heads, head_dim] BF16 (roped)
// ring    [ring_cap, head_dim]    BF16 — main_kv sliding window; slots
//                                 0..ring_vis-1 are valid (reference
//                                 topk_idxs: arange(min(win, pos+1)))
// blk_kv  [rows, head_dim]        BF16 — the block's own KV rows
// sink    [heads] F32             — per-head sink logit: joins the softmax
//                                 denominator, contributes no value
// o       [rows, heads, head_dim] BF16
// Every query row sees ring[0..ring_vis) ∥ blk_kv[0..rows) — NO causal mask
// inside the block (reference: full bidirectional). V = K (the full 512-d
// row; the caller de-rotates o's tail afterwards).
// Grid: (rows, heads)  Block: (128, 1, 1).
#define DSPARK_MAX_KEYS 160   // 128-window + block rows; production is 133
extern "C" __global__ void dspark_attn(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ ring,
    const __nv_bfloat16* __restrict__ blk_kv,
    const float* __restrict__ sink,
    __nv_bfloat16* __restrict__ o,
    const unsigned int rows,
    const unsigned int heads,
    const unsigned int head_dim,
    const unsigned int ring_vis,
    const float scale
) {
    const unsigned int r = blockIdx.x;
    const unsigned int hh = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int nk = ring_vis + rows;

    __shared__ float s_scores[DSPARK_MAX_KEYS];
    __shared__ float s_red[128];
    __shared__ float s_max, s_denom;

    const __nv_bfloat16* qp = q + ((size_t)r * heads + hh) * head_dim;

    // Pass 1: raw scores, one key per thread stride.
    for (unsigned int k = tid; k < nk; k += blockDim.x) {
        const __nv_bfloat16* kv = (k < ring_vis)
            ? ring + (size_t)k * head_dim
            : blk_kv + (size_t)(k - ring_vis) * head_dim;
        float acc = 0.f;
        for (unsigned int d = 0; d < head_dim; ++d) {
            acc += __bfloat162float(qp[d]) * __bfloat162float(kv[d]);
        }
        s_scores[k] = acc * scale;
    }
    __syncthreads();

    // Max over keys. The sink joins the DENOMINATOR only, but it must join
    // the max too so its exponent is computed on the same shifted basis the
    // reference uses (sum_exp += exp(sink - max)).
    float m = sink[hh];
    for (unsigned int k = tid; k < nk; k += blockDim.x) m = fmaxf(m, s_scores[k]);
    s_red[tid] = m;
    __syncthreads();
    for (unsigned int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (tid < w) s_red[tid] = fmaxf(s_red[tid], s_red[tid + w]);
        __syncthreads();
    }
    if (tid == 0) s_max = s_red[0];
    __syncthreads();
    const float mx = s_max;

    // exp + denominator (with the sink term).
    float dsum = 0.f;
    for (unsigned int k = tid; k < nk; k += blockDim.x) {
        const float e = expf(s_scores[k] - mx);
        s_scores[k] = e;
        dsum += e;
    }
    s_red[tid] = dsum;
    __syncthreads();
    for (unsigned int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (tid < w) s_red[tid] += s_red[tid + w];
        __syncthreads();
    }
    if (tid == 0) s_denom = s_red[0] + expf(sink[hh] - mx);
    __syncthreads();
    const float inv_denom = 1.0f / s_denom;

    // Pass 2: o[d] = Σ_k w_k · kv[k][d], threads own dims.
    __nv_bfloat16* op = o + ((size_t)r * heads + hh) * head_dim;
    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (unsigned int k = 0; k < nk; ++k) {
            const __nv_bfloat16* kv = (k < ring_vis)
                ? ring + (size_t)k * head_dim
                : blk_kv + (size_t)(k - ring_vis) * head_dim;
            acc += s_scores[k] * __bfloat162float(kv[d]);
        }
        op[d] = __float2bfloat16(acc * inv_denom);
    }
}
