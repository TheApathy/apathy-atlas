// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1 DSpark draft/verify glue: the Markov-head draft loop, greedy row argmax, and the
// greedy accept rule. Everything reads its token ids from device memory, so a draft step and a
// verify step need no host round trip until the single accept readback.
//
// Reference: engine/model.py::dspark_draft (Markov bias) and v41_engine.py lean greedy verify.

#include <cuda_bf16.h>
#include <stdint.h>

#define DS_WARPS 8
#define DS_RANK 256

// lg[v] = f32(logits_row[v]) + f32(bf16(markov_head[v] . markov_embed[ids[i]]))   (fp32 accumulate)
// F.linear(e.bf16, markov_head) rounds its output to bf16 before `.float()`, as done here.
// Grid (ceil(V / DS_WARPS)), Block (32 * DS_WARPS). Warp -> one vocab row.
extern "C" __global__ void __launch_bounds__(32 * DS_WARPS) dsv41_markov_bias(
    const __nv_bfloat16* __restrict__ logits_row, const __nv_bfloat16* __restrict__ embed,
    const __nv_bfloat16* __restrict__ head, const uint32_t* __restrict__ ids, const int i,
    float* __restrict__ lg, const unsigned V) {
    const unsigned warp = threadIdx.x >> 5, L = threadIdx.x & 31;
    const unsigned v = blockIdx.x * DS_WARPS + warp;
    if (v >= V) return;
    const uint32_t tok = ids[i];
    // 256 = 32 lanes x 8: one uint4 of the embedding row and of the head row per lane.
    const uint4 e = __ldg(reinterpret_cast<const uint4*>(embed + (size_t)tok * DS_RANK) + L);
    const uint4 w = __ldg(reinterpret_cast<const uint4*>(head + (size_t)v * DS_RANK) + L);
    const uint32_t ee[4] = {e.x, e.y, e.z, e.w}, ww[4] = {w.x, w.y, w.z, w.w};
    float acc = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        acc = __fadd_rn(acc, __fmul_rn(__uint_as_float(ee[q] << 16), __uint_as_float(ww[q] << 16)));
        acc = __fadd_rn(acc, __fmul_rn(__uint_as_float(ee[q] & 0xffff0000u), __uint_as_float(ww[q] & 0xffff0000u)));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (L == 0) lg[v] = __bfloat162float(logits_row[v]) + __bfloat162float(__float2bfloat16(acc));
}

// Argmax of one fp32 row -> out[slot] (ties: the LOWEST index, as torch.argmax). Grid (1), Block 1024.
extern "C" __global__ void __launch_bounds__(1024) dsv41_argmax_f32(const float* __restrict__ x, const unsigned V,
                                                                    uint32_t* __restrict__ out, const int slot) {
    __shared__ float sv[32];
    __shared__ unsigned si[32];
    float best = -INFINITY;
    unsigned bi = 0xffffffffu;
    for (unsigned v = threadIdx.x; v < V; v += blockDim.x) {
        const float f = x[v];
        if (f > best) { best = f; bi = v; }  // v ascends per thread: strict > keeps the lowest index
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float of = __shfl_xor_sync(0xffffffffu, best, o);
        const unsigned oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (of > best || (of == best && oi < bi)) { best = of; bi = oi; }
    }
    const int w = threadIdx.x >> 5, L = threadIdx.x & 31;
    if (L == 0) { sv[w] = best; si[w] = bi; }
    __syncthreads();
    if (w == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        best = L < nw ? sv[L] : -INFINITY;
        bi = L < nw ? si[L] : 0xffffffffu;
        for (int o = 16; o > 0; o >>= 1) {
            const float of = __shfl_xor_sync(0xffffffffu, best, o);
            const unsigned oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (of > best || (of == best && oi < bi)) { best = of; bi = oi; }
        }
        if (L == 0) out[slot] = bi;
    }
}

// Argmax of each bf16 logits row r (stride V) -> am[r]. Grid (T), Block 1024.
extern "C" __global__ void __launch_bounds__(1024) dsv41_argmax_rows_bf16(const __nv_bfloat16* __restrict__ x,
                                                                          const unsigned V, uint32_t* __restrict__ am) {
    __shared__ float sv[32];
    __shared__ unsigned si[32];
    const __nv_bfloat16* row = x + (size_t)blockIdx.x * V;
    float best = -INFINITY;
    unsigned bi = 0xffffffffu;
    for (unsigned v = threadIdx.x; v < V; v += blockDim.x) {
        const float f = __bfloat162float(row[v]);
        if (f > best) { best = f; bi = v; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float of = __shfl_xor_sync(0xffffffffu, best, o);
        const unsigned oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (of > best || (of == best && oi < bi)) { best = of; bi = oi; }
    }
    const int w = threadIdx.x >> 5, L = threadIdx.x & 31;
    if (L == 0) { sv[w] = best; si[w] = bi; }
    __syncthreads();
    if (w == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        best = L < nw ? sv[L] : -INFINITY;
        bi = L < nw ? si[L] : 0xffffffffu;
        for (int o = 16; o > 0; o >>= 1) {
            const float of = __shfl_xor_sync(0xffffffffu, best, o);
            const unsigned oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (of > best || (of == best && oi < bi)) { best = of; bi = oi; }
        }
        if (L == 0) am[blockIdx.x] = bi;
    }
}

// Greedy accept: a = number of LEADING drafts[i] == am[i] (i < B), bonus = am[a].
// res = {a, am[0..=B], drafts[0..B]} (2B + 2 u32): ONE readback carries everything the host needs.
// Grid (1), Block (1).
extern "C" __global__ void dsv41_dspark_accept(const uint32_t* __restrict__ am, const uint32_t* __restrict__ drafts,
                                               const int B, uint32_t* __restrict__ res) {
    int a = 0;
    while (a < B && am[a] == drafts[a]) ++a;
    res[0] = (uint32_t)a;
    for (int i = 0; i <= B; ++i) res[1 + i] = am[i];
    for (int i = 0; i < B; ++i) res[2 + B + i] = drafts[i];
}

// Sampled draft (T > 0): q = softmax(lg / T) over the vocab (fp32, as Python's
// `torch.softmax(lg / temperature, -1)`, no top_p), written to `q`; then ids[slot] = the
// inverse-CDF sample of q at uniform `u` (a host/request RNG value, so a step is reproducible).
// Deterministic: fixed thread->range split, an in-order block scan. Grid (1), Block 1024.
extern "C" __global__ void __launch_bounds__(1024) dsv41_sample_softmax(const float* __restrict__ lg, const unsigned V,
                                                                         const float temperature, const float u,
                                                                         float* __restrict__ q, uint32_t* __restrict__ ids,
                                                                         const int slot) {
    __shared__ float red[1024];
    __shared__ float part[1024];
    const int tid = threadIdx.x, nt = blockDim.x;
    const float inv_t = 1.0f / temperature;
    float m = -INFINITY;
    for (unsigned v = tid; v < V; v += nt) m = fmaxf(m, lg[v] * inv_t);
    red[tid] = m;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float mx = red[0];
    __syncthreads();
    // Contiguous ranges per thread so the CDF order is the vocab order.
    const unsigned chunk = (V + nt - 1) / nt;
    const unsigned lo = tid * chunk, hi = min(V, lo + chunk);
    float s = 0.0f;
    for (unsigned v = lo; v < hi; ++v) {
        const float e = expf(lg[v] * inv_t - mx);
        q[v] = e;
        s += e;
    }
    part[tid] = s;
    __syncthreads();
    if (tid == 0) {  // in-order exclusive scan
        float acc = 0.0f;
        for (int i = 0; i < nt; ++i) { const float x = part[i]; part[i] = acc; acc += x; }
        red[0] = acc;
    }
    __syncthreads();
    const float total = red[0];
    const float inv = 1.0f / total;
    const float target = u * total;
    for (unsigned v = lo; v < hi; ++v) q[v] *= inv;
    // The thread whose range holds the target walks it. (Exactly one does; the last thread
    // with a non-empty range takes u rounding past the total.)
    const float base = part[tid];
    const float end = tid + 1 < nt ? part[tid + 1] : total;
    const bool mine = (target >= base && target < end) || (tid == nt - 1 && target >= total);
    if (mine && lo < hi) {
        float acc = base;
        unsigned pick = hi - 1;
        for (unsigned v = lo; v < hi; ++v) {
            acc += q[v] * total;
            if (acc > target) { pick = v; break; }
        }
        ids[slot] = pick;
    } else if (mine) {
        ids[slot] = V - 1;
    }
}
