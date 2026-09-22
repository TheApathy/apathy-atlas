// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1 routed MoE for DECODE-SIZED passes (T <= 8 tokens, <= 8 rows per expert):
// GPU routing + a CB3 grouped GEMV that decodes the 3-bit codes in registers and never
// writes a bf16 copy of an expert.
//
// Why a separate path from `cb3_reconstruct_bf16` + cuBLASLt: at T = 1 each routed expert
// feeds ONE row, so the reconstruct wrote and re-read ~70.8 MB of bf16 per expert to serve
// a matrix-VECTOR product (~250 ms/token over 40 layers). This path reads the packed planes
// once: 6 x 14.45 MB = 87 MB per layer at T = 1, ~0.36 ms at 240 GB/s.
//
// NUMERICS. A CB3 weight is fp4[code] * 2^(s-127): exact in fp32 (and bf16). Its product with
// a bf16 activation is exact in fp32 (2 + 8 significant bits). So every dot product below
// differs from the reconstruct + cuBLASLt path ONLY in fp32 accumulation order. The rounding
// points are the engine's (`dsv41_moe_combine.cu`):
//     h   = bf16( silu(min(g, L)) * clamp(u, -L, L) * route_w )
//     out = bf16( sum_{k = 0..5, in pick order} down_k )           (fp32 sum)
// Scaling a partial sum by the group's 2^(s-127) instead of each weight is exact (power of
// two, no subnormals at these magnitudes), so it is not a numerics change.
//
// Routing (`dsv41_route_decode`) is a port of `routing.rs::select_experts_multimodal` with
// the score of `score_of` / torch softplus: sqrt(x > 20 ? x : log1p(exp(x))). Masked experts
// are -inf before top-k; ties break to the LOWER expert id; weights come from the SCORES
// (not the biased keys), summed in pick order, / (sum + 1e-20) * route_scale.
//
// Layouts:
//   y       bf16 [T, 5120]          (post ffn_norm; MM_TILE slack rows are never read)
//   logits  f32  [T, 384]
//   groups  i32  [1 + 48 * GROUP_INTS]: n_groups, then per group {slot, nrows, row[8]}
//           where row = t * 6 + k (the (token, pick) it serves)
//   row_w   f32  [T * 6]            route weight of (t, k)
//   sel     i32  [T * 6]            routed expert id of (t, k)   (for gates; not consumed)
//   h       bf16 [T * 6, 2304]
//   down    f32  [T * 6, 5120]
//   out     bf16 [T, 5120]
// CB3 planes are passed as the slot-0 address of each resident plane plus the per-slot
// stride in bytes (the arena is expert-major per plane; see cb3_arena.rs).

#include <cuda_bf16.h>
#include <stdint.h>

#define DEC_TOPK 6
#define DEC_EXPERTS 384
#define DEC_MAXR 8
#define DEC_MAXT 8
#define DEC_GROUP_INTS (2 + DEC_MAXR)
#define DEC_WARPS 8

__constant__ float kDecFp4[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float dec_pow2(uint32_t s) { return ldexpf(1.0f, (int)s - 127); }

__device__ __forceinline__ float dec_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// 8 consecutive bf16 -> float.
__device__ __forceinline__ void dec_load8(const __nv_bfloat16* p, float* f) {
    const uint4 v = __ldg(reinterpret_cast<const uint4*>(p));
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i] = __uint_as_float(w[i] << 16);
        f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

// The row's 8-entry codebook as bf16 (fp4 value, UNscaled: exact), split into a low-byte and a
// high-byte table so `prmt` can select four weights at once (engine2's cb3_moe_gemm.cu trick,
// ece90b487). Selected values are exactly kDecFp4[code] -- the same floats the previous
// shared-memory lookup produced -- so every product and sum below is unchanged.
struct DecCb {
    uint32_t lx, ly, hx, hy;
};

__device__ __forceinline__ uint32_t dec_prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}

__device__ __forceinline__ DecCb dec_cb_tables(const uint8_t* __restrict__ cb_row) {
    const uint2 c = __ldg(reinterpret_cast<const uint2*>(cb_row));
    uint32_t e[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t w = i < 4 ? c.x : c.y;
        e[i] = __float_as_uint(kDecFp4[(w >> (8 * (i % 4))) & 0xfu]) >> 16;  // fp4 is exact in bf16
    }
    DecCb t;
    t.lx = (e[0] & 0xffu) | ((e[1] & 0xffu) << 8) | ((e[2] & 0xffu) << 16) | ((e[3] & 0xffu) << 24);
    t.ly = (e[4] & 0xffu) | ((e[5] & 0xffu) << 8) | ((e[6] & 0xffu) << 16) | ((e[7] & 0xffu) << 24);
    t.hx = (e[0] >> 8) | ((e[1] >> 8) << 8) | ((e[2] >> 8) << 16) | ((e[3] >> 8) << 24);
    t.hy = (e[4] >> 8) | ((e[5] >> 8) << 8) | ((e[6] >> 8) << 16) | ((e[7] >> 8) << 24);
    return t;
}

// Four weights (bytes j = 0..3 of lo/hi) for one (g parity, r): their fp32 values in w[0..3].
// lo bits at shift SH = 4*godd + 2r, hi bit at HB = nib + 2*godd + r.
__device__ __forceinline__ void dec_pick4(uint32_t lo, uint32_t hi, int sh, int hb, const DecCb& t, float* w) {
    const uint32_t a = (lo >> sh) & 0x03030303u;
    const uint32_t b = ((hi >> hb) & 0x01010101u) << 2;
    const uint32_t ie = a | b;                 // one index 0..7 per byte
    const uint32_t u = ie | (ie >> 4);
    const uint32_t sel = (u & 0xffu) | ((u >> 8) & 0xff00u);
    const uint32_t lb = dec_prmt(t.lx, t.ly, sel), hb4 = dec_prmt(t.hx, t.hy, sel);
    const uint32_t p01 = dec_prmt(lb, hb4, 0x5140), p23 = dec_prmt(lb, hb4, 0x7362);
    w[0] = __uint_as_float(p01 << 16);
    w[1] = __uint_as_float(p01 & 0xffff0000u);
    w[2] = __uint_as_float(p23 << 16);
    w[3] = __uint_as_float(p23 & 0xffff0000u);
}

// One lane's 16 weights of one 512-wide K block (see CB3_FORMAT.md):
//   lane L reads lo u32 at block*128 + 4L  -> lo bytes (gg = L/4, lane0 = 4*(L%4) .. +3)
//   hi u32 at block*64 + (L/8)*16 + 4*(L%4), nibble gg%2
//   byte j of those words covers k = block*512 + g*32 + (lane0+j)*2 + r, g in {2gg, 2gg+1}
// so the activations are two contiguous 8-element runs: xe = block*512 + 64*gg + 8*(L%4)
// (even g) and xe + 32 (odd g). Adds sum(w*x) for each row into acc[r].
template <int R>
__device__ __forceinline__ void dec_cb3_chunk(uint32_t lo, uint32_t hi, uint32_t sc16, int gg,
                                              const DecCb& cbt,
                                              const __nv_bfloat16* const* xr, int xe, int nr,
                                              float* acc) {
    const int nib = (gg & 1) * 4;
    // we/wo[2j + r] = weight of byte j, sub-position r, even/odd scale group (same order as before).
    float we[8], wo[8], t4[4];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        dec_pick4(lo, hi, 2 * r, nib + r, cbt, t4);
#pragma unroll
        for (int j = 0; j < 4; ++j) we[2 * j + r] = t4[j];
        dec_pick4(lo, hi, 4 + 2 * r, nib + 2 + r, cbt, t4);
#pragma unroll
        for (int j = 0; j < 4; ++j) wo[2 * j + r] = t4[j];
    }
    const float se = dec_pow2(sc16 & 0xffu);
    const float so = dec_pow2((sc16 >> 8) & 0xffu);
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (r >= nr) break;
        float xa[8], xb[8];
        dec_load8(xr[r] + xe, xa);
        dec_load8(xr[r] + xe + 32, xb);
        float pe = 0.0f, po = 0.0f;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            pe = __fadd_rn(pe, __fmul_rn(we[q], xa[q]));
            po = __fadd_rn(po, __fmul_rn(wo[q], xb[q]));
        }
        acc[r] = __fadd_rn(acc[r], __fmul_rn(pe, se));
        acc[r] = __fadd_rn(acc[r], __fmul_rn(po, so));
    }
}

// Dot products of one CB3 row (K = NB*512 + (TAIL ? 256 : 0)) against R activation rows.
template <int R, int NB, bool TAIL>
__device__ __forceinline__ void dec_cb3_row(const uint8_t* __restrict__ lo_row,
                                            const uint8_t* __restrict__ hi_row,
                                            const uint8_t* __restrict__ sc_row,
                                            const DecCb& cbt,
                                            const __nv_bfloat16* const* xr, int nr, float* acc) {
    const int L = threadIdx.x & 31;
    const int gg = L >> 2;
    constexpr int NT = NB + (TAIL ? 1 : 0);
    uint32_t lo[NT], hi[NT], sc[NT];
    // Issue every load of the row before any math: ~2 KB in flight per warp.
#pragma unroll
    for (int b = 0; b < NT; ++b) {
        const bool live = b < NB || L < 16;
        lo[b] = live ? __ldg(reinterpret_cast<const uint32_t*>(lo_row + b * 128 + 4 * L)) : 0u;
        hi[b] = live ? __ldg(reinterpret_cast<const uint32_t*>(hi_row + b * 64 + (L >> 3) * 16 + 4 * (L & 3))) : 0u;
        sc[b] = live ? (uint32_t)__ldg(reinterpret_cast<const uint16_t*>(sc_row + b * 16 + 2 * gg)) : 0u;
    }
#pragma unroll
    for (int b = 0; b < NT; ++b) {
        if (b < NB || L < 16)
            dec_cb3_chunk<R>(lo[b], hi[b], sc[b], gg, cbt, xr, b * 512 + 64 * gg + 8 * (L & 3), nr, acc);
    }
}

// ── routing ──────────────────────────────────────────────────────────────────
// Grid (1), Block (32 * T). Warp t routes token t; then thread 0 groups the picks by slot
// (first-occurrence order) for the GEMVs.
extern "C" __global__ void dsv41_route_decode(
    const float* __restrict__ logits,        // [T, 384]
    const float* __restrict__ bias,          // [384]
    const float* __restrict__ bias_vl,       // [384]
    const uint32_t* __restrict__ ids,        // [T] token ids (image rows use bias_vl)
    const uint8_t* __restrict__ resident,    // [384] routing mask
    const int* __restrict__ slot_of,         // [384] routed id -> slot, -1 = none
    int* __restrict__ groups,                // [1 + 48 * GROUP_INTS]
    float* __restrict__ row_w,               // [T * 6]
    int* __restrict__ sel,                   // [T * 6]
    int* __restrict__ err,                   // [1], set nonzero on an impossible pick
    const int T, const float route_scale) {
    __shared__ int s_slot[DEC_MAXT * DEC_TOPK];
    const int t = threadIdx.x >> 5, L = threadIdx.x & 31;
    if (t < T) {
        const uint32_t id = ids[t];
        const float* b = (id == 129264u || id == 129265u) ? bias_vl : bias;
        float sc[12], key[12];
#pragma unroll
        for (int i = 0; i < 12; ++i) {
            const int e = L + 32 * i;
            const float x = logits[t * DEC_EXPERTS + e];
            const float sp = x > 20.0f ? x : log1pf(expf(x));
            sc[i] = sqrtf(sp);
            key[i] = resident[e] ? sc[i] + b[e] : -INFINITY;
        }
        float picked_score[DEC_TOPK];
        int picked[DEC_TOPK];
        for (int k = 0; k < DEC_TOPK; ++k) {
            // Lane-local best: ids ascend with i, so a strict > keeps the lower id on a tie.
            // Removed and masked entries are -inf and never win (bi stays INT_MAX).
            float bk = -INFINITY;
            int bi = 0x7fffffff;
#pragma unroll
            for (int i = 0; i < 12; ++i) {
                if (key[i] > bk) {
                    bk = key[i];
                    bi = L + 32 * i;
                }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ok = __shfl_xor_sync(0xffffffffu, bk, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ok > bk || (ok == bk && oi < bi)) {
                    bk = ok;
                    bi = oi;
                }
            }
            if (bi == 0x7fffffff) {  // fewer than 6 resident: never pick a masked expert
                if (L == 0) atomicExch(err, 1);
                bi = 0;
            }
            picked[k] = bi;
            // The owner lane removes it and publishes its SCORE.
            float s = 0.0f;
            if ((bi & 31) == L) {
#pragma unroll
                for (int i = 0; i < 12; ++i)
                    if (L + 32 * i == bi) {
                        s = sc[i];
                        key[i] = -INFINITY;
                    }
            }
            picked_score[k] = __shfl_sync(0xffffffffu, s, bi & 31);
        }
        if (L == 0) {
            float sum = 0.0f;
            for (int k = 0; k < DEC_TOPK; ++k) sum += picked_score[k];
            const float den = sum + 1e-20f;
            for (int k = 0; k < DEC_TOPK; ++k) {
                row_w[t * DEC_TOPK + k] = picked_score[k] / den * route_scale;
                sel[t * DEC_TOPK + k] = picked[k];
                const int s = slot_of[picked[k]];
                if (s < 0) atomicExch(err, 2);
                s_slot[t * DEC_TOPK + k] = s < 0 ? 0 : s;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int n = 0;
        for (int p = 0; p < T * DEC_TOPK; ++p) {
            int g = 0;
            while (g < n && groups[1 + g * DEC_GROUP_INTS] != s_slot[p]) ++g;
            int* G = groups + 1 + g * DEC_GROUP_INTS;
            if (g == n) {
                G[0] = s_slot[p];
                G[1] = 0;
                ++n;
            }
            G[2 + G[1]] = p;
            G[1] += 1;
        }
        groups[0] = n;
    }
}

// ── gate + up + SwiGLU ───────────────────────────────────────────────────────
// Grid (2304 / DEC_WARPS, max_groups), Block (32 * DEC_WARPS). Warp -> one output row n of
// one expert group: both W1[n] and W3[n] against the group's rows.
template <int R>
__device__ __forceinline__ void dec_gateup(
    const uint8_t* __restrict__ w1lo, const uint8_t* __restrict__ w1hi,
    const uint8_t* __restrict__ w1cb, const uint8_t* __restrict__ s1,
    const uint8_t* __restrict__ w3lo, const uint8_t* __restrict__ w3hi,
    const uint8_t* __restrict__ w3cb, const uint8_t* __restrict__ s3,
    unsigned long long lo_stride, unsigned long long hi_stride,
    unsigned long long cb_stride, unsigned long long sc_stride,
    const __nv_bfloat16* __restrict__ y, const int* __restrict__ groups,
    const float* __restrict__ row_w, __nv_bfloat16* __restrict__ h, float limit) {
    constexpr int K = 5120, N = 2304;
    const int g = blockIdx.y;
    if (g >= groups[0]) return;
    const int warp = threadIdx.x >> 5, L = threadIdx.x & 31;
    const int n = blockIdx.x * DEC_WARPS + warp;
    if (n >= N) return;
    const int* G = groups + 1 + g * DEC_GROUP_INTS;
    const unsigned long long slot = (unsigned long long)G[0];
    const int nr = min(G[1], R);
    const unsigned long long rw = (unsigned long long)n;
    const DecCb cb1 = dec_cb_tables(w1cb + slot * cb_stride + rw * 8);
    const DecCb cb3 = dec_cb_tables(w3cb + slot * cb_stride + rw * 8);
    const __nv_bfloat16* xr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) xr[r] = y + (size_t)(G[2 + (r < nr ? r : 0)] / DEC_TOPK) * K;
    float ag[R], au[R];
#pragma unroll
    for (int r = 0; r < R; ++r) ag[r] = au[r] = 0.0f;
    dec_cb3_row<R, 10, false>(w1lo + slot * lo_stride + rw * (K / 4), w1hi + slot * hi_stride + rw * (K / 8),
                              s1 + slot * sc_stride + rw * (K / 32), cb1, xr, nr, ag);
    dec_cb3_row<R, 10, false>(w3lo + slot * lo_stride + rw * (K / 4), w3hi + slot * hi_stride + rw * (K / 8),
                              s3 + slot * sc_stride + rw * (K / 32), cb3, xr, nr, au);
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (r >= nr) break;
        const float gs = dec_warp_sum(ag[r]);
        const float us = dec_warp_sum(au[r]);
        if (L == 0) {
            const int row = G[2 + r];
            const float gv = fminf(gs, limit);
            const float uv = fminf(fmaxf(us, -limit), limit);
            const float sig = 1.0f / (1.0f + expf(-gv));
            h[(size_t)row * N + n] = __float2bfloat16(gv * sig * uv * row_w[row]);
        }
    }
}

#define DEC_GATEUP_ENTRY(NAME, R)                                                                   \
    extern "C" __global__ void __launch_bounds__(32 * DEC_WARPS) NAME(                             \
        const uint8_t* w1lo, const uint8_t* w1hi, const uint8_t* w1cb, const uint8_t* s1,          \
        const uint8_t* w3lo, const uint8_t* w3hi, const uint8_t* w3cb, const uint8_t* s3,          \
        unsigned long long lo_stride, unsigned long long hi_stride, unsigned long long cb_stride,  \
        unsigned long long sc_stride, const __nv_bfloat16* y, const int* groups,                  \
        const float* row_w, __nv_bfloat16* h, float limit) {                                       \
        dec_gateup<R>(w1lo, w1hi, w1cb, s1, w3lo, w3hi, w3cb, s3, lo_stride, hi_stride, cb_stride, \
                      sc_stride, y, groups, row_w, h, limit);                                      \
    }
DEC_GATEUP_ENTRY(dsv41_cb3_gateup_decode_r1, 1)
DEC_GATEUP_ENTRY(dsv41_cb3_gateup_decode_r8, 8)

// ── down ─────────────────────────────────────────────────────────────────────
// Grid (5120 / DEC_WARPS, max_groups), Block (32 * DEC_WARPS). K = 2304 = 4 x 512 + 256.
template <int R>
__device__ __forceinline__ void dec_down(
    const uint8_t* __restrict__ lo, const uint8_t* __restrict__ hi,
    const uint8_t* __restrict__ cb, const uint8_t* __restrict__ s2,
    unsigned long long lo_stride, unsigned long long hi_stride,
    unsigned long long cb_stride, unsigned long long sc_stride,
    const __nv_bfloat16* __restrict__ h, const int* __restrict__ groups, float* __restrict__ down) {
    constexpr int K = 2304, N = 5120;
    const int g = blockIdx.y;
    if (g >= groups[0]) return;
    const int warp = threadIdx.x >> 5, L = threadIdx.x & 31;
    const int n = blockIdx.x * DEC_WARPS + warp;
    if (n >= N) return;
    const int* G = groups + 1 + g * DEC_GROUP_INTS;
    const unsigned long long slot = (unsigned long long)G[0];
    const int nr = min(G[1], R);
    const unsigned long long rw = (unsigned long long)n;
    const DecCb cbt = dec_cb_tables(cb + slot * cb_stride + rw * 8);
    const __nv_bfloat16* xr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) xr[r] = h + (size_t)G[2 + (r < nr ? r : 0)] * K;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    dec_cb3_row<R, 4, true>(lo + slot * lo_stride + rw * (K / 4), hi + slot * hi_stride + rw * (K / 8),
                            s2 + slot * sc_stride + rw * (K / 32), cbt, xr, nr, acc);
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (r >= nr) break;
        const float v = dec_warp_sum(acc[r]);
        if (L == 0) down[(size_t)G[2 + r] * N + n] = v;
    }
}

#define DEC_DOWN_ENTRY(NAME, R)                                                                    \
    extern "C" __global__ void __launch_bounds__(32 * DEC_WARPS) NAME(                             \
        const uint8_t* lo, const uint8_t* hi, const uint8_t* cb, const uint8_t* s2,                \
        unsigned long long lo_stride, unsigned long long hi_stride, unsigned long long cb_stride,  \
        unsigned long long sc_stride, const __nv_bfloat16* h, const int* groups, float* down) {    \
        dec_down<R>(lo, hi, cb, s2, lo_stride, hi_stride, cb_stride, sc_stride, h, groups, down);  \
    }
DEC_DOWN_ENTRY(dsv41_cb3_down_decode_r1, 1)
DEC_DOWN_ENTRY(dsv41_cb3_down_decode_r8, 8)

// ── sum over the six picks ───────────────────────────────────────────────────
// Grid (ceil(5120 / 256), T), Block (256). out[t, c] = bf16(sum_{k=0..5} down[t*6+k, c]).
extern "C" __global__ void dsv41_moe_sum_decode(const float* __restrict__ down,
                                                __nv_bfloat16* __restrict__ out) {
    constexpr int N = 5120;
    const int t = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < DEC_TOPK; ++k) acc += down[(size_t)(t * DEC_TOPK + k) * N + c];
    out[(size_t)t * N + c] = __float2bfloat16(acc);
}

// ── router logits ────────────────────────────────────────────────────────────
// logits[t, e] = sum_k f32(y[t, k]) * gate_w[e, k], fp32 (bf16 -> fp32 is exact).
// Grid (384 / DEC_WARPS, T), Block (32 * DEC_WARPS). Warp -> one (t, e).
extern "C" __global__ void __launch_bounds__(32 * DEC_WARPS) dsv41_router_logits_decode(
    const __nv_bfloat16* __restrict__ y, const float* __restrict__ gate_w,
    float* __restrict__ logits) {
    constexpr int K = 5120;
    const int t = blockIdx.y;
    const int warp = threadIdx.x >> 5, L = threadIdx.x & 31;
    const int e = blockIdx.x * DEC_WARPS + warp;
    if (e >= DEC_EXPERTS) return;
    const float4* w = reinterpret_cast<const float4*>(gate_w + (size_t)e * K);
    const uint2* x = reinterpret_cast<const uint2*>(y + (size_t)t * K);
    float acc = 0.0f;
#pragma unroll 8
    for (int i = L; i < K / 4; i += 32) {
        const float4 wv = __ldg(w + i);
        const uint2 xv = __ldg(x + i);
        acc = __fadd_rn(acc, __fmul_rn(wv.x, __uint_as_float(xv.x << 16)));
        acc = __fadd_rn(acc, __fmul_rn(wv.y, __uint_as_float(xv.x & 0xffff0000u)));
        acc = __fadd_rn(acc, __fmul_rn(wv.z, __uint_as_float(xv.y << 16)));
        acc = __fadd_rn(acc, __fmul_rn(wv.w, __uint_as_float(xv.y & 0xffff0000u)));
    }
    acc = dec_warp_sum(acc);
    if (L == 0) logits[t * DEC_EXPERTS + e] = acc;
}
