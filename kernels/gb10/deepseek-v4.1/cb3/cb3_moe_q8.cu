// SPDX-License-Identifier: AGPL-3.0-only
//
// OPT-IN fp8-ACTIVATION routed MoE for DeepSeek-V4.1 (ATLAS_DSV41_MOE_FP8_ACT=1; default OFF).
//
// NOT EXACT. The default path is cb3_moe_gemm.cu (bf16 activations, byte-identical to the
// engine). This path quantizes the MoE input and the gate/up output h to e4m3 with one UE8M0
// scale per row per 32 K (tools/v41_ref.act_qdq_fp8, i.e. DeepSeek's act_quant ue8m0: amax
// floored at 1e-4, scale = 2^ceil(log2(amax / 448)), clamp +-448, round to nearest e4m3) and
// multiplies on the block-scaled tensor core:
//   mma.sync m16n8k32 kind::mxf8f6f4 block_scale  (A e4m3 x B e2m1, ue8m0 per 32 K, fp32 acc)
// A CB3 weight is an e2m1 code times its group's UE8M0 scale, so the group scale IS the B scale
// and the decode is a byte-table lookup: no per-group scale arithmetic, 1 byte per weight.
// Every scale depends on its own row only, so the result is chunk-invariant.
//
// GB10 fragment layout (probed, dsv41-prefill-work/tools/cuda/MMA_LAYOUT_SM121.md): A = the
// standard fp8 m16n8k32 layout; B byte j of b0 = B[4 tig + j, gid], b1 = k + 16; the e2m1 value
// sits at bits [5:2] of its byte (the hardware reads e2m3); with thread-id selector 0 the scale
// of row gid comes from lane tig 0, of row gid + 8 from tig 1, of column gid from tig 0.
//
// WEIGHTS STAY EXACT: every codebook byte of the served pack is an e2m1 code (0..15; scanned
// 2026-09-23 over all 40 layers, 479M bytes, none > 15) and every group scale is a UE8M0 byte
// (117..127), so a CB3 weight IS e2m1 x UE8M0, the value the exact kernel decodes. Only the
// activations (MoE input, h) are approximated.
// GATE (cb3_moe_oracle_microtest --fp8-act): --kernel fused vs --kernel reconstruct, which runs
// dsv41_qdq_e4m3_bf16 / dsv41_swiglu_qdq + bf16 cuBLASLt: rel_l2 2e-6 .. 1e-4 (the block-scaled
// MMA's accumulation is not order-exact; an fp32 difference can flip an e4m3 rounding of h),
// against 2.1-2.4e-2 for the quantization itself vs the exact oracle.
// MEASURED (6 interleaved rounds, ms/layer, exact -> fp8): T=4096 43.40 -> 37.81, T=2048
// 26.84 -> 24.56, T=512 15.72 -> 15.70, T=128 11.59 -> 11.85. ncu: -29% instructions but the
// same load-latency bound as the exact kernel, so the wall moves little.
//
// Structure mirrors cb3_moe_gemm.cu (whose header records why): one small smem stage, plane
// words held in registers per K-step pair, a row's two groups decoded by adjacent lanes,
// 128-row tiles and 32-row tiles for small experts. A K step is 64 k = two k32 MMAs.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int BN = 64;
constexpr int BK = 64;          // bytes of K per row per step (e4m3 / e2m1 container: 1 byte each)
constexpr int THREADS = 256;
constexpr float FP8_MAX = 448.0f;
constexpr float AMAX_FLOOR = 1e-4f;

struct Cb3Planes {
    const uint8_t* lo;
    const uint8_t* hi;
    const uint8_t* cb;
    const uint8_t* sc;
};

__device__ __forceinline__ Cb3Planes planes_for(const uint8_t* lo, const uint8_t* hi, const uint8_t* cb,
                                                const uint8_t* sc, unsigned long long lo_s,
                                                unsigned long long hi_s, unsigned long long cb_s,
                                                unsigned long long sc_s, int slot) {
    return {lo + slot * lo_s, hi + slot * hi_s, cb + slot * cb_s, sc + slot * sc_s};
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}

/// Four 3-bit codebook indices -> a prmt selector (as cb3_moe_gemm.cu).
__device__ __forceinline__ uint32_t cb3_selector(uint32_t L, uint32_t H, int shift, int bit) {
    const uint32_t a = (L >> shift) & 0x03030303u;
    const uint32_t b = ((H >> bit) << 2) & 0x04040404u;
    const uint32_t ie = a | b;
    const uint32_t t = ie | (ie >> 4);
    return (t & 0xFFu) | ((t >> 8) & 0xFF00u);
}

/// One row's plane bytes for a PAIR of K steps (k0 % 128 == 0), as cb3_moe_gemm.cu.
struct RawPair {
    uint4 lo0, lo1;
    uint4 hi;
    uint32_t sc4;
};

__device__ __forceinline__ RawPair fetch_pair(const Cb3Planes& p, long long n, int k0, int K) {
    const int block = k0 / 512;
    const int g = (k0 % 512) / 32;
    const uint8_t* lo = p.lo + n * (K / 4) + block * 128 + (g / 2) * 16;
    RawPair r;
    r.lo0 = *reinterpret_cast<const uint4*>(lo);
    r.lo1 = *reinterpret_cast<const uint4*>(lo + 16);
    r.hi = *reinterpret_cast<const uint4*>(p.hi + n * (K / 8) + block * 64 + (g / 4) * 16);
    r.sc4 = *reinterpret_cast<const uint32_t*>(p.sc + n * (K / 32) + k0 / 32);
    return r;
}

/// A row's codebook as e2m1 containers (code << 2, the value at bits [5:2]), 4 per word.
__device__ __forceinline__ uint2 container_table(uint2 cb) { return make_uint2(cb.x << 2, cb.y << 2); }

/// Decode one 32-weight group into 32 container bytes in K order (8 words).
__device__ __forceinline__ void decode32_q8(uint4 lo, uint4 hi, uint2 tab, int k0, uint32_t (&out)[8]) {
    const int g = (k0 % 512) / 32;
    const int lo_shift = 4 * (g % 2);
    const int hi_bit = ((g / 2) % 2) * 4 + (g % 2) * 2;
    const uint32_t lo_w[4] = {lo.x, lo.y, lo.z, lo.w};
    const uint32_t hi_w[4] = {hi.x, hi.y, hi.z, hi.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const uint32_t c0 = prmt(tab.x, tab.y, cb3_selector(lo_w[q], hi_w[q], lo_shift, hi_bit));          // r = 0
        const uint32_t c1 = prmt(tab.x, tab.y, cb3_selector(lo_w[q], hi_w[q], lo_shift + 2, hi_bit + 1));  // r = 1
        // lane j of this word holds k = 2 j (r = 0) and 2 j + 1 (r = 1).
        out[2 * q] = prmt(c0, c1, 0x5140);
        out[2 * q + 1] = prmt(c0, c1, 0x7362);
    }
}

/// Swizzled 16-byte chunk index: a row of a step is 64 bytes = 4 chunks; XOR by (row / 2) % 4
/// puts the 8 rows one ldmatrix phase reads on 8 different 16-byte bank groups.
__device__ __forceinline__ int swz(int row, int chunk) { return row * 4 + (chunk ^ ((row >> 1) & 3)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const uint4* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1, const uint4* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(a));
}

/// d += (A x 2^(sa-127)) (B x 2^(sb-127)), m16n8k32, e4m3 x e2m1, thread-id selector 0, byte 0.
__device__ __forceinline__ void qmma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                     uint32_t sb) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}

/// Smallest e with 2^e >= v (v > 0 normal), from the bit pattern (tools/fp8mma_moe._pow2_ceil_exp).
__device__ __forceinline__ int pow2_ceil_exp(float v) {
    const uint32_t b = __float_as_uint(v);
    return (int)((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) != 0u);
}

/// Two floats -> two e4m3 bytes (lo in byte 0), round to nearest, saturating at +-448.
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}

/// Tile shape per M-tile height (as cb3_moe_gemm.cu): 128 -> 4 x 2 warps of 32 x 32; 32 -> 1 x 8
/// warps of 32 x 8. A stage: [A TBM x 64 B | W1 64 x 64 B | W3 64 x 64 B | sA TBM x 2 | sW1 | sW3].
template <int TBM>
struct Shape {
    static constexpr int WARPS_N = 8 / (TBM / 32);
    static constexpr int WN = BN / WARPS_N;
    static constexpr int NB = WN / 8;
    static constexpr int ACT = (TBM * 4 + THREADS - 1) / THREADS;  // uint4 activation chunks per thread
    static constexpr int A_U4 = TBM * 4;
    static constexpr int W_U4 = BN * 4;
    // byte offsets
    static constexpr int GU_SA = (A_U4 + 2 * W_U4) * 16;
    static constexpr int GU_SW1 = GU_SA + TBM * 2;
    static constexpr int GU_SW3 = GU_SW1 + BN * 2;
    static constexpr int GU_BYTES = GU_SW3 + BN * 2;
    static constexpr int DN_SA = (A_U4 + W_U4) * 16;
    static constexpr int DN_SW = DN_SA + TBM * 2;
    static constexpr int DN_BYTES = DN_SW + BN * 2;
    static constexpr int EPI_BYTES = TBM * WARPS_N * 4;  // per-(row, warp column) amax
};

/// Source row of tile row `r`: through `tok` (permuted row -> token, so the MoE input is
/// quantized once per TOKEN and never permuted) or, with `tok == nullptr`, the row itself.
__device__ __forceinline__ long long src_row(const int* __restrict__ tok, int row_begin, int r) {
    return tok ? (long long)tok[row_begin + r] : (long long)(row_begin + r);
}

template <int TBM>
__device__ __forceinline__ void fetch_act(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ sx,
                                          const int* __restrict__ tok, int row_begin, int rows, int k0, int K,
                                          uint4 (&v)[Shape<TBM>::ACT], uint32_t& s) {
#pragma unroll
    for (int e = 0; e < Shape<TBM>::ACT; ++e) {
        const int c = threadIdx.x + e * THREADS;
        const int row = c / 4;
        v[e] = (c < TBM * 4 && row < rows)
                   ? __ldcg(reinterpret_cast<const uint4*>(xq + src_row(tok, row_begin, row) * K + k0 + (c % 4) * 16))
                   : make_uint4(0, 0, 0, 0);
    }
    // The two scale bytes of row threadIdx.x for this step (a padding row gets scale 2^0).
    s = (threadIdx.x < rows && threadIdx.x < TBM)
            ? *reinterpret_cast<const uint16_t*>(sx + src_row(tok, row_begin, threadIdx.x) * (K / 32) + k0 / 32)
            : 0x7F7Fu;
}

template <int TBM>
__device__ __forceinline__ void store_act(const uint4 (&v)[Shape<TBM>::ACT], uint32_t s, uint4* stage,
                                          uint8_t* s_sa) {
#pragma unroll
    for (int e = 0; e < Shape<TBM>::ACT; ++e) {
        const int c = threadIdx.x + e * THREADS;
        if (c < TBM * 4) stage[swz(c / 4, c % 4)] = v[e];
    }
    if (threadIdx.x < TBM) *reinterpret_cast<uint16_t*>(s_sa + threadIdx.x * 2) = (uint16_t)s;
}

/// Decode this thread's group `dgrp` of step `step` of the pair into its weight row.
__device__ __forceinline__ void decode_into(const RawPair& r, int step, int dgrp, uint2 tab, int ks, uint4* w,
                                            uint8_t* s_sw, int row) {
    uint32_t out[8];
    decode32_q8(step ? r.lo1 : r.lo0, r.hi, tab, ks + dgrp * 32, out);
    w[swz(row, 2 * dgrp)] = make_uint4(out[0], out[1], out[2], out[3]);
    w[swz(row, 2 * dgrp + 1)] = make_uint4(out[4], out[5], out[6], out[7]);
    s_sw[row * 2 + dgrp] = (uint8_t)(r.sc4 >> (8 * (2 * step + dgrp)));
}

/// One K step (two k32 MMAs) of a warp's 32 x WN strip against one weight matrix.
template <int NB>
__device__ __forceinline__ void mma_step(float (&acc)[2][NB][4], const uint4* s_a, const uint8_t* s_sa,
                                         const uint4* s_w, const uint8_t* s_sw, int wm, int wn) {
    const int lane = threadIdx.x % 32, gid = lane / 4, tig = lane % 4;
#pragma unroll
    for (int kg = 0; kg < 2; ++kg) {
        uint32_t a[2][4], sa[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            ldmatrix_x4(a[i], s_a + swz(wm + i * 16 + lane % 16, 2 * kg + lane / 16));
            sa[i] = s_sa[(wm + i * 16 + gid + 8 * (tig & 1)) * 2 + kg];
        }
        uint32_t b[NB][2];
        if constexpr (NB == 1) {
            ldmatrix_x2(b[0][0], b[0][1], s_w + swz(wn + lane % 8, 2 * kg + (lane / 8) % 2));
        } else {
#pragma unroll
            for (int j = 0; j < NB / 2; ++j) {
                uint32_t r[4];
                ldmatrix_x4(r, s_w + swz(wn + j * 16 + lane % 8 + (lane / 16) * 8, 2 * kg + (lane / 8) % 2));
                b[2 * j][0] = r[0];
                b[2 * j][1] = r[1];
                b[2 * j + 1][0] = r[2];
                b[2 * j + 1][1] = r[3];
            }
        }
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
            const uint32_t sb = s_sw[(wn + nb * 8 + gid) * 2 + kg];
#pragma unroll
            for (int i = 0; i < 2; ++i) qmma(acc[i][nb], a[i], b[nb][0], b[nb][1], sa[i], sb);
        }
    }
}

template <int TBM>
__device__ __forceinline__ void gate_up_q8_body(
    const uint8_t* __restrict__ xq, const uint8_t* __restrict__ sx, const int* __restrict__ tok,
    const int4* __restrict__ tiles, const uint8_t* w1_lo, const uint8_t* w1_hi, const uint8_t* w1_cb, const uint8_t* w1_sc,
    const uint8_t* w3_lo, const uint8_t* w3_hi, const uint8_t* w3_cb, const uint8_t* w3_sc,
    unsigned long long lo_s, unsigned long long hi_s, unsigned long long cb_s, unsigned long long sc_s,
    const float* __restrict__ row_weight, uint8_t* __restrict__ hq, uint8_t* __restrict__ hs, int N, int K,
    float limit) {
    using Sh = Shape<TBM>;
    constexpr int NB = Sh::NB;
    extern __shared__ __align__(128) uint4 smem[];
    uint8_t* bytes = reinterpret_cast<uint8_t*>(smem);
    uint4* s_a = smem;
    uint4* s_w1 = smem + Sh::A_U4;
    uint4* s_w3 = smem + Sh::A_U4 + Sh::W_U4;
    uint8_t* s_sa = bytes + Sh::GU_SA;
    uint8_t* s_sw1 = bytes + Sh::GU_SW1;
    uint8_t* s_sw3 = bytes + Sh::GU_SW3;

    const int4 tile = tiles[blockIdx.y];
    const int row_begin = tile.x, rows = tile.y, slot = tile.z;
    const int n0 = blockIdx.x * BN;
    const Cb3Planes p1 = planes_for(w1_lo, w1_hi, w1_cb, w1_sc, lo_s, hi_s, cb_s, sc_s, slot);
    const Cb3Planes p3 = planes_for(w3_lo, w3_hi, w3_cb, w3_sc, lo_s, hi_s, cb_s, sc_s, slot);

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = (warp / Sh::WARPS_N) * 32, wn = (warp % Sh::WARPS_N) * Sh::WN;
    const bool live = wm < rows;
    float acc_g[2][NB][4], acc_u[2][NB][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < NB; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc_g[i][j][e] = acc_u[i][j][e] = 0.0f;

    // Thread t decodes group t % 2 of row (t % 128) / 2 of w1 (t < 128) or w3 (t >= 128).
    const bool up = threadIdx.x >= 128;
    const Cb3Planes& mine = up ? p3 : p1;
    uint4* w_mine = up ? s_w3 : s_w1;
    uint8_t* sw_mine = up ? s_sw3 : s_sw1;
    const int drow = (threadIdx.x % 128) / 2;
    const int dgrp = threadIdx.x % 2;
    const long long dn = n0 + drow;
    const uint2 tab = container_table(*reinterpret_cast<const uint2*>(mine.cb + dn * 8));
    RawPair cur = fetch_pair(mine, dn, 0, K);
    uint4 a_reg[Sh::ACT];
    uint32_t a_sc;
    fetch_act<TBM>(xq, sx, tok, row_begin, rows, 0, K, a_reg, a_sc);

    for (int k0 = 0; k0 < K; k0 += 2 * BK) {
        RawPair nxt = cur;
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            const int ks = k0 + step * BK;
            store_act<TBM>(a_reg, a_sc, s_a, s_sa);
            decode_into(cur, step, dgrp, tab, ks, w_mine, sw_mine, drow);
            __syncthreads();
            if (step == 0 && k0 + 2 * BK < K) nxt = fetch_pair(mine, dn, k0 + 2 * BK, K);
            if (ks + BK < K) fetch_act<TBM>(xq, sx, tok, row_begin, rows, ks + BK, K, a_reg, a_sc);
            if (live) {
                mma_step<NB>(acc_g, s_a, s_sa, s_w1, s_sw1, wm, wn);
                mma_step<NB>(acc_u, s_a, s_sa, s_w3, s_sw3, wm, wn);
            }
            __syncthreads();
        }
        cur = nxt;
    }

    // Epilogue: h = ((g * sigmoid(g)) * u) * w in fp32 (tools/cb3_scaled._up_cb3_kernel), then
    // e4m3 with one UE8M0 scale per row per 32 columns. A group's 32 columns span WN-wide warp
    // strips, so each warp's per-row amax goes through shared memory.
    const int gid = lane / 4, tig = lane % 4;
    float* s_amax = reinterpret_cast<float*>(smem);  // stage is free after the last barrier
    float hv[2][NB][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = wm + i * 16 + gid + half * 8;
            const float w = (live && row < rows) ? row_weight[row_begin + row] : 0.0f;
            float m = 0.0f;
#pragma unroll
            for (int j = 0; j < NB; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float g = fminf(acc_g[i][j][2 * half + e], limit);
                    const float u = fminf(fmaxf(acc_u[i][j][2 * half + e], -limit), limit);
                    const float sig = 1.0f / (1.0f + expf(-g));
                    const float v = g * sig * u * w;
                    hv[i][j][2 * half + e] = v;
                    m = fmaxf(m, fabsf(v));
                }
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 1));
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 2));
            if (tig == 0) s_amax[row * Sh::WARPS_N + warp % Sh::WARPS_N] = m;
        }
    __syncthreads();
    if (!live) return;
    constexpr int PER_GROUP = 32 / Sh::WN;  // warp strips per 32-column group
    const int w0 = ((warp % Sh::WARPS_N) / PER_GROUP) * PER_GROUP;
    const int grp = (n0 + wn) / 32;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = wm + i * 16 + gid + half * 8;
            if (row >= rows) continue;
            float amax = 0.0f;
#pragma unroll
            for (int p = 0; p < PER_GROUP; ++p) amax = fmaxf(amax, s_amax[row * Sh::WARPS_N + w0 + p]);
            const int e = pow2_ceil_exp(fmaxf(amax, AMAX_FLOOR) / FP8_MAX);
            const float inv = __uint_as_float((uint32_t)(127 - e) << 23);  // 2^-e, exact
            const long long prow = row_begin + row;
#pragma unroll
            for (int j = 0; j < NB; ++j) {
                const int col = n0 + wn + j * 8 + tig * 2;
                const uint32_t q = e4m3x2(hv[i][j][2 * half] * inv, hv[i][j][2 * half + 1] * inv);
                *reinterpret_cast<uint16_t*>(hq + prow * N + col) = (uint16_t)q;
            }
            if (tig == 0 && (warp % Sh::WARPS_N) == w0) hs[prow * (N / 32) + grp] = (uint8_t)(e + 127);
        }
}

template <int TBM>
__device__ __forceinline__ void down_q8_body(
    const uint8_t* __restrict__ hq, const uint8_t* __restrict__ hs, const int4* __restrict__ tiles,
    const uint8_t* w2_lo, const uint8_t* w2_hi, const uint8_t* w2_cb, const uint8_t* w2_sc,
    unsigned long long lo_s, unsigned long long hi_s, unsigned long long cb_s, unsigned long long sc_s,
    float* __restrict__ out, int N, int K) {
    using Sh = Shape<TBM>;
    constexpr int NB = Sh::NB;
    extern __shared__ __align__(128) uint4 smem[];
    uint8_t* bytes = reinterpret_cast<uint8_t*>(smem);
    uint4* s_a = smem;
    uint4* s_w = smem + Sh::A_U4;
    uint8_t* s_sa = bytes + Sh::DN_SA;
    uint8_t* s_sw = bytes + Sh::DN_SW;

    const int4 tile = tiles[blockIdx.y];
    const int row_begin = tile.x, rows = tile.y, slot = tile.z;
    const int n0 = blockIdx.x * BN;
    const Cb3Planes p2 = planes_for(w2_lo, w2_hi, w2_cb, w2_sc, lo_s, hi_s, cb_s, sc_s, slot);

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = (warp / Sh::WARPS_N) * 32, wn = (warp % Sh::WARPS_N) * Sh::WN;
    const bool live = wm < rows;
    float acc[2][NB][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < NB; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    const bool decoder = threadIdx.x < 2 * BN;
    const int drow = (threadIdx.x % 128) / 2;
    const int dgrp = threadIdx.x % 2;
    const long long dn = n0 + drow;
    uint2 tab = make_uint2(0, 0);
    RawPair cur{};
    if (decoder) {
        tab = container_table(*reinterpret_cast<const uint2*>(p2.cb + dn * 8));
        cur = fetch_pair(p2, dn, 0, K);
    }
    uint4 a_reg[Sh::ACT];
    uint32_t a_sc;
    fetch_act<TBM>(hq, hs, nullptr, row_begin, rows, 0, K, a_reg, a_sc);

    for (int k0 = 0; k0 < K; k0 += 2 * BK) {
        RawPair nxt = cur;
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            const int ks = k0 + step * BK;
            store_act<TBM>(a_reg, a_sc, s_a, s_sa);
            if (decoder) decode_into(cur, step, dgrp, tab, ks, s_w, s_sw, drow);
            __syncthreads();
            if (decoder && step == 0 && k0 + 2 * BK < K) nxt = fetch_pair(p2, dn, k0 + 2 * BK, K);
            if (ks + BK < K) fetch_act<TBM>(hq, hs, nullptr, row_begin, rows, ks + BK, K, a_reg, a_sc);
            if (live) mma_step<NB>(acc, s_a, s_sa, s_w, s_sw, wm, wn);
            __syncthreads();
        }
        cur = nxt;
    }

    if (!live) return;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = wm + i * 16 + lane / 4 + half * 8;
            if (row >= rows) continue;
            float* dst = out + (long long)(row_begin + row) * N + n0 + wn + (lane % 4) * 2;
#pragma unroll
            for (int j = 0; j < NB; ++j)
                *reinterpret_cast<float2*>(dst + j * 8) = make_float2(acc[i][j][2 * half], acc[i][j][2 * half + 1]);
        }
}

}  // namespace

// Dynamic shared memory the launcher passes (moe.rs FUSED_Q8_*_SMEM): stage (+ epilogue amax).
static_assert(Shape<128>::GU_BYTES == 16896 && Shape<128>::DN_BYTES == 12672, "moe.rs sizes");
static_assert(Shape<32>::GU_BYTES == 10560 && Shape<32>::DN_BYTES == 6336, "moe.rs sizes");
static_assert(Shape<128>::EPI_BYTES <= Shape<128>::GU_BYTES && Shape<32>::EPI_BYTES <= Shape<32>::GU_BYTES, "epi");

/// bf16 [rows, K] -> e4m3 bytes [rows, K] + UE8M0 bytes [rows, K / 32]; one warp per 32-group.
extern "C" __global__ void __launch_bounds__(THREADS) dsv41_act_quant_e4m3(const __nv_bfloat16* __restrict__ x,
                                                                            uint8_t* __restrict__ xq,
                                                                            uint8_t* __restrict__ sx, int K) {
    const long long row = blockIdx.x;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int g = warp; g < K / 32; g += THREADS / 32) {
        const float v = __bfloat162float(x[row * K + g * 32 + lane]);
        float m = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
        const int e = pow2_ceil_exp(fmaxf(m, AMAX_FLOOR) / FP8_MAX);
        const float inv = __uint_as_float((uint32_t)(127 - e) << 23);
        xq[row * K + g * 32 + lane] = (uint8_t)(e4m3x2(v * inv, 0.0f) & 0xFFu);
        if (lane == 0) sx[row * (K / 32) + g] = (uint8_t)(e + 127);
    }
}

// ---- REFERENCE EMULATION (the reconstruct path with fp8 on): the same quantization applied
// in place as quantize-then-dequantize to bf16 (exact: an e4m3 value times a power of two fits
// bf16), so bf16 x bf16 cuBLASLt GEMMs compute the products the block-scaled MMA does. With
// <= 6-bit products and K <= 5120 the fp32 sums cannot round, so the fp8 kernels are gated
// against this byte for byte.

/// x [rows, K] bf16 -> qdq(x) in place, per row per 32.
extern "C" __global__ void __launch_bounds__(THREADS) dsv41_qdq_e4m3_bf16(__nv_bfloat16* __restrict__ x, int K) {
    const long long row = blockIdx.x;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int g = warp; g < K / 32; g += THREADS / 32) {
        const long long i = row * K + g * 32 + lane;
        const float v = __bfloat162float(x[i]);
        float m = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
        const int e = pow2_ceil_exp(fmaxf(m, AMAX_FLOOR) / FP8_MAX);
        const float inv = __uint_as_float((uint32_t)(127 - e) << 23);
        const uint32_t q = e4m3x2(v * inv, 0.0f) & 0xFFu;
        uint32_t h2;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((uint16_t)q));
        x[i] = __float2bfloat16(__half2float(__ushort_as_half((unsigned short)(h2 & 0xFFFFu))) *
                                __uint_as_float((uint32_t)(127 + e) << 23));
    }
}

/// h = qdq(((g * sigmoid(g)) * u) * w) as bf16, per row per 32 (the q8 gate/up epilogue's values).
extern "C" __global__ void __launch_bounds__(THREADS) dsv41_swiglu_qdq(const float* __restrict__ gate,
                                                                        const float* __restrict__ up,
                                                                        const float* __restrict__ row_weight,
                                                                        __nv_bfloat16* __restrict__ h, int N,
                                                                        float limit) {
    const long long row = blockIdx.x;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float w = row_weight[row];
    for (int g = warp; g < N / 32; g += THREADS / 32) {
        const long long i = row * N + g * 32 + lane;
        const float gv = fminf(gate[i], limit);
        const float uv = fminf(fmaxf(up[i], -limit), limit);
        const float sig = 1.0f / (1.0f + expf(-gv));
        const float v = gv * sig * uv * w;
        float m = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
        const int e = pow2_ceil_exp(fmaxf(m, AMAX_FLOOR) / FP8_MAX);
        const float inv = __uint_as_float((uint32_t)(127 - e) << 23);
        const uint32_t q = e4m3x2(v * inv, 0.0f) & 0xFFu;
        uint32_t h2;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((uint16_t)q));
        h[i] = __float2bfloat16(__half2float(__ushort_as_half((unsigned short)(h2 & 0xFFFFu))) *
                                __uint_as_float((uint32_t)(127 + e) << 23));
    }
}

#define ATLAS_CB3_Q8_GATE_UP_PARAMS                                                                       \
    const uint8_t* __restrict__ xq, const uint8_t* __restrict__ sx, const int* __restrict__ tok,         \
        const int4* __restrict__ tiles, const uint8_t* w1_lo, const uint8_t* w1_hi, const uint8_t* w1_cb, const uint8_t* w1_sc,           \
        const uint8_t* w3_lo, const uint8_t* w3_hi, const uint8_t* w3_cb, const uint8_t* w3_sc,           \
        unsigned long long lo_s, unsigned long long hi_s, unsigned long long cb_s, unsigned long long sc_s, \
        const float* __restrict__ row_weight, uint8_t* __restrict__ hq, uint8_t* __restrict__ hs, int N,  \
        int K, float limit
#define ATLAS_CB3_Q8_GATE_UP_ARGS xq, sx, tok, tiles, w1_lo, w1_hi, w1_cb, w1_sc, w3_lo, w3_hi, w3_cb, w3_sc, lo_s, \
    hi_s, cb_s, sc_s, row_weight, hq, hs, N, K, limit
#define ATLAS_CB3_Q8_DOWN_PARAMS                                                                          \
    const uint8_t* __restrict__ hq, const uint8_t* __restrict__ hs, const int4* __restrict__ tiles,       \
        const uint8_t* w2_lo, const uint8_t* w2_hi, const uint8_t* w2_cb, const uint8_t* w2_sc,           \
        unsigned long long lo_s, unsigned long long hi_s, unsigned long long cb_s, unsigned long long sc_s, \
        float* __restrict__ out, int N, int K
#define ATLAS_CB3_Q8_DOWN_ARGS hq, hs, tiles, w2_lo, w2_hi, w2_cb, w2_sc, lo_s, hi_s, cb_s, sc_s, out, N, K

extern "C" __global__ void __launch_bounds__(THREADS) cb3_moe_q8_gate_up(ATLAS_CB3_Q8_GATE_UP_PARAMS) {
    gate_up_q8_body<128>(ATLAS_CB3_Q8_GATE_UP_ARGS);
}
extern "C" __global__ void __launch_bounds__(THREADS) cb3_moe_q8_down(ATLAS_CB3_Q8_DOWN_PARAMS) {
    down_q8_body<128>(ATLAS_CB3_Q8_DOWN_ARGS);
}
extern "C" __global__ void __launch_bounds__(THREADS, 2) cb3_moe_q8_gate_up_m32(ATLAS_CB3_Q8_GATE_UP_PARAMS) {
    gate_up_q8_body<32>(ATLAS_CB3_Q8_GATE_UP_ARGS);
}
extern "C" __global__ void __launch_bounds__(THREADS, 2) cb3_moe_q8_down_m32(ATLAS_CB3_Q8_DOWN_PARAMS) {
    down_q8_body<32>(ATLAS_CB3_Q8_DOWN_ARGS);
}
