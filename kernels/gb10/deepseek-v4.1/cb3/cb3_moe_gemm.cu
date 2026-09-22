// SPDX-License-Identifier: AGPL-3.0-only
//
// Grouped CB3 expert GEMM with the weights decoded IN SHARED MEMORY, never in DRAM.
//
// Replaces reconstruct-to-bf16-scratch + cuBLASLt for the DeepSeek-V4.1 routed experts.
// Measured on one layer at T=2048 (dsv41_moe_bench): the scratch path wrote 70.8 MB of
// bf16 per expert and read it back — 129 of 147 ms per layer — against a floor of reading
// 1.79 GB of packed experts (~8 ms). Here each CTA decodes its [BN, BK] weight slice from
// the packed planes into shared memory and feeds bf16 tensor-core MMAs directly.
//
// NUMERICS are the reconstruct path's: a CB3 value is e2m1 x 2^(s-127), exact in bf16, so
// the decoded operand is bit-identical to what `cb3_reconstruct_bf16` wrote; only the fp32
// accumulation order differs. The gated kernel keeps gate and up in fp32 through the clamp,
// SwiGLU and route weight and rounds h ONCE, like the engine (tools/fp4_moe.py); the down
// kernel writes fp32 for `dsv41_unpermute_sum_f32`.
//
// K-ORDER (CB3_FORMAT.md, settled with a control): K in 512-blocks (a 256 tail for w2);
// for scale group g, lane 0..15, r 0..1:  k = block*512 + g*32 + lane*2 + r
//   lo byte = block*128 + (g/2)*16 + lane   shift 4*(g%2) + 2*r
//   hi byte = block*64  + (g/4)*16 + lane   bit   ((g/2)%2)*4 + (g%2)*2 + r
// BK = 32 is exactly one scale group, so one scale byte per row per K step.
//
// WORK LIST: one entry per (expert group, 128-row M tile): {row_begin, rows, slot, 0}.
// Grid: (N / BN, num_tiles) — N FASTEST. Consecutive CTAs are the N tiles of ONE expert
// tile, so its activation rows are read from DRAM once and then hit L2. With tiles fastest,
// every wave touched ~48 different experts' activations and re-read them per N tile:
// ~13 GB of activation traffic per layer at T=2048, more than the whole weight stream.
// MEASURED AND REJECTED (2026-09-22, interleaved A/B, 6 rounds, clean window):
//   - __launch_bounds__(256, 2) for 2 CTAs/SM: 128 regs + ~90 B spill, 14% SLOWER at T=2048.
//   - Warp specialisation (8 MMA warps + 4 producer warps decoding the next K step into a
//     second dynamic-smem stage, cp.async activations): bit-identical, but 38.2 vs 32.5 ms
//     at T=2048, 24.3 vs 18.8 at T=512, 17.0 vs 14.0 at T=128. Concentrating the decode on
//     4 warps made it the critical path: decode, not MMA, bounds this kernel (ncu: tensor
//     pipe 28.7%, IPC 1.12). The lever is cheaper or less redundant decode, not overlap.
//   - Deeper pipelining (3-stage cp.async activation ring + packed weight bytes fetched TWO
//     K steps ahead, 73.7 KB dynamic smem): bit-identical, but 35.3 vs 28.7 ms at T=2048,
//     21.7 vs 17.2 at T=512, 16.7 vs 13.0 at T=128 (6 interleaved rounds). Exposed load
//     latency is not what bounds a step either.
//   - Shaped tiles for few-row experts (32x256 / 64x128, all 8 warps in the MMA, every
//     thread decoding): chosen because ncu stall reasons at T=512 put BARRIER first (3.29
//     cycles/instr gate_up, 5.82 down). Bit-identical and chunk-invariant, but SLOWER at every
//     T: 19.55 vs 17.13 ms at T=512, 31.08 vs 28.67 at 2048, 9.51 vs 8.21 at 32 (6
//     interleaved rounds; 254 registers with 4 decode jobs per thread).
// Block: 256 threads (8 warps, each a 32x32 quadrant of 128x64).

#include <cstdint>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

namespace {

constexpr int BM = 128;
constexpr int BN = 64;
constexpr int BK = 64;   // two scale groups per K step
constexpr int THREADS = 256;
// Shared row stride in elements. BK + 8 so the 64 decoders (one row each) and the
// activation writes do not all land in one bank group: at a 128-byte stride every row
// starts on bank 0 and a warp's 16-byte stores serialise 32 ways.
constexpr int LDS = BK + 8;

/// e2m1 code -> float, by arithmetic (a __constant__ table indexed by divergent codes
/// serialises across the warp). Magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}; bit 3 is the sign.
__device__ __forceinline__ float fp4_value(uint32_t code) {
    const uint32_t m = code & 7u;
    uint32_t bits = m >= 2u ? (((126u + (m >> 1)) << 23) | ((m & 1u) << 22)) : (m ? (126u << 23) : 0u);
    bits |= (code & 8u) << 28;
    return __uint_as_float(bits);
}

/// One CB3 matrix's planes for expert `slot`: base + slot * per-slot stride.
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

__device__ __forceinline__ uint32_t bf16_bits(float v) {
    const __nv_bfloat16 b = __float2bfloat16(v);
    return (uint32_t)(*reinterpret_cast<const unsigned short*>(&b));
}

/// The packed bytes one thread needs for one (row, 32-weight group): fetched one K step
/// AHEAD so the global loads are in flight during the MMAs of the current step.
struct Raw {
    uint4 lo;
    uint4 hi;
    uint32_t sc;
};

__device__ __forceinline__ Raw fetch_group(const Cb3Planes& p, long long n, int k0, int K) {
    const int block = k0 / 512;
    const int g = (k0 % 512) / 32;
    Raw r;
    r.lo = *reinterpret_cast<const uint4*>(p.lo + n * (K / 4) + block * 128 + (g / 2) * 16);
    r.hi = *reinterpret_cast<const uint4*>(p.hi + n * (K / 8) + block * 64 + (g / 4) * 16);
    r.sc = p.sc[n * (K / 32) + k0 / 32];
    return r;
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}

/// Four 3-bit codebook indices -> a prmt selector (one index per nibble, bit 3 clear).
/// Byte j of L/H belongs to lane j of this word; `shift`/`bit` pick sub-position r.
__device__ __forceinline__ uint32_t cb3_selector(uint32_t L, uint32_t H, int shift, int bit) {
    const uint32_t a = (L >> shift) & 0x03030303u;
    const uint32_t b = (bit >= 2 ? (H >> (bit - 2)) : (H << (2 - bit))) & 0x04040404u;
    const uint32_t ie = a | b;             // one index (0..7) per byte
    const uint32_t t = ie | (ie >> 4);     // bytes 0 and 2 now hold two indices each
    return (t & 0xFFu) | ((t >> 8) & 0xFF00u);
}

/// Decode one row's 32-weight scale group into 16 packed bf16 pairs (K order).
///
/// The row codebook times the group scale is an 8-entry bf16 table, split into a low-byte
/// table and a high-byte table (two u32 each). Each weight index then selects its two bytes
/// with `prmt` — four weights per selector, no per-weight arithmetic, no dynamic register
/// indexing. Values are exactly `cb3_reconstruct_bf16`'s (fp4 x 2^(s-127), exact in bf16);
/// this replaced a per-weight predicated select that cost ~3x the instructions.
__device__ __forceinline__ void decode32(const Raw& r, uint2 cb, int k0, uint32_t (&out)[16]) {
    const int g = (k0 % 512) / 32;
    const float scale = exp2f((float)r.sc - 127.0f);
    uint32_t e[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t word = (i < 4) ? cb.x : cb.y;
        e[i] = bf16_bits(fp4_value((word >> ((i % 4) * 8)) & 0x0Fu) * scale);
    }
    const uint32_t tlo_x = (e[0] & 0xFFu) | ((e[1] & 0xFFu) << 8) | ((e[2] & 0xFFu) << 16) | ((e[3] & 0xFFu) << 24);
    const uint32_t tlo_y = (e[4] & 0xFFu) | ((e[5] & 0xFFu) << 8) | ((e[6] & 0xFFu) << 16) | ((e[7] & 0xFFu) << 24);
    const uint32_t thi_x = (e[0] >> 8) | ((e[1] >> 8) << 8) | ((e[2] >> 8) << 16) | ((e[3] >> 8) << 24);
    const uint32_t thi_y = (e[4] >> 8) | ((e[5] >> 8) << 8) | ((e[6] >> 8) << 16) | ((e[7] >> 8) << 24);
    const int lo_shift = 4 * (g % 2);
    const int hi_bit = ((g / 2) % 2) * 4 + (g % 2) * 2;
    const uint32_t lo_w[4] = {r.lo.x, r.lo.y, r.lo.z, r.lo.w};
    const uint32_t hi_w[4] = {r.hi.x, r.hi.y, r.hi.z, r.hi.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const uint32_t s0 = cb3_selector(lo_w[q], hi_w[q], lo_shift, hi_bit);          // r = 0
        const uint32_t s1 = cb3_selector(lo_w[q], hi_w[q], lo_shift + 2, hi_bit + 1);  // r = 1
        const uint32_t lo0 = prmt(tlo_x, tlo_y, s0), hi0 = prmt(thi_x, thi_y, s0);
        const uint32_t lo1 = prmt(tlo_x, tlo_y, s1), hi1 = prmt(thi_x, thi_y, s1);
        // Interleave to bf16(lane j, r=0) | bf16(lane j, r=1) << 16 for j = 0..3.
        const uint32_t x01 = prmt(lo0, lo1, 0x5140), x23 = prmt(lo0, lo1, 0x7362);
        const uint32_t y01 = prmt(hi0, hi1, 0x5140), y23 = prmt(hi0, hi1, 0x7362);
        out[4 * q + 0] = prmt(x01, y01, 0x5140);
        out[4 * q + 1] = prmt(x01, y01, 0x7362);
        out[4 * q + 2] = prmt(x23, y23, 0x5140);
        out[4 * q + 3] = prmt(x23, y23, 0x7362);
    }
}

/// Decode one row's one 32-weight scale group into dst[0..32) (bf16, K order).
__device__ __forceinline__ void decode_group(const Raw& r, uint2 cb, int k0, __nv_bfloat16* dst) {
    uint32_t out[16];
    decode32(r, cb, k0, out);
    uint4* d = reinterpret_cast<uint4*>(dst);
#pragma unroll
    for (int q = 0; q < 4; ++q) d[q] = make_uint4(out[4 * q], out[4 * q + 1], out[4 * q + 2], out[4 * q + 3]);
}

/// This thread's activation chunks for one K step: BM x BK / 8 = 1024 uint4 over 256 threads.
constexpr int ACT_CHUNKS = BM * (BK / 8) / THREADS;

__device__ __forceinline__ void fetch_act(const __nv_bfloat16* __restrict__ act, int row_begin, int rows,
                                          int k0, int K, uint4 (&v)[ACT_CHUNKS]) {
#pragma unroll
    for (int e = 0; e < ACT_CHUNKS; ++e) {
        const int c = threadIdx.x + e * THREADS;
        const int row = c / (BK / 8);
        const int col = (c % (BK / 8)) * 8;
        v[e] = row < rows
                   ? *reinterpret_cast<const uint4*>(act + (long long)(row_begin + row) * K + k0 + col)
                   : make_uint4(0, 0, 0, 0);
    }
}

__device__ __forceinline__ void store_act(const uint4 (&v)[ACT_CHUNKS], __nv_bfloat16 (*dst)[LDS]) {
#pragma unroll
    for (int e = 0; e < ACT_CHUNKS; ++e) {
        const int c = threadIdx.x + e * THREADS;
        *reinterpret_cast<uint4*>(&dst[c / (BK / 8)][(c % (BK / 8)) * 8]) = v[e];
    }
}

// Shared layout: mainloop [s_a BM x LDS | s_w1 BN x LDS | s_w3 BN x LDS] = 36 KB, reused as
// the fp32 epilogue tile [BM][BN] = 32 KB after the loop.
constexpr int SMEM_BYTES = BM * LDS * 2 + 2 * BN * LDS * 2;
static_assert(SMEM_BYTES >= BM * BN * 4, "epilogue tile must fit in the mainloop buffer");

}  // namespace

/// Gate + up + SwiGLU for one (expert tile, N tile):
///   h[row, n] = bf16( silu(min(g, L)) * clamp(u, -L, L) * route_w[row] ),  g/u fp32.
namespace {
/// The gate/up body, shared by the 1- and 2-CTA-per-SM entry points below.
__device__ __forceinline__ void gate_up_body(
    const __nv_bfloat16* __restrict__ act,  // [P, K] permuted rows
    const int4* __restrict__ tiles,         // {row_begin, rows, slot, 0}
    const uint8_t* w1_lo, const uint8_t* w1_hi, const uint8_t* w1_cb, const uint8_t* w1_sc,
    const uint8_t* w3_lo, const uint8_t* w3_hi, const uint8_t* w3_cb, const uint8_t* w3_sc,
    unsigned long long lo_s, unsigned long long hi_s, unsigned long long cb_s, unsigned long long sc_s,
    const float* __restrict__ row_weight,   // [P]
    __nv_bfloat16* __restrict__ h,          // [P, N]
    int N, int K, float limit) {
    __shared__ __align__(128) unsigned char smem[SMEM_BYTES];
    auto s_a = reinterpret_cast<__nv_bfloat16 (*)[LDS]>(smem);
    auto s_w1 = reinterpret_cast<__nv_bfloat16 (*)[LDS]>(smem + BM * LDS * 2);
    auto s_w3 = reinterpret_cast<__nv_bfloat16 (*)[LDS]>(smem + BM * LDS * 2 + BN * LDS * 2);
    auto s_out = reinterpret_cast<float (*)[BN]>(smem);

    const int4 tile = tiles[blockIdx.y];
    const int row_begin = tile.x, rows = tile.y, slot = tile.z;
    const int n0 = blockIdx.x * BN;
    const Cb3Planes p1 = planes_for(w1_lo, w1_hi, w1_cb, w1_sc, lo_s, hi_s, cb_s, sc_s, slot);
    const Cb3Planes p3 = planes_for(w3_lo, w3_hi, w3_cb, w3_sc, lo_s, hi_s, cb_s, sc_s, slot);

    // 8 warps: 4 along M x 2 along N, each a 32x32 quadrant.
    const int warp = threadIdx.x / 32;
    const int wm = (warp / 2) * 32, wn = (warp % 2) * 32;
    // Rows beyond `rows` are zero; a warp whose whole quadrant is padding skips the MMAs.
    const bool live = wm < rows;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_g[2][2], acc_u[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(acc_g[i][j], 0.0f);
            wmma::fill_fragment(acc_u[i][j], 0.0f);
        }

    // Thread t decodes (row t % 64, group (t / 64) % 2) of w1 (t < 128) or w3 (t >= 128).
    const Cb3Planes& mine = threadIdx.x < 128 ? p1 : p3;
    __nv_bfloat16 (*s_mine)[LDS] = threadIdx.x < 128 ? s_w1 : s_w3;
    const int drow = threadIdx.x % BN;
    const int dgrp = (threadIdx.x / BN) % 2;
    const long long dn = n0 + drow;
    const uint2 cb = *reinterpret_cast<const uint2*>(mine.cb + dn * 8);
    Raw raw = fetch_group(mine, dn, dgrp * 32, K);
    uint4 a_reg[ACT_CHUNKS];
    fetch_act(act, row_begin, rows, 0, K, a_reg);

    for (int k0 = 0; k0 < K; k0 += BK) {
        store_act(a_reg, s_a);
        decode_group(raw, cb, k0 + dgrp * 32, &s_mine[drow][dgrp * 32]);
        __syncthreads();
        if (k0 + BK < K) {
            // Next step's loads go out now and land during this step's MMAs.
            raw = fetch_group(mine, dn, k0 + BK + dgrp * 32, K);
            fetch_act(act, row_begin, rows, k0 + BK, K, a_reg);
        }
        if (live) {
#pragma unroll
            for (int kk = 0; kk < BK; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b1[2], b3[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &s_a[wm + i * 16][kk], LDS);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    // W tile is [n][k] row-major == B[k][n] col-major.
                    wmma::load_matrix_sync(b1[j], &s_w1[wn + j * 16][kk], LDS);
                    wmma::load_matrix_sync(b3[j], &s_w3[wn + j * 16][kk], LDS);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        wmma::mma_sync(acc_g[i][j], a[i], b1[j], acc_g[i][j]);
                        wmma::mma_sync(acc_u[i][j], a[i], b3[j], acc_u[i][j]);
                    }
            }
        }
        __syncthreads();
    }

    // Epilogue. gate and up fragments share one (unspecified but identical) element layout,
    // so the clamp and SwiGLU run element-wise in registers; only silu(g)*u goes through the
    // fp32 staging tile, where the row is known for the route weight. The association
    // ((g * sig) * u) * w is the reference's, and h is rounded to bf16 exactly once.
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int e = 0; e < acc_g[i][j].num_elements; ++e) {
                const float g = fminf(acc_g[i][j].x[e], limit);
                const float u = fminf(fmaxf(acc_u[i][j].x[e], -limit), limit);
                const float sig = 1.0f / (1.0f + expf(-g));
                acc_g[i][j].x[e] = g * sig * u;
            }
            wmma::store_matrix_sync(&s_out[wm + i * 16][wn + j * 16], acc_g[i][j], BN, wmma::mem_row_major);
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < BM * BN; idx += THREADS) {
        const int row = idx / BN, col = idx % BN;
        if (row >= rows) continue;
        const long long prow = row_begin + row;
        h[prow * N + n0 + col] = __float2bfloat16(s_out[row][col] * row_weight[prow]);
    }
}

}  // namespace

#define ATLAS_CB3_GATE_UP_ARGS act, tiles, w1_lo, w1_hi, w1_cb, w1_sc, w3_lo, w3_hi, w3_cb, w3_sc, \
    lo_s, hi_s, cb_s, sc_s, row_weight, h, N, K, limit

/// Gate + up + SwiGLU, one CTA per SM (168 registers, no spills).
///
/// Measured, interleaved A/B (6 rounds, clean window): the same body at
/// __launch_bounds__(256, 2) — 128 registers, 2 CTAs/SM, a ~90-byte spill — was 14% SLOWER
/// (35.9 vs 31.5 ms experts at T=2048) despite ncu showing this variant at 16.7% occupancy.
extern "C" __global__ void __launch_bounds__(THREADS) cb3_moe_gate_up(
    const __nv_bfloat16* __restrict__ act,  // [P, K] permuted rows
    const int4* __restrict__ tiles,         // {row_begin, rows, slot, 0}
    const uint8_t* w1_lo, const uint8_t* w1_hi, const uint8_t* w1_cb, const uint8_t* w1_sc,
    const uint8_t* w3_lo, const uint8_t* w3_hi, const uint8_t* w3_cb, const uint8_t* w3_sc,
    unsigned long long lo_s, unsigned long long hi_s, unsigned long long cb_s, unsigned long long sc_s,
    const float* __restrict__ row_weight,   // [P]
    __nv_bfloat16* __restrict__ h,          // [P, N]
    int N, int K, float limit) {
    gate_up_body(ATLAS_CB3_GATE_UP_ARGS);
}

/// Down projection for one (expert tile, N tile): out[row, n] = h[row, :] . w2[n, :], fp32.
extern "C" __global__ void __launch_bounds__(THREADS) cb3_moe_down(
    const __nv_bfloat16* __restrict__ h,    // [P, K]
    const int4* __restrict__ tiles,
    const uint8_t* w2_lo, const uint8_t* w2_hi, const uint8_t* w2_cb, const uint8_t* w2_sc,
    unsigned long long lo_s, unsigned long long hi_s, unsigned long long cb_s, unsigned long long sc_s,
    float* __restrict__ out,                // [P, N]
    int N, int K) {
    __shared__ __align__(128) unsigned char smem[SMEM_BYTES];
    auto s_a = reinterpret_cast<__nv_bfloat16 (*)[LDS]>(smem);
    auto s_w = reinterpret_cast<__nv_bfloat16 (*)[LDS]>(smem + BM * LDS * 2);
    auto s_out = reinterpret_cast<float (*)[BN]>(smem);

    const int4 tile = tiles[blockIdx.y];
    const int row_begin = tile.x, rows = tile.y, slot = tile.z;
    const int n0 = blockIdx.x * BN;
    const Cb3Planes p2 = planes_for(w2_lo, w2_hi, w2_cb, w2_sc, lo_s, hi_s, cb_s, sc_s, slot);

    const int warp = threadIdx.x / 32;
    const int wm = (warp / 2) * 32, wn = (warp % 2) * 32;
    const bool live = wm < rows;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    // Threads 0..127 decode (row t % 64, group t / 64); 128..255 only move activations.
    const bool decoder = threadIdx.x < BN * (BK / 32);
    const int drow = threadIdx.x % BN;
    const int dgrp = (threadIdx.x / BN) % 2;
    const long long dn = n0 + drow;
    uint2 cb = make_uint2(0, 0);
    Raw raw{};
    if (decoder) {
        cb = *reinterpret_cast<const uint2*>(p2.cb + dn * 8);
        raw = fetch_group(p2, dn, dgrp * 32, K);
    }
    uint4 a_reg[ACT_CHUNKS];
    fetch_act(h, row_begin, rows, 0, K, a_reg);

    for (int k0 = 0; k0 < K; k0 += BK) {
        store_act(a_reg, s_a);
        if (decoder) decode_group(raw, cb, k0 + dgrp * 32, &s_w[drow][dgrp * 32]);
        __syncthreads();
        if (k0 + BK < K) {
            if (decoder) raw = fetch_group(p2, dn, k0 + BK + dgrp * 32, K);
            fetch_act(h, row_begin, rows, k0 + BK, K, a_reg);
        }
        if (live) {
#pragma unroll
            for (int kk = 0; kk < BK; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &s_a[wm + i * 16][kk], LDS);
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &s_w[wn + j * 16][kk], LDS);
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&s_out[wm + i * 16][wn + j * 16], acc[i][j], BN, wmma::mem_row_major);
    __syncthreads();
    for (int idx = threadIdx.x; idx < BM * BN; idx += THREADS) {
        const int row = idx / BN, col = idx % BN;
        if (row < rows) out[(long long)(row_begin + row) * N + n0 + col] = s_out[row][col];
    }
}
