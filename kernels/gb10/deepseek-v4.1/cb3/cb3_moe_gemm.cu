// SPDX-License-Identifier: AGPL-3.0-only
//
// Grouped CB3 expert GEMM with the weights decoded IN SHARED MEMORY, never in DRAM.
//
// Replaces reconstruct-to-bf16-scratch + cuBLASLt for the DeepSeek-V4.1 routed experts.
// Measured on one layer at T=2048 (dsv41_moe_bench): the scratch path wrote 70.8 MB of
// bf16 per expert and read it back — 129 of 147 ms per layer — against a floor of reading
// 1.79 GB of packed experts (~8 ms). Here each CTA decodes its [BN, BK] weight slice from
// the packed planes into shared memory and feeds bf16 tensor-core MMAs (mma.sync) directly.
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
//   - Register-fragment kernels (CB3 decoded straight into mma.sync B fragments, no B smem
//     tile, no main-loop barrier, scale as a __vadd2 exponent add on a per-column table,
//     A fragments from global): bit-identical and chunk-invariant, but 2-3x SLOWER — 85.7 vs
//     28.8 ms at T=2048, 34.2 vs 17.2 at T=512 (6 interleaved rounds; 244 registers, each
//     thread's 16-byte plane loads scattered over 8 rows with no prefetch). Removing the
//     barrier did not pay for losing the shared, coalesced decode.
// KEPT (2026-09-22): the group scale as a __vadd2 exponent add on a per-row table built once,
// and the group's selector shifts as template constants. Per-group decode SASS fell from ~290
// to ~115 instructions, and the output is byte-identical on runA/runF L0,L2/runE_image/runC_2048.
// Wall time barely moved, though (6 interleaved rounds): T=128 13.38 vs 13.55 ms (-1.3%),
// T=512 18.01 vs 18.12 (-0.6%), T=2048 31.03 vs 31.00, T=4096 51.80 vs 51.72 (noise). Together
// with the warp-specialisation result this says decode instructions are no longer what bounds
// a step; the barrier stall is.
// L1 IS THE LEVER (2026-09-23). The decoders re-read their packed-plane lines across K steps,
// and those reads hit in L1 only while L1 is large enough. Any shared memory the kernel
// reserves shrinks L1: two stages (73.7 KB, and again at 64 KB in the swizzled layout) measured
// 10-17% and ~1% SLOWER, not faster. What won, all byte-identical (6 interleaved rounds, ms/layer):
//   - Unpadded XOR-swizzled 32 KB stage + ldmatrix + mma.sync m16n8k16 (the HMMA wmma issued)
//     + an epilogue straight from registers: T=2048 30.78 -> 28.81, T=128 13.31 -> 12.96.
//   - A row's two groups decoded by ADJACENT lanes, so their shared lo/hi words are one request
//     per warp, not one per warp pair relying on L1; activations loaded L2-only (__ldcg):
//     T=2048 28.84 -> 27.23, T=128 12.93 -> 12.45, T=4096 47.03 -> 44.09.
//   - Also measured and rejected: spreading the MMAs over live 16-row fragments so small tiles
//     use more warps (barrier stall 4.26 -> 2.36 but +20% instructions, 9-14% slower).
//   - Each row's plane words for a PAIR of K steps (lo 32 B + hi 16 B + 4 scale bytes) loaded
//     once into registers, prefetched two steps ahead: T=2048 27.20 -> 26.84, T=512
//     16.57 -> 16.05, T=128 12.44 -> 12.02, T=4096 44.23 -> 43.45 (174 registers, 1 CTA/SM).
// Block: 256 threads (8 warps, each a 32x32 quadrant of 128x64).

#include <cstdint>
#include <cuda_bf16.h>

namespace {

constexpr int BM = 128;
constexpr int BN = 64;
constexpr int BK = 64;   // two scale groups per K step
constexpr int THREADS = 256;
#ifndef CB3_ACT_LDCG
#define CB3_ACT_LDCG 1
#endif

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

/// The packed bytes one thread decodes for one (row, 32-weight group).
struct Raw {
    uint4 lo;
    uint4 hi;
    uint32_t sc;
};

/// One row's plane bytes for a PAIR of K steps (k0 % 128 == 0): the 32 lo bytes, the 16 hi
/// bytes and the 4 scale bytes that both steps' two groups use, each loaded exactly once and
/// kept in registers, so no plane byte depends on L1 surviving from one K step to the next.
struct RawPair {
    uint4 lo0, lo1;
    uint4 hi;
    uint32_t sc4;
};

__device__ __forceinline__ RawPair fetch_pair(const Cb3Planes& p, long long n, int k0, int K) {
    const int block = k0 / 512;
    const int g = (k0 % 512) / 32;  // a multiple of 4
    const uint8_t* lo = p.lo + n * (K / 4) + block * 128 + (g / 2) * 16;
    RawPair r;
    r.lo0 = *reinterpret_cast<const uint4*>(lo);
    r.lo1 = *reinterpret_cast<const uint4*>(lo + 16);
    r.hi = *reinterpret_cast<const uint4*>(p.hi + n * (K / 8) + block * 64 + (g / 4) * 16);
    r.sc4 = *reinterpret_cast<const uint32_t*>(p.sc + n * (K / 32) + k0 / 32);
    return r;
}

/// Group `dgrp` of step `step` (0/1) of a pair.
__device__ __forceinline__ Raw pick(const RawPair& r, int step, int dgrp) {
    Raw x;
    x.lo = step ? r.lo1 : r.lo0;
    x.hi = r.hi;
    x.sc = (r.sc4 >> (8 * (2 * step + dgrp))) & 0xFFu;
    return x;
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
    const uint32_t b = ((H >> bit) << 2) & 0x04040404u;
    const uint32_t ie = a | b;             // one index (0..7) per byte
    const uint32_t t = ie | (ie >> 4);     // bytes 0 and 2 now hold two indices each
    return (t & 0xFFu) | ((t >> 8) & 0xFF00u);
}

/// A row's 8 codebook entries as UNSCALED bf16 (fp4 values, exact), two per u32, plus a
/// mask of the non-zero halves. Built once per thread: the codebook is per row, and only
/// the group scale changes along K.
struct RowTable {
    uint32_t pair[4];
    uint32_t nz[4];
};

__device__ __forceinline__ RowTable row_table(uint2 cb) {
    RowTable t;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t word = (j < 2) ? cb.x : cb.y;
        const uint32_t e0 = bf16_bits(fp4_value((word >> ((2 * j % 4) * 8)) & 0x0Fu));
        const uint32_t e1 = bf16_bits(fp4_value((word >> (((2 * j + 1) % 4) * 8)) & 0x0Fu));
        t.pair[j] = e0 | (e1 << 16);
        t.nz[j] = ((e0 & 0x7FFFu) ? 0x0000FFFFu : 0u) | ((e1 & 0x7FFFu) ? 0xFFFF0000u : 0u);
    }
    return t;
}

/// The row table times 2^(s-127), split into low-byte and high-byte tables (two u32 each).
///
/// fp4 magnitudes have bf16 exponents 126..129, so for 2 <= s <= 252 every non-zero entry
/// stays normal and the scale is an exponent add: one __vadd2 per pair of entries. Zeros
/// (+0 and the -0 code) are masked out of the add. Outside that range (never seen in the
/// shipped weights) it falls back to the float multiply, which is what the fast path
/// reproduces bit for bit inside it.
__device__ __forceinline__ void scaled_table(const RowTable& t, uint32_t s, uint32_t (&tl)[2], uint32_t (&th)[2]) {
    uint32_t p[4];
    if (s - 2u <= 250u) {
        const uint32_t d = ((s - 127u) << 7) & 0xFFFFu;
        const uint32_t dd = d | (d << 16);
#pragma unroll
        for (int j = 0; j < 4; ++j) p[j] = __vadd2(t.pair[j], dd & t.nz[j]);
    } else {
        const float scale = exp2f((float)s - 127.0f);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t e0 = bf16_bits(__uint_as_float(t.pair[j] << 16) * scale);
            const uint32_t e1 = bf16_bits(__uint_as_float(t.pair[j] & 0xFFFF0000u) * scale);
            p[j] = e0 | (e1 << 16);
        }
    }
    tl[0] = prmt(p[0], p[1], 0x6420);
    th[0] = prmt(p[0], p[1], 0x7531);
    tl[1] = prmt(p[2], p[3], 0x6420);
    th[1] = prmt(p[2], p[3], 0x7531);
}

/// Decode one row's 32-weight scale group into 16 packed bf16 pairs (K order).
///
/// Each weight index selects its two bytes from the scaled table with `prmt` — four
/// weights per selector, no per-weight arithmetic, no dynamic register indexing. Values
/// are exactly `cb3_reconstruct_bf16`'s (fp4 x 2^(s-127), exact in bf16). The group's
/// sub-positions (LO_SHIFT = 4 * (g % 2), HI_BIT = ((g / 2) % 2) * 4 + (g % 2) * 2) are
/// template constants so the selector shifts fold.
__device__ __forceinline__ void decode32(const Raw& r, const RowTable& t, int lo_shift, int hi_bit,
                                         uint32_t (&out)[16]) {
    uint32_t tl[2], th[2];
    scaled_table(t, r.sc, tl, th);
    const uint32_t lo_w[4] = {r.lo.x, r.lo.y, r.lo.z, r.lo.w};
    const uint32_t hi_w[4] = {r.hi.x, r.hi.y, r.hi.z, r.hi.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const uint32_t s0 = cb3_selector(lo_w[q], hi_w[q], lo_shift, hi_bit);          // r = 0
        const uint32_t s1 = cb3_selector(lo_w[q], hi_w[q], lo_shift + 2, hi_bit + 1);  // r = 1
        const uint32_t lo0 = prmt(tl[0], tl[1], s0), hi0 = prmt(th[0], th[1], s0);
        const uint32_t lo1 = prmt(tl[0], tl[1], s1), hi1 = prmt(th[0], th[1], s1);
        // Interleave to bf16(lane j, r=0) | bf16(lane j, r=1) << 16 for j = 0..3.
        const uint32_t x01 = prmt(lo0, lo1, 0x5140), x23 = prmt(lo0, lo1, 0x7362);
        const uint32_t y01 = prmt(hi0, hi1, 0x5140), y23 = prmt(hi0, hi1, 0x7362);
        out[4 * q + 0] = prmt(x01, y01, 0x5140);
        out[4 * q + 1] = prmt(x01, y01, 0x7362);
        out[4 * q + 2] = prmt(x23, y23, 0x5140);
        out[4 * q + 3] = prmt(x23, y23, 0x7362);
    }
}

/// This thread's activation chunks for one K step: BM x BK / 8 = 1024 uint4 over 256 threads.
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
#if CB3_ACT_LDCG
                   // L2 only: a CTA reads each activation chunk once; keep L1 for the planes.
                   ? __ldcg(reinterpret_cast<const uint4*>(act + (long long)(row_begin + row) * K + k0 + col))
#else
                   ? *reinterpret_cast<const uint4*>(act + (long long)(row_begin + row) * K + k0 + col)
#endif
                   : make_uint4(0, 0, 0, 0);
    }
}

// SHARED LAYOUT. A row of a K step is BK = 64 bf16 = 128 bytes = 8 16-byte chunks, stored
// UNPADDED with the chunk index XOR-swizzled by row % 8. ldmatrix reads 8 rows of one chunk
// column per phase, and the decoders write 8 consecutive rows of one chunk column per
// quarter-warp: both land on 8 different 16-byte bank groups. No padding keeps a stage at
// 32 KB (gate/up) / 24 KB (down) instead of 36 KB.
__device__ __forceinline__ int swz(int row, int chunk) { return row * (BK / 8) + (chunk ^ (row & 7)); }

__device__ __forceinline__ void store_act(const uint4 (&v)[ACT_CHUNKS], uint4* dst) {
#pragma unroll
    for (int e = 0; e < ACT_CHUNKS; ++e) {
        const int c = threadIdx.x + e * THREADS;
        dst[swz(c / (BK / 8), c % (BK / 8))] = v[e];
    }
}

/// Decode one row's one 32-weight scale group into chunks [4 * half, 4 * half + 4) of `row`.
__device__ __forceinline__ void decode_group_swz(const Raw& r, const RowTable& t, int k0, uint4* dst, int row, int half) {
    uint32_t out[16];
    // The two groups of a K step sit in adjacent lanes, so the sub-position is a runtime
    // shift, not a (divergent) switch over template instances.
    const int g = (k0 % 512) / 32;
    decode32(r, t, 4 * (g % 2), ((g / 2) % 2) * 4 + (g % 2) * 2, out);
#pragma unroll
    for (int q = 0; q < 4; ++q)
        dst[swz(row, 4 * half + q)] = make_uint4(out[4 * q], out[4 * q + 1], out[4 * q + 2], out[4 * q + 3]);
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const uint4* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

/// d += a * b, m16n8k16, bf16 in, fp32 accumulate: the instruction wmma's 16x16x16 bf16
/// fragment issued twice (once per 8-column half), so per-element results are unchanged.
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/// A fragments (two 16-row halves of this warp's 32 rows) for K sub-step kk of one stage.
__device__ __forceinline__ void load_a(uint32_t (&a)[2][4], const uint4* s_a, int wm, int kk) {
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int i = 0; i < 2; ++i) ldmatrix_x4(a[i], s_a + swz(wm + i * 16 + lane % 16, kk / 8 + lane / 16));
}

/// B fragments for this warp's 32 columns (four n8 blocks) of a [n][k] weight tile.
__device__ __forceinline__ void load_b(uint32_t (&b)[2][4], const uint4* s_w, int wn, int kk) {
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int j = 0; j < 2; ++j)
        ldmatrix_x4(b[j], s_w + swz(wn + j * 16 + lane % 8 + (lane / 16) * 8, kk / 8 + (lane / 8) % 2));
}

// b[j] = {n-block 2j: k 0-7, k 8-15, n-block 2j+1: k 0-7, k 8-15}.
#define ATLAS_CB3_MMA_32x32(acc, a, b)                                                   \
    _Pragma("unroll") for (int i = 0; i < 2; ++i)                                        \
        _Pragma("unroll") for (int j = 0; j < 2; ++j) {                                  \
            mma_bf16(acc[i][2 * j], a[i], b[j][0], b[j][1]);                             \
            mma_bf16(acc[i][2 * j + 1], a[i], b[j][2], b[j][3]);                         \
        }

// Stage sizes in uint4. Gate/up: [s_a BM rows | s_w1 BN rows | s_w3 BN rows].
constexpr int ROW_U4 = BK / 8;
constexpr int GU_STAGE = (BM + 2 * BN) * ROW_U4;  // 2048 uint4 = 32 KB
constexpr int DN_STAGE = (BM + BN) * ROW_U4;      // 1536 uint4 = 24 KB

}  // namespace

// Dynamic shared memory the launcher passes: 32 KB (gate/up), 24 KB (down) — moe.rs
// FUSED_GATE_UP_SMEM / FUSED_DOWN_SMEM. One stage on purpose: see "L1 IS THE LEVER" above.
static_assert(GU_STAGE * 16 == 32768 && DN_STAGE * 16 == 24576, "keep moe.rs smem sizes in sync");

/// Gate + up + SwiGLU for one (expert tile, N tile):
///   h[row, n] = bf16( silu(min(g, L)) * clamp(u, -L, L) * route_w[row] ),  g/u fp32.
extern "C" __global__ void __launch_bounds__(THREADS) cb3_moe_gate_up(
    const __nv_bfloat16* __restrict__ act,  // [P, K] permuted rows
    const int4* __restrict__ tiles,         // {row_begin, rows, slot, 0}
    const uint8_t* w1_lo, const uint8_t* w1_hi, const uint8_t* w1_cb, const uint8_t* w1_sc,
    const uint8_t* w3_lo, const uint8_t* w3_hi, const uint8_t* w3_cb, const uint8_t* w3_sc,
    unsigned long long lo_s, unsigned long long hi_s, unsigned long long cb_s, unsigned long long sc_s,
    const float* __restrict__ row_weight,   // [P]
    __nv_bfloat16* __restrict__ h,          // [P, N]
    int N, int K, float limit) {
    extern __shared__ __align__(128) uint4 smem[];

    const int4 tile = tiles[blockIdx.y];
    const int row_begin = tile.x, rows = tile.y, slot = tile.z;
    const int n0 = blockIdx.x * BN;
    const Cb3Planes p1 = planes_for(w1_lo, w1_hi, w1_cb, w1_sc, lo_s, hi_s, cb_s, sc_s, slot);
    const Cb3Planes p3 = planes_for(w3_lo, w3_hi, w3_cb, w3_sc, lo_s, hi_s, cb_s, sc_s, slot);

    // 8 warps: 4 along M x 2 along N, each a 32x32 quadrant.
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = (warp / 2) * 32, wn = (warp % 2) * 32;
    // Rows beyond `rows` are zero; a warp whose whole quadrant is padding skips the MMAs.
    const bool live = wm < rows;
    float acc_g[2][4][4], acc_u[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc_g[i][j][e] = acc_u[i][j][e] = 0.0f;

    // Thread t decodes (row t % 64, group (t / 64) % 2) of w1 (t < 128) or w3 (t >= 128).
    const Cb3Planes& mine = threadIdx.x < 128 ? p1 : p3;
    const int mine_off = (threadIdx.x < 128 ? BM : BM + BN) * ROW_U4;
    // A row's two groups go to adjacent lanes: their shared 16-byte lo word (and the hi word
    // shared with the next K step) is one request per warp instead of one per warp pair.
    const int drow = (threadIdx.x % 128) / 2;
    const int dgrp = threadIdx.x % 2;
    const long long dn = n0 + drow;
    const RowTable tab = row_table(*reinterpret_cast<const uint2*>(mine.cb + dn * 8));
    RawPair cur = fetch_pair(mine, dn, 0, K);
    uint4 a_reg[ACT_CHUNKS];
    fetch_act(act, row_begin, rows, 0, K, a_reg);

    auto mma_step = [&](const uint4* st) {
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t a[2][4], b1[2][4], b3[2][4];
            load_a(a, st, wm, kk);
            load_b(b1, st + BM * ROW_U4, wn, kk);
            load_b(b3, st + (BM + BN) * ROW_U4, wn, kk);
            ATLAS_CB3_MMA_32x32(acc_g, a, b1)
            ATLAS_CB3_MMA_32x32(acc_u, a, b3)
        }
    };
    for (int k0 = 0; k0 < K; k0 += 2 * BK) {
        RawPair nxt = cur;
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            const int ks = k0 + step * BK;
            store_act(a_reg, smem);
            decode_group_swz(pick(cur, step, dgrp), tab, ks + dgrp * 32, smem + mine_off, drow, dgrp);
            __syncthreads();
            // Next loads go out now and land during the MMAs: the plane pair two steps
            // ahead, the activations one step ahead.
            if (step == 0 && k0 + 2 * BK < K) nxt = fetch_pair(mine, dn, k0 + 2 * BK, K);
            if (ks + BK < K) fetch_act(act, row_begin, rows, ks + BK, K, a_reg);
            if (live) mma_step(smem);
            __syncthreads();
        }
        cur = nxt;
    }

    // Epilogue straight from the accumulators: gate and up share the fragment layout, so the
    // clamp and SwiGLU are element-wise; the association ((g * sig) * u) * w is the
    // reference's, and h is rounded to bf16 exactly once. Element e of acc[i][j] is row
    // wm + 16i + lane/4 (+8 for e >= 2), column wn + 8j + 2(lane%4) + e%2.
    if (!live) return;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = wm + i * 16 + lane / 4 + half * 8;
            if (row >= rows) continue;
            const long long prow = row_begin + row;
            const float w = row_weight[prow];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s[2];
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float g = fminf(acc_g[i][j][2 * half + e], limit);
                    const float u = fminf(fmaxf(acc_u[i][j][2 * half + e], -limit), limit);
                    const float sig = 1.0f / (1.0f + expf(-g));
                    s[e] = g * sig * u;
                }
                const int col = n0 + wn + j * 8 + (lane % 4) * 2;
                *reinterpret_cast<__nv_bfloat162*>(&h[prow * N + col]) =
                    __halves2bfloat162(__float2bfloat16(s[0] * w), __float2bfloat16(s[1] * w));
            }
        }
}

/// Down projection for one (expert tile, N tile): out[row, n] = h[row, :] . w2[n, :], fp32.
extern "C" __global__ void __launch_bounds__(THREADS) cb3_moe_down(
    const __nv_bfloat16* __restrict__ h,    // [P, K]
    const int4* __restrict__ tiles,
    const uint8_t* w2_lo, const uint8_t* w2_hi, const uint8_t* w2_cb, const uint8_t* w2_sc,
    unsigned long long lo_s, unsigned long long hi_s, unsigned long long cb_s, unsigned long long sc_s,
    float* __restrict__ out,                // [P, N]
    int N, int K) {
    extern __shared__ __align__(128) uint4 smem[];

    const int4 tile = tiles[blockIdx.y];
    const int row_begin = tile.x, rows = tile.y, slot = tile.z;
    const int n0 = blockIdx.x * BN;
    const Cb3Planes p2 = planes_for(w2_lo, w2_hi, w2_cb, w2_sc, lo_s, hi_s, cb_s, sc_s, slot);

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = (warp / 2) * 32, wn = (warp % 2) * 32;
    const bool live = wm < rows;
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    // Threads 0..127 decode (row t % 64, group t / 64); 128..255 only move activations.
    const bool decoder = threadIdx.x < BN * (BK / 32);
    // A row's two groups go to adjacent lanes: their shared 16-byte lo word (and the hi word
    // shared with the next K step) is one request per warp instead of one per warp pair.
    const int drow = (threadIdx.x % 128) / 2;
    const int dgrp = threadIdx.x % 2;
    const long long dn = n0 + drow;
    constexpr int W_OFF = BM * ROW_U4;
    RowTable tab{};
    RawPair cur{};
    if (decoder) {
        tab = row_table(*reinterpret_cast<const uint2*>(p2.cb + dn * 8));
        cur = fetch_pair(p2, dn, 0, K);
    }
    uint4 a_reg[ACT_CHUNKS];
    fetch_act(h, row_begin, rows, 0, K, a_reg);

    auto mma_step = [&](const uint4* st) {
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t a[2][4], b[2][4];
            load_a(a, st, wm, kk);
            load_b(b, st + W_OFF, wn, kk);
            ATLAS_CB3_MMA_32x32(acc, a, b)
        }
    };
    for (int k0 = 0; k0 < K; k0 += 2 * BK) {
        RawPair nxt = cur;
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            const int ks = k0 + step * BK;
            store_act(a_reg, smem);
            if (decoder) decode_group_swz(pick(cur, step, dgrp), tab, ks + dgrp * 32, smem + W_OFF, drow, dgrp);
            __syncthreads();
            if (decoder && step == 0 && k0 + 2 * BK < K) nxt = fetch_pair(p2, dn, k0 + 2 * BK, K);
            if (ks + BK < K) fetch_act(h, row_begin, rows, ks + BK, K, a_reg);
            if (live) mma_step(smem);
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
            for (int j = 0; j < 4; ++j)
                *reinterpret_cast<float2*>(dst + j * 8) = make_float2(acc[i][j][2 * half], acc[i][j][2 * half + 1]);
        }
}
