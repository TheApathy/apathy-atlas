// SPDX-License-Identifier: AGPL-3.0-only
//
// EXL3 trellis-coded 3.0 bpw decode GEMV for SM121 (GB10) — M=1 decode path.
//
// Format (EXL3 / ExLlamaV3, QTIP-style trellis, verified against the vendored
// reference at sparkinfer-upstream/b12x/gemm/trellis_linear/csrc/vendor/quant/):
//
//   trellis: I16 [K/16, N/16, 48]   48 u16 = 96 B per 16x16 tile = 3.000 bpw.
//            The 48 u16 (read as 24 LE u32) form a CIRCULAR 768-bit stream.
//            Weight t (t = 0..255 within a tile) is the 16-bit window ENDING
//            at bit ((t + 257) * 3) mod 768; within each u32 the stream runs
//            MSB -> LSB (bit g lives in word g/32 at bit position 31 - g%32).
//   suh:     F16 [K]  input-side sign vector (applied BEFORE the input Hadamard)
//   svh:     F16 [N]  output-side sign vector (applied AFTER the output Hadamard)
//   mcg:     I32 scalar = 0xCBAC1FED (the 3INST cb=1 multiplicative-congruential
//            codebook constant; compile-time here, matching the reference).
//
//   Decode ("3INST", codebook.cuh cb==1):
//     x  = w16 * 0xCBAC1FED            (u32 wrap)
//     x  = (x & 0x8fff8fff) ^ 0x3b603b60    (lop3 immLut 0x6a == (a&b)^c)
//     w  = fp16(x.lo16) + fp16(x.hi16)      (IEEE fp16 add, RN)
//   Decoded weights lie in (-4, 4); there are NO group scales / zero points.
//
//   Tile linear order t -> (k, n) inside the 16x16 tile (this is the
//   m16n8k16 MMA B-fragment order the quantizer packed for; lane = t/8,
//   s = t%8):
//     n_in_tile = 8*(s/4) + lane/4
//     k_in_tile = 2*(lane%4) + (s%2) + 8*((s%4)/2)
//
//   End-to-end math (b = decoded tile matrix, B[k,n], W = b^T):
//     x' = H128( diag(suh) . x ) / sqrt(128)     (blockwise-128 Sylvester
//     y0 = B^T x'                                 Hadamard along K)
//     y  = diag(svh) . H128( y0 ) / sqrt(128)    (blockwise-128 along N)
//   The Hadamard is a SYLVESTER transform per aligned 128-chunk:
//   H[i][j] = (-1)^popcount(i & j), NOT a 16-point per-tile transform.
//
// Kernel design (exl3_gemv_m1):
//   - grid.x = N/128 output strips (8 tile-columns each), grid.y = SPLIT_K.
//   - block = 256 threads = 8 warps. Warp w owns tile-column w of the strip;
//     lane accumulates 2 outputs (n = 16w + lane/4 and +8) in fp32.
//   - Phase 1: block computes x' for one SUPERBLOCK (up to 8 chunks
//     = 1024 k) of its K-slice into smem, one 128-chunk per warp via the
//     warp-shuffle Hadamard (4 elems/lane, fp32 math); refilled per
//     superblock, so the K-slice length is unbounded. x' is STORED as
//     packed __half2 (k, k+1) pairs — 2 KB — feeding the half2 dot path
//     below (round 3; the fp16 store is a quantization of x', part of the
//     documented numerics tier, see docs/kernels/exl3-gemv.md §3b).
//     At the production geometry a CTA runs 4-6 stages, so the refill never
//     fires after the prologue.
//   - Phase 2: trellis is streamed through a 3-stage cp.async smem ring;
//     each stage is 8 tile-rows x 8 tile-cols = 6 KB (one 128-k chunk),
//     issued as 16-B cp.async per thread — wide, coalesced, contiguous 768-B
//     runs per tile-row. One warp decodes one 96-B tile per iteration with
//     the dq8 batching (8 weights per lane from two u32 loads). The dot is
//     accumulated in __half2 HFMA2 chains (4 independent chains: {acc0,acc1}
//     x even/odd tile-row), converted to fp32 ONCE PER 128-k CHUNK in fixed
//     order — this removes the 8 HADD2.F32 cvts + 8 FFMAs per tile that the
//     round-2 SASS showed to be 24% of the issue stream. The ring re-arms
//     stage s+2 at the TOP of iteration s (legal because the third buffer
//     holds the stage consumed at s-1, which the top barrier already ordered)
//     — ONE block barrier per 6 KB stage, prefetch window = two compute
//     periods. Round 4; rationale below.
//   - Phase 3: quad shuffle-reduce, strip partial in smem, then (split 0 of 1,
//     or the LAST split to finish, elected by an atomic counter) applies the
//     output Hadamard-128 + svh and stores bf16. Split partials are combined
//     in fixed split order, so the result is deterministic for a fixed grid.
//
// Occupancy: smem = 2 KB x' + 3 x 6 KB stages + 0.5 KB partials = 20,996 B
// (ptxas), 64 regs, 0 spills -> __launch_bounds__(256, 4) = 4 CTAs/SM
// (32 warps) on GB10's 100 KB SMs: 4 x (20,996 + 1 KB driver reserve) =
// 88 KB. Registers, not smem, are the binding limit (64 x 256 x 4 = the full
// 64 K file), so the extra stage buffer is free.
//
// ROUND-4 DIAGNOSIS (docs/kernels/exl3-gemv.md §3c). Rounds 1-3 chased the
// instruction stream and stalled at 156 -> 168 -> 168 GB/s. The SASS cycle
// budget says why: at the measured 19.0 us / 166 GB/s the SM issues ~39 K
// warp-instructions against ~45.6 K cycles of 4-wide issue = 21.5% issue
// utilisation, and LDS is 6%. ~78% of the wall clock is memory stall, so
// instruction count cannot bind and round 3's null result was expected.
// Wave quantisation is also dead: the microtest sweep already covered the
// split that exactly fills 192 CTA slots and it gained ~1%, and the fused
// production grid (strips x split x groups) is already an exact multiple of
// 192 for gate/up at every top_k. What is left is the ONE structural
// difference from every fast GEMV in this directory: those are barrier-free
// warp-private LDG loops (0 barriers, or 1 per 32 KB) and reach 194-206 GB/s;
// this kernel paid 2 block barriers per 6 KB and capped its fetch window at
// one compute period. Round 4 halves the first and doubles the second without
// touching the arithmetic.
//
// Constraints: K % 128 == 0, N % 128 == 0, gridDim.y splits K in 128-aligned
// chunks (any split works; the x' superblock loop removes the old 4096-K
// per-slice cap). Expert shapes (N=2048 K=4096, N=4096 K=2048) satisfy this.
//
// The trellis decode and Hadamard logic is ported from ExLlamaV3
// (https://github.com/turboderp-org/exllamav3, MIT License, Copyright (c)
// 2025 Turboderp), as vendored at revision 704aefd7 / checkpoint revision
// 787d1582 in sparkinfer-upstream b12x/gemm/trellis_linear/csrc/vendor/
// (codebook.cuh, exl3_dq.cuh, hadamard_inner.cuh). See LICENSE.exllamav3
// there; the MIT attribution above must be preserved in derived ports.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "moe_batched_blend.cuh"

#ifndef EXL3_FIXED_BITS
#define EXL3_FIXED_BITS 0
#endif
#ifndef EXL3_HROW_DSV4_FIXED
#define EXL3_HROW_DSV4_FIXED 0
#endif
static_assert(EXL3_HROW_DSV4_FIXED == 0 || EXL3_HROW_DSV4_FIXED == 1,
              "EXL3 fixed H-row mode must be boolean");
#if EXL3_FIXED_BITS
static_assert(EXL3_FIXED_BITS == 2 || EXL3_FIXED_BITS == 3,
              "EXL3 fixed bitrate must be K2 or K3");
#define EXL3_REQUIRE_BITS(bits_) \
    do { if ((bits_) != EXL3_FIXED_BITS) return; } while (0)
#define EXL3_BIT_WIDTH(bits_) EXL3_FIXED_BITS
#else
#define EXL3_REQUIRE_BITS(bits_) do { } while (0)
#define EXL3_BIT_WIDTH(bits_) (bits_)
#endif

#define EXL3_MAX_BITS 3
#define EXL3_MCG_MULT 0xCBAC1FEDu
#define EXL3_BLOCK 256
#define EXL3_NSTRIP 128     // output columns per block (8 tiles of 16)
#define EXL3_STAGE_ROWS 8   // tile-rows per cp.async stage (8*8*96 B = 6 KB = 1 chunk)
#define EXL3_MAX_STAGE_U4 (EXL3_STAGE_ROWS * 8 * 2 * EXL3_MAX_BITS)
#define EXL3_STAGES 3        // cp.async ring depth (round 4; see the barrier note below)
#define EXL3_MAX_XCHUNKS 8   // x' smem superblock: 1024 k as __half2 pairs, 2 KB (refilled per superblock)
#define EXL3_RSQRT128 0.088388347648f

// ---------------------------------------------------------------------------
// Decode primitives (port of exl3_dq.cuh / codebook.cuh, bits=3, cb=1)
// ---------------------------------------------------------------------------

union exl3_h2u32 {
    unsigned int u;
    __half2 h2;
};

// 3INST decode of two 16-bit windows -> half2 (w0 in .x, w1 in .y).
__device__ __forceinline__ __half2 exl3_decode2(unsigned int x0, unsigned int x1) {
    x0 *= EXL3_MCG_MULT;
    x1 *= EXL3_MCG_MULT;
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
    exl3_h2u32 u0, u1;
    u0.u = x0;
    u1.u = x1;
    __half2 lo = __lows2half2(u0.h2, u1.h2);
    __half2 hi = __highs2half2(u0.h2, u1.h2);
    return __hadd2(lo, hi);
}

// Per-lane window geometry for t_offset = lane*8 (compile-time function of
// lane; hoist out of the K loop). Circular within the 24-u32 tile.
//
// For t = 8*lane the geometry is provably benign:
//   b1 = (8*lane + 257)*3 = 24*lane + 771, b0 = b1 - 16, b2 = b1 + 21.
//   b0 % 32 cycles {19, 11, 3, 27} (all <= 27), so the 37-bit span of the 8
//   windows never crosses THREE u32 words: ib == ia + 1 (mod 24) always, and
//   s2 = (i2+1)*32 - b2 is always one of {0, 8, 16, 24} — in particular
//   s2 < 32, which exl3_dq8 relies on for its single-funnel-shift alignment.
struct Exl3LaneGeom {
    int ia, ib, s2;
};

__device__ __forceinline__ Exl3LaneGeom exl3_lane_geom(int lane, int bits) {
    int t = lane * 8;
    int b1 = (t + 257) * bits;  // end of first window
    int b0 = b1 - 16;                // start of first window
    int b2 = b1 + 7 * bits;     // end of last window
    int i0 = b0 >> 5;
    int i2 = (b2 - 1) >> 5;
    Exl3LaneGeom g;
    g.ia = i0 % (8 * bits);
    g.ib = i2 % (8 * bits);
    g.s2 = (i2 + 1) * 32 - b2;
    return g;
}

// dq8: 8 weights per lane from two u32 loads. d01=(t0,t1) d23=(t2,t3)
// d45=(t4,t5) d67=(t6,t7); t = lane*8 + s.
//
// ILP restructure vs the vendored align=4 dq8: the reference derives the
// even windows by a SERIAL shift chain (w7 -> w6 -> w5 -> w4), which puts 7
// dependent ops in front of the last window's decode and caps the warp at
// IPC ~1 (measured 156 GB/s plateau, issue-latency bound). Instead we align
// the whole 37-bit window span ONCE (m = merged >> s2; s2 < 32 by the lane
// geometry above, so m_lo is a single funnel shift), then extract the four
// odd windows with INDEPENDENT immediate funnel shifts and the four even
// windows with independent `>> 3`s. Every window is a pure function of
// (m_lo, m_hi), so the four decode2 chains interleave in the scoreboard.
// Bit-exact vs the serial chain: all windows are masked to 16 bits, and
// bits [s2+3j, s2+3j+16) of `merged` are identical whether reached by one
// funnel shift or by truncate-then-shift.
__device__ __forceinline__ void exl3_dq8(const unsigned int* __restrict__ tile,
                                         const Exl3LaneGeom g, __half2& d01, __half2& d23,
                                         __half2& d45, __half2& d67, int bits) {
    unsigned int a = tile[g.ia];
    unsigned int b = tile[g.ib];
    unsigned int mlo = __funnelshift_r(b, a, g.s2);  // s2 in {0,8,16,24} < 32
    unsigned int mhi = a >> g.s2;
    unsigned int w7 = mlo;
    unsigned int w5 = __funnelshift_r(mlo, mhi, 2 * bits);
    unsigned int w3 = __funnelshift_r(mlo, mhi, 4 * bits);
    unsigned int w1 = __funnelshift_r(mlo, mhi, 6 * bits);
    unsigned int w6 = w7 >> bits;
    unsigned int w4 = w5 >> bits;
    unsigned int w2 = w3 >> bits;
    unsigned int w0 = w1 >> bits;
    d01 = exl3_decode2(w0 & 0xffff, w1 & 0xffff);
    d23 = exl3_decode2(w2 & 0xffff, w3 & 0xffff);
    d45 = exl3_decode2(w4 & 0xffff, w5 & 0xffff);
    d67 = exl3_decode2(w6 & 0xffff, w7 & 0xffff);
}

// ---------------------------------------------------------------------------
// Warp-shuffle Sylvester Hadamard, 128 elements (4 fp32 per lane), port of
// hadamard_inner.cuh shuffle_had_f4x32 + the in-register 4-point stage.
// Elements: lane owns [4*lane .. 4*lane+3]. Output is UNNORMALIZED (caller
// multiplies by 1/sqrt(128)).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void exl3_had128(float& h0, float& h1, float& h2, float& h3,
                                            const int lane) {
    // 4-point stage in registers
    float s0 = h0 + h1;
    float d0 = h0 - h1;
    float s1 = h2 + h3;
    float d1 = h2 - h3;
    h0 = s0 + s1;
    h1 = d0 + d1;
    h2 = s0 - s1;
    h3 = d0 - d1;
    // 32-point stage across lanes
#pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        float p0 = __shfl_xor_sync(0xffffffffu, h0, i);
        float p1 = __shfl_xor_sync(0xffffffffu, h1, i);
        float p2 = __shfl_xor_sync(0xffffffffu, h2, i);
        float p3 = __shfl_xor_sync(0xffffffffu, h3, i);
        float sgn = (lane & i) ? -1.0f : 1.0f;
        h0 = __fmaf_rn(sgn, h0, p0);
        h1 = __fmaf_rn(sgn, h1, p1);
        h2 = __fmaf_rn(sgn, h2, p2);
        h3 = __fmaf_rn(sgn, h3, p3);
    }
}

// ---------------------------------------------------------------------------
// cp.async helpers (same pattern as fp8_gemm_t_blockscaled.cu; .cg to bypass
// L1 — the trellis stream is read exactly once). Round 4 dropped the
// src-size tail predication: every stage is provably a full
// EXL3_STAGE_ROWS-row stage (proof at exl3_load_stage), so the copy is
// unconditional and the issue block carries no branches.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void exl3_cp_async_16(void* dst_smem, const void* src_gmem) {
    unsigned int dst = (unsigned int)__cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src_gmem));
}

__device__ __forceinline__ void exl3_cp_commit() { asm volatile("cp.async.commit_group;"); }

// Issue one stage: tile-rows [r0, r0+EXL3_STAGE_ROWS) of the block's 8-tile
// strip. Each tile-row is a contiguous 768-B run (48 uint4). 384 copies over
// 256 threads: iteration 0 is full, iteration 1 covers threads 0..127.
//
// EVERY stage is exactly EXL3_STAGE_ROWS full tile-rows, so there is no tail
// predication (round 4 removed it). Proof: splits are 128-k-chunk aligned
// (rows_lo = 8*c_lo, rows_hi = 8*c_hi), one stage is exactly one chunk
// (EXL3_STAGE_ROWS == 8), nstages == c_hi - c_lo, and stage s is only ever
// issued for s < nstages -> r0 + 8 = 8*(c_lo + s + 1) <= 8*c_hi = rows_hi.
//
// Lane geometry of the copy: thread t handles (r = t/48, o = t%48), so 2 of
// every 3 warps issue ONE 512-B contiguous request and the third issues two
// 256-B requests — averaging ~427 B per warp-request. That width is why the
// stage is block-cooperative rather than warp-private: a warp-private variant
// would have to fetch 96-B tiles (one tile-column, stride N/16*96), and the
// measured GB10 law in moe_shared_expert_fused_t.cu (:65-75) is that a
// 32-B/warp request pins a GEMV at ~130 GB/s while a 128-B request has no such
// ceiling. Narrowing the request to buy barrier-freedom would lose more than
// it gains.
__device__ __forceinline__ void exl3_load_stage(uint4* __restrict__ dst,
                                                const uint4* __restrict__ trellis_u4, int r0,
                                                int n_tiles_row, int nb0, int bits) {
    const int tile_u4 = 2 * bits;
    const int row_u4 = 8 * tile_u4;
    const int stage_u4 = EXL3_STAGE_ROWS * row_u4;
    const size_t base = ((size_t)r0 * n_tiles_row + nb0) * tile_u4;
    for (int i = 0; i < (EXL3_MAX_STAGE_U4 + EXL3_BLOCK - 1) / EXL3_BLOCK; ++i) {
        int idx = i * EXL3_BLOCK + (int)threadIdx.x;
        if (idx >= stage_u4) break;
        int r = idx / row_u4;
        int o = idx - r * row_u4;
        exl3_cp_async_16(dst + idx, trellis_u4 + base + (size_t)r * n_tiles_row * tile_u4 + o);
    }
    exl3_cp_commit();
}

// Phase-1 input pass for one x' superblock: x' = H128(diag(suh) . x)/sqrt(128)
// computed in fp32 for chunks [c0, c1), stored as packed __half2 (k, k+1)
// pairs at s_x[(c - c0)*64 ..]. One 128-chunk per warp per iteration. The
// fp16 store quantizes x' (rel. 2^-11 per element, incoherent across k) —
// covered by the GEMV cosine gate, see docs/kernels/exl3-gemv.md §3b.
__device__ __forceinline__ void exl3_input_pass(const __nv_bfloat16* __restrict__ A,
                                                const __half* __restrict__ suh,
                                                __half2* __restrict__ s_x, int c0, int c1,
                                                int warp, int lane) {
    for (int c = c0 + warp; c < c1; c += EXL3_BLOCK / 32) {
        int base = (c << 7) + lane * 4;
        float h0 = __bfloat162float(A[base + 0]) * __half2float(suh[base + 0]);
        float h1 = __bfloat162float(A[base + 1]) * __half2float(suh[base + 1]);
        float h2 = __bfloat162float(A[base + 2]) * __half2float(suh[base + 2]);
        float h3 = __bfloat162float(A[base + 3]) * __half2float(suh[base + 3]);
        exl3_had128(h0, h1, h2, h3, lane);
        int lb = ((c - c0) << 6) + lane * 2;
        s_x[lb + 0] = __floats2half2_rn(h0 * EXL3_RSQRT128, h1 * EXL3_RSQRT128);
        s_x[lb + 1] = __floats2half2_rn(h2 * EXL3_RSQRT128, h3 * EXL3_RSQRT128);
    }
}

// ---------------------------------------------------------------------------
// exl3_gemv_m1: C[N] (bf16) = EXL3-decode GEMV of A[K] (bf16).
//
//   grid  = (N/128, SPLIT_K, 1), block = (256, 1, 1)
//   ws    : fp32 [SPLIT_K, N] scratch  (untouched when gridDim.y == 1)
//   counters: int [N/128], must be 0 before FIRST launch; the kernel resets
//             them to 0 on completion, so back-to-back launches are safe.
// ---------------------------------------------------------------------------

// (body shared by exl3_gemv_m1 and exl3_gemv_m1_idx; the 4-CTA/SM occupancy
// bound from the ILP-tuning pass lives on the __global__ wrappers below.)
__device__ __forceinline__ void exl3_gemv_m1_body(
    const __nv_bfloat16* __restrict__ A,         // [K]
    const unsigned short* __restrict__ trellis,  // [K/16, N/16, 48]
    const __half* __restrict__ suh,              // [K]
    const __half* __restrict__ svh,              // [N]
    __nv_bfloat16* __restrict__ C,               // [N]
    float* __restrict__ ws,                      // [gridDim.y, N]
    int* __restrict__ counters,                  // [N/128]
    unsigned int N, unsigned int K, unsigned int bits) {
    __shared__ __align__(16) __half2 s_x[EXL3_MAX_XCHUNKS * 64];            // 2 KB
    __shared__ __align__(16) unsigned short s_stage[EXL3_STAGES][EXL3_STAGE_ROWS * 8 * 48];  // 18 KB
    __shared__ float s_y[EXL3_NSTRIP];
    __shared__ int s_elect;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n0 = blockIdx.x * EXL3_NSTRIP;
    const int nb0 = n0 >> 4;  // first tile-column of the strip
    const int n_tiles_row = N >> 4;
    const int S = gridDim.y;
    const int split = blockIdx.y;

    // 128-aligned K-slice for this split
    const int chunks_total = K >> 7;
    const int c_lo = (int)(((long long)chunks_total * split) / S);
    const int c_hi = (int)(((long long)chunks_total * (split + 1)) / S);
    const int rows_lo = c_lo * 8;  // tile-rows (16 k each)
    // One stage per 128-k chunk (EXL3_STAGE_ROWS == 8), so every stage is
    // exactly full (rows_lo + 8*nstages == 8*c_hi) and nstages == chunks in
    // the slice. exl3_load_stage relies on this: no tail predication.
    const int nstages = c_hi - c_lo;

    // Kick off the trellis pipeline before the x' pass so the DRAM stream
    // starts immediately.
    // Two stages in flight from the prologue; the third buffer stays free so
    // the steady-state loop can re-arm it BEFORE it computes (see below).
    const uint4* trellis_u4 = (const uint4*)trellis;
    exl3_load_stage((uint4*)s_stage[0], trellis_u4, rows_lo, n_tiles_row, nb0, bits);
    if (nstages > 1)
        exl3_load_stage((uint4*)s_stage[1], trellis_u4, rows_lo + EXL3_STAGE_ROWS,
                        n_tiles_row, nb0, bits);

    // ---- Phase 1: x' for the first superblock (up to 1024 k) ----
    {
        int sb1 = c_lo + EXL3_MAX_XCHUNKS;
        if (sb1 > c_hi) sb1 = c_hi;
        exl3_input_pass(A, suh, s_x, c_lo, sb1, warp, lane);
    }
    __syncthreads();

    // ---- Phase 2: stream + decode + accumulate ----
    // The dot runs in __half2 within each 128-k chunk (= one stage): four
    // independent HFMA2 chains ({acc0,acc1} x even/odd tile-row, depth 4
    // each), combined in FIXED order into fp32 once per chunk. Chunks never
    // straddle a split boundary (splits are chunk-aligned), so the grouping
    // — and therefore the fp32 result — is deterministic for a fixed grid.
    float acc0 = 0.0f;  // n = 16*warp + lane/4       (fp32 across chunks)
    float acc1 = 0.0f;  // n = 16*warp + lane/4 + 8
    const Exl3LaneGeom g = exl3_lane_geom(lane, bits);
    const int xkh = lane & 3;  // half2 (k-pair) index within the 16-k tile row

    // ROUND 4 — ONE block barrier per stage, prefetch distance 2.
    //
    // Rounds 1-3 ran a 2-buffer pipeline with TWO __syncthreads() per 6 KB
    // stage: one after `cp.async.wait_group` (stage s landed) and one after
    // the compute (so no warp is still reading s_stage[s&1] when the same
    // buffer is re-armed for stage s+2). The re-arm therefore had to sit at
    // the BOTTOM of the iteration, giving the fetch only ONE compute window
    // to land in.
    //
    // With a THIRD buffer the re-arm target at iteration s is buffer
    // (s+2) % 3 == (s-1) % 3 — the buffer that was consumed at iteration
    // s-1. The single barrier at the top of iteration s already orders every
    // warp past that consume, so the re-arm is legal immediately after it and
    // the trailing barrier is redundant. Net per stage: 2 barriers -> 1, and
    // the fetch for stage s+2 now has compute(s) + compute(s+1) to land in.
    // The wait_group immediates are unchanged (1 in steady state, 0 on the
    // last stage), because the ring still holds exactly two groups in flight.
    //
    // Nothing about the arithmetic moves: the same 8 tiles are decoded from
    // the same bytes in the same order, the fp16 chains are still grouped per
    // 128-k chunk, and the per-chunk fp32 combine and the split combine keep
    // their fixed order. For a fixed grid this is BIT-IDENTICAL to round 3.
    static_assert(EXL3_STAGES == 3,
                  "the single-barrier schedule below is written for a 3-deep ring with "
                  "2 groups in flight: it re-arms stage s+2 into buffer (s+2)%3 == (s-1)%3, "
                  "which is exactly the buffer the top barrier just freed");
    int buf_cur = 0;                  // buffer holding stage s
    int buf_fre = EXL3_STAGES - 1;    // buffer to re-arm with stage s+2
    for (int s = 0; s < nstages; ++s) {
        if (s + 1 < nstages)
            asm volatile("cp.async.wait_group 1;");
        else
            asm volatile("cp.async.wait_group 0;");
        // The ONE barrier: (i) stage s has landed for every warp, and
        // (ii) every warp is past compute(s-1), which frees buf_fre and the
        // old x' superblock.
        __syncthreads();

        if (s + 2 < nstages)
            exl3_load_stage((uint4*)s_stage[buf_fre], trellis_u4,
                            rows_lo + (s + 2) * EXL3_STAGE_ROWS, n_tiles_row, nb0, bits);

        if (s != 0 && (s & (EXL3_MAX_XCHUNKS - 1)) == 0) {
            // Superblock boundary: refill x' for chunks [c_lo+s, +XCHUNKS).
            // The barrier above ordered all reads of the old superblock
            // before this overwrite; the in-flight cp.async stages are
            // unaffected (s_stage only). Costs one extra barrier once every
            // EXL3_MAX_XCHUNKS stages — never taken at the production
            // geometry, where a CTA runs 4-6 stages.
            int sb0 = c_lo + s;
            int sb1 = sb0 + EXL3_MAX_XCHUNKS;
            if (sb1 > c_hi) sb1 = c_hi;
            exl3_input_pass(A, suh, s_x, sb0, sb1, warp, lane);
            __syncthreads();
        }

        const unsigned int* stage32 = (const unsigned int*)s_stage[buf_cur];
        const __half2* xrow = s_x + ((s & (EXL3_MAX_XCHUNKS - 1)) << 6) + xkh;

        const __half2 hz = __float2half2_rn(0.0f);
        __half2 hacc0e = hz, hacc0o = hz;  // acc0, even/odd tile-row
        __half2 hacc1e = hz, hacc1o = hz;  // acc1, even/odd tile-row
#pragma unroll
        for (int r = 0; r < EXL3_STAGE_ROWS; ++r) {
            const unsigned int* tile = stage32 + (r * 8 + warp) * (8 * bits);
            __half2 d01, d23, d45, d67;
            exl3_dq8(tile, g, d01, d23, d45, d67, bits);
            __half2 xa = xrow[(r << 3)];      // k, k+1
            __half2 xc = xrow[(r << 3) + 4];  // k+8, k+9
            if (r & 1) {
                hacc0o = __hfma2(d01, xa, hacc0o);
                hacc0o = __hfma2(d23, xc, hacc0o);
                hacc1o = __hfma2(d45, xa, hacc1o);
                hacc1o = __hfma2(d67, xc, hacc1o);
            } else {
                hacc0e = __hfma2(d01, xa, hacc0e);
                hacc0e = __hfma2(d23, xc, hacc0e);
                hacc1e = __hfma2(d45, xa, hacc1e);
                hacc1e = __hfma2(d67, xc, hacc1e);
            }
        }
        // Fixed-order per-chunk combine into fp32 (deterministic).
        __half2 h0 = __hadd2(hacc0e, hacc0o);
        __half2 h1 = __hadd2(hacc1e, hacc1o);
        float2 f0 = __half22float2(h0);
        float2 f1 = __half22float2(h1);
        acc0 += f0.x;
        acc0 += f0.y;
        acc1 += f1.x;
        acc1 += f1.y;

        // Rotate the ring (no trailing barrier — see the round-4 note above).
        buf_cur = (buf_cur + 1 == EXL3_STAGES) ? 0 : buf_cur + 1;
        buf_fre = (buf_fre + 1 == EXL3_STAGES) ? 0 : buf_fre + 1;
    }

    // ---- Phase 3: reduce + (elected) output Hadamard + svh + store ----
    // Quad reduction: lanes with equal lane/4 hold the same n over disjoint k.
    float acc0f = acc0;
    float acc1f = acc1;
    acc0f += __shfl_xor_sync(0xffffffffu, acc0f, 1);
    acc0f += __shfl_xor_sync(0xffffffffu, acc0f, 2);
    acc1f += __shfl_xor_sync(0xffffffffu, acc1f, 1);
    acc1f += __shfl_xor_sync(0xffffffffu, acc1f, 2);
    if ((lane & 3) == 0) {
        s_y[warp * 16 + (lane >> 2)] = acc0f;
        s_y[warp * 16 + (lane >> 2) + 8] = acc1f;
    }
    __syncthreads();

    if (S > 1) {
        // Publish this split's raw (pre-Hadamard) partial, then elect the
        // LAST split to finish for the final combine. Fixed combine order
        // (split 0..S-1) keeps the fp32 sum deterministic for a fixed grid.
        if (threadIdx.x < EXL3_NSTRIP)
            ws[(size_t)split * N + n0 + threadIdx.x] = s_y[threadIdx.x];
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            int prev = atomicAdd(&counters[blockIdx.x], 1);
            s_elect = (prev == S - 1) ? 1 : 0;
        }
        __syncthreads();
        if (!s_elect) return;
        __threadfence();
        if (threadIdx.x < EXL3_NSTRIP) {
            float sum = 0.0f;
            for (int p = 0; p < S; ++p) sum += ws[(size_t)p * N + n0 + threadIdx.x];
            s_y[threadIdx.x] = sum;
        }
        if (threadIdx.x == 0) counters[blockIdx.x] = 0;  // re-arm for next launch
        __syncthreads();
    }

    if (warp == 0) {
        int nb = n0 + lane * 4;
        float h0 = s_y[lane * 4 + 0];
        float h1 = s_y[lane * 4 + 1];
        float h2 = s_y[lane * 4 + 2];
        float h3 = s_y[lane * 4 + 3];
        exl3_had128(h0, h1, h2, h3, lane);
        C[nb + 0] = __float2bfloat16(h0 * EXL3_RSQRT128 * __half2float(svh[nb + 0]));
        C[nb + 1] = __float2bfloat16(h1 * EXL3_RSQRT128 * __half2float(svh[nb + 1]));
        C[nb + 2] = __float2bfloat16(h2 * EXL3_RSQRT128 * __half2float(svh[nb + 2]));
        C[nb + 3] = __float2bfloat16(h3 * EXL3_RSQRT128 * __half2float(svh[nb + 3]));
    }
}

extern "C" __global__ void __launch_bounds__(EXL3_BLOCK, 4) exl3_gemv_m1(
    const __nv_bfloat16* __restrict__ A,         // [K]
    const unsigned short* __restrict__ trellis,  // [K/16, N/16, 48]
    const __half* __restrict__ suh,              // [K]
    const __half* __restrict__ svh,              // [N]
    __nv_bfloat16* __restrict__ C,               // [N]
    float* __restrict__ ws,                      // [gridDim.y, N]
    int* __restrict__ counters,                  // [N/128]
    unsigned int N, unsigned int K, unsigned int bits) {
    EXL3_REQUIRE_BITS(bits);
    exl3_gemv_m1_body(
        A, trellis, suh, svh, C, ws, counters, N, K, EXL3_BIT_WIDTH(bits));
}

// ---------------------------------------------------------------------------
// exl3_gemv_m1_idx: device-indexed variant for the MoE decode dispatch.
//
// Instead of per-expert pointers it takes device POINTER TABLES (one u64 per
// expert, built by the loader) plus the routed `indices` buffer the top-k
// kernel wrote, and a compile-side `slot` (0..top_k-1). The expert id is read
// ON DEVICE (e = indices[slot]) so the launch sequence is identical every
// step — no D2H of the routing, safe under CUDA graph capture.
//
//   grid/block/ws/counters: exactly as exl3_gemv_m1.
// ---------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(EXL3_BLOCK, 4) exl3_gemv_m1_idx(
    const __nv_bfloat16* __restrict__ A,              // [K]
    const unsigned long long* __restrict__ trellis_tab,  // [num_experts]
    const unsigned long long* __restrict__ suh_tab,      // [num_experts]
    const unsigned long long* __restrict__ svh_tab,      // [num_experts]
    const unsigned int* __restrict__ indices,         // [top_k] routed ids
    unsigned int slot,                                // which routed slot
    __nv_bfloat16* __restrict__ C,                    // [N]
    float* __restrict__ ws,                           // [gridDim.y, N]
    int* __restrict__ counters,                       // [N/128]
    unsigned int N, unsigned int K, unsigned int bits) {
    EXL3_REQUIRE_BITS(bits);
    const unsigned int e = indices[slot];
    exl3_gemv_m1_body(A, (const unsigned short*)trellis_tab[e],
                      (const __half*)suh_tab[e], (const __half*)svh_tab[e], C, ws,
                      counters, N, K, EXL3_BIT_WIDTH(bits));
}

// ---------------------------------------------------------------------------
// FUSED MoE decode entry points: ONE launch for ALL routed slots.
//
// `exl3_gemv_m1_idx` above is the bring-up shape — one launch per (slot,
// projection), i.e. 3·top_k GEMV launches per layer. At 43 layers × top_k 6-8
// that is ~800-1000 launches per decoded token, and the EXL3 byte win (2.43
// GB/token vs MXFP4's 3.45) is eaten by launch + tail underfill.
//
// The fused pair below mirrors the MXFP4/BF16 fused decode organization
// (`moe_expert_gate_up_shared_bf16`: expert slot on blockIdx.y, projection on
// blockIdx.z, expert id resolved per-CTA from the routing buffer). EXL3 needs
// blockIdx.y for SPLIT_K, so the (slot, projection) pair rides blockIdx.z:
//
//   exl3_gemv_m1_fused_gate_up  grid = (N/128, SPLIT_K, 2·top_k)
//                               z = 2·slot + proj   (proj 0 = gate, 1 = up)
//   exl3_gemv_m1_fused_down     grid = (N/128, SPLIT_K, top_k)
//                               z = slot
//
// Per-CTA work is IDENTICAL to the per-slot kernel — same body, same K-slice
// for a given (blockIdx.y, gridDim.y), same fixed-order per-chunk fp32
// combine, same fixed-order split combine. Fusing changes only WHICH CTA does
// which (slot, strip, split) triple, so the outputs are BIT-IDENTICAL to the
// per-slot chain at equal SPLIT_K (gated in exl3_gemv_microtest.rs, GATE8).
//
// SPLIT-K SCRATCH ACROSS SLOTS (the one thing that must not be got wrong):
// per-slot launches serialize on the stream and can therefore share ONE
// ws/counters scratch. Fused slots run CONCURRENTLY, so every group gets a
// private region:
//
//   ws       + group · gridDim.y · N        (fp32 [group][SPLIT_K][N])
//   counters + group · (N / 128)            (i32  [group][N/128])
//
// Both offsets are computed from the launch geometry, so the host only has to
// size the allocation for the widest (groups, SPLIT_K, N) it will ever launch.
// The self-resetting counter logic is untouched: within a group exactly
// SPLIT_K CTAs hit counters[group][strip], the last one elected re-arms it to
// 0, and groups never touch each other's counters — so back-to-back launches
// (and CUDA-graph replays) still start from an all-zero counter array.
// ---------------------------------------------------------------------------

// Fused gate+up over all routed slots. Every slot reads the SAME activation
// A[K]; outputs land in the per-slot rows the existing SwiGLU expects
// (gate_out[slot·N .. ], up_out[slot·N .. ]).
extern "C" __global__ void __launch_bounds__(EXL3_BLOCK, 4) exl3_gemv_m1_fused_gate_up(
    const __nv_bfloat16* __restrict__ A,                    // [K] (shared by all slots)
    const unsigned long long* __restrict__ gate_trellis_tab,  // [num_experts]
    const unsigned long long* __restrict__ gate_suh_tab,      // [num_experts]
    const unsigned long long* __restrict__ gate_svh_tab,      // [num_experts]
    const unsigned long long* __restrict__ up_trellis_tab,    // [num_experts]
    const unsigned long long* __restrict__ up_suh_tab,        // [num_experts]
    const unsigned long long* __restrict__ up_svh_tab,        // [num_experts]
    const unsigned int* __restrict__ indices,               // [top_k] routed ids
    __nv_bfloat16* __restrict__ gate_out,                   // [top_k, N]
    __nv_bfloat16* __restrict__ up_out,                     // [top_k, N]
    float* __restrict__ ws,                                 // [gridDim.z, gridDim.y, N]
    int* __restrict__ counters,                             // [gridDim.z, N/128]
    unsigned int N, unsigned int K, unsigned int bits) {
    EXL3_REQUIRE_BITS(bits);
    const unsigned int group = blockIdx.z;
    const unsigned int slot = group >> 1;
    const unsigned int proj = group & 1u;  // 0 = gate, 1 = up
    const unsigned int e = indices[slot];
    const unsigned long long* tt = proj ? up_trellis_tab : gate_trellis_tab;
    const unsigned long long* su = proj ? up_suh_tab : gate_suh_tab;
    const unsigned long long* sv = proj ? up_svh_tab : gate_svh_tab;
    __nv_bfloat16* C = (proj ? up_out : gate_out) + (size_t)slot * N;
    exl3_gemv_m1_body(A, (const unsigned short*)tt[e], (const __half*)su[e],
                      (const __half*)sv[e], C,
                      ws + (size_t)group * gridDim.y * N, counters + group * (N >> 7), N,
                      K, EXL3_BIT_WIDTH(bits));
}

// Fused down over all routed slots. Slot s consumes the SwiGLU activation row
// `act + s·K` (K = intermediate size) and writes `down_out + s·N`.
extern "C" __global__ void __launch_bounds__(EXL3_BLOCK, 4) exl3_gemv_m1_fused_down(
    const __nv_bfloat16* __restrict__ act,               // [top_k, K] per-slot activations
    const unsigned long long* __restrict__ trellis_tab,  // [num_experts]
    const unsigned long long* __restrict__ suh_tab,      // [num_experts]
    const unsigned long long* __restrict__ svh_tab,      // [num_experts]
    const unsigned int* __restrict__ indices,            // [top_k] routed ids
    __nv_bfloat16* __restrict__ down_out,                // [top_k, N]
    float* __restrict__ ws,                              // [gridDim.z, gridDim.y, N]
    int* __restrict__ counters,                          // [gridDim.z, N/128]
    unsigned int N, unsigned int K, unsigned int bits) {
    EXL3_REQUIRE_BITS(bits);
    const unsigned int slot = blockIdx.z;
    const unsigned int e = indices[slot];
    exl3_gemv_m1_body(act + (size_t)slot * K, (const unsigned short*)trellis_tab[e],
                      (const __half*)suh_tab[e], (const __half*)svh_tab[e],
                      down_out + (size_t)slot * N, ws + (size_t)slot * gridDim.y * N,
                      counters + slot * (N >> 7), N, K, EXL3_BIT_WIDTH(bits));
}

// ===========================================================================
// M-ROW (speculative verify) decode GEMV — the EXL3 twin of the MXFP4
// `exp_splitk_m_t` family in moe_shared_expert_fused_t.cu.
//
// WHY. At the γ-verify the MoE runs num_tokens rows through the SAME routed
// expert set. Without dedup every row re-streams the whole set: measured on
// GB10 the m=6 MXFP4 expert union is 54.1 ms of a ~113 ms verify step, the
// single largest bucket and the only one that scales with verify width. The
// MXFP4 `_m` kernels elect ONE leader block per distinct expert id across all
// rows and FMA the decoded weight into every row that selected it, so the
// weight bytes are read once per expert instead of once per (expert, row).
// This file's twin does the same for the 3.0 bpw trellis stream.
//
// CONTRACT MIRRORED FROM `exp_splitk_m_t` (so the scheduler above is unchanged):
//
//  1. ROW -> EXPERT. `indices` is the flat `[num_tokens*top_k]` routing buffer
//     the per-row top-k kernels wrote. Flat slot y holds token `y / top_k`'s
//     `y % top_k`-th expert. `total_routed = num_tokens * top_k`.
//  2. UNION / DEDUP. One CTA-set per flat slot (gate_up rides `2*y + proj` on
//     grid.z, down rides `y`). The FIRST slot holding a given expert id is the
//     LEADER and computes every slot routed to that expert; later duplicates
//     exit before touching memory. Election and the gather are done ON DEVICE
//     by thread 0 over a cooperatively staged copy of `indices`.
//  3. OUTPUT LAYOUT. Row m writes `out + slots[m]*N` — the same flat routed
//     slot row the per-row m=1 chain writes, so SwiGLU and the blend are
//     untouched.
//  4. SPLIT-K PARTIALS. `ws` is indexed by the OUTPUT ROW, not by the launch
//     group: region `ws_row = ws_row_mul*slots[m] + ws_row_add` (gate_up:
//     `2*slot + proj`; down: `slot`), each `[S][N]` fp32. Flat routed slots
//     are unique across leaders, so leaders never collide. `counters` stays
//     indexed by launch group (`N/128` ints each) and is self-resetting, so
//     back-to-back launches and graph replays still start from all-zero.
//  5. GRAPH SAFETY. Expert ids are read on device; the launch geometry is a
//     function of (num_tokens, top_k, N, K) only, never of the routing
//     content. No D2H, identical launch sequence every step.
//
// THE EXACT-GEMV LAW (docs/DECODE-WATERFALL-2026-08-10.md, and the measured
// o-proj result: partial exactness scored 2.54 tok/step against 2.83 for none
// and 2.92-3.01 for full). Every verify row's expert output MUST be BIT-
// IDENTICAL to what the m=1 fused path computes for that same token. This
// kernel guarantees that STRUCTURALLY, not statistically:
//
//   * x' is produced by the SAME `exl3_input_pass` device function, on the
//     same activation row and the same per-expert `suh`. The Hadamard is per
//     aligned 128-chunk, so the x' bits for a chunk depend only on that chunk
//     — the smaller m-row superblock (EXL3_M_XCHUNKS vs EXL3_MAX_XCHUNKS,
//     which only trades smem for refill frequency) cannot move a single bit.
//   * the K-slice per split uses the SAME `chunks_total*split/S` formula at
//     the SAME S (the host passes the m=1 `split_for(N)`), so every split sees
//     the identical 128-aligned chunk range.
//   * per row the k order, the four HFMA2 chains, their even/odd tile-row
//     assignment, the per-128-k-chunk `__hadd2` + `__half22float2` + four fp32
//     adds, the quad shuffle-reduce, the fixed split-order combine (p = 0..S-1)
//     and the output Hadamard/svh/bf16 store are the SAME op sequence, in the
//     same order, as `exl3_gemv_m1_body`.
//   * rows NEVER interact arithmetically. Each row owns a private accumulator
//     chain; dedup changes only WHICH CTA evaluates a row, exactly like the
//     m=1 -> fused collapse gated by GATE8. FP results are invariant under
//     instruction scheduling, so register allocation and occupancy cannot move
//     them either.
//
// Gated by GATE9 of `exl3_gemv_microtest` (every row byte-identical to the m=1
// fused path at m in {2,4,6,8} with a non-slot-ordered, duplicate-heavy index
// list, plus relaunch byte-identity across concurrent groups).
// ===========================================================================

// x' superblock per gathered row: 8 chunks = 1024 k, stored as packed __half2
// = 2 KB/row. Eight chunks is exactly one per warp, so the input pass fills a
// superblock in ONE fully-occupied iteration per row. Smaller than the m=1
// kernel's 16 purely to keep MROW slices affordable in smem (MROW=6 -> 12 KB
// x' + 12 KB stage + 3 KB s_y ~ 27 KB, 3 CTAs/SM on GB10's 100 KB). It is a
// power of two, which the refill/index arithmetic requires, and it is
// numerically inert (see the law above). MROW=16 uses four chunks: eight would
// require 55,368 B of static shared memory after the inlined routing gather,
// above SM121's 48 KiB per-block static limit. With four, ptxas reports
// 40,008 B; it merely adds a storage refill after the fourth 128-k chunk, and
// arithmetic order is unchanged.
#define EXL3_M_XCHUNKS 8
#define EXL3_M16_XCHUNKS 4

// Leader election + slot gather. Algorithmic twin of `mrow_gather_slots` in
// moe_shared_expert_fused_t.cu, minus the shared-expert block-set (EXL3
// checkpoints serve the shared expert through the NVFP4 `w4a16_gemv` chain,
// outside this family).
//
// Returns a pointer to the block's shared `slots[MROW]`, or nullptr when this
// block is a duplicate and must exit. Surplus ladder entries (MROW > gathered
// count) are filled with slots[0] so every `s_slot[m]` read is defined; their
// results are discarded at emit.
//
// `MROW*32` routing staging bounds `total_routed = num_tokens*top_k`: the host
// picks MROW >= num_tokens, so the invariant is top_k <= 32 (EXL3_MAX_TOP_K).
// Every branch below is block-uniform (y comes from blockIdx, total_routed from
// the launch), so the __syncthreads are reached by all threads or none.
template <int MROW>
__device__ __forceinline__ const unsigned int* exl3_mrow_gather(
    const unsigned int* __restrict__ indices, unsigned int y, unsigned int total_routed,
    unsigned int& m_out) {
    __shared__ unsigned int s_idx[MROW * 32];
    __shared__ unsigned int s_slot[MROW];
    __shared__ unsigned int s_m;
    // Stage the routing cooperatively — the scan below is serial on thread 0,
    // and leaving it against global memory makes every block pay up to `y`
    // dependent loads with the rest of the block parked at the barrier.
    for (unsigned int i = threadIdx.x; i < total_routed; i += blockDim.x) {
        s_idx[i] = indices[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int m = 0;
        const unsigned int e = s_idx[y];
        bool leader = true;
        for (unsigned int s = 0; s < y; ++s) {
            if (s_idx[s] == e) {
                leader = false;
                break;
            }
        }
        if (leader) {
            for (unsigned int s = y; s < total_routed; ++s) {
                if (s_idx[s] == e) {
                    if (m < MROW) s_slot[m] = s;
                    ++m;
                }
            }
        }
        s_m = m;  // 0 => duplicate slot, nothing to do
        // Alias the ladder's surplus rows onto row 0 (see the header note).
        if (m > 0) {
            for (unsigned int i = m; i < MROW; ++i) s_slot[i] = s_slot[0];
        }
    }
    __syncthreads();
    if (s_m == 0) return nullptr;
    m_out = min(s_m, (unsigned int)MROW);
    return s_slot;
}

// The m-row body. `A_IS_TOKEN` picks the activation row mapping:
//   true  (gate/up): A is `[num_tokens, K]`, row = slots[m] / top_k
//   false (down)   : A is `[total_routed, K]`, row = slots[m]
template <int MROW, bool A_IS_TOKEN>
__device__ __forceinline__ void exl3_gemv_mrow_body(
    const __nv_bfloat16* __restrict__ A,         // activation base (see A_IS_TOKEN)
    const unsigned short* __restrict__ trellis,  // [K/16, N/16, 48] for this expert
    const __half* __restrict__ suh,              // [K]
    const __half* __restrict__ svh,              // [N]
    __nv_bfloat16* __restrict__ C,               // [total_routed, N] output base
    float* __restrict__ ws,                      // base; row region = [S, N]
    int* __restrict__ counters,                  // pre-offset by launch group
    const unsigned int* __restrict__ s_slot,     // shared [MROW] from the gather
    unsigned int M,                              // gathered rows (<= MROW)
    unsigned int n_block, unsigned int split, unsigned int splits,
    unsigned int ws_row_mul, unsigned int ws_row_add, unsigned int top_k, unsigned int N,
    unsigned int K, unsigned int bits) {
    constexpr int XCHUNKS = MROW == 16 ? EXL3_M16_XCHUNKS : EXL3_M_XCHUNKS;
    __shared__ __align__(16) __half2 s_x[MROW * XCHUNKS * 64];
    __shared__ __align__(16) unsigned short s_stage[2][EXL3_STAGE_ROWS * 8 * 48];  // 12 KB
    __shared__ float s_y[MROW][EXL3_NSTRIP];
    __shared__ int s_elect;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n0 = n_block * EXL3_NSTRIP;
    const int nb0 = n0 >> 4;  // first tile-column of the strip
    const int n_tiles_row = N >> 4;
    const int S = splits;

    // 128-aligned K-slice for this split — IDENTICAL formula and identical S
    // to exl3_gemv_m1_body, which is half the exact-GEMV argument.
    const int chunks_total = K >> 7;
    const int c_lo = (int)(((long long)chunks_total * split) / S);
    const int c_hi = (int)(((long long)chunks_total * (split + 1)) / S);
    const int rows_lo = c_lo * 8;
    const int nstages = c_hi - c_lo;

    // Kick the trellis pipeline before the x' pass — this is the ONE stream
    // the dedup exists to read once, so it starts first.
    const uint4* trellis_u4 = (const uint4*)trellis;
    exl3_load_stage((uint4*)s_stage[0], trellis_u4, rows_lo, n_tiles_row, nb0, bits);
    if (nstages > 1)
        exl3_load_stage((uint4*)s_stage[1], trellis_u4, rows_lo + EXL3_STAGE_ROWS,
                        n_tiles_row, nb0, bits);

    // Per-row activation base. Read from smem at the point of use rather than
    // cached in an `A_row[MROW]` register array: it is touched only in the
    // (cold) input pass, and MROW live pointers through the k sweep is exactly
    // the register pressure the MXFP4 twin's ladder note warns about.
#define EXL3_M_AROW(m_) \
    (A + (unsigned long long)(A_IS_TOKEN ? (s_slot[(m_)] / top_k) : s_slot[(m_)]) * K)

    // ---- Phase 1: x' for the first superblock, one slice per gathered row ----
    {
        int sb1 = c_lo + XCHUNKS;
        if (sb1 > c_hi) sb1 = c_hi;
#pragma unroll
        for (int m = 0; m < MROW; ++m) {
            if (m >= (int)M) break;
            exl3_input_pass(EXL3_M_AROW(m), suh, s_x + m * (XCHUNKS * 64), c_lo, sb1,
                            warp, lane);
        }
    }
    __syncthreads();

    // ---- Phase 2: stream + decode ONCE + accumulate into every row ----
    float acc0[MROW];  // n = 16*warp + lane/4
    float acc1[MROW];  // n = 16*warp + lane/4 + 8
#pragma unroll
    for (int m = 0; m < MROW; ++m) {
        acc0[m] = 0.0f;
        acc1[m] = 0.0f;
    }
    const Exl3LaneGeom g = exl3_lane_geom(lane, bits);
    const int xkh = lane & 3;  // half2 (k-pair) index within the 16-k tile row
    // Surplus ladder rows alias x' slice 0 — defined arithmetic, discarded at
    // emit (the twin of the MXFP4 `act_row[m]` slice-0 aliasing).
    int xoff[MROW];
#pragma unroll
    for (int m = 0; m < MROW; ++m)
        xoff[m] = ((m < (int)M) ? m : 0) * (XCHUNKS * 64);

    for (int s = 0; s < nstages; ++s) {
        if (s != 0 && (s & (XCHUNKS - 1)) == 0) {
            // Superblock boundary: refill every row's x' for chunks
            // [c_lo+s, +XCHUNKS). The previous iteration's trailing
            // __syncthreads() ordered all reads of the old superblock before
            // this overwrite; the two in-flight cp.async stages are unaffected.
            int sb0 = c_lo + s;
            int sb1 = sb0 + XCHUNKS;
            if (sb1 > c_hi) sb1 = c_hi;
#pragma unroll
            for (int m = 0; m < MROW; ++m) {
                if (m >= (int)M) break;
                exl3_input_pass(EXL3_M_AROW(m), suh, s_x + m * (XCHUNKS * 64), sb0,
                                sb1, warp, lane);
            }
            __syncthreads();
        }
        if (s + 1 < nstages)
            asm volatile("cp.async.wait_group 1;");
        else
            asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        const unsigned int* stage32 = (const unsigned int*)s_stage[s & 1];
        const int xc = ((s & (XCHUNKS - 1)) << 6) + xkh;

        const __half2 hz = __float2half2_rn(0.0f);
        __half2 hacc0e[MROW], hacc0o[MROW], hacc1e[MROW], hacc1o[MROW];
#pragma unroll
        for (int m = 0; m < MROW; ++m) {
            hacc0e[m] = hz;
            hacc0o[m] = hz;
            hacc1e[m] = hz;
            hacc1o[m] = hz;
        }
#pragma unroll
        for (int r = 0; r < EXL3_STAGE_ROWS; ++r) {
            const unsigned int* tile = stage32 + (r * 8 + warp) * (8 * bits);
            __half2 d01, d23, d45, d67;
            // THE WIN: one 96-B tile decoded once, FMA'd into all M rows.
            exl3_dq8(tile, g, d01, d23, d45, d67, bits);
#pragma unroll
            for (int m = 0; m < MROW; ++m) {
                const __half2* xrow = s_x + xoff[m] + xc;
                __half2 xa = xrow[(r << 3)];      // k, k+1
                __half2 xb = xrow[(r << 3) + 4];  // k+8, k+9
                // Per row: the SAME four chains, the SAME even/odd tile-row
                // assignment, the SAME operand order as exl3_gemv_m1_body.
                if (r & 1) {
                    hacc0o[m] = __hfma2(d01, xa, hacc0o[m]);
                    hacc0o[m] = __hfma2(d23, xb, hacc0o[m]);
                    hacc1o[m] = __hfma2(d45, xa, hacc1o[m]);
                    hacc1o[m] = __hfma2(d67, xb, hacc1o[m]);
                } else {
                    hacc0e[m] = __hfma2(d01, xa, hacc0e[m]);
                    hacc0e[m] = __hfma2(d23, xb, hacc0e[m]);
                    hacc1e[m] = __hfma2(d45, xa, hacc1e[m]);
                    hacc1e[m] = __hfma2(d67, xb, hacc1e[m]);
                }
            }
        }
        // Fixed-order per-chunk combine into fp32, per row (deterministic, and
        // the same expression sequence the single-row kernel uses).
#pragma unroll
        for (int m = 0; m < MROW; ++m) {
            __half2 h0 = __hadd2(hacc0e[m], hacc0o[m]);
            __half2 h1 = __hadd2(hacc1e[m], hacc1o[m]);
            float2 f0 = __half22float2(h0);
            float2 f1 = __half22float2(h1);
            acc0[m] += f0.x;
            acc0[m] += f0.y;
            acc1[m] += f1.x;
            acc1[m] += f1.y;
        }

        __syncthreads();
        if (s + 2 < nstages)
            exl3_load_stage((uint4*)s_stage[s & 1], trellis_u4,
                            rows_lo + (s + 2) * EXL3_STAGE_ROWS, n_tiles_row, nb0, bits);
    }

    // ---- Phase 3: reduce + (elected) output Hadamard + svh + store ----
    // Quad reduction per row: lanes with equal lane/4 hold the same n over
    // disjoint k. Run for ALL MROW rows — the shuffles need full-warp
    // participation and MROW is a literal, so no lane can diverge.
#pragma unroll
    for (int m = 0; m < MROW; ++m) {
        float a0 = acc0[m];
        float a1 = acc1[m];
        a0 += __shfl_xor_sync(0xffffffffu, a0, 1);
        a0 += __shfl_xor_sync(0xffffffffu, a0, 2);
        a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
        a1 += __shfl_xor_sync(0xffffffffu, a1, 2);
        if ((lane & 3) == 0) {
            s_y[m][warp * 16 + (lane >> 2)] = a0;
            s_y[m][warp * 16 + (lane >> 2) + 8] = a1;
        }
    }
    __syncthreads();

    if (S > 1) {
        // Publish each row's raw (pre-Hadamard) partial into ITS OWN region —
        // keyed by the flat routed slot, which is unique across leaders — then
        // elect the LAST split of this launch group to finish. Fixed combine
        // order (split 0..S-1) keeps the fp32 sum deterministic, and identical
        // to the single-row kernel's.
#pragma unroll
        for (int m = 0; m < MROW; ++m) {
            if (m >= (int)M) break;
            if (threadIdx.x < EXL3_NSTRIP) {
                const unsigned long long wr =
                    (unsigned long long)ws_row_mul * s_slot[m] + ws_row_add;
                ws[(wr * S + split) * N + n0 + threadIdx.x] = s_y[m][threadIdx.x];
            }
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            int prev = atomicAdd(&counters[n_block], 1);
            s_elect = (prev == S - 1) ? 1 : 0;
        }
        __syncthreads();
        if (!s_elect) return;
        __threadfence();
#pragma unroll
        for (int m = 0; m < MROW; ++m) {
            if (m >= (int)M) break;
            if (threadIdx.x < EXL3_NSTRIP) {
                const unsigned long long wr =
                    (unsigned long long)ws_row_mul * s_slot[m] + ws_row_add;
                float sum = 0.0f;
                for (int p = 0; p < S; ++p) sum += ws[(wr * S + p) * N + n0 + threadIdx.x];
                s_y[m][threadIdx.x] = sum;
            }
        }
        if (threadIdx.x == 0) counters[n_block] = 0;  // re-arm for next logical strip
        __syncthreads();
    }

    if (warp == 0) {
#pragma unroll
        for (int m = 0; m < MROW; ++m) {
            if (m >= (int)M) break;  // warp-uniform: had128's shuffles stay full
            const int nb = n0 + lane * 4;
            float h0 = s_y[m][lane * 4 + 0];
            float h1 = s_y[m][lane * 4 + 1];
            float h2 = s_y[m][lane * 4 + 2];
            float h3 = s_y[m][lane * 4 + 3];
            exl3_had128(h0, h1, h2, h3, lane);
            __nv_bfloat16* dst = C + (unsigned long long)s_slot[m] * N;
            dst[nb + 0] = __float2bfloat16(h0 * EXL3_RSQRT128 * __half2float(svh[nb + 0]));
            dst[nb + 1] = __float2bfloat16(h1 * EXL3_RSQRT128 * __half2float(svh[nb + 1]));
            dst[nb + 2] = __float2bfloat16(h2 * EXL3_RSQRT128 * __half2float(svh[nb + 2]));
            dst[nb + 3] = __float2bfloat16(h3 * EXL3_RSQRT128 * __half2float(svh[nb + 3]));
        }
    }
#undef EXL3_M_AROW
}

// Register-budget hint for the whole m-row family, mirroring the MXFP4 twin's
// `MOE_M_LB`. The m-row arm carries MROW accumulator chains and is
// load-latency bound, so it wants registers to keep loads in flight; smem
// already caps it at 3 CTAs/SM at MROW=6, so "(256, 2)" is the honest floor
// and lets ptxas spend up to 128 registers where that pays. Scheduling hint
// only — no op changes, so the exact-GEMV law is untouched.
#define EXL3_MROW_LB __launch_bounds__(EXL3_BLOCK, 2)

// Fused gate+up over all routed slots of the whole verify block.
//   grid = (N/128, SPLIT_K, 2*num_tokens*top_k), z = 2*slot + proj
//   ws   >= 2*num_tokens*top_k * SPLIT_K * N floats
//   counters >= 2*num_tokens*top_k * (N/128) ints, zero before the FIRST launch
#define EXL3_MROW_GATE_UP_ENTRY(NAME, MROW_)                                                 \
    extern "C" __global__ void EXL3_MROW_LB NAME(                                            \
        const __nv_bfloat16* __restrict__ A,                    /* [num_tokens, K] */        \
        const unsigned long long* __restrict__ gate_trellis_tab,                             \
        const unsigned long long* __restrict__ gate_suh_tab,                                 \
        const unsigned long long* __restrict__ gate_svh_tab,                                 \
        const unsigned long long* __restrict__ up_trellis_tab,                               \
        const unsigned long long* __restrict__ up_suh_tab,                                   \
        const unsigned long long* __restrict__ up_svh_tab,                                   \
        const unsigned int* __restrict__ indices,               /* [num_tokens*top_k] */     \
        __nv_bfloat16* __restrict__ gate_out,                   /* [num_tokens*top_k, N] */  \
        __nv_bfloat16* __restrict__ up_out,                                                  \
        float* __restrict__ ws, int* __restrict__ counters, unsigned int N, unsigned int K,  \
        unsigned int top_k, unsigned int num_tokens, unsigned int bits) {                    \
        EXL3_REQUIRE_BITS(bits);                                                              \
        const unsigned int total_routed = num_tokens * top_k;                                \
        const unsigned int group = blockIdx.z;                                               \
        const unsigned int y = group >> 1;                                                   \
        const unsigned int proj = group & 1u; /* 0 = gate, 1 = up */                         \
        unsigned int M = 0;                                                                  \
        const unsigned int* slots = exl3_mrow_gather<(MROW_)>(indices, y, total_routed, M);  \
        if (!slots) return; /* duplicate expert: the leader covers this slot */              \
        const unsigned int e = indices[y];                                                   \
        const unsigned long long* tt = proj ? up_trellis_tab : gate_trellis_tab;             \
        const unsigned long long* su = proj ? up_suh_tab : gate_suh_tab;                     \
        const unsigned long long* sv = proj ? up_svh_tab : gate_svh_tab;                     \
        exl3_gemv_mrow_body<(MROW_), true>(                                                  \
            A, (const unsigned short*)tt[e], (const __half*)su[e], (const __half*)sv[e],     \
            proj ? up_out : gate_out, ws, counters + group * (N >> 7), slots, M,             \
            blockIdx.x, blockIdx.y, gridDim.y, 2u, proj, top_k, N, K, EXL3_BIT_WIDTH(bits)); \
    }

// Fused down over all routed slots. Slot s consumes the SwiGLU activation row
// `act + s*K` (K = intermediate size) and writes `down_out + s*N`.
//   grid = (N/128, SPLIT_K, num_tokens*top_k), z = slot
#define EXL3_MROW_DOWN_ENTRY(NAME, MROW_)                                                    \
    extern "C" __global__ void EXL3_MROW_LB NAME(                                            \
        const __nv_bfloat16* __restrict__ act,                  /* [num_tokens*top_k, K] */  \
        const unsigned long long* __restrict__ trellis_tab,                                  \
        const unsigned long long* __restrict__ suh_tab,                                      \
        const unsigned long long* __restrict__ svh_tab,                                      \
        const unsigned int* __restrict__ indices,                                            \
        __nv_bfloat16* __restrict__ down_out,                   /* [num_tokens*top_k, N] */  \
        float* __restrict__ ws, int* __restrict__ counters, unsigned int N, unsigned int K,  \
        unsigned int top_k, unsigned int num_tokens, unsigned int bits) {                    \
        EXL3_REQUIRE_BITS(bits);                                                              \
        const unsigned int total_routed = num_tokens * top_k;                                \
        const unsigned int y = blockIdx.z;                                                   \
        unsigned int M = 0;                                                                  \
        const unsigned int* slots = exl3_mrow_gather<(MROW_)>(indices, y, total_routed, M);  \
        if (!slots) return;                                                                  \
        const unsigned int e = indices[y];                                                   \
        exl3_gemv_mrow_body<(MROW_), false>(                                                 \
            act, (const unsigned short*)trellis_tab[e], (const __half*)suh_tab[e],           \
            (const __half*)svh_tab[e], down_out, ws, counters + y * (N >> 7), slots, M,      \
            blockIdx.x, blockIdx.y, gridDim.y, 1u, 0u, top_k, N, K, EXL3_BIT_WIDTH(bits));   \
    }

// The ladder. An MROW=R entry is correct for ANY num_tokens <= R (an expert can
// be duplicated at most num_tokens times, since a token's top-k ids are
// distinct), so the host picks the SMALLEST rung >= num_tokens and never over-
// provisions the accumulator array. Going the other way is a CORRECTNESS bug,
// not a slowdown: a gather of more than MROW slots silently drops the tail and
// those rows come out unwritten.
//
// MROW=1 exists solely as the microtest's bit-exactness reference against the
// shipping `exl3_gemv_m1_fused_*` pair (same discipline as the MXFP4
// `_m1v2s4` entries).
EXL3_MROW_GATE_UP_ENTRY(exl3_gemv_mrow_fused_gate_up_m1, 1)
EXL3_MROW_GATE_UP_ENTRY(exl3_gemv_mrow_fused_gate_up_m2, 2)
EXL3_MROW_GATE_UP_ENTRY(exl3_gemv_mrow_fused_gate_up_m4, 4)
EXL3_MROW_GATE_UP_ENTRY(exl3_gemv_mrow_fused_gate_up_m6, 6)
EXL3_MROW_GATE_UP_ENTRY(exl3_gemv_mrow_fused_gate_up_m8, 8)
EXL3_MROW_GATE_UP_ENTRY(exl3_gemv_mrow_fused_gate_up_m16, 16)

EXL3_MROW_DOWN_ENTRY(exl3_gemv_mrow_fused_down_m1, 1)
EXL3_MROW_DOWN_ENTRY(exl3_gemv_mrow_fused_down_m2, 2)
EXL3_MROW_DOWN_ENTRY(exl3_gemv_mrow_fused_down_m4, 4)
EXL3_MROW_DOWN_ENTRY(exl3_gemv_mrow_fused_down_m6, 6)
EXL3_MROW_DOWN_ENTRY(exl3_gemv_mrow_fused_down_m8, 8)
EXL3_MROW_DOWN_ENTRY(exl3_gemv_mrow_fused_down_m16, 16)

// ---------------------------------------------------------------------------
// Device-built persistent M6 worklist (experimental production opt-in).
//
// The regular m-row grid launches one z-group per routed slot and lets each
// duplicate stage + scan the route before exiting. This builder emits one
// 32-byte item per distinct expert. The fixed 96-CTA consumers grid-stride
// logical (work, projection, split, N-strip) tasks and invoke the exact same
// m-row body. No host sync or device->host count copy is required.
// ---------------------------------------------------------------------------

struct __align__(16) Exl3PersistentM6Work {
    unsigned int expert;
    unsigned int count;
    unsigned int slots[6];
};
static_assert(sizeof(Exl3PersistentM6Work) == 32,
              "EXL3 persistent work record must be 32 bytes");

struct __align__(16) Exl3PersistentM6State {
    unsigned int count;
    unsigned int status;
    unsigned int reserved0;
    unsigned int reserved1;
};
static_assert(sizeof(Exl3PersistentM6State) == 16,
              "EXL3 persistent state must be 16 bytes");

enum Exl3PersistentStatus : unsigned int {
    EXL3_PERSIST_OK = 0,
    EXL3_PERSIST_BAD_SHAPE = 1,
    EXL3_PERSIST_BAD_EXPERT = 2,
    EXL3_PERSIST_DUPLICATE_ROW_EXPERT = 3,
    EXL3_PERSIST_CAPACITY = 4,
};

// One serial lane is intentional: the route has at most 6*32 entries. For the
// accepted M6/top-6/capacity>=36 shape, deterministic first-seen ordering
// matches `persistent_work::try_exl3_worklist`. The host oracle additionally
// models smaller capacities, which this fixed-shape device entry rejects.
// Same-stream launch ordering publishes records and zeroed queues to consumers.
extern "C" __global__ void exl3_build_m6_worklist(
    const unsigned int* __restrict__ indices,
    Exl3PersistentM6Work* __restrict__ worklist,
    Exl3PersistentM6State* __restrict__ state,
    unsigned int num_tokens, unsigned int top_k,
    unsigned int num_experts, unsigned int capacity) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
        threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) return;
    state->count = 0;
    state->status = EXL3_PERSIST_OK;
    state->reserved0 = 0;
    state->reserved1 = 0;
    if (gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1 ||
        blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 ||
        num_tokens != 6 || top_k != 6 || capacity < 36) {
        state->status = EXL3_PERSIST_BAD_SHAPE;
        return;
    }

    const unsigned int total = num_tokens * top_k;
    // Validate the complete route before writing any record. A late duplicate
    // can otherwise make an early leader gather seven occurrences and cross
    // the fixed slots[6] record boundary before the duplicate row is visited.
    for (unsigned int y = 0; y < total; ++y) {
        const unsigned int expert = indices[y];
        if (expert >= num_experts) {
            state->status = EXL3_PERSIST_BAD_EXPERT;
            return;
        }
        const unsigned int row_begin = (y / top_k) * top_k;
        for (unsigned int prior = row_begin; prior < y; ++prior) {
            if (indices[prior] == expert) {
                state->status = EXL3_PERSIST_DUPLICATE_ROW_EXPERT;
                return;
            }
        }
    }

    unsigned int emitted = 0;
    for (unsigned int y = 0; y < total; ++y) {
        const unsigned int expert = indices[y];
        bool leader = true;
        for (unsigned int prior = 0; prior < y; ++prior) {
            if (indices[prior] == expert) {
                leader = false;
                break;
            }
        }
        if (!leader) continue;
        if (emitted >= capacity) {
            state->status = EXL3_PERSIST_CAPACITY;
            return;
        }

        Exl3PersistentM6Work& work = worklist[emitted];
        work.expert = expert;
        unsigned int count = 0;
        for (unsigned int slot = y; slot < total; ++slot) {
            if (indices[slot] != expert) continue;
            if (count == 6) {
                state->count = 0;
                state->status = EXL3_PERSIST_DUPLICATE_ROW_EXPERT;
                return;
            }
            work.slots[count++] = slot;
        }
        work.count = count;
        for (unsigned int row = count; row < 6; ++row) work.slots[row] = work.slots[0];
        ++emitted;
    }
    state->count = emitted;
}

template <unsigned int WIDTH>
__device__ __forceinline__ void exl3_zero_m6_routed(__nv_bfloat16* out) {
    const unsigned int thread_linear =
        threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    const unsigned int block_linear =
        blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const unsigned int total_blocks = gridDim.x * gridDim.y * gridDim.z;
    const size_t first = (size_t)block_linear * threads_per_block + thread_linear;
    const size_t stride = (size_t)total_blocks * threads_per_block;
    for (size_t index = first; index < 36u * WIDTH; index += stride) {
        out[index] = __float2bfloat16(0.0f);
    }
}

extern "C" __global__ void EXL3_MROW_LB exl3_gemv_mrow_persistent_gate_up_m6(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_trellis_tab,
    const unsigned long long* __restrict__ gate_suh_tab,
    const unsigned long long* __restrict__ gate_svh_tab,
    const unsigned long long* __restrict__ up_trellis_tab,
    const unsigned long long* __restrict__ up_suh_tab,
    const unsigned long long* __restrict__ up_svh_tab,
    const Exl3PersistentM6Work* __restrict__ worklist,
    Exl3PersistentM6State* __restrict__ state,
    __nv_bfloat16* __restrict__ gate_out,
    __nv_bfloat16* __restrict__ up_out,
    float* __restrict__ ws, int* __restrict__ counters,
    unsigned int N, unsigned int K, unsigned int top_k,
    unsigned int splits, unsigned int bits) {
    bool bits_valid = bits == 2 || bits == 3;
#if EXL3_FIXED_BITS
    bits_valid = bits == EXL3_FIXED_BITS;
#endif
    const bool launch_valid =
        gridDim.x == 96 && gridDim.y == 1 && gridDim.z == 1 &&
        blockDim.x == 256 && blockDim.y == 1 && blockDim.z == 1 &&
        blockIdx.y == 0 && blockIdx.z == 0 && N == 2048 && K == 4096 &&
        top_k == 6 && splits != 0 && splits <= 12 && bits_valid;
    const bool state_valid = state->status == EXL3_PERSIST_OK &&
                             state->count != 0 && state->count <= 36;
    if (!launch_valid || !state_valid) {
        exl3_zero_m6_routed<2048>(gate_out);
        exl3_zero_m6_routed<2048>(up_out);
        return;
    }
    __shared__ Exl3PersistentM6Work work;
    const unsigned int n_blocks = N >> 7;
    const unsigned int total_tasks = state->count * 2u * splits * n_blocks;
    if (total_tasks == 0) return;

    for (unsigned int logical = blockIdx.x; logical < total_tasks; logical += gridDim.x) {
        unsigned int q = logical;
        const unsigned int n_block = q % n_blocks;
        q /= n_blocks;
        const unsigned int split = q % splits;
        q /= splits;
        const unsigned int proj = q & 1u;
        const unsigned int work_index = q >> 1;
        if (threadIdx.x < 8) ((unsigned int*)&work)[threadIdx.x] =
            ((const unsigned int*)&worklist[work_index])[threadIdx.x];
        __syncthreads();

        const unsigned int e = work.expert;
        const unsigned int group = 2u * work_index + proj;
        const unsigned long long* tt = proj ? up_trellis_tab : gate_trellis_tab;
        const unsigned long long* su = proj ? up_suh_tab : gate_suh_tab;
        const unsigned long long* sv = proj ? up_svh_tab : gate_svh_tab;
        exl3_gemv_mrow_body<6, true>(
            A, (const unsigned short*)tt[e], (const __half*)su[e], (const __half*)sv[e],
            proj ? up_out : gate_out, ws, counters + group * n_blocks,
            work.slots, work.count, n_block, split, splits, 2u, proj,
            top_k, N, K, EXL3_BIT_WIDTH(bits));
        __syncthreads();
    }
}

extern "C" __global__ void EXL3_MROW_LB exl3_gemv_mrow_persistent_down_m6(
    const __nv_bfloat16* __restrict__ act,
    const unsigned long long* __restrict__ trellis_tab,
    const unsigned long long* __restrict__ suh_tab,
    const unsigned long long* __restrict__ svh_tab,
    const Exl3PersistentM6Work* __restrict__ worklist,
    Exl3PersistentM6State* __restrict__ state,
    __nv_bfloat16* __restrict__ down_out,
    float* __restrict__ ws, int* __restrict__ counters,
    unsigned int N, unsigned int K, unsigned int top_k,
    unsigned int splits, unsigned int bits) {
    bool bits_valid = bits == 2 || bits == 3;
#if EXL3_FIXED_BITS
    bits_valid = bits == EXL3_FIXED_BITS;
#endif
    const bool launch_valid =
        gridDim.x == 96 && gridDim.y == 1 && gridDim.z == 1 &&
        blockDim.x == 256 && blockDim.y == 1 && blockDim.z == 1 &&
        blockIdx.y == 0 && blockIdx.z == 0 && N == 4096 && K == 2048 &&
        top_k == 6 && splits != 0 && splits <= 12 && bits_valid;
    const bool state_valid = state->status == EXL3_PERSIST_OK &&
                             state->count != 0 && state->count <= 36;
    if (!launch_valid || !state_valid) {
        exl3_zero_m6_routed<4096>(down_out);
        return;
    }
    __shared__ Exl3PersistentM6Work work;
    const unsigned int n_blocks = N >> 7;
    const unsigned int total_tasks = state->count * splits * n_blocks;
    if (total_tasks == 0) return;

    for (unsigned int logical = blockIdx.x; logical < total_tasks; logical += gridDim.x) {
        unsigned int q = logical;
        const unsigned int n_block = q % n_blocks;
        q /= n_blocks;
        const unsigned int split = q % splits;
        const unsigned int work_index = q / splits;
        if (threadIdx.x < 8) ((unsigned int*)&work)[threadIdx.x] =
            ((const unsigned int*)&worklist[work_index])[threadIdx.x];
        __syncthreads();

        const unsigned int e = work.expert;
        exl3_gemv_mrow_body<6, false>(
            act, (const unsigned short*)trellis_tab[e], (const __half*)suh_tab[e],
            (const __half*)svh_tab[e], down_out, ws, counters + work_index * n_blocks,
            work.slots, work.count, n_block, split, splits, 1u, 0u,
            top_k, N, K, EXL3_BIT_WIDTH(bits));
        __syncthreads();
    }
}

// The M16 record is intentionally a distinct ABI.  Keeping the M6 entry
// points and 32-byte stride unchanged lets previously qualified M6 captures
// remain valid while the DFlash2 16-row arm uses the wider slot vector.
struct __align__(16) Exl3PersistentM16Work {
    unsigned int expert;
    unsigned int count;
    unsigned int slots[16];
    unsigned int reserved[2];
};
static_assert(sizeof(Exl3PersistentM16Work) == 80,
              "EXL3 M16 persistent work record must be 80 bytes");

extern "C" __global__ void exl3_build_m16_worklist(
    const unsigned int* __restrict__ indices,
    Exl3PersistentM16Work* __restrict__ worklist,
    Exl3PersistentM6State* __restrict__ state,
    unsigned int num_tokens, unsigned int top_k,
    unsigned int num_experts, unsigned int capacity) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
        threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) return;
    state->count = 0;
    state->status = EXL3_PERSIST_OK;
    state->reserved0 = 0;
    state->reserved1 = 0;
    if (gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1 ||
        blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 ||
        num_tokens != 16 || top_k != 6 || capacity < 96) {
        state->status = EXL3_PERSIST_BAD_SHAPE;
        return;
    }

    const unsigned int total = num_tokens * top_k;
    // Validate the complete route before writing any record.  In particular,
    // reject a duplicate in the final row before an earlier leader could
    // overrun slots[16].
    for (unsigned int y = 0; y < total; ++y) {
        const unsigned int expert = indices[y];
        if (expert >= num_experts) {
            state->status = EXL3_PERSIST_BAD_EXPERT;
            return;
        }
        const unsigned int row_begin = (y / top_k) * top_k;
        for (unsigned int prior = row_begin; prior < y; ++prior) {
            if (indices[prior] == expert) {
                state->status = EXL3_PERSIST_DUPLICATE_ROW_EXPERT;
                return;
            }
        }
    }

    unsigned int emitted = 0;
    for (unsigned int y = 0; y < total; ++y) {
        const unsigned int expert = indices[y];
        bool leader = true;
        for (unsigned int prior = 0; prior < y; ++prior) {
            if (indices[prior] == expert) {
                leader = false;
                break;
            }
        }
        if (!leader) continue;
        if (emitted >= capacity) {
            state->status = EXL3_PERSIST_CAPACITY;
            return;
        }

        Exl3PersistentM16Work& work = worklist[emitted];
        work.expert = expert;
        unsigned int count = 0;
        for (unsigned int slot = y; slot < total; ++slot) {
            if (indices[slot] != expert) continue;
            if (count == 16) {
                state->count = 0;
                state->status = EXL3_PERSIST_DUPLICATE_ROW_EXPERT;
                return;
            }
            work.slots[count++] = slot;
        }
        work.count = count;
        for (unsigned int row = count; row < 16; ++row) work.slots[row] = work.slots[0];
        work.reserved[0] = 0;
        work.reserved[1] = 0;
        ++emitted;
    }
    state->count = emitted;
}

template <unsigned int ROWS, unsigned int WIDTH>
__device__ __forceinline__ void exl3_zero_routed(__nv_bfloat16* out) {
    const unsigned int thread_linear =
        threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    const unsigned int block_linear =
        blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const unsigned int total_blocks = gridDim.x * gridDim.y * gridDim.z;
    const size_t first = (size_t)block_linear * threads_per_block + thread_linear;
    const size_t stride = (size_t)total_blocks * threads_per_block;
    for (size_t index = first; index < (size_t)(ROWS * 6u) * WIDTH; index += stride) {
        out[index] = __float2bfloat16(0.0f);
    }
}

extern "C" __global__ void EXL3_MROW_LB exl3_gemv_mrow_persistent_gate_up_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_trellis_tab,
    const unsigned long long* __restrict__ gate_suh_tab,
    const unsigned long long* __restrict__ gate_svh_tab,
    const unsigned long long* __restrict__ up_trellis_tab,
    const unsigned long long* __restrict__ up_suh_tab,
    const unsigned long long* __restrict__ up_svh_tab,
    const Exl3PersistentM16Work* __restrict__ worklist,
    Exl3PersistentM6State* __restrict__ state,
    __nv_bfloat16* __restrict__ gate_out,
    __nv_bfloat16* __restrict__ up_out,
    float* __restrict__ ws, int* __restrict__ counters,
    unsigned int N, unsigned int K, unsigned int top_k,
    unsigned int splits, unsigned int bits) {
    bool bits_valid = bits == 2 || bits == 3;
#if EXL3_FIXED_BITS
    bits_valid = bits == EXL3_FIXED_BITS;
#endif
    const bool launch_valid =
        gridDim.x == 96 && gridDim.y == 1 && gridDim.z == 1 &&
        blockDim.x == 256 && blockDim.y == 1 && blockDim.z == 1 &&
        blockIdx.y == 0 && blockIdx.z == 0 && N == 2048 && K == 4096 &&
        top_k == 6 && splits != 0 && splits <= 12 && bits_valid;
    const bool state_valid = state->status == EXL3_PERSIST_OK &&
                             state->count != 0 && state->count <= 96;
    if (!launch_valid || !state_valid) {
        exl3_zero_routed<16, 2048>(gate_out);
        exl3_zero_routed<16, 2048>(up_out);
        return;
    }
    __shared__ Exl3PersistentM16Work work;
    const unsigned int n_blocks = N >> 7;
    const unsigned int total_tasks = state->count * 2u * splits * n_blocks;
    if (total_tasks == 0) return;

    for (unsigned int logical = blockIdx.x; logical < total_tasks; logical += gridDim.x) {
        unsigned int q = logical;
        const unsigned int n_block = q % n_blocks;
        q /= n_blocks;
        const unsigned int split = q % splits;
        q /= splits;
        const unsigned int proj = q & 1u;
        const unsigned int work_index = q >> 1;
        if (threadIdx.x < 20) ((unsigned int*)&work)[threadIdx.x] =
            ((const unsigned int*)&worklist[work_index])[threadIdx.x];
        __syncthreads();

        const unsigned int e = work.expert;
        const unsigned int group = 2u * work_index + proj;
        const unsigned long long* tt = proj ? up_trellis_tab : gate_trellis_tab;
        const unsigned long long* su = proj ? up_suh_tab : gate_suh_tab;
        const unsigned long long* sv = proj ? up_svh_tab : gate_svh_tab;
        exl3_gemv_mrow_body<16, true>(
            A, (const unsigned short*)tt[e], (const __half*)su[e], (const __half*)sv[e],
            proj ? up_out : gate_out, ws, counters + group * n_blocks,
            work.slots, work.count, n_block, split, splits, 2u, proj,
            top_k, N, K, EXL3_BIT_WIDTH(bits));
        __syncthreads();
    }
}

extern "C" __global__ void EXL3_MROW_LB exl3_gemv_mrow_persistent_down_m16(
    const __nv_bfloat16* __restrict__ act,
    const unsigned long long* __restrict__ trellis_tab,
    const unsigned long long* __restrict__ suh_tab,
    const unsigned long long* __restrict__ svh_tab,
    const Exl3PersistentM16Work* __restrict__ worklist,
    Exl3PersistentM6State* __restrict__ state,
    __nv_bfloat16* __restrict__ down_out,
    float* __restrict__ ws, int* __restrict__ counters,
    unsigned int N, unsigned int K, unsigned int top_k,
    unsigned int splits, unsigned int bits) {
    bool bits_valid = bits == 2 || bits == 3;
#if EXL3_FIXED_BITS
    bits_valid = bits == EXL3_FIXED_BITS;
#endif
    const bool launch_valid =
        gridDim.x == 96 && gridDim.y == 1 && gridDim.z == 1 &&
        blockDim.x == 256 && blockDim.y == 1 && blockDim.z == 1 &&
        blockIdx.y == 0 && blockIdx.z == 0 && N == 4096 && K == 2048 &&
        top_k == 6 && splits != 0 && splits <= 12 && bits_valid;
    const bool state_valid = state->status == EXL3_PERSIST_OK &&
                             state->count != 0 && state->count <= 96;
    if (!launch_valid || !state_valid) {
        exl3_zero_routed<16, 4096>(down_out);
        return;
    }
    __shared__ Exl3PersistentM16Work work;
    const unsigned int n_blocks = N >> 7;
    const unsigned int total_tasks = state->count * splits * n_blocks;
    if (total_tasks == 0) return;

    for (unsigned int logical = blockIdx.x; logical < total_tasks; logical += gridDim.x) {
        unsigned int q = logical;
        const unsigned int n_block = q % n_blocks;
        q /= n_blocks;
        const unsigned int split = q % splits;
        const unsigned int work_index = q / splits;
        if (threadIdx.x < 20) ((unsigned int*)&work)[threadIdx.x] =
            ((const unsigned int*)&worklist[work_index])[threadIdx.x];
        __syncthreads();

        const unsigned int e = work.expert;
        exl3_gemv_mrow_body<16, false>(
            act, (const unsigned short*)trellis_tab[e], (const __half*)suh_tab[e],
            (const __half*)svh_tab[e], down_out, ws, counters + work_index * n_blocks,
            work.slots, work.count, n_block, split, splits, 1u, 0u,
            top_k, N, K, EXL3_BIT_WIDTH(bits));
        __syncthreads();
    }
}

#undef EXL3_MROW_GATE_UP_ENTRY
#undef EXL3_MROW_DOWN_ENTRY

// ---------------------------------------------------------------------------
// exl3_dequant_dump: debug oracle — decode every tile and store the raw fp16
// weights as W[n][k] row-major [N, K] (NO Hadamard / suh / svh applied). The
// microtest byte-compares this against the CPU reference decode: the gate is
// bitdiff == 0.
//
//   grid = (N/16, K/16, 1), block = (32, 1, 1)  — one warp per tile.
// ---------------------------------------------------------------------------

extern "C" __global__ void exl3_dequant_dump(const unsigned short* __restrict__ trellis,
                                             __half* __restrict__ Wout,  // [N, K]
                                             unsigned int N, unsigned int K,
                                             unsigned int bits) {
    const int nb = blockIdx.x;
    const int kb = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const unsigned int* tile =
        (const unsigned int*)(trellis + ((size_t)kb * (N >> 4) + nb) * (16 * bits));
    Exl3LaneGeom g = exl3_lane_geom(lane, bits);
    __half2 d01, d23, d45, d67;
    exl3_dq8(tile, g, d01, d23, d45, d67, bits);
    size_t krow = (size_t)kb * 16 + 2 * (lane & 3);
    size_t na = (size_t)nb * 16 + (lane >> 2);
    size_t nc = na + 8;
    __half* wa = Wout + na * K + krow;
    __half* wc = Wout + nc * K + krow;
    wa[0] = __low2half(d01);   // s=0: k+0
    wa[1] = __high2half(d01);  // s=1: k+1
    wa[8] = __low2half(d23);   // s=2: k+8
    wa[9] = __high2half(d23);  // s=3: k+9
    wc[0] = __low2half(d45);   // s=4
    wc[1] = __high2half(d45);  // s=5
    wc[8] = __low2half(d67);   // s=6
    wc[9] = __high2half(d67);  // s=7
}

// ---------------------------------------------------------------------------
// P1 prefill kernels (plan §3 / exl3-gemv.md §6): scratch-dequant + M-row
// activation rotations. The trellis stores W in the ROTATED space; instead of
// baking rotations into the scratch weights, the raw decoded W feeds the
// existing BF16 grouped GEMM and the rotations ride on the ACTIVATIONS —
// exactly the composition the M=1 GEMV applies (verified f64 oracle,
// exl3_gemv_microtest.rs):
//
//   x'  = H128( diag(suh_e) · x ) / sqrt(128)     (pre,  along K, per row)
//   y0  = W_decoded · x'                          (BF16 grouped GEMM)
//   y   = diag(svh_e) · H128( y0 ) / sqrt(128)    (post, along N, per row)
//
// suh/svh are PER EXPERT, so a token routed to k experts needs k different
// pre-rotations: the pre kernel writes the EXPANDED sorted-layout activation
// (one row per (token, slot) pair, gathered via sorted_token_ids), and the
// grouped GEMM then runs with sorted_token_ids = NULL (identity gather).
// Both kernels reuse exl3_had128 — one warp per aligned 128-chunk, 4 fp32
// per lane, the SAME op order as the m1 GEMV's output pass.
// ---------------------------------------------------------------------------

#define EXL3_HROW_WARPS 8  // 128-chunks per 256-thread block

__device__ __forceinline__ void exl3_h128_pre_values4(
    float a0, float a1, float a2, float a3,
    const __half* __restrict__ suh, unsigned int lane,
    float& h0, float& h1, float& h2, float& h3) {
    h0 = a0 * __half2float(suh[4 * lane + 0]);
    h1 = a1 * __half2float(suh[4 * lane + 1]);
    h2 = a2 * __half2float(suh[4 * lane + 2]);
    h3 = a3 * __half2float(suh[4 * lane + 3]);
    exl3_had128(h0, h1, h2, h3, lane);
}

__device__ __forceinline__ void exl3_h128_pre_transform4(
    float a0, float a1, float a2, float a3,
    const __half* __restrict__ suh, unsigned int lane,
    __nv_bfloat16* __restrict__ out) {
    float h0, h1, h2, h3;
    exl3_h128_pre_values4(a0, a1, a2, a3, suh, lane, h0, h1, h2, h3);
    out[0] = __float2bfloat16(h0 * EXL3_RSQRT128);
    out[1] = __float2bfloat16(h1 * EXL3_RSQRT128);
    out[2] = __float2bfloat16(h2 * EXL3_RSQRT128);
    out[3] = __float2bfloat16(h3 * EXL3_RSQRT128);
}

#if EXL3_HROW_DSV4_FIXED
__device__ __forceinline__ void exl3_h128_pre_transform4_packed(
    float a0, float a1, float a2, float a3,
    const __half* __restrict__ suh, unsigned int lane,
    __nv_bfloat16* __restrict__ out) {
    float h0, h1, h2, h3;
    exl3_h128_pre_values4(a0, a1, a2, a3, suh, lane, h0, h1, h2, h3);
    __nv_bfloat162* packed = reinterpret_cast<__nv_bfloat162*>(out);
    packed[0] = __floats2bfloat162_rn(
        h0 * EXL3_RSQRT128, h1 * EXL3_RSQRT128);
    packed[1] = __floats2bfloat162_rn(
        h2 * EXL3_RSQRT128, h3 * EXL3_RSQRT128);
}
#endif

// x' rows for the grouped GEMM: Aout[row] = H128(suh_e ⊙ A[token]) / sqrt(128).
//   row    = blockIdx.y (expanded sorted-layout row)
//   token  = sorted_token_ids[row] (NULL → identity: token = row)
//   e      = sorted_expert_ids[row]
// In-place legal iff sorted_token_ids == NULL (each warp reads only the
// 128-chunk it overwrites, all in registers). Grid: (rows, ceil(K/1024)) —
// rows on grid.x (2^31 cap; grid.y's 65535 would clip a >10.9K-token chunk).
extern "C" __global__ void exl3_h128_pre_rows(
    const __nv_bfloat16* __restrict__ A,             // [num_tokens, K] token-major
    const int* __restrict__ sorted_token_ids,        // [rows] or NULL
    const int* __restrict__ sorted_expert_ids,       // [rows]
    const unsigned long long* __restrict__ suh_tab,  // [num_experts] → F16 [K]
    __nv_bfloat16* __restrict__ Aout,                // [rows, K] sorted layout
    unsigned int K) {
    const unsigned int row = blockIdx.x;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int chunk = blockIdx.y * EXL3_HROW_WARPS + warp;
    if (chunk * 128 >= K) return;
    const int e = sorted_expert_ids[row];
    const __half* suh = (const __half*)suh_tab[e] + chunk * 128;
    const long long tok = sorted_token_ids ? (long long)sorted_token_ids[row] : (long long)row;
    const __nv_bfloat16* a = A + (unsigned long long)tok * K + chunk * 128 + 4 * lane;
    __nv_bfloat16* o = Aout + (unsigned long long)row * K + chunk * 128 + 4 * lane;
    exl3_h128_pre_transform4(
        __bfloat162float(a[0]), __bfloat162float(a[1]),
        __bfloat162float(a[2]), __bfloat162float(a[3]), suh, lane, o);
}

#if EXL3_HROW_DSV4_FIXED
// Exact DeepSeek-V4 gate/up pre-rotation. Both projections read the same
// expanded token row, so keep that BF16 input in registers and apply the two
// independent expert sign tables before writing distinct sorted-layout rows.
extern "C" __global__ void exl3_h128_pre_dual_rows_h4096(
    const __nv_bfloat16* __restrict__ A,             // [num_tokens, 4096]
    const int* __restrict__ sorted_token_ids,        // [rows] or NULL
    const int* __restrict__ sorted_expert_ids,       // [rows]
    const unsigned long long* __restrict__ gate_suh_tab,
    const unsigned long long* __restrict__ up_suh_tab,
    __nv_bfloat16* __restrict__ gate_out,            // [rows, 4096]
    __nv_bfloat16* __restrict__ up_out,              // [rows, 4096]
    unsigned int K, unsigned int rows) {
    if (K != 4096) return;
    asm volatile(
        "{ .reg .pred p; .reg .u32 n;\n\t"
        "mov.u32 n, %ntid.x; setp.ne.u32 p, n, 256; @p exit;\n\t"
        "mov.u32 n, %ntid.y; setp.ne.u32 p, n, 1; @p exit;\n\t"
        "mov.u32 n, %ntid.z; setp.ne.u32 p, n, 1; @p exit; }");
    if (gridDim.x != rows || gridDim.y != 4 || gridDim.z != 1) return;

    const unsigned int row = blockIdx.x;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int chunk = blockIdx.y * EXL3_HROW_WARPS + warp;
    const int expert = sorted_expert_ids[row];
    const long long token = sorted_token_ids
        ? (long long)sorted_token_ids[row] : (long long)row;
    const __nv_bfloat16* a = A + (unsigned long long)token * 4096
        + chunk * 128 + 4 * lane;
    const float a0 = __bfloat162float(a[0]);
    const float a1 = __bfloat162float(a[1]);
    const float a2 = __bfloat162float(a[2]);
    const float a3 = __bfloat162float(a[3]);
    const __half* gate_suh = (const __half*)gate_suh_tab[expert] + chunk * 128;
    const __half* up_suh = (const __half*)up_suh_tab[expert] + chunk * 128;
    __nv_bfloat16* gate_o = gate_out + (unsigned long long)row * 4096
        + chunk * 128 + 4 * lane;
    __nv_bfloat16* up_o = up_out + (unsigned long long)row * 4096
        + chunk * 128 + 4 * lane;
    exl3_h128_pre_transform4_packed(a0, a1, a2, a3, gate_suh, lane, gate_o);
    exl3_h128_pre_transform4_packed(a0, a1, a2, a3, up_suh, lane, up_o);
}
#endif

// Output pass, IN PLACE over the sorted-layout GEMM result:
//   Y[row] = svh_e ⊙ H128(Y[row]) / sqrt(128),  e = sorted_expert_ids[row].
// Grid: (rows, ceil(N/1024)). In-place safe (warp-private 128-chunk).
extern "C" __global__ void exl3_h128_post_rows(
    __nv_bfloat16* __restrict__ Y,                   // [rows, N] sorted layout
    const int* __restrict__ sorted_expert_ids,       // [rows]
    const unsigned long long* __restrict__ svh_tab,  // [num_experts] → F16 [N]
    unsigned int N) {
    const unsigned int row = blockIdx.x;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int chunk = blockIdx.y * EXL3_HROW_WARPS + warp;
    if (chunk * 128 >= N) return;
    const int e = sorted_expert_ids[row];
    const __half* svh = (const __half*)svh_tab[e] + chunk * 128;
    __nv_bfloat16* y = Y + (unsigned long long)row * N + chunk * 128 + 4 * lane;
    float h0 = __bfloat162float(y[0]);
    float h1 = __bfloat162float(y[1]);
    float h2 = __bfloat162float(y[2]);
    float h3 = __bfloat162float(y[3]);
    exl3_had128(h0, h1, h2, h3, lane);
    y[0] = __float2bfloat16(h0 * EXL3_RSQRT128 * __half2float(svh[4 * lane + 0]));
    y[1] = __float2bfloat16(h1 * EXL3_RSQRT128 * __half2float(svh[4 * lane + 1]));
    y[2] = __float2bfloat16(h2 * EXL3_RSQRT128 * __half2float(svh[4 * lane + 2]));
    y[3] = __float2bfloat16(h3 * EXL3_RSQRT128 * __half2float(svh[4 * lane + 3]));
}

// Fused gate/up output rotations, routed DeepSeek SwiGLU, and down input
// rotation. Explicit register BF16 barriers preserve the legacy stores after
// each post-rotation and after SwiGLU, making this byte-identical to
// post(gate) + post(up) + moe_silu_mul + pre(down) while removing their
// intermediate global writes and rereads.
#define EXL3_SWIGLU_LIMIT 10.0f
extern "C" __global__ void exl3_h128_post_silu_pre_rows(
    __nv_bfloat16* __restrict__ gate,               // [rows, N], output in place
    const __nv_bfloat16* __restrict__ up,            // [rows, N]
    const int* __restrict__ sorted_expert_ids,       // [rows]
    const unsigned long long* __restrict__ gate_svh_tab,
    const unsigned long long* __restrict__ up_svh_tab,
    const unsigned long long* __restrict__ down_suh_tab,
    unsigned int N) {
    const unsigned int row = blockIdx.x;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int chunk = blockIdx.y * EXL3_HROW_WARPS + warp;
    if (chunk * 128 >= N) return;
    const int e = sorted_expert_ids[row];
    const __half* gate_svh = (const __half*)gate_svh_tab[e] + chunk * 128;
    const __half* up_svh = (const __half*)up_svh_tab[e] + chunk * 128;
    const __half* down_suh = (const __half*)down_suh_tab[e] + chunk * 128;
    __nv_bfloat16* g = gate + (unsigned long long)row * N + chunk * 128 + 4 * lane;
    const __nv_bfloat16* u = up + (unsigned long long)row * N + chunk * 128 + 4 * lane;

    float g0 = __bfloat162float(g[0]);
    float g1 = __bfloat162float(g[1]);
    float g2 = __bfloat162float(g[2]);
    float g3 = __bfloat162float(g[3]);
    float u0 = __bfloat162float(u[0]);
    float u1 = __bfloat162float(u[1]);
    float u2 = __bfloat162float(u[2]);
    float u3 = __bfloat162float(u[3]);
    exl3_had128(g0, g1, g2, g3, lane);
    exl3_had128(u0, u1, u2, u3, lane);

    // Preserve the legacy post-kernel stores followed by silu-kernel loads.
    g0 = __bfloat162float(__float2bfloat16(
        g0 * EXL3_RSQRT128 * __half2float(gate_svh[4 * lane + 0])));
    g1 = __bfloat162float(__float2bfloat16(
        g1 * EXL3_RSQRT128 * __half2float(gate_svh[4 * lane + 1])));
    g2 = __bfloat162float(__float2bfloat16(
        g2 * EXL3_RSQRT128 * __half2float(gate_svh[4 * lane + 2])));
    g3 = __bfloat162float(__float2bfloat16(
        g3 * EXL3_RSQRT128 * __half2float(gate_svh[4 * lane + 3])));
    u0 = __bfloat162float(__float2bfloat16(
        u0 * EXL3_RSQRT128 * __half2float(up_svh[4 * lane + 0])));
    u1 = __bfloat162float(__float2bfloat16(
        u1 * EXL3_RSQRT128 * __half2float(up_svh[4 * lane + 1])));
    u2 = __bfloat162float(__float2bfloat16(
        u2 * EXL3_RSQRT128 * __half2float(up_svh[4 * lane + 2])));
    u3 = __bfloat162float(__float2bfloat16(
        u3 * EXL3_RSQRT128 * __half2float(up_svh[4 * lane + 3])));

    g0 = fminf(g0, EXL3_SWIGLU_LIMIT);
    g1 = fminf(g1, EXL3_SWIGLU_LIMIT);
    g2 = fminf(g2, EXL3_SWIGLU_LIMIT);
    g3 = fminf(g3, EXL3_SWIGLU_LIMIT);
    u0 = fminf(fmaxf(u0, -EXL3_SWIGLU_LIMIT), EXL3_SWIGLU_LIMIT);
    u1 = fminf(fmaxf(u1, -EXL3_SWIGLU_LIMIT), EXL3_SWIGLU_LIMIT);
    u2 = fminf(fmaxf(u2, -EXL3_SWIGLU_LIMIT), EXL3_SWIGLU_LIMIT);
    u3 = fminf(fmaxf(u3, -EXL3_SWIGLU_LIMIT), EXL3_SWIGLU_LIMIT);
    // The BF16 values are the exact legacy moe_silu_mul outputs consumed by
    // exl3_h128_pre_rows. Re-expand only after rounding, then apply down suh.
    const __nv_bfloat16 a0 =
        __float2bfloat16(g0 * (1.0f / (1.0f + __expf(-g0))) * u0);
    const __nv_bfloat16 a1 =
        __float2bfloat16(g1 * (1.0f / (1.0f + __expf(-g1))) * u1);
    const __nv_bfloat16 a2 =
        __float2bfloat16(g2 * (1.0f / (1.0f + __expf(-g2))) * u2);
    const __nv_bfloat16 a3 =
        __float2bfloat16(g3 * (1.0f / (1.0f + __expf(-g3))) * u3);
    float d0 = __bfloat162float(a0) * __half2float(down_suh[4 * lane + 0]);
    float d1 = __bfloat162float(a1) * __half2float(down_suh[4 * lane + 1]);
    float d2 = __bfloat162float(a2) * __half2float(down_suh[4 * lane + 2]);
    float d3 = __bfloat162float(a3) * __half2float(down_suh[4 * lane + 3]);
    exl3_had128(d0, d1, d2, d3, lane);
    g[0] = __float2bfloat16(d0 * EXL3_RSQRT128);
    g[1] = __float2bfloat16(d1 * EXL3_RSQRT128);
    g[2] = __float2bfloat16(d2 * EXL3_RSQRT128);
    g[3] = __float2bfloat16(d3 * EXL3_RSQRT128);
}

// Fuse the final down H128/svh pass with indexed weighted unpermute. One warp
// owns one 128-column chunk of one original token and visits routed slots in
// the same top-k order as moe_unpermute_reduce_indexed. Each rotated expert
// value is explicitly rounded to BF16 before weighting, preserving the legacy
// post-kernel store followed by the unpermute-kernel load.
template <unsigned int FIXED_H>
__device__ __forceinline__ void exl3_h128_post_unpermute_chunk(
    const __nv_bfloat16* __restrict__ expert_output,  // [rows, H], raw down GEMM
    const int* __restrict__ token_to_perm,            // [num_tokens, topk]
    const float* __restrict__ topk_weights,           // [num_tokens, topk]
    const int* __restrict__ sorted_expert_ids,        // [rows]
    const unsigned long long* __restrict__ svh_tab,   // [num_experts] -> F16 [H]
    unsigned int token, unsigned int chunk, unsigned int lane,
    unsigned int H, unsigned int topk,
    float& a0, float& a1, float& a2, float& a3) {
    const unsigned int h_extent = FIXED_H ? FIXED_H : H;

    a0 = 0.0f;
    a1 = 0.0f;
    a2 = 0.0f;
    a3 = 0.0f;
    for (unsigned int k = 0; k < topk; ++k) {
        const unsigned int slot = token * topk + k;
        const int perm_row = token_to_perm[slot];
        const float weight = topk_weights[slot];
        const int e = sorted_expert_ids[perm_row];
        const __half* svh = (const __half*)svh_tab[e] + chunk * 128;
        const __nv_bfloat16* y = expert_output +
            (unsigned long long)perm_row * h_extent + chunk * 128 + 4 * lane;
        float h0 = __bfloat162float(y[0]);
        float h1 = __bfloat162float(y[1]);
        float h2 = __bfloat162float(y[2]);
        float h3 = __bfloat162float(y[3]);
        exl3_had128(h0, h1, h2, h3, lane);
        h0 = __bfloat162float(__float2bfloat16(
            h0 * EXL3_RSQRT128 * __half2float(svh[4 * lane + 0])));
        h1 = __bfloat162float(__float2bfloat16(
            h1 * EXL3_RSQRT128 * __half2float(svh[4 * lane + 1])));
        h2 = __bfloat162float(__float2bfloat16(
            h2 * EXL3_RSQRT128 * __half2float(svh[4 * lane + 2])));
        h3 = __bfloat162float(__float2bfloat16(
            h3 * EXL3_RSQRT128 * __half2float(svh[4 * lane + 3])));
        a0 += weight * h0;
        a1 += weight * h1;
        a2 += weight * h2;
        a3 += weight * h3;
    }
}

template <unsigned int FIXED_H>
__device__ __forceinline__ void exl3_h128_post_unpermute_rows_body(
    const __nv_bfloat16* __restrict__ expert_output,  // [rows, H], raw down GEMM
    __nv_bfloat16* __restrict__ output,               // [num_tokens, H]
    const int* __restrict__ token_to_perm,            // [num_tokens, topk]
    const float* __restrict__ topk_weights,           // [num_tokens, topk]
    const int* __restrict__ sorted_expert_ids,        // [rows]
    const unsigned long long* __restrict__ svh_tab,   // [num_experts] -> F16 [H]
    unsigned int H, unsigned int num_tokens, unsigned int topk) {
    const unsigned int token = blockIdx.x;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int chunk = blockIdx.y * EXL3_HROW_WARPS + warp;
    if (FIXED_H == 0 && (token >= num_tokens || chunk * 128 >= H)) return;
    const unsigned int h_extent = FIXED_H ? FIXED_H : H;
    float a0, a1, a2, a3;
    exl3_h128_post_unpermute_chunk<FIXED_H>(
        expert_output, token_to_perm, topk_weights, sorted_expert_ids, svh_tab,
        token, chunk, lane, H, topk, a0, a1, a2, a3);
    __nv_bfloat16* out = output +
        (unsigned long long)token * h_extent + chunk * 128 + 4 * lane;
    out[0] = __float2bfloat16(a0);
    out[1] = __float2bfloat16(a1);
    out[2] = __float2bfloat16(a2);
    out[3] = __float2bfloat16(a3);
}

extern "C" __global__ void exl3_h128_post_unpermute_rows(
    const __nv_bfloat16* __restrict__ expert_output,  // [rows, H], raw down GEMM
    __nv_bfloat16* __restrict__ output,               // [num_tokens, H]
    const int* __restrict__ token_to_perm,            // [num_tokens, topk]
    const float* __restrict__ topk_weights,           // [num_tokens, topk]
    const int* __restrict__ sorted_expert_ids,        // [rows]
    const unsigned long long* __restrict__ svh_tab,   // [num_experts] -> F16 [H]
    unsigned int H, unsigned int num_tokens, unsigned int topk) {
    exl3_h128_post_unpermute_rows_body<0>(
        expert_output, output, token_to_perm, topk_weights,
        sorted_expert_ids, svh_tab, H, num_tokens, topk);
}

#if EXL3_HROW_DSV4_FIXED
extern "C" __global__ void exl3_h128_post_unpermute_rows_h4096(
    const __nv_bfloat16* __restrict__ expert_output,  // [rows, H], raw down GEMM
    __nv_bfloat16* __restrict__ output,               // [num_tokens, H]
    const int* __restrict__ token_to_perm,            // [num_tokens, topk]
    const float* __restrict__ topk_weights,           // [num_tokens, topk]
    const int* __restrict__ sorted_expert_ids,        // [rows]
    const unsigned long long* __restrict__ svh_tab,   // [num_experts] -> F16 [H]
    unsigned int H, unsigned int num_tokens, unsigned int topk) {
    if (H != 4096) return;
    asm volatile(
        "{ .reg .pred p; .reg .u32 n;\n\t"
        "mov.u32 n, %ntid.x; setp.ne.u32 p, n, 256; @p exit;\n\t"
        "mov.u32 n, %ntid.y; setp.ne.u32 p, n, 1; @p exit;\n\t"
        "mov.u32 n, %ntid.z; setp.ne.u32 p, n, 1; @p exit; }");
    if (gridDim.x != num_tokens || gridDim.y != 4 || gridDim.z != 1) return;
    exl3_h128_post_unpermute_rows_body<4096>(
        expert_output, output, token_to_perm, topk_weights,
        sorted_expert_ids, svh_tab, H, num_tokens, topk);
}

// Exact DeepSeek-V4 shared-expert tail. One CTA owns one token so its eight
// warps can first reproduce the standalone gate reduction, then cover all 32
// H128 chunks without materializing/reloading the routed BF16 row.
extern "C" __global__ void exl3_h128_post_unpermute_blend_h4096(
    const __nv_bfloat16* __restrict__ expert_output,  // [rows, 4096], raw down GEMM
    __nv_bfloat16* __restrict__ output,               // [num_tokens, 4096]
    const int* __restrict__ token_to_perm,            // [num_tokens, topk]
    const float* __restrict__ topk_weights,           // [num_tokens, topk]
    const int* __restrict__ sorted_expert_ids,        // [rows]
    const unsigned long long* __restrict__ svh_tab,   // [num_experts] -> F16 [4096]
    const __nv_bfloat16* __restrict__ shared_out,     // [num_tokens, 4096]
    const __nv_bfloat16* __restrict__ normed,         // [num_tokens, 4096]
    const __nv_bfloat16* __restrict__ gate_weight,    // [4096], nullable
    unsigned int H, unsigned int num_tokens, unsigned int topk) {
    if (H != 4096 || topk != 6) return;
    asm volatile(
        "{ .reg .pred p; .reg .u32 n;\n\t"
        "mov.u32 n, %ntid.x; setp.ne.u32 p, n, 256; @p exit;\n\t"
        "mov.u32 n, %ntid.y; setp.ne.u32 p, n, 1; @p exit;\n\t"
        "mov.u32 n, %ntid.z; setp.ne.u32 p, n, 1; @p exit; }");
    if (gridDim.x != num_tokens || gridDim.y != 1 || gridDim.z != 1) return;

    __shared__ float gate_partials[8];
    const unsigned int token = blockIdx.x;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const __nv_bfloat16* my_normed = normed + (unsigned long long)token * 4096;
    const float gate_scalar = atlas_moe_shared_gate_scalar_256(
        my_normed, gate_weight, 4096, gate_partials);

    for (unsigned int chunk = warp; chunk < 32; chunk += 8) {
        float a0, a1, a2, a3;
        exl3_h128_post_unpermute_chunk<4096>(
            expert_output, token_to_perm, topk_weights, sorted_expert_ids, svh_tab,
            token, chunk, lane, H, topk, a0, a1, a2, a3);
        const unsigned long long base =
            (unsigned long long)token * 4096 + chunk * 128 + 4 * lane;
        const float routed0 = atlas_round_bf16_to_f32(a0);
        const float routed1 = atlas_round_bf16_to_f32(a1);
        const float routed2 = atlas_round_bf16_to_f32(a2);
        const float routed3 = atlas_round_bf16_to_f32(a3);
        output[base + 0] = atlas_moe_blend_from_routed_bf16(
            routed0, shared_out[base + 0], gate_scalar);
        output[base + 1] = atlas_moe_blend_from_routed_bf16(
            routed1, shared_out[base + 1], gate_scalar);
        output[base + 2] = atlas_moe_blend_from_routed_bf16(
            routed2, shared_out[base + 2], gate_scalar);
        output[base + 3] = atlas_moe_blend_from_routed_bf16(
            routed3, shared_out[base + 3], gate_scalar);
    }
}
#endif

// Chunked scratch dequant for the grouped BF16 prefill GEMM: decode experts
// [e0, e0+count) into slot-major BF16 scratch (slot z = expert e0+z at
// Wout + z·N·K, layout [N, K] — the exact layout moe_bf16_grouped_gemm's
// pointer table expects). Same per-tile decode as exl3_dequant_dump, with a
// final fp16 → bf16 RN convert. NO rotations applied (they ride on the
// activations, above).
//
//   grid = (N/16, K/16, count), block = (32, 1, 1) — one warp per tile.
extern "C" __global__ void exl3_dequant_chunk_bf16(
    const unsigned long long* __restrict__ trellis_tab,  // [num_experts]
    unsigned int e0, unsigned int count,
    __nv_bfloat16* __restrict__ Wout,  // [count, N, K] slot-major scratch
    unsigned int N, unsigned int K, unsigned int bits) {
    const unsigned int slot = blockIdx.z;
    if (slot >= count) return;
    const int nb = blockIdx.x;
    const int kb = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const unsigned short* trellis = (const unsigned short*)trellis_tab[e0 + slot];
    const unsigned int* tile =
        (const unsigned int*)(trellis + ((size_t)kb * (N >> 4) + nb) * (16 * bits));
    Exl3LaneGeom g = exl3_lane_geom(lane, bits);
    __half2 d01, d23, d45, d67;
    exl3_dq8(tile, g, d01, d23, d45, d67, bits);
    __nv_bfloat16* W = Wout + (unsigned long long)slot * N * K;
    size_t krow = (size_t)kb * 16 + 2 * (lane & 3);
    size_t na = (size_t)nb * 16 + (lane >> 2);
    size_t nc = na + 8;
    __nv_bfloat16* wa = W + na * K + krow;
    __nv_bfloat16* wc = W + nc * K + krow;
    wa[0] = __float2bfloat16(__half2float(__low2half(d01)));   // s=0: k+0
    wa[1] = __float2bfloat16(__half2float(__high2half(d01)));  // s=1: k+1
    wa[8] = __float2bfloat16(__half2float(__low2half(d23)));   // s=2: k+8
    wa[9] = __float2bfloat16(__half2float(__high2half(d23)));  // s=3: k+9
    wc[0] = __float2bfloat16(__half2float(__low2half(d45)));   // s=4
    wc[1] = __float2bfloat16(__half2float(__high2half(d45)));  // s=5
    wc[8] = __float2bfloat16(__half2float(__low2half(d67)));   // s=6
    wc[9] = __float2bfloat16(__half2float(__high2half(d67)));  // s=7
}
