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
//   - Phase 1: block computes x' for one SUPERBLOCK (up to 16 chunks
//     = 2048 k) of its K-slice into smem, one 128-chunk per warp via the
//     warp-shuffle Hadamard (4 elems/lane, fp32 math); refilled per
//     superblock, so the K-slice length is unbounded. x' is STORED as
//     packed __half2 (k, k+1) pairs — 4 KB — feeding the half2 dot path
//     below (round 3; the fp16 store is a quantization of x', part of the
//     documented numerics tier, see docs/kernels/exl3-gemv.md §3b).
//   - Phase 2: trellis is streamed through a 2-stage cp.async smem pipeline;
//     each stage is 8 tile-rows x 8 tile-cols = 6 KB (one 128-k chunk),
//     issued as 16-B cp.async per thread — wide, coalesced, contiguous 768-B
//     runs per tile-row. One warp decodes one 96-B tile per iteration with
//     the dq8 batching (8 weights per lane from two u32 loads). The dot is
//     accumulated in __half2 HFMA2 chains (4 independent chains: {acc0,acc1}
//     x even/odd tile-row), converted to fp32 ONCE PER 128-k CHUNK in fixed
//     order — this removes the 8 HADD2.F32 cvts + 8 FFMAs per tile that the
//     round-2 SASS showed to be 24% of the issue stream (the kernel is
//     warp-issue-bound, ~67 ops per 96-B tile; LDS runs at ~17% of its
//     bandwidth at the DRAM target and is NOT the constraint).
//   - Phase 3: quad shuffle-reduce, strip partial in smem, then (split 0 of 1,
//     or the LAST split to finish, elected by an atomic counter) applies the
//     output Hadamard-128 + svh and stores bf16. Split partials are combined
//     in fixed split order, so the result is deterministic for a fixed grid.
//
// Occupancy: smem = 4 KB x' + 2 x 6 KB stages + 0.5 KB partials ~ 16.5 KB,
// __launch_bounds__(256, 4) -> 4 CTAs/SM (32 warps) on GB10's 100 KB SMs,
// vs 2 CTAs/SM for the first bring-up (42.5 KB). The kernel is issue-latency
// bound, not DRAM-bound, at 2 CTAs/SM — the extra warps hide the dependent
// dequant chains.
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

#define EXL3_BITS 3
#define EXL3_TILE_U32 24  // 96 B per 16x16 tile
#define EXL3_MCG_MULT 0xCBAC1FEDu
#define EXL3_BLOCK 256
#define EXL3_NSTRIP 128     // output columns per block (8 tiles of 16)
#define EXL3_STAGE_ROWS 8   // tile-rows per cp.async stage (8*8*96 B = 6 KB = 1 chunk)
#define EXL3_STAGE_U4 (EXL3_STAGE_ROWS * 8 * 6)  // 16-B copies per stage (384)
#define EXL3_MAX_XCHUNKS 16  // x' smem superblock: 2048 k as __half2 pairs, 4 KB (refilled per superblock)
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

__device__ __forceinline__ Exl3LaneGeom exl3_lane_geom(int lane) {
    int t = lane * 8;
    int b1 = (t + 257) * EXL3_BITS;  // end of first window
    int b0 = b1 - 16;                // start of first window
    int b2 = b1 + 7 * EXL3_BITS;     // end of last window
    int i0 = b0 >> 5;
    int i2 = (b2 - 1) >> 5;
    Exl3LaneGeom g;
    g.ia = i0 % EXL3_TILE_U32;
    g.ib = i2 % EXL3_TILE_U32;
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
                                         __half2& d45, __half2& d67) {
    unsigned int a = tile[g.ia];
    unsigned int b = tile[g.ib];
    unsigned int mlo = __funnelshift_r(b, a, g.s2);  // s2 in {0,8,16,24} < 32
    unsigned int mhi = a >> g.s2;
    unsigned int w7 = mlo;
    unsigned int w5 = __funnelshift_r(mlo, mhi, 2 * EXL3_BITS);
    unsigned int w3 = __funnelshift_r(mlo, mhi, 4 * EXL3_BITS);
    unsigned int w1 = __funnelshift_r(mlo, mhi, 6 * EXL3_BITS);
    unsigned int w6 = w7 >> EXL3_BITS;
    unsigned int w4 = w5 >> EXL3_BITS;
    unsigned int w2 = w3 >> EXL3_BITS;
    unsigned int w0 = w1 >> EXL3_BITS;
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
// L1 — the trellis stream is read exactly once). src-size predication makes
// the copy a zero-fill no-op for tail rows.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void exl3_cp_async_16(void* dst_smem, const void* src_gmem,
                                                 bool pred) {
    unsigned int dst = (unsigned int)__cvta_generic_to_shared(dst_smem);
    int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src_gmem),
                 "r"(src_bytes));
}

__device__ __forceinline__ void exl3_cp_commit() { asm volatile("cp.async.commit_group;"); }

// Issue one stage: tile-rows [r0, r0+EXL3_STAGE_ROWS) of the block's 8-tile
// strip. Each tile-row is a contiguous 768-B run (48 uint4). 384 copies over
// 256 threads: iteration 0 is full, iteration 1 is half-predicated.
__device__ __forceinline__ void exl3_load_stage(uint4* __restrict__ dst,
                                                const uint4* __restrict__ trellis_u4, int r0,
                                                int rows_hi, int n_tiles_row, int nb0) {
    int nrows = rows_hi - r0;
    if (nrows > EXL3_STAGE_ROWS) nrows = EXL3_STAGE_ROWS;
#pragma unroll
    for (int i = 0; i < (EXL3_STAGE_U4 + EXL3_BLOCK - 1) / EXL3_BLOCK; ++i) {
        int idx = i * EXL3_BLOCK + (int)threadIdx.x;
        if (idx >= EXL3_STAGE_U4) break;
        int r = idx / 48;
        int o = idx % 48;
        bool p = r < nrows;
        const uint4* src =
            trellis_u4 + ((size_t)(r0 + (p ? r : 0)) * n_tiles_row + nb0) * 6 + o;
        exl3_cp_async_16(dst + idx, src, p);
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
    unsigned int N, unsigned int K) {
    __shared__ __align__(16) __half2 s_x[EXL3_MAX_XCHUNKS * 64];            // 4 KB
    __shared__ __align__(16) unsigned short s_stage[2][EXL3_STAGE_ROWS * 8 * 48];  // 12 KB
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
    const int rows_hi = c_hi * 8;
    // One stage per 128-k chunk (EXL3_STAGE_ROWS == 8), so every stage is
    // exactly full and nstages == chunks in the slice.
    const int nstages = c_hi - c_lo;

    // Kick off the trellis pipeline before the x' pass so the DRAM stream
    // starts immediately.
    const uint4* trellis_u4 = (const uint4*)trellis;
    exl3_load_stage((uint4*)s_stage[0], trellis_u4, rows_lo, rows_hi, n_tiles_row, nb0);
    if (nstages > 1)
        exl3_load_stage((uint4*)s_stage[1], trellis_u4, rows_lo + EXL3_STAGE_ROWS, rows_hi,
                        n_tiles_row, nb0);

    // ---- Phase 1: x' for the first superblock (up to 2048 k) ----
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
    const Exl3LaneGeom g = exl3_lane_geom(lane);
    const int xkh = lane & 3;  // half2 (k-pair) index within the 16-k tile row

    for (int s = 0; s < nstages; ++s) {
        if (s != 0 && (s & (EXL3_MAX_XCHUNKS - 1)) == 0) {
            // Superblock boundary: refill x' for chunks [c_lo+s, +16). The
            // previous iteration's trailing __syncthreads() ordered all
            // reads of the old superblock before this overwrite; the two
            // in-flight cp.async stages are unaffected (s_stage only).
            int sb0 = c_lo + s;
            int sb1 = sb0 + EXL3_MAX_XCHUNKS;
            if (sb1 > c_hi) sb1 = c_hi;
            exl3_input_pass(A, suh, s_x, sb0, sb1, warp, lane);
            __syncthreads();
        }
        if (s + 1 < nstages)
            asm volatile("cp.async.wait_group 1;");
        else
            asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        const unsigned int* stage32 = (const unsigned int*)s_stage[s & 1];
        const __half2* xrow = s_x + ((s & (EXL3_MAX_XCHUNKS - 1)) << 6) + xkh;

        const __half2 hz = __float2half2_rn(0.0f);
        __half2 hacc0e = hz, hacc0o = hz;  // acc0, even/odd tile-row
        __half2 hacc1e = hz, hacc1o = hz;  // acc1, even/odd tile-row
#pragma unroll
        for (int r = 0; r < EXL3_STAGE_ROWS; ++r) {
            const unsigned int* tile = stage32 + (r * 8 + warp) * EXL3_TILE_U32;
            __half2 d01, d23, d45, d67;
            exl3_dq8(tile, g, d01, d23, d45, d67);
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

        __syncthreads();
        if (s + 2 < nstages)
            exl3_load_stage((uint4*)s_stage[s & 1], trellis_u4,
                            rows_lo + (s + 2) * EXL3_STAGE_ROWS, rows_hi, n_tiles_row, nb0);
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
    unsigned int N, unsigned int K) {
    exl3_gemv_m1_body(A, trellis, suh, svh, C, ws, counters, N, K);
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
    unsigned int N, unsigned int K) {
    const unsigned int e = indices[slot];
    exl3_gemv_m1_body(A, (const unsigned short*)trellis_tab[e],
                      (const __half*)suh_tab[e], (const __half*)svh_tab[e], C, ws,
                      counters, N, K);
}

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
                                             unsigned int N, unsigned int K) {
    const int nb = blockIdx.x;
    const int kb = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const unsigned int* tile =
        (const unsigned int*)(trellis + ((size_t)kb * (N >> 4) + nb) * 48);
    Exl3LaneGeom g = exl3_lane_geom(lane);
    __half2 d01, d23, d45, d67;
    exl3_dq8(tile, g, d01, d23, d45, d67);
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
    float h0 = __bfloat162float(a[0]) * __half2float(suh[4 * lane + 0]);
    float h1 = __bfloat162float(a[1]) * __half2float(suh[4 * lane + 1]);
    float h2 = __bfloat162float(a[2]) * __half2float(suh[4 * lane + 2]);
    float h3 = __bfloat162float(a[3]) * __half2float(suh[4 * lane + 3]);
    exl3_had128(h0, h1, h2, h3, lane);
    o[0] = __float2bfloat16(h0 * EXL3_RSQRT128);
    o[1] = __float2bfloat16(h1 * EXL3_RSQRT128);
    o[2] = __float2bfloat16(h2 * EXL3_RSQRT128);
    o[3] = __float2bfloat16(h3 * EXL3_RSQRT128);
}

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
    unsigned int N, unsigned int K) {
    const unsigned int slot = blockIdx.z;
    if (slot >= count) return;
    const int nb = blockIdx.x;
    const int kb = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const unsigned short* trellis = (const unsigned short*)trellis_tab[e0 + slot];
    const unsigned int* tile =
        (const unsigned int*)(trellis + ((size_t)kb * (N >> 4) + nb) * 48);
    Exl3LaneGeom g = exl3_lane_geom(lane);
    __half2 d01, d23, d45, d67;
    exl3_dq8(tile, g, d01, d23, d45, d67);
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
