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
//   - Phase 1: block computes x' (fp32) for its K-slice into smem, one
//     128-chunk per warp via the warp-shuffle Hadamard (4 elems/lane).
//   - Phase 2: trellis is streamed through a 2-stage cp.async smem pipeline;
//     each stage is 16 tile-rows x 8 tile-cols = 12 KB, issued as 16-B
//     cp.async per thread (3 per thread per stage) — wide, coalesced,
//     contiguous 768-B runs per tile-row. One warp decodes one 96-B tile per
//     iteration with the dq8 batching (8 weights per lane from two u32 loads).
//   - Phase 3: quad shuffle-reduce, strip partial in smem, then (split 0 of 1,
//     or the LAST split to finish, elected by an atomic counter) applies the
//     output Hadamard-128 + svh and stores bf16. Split partials are combined
//     in fixed split order, so the result is deterministic for a fixed grid.
//
// Constraints: K % 128 == 0, N % 128 == 0, gridDim.y splits K in 128-aligned
// chunks; K-slice per block <= EXL3_MAX_XCHUNKS*128 = 4096 (use gridDim.y > 1
// for larger K). Expert shapes (N=2048 K=4096, N=4096 K=2048) all satisfy this.
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
#define EXL3_NSTRIP 128      // output columns per block (8 tiles of 16)
#define EXL3_STAGE_ROWS 16   // tile-rows per cp.async stage (16*8*96 B = 12 KB)
#define EXL3_MAX_XCHUNKS 32  // fp32 x' smem: up to 4096 K per block K-slice
#define EXL3_RSQRT128 0.088388347648f

// ---------------------------------------------------------------------------
// Decode primitives (port of exl3_dq.cuh / codebook.cuh, bits=3, cb=1)
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned int exl3_fshift(unsigned int b, unsigned int a, int shift) {
    unsigned long long merged = ((unsigned long long)a << 32) | (unsigned long long)b;
    return (unsigned int)(merged >> shift);
}

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

// dq8 (align=4 variant, exl3_dq.cuh): 8 weights per lane from two u32 loads.
// d01=(t0,t1) d23=(t2,t3) d45=(t4,t5) d67=(t6,t7); t = lane*8 + s.
__device__ __forceinline__ void exl3_dq8(const unsigned int* __restrict__ tile,
                                         const Exl3LaneGeom g, __half2& d01, __half2& d23,
                                         __half2& d45, __half2& d67) {
    unsigned int a = tile[g.ia];
    unsigned int b = tile[g.ib];
    unsigned int w7 = exl3_fshift(b, a, g.s2);
    unsigned int w6 = w7 >> EXL3_BITS;
    unsigned int w5 = w6 >> EXL3_BITS;
    unsigned int w4 = w5 >> EXL3_BITS;
    unsigned int w3 = exl3_fshift(b, a, g.s2 + 4 * EXL3_BITS);
    unsigned int w2 = w3 >> EXL3_BITS;
    unsigned int w1 = w2 >> EXL3_BITS;
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
// strip. Each tile-row is a contiguous 768-B run (48 uint4).
__device__ __forceinline__ void exl3_load_stage(uint4* __restrict__ dst,
                                                const uint4* __restrict__ trellis_u4, int r0,
                                                int rows_hi, int n_tiles_row, int nb0) {
    int nrows = rows_hi - r0;
    if (nrows > EXL3_STAGE_ROWS) nrows = EXL3_STAGE_ROWS;
#pragma unroll
    for (int i = 0; i < (EXL3_STAGE_ROWS * 48) / EXL3_BLOCK; ++i) {
        int idx = i * EXL3_BLOCK + (int)threadIdx.x;
        int r = idx / 48;
        int o = idx % 48;
        bool p = r < nrows;
        const uint4* src =
            trellis_u4 + ((size_t)(r0 + (p ? r : 0)) * n_tiles_row + nb0) * 6 + o;
        exl3_cp_async_16(dst + idx, src, p);
    }
    exl3_cp_commit();
}

// ---------------------------------------------------------------------------
// exl3_gemv_m1: C[N] (bf16) = EXL3-decode GEMV of A[K] (bf16).
//
//   grid  = (N/128, SPLIT_K, 1), block = (256, 1, 1)
//   ws    : fp32 [SPLIT_K, N] scratch  (untouched when gridDim.y == 1)
//   counters: int [N/128], must be 0 before FIRST launch; the kernel resets
//             them to 0 on completion, so back-to-back launches are safe.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void exl3_gemv_m1_body(
    const __nv_bfloat16* __restrict__ A,         // [K]
    const unsigned short* __restrict__ trellis,  // [K/16, N/16, 48]
    const __half* __restrict__ suh,              // [K]
    const __half* __restrict__ svh,              // [N]
    __nv_bfloat16* __restrict__ C,               // [N]
    float* __restrict__ ws,                      // [gridDim.y, N]
    int* __restrict__ counters,                  // [N/128]
    unsigned int N, unsigned int K) {
    __shared__ __align__(16) float s_x[EXL3_MAX_XCHUNKS * 128];             // 16 KB
    __shared__ __align__(16) unsigned short s_stage[2][EXL3_STAGE_ROWS * 8 * 48];  // 24 KB
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
    const int nstages = (rows_hi - rows_lo + EXL3_STAGE_ROWS - 1) / EXL3_STAGE_ROWS;

    // Kick off the trellis pipeline before the x' pass so the DRAM stream
    // starts immediately.
    const uint4* trellis_u4 = (const uint4*)trellis;
    exl3_load_stage((uint4*)s_stage[0], trellis_u4, rows_lo, rows_hi, n_tiles_row, nb0);
    if (nstages > 1)
        exl3_load_stage((uint4*)s_stage[1], trellis_u4, rows_lo + EXL3_STAGE_ROWS, rows_hi,
                        n_tiles_row, nb0);

    // ---- Phase 1: x' = H128(diag(suh) . x) / sqrt(128), fp32, into smem ----
    for (int c = c_lo + warp; c < c_hi; c += EXL3_BLOCK / 32) {
        int base = (c << 7) + lane * 4;
        float h0 = __bfloat162float(A[base + 0]) * __half2float(suh[base + 0]);
        float h1 = __bfloat162float(A[base + 1]) * __half2float(suh[base + 1]);
        float h2 = __bfloat162float(A[base + 2]) * __half2float(suh[base + 2]);
        float h3 = __bfloat162float(A[base + 3]) * __half2float(suh[base + 3]);
        exl3_had128(h0, h1, h2, h3, lane);
        int lb = ((c - c_lo) << 7) + lane * 4;
        s_x[lb + 0] = h0 * EXL3_RSQRT128;
        s_x[lb + 1] = h1 * EXL3_RSQRT128;
        s_x[lb + 2] = h2 * EXL3_RSQRT128;
        s_x[lb + 3] = h3 * EXL3_RSQRT128;
    }
    __syncthreads();

    // ---- Phase 2: stream + decode + accumulate ----
    float acc0 = 0.0f, acc1 = 0.0f;  // n = 16*warp + lane/4, and +8
    const Exl3LaneGeom g = exl3_lane_geom(lane);
    const int xk = 2 * (lane & 3);

    for (int s = 0; s < nstages; ++s) {
        if (s + 1 < nstages)
            asm volatile("cp.async.wait_group 1;");
        else
            asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        int r0 = rows_lo + s * EXL3_STAGE_ROWS;
        int nrows = rows_hi - r0;
        if (nrows > EXL3_STAGE_ROWS) nrows = EXL3_STAGE_ROWS;
        const unsigned int* stage32 = (const unsigned int*)s_stage[s & 1];

#pragma unroll 4
        for (int r = 0; r < nrows; ++r) {
            const unsigned int* tile = stage32 + (r * 8 + warp) * EXL3_TILE_U32;
            __half2 d01, d23, d45, d67;
            exl3_dq8(tile, g, d01, d23, d45, d67);
            int xb = ((r0 - rows_lo + r) << 4) + xk;
            float2 xa = *(const float2*)&s_x[xb];      // k, k+1
            float2 xc = *(const float2*)&s_x[xb + 8];  // k+8, k+9
            float2 f;
            f = __half22float2(d01);
            acc0 = __fmaf_rn(f.x, xa.x, acc0);
            acc0 = __fmaf_rn(f.y, xa.y, acc0);
            f = __half22float2(d23);
            acc0 = __fmaf_rn(f.x, xc.x, acc0);
            acc0 = __fmaf_rn(f.y, xc.y, acc0);
            f = __half22float2(d45);
            acc1 = __fmaf_rn(f.x, xa.x, acc1);
            acc1 = __fmaf_rn(f.y, xa.y, acc1);
            f = __half22float2(d67);
            acc1 = __fmaf_rn(f.x, xc.x, acc1);
            acc1 = __fmaf_rn(f.y, xc.y, acc1);
        }
        __syncthreads();
        if (s + 2 < nstages)
            exl3_load_stage((uint4*)s_stage[s & 1], trellis_u4,
                            rows_lo + (s + 2) * EXL3_STAGE_ROWS, rows_hi, n_tiles_row, nb0);
    }

    // ---- Phase 3: reduce + (elected) output Hadamard + svh + store ----
    // Quad reduction: lanes {q, q+1, q+2, q+3} share n = 16*warp + q/... —
    // lanes with equal lane/4 hold the same n over disjoint k.
    acc0 += __shfl_xor_sync(0xffffffffu, acc0, 1);
    acc0 += __shfl_xor_sync(0xffffffffu, acc0, 2);
    acc1 += __shfl_xor_sync(0xffffffffu, acc1, 1);
    acc1 += __shfl_xor_sync(0xffffffffu, acc1, 2);
    if ((lane & 3) == 0) {
        s_y[warp * 16 + (lane >> 2)] = acc0;
        s_y[warp * 16 + (lane >> 2) + 8] = acc1;
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

extern "C" __global__ void __launch_bounds__(EXL3_BLOCK) exl3_gemv_m1(
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

extern "C" __global__ void __launch_bounds__(EXL3_BLOCK) exl3_gemv_m1_idx(
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
