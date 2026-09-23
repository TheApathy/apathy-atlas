// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1 dense FP8 linear, fused (no bf16 dequant copy), v3 main loop:
// C[M, N] (bf16) = A[M, K] (bf16) @ dequant(W[N, K])^T, W e4m3 + UE8M0 scales per 32x32 block.
//
// Numerics contract (as dsv41_fp8_gemm.cu v1/v2): every output is ONE fp32 accumulation chain of
// mma.sync.m16n8k16 bf16 steps over k ascending in 16s, no split-K, B = bf16(fp8 * 2^(s-127))
// (exact). Byte-identical to dequant + the no-split-K cuBLASLt bf16 GEMM, and independent of M.
//
// What v3 changes is only the schedule: templated CTA tile / warp grid / k-tile / pipeline depth,
// dynamic shared memory (> 48 KB), and the FP8 tile converted into B fragments in registers.

#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace dsv41_fp8gemm3 {

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], unsigned saddr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(saddr));
}
__device__ __forceinline__ void cp_async16(unsigned saddr, const void* gmem, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(saddr), "l"(gmem), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ uint16_t lds_u16(unsigned saddr) {
    uint16_t v;
    asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(v) : "r"(saddr));
    return v;
}
// Two e4m3 (low byte = element 0) -> bf16x2 of fp8 * sc. fp8 -> f16 is exact (cvt.rn on an exact
// value), f16 -> f32 exact, * 2^k exact, f32 -> bf16 exact (the value has <= 4 significant bits).
__device__ __forceinline__ uint32_t fp8x2_bf16x2(uint16_t two, float sc) {
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(h) : "h"(two));
    const __half2 hh = *reinterpret_cast<const __half2*>(&h);
    const float2 f = __half22float2(hh);
    const __nv_bfloat162 v = __floats2bfloat162_rn(f.x * sc, f.y * sc);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// 2^(s - 127) exactly: the float with exponent field s (s = 0 is the subnormal 2^-127).
__device__ __forceinline__ float ue8m0_scale(uint8_t s) {
    return s ? __int_as_float((int)s << 23) : exp2f(-127.f);
}

// L2-grouped rasterization (numerics-neutral): a 1D grid walks GROUP_M m-tiles down each n-tile
// column before moving right, so the CTAs resident at once share A rows and W columns in L2
// (plain n-fastest order streams all of W from DRAM once per m-tile: wq_b's 84 MB x 16 at M=2048).
constexpr int GROUP_M = 8;
__device__ __forceinline__ void tile_coords(int M, int BM, int BN, int N, int& m0, int& n0) {
    const int pm = (M + BM - 1) / BM, pn = N / BN, pid = blockIdx.x;
    const int per_group = GROUP_M * pn, first = (pid / per_group) * GROUP_M;
    const int gsize = min(pm - first, GROUP_M);
    m0 = (first + (pid % per_group) % gsize) * BM;
    n0 = ((pid % per_group) / gsize) * BN;
}

template <int BM_, int BN_, int BK_, int STAGES_, int WM_, int WN_>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, STAGES = STAGES_, WM = WM_, WN = WN_;
    static constexpr int THREADS = 32 * WM * WN;
    static constexpr int TM = BM / WM, TN = BN / WN;     // warp tile
    static constexpr int MI = TM / 16, NI = TN / 8;      // mma tiles per warp
    static constexpr int ALD = BK + 8;                   // bf16 elements per A smem row
    static constexpr int WLD = BK + 16;                  // bytes per raw FP8 smem row
    static constexpr int A_STAGE = BM * ALD * 2, W_STAGE = BN * WLD;
    static constexpr int SMEM = STAGES * (A_STAGE + W_STAGE);
    static_assert(TM % 16 == 0 && TN % 8 == 0 && BK % 32 == 0, "tile shape");
};

template <class C>
__device__ __forceinline__ void gemm_body(const __nv_bfloat16* __restrict__ A, int lda,
                                          const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int s_ld,
                                          __nv_bfloat16* __restrict__ Cout, int ldc, int M, int N, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned sA = sbase, sW = sbase + C::STAGES * C::A_STAGE;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / C::WN, wn = warp % C::WN;
    int m0, n0;
    tile_coords(M, C::BM, C::BN, N, m0, n0);
    const int g = lane >> 2, q = lane & 3;

    auto load = [&](int kt, int st) {
        const int k0 = kt * C::BK;
        const unsigned a_st = sA + st * C::A_STAGE, w_st = sW + st * C::W_STAGE;
#pragma unroll
        for (int i = tid; i < C::BM * C::BK / 8; i += C::THREADS) {
            const int r = i / (C::BK / 8), c = (i % (C::BK / 8)) * 8;
            const int m = m0 + r;
            cp_async16(a_st + (r * C::ALD + c) * 2, A + (size_t)(m < M ? m : 0) * lda + k0 + c, m < M);
        }
#pragma unroll
        for (int i = tid; i < C::BN * C::BK / 16; i += C::THREADS) {
            const int r = i / (C::BK / 16), c = (i % (C::BK / 16)) * 16;
            cp_async16(w_st + r * C::WLD + c, W + (size_t)(n0 + r) * K + k0 + c, true);
        }
    };

    float acc[C::MI][C::NI][4];
#pragma unroll
    for (int i = 0; i < C::MI; ++i)
#pragma unroll
        for (int j = 0; j < C::NI; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const int nk = K / C::BK;
#pragma unroll
    for (int st = 0; st < C::STAGES - 1; ++st) {
        if (st < nk) load(st, st);
        asm volatile("cp.async.commit_group;\n");
    }
    // This thread's B rows (n within the CTA tile) and their scale-row offsets.
    int nrow[C::NI];
    const uint8_t* srow[C::NI];
#pragma unroll
    for (int j = 0; j < C::NI; ++j) {
        nrow[j] = wn * C::TN + j * 8 + g;
        srow[j] = S + (size_t)((n0 + nrow[j]) / 32) * s_ld;
    }
    // ldmatrix A address pieces: row = wm*TM + i*16 + (lane & 15), col = kk + (lane >> 4) * 8.
    const int a_row = wm * C::TM + (lane & 15), a_col = (lane >> 4) * 8;

    for (int kt = 0; kt < nk; ++kt) {
        const int st = kt % C::STAGES;
        asm volatile("cp.async.wait_group %0;\n" :: "n"(C::STAGES - 2));
        __syncthreads();
        {
            const int nx = kt + C::STAGES - 1;
            if (nx < nk) load(nx, nx % C::STAGES);
            asm volatile("cp.async.commit_group;\n");
        }
        const unsigned a_st = sA + st * C::A_STAGE, w_st = sW + st * C::W_STAGE;
#pragma unroll
        for (int kb = 0; kb < C::BK / 32; ++kb) {   // one scale block (32 k) at a time
            float sc[C::NI];
#pragma unroll
            for (int j = 0; j < C::NI; ++j) sc[j] = exp2f((float)srow[j][kt * (C::BK / 32) + kb] - 127.f);
#pragma unroll
            for (int kh = 0; kh < 32; kh += 16) {
                const int kk = kb * 32 + kh;
                uint32_t af[C::MI][4];
#pragma unroll
                for (int i = 0; i < C::MI; ++i)
                    ldmatrix_x4(af[i], a_st + ((a_row + i * 16) * C::ALD + kk + a_col) * 2);
#pragma unroll
                for (int j = 0; j < C::NI; ++j) {
                    const unsigned wr = w_st + nrow[j] * C::WLD + kk + 2 * q;
                    const uint32_t b0 = fp8x2_bf16x2(lds_u16(wr), sc[j]);
                    const uint32_t b1 = fp8x2_bf16x2(lds_u16(wr + 8), sc[j]);
#pragma unroll
                    for (int i = 0; i < C::MI; ++i) mma16816(acc[i][j], af[i], b0, b1);
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n");

#pragma unroll
    for (int i = 0; i < C::MI; ++i)
#pragma unroll
        for (int j = 0; j < C::NI; ++j) {
            const int m = m0 + wm * C::TM + i * 16 + g;
            const int n = n0 + wn * C::TN + j * 8 + 2 * q;
            if (m < M)
                *reinterpret_cast<__nv_bfloat162*>(Cout + (size_t)m * ldc + n) = __floats2bfloat162_rn(acc[i][j][0], acc[i][j][1]);
            if (m + 8 < M)
                *reinterpret_cast<__nv_bfloat162*>(Cout + (size_t)(m + 8) * ldc + n) = __floats2bfloat162_rn(acc[i][j][2], acc[i][j][3]);
        }
}


// v4 schedule: the FP8 tile is converted ONCE per CTA into a bf16 B tile (no per-warp redundant
// conversion), then both operands go through ldmatrix like an sm80 bf16 GEMM. Conversion of
// tile kt+1 overlaps the MMAs of tile kt (double-buffered bf16 B), one barrier per k-tile.
template <int BM_, int BN_, int BK_, int STAGES_, int WM_, int WN_>
struct Cfg4 {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, STAGES = STAGES_, WM = WM_, WN = WN_;
    static constexpr int THREADS = 32 * WM * WN;
    static constexpr int TM = BM / WM, TN = BN / WN;
    static constexpr int MI = TM / 16, NI = TN / 8;
    static constexpr int ALD = BK + 8, BLD = BK + 8, WLD = BK + 16;
    static constexpr int A_STAGE = BM * ALD * 2, W_STAGE = BN * WLD, B_BUF = BN * BLD * 2;
    static constexpr int SMEM = STAGES * (A_STAGE + W_STAGE) + 2 * B_BUF;
    static_assert(TM % 16 == 0 && TN % 16 == 0 && BK % 32 == 0, "tile shape");
    static_assert((BN * BK / 16) % THREADS == 0, "conversion split");
};

template <class C>
__device__ __forceinline__ void gemm_body4(const __nv_bfloat16* __restrict__ A, int lda,
                                           const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int s_ld,
                                           __nv_bfloat16* __restrict__ Cout, int ldc, int M, int N, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned sA = sbase, sW = sA + C::STAGES * C::A_STAGE, sB = sW + C::STAGES * C::W_STAGE;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / C::WN, wn = warp % C::WN;
    int m0, n0;
    tile_coords(M, C::BM, C::BN, N, m0, n0);
    const int g = lane >> 2, q = lane & 3;

    auto load = [&](int kt, int st) {
        const int k0 = kt * C::BK;
        const unsigned a_st = sA + st * C::A_STAGE, w_st = sW + st * C::W_STAGE;
#pragma unroll
        for (int i = tid; i < C::BM * C::BK / 8; i += C::THREADS) {
            const int r = i / (C::BK / 8), c = (i % (C::BK / 8)) * 8;
            const int m = m0 + r;
            cp_async16(a_st + (r * C::ALD + c) * 2, A + (size_t)(m < M ? m : 0) * lda + k0 + c, m < M);
        }
#pragma unroll
        for (int i = tid; i < C::BN * C::BK / 16; i += C::THREADS) {
            const int r = i / (C::BK / 16), c = (i % (C::BK / 16)) * 16;
            cp_async16(w_st + r * C::WLD + c, W + (size_t)(n0 + r) * K + k0 + c, true);
        }
    };
    // raw FP8 stage st (k-tile kt) -> bf16 B buffer bb: 16 bytes per thread per step.
    auto convert = [&](int kt, int st, int bb) {
        const unsigned w_st = sW + st * C::W_STAGE, b_st = sB + bb * C::B_BUF;
#pragma unroll
        for (int i = tid; i < C::BN * C::BK / 16; i += C::THREADS) {
            const int r = i / (C::BK / 16), c = (i % (C::BK / 16)) * 16;
            const float sc = exp2f((float)S[(size_t)((n0 + r) / 32) * s_ld + (kt * C::BK + c) / 32] - 127.f);
            uint32_t w4[4];
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(w4[0]), "=r"(w4[1]), "=r"(w4[2]), "=r"(w4[3]) : "r"(w_st + r * C::WLD + c));
            uint32_t o[8];
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                o[2 * h] = fp8x2_bf16x2((uint16_t)(w4[h] & 0xffffu), sc);
                o[2 * h + 1] = fp8x2_bf16x2((uint16_t)(w4[h] >> 16), sc);
            }
            const unsigned dst = b_st + (r * C::BLD + c) * 2;
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst), "r"(o[0]), "r"(o[1]), "r"(o[2]), "r"(o[3]));
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst + 16), "r"(o[4]), "r"(o[5]), "r"(o[6]), "r"(o[7]));
        }
    };

    float acc[C::MI][C::NI][4];
#pragma unroll
    for (int i = 0; i < C::MI; ++i)
#pragma unroll
        for (int j = 0; j < C::NI; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const int nk = K / C::BK;
#pragma unroll
    for (int st = 0; st < C::STAGES - 1; ++st) {
        if (st < nk) load(st, st);
        asm volatile("cp.async.commit_group;\n");
    }
    // Prologue: tile 0 landed -> convert into B buffer 0.
    asm volatile("cp.async.wait_group %0;\n" :: "n"(C::STAGES - 2));
    __syncthreads();
    convert(0, 0, 0);

    const int a_row = wm * C::TM + (lane & 15), a_col = (lane >> 4) * 8;
    // B ldmatrix x4 over two n8 tiles: lanes 0-7 (tile 2p, k lo), 8-15 (2p, k hi), 16-23 (2p+1, k lo), 24-31 (2p+1, k hi).
    const int b_row = wn * C::TN + ((lane >> 4) << 3) + (lane & 7), b_col = ((lane >> 3) & 1) * 8;

    for (int kt = 0; kt < nk; ++kt) {
        const int st = kt % C::STAGES, bb = kt & 1;
        // tile kt+1's raw data must have landed before it is converted below.
        asm volatile("cp.async.wait_group %0;\n" :: "n"(C::STAGES - 3 >= 0 ? C::STAGES - 3 : 0));
        __syncthreads();   // B[bb] (converted last iteration) visible; stage (kt-1)%STAGES and B[bb^1] free
        {
            const int nx = kt + C::STAGES - 1;
            if (nx < nk) load(nx, nx % C::STAGES);
            asm volatile("cp.async.commit_group;\n");
        }
        if (kt + 1 < nk) convert(kt + 1, (kt + 1) % C::STAGES, bb ^ 1);
        const unsigned a_st = sA + st * C::A_STAGE, b_st = sB + bb * C::B_BUF;
#pragma unroll
        for (int kk = 0; kk < C::BK; kk += 16) {
            uint32_t af[C::MI][4], bf[C::NI / 2][4];
#pragma unroll
            for (int i = 0; i < C::MI; ++i) ldmatrix_x4(af[i], a_st + ((a_row + i * 16) * C::ALD + kk + a_col) * 2);
#pragma unroll
            for (int p = 0; p < C::NI / 2; ++p) ldmatrix_x4(bf[p], b_st + ((b_row + p * 16) * C::BLD + kk + b_col) * 2);
#pragma unroll
            for (int i = 0; i < C::MI; ++i)
#pragma unroll
                for (int j = 0; j < C::NI; ++j) mma16816(acc[i][j], af[i], bf[j >> 1][(j & 1) * 2], bf[j >> 1][(j & 1) * 2 + 1]);
        }
    }
    asm volatile("cp.async.wait_group 0;\n");

#pragma unroll
    for (int i = 0; i < C::MI; ++i)
#pragma unroll
        for (int j = 0; j < C::NI; ++j) {
            const int m = m0 + wm * C::TM + i * 16 + g;
            const int n = n0 + wn * C::TN + j * 8 + 2 * q;
            if (m < M)
                *reinterpret_cast<__nv_bfloat162*>(Cout + (size_t)m * ldc + n) = __floats2bfloat162_rn(acc[i][j][0], acc[i][j][1]);
            if (m + 8 < M)
                *reinterpret_cast<__nv_bfloat162*>(Cout + (size_t)(m + 8) * ldc + n) = __floats2bfloat162_rn(acc[i][j][2], acc[i][j][3]);
        }
}

// v5: CUTLASS-sm80-style main loop (MmaMultistage): fragments double-buffered across the k16 steps
// AND across the k-tile boundary, the next tile's cp.async issued inside the k16 loop, one barrier
// per k-tile placed before the first fragment load of the next stage. FP8B = false is a pure bf16
// control (B is a pre-dequantized bf16 [N, K] matrix, ldmatrix'd) to measure the skeleton alone.
template <int BM_, int BN_, int BK_, int STAGES_, int WM_, int WN_, bool FP8B_, int PROBE_ = 0>
struct Cfg5 {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, STAGES = STAGES_, WM = WM_, WN = WN_;
    static constexpr bool FP8B = FP8B_;
    // TIMING PROBES ONLY (wrong values): 1 = no scale loads (sc = 1), 2 = no conversion at all.
    static constexpr int PROBE = PROBE_;
    static constexpr int THREADS = 32 * WM * WN;
    static constexpr int TM = BM / WM, TN = BN / WN;
    static constexpr int MI = TM / 16, NI = TN / 8, KK = BK / 16;
    static constexpr int ALD = BK + 8;
    static constexpr int BLDB = FP8B ? BK + 16 : (BK + 8) * 2;   // B row stride in BYTES
    static constexpr int A_STAGE = BM * ALD * 2, B_STAGE = BN * BLDB;
    static constexpr int SMEM = STAGES * (A_STAGE + B_STAGE);
    static_assert(TM % 16 == 0 && TN % 16 == 0 && BK % 32 == 0 && STAGES >= 2, "tile shape");
};

template <class C>
__device__ __forceinline__ void gemm_body5(const __nv_bfloat16* __restrict__ A, int lda,
                                           const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int s_ld,
                                           __nv_bfloat16* __restrict__ Cout, int ldc, int M, int N, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    const unsigned sA = (unsigned)__cvta_generic_to_shared(smem), sB = sA + C::STAGES * C::A_STAGE;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / C::WN, wn = warp % C::WN;
    int m0, n0;
    tile_coords(M, C::BM, C::BN, N, m0, n0);
    const int g = lane >> 2, q = lane & 3;
    const int nk = K / C::BK;

    auto load = [&](int kt, int st) {
        const int k0 = kt * C::BK;
        const unsigned a_st = sA + st * C::A_STAGE, b_st = sB + st * C::B_STAGE;
#pragma unroll
        for (int i = tid; i < C::BM * C::BK / 8; i += C::THREADS) {
            const int r = i / (C::BK / 8), c = (i % (C::BK / 8)) * 8;
            const int m = m0 + r;
            cp_async16(a_st + (r * C::ALD + c) * 2, A + (size_t)(m < M ? m : 0) * lda + k0 + c, m < M);
        }
        if constexpr (C::FP8B) {
#pragma unroll
            for (int i = tid; i < C::BN * C::BK / 16; i += C::THREADS) {
                const int r = i / (C::BK / 16), c = (i % (C::BK / 16)) * 16;
                cp_async16(b_st + r * C::BLDB + c, W + (size_t)(n0 + r) * K + k0 + c, true);
            }
        } else {
            const __nv_bfloat16* Wb = reinterpret_cast<const __nv_bfloat16*>(W);
#pragma unroll
            for (int i = tid; i < C::BN * C::BK / 8; i += C::THREADS) {
                const int r = i / (C::BK / 8), c = (i % (C::BK / 8)) * 8;
                cp_async16(b_st + r * C::BLDB + c * 2, Wb + (size_t)(n0 + r) * K + k0 + c, true);
            }
        }
    };

    const int a_row = wm * C::TM + (lane & 15), a_col = (lane >> 4) * 8;
    const int b_row = wn * C::TN + ((lane >> 4) << 3) + (lane & 7), b_col = ((lane >> 3) & 1) * 8;
    int nrow[C::NI];
#pragma unroll
    for (int j = 0; j < C::NI; ++j) nrow[j] = wn * C::TN + j * 8 + g;
    // A thread's B rows j*8+g (j < NI) fall in TN/32 scale rows (8-row groups never straddle a
    // 32-row block): the scales of k-tile kt+1 are loaded ONE tile ahead (a per-k16 global load
    // on the fragment path cost ~30%: the sc=1 probe ran at the bf16 control's speed).
    constexpr int NSB = (C::TN + 31) / 32, KSB = C::BK / 32;
    const uint8_t* sbase = S + (size_t)((n0 + wn * C::TN) / 32) * s_ld;
    uint32_t scur[NSB][KSB], snext[NSB][KSB];   // raw scale bytes
    auto load_scales = [&](int kt, uint32_t (&dst)[NSB][KSB]) {
#pragma unroll
        for (int b = 0; b < NSB; ++b)
#pragma unroll
            for (int kb = 0; kb < KSB; ++kb) dst[b][kb] = sbase[(size_t)b * s_ld + kt * KSB + kb];
    };
    if constexpr (C::FP8B) load_scales(0, scur);
    uint32_t af[2][C::MI][4], bf[2][C::NI][2];
    // Fragments of k16 step kk of the k-tile kt held in stage st (scales sc of that tile).
    auto frags = [&](int buf, int kt, int st, int kk, const uint32_t (&sc)[NSB][KSB]) {
        const unsigned a_st = sA + st * C::A_STAGE, b_st = sB + st * C::B_STAGE;
#pragma unroll
        for (int i = 0; i < C::MI; ++i) ldmatrix_x4(af[buf][i], a_st + ((a_row + i * 16) * C::ALD + kk * 16 + a_col) * 2);
        if constexpr (C::FP8B) {
#pragma unroll
            for (int j = 0; j < C::NI; ++j) {
                const unsigned wr = b_st + nrow[j] * C::BLDB + kk * 16 + 2 * q;
                if constexpr (C::PROBE == 2) {
                    bf[buf][j][0] = lds_u16(wr);
                    bf[buf][j][1] = lds_u16(wr + 8);
                } else {
                    // (an integer f16->bf16 re-bias instead of the f32 round trip measured SLOWER, 42 vs
                    // 61 TF/s on wq_b, and is wrong for the NaN encodings: not used)
                    const float s1 = ue8m0_scale((uint8_t)sc[(j * 8) / 32][(kk * 16) / 32]);
                    bf[buf][j][0] = fp8x2_bf16x2(lds_u16(wr), s1);
                    bf[buf][j][1] = fp8x2_bf16x2(lds_u16(wr + 8), s1);
                }
            }
        } else {
#pragma unroll
            for (int p = 0; p < C::NI / 2; ++p) {
                uint32_t r4[4];
                ldmatrix_x4(r4, b_st + (b_row + p * 16) * C::BLDB + (kk * 16 + b_col) * 2);
                bf[buf][2 * p][0] = r4[0]; bf[buf][2 * p][1] = r4[1];
                bf[buf][2 * p + 1][0] = r4[2]; bf[buf][2 * p + 1][1] = r4[3];
            }
        }
    };

    float acc[C::MI][C::NI][4];
#pragma unroll
    for (int i = 0; i < C::MI; ++i)
#pragma unroll
        for (int j = 0; j < C::NI; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

#pragma unroll
    for (int st = 0; st < C::STAGES - 1; ++st) {
        if (st < nk) load(st, st);
        asm volatile("cp.async.commit_group;\n");
    }
    asm volatile("cp.async.wait_group %0;\n" :: "n"(C::STAGES - 2));
    __syncthreads();
    frags(0, 0, 0, 0, scur);

    for (int kt = 0; kt < nk; ++kt) {
        const int st = kt % C::STAGES;
#pragma unroll
        for (int kk = 0; kk < C::KK; ++kk) {
            const int cur = kk & 1;
            if (kk == 0) {   // refill the stage consumed by tile kt-1 (every warp passed the barrier)
                const int nx = kt + C::STAGES - 1;
                if (nx < nk) load(nx, nx % C::STAGES);
                asm volatile("cp.async.commit_group;\n");
                if constexpr (C::FP8B) if (kt + 1 < nk) load_scales(kt + 1, snext);
            }
            if (kk == C::KK - 1) {
                // next tile: its stage must have landed before its first fragments are read
                asm volatile("cp.async.wait_group %0;\n" :: "n"(C::STAGES - 2));
                __syncthreads();
                if constexpr (C::FP8B) {
#pragma unroll
                    for (int b = 0; b < NSB; ++b)
#pragma unroll
                        for (int kb = 0; kb < KSB; ++kb) scur[b][kb] = snext[b][kb];
                }
                if (kt + 1 < nk) frags(cur ^ 1, kt + 1, (kt + 1) % C::STAGES, 0, scur);
            } else {
                frags(cur ^ 1, kt, st, kk + 1, scur);
            }
#pragma unroll
            for (int i = 0; i < C::MI; ++i)
#pragma unroll
                for (int j = 0; j < C::NI; ++j) mma16816(acc[i][j], af[cur][i], bf[cur][j][0], bf[cur][j][1]);
        }
    }
    asm volatile("cp.async.wait_group 0;\n");

#pragma unroll
    for (int i = 0; i < C::MI; ++i)
#pragma unroll
        for (int j = 0; j < C::NI; ++j) {
            const int m = m0 + wm * C::TM + i * 16 + g;
            const int n = n0 + wn * C::TN + j * 8 + 2 * q;
            if (m < M)
                *reinterpret_cast<__nv_bfloat162*>(Cout + (size_t)m * ldc + n) = __floats2bfloat162_rn(acc[i][j][0], acc[i][j][1]);
            if (m + 8 < M)
                *reinterpret_cast<__nv_bfloat162*>(Cout + (size_t)(m + 8) * ldc + n) = __floats2bfloat162_rn(acc[i][j][2], acc[i][j][3]);
        }
}

// v6: the v5 main loop with the FP8 tile converted ONCE per CTA: raw FP8 stages (cp.async) ->
// all threads convert k-tile kt+1 into a double-buffered bf16 B tile during k-tile kt -> both
// operands ldmatrix'd exactly as the bf16 control. Same MMAs, same order: byte-identical.
template <int BM_, int BN_, int BK_, int STAGES_, int WM_, int WN_>
struct Cfg6 {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, STAGES = STAGES_, WM = WM_, WN = WN_;
    static constexpr int THREADS = 32 * WM * WN;
    static constexpr int TM = BM / WM, TN = BN / WN;
    static constexpr int MI = TM / 16, NI = TN / 8, KK = BK / 16;
    static constexpr int ALD = BK + 8, BLD = BK + 8, WLD = BK;   // raw rows unpadded: read linearly
    static constexpr int A_STAGE = BM * ALD * 2, W_STAGE = BN * WLD, B_BUF = BN * BLD * 2;
    static constexpr int SMEM = STAGES * (A_STAGE + W_STAGE) + 2 * B_BUF;
    static_assert(TM % 16 == 0 && TN % 16 == 0 && BK % 32 == 0 && STAGES >= 3, "tile shape");
    static_assert((BN * BK / 16) % THREADS == 0 || THREADS % (BN * BK / 16) == 0, "conversion split");
};

template <class C>
__device__ __forceinline__ void gemm_body6(const __nv_bfloat16* __restrict__ A, int lda,
                                           const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int s_ld,
                                           __nv_bfloat16* __restrict__ Cout, int ldc, int M, int N, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    const unsigned sA = (unsigned)__cvta_generic_to_shared(smem), sW = sA + C::STAGES * C::A_STAGE,
                   sB = sW + C::STAGES * C::W_STAGE;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / C::WN, wn = warp % C::WN;
    int m0, n0;
    tile_coords(M, C::BM, C::BN, N, m0, n0);
    const int g = lane >> 2, q = lane & 3;
    const int nk = K / C::BK;

    auto load = [&](int kt, int st) {
        const int k0 = kt * C::BK;
        const unsigned a_st = sA + st * C::A_STAGE, w_st = sW + st * C::W_STAGE;
#pragma unroll
        for (int i = tid; i < C::BM * C::BK / 8; i += C::THREADS) {
            const int r = i / (C::BK / 8), c = (i % (C::BK / 8)) * 8;
            const int m = m0 + r;
            cp_async16(a_st + (r * C::ALD + c) * 2, A + (size_t)(m < M ? m : 0) * lda + k0 + c, m < M);
        }
#pragma unroll
        for (int i = tid; i < C::BN * C::BK / 16; i += C::THREADS) {
            const int r = i / (C::BK / 16), c = (i % (C::BK / 16)) * 16;
            cp_async16(w_st + r * C::WLD + c, W + (size_t)(n0 + r) * K + k0 + c, true);
        }
    };
    auto convert = [&](int kt, int st, int bb) {
        const unsigned w_st = sW + st * C::W_STAGE, b_st = sB + bb * C::B_BUF;
#pragma unroll
        for (int i = tid; i < C::BN * C::BK / 16; i += C::THREADS) {
            const int r = i / (C::BK / 16), c = (i % (C::BK / 16)) * 16;
            const uint8_t sb = S[(size_t)((n0 + r) / 32) * s_ld + (kt * C::BK + c) / 32];
            uint32_t w4[4];
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(w4[0]), "=r"(w4[1]), "=r"(w4[2]), "=r"(w4[3]) : "r"(w_st + r * C::WLD + c));
            uint32_t o[8];
            const float sc = ue8m0_scale(sb);
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                o[2 * h] = fp8x2_bf16x2((uint16_t)(w4[h] & 0xffffu), sc);
                o[2 * h + 1] = fp8x2_bf16x2((uint16_t)(w4[h] >> 16), sc);
            }
            const unsigned dst = b_st + (r * C::BLD + c) * 2;
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst), "r"(o[0]), "r"(o[1]), "r"(o[2]), "r"(o[3]));
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst + 16), "r"(o[4]), "r"(o[5]), "r"(o[6]), "r"(o[7]));
        }
    };

    const int a_row = wm * C::TM + (lane & 15), a_col = (lane >> 4) * 8;
    const int b_row = wn * C::TN + ((lane >> 4) << 3) + (lane & 7), b_col = ((lane >> 3) & 1) * 8;
    uint32_t af[2][C::MI][4], bf[2][C::NI][2];
    auto frags = [&](int buf, int st, int bb, int kk) {
        const unsigned a_st = sA + st * C::A_STAGE, b_st = sB + bb * C::B_BUF;
#pragma unroll
        for (int i = 0; i < C::MI; ++i) ldmatrix_x4(af[buf][i], a_st + ((a_row + i * 16) * C::ALD + kk * 16 + a_col) * 2);
#pragma unroll
        for (int p = 0; p < C::NI / 2; ++p) {
            uint32_t r4[4];
            ldmatrix_x4(r4, b_st + ((b_row + p * 16) * C::BLD + kk * 16 + b_col) * 2);
            bf[buf][2 * p][0] = r4[0]; bf[buf][2 * p][1] = r4[1];
            bf[buf][2 * p + 1][0] = r4[2]; bf[buf][2 * p + 1][1] = r4[3];
        }
    };

    float acc[C::MI][C::NI][4];
#pragma unroll
    for (int i = 0; i < C::MI; ++i)
#pragma unroll
        for (int j = 0; j < C::NI; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

#pragma unroll
    for (int st = 0; st < C::STAGES - 1; ++st) {
        if (st < nk) load(st, st);
        asm volatile("cp.async.commit_group;\n");
    }
    // tile 0 into B[0]; tile 1 must also be resident before the loop converts it at kt = 0.
    asm volatile("cp.async.wait_group %0;\n" :: "n"(C::STAGES - 2));
    __syncthreads();
    convert(0, 0, 0);
    asm volatile("cp.async.wait_group %0;\n" :: "n"(C::STAGES - 3));
    __syncthreads();
    frags(0, 0, 0, 0);

    for (int kt = 0; kt < nk; ++kt) {
        const int st = kt % C::STAGES, bb = kt & 1;
#pragma unroll
        for (int kk = 0; kk < C::KK; ++kk) {
            const int cur = kk & 1;
            if (kk == 0) {
                const int nx = kt + C::STAGES - 1;
                if (nx < nk) load(nx, nx % C::STAGES);
                asm volatile("cp.async.commit_group;\n");
                if (kt + 1 < nk) convert(kt + 1, (kt + 1) % C::STAGES, bb ^ 1);
            }
            if (kk == C::KK - 1) {
                // B[bb^1] (tile kt+1) written by every thread; raw tile kt+2 landed for the next convert.
                asm volatile("cp.async.wait_group %0;\n" :: "n"(C::STAGES - 3));
                __syncthreads();
                if (kt + 1 < nk) frags(cur ^ 1, (kt + 1) % C::STAGES, bb ^ 1, 0);
            } else {
                frags(cur ^ 1, st, bb, kk + 1);
            }
#pragma unroll
            for (int i = 0; i < C::MI; ++i)
#pragma unroll
                for (int j = 0; j < C::NI; ++j) mma16816(acc[i][j], af[cur][i], bf[cur][j][0], bf[cur][j][1]);
        }
    }
    asm volatile("cp.async.wait_group 0;\n");

#pragma unroll
    for (int i = 0; i < C::MI; ++i)
#pragma unroll
        for (int j = 0; j < C::NI; ++j) {
            const int m = m0 + wm * C::TM + i * 16 + g;
            const int n = n0 + wn * C::TN + j * 8 + 2 * q;
            if (m < M)
                *reinterpret_cast<__nv_bfloat162*>(Cout + (size_t)m * ldc + n) = __floats2bfloat162_rn(acc[i][j][0], acc[i][j][1]);
            if (m + 8 < M)
                *reinterpret_cast<__nv_bfloat162*>(Cout + (size_t)(m + 8) * ldc + n) = __floats2bfloat162_rn(acc[i][j][2], acc[i][j][3]);
        }
}
}  // namespace dsv41_fp8gemm3

#define DSV41_FP8GEMM3_ENTRY(NAME, BM, BN, BK, ST, WM, WN)                                                       \
    extern "C" __global__ void __launch_bounds__(32 * WM * WN) NAME(                                             \
        const __nv_bfloat16* __restrict__ A, int lda, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, \
        int s_ld, __nv_bfloat16* __restrict__ C, int ldc, int M, int N, int K) {                                 \
        dsv41_fp8gemm3::gemm_body<dsv41_fp8gemm3::Cfg<BM, BN, BK, ST, WM, WN>>(A, lda, W, S, s_ld, C, ldc, M, N, K); \
    }

#define DSV41_FP8GEMM4_ENTRY(NAME, BM, BN, BK, ST, WM, WN)                                                       \
    extern "C" __global__ void __launch_bounds__(32 * WM * WN) NAME(                                             \
        const __nv_bfloat16* __restrict__ A, int lda, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, \
        int s_ld, __nv_bfloat16* __restrict__ C, int ldc, int M, int N, int K) {                                 \
        dsv41_fp8gemm3::gemm_body4<dsv41_fp8gemm3::Cfg4<BM, BN, BK, ST, WM, WN>>(A, lda, W, S, s_ld, C, ldc, M, N, K); \
    }

#define DSV41_FP8GEMM5_ENTRY(NAME, BM, BN, BK, ST, WM, WN, FP8B, ...)                                                 \
    extern "C" __global__ void __launch_bounds__(32 * WM * WN) NAME(                                             \
        const __nv_bfloat16* __restrict__ A, int lda, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, \
        int s_ld, __nv_bfloat16* __restrict__ C, int ldc, int M, int N, int K) {                                 \
        dsv41_fp8gemm3::gemm_body5<dsv41_fp8gemm3::Cfg5<BM, BN, BK, ST, WM, WN, FP8B __VA_OPT__(,) __VA_ARGS__>>(A, lda, W, S, s_ld, C, ldc, M, N, K); \
    }

#define DSV41_FP8GEMM6_ENTRY(NAME, BM, BN, BK, ST, WM, WN)                                                       \
    extern "C" __global__ void __launch_bounds__(32 * WM * WN) NAME(                                             \
        const __nv_bfloat16* __restrict__ A, int lda, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, \
        int s_ld, __nv_bfloat16* __restrict__ C, int ldc, int M, int N, int K) {                                 \
        dsv41_fp8gemm3::gemm_body6<dsv41_fp8gemm3::Cfg6<BM, BN, BK, ST, WM, WN>>(A, lda, W, S, s_ld, C, ldc, M, N, K); \
    }
