// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1 dense FP8 linear, FUSED: C[M, N] (bf16) = A[M, K] (bf16) @ dequant(W[N, K])^T with
// W stored e4m3 + UE8M0 scales per 32x32 block. The weight is read as FP8 and converted to bf16
// in shared memory tile by tile (exact: e4m3 x 2^k fits bf16), so the separate dequant pass --
// which writes a full bf16 copy of every weight, ~168 ms of a warm prefill -- disappears.
//
// mma.sync.m16n8k16 bf16 -> fp32. Tile 128 x 128 x 32, 8 warps (2 x 4), warp tile 64 x 32,
// double-buffered cp.async for A and the raw FP8 tile. FIXED tile config and NO split-K: every
// output is one fp32 accumulation chain over k in ascending 16-steps, so a row's result does not
// depend on M or on the rows it shares a launch with (chunk invariance by construction).
// lda / ldc allow column slices (the grouped wo_a). Requires N % 128 == 0 and K % 32 == 0.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace dsv41_fp8gemm {
constexpr int BM = 128, BN = 128, BK = 32, THREADS = 256, PAD = 8;
constexpr int LDS = BK + PAD;   // smem row stride (bf16 elements), 80 B: 16-byte aligned rows

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(pred ? 16 : 0));
}
}  // namespace dsv41_fp8gemm

extern "C" __global__ void __launch_bounds__(dsv41_fp8gemm::THREADS) dsv41_fp8_gemm_nt(
    const __nv_bfloat16* __restrict__ A, int lda,
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int s_ld,
    __nv_bfloat16* __restrict__ C, int ldc, int M, int N, int K)
{
    using namespace dsv41_fp8gemm;
    __shared__ __align__(16) __nv_bfloat16 as[2][BM][LDS];
    __shared__ __align__(16) uint8_t wraw[2][BN][BK];
    __shared__ __align__(16) __nv_bfloat16 bs[BN][LDS];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;               // warp tile origin (64 x 32)
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int g = lane >> 2, q = lane & 3;

    auto load = [&](int kt, int b) {
        const int k0 = kt * BK;
        // A: 128 rows x 32 bf16 = 512 x 16 B chunks, 2 per thread; rows >= M zero-filled.
        for (int i = tid; i < BM * BK / 8; i += THREADS) {
            const int r = i / (BK / 8), c = (i % (BK / 8)) * 8;
            const int m = m0 + r;
            cp_async16(&as[b][r][c], A + (size_t)(m < M ? m : 0) * lda + k0 + c, m < M);
        }
        // W: 128 rows x 32 fp8 = 256 x 16 B chunks, 1 per thread.
        for (int i = tid; i < BN * BK / 16; i += THREADS) {
            const int r = i / (BK / 16), c = (i % (BK / 16)) * 16;
            cp_async16(&wraw[b][r][c], W + (size_t)(n0 + r) * K + k0 + c, true);
        }
        asm volatile("cp.async.commit_group;\n");
    };

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const int nk = K / BK;
    load(0, 0);
    for (int kt = 0; kt < nk; ++kt) {
        const int b = kt & 1;
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        if (kt + 1 < nk) load(kt + 1, b ^ 1);
        // Convert this tile's FP8 weights to bf16 (exact), one 32x32 scale block per 32 rows.
        {
            const int r = tid >> 1, c = (tid & 1) * 16;
            const float sc = exp2f((float)S[(size_t)((n0 + r) / 32) * s_ld + kt] - 127.f);
            const uint8_t* src = &wraw[b][r][c];
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                __nv_fp8_e4m3 f0, f1;
                f0.__x = src[j];
                f1.__x = src[j + 1];
                const __nv_bfloat162 v = __floats2bfloat162_rn(float(f0) * sc, float(f1) * sc);
                *reinterpret_cast<__nv_bfloat162*>(&bs[r][c + j]) = v;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t af[4][4], bfr[2][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                ldmatrix_x4(af[i], &as[b][wm * 64 + i * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                // x4 over two n8 tiles: lanes 0-7 (n-tile 2j, k lo), 8-15 (2j, k hi),
                // 16-23 (2j+1, k lo), 24-31 (2j+1, k hi).
                const int nrow = wn * 32 + j * 16 + ((lane >> 4) << 3) + (lane & 7);
                ldmatrix_x4(bfr[j], &bs[nrow][kk + ((lane >> 3) & 1) * 8]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t bb[2] = {bfr[j >> 1][(j & 1) * 2], bfr[j >> 1][(j & 1) * 2 + 1]};
                    mma16816(acc[i][j], af[i], bb);
                }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int m = m0 + wm * 64 + i * 16 + g;
            const int n = n0 + wn * 32 + j * 8 + 2 * q;
            if (m < M)
                *reinterpret_cast<__nv_bfloat162*>(C + (size_t)m * ldc + n) = __floats2bfloat162_rn(acc[i][j][0], acc[i][j][1]);
            if (m + 8 < M)
                *reinterpret_cast<__nv_bfloat162*>(C + (size_t)(m + 8) * ldc + n) = __floats2bfloat162_rn(acc[i][j][2], acc[i][j][3]);
        }
}

// v2: B converted ON THE FLY from the raw FP8 tile into mma fragments (no bf16 B tile, one
// barrier per k-tile instead of two) and a 3-stage cp.async pipeline. Same values into the same
// MMAs in the same order as dsv41_fp8_gemm_nt -> must be BYTE-IDENTICAL to it.
namespace dsv41_fp8gemm {
constexpr int STAGES = 3, WLD = BK + 16;   // raw FP8 row stride (bytes): 48, 16-byte aligned

__device__ __forceinline__ uint32_t fp8x2_to_bf16x2(uint16_t two, float sc) {
    __nv_fp8_e4m3 f0, f1;
    f0.__x = (uint8_t)(two & 0xff);
    f1.__x = (uint8_t)(two >> 8);
    const __nv_bfloat162 v = __floats2bfloat162_rn(float(f0) * sc, float(f1) * sc);
    return *reinterpret_cast<const uint32_t*>(&v);
}
}  // namespace dsv41_fp8gemm

template <bool CONVERT>
__device__ __forceinline__ void fp8_gemm_v2_body(
    const __nv_bfloat16* __restrict__ A, int lda,
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int s_ld,
    __nv_bfloat16* __restrict__ C, int ldc, int M, int N, int K)
{
    using namespace dsv41_fp8gemm;
    __shared__ __align__(16) __nv_bfloat16 as[STAGES][BM][LDS];
    __shared__ __align__(16) uint8_t wraw[STAGES][BN][WLD];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int g = lane >> 2, q = lane & 3;

    auto load = [&](int kt, int b) {
        const int k0 = kt * BK;
        for (int i = tid; i < BM * BK / 8; i += THREADS) {
            const int r = i / (BK / 8), c = (i % (BK / 8)) * 8;
            const int m = m0 + r;
            cp_async16(&as[b][r][c], A + (size_t)(m < M ? m : 0) * lda + k0 + c, m < M);
        }
        for (int i = tid; i < BN * BK / 16; i += THREADS) {
            const int r = i / (BK / 16), c = (i % (BK / 16)) * 16;
            cp_async16(&wraw[b][r][c], W + (size_t)(n0 + r) * K + k0 + c, true);
        }
    };

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const int nk = K / BK;
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < nk) load(st, st);
        asm volatile("cp.async.commit_group;\n");
    }
    // The 4 weight rows this thread's B fragments read: n = wn*32 + j*8 + g, j = 0..3.
    int nrow[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) nrow[j] = wn * 32 + j * 8 + g;

    for (int kt = 0; kt < nk; ++kt) {
        const int b = kt % STAGES;
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
        __syncthreads();
        {   // prefetch kt + STAGES - 1 into the slot freed by kt - 1
            const int nx = kt + STAGES - 1;
            if (nx < nk) load(nx, nx % STAGES);
            asm volatile("cp.async.commit_group;\n");
        }
        float sc[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) sc[j] = exp2f((float)S[(size_t)((n0 + nrow[j]) / 32) * s_ld + kt] - 127.f);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t af[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                ldmatrix_x4(af[i], &as[b][wm * 64 + i * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint8_t* wr = &wraw[b][nrow[j]][kk + 2 * q];
                uint32_t bb[2];
                if (CONVERT) {
                    bb[0] = fp8x2_to_bf16x2(*reinterpret_cast<const uint16_t*>(wr), sc[j]);
                    bb[1] = fp8x2_to_bf16x2(*reinterpret_cast<const uint16_t*>(wr + 8), sc[j]);
                } else {   // TIMING PROBE ONLY: raw bits, wrong values
                    bb[0] = *reinterpret_cast<const uint16_t*>(wr);
                    bb[1] = *reinterpret_cast<const uint16_t*>(wr + 8);
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) mma16816(acc[i][j], af[i], bb);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int m = m0 + wm * 64 + i * 16 + g;
            const int n = n0 + wn * 32 + j * 8 + 2 * q;
            if (m < M)
                *reinterpret_cast<__nv_bfloat162*>(C + (size_t)m * ldc + n) = __floats2bfloat162_rn(acc[i][j][0], acc[i][j][1]);
            if (m + 8 < M)
                *reinterpret_cast<__nv_bfloat162*>(C + (size_t)(m + 8) * ldc + n) = __floats2bfloat162_rn(acc[i][j][2], acc[i][j][3]);
        }
}

extern "C" __global__ void __launch_bounds__(dsv41_fp8gemm::THREADS) dsv41_fp8_gemm_nt_v2(
    const __nv_bfloat16* __restrict__ A, int lda,
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int s_ld,
    __nv_bfloat16* __restrict__ C, int ldc, int M, int N, int K)
{
    fp8_gemm_v2_body<true>(A, lda, W, S, s_ld, C, ldc, M, N, K);
}

#ifdef DSV41_FP8GEMM_GATE
// Timing probe: the same kernel with the FP8 -> bf16 conversion removed (wrong values).
extern "C" __global__ void __launch_bounds__(dsv41_fp8gemm::THREADS) dsv41_fp8_gemm_probe_noconv(
    const __nv_bfloat16* __restrict__ A, int lda,
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int s_ld,
    __nv_bfloat16* __restrict__ C, int ldc, int M, int N, int K)
{
    fp8_gemm_v2_body<false>(A, lda, W, S, s_ld, C, ldc, M, N, K);
}
#endif

// ── v7 (dsv41-decode) ───────────────────────────────────────────────────────────────────────────
// Same numeric contract as v1/v2 (one m16n8k16 bf16 fp32 chain per output, k ascending in 16s, B =
// bf16(fp8 * 2^(s-127)) exact, no split-K): BYTE-IDENTICAL to dequant + cuBLASLt (gate:
// fp8gemm3/gate.log, all 7 layer-2 shapes at M = 512 and 2048). The schedule is what changes:
//  - L2-grouped 1D raster (GROUP_M m-tiles down each n column): the old n-fastest grid re-read all of
//    W from DRAM once per m-tile;
//  - the FP8 tile converted ONCE per CTA into a double-buffered bf16 B tile, each thread converting
//    exactly the chunks its own cp.async wrote (its wait_group alone makes them visible), so ONE
//    barrier per k-tile and a second tile stays in flight;
//  - CUTLASS-sm80-style main loop: ldmatrix for both operands, fragments double-buffered across the
//    k16 steps and the k-tile boundary, dynamic shared memory (> 48 KB).
// At M = 2048: wq_b 79, wo_b 81, w1 90, w2 83, wq_a 81 TF/s (cuBLAS on pre-dequantized weights
// 88-94); 1.15-1.44x the dequant + GEMM path.
#include <cuda_fp16.h>
namespace dsv41_fp8gemm7 {
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

__device__ __forceinline__ uint32_t fp8x2_bf16x2(uint16_t two, float sc) {
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(h) : "h"(two));
    const __half2 hh = *reinterpret_cast<const __half2*>(&h);
    const float2 f = __half22float2(hh);
    const __nv_bfloat162 v = __floats2bfloat162_rn(f.x * sc, f.y * sc);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float ue8m0_scale(uint8_t s) {
    return s ? __int_as_float((int)s << 23) : exp2f(-127.f);
}

constexpr int GROUP_M = 8;
__device__ __forceinline__ void tile_coords(int M, int BM, int BN, int N, int& m0, int& n0) {
    const int pm = (M + BM - 1) / BM, pn = N / BN, pid = blockIdx.x;
    const int per_group = GROUP_M * pn, first = (pid / per_group) * GROUP_M;
    const int gsize = min(pm - first, GROUP_M);
    m0 = (first + (pid % per_group) % gsize) * BM;
    n0 = ((pid % per_group) / gsize) * BN;
}

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

// v7: v6 with the conversion moved to the end of the k-tile on each thread's OWN copies (keeps a
// second tile in flight; v6 had to drain to wait_group 0 at 3 stages).
template <class C>
__device__ __forceinline__ void gemm_body7(const __nv_bfloat16* __restrict__ A, int lda,
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
    // Each thread converts exactly the FP8 chunks its OWN cp.async wrote (same i -> (r, c) map
    // in load and convert), so its wait_group alone makes them visible: no extra barrier.
    asm volatile("cp.async.wait_group %0;\n" :: "n"(C::STAGES - 2));
    convert(0, 0, 0);
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
            }
            if (kk == C::KK - 1) {
                // tile kt+1 landed (own copies; the load issued this tile may stay in flight),
                // converted into B[bb^1], then ONE barrier publishes A(kt+1) and B[bb^1].
                asm volatile("cp.async.wait_group %0;\n" :: "n"(C::STAGES - 2));
                if (kt + 1 < nk) convert(kt + 1, (kt + 1) % C::STAGES, bb ^ 1);
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
}  // namespace dsv41_fp8gemm7

#define DSV41_FP8GEMM7_ENTRY(NAME, BM, BN, BK, ST, WM, WN)                                                           extern "C" __global__ void __launch_bounds__(32 * WM * WN) NAME(                                                     const __nv_bfloat16* __restrict__ A, int lda, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,         int s_ld, __nv_bfloat16* __restrict__ C, int ldc, int M, int N, int K) {                                         dsv41_fp8gemm7::gemm_body7<dsv41_fp8gemm7::Cfg6<BM, BN, BK, ST, WM, WN>>(A, lda, W, S, s_ld, C, ldc, M, N, K);     }
// 256 x 128 CTA (4 x 2 warps of 64 x 64) and 128 x 256 (2 x 4); BK 32, 3 stages; 94 / 96 KB smem.
DSV41_FP8GEMM7_ENTRY(dsv41_fp8_gemm_nt_v7_m256, 256, 128, 32, 3, 4, 2)
DSV41_FP8GEMM7_ENTRY(dsv41_fp8_gemm_nt_v7_n256, 128, 256, 32, 3, 2, 4)
