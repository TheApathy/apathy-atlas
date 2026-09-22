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

extern "C" __global__ void __launch_bounds__(dsv41_fp8gemm::THREADS) dsv41_fp8_gemm_nt_v2(
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
                const uint32_t bb[2] = {fp8x2_to_bf16x2(*reinterpret_cast<const uint16_t*>(wr), sc[j]),
                                        fp8x2_to_bf16x2(*reinterpret_cast<const uint16_t*>(wr + 8), sc[j])};
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
