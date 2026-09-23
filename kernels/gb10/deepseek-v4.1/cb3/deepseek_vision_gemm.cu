// SPDX-License-Identifier: AGPL-3.0-only
// DeepSeek vision native BF16 linear/SDPA matmuls. Four 16x16 WMMA tiles
// form one 32x32 CTA. All K tails are zero masked; bias is added to the
// FP32 accumulator before a single BF16 output rounding.
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

template <bool FLOAT_OUTPUT>
__device__ __forceinline__ void dsv_matmul(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    const __nv_bfloat16* bias, void* output,
    unsigned m, unsigned n, unsigned k, unsigned ldc) {
    __shared__ __align__(32) __nv_bfloat16 sa[32 * 16];
    __shared__ __align__(32) __nv_bfloat16 sb[32 * 16];
    __shared__ __align__(32) float sc[32 * 32];
    const unsigned tid = threadIdx.x, warp = tid / 32;
    const unsigned rm = blockIdx.y * 32, cn = blockIdx.x * 32;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);
    for (unsigned kb = 0; kb < k; kb += 16) {
        for (unsigned i = tid; i < 32 * 16; i += 128) {
            unsigned r = i / 16, c = i % 16;
            sa[i] = rm + r < m && kb + c < k
                ? a[(unsigned long long)(rm + r) * k + kb + c] : __float2bfloat16(0.0f);
            // B is globally [N,K]; this tile is column-major [K,N].
            sb[i] = cn + r < n && kb + c < k
                ? b[(unsigned long long)(cn + r) * k + kb + c] : __float2bfloat16(0.0f);
        }
        __syncthreads();
        wmma::load_matrix_sync(fa, sa + (warp / 2) * 16 * 16, 16);
        wmma::load_matrix_sync(fb, sb + (warp % 2) * 16 * 16, 16);
        wmma::mma_sync(fc, fa, fb, fc);
        __syncthreads();
    }
    wmma::store_matrix_sync(sc + (warp / 2) * 16 * 32 + (warp % 2) * 16,
        fc, 32, wmma::mem_row_major);
    __syncthreads();
    for (unsigned i = tid; i < 32 * 32; i += 128) {
        const unsigned row = rm + i / 32, col = cn + i % 32;
        if (row < m && col < n) {
            const unsigned long long index = (unsigned long long)row * ldc + col;
            if constexpr (FLOAT_OUTPUT) {
                static_cast<float*>(output)[index] = sc[i];
            } else {
                float value = sc[i] + (bias ? __bfloat162float(bias[col]) : 0.0f);
                static_cast<__nv_bfloat16*>(output)[index] = __float2bfloat16_rn(value);
            }
        }
    }
}

extern "C" __global__ void deepseek_vision_linear(
    const __nv_bfloat16* a, const __nv_bfloat16* b, const __nv_bfloat16* bias,
    __nv_bfloat16* c, unsigned m, unsigned n, unsigned k, unsigned ldc) {
    dsv_matmul<false>(a, b, bias, c, m, n, k, ldc);
}

extern "C" __global__ void deepseek_vision_scores(
    const __nv_bfloat16* a, const __nv_bfloat16* b, float* c,
    unsigned m, unsigned n, unsigned k, unsigned ldc) {
    dsv_matmul<true>(a, b, nullptr, c, m, n, k, ldc);
}
