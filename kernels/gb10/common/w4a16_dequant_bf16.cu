// SPDX-License-Identifier: AGPL-3.0-only
//
// NVFP4 (E2M1 nibbles, 16-wide FP8-E4M3 block scales, per-tensor FP32 scale2)
// -> row-major BF16 [N, K] dequantization. Produces exactly the BF16 weight
// operand that the W4A16 tensor-core GEMM (`w4a16_gemm.cu`) forms in shared
// memory: `bf16(LUT[nibble] * float(fp8_scale) * scale2)`. The dequantized
// matrix feeds a cuBLASLt BF16 GEMM (FP32 accumulation), so the only
// numerical difference from `w4a16_gemm` is accumulation order.
//
// Layout (unchanged from the exact GEMV family):
//   packed[n * (K/2) + k/2]  low nibble = even k, high nibble = odd k
//   scales[n * (K/16) + k/16]
//
// One thread dequantizes one 16-wide group: 8 packed bytes + 1 scale byte in,
// 16 BF16 (two uint4) out. Grid = ceil(N*K/16 / 256) blocks of 256 threads.
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __constant__ float W4A16_DQ_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

extern "C" __global__ void w4a16_dequant_bf16(
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scales,
    const float scale2,
    __nv_bfloat16* __restrict__ out,
    unsigned int N,
    unsigned int K,
    unsigned int qg_heads,
    unsigned int qg_head_dim)
{
    const unsigned long long groups_per_row = K / 16u;
    const unsigned long long total = (unsigned long long)N * groups_per_row;
    const unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total || (K & 15u) != 0u) return;
    const unsigned long long n = g / groups_per_row;
    const unsigned long long gi = g - n * groups_per_row;

    const unsigned long long p8 =
        *reinterpret_cast<const unsigned long long*>(packed + n * (K / 2u) + gi * 8ull);
    __nv_fp8_e4m3 fp8;
    *reinterpret_cast<unsigned char*>(&fp8) = scales[n * groups_per_row + gi];
    const float s = (float)fp8;

    unsigned short vals[16];
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        const unsigned char byte_val = (unsigned char)(p8 >> (b * 8));
        const __nv_bfloat16 lo = __float2bfloat16(W4A16_DQ_E2M1_LUT[byte_val & 0xF] * s * scale2);
        const __nv_bfloat16 hi = __float2bfloat16(W4A16_DQ_E2M1_LUT[byte_val >> 4] * s * scale2);
        vals[2 * b] = *reinterpret_cast<const unsigned short*>(&lo);
        vals[2 * b + 1] = *reinterpret_cast<const unsigned short*>(&hi);
    }
    // Optional gated-Q row permutation (matches w4a16_gemv_qg_exact's output
    // order): weight row n = head h, index idx in [0, 2*hd); q part goes to
    // h*hd+idx, gate part to heads*hd + h*hd + (idx-hd).
    unsigned long long out_row = n;
    if (qg_heads != 0u) {
        const unsigned long long group_dim = 2ull * qg_head_dim;
        const unsigned long long h = n / group_dim;
        const unsigned long long idx = n - h * group_dim;
        out_row = idx < qg_head_dim
            ? h * qg_head_dim + idx
            : (unsigned long long)qg_heads * qg_head_dim + h * qg_head_dim + (idx - qg_head_dim);
    }
    uint4* dst = reinterpret_cast<uint4*>(out + out_row * (unsigned long long)K + gi * 16ull);
    uint4 v0, v1;
    v0.x = (unsigned)vals[0] | ((unsigned)vals[1] << 16);
    v0.y = (unsigned)vals[2] | ((unsigned)vals[3] << 16);
    v0.z = (unsigned)vals[4] | ((unsigned)vals[5] << 16);
    v0.w = (unsigned)vals[6] | ((unsigned)vals[7] << 16);
    v1.x = (unsigned)vals[8] | ((unsigned)vals[9] << 16);
    v1.y = (unsigned)vals[10] | ((unsigned)vals[11] << 16);
    v1.z = (unsigned)vals[12] | ((unsigned)vals[13] << 16);
    v1.w = (unsigned)vals[14] | ((unsigned)vals[15] << 16);
    dst[0] = v0;
    dst[1] = v1;
}
