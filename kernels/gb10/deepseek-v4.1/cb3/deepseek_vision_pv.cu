// SPDX-License-Identifier: AGPL-3.0-only
// Native DeepSeek vision math-SDPA P@V: FP32 probabilities/accumulators,
// exactly widened BF16 values, and one final BF16 rounding. No TF32 path.
#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void deepseek_vision_attention_value(
    const float* probabilities, const __nv_bfloat16* value,
    __nv_bfloat16* output, unsigned patches, unsigned ldc) {
    // One CTA computes 16 patch rows x the complete 64-channel head.
    // Global V is [64,patches]; shared V is transposed for contiguous reads.
    __shared__ float sp[16 * 32];
    __shared__ float sv[32 * 64];
    const unsigned tid = threadIdx.x, col = tid % 64, row = tid / 64;
    const unsigned first = blockIdx.x * 16;
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned kb = 0; kb < patches; kb += 32) {
        for (unsigned i = tid; i < 16 * 32; i += 256) {
            const unsigned r = i / 32, k = i % 32;
            sp[i] = first + r < patches && kb + k < patches
                ? probabilities[(unsigned long long)(first + r) * patches + kb + k]
                : 0.0f;
        }
        for (unsigned i = tid; i < 64 * 32; i += 256) {
            const unsigned c = i / 32, k = i % 32;
            sv[k * 64 + c] = kb + k < patches
                ? __bfloat162float(value[(unsigned long long)c * patches + kb + k])
                : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned k = 0; k < 32; ++k) {
            const float v = sv[k * 64 + col];
            #pragma unroll
            for (unsigned r = 0; r < 4; ++r)
                accum[r] = fmaf(sp[(row + r * 4) * 32 + k], v, accum[r]);
        }
        // Every reader finishes before the next tile overwrites shared arrays.
        __syncthreads();
    }
    #pragma unroll
    for (unsigned r = 0; r < 4; ++r) {
        const unsigned output_row = first + row + r * 4;
        if (output_row < patches)
            output[(unsigned long long)output_row * ldc + col] = __float2bfloat16_rn(accum[r]);
    }
}
