// SPDX-License-Identifier: AGPL-3.0-only
// Exact dtype bridge between Atlas BF16 activations and ExLlamaV3 F16 GEMM.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define GLM53_EXL3_CAST_THREADS 256U

extern "C" __global__ void __launch_bounds__(GLM53_EXL3_CAST_THREADS, 1)
atlas_glm53_exl3_bf16_to_f16(
        const __nv_bfloat16 * __restrict__ input,
        __half * __restrict__ output,
        unsigned int elements) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = __float2half_rn(__bfloat162float(input[index]));
    }
}

extern "C" __global__ void __launch_bounds__(GLM53_EXL3_CAST_THREADS, 1)
atlas_glm53_exl3_f16_to_bf16(
        const __half * __restrict__ input,
        __nv_bfloat16 * __restrict__ output,
        unsigned int elements) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = __float2bfloat16_rn(__half2float(input[index]));
    }
}
