// SPDX-License-Identifier: AGPL-3.0-only
// GLM-5.3 DFlash2 grouped dynamic causal convolution.

#include <cuda_bf16.h>

#define GLM53_DFLASH2_HIDDEN 4096U
#define GLM53_DFLASH2_GROUP_SIZE 16U
#define GLM53_DFLASH2_GROUPS 256U
#define GLM53_DFLASH2_KERNEL 2U
#define GLM53_DFLASH2_PHASES 2U
#define GLM53_DFLASH2_THREADS 256U

extern "C" __global__ void __launch_bounds__(GLM53_DFLASH2_THREADS, 1)
atlas_glm53_dflash2_grouped_causal_conv(
        const __nv_bfloat16 * __restrict__ input,
        const __nv_bfloat16 * __restrict__ dynamic,
        const __nv_bfloat16 * __restrict__ base,
        __nv_bfloat16 * __restrict__ output,
        unsigned int batch, unsigned int tokens, unsigned int hidden_size,
        unsigned int group_size, unsigned int kernel_size,
        unsigned int phase) {
    if (batch == 0U || tokens == 0U || tokens > 8U ||
        hidden_size != GLM53_DFLASH2_HIDDEN ||
        group_size != GLM53_DFLASH2_GROUP_SIZE ||
        kernel_size != GLM53_DFLASH2_KERNEL ||
        phase >= GLM53_DFLASH2_PHASES) {
        return;
    }
    const unsigned long long elements =
        (unsigned long long) batch * tokens * GLM53_DFLASH2_HIDDEN;
    unsigned long long index =
        (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long) gridDim.x * blockDim.x;
    for (; index < elements; index += stride) {
        const unsigned int channel =
            (unsigned int) (index % GLM53_DFLASH2_HIDDEN);
        const unsigned long long row = index / GLM53_DFLASH2_HIDDEN;
        const unsigned int token = (unsigned int) (row % tokens);
        const unsigned int group = channel / GLM53_DFLASH2_GROUP_SIZE;
        float accumulated = 0.0f;
        #pragma unroll
        for (unsigned int offset = 0U; offset < GLM53_DFLASH2_KERNEL; ++offset) {
            float value = 0.0f;
            if (token >= offset) {
                const unsigned long long source =
                    index - (unsigned long long) offset * GLM53_DFLASH2_HIDDEN;
                value = __bfloat162float(input[source]);
            }
            const unsigned long long base_index =
                ((unsigned long long) phase * GLM53_DFLASH2_KERNEL + offset) *
                    GLM53_DFLASH2_HIDDEN + channel;
            const unsigned long long dynamic_index =
                ((((row * GLM53_DFLASH2_PHASES) + phase) *
                    GLM53_DFLASH2_KERNEL + offset) *
                    GLM53_DFLASH2_GROUPS) + group;
            const float base_product = __bfloat162float(
                __float2bfloat16_rn(__bfloat162float(base[base_index]) * value));
            accumulated = __bfloat162float(
                __float2bfloat16_rn(accumulated + base_product));
            accumulated = __bfloat162float(__float2bfloat16_rn(
                accumulated + __bfloat162float(dynamic[dynamic_index]) * value));
        }
        output[index] = __float2bfloat16_rn(accumulated);
    }
}
