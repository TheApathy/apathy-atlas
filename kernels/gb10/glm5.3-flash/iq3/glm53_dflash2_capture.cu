// SPDX-License-Identifier: AGPL-3.0-only
// Contract GLM-5.3 post-layer mHC state into DFlash2 target-capture slots.

#include <cuda_bf16.h>

#define GLM53_DFLASH2_HIDDEN 4096U
#define GLM53_DFLASH2_HC 4U
#define GLM53_DFLASH2_CAPTURES 5U
#define GLM53_DFLASH2_THREADS 256U

__device__ __forceinline__ bool glm53_dflash2_capture_pair(
        unsigned int post_layer, unsigned int slot) {
    // Config IDs are one-based; the Atlas walk passes zero-based layer indices.
    return (post_layer == 4U && slot == 0U) ||
        (post_layer == 13U && slot == 1U) ||
        (post_layer == 23U && slot == 2U) ||
        (post_layer == 32U && slot == 3U) ||
        (post_layer == 41U && slot == 4U);
}

extern "C" __global__ void __launch_bounds__(GLM53_DFLASH2_THREADS, 1)
atlas_glm53_dflash2_capture_mean(
        const __nv_bfloat16 * __restrict__ streams,
        __nv_bfloat16 * __restrict__ captures,
        unsigned int batch, unsigned int tokens, unsigned int hidden_size,
        unsigned int hc, unsigned int post_layer, unsigned int slot) {
    if (batch == 0U || tokens == 0U ||
        hidden_size != GLM53_DFLASH2_HIDDEN || hc != GLM53_DFLASH2_HC ||
        !glm53_dflash2_capture_pair(post_layer, slot)) {
        return;
    }
    const unsigned long long rows = (unsigned long long) batch * tokens;
    const unsigned long long row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned long long input_base =
        row * GLM53_DFLASH2_HC * GLM53_DFLASH2_HIDDEN;
    const unsigned long long output_base =
        (row * GLM53_DFLASH2_CAPTURES + slot) * GLM53_DFLASH2_HIDDEN;
    for (unsigned int column = threadIdx.x; column < GLM53_DFLASH2_HIDDEN;
         column += GLM53_DFLASH2_THREADS) {
        float sum = 0.0f;
        #pragma unroll
        for (unsigned int stream = 0U; stream < GLM53_DFLASH2_HC; ++stream) {
            sum += __bfloat162float(
                streams[input_base +
                    (unsigned long long) stream * GLM53_DFLASH2_HIDDEN + column]);
        }
        captures[output_base + column] = __float2bfloat16_rn(sum * 0.25f);
    }
}
