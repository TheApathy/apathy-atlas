// SPDX-License-Identifier: AGPL-3.0-only
// One-time Q6_K to BF16 materialization for the pinned GLM token embedding.
// Q6 index/arithmetic order follows ggml-org/llama.cpp dequantize_q6_K at
// ca3d5a3e10d53f7ea672cb9b6178faca3e2807bc.

#include <cuda_bf16.h>
#include "../../qwen3.6-27b/nvfp4/q4k_vendor/mmq.cuh"

extern "C" __global__ void __launch_bounds__(256, 1)
atlas_q6_k_embedding_bf16(
        const block_q6_K * __restrict__ source,
        __nv_bfloat16 * __restrict__ destination,
        unsigned int rows, unsigned int columns) {
    if (rows == 0 || columns == 0 || columns % QK_K != 0) {
        return;
    }
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned long long blocks_per_row = (unsigned int) columns / QK_K;
    for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int within = (int) (column % QK_K);
        const int half = within / 128;
        const int segment = (within % 128) / 32;
        const int lane = within % 32;
        const unsigned long long block_index =
            (unsigned long long) row * blocks_per_row + column / QK_K;
        const block_q6_K * block = source + block_index;
        const int low_index = 64 * half + lane + 32 * (segment & 1);
        const int low = (block->ql[low_index] >> (4 * (segment / 2))) & 0x0f;
        const int high = (block->qh[32 * half + lane] >> (2 * segment)) & 0x03;
        const int quant = (low | (high << 4)) - 32;
        const int scale_index = 8 * half + lane / 16 + 2 * segment;
        float value = __half2float(block->d);
        value *= (float) block->scales[scale_index];
        value *= (float) quant;
        destination[(unsigned long long) row * columns + column] =
            __float2bfloat16_rn(value);
    }
}
