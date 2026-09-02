// SPDX-License-Identifier: AGPL-3.0-only
// One-time GGML Q8_0 to F32 materialization for GLM mHC projections.

#include <cuda_fp16.h>
#define GGML_COMMON_DECL_CUDA
#include "../../qwen3.6-27b/nvfp4/q4k_vendor/ggml-common.h"

extern "C" __global__ void __launch_bounds__(256, 1)
atlas_q8_0_to_f32(
        const block_q8_0 * __restrict__ source,
        float * __restrict__ destination,
        unsigned int rows, unsigned int columns) {
    if (rows == 0 || columns == 0 || columns % QK8_0 != 0) {
        return;
    }
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned long long blocks_per_row = columns / QK8_0;
    for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
        const unsigned long long block_index =
            (unsigned long long) row * blocks_per_row + column / QK8_0;
        const block_q8_0 * block = source + block_index;
        destination[(unsigned long long) row * columns + column] =
            __half2float(block->d) * (float) block->qs[column % QK8_0];
    }
}
