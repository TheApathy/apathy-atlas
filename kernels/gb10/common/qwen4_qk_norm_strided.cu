// SPDX-License-Identifier: AGPL-3.0-only
//
// Per-head RMS norm over a strided [rows, row_stride] BF16 buffer: block
// (row, head) normalizes head_dim elements at x + row*row_stride + head*head_dim
// in place. Same arithmetic as `rms_norm` (FP32 sum of squares, rsqrtf,
// x * rms * (1 + w)), batched over all rows and heads in one launch.
#include <cuda_bf16.h>

extern "C" __global__ void qwen4_qk_norm_strided(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    unsigned int heads,
    unsigned int head_dim,
    unsigned int row_stride,
    float eps)
{
    const unsigned int row = blockIdx.x / heads;
    const unsigned int head = blockIdx.x - row * heads;
    __nv_bfloat16* v = x + (unsigned long long)row * row_stride + (unsigned long long)head * head_dim;
    const unsigned int tid = threadIdx.x;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < head_dim; i += blockDim.x) {
        const float val = __bfloat162float(v[i]);
        sum_sq += val * val;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFFu, sum_sq, offset);
    __shared__ float warp_sums[32];
    const unsigned int warp_id = tid / 32, lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();
    const float rms = rsqrtf(warp_sums[0] / (float)head_dim + eps);
    for (unsigned int i = tid; i < head_dim; i += blockDim.x) {
        const float val = __bfloat162float(v[i]);
        const float w = __bfloat162float(weight[i]);
        v[i] = __float2bfloat16(val * rms * (1.0f + w));
    }
}
