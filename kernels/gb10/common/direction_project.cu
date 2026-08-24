// SPDX-License-Identifier: AGPL-3.0-only

// Directional projection on the residual stream:
//
//     h' = h - alpha * (h . d_hat) * d_hat
//
// Removes the component of each row of the residual stream that lies along a
// unit direction `d_hat`. This is the runtime form of a rank-1 weight edit:
// projecting the activation is arithmetically identical to editing every
// matrix that writes into the residual stream with
//
//     dW = -alpha * d_hat (d_hat^T W)
//
// so the modification can ship as a small vector (hidden_size floats) applied
// at load/serve time instead of a redistributed checkpoint.
//
// The operation is self-limiting: a row carrying no component along `d_hat`
// has zero subtracted from it and is bit-identical on output. That property is
// what makes it safe to apply unconditionally across every row — rows the
// direction does not describe are untouched rather than perturbed.
//
// One block per row, blockDim.x threads striding the hidden dimension. The dot
// product is reduced in FP32 regardless of storage dtype: the residual stream
// is the accumulation path, and reducing it in BF16 would lose more precision
// than the projection removes.

#include <cuda_bf16.h>

// Block-wide sum reduction into lane 0 via shared memory.
// `smem` must hold at least blockDim.x floats.
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    const unsigned int t = threadIdx.x;
    smem[t] = v;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (t < stride) {
            smem[t] += smem[t + stride];
        }
        __syncthreads();
    }
    return smem[0];
}

// BF16 residual stream, in-place.
//
//   hidden : [rows, hidden_size] BF16, modified in place
//   d_hat  : [hidden_size] FP32, MUST be L2-normalised by the caller
//   alpha  : projection strength; 1.0 removes the component entirely
extern "C" __global__ void bf16_direction_project(
    __nv_bfloat16* __restrict__ hidden,
    const float* __restrict__ d_hat,
    float alpha,
    unsigned int hidden_size
) {
    extern __shared__ float smem[];

    __nv_bfloat16* row = hidden + (size_t)blockIdx.x * hidden_size;

    // dot = h . d_hat, accumulated in FP32.
    float partial = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        partial = fmaf(__bfloat162float(row[i]), d_hat[i], partial);
    }
    const float dot = block_reduce_sum(partial, smem);

    // h -= alpha * dot * d_hat
    const float scale = alpha * dot;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float h = __bfloat162float(row[i]);
        row[i] = __float2bfloat16(h - scale * d_hat[i]);
    }
}

// FP32 residual stream, in-place. Same contract as the BF16 variant.
extern "C" __global__ void f32_direction_project(
    float* __restrict__ hidden,
    const float* __restrict__ d_hat,
    float alpha,
    unsigned int hidden_size
) {
    extern __shared__ float smem[];

    float* row = hidden + (size_t)blockIdx.x * hidden_size;

    float partial = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        partial = fmaf(row[i], d_hat[i], partial);
    }
    const float dot = block_reduce_sum(partial, smem);

    const float scale = alpha * dot;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row[i] -= scale * d_hat[i];
    }
}
