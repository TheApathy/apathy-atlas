// SPDX-License-Identifier: AGPL-3.0-only
// Exact GLM-5.3 DSA norms and F32 index projection to BF16.

#include <cuda_bf16.h>
#include <math.h>

#define GLM53_KV_RANK 512U
#define GLM53_Q_RANK 1536U
#define GLM53_INDEX_DIM 128U
#define GLM53_HIDDEN 4096U
#define GLM53_INDEX_HEADS 32U
#define GLM53_MAX_ROWS 65520U

__device__ __forceinline__ float glm53_dsa_sum_256(
        float value, float *scratch, unsigned int tid) {
    scratch[tid] = value;
    __syncthreads();
    for (unsigned int stride = 128U; stride > 0U; stride >>= 1U) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

extern "C" __global__ void __launch_bounds__(256, 1)
atlas_glm53_dsa_absolute_rms_norm_bf16(
        const __nv_bfloat16 * __restrict__ input,
        const float * __restrict__ weight,
        __nv_bfloat16 * __restrict__ output,
        unsigned int rows, unsigned int width, float eps) {
    if (blockDim.x != 256U || blockDim.y != 1U || blockDim.z != 1U ||
        gridDim.x != rows || gridDim.y != 1U || gridDim.z != 1U ||
        rows == 0U || rows > GLM53_MAX_ROWS ||
        (width != GLM53_KV_RANK && width != GLM53_Q_RANK) ||
        eps != 1.0e-5f) {
        return;
    }
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int tid = threadIdx.x;
    const unsigned long long base = (unsigned long long) row * width;
    float square_sum = 0.0f;
    for (unsigned int column = tid; column < width; column += 256U) {
        const float value = __bfloat162float(input[base + column]);
        square_sum = fmaf(value, value, square_sum);
    }
    __shared__ float scratch[256];
    const float total = glm53_dsa_sum_256(square_sum, scratch, tid);
    __shared__ float inverse_rms;
    if (tid == 0U) {
        inverse_rms = rsqrtf(total / (float) width + eps);
    }
    __syncthreads();
    for (unsigned int column = tid; column < width; column += 256U) {
        const float normalized =
            __bfloat162float(input[base + column]) * inverse_rms;
        const __nv_bfloat16 normalized_bf16 = __float2bfloat16_rn(normalized);
        const __nv_bfloat16 weight_bf16 = __float2bfloat16_rn(weight[column]);
        output[base + column] = __float2bfloat16_rn(
            __bfloat162float(normalized_bf16) * __bfloat162float(weight_bf16));
    }
}

extern "C" __global__ void __launch_bounds__(128, 1)
atlas_glm53_dsa_biased_layer_norm_bf16(
        const __nv_bfloat16 * __restrict__ input,
        const float * __restrict__ weight,
        const float * __restrict__ bias,
        __nv_bfloat16 * __restrict__ output,
        unsigned int rows, unsigned int width, float eps) {
    if (blockDim.x != GLM53_INDEX_DIM || blockDim.y != 1U || blockDim.z != 1U ||
        gridDim.x != rows || gridDim.y != 1U || gridDim.z != 1U ||
        rows == 0U || rows > GLM53_MAX_ROWS ||
        width != GLM53_INDEX_DIM || eps != 1.0e-6f) {
        return;
    }
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int column = threadIdx.x;
    const unsigned long long base =
        (unsigned long long) row * GLM53_INDEX_DIM;
    const float value = __bfloat162float(input[base + column]);
    __shared__ float scratch[GLM53_INDEX_DIM];
    scratch[column] = value;
    __syncthreads();
    for (unsigned int stride = 64U; stride > 0U; stride >>= 1U) {
        if (column < stride) {
            scratch[column] += scratch[column + stride];
        }
        __syncthreads();
    }
    __shared__ float mean;
    if (column == 0U) {
        mean = scratch[0] / (float) GLM53_INDEX_DIM;
    }
    __syncthreads();
    const float centered = value - mean;
    scratch[column] = centered * centered;
    __syncthreads();
    for (unsigned int stride = 64U; stride > 0U; stride >>= 1U) {
        if (column < stride) {
            scratch[column] += scratch[column + stride];
        }
        __syncthreads();
    }
    __shared__ float inverse_std;
    if (column == 0U) {
        inverse_std = rsqrtf(
            scratch[0] / (float) GLM53_INDEX_DIM + eps);
    }
    __syncthreads();
    const float normalized = centered * inverse_std;
    output[base + column] =
        __float2bfloat16_rn(fmaf(normalized, weight[column], bias[column]));
}

extern "C" __global__ void __launch_bounds__(32, 1)
atlas_glm53_dsa_index_projection_f32_bf16(
        const __nv_bfloat16 * __restrict__ input,
        const float * __restrict__ weight,
        __nv_bfloat16 * __restrict__ output,
        unsigned int rows, unsigned int inner, unsigned int heads) {
    if (blockDim.x != GLM53_INDEX_HEADS || blockDim.y != 1U || blockDim.z != 1U ||
        gridDim.x != rows || gridDim.y != 1U || gridDim.z != 1U ||
        rows == 0U || rows > GLM53_MAX_ROWS ||
        inner != GLM53_HIDDEN || heads != GLM53_INDEX_HEADS) {
        return;
    }
    const unsigned int row = blockIdx.x;
    const unsigned int head = threadIdx.x;
    if (row >= rows || head >= heads) {
        return;
    }
    const unsigned long long input_base =
        (unsigned long long) row * GLM53_HIDDEN;
    float sum = 0.0f;
    for (unsigned int column = 0U; column < GLM53_HIDDEN; ++column) {
        const float input_value = __bfloat162float(input[input_base + column]);
        const float weight_value = weight[
            (unsigned long long) head * GLM53_HIDDEN + column];
        sum = fmaf(weight_value, input_value, sum);
    }
    output[(unsigned long long) row * GLM53_INDEX_HEADS + head] =
        __float2bfloat16_rn(sum);
}

// ---------------------------------------------------------------------------
// Wide DSA index projection (ATLAS_GLM53_DSA_INDEX_PROJ_WIDE=1), bit-exact.
//
// `atlas_glm53_dsa_index_projection_f32_bf16` launches one block per ROW with
// 32 threads, one thread per head. At decode rows==1, so a [32,4096]x[4096]
// f32 GEMV -- 512 KiB of `weight` -- is computed by a SINGLE WARP on a single
// SM, and the per-thread stride of 4096 floats means no two lanes ever share a
// cache line. Measured on GB10: 183.1 us per launch, 11 launches per token,
// 2.0 ms of a 66.6 ms token, against a ~0.02 ms bandwidth floor.
//
// Here each (row, head) gets its OWN block, so the 32 heads occupy 32 SMs, and
// within a block the warp prefetches each tile cooperatively (coalesced) before
// the accumulating lane walks it.
//
// EXACTNESS. Lane 0 performs exactly the original serial chain: columns 0..4095
// ascending, `sum = fmaf(weight_value, input_value, sum)`, same operand order,
// same bf16->f32 conversion of the input. Staging the operands through shared
// memory does not change any value, and no partial sums are ever combined.
// ---------------------------------------------------------------------------

#define GLM53_INDEX_PROJ_TILE 512U

extern "C" __global__ void __launch_bounds__(32, 1)
atlas_glm53_dsa_index_projection_f32_bf16_wide(
        const __nv_bfloat16 * __restrict__ input,
        const float * __restrict__ weight,
        __nv_bfloat16 * __restrict__ output,
        unsigned int rows, unsigned int inner, unsigned int heads) {
    if (blockDim.x != 32U || blockDim.y != 1U || blockDim.z != 1U ||
        gridDim.x != heads || gridDim.y != rows || gridDim.z != 1U ||
        rows == 0U || rows > GLM53_MAX_ROWS ||
        inner != GLM53_HIDDEN || heads != GLM53_INDEX_HEADS) {
        return;
    }
    const unsigned int head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int lane = threadIdx.x;

    __shared__ float tile_weight[GLM53_INDEX_PROJ_TILE];
    __shared__ float tile_input[GLM53_INDEX_PROJ_TILE];

    const unsigned long long input_base = (unsigned long long) row * GLM53_HIDDEN;
    const unsigned long long weight_base = (unsigned long long) head * GLM53_HIDDEN;

    float sum = 0.0f;
    for (unsigned int tile = 0U; tile < GLM53_HIDDEN;
         tile += GLM53_INDEX_PROJ_TILE) {
        for (unsigned int slot = lane; slot < GLM53_INDEX_PROJ_TILE; slot += 32U) {
            tile_weight[slot] = weight[weight_base + tile + slot];
            tile_input[slot] =
                __bfloat162float(input[input_base + tile + slot]);
        }
        __syncwarp();
        if (lane == 0U) {
            for (unsigned int slot = 0U; slot < GLM53_INDEX_PROJ_TILE; ++slot) {
                sum = fmaf(tile_weight[slot], tile_input[slot], sum);
            }
        }
        __syncwarp();
    }
    if (lane == 0U) {
        output[(unsigned long long) row * GLM53_INDEX_HEADS + head] =
            __float2bfloat16_rn(sum);
    }
}
