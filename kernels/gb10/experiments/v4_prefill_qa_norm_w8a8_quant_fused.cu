// SPDX-License-Identifier: AGPL-3.0-only

// Compile-only DeepSeek-V4 experiment: fuse the weighted q_a RMSNorm into the
// row-scaled E4M3 activation quantizer used by the already-eligible W8A8 wq_b
// projection. This file is deliberately unreachable from the kernel registry.
//
// The incumbent kernels use incompatible lane ownership:
//   rms_norm_vanilla: one packed BF16 pair for each of tids 0..511 in a
//                     1024-thread CTA
//   quantize_a_fp8_rows: scalar columns k = tid + j * 256
// A 1024-element BF16 shared row is the exact hand-off between those mappings.
// It retains both reduction trees and the incumbent BF16 store/load boundary
// without materializing that row in global memory.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define V4_QA_DIM 1024u
#define V4_QA_BLOCK 1024u
#define V4_QA_RMS_WARPS (V4_QA_BLOCK / 32u)
#define V4_QA_QUANT_BLOCK 256u
#define V4_QA_QUANT_WARPS (V4_QA_QUANT_BLOCK / 32u)
#define V4_QA_E4M3_MAX 448.0f
#define V4_QA_SCALE_FLOOR 1.0e-8f

static_assert(V4_QA_DIM == V4_QA_BLOCK,
              "the incumbent RMSNorm launches one thread per q_a dimension");
static_assert(V4_QA_DIM == 4u * V4_QA_QUANT_BLOCK,
              "one quantizer thread must own four columns");
static_assert(V4_QA_RMS_WARPS == 32u, "the incumbent RMSNorm uses 32 warps");
static_assert(V4_QA_QUANT_WARPS == 8u, "the incumbent quantizer uses eight warps");

__device__ __forceinline__ void v4_qa_unpack_bf16x2(
    const unsigned int packed,
    float& first,
    float& second) {
    first = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(packed)));
    second = __bfloat162float(
        __ushort_as_bfloat16(static_cast<unsigned short>(packed >> 16)));
}

__device__ __forceinline__ float v4_qa_warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xFFFFFFFFu, value, offset);
    }
    return value;
}

// Grid: (num_tokens, 1, 1). Block: (1024, 1, 1).
// input/weight are BF16 [num_tokens,1024]/[1024]; output is E4M3 bytes
// [num_tokens,1024] with one FP32 dequantization scale per row.
extern "C" __global__ __launch_bounds__(V4_QA_BLOCK) void
v4_prefill_qa_norm_w8a8_quant_fused(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    unsigned char* __restrict__ output_fp8,
    float* __restrict__ row_scale,
    const unsigned int num_tokens,
    const unsigned int hidden_size,
    const float eps) {
    if (num_tokens == 0 || input == nullptr || weight == nullptr ||
        output_fp8 == nullptr || row_scale == nullptr ||
        hidden_size != V4_QA_DIM || !isfinite(eps) || eps <= 0.0f ||
        blockDim.x != V4_QA_BLOCK || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.x != num_tokens || gridDim.y != 1 || gridDim.z != 1) {
        return;
    }

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int token = blockIdx.x;
    const unsigned long long row_offset =
        static_cast<unsigned long long>(token) * V4_QA_DIM;
    const __nv_bfloat16* const row = input + row_offset;

    // Match rms_norm_vanilla's packed-pair loop at blockDim.x == 1024.
    const unsigned int pair0 = tid;
    const unsigned int* const row32 = reinterpret_cast<const unsigned int*>(row);
    float x0 = 0.0f;
    float x1 = 0.0f;
    float sum_sq = 0.0f;
    if (pair0 < V4_QA_DIM / 2u) {
        v4_qa_unpack_bf16x2(row32[pair0], x0, x1);
        sum_sq += x0 * x0 + x1 * x1;
    }
    sum_sq = v4_qa_warp_sum(sum_sq);

    __shared__ float reduction[32];
    __shared__ __align__(4) __nv_bfloat16 normed[V4_QA_DIM];
    if (lane == 0) {
        reduction[warp] = sum_sq;
    }
    __syncthreads();
    if (warp == 0) {
        float value = reduction[lane];
        value = v4_qa_warp_sum(value);
        if (lane == 0) {
            reduction[0] = value;
        }
    }
    __syncthreads();

    const float rms = rsqrtf(reduction[0] / static_cast<float>(V4_QA_DIM) + eps);
    const unsigned int* const weight32 =
        reinterpret_cast<const unsigned int*>(weight);
    if (pair0 < V4_QA_DIM / 2u) {
        float w0, w1;
        v4_qa_unpack_bf16x2(weight32[pair0], w0, w1);
        // Round exactly where the standalone RMSNorm writes BF16. The shared
        // row changes ownership without changing row-major byte layout.
        normed[2u * pair0] = __float2bfloat16(x0 * rms * w0);
        normed[2u * pair0 + 1u] = __float2bfloat16(x1 * rms * w1);
    }
    __syncthreads();

    // Match quantize_a_fp8_rows' scalar, four-column lane ownership and its
    // down-shuffle + serial warp-max reduction exactly.
    const unsigned int q0 = tid;
    const unsigned int q1 = tid + V4_QA_QUANT_BLOCK;
    const unsigned int q2 = tid + 2u * V4_QA_QUANT_BLOCK;
    const unsigned int q3 = tid + 3u * V4_QA_QUANT_BLOCK;
    unsigned short y0_bits = 0;
    unsigned short y1_bits = 0;
    unsigned short y2_bits = 0;
    unsigned short y3_bits = 0;
    float amax = 0.0f;
    if (tid < V4_QA_QUANT_BLOCK) {
        y0_bits = __bfloat16_as_ushort(normed[q0]);
        y1_bits = __bfloat16_as_ushort(normed[q1]);
        y2_bits = __bfloat16_as_ushort(normed[q2]);
        y3_bits = __bfloat16_as_ushort(normed[q3]);
        amax = fmaxf(amax, fabsf(__bfloat162float(__ushort_as_bfloat16(y0_bits))));
        amax = fmaxf(amax, fabsf(__bfloat162float(__ushort_as_bfloat16(y1_bits))));
        amax = fmaxf(amax, fabsf(__bfloat162float(__ushort_as_bfloat16(y2_bits))));
        amax = fmaxf(amax, fabsf(__bfloat162float(__ushort_as_bfloat16(y3_bits))));
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_down_sync(0xFFFFFFFFu, amax, offset));
        }
        if (lane == 0) {
            reduction[warp] = amax;
        }
    }
    __syncthreads();
    if (tid == 0) {
        float block_max = reduction[0];
        #pragma unroll
        for (unsigned int w = 1; w < V4_QA_QUANT_WARPS; ++w) {
            block_max = fmaxf(block_max, reduction[w]);
        }
        reduction[0] = fmaxf(block_max, V4_QA_SCALE_FLOOR) / V4_QA_E4M3_MAX;
        row_scale[token] = reduction[0];
    }
    __syncthreads();

    if (tid < V4_QA_QUANT_BLOCK) {
        const float inverse_scale = 1.0f / reduction[0];
        unsigned char* const output = output_fp8 + row_offset;
        const __nv_fp8_e4m3 out0(
            __bfloat162float(__ushort_as_bfloat16(y0_bits)) * inverse_scale);
        const __nv_fp8_e4m3 out1(
            __bfloat162float(__ushort_as_bfloat16(y1_bits)) * inverse_scale);
        const __nv_fp8_e4m3 out2(
            __bfloat162float(__ushort_as_bfloat16(y2_bits)) * inverse_scale);
        const __nv_fp8_e4m3 out3(
            __bfloat162float(__ushort_as_bfloat16(y3_bits)) * inverse_scale);
        output[q0] = *reinterpret_cast<const unsigned char*>(&out0);
        output[q1] = *reinterpret_cast<const unsigned char*>(&out1);
        output[q2] = *reinterpret_cast<const unsigned char*>(&out2);
        output[q3] = *reinterpret_cast<const unsigned char*>(&out3);
    }
}
