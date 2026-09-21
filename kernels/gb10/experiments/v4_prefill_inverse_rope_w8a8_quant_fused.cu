// SPDX-License-Identifier: AGPL-3.0-only

// Compile-only DeepSeek-V4 experiment: inverse-rotate the trailing attention
// output channels and immediately quantize the full row for the already-
// eligible grouped wo_a W8A8-inplace arm. The BF16 attention input is not
// mutated. This file is deliberately unreachable from registry and serving.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define V4_INV_QUANT_NQ 64u
#define V4_INV_QUANT_HEAD_DIM 512u
#define V4_INV_QUANT_NOPE_DIM 448u
#define V4_INV_QUANT_ROPE_DIM 64u
#define V4_INV_QUANT_ROPE_PAIRS 32u
#define V4_INV_QUANT_TOTAL_PAIRS 2048u
#define V4_INV_QUANT_TAIL_VALUES 4096u
#define V4_INV_QUANT_ROW_WIDTH 32768u
#define V4_INV_QUANT_BLOCK 256u
#define V4_INV_QUANT_WARPS 8u
#define V4_INV_QUANT_SCALE_FLOOR 1.0e-8f
#define V4_INV_QUANT_E4M3_MAX 448.0f

static_assert(V4_INV_QUANT_NQ * V4_INV_QUANT_HEAD_DIM == V4_INV_QUANT_ROW_WIDTH,
              "the quantizer consumes one complete V4 attention-output row");
static_assert(V4_INV_QUANT_NOPE_DIM + V4_INV_QUANT_ROPE_DIM ==
                  V4_INV_QUANT_HEAD_DIM,
              "inverse RoPE owns the exact trailing channels");
static_assert(V4_INV_QUANT_NQ * V4_INV_QUANT_ROPE_PAIRS ==
                  V4_INV_QUANT_TOTAL_PAIRS,
              "every Q head contributes 32 inverse-RoPE pairs");
static_assert(2u * V4_INV_QUANT_TOTAL_PAIRS == V4_INV_QUANT_TAIL_VALUES,
              "shared memory preserves every BF16-rounded tail scalar");
static_assert(V4_INV_QUANT_ROW_WIDTH % V4_INV_QUANT_BLOCK == 0,
              "quantize_a_fp8_rows requires a complete scalar traversal");

__device__ __forceinline__ __nv_bfloat16 v4_inv_quant_load(
    const __nv_bfloat16* __restrict__ row,
    const __nv_bfloat16* __restrict__ rotated_tail,
    const unsigned int k) {
    const unsigned int head = k / V4_INV_QUANT_HEAD_DIM;
    const unsigned int dimension = k - head * V4_INV_QUANT_HEAD_DIM;
    if (dimension >= V4_INV_QUANT_NOPE_DIM) {
        return rotated_tail[head * V4_INV_QUANT_ROPE_DIM +
                            dimension - V4_INV_QUANT_NOPE_DIM];
    }
    return row[k];
}

// Applicability boundary for any future host dispatch: decide the existing
// w8a8_inplace eligibility before launch and require diag_this == false.
// Diagnostics currently inspect materialized inverse-rotated BF16 attn_out
// before wo_a; because this candidate leaves attn_out immutable, diagnostics
// and dumps must bypass it and run the incumbent inverse-RoPE + quantize chain.
// Grid: (num_tokens,1,1). Block: (256,1,1).
// input is immutable BF16 [N,64,512]. output_fp8 is E4M3 [N,32768],
// row_scale is FP32 [N]. One CTA owns a complete token row.
extern "C" __global__ __launch_bounds__(V4_INV_QUANT_BLOCK) void
v4_prefill_inverse_rope_w8a8_quant_fused(
    const __nv_bfloat16* __restrict__ input,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ inv_freq,
    unsigned char* __restrict__ output_fp8,
    float* __restrict__ row_scale,
    const unsigned int num_tokens,
    const unsigned int num_q_heads,
    const unsigned int head_dim,
    const unsigned int nope_dim,
    const unsigned int rotary_dim,
    const unsigned int row_width,
    const unsigned int quant_block,
    const float mscale) {
    if (num_tokens == 0 || input == nullptr || positions == nullptr ||
        inv_freq == nullptr || output_fp8 == nullptr || row_scale == nullptr ||
        reinterpret_cast<const void*>(input) ==
            reinterpret_cast<const void*>(output_fp8) ||
        num_q_heads != V4_INV_QUANT_NQ || head_dim != V4_INV_QUANT_HEAD_DIM ||
        nope_dim != V4_INV_QUANT_NOPE_DIM || rotary_dim != V4_INV_QUANT_ROPE_DIM ||
        row_width != V4_INV_QUANT_ROW_WIDTH || quant_block != V4_INV_QUANT_BLOCK ||
        !isfinite(mscale) || blockDim.x != V4_INV_QUANT_BLOCK || blockDim.y != 1 ||
        blockDim.z != 1 || gridDim.x != num_tokens || gridDim.y != 1 || gridDim.z != 1) {
        return;
    }

    const unsigned int token = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp_id = tid >> 5;
    const unsigned long long row_offset =
        static_cast<unsigned long long>(token) * V4_INV_QUANT_ROW_WIDTH;
    const __nv_bfloat16* const row = input + row_offset;

    __shared__ __align__(4) __nv_bfloat16 rotated_tail[V4_INV_QUANT_TAIL_VALUES];
    __shared__ float warp_max[V4_INV_QUANT_WARPS];

    // Direct inverse interleaved YaRN. Each thread owns eight of the 2,048
    // head-local pairs. Store the rounded result in the same BF16 boundary the
    // standalone inverse kernel would expose to quantize_a_fp8_rows.
    #pragma unroll 1
    for (unsigned int linear_pair = tid; linear_pair < V4_INV_QUANT_TOTAL_PAIRS;
         linear_pair += V4_INV_QUANT_BLOCK) {
        const unsigned int head = linear_pair / V4_INV_QUANT_ROPE_PAIRS;
        const unsigned int pair = linear_pair - head * V4_INV_QUANT_ROPE_PAIRS;
        const unsigned int input_index =
            head * V4_INV_QUANT_HEAD_DIM + V4_INV_QUANT_NOPE_DIM + 2u * pair;
        const float x0 = __bfloat162float(row[input_index]);
        const float x1 = __bfloat162float(row[input_index + 1u]);
        const float angle = static_cast<float>(positions[token]) * inv_freq[pair];
        const float cos_value = cosf(angle) * mscale;
        const float sin_value = sinf(angle) * mscale;
        rotated_tail[2u * linear_pair] =
            __float2bfloat16(x0 * cos_value + x1 * sin_value);
        rotated_tail[2u * linear_pair + 1u] =
            __float2bfloat16(x1 * cos_value - x0 * sin_value);
    }
    __syncthreads();

    // Verbatim scalar lane order from quantize_a_fp8_rows: each thread visits
    // k=tid+j*256 in increasing j. Tail loads come from the BF16 shared seam;
    // the 448 non-RoPE channels per head remain direct immutable input loads.
    float amax = 0.0f;
    #pragma unroll 1
    for (unsigned int k = tid; k < V4_INV_QUANT_ROW_WIDTH;
         k += V4_INV_QUANT_BLOCK) {
        const __nv_bfloat16 value = v4_inv_quant_load(row, rotated_tail, k);
        amax = fmaxf(amax, fabsf(__bfloat162float(value)));
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xFFFFFFFFu, amax, offset));
    }
    if (lane == 0) {
        warp_max[warp_id] = amax;
    }
    __syncthreads();
    if (tid == 0) {
        float block_max = warp_max[0];
        #pragma unroll
        for (unsigned int warp = 1; warp < V4_INV_QUANT_WARPS; ++warp) {
            block_max = fmaxf(block_max, warp_max[warp]);
        }
        warp_max[0] =
            fmaxf(block_max, V4_INV_QUANT_SCALE_FLOOR) / V4_INV_QUANT_E4M3_MAX;
        row_scale[token] = warp_max[0];
    }
    __syncthreads();

    const float inverse_scale = 1.0f / warp_max[0];
    unsigned char* const output = output_fp8 + row_offset;
    #pragma unroll 1
    for (unsigned int k = tid; k < V4_INV_QUANT_ROW_WIDTH;
         k += V4_INV_QUANT_BLOCK) {
        const __nv_bfloat16 value = v4_inv_quant_load(row, rotated_tail, k);
        const __nv_fp8_e4m3 quantized(__bfloat162float(value) * inverse_scale);
        output[k] = *reinterpret_cast<const unsigned char*>(&quantized);
    }
}
