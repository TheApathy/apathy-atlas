// SPDX-License-Identifier: AGPL-3.0-only

// Compile-only DeepSeek-V4 experiment: apply interleaved YaRN RoPE directly to
// the trailing channels of the resident Q/K or attention-output tensors.
// This file is deliberately unreachable from the kernel registry and serving.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define V4_ROPE_NQ 64
#define V4_ROPE_NKV 1
#define V4_ROPE_HEAD_DIM 512
#define V4_ROPE_NOPE_DIM 448
#define V4_ROPE_DIM 64
#define V4_ROPE_PAIRS (V4_ROPE_DIM / 2)

static_assert(V4_ROPE_NOPE_DIM + V4_ROPE_DIM == V4_ROPE_HEAD_DIM,
              "RoPE must occupy the exact trailing DeepSeek-V4 channels");
static_assert(V4_ROPE_PAIRS == 32, "one warp must own one head's RoPE pairs");

__device__ __forceinline__ void v4_rope_pair_forward(
    __nv_bfloat16* __restrict__ ptr,
    const unsigned int pair_idx,
    const unsigned int abs_pos,
    const float* __restrict__ inv_freq,
    const float mscale) {
    const float freq = inv_freq[pair_idx];
    const float angle = static_cast<float>(abs_pos) * freq;
    const float cos_val = cosf(angle) * mscale;
    const float sin_val = sinf(angle) * mscale;
    const unsigned int d0 = V4_ROPE_NOPE_DIM + 2 * pair_idx;
    const unsigned int d1 = d0 + 1;
    const float x0 = static_cast<float>(ptr[d0]);
    const float x1 = static_cast<float>(ptr[d1]);
    const float y0 = x0 * cos_val - x1 * sin_val;
    const float y1 = x1 * cos_val + x0 * sin_val;
    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}

__device__ __forceinline__ void v4_rope_pair_inverse(
    __nv_bfloat16* __restrict__ ptr,
    const unsigned int pair_idx,
    const unsigned int abs_pos,
    const float* __restrict__ inv_freq,
    const float mscale) {
    const float freq = inv_freq[pair_idx];
    const float angle = static_cast<float>(abs_pos) * freq;
    const float cos_val = cosf(angle) * mscale;
    const float sin_val = sinf(angle) * mscale;
    const unsigned int d0 = V4_ROPE_NOPE_DIM + 2 * pair_idx;
    const unsigned int d1 = d0 + 1;
    const float x0 = static_cast<float>(ptr[d0]);
    const float x1 = static_cast<float>(ptr[d1]);
    const float y0 = x0 * cos_val + x1 * sin_val;
    const float y1 = x1 * cos_val - x0 * sin_val;
    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}

// Grid (num_tokens, 65, 1), block (32, 1, 1). blockIdx.y 0..63 owns Q;
// blockIdx.y 64 owns K. Q/K are [num_tokens, heads, 512] BF16 tensors.
extern "C" __global__ __launch_bounds__(32) void v4_prefill_rope_fused_forward(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ inv_freq,
    const unsigned int num_tokens,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int nope_dim,
    const unsigned int rotary_dim,
    const float mscale) {
    if (num_tokens == 0 || Q == nullptr || K == nullptr || positions == nullptr ||
        inv_freq == nullptr || Q == K || !isfinite(mscale) || mscale <= 0.0f ||
        num_q_heads != V4_ROPE_NQ ||
        num_kv_heads != V4_ROPE_NKV || head_dim != V4_ROPE_HEAD_DIM ||
        nope_dim != V4_ROPE_NOPE_DIM || rotary_dim != V4_ROPE_DIM ||
        blockDim.x != 32 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.x != num_tokens || gridDim.y != V4_ROPE_NQ + V4_ROPE_NKV ||
        gridDim.z != 1) {
        return;
    }

    const unsigned int token = blockIdx.x;
    const unsigned int head_slot = blockIdx.y;
    const unsigned int pair_idx = threadIdx.x;
    const bool is_q = head_slot < V4_ROPE_NQ;
    const unsigned int head = is_q ? head_slot : 0;
    const unsigned long long tensor_heads = is_q ? V4_ROPE_NQ : V4_ROPE_NKV;
    __nv_bfloat16* const base = is_q ? Q : K;
    const unsigned long long offset =
        (static_cast<unsigned long long>(token) * tensor_heads + head) * V4_ROPE_HEAD_DIM;
    v4_rope_pair_forward(base + offset, pair_idx, positions[token], inv_freq, mscale);
}

// Grid (num_tokens, 64, 1), block (32, 1, 1). This conjugate rotation is
// applied directly to [num_tokens, 64, 512] attention output. num_kv_heads=0
// distinguishes the inverse ABI and fails closed if the forward shape is used.
extern "C" __global__ __launch_bounds__(32) void v4_prefill_rope_fused_inverse(
    __nv_bfloat16* __restrict__ Q,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ inv_freq,
    const unsigned int num_tokens,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int nope_dim,
    const unsigned int rotary_dim,
    const float mscale) {
    if (num_tokens == 0 || Q == nullptr || positions == nullptr || inv_freq == nullptr ||
        !isfinite(mscale) || mscale <= 0.0f || num_q_heads != V4_ROPE_NQ ||
        num_kv_heads != 0 ||
        head_dim != V4_ROPE_HEAD_DIM || nope_dim != V4_ROPE_NOPE_DIM ||
        rotary_dim != V4_ROPE_DIM || blockDim.x != 32 || blockDim.y != 1 ||
        blockDim.z != 1 || gridDim.x != num_tokens || gridDim.y != V4_ROPE_NQ ||
        gridDim.z != 1) {
        return;
    }

    const unsigned int token = blockIdx.x;
    const unsigned int head = blockIdx.y;
    const unsigned int pair_idx = threadIdx.x;
    const unsigned long long offset =
        (static_cast<unsigned long long>(token) * V4_ROPE_NQ + head) * V4_ROPE_HEAD_DIM;
    v4_rope_pair_inverse(Q + offset, pair_idx, positions[token], inv_freq, mscale);
}
