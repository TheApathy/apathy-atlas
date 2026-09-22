// SPDX-License-Identifier: AGPL-3.0-only
//
// Token-parallel twin of `causal_conv1d_update_l2norm_f32_sequence`. The
// depthwise conv has no recurrence beyond a d_conv-deep shift register, so
// every (token, channel) output depends only on the d_conv most recent
// inputs: identical per-element arithmetic (bias + ordered k-loop, __expf
// SiLU, per-head L2 over the same warp-shuffle / 4-warp reduction order) run
// with one block per (channel block, token) instead of one block looping
// over all tokens. The final conv_state is committed by a second kernel
// after every reader is done. Bit-exact with the sequence kernel.
//
// Grid: (ceil(dim/256), num_tokens)  Block: (256, 1, 1)   [2 heads of 128]
#include <cuda_bf16.h>

extern "C" __global__ void causal_conv1d_update_l2norm_f32_prefill_parallel(
    const float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    unsigned int num_tokens,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps,
    unsigned int input_stride,
    unsigned int output_stride
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int token = blockIdx.y;
    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = block_start < qk_channels;
    const bool valid = ch < dim && token < num_tokens;
    __shared__ float warp_sums[8];

    float silu = 0.0f;
    if (valid) {
        // state[k] == input at token (token - (d_conv-1) + k); negative
        // indices come from the pre-chunk shift register (old state[k+token+1]).
        float state[8];
        for (unsigned int k = 0; k < d_conv; k++) {
            const int idx = (int)token - (int)(d_conv - 1) + (int)k;
            if (idx >= 0) {
                state[k] = (float)new_input[(unsigned long long)idx * input_stride + ch];
            } else {
                // old register after `token+1` shifts: old[k + token + 1]
                state[k] = conv_state[(unsigned long long)ch * d_conv + (k + token + 1)];
            }
        }
        const __nv_bfloat16* w = weight + (unsigned long long)ch * d_conv;
        float acc = (bias != nullptr) ? bias[ch] : 0.0f;
        for (unsigned int k = 0; k < d_conv; k++)
            acc += state[k] * (float)w[k];
        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        silu = acc * sigmoid_acc;
    }

    if (block_needs_l2) {
        float sq = valid ? silu * silu : 0.0f;
        const unsigned int warp_id = tid / 32;
        const unsigned int lane = tid % 32;
        for (int offset = 16; offset >= 1; offset >>= 1)
            sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
        if (lane == 0) warp_sums[warp_id] = sq;
        __syncthreads();
        const unsigned int head_in_block = tid / head_dim;
        const unsigned int base_warp = head_in_block * (head_dim / 32);
        if (tid == 0 || tid == head_dim) {
            float total = warp_sums[base_warp] + warp_sums[base_warp + 1]
                        + warp_sums[base_warp + 2] + warp_sums[base_warp + 3];
            warp_sums[base_warp] = rsqrtf(total + l2_eps);
        }
        __syncthreads();
        if (valid) silu *= warp_sums[base_warp];
    }

    if (valid) {
        output[(unsigned long long)token * output_stride + ch] = silu;
    }
}

// Commits the post-chunk shift register: state[k] = input[T - d_conv + k],
// or the old register shifted by T when T < d_conv. Grid: ceil(dim/256).
extern "C" __global__ void causal_conv1d_prefill_state_commit(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    unsigned int num_tokens,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int input_stride
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;
    float* state = conv_state + (unsigned long long)ch * d_conv;
    float old[8];
    for (unsigned int k = 0; k < d_conv; k++) old[k] = state[k];
    for (unsigned int k = 0; k < d_conv; k++) {
        const int idx = (int)num_tokens - (int)d_conv + (int)k;
        state[k] = idx >= 0
            ? (float)new_input[(unsigned long long)idx * input_stride + ch]
            : old[k + num_tokens];
    }
}
