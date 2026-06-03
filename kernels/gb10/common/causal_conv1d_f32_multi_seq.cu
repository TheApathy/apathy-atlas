// SPDX-License-Identifier: AGPL-3.0-only
//
// FP32-output variants of the multi-seq conv1d kernels.
//
// Same compute as `causal_conv1d_update_l2norm_multi_seq` but writes
// FP32 output instead of BF16. This is the precision-preserving variant
// needed to wire ATLAS_SSM_MULTI_SEQ_KERNEL into the production decode
// path on AEON-Q36-27B — the single-seq path uses FP32 output via
// `causal_conv1d_update_l2norm_f32`, and the model is calibrated for
// that precision (BF16 truncation in the recurrent path compounds to
// noise at 8k+ tokens; the FP32 variant preserves the
// gated_delta_rule's numerical regime).
//
// Strides: `input_stride` is BF16 ELEMENTS, `output_stride` is FP32
// ELEMENTS, between successive sequences.
//
// Grid: (ceil(dim/256), num_seqs, 1)  Block: (256, 1, 1)

#include <cuda_bf16.h>

// ============================================================
// DECODE multi-seq: conv1d sliding window update + SiLU (FP32 output).
// ============================================================
extern "C" __global__ void causal_conv1d_update_f32_multi_seq(
    float* const* __restrict__ conv_states,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    unsigned int num_seqs,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int input_stride,   // BF16 elements between seqs in new_input
    unsigned int output_stride   // FP32 elements between seqs in output
) {
    const unsigned int ch  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int seq = blockIdx.y;
    if (ch >= dim || seq >= num_seqs) return;

    float* state = conv_states[seq] + ch * d_conv;

    for (unsigned int i = 0; i < d_conv - 1; i++) {
        state[i] = state[i + 1];
    }
    state[d_conv - 1] = (float)new_input[(unsigned long long)seq * input_stride + ch];

    const __nv_bfloat16* w = weight + ch * d_conv;
    float acc = (bias != nullptr) ? bias[ch] : 0.0f;
    for (unsigned int k = 0; k < d_conv; k++) {
        acc += state[k] * (float)w[k];
    }

    float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
    float silu = acc * sigmoid_acc;

    output[(unsigned long long)seq * output_stride + ch] = silu;
}

// ============================================================
// DECODE multi-seq: conv1d + SiLU + L2-norm for Q/K channels (FP32 output).
// ============================================================
extern "C" __global__ void causal_conv1d_update_l2norm_f32_multi_seq(
    float* const* __restrict__ conv_states,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    unsigned int num_seqs,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps,
    unsigned int input_stride,   // BF16 elements between seqs in new_input
    unsigned int output_stride   // FP32 elements between seqs in output
) {
    const unsigned int ch  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int seq = blockIdx.y;
    const unsigned int tid = threadIdx.x;

    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = (block_start < qk_channels);

    const bool valid = (ch < dim && seq < num_seqs);
    float silu = 0.0f;

    if (valid) {
        float* state = conv_states[seq] + ch * d_conv;

        for (unsigned int i = 0; i < d_conv - 1; i++)
            state[i] = state[i + 1];
        state[d_conv - 1] = (float)new_input[(unsigned long long)seq * input_stride + ch];

        const __nv_bfloat16* w = weight + ch * d_conv;
        float acc = (bias != nullptr) ? bias[ch] : 0.0f;
        for (unsigned int k = 0; k < d_conv; k++)
            acc += state[k] * (float)w[k];

        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        silu = acc * sigmoid_acc;
    }

    if (block_needs_l2) {
        float sq = valid ? (silu * silu) : 0.0f;

        const unsigned int warp_id = tid / 32;
        const unsigned int lane = tid % 32;
        for (int offset = 16; offset >= 1; offset >>= 1)
            sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);

        __shared__ float warp_sums[8];
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

        if (valid) {
            silu *= warp_sums[base_warp];
        }
    }

    if (valid) {
        output[(unsigned long long)seq * output_stride + ch] = silu;
    }
}
