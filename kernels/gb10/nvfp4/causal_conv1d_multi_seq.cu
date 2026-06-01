// SPDX-License-Identifier: AGPL-3.0-only

// Multi-sequence causal conv1d update — parallel state advance across
// `num_seqs` independent sequences.
//
// Background: the single-seq `causal_conv1d_update` kernel updates one
// per-sequence sliding-window state. To advance c concurrent sequences
// the calling code was running c launches in a sequential loop, leaving
// the 48-SM GB10 dramatically under-utilized for c=1 launches (~256/256
// threads × ceil(dim/256)=32 CTAs = 32 SMs touched, then a serial loop
// of c such launches).
//
// This kernel fuses all c launches into one: the grid `y` axis is the
// sequence index, and each (block_x, seq_idx) CTA reads its own
// per-sequence `conv_state` pointer from a small device-resident array.
//
// Strides: `input_stride` and `output_stride` are in BF16 ELEMENTS
// between successive sequences. They are explicit because the calling
// code feeds slabs of a wider per-seq tensor (e.g., AEON's `qkvz_size`
// row layout where each seq's QKV occupies just the first conv_dim
// elements of a qkvz_size-strided row).
//
// Per-seq state pointers are scattered because each sequence owns an
// arbitrary pool slot.
//
// Grid: (ceil(dim/256), num_seqs, 1)  Block: (256, 1, 1)

#include <cuda_bf16.h>

// ============================================================
// DECODE multi-seq: conv1d sliding window update + SiLU.
// ============================================================
extern "C" __global__ void causal_conv1d_update_multi_seq(
    // Per-seq state pointers (device-resident, length num_seqs).
    float* const* __restrict__ conv_states,
    // Per-seq input/output (interpreted with explicit strides).
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,   // [dim, d_conv] BF16, shared
    const float* __restrict__ bias,             // [dim] or nullptr, shared
    __nv_bfloat16* __restrict__ output,
    unsigned int num_seqs,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int input_stride,   // BF16 elements between seqs in new_input
    unsigned int output_stride   // BF16 elements between seqs in output
) {
    const unsigned int ch  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int seq = blockIdx.y;
    if (ch >= dim || seq >= num_seqs) return;

    // Each sequence reads its own state pointer (one u64 load).
    float* state = conv_states[seq] + ch * d_conv;

    // 1. Shift state left by 1.
    for (unsigned int i = 0; i < d_conv - 1; i++) {
        state[i] = state[i + 1];
    }
    // 2. Insert new token.
    state[d_conv - 1] = (float)new_input[(unsigned long long)seq * input_stride + ch];

    // 3. Depthwise conv.
    const __nv_bfloat16* w = weight + ch * d_conv;
    float acc = (bias != nullptr) ? bias[ch] : 0.0f;
    for (unsigned int k = 0; k < d_conv; k++) {
        acc += state[k] * (float)w[k];
    }

    // 4. SiLU.
    float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
    float silu = acc * sigmoid_acc;

    output[(unsigned long long)seq * output_stride + ch] = __float2bfloat16(silu);
}

// ============================================================
// DECODE multi-seq: conv1d + SiLU + L2-norm for Q/K channels.
// ============================================================
// Fused variant matching `causal_conv1d_update_l2norm` semantics.
// One block per (ch_chunk, seq). L2 norm reduces across BLOCK_SIZE=256
// threads within the block (= 2 heads of head_dim=128 each).
extern "C" __global__ void causal_conv1d_update_l2norm_multi_seq(
    float* const* __restrict__ conv_states,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_seqs,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps,
    unsigned int input_stride,
    unsigned int output_stride
) {
    const unsigned int ch  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int seq = blockIdx.y;
    const unsigned int tid = threadIdx.x;

    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = (block_start < qk_channels);

    const bool valid = (ch < dim && seq < num_seqs);
    float silu = 0.0f;

    // ── Step 1: conv1d update + SiLU ──
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

    // ── Step 2: L2 normalize Q/K channels per head ──
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
        output[(unsigned long long)seq * output_stride + ch] = __float2bfloat16(silu);
    }
}
