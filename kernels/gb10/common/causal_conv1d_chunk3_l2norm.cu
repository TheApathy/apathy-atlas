// SPDX-License-Identifier: AGPL-3.0-only

// K=3 fused conv1d-update + SiLU + L2 norm + intermediate-state save.
//
// At MTP K=3 verify, the decode_batched path runs conv1d_update_l2norm
// 3 times sequentially (state evolves token-by-token) AND issues 3 d2d
// copies of conv_state into per-step intermediates. That's 6 launches
// per SSM layer × 48 SSM layers = 288 launches per token spent on
// conv1d alone. This kernel collapses the chain into a SINGLE launch
// by keeping conv_state in registers across the 3 sequential updates
// and writing each intermediate + the final state directly to DRAM.
//
// Saves 5 launches per SSM layer per K=3 verify step (288 → 48 launches
// per token), ≈4.8 ms of pure launch-overhead removed at K=3 × 48
// layers × ~20 μs/launch on GB10.
//
// Output is BF16 (matches the existing K=3 verify GDN input contract:
// gdn_wy3 reads BF16 query/key/value from conv_out_buf). FP32 single-
// seq decode keeps its dedicated kernel; this is the K=3 verify path
// equivalent of the chunk2 fused kernel that already exists.
//
// Grid: (ceil(dim/256), batch, 1)  Block: (256, 1, 1)
//
// L2 norm grouping: BLOCK_SIZE=256, head_dim=128 → exactly 2 heads per
// block (warps 0-3 = head A, warps 4-7 = head B). Requires
// qk_channels % 256 == 0 (always true: qk_channels = 2*key_dim = 4096
// for AEON-Q36-27B-XS).

#include <cuda_bf16.h>

extern "C" __global__ void causal_conv1d_update_l2norm_chunk3(
    float* __restrict__ conv_state,              // [batch, dim, d_conv] FP32 in/out (final state)
    const __nv_bfloat16* __restrict__ new_input, // BF16, stride `input_stride` between tokens
    const __nv_bfloat16* __restrict__ weight,    // [dim, d_conv] BF16
    const float* __restrict__ bias,              // [dim] or nullptr
    __nv_bfloat16* __restrict__ output,          // BF16, stride `output_stride` between tokens
    float* __restrict__ state_inter_0,           // [batch, dim, d_conv] FP32 (after token 0)
    float* __restrict__ state_inter_1,           // [batch, dim, d_conv] FP32 (after token 1)
    unsigned int batch,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps,
    unsigned int input_stride,  // BF16 elements between successive tokens in new_input
    unsigned int output_stride  // BF16 elements between successive tokens in output
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b  = blockIdx.y;
    const unsigned int tid = threadIdx.x;

    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = (block_start < qk_channels);
    const bool valid = (ch < dim && b < batch);

    // Load weights + bias into registers (used 3 times).
    float w_reg[4] = {0.f, 0.f, 0.f, 0.f};
    float b_val = 0.0f;
    if (valid) {
        const __nv_bfloat16* w = weight + ch * d_conv;
        for (unsigned int k = 0; k < d_conv && k < 4; k++) {
            w_reg[k] = (float)w[k];
        }
        b_val = (bias != nullptr) ? bias[ch] : 0.0f;
    }

    // Load conv_state into registers (d_conv = 4 for Qwen3-Next).
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    if (valid) {
        float* state = conv_state + (b * dim + ch) * d_conv;
        for (unsigned int k = 0; k < d_conv && k < 4; k++) {
            s[k] = state[k];
        }
    }

    // Per-block scratch for warp-level L2 reductions (shared across the 3
    // tokens — each token recomputes via warp_sums[base_warp]).
    __shared__ float warp_sums[8];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane    = tid % 32;
    const unsigned int head_in_block = tid / head_dim;          // 0 or 1
    const unsigned int base_warp     = head_in_block * (head_dim / 32);

    // Loop body for one token: shift+insert, save state, conv+silu, optional
    // L2 norm, write output. Saves to `state_out_ptr` if non-null (used for
    // intermediates after tokens 0 and 1).
    #define PROCESS_TOKEN(in_idx, state_out_ptr, out_idx)                      \
    do {                                                                       \
        float in_val = 0.0f;                                                   \
        if (valid) {                                                           \
            in_val = (float)new_input[                                         \
                (unsigned long long)b * 3 * input_stride                       \
              + (unsigned long long)(in_idx) * input_stride                    \
              + ch];                                                           \
        }                                                                      \
        float new_s0 = s[1], new_s1 = s[2], new_s2 = s[3], new_s3 = in_val;    \
                                                                               \
        if (valid && (state_out_ptr) != nullptr) {                             \
            float* st_out = (state_out_ptr) + (b * dim + ch) * d_conv;         \
            st_out[0] = new_s0;                                                \
            st_out[1] = new_s1;                                                \
            st_out[2] = new_s2;                                                \
            st_out[3] = new_s3;                                                \
        }                                                                      \
                                                                               \
        s[0] = new_s0; s[1] = new_s1; s[2] = new_s2; s[3] = new_s3;            \
                                                                               \
        float silu = 0.0f;                                                     \
        if (valid) {                                                           \
            float acc = b_val + s[0]*w_reg[0] + s[1]*w_reg[1]                  \
                              + s[2]*w_reg[2] + s[3]*w_reg[3];                 \
            float sig = 1.0f / (1.0f + __expf(-acc));                          \
            silu = acc * sig;                                                  \
        }                                                                      \
                                                                               \
        if (block_needs_l2) {                                                  \
            float sq = valid ? (silu * silu) : 0.0f;                           \
            for (int offset = 16; offset >= 1; offset >>= 1)                   \
                sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);                \
            if (lane == 0) warp_sums[warp_id] = sq;                            \
            __syncthreads();                                                   \
            if (tid == 0 || tid == head_dim) {                                 \
                float total = warp_sums[base_warp] + warp_sums[base_warp + 1] \
                            + warp_sums[base_warp + 2] + warp_sums[base_warp + 3]; \
                warp_sums[base_warp] = rsqrtf(total + l2_eps);                 \
            }                                                                  \
            __syncthreads();                                                   \
            if (valid) silu *= warp_sums[base_warp];                           \
        }                                                                      \
                                                                               \
        if (valid) {                                                           \
            output[                                                            \
                (unsigned long long)b * 3 * output_stride                      \
              + (unsigned long long)(out_idx) * output_stride                  \
              + ch] = __float2bfloat16(silu);                                  \
        }                                                                      \
    } while (0)

    PROCESS_TOKEN(0, state_inter_0, 0);
    PROCESS_TOKEN(1, state_inter_1, 1);

    // Final token: write state back to `conv_state` (the "live" state pointer)
    // — same role as the original conv1d_update_l2norm output.
    PROCESS_TOKEN(2, conv_state, 2);

    #undef PROCESS_TOKEN
}
