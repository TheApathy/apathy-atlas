// SPDX-License-Identifier: AGPL-3.0-only

// Multi-sequence GDN gates compute — parallel across `num_seqs`.
//
// The single-seq `compute_gdn_gates` is launched as grid (num_tokens=1,
// 1, 1), block (num_v_heads=32, 1, 1). For a c-sequence decode the
// caller previously looped, paying c kernel-launch overheads. This
// kernel collapses the loop into one launch with grid (num_seqs, 1, 1):
// one CTA per sequence, threads = num_v_heads.
//
// All buffers are contiguous per-sequence with explicit strides:
//   ba_interleaved [num_seqs, ba_stride] BF16
//   gate_out       [num_seqs, gate_beta_stride] FP32
//   beta_out       [num_seqs, gate_beta_stride] FP32 (typically same
//                  buffer as gate_out with a +num_v_heads offset
//                  baked into beta_out_ptr)
//
// Functionally identical to compute_gdn_gates per (seq, vh) — only the
// outer loop is moved from CPU to GPU grid.

#include <cuda_bf16.h>

extern "C" __global__ void compute_gdn_gates_multi_seq(
    const __nv_bfloat16* __restrict__ ba_interleaved,
    const float* __restrict__ A_log,          // [num_v_heads]
    const float* __restrict__ dt_bias,        // [num_v_heads]
    float* __restrict__ gate_out,             // [num_seqs, gate_beta_stride] FP32
    float* __restrict__ beta_out,             // [num_seqs, gate_beta_stride] FP32
    unsigned int num_seqs,
    unsigned int num_v_heads,        // 32
    unsigned int num_groups,         // 16
    unsigned int vheads_per_group,   // 2
    unsigned int ba_stride,          // BF16 elements between seqs in ba_interleaved
    unsigned int gate_beta_stride    // FP32 elements between seqs in gate_out/beta_out
) {
    unsigned int seq = blockIdx.x;
    unsigned int vh  = threadIdx.x;
    if (vh >= num_v_heads || seq >= num_seqs) return;

    unsigned int group = vh / vheads_per_group;
    unsigned int local_idx = vh % vheads_per_group;
    unsigned int group_dim_ba = 2 * vheads_per_group;

    const __nv_bfloat16* ba_tok = ba_interleaved + (unsigned long long)seq * ba_stride;
    float* gate_tok = gate_out + (unsigned long long)seq * gate_beta_stride;
    float* beta_tok = beta_out + (unsigned long long)seq * gate_beta_stride;

    // BA layout per group: [B_0, B_1, A_0, A_1]
    float b_raw = (float)ba_tok[group * group_dim_ba + local_idx];
    float a_raw = (float)ba_tok[group * group_dim_ba + vheads_per_group + local_idx];

    float a_log_val = A_log[vh];
    float dt_b = dt_bias[vh];

    float A_val = __expf(fminf(a_log_val, 20.0f));
    float dt = __logf(1.0f + __expf(fminf(a_raw + dt_b, 20.0f)));  // softplus
    float g = -A_val * dt;
    gate_tok[vh] = __expf(g);

    beta_tok[vh] = 1.0f / (1.0f + __expf(-b_raw));
}
