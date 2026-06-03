// SPDX-License-Identifier: AGPL-3.0-only

// Multi-sequence Gated RMS Norm with FP32 input, PER-HEAD normalization,
// and per-seq strides.
//
// The single-seq `gated_rms_norm_f32_input` is launched as
//   grid (num_v_heads, 1, 1)  block (head_dim, 1, 1)
// — i.e. one CTA per (seq, head) pair. For the multi-seq decode buffer
// layout (see trait_decode_multi_seq.rs), the GDN output lives at
// `conv_out_f32 + gdn_local_offset` with stride `qkvz_size` FP32 between
// seqs, the Z gate lives at `deinterleaved + Z offset` with stride
// `qkvz_size` BF16, and we want output written into a VALUE_DIM-CONTIGUOUS
// scratch buffer so the subsequent out_proj can fire as a single batched
// w4a16_gemm at M=num_seqs.
//
// Grid: (num_v_heads, num_seqs, 1)   Block: (head_dim, 1, 1)
//
// Functionally identical to gated_rms_norm_f32_input per (seq, head) —
// only the outer per-seq loop is moved from CPU to GPU grid, and the
// per-seq row strides for input/gate/output are parameterised so the
// in-bounds slices stay correct under the multi-seq layout.

#include <cuda_bf16.h>

extern "C" __global__ void gated_rms_norm_f32_multi_seq(
    const float* __restrict__ input,              // base: GDN output region
    const __nv_bfloat16* __restrict__ gate,       // base: Z gate region
    const __nv_bfloat16* __restrict__ weight,     // [head_dim]
    __nv_bfloat16* __restrict__ output,            // base: value_dim-contig out
    unsigned int head_dim,
    float eps,
    unsigned int input_stride,   // FP32 elements between seqs in input
    unsigned int gate_stride,    // BF16 elements between seqs in gate
    unsigned int output_stride   // BF16 elements between seqs in output
) {
    unsigned int head = blockIdx.x;
    unsigned int seq  = blockIdx.y;
    unsigned int tid  = threadIdx.x;

    const float* x = input  + (unsigned long long)seq * input_stride  + head * head_dim;
    const __nv_bfloat16* g = gate   + (unsigned long long)seq * gate_stride   + head * head_dim;
    __nv_bfloat16* out     = output + (unsigned long long)seq * output_stride + head * head_dim;

    // Pass 1: sum of squares within this head's slice.
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < head_dim; i += blockDim.x) {
        float f = x[i];
        sum_sq += f * f;
    }

    // Warp + block reduction. block = head_dim (typically 128 ≤ 1024), so
    // ceil(blockDim/32) ≤ 32 warps.
    for (int off = 16; off > 0; off >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, off);
    }
    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, off);
        }
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)head_dim + eps);

    // Pass 2: apply norm × weight × SiLU(gate). Scalar form (head_dim is
    // typically 128 which fits in one warp pass per thread).
    for (unsigned int i = tid; i < head_dim; i += blockDim.x) {
        float f  = x[i];
        float w  = __bfloat162float(weight[i]);
        float gv = __bfloat162float(g[i]);
        float s  = gv / (1.0f + __expf(-gv));
        out[i] = __float2bfloat16(f * rms * w * s);
    }
}
