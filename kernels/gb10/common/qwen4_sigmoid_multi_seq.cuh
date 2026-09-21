// SPDX-License-Identifier: AGPL-3.0-only

#ifndef ATLAS_QWEN4_SIGMOID_MULTI_SEQ_CUH
#define ATLAS_QWEN4_SIGMOID_MULTI_SEQ_CUH

// Include after qwen4_gated_rms_sigmoid_body. Keep its per-head reduction,
// expf, multiplication/division order and BF16 boundary unchanged. Strides
// are elements of each pointer's type, not bytes; grid.x=head, grid.y=row.
extern "C" __global__ void qwen4_gated_rms_norm_sigmoid_f32_multi_seq(
    const float* input,
    const __nv_bfloat16* gate,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    unsigned int head_dim,
    float eps,
    unsigned int input_stride,
    unsigned int gate_stride,
    unsigned int output_stride) {
    const unsigned long long seq = blockIdx.y;
    qwen4_gated_rms_sigmoid_body(
        input + seq * input_stride, gate + seq * gate_stride, weight,
        output + seq * output_stride, head_dim, eps, head_dim);
}

#endif
