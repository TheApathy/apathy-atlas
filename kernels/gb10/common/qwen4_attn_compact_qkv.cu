// SPDX-License-Identifier: AGPL-3.0-only

// Compact the Qwen4 prefill QKV scratch into the contiguous per-tensor layout
// the Flash-Attention prefill kernel expects.
//
// The fast prefill projection writes one row per token as
//     [ Q (nq*hd) | gate (nq*hd) | K (nkv*hd) | V (nkv*hd) ]
// with an arbitrary element stride `row_stride` (13312 for this model), and
// MRoPE is applied in place. `inferspark_prefill` instead wants three separate
// contiguous tensors with strides nq*hd and nkv*hd.
//
// This is a pure strided copy: no arithmetic, no rounding, no reordering within
// a row. It moves ~29 MB per attention layer.

#include <cuda_bf16.h>

extern "C" __global__ void qwen4_attn_compact_qkv(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    unsigned int num_tokens,
    unsigned int row_stride,   // elements between consecutive qkv rows
    unsigned int q_elems,      // nq * hd
    unsigned int kv_elems,     // nkv * hd
    unsigned int k_offset,     // element offset of K within a qkv row
    unsigned int v_offset)     // element offset of V within a qkv row
{
    const unsigned int token = blockIdx.y;
    if (token >= num_tokens) return;
    const unsigned long long src = (unsigned long long)token * row_stride;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;

    for (unsigned int c = i; c < q_elems; c += stride) {
        q_out[(unsigned long long)token * q_elems + c] = qkv[src + c];
    }
    for (unsigned int c = i; c < kv_elems; c += stride) {
        k_out[(unsigned long long)token * kv_elems + c] = qkv[src + k_offset + c];
        v_out[(unsigned long long)token * kv_elems + c] = qkv[src + v_offset + c];
    }
}
