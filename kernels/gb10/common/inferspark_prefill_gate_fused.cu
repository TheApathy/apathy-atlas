// SPDX-License-Identifier: AGPL-3.0-only

// Exact chunk-0 BR64 attention + sigmoid-gate shadow for dense Qwen3.8.
// The included parent retains every QK/softmax/PV operation. Only the final
// packed BF16x2 store is specialized: attention is explicitly rounded to
// BF16 and widened before the parent gate expression and final BF16 store.
// The included BR64 parent ABI carries seq_len, query_start, and
// query_len_total before the head geometry; the Rust launcher supplies the
// full-range pair (0, seq_len) for this chunk-0 route.

#define ATLAS_PREFILL_64_KERNEL_NAME inferspark_prefill_64_gate_fused
#define ATLAS_PREFILL_64_EXTRA_ARGS \
    , const __nv_bfloat16* __restrict__ Gate, const unsigned int gate_stride
#define ATLAS_PREFILL_64_GATE_SETUP \
    const __nv_bfloat16* Gate_batch = Gate + batch * seq_len * gate_stride;
#define ATLAS_PREFILL_64_STORE_PAIR(base, row, stride, col, value0, value1, inv_l) do { \
    __nv_bfloat16 attn0_bf16 = __float2bfloat16((value0) * (inv_l));                  \
    __nv_bfloat16 attn1_bf16 = __float2bfloat16((value1) * (inv_l));                  \
    float x0 = __bfloat162float(attn0_bf16);                                          \
    float x1 = __bfloat162float(attn1_bf16);                                          \
    unsigned int gate_idx = (row) * gate_stride + q_head * head_dim + (col);          \
    float g0 = __bfloat162float(Gate_batch[gate_idx]);                                \
    float g1 = __bfloat162float(Gate_batch[gate_idx + 1]);                            \
    float sigmoid_g0 = 1.0f / (1.0f + expf(-g0));                                    \
    float sigmoid_g1 = 1.0f / (1.0f + expf(-g1));                                    \
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(                             \
        __float2bfloat16(x0 * sigmoid_g0));                                           \
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(                             \
        __float2bfloat16(x1 * sigmoid_g1));                                           \
    *(unsigned int*)&(base)[(row) * (stride) + (col)] = lo | (hi << 16);              \
} while (0)

#include "inferspark_prefill.cu"
