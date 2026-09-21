// SPDX-License-Identifier: AGPL-3.0-only

// F32 derivative: preserve F29's ordered K16 BF16 MMA arithmetic while
// addressing the weight-neutral replacement layout as [K/2, N] packed
// nibbles and [K/16, N] scales.
#define ATLAS_QWEN4_MOE_COMPACT_CONTRACT 1
#define OI640_STEP_K 32
#define OI640_TRANSPOSED 1
#define flash_next_orig_i640_prefill flash_next_f32_transposed_compact_k32
#define moe_w4a16_orig_i640_compact_prefill_plan qwen4_moe_compact_t_plan_k32
#define moe_w4a16_orig_i640_compact_prefill_gemm qwen4_moe_compact_t_gemm_k32
#include "moe_w4a16_orig_i640_compact_prefill.cu"
#undef moe_w4a16_orig_i640_compact_prefill_gemm
#undef moe_w4a16_orig_i640_compact_prefill_plan
#undef flash_next_orig_i640_prefill
#undef OI640_TRANSPOSED
#undef OI640_STEP_K
#undef ATLAS_QWEN4_MOE_COMPACT_CONTRACT
