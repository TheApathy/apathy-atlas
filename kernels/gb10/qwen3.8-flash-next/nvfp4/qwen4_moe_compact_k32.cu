// SPDX-License-Identifier: AGPL-3.0-only

// Default-off F25 derivative: two ordered K16 MMA steps per shared-memory
// residency. The arithmetic order is unchanged; only the barrier cadence is
// reduced from one round per 16 K values to one round per 32 K values.
#define ATLAS_QWEN4_MOE_COMPACT_CONTRACT 1
#define OI640_STEP_K 32
#define flash_next_orig_i640_prefill flash_next_f8_orig_compact_k32
#define moe_w4a16_orig_i640_compact_prefill_plan qwen4_moe_compact_plan_k32
#define moe_w4a16_orig_i640_compact_prefill_gemm qwen4_moe_compact_gemm_k32
#include "moe_w4a16_orig_i640_compact_prefill.cu"
#undef moe_w4a16_orig_i640_compact_prefill_gemm
#undef moe_w4a16_orig_i640_compact_prefill_plan
#undef flash_next_orig_i640_prefill
#undef OI640_STEP_K
#undef ATLAS_QWEN4_MOE_COMPACT_CONTRACT
