// SPDX-License-Identifier: AGPL-3.0-only

// F8-only production derivative. Retain the planner and N64/K16 BF16-MMA
// implementation as the single arithmetic source; only row admission differs.
#define ATLAS_QWEN4_MOE_COMPACT_CONTRACT 1
#define flash_next_orig_i640_prefill flash_next_f8_orig_compact
#define moe_w4a16_orig_i640_compact_prefill_plan qwen4_moe_compact_plan
#define moe_w4a16_orig_i640_compact_prefill_gemm qwen4_moe_compact_gemm
#include "moe_w4a16_orig_i640_compact_prefill.cu"
#undef moe_w4a16_orig_i640_compact_prefill_gemm
#undef moe_w4a16_orig_i640_compact_prefill_plan
#undef flash_next_orig_i640_prefill
#undef ATLAS_QWEN4_MOE_COMPACT_CONTRACT
