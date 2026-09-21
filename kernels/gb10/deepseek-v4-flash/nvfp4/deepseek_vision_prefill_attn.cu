// SPDX-License-Identifier: AGPL-3.0-only

// Reuse the shipping TC2 arithmetic, changing only its raw visibility ABI.
#include "deepseek_vision_bounds.cuh"
#define ATLAS_DEEPSEEK_VISION_TC2 1
#define prefill_attn_compressed_tc2 deepseek_vision_prefill_attn
#include "prefill_attn_compressed.cu"
