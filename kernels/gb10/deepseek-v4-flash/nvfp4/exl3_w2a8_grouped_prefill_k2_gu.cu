// SPDX-License-Identifier: AGPL-3.0-only
// DeepSeek-V4 exact K2 gate/up W2A8 grouped-prefill wrapper.

#define W2A8_FIXED_N 2048
#define W2A8_FIXED_K 4096
#define W2A8_KERNEL_NAME exl3_w2a8_grouped_prefill_k2_gu
#include "../../experiments/exl3_w2a8_grouped_prefill.cu"
