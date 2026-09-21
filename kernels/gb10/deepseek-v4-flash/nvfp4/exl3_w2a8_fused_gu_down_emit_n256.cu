// SPDX-License-Identifier: AGPL-3.0-only
// DeepSeek-V4 exact K2 fused gate/up-to-down-A8 M32xN256 wrapper.

#define W2A8_FIXED_N 2048
#define W2A8_FIXED_K 4096
#define W2A8_KERNEL_NAME exl3_w2a8_fused_gu_down_emit_n256
#include "../../experiments/exl3_w2a8_fused_gu_down_emit_n256.cu"
