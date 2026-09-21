// SPDX-License-Identifier: AGPL-3.0-only
// DeepSeek-V4 gate/up: persistent M64/K16, fixed N=2048, K=4096, K2.

#define EXL3_PF_FIXED_BITS 2
#define EXL3_PF_FIXED_N 2048
#define EXL3_PF_FIXED_K 4096
#define EXL3_PF_FIXED_PERSISTENT 1
#define EXL3_PF_FIXED_IDENTITY_ROWS 1
#define EXL3_PF_EXACT_FULL_GRID 1
#define EXL3_PF_LAUNCH_BOUNDS 128
#define EXL3_PF_KERNEL_NAME exl3_grouped_prefill_k2_gu
#include "exl3_grouped_prefill.cu"
