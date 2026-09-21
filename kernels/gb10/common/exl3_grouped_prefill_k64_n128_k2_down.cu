// SPDX-License-Identifier: AGPL-3.0-only
// DeepSeek-V4 down: persistent M64/N128/K64, fixed N=4096, K=2048, K2.

#define EXL3_PF_M_TILE 64
#define EXL3_PF_N_TILE 128
#define EXL3_PF_FIXED_BITS 2
#define EXL3_PF_FIXED_N 4096
#define EXL3_PF_FIXED_K 2048
#define EXL3_PF_FIXED_PERSISTENT 1
#define EXL3_PF_FIXED_IDENTITY_ROWS 1
#define EXL3_PF_EXACT_FULL_GRID 1
#define EXL3_PF_LAUNCH_BOUNDS 256
#define EXL3_PF_PACKED_BF16_STORE 1
#define EXL3_PF_K2_LO_WINDOWS 1
#define EXL3_PF_K_STEP 64
#define EXL3_PF_ASYNC_STAGE 1
#define EXL3_PF_KERNEL_NAME exl3_grouped_prefill_k64_n128_k2_down
#include "exl3_grouped_prefill.cu"
