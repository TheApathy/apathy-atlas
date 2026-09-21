// SPDX-License-Identifier: AGPL-3.0-only
// DeepSeek-V4 gate/up: persistent M64/N256/K64, fixed N=2048, K=4096, K2.

#define EXL3_PF_M_TILE 64
#define EXL3_PF_N_TILE 256
#define EXL3_PF_FIXED_BITS 2
#define EXL3_PF_FIXED_N 2048
#define EXL3_PF_FIXED_K 4096
#define EXL3_PF_FIXED_PERSISTENT 1
#define EXL3_PF_FIXED_IDENTITY_ROWS 1
#define EXL3_PF_EXACT_FULL_GRID 1
#define EXL3_PF_LAUNCH_BOUNDS 512
#define EXL3_PF_PACKED_BF16_STORE 1
#define EXL3_PF_K2_LO_WINDOWS 1
#define EXL3_PF_K_STEP 64
#define EXL3_PF_ASYNC_STAGE 1
#define EXL3_PF_KERNEL_NAME exl3_grouped_prefill_k64_n256_k2_gu
#include "exl3_grouped_prefill.cu"
