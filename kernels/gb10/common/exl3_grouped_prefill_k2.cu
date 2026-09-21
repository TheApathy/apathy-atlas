// SPDX-License-Identifier: AGPL-3.0-only
// M64/K16 specialization for the serving checkpoint's fixed K2 trellis.

#define EXL3_PF_FIXED_BITS 2
#define EXL3_PF_KERNEL_NAME exl3_grouped_prefill_k2
#include "exl3_grouped_prefill.cu"
