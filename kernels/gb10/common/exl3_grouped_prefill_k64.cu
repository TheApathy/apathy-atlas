// SPDX-License-Identifier: AGPL-3.0-only
// M64 specialization that amortizes direct-prefill synchronization over K64.

#define EXL3_PF_K_STEP 64
#define EXL3_PF_ASYNC_STAGE 1
#define EXL3_PF_KERNEL_NAME exl3_grouped_prefill_k64
#include "exl3_grouped_prefill.cu"
