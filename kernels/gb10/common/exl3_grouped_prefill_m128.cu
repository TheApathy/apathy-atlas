// SPDX-License-Identifier: AGPL-3.0-only
// M128 specialization of the attributed EXL3 direct-prefill kernel.

#define EXL3_PF_M_TILE 128
#define EXL3_PF_KERNEL_NAME exl3_grouped_prefill_m128
#include "exl3_grouped_prefill.cu"
