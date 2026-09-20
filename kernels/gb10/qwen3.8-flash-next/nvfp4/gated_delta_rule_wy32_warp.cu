// SPDX-License-Identifier: AGPL-3.0-only

// Flash-Next WY32 recurrence with four-pair warp-parallel FP32 reductions.
// The arithmetic tree matches the generic kernel; only block barriers and
// idle-warp scheduling are removed.
#define ATLAS_GDN_KD_WARP 1
#define gated_delta_rule_prefill_wy64 gated_delta_rule_prefill_wy32_warp
#include "../../common/gated_delta_rule_wy64_prefill.cu"
