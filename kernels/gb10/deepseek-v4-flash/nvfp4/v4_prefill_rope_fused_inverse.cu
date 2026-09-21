// SPDX-License-Identifier: AGPL-3.0-only

// Production module wrapper for the strict default-off inverse-only V4 prefill
// arm. The experiment remains the single source of the numeric implementation;
// production resolves only its inverse entry point.
#include "../../experiments/v4_prefill_rope_fused.cu"
