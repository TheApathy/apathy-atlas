// SPDX-License-Identifier: AGPL-3.0-only

// Flash-Next uses the generic pointer-table MoE implementation, including
// the transposed and K=64 entry points required by MoeLayer construction.
// This implementation is byte-identical across the existing GB10 Qwen MoE
// targets; keep a single source include until it is promoted into common/.
#include "../../qwen3-next-80b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu"
