// SPDX-License-Identifier: AGPL-3.0-only

// Flash-Next uses the generic pointer-table MoE implementation. The attempted
// MiniMax M=128 import needs 58,048 bytes of static shared memory and is
// rejected by sm_121 ptxas (49,152-byte static limit), so it remains
// unavailable rather than silently shipping an unbuildable kernel.
#include "../../qwen3-next-80b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu"

// Flash-Next-only same-stream compact-worklist shadows. Kept separate so the
// imported parent symbols and arithmetic remain byte-for-byte untouched.
#include "moe_w4a16_worklist.cuh"
