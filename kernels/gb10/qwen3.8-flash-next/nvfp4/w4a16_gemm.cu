// SPDX-License-Identifier: AGPL-3.0-only

// Use the generic GB10 W4A16 module that also exports the FP8 staging helpers
// required by the shared MoE constructor.  No dimensions are compiled into
// these kernels; Flash-Next supplies its 2560/640 geometry at launch time.
#include "../../qwen3-next-80b-a3b/nvfp4/w4a16_gemm.cu"
