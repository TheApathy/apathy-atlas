// SPDX-License-Identifier: AGPL-3.0-only

// Flash-Next ships the complete 27-block vision tower. These kernels receive
// every vision dimension at launch and therefore support its 1152-wide tower
// without model pruning or a geometry-specific specialization.
#include "../../qwen3.8-27b/nvfp4/vision_encoder.cu"
