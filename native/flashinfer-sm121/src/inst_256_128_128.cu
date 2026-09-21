/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_bf16.h>
#include "flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h"

namespace flashinfer {
namespace gemm {
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 256, 128, 128, 1, 1, 1,
                                     _1SM)
}  // namespace gemm
}  // namespace flashinfer
