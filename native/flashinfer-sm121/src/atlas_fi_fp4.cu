/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Isolated Atlas/FlashInfer C ABI prototype. CUTLASS headers are BSD-3-Clause.
 */
#include "atlas_fi_fp4.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <exception>
#include <string>
#include <vector>

#include "flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h"

namespace {

using Runner = flashinfer::gemm::CutlassFp4GemmRunner<
    __nv_bfloat16,
    flashinfer::gemm::FP4GemmType::W4A4_NVFP4_NVFP4>;

thread_local std::string g_last_error;

int fail(int code, const char* message) noexcept {
  g_last_error = message ? message : "unknown error";
  return code;
}

bool valid_shape(int tactic, int m, int n, int k, int batch_count) noexcept {
  return tactic >= 0 && tactic < 6 && m > 0 && n > 0 && k > 0 &&
         batch_count > 0 && (k % 32) == 0;
}

flashinfer::gemm::CutlassGemmConfig config_for(int tactic) {
  Runner runner;
  auto configs = runner.getConfigs();
  if (tactic < 0 || static_cast<size_t>(tactic) >= configs.size()) {
    throw std::invalid_argument("tactic must be in [0, 5]");
  }
  return configs[static_cast<size_t>(tactic)];
}

}  // namespace

extern "C" const char* atlas_fi_nvfp4_sm121_last_error(void) {
  return g_last_error.c_str();
}

extern "C" int atlas_fi_nvfp4_sm121_workspace_size(
    int tactic, int m, int n, int k, int batch_count,
    size_t* workspace_bytes_out) {
  g_last_error.clear();
  if (!workspace_bytes_out || !valid_shape(tactic, m, n, k, batch_count)) {
    return fail(ATLAS_FI_FP4_INVALID_ARGUMENT,
                "invalid tactic, shape, batch count, or output pointer");
  }
  *workspace_bytes_out = 0;
  try {
    auto config = config_for(tactic);
    *workspace_bytes_out =
        flashinfer::gemm::dispatchNVFP4xNVFP4GemmCTAShapeSm120<
            __nv_bfloat16>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                           m, n, k, batch_count, config, nullptr, 0, nullptr);
    return ATLAS_FI_FP4_OK;
  } catch (const std::exception& error) {
    return fail(ATLAS_FI_FP4_CUTLASS_ERROR, error.what());
  } catch (...) {
    return fail(ATLAS_FI_FP4_UNKNOWN_ERROR, "unknown C++ exception");
  }
}

extern "C" int atlas_fi_nvfp4_sm121_bf16(
    int tactic, void* d, const void* a, const void* b, const void* a_sf,
    const void* b_sf, const float* global_sf, int m, int n, int k,
    int batch_count, void* workspace, size_t workspace_bytes, void* stream) {
  g_last_error.clear();
  if (!valid_shape(tactic, m, n, k, batch_count) || !d || !a || !b ||
      !a_sf || !b_sf || !global_sf || (workspace_bytes != 0 && !workspace)) {
    return fail(ATLAS_FI_FP4_INVALID_ARGUMENT,
                "invalid tactic, shape, or required device pointer");
  }
  try {
    Runner runner;
    runner.gemm(d, a, b, a_sf, b_sf, global_sf, m, n, k, batch_count,
                config_for(tactic), static_cast<char*>(workspace),
                workspace_bytes, reinterpret_cast<cudaStream_t>(stream));
    return ATLAS_FI_FP4_OK;
  } catch (const std::exception& error) {
    return fail(ATLAS_FI_FP4_CUTLASS_ERROR, error.what());
  } catch (...) {
    return fail(ATLAS_FI_FP4_UNKNOWN_ERROR, "unknown C++ exception");
  }
}
