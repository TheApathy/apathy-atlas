// SPDX-License-Identifier: AGPL-3.0-only

// Test-only host ABI for the exact current Atlas WY32 gate-cache source.
#include <cuda_runtime.h>
#include <cstdio>

#ifndef ATLAS_WY32_SOURCE_SHA256
#error "build must inject ATLAS_WY32_SOURCE_SHA256"
#endif
#ifndef ATLAS_WY32_BRIDGE_SHA256
#error "build must inject ATLAS_WY32_BRIDGE_SHA256"
#endif

#include "../../../kernels/gb10/common/gated_delta_rule_wy32_gatecache.cu"

namespace {
thread_local char wy32_error[512] = "ok";
int wy32_fail(cudaError_t status, const char* operation) {
    std::snprintf(wy32_error, sizeof(wy32_error), "%s: %s", operation,
                  cudaGetErrorString(status));
    return static_cast<int>(status);
}
}  // namespace

extern "C" const char* atlas_gdn_wy32_last_error() { return wy32_error; }

extern "C" const char* atlas_gdn_wy32_abi_identity() {
    return "atlas-gdn-wy32-bridge-v1:" ATLAS_WY32_SOURCE_SHA256 ":"
           ATLAS_WY32_BRIDGE_SHA256;
}

extern "C" int atlas_gdn_wy32_launch(
    float* state, const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const float* alpha, const float* beta,
    __nv_bfloat16* output, unsigned int seq_len, void* stream_raw) {
    if (!state || !query || !key || !value || !alpha || !beta || !output ||
        seq_len == 0) {
        std::snprintf(wy32_error, sizeof(wy32_error), "invalid null/zero argument");
        return -1;
    }
    constexpr unsigned int shared_bytes = 95232;
    cudaError_t status = cudaFuncSetAttribute(
        gated_delta_rule_prefill_wy32_gatecache,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    if (status != cudaSuccess) return wy32_fail(status, "set WY32 smem");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    gated_delta_rule_prefill_wy32_gatecache<<<dim3(48, 1, 1), 128,
                                              shared_bytes, stream>>>(
        state, query, key, value, alpha, beta, output, 1, seq_len, 16, 48, 128,
        128, 16 * 128, 48 * 128, 48);
    status = cudaGetLastError();
    if (status != cudaSuccess) return wy32_fail(status, "launch WY32");
    std::snprintf(wy32_error, sizeof(wy32_error), "ok");
    return 0;
}
