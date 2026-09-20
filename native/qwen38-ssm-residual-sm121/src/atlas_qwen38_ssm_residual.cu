// SPDX-License-Identifier: AGPL-3.0-only
#include "atlas_qwen38_ssm_residual.h"
#include "atlas_qwen38_ssm_residual_admission.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr int kResidualCols = 5120;
constexpr int kProjectionCols = 16384;
constexpr int kGroupSize = 16;
constexpr int kThreads = 256;
thread_local const char* g_last_error = "";
struct PendingReceipt {
  cudaStream_t stream;
  unsigned long long nonce;
  bool active;
};
thread_local PendingReceipt g_pending{};
std::atomic<unsigned long long> g_next_nonce{1};

bool valid_rows(int rows) { return rows == 2079 || rows == 8192; }

int fail(int code, const char* message) {
  g_last_error = message;
  return code;
}

bool prelaunch_status() {
  cudaError_t status = cudaGetLastError();
  if (status == cudaSuccess) return true;
  fail(ATLAS_Q38_CUDA_ERROR, cudaGetErrorString(status));
  return false;
}

int launch_status() {
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) return fail(ATLAS_Q38_CUDA_ERROR, cudaGetErrorString(status));
  return ATLAS_Q38_OK;
}

unsigned long long reserve_nonce() {
  auto nonce = g_next_nonce.load(std::memory_order_relaxed);
  const auto exhausted = std::numeric_limits<unsigned long long>::max();
  while (nonce != exhausted) {
    if (g_next_nonce.compare_exchange_weak(
            nonce, nonce + 1, std::memory_order_relaxed)) return nonce;
  }
  return 0;
}

int publish_after_launch(cudaStream_t stream, unsigned long long nonce) {
  int status = launch_status();
  if (status != ATLAS_Q38_OK) return status;
  g_pending = {stream, nonce, true};
  return ATLAS_Q38_OK;
}

__device__ __constant__ float kE2m1[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

__device__ __forceinline__ float decode_e2m1(unsigned int nibble) {
  float magnitude = kE2m1[nibble & 7U];
  return (nibble & 8U) != 0U ? -magnitude : magnitude;
}

__device__ __forceinline__ float decode_e4m3(unsigned char byte) {
  unsigned int sign = byte >> 7;
  unsigned int exponent = (byte >> 3) & 15U;
  unsigned int mantissa = byte & 7U;
  float decoded;
  if (exponent == 0U) {
    decoded = static_cast<float>(mantissa) * 0.001953125f;
  } else if (exponent == 15U && mantissa == 7U) {
    decoded = 0.0f;
  } else {
    decoded = __uint_as_float(((exponent + 120U) << 23) | (mantissa << 20));
  }
  return sign != 0U ? -decoded : decoded;
}

__device__ __forceinline__ unsigned long long scale_offset_128x4(
    unsigned int row, unsigned int group, unsigned int groups) {
  unsigned int group_blocks = (groups + 3) / 4;
  return (((((static_cast<unsigned long long>(row / 128) * group_blocks + group / 4) * 32
              + row % 32) * 4 + (row % 128) / 32) * 4) + group % 4);
}

__global__ void residual_kernel(
    __nv_bfloat16* residual, const __nv_bfloat16* input,
    const unsigned char* packed, const unsigned char* physical_scales,
    float scale2, unsigned int rows, unsigned int cols) {
  unsigned long long total = static_cast<unsigned long long>(rows) * cols;
  unsigned int groups = cols / kGroupSize;
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total; index += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
    unsigned int row = index / cols;
    unsigned int col = index - static_cast<unsigned long long>(row) * cols;
    unsigned char byte = packed[static_cast<unsigned long long>(row) * (cols / 2) + col / 2];
    unsigned int nibble = (col & 1U) == 0U ? byte & 0x0fU : byte >> 4;
    unsigned char scale_byte =
        physical_scales[scale_offset_128x4(row, col / kGroupSize, groups)];
    float original = __bfloat162float(input[index]);
    float dequantized = decode_e2m1(nibble) * decode_e4m3(scale_byte) * scale2;
    residual[index] = __float2bfloat16_rn(original - dequantized);
  }
}

__global__ void add_kernel(
    __nv_bfloat16* output, const __nv_bfloat16* first,
    const __nv_bfloat16* second, unsigned long long elements) {
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
    output[index] = __float2bfloat16_rn(
        __bfloat162float(first[index]) + __bfloat162float(second[index]));
  }
}

int blocks_for(unsigned long long elements) {
  unsigned long long blocks = (elements + kThreads - 1) / kThreads;
  return static_cast<int>(blocks > 65535 ? 65535 : blocks);
}

}  // namespace

extern "C" const char* atlas_qwen38_ssm_residual_last_error(void) {
  return g_last_error;
}

extern "C" int atlas_qwen38_ssm_residual_bf16(
    void* residual_bf16, size_t residual_bf16_bytes,
    const void* input_bf16, size_t input_bf16_bytes,
    const unsigned char* packed_e2m1, size_t packed_e2m1_bytes,
    const unsigned char* physical_e4m3_scales,
    size_t physical_e4m3_scales_bytes, float scale2, int rows, int cols,
    void* stream) {
  g_last_error = "";
  if (stream == nullptr || g_pending.active || !valid_rows(rows) || cols != kResidualCols ||
      !std::isfinite(scale2) || scale2 <= 0.0f ||
      !aligned_bf16(residual_bf16) || !aligned_bf16(input_bf16)) {
    return fail(ATLAS_Q38_INVALID_ARGUMENT, "invalid residual stream, shape, scale, or BF16 pointer");
  }
  size_t values = static_cast<size_t>(rows) * cols;
  size_t padded_rows = (static_cast<size_t>(rows) + 127) / 128 * 128;
  size_t value_bytes = values * 2;
  size_t packed_bytes = values / 2;
  size_t scale_bytes = padded_rows * (cols / kGroupSize);
  Span spans[4];
  if (!exact_cuda_allocation(residual_bf16, residual_bf16_bytes, value_bytes, &spans[0]) ||
      !exact_cuda_allocation(input_bf16, input_bf16_bytes, value_bytes, &spans[1]) ||
      !exact_cuda_allocation(packed_e2m1, packed_e2m1_bytes, packed_bytes, &spans[2]) ||
      !exact_cuda_allocation(physical_e4m3_scales, physical_e4m3_scales_bytes, scale_bytes,
                             &spans[3]) ||
      !pairwise_disjoint(spans, 4)) {
    return fail(ATLAS_Q38_INVALID_ARGUMENT,
                "residual buffers must be exact disjoint CUDA allocations");
  }
  if (!prelaunch_status()) return ATLAS_Q38_CUDA_ERROR;
  unsigned long long nonce = reserve_nonce();
  if (nonce == 0) return fail(ATLAS_Q38_CUDA_ERROR, "receipt nonce exhausted");
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  residual_kernel<<<blocks_for(values), kThreads, 0, cuda_stream>>>(
      static_cast<__nv_bfloat16*>(residual_bf16),
      static_cast<const __nv_bfloat16*>(input_bf16), packed_e2m1,
      physical_e4m3_scales, scale2, rows, cols);
  return publish_after_launch(cuda_stream, nonce);
}

extern "C" int atlas_qwen38_ssm_add_bf16(
    void* output_bf16, size_t output_bf16_bytes,
    const void* first_bf16, size_t first_bf16_bytes,
    const void* second_bf16, size_t second_bf16_bytes, int rows, int cols,
    void* stream) {
  g_last_error = "";
  if (stream == nullptr || g_pending.active || !valid_rows(rows) || cols != kProjectionCols ||
      !aligned_bf16(output_bf16) || !aligned_bf16(first_bf16) ||
      !aligned_bf16(second_bf16)) {
    return fail(ATLAS_Q38_INVALID_ARGUMENT, "invalid add stream, shape, or BF16 pointer");
  }
  size_t values = static_cast<size_t>(rows) * cols;
  size_t value_bytes = values * 2;
  Span spans[3];
  if (!exact_cuda_allocation(output_bf16, output_bf16_bytes, value_bytes, &spans[0]) ||
      !exact_cuda_allocation(first_bf16, first_bf16_bytes, value_bytes, &spans[1]) ||
      !exact_cuda_allocation(second_bf16, second_bf16_bytes, value_bytes, &spans[2]) ||
      !pairwise_disjoint(spans, 3)) {
    return fail(ATLAS_Q38_INVALID_ARGUMENT, "add buffers must be exact disjoint CUDA allocations");
  }
  if (!prelaunch_status()) return ATLAS_Q38_CUDA_ERROR;
  unsigned long long nonce = reserve_nonce();
  if (nonce == 0) return fail(ATLAS_Q38_CUDA_ERROR, "receipt nonce exhausted");
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  add_kernel<<<blocks_for(values), kThreads, 0, cuda_stream>>>(
      static_cast<__nv_bfloat16*>(output_bf16),
      static_cast<const __nv_bfloat16*>(first_bf16),
      static_cast<const __nv_bfloat16*>(second_bf16), values);
  return publish_after_launch(cuda_stream, nonce);
}

extern "C" unsigned long long atlas_qwen38_ssm_pending_receipt(void) {
  return g_pending.active ? g_pending.nonce : 0;
}

extern "C" int atlas_qwen38_ssm_stream_synchronize(
    void* stream, unsigned long long nonce) {
  g_last_error = "";
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (stream == nullptr || nonce == 0 || !g_pending.active ||
      g_pending.stream != cuda_stream || g_pending.nonce != nonce) {
    return fail(ATLAS_Q38_INVALID_ARGUMENT, "receipt or bound stream mismatch");
  }
  g_pending = {};
  cudaError_t status = cudaStreamSynchronize(cuda_stream);
  if (status != cudaSuccess) return fail(ATLAS_Q38_CUDA_ERROR, cudaGetErrorString(status));
  return ATLAS_Q38_OK;
}
