// SPDX-License-Identifier: AGPL-3.0-only
#ifndef ATLAS_QWEN38_SSM_RESIDUAL_ADMISSION_CUH_
#define ATLAS_QWEN38_SSM_RESIDUAL_ADMISSION_CUH_

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

struct Span {
  uintptr_t begin;
  uintptr_t end;
};

inline bool make_span(const void* pointer, size_t bytes, Span* span) {
  if (pointer == nullptr || span == nullptr || bytes == 0) return false;
  uintptr_t begin = reinterpret_cast<uintptr_t>(pointer);
  if (begin > std::numeric_limits<uintptr_t>::max() - bytes) return false;
  *span = {begin, begin + bytes};
  return true;
}

inline bool exact_cuda_allocation(const void* pointer, size_t claimed_bytes,
                                  size_t expected_bytes, Span* span) {
  if (claimed_bytes != expected_bytes || !make_span(pointer, claimed_bytes, span)) return false;
  cudaPointerAttributes attributes{};
  cudaError_t runtime_status = cudaPointerGetAttributes(&attributes, pointer);
  if (runtime_status != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  if (attributes.type != cudaMemoryTypeDevice && attributes.type != cudaMemoryTypeManaged) {
    return false;
  }
  CUdeviceptr address = reinterpret_cast<CUdeviceptr>(pointer);
  CUdeviceptr base = 0;
  size_t allocation_bytes = 0;
  if (cuPointerGetAttribute(&base, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, address) !=
          CUDA_SUCCESS ||
      cuPointerGetAttribute(&allocation_bytes, CU_POINTER_ATTRIBUTE_RANGE_SIZE, address) !=
          CUDA_SUCCESS ||
      base != address || allocation_bytes != claimed_bytes) {
    return false;
  }
  return true;
}

inline bool pairwise_disjoint(const Span* spans, size_t count) {
  for (size_t left = 0; left < count; ++left) {
    for (size_t right = left + 1; right < count; ++right) {
      if (spans[left].begin < spans[right].end && spans[right].begin < spans[left].end) {
        return false;
      }
    }
  }
  return true;
}

inline bool aligned_bf16(const void* pointer) {
  return (reinterpret_cast<uintptr_t>(pointer) & 1U) == 0;
}

#endif
