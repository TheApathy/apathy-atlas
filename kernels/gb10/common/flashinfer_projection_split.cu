// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Split the row-major FlashInfer [M, QG|K|V] BF16 result into Atlas's
// existing three prefill destinations. All widths are multiples of uint4, so
// one CTA owns one row and performs only aligned 16-byte transactions.
extern "C" __global__ void flashinfer_projection_split_qgkv(
    const __nv_bfloat16 *__restrict__ merged,
    __nv_bfloat16 *__restrict__ query_gate,
    __nv_bfloat16 *__restrict__ key,
    __nv_bfloat16 *__restrict__ value,
    uint32_t rows) {
  constexpr uint32_t QG = 12288;
  constexpr uint32_t KV = 1024;
  constexpr uint32_t TOTAL = QG + 2 * KV;
  constexpr uint32_t BF16_PER_VECTOR = sizeof(uint4) / sizeof(__nv_bfloat16);
  static_assert(QG % BF16_PER_VECTOR == 0, "QG must be uint4-aligned");
  static_assert(KV % BF16_PER_VECTOR == 0, "KV must be uint4-aligned");

  const uint32_t row = blockIdx.x;
  if (row >= rows) return;

  const auto *src = reinterpret_cast<const uint4 *>(merged + row * TOTAL);
  auto *qg = reinterpret_cast<uint4 *>(query_gate + row * QG);
  auto *k = reinterpret_cast<uint4 *>(key + row * KV);
  auto *v = reinterpret_cast<uint4 *>(value + row * KV);
  constexpr uint32_t QG_VECTORS = QG / BF16_PER_VECTOR;
  constexpr uint32_t KV_VECTORS = KV / BF16_PER_VECTOR;

  for (uint32_t vector = threadIdx.x; vector < QG_VECTORS;
       vector += blockDim.x) {
    qg[vector] = src[vector];
  }
  for (uint32_t vector = threadIdx.x; vector < KV_VECTORS;
       vector += blockDim.x) {
    k[vector] = src[QG_VECTORS + vector];
    v[vector] = src[QG_VECTORS + KV_VECTORS + vector];
  }
}

// Split the exact Qwen3.8 dense-FFN FlashInfer [M, gate|up] BF16 result
// into Atlas's existing row-major gate and up destinations. A CTA owns one
// row and all three spans are 16-byte aligned by host admission.
extern "C" __global__ void flashinfer_projection_split_ffn_gate_up(
    const __nv_bfloat16 *__restrict__ merged,
    __nv_bfloat16 *__restrict__ gate,
    __nv_bfloat16 *__restrict__ up,
    uint32_t rows) {
  constexpr uint32_t INTERMEDIATE = 17408;
  constexpr uint32_t TOTAL = 2 * INTERMEDIATE;
  constexpr uint32_t BF16_PER_VECTOR = sizeof(uint4) / sizeof(__nv_bfloat16);
  static_assert(INTERMEDIATE % BF16_PER_VECTOR == 0,
                "FFN intermediate must be uint4-aligned");

  const uint32_t row = blockIdx.x;
  if (row >= rows) return;

  const auto *src = reinterpret_cast<const uint4 *>(merged + row * TOTAL);
  auto *gate_dst = reinterpret_cast<uint4 *>(gate + row * INTERMEDIATE);
  auto *up_dst = reinterpret_cast<uint4 *>(up + row * INTERMEDIATE);
  constexpr uint32_t VECTORS = INTERMEDIATE / BF16_PER_VECTOR;

  for (uint32_t vector = threadIdx.x; vector < VECTORS;
       vector += blockDim.x) {
    gate_dst[vector] = src[vector];
    up_dst[vector] = src[VECTORS + vector];
  }
}
