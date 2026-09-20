/* SPDX-License-Identifier: AGPL-3.0-only */
#ifndef ATLAS_QWEN38_SSM_RESIDUAL_H_
#define ATLAS_QWEN38_SSM_RESIDUAL_H_

#include <stddef.h>

#if defined(_WIN32)
#define ATLAS_Q38_API __declspec(dllexport)
#else
#define ATLAS_Q38_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
  ATLAS_Q38_OK = 0,
  ATLAS_Q38_INVALID_ARGUMENT = -1,
  ATLAS_Q38_CUDA_ERROR = -2,
};

/*
 * Recover BF16(A - dequant(Q(A))) from logical packed E2M1 values and
 * FlashInfer/CUTLASS physical 128-row x 4-group E4M3 scales.
 * Exact admitted shapes are rows=2079|8192 and cols=5120.
 */
ATLAS_Q38_API int atlas_qwen38_ssm_residual_bf16(
    void* residual_bf16, size_t residual_bf16_bytes,
    const void* input_bf16, size_t input_bf16_bytes,
    const unsigned char* packed_e2m1, size_t packed_e2m1_bytes,
    const unsigned char* physical_e4m3_scales,
    size_t physical_e4m3_scales_bytes, float scale2, int rows, int cols,
    void* stream);

/*
 * Materialize BF16(first + second) for the two W4A4 projection outputs.
 * Exact admitted shapes are rows=2079|8192 and cols=16384.
 */
ATLAS_Q38_API int atlas_qwen38_ssm_add_bf16(
    void* output_bf16, size_t output_bf16_bytes,
    const void* first_bf16, size_t first_bf16_bytes,
    const void* second_bf16, size_t second_bf16_bytes, int rows, int cols,
    void* stream);

/* Issued only after a successful enqueue on this calling thread. */
ATLAS_Q38_API unsigned long long atlas_qwen38_ssm_pending_receipt(void);

/* Consumes the exact pending receipt and synchronizes its bound stream once. */
ATLAS_Q38_API int atlas_qwen38_ssm_stream_synchronize(
    void* stream, unsigned long long nonce);

ATLAS_Q38_API const char* atlas_qwen38_ssm_residual_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
