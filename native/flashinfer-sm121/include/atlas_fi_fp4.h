#ifndef ATLAS_FI_FP4_H_
#define ATLAS_FI_FP4_H_

#include <stddef.h>

#if defined(_WIN32)
#define ATLAS_FI_API __declspec(dllexport)
#else
#define ATLAS_FI_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
  ATLAS_FI_FP4_OK = 0,
  ATLAS_FI_FP4_INVALID_ARGUMENT = -1,
  ATLAS_FI_FP4_CUTLASS_ERROR = -2,
  ATLAS_FI_FP4_UNKNOWN_ERROR = -3,
};

/*
 * Tactics 0..5 are, in order:
 *   128x128x128 DP, 128x128x128 StreamK,
 *   128x128x256 DP, 128x128x256 StreamK,
 *   256x128x128 DP, 256x128x128 StreamK.
 *
 * This query performs CUTLASS host-side argument construction only. The caller
 * must not assume the result is shape-independent.
 */
ATLAS_FI_API int atlas_fi_nvfp4_sm121_workspace_size(
    int tactic, int m, int n, int k, int batch_count,
    size_t* workspace_bytes_out);

/*
 * D: BF16 row-major [batch,m,n]
 * A: packed E2M1 row-major [batch,m,k/2]
 * B: packed E2M1 column-major logical [batch,n,k/2]
 * A_sf/B_sf: UE4M3 FlashInfer/CUTLASS 128x4 interleaved scale layouts
 * global_sf: device pointer to one FP32 value
 * workspace: device pointer with at least workspace_bytes bytes
 * stream: cudaStream_t represented as an opaque pointer
 */
ATLAS_FI_API int atlas_fi_nvfp4_sm121_bf16(
    int tactic, void* d, const void* a, const void* b, const void* a_sf,
    const void* b_sf, const float* global_sf, int m, int n, int k,
    int batch_count, void* workspace, size_t workspace_bytes, void* stream);

/* Thread-local diagnostic for the most recent call on this host thread. */
ATLAS_FI_API const char* atlas_fi_nvfp4_sm121_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
