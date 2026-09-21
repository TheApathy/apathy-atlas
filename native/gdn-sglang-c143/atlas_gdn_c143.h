// SPDX-License-Identifier: AGPL-3.0-only
#ifndef ATLAS_GDN_C143_H
#define ATLAS_GDN_C143_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Thread-local diagnostic for the most recent ABI call on this host thread.
const char* atlas_gdn_c143_last_error(void);

// Build-script-injected identity for binding a loaded library to the exact
// candidate source bytes. Callers must require the fixed ABI prefix and compare
// the suffix with the source SHA256 before querying workspace or launching.
const char* atlas_gdn_c143_abi_identity(void);

// v1: compact BF16 Q[M,2048], K[M,2048], V[M,6144], FP32
// log_decay[M,48]/beta[M,48], FP32 state[48,128,128], and compact BF16
// output[M,6144]. All work is enqueued on stream_raw.
size_t atlas_gdn_c143_workspace_size(uint32_t seq_len);
int atlas_gdn_c143_launch(
    float* state_kv, const void* query_bf16, const void* key_bf16,
    const void* value_bf16, const float* log_decay, const float* beta,
    void* output_bf16, void* workspace, size_t workspace_bytes,
    uint32_t seq_len, void* stream_raw);

// v2: Atlas production layout. Q/K/V are BF16 views into qkv_base. Offsets
// and row strides are in BF16 elements, not bytes. For Qwen3.8 C1 the exact
// packed row is offsets 0/2048/4096 with strides 10240/10240/10240.
// gate_beta is FP32 [M,gate_beta_row_stride], where each row begins with
// alpha[48] followed by beta[48]; the exact production stride is 96.
//
// v2 appends compact log(max(alpha,1e-30)) scratch to the v1 workspace. The
// queried allocation may be shared by sequential layer calls on the same
// stream, but must not be reused concurrently or released before completion.
size_t atlas_gdn_c143_workspace_size_v2(uint32_t seq_len);
int atlas_gdn_c143_launch_v2(
    float* state_kv, const void* qkv_base_bf16,
    size_t query_base_offset, size_t key_base_offset, size_t value_base_offset,
    uint32_t query_row_stride, uint32_t key_row_stride,
    uint32_t value_row_stride, const float* gate_beta,
    uint32_t gate_beta_row_stride, void* output_bf16, void* workspace,
    size_t workspace_bytes, uint32_t seq_len, void* stream_raw);

// v3 retains the exact v2 production signature and fail-closed admission, but
// uses the pinned-c143 tile geometry internally: (NT,H) KKT/WU programs,
// four BV=32 recurrence programs per head, and two BV=64 output programs per
// chunk/head. v1/v2 remain available as numerical oracles. The queried v3
// workspace has the same caller-owned, same-stream lifetime contract as v2.
// v3 requires an explicit non-null stream; it never silently uses CUDA's
// process-global default stream.
size_t atlas_gdn_c143_workspace_size_v3(uint32_t seq_len);
int atlas_gdn_c143_launch_v3(
    float* state_kv, const void* qkv_base_bf16,
    size_t query_base_offset, size_t key_base_offset, size_t value_base_offset,
    uint32_t query_row_stride, uint32_t key_row_stride,
    uint32_t value_row_stride, const float* gate_beta,
    uint32_t gate_beta_row_stride, void* output_bf16, void* workspace,
    size_t workspace_bytes, uint32_t seq_len, void* stream_raw);

#ifdef __cplusplus
}
#endif

#endif  // ATLAS_GDN_C143_H
