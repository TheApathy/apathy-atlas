// SPDX-License-Identifier: AGPL-3.0-only
//
// Transposed-layout decode MoE — K=2 batch variant. Same semantics as
// moe_shared_expert_fused_batch2 but reads weight in `[K/2, N]` layout
// (input-major, prefill-coalesced) instead of `[N, K/2]`. See
// moe_shared_expert_fused_t.cu for the layout rationale.
//
// blockIdx.y: 0..2*top_k-1 routed (token = y/top_k, slot = y%top_k);
//             2*top_k..2*top_k+1 shared (token = y - 2*top_k).
// For gate_up: blockIdx.z = proj (0=gate, 1=up). silu_down has no z.
// Block: (128); each thread owns one output position `n`; lanes adjacent.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ARM-2 Phase-K RIDER 1: ONE E8M0 scale primitive shared by every native-MXFP4
// kernel entry (mx_block_scale / atlas_dec_e4m3). Previously this file carried
// a private atlas_dec_e4m3 copy; the header is the SSOT so the batched verify
// dequantizes bit-identically to the single-token decode it must agree with —
// a numeric skew between them shows up directly as lost MTP acceptance.
#include "mx_block_scale.cuh"

#define BLOCK_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_BATCH2_T[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ============================================================================
// DUAL-FORMAT templated impls (ARM-2 Phase-K RIDER A, ported to the batched
// verify). Routed experts (GS_R/E8M0_R) and the shared expert (GS_S/E8M0_S)
// may carry different quant formats — the native DeepSeek-V4 checkpoint is
// heterogeneous: routed experts are MXFP4 with per-32 E8M0 exponent scales,
// the shared expert is FP8->NVFP4 with per-16 E4M3 scales and a per-tensor
// global. Before this, the batched entries existed ONLY in the all-NVFP4
// flavor, so the m-row speculative verify had to fall back to firing the
// single-token kernel once per row (multi_seq/mod.rs), which re-reads every
// expert, the shared expert and the gate for each row.
//
// That fallback is not free: measured with ATLAS_MOE_OVERLAP=1 on
// DeepSeek-V4-Flash-162B at m=2/top_k=6, the two verify rows' expert sets
// overlap enough to leave 1.28x (learned-gate layers, ~93% of fires) to 2.01x
// (hash-routed layers, where both rows select the IDENTICAL expert set) of
// weight-read amortization on the table, plus the always-duplicated shared
// expert. The long-standing "the two verify tokens' expert sets are mostly
// disjoint" note in forward_k2.rs was an assumption, and it was wrong.
//
// `num_tokens` is a runtime parameter, so batch2 and batchN share one body.
// Grid: (ceil(N/BLOCK_SIZE), num_tokens*top_k + num_tokens, z).
//   blockIdx.y < num_tokens*top_k -> routed (token = y/top_k, slot = y%top_k)
//   blockIdx.y >= num_tokens*top_k -> shared (token = y - num_tokens*top_k)
// ============================================================================

template<int GS_R, bool E8M0_R, int GS_S, bool E8M0_S>
__device__ __forceinline__ void gate_up_shared_batchN_t_impl(
    const __nv_bfloat16* __restrict__ A,                    // [num_tokens, K]
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,                   // [num_tokens*top_k, N]
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,                     // [num_tokens*top_k, N]
    const unsigned int* __restrict__ expert_indices,        // [num_tokens*top_k]
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,                // [num_tokens, N]
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,                  // [num_tokens, N]
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int num_tokens
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y >= total_routed);

    unsigned int token, expert_slot;
    if (is_shared) {
        token = y - total_routed;
        expert_slot = 0;
    } else {
        token = y / top_k;
        expert_slot = y % top_k;
    }

    const __nv_bfloat16* A_token = A + (unsigned long long)token * K;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;
    unsigned long long c_offset = 0;

    if (is_shared) {
        if (proj == 0) {
            if (sh_gate_t_packed == 0) {
                const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
                if (n < N) sh_gate_out[(unsigned long long)token * N + n] = __float2bfloat16(0.0f);
                return;
            }
            B_packed = sh_gate_t_packed; B_scale = sh_gate_t_scale; s2 = sh_gate_s2;
            C = sh_gate_out; c_offset = (unsigned long long)token * N;
        } else {
            if (sh_up_t_packed == 0) {
                const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
                if (n < N) sh_up_out[(unsigned long long)token * N + n] = __float2bfloat16(0.0f);
                return;
            }
            B_packed = sh_up_t_packed; B_scale = sh_up_t_scale; s2 = sh_up_s2;
            C = sh_up_out; c_offset = (unsigned long long)token * N;
        }
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_t_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id];
            C = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_t_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id];
            C = up_out;
        }
        c_offset = (unsigned long long)flat_slot * N;
        if (B_packed == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) C[c_offset + n] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool valid = (n < N);

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH2_T[threadIdx.x];
    __syncthreads();
    if (!valid) return;

    float acc = 0.0f;
    // ONE parameterized accumulation body; the same-format instantiation
    // collapses it to a single branchless loop so the NVFP4 entries stay
    // PTX-identical to the pre-template kernel.
    #define GATEUP_BATCHN_ACCUM(GS_, E8M0_) do { \
        const unsigned int num_groups = K / (GS_); \
        for (unsigned int sg = 0; sg < num_groups; sg++) { \
            unsigned char sb = B_scale[(unsigned long long)sg * N + n]; \
            float sc = mx_block_scale<(E8M0_)>(sb, s2); \
            const unsigned int kh_base = sg * ((GS_) / 2); \
            _Pragma("unroll") \
            for (unsigned int kh_off = 0; kh_off < ((GS_) / 2); kh_off++) { \
                unsigned int k_half = kh_base + kh_off; \
                unsigned char byte = B_packed[(unsigned long long)k_half * N + n]; \
                float a_lo = __bfloat162float(A_token[k_half * 2]); \
                float a_hi = __bfloat162float(A_token[k_half * 2 + 1]); \
                float w_lo = s_lut[byte & 0xFu] * sc; \
                float w_hi = s_lut[(byte >> 4) & 0xFu] * sc; \
                acc += a_lo * w_lo + a_hi * w_hi; \
            } \
        } \
    } while(0)
    if constexpr (GS_R == GS_S && E8M0_R == E8M0_S) {
        GATEUP_BATCHN_ACCUM(GS_R, E8M0_R);
    } else if (is_shared) {
        GATEUP_BATCHN_ACCUM(GS_S, E8M0_S);
    } else {
        GATEUP_BATCHN_ACCUM(GS_R, E8M0_R);
    }
    #undef GATEUP_BATCHN_ACCUM
    C[c_offset + n] = __float2bfloat16(acc);
}

template<int GS_R, bool E8M0_R, int GS_S, bool E8M0_S>
__device__ __forceinline__ void silu_down_shared_batchN_t_impl(
    const __nv_bfloat16* __restrict__ gate_out,             // [num_tokens*top_k, K]
    const __nv_bfloat16* __restrict__ up_out,               // [num_tokens*top_k, K]
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,                          // [num_tokens*top_k, N]
    const unsigned int* __restrict__ expert_indices,        // [num_tokens*top_k]
    const __nv_bfloat16* __restrict__ sh_gate_in,           // [num_tokens, K]
    const __nv_bfloat16* __restrict__ sh_up_in,             // [num_tokens, K]
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,                // [num_tokens, N]
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int num_tokens
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);
    unsigned int token, expert_slot;
    if (is_shared) {
        token = y - total_routed;
        expert_slot = 0;
    } else {
        token = y / top_k;
        expert_slot = y % top_k;
    }

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;
    unsigned long long c_offset;

    if (is_shared) {
        if (sh_down_t_packed == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) sh_down_out[(unsigned long long)token * N + n] = __float2bfloat16(0.0f);
            return;
        }
        B_packed = sh_down_t_packed; B_scale = sh_down_t_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in + (unsigned long long)token * K;
        u_ptr = sh_up_in + (unsigned long long)token * K;
        c_offset = (unsigned long long)token * N;  // sh_down_out
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        B_packed = (const unsigned char*)packed_t_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_t_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)flat_slot * K;
        u_ptr = up_out + (unsigned long long)flat_slot * K;
        c_offset = (unsigned long long)flat_slot * N;
        if (B_packed == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) C[c_offset + n] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool valid = (n < N);

    extern __shared__ float s_act[];
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH2_T[threadIdx.x];
    __syncthreads();
    if (!valid) return;

    float acc = 0.0f;
    #define SILUDOWN_BATCHN_ACCUM(GS_, E8M0_) do { \
        const unsigned int num_groups = K / (GS_); \
        for (unsigned int sg = 0; sg < num_groups; sg++) { \
            unsigned char sb = B_scale[(unsigned long long)sg * N + n]; \
            float sc = mx_block_scale<(E8M0_)>(sb, s2); \
            const unsigned int kh_base = sg * ((GS_) / 2); \
            _Pragma("unroll") \
            for (unsigned int kh_off = 0; kh_off < ((GS_) / 2); kh_off++) { \
                unsigned int k_half = kh_base + kh_off; \
                unsigned char byte = B_packed[(unsigned long long)k_half * N + n]; \
                float w_lo = s_lut[byte & 0xFu] * sc; \
                float w_hi = s_lut[(byte >> 4) & 0xFu] * sc; \
                acc += s_act[k_half * 2] * w_lo + s_act[k_half * 2 + 1] * w_hi; \
            } \
        } \
    } while(0)
    if constexpr (GS_R == GS_S && E8M0_R == E8M0_S) {
        SILUDOWN_BATCHN_ACCUM(GS_R, E8M0_R);
    } else if (is_shared) {
        SILUDOWN_BATCHN_ACCUM(GS_S, E8M0_S);
    } else {
        SILUDOWN_BATCHN_ACCUM(GS_R, E8M0_R);
    }
    #undef SILUDOWN_BATCHN_ACCUM

    if (is_shared) {
        sh_down_out[c_offset + n] = __float2bfloat16(acc);
    } else {
        C[c_offset + n] = __float2bfloat16(acc);
    }
}

// ── Entry macros ────────────────────────────────────────────────────────────
// NTOK_: 2 for the fixed batch2 entries, `num_tokens` for the batchN entries.
#define GATEUP_BATCHN_ENTRY(NAME, GS_R_, E8M0_R_, NTOK_, NTOK_PARAM_)          \
extern "C" __global__ void NAME(                                               \
    const __nv_bfloat16* __restrict__ A,                                       \
    const unsigned long long* __restrict__ gate_packed_t_ptrs,                 \
    const unsigned long long* __restrict__ gate_scale_t_ptrs,                  \
    const float* __restrict__ gate_scale2_vals,                                \
    __nv_bfloat16* __restrict__ gate_out,                                      \
    const unsigned long long* __restrict__ up_packed_t_ptrs,                   \
    const unsigned long long* __restrict__ up_scale_t_ptrs,                    \
    const float* __restrict__ up_scale2_vals,                                  \
    __nv_bfloat16* __restrict__ up_out,                                        \
    const unsigned int* __restrict__ expert_indices,                           \
    const unsigned char* __restrict__ sh_gate_t_packed,                        \
    const unsigned char* __restrict__ sh_gate_t_scale,                         \
    float sh_gate_s2,                                                          \
    __nv_bfloat16* __restrict__ sh_gate_out,                                   \
    const unsigned char* __restrict__ sh_up_t_packed,                          \
    const unsigned char* __restrict__ sh_up_t_scale,                           \
    float sh_up_s2,                                                            \
    __nv_bfloat16* __restrict__ sh_up_out,                                     \
    unsigned int N, unsigned int K, unsigned int top_k NTOK_PARAM_             \
) {                                                                            \
    gate_up_shared_batchN_t_impl<(GS_R_), (E8M0_R_), GROUP_SIZE, false>(       \
        A, gate_packed_t_ptrs, gate_scale_t_ptrs, gate_scale2_vals, gate_out,  \
        up_packed_t_ptrs, up_scale_t_ptrs, up_scale2_vals, up_out,             \
        expert_indices, sh_gate_t_packed, sh_gate_t_scale, sh_gate_s2,         \
        sh_gate_out, sh_up_t_packed, sh_up_t_scale, sh_up_s2, sh_up_out,       \
        N, K, top_k, (NTOK_));                                                 \
}

#define SILUDOWN_BATCHN_ENTRY(NAME, GS_R_, E8M0_R_, NTOK_, NTOK_PARAM_)        \
extern "C" __global__ void NAME(                                               \
    const __nv_bfloat16* __restrict__ gate_out,                                \
    const __nv_bfloat16* __restrict__ up_out,                                  \
    const unsigned long long* __restrict__ packed_t_ptrs,                      \
    const unsigned long long* __restrict__ scale_t_ptrs,                       \
    const float* __restrict__ scale2_vals,                                     \
    __nv_bfloat16* __restrict__ C,                                             \
    const unsigned int* __restrict__ expert_indices,                           \
    const __nv_bfloat16* __restrict__ sh_gate_in,                              \
    const __nv_bfloat16* __restrict__ sh_up_in,                                \
    const unsigned char* __restrict__ sh_down_t_packed,                        \
    const unsigned char* __restrict__ sh_down_t_scale,                         \
    float sh_down_s2,                                                          \
    __nv_bfloat16* __restrict__ sh_down_out,                                   \
    unsigned int N, unsigned int K, unsigned int top_k NTOK_PARAM_             \
) {                                                                            \
    silu_down_shared_batchN_t_impl<(GS_R_), (E8M0_R_), GROUP_SIZE, false>(     \
        gate_out, up_out, packed_t_ptrs, scale_t_ptrs, scale2_vals, C,         \
        expert_indices, sh_gate_in, sh_up_in, sh_down_t_packed,                \
        sh_down_t_scale, sh_down_s2, sh_down_out, N, K, top_k, (NTOK_));       \
}

#define NTOK_FIXED
#define NTOK_RUNTIME , unsigned int num_tokens

// NVFP4 (default): routed and shared both per-16 E4M3 + per-tensor global.
GATEUP_BATCHN_ENTRY(moe_expert_gate_up_shared_batch2_t, GROUP_SIZE, false, 2u, NTOK_FIXED)
SILUDOWN_BATCHN_ENTRY(moe_expert_silu_down_shared_batch2_t, GROUP_SIZE, false, 2u, NTOK_FIXED)
GATEUP_BATCHN_ENTRY(moe_expert_gate_up_shared_batchN_t, GROUP_SIZE, false, num_tokens, NTOK_RUNTIME)
SILUDOWN_BATCHN_ENTRY(moe_expert_silu_down_shared_batchN_t, GROUP_SIZE, false, num_tokens, NTOK_RUNTIME)

// Native MXFP4 (ARM-2): ROUTED experts E8M0 per-32 (no per-tensor global);
// SHARED expert stays NVFP4 — the native V4 checkpoint ships the shared expert
// FP8->NVFP4, not MXFP4. Mirrors moe_expert_gate_up_shared_t_e8m0 exactly, so
// the batched verify and the single-token decode dequantize identically.
GATEUP_BATCHN_ENTRY(moe_expert_gate_up_shared_batch2_t_e8m0, 32, true, 2u, NTOK_FIXED)
SILUDOWN_BATCHN_ENTRY(moe_expert_silu_down_shared_batch2_t_e8m0, 32, true, 2u, NTOK_FIXED)
GATEUP_BATCHN_ENTRY(moe_expert_gate_up_shared_batchN_t_e8m0, 32, true, num_tokens, NTOK_RUNTIME)
SILUDOWN_BATCHN_ENTRY(moe_expert_silu_down_shared_batchN_t_e8m0, 32, true, num_tokens, NTOK_RUNTIME)
