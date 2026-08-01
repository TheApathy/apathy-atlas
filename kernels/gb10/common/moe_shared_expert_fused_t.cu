// SPDX-License-Identifier: AGPL-3.0-only
//
// Transposed-layout decode MoE GEMV — Phase 8a unified-layout MoE.
//
// Same semantics as moe_expert_silu_down_shared / moe_expert_gate_up_shared,
// but reads the weight tensor in transposed `[K/2, N]` layout (input-major)
// instead of the current `[N, K/2]` layout (output-major). Goal: a single
// weight layout shared between prefill and decode so we can free the
// untransposed copies (~59 GB on MiniMax M2.7-NVFP4 EP=2) and either
// run the persistent down transpose or use the headroom for KV cache.
//
// Coalescence strategy is inverted vs the original kernel:
//   - original: each thread owns ONE output (sub-group reduces over K)
//     → reads B[n_lane * (K/2) + k_iter*8] = 8 bytes contiguous per lane
//   - transposed: each thread STILL owns one output, but threads in a
//     warp have ADJACENT n's (lane = n within warp). Read B_t[k_half * N + n]:
//     32 bytes contiguous per warp = 1 cache line per K iter. No warp
//     reduction needed (each lane has its own accumulator + own output).
//
// Block: (128, 1, 1)  Grid: (ceil(N/128), top_k+1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ARM-2 Phase-K RIDER 1: the E8M0 scale primitive (mx_block_scale / atlas_dec_e4m3)
// lives in ONE shared header, included by both Family A (this file) and Family B
// (../qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu) — bit-identical across
// families, no second copy.
#include "mx_block_scale.cuh"

// Tuning note: 32-thread blocks (1 warp per block) outperform 128-thread blocks
// on GB10 for the transposed decode silu_down — more blocks → better SM
// occupancy, the s_act shared-mem precompute parallelism is unchanged because
// the K loop dominates (block_size irrelevant once each thread is 1 output).
// The kernels now index off `blockDim.x` rather than this constant, so the
// launch side (`T_BLOCK` in `fp8_moe.rs`) owns the choice; keep the two in sync
// if you sweep it.
#define GROUP_SIZE 16

// atlas_dec_e4m3 + mx_block_scale<E8M0> now live in mx_block_scale.cuh (RIDER 1).

// Branch-free E2M1 nibble -> float. Reproduces the retired E2M1_LUT_T table
//   { 0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6 }
// exactly for all 16 codes, signed zero included; the microtest checks all 16.
//
// This replaces a `__shared__ float s_lut[16]` lookup. The table version costs
// an LDS per nibble — two per weight byte — and the index is the freshly loaded
// weight nibble, so every global load carries a dependent shared-memory load
// before its FMA. Worse, 32 lanes index 16 entries at random, so those LDS ops
// bank-conflict several ways. That chain is what held the decode MoE GEMV at
// ~127 GB/s: the kernel could not keep enough loads in flight to cover DRAM
// latency. Pure ALU has no such dependency.
//
// Encoding: nib = [s|e1 e0|m]. e==0 -> subnormal (0.0 or 0.5); otherwise the
// IEEE-754 fields fall out directly as exponent 126+e, mantissa m<<22.
__device__ __forceinline__ float e2m1_decode(unsigned int nib) {
    const unsigned int m = nib & 1u;
    const unsigned int e = (nib >> 1) & 3u;
    const unsigned int s = (nib >> 3) & 1u;
    const unsigned int mag = (e == 0u) ? (m ? 0x3F000000u : 0u)
                                       : (((126u + e) << 23) | (m << 22));
    return __int_as_float(mag | (s << 31));
}

// VEC = outputs owned per thread (adjacent in N). VEC=1 is the original
// 1-byte-per-lane load: 32 contiguous bytes per warp per K iteration. Measured
// on GB10 (moe_unified_t_v4_microtest), a 32-byte-per-warp request pins this
// GEMV at ~130 GB/s no matter how many warps are resident — 4× the warps moved
// it only 128 -> 136 GB/s. A 128-byte request (VEC=4) has no such ceiling: the
// same sweep ran 67 -> 118 -> 194 GB/s purely on added warps. So width sets the
// ceiling and warp count sets how close you get to it, and VEC>1 must be paired
// with enough parallelism to pay for the warps it gives up.
//
// The per-output accumulation order is untouched (same k sequence, same FMA
// order per n), so every VEC is bit-identical; the microtest gates on that.
template<int VEC>
__device__ __forceinline__ void load_vec_u8(const unsigned char* __restrict__ p,
                                            unsigned char (&out)[VEC]) {
    if constexpr (VEC == 4) {
        const uchar4 v = *reinterpret_cast<const uchar4*>(p);
        out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
    } else if constexpr (VEC == 2) {
        const uchar2 v = *reinterpret_cast<const uchar2*>(p);
        out[0] = v.x; out[1] = v.y;
    } else {
        #pragma unroll
        for (int i = 0; i < VEC; ++i) out[i] = p[i];
    }
}

// Store VEC consecutive bf16 outputs in one instruction (VEC=4 -> 8 bytes,
// VEC=2 -> 4 bytes) rather than VEC separate 2-byte stores.
template<int VEC>
__device__ __forceinline__ void store_vec_bf16(__nv_bfloat16* __restrict__ p,
                                               const float (&acc)[VEC]) {
    if constexpr (VEC == 4 || VEC == 2) {
        __nv_bfloat16 b[VEC];
        #pragma unroll
        for (int v = 0; v < VEC; ++v) b[v] = __float2bfloat16(acc[v]);
        if constexpr (VEC == 4) {
            *reinterpret_cast<uint2*>(p) = *reinterpret_cast<const uint2*>(b);
        } else {
            *reinterpret_cast<unsigned int*>(p) = *reinterpret_cast<const unsigned int*>(b);
        }
    } else {
        #pragma unroll
        for (int v = 0; v < VEC; ++v) p[v] = __float2bfloat16(acc[v]);
    }
}

// Transposed-layout fused gate+up decode kernel.
//
// Single-token GEMV: for each routed expert slot (top_k of them) plus the
// shared expert, compute gate_out = A @ W_gate^T and up_out = A @ W_up^T.
// blockIdx.z selects gate (0) or up (1). Same coalescence strategy as
// silu_down_t — each thread owns one output, lanes within warp adjacent
// in N.
// ARM-2 Phase-K RIDER A: DUAL-FORMAT. Routed experts (E8M0_R/GS_R) and the
// shared expert (E8M0_S/GS_S) can carry DIFFERENT quant formats — the native
// V4 ckpt is heterogeneous (routed MXFP4-E8M0, shared FP8→NVFP4). The format is
// keyed off the weight's `WeightQuantFormat` tag (Rust dispatch, asserted via
// `expect`), NOT positionally; `is_shared` only selects the region. The branch
// is BLOCK-UNIFORM (grid.y = expert_slot, one expert per block — RIDER A2).
//
// SPLIT>1 (split-K): grid.z is 2*SPLIT instead of 2, and block (proj, ks) owns
// only k in [ks*K/SPLIT, (ks+1)*K/SPLIT). That is how a VEC>1 kernel gets its
// warps back — widening the request costs parallelism, and on GB10's 25 SMs the
// VEC=2/VEC=4 kernels are warp-starved at production top_k=6 (see the sweep note
// on load_vec_u8). Each block writes an f32 partial and
// `moe_gate_up_partial_finalize` sums the SPLIT of them in ascending-ks order —
// a fixed order, not atomicAdd, so decode stays bit-reproducible run to run.
// It is NOT bit-equal to SPLIT==1, which adds the same terms in one sweep.
template<int GS_R, bool E8M0_R, int GS_S, bool E8M0_S, int VEC = 1, int SPLIT = 1>
__device__ __forceinline__ void gate_up_shared_t_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert direct pointers (transposed).
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k,
    // SPLIT>1 only: f32 scratch `[2, SPLIT, top_k+1, N]`. Ignored when SPLIT==1.
    float* __restrict__ partial = nullptr
) {
    const unsigned int expert_slot = blockIdx.y;
    // blockIdx.z packs (proj, ksplit) in that order, so the partial buffer is
    // laid out [proj][ks][slot][n] with blockIdx.z as its leading index.
    const unsigned int proj = (SPLIT == 1) ? blockIdx.z : (blockIdx.z / SPLIT);
    const unsigned int ks = (SPLIT == 1) ? 0u : (blockIdx.z % SPLIT);
    const bool is_shared = (expert_slot == top_k);
    // First of the VEC adjacent outputs this thread owns.
    const unsigned int n = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;

    float* const out_f32 = (SPLIT == 1) ? nullptr
        : partial + ((unsigned long long)blockIdx.z * (top_k + 1) + expert_slot) * N;
    // Row base for the SPLIT==1 store; unused (and left null) under split-K.
    __nv_bfloat16* C_row = nullptr;
    const auto emit = [&](const float (&vals)[VEC]) {
        if constexpr (SPLIT == 1) {
            store_vec_bf16<VEC>(C_row + n, vals);
        } else {
            #pragma unroll
            for (int v = 0; v < VEC; ++v) out_f32[n + v] = vals[v];
        }
    };
    // Runs before the VEC alignment guard below, so it masks per element.
    const auto emit_zero = [&]() {
        #pragma unroll
        for (int v = 0; v < VEC; ++v) {
            if (n + v >= N) continue;
            if constexpr (SPLIT == 1) C_row[n + v] = __float2bfloat16(0.0f);
            else out_f32[n + v] = 0.0f;
        }
    };

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;

    if (is_shared) {
        if (proj == 0) {
            C_row = sh_gate_out;
            if (sh_gate_t_packed == 0) {
                emit_zero();
                return;
            }
            B_packed = sh_gate_t_packed;
            B_scale = sh_gate_t_scale;
            s2 = sh_gate_s2;
        } else {
            C_row = sh_up_out;
            if (sh_up_t_packed == 0) {
                emit_zero();
                return;
            }
            B_packed = sh_up_t_packed;
            B_scale = sh_up_t_scale;
            s2 = sh_up_s2;
        }
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_t_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id];
            C_row = gate_out + (unsigned long long)expert_slot * N;
        } else {
            B_packed = (const unsigned char*)up_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_t_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id];
            C_row = up_out + (unsigned long long)expert_slot * N;
        }
        if (B_packed == 0) {
            emit_zero();
            return;
        }
    }

    // VEC>1 loads/stores the whole group with one instruction, so a partial
    // group cannot be served. Callers must launch VEC>1 only when N divides
    // evenly into blockDim.x*VEC (`fp8_moe.rs` enforces this); the guard below
    // is what makes a mis-sized launch drop work instead of corrupting memory.
    const bool valid = (VEC == 1) ? (n < N) : (n + VEC <= N);
    if (!valid) return;

    // GEMV: C[n] = sum_k A[k] * W[n, k]. With transposed weight stored as
    // [K/2, N] packed: each byte at (k_half, n) holds two consecutive k
    // nibbles for output position n. Iterate by scale-group (16 K) to
    // cache the per-group scale.
    // Block-uniform dual-format accumulation. ONE parameterized macro so the
    // shared and routed paths run byte-identical logic differing only in
    // (GS, E8M0) — the shared NVFP4 branch stays bit-identical to the baseline
    // kernel (RIDER A3); the compile-time template keeps each fully unrolled.
    float acc[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) acc[v] = 0.0f;
    #define GATEUP_ACCUM(GS_, E8M0_) do { \
        const unsigned int gpk = K / (GS_) / SPLIT; \
        for (unsigned int sg = ks * gpk; sg < (ks + 1) * gpk; sg++) { \
            unsigned char sb[VEC]; \
            load_vec_u8<VEC>(B_scale + (unsigned long long)sg * N + n, sb); \
            float sc[VEC]; \
            _Pragma("unroll") \
            for (int v = 0; v < VEC; ++v) sc[v] = mx_block_scale<(E8M0_)>(sb[v], s2); \
            const unsigned int kh_base = sg * ((GS_) / 2); \
            _Pragma("unroll") \
            for (unsigned int kh_off = 0; kh_off < ((GS_) / 2); kh_off++) { \
                unsigned int k_half = kh_base + kh_off; \
                unsigned char byte[VEC]; \
                load_vec_u8<VEC>(B_packed + (unsigned long long)k_half * N + n, byte); \
                float a_lo = __bfloat162float(A[k_half * 2]); \
                float a_hi = __bfloat162float(A[k_half * 2 + 1]); \
                _Pragma("unroll") \
                for (int v = 0; v < VEC; ++v) { \
                    float w_lo = e2m1_decode(byte[v] & 0xFu) * sc[v]; \
                    float w_hi = e2m1_decode((byte[v] >> 4) & 0xFu) * sc[v]; \
                    acc[v] += a_lo * w_lo + a_hi * w_hi; \
                } \
            } \
        } \
    } while(0)
    // Same-format wrappers (NVFP4 default) collapse to a SINGLE loop — no
    // branch, PTX-identical to the baseline kernel (RIDER A3). Only the
    // heterogeneous e8m0 wrapper (routed≠shared) emits the block-uniform branch.
    if constexpr (GS_R == GS_S && E8M0_R == E8M0_S) {
        GATEUP_ACCUM(GS_R, E8M0_R);
    } else {
        if (is_shared) { GATEUP_ACCUM(GS_S, E8M0_S); }
        else           { GATEUP_ACCUM(GS_R, E8M0_R); }
    }
    #undef GATEUP_ACCUM

    emit(acc);
}

// NVFP4 (default): FP8-E4M3 per-16 scales × per-tensor global.
extern "C" __global__ void moe_expert_gate_up_shared_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    gate_up_shared_t_impl<GROUP_SIZE, false, GROUP_SIZE, false>(
        A, gate_packed_t_ptrs, gate_scale_t_ptrs, gate_scale2_vals, gate_out,
        up_packed_t_ptrs, up_scale_t_ptrs, up_scale2_vals, up_out, expert_indices,
        sh_gate_t_packed, sh_gate_t_scale, sh_gate_s2, sh_gate_out,
        sh_up_t_packed, sh_up_t_scale, sh_up_s2, sh_up_out, N, K, top_k);
}

// Native MXFP4 (ARM-2): ROUTED experts E8M0 per-32 (no global); SHARED expert
// stays NVFP4 (`<GROUP_SIZE,false>`) — the native V4 ckpt ships the shared
// expert FP8→NVFP4, NOT MXFP4. Routed buffers are the E8M0-tagged
// (`WeightQuantFormat::Mxfp4E8m0`) transcode-free loader output; sh_* are NVFP4.
extern "C" __global__ void moe_expert_gate_up_shared_t_e8m0(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    gate_up_shared_t_impl<32, true, GROUP_SIZE, false>(
        A, gate_packed_t_ptrs, gate_scale_t_ptrs, gate_scale2_vals, gate_out,
        up_packed_t_ptrs, up_scale_t_ptrs, up_scale2_vals, up_out, expert_indices,
        sh_gate_t_packed, sh_gate_t_scale, sh_gate_s2, sh_gate_out,
        sh_up_t_packed, sh_up_t_scale, sh_up_s2, sh_up_out, N, K, top_k);
}

// ── VEC=4 variants ──
// Identical math and identical per-output accumulation order; each thread owns
// 4 adjacent n so a warp requests 128 contiguous bytes per K iteration instead
// of 32. Launch with grid.x = N/(blockDim.x*4); `fp8_moe.rs` only dispatches
// these when that divides evenly.
extern "C" __global__ void moe_expert_gate_up_shared_t_v4(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    gate_up_shared_t_impl<GROUP_SIZE, false, GROUP_SIZE, false, 4>(
        A, gate_packed_t_ptrs, gate_scale_t_ptrs, gate_scale2_vals, gate_out,
        up_packed_t_ptrs, up_scale_t_ptrs, up_scale2_vals, up_out, expert_indices,
        sh_gate_t_packed, sh_gate_t_scale, sh_gate_s2, sh_gate_out,
        sh_up_t_packed, sh_up_t_scale, sh_up_s2, sh_up_out, N, K, top_k);
}

extern "C" __global__ void moe_expert_gate_up_shared_t_e8m0_v4(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    gate_up_shared_t_impl<32, true, GROUP_SIZE, false, 4>(
        A, gate_packed_t_ptrs, gate_scale_t_ptrs, gate_scale2_vals, gate_out,
        up_packed_t_ptrs, up_scale_t_ptrs, up_scale2_vals, up_out, expert_indices,
        sh_gate_t_packed, sh_gate_t_scale, sh_gate_s2, sh_gate_out,
        sh_up_t_packed, sh_up_t_scale, sh_up_s2, sh_up_out, N, K, top_k);
}

// ── VEC=2 variants ──
// 64 bytes per warp per K iteration: half the request width of the _v4 pair,
// but only half the parallelism given up. Probe for whether the ~130 GB/s
// 32-byte ceiling lifts before the full 128-byte width.
extern "C" __global__ void moe_expert_gate_up_shared_t_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    gate_up_shared_t_impl<GROUP_SIZE, false, GROUP_SIZE, false, 2>(
        A, gate_packed_t_ptrs, gate_scale_t_ptrs, gate_scale2_vals, gate_out,
        up_packed_t_ptrs, up_scale_t_ptrs, up_scale2_vals, up_out, expert_indices,
        sh_gate_t_packed, sh_gate_t_scale, sh_gate_s2, sh_gate_out,
        sh_up_t_packed, sh_up_t_scale, sh_up_s2, sh_up_out, N, K, top_k);
}

extern "C" __global__ void moe_expert_gate_up_shared_t_e8m0_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    gate_up_shared_t_impl<32, true, GROUP_SIZE, false, 2>(
        A, gate_packed_t_ptrs, gate_scale_t_ptrs, gate_scale2_vals, gate_out,
        up_packed_t_ptrs, up_scale_t_ptrs, up_scale2_vals, up_out, expert_indices,
        sh_gate_t_packed, sh_gate_t_scale, sh_gate_s2, sh_gate_out,
        sh_up_t_packed, sh_up_t_scale, sh_up_s2, sh_up_out, N, K, top_k);
}

// Transposed-layout silu_down decode kernel.
//
// Per-expert weight buffers `[K/2, N]` packed NVFP4 + `[K/16, N]` FP8 scales.
// Input gate_out / up_out: `[(top_k+1), K]` BF16 (per-slot). top_k slot is
// the shared-expert input. Output C: `[top_k, N]` BF16; shared-expert
// output goes to `sh_down_out: [N]`.
// ARM-2 Phase-K RIDER A: DUAL-FORMAT (see gate_up_shared_t_impl). Routed
// (GS_R/E8M0_R) vs shared (GS_S/E8M0_S), block-uniform on is_shared.
// SPLIT>1: blockIdx.z is the k-split index (grid.z = SPLIT). Beyond restoring
// warps (see gate_up_shared_t_impl), split-K shrinks the `s_act` shared buffer
// from K*4 to K*4/SPLIT bytes, which is what was capping blocks-per-SM here —
// 8 KB/block at K=2048 allows only ~12 blocks/SM.
template<int GS_R, bool E8M0_R, int GS_S, bool E8M0_S, int VEC = 1, int SPLIT = 1>
__device__ __forceinline__ void silu_down_shared_t_impl(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,   // [num_experts] device ptrs to [K/2 * N] bytes
    const unsigned long long* __restrict__ scale_t_ptrs,    // [num_experts] device ptrs to [K/16 * N] bytes
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert direct pointers (transposed layout)
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k,
    // SPLIT>1 only: f32 scratch `[SPLIT, top_k+1, N]`. Ignored when SPLIT==1.
    float* __restrict__ partial = nullptr
) {
    const unsigned int expert_slot = blockIdx.y;
    const unsigned int ks = (SPLIT == 1) ? 0u : blockIdx.z;
    const bool is_shared = (expert_slot == top_k);
    const unsigned int n = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;

    float* const out_f32 = (SPLIT == 1) ? nullptr
        : partial + ((unsigned long long)ks * (top_k + 1) + expert_slot) * N;
    __nv_bfloat16* C_row = nullptr;
    const auto emit = [&](const float (&vals)[VEC]) {
        if constexpr (SPLIT == 1) {
            store_vec_bf16<VEC>(C_row + n, vals);
        } else {
            #pragma unroll
            for (int v = 0; v < VEC; ++v) out_f32[n + v] = vals[v];
        }
    };
    const auto emit_zero = [&]() {
        #pragma unroll
        for (int v = 0; v < VEC; ++v) {
            if (n + v >= N) continue;
            if constexpr (SPLIT == 1) C_row[n + v] = __float2bfloat16(0.0f);
            else out_f32[n + v] = 0.0f;
        }
    };

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;
    if (is_shared) {
        C_row = sh_down_out;
        if (sh_down_t_packed == 0) {
            // No shared expert — write zeros.
            emit_zero();
            return;
        }
        B_packed = sh_down_t_packed;
        B_scale = sh_down_t_scale;
        s2 = sh_down_s2;
        g_ptr = sh_gate_in;
        u_ptr = sh_up_in;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_packed = (const unsigned char*)packed_t_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_t_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)expert_slot * K;
        u_ptr = up_out + (unsigned long long)expert_slot * K;
        C_row = C + (unsigned long long)expert_slot * N;
        // EP remote expert: NULL pointer → write zero output and return.
        if (B_packed == 0) {
            emit_zero();
            return;
        }
    }

    // See the VEC note on gate_up_shared_t_impl: a partial group cannot be
    // served by the vector load/store, so VEC>1 needs N % (blockDim.x*VEC) == 0.
    const bool valid = (VEC == 1) ? (n < N) : (n + VEC <= N);

    // Phase 1: cooperatively precompute s_act = SiLU(gate) * up over this
    // block's k slice only — [k_lo, k_lo + K/SPLIT). Callers size the dynamic
    // shared memory to K*4/SPLIT to match.
    const unsigned int k_len = K / SPLIT;
    const unsigned int k_lo = (SPLIT == 1) ? 0u : ks * k_len;
    extern __shared__ float s_act[];
    for (unsigned int i = threadIdx.x; i < k_len; i += blockDim.x) {
        float gf = __bfloat162float(g_ptr[k_lo + i]);
        float uf = __bfloat162float(u_ptr[k_lo + i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }

    __syncthreads(); // s_act is read by every thread below

    if (!valid) return;

    // Phase 2: per-thread accumulate over K_half iterations. Each thread
    // owns VEC adjacent output positions `n`; lanes in a warp have adjacent
    // n's so `B_packed[k_half * N + n]` reads are coalesced (VEC bytes per
    // lane, 32*VEC bytes contiguous per warp per iter).
    const unsigned int K_half = K / 2;
    float acc[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) acc[v] = 0.0f;

    // Block-uniform dual-format accumulation (RIDER A). Cache per-group scale
    // in a register; iterate GS/2 K_half iters per group.
    #define SILUDOWN_ACCUM(GS_, E8M0_) do { \
        const unsigned int gpk = K / (GS_) / SPLIT; \
        for (unsigned int sg = ks * gpk; sg < (ks + 1) * gpk; sg++) { \
            unsigned char sb[VEC]; \
            load_vec_u8<VEC>(B_scale + (unsigned long long)sg * N + n, sb); \
            float sc[VEC]; \
            _Pragma("unroll") \
            for (int v = 0; v < VEC; ++v) sc[v] = mx_block_scale<(E8M0_)>(sb[v], s2); \
            const unsigned int kh_base = sg * ((GS_) / 2); \
            _Pragma("unroll") \
            for (unsigned int kh_off = 0; kh_off < ((GS_) / 2); kh_off++) { \
                unsigned int k_half = kh_base + kh_off; \
                unsigned char byte[VEC]; \
                load_vec_u8<VEC>(B_packed + (unsigned long long)k_half * N + n, byte); \
                float a_lo = s_act[k_half * 2 - k_lo]; \
                float a_hi = s_act[k_half * 2 + 1 - k_lo]; \
                _Pragma("unroll") \
                for (int v = 0; v < VEC; ++v) { \
                    float w_lo = e2m1_decode(byte[v] & 0xFu) * sc[v]; \
                    float w_hi = e2m1_decode((byte[v] >> 4) & 0xFu) * sc[v]; \
                    acc[v] += a_lo * w_lo + a_hi * w_hi; \
                } \
            } \
            if (kh_base + ((GS_) / 2) > K_half) break; \
        } \
    } while(0)
    if constexpr (GS_R == GS_S && E8M0_R == E8M0_S) {
        SILUDOWN_ACCUM(GS_R, E8M0_R);
    } else {
        if (is_shared) { SILUDOWN_ACCUM(GS_S, E8M0_S); }
        else           { SILUDOWN_ACCUM(GS_R, E8M0_R); }
    }
    #undef SILUDOWN_ACCUM

    // Output offset: routed → C[expert_slot * N + n]; shared → sh_down_out[n].
    emit(acc);
}

// NVFP4 (default): FP8-E4M3 per-16 scales × per-tensor global.
extern "C" __global__ void moe_expert_silu_down_shared_t(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    silu_down_shared_t_impl<GROUP_SIZE, false, GROUP_SIZE, false>(
        gate_out, up_out, packed_t_ptrs, scale_t_ptrs, scale2_vals, C,
        expert_indices, sh_gate_in, sh_up_in, sh_down_t_packed, sh_down_t_scale,
        sh_down_s2, sh_down_out, N, K, top_k);
}

// Native MXFP4 (ARM-2): ROUTED experts E8M0 per-32; SHARED expert stays NVFP4.
extern "C" __global__ void moe_expert_silu_down_shared_t_e8m0(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    silu_down_shared_t_impl<32, true, GROUP_SIZE, false>(
        gate_out, up_out, packed_t_ptrs, scale_t_ptrs, scale2_vals, C,
        expert_indices, sh_gate_in, sh_up_in, sh_down_t_packed, sh_down_t_scale,
        sh_down_s2, sh_down_out, N, K, top_k);
}

// ── VEC=4 variants (see the gate_up pair above) ──
extern "C" __global__ void moe_expert_silu_down_shared_t_v4(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    silu_down_shared_t_impl<GROUP_SIZE, false, GROUP_SIZE, false, 4>(
        gate_out, up_out, packed_t_ptrs, scale_t_ptrs, scale2_vals, C,
        expert_indices, sh_gate_in, sh_up_in, sh_down_t_packed, sh_down_t_scale,
        sh_down_s2, sh_down_out, N, K, top_k);
}

extern "C" __global__ void moe_expert_silu_down_shared_t_e8m0_v4(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    silu_down_shared_t_impl<32, true, GROUP_SIZE, false, 4>(
        gate_out, up_out, packed_t_ptrs, scale_t_ptrs, scale2_vals, C,
        expert_indices, sh_gate_in, sh_up_in, sh_down_t_packed, sh_down_t_scale,
        sh_down_s2, sh_down_out, N, K, top_k);
}

// ── VEC=2 variants (see the gate_up pair above) ──
extern "C" __global__ void moe_expert_silu_down_shared_t_v2(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    silu_down_shared_t_impl<GROUP_SIZE, false, GROUP_SIZE, false, 2>(
        gate_out, up_out, packed_t_ptrs, scale_t_ptrs, scale2_vals, C,
        expert_indices, sh_gate_in, sh_up_in, sh_down_t_packed, sh_down_t_scale,
        sh_down_s2, sh_down_out, N, K, top_k);
}

extern "C" __global__ void moe_expert_silu_down_shared_t_e8m0_v2(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    silu_down_shared_t_impl<32, true, GROUP_SIZE, false, 2>(
        gate_out, up_out, packed_t_ptrs, scale_t_ptrs, scale2_vals, C,
        expert_indices, sh_gate_in, sh_up_in, sh_down_t_packed, sh_down_t_scale,
        sh_down_s2, sh_down_out, N, K, top_k);
}

// ── split-K variants ──
//
// One entry point per (format, VEC, SPLIT). The parameter lists are the
// SPLIT==1 ones plus a trailing f32 `partial` scratch pointer, so a macro is
// used rather than eight more verbatim copies. Launch contract:
//   gate_up:  grid = [N/(block*VEC), top_k+1, 2*SPLIT]
//   down:     grid = [N/(block*VEC), top_k+1, SPLIT], smem = K*4/SPLIT
// then `moe_gate_up_partial_finalize` / `moe_down_partial_finalize`.
#define GATEUP_SPLIT_ENTRY(NAME, GS_R_, E8M0_R_, VEC_, SPLIT_)                 \
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
    unsigned int N, unsigned int K, unsigned int top_k,                        \
    float* __restrict__ partial                                                \
) {                                                                            \
    gate_up_shared_t_impl<(GS_R_), (E8M0_R_), GROUP_SIZE, false, (VEC_), (SPLIT_)>( \
        A, gate_packed_t_ptrs, gate_scale_t_ptrs, gate_scale2_vals, gate_out,  \
        up_packed_t_ptrs, up_scale_t_ptrs, up_scale2_vals, up_out,             \
        expert_indices, sh_gate_t_packed, sh_gate_t_scale, sh_gate_s2,         \
        sh_gate_out, sh_up_t_packed, sh_up_t_scale, sh_up_s2, sh_up_out,       \
        N, K, top_k, partial);                                                 \
}

#define DOWN_SPLIT_ENTRY(NAME, GS_R_, E8M0_R_, VEC_, SPLIT_)                   \
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
    unsigned int N, unsigned int K, unsigned int top_k,                        \
    float* __restrict__ partial                                                \
) {                                                                            \
    silu_down_shared_t_impl<(GS_R_), (E8M0_R_), GROUP_SIZE, false, (VEC_), (SPLIT_)>( \
        gate_out, up_out, packed_t_ptrs, scale_t_ptrs, scale2_vals, C,         \
        expert_indices, sh_gate_in, sh_up_in, sh_down_t_packed,                \
        sh_down_t_scale, sh_down_s2, sh_down_out, N, K, top_k, partial);       \
}

GATEUP_SPLIT_ENTRY(moe_expert_gate_up_shared_t_v2s2,       GROUP_SIZE, false, 2, 2)
GATEUP_SPLIT_ENTRY(moe_expert_gate_up_shared_t_e8m0_v2s2,  32,         true,  2, 2)
GATEUP_SPLIT_ENTRY(moe_expert_gate_up_shared_t_v2s4,       GROUP_SIZE, false, 2, 4)
GATEUP_SPLIT_ENTRY(moe_expert_gate_up_shared_t_e8m0_v2s4,  32,         true,  2, 4)
GATEUP_SPLIT_ENTRY(moe_expert_gate_up_shared_t_v4s2,       GROUP_SIZE, false, 4, 2)
GATEUP_SPLIT_ENTRY(moe_expert_gate_up_shared_t_e8m0_v4s2,  32,         true,  4, 2)
GATEUP_SPLIT_ENTRY(moe_expert_gate_up_shared_t_v4s4,       GROUP_SIZE, false, 4, 4)
GATEUP_SPLIT_ENTRY(moe_expert_gate_up_shared_t_e8m0_v4s4,  32,         true,  4, 4)

DOWN_SPLIT_ENTRY(moe_expert_silu_down_shared_t_v2s2,       GROUP_SIZE, false, 2, 2)
DOWN_SPLIT_ENTRY(moe_expert_silu_down_shared_t_e8m0_v2s2,  32,         true,  2, 2)
DOWN_SPLIT_ENTRY(moe_expert_silu_down_shared_t_v2s4,       GROUP_SIZE, false, 2, 4)
DOWN_SPLIT_ENTRY(moe_expert_silu_down_shared_t_e8m0_v2s4,  32,         true,  2, 4)
DOWN_SPLIT_ENTRY(moe_expert_silu_down_shared_t_v4s2,       GROUP_SIZE, false, 4, 2)
DOWN_SPLIT_ENTRY(moe_expert_silu_down_shared_t_e8m0_v4s2,  32,         true,  4, 2)
DOWN_SPLIT_ENTRY(moe_expert_silu_down_shared_t_v4s4,       GROUP_SIZE, false, 4, 4)
DOWN_SPLIT_ENTRY(moe_expert_silu_down_shared_t_e8m0_v4s4,  32,         true,  4, 4)

#undef GATEUP_SPLIT_ENTRY
#undef DOWN_SPLIT_ENTRY

// Sum the split-K partials and write the model's bf16 buffers. `split` is a
// runtime argument here (unlike SPLIT in the GEMV) because this kernel is
// bandwidth-trivial — ~0.3 MB against the ~94 MB/layer the GEMVs stream — so
// there is nothing to gain from unrolling it per split factor.
//
// The `ks` loop runs in ascending order for every launch, so the sum is
// bit-reproducible; that is the property the per-arm reference hashes need.
// Launch: grid = [ceil(N/block), top_k+1, 2], block = [block,1,1].
extern "C" __global__ void moe_gate_up_partial_finalize(
    const float* __restrict__ partial,          // [2, split, top_k+1, N]
    __nv_bfloat16* __restrict__ gate_out,       // [top_k, N]
    __nv_bfloat16* __restrict__ sh_gate_out,    // [N]
    __nv_bfloat16* __restrict__ up_out,         // [top_k, N]
    __nv_bfloat16* __restrict__ sh_up_out,      // [N]
    unsigned int N, unsigned int top_k, unsigned int split
) {
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const unsigned int slot = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const unsigned int slots = top_k + 1;

    float s = 0.0f;
    for (unsigned int ks = 0; ks < split; ++ks) {
        s += partial[(((unsigned long long)proj * split + ks) * slots + slot) * N + n];
    }
    __nv_bfloat16* dst = (slot == top_k)
        ? ((proj == 0) ? sh_gate_out : sh_up_out)
        : (((proj == 0) ? gate_out : up_out) + (unsigned long long)slot * N);
    if (dst == 0) return;   // absent shared expert; the GEMV wrote zeros nowhere
    dst[n] = __float2bfloat16(s);
}

// Launch: grid = [ceil(N/block), top_k+1, 1], block = [block,1,1].
extern "C" __global__ void moe_down_partial_finalize(
    const float* __restrict__ partial,          // [split, top_k+1, N]
    __nv_bfloat16* __restrict__ C,              // [top_k, N]
    __nv_bfloat16* __restrict__ sh_down_out,    // [N]
    unsigned int N, unsigned int top_k, unsigned int split
) {
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const unsigned int slot = blockIdx.y;
    const unsigned int slots = top_k + 1;

    float s = 0.0f;
    for (unsigned int ks = 0; ks < split; ++ks) {
        s += partial[((unsigned long long)ks * slots + slot) * N + n];
    }
    __nv_bfloat16* dst = (slot == top_k) ? sh_down_out : (C + (unsigned long long)slot * N);
    if (dst == 0) return;
    dst[n] = __float2bfloat16(s);
}
