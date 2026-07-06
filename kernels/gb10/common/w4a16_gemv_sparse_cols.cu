// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A16 column-sparse GEMV — PROTOTYPE for sparsity-drafted
// self-speculation (TEAL-style activation sparsity). COMPILE-ONLY,
// default-off, NOT wired end-to-end. The DRAFT need not be bit-exact
// (the dense verify is the oracle), so this kernel has latitude to skip
// weight columns whose activation is below threshold.
//
// IDEA:  y[n] = sum_j A[j] * dequant(B[n, j]).  For a decode row A[K],
//        if |A[j]| < tau we skip reading B[:, j] entirely — fewer weight
//        BYTES moved, which is the ONLY lever that beats the bandwidth wall.
//
// GRANULARITY:  The fused GEMV packs 8 E2M1 nibbles per uint (4 packed
//        bytes) and one FP8 scale per GROUP_SIZE=16. We skip at the k8 chunk
//        (8 activations) granularity so the packed 4-byte weight read for
//        that chunk is elided. The host precomputes `keep_idx` — the sorted
//        list of surviving k8 chunk indices — and `keep_len` from a
//        thresholding pass over A (the measurement kernel's predicate). A
//        chunk survives iff ANY of its 8 activations is >= tau.
//
// WEIGHT-BYTE SAVINGS:  B_packed is row-major [N, K/2]; each output row n
//        streams its own K/2 bytes. Skipping keep-complement chunks skips a
//        `keep_len/K8` fraction of EVERY row's weight bytes — i.e. the packed
//        read AND the per-group scale read both drop proportionally. The
//        skipped reads are strided (4 bytes per surviving chunk) not
//        contiguous, so effective bandwidth depends on L2/coalescing, but the
//        DRAM-traffic floor drops by exactly the skipped fraction.
//
// CORRECTNESS NOTE:  This is an APPROXIMATION of the dense GEMV (it drops the
//        small-activation contributions). That is intentional and safe: the
//        output feeds a DRAFT token proposal that the dense verify re-checks.
//
// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)   [mirrors w4a16_gemv]

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define SP_BLOCK 256
#define SP_NPB 4
#define SP_WARP 32
#define SP_GROUP 16

__device__ __constant__ float E2M1_LUT_SPARSE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// A:         [1, K] BF16 activation.
// B_packed:  [N, K/2] NVFP4 nibbles.
// B_scale:   [N, K/GROUP] FP8-E4M3 group scales.
// keep_idx:  [keep_len] u32 — surviving k8 chunk indices (0..K/8), sorted.
// keep_len:  number of surviving chunks.
// C:         [1, N] BF16 output.
extern "C" __global__ void w4a16_gemv_sparse_cols(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    const unsigned int* __restrict__ keep_idx,
    unsigned int keep_len,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = SP_BLOCK / SP_NPB;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * SP_NPB + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / SP_GROUP;

    __shared__ float s_lut[16];
    __shared__ float smem[SP_NPB * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_SPARSE[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    // Iterate ONLY over surviving k8 chunks. Each surviving chunk still does
    // the same packed-4-byte read + FMA as the dense kernel — we've simply
    // dropped the loop iterations (and their weight reads) for skipped chunks.
    for (unsigned int idx = lane; idx < keep_len; idx += threads_per_out) {
        const unsigned int k8 = keep_idx[idx];
        const unsigned int base_k = k8 * 8;

        uint4 a_data = ((const uint4*)A)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};

        unsigned int packed4 = *(const unsigned int*)(
            B_packed + (unsigned long long)n * half_K + k8 * 4);

        unsigned int scale_group = base_k / SP_GROUP;
        unsigned char scale_byte = B_scale[
            (unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo) * w_lo;
            acc += __bfloat162float(a_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % SP_WARP;
    #pragma unroll
    for (int offset = SP_WARP / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    if (warp_lane == 0) {
        smem[local_out * 2 + (lane / SP_WARP)] = acc;
    }
    __syncthreads();
    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

// Companion host-side-driven chunk selector run ON DEVICE: threshold A[K]
// (per-row max-abs * tau) and emit the surviving k8-chunk index list +
// count. One CTA. Writes keep_idx[] (dense prefix) and keep_len[0].
//
// A chunk (8 contiguous activations) survives iff its max-abs >= tau*rowmax.
// This lets the sparse GEMV above avoid a host round-trip: propose runs
//   ffn_build_keep_chunks(A, tau, keep_idx, keep_len, K)
//   w4a16_gemv_sparse_cols(A, ..., keep_idx, keep_len[0], ...)
// back-to-back on the same stream.
extern "C" __global__ void ffn_build_keep_chunks(
    const __nv_bfloat16* __restrict__ A,
    float tau,                         // fraction of rowmax (e.g. 0.01)
    unsigned int* __restrict__ keep_idx,   // [K/8] capacity
    unsigned int* __restrict__ keep_len,   // [1]
    unsigned int K
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int K8 = K / 8;

    __shared__ float s_max;
    __shared__ float s_warpmax[SP_BLOCK / 32];
    __shared__ unsigned int s_cursor;

    if (tid == 0) { keep_len[0] = 0u; s_cursor = 0u; }

    // Pass 1: rowmax.
    float local_max = 0.0f;
    for (unsigned int j = tid; j < K; j += SP_BLOCK)
        local_max = fmaxf(local_max, fabsf(__bfloat162float(A[j])));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, off));
    const unsigned int wid = tid / 32, wlane = tid % 32;
    if (wlane == 0) s_warpmax[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        float v = (wlane < (SP_BLOCK / 32)) ? s_warpmax[wlane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, off));
        if (wlane == 0) s_max = v;
    }
    __syncthreads();

    const float cut = tau * s_max;

    // Pass 2: each thread tests strided k8 chunks; survivors are appended to
    // keep_idx via an atomic cursor. Order is not guaranteed sorted — the
    // sparse GEMV does not require sorted indices (random-access per chunk),
    // so we skip a sort. (A sorted variant would improve coalescing.)
    for (unsigned int k8 = tid; k8 < K8; k8 += SP_BLOCK) {
        uint4 a_data = ((const uint4*)A)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        float cmax = 0.0f;
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            cmax = fmaxf(cmax, fabsf(__bfloat162float(a_lo)));
            cmax = fmaxf(cmax, fabsf(__bfloat162float(a_hi)));
        }
        if (cmax >= cut) {
            unsigned int slot = atomicAdd(&s_cursor, 1u);
            keep_idx[slot] = k8;
        }
    }
    __syncthreads();
    if (tid == 0) keep_len[0] = s_cursor;
}
