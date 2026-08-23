// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A16 GEMV — Fused NVFP4 weight dequant + BF16 GEMV for M=1 decode.
//
// out[n] = dot(A[0,:], dequant(B_fp4[n,:]))
//
// Specialized for M=1 decode: replaces w4a16_gemm which wastes ~98% of
// threads at M=1 with 64x64 tiles + MMA tensor cores (MMA requires M>=16).
//
// Vectorized: reads 4 packed weight bytes (uint32_t = 8 FP4 values) and
// 8 BF16 activations (uint4 = 16 bytes) per iteration for better bandwidth.
//
// NVFP4 weight format (HuggingFace/compressed-tensors):
//   B_packed: [N, K/2] uint8 — byte at [n, j] holds W[n, 2j] (low) and W[n, 2j+1] (high)
//   B_scale:  [N, K/GROUP_SIZE] FP8-E4M3 — one scale per group of 16 K-dim values
//   scale2:   scalar FP32 — per-tensor second-level scale
//
// K-dim packing: each byte holds 2 consecutive input features for the same output.
// Vectorized reads of 4 bytes = 8 weight values, coalesced across warps.
//
// 4 outputs per block, 64 threads (2 warps) per output. Cross-warp smem reduction.
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

// E2M1 lookup table (same as w4a16_gemm.cu)
__device__ __constant__ float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// PR #519: a data-dependent index into __constant__ memory serializes a
// warp once per distinct nibble. Stage one bit-exact LUT copy per CUDA warp;
// the SCALE/HIP symlinked builds keep their established constant-memory path
// because they do not provide CUDA's __syncwarp intrinsic.
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
#define ATLAS_WARP_LUT_STAGED 0
#else
#define ATLAS_WARP_LUT_STAGED 1
#endif

__device__ __forceinline__ void stage_e2m1_lut_warp(
    float* s_lut, unsigned int lane)
{
#if ATLAS_WARP_LUT_STAGED
    if (lane < 16u) s_lut[lane] = E2M1_LUT[lane];
    __syncwarp();
#else
    (void)s_lut;
    (void)lane;
#endif
}

// One base-kernel lane's K16 partial.
//
// `orig_lane` is the 64-thread kernel's lane (0..63): the loop starts at
// k16 = orig_lane and strides by threads_per_out = BLOCK_SIZE / N_PER_BLOCK = 64.
//
// SSOT: `w4a16_gemv` and `w4a16_gemv_sw` share this body so the single-warp
// path cannot drift off the base association. Upstream learned this the hard
// way — a hand-copied SW body that used a different K-chunk association was
// 1 ULP lossy against its base kernel
// (`upstream-latest/kernels/gb10/common/w4a16_gemv.cu:64-68`).
//
// `lut` is either the base kernel's shared-memory E2M1 table or `E2M1_LUT`
// itself; the 16 values are identical, so the arithmetic is bit-identical
// whichever address space it comes from. Passing it keeps the base kernel's
// existing shared-memory read untouched.
//
// NOTE: this is the fork's *sequential* `acc += a*w` association, NOT upstream's
// 2-chunk pipelined `w4a16_gemv_partial`. Bit-parity here is against OUR base
// kernel, which is what keeps the committed token stream unchanged.
__device__ __forceinline__ float w4a16_gemv_partial(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    const float* lut,
    unsigned int n, unsigned int half_K, unsigned int num_groups,
    unsigned int K16, unsigned int orig_lane)
{
    float acc = 0.0f;

    // Vectorized: process 16 K-values per iteration (2× uint4 activation + uint64 weight)
    // One scale per GROUP_SIZE=16, so each iteration uses exactly 1 scale lookup.
    for (unsigned int k16 = orig_lane; k16 < K16; k16 += 64u) {
        const unsigned int base_k = k16 * 16;

        // Load 16 BF16 activations as 2× uint4 (256-bit total)
        uint4 a_lo = ((const uint4*)A)[k16 * 2];
        uint4 a_hi = ((const uint4*)A)[k16 * 2 + 1];
        const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};

        // Load 8 packed weight bytes as uint64 (16 FP4 values)
        unsigned long long packed8 = *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + k16 * 8);

        // Load single FP8 scale — 16 values = exactly 1 group
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        // Unpack 8 bytes × 2 nibbles = 16 weight values, FMA with activations
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
            float w_lo = lut[byte_val & 0xF] * scale;
            float w_hi = lut[byte_val >> 4] * scale;

            __nv_bfloat16 a_lo_bf, a_hi_bf;
            *(unsigned short*)&a_lo_bf = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi_bf = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo_bf) * w_lo;
            acc += __bfloat162float(a_hi_bf) * w_hi;
        }
    }
    return acc;
}

// W4A16 GEMV: C[n] = sum_k A[k] * dequant(B_fp4[n, k])
//
// Vectorized: processes 8 K-values per iteration.
// - 4 packed weight bytes (uint32_t) → 8 FP4 values via E2M1 LUT
// - 8 BF16 activations (uint4 = 128-bit load)
// - 1 FP8 scale (all 8 values in same group since GROUP_SIZE=16, stride=8)
//
// Coalescing: within a warp, consecutive threads read consecutive 4-byte
// weight chunks and consecutive 16-byte activation chunks. Perfectly coalesced.
extern "C" __global__ void w4a16_gemv(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [1, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = n < N;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];  // cross-warp reduction
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;
    if (valid) {
        acc = w4a16_gemv_partial(A, B_packed, B_scale, scale2, s_lut,
                                 n, half_K, num_groups, K16, lane);
    }

    // Warp shuffle reduction within each group of 64 threads
    // First reduce within each warp (32 threads)
    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // threads_per_out=64 means 2 warps per output. Use shared memory for cross-warp reduce.
    if (warp_lane == 0) {
        // Each warp writes its partial sum
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    // First thread of each output group writes final result
    if (valid && lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

// ============================================================
// W4A16 exact multi-row GEMV for large-N LM-head verification.
// ============================================================
//
// This family reads each NVFP4 weight row once while reproducing M independent
// `w4a16_gemv` launches byte-for-byte. It deliberately keeps the base kernel's
// K16 lane assignment, scaled-weight arithmetic, two ordered accumulator
// updates per packed byte, two-warp shuffle trees, and ordered cross-warp add.
// It must not be replaced with the numerically different K8 batch kernels or
// the E4M3 tensor-core small-M GEMM.
//
// Four register-sized entry points avoid paying the MAX_M=32 register cost for
// common smaller verify widths. Production selects the smallest fitting tier:
//   M=2..4   -> w4a16_gemv_batch_logits_exact_m4
//   M=5..8   -> w4a16_gemv_batch_logits_exact_m8
//   M=9..17  -> w4a16_gemv_batch_logits_exact_m17
//   M=18..32 -> w4a16_gemv_batch_logits_exact_m32
//
// A:         [M, K] BF16 contiguous
// B_packed:  [N, K/2] NVFP4 packed, row-major
// B_scale:   [N, K/16] FP8-E4M3
// C:         [M, N] BF16 contiguous
// Grid:      (ceil(N / 4), 1, 1)
// Block:     (256, 1, 1), 64 threads / output
//
// Tail output groups never return before either block barrier. Invalid groups
// take part in both barriers and both reductions with zero accumulators, while
// all row-address formation and global loads/stores remain predicated.

template <int MAX_M>
__device__ __forceinline__ void w4a16_gemv_batch_logits_exact_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    float* s_lut,
    float* smem)
{
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int warp_lane = lane % WARP_SIZE;
    const unsigned int warp_idx = lane / WARP_SIZE;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool rows_valid = M > 0 && M <= (unsigned int)MAX_M;
    const bool valid = rows_valid && n < N;

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc[MAX_M];
    #pragma unroll
    for (int row = 0; row < MAX_M; ++row) acc[row] = 0.0f;

    if (valid) {
        const unsigned int half_K = K / 2;
        const unsigned int num_groups = K / GROUP_SIZE;
        const unsigned int K16 = K / 16;

        // Keep the ordinary K=1 lane ownership exactly: k16=lane, stride 64.
        for (unsigned int k16 = lane; k16 < K16; k16 += 64u) {
            const unsigned int base_k = k16 * 16;
            const unsigned long long weight_row =
                (unsigned long long)n * half_K;
            const unsigned long long scale_row =
                (unsigned long long)n * num_groups;
            const unsigned long long packed8 =
                *(const unsigned long long*)(B_packed + weight_row + k16 * 8);

            const unsigned int scale_group = base_k / GROUP_SIZE;
            const unsigned char scale_byte = B_scale[scale_row + scale_group];
            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = scale_byte;
            const float scale = (float)fp8 * scale2;

            // Unpack and scale once, then reuse the exact FP32 operands for
            // every active activation row.
            float w_lo[8];
            float w_hi[8];
            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                const unsigned char byte_val =
                    (unsigned char)(packed8 >> (b * 8));
                w_lo[b] = s_lut[byte_val & 0xF] * scale;
                w_hi[b] = s_lut[byte_val >> 4] * scale;
            }

            #pragma unroll
            for (int row = 0; row < MAX_M; ++row) {
                if ((unsigned int)row < M) {
                    const __nv_bfloat16* __restrict__ A_row =
                        A + (unsigned long long)row * K;
                    const uint4 a_lo = ((const uint4*)A_row)[k16 * 2];
                    const uint4 a_hi = ((const uint4*)A_row)[k16 * 2 + 1];
                    const unsigned int a_raw[8] = {
                        a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                        a_hi.x, a_hi.y, a_hi.z, a_hi.w
                    };

                    #pragma unroll
                    for (int b = 0; b < 8; ++b) {
                        __nv_bfloat16 a_lo_bf, a_hi_bf;
                        *(unsigned short*)&a_lo_bf =
                            (unsigned short)(a_raw[b] & 0xFFFF);
                        *(unsigned short*)&a_hi_bf =
                            (unsigned short)(a_raw[b] >> 16);
                        acc[row] += __bfloat162float(a_lo_bf) * w_lo[b];
                        acc[row] += __bfloat162float(a_hi_bf) * w_hi[b];
                    }
                }
            }
        }
    }

    // Reproduce the ordinary K=1 five-step tree independently for every row.
    #pragma unroll
    for (int row = 0; row < MAX_M; ++row) {
        if ((unsigned int)row < M) {
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                acc[row] += __shfl_down_sync(0xFFFFFFFF, acc[row], offset);
            }
            if (warp_lane == 0) {
                smem[(row * N_PER_BLOCK + local_out) * 2 + warp_idx] =
                    acc[row];
            }
        }
    }
    __syncthreads();

    if (valid && lane == 0) {
        #pragma unroll
        for (int row = 0; row < MAX_M; ++row) {
            if ((unsigned int)row < M) {
                const unsigned int base =
                    (row * N_PER_BLOCK + local_out) * 2;
                C[(unsigned long long)row * N + n] =
                    __float2bfloat16(smem[base] + smem[base + 1]);
            }
        }
    }
}

#define DEFINE_W4A16_GEMV_BATCH_LOGITS_EXACT(NAME, MAX_ROWS)                 \
extern "C" __global__ void NAME(                                             \
    const __nv_bfloat16* __restrict__ A,                                     \
    const unsigned char* __restrict__ B_packed,                              \
    const unsigned char* __restrict__ B_scale,                               \
    const float scale2,                                                       \
    __nv_bfloat16* __restrict__ C,                                            \
    unsigned int M, unsigned int N, unsigned int K)                           \
{                                                                            \
    __shared__ float s_lut[16];                                               \
    __shared__ float smem[(MAX_ROWS) * N_PER_BLOCK * 2];                      \
    w4a16_gemv_batch_logits_exact_body<MAX_ROWS>(                             \
        A, B_packed, B_scale, scale2, C, M, N, K, s_lut, smem);              \
}

DEFINE_W4A16_GEMV_BATCH_LOGITS_EXACT(
    w4a16_gemv_batch_logits_exact_m4, 4)
DEFINE_W4A16_GEMV_BATCH_LOGITS_EXACT(
    w4a16_gemv_batch_logits_exact_m8, 8)
DEFINE_W4A16_GEMV_BATCH_LOGITS_EXACT(
    w4a16_gemv_batch_logits_exact_m17, 17)
DEFINE_W4A16_GEMV_BATCH_LOGITS_EXACT(
    w4a16_gemv_batch_logits_exact_m32, 32)

#undef DEFINE_W4A16_GEMV_BATCH_LOGITS_EXACT

// ============================================================
// W4A16 GEMV — SINGLE-WARP-PER-OUTPUT variant (lossless; default ON,
// kill with ATLAS_NO_GEMV_SW=1).
//
// Ported from upstream `w4a16_gemv_sw`
// (upstream-latest/kernels/gb10/common/w4a16_gemv.cu:161-215), re-derived
// against THIS tree's base kernel rather than upstream's — see the note on
// `w4a16_gemv_partial` above.
//
// 32 threads (1 warp) per output instead of 64 (2 warps). 8 outputs per
// 256-thread block instead of 4. The cross-warp __syncthreads() + smem
// round-trip is gone: the final combine is one FP32 add of two warp-shuffle
// reductions.
//
// BIT-IDENTICALITY against `w4a16_gemv`: in the base kernel, the 64 threads
// that own output `n` are hardware warp A (base lanes 0..31) and warp B (base
// lanes 32..63), each shuffle-reduced independently into
// smem[local_out*2 + 0] and smem[local_out*2 + 1], then added.
// Here one warp holds both partials:
//   acc_a[lane] == base acc[lane]        (k16 = lane,      stride 64)
//   acc_b[lane] == base acc[lane + 32]   (k16 = lane + 32, stride 64)
// Reducing each in the same 5-step shuffle tree gives reduced_a == smem[0]
// and reduced_b == smem[1]; `reduced_a + reduced_b` is the base kernel's final
// add, operand-for-operand.
//
// The early `n >= N` return is safe here (unlike in the base kernel, which
// returns before a __syncthreads()) because this kernel has no barrier at all.
//
// Grid: (ceil(N / 8), 1, 1)   Block: (256, 1, 1)
// ============================================================

#define N_PER_BLOCK_SW 8

extern "C" __global__ void w4a16_gemv_sw(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [1, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int local_out = threadIdx.x / WARP_SIZE;  // 0..7
    const unsigned int lane = threadIdx.x % WARP_SIZE;       // 0..31
    const unsigned int n = blockIdx.x * N_PER_BLOCK_SW + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    // The early return above is warp-uniform but not block-uniform, so a
    // block barrier would be invalid. Each warp publishes its own 64-byte row.
    __shared__ float s_lut[N_PER_BLOCK_SW][16];
    stage_e2m1_lut_warp(s_lut[local_out], lane);
#if ATLAS_WARP_LUT_STAGED
    const float* __restrict__ warp_lut = s_lut[local_out];
#else
    const float* __restrict__ warp_lut = E2M1_LUT;
#endif

    // acc_a reproduces base lane `lane` (warp A); acc_b reproduces base lane
    // `lane + 32` (warp B). Same operands, same order as the 64-thread kernel.
    float acc_a = w4a16_gemv_partial(A, B_packed, B_scale, scale2, warp_lut,
                                     n, half_K, num_groups, K16, lane);
    float acc_b = w4a16_gemv_partial(A, B_packed, B_scale, scale2, warp_lut,
                                     n, half_K, num_groups, K16, lane + 32u);

    // Reduce each accumulator within the warp in the SAME tree order as base.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xFFFFFFFF, acc_a, offset);
        acc_b += __shfl_down_sync(0xFFFFFFFF, acc_b, offset);
    }

    // lane 0 holds reduced warp-A and warp-B. This add == smem[0] + smem[1].
    if (lane == 0) {
        C[n] = __float2bfloat16(acc_a + acc_b);
    }
}

// ============================================================
// W4A16 GEMV with FP32 output (for LM head logits).
// Identical to w4a16_gemv but writes float instead of BF16.
// FP32 logits are critical for sampling quality — BF16 collapses
// similar logit values, making stochastic sampling random.
// ============================================================
extern "C" __global__ void w4a16_gemv_logits(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    float* __restrict__ C,  // FP32 output
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = n < N;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;
    if (valid) {
        for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
            const unsigned int base_k = k16 * 16;
            uint4 a_lo = ((const uint4*)A)[k16 * 2];
            uint4 a_hi = ((const uint4*)A)[k16 * 2 + 1];
            const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                            a_hi.x, a_hi.y, a_hi.z, a_hi.w};
            unsigned long long packed8 = *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + k16 * 8);
            unsigned int scale_group = base_k / GROUP_SIZE;
            unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = scale_byte;
            float scale = (float)fp8 * scale2;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
                float w_lo = s_lut[byte_val & 0xF] * scale;
                float w_hi = s_lut[byte_val >> 4] * scale;
                __nv_bfloat16 a_lo_bf, a_hi_bf;
                *(unsigned short*)&a_lo_bf = (unsigned short)(a_raw[b] & 0xFFFF);
                *(unsigned short*)&a_hi_bf = (unsigned short)(a_raw[b] >> 16);
                acc += __bfloat162float(a_lo_bf) * w_lo;
                acc += __bfloat162float(a_hi_bf) * w_hi;
            }
        }
    }
    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();
    if (valid && lane == 0) {
        C[n] = smem[local_out * 2] + smem[local_out * 2 + 1]; // FP32 output!
    }
}

// ============================================================
// W4A16 double-GEMV (M=2): reads weights once, computes 2 outputs
// ============================================================
// For K=2 speculative verification: processes 2 input vectors through
// the same weight matrix in a single pass. Eliminates the GEMM M=2
// tile waste (64x64 tiles at 3% M-utilization).
//
// A: [2, K] BF16 contiguous (row 0 and row 1)
// B: [N, K/2] NVFP4 packed weights
// C: [2, N] BF16 contiguous (row 0 and row 1)
//
// Same memory bandwidth as M=1 GEMV (weights dominate, read once).
// Extra cost: 2x activation reads (K*2 bytes per vector, fits in L1/L2).
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_batch2(
    const __nv_bfloat16* __restrict__ A,        // [2, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [2, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = n < N;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    // Pointers to second input/output rows
    const __nv_bfloat16* __restrict__ A1 = A + K;  // second input vector
    __nv_bfloat16* __restrict__ C1 = C + N;         // second output vector

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 4];  // 2 warps × 2 accumulators per output
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f;  // accumulator for first input
    float acc1 = 0.0f;  // accumulator for second input

    if (valid) {
    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        // Load 8 BF16 activations from BOTH input vectors
        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};

        // Load 4 packed weight bytes (SHARED between both inputs)
        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);

        // Load single FP8 scale
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        // Unpack weights and FMA with BOTH activation vectors
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            // First input vector
            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo;
            acc0 += __bfloat162float(a0_hi) * w_hi;

            // Second input vector (same weights, different activations)
            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo;
            acc1 += __bfloat162float(a1_hi) * w_hi;
        }
    }
    }

    // Warp shuffle reduction for both accumulators
    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }

    // Cross-warp reduction via shared memory (2 warps per output × 2 accumulators)
    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 4 + warp_idx * 2]     = acc0;
        smem[local_out * 4 + warp_idx * 2 + 1] = acc1;
    }
    __syncthreads();

    // First thread of each output group writes both results
    if (valid && lane == 0) {
        float result0 = smem[local_out * 4]     + smem[local_out * 4 + 2];
        float result1 = smem[local_out * 4 + 1] + smem[local_out * 4 + 3];
        C[n]  = __float2bfloat16(result0);
        C1[n] = __float2bfloat16(result1);
    }
}

// ============================================================
// W4A16 GEMV with inline Q/Gate deinterleave on output write
// ============================================================
// Same GEMV as w4a16_gemv but writes Q and Gate to separate halves.
// Eliminates the separate deinterleave_qg kernel (saves 12 graph nodes).
//
// Input layout (interleaved per head): [Q_h0(hd), G_h0(hd), Q_h1(hd), G_h1(hd), ...]
// Output layout (deinterleaved): [Q_h0..Q_nh | G_h0..G_nh]
//
// N = num_heads * head_dim * 2  (total Q+Gate elements)
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_qg(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [Q | G] deinterleaved
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = n < N;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    if (valid) {
    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;
        uint4 a_data = ((const uint4*)A)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
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
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        smem[local_out * 2 + (lane / WARP_SIZE)] = acc;
    }
    __syncthreads();

    if (valid && lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];

        // Deinterleave: n indexes interleaved [Q_h0(hd), G_h0(hd), Q_h1(hd), ...]
        // head = n / (2 * head_dim), is_gate = (n % (2 * head_dim)) >= head_dim
        unsigned int group_dim = 2 * head_dim;
        unsigned int h = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_heads * head_dim;

        unsigned int out_idx;
        if (idx < head_dim) {
            out_idx = h * head_dim + idx;             // Q region
        } else {
            out_idx = q_total + h * head_dim + (idx - head_dim);  // Gate region
        }
        C[out_idx] = __float2bfloat16(result);
    }
}

// ============================================================
// W4A16 GEMV with inline QKVZ deinterleave on output write
// ============================================================
// Same GEMV as w4a16_gemv but writes to deinterleaved output locations.
// Eliminates the separate deinterleave_qkvz kernel (saves 36 graph nodes).
//
// QKVZ interleaved layout (N=12288, 16 groups of 768):
//   Group g: [Q_{g*128..128} | K_{g*128..128} | V_{g*256..256} | Z_{g*256..256}]
//
// Deinterleaved output: [Q_2048 | K_2048 | V_4096 | Z_4096]
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_qkvz(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [Q|K|V|Z] deinterleaved
    unsigned int N,
    unsigned int K,
    // Deinterleave params:
    unsigned int num_groups,        // 16
    unsigned int head_k_dim,        // 128
    unsigned int vheads_per_group,  // 2
    unsigned int head_v_dim         // 128
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = n < N;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups_k = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    if (valid) {
    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;
        uint4 a_data = ((const uint4*)A)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups_k + scale_group];
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
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        smem[local_out * 2 + (lane / WARP_SIZE)] = acc;
    }
    __syncthreads();

    if (valid && lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];

        // Compute deinterleaved output index
        unsigned int v_group_size = vheads_per_group * head_v_dim;
        unsigned int group_dim = 2 * head_k_dim + 2 * v_group_size;
        unsigned int g = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_groups * head_k_dim;
        unsigned int k_total = num_groups * head_k_dim;

        unsigned int out_idx;
        if (idx < head_k_dim) {
            out_idx = g * head_k_dim + idx;
        } else if (idx < 2 * head_k_dim) {
            out_idx = q_total + g * head_k_dim + (idx - head_k_dim);
        } else if (idx < 2 * head_k_dim + v_group_size) {
            out_idx = q_total + k_total + g * v_group_size + (idx - 2 * head_k_dim);
        } else {
            out_idx = q_total + k_total + num_groups * v_group_size
                    + g * v_group_size + (idx - 2 * head_k_dim - v_group_size);
        }
        C[out_idx] = __float2bfloat16(result);
    }
}

// ============================================================
// W4A16 GEMV batch2 with inline Q/Gate deinterleave
// ============================================================
// Combines w4a16_gemv_batch2 (2-input) with w4a16_gemv_qg (deinterleave).
// Reads Q+Gate weight matrix once for 2 input tokens, produces 2 deinterleaved
// output vectors [Q_all | Gate_all] per token.
//
// Input:  A[2, K] BF16 (2 token hidden states)
// Output: C[2, N] BF16 (deinterleaved: C[0] = [Q0|G0], C[1] = [Q1|G1])
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_qg_batch2(
    const __nv_bfloat16* __restrict__ A,        // [2, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [2, N] deinterleaved [Q|G] per token
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = n < N;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    const __nv_bfloat16* __restrict__ A1 = A + K;
    __nv_bfloat16* __restrict__ C1 = C + N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 4];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    if (valid) {
    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo + __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo + __bfloat162float(a1_hi) * w_hi;
        }
    }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 4 + warp_idx * 2]     = acc0;
        smem[local_out * 4 + warp_idx * 2 + 1] = acc1;
    }
    __syncthreads();

    if (valid && lane == 0) {
        float result0 = smem[local_out * 4]     + smem[local_out * 4 + 2];
        float result1 = smem[local_out * 4 + 1] + smem[local_out * 4 + 3];

        // Deinterleave: n indexes interleaved [Q_h0(hd), G_h0(hd), Q_h1(hd), ...]
        unsigned int group_dim = 2 * head_dim;
        unsigned int h = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_heads * head_dim;

        unsigned int out_idx;
        if (idx < head_dim) {
            out_idx = h * head_dim + idx;
        } else {
            out_idx = q_total + h * head_dim + (idx - head_dim);
        }
        C[out_idx]  = __float2bfloat16(result0);
        C1[out_idx] = __float2bfloat16(result1);
    }
}

// ============================================================
// W4A16 GEMV dual batch2: K+V for 2 input tokens in one launch
// ============================================================
// Processes 2 separate weight matrices (K and V) with 2 input vectors each.
// blockIdx.z selects K (0) or V (1). Both projections compute 2 outputs.
//
// Input:  A[2, K_in] BF16 (2 token hidden states)
// Output: C[2, N] where blockIdx.z=0 writes K, blockIdx.z=1 writes V
//
// Grid: (ceil(N / 4), 1, 2)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_dual_batch2(
    const __nv_bfloat16* __restrict__ A,         // [2, K_in] BF16
    const unsigned char* __restrict__ B0_packed,  // [N, K_in/2] first projection
    const unsigned char* __restrict__ B0_scale,
    float B0_scale2,
    __nv_bfloat16* __restrict__ C0,              // [2, N] first projection output
    const unsigned char* __restrict__ B1_packed,  // [N, K_in/2] second projection
    const unsigned char* __restrict__ B1_scale,
    float B1_scale2,
    __nv_bfloat16* __restrict__ C1,              // [2, N] second projection output
    unsigned int N,
    unsigned int K_in
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = (proj == 0) ? B0_packed : B1_packed;
    const unsigned char* B_scale = (proj == 0) ? B0_scale : B1_scale;
    float s2 = (proj == 0) ? B0_scale2 : B1_scale2;
    __nv_bfloat16* C_out = (proj == 0) ? C0 : C1;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = n < N;

    const unsigned int half_K = K_in / 2;
    const unsigned int num_groups = K_in / GROUP_SIZE;
    const unsigned int K8 = K_in / 8;

    const __nv_bfloat16* A1 = A + K_in;
    __nv_bfloat16* C_out1 = C_out + N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 4];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f, acc1 = 0.0f;

    if (valid) {
    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb = B_scale[(unsigned long long)n * num_groups + sg];
        __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
        float scale = (float)fp8 * s2;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char bv = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[bv & 0xF] * scale;
            float w_hi = s_lut[bv >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo + __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo + __bfloat162float(a1_hi) * w_hi;
        }
    }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 4 + warp_idx * 2]     = acc0;
        smem[local_out * 4 + warp_idx * 2 + 1] = acc1;
    }
    __syncthreads();

    if (valid && lane == 0) {
        float result0 = smem[local_out * 4]     + smem[local_out * 4 + 2];
        float result1 = smem[local_out * 4 + 1] + smem[local_out * 4 + 3];
        C_out[n]  = __float2bfloat16(result0);
        C_out1[n] = __float2bfloat16(result1);
    }
}

// ============================================================
// W4A16 triple-GEMV (M=3): reads weights once, computes 3 outputs
// ============================================================
// For K=3 speculative verification: processes 3 input vectors through
// the same weight matrix in a single pass.
//
// A: [3, K] BF16 contiguous (row 0, 1, 2)
// B: [N, K/2] NVFP4 packed weights
// C: [3, N] BF16 contiguous (row 0, 1, 2)
//
// Same memory bandwidth as M=1 GEMV (weights dominate, read once).
// Extra cost: 3x activation reads (K*2 bytes per vector, fits in L1/L2).
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_batch3(
    const __nv_bfloat16* __restrict__ A,        // [3, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [3, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = n < N;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    const __nv_bfloat16* __restrict__ A1 = A + K;
    const __nv_bfloat16* __restrict__ A2 = A + 2 * K;
    __nv_bfloat16* __restrict__ C1 = C + N;
    __nv_bfloat16* __restrict__ C2 = C + 2 * N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 6];  // 2 warps × 3 accumulators per output
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;

    if (valid) {
    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        uint4 a2_data = ((const uint4*)A2)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int a2_raw[4] = {a2_data.x, a2_data.y, a2_data.z, a2_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);

        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo + __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo + __bfloat162float(a1_hi) * w_hi;

            __nv_bfloat16 a2_lo, a2_hi;
            *(unsigned short*)&a2_lo = (unsigned short)(a2_raw[b] & 0xFFFF);
            *(unsigned short*)&a2_hi = (unsigned short)(a2_raw[b] >> 16);
            acc2 += __bfloat162float(a2_lo) * w_lo + __bfloat162float(a2_hi) * w_hi;
        }
    }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 6 + warp_idx * 3]     = acc0;
        smem[local_out * 6 + warp_idx * 3 + 1] = acc1;
        smem[local_out * 6 + warp_idx * 3 + 2] = acc2;
    }
    __syncthreads();

    if (valid && lane == 0) {
        float result0 = smem[local_out * 6]     + smem[local_out * 6 + 3];
        float result1 = smem[local_out * 6 + 1] + smem[local_out * 6 + 4];
        float result2 = smem[local_out * 6 + 2] + smem[local_out * 6 + 5];
        C[n]  = __float2bfloat16(result0);
        C1[n] = __float2bfloat16(result1);
        C2[n] = __float2bfloat16(result2);
    }
}

// ============================================================
// W4A16 GEMV batch3 with inline Q/Gate deinterleave
// ============================================================
// Combines w4a16_gemv_batch3 (3-input) with Q/Gate deinterleave.
// Reads Q+Gate weight matrix once for 3 input tokens, produces 3 deinterleaved
// output vectors [Q_all | Gate_all] per token.
//
// Input:  A[3, K] BF16 (3 token hidden states)
// Output: C[3, N] BF16 (deinterleaved: C[i] = [Qi|Gi])
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_qg_batch3(
    const __nv_bfloat16* __restrict__ A,        // [3, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [3, N] deinterleaved [Q|G] per token
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = n < N;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    const __nv_bfloat16* __restrict__ A1 = A + K;
    const __nv_bfloat16* __restrict__ A2 = A + 2 * K;
    __nv_bfloat16* __restrict__ C1 = C + N;
    __nv_bfloat16* __restrict__ C2 = C + 2 * N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 6];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;

    if (valid) {
    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        uint4 a2_data = ((const uint4*)A2)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int a2_raw[4] = {a2_data.x, a2_data.y, a2_data.z, a2_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo + __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo + __bfloat162float(a1_hi) * w_hi;

            __nv_bfloat16 a2_lo, a2_hi;
            *(unsigned short*)&a2_lo = (unsigned short)(a2_raw[b] & 0xFFFF);
            *(unsigned short*)&a2_hi = (unsigned short)(a2_raw[b] >> 16);
            acc2 += __bfloat162float(a2_lo) * w_lo + __bfloat162float(a2_hi) * w_hi;
        }
    }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 6 + warp_idx * 3]     = acc0;
        smem[local_out * 6 + warp_idx * 3 + 1] = acc1;
        smem[local_out * 6 + warp_idx * 3 + 2] = acc2;
    }
    __syncthreads();

    if (valid && lane == 0) {
        float result0 = smem[local_out * 6]     + smem[local_out * 6 + 3];
        float result1 = smem[local_out * 6 + 1] + smem[local_out * 6 + 4];
        float result2 = smem[local_out * 6 + 2] + smem[local_out * 6 + 5];

        unsigned int group_dim = 2 * head_dim;
        unsigned int h = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_heads * head_dim;

        unsigned int out_idx;
        if (idx < head_dim) {
            out_idx = h * head_dim + idx;
        } else {
            out_idx = q_total + h * head_dim + (idx - head_dim);
        }
        C[out_idx]  = __float2bfloat16(result0);
        C1[out_idx] = __float2bfloat16(result1);
        C2[out_idx] = __float2bfloat16(result2);
    }
}

// ============================================================
// W4A16 GEMV dual batch3: K+V for 3 input tokens in one launch
// ============================================================
// Processes 2 separate weight matrices (K and V) with 3 input vectors each.
// blockIdx.z selects K (0) or V (1). Both projections compute 3 outputs.
//
// Input:  A[3, K_in] BF16 (3 token hidden states)
// Output: C[3, N] where blockIdx.z=0 writes K, blockIdx.z=1 writes V
//
// Grid: (ceil(N / 4), 1, 2)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_dual_batch3(
    const __nv_bfloat16* __restrict__ A,         // [3, K_in] BF16
    const unsigned char* __restrict__ B0_packed,  // [N, K_in/2] first projection
    const unsigned char* __restrict__ B0_scale,
    float B0_scale2,
    __nv_bfloat16* __restrict__ C0,              // [3, N] first projection output
    const unsigned char* __restrict__ B1_packed,  // [N, K_in/2] second projection
    const unsigned char* __restrict__ B1_scale,
    float B1_scale2,
    __nv_bfloat16* __restrict__ C1,              // [3, N] second projection output
    unsigned int N,
    unsigned int K_in
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = (proj == 0) ? B0_packed : B1_packed;
    const unsigned char* B_scale = (proj == 0) ? B0_scale : B1_scale;
    float s2 = (proj == 0) ? B0_scale2 : B1_scale2;
    __nv_bfloat16* C_out = (proj == 0) ? C0 : C1;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = n < N;

    const unsigned int half_K = K_in / 2;
    const unsigned int num_groups = K_in / GROUP_SIZE;
    const unsigned int K8 = K_in / 8;

    const __nv_bfloat16* A1 = A + K_in;
    const __nv_bfloat16* A2 = A + 2 * K_in;
    __nv_bfloat16* C_out1 = C_out + N;
    __nv_bfloat16* C_out2 = C_out + 2 * N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 6];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;

    if (valid) {
    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        uint4 a2_data = ((const uint4*)A2)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int a2_raw[4] = {a2_data.x, a2_data.y, a2_data.z, a2_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb = B_scale[(unsigned long long)n * num_groups + sg];
        __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
        float scale = (float)fp8 * s2;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char bv = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[bv & 0xF] * scale;
            float w_hi = s_lut[bv >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo + __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo + __bfloat162float(a1_hi) * w_hi;

            __nv_bfloat16 a2_lo, a2_hi;
            *(unsigned short*)&a2_lo = (unsigned short)(a2_raw[b] & 0xFFFF);
            *(unsigned short*)&a2_hi = (unsigned short)(a2_raw[b] >> 16);
            acc2 += __bfloat162float(a2_lo) * w_lo + __bfloat162float(a2_hi) * w_hi;
        }
    }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 6 + warp_idx * 3]     = acc0;
        smem[local_out * 6 + warp_idx * 3 + 1] = acc1;
        smem[local_out * 6 + warp_idx * 3 + 2] = acc2;
    }
    __syncthreads();

    if (valid && lane == 0) {
        float result0 = smem[local_out * 6]     + smem[local_out * 6 + 3];
        float result1 = smem[local_out * 6 + 1] + smem[local_out * 6 + 4];
        float result2 = smem[local_out * 6 + 2] + smem[local_out * 6 + 5];
        C_out[n]  = __float2bfloat16(result0);
        C_out1[n] = __float2bfloat16(result1);
        C_out2[n] = __float2bfloat16(result2);
    }
}

// ============================================================
// M=1 GEMV tuning sweep — 2026-05-19
// ============================================================
// Baseline w4a16_gemv (N_PER_BLOCK=4, threads_per_out=64) profiled at
// 59-60% of GB10's 273 GB/s LPDDR5X bandwidth on the M=1 small-projection
// path (Qwen3.6-27B SSM qkvz_proj, attn Q/K/V/O), while the fused FFN
// kernels (w4a16_gemv_fused.cu) achieve 85%. The single-output kernel is
// the bandwidth-bound bleeder on the K=3 verify path.
//
// Three variants below try different (N_PER_BLOCK, threads_per_out)
// combinations:
//   v1 (N=2, t=128) — 2× more blocks, more threads per output, better K-dim coalescing
//   v2 (N=1, t=256) — 4× more blocks, single output per CTA
//   v3 (N=8, t=32)  — half the blocks but 1 warp/output (no cross-warp reduce)
//
// All variants reuse the same vectorized inner loop: 16 BF16 acts as 2×uint4,
// 8 packed weight bytes as uint64, 1 FP8 scale (since GROUP_SIZE=16).
//
// Generalized cross-warp reduction handles any warps_per_out in {1,2,4,8}.
//
// Each Rust launcher MUST set grid = ceil(N / N_PER_BLOCK_VARIANT), block = 256.

// Shared inner-loop FMA body (16 K-values per iteration, 1 scale, 1 packed8)
#define W4A16_INNER_FMA(ACC, LANE, THREADS_PER_OUT)                                              \
    for (unsigned int k16 = (LANE); k16 < K16; k16 += (THREADS_PER_OUT)) {                       \
        const unsigned int base_k = k16 * 16;                                                    \
        uint4 a_lo = ((const uint4*)A)[k16 * 2];                                                 \
        uint4 a_hi = ((const uint4*)A)[k16 * 2 + 1];                                             \
        const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,                           \
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};                         \
        unsigned long long packed8 = *(const unsigned long long*)(                               \
            B_packed + (unsigned long long)n * half_K + k16 * 8);                                \
        unsigned int scale_group = base_k / GROUP_SIZE;                                          \
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];    \
        __nv_fp8_e4m3 fp8;                                                                       \
        *(unsigned char*)&fp8 = scale_byte;                                                      \
        float scale = (float)fp8 * scale2;                                                       \
        _Pragma("unroll")                                                                        \
        for (int b = 0; b < 8; b++) {                                                            \
            unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));                        \
            float w_lo = s_lut[byte_val & 0xF] * scale;                                          \
            float w_hi = s_lut[byte_val >> 4] * scale;                                           \
            __nv_bfloat16 a_lo_bf, a_hi_bf;                                                      \
            *(unsigned short*)&a_lo_bf = (unsigned short)(a_raw[b] & 0xFFFF);                    \
            *(unsigned short*)&a_hi_bf = (unsigned short)(a_raw[b] >> 16);                       \
            (ACC) += __bfloat162float(a_lo_bf) * w_lo;                                           \
            (ACC) += __bfloat162float(a_hi_bf) * w_hi;                                           \
        }                                                                                        \
    }

// ── Variant 1: (N_PER_BLOCK=2, threads_per_out=128) ──
// 4 warps per output; cross-warp reduce via smem with 4 partials.
extern "C" __global__ void w4a16_gemv_v1(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    constexpr unsigned int NB = 2;
    constexpr unsigned int TPO = BLOCK_SIZE / NB;  // 128
    constexpr unsigned int WARPS_PER_OUT = TPO / WARP_SIZE;  // 4

    const unsigned int local_out = threadIdx.x / TPO;
    const unsigned int lane = threadIdx.x % TPO;

    const unsigned int n = blockIdx.x * NB + local_out;
    const bool valid = n < N;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float smem[NB * WARPS_PER_OUT];  // 8 floats
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;
    if (valid) {
    W4A16_INNER_FMA(acc, lane, TPO)
    }

    // Warp shuffle reduction (32 threads per warp)
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // Cross-warp reduction via smem
    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    const unsigned int warp_idx = lane / WARP_SIZE;  // 0..WARPS_PER_OUT-1
    if (warp_lane == 0) {
        smem[local_out * WARPS_PER_OUT + warp_idx] = acc;
    }
    __syncthreads();

    if (valid && lane == 0) {
        float result = 0.0f;
        #pragma unroll
        for (unsigned int w = 0; w < WARPS_PER_OUT; w++) {
            result += smem[local_out * WARPS_PER_OUT + w];
        }
        C[n] = __float2bfloat16(result);
    }
}

// ── Variant 2: (N_PER_BLOCK=1, threads_per_out=256) ──
// 8 warps per output; one output per CTA. Max grid → max blocks resident.
extern "C" __global__ void w4a16_gemv_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    constexpr unsigned int NB = 1;
    constexpr unsigned int TPO = BLOCK_SIZE / NB;  // 256
    constexpr unsigned int WARPS_PER_OUT = TPO / WARP_SIZE;  // 8

    const unsigned int local_out = 0;
    const unsigned int lane = threadIdx.x;

    const unsigned int n = blockIdx.x;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float smem[WARPS_PER_OUT];  // 8 floats
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;
    W4A16_INNER_FMA(acc, lane, TPO)

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    const unsigned int warp_idx = threadIdx.x / WARP_SIZE;
    if (warp_lane == 0) {
        smem[warp_idx] = acc;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float result = 0.0f;
        #pragma unroll
        for (unsigned int w = 0; w < WARPS_PER_OUT; w++) {
            result += smem[w];
        }
        C[n] = __float2bfloat16(result);
        (void)local_out;
    }
}

// ── Variant 3: (N_PER_BLOCK=8, threads_per_out=32) ──
// 1 warp per output — pure warp shuffle, no cross-warp smem reduce.
extern "C" __global__ void w4a16_gemv_v3(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    constexpr unsigned int NB = 8;
    constexpr unsigned int TPO = BLOCK_SIZE / NB;  // 32

    const unsigned int local_out = threadIdx.x / TPO;
    const unsigned int lane = threadIdx.x % TPO;

    const unsigned int n = blockIdx.x * NB + local_out;
    const bool valid = n < N;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;
    if (valid) {
    W4A16_INNER_FMA(acc, lane, TPO)
    }

    // Pure warp shuffle (no cross-warp reduce since 1 warp == 1 output)
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (valid && lane == 0) {
        C[n] = __float2bfloat16(acc);
    }
}

// ============================================================
// W4A16 GEMV batch3 LOGITS: M=3 GEMV specialized for LM head.
// ============================================================
// Same M=3 algorithm as `w4a16_gemv_batch3` (reads each NVFP4 weight row
// ONCE and FMAs against 3 input rows), but tuned for the large-N LM-head
// case (vocab=248320 vs ~12k for SSM QKVZ):
//
//  * `N_PER_BLOCK = 8` → grid shrinks 2×  (62080 → 31040 CTAs)
//  * `threads_per_out = 32`  → exactly 1 warp per output → NO smem
//    cross-warp reduce. Only the 16-float LUT lives in smem.
//  * Output is BF16 [3, N] contiguous (row-major, vocab is the inner dim),
//    matching what the existing K=3 verify argmax expects.
//
// Bandwidth analysis (Qwen3.6-27B lm_head, hidden=5120 vocab=248320):
//   Weight read   = 5120 × 248320 × 0.5625 B = 715 MB
//   Activation rd = 3 × 5120 × 2 B           ≈  30 KB  (L1 / L2 cached)
//   Output write  = 3 × 248320 × 2 B         ≈ 1.5 MB
// Theoretical at 273 GB/s: ~2.6 ms. The 18.7 ms measured by the M=3
// fallback through `w4a16_gemm` was 95% wasted M-tile (64×64 tile reads
// the weight but only 3 of 64 M-rows are valid).
//
// Grid: (ceil(N / N_PER_BLOCK), 1, 1)   Block: (BLOCK_SIZE, 1, 1)
//
// Input layout:
//   A           [3, K]      BF16 contiguous (3 token hidden states)
//   B_packed    [N, K/2]    NVFP4 packed (2 weights per byte)
//   B_scale     [N, K/16]   FP8-E4M3 (one scale per 16 K)
//   C           [3, N]      BF16 contiguous (3 token logits)
extern "C" __global__ void w4a16_gemv_batch3_logits(
    const __nv_bfloat16* __restrict__ A,        // [3, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [3, N]
    unsigned int N,
    unsigned int K
) {
    constexpr unsigned int NB = 8;                  // 8 output cols per CTA
    constexpr unsigned int TPO = BLOCK_SIZE / NB;   // 32 → 1 warp per output

    const unsigned int local_out = threadIdx.x / TPO;
    const unsigned int lane = threadIdx.x % TPO;

    const unsigned int n = blockIdx.x * NB + local_out;
    const bool valid = n < N;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    const __nv_bfloat16* __restrict__ A1 = A + K;
    const __nv_bfloat16* __restrict__ A2 = A + 2 * K;
    __nv_bfloat16* __restrict__ C1 = C + N;
    __nv_bfloat16* __restrict__ C2 = C + 2 * N;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;

    // Process 8 K-values per iteration — matches the batch3 kernel layout
    // (uint4 activation read = 8 BF16 values; uint32_t weight read = 4 bytes
    // = 8 FP4 values; exactly half of GROUP_SIZE → 2 iters per scale).
    if (valid) {
    for (unsigned int k8 = lane; k8 < K8; k8 += TPO) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        uint4 a2_data = ((const uint4*)A2)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int a2_raw[4] = {a2_data.x, a2_data.y, a2_data.z, a2_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);

        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo + __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo + __bfloat162float(a1_hi) * w_hi;

            __nv_bfloat16 a2_lo, a2_hi;
            *(unsigned short*)&a2_lo = (unsigned short)(a2_raw[b] & 0xFFFF);
            *(unsigned short*)&a2_hi = (unsigned short)(a2_raw[b] >> 16);
            acc2 += __bfloat162float(a2_lo) * w_lo + __bfloat162float(a2_hi) * w_hi;
        }
    }
    }

    // 1 warp per output → pure shuffle reduce, no smem cross-warp step.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    if (valid && lane == 0) {
        C[n]  = __float2bfloat16(acc0);
        C1[n] = __float2bfloat16(acc1);
        C2[n] = __float2bfloat16(acc2);
    }
}

// ── Variant 4: BLOCK_SIZE=128, N_PER_BLOCK=2, threads_per_out=64 ──
// Same threads-per-output as baseline (64, so 2 warps/output, identical
// cross-warp reduce cost), but block is HALF the size — so grid count
// is 2× baseline. More resident blocks at the same per-thread workload
// can help saturate LPDDR5X by overlapping load latency across more CTAs.
extern "C" __global__ void w4a16_gemv_v4(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    constexpr unsigned int BS = 128;
    constexpr unsigned int NB = 2;
    constexpr unsigned int TPO = BS / NB;  // 64
    constexpr unsigned int WARPS_PER_OUT = TPO / WARP_SIZE;  // 2

    const unsigned int local_out = threadIdx.x / TPO;
    const unsigned int lane = threadIdx.x % TPO;

    const unsigned int n = blockIdx.x * NB + local_out;
    const bool valid = n < N;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float smem[NB * WARPS_PER_OUT];  // 4 floats
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;
    if (valid) {
    W4A16_INNER_FMA(acc, lane, TPO)
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    const unsigned int warp_idx = lane / WARP_SIZE;
    if (warp_lane == 0) {
        smem[local_out * WARPS_PER_OUT + warp_idx] = acc;
    }
    __syncthreads();

    if (valid && lane == 0) {
        float result = 0.0f;
        #pragma unroll
        for (unsigned int w = 0; w < WARPS_PER_OUT; w++) {
            result += smem[local_out * WARPS_PER_OUT + w];
        }
        C[n] = __float2bfloat16(result);
    }
}


// ============================================================
// Strided variants of w4a16_gemv_qg_batch3 and w4a16_gemv_dual_batch3
// ============================================================
// These accept an additional `out_stride` (in BF16 elements) so that the
// per-token outputs land directly in an interleaved [Q|K|V] layout
// (qkv_buf) without an intermediate scratch + d2d copy. When
// out_stride == N the layout matches the non-strided kernel.
//
// Used by the attention Q/K/V projection path when
// ATLAS_ATTN_QKV_FUSED=1.

extern "C" __global__ void w4a16_gemv_qg_batch3_strided(
    const __nv_bfloat16* __restrict__ A,        // [3, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // base pointer; token i at C + i*out_stride
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int out_stride                       // BF16 elements between successive tokens
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = n < N;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    const __nv_bfloat16* __restrict__ A1 = A + K;
    const __nv_bfloat16* __restrict__ A2 = A + 2 * K;
    __nv_bfloat16* __restrict__ C1 = C + out_stride;
    __nv_bfloat16* __restrict__ C2 = C + 2 * out_stride;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 6];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;

    if (valid) {
    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        uint4 a2_data = ((const uint4*)A2)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int a2_raw[4] = {a2_data.x, a2_data.y, a2_data.z, a2_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo + __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo + __bfloat162float(a1_hi) * w_hi;

            __nv_bfloat16 a2_lo, a2_hi;
            *(unsigned short*)&a2_lo = (unsigned short)(a2_raw[b] & 0xFFFF);
            *(unsigned short*)&a2_hi = (unsigned short)(a2_raw[b] >> 16);
            acc2 += __bfloat162float(a2_lo) * w_lo + __bfloat162float(a2_hi) * w_hi;
        }
    }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 6 + warp_idx * 3]     = acc0;
        smem[local_out * 6 + warp_idx * 3 + 1] = acc1;
        smem[local_out * 6 + warp_idx * 3 + 2] = acc2;
    }
    __syncthreads();

    if (valid && lane == 0) {
        float result0 = smem[local_out * 6]     + smem[local_out * 6 + 3];
        float result1 = smem[local_out * 6 + 1] + smem[local_out * 6 + 4];
        float result2 = smem[local_out * 6 + 2] + smem[local_out * 6 + 5];

        unsigned int group_dim = 2 * head_dim;
        unsigned int h = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_heads * head_dim;

        unsigned int out_idx;
        if (idx < head_dim) {
            out_idx = h * head_dim + idx;
        } else {
            out_idx = q_total + h * head_dim + (idx - head_dim);
        }
        C[out_idx]  = __float2bfloat16(result0);
        C1[out_idx] = __float2bfloat16(result1);
        C2[out_idx] = __float2bfloat16(result2);
    }
}

extern "C" __global__ void w4a16_gemv_dual_batch3_strided(
    const __nv_bfloat16* __restrict__ A,         // [3, K_in] BF16
    const unsigned char* __restrict__ B0_packed,
    const unsigned char* __restrict__ B0_scale,
    float B0_scale2,
    __nv_bfloat16* __restrict__ C0,              // base; token i at C0 + i*out_stride
    const unsigned char* __restrict__ B1_packed,
    const unsigned char* __restrict__ B1_scale,
    float B1_scale2,
    __nv_bfloat16* __restrict__ C1,              // base; token i at C1 + i*out_stride
    unsigned int N,
    unsigned int K_in,
    unsigned int out_stride                       // BF16 elements between successive tokens
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = (proj == 0) ? B0_packed : B1_packed;
    const unsigned char* B_scale = (proj == 0) ? B0_scale : B1_scale;
    float s2 = (proj == 0) ? B0_scale2 : B1_scale2;
    __nv_bfloat16* C_out = (proj == 0) ? C0 : C1;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = n < N;

    const unsigned int half_K = K_in / 2;
    const unsigned int num_groups = K_in / GROUP_SIZE;
    const unsigned int K8 = K_in / 8;

    const __nv_bfloat16* A1 = A + K_in;
    const __nv_bfloat16* A2 = A + 2 * K_in;
    __nv_bfloat16* C_out1 = C_out + out_stride;
    __nv_bfloat16* C_out2 = C_out + 2 * out_stride;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 6];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;

    if (valid) {
    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        uint4 a2_data = ((const uint4*)A2)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int a2_raw[4] = {a2_data.x, a2_data.y, a2_data.z, a2_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb = B_scale[(unsigned long long)n * num_groups + sg];
        __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
        float scale = (float)fp8 * s2;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char bv = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[bv & 0xF] * scale;
            float w_hi = s_lut[bv >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo + __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo + __bfloat162float(a1_hi) * w_hi;

            __nv_bfloat16 a2_lo, a2_hi;
            *(unsigned short*)&a2_lo = (unsigned short)(a2_raw[b] & 0xFFFF);
            *(unsigned short*)&a2_hi = (unsigned short)(a2_raw[b] >> 16);
            acc2 += __bfloat162float(a2_lo) * w_lo + __bfloat162float(a2_hi) * w_hi;
        }
    }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 6 + warp_idx * 3]     = acc0;
        smem[local_out * 6 + warp_idx * 3 + 1] = acc1;
        smem[local_out * 6 + warp_idx * 3 + 2] = acc2;
    }
    __syncthreads();

    if (valid && lane == 0) {
        float result0 = smem[local_out * 6]     + smem[local_out * 6 + 3];
        float result1 = smem[local_out * 6 + 1] + smem[local_out * 6 + 4];
        float result2 = smem[local_out * 6 + 2] + smem[local_out * 6 + 5];
        C_out[n]  = __float2bfloat16(result0);
        C_out1[n] = __float2bfloat16(result1);
        C_out2[n] = __float2bfloat16(result2);
    }
}

// ============================================================
// W4A16 GEMV dual batch3 TUNED — fused gate+up, wider inner loop
// ============================================================
// Drop-in replacement for `w4a16_gemv_dual_batch3` gated behind
// `ATLAS_FFN_DUAL_TUNED=1`. Two stacked optimisations vs the baseline:
//
//   1. Fused gate+up dispatch — both projections compute in the SAME CTA
//      (8 outputs / CTA: 4 gate + 4 up, dispatched by warp id) instead of
//      the baseline `blockIdx.z = 2` fan-out. Grid drops from
//      (ceil(N/4), 1, 2) to (ceil(N/4), 1, 1) — exactly half as many CTAs
//      across the kernel. The 3-token activation vector `A[3, K_in]` is
//      L1-shared across all 8 outputs in the CTA instead of being read
//      once per (CTA, projection), halving the L2 A-read traffic.
//
//   2. K=16 inner loop — each iteration consumes 16 K-values (8-byte
//      weight load + 2× uint4 act loads per token) instead of 8. Halves
//      the K-loop trip count, the integer index-math overhead, and the
//      FP8-scale fetch frequency (one scale spans 16 K-vals exactly at
//      GROUP_SIZE=16).
//
// Geometry:
//   - 8 outputs per CTA, 64 threads per output (2 warps), 512 thr/block
//   - Grid:  (ceil(N/4), 1, 1)   half the baseline CTA count
//   - Block: (512, 1, 1)
//   - `__launch_bounds__(512, 2)` requests at least 2 resident blocks/SM
//
// Mapping:
//   local_out = threadIdx.x / 64   in [0..8)
//   proj      = local_out & 1      0=gate, 1=up
//   n_local   = local_out >> 1     in [0..4)
//   n         = blockIdx.x * 4 + n_local
//
// Output buffer layout is identical to `w4a16_gemv_dual_batch3` (gate to
// C0, up to C1) so the downstream silu_mul + down kernels need no changes
// and the bytes are equivalent at FP arithmetic precision.
//
// Resources (ptxas --gpu-name sm_121 + cuobjdump --dump-resource-usage):
//   64 registers / thread, 256 B static smem, 0 stack, 0 spills.
//   2 blocks/SM resident (vs 6 for baseline at 40 reg × 256 thr),
//   so 1024 threads/SM (vs 1536) — losing 33% thread-level occupancy in
//   exchange for halving the CTA count and the activation traffic.
//
// Measured impact on Qwen3.6-27B (N=12288, K=5120) counting prompt:
//   ffn_gate_up_dual_batch3 per-call: 477 us (baseline) -> 473-474 us
//   (tuned), ~1% improvement. End-to-end server tok/s: 33.5 vs 33.4 — at
//   the floor of LPDDR5X bandwidth + launch-overhead noise for this CTA
//   geometry. The structural changes are correct; further headroom needs
//   either pipelined activation prefetch (cp.async) or a fundamentally
//   different weight-layout (eg interleaved gate/up rows for one bus burst
//   per weight pair) which are out of scope for this drop-in replacement.
extern "C" __global__ __launch_bounds__(512, 2)
void w4a16_gemv_dual_batch3_tuned(
    const __nv_bfloat16* __restrict__ A,         // [3, K_in] BF16 (shared)
    const unsigned char* __restrict__ B0_packed,  // [N, K_in/2] gate
    const unsigned char* __restrict__ B0_scale,
    float B0_scale2,
    __nv_bfloat16* __restrict__ C0,              // [3, N] gate output
    const unsigned char* __restrict__ B1_packed,  // [N, K_in/2] up
    const unsigned char* __restrict__ B1_scale,
    float B1_scale2,
    __nv_bfloat16* __restrict__ C1,              // [3, N] up output
    unsigned int N,
    unsigned int K_in
) {
    // 8 outputs per CTA (4 gate + 4 up), 64 threads per output (2 warps).
    // 512 threads/block. The activation vector A[3, K_in] is read once per CTA
    // (4 gate + 4 up outputs share it via L1) instead of once per CTA per
    // projection like the baseline blockIdx.z=2 dispatch -> halves L2 A-read
    // traffic across the kernel (12288 baseline CTAs -> 3072 tuned CTAs).
    constexpr unsigned int TPO = 64u;
    const unsigned int local_out = threadIdx.x / TPO;        // 0..8
    const unsigned int lane      = threadIdx.x % TPO;        // 0..64
    const unsigned int proj      = local_out & 1u;
    const unsigned int n_local   = local_out >> 1;           // 0..4

    const unsigned char* B_packed = (proj == 0u) ? B0_packed : B1_packed;
    const unsigned char* B_scale  = (proj == 0u) ? B0_scale  : B1_scale;
    const float          s2       = (proj == 0u) ? B0_scale2 : B1_scale2;
    __nv_bfloat16*       C_out    = (proj == 0u) ? C0        : C1;

    const unsigned int n = blockIdx.x * 4u + n_local;
    const bool valid = n < N;

    const unsigned int half_K     = K_in / 2;
    const unsigned int num_groups = K_in / GROUP_SIZE;
    const unsigned int K16        = K_in / 16;  // 16 K-vals per iter

    const __nv_bfloat16* A1 = A  + K_in;
    const __nv_bfloat16* A2 = A1 + K_in;
    __nv_bfloat16* C_out1 = C_out + N;
    __nv_bfloat16* C_out2 = C_out + 2 * N;

    // 8 outputs/CTA × 2 warps/out × 3 tokens = 48 floats for cross-warp reduce.
    __shared__ float s_lut[16];
    __shared__ float smem[8 * 6];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;

    // 16 K-values per iteration: 2× uint4 acts (256-bit) + uint64 weight (8 bytes)
    // + 1 scale byte (GROUP_SIZE=16 -> exactly one scale spans the whole iter).
    if (valid) {
    for (unsigned int k16 = lane; k16 < K16; k16 += TPO) {
        const unsigned int base_k = k16 * 16;

        // Two adjacent 128-bit loads -> 16 BF16 acts per token (8 uint per token).
        uint4 a0_lo4 = ((const uint4*)A )[k16 * 2];
        uint4 a0_hi4 = ((const uint4*)A )[k16 * 2 + 1];
        uint4 a1_lo4 = ((const uint4*)A1)[k16 * 2];
        uint4 a1_hi4 = ((const uint4*)A1)[k16 * 2 + 1];
        uint4 a2_lo4 = ((const uint4*)A2)[k16 * 2];
        uint4 a2_hi4 = ((const uint4*)A2)[k16 * 2 + 1];
        const unsigned int a0_raw[8] = {a0_lo4.x, a0_lo4.y, a0_lo4.z, a0_lo4.w,
                                         a0_hi4.x, a0_hi4.y, a0_hi4.z, a0_hi4.w};
        const unsigned int a1_raw[8] = {a1_lo4.x, a1_lo4.y, a1_lo4.z, a1_lo4.w,
                                         a1_hi4.x, a1_hi4.y, a1_hi4.z, a1_hi4.w};
        const unsigned int a2_raw[8] = {a2_lo4.x, a2_lo4.y, a2_lo4.z, a2_lo4.w,
                                         a2_hi4.x, a2_hi4.y, a2_hi4.z, a2_hi4.w};

        // 8 packed bytes -> 16 weight values.
        unsigned long long packed8 = *(const unsigned long long*)(
            B_packed + (unsigned long long)n * half_K + k16 * 8);

        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb = B_scale[(unsigned long long)n * num_groups + sg];
        __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
        float scale = (float)fp8 * s2;

        #pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char bv = (unsigned char)(packed8 >> (b * 8));
            float w_lo = s_lut[bv & 0xF] * scale;
            float w_hi = s_lut[bv >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo + __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo + __bfloat162float(a1_hi) * w_hi;

            __nv_bfloat16 a2_lo, a2_hi;
            *(unsigned short*)&a2_lo = (unsigned short)(a2_raw[b] & 0xFFFF);
            *(unsigned short*)&a2_hi = (unsigned short)(a2_raw[b] >> 16);
            acc2 += __bfloat162float(a2_lo) * w_lo + __bfloat162float(a2_hi) * w_hi;
        }
    }
    }

    // 2 warps per output -> warp shuffle then smem cross-warp reduce (3 acc each).
    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 6 + warp_idx * 3]     = acc0;
        smem[local_out * 6 + warp_idx * 3 + 1] = acc1;
        smem[local_out * 6 + warp_idx * 3 + 2] = acc2;
    }
    __syncthreads();

    if (valid && lane == 0) {
        float r0 = smem[local_out * 6]     + smem[local_out * 6 + 3];
        float r1 = smem[local_out * 6 + 1] + smem[local_out * 6 + 4];
        float r2 = smem[local_out * 6 + 2] + smem[local_out * 6 + 5];
        C_out [n] = __float2bfloat16(r0);
        C_out1[n] = __float2bfloat16(r1);
        C_out2[n] = __float2bfloat16(r2);
    }
}
// w4a16_gemv_dual_batch3_tuned
