// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Dense BF16 multi-row GEMV (small M) for SM121 (GB10).
//
// The runtime-M sibling of dense_gemv_bf16 / dense_gemv_bf16_batch2:
//
//   C[t, n] = dot(A[t, :], B[n, :])   for t in [0, M)
//
//   A: [M, K] BF16, row t at A + t*a_stride
//   B: [N, K] BF16 (weights, row-major — standard HuggingFace layout)
//   C: [M, N] BF16, row t at C + t*out_stride
//
// Why this exists: the DFlash verify step runs the attention head-gate
// projection at M = gamma+1 (7 at gamma=6) with N = num_heads = 48. Routed
// through the prefill tensor-core GEMM (16 M-rows x 64 N-cols per tile) that
// is grid (ceil(48/64), ceil(7/16), 1) = ONE CTA — 128 threads, on 121 SMs,
// dragging the whole 294 KB BF16 weight through a single SM's L1. Same
// pathology the M=1 decode path had before ATLAS_DECODE_GPROJ_GEMV.
//
// Here the parallel axis is N (ceil(N/4) CTAs) and M rides along in
// registers, so the weight is still read exactly once but by 12 CTAs instead
// of 1. Numerically this is the M=1 dense_gemv_bf16 run M times: each row
// keeps its own accumulator over the identical K-iteration and reduction
// order, so row 0 is bit-identical to a dense_gemv_bf16 call.
//
// M is a runtime argument bounded by MAX_M; callers above that bound must
// stay on the tensor-core GEMM (which is the right kernel once M is large
// enough to fill the machine anyway).
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define VEC_SIZE 8  // BF16 values per vectorized load (uint4 = 16 bytes)
#define MAX_M 16    // gamma+1 for every speculative depth we run

extern "C" __global__ void dense_gemv_bf16_batchm(
    const __nv_bfloat16* __restrict__ A,  // [M, K], row t at A + t*a_stride
    const __nv_bfloat16* __restrict__ B,  // [N, K]
    __nv_bfloat16* __restrict__ C,        // [M, N], row t at C + t*out_stride
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_stride,                // BF16 elements between A rows
    unsigned int out_stride               // BF16 elements between C rows
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    // NOTE: no early return — every thread must reach the __syncthreads()
    // below. Out-of-range lanes carry zero accumulators and skip the store.
    const bool active = (n < N);

    float acc[MAX_M];
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;

    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)(active ? n : 0) * K);

    for (unsigned int kv = lane; kv < K_VEC && active; kv += threads_per_out) {
        const uint4 b_data = B_vec[kv];
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

        // Hoist the weight unpack out of the M loop: B is the bandwidth-bound
        // operand, A is tiny (M*K BF16, resident in L1/L2 across all CTAs).
        float bf[8];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 b_lo, b_hi;
            *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
            *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
            bf[2 * i] = __bfloat162float(b_lo);
            bf[2 * i + 1] = __bfloat162float(b_hi);
        }

        // Guarded, NOT `break`: a data-dependent exit would make `acc[t]` a
        // dynamic index and spill the accumulators to local memory. Predicated
        // full unroll keeps all MAX_M accumulators in registers.
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t < M) {
                const uint4 a_data =
                    ((const uint4*)(A + (unsigned long long)t * a_stride))[kv];
                const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    __nv_bfloat16 a_lo, a_hi;
                    *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                    *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
                    acc[t] += __bfloat162float(a_lo) * bf[2 * i];
                    acc[t] += __bfloat162float(a_hi) * bf[2 * i + 1];
                }
            }
        }
    }

    // Scalar tail for K not divisible by VEC_SIZE (never hits for model dims).
    if (active) {
        const unsigned int tail_start = K_VEC * VEC_SIZE;
        const __nv_bfloat16* B_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += threads_per_out) {
            const float b = __bfloat162float(B_row[k]);
            #pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned int)t < M) {
                    acc[t] += __bfloat162float(A[(unsigned long long)t * a_stride + k]) * b;
                }
            }
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t < M) {
                acc[t] += __shfl_down_sync(0xFFFFFFFF, acc[t], offset);
            }
        }
    }

    // 2 warps per output: cross-warp reduce through shared memory, per row.
    __shared__ float smem[MAX_M][N_PER_BLOCK * 2];

    if (warp_lane == 0) {
        const unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t < M) smem[t][smem_idx] = acc[t];
        }
    }
    __syncthreads();

    if (lane == 0 && active) {
        for (unsigned int t = 0; t < M; t++) {
            const float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            C[(unsigned long long)t * out_stride + n] = __float2bfloat16(r);
        }
    }
}

// ── GROUPED (block-diagonal) sibling ────────────────────────────────────────
//
// One launch for a block-diagonal projection: output row n belongs to group
// g = n / rows_per_group and reads A columns [g*K, (g+1)*K) of each activation
// row. The DSpark drafter's wo_a (o_groups=8 x [o_lora=1024, group_in=4096],
// weight rows contiguous) previously ran 8 x dense_gemm(16x64 TC tiles at
// M<=6 — 27 GB/s) PLUS 16 gather/scatter 2D copies per stage; this kernel
// replaces the whole block with one weight-streaming launch reading A/C
// strided in place. rows_per_group = N degenerates to plain batchm (used for
// the drafter wo_b so both projections leave the tensor-core GEMM).
//
// Per-row K-iteration and reduction order are identical to
// dense_gemv_bf16_batchm (and therefore to M x dense_gemv_bf16 on the same
// slice).
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void dense_gemv_bf16_grouped_batchm(
    const __nv_bfloat16* __restrict__ A,  // [M, a_stride], group cols inside
    const __nv_bfloat16* __restrict__ B,  // [N, K] rows contiguous across groups
    __nv_bfloat16* __restrict__ C,        // [M, out_stride]
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_stride,
    unsigned int out_stride,
    unsigned int rows_per_group
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool active = (n < N);

    // The one delta vs dense_gemv_bf16_batchm: this output row's activation
    // column segment.
    const unsigned long long a_col =
        (unsigned long long)(active ? n / rows_per_group : 0) * K;

    float acc[MAX_M];
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;

    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)(active ? n : 0) * K);

    for (unsigned int kv = lane; kv < K_VEC && active; kv += threads_per_out) {
        const uint4 b_data = B_vec[kv];
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

        float bf[8];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 b_lo, b_hi;
            *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
            *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
            bf[2 * i] = __bfloat162float(b_lo);
            bf[2 * i + 1] = __bfloat162float(b_hi);
        }

        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t < M) {
                const uint4 a_data = ((const uint4*)(
                    A + (unsigned long long)t * a_stride + a_col))[kv];
                const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    __nv_bfloat16 a_lo, a_hi;
                    *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                    *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
                    acc[t] += __bfloat162float(a_lo) * bf[2 * i];
                    acc[t] += __bfloat162float(a_hi) * bf[2 * i + 1];
                }
            }
        }
    }

    if (active) {
        const unsigned int tail_start = K_VEC * VEC_SIZE;
        const __nv_bfloat16* B_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += threads_per_out) {
            const float b = __bfloat162float(B_row[k]);
            #pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned int)t < M) {
                    acc[t] += __bfloat162float(
                        A[(unsigned long long)t * a_stride + a_col + k]) * b;
                }
            }
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t < M) {
                acc[t] += __shfl_down_sync(0xFFFFFFFF, acc[t], offset);
            }
        }
    }

    __shared__ float smem_g[MAX_M][N_PER_BLOCK * 2];
    if (warp_lane == 0) {
        const unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t < M) smem_g[t][smem_idx] = acc[t];
        }
    }
    __syncthreads();

    if (lane == 0 && active) {
        for (unsigned int t = 0; t < M; t++) {
            const float r = smem_g[t][local_out * 2] + smem_g[t][local_out * 2 + 1];
            C[(unsigned long long)t * out_stride + n] = __float2bfloat16(r);
        }
    }
}
