// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A8 Pipelined GEMM — FP8-native MMA (m16n8k32.e4m3) rewrite of
// w8a16_gemm_pipelined.
//
// C[M,N] = a_row_scale[m] * ( A_fp8[M,K] · B[N,K]^T ) with 2D block scales
//
// Why: w8a16_gemm_pipelined is ISSUE-bound (its own header: occupancy/issue,
// not DRAM), at ~12 TFLOP/s. FP8-native MMA attacks exactly that:
//   - ONE m16n8k32 MMA per 32-K step where BF16 needed TWO m16n8k16 → half
//     the MMA issues per K traversed.
//   - raw E4M3 bytes are the MMA operand — the cooperative LUT-dequant
//     phase, its smem_B buffer, the smem LUT, and one __syncthreads per
//     K-step are all DELETED.
//   - A is E4M3 too (quantized per-row by quantize_a_fp8_rows below): half
//     the A smem + cp.async traffic.
//
// Numerics: A carries one FP32 scale per row (absmax/448), folded once at
// the epilogue; the two-level FP32 accumulation from w8a16 is preserved
// (inner per 128-K block, outer += inner * block_scale at the boundary).
// Activation quantization to E4M3 is LOSSY (~2 decimal digits/element) —
// this path is behaviour-gated (tool-eval-bench) like every prefill
// numerics change; per-row scales keep the relative error bounded.
//
// Layouts: A_fp8 [M, K] row-major bytes; B [N, K] row-major bytes (same as
// w8a16); block_scale [N/128, K/128] FP32; a_row_scale [M] FP32.
// smem rows are padded to 48 B (multiple of 16 for cp.async.cg; 12-word
// stride makes BOTH operand reads bank-conflict-free across a warp).
//
// Grid: (ceil(N/32), ceil(M/128), 1)  Block: (256,1,1) = 8 warps.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define P8_M_TILE 128
#define P8_N_TILE 32
#define P8_K_STEP 32
#define P8_STRIDE 48   // 32 K-bytes + 16 pad; multiple of 16 for cp.async
#define P8_FP8_BLOCK 128
#define P8_WARPS 8
#define P8_THREADS (P8_WARPS * 32)
#define P8_N_TILES_PER_WARP (P8_N_TILE / 8)
#define P8_STAGES 2

__device__ __forceinline__ void p8_cp_async_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void p8_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void p8_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void p8_cp_wait_le(unsigned int n) {
    switch (n) {
        case 0:  p8_cp_wait<0>(); break;
        case 1:  p8_cp_wait<1>(); break;
        default: p8_cp_wait<2>(); break;
    }
}

extern "C" __global__ void w8a8_gemm_pipelined(
    const unsigned char* __restrict__ A,        // [M, K] FP8 E4M3 (pre-quantized)
    const float* __restrict__ a_row_scale,      // [M] FP32
    const unsigned char* __restrict__ B,         // [N, K] FP8 E4M3
    const float* __restrict__ block_scale,       // [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,               // [M, N] BF16
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * P8_M_TILE;
    const unsigned int cta_n = blockIdx.x * P8_N_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __align__(16) unsigned char smem_A[P8_STAGES][P8_M_TILE][P8_STRIDE];
    __shared__ __align__(16) unsigned char smem_B[P8_STAGES][P8_N_TILE][P8_STRIDE];

    float inner_acc[P8_N_TILES_PER_WARP][4];
    float outer_acc[P8_N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < P8_N_TILES_PER_WARP; i++) {
        inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
        inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
        outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
        outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
    }

    const unsigned int k_blocks = K / P8_FP8_BLOCK;
    const unsigned int k_steps_per_block = P8_FP8_BLOCK / P8_K_STEP;
    const unsigned int n_block = cta_n / P8_FP8_BLOCK;
    const unsigned int n_steps = (K + P8_K_STEP - 1) / P8_K_STEP;

    // A: 128 rows × 32 K-bytes = two 16-B chunks/row = 256 chunks.
    // B:  32 rows × 32 K-bytes = 64 chunks.
    auto prefetch = [&](unsigned int step, unsigned int stage) {
        const unsigned int k_base = step * P8_K_STEP;
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < (P8_M_TILE * P8_K_STEP) / 16;
             c += P8_THREADS) {
            const unsigned int row = (c * 16) / P8_K_STEP;
            const unsigned int col = (c * 16) % P8_K_STEP;
            const unsigned int gr = cta_m + row;
            const unsigned int gc = k_base + col;
            unsigned char* dst = &smem_A[stage][row][col];
            if (gr < M && gc + 16 <= K) {
                p8_cp_async_16(dst, &A[(unsigned long long)gr * K + gc]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 16; e++) {
                    const unsigned int gce = gc + e;
                    dst[e] = (gr < M && gce < K)
                        ? A[(unsigned long long)gr * K + gce]
                        : 0;
                }
            }
        }
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < (P8_N_TILE * P8_K_STEP) / 16;
             c += P8_THREADS) {
            const unsigned int nrow = (c * 16) / P8_K_STEP;
            const unsigned int kcol = (c * 16) % P8_K_STEP;
            const unsigned int gn = cta_n + nrow;
            const unsigned int gk = k_base + kcol;
            unsigned char* dst = &smem_B[stage][nrow][kcol];
            if (gn < N && gk + 16 <= K) {
                p8_cp_async_16(dst, &B[(unsigned long long)gn * K + gk]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 16; e++) {
                    const unsigned int gke = gk + e;
                    dst[e] = (gn < N && gke < K)
                        ? B[(unsigned long long)gn * K + gke]
                        : 0;
                }
            }
        }
        p8_cp_commit();
    };

    #pragma unroll
    for (unsigned int p = 0; p < P8_STAGES - 1; p++) {
        if (p < n_steps) prefetch(p, p % P8_STAGES);
    }
    unsigned int k_step_in_block = 0;

    for (unsigned int step = 0; step < n_steps; step++) {
        const unsigned int cur = step % P8_STAGES;
        const unsigned int ahead = step + (P8_STAGES - 1);
        if (ahead < n_steps) prefetch(ahead, ahead % P8_STAGES);
        const unsigned int committed = min(n_steps, P8_STAGES + step);
        p8_cp_wait_le(committed - (step + 1));
        __syncthreads();   // smem_[AB][cur] resident for all threads

        // ── ONE m16n8k32 e4m3 MMA per N-tile per K-step ──
        {
            const unsigned char* sA = &smem_A[cur][0][0];
            const unsigned char* sB = &smem_B[cur][0][0];
            const unsigned int r0 = warp_m_offset + group_id;
            const unsigned int r1 = r0 + 8;
            // A [16x32] row-major: 4 consecutive K-bytes per u32, threads
            // cover k = tid*4 (+16 for the high half).
            const unsigned int a0 = *(const unsigned int*)&sA[r0 * P8_STRIDE + tid * 4];
            const unsigned int a1 = *(const unsigned int*)&sA[r1 * P8_STRIDE + tid * 4];
            const unsigned int a2 = *(const unsigned int*)&sA[r0 * P8_STRIDE + tid * 4 + 16];
            const unsigned int a3 = *(const unsigned int*)&sA[r1 * P8_STRIDE + tid * 4 + 16];
            #pragma unroll
            for (int n_tile = 0; n_tile < P8_N_TILES_PER_WARP; n_tile++) {
                const unsigned int n_col = n_tile * 8 + group_id;
                const unsigned int b0 =
                    *(const unsigned int*)&sB[n_col * P8_STRIDE + tid * 4];
                const unsigned int b1 =
                    *(const unsigned int*)&sB[n_col * P8_STRIDE + tid * 4 + 16];
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%10, %11, %12, %13};"
                    : "=f"(inner_acc[n_tile][0]), "=f"(inner_acc[n_tile][1]),
                      "=f"(inner_acc[n_tile][2]), "=f"(inner_acc[n_tile][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1),
                      "f"(inner_acc[n_tile][0]), "f"(inner_acc[n_tile][1]),
                      "f"(inner_acc[n_tile][2]), "f"(inner_acc[n_tile][3]));
            }
        }
        __syncthreads();   // done reading smem_*[cur]

        k_step_in_block++;
        if (k_step_in_block == k_steps_per_block) {
            const unsigned int k_block = (step * P8_K_STEP) / P8_FP8_BLOCK;
            const float scale = block_scale[n_block * k_blocks + k_block];
            #pragma unroll
            for (int i = 0; i < P8_N_TILES_PER_WARP; i++) {
                outer_acc[i][0] += inner_acc[i][0] * scale;
                outer_acc[i][1] += inner_acc[i][1] * scale;
                outer_acc[i][2] += inner_acc[i][2] * scale;
                outer_acc[i][3] += inner_acc[i][3] * scale;
                inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
                inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
            }
            k_step_in_block = 0;
        }
    }
    if (k_step_in_block != 0) {
        const unsigned int k_block = (K - 1) / P8_FP8_BLOCK;
        const float scale = block_scale[n_block * k_blocks + k_block];
        #pragma unroll
        for (int i = 0; i < P8_N_TILES_PER_WARP; i++) {
            outer_acc[i][0] += inner_acc[i][0] * scale;
            outer_acc[i][1] += inner_acc[i][1] * scale;
            outer_acc[i][2] += inner_acc[i][2] * scale;
            outer_acc[i][3] += inner_acc[i][3] * scale;
        }
    }

    // ── Epilogue: fold the per-row A scale, store BF16 ──
    #pragma unroll
    for (int n_tile = 0; n_tile < P8_N_TILES_PER_WARP; n_tile++) {
        const unsigned int base_n = cta_n + n_tile * 8;
        const unsigned int col0 = base_n + tid * 2;
        const unsigned int col1 = col0 + 1;
        const unsigned int row0 = cta_m + warp_m_offset + group_id;
        const unsigned int row1 = row0 + 8;
        const float s0 = (row0 < M) ? a_row_scale[row0] : 0.0f;
        const float s1 = (row1 < M) ? a_row_scale[row1] : 0.0f;
        if (row0 < M && col0 < N)
            C[row0 * N + col0] = __float2bfloat16(outer_acc[n_tile][0] * s0);
        if (row0 < M && col1 < N)
            C[row0 * N + col1] = __float2bfloat16(outer_acc[n_tile][1] * s0);
        if (row1 < M && col0 < N)
            C[row1 * N + col0] = __float2bfloat16(outer_acc[n_tile][2] * s1);
        if (row1 < M && col1 < N)
            C[row1 * N + col1] = __float2bfloat16(outer_acc[n_tile][3] * s1);
    }
}

// ── Activation row-quantizer: BF16 [M, K] → E4M3 [M, K] + FP32 scale[M] ──
//
// scale[m] = max(absmax(A[m,:]), eps) / 448 (E4M3 max normal), quantize
// a / scale[m]. One block per row, 256 threads, shfl+smem absmax reduce.
extern "C" __global__ void quantize_a_fp8_rows(
    const __nv_bfloat16* __restrict__ A,   // [M, K] BF16
    unsigned char* __restrict__ A_fp8,      // [M, K] E4M3 out
    float* __restrict__ row_scale,          // [M] FP32 out
    unsigned int M,
    unsigned int K
) {
    const unsigned int m = blockIdx.x;
    if (m >= M) return;
    const __nv_bfloat16* row = A + (unsigned long long)m * K;

    float amax = 0.0f;
    for (unsigned int k = threadIdx.x; k < K; k += blockDim.x) {
        amax = fmaxf(amax, fabsf(__bfloat162float(row[k])));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xFFFFFFFFu, amax, off));
    }
    __shared__ float warp_max[8];
    if ((threadIdx.x & 31u) == 0) warp_max[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float b = warp_max[0];
        #pragma unroll
        for (int w = 1; w < 8; w++) b = fmaxf(b, warp_max[w]);
        warp_max[0] = fmaxf(b, 1e-8f) / 448.0f;
        row_scale[m] = warp_max[0];
    }
    __syncthreads();
    const float inv = 1.0f / warp_max[0];

    unsigned char* out = A_fp8 + (unsigned long long)m * K;
    for (unsigned int k = threadIdx.x; k < K; k += blockDim.x) {
        const __nv_fp8_e4m3 q(__bfloat162float(row[k]) * inv);
        out[k] = *(const unsigned char*)&q;
    }
}
