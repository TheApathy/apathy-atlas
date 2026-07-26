// SPDX-License-Identifier: AGPL-3.0-only
//
// ═══════════════════════════════════════════════════════════════════
// dense_gemm_bf16_mtile16 — small-M (M ≤ 16) BF16 weight-streaming GEMM
// for the DFlash drafter propose path (ATLAS_DFLASH_DRAFTER_FASTGEMM=1).
//
//   C[M,N] = A[M,K] (BF16) · B[N,K]^T (BF16), FP32 accumulate, BF16 out.
//
// Modeled EXACTLY on kernels/gb10/laguna-s-2.1/nvfp4/w4a16_gemm.cu::
// fp8_gemm_t_row_scaled_mtile8 (proven 1.8-4.5× at M=7), adapted to:
//   - BF16 weights (2 bytes/elem) instead of FP8 → B stage = 8 KB.
//   - M ≤ 16 (the drafter's γ=16 block) instead of M ≤ 8: one full
//     m16n8k16 BF16 MMA row-block, no dead A-fragment rows.
//   - mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 — the SAME
//     instruction, K-chunk width (16) and ascending-K accumulation
//     chain as dense_gemm_bf16_pipelined (dense_gemm_bf16.cu), so the
//     output is expected BIT-IDENTICAL to that kernel (verified by the
//     dflash_bf16gemm_smallm_microtest bytewise gate).
//
// Why the existing dense_gemm_bf16_pipelined is slow at the drafter's
// shapes: its 128×128 CTA tile puts only N/128 CTAs on the grid —
// o_proj/down (N=3072) = 24 CTAs, k/v (N=1024) = 8 CTAs on a 48-SM
// GB10, and its 2-stage cp.async ring keeps just ~8 KB of B in flight
// per CTA — far too little to hide LPDDR5x latency at small M where
// the kernel is purely weight-read-bound.
//
// This kernel: each CTA streams a contiguous N_TILE=64 slice of B
// exactly once over the full K range with a 4-stage cp.async ring,
// K_STEP=64 → 3 stages × 8 KB of B permanently in flight per CTA.
// Grid ceil(N/64): q=144, o/down=48, gate/up=192, lm_head=1568 CTAs —
// multiples of the 48 SMs for every major drafter shape. The [M≤16,K]
// BF16 activation streams alongside (2 KB/stage; A is tiny and shared
// by all CTAs → L2-resident, no extra DRAM traffic).
//
// 4 warps × 16 cols = 64 cols/CTA; each warp does 4 K-subchunks × 2
// n8 tiles = 8 MMAs per 64-wide K step.
//
// smem: A 4×16×72×2 = 9216 B + B 4×64×72×2 = 36864 B = 46080 B/CTA →
// 2 CTAs/SM on GB10's 101 KB.
//
// Contract: K % 8 == 0 (16-B cp.async row alignment; host dispatch
// guarantees — all drafter K's are {3072, 9216, 12288, 18432}).
// M ≤ 16 (rows ≥ 16 are never computed). Pure device args — CUDA
// graph-capture compatible.
//
// A: [M, K] BF16, B: [N, K] BF16 (HF layout), C: [M, N] BF16.
// Grid: (ceil(N/64), 1, 1)  Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════

#include <cuda_bf16.h>

#define DM16_N_TILE 64
#define DM16_K_STEP 64
#define DM16_STAGES 4
// 64 + 8 pad elems: 144-B rows, 16-B aligned, breaks bank conflicts on
// the MMA's u32 fragment reads.
#define DM16_STRIDE 72

__device__ __forceinline__ void dm16_cp_async_pred_16(
    void* dst_smem, const void* src_gmem, bool pred
) {
    unsigned int dst = (unsigned int)__cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}
__device__ __forceinline__ void dm16_cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::);
}
__device__ __forceinline__ void dm16_cp_async_wait_prior_2() {
    // With the ring below there are always exactly 3 outstanding groups
    // before this call; waiting to ≤2 waits precisely for the oldest
    // (the stage about to be computed).
    asm volatile("cp.async.wait_group 2;");
}

extern "C" __global__ void dense_gemm_bf16_mtile16(
    const __nv_bfloat16* __restrict__ A,   // [M, K] BF16, M <= 16
    const __nv_bfloat16* __restrict__ B,   // [N, K] BF16 weights
    __nv_bfloat16* __restrict__ C,         // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * DM16_N_TILE;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __align__(16) __nv_bfloat16 smem_A[DM16_STAGES][16][DM16_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 smem_B[DM16_STAGES][DM16_N_TILE][DM16_STRIDE];

    // 2 n8 tiles per warp; all four accumulator rows are live (M ≤ 16).
    float acc[2][4];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int num_k = (K + DM16_K_STEP - 1) / DM16_K_STEP;

    // Stage load:
    //   A = 16 rows × 64 cols BF16 = 128 × 16-B chunks → 1 chunk/thread.
    //   B = 64 rows × 64 cols BF16 = 512 × 16-B chunks → 4 chunks/thread.
    // False predicates zero-fill (cp.async src_bytes=0), so M/N/K edges
    // contribute exact +0.0f — identical to the pipelined kernel's
    // zero-fill fallback.
    #define DM16_LOAD(stage, kb) do { \
        { \
            unsigned int ar = threadIdx.x >> 3; \
            unsigned int ac = (threadIdx.x & 7) << 3; \
            unsigned int gc = (kb) + ac; \
            dm16_cp_async_pred_16(&smem_A[(stage)][ar][ac], \
                &A[(unsigned long long)ar * K + gc], \
                (ar < M) && (gc + 8 <= K)); \
        } \
        _Pragma("unroll") \
        for (int rnd = 0; rnd < 4; rnd++) { \
            unsigned int ch = (threadIdx.x << 2) + rnd; \
            unsigned int br = ch >> 3; \
            unsigned int bo = (ch & 7) << 3; \
            unsigned int gn = cta_n + br; \
            unsigned int gk = (kb) + bo; \
            dm16_cp_async_pred_16(&smem_B[(stage)][br][bo], \
                &B[(unsigned long long)gn * K + gk], \
                (gn < N) && (gk + 8 <= K)); \
        } \
    } while (0)

    // Stage compute: 4 k16 sub-chunks × 2 n8 tiles = 8 MMAs per warp,
    // ascending K — the same chain order as dense_gemm_bf16_pipelined's
    // dm_mma_kstep, so accumulation is bit-identical.
    #define DM16_COMPUTE(stage) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(stage)]; \
        const unsigned short* sB = (const unsigned short*)smem_B[(stage)]; \
        _Pragma("unroll") \
        for (int kc = 0; kc < DM16_K_STEP / 16; kc++) { \
            unsigned int k_off = kc * 16; \
            unsigned int c0 = k_off + tid * 2; \
            unsigned int c1 = k_off + tid * 2 + 8; \
            unsigned int a0 = *(const unsigned int*)&sA[group_id * DM16_STRIDE + c0]; \
            unsigned int a1 = *(const unsigned int*)&sA[(group_id + 8) * DM16_STRIDE + c0]; \
            unsigned int a2 = *(const unsigned int*)&sA[group_id * DM16_STRIDE + c1]; \
            unsigned int a3 = *(const unsigned int*)&sA[(group_id + 8) * DM16_STRIDE + c1]; \
            _Pragma("unroll") \
            for (int nt = 0; nt < 2; nt++) { \
                unsigned int nc = warp_id * 16 + nt * 8 + group_id; \
                unsigned int b0 = *(const unsigned int*)&sB[nc * DM16_STRIDE + c0]; \
                unsigned int b1 = *(const unsigned int*)&sB[nc * DM16_STRIDE + c1]; \
                asm volatile( \
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                    : "=f"(acc[nt][0]), "=f"(acc[nt][1]), \
                      "=f"(acc[nt][2]), "=f"(acc[nt][3]) \
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), \
                      "r"(b0), "r"(b1), \
                      "f"(acc[nt][0]), "f"(acc[nt][1]), \
                      "f"(acc[nt][2]), "f"(acc[nt][3])); \
            } \
        } \
    } while (0)

    // Prologue: always commit STAGES-1 groups (empty groups are legal),
    // so the in-loop wait_group 2 invariant holds even when num_k < 3.
    #pragma unroll
    for (unsigned int s = 0; s < DM16_STAGES - 1; s++) {
        if (s < num_k) DM16_LOAD(s, s * DM16_K_STEP);
        dm16_cp_async_commit();
    }

    for (unsigned int s = 0; s < num_k; s++) {
        dm16_cp_async_wait_prior_2();   // stage s landed
        __syncthreads();                // …and is visible to all warps; all
                                        // warps also finished stage s-1, so
                                        // its ring slot can be reloaded.
        unsigned int pf = s + DM16_STAGES - 1;
        if (pf < num_k) DM16_LOAD(pf & (DM16_STAGES - 1), pf * DM16_K_STEP);
        dm16_cp_async_commit();         // always commit: keeps 3 groups in flight
        DM16_COMPUTE(s & (DM16_STAGES - 1));
    }

    #undef DM16_LOAD
    #undef DM16_COMPUTE

    // BF16 write-out. acc rows: group_id (0..7) and group_id+8 (8..15).
    #pragma unroll
    for (int nt = 0; nt < 2; nt++) {
        unsigned int c0 = cta_n + warp_id * 16 + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = group_id;
        unsigned int r1 = group_id + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// dense_gemm_bf16_mtile16_n128 — wide-stream variant: N_TILE=128,
// 8 warps (block 256), K_STEP=32, 4-stage ring. Grid ceil(N/128).
//
// Rationale: on LPDDR5x, MANY concurrent 64-row B streams (96+ CTAs)
// thrash DRAM page locality; a 128-row slice per CTA halves the number
// of concurrent streams and doubles each stream's contiguity — the same
// geometry the production pipelined kernel uses (so it inherits its
// DRAM behaviour) while dropping the 128-row A tile (16 rows only, no
// dead cp.asyncs) and deepening the ring 2 → 4 stages.
//
// smem: A 4×16×40×2 = 5120 B + B 4×128×40×2 = 40960 B = 46080 B/CTA →
// 2 CTAs/SM. Same ascending-K m16n8k16 chain → bit-identical output.
//
// A: [M, K] BF16 (M ≤ 16), B: [N, K] BF16, C: [M, N] BF16.
// Grid: (ceil(N/128), 1, 1)  Block: (256, 1, 1)
// ═══════════════════════════════════════════════════════════════════

#define W128_N_TILE 128
#define W128_K_STEP 32
#define W128_STAGES 4
#define W128_STRIDE 40   // 32 + 8 pad elems: 80-B rows, 16-B aligned

extern "C" __global__ void dense_gemm_bf16_mtile16_n128(
    const __nv_bfloat16* __restrict__ A,   // [M, K] BF16, M <= 16
    const __nv_bfloat16* __restrict__ B,   // [N, K] BF16 weights
    __nv_bfloat16* __restrict__ C,         // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * W128_N_TILE;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __align__(16) __nv_bfloat16 smem_A[W128_STAGES][16][W128_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 smem_B[W128_STAGES][W128_N_TILE][W128_STRIDE];

    float acc[2][4];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int num_k = (K + W128_K_STEP - 1) / W128_K_STEP;

    // A = 16 rows × 32 cols = 64 × 16-B chunks (threads 0..63, 1 each).
    // B = 128 rows × 32 cols = 512 chunks → 2 chunks/thread (256 threads).
    #define W128_LOAD(stage, kb) do { \
        if (threadIdx.x < 64) { \
            unsigned int ar = threadIdx.x >> 2; \
            unsigned int ac = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + ac; \
            dm16_cp_async_pred_16(&smem_A[(stage)][ar][ac], \
                &A[(unsigned long long)ar * K + gc], \
                (ar < M) && (gc + 8 <= K)); \
        } \
        _Pragma("unroll") \
        for (int rnd = 0; rnd < 2; rnd++) { \
            unsigned int ch = (threadIdx.x << 1) + rnd; \
            unsigned int br = ch >> 2; \
            unsigned int bo = (ch & 3) << 3; \
            unsigned int gn = cta_n + br; \
            unsigned int gk = (kb) + bo; \
            dm16_cp_async_pred_16(&smem_B[(stage)][br][bo], \
                &B[(unsigned long long)gn * K + gk], \
                (gn < N) && (gk + 8 <= K)); \
        } \
    } while (0)

    // 2 k16 sub-chunks × 2 n8 tiles = 4 MMAs per warp per stage,
    // ascending K — bit-identical accumulate chain.
    #define W128_COMPUTE(stage) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(stage)]; \
        const unsigned short* sB = (const unsigned short*)smem_B[(stage)]; \
        _Pragma("unroll") \
        for (int kc = 0; kc < W128_K_STEP / 16; kc++) { \
            unsigned int k_off = kc * 16; \
            unsigned int c0 = k_off + tid * 2; \
            unsigned int c1 = k_off + tid * 2 + 8; \
            unsigned int a0 = *(const unsigned int*)&sA[group_id * W128_STRIDE + c0]; \
            unsigned int a1 = *(const unsigned int*)&sA[(group_id + 8) * W128_STRIDE + c0]; \
            unsigned int a2 = *(const unsigned int*)&sA[group_id * W128_STRIDE + c1]; \
            unsigned int a3 = *(const unsigned int*)&sA[(group_id + 8) * W128_STRIDE + c1]; \
            _Pragma("unroll") \
            for (int nt = 0; nt < 2; nt++) { \
                unsigned int nc = warp_id * 16 + nt * 8 + group_id; \
                unsigned int b0 = *(const unsigned int*)&sB[nc * W128_STRIDE + c0]; \
                unsigned int b1 = *(const unsigned int*)&sB[nc * W128_STRIDE + c1]; \
                asm volatile( \
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                    : "=f"(acc[nt][0]), "=f"(acc[nt][1]), \
                      "=f"(acc[nt][2]), "=f"(acc[nt][3]) \
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), \
                      "r"(b0), "r"(b1), \
                      "f"(acc[nt][0]), "f"(acc[nt][1]), \
                      "f"(acc[nt][2]), "f"(acc[nt][3])); \
            } \
        } \
    } while (0)

    #pragma unroll
    for (unsigned int s = 0; s < W128_STAGES - 1; s++) {
        if (s < num_k) W128_LOAD(s, s * W128_K_STEP);
        dm16_cp_async_commit();
    }

    for (unsigned int s = 0; s < num_k; s++) {
        dm16_cp_async_wait_prior_2();
        __syncthreads();
        unsigned int pf = s + W128_STAGES - 1;
        if (pf < num_k) W128_LOAD(pf & (W128_STAGES - 1), pf * W128_K_STEP);
        dm16_cp_async_commit();
        W128_COMPUTE(s & (W128_STAGES - 1));
    }

    #undef W128_LOAD
    #undef W128_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 2; nt++) {
        unsigned int c0 = cta_n + warp_id * 16 + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = group_id;
        unsigned int r1 = group_id + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc[nt][3]);
    }
}
