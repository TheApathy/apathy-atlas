// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W3A16 GEMM — 3-bit-weight clone of `w4a16_gemm_t_m32_n64` for the
// DFlash K=γ+1 verify FFN on W3-gated layers (ATLAS_FFN_W3_LAYERS).
//
// W3 FORMAT (v1 — must match local/tools/repack_w3.py and w3a16_gemv.cu):
//   * W3_LUT[8] = {0, 1, 2, 4, -0, -1, -2, -4} (e2m1-subset, sign bit 2)
//   * 8 weights -> 3 bytes little-endian (code_i at bits 3i of a 24-bit word)
//   * TRANSPOSED layout (built by the Rust loader from the sidecar):
//       B_packed3 [3*K/8, N_pad64] u8 — row (3j+b) = byte-plane b of octet
//       j (k = 8j..8j+7); column n. N padded to 64 for cp.async alignment.
//       B_scale  [K/16, N_pad64] u8 FP8-E4M3 (same scheme as nvfp4_t).
//   * Dequant: sv = (float)e4m3(scale) * scale2; w = W3_LUT[code] * sv;
//     then the SAME FP8-E4M3 round-trip (cvt.rn.satfinite.e4m3x2) and the
//     SAME m16n8k32 e4m3 MMA + BF16 output stores as the W4 parent — the
//     only numeric difference vs W4 is the requantized weights themselves.
//
// Geometry — identical to w4a16_gemm_t_m32_n64:
//   Grid (ceil(N/64), ceil(M/32), 1)  Block (128, 1, 1) = 4 warps.
//   Per K_STEP_T=32 tile the packed stage is 12 rows x 64 cols (vs 16 rows
//   for W4): threads 0..47 issue the 48 cp.async.16 Bp3 loads, threads
//   64..71 issue the 8 scale loads (disjoint ranges, same load pattern).
//
// SMEM: A 2x32x40x2 = 5120B, Bp3 2x12x80 = 1920B, Bs 2x2x80 = 320B,
//       B_fp8 64x32 = 2048B, LUT 32B  ≈ 9.4 KB (vs 10.1 KB for W4).

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define M_TILE_W3 32
#define N_TILE_W3 64
#define K_STEP_W3 32
#define PAD_T_W3 8      // (32+8)*2 = 80B rows, 16B-aligned for cp.async
#define BP_PAD_W3 16    // 64+16 = 80B rows, 16B-aligned
#define GROUP_SIZE_W3 16

__device__ __constant__ float W3_LUT_GEMM[8] = {
    0.0f, 1.0f, 2.0f, 4.0f,
    -0.0f, -1.0f, -2.0f, -4.0f
};

// cp.async helpers (same as w4a16_gemm.cu)
__device__ __forceinline__ void w3_cp_async_pred_16(void* dst_smem, const void* src_gmem, bool pred) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}
__device__ __forceinline__ void w3_cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}
__device__ __forceinline__ void w3_cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;");
}

// Convert 4 BF16 values from smem to packed uint32 of 4 E4M3 values
// (verbatim from w4a16_gemm.cu — keeps the A-side round-trip identical).
__device__ __forceinline__ unsigned int w3_bf16x4_to_e4m3x4(const unsigned short* src) {
    unsigned int p0 = *(const unsigned int*)src;
    unsigned int p1 = *(const unsigned int*)(src + 2);
    unsigned short bf0 = (unsigned short)(p0 & 0xFFFFu);
    unsigned short bf1 = (unsigned short)(p0 >> 16);
    unsigned short bf2 = (unsigned short)(p1 & 0xFFFFu);
    unsigned short bf3 = (unsigned short)(p1 >> 16);
    float f0, f1, f2, f3;
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f0) : "h"(bf0));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f1) : "h"(bf1));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f2) : "h"(bf2));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f3) : "h"(bf3));
    unsigned short h0, h1;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h0) : "f"(f1), "f"(f0));
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h1) : "f"(f3), "f"(f2));
    return ((unsigned int)h1 << 16) | (unsigned int)h0;
}

// Caller contract mirrors `w4a16_gemm_t_m32_n64` (same arg list; B ptrs are
// the W3 transposed sidecar buffers). Accepts any M ≤ 32 per M-tile row;
// output bounds-check discards padding rows.
extern "C" __global__ void w3a16_gemm_t_m32_n64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed3,   // [3*K/8, ldb]
    const unsigned char* __restrict__ B_scale,     // [K/16, ldb]
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int ldb                               // padded row stride (N_pad64)
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_W3;
    const unsigned int cta_m = blockIdx.y * M_TILE_W3;
    const unsigned int warp_id = threadIdx.x / 32;  // 0..3
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int group_id = lane_id >> 2;     // 0..7
    const unsigned int tid = lane_id & 3;           // 0..3

    __shared__ __nv_bfloat16 smem_A[2][M_TILE_W3][K_STEP_W3 + PAD_T_W3];          // 5120B
    __shared__ unsigned char smem_Bp3[2][(K_STEP_W3 / 8) * 3][N_TILE_W3 + BP_PAD_W3]; // 1920B
    __shared__ unsigned char smem_Bs[2][K_STEP_W3 / GROUP_SIZE_W3][N_TILE_W3 + BP_PAD_W3]; // 320B
    __shared__ unsigned char smem_B_fp8[N_TILE_W3][K_STEP_W3];                     // 2048B
    __shared__ float smem_LUT[8];

    if (threadIdx.x < 8) smem_LUT[threadIdx.x] = W3_LUT_GEMM[threadIdx.x];

    // Per-warp accumulators: [m_frag 0..1][n_subtile 0..1][4 fp32] — same
    // as the W4 parent.
    float acc[2][2][4];
    #pragma unroll
    for (int mf = 0; mf < 2; mf++)
        #pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            acc[mf][sub][0] = 0.0f; acc[mf][sub][1] = 0.0f;
            acc[mf][sub][2] = 0.0f; acc[mf][sub][3] = 0.0f;
        }

    const unsigned int a_stride = K_STEP_W3 + PAD_T_W3;

    #define W3_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row = threadIdx.x >> 2;        /* 0..31 */ \
            unsigned int a_col = (threadIdx.x & 3) << 3;  /* 0/8/16/24 */ \
            unsigned int gc = (kb) + a_col; \
            unsigned int gr = cta_m + a_row; \
            w3_cp_async_pred_16(&smem_A[(buf)][a_row][a_col], \
                &A[(unsigned long long)gr * K + gc], \
                (gr < M) && (gc + 7 < K)); \
        } \
        if (threadIdx.x < 48) { \
            /* 12 byte-plane rows x 64 cols: kp3 = local row (0..11), */ \
            /* global row = (kb/8)*3 + kp3. Octet j = kp3/3 covers */ \
            /* k = kb + 8j .. kb + 8j+7. */ \
            unsigned int kp3 = threadIdx.x >> 2;         /* 0..11 */ \
            unsigned int ns = (threadIdx.x & 3) << 4;    /* 0/16/32/48 */ \
            unsigned int gk0 = (kb) + ((kp3 / 3) << 3); \
            unsigned int gns = cta_n + ns; \
            w3_cp_async_pred_16(&smem_Bp3[(buf)][kp3][ns], \
                &B_packed3[(unsigned long long)(((kb) >> 3) * 3 + kp3) * ldb + gns], \
                (gk0 + 7 < K) && (gns + 15 < ldb)); \
        } else if (threadIdx.x >= 64 && threadIdx.x < 64 + 8) { \
            unsigned int kp = threadIdx.x - 64;          /* 0..7 */ \
            unsigned int sg_row = kp >> 2;               /* 0..1 */ \
            unsigned int sg_ns = (kp & 3) << 4;          /* 0/16/32/48 */ \
            unsigned int sg = (kb) / GROUP_SIZE_W3 + sg_row; \
            w3_cp_async_pred_16(&smem_Bs[(buf)][sg_row][sg_ns], \
                &B_scale[(unsigned long long)sg * ldb + cta_n + sg_ns], \
                (cta_n + sg_ns + 15 < ldb)); \
        } \
    } while(0)

    // Dequant: 2 threads per col; thread half h handles K-half h (values
    // 16h..16h+15 = octets 2h, 2h+1) with scale group h. Writes the SAME
    // smem_B_fp8[n][k] layout as the W4 parent via the SAME
    // cvt.rn.satfinite.e4m3x2 round-trip.
    #define W3_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x >> 1;            /* 0..63 */ \
        unsigned int half = threadIdx.x & 1;             /* 0..1  */ \
        unsigned char sb = smem_Bs[(buf)][half][my_n]; \
        __nv_fp8_e4m3 f; \
        *(unsigned char*)&f = sb; \
        float sv = (float)f * scale2; \
        _Pragma("unroll") \
        for (unsigned int j = half * 2; j < half * 2 + 2; j++) { \
            unsigned int u24 = (unsigned int)smem_Bp3[(buf)][j * 3][my_n] \
                             | ((unsigned int)smem_Bp3[(buf)][j * 3 + 1][my_n] << 8) \
                             | ((unsigned int)smem_Bp3[(buf)][j * 3 + 2][my_n] << 16); \
            _Pragma("unroll") \
            for (unsigned int p = 0; p < 4; p++) { \
                float lo = smem_LUT[(u24 >> (6 * p)) & 7] * sv; \
                float hi = smem_LUT[(u24 >> (6 * p + 3)) & 7] * sv; \
                unsigned short fp8_pair; \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
                *(unsigned short*)&smem_B_fp8[my_n][j * 8 + p * 2] = fp8_pair; \
            } \
        } \
    } while(0)

    // FP8 MMA — verbatim structure from the W4 parent: 4 warps x
    // (2 M-frags x 2 N-subtiles), m16n8k32 e4m3.
    #define W3_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = group_id; \
        unsigned int fr1 = fr0 + 8; \
        unsigned int a0 = w3_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = w3_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = w3_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = w3_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        unsigned int b0 = w3_bf16x4_to_e4m3x4(&sA[(fr0 + 16) * a_stride + tid * 4]); \
        unsigned int b1 = w3_bf16x4_to_e4m3x4(&sA[(fr1 + 16) * a_stride + tid * 4]); \
        unsigned int b2 = w3_bf16x4_to_e4m3x4(&sA[(fr0 + 16) * a_stride + 16 + tid * 4]); \
        unsigned int b3 = w3_bf16x4_to_e4m3x4(&sA[(fr1 + 16) * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int sub = 0; sub < 2; sub++) { \
            unsigned int nt = warp_id * 2 + sub; \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int v0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int v1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[0][sub][0]),"=f"(acc[0][sub][1]),"=f"(acc[0][sub][2]),"=f"(acc[0][sub][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(v0),"r"(v1), \
                 "f"(acc[0][sub][0]),"f"(acc[0][sub][1]),"f"(acc[0][sub][2]),"f"(acc[0][sub][3])); \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[1][sub][0]),"=f"(acc[1][sub][1]),"=f"(acc[1][sub][2]),"=f"(acc[1][sub][3]) \
                :"r"(b0),"r"(b1),"r"(b2),"r"(b3),"r"(v0),"r"(v1), \
                 "f"(acc[1][sub][0]),"f"(acc[1][sub][1]),"f"(acc[1][sub][2]),"f"(acc[1][sub][3])); \
        } \
    } while(0)

    W3_ISSUE_LOADS(0, 0);
    w3_cp_async_commit();
    w3_cp_async_wait_all();
    __syncthreads();
    W3_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_W3; k_base < K; k_base += K_STEP_W3) {
        int nxt = 1 - cur;
        W3_ISSUE_LOADS(nxt, k_base);
        w3_cp_async_commit();
        W3_COMPUTE_MMA(cur);
        w3_cp_async_wait_all();
        __syncthreads();
        W3_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    W3_COMPUTE_MMA(cur);

    #undef W3_ISSUE_LOADS
    #undef W3_DEQUANT
    #undef W3_COMPUTE_MMA

    // Output: each warp writes 2 N sub-tiles x 4 row groups, bounds-checked
    // against M — identical to the W4 parent.
    #pragma unroll
    for (int sub = 0; sub < 2; sub++) {
        unsigned int nt = warp_id * 2 + sub;
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + group_id;
        unsigned int r1 = r0 + 8;
        unsigned int r2 = r0 + 16;
        unsigned int r3 = r0 + 24;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[0][sub][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[0][sub][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[0][sub][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[0][sub][3]);
        if (r2 < M && c0 < N) C[r2*N+c0] = __float2bfloat16(acc[1][sub][0]);
        if (r2 < M && c1 < N) C[r2*N+c1] = __float2bfloat16(acc[1][sub][1]);
        if (r3 < M && c0 < N) C[r3*N+c0] = __float2bfloat16(acc[1][sub][2]);
        if (r3 < M && c1 < N) C[r3*N+c1] = __float2bfloat16(acc[1][sub][3]);
    }
}
