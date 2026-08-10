// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A16 GEMM — 35B model shadow.
//
// Optimizations:
// - w4a16_gemm_t: cp.async 2-stage double-buffered pipeline (overlaps next tile
//   loads with current tile compute), prmt BF16 packing, BP_PAD bank conflict fix
// - Vectorized uint4 (128-bit) B_packed loads
// - Both-nibble extraction from packed bytes
// - N_TILE=128 for reduced A bandwidth

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define M_TILE 64
#define N_TILE_SM 64
#define N_TILE_LG 128
#define K_STEP 16
#define K_STEP_T 32
#define PAD 2
#define PAD_T 8        // cp.async needs 16-byte aligned rows: (32+8)*2=80, 80%16=0
#define BP_PAD 16      // smem_Bp row padding: stride 144 is 16-byte aligned, eliminates 4-way bank conflict
#define B_PAD 2        // BF16 padding for bank-conflict-free smem_B_bf16 (stride 17 coprime with 32)
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Original layout w4a16_gemm: unchanged, N_TILE=64
extern "C" __global__ void w4a16_gemm(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int cta_n = blockIdx.x * N_TILE_SM;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE_SM + PAD];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE_SM + PAD;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        {
            const unsigned int ept = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < ept; i++) {
                unsigned int idx = threadIdx.x * ept + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int gr = cta_m + row;
                unsigned int gc = k_base + col;
                smem_A[row][col] = (gr < M && gc < K) ? A[gr * K + gc] : __float2bfloat16(0.0f);
            }
        }
        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int k = idx / N_TILE_SM;
                unsigned int n = idx % N_TILE_SM;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;
                if (gk < K && gn < N) {
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_packed[(unsigned long long)gn * half_K + k_pair];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);
                    unsigned int sg = gk / GROUP_SIZE;
                    unsigned char sb = B_scale[(unsigned long long)gn * num_groups + sg];
                    __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
                    smem_B[k][n] = __float2bfloat16(E2M1_LUT[nibble] * (float)fp8 * scale2);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        const unsigned short* sA = (const unsigned short*)smem_A;
        const unsigned short* sB = (const unsigned short*)smem_B;
        unsigned int fr0 = warp_m_offset + group_id;
        unsigned int fr1 = fr0 + 8;
        unsigned int fc0 = tid * 2, fc1 = fc0 + 8;
        unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0];
        unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0];
        unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1];
        unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1];
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            unsigned int nc = nt * 8 + group_id;
            unsigned int k0 = tid * 2, k1 = k0 + 8;
            unsigned int b0 = ((unsigned int)sB[(k0+1)*b_stride+nc]<<16) | (unsigned int)sB[k0*b_stride+nc];
            unsigned int b1 = ((unsigned int)sB[(k1+1)*b_stride+nc]<<16) | (unsigned int)sB[k1*b_stride+nc];
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3])
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]));
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// cp.async 2-stage double-buffered transposed GEMM.
//
// Overlaps global→smem loads for tile N+1 with MMA compute on tile N.
// All loads (A, Bp, Bs) use cp.async.16 for register-free transfers.
//
// smem (double-buffered):
//   A:  2 × 64 × 40 × 2B = 10240B
//   Bp: 2 × 16 × 144     =  4608B
//   Bs: 2 × 2  × 144     =   576B
//   LUT: 64B
//   Total: ~15.5KB → register-limited at ~6 CTAs/SM (unchanged)
//
// B_packed[K/2, N], B_scale[K/GROUP_SIZE, N].
// ═══════════════════════════════════════════════════════════════════

// cp.async helpers (SM80+)
__device__ __forceinline__ void cp_async_pred_16(void* dst_smem, const void* src_gmem, bool pred) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;");
}

__device__ __forceinline__ unsigned int pack_bf16_pair(float lo, float hi) {
    unsigned int result;
    asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(result)
        : "r"(__float_as_uint(lo)), "r"(__float_as_uint(hi)));
    return result;
}

// ═══════════════════════════════════════════════════════════════════
// FP8-MMA transposed dense GEMM.
//
// Dequant B to FP8 E4M3 (not BF16). Convert A from BF16→FP8 in
// registers. Use mma.sync.m16n8k32.e4m3.e4m3 — processes full K=32
// per instruction (2x fewer MMA instructions vs BF16 m16n8k16).
//
// Pipeline: load[nxt] || MMA[cur] → wait → dequant[nxt] → sync
//
// smem: A 2×64×40×2=10240B, Bp 2×16×144=4608B, Bs 2×2×144=576B,
//       B_fp8 128×32=4096B, LUT 64B = ~19.6KB
// ═══════════════════════════════════════════════════════════════════

// Convert 4 BF16 values from smem to packed uint32 of 4 E4M3 values
__device__ __forceinline__ unsigned int bf16x4_to_e4m3x4(const unsigned short* src) {
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

extern "C" __global__ void w4a16_gemm_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_B_fp8[N_TILE_LG][K_STEP_T];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    #define ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[gr * K + gc], (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    // Dequant B: FP4 → FP8 E4M3 (cvt.rn.satfinite.e4m3x2.f32)
    #define DEQUANT_T(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv0; \
            float hi = smem_LUT[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv1; \
            float hi = smem_LUT[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    // FP8 MMA: convert A BF16→E4M3 in registers, single m16n8k32 per N-tile
    #define COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    ISSUE_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    DEQUANT_T(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        ISSUE_LOADS(nxt, k_base);
        cp_async_commit();
        COMPUTE_MMA(cur);
        cp_async_wait_all();
        __syncthreads();
        DEQUANT_T(nxt);
        __syncthreads();
        cur = nxt;
    }

    COMPUTE_MMA(cur);

    #undef ISSUE_LOADS
    #undef DEQUANT_T
    #undef COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pre-dequanted FP8 GEMM (prefill).
//
// B_fp8 is pre-dequanted at load time: NVFP4 → FP8 E4M3 once.
// Eliminates the per-inference DEQUANT phase entirely.
// B_fp8[N, K] layout — each row is one output neuron, K consecutive.
//
// Pipeline: LOAD(A+B_fp8) || COMPUTE_MMA — only 1 sync per K step.
//
// smem: A 2×64×40×2=10240B, B_fp8 2×128×32=8192B = ~18.4KB
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__ void fp8_gemm_t(
    const __nv_bfloat16* __restrict__ A,       // [M, K] BF16
    const unsigned char* __restrict__ B_fp8,   // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,             // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_B[2][N_TILE_LG][K_STEP_T];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // Load A (BF16) + B (FP8, pre-dequanted) via cp.async
    #define FP8_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_B[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_B[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    // FP8 MMA — identical to w4a16_gemm_t COMPUTE_MMA
    #define FP8_COMPUTE(a_buf, b_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(b_buf)][nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]), \
                 "=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3), \
                 "r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]), \
                 "f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    // Prolog: load first tile, wait, no dequant needed
    FP8_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // Main loop: LOAD(nxt) || COMPUTE(cur) → wait → sync
    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FP8_LOADS(nxt, k_base);
        cp_async_commit();
        FP8_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FP8_COMPUTE(cur, cur);

    #undef FP8_LOADS
    #undef FP8_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pre-dequant: NVFP4 [N, K/2] + scales [N, K/GROUP_SIZE] → FP8 [N, K]
//
// One-time conversion at model load. Each thread processes 1 packed
// byte (2 FP4 values) → 2 FP8 E4M3 values.
// Grid: (ceil(N * K/2 / 256), 1, 1)  Block: (256, 1, 1)
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__ void predequant_nvfp4_to_fp8(
    const unsigned char* __restrict__ B_packed,  // [N, K/2]
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE]
    float scale2,
    unsigned char* __restrict__ B_fp8,           // [N, K]
    unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half_K = K / 2;
    unsigned int total = N * half_K;
    if (idx >= total) return;

    unsigned int n = idx / half_K;
    unsigned int k_pair = idx % half_K;
    unsigned int k_even = k_pair * 2;

    unsigned char packed = B_packed[(unsigned long long)n * half_K + k_pair];
    unsigned int group = k_even / GROUP_SIZE;
    unsigned char sb = B_scale[(unsigned long long)n * (K / GROUP_SIZE) + group];
    __nv_fp8_e4m3 fp8_scale;
    *(unsigned char*)&fp8_scale = sb;
    float sv = (float)fp8_scale * scale2;

    float val_lo = E2M1_LUT[packed & 0xF] * sv;
    float val_hi = E2M1_LUT[packed >> 4] * sv;

    unsigned short fp8_pair;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(fp8_pair) : "f"(val_hi), "f"(val_lo));

    *(unsigned short*)&B_fp8[(unsigned long long)n * K + k_even] = fp8_pair;
}

// ═══════════════════════════════════════════════════════════════════
// BF16 → FP8 E4M3 activation conversion.
// Converts [M, K] BF16 activations to [M, K] FP8 E4M3 in-place or
// out-of-place. Grid: (ceil(M*K/2 / 256), 1, 1)  Block: (256, 1, 1)
// Each thread converts 2 BF16 values → 2 FP8 values via cvt.e4m3x2.
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void bf16_to_fp8(
    const __nv_bfloat16* __restrict__ src,   // [M, K] BF16
    unsigned char* __restrict__ dst,          // [M, K] FP8 E4M3
    unsigned int total_elements               // M * K (must be even)
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= total_elements) return;

    unsigned int p = *(const unsigned int*)&src[idx];
    unsigned short bf0 = (unsigned short)(p & 0xFFFFu);
    unsigned short bf1 = (unsigned short)(p >> 16);
    float f0, f1;
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f0) : "h"(bf0));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f1) : "h"(bf1));
    unsigned short fp8_pair;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(fp8_pair) : "f"(f1), "f"(f0));
    *(unsigned short*)&dst[idx] = fp8_pair;
}

// ═══════════════════════════════════════════════════════════════════
// FP8×FP8 GEMM: A [M, K] FP8 E4M3 × B [N, K] FP8 E4M3 → C [M, N] BF16
//
// Both A and B are pre-converted to FP8. No BF16→FP8 conversion in
// the inner loop — pure cp.async loads + FP8 MMA.
// Same tiling as fp8_gemm_t: M_TILE=64, N_TILE=128, K_STEP=32.
// A smem is FP8 (half the size of BF16 variant), no PAD needed.
// Grid: (ceil(N/128), ceil(M/64))  Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════
#define A_FP8_STRIDE 32  // K_STEP_T = 32 bytes per row for FP8

extern "C" __global__ void fp8_fp8_gemm_t(
    const unsigned char* __restrict__ A_fp8,  // [M, K] FP8 E4M3
    const unsigned char* __restrict__ B_fp8,  // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,            // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // A smem: FP8 [64][32] = 2 KB per buffer (vs 5 KB BF16)
    __shared__ unsigned char smem_Af[2][M_TILE][A_FP8_STRIDE];
    __shared__ unsigned char smem_Bf[2][N_TILE_LG][K_STEP_T];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    // Load A (FP8) + B (FP8) via cp.async — both 1 byte per element
    #define FF_LOADS(buf, kb) do { \
        { \
            /* 128 threads load 64 rows × 32 bytes: each thread loads 16 bytes */ \
            unsigned int a_row_base = threadIdx.x >> 1; \
            unsigned int a_col = (threadIdx.x & 1) << 4; \
            unsigned int gc = (kb) + a_col; \
            unsigned int row = a_row_base; \
            unsigned int gr = cta_m + row; \
            cp_async_pred_16(&smem_Af[(buf)][row][a_col], \
                &A_fp8[(unsigned long long)gr * K + gc], \
                (gr < M) && (gc + 15 < K)); \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    // FP8×FP8 MMA — no conversion needed, read A directly as FP8
    #define FF_COMPUTE(a_buf, b_buf) do { \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        /* A fragments: 4 bytes = 4 FP8 elements per register, need 8 regs (m16×k32) */ \
        unsigned int a0 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][4 * tid]; \
        unsigned int a1 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][4 * tid]; \
        unsigned int a2 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][16 + 4 * tid]; \
        unsigned int a3 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][16 + 4 * tid]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]), \
                 "=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3), \
                 "r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]), \
                 "f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    // Prolog: load first tile, wait
    FF_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // Main loop: LOAD(nxt) || COMPUTE(cur) → wait → sync
    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FF_LOADS(nxt, k_base);
        cp_async_commit();
        FF_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FF_COMPUTE(cur, cur);

    #undef FF_LOADS
    #undef FF_COMPUTE

    // Write results
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// K64 FP8-MMA transposed dense GEMM — halves outer K-loop vs K32.
//
// Same algorithm as w4a16_gemm_t but K_STEP_T64=64: 32 outer iterations
// instead of 64 for K=2048. Two m16n8k32 MMAs per N-tile per step.
// Reduces loop overhead and better amortizes DMA startup cost.
//
// K must be divisible by 64.
//
// smem: A 2×64×72×2=18432B, Bp 2×32×144=9216B, Bs 2×4×144=1152B,
//       B_fp8 128×80=10240B, LUT 64B = ~38.4KB
// ═══════════════════════════════════════════════════════════════════
#define K_STEP_T64 64
#define PAD_T64    8   // (64+8)*2=144, 144%16=0 ✓

extern "C" __global__ void w4a16_gemm_t_k64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // B_fp8 row stride 80 = K64+16: avoids 4-way bank conflicts.
    __shared__ __nv_bfloat16 smem_A_k64[2][M_TILE][K_STEP_T64 + PAD_T64];
    __shared__ unsigned char smem_Bp_k64[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs_k64[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_B_fp8_k64[N_TILE_LG][K_STEP_T64 + 16];
    __shared__ float smem_LUT_k64[16];

    if (threadIdx.x < 16) smem_LUT_k64[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int ast64 = K_STEP_T64 + PAD_T64;

    // A: 4 rounds × 16 rows = 64 rows (M_TILE); each thread: 8 BF16 = 16 bytes.
    // Bp: 2 rounds × 16 rows = 32 rows (K64/2); each thread: 16 bytes per ns chunk.
    // Bs: inline with Bp when kp_cur < K_STEP_T64/GROUP_SIZE (4 scale groups).
    #define K64_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 3; \
            unsigned int a_col = (threadIdx.x & 7) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = rnd * 16 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A_k64[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gns = cta_n + ns; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int kp_cur = rnd * 16 + kp; \
                unsigned int gke = (kb) + (kp_cur << 1); \
                cp_async_pred_16(&smem_Bp_k64[(buf)][kp_cur][ns], \
                    &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 <= K) && (gns + 15 < N)); \
                if (kp_cur < K_STEP_T64 / GROUP_SIZE) { \
                    unsigned int sg = (kb) / GROUP_SIZE + kp_cur; \
                    cp_async_pred_16(&smem_Bs_k64[(buf)][kp_cur][ns], \
                        &B_scale[(unsigned long long)sg * N + gns], \
                        (gns + 15 < N)); \
                } \
            } \
        } \
    } while(0)

    // 4 scale groups, 32 dequant iters: sv0→K{0..15}, sv1→K{16..31},
    // sv2→K{32..47}, sv3→K{48..63}.
    #define K64_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        __nv_fp8_e4m3 f0, f1, f2, f3; \
        *(unsigned char*)&f0 = smem_Bs_k64[(buf)][0][my_n]; \
        *(unsigned char*)&f1 = smem_Bs_k64[(buf)][1][my_n]; \
        *(unsigned char*)&f2 = smem_Bs_k64[(buf)][2][my_n]; \
        *(unsigned char*)&f3 = smem_Bs_k64[(buf)][3][my_n]; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        float sv2 = (float)f2 * scale2, sv3 = (float)f3 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv0; \
            float hi = smem_LUT_k64[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv1; \
            float hi = smem_LUT_k64[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 16; kp < 24; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv2; \
            float hi = smem_LUT_k64[packed >> 4] * sv2; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 24; kp < 32; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv3; \
            float hi = smem_LUT_k64[packed >> 4] * sv3; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    // Two m16n8k32 MMA calls per N-tile: first covers K=0..31, second K=32..63.
    #define K64_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A_k64[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_k64[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_k64[nc][16 + 4 * tid]; \
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
        unsigned int a4 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 32 + tid * 4]); \
        unsigned int a5 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 32 + tid * 4]); \
        unsigned int a6 = bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 48 + tid * 4]); \
        unsigned int a7 = bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 48 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_k64[nc][32 + 4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_k64[nc][48 + 4 * tid]; \
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a4),"r"(a5),"r"(a6),"r"(a7),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    K64_ISSUE_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    K64_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T64; k_base < K; k_base += K_STEP_T64) {
        int nxt = 1 - cur;
        K64_ISSUE_LOADS(nxt, k_base);
        cp_async_commit();
        K64_COMPUTE_MMA(cur);
        cp_async_wait_all();
        __syncthreads();
        K64_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    K64_COMPUTE_MMA(cur);

    #undef K64_ISSUE_LOADS
    #undef K64_DEQUANT
    #undef K64_COMPUTE_MMA
    #undef K_STEP_T64
    #undef PAD_T64

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// M128 variant: 2 consecutive 64-row M-chunks per CTA.
//
// For large-M prefill (e.g. ISL=1016, N=12288):
//   M_TILE=64: grid=(96,16,1)=1536 blocks, 16 weight re-reads  → 227MB B DRAM
//   M_TILE2=128: grid=(96,8,1)=768 blocks, 8 weight re-reads   → 114MB B DRAM
//
// SMEM: A 2×128×40×2=20480B, Bp 2×16×144=4608B, Bs 2×2×144=576B,
//       B_fp8 128×32=4096B, LUT 64B ≈ 29.8KB → 3 blocks/SM.
//
// For qkvz (K=2048,N=12288): ~2× speedup at ISL>128 vs w4a16_gemm_t.
// ═══════════════════════════════════════════════════════════════════

extern "C" __global__
__launch_bounds__(128, 3)
void w4a16_gemm_t_m128(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n  = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m  = blockIdx.y * (2 * M_TILE);  // base row for this block
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // A is 2× larger (128 rows instead of 64); B/LUT/dequant identical to w4a16_gemm_t.
    __shared__ __nv_bfloat16 smem_A[2][2 * M_TILE][K_STEP_T + PAD_T];   // 20480 B
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD]; // 4608 B
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD]; // 576 B
    __shared__ unsigned char smem_B_fp8[N_TILE_LG][K_STEP_T];             // 4096 B
    __shared__ float smem_LUT[16];                                         //   64 B
    // Total ≈ 29.8 KB → 3 blocks/SM

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    // Two sets of accumulators: chunk0 = rows [cta_m..cta_m+63],
    //                           chunk1 = rows [cta_m+64..cta_m+127].
    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0] = 0.f; acc0[i][1] = 0.f; acc0[i][2] = 0.f; acc0[i][3] = 0.f;
        acc1[i][0] = 0.f; acc1[i][1] = 0.f; acc1[i][2] = 0.f; acc1[i][3] = 0.f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // Load A (4 rounds → 128 rows) + B (same as w4a16_gemm_t).
    #define M128_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col      = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp  = threadIdx.x >> 3; \
            unsigned int ns  = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    // Dequant B tile: identical to w4a16_gemm_t's DEQUANT_T.
    #define M128_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv0; \
            float hi = smem_LUT[packed >> 4]  * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv1; \
            float hi = smem_LUT[packed >> 4]  * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    // MMA for both M-chunks; B tile (smem_B_fp8) loaded once, reused by both.
    #define M128_COMPUTE(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0, fr1, a0, a1, a2, a3; \
        /* Chunk 0: smem rows 0..63 */ \
        fr0 = warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc0[nt][0]),"=f"(acc0[nt][1]),"=f"(acc0[nt][2]),"=f"(acc0[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc0[nt][0]),"f"(acc0[nt][1]),"f"(acc0[nt][2]),"f"(acc0[nt][3])); \
        } \
        /* Chunk 1: smem rows 64..127 (offset M_TILE=64) */ \
        fr0 = M_TILE + warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc1[nt][0]),"=f"(acc1[nt][1]),"=f"(acc1[nt][2]),"=f"(acc1[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc1[nt][0]),"f"(acc1[nt][1]),"f"(acc1[nt][2]),"f"(acc1[nt][3])); \
        } \
    } while(0)

    // Pipeline: same double-buffer structure as w4a16_gemm_t.
    M128_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    M128_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        M128_LOADS(nxt, k_base);
        cp_async_commit();
        M128_COMPUTE(cur);
        cp_async_wait_all();
        __syncthreads();
        M128_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    M128_COMPUTE(cur);

    #undef M128_LOADS
    #undef M128_DEQUANT
    #undef M128_COMPUTE

    // Write chunk 0: rows [cta_m..cta_m+63]
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc0[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc0[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc0[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc0[nt][3]);
    }
    // Write chunk 1: rows [cta_m+64..cta_m+127]
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + M_TILE + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc1[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc1[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc1[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc1[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// M128 variant of fp8_gemm_t: BF16 A × FP8 B, 2 M-chunks per CTA.
//
// For out_proj (K=2048, N=2048) and paged Q/K/V: halves the number of
// times B is read from DRAM (8 m-tile groups vs 16 at M=1015).
//
// SMEM: A 2×128×40×2=20480B, B 2×128×32=8192B ≈ 28.7KB → 3 blocks/SM.
// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(128, 3)
void fp8_gemm_t_m128(
    const __nv_bfloat16* __restrict__ A,       // [M, K] BF16
    const unsigned char* __restrict__ B_fp8,   // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,             // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][2 * M_TILE][K_STEP_T + PAD_T];  // 20480 B
    __shared__ unsigned char  smem_B[2][N_TILE_LG][K_STEP_T];            //  8192 B

    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0] = 0.f; acc0[i][1] = 0.f; acc0[i][2] = 0.f; acc0[i][3] = 0.f;
        acc1[i][0] = 0.f; acc1[i][1] = 0.f; acc1[i][2] = 0.f; acc1[i][3] = 0.f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // Load A (BF16, 4 rounds → 128 rows) + B (FP8, same as fp8_gemm_t).
    #define FGM128_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_B[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_B[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    // FP8 MMA for both M-chunks; B tile loaded once and reused.
    #define FGM128_COMPUTE(a_buf, b_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0, fr1, a0, a1, a2, a3; \
        /* Chunk 0: smem rows 0..63 */ \
        fr0 = warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(b_buf)][nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc0[nt][0]),"=f"(acc0[nt][1]),"=f"(acc0[nt][2]),"=f"(acc0[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc0[nt][0]),"f"(acc0[nt][1]),"f"(acc0[nt][2]),"f"(acc0[nt][3])); \
        } \
        /* Chunk 1: smem rows 64..127 */ \
        fr0 = M_TILE + warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B[(b_buf)][nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc1[nt][0]),"=f"(acc1[nt][1]),"=f"(acc1[nt][2]),"=f"(acc1[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc1[nt][0]),"f"(acc1[nt][1]),"f"(acc1[nt][2]),"f"(acc1[nt][3])); \
        } \
    } while(0)

    FGM128_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FGM128_LOADS(nxt, k_base);
        cp_async_commit();
        FGM128_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FGM128_COMPUTE(cur, cur);

    #undef FGM128_LOADS
    #undef FGM128_COMPUTE

    // Write chunk 0: rows [cta_m..cta_m+63]
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc0[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc0[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc0[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc0[nt][3]);
    }
    // Write chunk 1: rows [cta_m+64..cta_m+127]
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + M_TILE + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc1[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc1[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc1[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc1[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// M128 variant of fp8_fp8_gemm_t: FP8 A × FP8 B, 2 M-chunks per CTA.
//
// For Q/K/V projections in cache-skip prefill path (FP8 activations):
// halves B re-reads. Uses 3 blocks/SM (not 6) to avoid register spilling:
// dual acc0+acc1 need ~145 regs/thread; 3 blocks allows 170 regs/thread.
//
// SMEM: Af 2×128×32=8192B, Bf 2×128×32=8192B ≈ 16KB, 3 blocks → 48KB/SM.
// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(128, 3)
void fp8_fp8_gemm_t_m128(
    const unsigned char* __restrict__ A_fp8,  // [M, K] FP8 E4M3
    const unsigned char* __restrict__ B_fp8,  // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,            // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ unsigned char smem_Af[2][2 * M_TILE][A_FP8_STRIDE];  //  8192 B
    __shared__ unsigned char smem_Bf[2][N_TILE_LG][K_STEP_T];        //  8192 B

    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0] = 0.f; acc0[i][1] = 0.f; acc0[i][2] = 0.f; acc0[i][3] = 0.f;
        acc1[i][0] = 0.f; acc1[i][1] = 0.f; acc1[i][2] = 0.f; acc1[i][3] = 0.f;
    }

    // Load A (FP8, 2 rounds → 128 rows) + B (FP8, same as fp8_fp8_gemm_t).
    #define FFM128_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 1; \
            unsigned int a_col = (threadIdx.x & 1) << 4; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 64) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                cp_async_pred_16(&smem_Af[(buf)][row][a_col], \
                    &A_fp8[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 15 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    // FP8×FP8 MMA for both M-chunks; B loaded once, reused by both.
    #define FFM128_COMPUTE(a_buf, b_buf) do { \
        unsigned int fr0, fr1, a0, a1, a2, a3; \
        /* Chunk 0: smem rows 0..63 */ \
        fr0 = warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][4 * tid]; \
        a1 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][4 * tid]; \
        a2 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][16 + 4 * tid]; \
        a3 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][16 + 4 * tid]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc0[nt][0]),"=f"(acc0[nt][1]),"=f"(acc0[nt][2]),"=f"(acc0[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc0[nt][0]),"f"(acc0[nt][1]),"f"(acc0[nt][2]),"f"(acc0[nt][3])); \
        } \
        /* Chunk 1: smem rows 64..127 */ \
        fr0 = M_TILE + warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][4 * tid]; \
        a1 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][4 * tid]; \
        a2 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][16 + 4 * tid]; \
        a3 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][16 + 4 * tid]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc1[nt][0]),"=f"(acc1[nt][1]),"=f"(acc1[nt][2]),"=f"(acc1[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc1[nt][0]),"f"(acc1[nt][1]),"f"(acc1[nt][2]),"f"(acc1[nt][3])); \
        } \
    } while(0)

    FFM128_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FFM128_LOADS(nxt, k_base);
        cp_async_commit();
        FFM128_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;
    }
    FFM128_COMPUTE(cur, cur);

    #undef FFM128_LOADS
    #undef FFM128_COMPUTE

    // Write chunk 0: rows [cta_m..cta_m+63]
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc0[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc0[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc0[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc0[nt][3]);
    }
    // Write chunk 1: rows [cta_m+64..cta_m+127]
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + M_TILE + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc1[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc1[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc1[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc1[nt][3]);
    }
}

// ── Ported verbatim from kernels/gb10/laguna-s-2.1/nvfp4/w4a16_gemm.cu:1706-1836.
// This model-specific override REPLACES the common w4a16_gemm.cu, so every
// caller that gates on fp8_gemm_row_scaled_mtile8 (impl_a3::lm_head's
// serial-mirror path, dflash_head's M=γ tail, the γ-row verify lm_head in
// model/trait_impl/decode_b2.rs, and dspark_head's block tail) was silently
// resolving KernelHandle(0) on deepseek-v4-flash and falling back to the
// per-row GEMV loop. Nothing about the kernel is model-specific.
// ═══════════════════════════════════════════════════════════════════

#define M8_N_TILE 64
#define M8_K_STEP 64
#define M8_STAGES 4
#define M8_A_STRIDE 72   // 64 + 8 pad elems: 144B rows, 16B-aligned, kills the
                         // 8-way bank conflict a 128B stride would produce.

__device__ __forceinline__ void cp_async_wait_prior_2() {
    // Wait until at most 2 commit-groups are pending. With the ring below
    // there are always exactly 3 outstanding groups before this call, so
    // this waits precisely for the oldest (the stage about to be computed).
    asm volatile("cp.async.wait_group 2;");
}

extern "C" __global__ void fp8_gemm_t_row_scaled_mtile8(
    const __nv_bfloat16* __restrict__ A,        // [M, K] BF16, M <= 8
    const unsigned char* __restrict__ B_fp8,    // [N, K] FP8 E4M3
    const float* __restrict__ row_scale,        // [N] f32
    __nv_bfloat16* __restrict__ C,              // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * M8_N_TILE;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M8_STAGES][8][M8_A_STRIDE];
    __shared__ unsigned char smem_B[M8_STAGES][M8_N_TILE][M8_K_STEP];

    // Per-lane accumulators: 2 n8 tiles per warp. acc[nt][2..3] hold mma
    // rows 8..15 — dead for M≤8, but the mma writes them regardless.
    float acc[2][4];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int num_k = (K + M8_K_STEP - 1) / M8_K_STEP;

    // Stage load: A = 8 rows × 64 cols BF16 (64 × 16B chunks, threads 0..63),
    // B = 64 rows × 64 B (256 × 16B chunks, 2 per thread). False predicates
    // zero-fill (cp.async src_bytes=0), so M/N/K edges contribute exact 0.
    #define M8_LOAD(stage, kb) do { \
        if (threadIdx.x < 64) { \
            unsigned int ar = threadIdx.x >> 3; \
            unsigned int ac = (threadIdx.x & 7) << 3; \
            unsigned int gc = (kb) + ac; \
            cp_async_pred_16(&smem_A[(stage)][ar][ac], \
                &A[(unsigned long long)ar * K + gc], \
                (ar < M) && (gc + 7 < K)); \
        } \
        _Pragma("unroll") \
        for (int rnd = 0; rnd < 2; rnd++) { \
            unsigned int ch = (threadIdx.x << 1) + rnd; \
            unsigned int br = ch >> 2; \
            unsigned int bo = (ch & 3) << 4; \
            unsigned int gn = cta_n + br; \
            unsigned int gk = (kb) + bo; \
            cp_async_pred_16(&smem_B[(stage)][br][bo], \
                &B_fp8[(unsigned long long)gn * K + gk], \
                (gn < N) && (gk + 15 < K)); \
        } \
    } while (0)

    // Stage compute: 2 k32 chunks × 2 n8 tiles = 4 MMAs per warp.
    // A-fragment upper rows (a1, a3 = rows 8..15) are constant zero.
    #define M8_COMPUTE(stage) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(stage)]; \
        _Pragma("unroll") \
        for (int kc = 0; kc < 2; kc++) { \
            unsigned int kb32 = kc * 32; \
            unsigned int a0 = bf16x4_to_e4m3x4( \
                &sA[group_id * M8_A_STRIDE + kb32 + tid * 4]); \
            unsigned int a2 = bf16x4_to_e4m3x4( \
                &sA[group_id * M8_A_STRIDE + kb32 + 16 + tid * 4]); \
            _Pragma("unroll") \
            for (int nt = 0; nt < 2; nt++) { \
                unsigned int nc = warp_id * 16 + nt * 8 + group_id; \
                unsigned int b0 = *(const unsigned int*)&smem_B[(stage)][nc][kb32 + 4 * tid]; \
                unsigned int b1 = *(const unsigned int*)&smem_B[(stage)][nc][kb32 + 16 + 4 * tid]; \
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                    :"=f"(acc[nt][0]),"=f"(acc[nt][1]), \
                     "=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                    :"r"(a0),"r"(0u),"r"(a2),"r"(0u), \
                     "r"(b0),"r"(b1), \
                     "f"(acc[nt][0]),"f"(acc[nt][1]), \
                     "f"(acc[nt][2]),"f"(acc[nt][3])); \
            } \
        } \
    } while (0)

    // Prologue: always commit STAGES-1 groups (empty groups are legal), so
    // the in-loop wait_group 2 invariant holds even when num_k < STAGES-1.
    #pragma unroll
    for (unsigned int s = 0; s < M8_STAGES - 1; s++) {
        if (s < num_k) M8_LOAD(s, s * M8_K_STEP);
        cp_async_commit();
    }

    for (unsigned int s = 0; s < num_k; s++) {
        cp_async_wait_prior_2();   // stage s landed
        __syncthreads();           // …and is visible to all warps; also all
                                   // warps are done computing stage s-1, so
                                   // its buffer (= (s+3)&3) can be reloaded.
        unsigned int pf = s + M8_STAGES - 1;
        if (pf < num_k) M8_LOAD(pf & (M8_STAGES - 1), pf * M8_K_STEP);
        cp_async_commit();         // always commit: keeps 3 groups in flight
        M8_COMPUTE(s & (M8_STAGES - 1));
    }

    #undef M8_LOAD
    #undef M8_COMPUTE

    // Per-column scale multiply + BF16 write-out. Only mma rows 0..7 are
    // real for M≤8 (acc[nt][2..3] are rows 8..15 — never emitted).
    #pragma unroll
    for (int nt = 0; nt < 2; nt++) {
        unsigned int c0 = cta_n + warp_id * 16 + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = group_id;   // 0..7
        float sc0 = (c0 < N) ? row_scale[c0] : 0.0f;
        float sc1 = (c1 < N) ? row_scale[c1] : 0.0f;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc[nt][0] * sc0);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc[nt][1] * sc1);
    }
}

// ═══════════════════════════════════════════════════════════════════
// w4a16_gemm_t_v2 — w4a16_gemm_t with the shared-memory B_fp8 staging
// buffer RE-BANKED and the dequant stores VECTORIZED.
//
// BIT-IDENTICAL to w4a16_gemm_t: same LUT, same `* scale2`, same
// cvt.rn.satfinite.e4m3x2.f32 operand order, same K_STEP=32 pipeline,
// same MMA fragment addressing, same accumulate order. The ONLY changes
// are (a) the B_fp8 row stride 32 → 48 bytes and (b) 16 STS.U16 per
// thread per K-step → 2 STS.128.
//
// WHY (verified in SASS, `cuobjdump -sass`):
//   w4a16_gemm_t emits 32 STS.U16 for its two DEQUANT_T instances —
//   nvcc does NOT coalesce them. With a 32-byte row stride the store
//   address is my_n*8 + kp/2 words, and gcd(8,32) = 4, so 32 lanes land
//   on only 4 distinct banks at 4 distinct addresses each: an 8-WAY BANK
//   CONFLICT on every dequant store. The MMA-side reload
//   `&smem_B_fp8[nt*8 + group_id][4*tid]` is (nc*8 + tid) mod 32, which
//   with group_id ∈ 0..7 collapses onto 4 bank groups → 2-WAY CONFLICT.
//   Together those two are ~77% of this kernel's shared-memory
//   transactions, which is why it lands at ~35 TFLOP/s (14% of dense-FP8
//   peak) on a GEMM with ~4600 FLOP/byte of arithmetic intensity: it is
//   LSU-bound, not tensor-core bound and not DRAM bound.
//
//   Stride 48 B = 12 words fixes both. gcd(12,32) = 4 but each lane of a
//   128-bit store owns 4 consecutive banks, so the 8 lanes of a store
//   phase cover {0,12,24,4,16,28,8,20} + {0..3} = all 32 banks exactly
//   once → conflict-free. The MMA reload is nc*12 + tid, and
//   group_id*12 mod 32 + tid ∈ 0..3 is likewise a permutation of 0..31
//   → conflict-free.
//
//   Per warp per K=64 of work, shared-memory transactions:
//     w4a16_gemm_t     : 64 LUT + 32 Bp + 256 STS + 128 MMA + 16 A = 496
//     w4a16_gemm_t_v2  : 64 LUT + 32 Bp +  16 STS +  64 MMA + 16 A = 192
//   i.e. 2.6× fewer, for +2048 B of smem (19584 → 21632, still 4 CTAs/SM
//   at the 100 KB/SM sm_120 limit vs 5 today — a far smaller occupancy
//   give-up than w4a16_gemm_t_k64's 39104 B / 2 CTAs per SM).
//
// Same signature/grid/block as w4a16_gemm_t:
//   Grid: (ceil(N/128), ceil(M/64), 1)  Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════
#define TV2_BSTRIDE 48   // 32 + 16: 12 words; 48 % 16 == 0 for STS.128

extern "C" __global__ void w4a16_gemm_t_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A_tv2[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp_tv2[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs_tv2[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_B_fp8_tv2[N_TILE_LG][TV2_BSTRIDE];
    __shared__ float smem_LUT_tv2[16];

    if (threadIdx.x < 16) smem_LUT_tv2[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // Byte-for-byte the w4a16_gemm_t loader.
    #define TV2_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A_tv2[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp_tv2[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs_tv2[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    // Two scale groups per K=32 step (GROUP_SIZE=16), each covering 8 packed
    // bytes = 16 output FP8 bytes = exactly one 16-byte-aligned STS.128.
    //
    // Byte-order proof vs the scalar DEQUANT_T: it stores
    //   *(u16*)&B_fp8[my_n][kp*2] = cvt(hi = packed>>4, lo = packed&0xF)
    // and `cvt.e4m3x2.f32 d, a, b` puts operand b in the LOW byte, so byte
    // kp*2 = low nibble (K index kp*2) and kp*2+1 = high nibble. Here w[j]
    // packs pairs (kp, kp+1) little-endian as
    //   [lo(kp), hi(kp), lo(kp+1), hi(kp+1)] — the identical byte sequence.
    #define TV2_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = smem_Bs_tv2[(buf)][0][my_n]; \
        *(unsigned char*)&f1 = smem_Bs_tv2[(buf)][1][my_n]; \
        float sv[2]; \
        sv[0] = (float)f0 * scale2; sv[1] = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int blk = 0; blk < 2; blk++) { \
            float s = sv[blk]; \
            unsigned int w[4]; \
            _Pragma("unroll") \
            for (int j = 0; j < 4; j++) { \
                unsigned int kp = blk * 8 + j * 2; \
                unsigned char p0 = smem_Bp_tv2[(buf)][kp][my_n]; \
                unsigned char p1 = smem_Bp_tv2[(buf)][kp + 1][my_n]; \
                float lo0 = smem_LUT_tv2[p0 & 0xF] * s; \
                float hi0 = smem_LUT_tv2[p0 >> 4] * s; \
                float lo1 = smem_LUT_tv2[p1 & 0xF] * s; \
                float hi1 = smem_LUT_tv2[p1 >> 4] * s; \
                unsigned short e0, e1; \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(e0) : "f"(hi0), "f"(lo0)); \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(e1) : "f"(hi1), "f"(lo1)); \
                w[j] = ((unsigned int)e1 << 16) | (unsigned int)e0; \
            } \
            uint4 vv; vv.x = w[0]; vv.y = w[1]; vv.z = w[2]; vv.w = w[3]; \
            *(uint4*)&smem_B_fp8_tv2[my_n][blk * 16] = vv; \
        } \
    } while(0)

    // Identical to w4a16_gemm_t's COMPUTE_MMA modulo the B row stride.
    #define TV2_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A_tv2[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_tv2[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_tv2[nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    TV2_ISSUE_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    TV2_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        TV2_ISSUE_LOADS(nxt, k_base);
        cp_async_commit();
        TV2_COMPUTE_MMA(cur);
        cp_async_wait_all();
        __syncthreads();
        TV2_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    TV2_COMPUTE_MMA(cur);

    #undef TV2_ISSUE_LOADS
    #undef TV2_DEQUANT
    #undef TV2_COMPUTE_MMA
    #undef TV2_BSTRIDE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// w4a16_gemm_t_k64_v2 — w4a16_gemm_t_k64 with an explicitly VECTORIZED
// dequant store. BIT-IDENTICAL output to w4a16_gemm_t_k64 (and hence to
// w4a16_gemm_t): identical LUT, identical `* scale2`, identical
// cvt.rn.satfinite.e4m3x2.f32 operand order, identical K accumulation
// order. The ONLY difference is the width of the shared-memory stores
// that publish the dequanted B tile.
//
// WHY (measured mechanism, prefill M=2410 shared expert):
//   w4a16_gemm_t is shared-memory-transaction bound, not tensor-core
//   bound. Per warp per K=32 of work it issues roughly:
//     LUT LDS.32          32 tx   (2 per packed byte — irreducible)
//     B_packed LDS.U8     16 tx
//     dequant STS         16 tx   (stride-32 B_fp8 rows → 2-way conflict)
//     MMA operand LDS.32  64 tx   (stride-32 B_fp8 rows → 2-way conflict)
//     A-fragment LDS.32    8 tx
//   K64 already fixes the two conflicts by padding the B_fp8 row stride
//   to 80 B (20 words, so nc*20 + tid spans all 32 banks) and halves the
//   outer-loop/barrier count: 96 tx per K=32 of work instead of 136.
//
//   This variant additionally guarantees the dequant STS is emitted as
//   four 16-byte STS.128 per thread per K-step instead of thirty-two
//   STS.U16. With the 80-byte row stride each 8-lane phase of a 128-bit
//   store covers banks {0,20,8,28,16,4,24,12}+{0..3} = all 32 → fully
//   conflict-free, 16 tx per K64 step. nvcc *may* already coalesce the
//   scalar stores in w4a16_gemm_t_k64; if it does, this kernel is a wash
//   and the A/B will say so. If it does not, this removes ~8% of the
//   remaining shared-memory traffic.
//
// K must be divisible by 64. Same signature/grid/block as w4a16_gemm_t:
//   Grid: (ceil(N/128), ceil(M/64), 1)  Block: (128, 1, 1)
//
// smem: A 2×64×72×2=18432B, Bp 2×32×144=9216B, Bs 2×4×144=1152B,
//       B_fp8 128×80=10240B, LUT 64B ≈ 38.2KB (< 48KB static cap).
// ═══════════════════════════════════════════════════════════════════
#define KV2_K_STEP 64
#define KV2_PAD    8   // (64+8)*2 = 144 B rows, 144 % 16 == 0 ✓
#define KV2_BSTRIDE 80 // 64 + 16: 20 words, coprime-enough with 32 banks

extern "C" __global__ void w4a16_gemm_t_k64_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A_v2[2][M_TILE][KV2_K_STEP + KV2_PAD];
    __shared__ unsigned char smem_Bp_v2[2][KV2_K_STEP / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs_v2[2][KV2_K_STEP / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_B_fp8_v2[N_TILE_LG][KV2_BSTRIDE];
    __shared__ float smem_LUT_v2[16];

    if (threadIdx.x < 16) smem_LUT_v2[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int astv2 = KV2_K_STEP + KV2_PAD;

    // Byte-for-byte the K64 loader: A = 4 rounds × 16 rows, Bp = 2 rounds ×
    // 16 kp-rows, Bs inline for the 4 scale groups of a K=64 step.
    #define KV2_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 3; \
            unsigned int a_col = (threadIdx.x & 7) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = rnd * 16 + a_row_base; \
                unsigned int gr = cta_m + row; \
                cp_async_pred_16(&smem_A_v2[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gns = cta_n + ns; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int kp_cur = rnd * 16 + kp; \
                unsigned int gke = (kb) + (kp_cur << 1); \
                cp_async_pred_16(&smem_Bp_v2[(buf)][kp_cur][ns], \
                    &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 <= K) && (gns + 15 < N)); \
                if (kp_cur < KV2_K_STEP / GROUP_SIZE) { \
                    unsigned int sg = (kb) / GROUP_SIZE + kp_cur; \
                    cp_async_pred_16(&smem_Bs_v2[(buf)][kp_cur][ns], \
                        &B_scale[(unsigned long long)sg * N + gns], \
                        (gns + 15 < N)); \
                } \
            } \
        } \
    } while(0)

    // Dequant, vectorized. blk selects the scale group exactly as the K64
    // kernel's four unrolled 8-iteration bodies do (kp 0-7 → sv0, 8-15 →
    // sv1, 16-23 → sv2, 24-31 → sv3). Each blk produces 16 consecutive
    // bytes of the row → one 16-byte-aligned STS.128 (row stride 80 % 16 == 0).
    //
    // Byte order proof vs the scalar version: the scalar body stores
    //   *(u16*)&B_fp8[my_n][kp*2] = cvt(hi=packed>>4, lo=packed&0xF)
    // and cvt.e4m3x2.f32 d,a,b puts b in the LOW byte. So byte kp*2 holds
    // the low nibble (K index kp*2) and byte kp*2+1 the high nibble. Here
    // v.x/.y/.z/.w pack pairs (kp, kp+1) little-endian as
    //   [lo(kp), hi(kp), lo(kp+1), hi(kp+1)] — the identical byte sequence.
    #define KV2_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        __nv_fp8_e4m3 f0, f1, f2, f3; \
        *(unsigned char*)&f0 = smem_Bs_v2[(buf)][0][my_n]; \
        *(unsigned char*)&f1 = smem_Bs_v2[(buf)][1][my_n]; \
        *(unsigned char*)&f2 = smem_Bs_v2[(buf)][2][my_n]; \
        *(unsigned char*)&f3 = smem_Bs_v2[(buf)][3][my_n]; \
        float sv[4]; \
        sv[0] = (float)f0 * scale2; sv[1] = (float)f1 * scale2; \
        sv[2] = (float)f2 * scale2; sv[3] = (float)f3 * scale2; \
        _Pragma("unroll") \
        for (int blk = 0; blk < 4; blk++) { \
            float s = sv[blk]; \
            unsigned int w[4]; \
            _Pragma("unroll") \
            for (int j = 0; j < 4; j++) { \
                unsigned int kp = blk * 8 + j * 2; \
                unsigned char p0 = smem_Bp_v2[(buf)][kp][my_n]; \
                unsigned char p1 = smem_Bp_v2[(buf)][kp + 1][my_n]; \
                float lo0 = smem_LUT_v2[p0 & 0xF] * s; \
                float hi0 = smem_LUT_v2[p0 >> 4] * s; \
                float lo1 = smem_LUT_v2[p1 & 0xF] * s; \
                float hi1 = smem_LUT_v2[p1 >> 4] * s; \
                unsigned short e0, e1; \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(e0) : "f"(hi0), "f"(lo0)); \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(e1) : "f"(hi1), "f"(lo1)); \
                w[j] = ((unsigned int)e1 << 16) | (unsigned int)e0; \
            } \
            uint4 vv; vv.x = w[0]; vv.y = w[1]; vv.z = w[2]; vv.w = w[3]; \
            *(uint4*)&smem_B_fp8_v2[my_n][blk * 16] = vv; \
        } \
    } while(0)

    // Two m16n8k32 MMAs per N-tile — identical fragment addressing and
    // identical accumulate order to w4a16_gemm_t_k64.
    #define KV2_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A_v2[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * astv2 + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * astv2 + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * astv2 + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * astv2 + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_v2[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_v2[nc][16 + 4 * tid]; \
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
        unsigned int a4 = bf16x4_to_e4m3x4(&sA[fr0 * astv2 + 32 + tid * 4]); \
        unsigned int a5 = bf16x4_to_e4m3x4(&sA[fr1 * astv2 + 32 + tid * 4]); \
        unsigned int a6 = bf16x4_to_e4m3x4(&sA[fr0 * astv2 + 48 + tid * 4]); \
        unsigned int a7 = bf16x4_to_e4m3x4(&sA[fr1 * astv2 + 48 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_v2[nc][32 + 4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_v2[nc][48 + 4 * tid]; \
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a4),"r"(a5),"r"(a6),"r"(a7),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    KV2_ISSUE_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    KV2_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = KV2_K_STEP; k_base < K; k_base += KV2_K_STEP) {
        int nxt = 1 - cur;
        KV2_ISSUE_LOADS(nxt, k_base);
        cp_async_commit();
        KV2_COMPUTE_MMA(cur);
        cp_async_wait_all();
        __syncthreads();
        KV2_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    KV2_COMPUTE_MMA(cur);

    #undef KV2_ISSUE_LOADS
    #undef KV2_DEQUANT
    #undef KV2_COMPUTE_MMA
    #undef KV2_K_STEP
    #undef KV2_PAD
    #undef KV2_BSTRIDE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}
