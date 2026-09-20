// SPDX-License-Identifier: AGPL-3.0-only
//
// Default-unrouted SM121 W4A4 long-prefill experiment.
//
// This is deliberately an ABI-separated shadow.  The host launches at most
// one CTA per SM; each CTA walks output tiles with a static grid-stride.  A
// logical K=256 tile is loaded and transposed once, then consumed as four
// ordered K=64 block-scaled OMMAs.  That preserves the promoted K-major
// kernel's FP32 accumulation order and BF16 epilogue.
//
// This is TMA-shaped, not TMA: cp.async.bulk.tensor requires a CUtensorMap
// descriptor encoded by the host.  Atlas's standalone .cu -> PTX modules are
// launched with raw pointers/dimensions and expose no tensor-map construction
// ABI.  Adding real TMA therefore requires a host/runtime integration change,
// which is intentionally outside this isolated prototype.
//
// ABI:
//   grid  = (min(tile_count, resident_ctas), 1, 1)
//   block = (256, 1, 1)
//   A     = [M,K/2], A_sf = [M,K/16]
//   B     = [K/2,N], B_sf = [K/16,N]
// Preconditions: M>0, N%128==0, K%256==0.

#include <cuda_bf16.h>

#define PERSIST_M 128
#define PERSIST_N 128
#define PERSIST_K 256
#define MMA_K 64
#define GROUP_SIZE 16

__device__ __forceinline__ void persist_cp_async_16(
    void* dst_smem, const void* src_gmem, bool pred
) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16u : 0u;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}

__device__ __forceinline__ void persist_cp_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void persist_cp_wait_all() {
    asm volatile("cp.async.wait_group 0;");
}

extern "C" __global__
__launch_bounds__(256, 2)
void nvfp4_nvfp4_gemm_kmajor_k256_persistent(
    const unsigned char* __restrict__ A_packed,
    const unsigned char* __restrict__ A_scale,
    const unsigned char* __restrict__ B_packed_t,
    const unsigned char* __restrict__ B_scale_t,
    const float scale2_ab,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    // Single-buffering needs 55,296 bytes.  That exceeds the 48-KiB static
    // PTX limit, so the isolated host gate must explicitly opt this function
    // into 55,296 bytes of dynamic shared memory before launch.  A second
    // K256 buffer would need 110,592 bytes and is intentionally rejected.
    extern __shared__ __align__(16) unsigned char storage[];
    unsigned char (*smem_Ap)[PERSIST_K / 2] =
        (unsigned char (*)[PERSIST_K / 2])(storage + 0);
    unsigned char (*smem_As)[PERSIST_K / GROUP_SIZE] =
        (unsigned char (*)[PERSIST_K / GROUP_SIZE])(storage + 16384);
    unsigned char (*smem_Bp_stage)[PERSIST_N] =
        (unsigned char (*)[PERSIST_N])(storage + 18432);
    unsigned char (*smem_Bs_stage)[PERSIST_N] =
        (unsigned char (*)[PERSIST_N])(storage + 34816);
    unsigned char (*smem_Bp)[PERSIST_K / 2] =
        (unsigned char (*)[PERSIST_K / 2])(storage + 36864);
    unsigned char (*smem_Bs)[PERSIST_K / GROUP_SIZE] =
        (unsigned char (*)[PERSIST_K / GROUP_SIZE])(storage + 53248);

    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31u;
    const unsigned int warp_m_offset = warp_id * 16u;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3u;
    const unsigned int n_tiles = N / PERSIST_N;
    const unsigned int m_tiles = (M + PERSIST_M - 1u) / PERSIST_M;
    const unsigned int tile_count = n_tiles * m_tiles;

    for (unsigned int tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
        const unsigned int tile_n = tile % n_tiles;
        const unsigned int tile_m = tile / n_tiles;
        const unsigned int cta_n = tile_n * PERSIST_N;
        const unsigned int cta_m = tile_m * PERSIST_M;

        float acc[16][4];
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            acc[nt][0] = 0.0f;
            acc[nt][1] = 0.0f;
            acc[nt][2] = 0.0f;
            acc[nt][3] = 0.0f;
        }

        for (unsigned int k_base = 0; k_base < K; k_base += PERSIST_K) {
            // A packed: 128 rows * 128 bytes = 1024 sixteen-byte copies.
            for (unsigned int pass = 0; pass < 4; ++pass) {
                unsigned int chunk = threadIdx.x + pass * 256u;
                unsigned int row = chunk >> 3;
                unsigned int col = (chunk & 7u) << 4;
                unsigned int gr = cta_m + row;
                unsigned int gc = k_base / 2u + col;
                bool valid = gr < M;
                persist_cp_async_16(
                    &smem_Ap[row][col],
                    &A_packed[(unsigned long long)gr * (K / 2u) + gc],
                    valid
                );
            }

            // A scales: 128 rows * 16 bytes.  Naturally aligned u32 words.
            for (unsigned int pass = 0; pass < 2; ++pass) {
                unsigned int word_index = threadIdx.x + pass * 256u;
                unsigned int row = word_index >> 2;
                unsigned int word = word_index & 3u;
                unsigned int gr = cta_m + row;
                unsigned int packed = 0;
                if (gr < M) {
                    packed = *(const unsigned int*)&A_scale[
                        (unsigned long long)gr * (K / GROUP_SIZE) +
                        k_base / GROUP_SIZE + word * 4u
                    ];
                }
                *(unsigned int*)&smem_As[row][word * 4u] = packed;
            }

            // K-major B: 128 packed-K bytes * 128 contiguous N bytes.
            for (unsigned int pass = 0; pass < 4; ++pass) {
                unsigned int chunk = threadIdx.x + pass * 256u;
                unsigned int kp = chunk >> 3;
                unsigned int ns = (chunk & 7u) << 4;
                unsigned int gkp = k_base / 2u + kp;
                unsigned int gn = cta_n + ns;
                persist_cp_async_16(
                    &smem_Bp_stage[kp][ns],
                    &B_packed_t[(unsigned long long)gkp * N + gn],
                    true
                );
            }
            // Eight copies/thread above is one complete cp.async group.
            persist_cp_commit();

            // K-major B scales: 16 groups * 128 contiguous N bytes.
            if (threadIdx.x < 128) {
                unsigned int kg = threadIdx.x >> 3;
                unsigned int ns = (threadIdx.x & 7u) << 4;
                unsigned int gg = k_base / GROUP_SIZE + kg;
                unsigned int gn = cta_n + ns;
                persist_cp_async_16(
                    &smem_Bs_stage[kg][ns],
                    &B_scale_t[(unsigned long long)gg * N + gn],
                    true
                );
            }
            persist_cp_commit();
            persist_cp_wait_all();

            // `wait_group` is thread-local: it guarantees completion only for
            // the cp.async operations issued by the calling thread.  The K256
            // transpose mapping is deliberately different from the load
            // mapping, so transposers consume bytes produced by other threads.
            // Publish the complete staging tiles before any cross-thread read.
            __syncthreads();

            // Convert coalesced K-major staging into the exact per-N fragment
            // layout used by the promoted M128/M256 kernels.
            for (unsigned int pass = 0; pass < 16; ++pass) {
                unsigned int index = threadIdx.x + pass * 256u;
                unsigned int kp = index >> 5;
                unsigned int n_word = index & 31u;
                unsigned int nbase = n_word * 4u;
                unsigned int word = *(const unsigned int*)&smem_Bp_stage[kp][nbase];
                smem_Bp[nbase + 0u][kp] = (unsigned char)(word >> 0);
                smem_Bp[nbase + 1u][kp] = (unsigned char)(word >> 8);
                smem_Bp[nbase + 2u][kp] = (unsigned char)(word >> 16);
                smem_Bp[nbase + 3u][kp] = (unsigned char)(word >> 24);
            }
            for (unsigned int pass = 0; pass < 2; ++pass) {
                unsigned int index = threadIdx.x + pass * 256u;
                unsigned int kg = index >> 5;
                unsigned int n_word = index & 31u;
                unsigned int nbase = n_word * 4u;
                unsigned int word = *(const unsigned int*)&smem_Bs_stage[kg][nbase];
                smem_Bs[nbase + 0u][kg] = (unsigned char)(word >> 0);
                smem_Bs[nbase + 1u][kg] = (unsigned char)(word >> 8);
                smem_Bs[nbase + 2u][kg] = (unsigned char)(word >> 16);
                smem_Bs[nbase + 3u][kg] = (unsigned char)(word >> 24);
            }
            __syncthreads();

            // Four K64 OMMAs, in the exact parent order.  The K256 tile only
            // changes staging/barrier granularity, never reduction order.
            for (unsigned int sub = 0; sub < PERSIST_K / MMA_K; ++sub) {
                unsigned int byte_base = sub * (MMA_K / 2u);
                unsigned int scale_base = sub * (MMA_K / GROUP_SIZE);
                unsigned int fr0 = warp_m_offset + group_id;
                unsigned int fr1 = fr0 + 8u;
                unsigned int a_col = byte_base + 4u * tid;
                unsigned int a0 = *(const unsigned int*)&smem_Ap[fr0][a_col];
                unsigned int a1 = *(const unsigned int*)&smem_Ap[fr1][a_col];
                unsigned int a2 = *(const unsigned int*)&smem_Ap[fr0][a_col + 16u];
                unsigned int a3 = *(const unsigned int*)&smem_Ap[fr1][a_col + 16u];
                unsigned int m_sfa = ((lane_id >> 2) + 8u * (lane_id & 1u)) + warp_m_offset;
                unsigned int sfa = *(const unsigned int*)&smem_As[m_sfa][scale_base];

                #pragma unroll
                for (int nt = 0; nt < 16; ++nt) {
                    unsigned int nc = (unsigned int)nt * 8u + group_id;
                    unsigned int b0 = *(const unsigned int*)&smem_Bp[nc][a_col];
                    unsigned int b1 = *(const unsigned int*)&smem_Bp[nc][a_col + 16u];
                    unsigned int sfb = *(const unsigned int*)&smem_Bs[nc][scale_base];
                    asm volatile(
                        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
                        "{%14},{%15,%16},{%17},{%18,%19};"
                        : "=f"(acc[nt][0]), "=f"(acc[nt][1]),
                          "=f"(acc[nt][2]), "=f"(acc[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc[nt][0]), "f"(acc[nt][1]),
                          "f"(acc[nt][2]), "f"(acc[nt][3]),
                          "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0),
                          "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0)
                    );
                }
            }
            // All warps must finish reading the single buffer before reuse.
            __syncthreads();
        }

        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            unsigned int c0 = cta_n + (unsigned int)nt * 8u + tid * 2u;
            unsigned int c1 = c0 + 1u;
            unsigned int r0 = cta_m + warp_m_offset + group_id;
            unsigned int r1 = r0 + 8u;
            if (r0 < M) {
                C[(unsigned long long)r0 * N + c0] = __float2bfloat16(acc[nt][0] * scale2_ab);
                C[(unsigned long long)r0 * N + c1] = __float2bfloat16(acc[nt][1] * scale2_ab);
            }
            if (r1 < M) {
                C[(unsigned long long)r1 * N + c0] = __float2bfloat16(acc[nt][2] * scale2_ab);
                C[(unsigned long long)r1 * N + c1] = __float2bfloat16(acc[nt][3] * scale2_ab);
            }
        }
        // Keep a CTA's next persistent tile from racing its own epilogue.
        __syncthreads();
    }
}
