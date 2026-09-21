// SPDX-License-Identifier: AGPL-3.0-only
//
// Compile-only SM121a EXL3 W2A8 N128 grouped-prefill component.
//
// A_fp8 is the sorted, post-H128 activation produced with one E4M3 scale per
// 128 K values. Trellis weights remain compressed and are decoded directly
// into registers. This file lives outside kernels/gb10/common on purpose: the
// build registry and serving host must not expose it before GPU numeric gates.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#ifndef W2A8_FIXED_N
#error "W2A8_FIXED_N must be explicit"
#endif
#ifndef W2A8_FIXED_K
#error "W2A8_FIXED_K must be explicit"
#endif
#ifndef W2A8_KERNEL_NAME
#error "W2A8_KERNEL_NAME must be explicit"
#endif

#define W2A8_M_TILE 64
#define W2A8_N_TILE 128
#define W2A8_K_STAGE 64
#define W2A8_K_PAIR 32
#define W2A8_SCALE_GROUP 128
#define W2A8_WEIGHT_SCALE 16.0f
#define W2A8_INV_WEIGHT_SCALE 0.0625f
#define W2A8_MCG_MULT 0xCBAC1FEDu

static_assert(W2A8_FIXED_N == 2048 || W2A8_FIXED_N == 4096,
              "W2A8 probe supports only DeepSeek gate/up/down N");
static_assert(W2A8_FIXED_K == 2048 || W2A8_FIXED_K == 4096,
              "W2A8 probe supports only DeepSeek gate/up/down K");
static_assert((W2A8_FIXED_N == 2048 && W2A8_FIXED_K == 4096) ||
              (W2A8_FIXED_N == 4096 && W2A8_FIXED_K == 2048),
              "W2A8 probe requires an exact DeepSeek gate/up or down shape");
static_assert(W2A8_FIXED_N % W2A8_N_TILE == 0, "N must be N128 aligned");
static_assert(W2A8_FIXED_K % W2A8_SCALE_GROUP == 0,
              "K must be activation-scale aligned");
static_assert(W2A8_SCALE_GROUP == 2 * W2A8_K_STAGE,
              "two K64 stages must share one activation scale");
static_assert(W2A8_K_STAGE == 2 * W2A8_K_PAIR,
              "each K64 stage must issue two K32 pairs");

union W2A8Half2Bits {
    unsigned int bits;
    __half2 values;
};

struct W2A8LaneGeom {
    int ia;
    int ib;
    int shift;
};

__device__ __forceinline__ W2A8LaneGeom w2a8_lane_geom(unsigned int lane) {
    constexpr int bits = 2;
    const int b1 = ((int)lane * 8 + 257) * bits;
    const int b0 = b1 - 16;
    const int b2 = b1 + 7 * bits;
    const int i0 = b0 >> 5;
    const int i2 = (b2 - 1) >> 5;
    return {i0 % 16, i2 % 16, (i2 + 1) * 32 - b2};
}
__device__ __forceinline__ __half2 w2a8_decode2(
    unsigned int x0, unsigned int x1) {
    x0 *= W2A8_MCG_MULT;
    x1 *= W2A8_MCG_MULT;
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
    W2A8Half2Bits u0 = {x0};
    W2A8Half2Bits u1 = {x1};
    return __hadd2(
        __lows2half2(u0.values, u1.values),
        __highs2half2(u0.values, u1.values));
}

__device__ __forceinline__ void w2a8_decode8(
    const unsigned int* tile, W2A8LaneGeom geom,
    __half2& d01, __half2& d23, __half2& d45, __half2& d67) {
    const unsigned int a = tile[geom.ia];
    const unsigned int b = tile[geom.ib];
    const unsigned int lo = __funnelshift_r(b, a, geom.shift);
    const unsigned int w7 = lo;
    const unsigned int w5 = lo >> 4;
    const unsigned int w3 = lo >> 8;
    const unsigned int w1 = lo >> 12;
    d01 = w2a8_decode2((w1 >> 2) & 0xffffu, w1 & 0xffffu);
    d23 = w2a8_decode2((w3 >> 2) & 0xffffu, w3 & 0xffffu);
    d45 = w2a8_decode2((w5 >> 2) & 0xffffu, w5 & 0xffffu);
    d67 = w2a8_decode2((w7 >> 2) & 0xffffu, w7 & 0xffffu);
}

__device__ __forceinline__ unsigned short w2a8_encode_pair(__half2 pair) {
    const float value0 = __half2float(__low2half(pair)) * W2A8_WEIGHT_SCALE;
    const float value1 = __half2float(__high2half(pair)) * W2A8_WEIGHT_SCALE;
    unsigned short encoded;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(encoded) : "f"(value1), "f"(value0));
    return encoded;
}
// END w2a8_encode_pair

__device__ __forceinline__ unsigned int w2a8_repack_b(
    unsigned int source_pairs, unsigned int tid) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int quad_base = lane & ~3u;
    const int src0 = (int)(quad_base + 2 * (tid & 1));
    const int src1 = src0 + 1;
    const unsigned int source0 = __shfl_sync(0xffffffffu, source_pairs, src0);
    const unsigned int source1 = __shfl_sync(0xffffffffu, source_pairs, src1);
    const unsigned int shift = 16 * (tid >> 1);
    return ((source0 >> shift) & 0xffffu) | ((source1 >> shift) << 16);
}
// END w2a8_repack_b

__device__ __forceinline__ void w2a8_mma(
    float acc[4], unsigned int a0, unsigned int a1, unsigned int a2,
    unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
}

extern "C" __global__ __launch_bounds__(256) void W2A8_KERNEL_NAME(
    const unsigned char* __restrict__ A_fp8,
    const float* __restrict__ a_scale,
    const unsigned long long* __restrict__ trellis_tab,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    unsigned int num_experts, unsigned int N, unsigned int K,
    unsigned int bits, unsigned int persistent_mode) {
    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1) return;
    if (gridDim.y != 1 || gridDim.z != 1) return;
    if (N != W2A8_FIXED_N || K != W2A8_FIXED_K) return;
    if (bits != 2 || persistent_mode != 1) return;
    constexpr unsigned int n_tiles = W2A8_FIXED_N / W2A8_N_TILE;
    if ((unsigned long long)gridDim.x !=
        (unsigned long long)num_experts * n_tiles) return;

    const unsigned int n_tile = blockIdx.x % n_tiles;
    const unsigned int expert_id = blockIdx.x / n_tiles;
    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    if (m_start >= m_end) return;
    const unsigned short* trellis =
        (const unsigned short*)trellis_tab[expert_id];
    if (trellis == nullptr) return;

    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int group = lane >> 2;
    const unsigned int tid = lane & 3;
    const unsigned int n_base = n_tile * W2A8_N_TILE;
    const W2A8LaneGeom lane_geom = w2a8_lane_geom(lane);

    __shared__ __align__(16) uint4 smem_A[W2A8_M_TILE][5];
    __shared__ __align__(16) uint4 smem_T[4][32];

    for (int m_local = 0; m_start + m_local < m_end; m_local += W2A8_M_TILE) {
        float outer[4][2][4] = {};

        for (unsigned int k_block = 0; k_block < W2A8_FIXED_K;
             k_block += W2A8_SCALE_GROUP) {
            float inner[4][2][4] = {};

#pragma unroll 1
            for (unsigned int k_stage = 0; k_stage < W2A8_SCALE_GROUP;
                 k_stage += W2A8_K_STAGE) {
                for (unsigned int vector = threadIdx.x;
                     vector < W2A8_M_TILE * 4; vector += blockDim.x) {
                    const unsigned int row_local = vector >> 2;
                    const unsigned int vector_k = vector & 3;
                    const int row = m_start + m_local + (int)row_local;
                    uint4 packed = {0, 0, 0, 0};
                    if (row < m_end) {
                        const uint4* src = (const uint4*)(
                            A_fp8 + (unsigned long long)row * W2A8_FIXED_K +
                            k_block + k_stage + vector_k * 16);
                        packed = *src;
                    }
                    smem_A[row_local][vector_k] = packed;
                }

                for (unsigned int load = threadIdx.x; load < 128;
                     load += blockDim.x) {
                    const unsigned int k_tile = load >> 5;
                    const unsigned int word = load & 31;
                    const unsigned int kb = (k_block + k_stage) / 16 + k_tile;
                    const unsigned int nb = n_base / 16;
                    const uint4* src = (const uint4*)trellis +
                        ((unsigned long long)kb * (W2A8_FIXED_N / 16) + nb) * 4 + word;
                    smem_T[k_tile][word] = *src;
                }
                __syncthreads();

#pragma unroll
                for (unsigned int pair = 0; pair < 2; ++pair) {
                    const unsigned int* tile0 =
                        (const unsigned int*)smem_T[pair * 2] + warp * 16;
                    const unsigned int* tile1 =
                        (const unsigned int*)smem_T[pair * 2 + 1] + warp * 16;
                    __half2 d01_0, d23_0, d45_0, d67_0;
                    __half2 d01_1, d23_1, d45_1, d67_1;
                    w2a8_decode8(
                        tile0, lane_geom, d01_0, d23_0, d45_0, d67_0);
                    w2a8_decode8(
                        tile1, lane_geom, d01_1, d23_1, d45_1, d67_1);
                    const unsigned int low0 =
                        (unsigned int)w2a8_encode_pair(d01_0) |
                        ((unsigned int)w2a8_encode_pair(d23_0) << 16);
                    const unsigned int low1 =
                        (unsigned int)w2a8_encode_pair(d01_1) |
                        ((unsigned int)w2a8_encode_pair(d23_1) << 16);
                    const unsigned int high0 =
                        (unsigned int)w2a8_encode_pair(d45_0) |
                        ((unsigned int)w2a8_encode_pair(d67_0) << 16);
                    const unsigned int high1 =
                        (unsigned int)w2a8_encode_pair(d45_1) |
                        ((unsigned int)w2a8_encode_pair(d67_1) << 16);
                    const unsigned int b00 = w2a8_repack_b(low0, tid);
                    const unsigned int b01 = w2a8_repack_b(low1, tid);
                    const unsigned int b10 = w2a8_repack_b(high0, tid);
                    const unsigned int b11 = w2a8_repack_b(high1, tid);

#pragma unroll
                    for (unsigned int mt = 0; mt < 4; ++mt) {
                        const unsigned int row0 = mt * 16 + group;
                        const unsigned int row1 = row0 + 8;
                        const unsigned int byte_k = pair * W2A8_K_PAIR;
                        const unsigned int stride = 5 * sizeof(uint4);
                        const unsigned char* a = (const unsigned char*)smem_A;
                        // BEGIN native A fragments
                        const unsigned int a0 = *(const unsigned int*)(
                            a + row0 * stride + byte_k + 4 * tid);
                        const unsigned int a1 = *(const unsigned int*)(
                            a + row1 * stride + byte_k + 4 * tid);
                        const unsigned int a2 = *(const unsigned int*)(
                            a + row0 * stride + byte_k + 16 + 4 * tid);
                        const unsigned int a3 = *(const unsigned int*)(
                            a + row1 * stride + byte_k + 16 + 4 * tid);
                        // END native A fragments
                        w2a8_mma(inner[mt][0], a0, a1, a2, a3, b00, b01);
                        w2a8_mma(inner[mt][1], a0, a1, a2, a3, b10, b11);
                    }
                }
                __syncthreads();
            }

            // BEGIN scale fold
#pragma unroll
            for (unsigned int mt = 0; mt < 4; ++mt) {
                const unsigned int row0 = mt * 16 + group;
                const unsigned int row1 = row0 + 8;
                const int global0 = m_start + m_local + (int)row0;
                const int global1 = m_start + m_local + (int)row1;
                const unsigned int scales_per_row = W2A8_FIXED_K / W2A8_SCALE_GROUP;
                float scale0 = global0 < m_end
                    ? a_scale[(unsigned long long)global0 * scales_per_row +
                              k_block / W2A8_SCALE_GROUP]
                    : 0.0f;
                float scale1 = global1 < m_end
                    ? a_scale[(unsigned long long)global1 * scales_per_row +
                              k_block / W2A8_SCALE_GROUP]
                    : 0.0f;
                if (!(scale0 >= 1.0e-12f) || !isfinite(scale0)) scale0 = 0.0f;
                if (!(scale1 >= 1.0e-12f) || !isfinite(scale1)) scale1 = 0.0f;
                const float factor0 = scale0 * W2A8_INV_WEIGHT_SCALE;
                const float factor1 = scale1 * W2A8_INV_WEIGHT_SCALE;
#pragma unroll
                for (unsigned int nt = 0; nt < 2; ++nt) {
                    outer[mt][nt][0] += inner[mt][nt][0] * factor0;
                    outer[mt][nt][1] += inner[mt][nt][1] * factor0;
                    outer[mt][nt][2] += inner[mt][nt][2] * factor1;
                    outer[mt][nt][3] += inner[mt][nt][3] * factor1;
                }
            }
            // END scale fold
        }

#pragma unroll
        for (unsigned int mt = 0; mt < 4; ++mt) {
#pragma unroll
            for (unsigned int nt = 0; nt < 2; ++nt) {
                const unsigned int col = n_base + warp * 16 + nt * 8 + tid * 2;
                const unsigned int row0 = mt * 16 + group;
                const unsigned int row1 = row0 + 8;
                if (m_start + m_local + (int)row0 < m_end) {
                    __nv_bfloat16* out = C +
                        (unsigned long long)(m_start + m_local + row0) * W2A8_FIXED_N + col;
                    out[0] = __float2bfloat16(outer[mt][nt][0]);
                    out[1] = __float2bfloat16(outer[mt][nt][1]);
                }
                if (m_start + m_local + (int)row1 < m_end) {
                    __nv_bfloat16* out = C +
                        (unsigned long long)(m_start + m_local + row1) * W2A8_FIXED_N + col;
                    out[0] = __float2bfloat16(outer[mt][nt][2]);
                    out[1] = __float2bfloat16(outer[mt][nt][3]);
                }
            }
        }
        __syncthreads();
    }
}
