// SPDX-License-Identifier: AGPL-3.0-only
//
// Exact-shape DeepSeek K2 W2A8 experiment. One 256-thread CTA owns one
// expert/N128 tile, computes gate and up sequentially for M64 rows, retains
// their BF16 boundaries in shared memory, then emits the post-H128 + SwiGLU +
// down-pre-H128 activation as E4M3 with one FP32 scale per K128. Serving can
// reach it only through a model-specific strict default-off wrapper; native
// numeric and timing qualification remain pending.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <climits>

#ifndef W2A8_FIXED_N
#error "W2A8_FIXED_N must be explicit"
#endif
#ifndef W2A8_FIXED_K
#error "W2A8_FIXED_K must be explicit"
#endif
#ifndef W2A8_KERNEL_NAME
#error "W2A8_KERNEL_NAME must be explicit"
#endif
#ifndef W2A8_PACKED_E4M3_CANDIDATE
#define W2A8_PACKED_E4M3_CANDIDATE 0
#endif
static_assert(W2A8_PACKED_E4M3_CANDIDATE == 0 ||
              W2A8_PACKED_E4M3_CANDIDATE == 1,
              "packed E4M3 candidate must be boolean");
#ifndef W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE
#define W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE 0
#endif
static_assert(W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE == 0 ||
              W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE == 1,
              "fused N128 double-buffer candidate must be boolean");
#ifndef W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE
#define W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE 0
#endif
static_assert(W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE == 0 ||
              W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE == 1,
              "fused N128 continuous-ring candidate must be boolean");
#if W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE
static_assert(W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE == 1,
              "continuous K64 ring requires the two staging buffers");
static_assert(W2A8_PACKED_E4M3_CANDIDATE == 1,
              "continuous K64 ring freezes packed E4M3 on");
#endif

#define W2F_M_TILE 64
#define W2F_N_TILE 128
#define W2F_GATE_UP_N W2A8_FIXED_N
#define W2F_GATE_UP_K W2A8_FIXED_K
#define W2F_K_STAGE 64
#define W2F_K_GROUP 128
#define W2F_WEIGHT_SCALE 16.0f
#define W2F_INV_WEIGHT_SCALE 0.0625f
#define W2F_E4M3_MAX 448.0f
#define W2F_MIN_SCALE 1.0e-12f
#define W2F_MCG_MULT 0xCBAC1FEDu
#define W2F_HROW_WARPS 8
#define W2F_RSQRT128 0.088388347648f
#define W2F_SWIGLU_LIMIT 10.0f
#define W2F_EXPERTS 256

static_assert(W2F_GATE_UP_N == 2048 && W2F_GATE_UP_K == 4096,
              "fused component requires exact DeepSeek gate/up shape");

union W2FHalf2Bits {
    unsigned int bits;
    __half2 values;
};

struct W2FLaneGeom {
    int ia;
    int ib;
    int shift;
};

struct W2FGemmScratch {
    uint4 activation[W2F_M_TILE][5];
    uint4 trellis[4][32];
};

union W2FWork {
#if W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE
    W2FGemmScratch gemm[2];
#else
    W2FGemmScratch gemm[1];
#endif
    float abs_values[W2F_HROW_WARPS][W2F_K_GROUP];
};

struct __align__(16) W2FShared {
    __nv_bfloat16 gate[W2F_M_TILE][W2F_N_TILE];
    __nv_bfloat16 up[W2F_M_TILE][W2F_N_TILE];
    W2FWork work;
};

#if W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE
static_assert(sizeof(W2FShared) == 46 * 1024,
              "double-buffered fused N128 shared layout changed");
#else
static_assert(sizeof(W2FShared) == 39 * 1024,
              "fused N128 component shared layout changed");
#endif
static_assert(sizeof(W2FShared) <= 49152, "fused N128 component exceeds 48 KiB");

__device__ __forceinline__ void w2f_had128(
    float& h0, float& h1, float& h2, float& h3, int lane) {
    const float s0 = h0 + h1;
    const float d0 = h0 - h1;
    const float s1 = h2 + h3;
    const float d1 = h2 - h3;
    h0 = s0 + s1;
    h1 = d0 + d1;
    h2 = s0 - s1;
    h3 = d0 - d1;
#pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        const float p0 = __shfl_xor_sync(0xffffffffu, h0, i);
        const float p1 = __shfl_xor_sync(0xffffffffu, h1, i);
        const float p2 = __shfl_xor_sync(0xffffffffu, h2, i);
        const float p3 = __shfl_xor_sync(0xffffffffu, h3, i);
        const float sign = (lane & i) ? -1.0f : 1.0f;
        h0 = __fmaf_rn(sign, h0, p0);
        h1 = __fmaf_rn(sign, h1, p1);
        h2 = __fmaf_rn(sign, h2, p2);
        h3 = __fmaf_rn(sign, h3, p3);
    }
}

__device__ __forceinline__ W2FLaneGeom w2f_lane_geom(unsigned int lane) {
    const int b1 = ((int)lane * 8 + 257) * 2;
    const int b0 = b1 - 16;
    const int b2 = b1 + 14;
    const int i0 = b0 >> 5;
    const int i2 = (b2 - 1) >> 5;
    return {i0 % 16, i2 % 16, (i2 + 1) * 32 - b2};
}

__device__ __forceinline__ __half2 w2f_decode2(
    unsigned int x0, unsigned int x1) {
    x0 *= W2F_MCG_MULT;
    x1 *= W2F_MCG_MULT;
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
    W2FHalf2Bits u0 = {x0};
    W2FHalf2Bits u1 = {x1};
    return __hadd2(
        __lows2half2(u0.values, u1.values),
        __highs2half2(u0.values, u1.values));
}

__device__ __forceinline__ void w2f_decode8(
    const unsigned int* tile, W2FLaneGeom geom,
    __half2& d01, __half2& d23, __half2& d45, __half2& d67) {
    const unsigned int a = tile[geom.ia];
    const unsigned int b = tile[geom.ib];
    const unsigned int lo = __funnelshift_r(b, a, geom.shift);
    d01 = w2f_decode2(((lo >> 12) >> 2) & 0xffffu, (lo >> 12) & 0xffffu);
    d23 = w2f_decode2(((lo >> 8) >> 2) & 0xffffu, (lo >> 8) & 0xffffu);
    d45 = w2f_decode2(((lo >> 4) >> 2) & 0xffffu, (lo >> 4) & 0xffffu);
    d67 = w2f_decode2((lo >> 2) & 0xffffu, lo & 0xffffu);
}

__device__ __forceinline__ unsigned short w2f_encode_pair(__half2 pair) {
#if W2A8_PACKED_E4M3_CANDIDATE
    // K2 decode is finite and strictly inside (-4, 4), so scaling by 2^4 is
    // an exact half exponent shift. Converting the scaled packed halves is
    // therefore byte-identical to the incumbent half->float, *16, FP8 path.
    W2FHalf2Bits scale = {0x4c004c00u};  // half2(16.0, 16.0)
    W2FHalf2Bits scaled;
    scaled.values = __hmul2(pair, scale.values);
    unsigned short encoded;
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                 : "=h"(encoded) : "r"(scaled.bits));
    return encoded;
#else
    const float value0 = __half2float(__low2half(pair)) * W2F_WEIGHT_SCALE;
    const float value1 = __half2float(__high2half(pair)) * W2F_WEIGHT_SCALE;
    unsigned short encoded;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(encoded) : "f"(value1), "f"(value0));
    return encoded;
#endif
}

__device__ __forceinline__ unsigned int w2f_repack_b(
    unsigned int source_pairs, unsigned int tid) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int quad_base = lane & ~3u;
    const int src0 = (int)(quad_base + 2 * (tid & 1));
    const unsigned int source0 = __shfl_sync(0xffffffffu, source_pairs, src0);
    const unsigned int source1 = __shfl_sync(0xffffffffu, source_pairs, src0 + 1);
    const unsigned int shift = 16 * (tid >> 1);
    return ((source0 >> shift) & 0xffffu) | ((source1 >> shift) << 16);
}

__device__ __forceinline__ void w2f_mma(
    float acc[4], unsigned int a0, unsigned int a1, unsigned int a2,
    unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
}

#if W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE
// BEGIN double-buffer K64 async stage
__device__ __forceinline__ void w2f_cp_async_16(
    void* destination, const void* source, unsigned int valid_bytes) {
    const unsigned int shared_address =
        (unsigned int)__cvta_generic_to_shared(destination);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(shared_address), "l"(source), "r"(valid_bytes));
}

__device__ __forceinline__ void w2f_cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void w2f_cp_async_wait() {
    asm volatile("cp.async.wait_group 0;");
}

__device__ __forceinline__ void w2f_stage_async(
    W2FGemmScratch& scratch, const unsigned char* __restrict__ activation,
    const unsigned short* __restrict__ trellis, int m_start, int m_end,
    int m_local, unsigned int n_base, unsigned int absolute_k) {
    for (unsigned int vector = threadIdx.x;
         vector < W2F_M_TILE * 4; vector += blockDim.x) {
        const unsigned int row_local = vector >> 2;
        const unsigned int vector_k = vector & 3;
        const int row = m_start + m_local + (int)row_local;
        const void* source = activation;
        unsigned int valid_bytes = 0;
        if (row < m_end) {
            source = activation + (unsigned long long)row * W2F_GATE_UP_K +
                absolute_k + vector_k * 16;
            valid_bytes = 16;
        }
        w2f_cp_async_16(
            &scratch.activation[row_local][vector_k], source, valid_bytes);
    }
    for (unsigned int load = threadIdx.x; load < 128;
         load += blockDim.x) {
        const unsigned int k_tile = load >> 5;
        const unsigned int word = load & 31;
        const unsigned int kb = absolute_k / 16 + k_tile;
        const unsigned int nb = n_base / 16;
        const uint4* source = (const uint4*)trellis +
            ((unsigned long long)kb * (W2F_GATE_UP_N / 16) + nb) * 4 + word;
        w2f_cp_async_16(&scratch.trellis[k_tile][word], source, 16);
    }
}
// END double-buffer K64 async stage
#endif

__device__ __forceinline__ void w2f_compute_leg(
    const unsigned char* __restrict__ activation,
    const float* __restrict__ activation_scale,
    const unsigned short* __restrict__ trellis,
    __nv_bfloat16 output[W2F_M_TILE][W2F_N_TILE],
    W2FGemmScratch* scratch_buffers, int m_start, int m_end, int m_local,
    unsigned int n_base) {
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int group = lane >> 2;
    const unsigned int tid = lane & 3;
    const W2FLaneGeom geom = w2f_lane_geom(lane);
    float outer[4][2][4] = {};

#if W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE
    // BEGIN continuous K64 ring initial publication
    // This is the only unoverlapped stage. Every later K64 stage is issued
    // into the retired alternate buffer while the current buffer feeds MMA.
    w2f_stage_async(scratch_buffers[0], activation, trellis,
                    m_start, m_end, m_local, n_base, 0);
    w2f_cp_async_commit();
    w2f_cp_async_wait();
    __syncthreads();
    // END continuous K64 ring initial publication
    // BEGIN continuous K64 ring schedule
#endif
    for (unsigned int k_block = 0; k_block < W2F_GATE_UP_K;
         k_block += W2F_K_GROUP) {
        float inner[4][2][4] = {};
#if W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE && \
    !W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE
        // BEGIN double-buffer K128 schedule
        w2f_stage_async(scratch_buffers[0], activation, trellis,
                        m_start, m_end, m_local, n_base, k_block);
        w2f_cp_async_commit();
        w2f_cp_async_wait();
        __syncthreads();
#endif
#pragma unroll 1
        for (unsigned int k_stage = 0; k_stage < W2F_K_GROUP;
             k_stage += W2F_K_STAGE) {
            // BEGIN K64 staging selection
#if W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE
            const unsigned int absolute_k = k_block + k_stage;
            const unsigned int absolute_stage = absolute_k / W2F_K_STAGE;
            const unsigned int current_buffer = absolute_stage & 1;
            const unsigned int next_buffer = current_buffer ^ 1;
            const unsigned int next_k = absolute_k + W2F_K_STAGE;
            if (next_k < W2F_GATE_UP_K) {
                // The preceding block barrier retired every read from this
                // alternate buffer before the next async overwrite begins.
                w2f_stage_async(scratch_buffers[next_buffer], activation, trellis,
                                m_start, m_end, m_local, n_base, next_k);
                w2f_cp_async_commit();
            }
            W2FGemmScratch& scratch = scratch_buffers[current_buffer];
#elif W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE
            const unsigned int current_buffer = (k_stage / W2F_K_STAGE) & 1;
            const unsigned int next_buffer = current_buffer ^ 1;
            const unsigned int next_k_stage = k_stage + W2F_K_STAGE;
            if (next_k_stage < W2F_K_GROUP) {
                w2f_stage_async(scratch_buffers[next_buffer], activation, trellis,
                                m_start, m_end, m_local, n_base,
                                k_block + next_k_stage);
                w2f_cp_async_commit();
            }
            W2FGemmScratch& scratch = scratch_buffers[current_buffer];
#else
            W2FGemmScratch& scratch = scratch_buffers[0];
            for (unsigned int vector = threadIdx.x;
                 vector < W2F_M_TILE * 4; vector += blockDim.x) {
                const unsigned int row_local = vector >> 2;
                const unsigned int vector_k = vector & 3;
                const int row = m_start + m_local + (int)row_local;
                uint4 packed = {0, 0, 0, 0};
                if (row < m_end) {
                    const uint4* source = (const uint4*)(
                        activation + (unsigned long long)row * W2F_GATE_UP_K +
                        k_block + k_stage + vector_k * 16);
                    packed = *source;
                }
                scratch.activation[row_local][vector_k] = packed;
            }
            for (unsigned int load = threadIdx.x; load < 128;
                 load += blockDim.x) {
                const unsigned int k_tile = load >> 5;
                const unsigned int word = load & 31;
                const unsigned int kb = (k_block + k_stage) / 16 + k_tile;
                const unsigned int nb = n_base / 16;
                const uint4* source = (const uint4*)trellis +
                    ((unsigned long long)kb * (W2F_GATE_UP_N / 16) + nb) * 4 + word;
                scratch.trellis[k_tile][word] = *source;
            }
            __syncthreads();
#endif
            // END K64 staging selection

#pragma unroll
            for (unsigned int pair = 0; pair < 2; ++pair) {
                const unsigned int* tile0 =
                    (const unsigned int*)scratch.trellis[pair * 2] + warp * 16;
                const unsigned int* tile1 =
                    (const unsigned int*)scratch.trellis[pair * 2 + 1] + warp * 16;
                __half2 d01_0, d23_0, d45_0, d67_0;
                __half2 d01_1, d23_1, d45_1, d67_1;
                w2f_decode8(tile0, geom, d01_0, d23_0, d45_0, d67_0);
                w2f_decode8(tile1, geom, d01_1, d23_1, d45_1, d67_1);
                const unsigned int low0 = (unsigned int)w2f_encode_pair(d01_0) |
                    ((unsigned int)w2f_encode_pair(d23_0) << 16);
                const unsigned int low1 = (unsigned int)w2f_encode_pair(d01_1) |
                    ((unsigned int)w2f_encode_pair(d23_1) << 16);
                const unsigned int high0 = (unsigned int)w2f_encode_pair(d45_0) |
                    ((unsigned int)w2f_encode_pair(d67_0) << 16);
                const unsigned int high1 = (unsigned int)w2f_encode_pair(d45_1) |
                    ((unsigned int)w2f_encode_pair(d67_1) << 16);
                const unsigned int b00 = w2f_repack_b(low0, tid);
                const unsigned int b01 = w2f_repack_b(low1, tid);
                const unsigned int b10 = w2f_repack_b(high0, tid);
                const unsigned int b11 = w2f_repack_b(high1, tid);
#pragma unroll
                for (unsigned int mt = 0; mt < 4; ++mt) {
                    const unsigned int row0 = mt * 16 + group;
                    const unsigned int row1 = row0 + 8;
                    const unsigned int byte_k = pair * 32;
                    const unsigned int stride = 5 * sizeof(uint4);
                    const unsigned char* a = (const unsigned char*)scratch.activation;
                    const unsigned int a0 = *(const unsigned int*)(
                        a + row0 * stride + byte_k + 4 * tid);
                    const unsigned int a1 = *(const unsigned int*)(
                        a + row1 * stride + byte_k + 4 * tid);
                    const unsigned int a2 = *(const unsigned int*)(
                        a + row0 * stride + byte_k + 16 + 4 * tid);
                    const unsigned int a3 = *(const unsigned int*)(
                        a + row1 * stride + byte_k + 16 + 4 * tid);
                    w2f_mma(inner[mt][0], a0, a1, a2, a3, b00, b01);
                    w2f_mma(inner[mt][1], a0, a1, a2, a3, b10, b11);
                }
            }
#if W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE
            if (next_k < W2F_GATE_UP_K) w2f_cp_async_wait();
#elif W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE
            if (next_k_stage < W2F_K_GROUP) w2f_cp_async_wait();
#endif
            // Publishes the next stage and retires every current-buffer read
            // before the two buffers exchange producer/consumer ownership.
            __syncthreads();
        }
#if W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE
        // END continuous K64 ring schedule
#elif W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE
        // END double-buffer K128 schedule
#endif

#pragma unroll
        for (unsigned int mt = 0; mt < 4; ++mt) {
            const unsigned int row0 = mt * 16 + group;
            const unsigned int row1 = row0 + 8;
            const int global0 = m_start + m_local + (int)row0;
            const int global1 = m_start + m_local + (int)row1;
            float scale0 = global0 < m_end
                ? activation_scale[(unsigned long long)global0 * 32 + k_block / 128]
                : 0.0f;
            float scale1 = global1 < m_end
                ? activation_scale[(unsigned long long)global1 * 32 + k_block / 128]
                : 0.0f;
            if (!(scale0 >= 1.0e-12f) || !isfinite(scale0)) scale0 = 0.0f;
            if (!(scale1 >= 1.0e-12f) || !isfinite(scale1)) scale1 = 0.0f;
            const float factor0 = scale0 * W2F_INV_WEIGHT_SCALE;
            const float factor1 = scale1 * W2F_INV_WEIGHT_SCALE;
#pragma unroll
            for (unsigned int nt = 0; nt < 2; ++nt) {
                outer[mt][nt][0] += inner[mt][nt][0] * factor0;
                outer[mt][nt][1] += inner[mt][nt][1] * factor0;
                outer[mt][nt][2] += inner[mt][nt][2] * factor1;
                outer[mt][nt][3] += inner[mt][nt][3] * factor1;
            }
        }
    }

#pragma unroll
    for (unsigned int mt = 0; mt < 4; ++mt) {
#pragma unroll
        for (unsigned int nt = 0; nt < 2; ++nt) {
            const unsigned int col = warp * 16 + nt * 8 + tid * 2;
            const unsigned int row0 = mt * 16 + group;
            const unsigned int row1 = row0 + 8;
            if (m_start + m_local + (int)row0 < m_end) {
                output[row0][col] = __float2bfloat16(outer[mt][nt][0]);
                output[row0][col + 1] = __float2bfloat16(outer[mt][nt][1]);
            }
            if (m_start + m_local + (int)row1 < m_end) {
                output[row1][col] = __float2bfloat16(outer[mt][nt][2]);
                output[row1][col + 1] = __float2bfloat16(outer[mt][nt][3]);
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void w2f_emit_group(
    float value0, float value1, float value2, float value3,
    unsigned char* __restrict__ output_fp8,
    float* __restrict__ output_scale, unsigned int row,
    unsigned int chunk, unsigned int warp, unsigned int lane,
    float abs_values[W2F_HROW_WARPS][W2F_K_GROUP]) {
    // BEGIN fused down-a8 quant reduction
    // The explicit BF16 round/re-expand and reduction reproduce the standalone
    // down-A8 emitter for exactly one row/K128 group.
    const float rounded0 = __bfloat162float(__float2bfloat16(value0));
    const float rounded1 = __bfloat162float(__float2bfloat16(value1));
    const float rounded2 = __bfloat162float(__float2bfloat16(value2));
    const float rounded3 = __bfloat162float(__float2bfloat16(value3));
    abs_values[warp][4 * lane + 0] = fabsf(rounded0);
    abs_values[warp][4 * lane + 1] = fabsf(rounded1);
    abs_values[warp][4 * lane + 2] = fabsf(rounded2);
    abs_values[warp][4 * lane + 3] = fabsf(rounded3);
    __syncwarp();
    // fused down-a8 quant reduction
    float max0 = abs_values[warp][lane + 0];
    float max1 = abs_values[warp][lane + 32];
    float max2 = abs_values[warp][lane + 64];
    float max3 = abs_values[warp][lane + 96];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        max0 = fmaxf(max0, __shfl_down_sync(0xffffffffu, max0, offset));
        max1 = fmaxf(max1, __shfl_down_sync(0xffffffffu, max1, offset));
        max2 = fmaxf(max2, __shfl_down_sync(0xffffffffu, max2, offset));
        max3 = fmaxf(max3, __shfl_down_sync(0xffffffffu, max3, offset));
    }
    float scale = 0.0f;
    if (lane == 0) {
        float global_max = 0.0f;
        global_max = fmaxf(global_max, max0);
        global_max = fmaxf(global_max, max1);
        global_max = fmaxf(global_max, max2);
        global_max = fmaxf(global_max, max3);
        scale = global_max / W2F_E4M3_MAX;
        if (scale < W2F_MIN_SCALE) scale = W2F_MIN_SCALE;
        output_scale[(unsigned long long)row * 16 + chunk] = scale;
    }
    scale = __shfl_sync(0xffffffffu, scale, 0);
    uchar4 packed;
    packed.x = (unsigned char)__nv_cvt_float_to_fp8(
        fmaxf(fminf(rounded0 / scale, W2F_E4M3_MAX), -W2F_E4M3_MAX),
        __NV_SATFINITE, __NV_E4M3);
    packed.y = (unsigned char)__nv_cvt_float_to_fp8(
        fmaxf(fminf(rounded1 / scale, W2F_E4M3_MAX), -W2F_E4M3_MAX),
        __NV_SATFINITE, __NV_E4M3);
    packed.z = (unsigned char)__nv_cvt_float_to_fp8(
        fmaxf(fminf(rounded2 / scale, W2F_E4M3_MAX), -W2F_E4M3_MAX),
        __NV_SATFINITE, __NV_E4M3);
    packed.w = (unsigned char)__nv_cvt_float_to_fp8(
        fmaxf(fminf(rounded3 / scale, W2F_E4M3_MAX), -W2F_E4M3_MAX),
        __NV_SATFINITE, __NV_E4M3);
    *(uchar4*)(output_fp8 + (unsigned long long)row * W2F_GATE_UP_N +
               chunk * W2F_K_GROUP + 4 * lane) = packed;
    __syncwarp();
    // END fused down-a8 quant reduction
}

extern "C" __global__ __launch_bounds__(256)
void W2A8_KERNEL_NAME(
    const unsigned char* __restrict__ gate_fp8,
    const float* __restrict__ gate_scale,
    const unsigned char* __restrict__ up_fp8,
    const float* __restrict__ up_scale,
    const unsigned long long* __restrict__ gate_trellis_tab,
    const unsigned long long* __restrict__ up_trellis_tab,
    const unsigned long long* __restrict__ gate_svh_tab,
    const unsigned long long* __restrict__ up_svh_tab,
    const unsigned long long* __restrict__ down_suh_tab,
    unsigned char* __restrict__ down_fp8,
    float* __restrict__ down_scale,
    const int* __restrict__ expert_offsets,
    unsigned int num_experts, unsigned int total_rows, unsigned int N, unsigned int K,
    unsigned int bits, unsigned int persistent_mode) {
    // BEGIN fused exact-shape guards
    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1) return;
    if (gridDim.y != 1 || gridDim.z != 1) return;
    if (N != W2F_GATE_UP_N || K != W2F_GATE_UP_K) return;
    if (bits != 2 || persistent_mode != 1 || num_experts != W2F_EXPERTS ||
        total_rows == 0) return;
    constexpr unsigned int n_tiles = W2F_GATE_UP_N / W2F_N_TILE;
    if ((unsigned long long)gridDim.x !=
        (unsigned long long)num_experts * n_tiles) return;
    if (!gate_fp8 || !gate_scale || !up_fp8 || !up_scale ||
        !gate_trellis_tab || !up_trellis_tab || !gate_svh_tab ||
        !up_svh_tab || !down_suh_tab || !down_fp8 || !down_scale ||
        !expert_offsets) return;
    if (total_rows > (unsigned int)INT_MAX) return;

    // Every warp independently validates the complete fixed-size route before
    // any thread can resolve a table or store output. The immutable vector is
    // striped across its 32 lanes, avoiding a second launch, atomics, or a
    // block barrier in the measured candidate.
    const unsigned int route_lane = threadIdx.x & 31;
    bool routing_invalid = false;
#pragma unroll
    for (unsigned int routing_index = route_lane;
         routing_index < W2F_EXPERTS; routing_index += 32) {
        const int route_start = expert_offsets[routing_index];
        const int route_end = expert_offsets[routing_index + 1];
        if (route_start < 0 || route_end < route_start ||
            (unsigned int)route_end > total_rows ||
            (routing_index == 0 && route_start != 0) ||
            (routing_index + 1 == W2F_EXPERTS &&
             route_end != (int)total_rows)) {
            routing_invalid = true;
        }
    }
    if (__ballot_sync(0xffffffffu, routing_invalid) != 0) return;

    const unsigned int n_tile = blockIdx.x % n_tiles;
    const unsigned int expert = blockIdx.x / n_tiles;
    const int m_start = expert_offsets[expert];
    const int m_end = expert_offsets[expert + 1];
    if (m_start == m_end) return;
    const unsigned short* gate_trellis =
        (const unsigned short*)gate_trellis_tab[expert];
    const unsigned short* up_trellis =
        (const unsigned short*)up_trellis_tab[expert];
    const __half* gate_svh = (const __half*)gate_svh_tab[expert];
    const __half* up_svh = (const __half*)up_svh_tab[expert];
    const __half* down_suh = (const __half*)down_suh_tab[expert];
    if (!gate_trellis || !up_trellis || !gate_svh || !up_svh || !down_suh) return;
    // END fused exact-shape guards

    __shared__ W2FShared shared;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int n_base = n_tile * W2F_N_TILE;
    gate_svh += n_base;
    up_svh += n_base;
    down_suh += n_base;

    for (int m_local = 0; m_start + m_local < m_end; m_local += W2F_M_TILE) {
        // BEGIN fused bf16 round-reexpand barriers
        // BEGIN fused gate trellis pass
        // gate_trellis_tab resolved gate_trellis above; this leg alone owns it.
        w2f_compute_leg(gate_fp8, gate_scale, gate_trellis, shared.gate,
                        shared.work.gemm, m_start, m_end, m_local, n_base);
        // END fused gate trellis pass
        // w2f_compute_leg retires shared output with __syncthreads().
        // BEGIN fused up trellis pass
        // up_trellis_tab resolved up_trellis above; this leg alone owns it.
        w2f_compute_leg(up_fp8, up_scale, up_trellis, shared.up,
                        shared.work.gemm, m_start, m_end, m_local, n_base);
        // END fused up trellis pass

        // BEGIN fused h128 swiglu down-pre order
        for (unsigned int row_local = warp; row_local < W2F_M_TILE;
             row_local += W2F_HROW_WARPS) {
            const int row = m_start + m_local + (int)row_local;
            if (row >= m_end) continue;
            const unsigned int col = 4 * lane;
            float g0 = __bfloat162float(shared.gate[row_local][col + 0]);
            float g1 = __bfloat162float(shared.gate[row_local][col + 1]);
            float g2 = __bfloat162float(shared.gate[row_local][col + 2]);
            float g3 = __bfloat162float(shared.gate[row_local][col + 3]);
            float u0 = __bfloat162float(shared.up[row_local][col + 0]);
            float u1 = __bfloat162float(shared.up[row_local][col + 1]);
            float u2 = __bfloat162float(shared.up[row_local][col + 2]);
            float u3 = __bfloat162float(shared.up[row_local][col + 3]);
            w2f_had128(g0, g1, g2, g3, lane);
            w2f_had128(u0, u1, u2, u3, lane);
            g0 = __bfloat162float(__float2bfloat16(
                g0 * W2F_RSQRT128 * __half2float(gate_svh[col + 0])));
            g1 = __bfloat162float(__float2bfloat16(
                g1 * W2F_RSQRT128 * __half2float(gate_svh[col + 1])));
            g2 = __bfloat162float(__float2bfloat16(
                g2 * W2F_RSQRT128 * __half2float(gate_svh[col + 2])));
            g3 = __bfloat162float(__float2bfloat16(
                g3 * W2F_RSQRT128 * __half2float(gate_svh[col + 3])));
            u0 = __bfloat162float(__float2bfloat16(
                u0 * W2F_RSQRT128 * __half2float(up_svh[col + 0])));
            u1 = __bfloat162float(__float2bfloat16(
                u1 * W2F_RSQRT128 * __half2float(up_svh[col + 1])));
            u2 = __bfloat162float(__float2bfloat16(
                u2 * W2F_RSQRT128 * __half2float(up_svh[col + 2])));
            u3 = __bfloat162float(__float2bfloat16(
                u3 * W2F_RSQRT128 * __half2float(up_svh[col + 3])));
            g0 = fminf(g0, W2F_SWIGLU_LIMIT);
            g1 = fminf(g1, W2F_SWIGLU_LIMIT);
            g2 = fminf(g2, W2F_SWIGLU_LIMIT);
            g3 = fminf(g3, W2F_SWIGLU_LIMIT);
            u0 = fminf(fmaxf(u0, -W2F_SWIGLU_LIMIT), W2F_SWIGLU_LIMIT);
            u1 = fminf(fmaxf(u1, -W2F_SWIGLU_LIMIT), W2F_SWIGLU_LIMIT);
            u2 = fminf(fmaxf(u2, -W2F_SWIGLU_LIMIT), W2F_SWIGLU_LIMIT);
            u3 = fminf(fmaxf(u3, -W2F_SWIGLU_LIMIT), W2F_SWIGLU_LIMIT);
            const __nv_bfloat16 a0 =
                __float2bfloat16(g0 * (1.0f / (1.0f + __expf(-g0))) * u0);
            const __nv_bfloat16 a1 =
                __float2bfloat16(g1 * (1.0f / (1.0f + __expf(-g1))) * u1);
            const __nv_bfloat16 a2 =
                __float2bfloat16(g2 * (1.0f / (1.0f + __expf(-g2))) * u2);
            const __nv_bfloat16 a3 =
                __float2bfloat16(g3 * (1.0f / (1.0f + __expf(-g3))) * u3);
            float d0 = __bfloat162float(a0) * __half2float(down_suh[col + 0]);
            float d1 = __bfloat162float(a1) * __half2float(down_suh[col + 1]);
            float d2 = __bfloat162float(a2) * __half2float(down_suh[col + 2]);
            float d3 = __bfloat162float(a3) * __half2float(down_suh[col + 3]);
            w2f_had128(d0, d1, d2, d3, lane);
            w2f_emit_group(
                d0 * W2F_RSQRT128, d1 * W2F_RSQRT128,
                d2 * W2F_RSQRT128, d3 * W2F_RSQRT128,
                down_fp8, down_scale, (unsigned int)row, n_tile, warp, lane,
                shared.work.abs_values);
        }
        // END fused h128 swiglu down-pre order
        // END fused bf16 round-reexpand barriers
        __syncthreads();
    }
}
