// SPDX-License-Identifier: AGPL-3.0-only
//
// Compile-only SM121a EXL3 W2A8 N256 grouped-prefill component.
//
// A_fp8 is the sorted, post-H128 activation produced with one E4M3 scale per
// 128 K values. Trellis weights remain compressed and are decoded directly
// into registers. The algorithm stays outside kernels/gb10/common; one exact
// DeepSeek down-shape wrapper exposes it only behind a strict default-off host
// gate while native numeric and timing qualification remain pending.

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
#ifndef W2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE
#define W2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE 0
#endif
static_assert(W2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE == 0 ||
              W2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE == 1,
              "single-warp route guard candidate must be boolean");
#ifndef W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE
#define W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE 0
#endif
static_assert(W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE == 0 ||
              W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE == 1,
              "N256 down double-buffer candidate must be boolean");
#ifndef W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE
#define W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE 0
#endif
static_assert(W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE == 0 ||
              W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE == 1,
              "N256 down continuous-ring candidate must be boolean");
#if W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE
static_assert(W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE == 1,
              "N256 continuous ring requires the two staging buffers");
static_assert(W2A8_PACKED_E4M3_CANDIDATE == 1,
              "N256 continuous ring freezes packed E4M3 on");
#endif

#define W2A8_M_TILE 64
#define W2A8_N_TILE 256
#define W2A8_THREADS 512
#define W2A8_WARPS (W2A8_THREADS / 32)
#define W2A8_A_VECTORS (W2A8_M_TILE * 4)
#define W2A8_T_WORDS (W2A8_N_TILE / 4)
#define W2A8_T_VECTORS (4 * W2A8_T_WORDS)
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
static_assert(W2A8_FIXED_N % W2A8_N_TILE == 0, "N must be N256 aligned");
static_assert(W2A8_FIXED_K % W2A8_SCALE_GROUP == 0,
              "K must be activation-scale aligned");
static_assert(W2A8_SCALE_GROUP == 2 * W2A8_K_STAGE,
              "two K64 stages must share one activation scale");
static_assert(W2A8_K_STAGE == 2 * W2A8_K_PAIR,
              "each K64 stage must issue two K32 pairs");
static_assert(W2A8_WARPS * 16 == W2A8_N_TILE,
              "each of 16 warps must own one unique N16 strip");
static_assert(W2A8_A_VECTORS == W2A8_THREADS / 2,
              "the first 256 threads must stage one unique activation uint4");
static_assert(W2A8_T_VECTORS == W2A8_THREADS / 2,
              "the first 256 threads must stage one unique trellis uint4");
#if W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE
static_assert(W2A8_FIXED_N == 4096 && W2A8_FIXED_K == 2048,
              "N256 double buffering is an exact down-shape experiment");
#endif
#if W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE
static_assert(W2A8_FIXED_N == 4096 && W2A8_FIXED_K == 2048,
              "N256 continuous ring is an exact down-shape experiment");
#endif

union W2A8Half2Bits {
    unsigned int bits;
    __half2 values;
};

struct W2A8LaneGeom {
    int ia;
    int ib;
    int shift;
};

struct __align__(16) W2A8GemmScratch {
    uint4 activation[W2A8_M_TILE][5];
    uint4 trellis[4][W2A8_T_WORDS];
};

static_assert(sizeof(W2A8GemmScratch) == 9 * 1024,
              "one N256 K64 scratch stage must remain 9 KiB");

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
#if W2A8_PACKED_E4M3_CANDIDATE
    // K2 decode is finite and strictly inside (-4, 4), so scaling by 2^4 is
    // an exact half exponent shift. Converting the scaled packed halves is
    // therefore byte-identical to the incumbent half->float, *16, FP8 path.
    W2A8Half2Bits scale = {0x4c004c00u};  // half2(16.0, 16.0)
    W2A8Half2Bits scaled;
    scaled.values = __hmul2(pair, scale.values);
    unsigned short encoded;
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                 : "=h"(encoded) : "r"(scaled.bits));
    return encoded;
#else
    const float value0 = __half2float(__low2half(pair)) * W2A8_WEIGHT_SCALE;
    const float value1 = __half2float(__high2half(pair)) * W2A8_WEIGHT_SCALE;
    unsigned short encoded;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(encoded) : "f"(value1), "f"(value0));
    return encoded;
#endif
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

__device__ __forceinline__ bool w2a8_route_lane_invalid(
    unsigned int route_lane, unsigned int num_experts,
    unsigned int total_rows, const int* __restrict__ expert_offsets) {
    bool routing_invalid = false;
#pragma unroll 1
    for (unsigned int routing_index = route_lane;
         routing_index < num_experts; routing_index += 32) {
        const int route_start = expert_offsets[routing_index];
        const int route_end = expert_offsets[routing_index + 1];
        if (route_start < 0 || route_end < route_start ||
            (unsigned int)route_end > total_rows ||
            (routing_index == 0 && route_start != 0) ||
            (routing_index + 1 == num_experts &&
             route_end != (int)total_rows)) {
            routing_invalid = true;
        }
    }
    return routing_invalid;
}

#if W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE
// BEGIN N256 down double-buffer K64 async stage
__device__ __forceinline__ void w2a8_cp_async_16(
    void* destination, const void* source, unsigned int valid_bytes) {
    const unsigned int shared_address =
        (unsigned int)__cvta_generic_to_shared(destination);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(shared_address), "l"(source), "r"(valid_bytes));
}

__device__ __forceinline__ void w2a8_cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void w2a8_cp_async_wait() {
    asm volatile("cp.async.wait_group 0;");
}

__device__ __forceinline__ void w2a8_stage_async(
    W2A8GemmScratch& scratch,
    const unsigned char* __restrict__ activation,
    const unsigned short* __restrict__ trellis,
    int m_start, int m_end, int m_local,
    unsigned int n_base, unsigned int absolute_k) {
    // Every one of the 512 threads owns exactly one 16-byte copy. The branch is
    // warp-uniform: the lower eight warps stage A and the upper eight stage T.
    void* destination;
    const void* source;
    unsigned int valid_bytes;
    if (threadIdx.x < W2A8_A_VECTORS) {
        const unsigned int vector = threadIdx.x;
        const unsigned int row_local = vector >> 2;
        const unsigned int vector_k = vector & 3;
        const int row = m_start + m_local + (int)row_local;
        destination = &scratch.activation[row_local][vector_k];
        source = activation;
        valid_bytes = 0;
        if (row < m_end) {
            source = activation + (unsigned long long)row * W2A8_FIXED_K +
                absolute_k + vector_k * 16;
            valid_bytes = 16;
        }
    } else {
        const unsigned int load = threadIdx.x - W2A8_A_VECTORS;
        const unsigned int k_tile = load / W2A8_T_WORDS;
        const unsigned int word = load % W2A8_T_WORDS;
        const unsigned int kb = absolute_k / 16 + k_tile;
        const unsigned int nb = n_base / 16;
        destination = &scratch.trellis[k_tile][word];
        source = (const uint4*)trellis +
            ((unsigned long long)kb * (W2A8_FIXED_N / 16) + nb) * 4 + word;
        valid_bytes = 16;
    }
    w2a8_cp_async_16(destination, source, valid_bytes);
}
// END N256 down double-buffer K64 async stage
#endif

#if W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE
#define W2A8_CURRENT_A scratch.activation
#define W2A8_CURRENT_T scratch.trellis
#else
#define W2A8_CURRENT_A smem_A
#define W2A8_CURRENT_T smem_T
#endif

extern "C" __global__ __launch_bounds__(W2A8_THREADS) void W2A8_KERNEL_NAME(
    const unsigned char* __restrict__ A_fp8,
    const float* __restrict__ a_scale,
    const unsigned long long* __restrict__ trellis_tab,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    unsigned int num_experts, unsigned int total_rows, unsigned int N,
    unsigned int K,
    unsigned int bits, unsigned int persistent_mode) {
    if (blockDim.x != W2A8_THREADS || blockDim.y != 1 || blockDim.z != 1) return;
    if (gridDim.y != 1 || gridDim.z != 1) return;
    if (N != W2A8_FIXED_N || K != W2A8_FIXED_K) return;
    if (bits != 2 || persistent_mode != 1) return;
    if (num_experts == 0 || total_rows == 0) return;
    constexpr unsigned int n_tiles = W2A8_FIXED_N / W2A8_N_TILE;
    if ((unsigned long long)gridDim.x !=
        (unsigned long long)num_experts * n_tiles) return;
    if (A_fp8 == nullptr || a_scale == nullptr || trellis_tab == nullptr ||
        C == nullptr || expert_offsets == nullptr) return;
    if (total_rows > (unsigned int)INT_MAX) return;

    // BEGIN route validation selection
#if W2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE
    // BEGIN single-warp route guard candidate
    // Warp zero scans the complete route, publishes one uniform block result,
    // and synchronizes every thread before any pointer resolution or output.
    __shared__ unsigned int route_invalid_block;
    if (threadIdx.x < 32) {
        const bool routing_invalid = w2a8_route_lane_invalid(
            threadIdx.x, num_experts, total_rows, expert_offsets);
        const unsigned int invalid_mask =
            __ballot_sync(0xffffffffu, routing_invalid);
        if (threadIdx.x == 0) route_invalid_block = invalid_mask != 0;
    }
    __syncthreads();
    if (route_invalid_block != 0) return;
    // END single-warp route guard candidate
#else
    // BEGIN every-warp route guard incumbent
    // Every warp independently scans the complete route. The warp-local ballot
    // keeps the incumbent decision uniform without a block barrier.
    const unsigned int route_lane = threadIdx.x & 31;
    const bool routing_invalid = w2a8_route_lane_invalid(
        route_lane, num_experts, total_rows, expert_offsets);
    if (__ballot_sync(0xffffffffu, routing_invalid) != 0) return;
    // END every-warp route guard incumbent
#endif
    // END route validation selection

    const unsigned int n_tile = blockIdx.x % n_tiles;
    const unsigned int expert_id = blockIdx.x / n_tiles;
    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    if (m_start == m_end) return;
    const unsigned short* trellis =
        (const unsigned short*)trellis_tab[expert_id];
    if (trellis == nullptr) return;

    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int group = lane >> 2;
    const unsigned int tid = lane & 3;
    const unsigned int n_base = n_tile * W2A8_N_TILE;
    const W2A8LaneGeom lane_geom = w2a8_lane_geom(lane);

#if W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE
    __shared__ __align__(16) W2A8GemmScratch scratch_buffers[2];
#else
    __shared__ __align__(16) uint4 smem_A[W2A8_M_TILE][5];
    __shared__ __align__(16) uint4 smem_T[4][W2A8_T_WORDS];
#endif

    for (int m_local = 0; m_start + m_local < m_end; m_local += W2A8_M_TILE) {
        float outer[4][2][4] = {};

#if W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE
        // BEGIN N256 down continuous K64 ring initial publication
        w2a8_stage_async(scratch_buffers[0], A_fp8, trellis,
                         m_start, m_end, m_local, n_base, 0);
        w2a8_cp_async_commit();
        w2a8_cp_async_wait();
        __syncthreads();
        // END N256 down continuous K64 ring initial publication
        // BEGIN N256 down continuous K64 ring schedule
#endif
        for (unsigned int k_block = 0; k_block < W2A8_FIXED_K;
             k_block += W2A8_SCALE_GROUP) {
            float inner[4][2][4] = {};

#if W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE && \
    !W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE
            // BEGIN N256 down double-buffer K128 schedule
            w2a8_stage_async(scratch_buffers[0], A_fp8, trellis,
                             m_start, m_end, m_local, n_base, k_block);
            w2a8_cp_async_commit();
            w2a8_cp_async_wait();
            __syncthreads();
#endif
#pragma unroll 1
            for (unsigned int k_stage = 0; k_stage < W2A8_SCALE_GROUP;
                 k_stage += W2A8_K_STAGE) {
                // BEGIN K64 staging selection
#if W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE
                const unsigned int absolute_k = k_block + k_stage;
                const unsigned int absolute_stage = absolute_k / W2A8_K_STAGE;
                const unsigned int current_buffer = absolute_stage & 1;
                const unsigned int next_buffer = current_buffer ^ 1;
                const unsigned int next_k = absolute_k + W2A8_K_STAGE;
                if (next_k < W2A8_FIXED_K) {
                    // The preceding block barrier retired all reads from this
                    // alternate slot before its next async overwrite begins.
                    w2a8_stage_async(
                        scratch_buffers[next_buffer], A_fp8, trellis,
                        m_start, m_end, m_local, n_base, next_k);
                    w2a8_cp_async_commit();
                }
                W2A8GemmScratch& scratch = scratch_buffers[current_buffer];
#elif W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE
                const unsigned int current_buffer =
                    k_stage / W2A8_K_STAGE;
                if (k_stage == 0) {
                    w2a8_stage_async(
                        scratch_buffers[1], A_fp8, trellis,
                        m_start, m_end, m_local, n_base,
                        k_block + W2A8_K_STAGE);
                    w2a8_cp_async_commit();
                }
                W2A8GemmScratch& scratch = scratch_buffers[current_buffer];
#else
                // A ownership is a bijection: threads [0,256) each stage one
                // (row, K16) uint4; the upper eight warps issue no duplicate load.
                for (unsigned int vector = threadIdx.x;
                     vector < W2A8_A_VECTORS; vector += blockDim.x) {
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

                // Trellis ownership is likewise a bijection over four K16 rows
                // by 64 uint4 words (sixteen N16 strips) for this N256 tile.
                for (unsigned int load = threadIdx.x; load < W2A8_T_VECTORS;
                     load += blockDim.x) {
                    const unsigned int k_tile = load / W2A8_T_WORDS;
                    const unsigned int word = load % W2A8_T_WORDS;
                    const unsigned int kb = (k_block + k_stage) / 16 + k_tile;
                    const unsigned int nb = n_base / 16;
                    const uint4* src = (const uint4*)trellis +
                        ((unsigned long long)kb * (W2A8_FIXED_N / 16) + nb) * 4 + word;
                    smem_T[k_tile][word] = *src;
                }
                __syncthreads();
#endif
                // END K64 staging selection

#pragma unroll
                for (unsigned int pair = 0; pair < 2; ++pair) {
                    // warp*16 partitions each K16 row into sixteen disjoint
                    // 16-u32 fragments, exactly matching the N16 output owner.
                    const unsigned int* tile0 =
                        (const unsigned int*)W2A8_CURRENT_T[pair * 2] + warp * 16;
                    const unsigned int* tile1 =
                        (const unsigned int*)W2A8_CURRENT_T[pair * 2 + 1] + warp * 16;
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
                        const unsigned char* a =
                            (const unsigned char*)W2A8_CURRENT_A;
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
#if W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE
                if (next_k < W2A8_FIXED_K) w2a8_cp_async_wait();
#elif W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE
                if (k_stage == 0) w2a8_cp_async_wait();
#endif
                // Publishes the next K64 stage and retires every read from the
                // current slot before producer/consumer ownership exchanges.
                __syncthreads();
            }
#if W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE
            // END N256 down continuous K64 ring schedule
#endif
#if W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE && \
    !W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE
            // END N256 down double-buffer K128 schedule
#endif

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

#undef W2A8_CURRENT_A
#undef W2A8_CURRENT_T
