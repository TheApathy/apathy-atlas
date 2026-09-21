// SPDX-License-Identifier: AGPL-3.0-only
//
// EXL3 K2/K3 grouped prefill GEMM for SM121.
//
// This is the direct P2 bridge: each warp decodes one 16x16 trellis tile
// directly into BF16 B-fragment registers and reuses it across eight
// m16n8k16 MMAs. No decoded weight is materialized in shared or global
// memory. Input/output H128 rotations remain in the existing row kernels.
//
// Trellis decode is derived from ExLlamaV3 (MIT, Copyright 2025 Turboderp),
// vendored reference revision 704aefd7. Tile order and 3INST constants match
// kernels/gb10/common/exl3_gemv.cu and docs/kernels/exl3-gemv.md.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifndef EXL3_PF_M_TILE
#define EXL3_PF_M_TILE 64
#endif
#ifndef EXL3_PF_N_TILE
#define EXL3_PF_N_TILE 64
#endif
#ifndef EXL3_PF_KERNEL_NAME
#define EXL3_PF_KERNEL_NAME exl3_grouped_prefill
#endif
#ifndef EXL3_PF_K_STEP
#define EXL3_PF_K_STEP 16
#endif
#ifndef EXL3_PF_ASYNC_STAGE
#define EXL3_PF_ASYNC_STAGE 0
#endif
#ifndef EXL3_PF_FIXED_BITS
#define EXL3_PF_FIXED_BITS 0
#endif
#ifndef EXL3_PF_FIXED_N
#define EXL3_PF_FIXED_N 0
#endif
#ifndef EXL3_PF_FIXED_K
#define EXL3_PF_FIXED_K 0
#endif
#ifndef EXL3_PF_FIXED_PERSISTENT
#define EXL3_PF_FIXED_PERSISTENT 0
#endif
#ifndef EXL3_PF_FIXED_IDENTITY_ROWS
#define EXL3_PF_FIXED_IDENTITY_ROWS 0
#endif
#ifndef EXL3_PF_EXACT_FULL_GRID
#define EXL3_PF_EXACT_FULL_GRID 0
#endif
#ifndef EXL3_PF_LAUNCH_BOUNDS
#define EXL3_PF_LAUNCH_BOUNDS 0
#endif
#ifndef EXL3_PF_PACKED_BF16_STORE
#define EXL3_PF_PACKED_BF16_STORE 0
#endif
#ifndef EXL3_PF_K2_LO_WINDOWS
#define EXL3_PF_K2_LO_WINDOWS 0
#endif
#define EXL3_PF_N_WARPS (EXL3_PF_N_TILE / 16)
#define EXL3_PF_K_TILES (EXL3_PF_K_STEP / 16)
#define EXL3_PF_PAD 2
#define EXL3_PF_MAX_BITS 3
#if EXL3_PF_FIXED_BITS
#define EXL3_PF_SMEM_BITS EXL3_PF_FIXED_BITS
#else
#define EXL3_PF_SMEM_BITS EXL3_PF_MAX_BITS
#endif
#define EXL3_PF_MCG_MULT 0xCBAC1FEDu

static_assert(EXL3_PF_M_TILE == 64 || EXL3_PF_M_TILE == 128,
              "EXL3 grouped prefill supports only M64/M128");
static_assert(EXL3_PF_N_TILE == 64 || EXL3_PF_N_TILE == 128 ||
              EXL3_PF_N_TILE == 256,
              "EXL3 grouped prefill supports only N64/N128/N256");
static_assert(EXL3_PF_K_STEP == 16 || EXL3_PF_K_STEP == 64,
              "EXL3 grouped prefill supports only K16/K64 stages");
static_assert(EXL3_PF_ASYNC_STAGE == 0 || EXL3_PF_ASYNC_STAGE == 1,
              "EXL3 grouped prefill async-stage flag must be boolean");
static_assert(EXL3_PF_FIXED_BITS == 0 || EXL3_PF_FIXED_BITS == 2 ||
              EXL3_PF_FIXED_BITS == 3,
              "EXL3 grouped prefill fixed bits must be 0, 2, or 3");
static_assert(EXL3_PF_FIXED_N == 0 || EXL3_PF_FIXED_N % EXL3_PF_N_TILE == 0,
              "EXL3 grouped prefill fixed N must be N64 aligned");
static_assert(EXL3_PF_FIXED_K == 0 || EXL3_PF_FIXED_K % EXL3_PF_K_STEP == 0,
              "EXL3 grouped prefill fixed K must be stage aligned");
static_assert((EXL3_PF_FIXED_N == 0) == (EXL3_PF_FIXED_K == 0),
              "EXL3 grouped prefill fixed N and K must be paired");
static_assert(EXL3_PF_FIXED_PERSISTENT == 0 || EXL3_PF_FIXED_PERSISTENT == 1,
              "EXL3 grouped prefill fixed-persistent flag must be boolean");
static_assert(!EXL3_PF_FIXED_PERSISTENT || EXL3_PF_FIXED_N != 0,
              "EXL3 grouped prefill fixed-persistent needs a fixed shape");
static_assert(EXL3_PF_FIXED_IDENTITY_ROWS == 0 ||
              EXL3_PF_FIXED_IDENTITY_ROWS == 1,
              "EXL3 grouped prefill fixed-identity-rows flag must be boolean");
static_assert(EXL3_PF_EXACT_FULL_GRID == 0 || EXL3_PF_EXACT_FULL_GRID == 1,
              "EXL3 grouped prefill exact-full-grid flag must be boolean");
static_assert(!EXL3_PF_EXACT_FULL_GRID || EXL3_PF_FIXED_BITS == 2,
              "EXL3 exact-full-grid requires fixed K2");
static_assert(!EXL3_PF_EXACT_FULL_GRID || EXL3_PF_M_TILE == 64,
              "EXL3 exact-full-grid requires M64");
static_assert(!EXL3_PF_EXACT_FULL_GRID || EXL3_PF_FIXED_PERSISTENT == 1,
              "EXL3 exact-full-grid requires fixed persistent mode");
static_assert(!EXL3_PF_EXACT_FULL_GRID || EXL3_PF_FIXED_N != 0,
              "EXL3 exact-full-grid requires fixed N");
static_assert(!EXL3_PF_EXACT_FULL_GRID || EXL3_PF_FIXED_K != 0,
              "EXL3 exact-full-grid requires fixed K");
static_assert(!EXL3_PF_EXACT_FULL_GRID ||
              (((EXL3_PF_FIXED_N / EXL3_PF_N_TILE) &
                ((EXL3_PF_FIXED_N / EXL3_PF_N_TILE) - 1)) == 0),
              "EXL3 fixed N-tile count must be a power of two");
static_assert(EXL3_PF_LAUNCH_BOUNDS == 0 || EXL3_PF_LAUNCH_BOUNDS == 128 ||
              EXL3_PF_LAUNCH_BOUNDS == 256 || EXL3_PF_LAUNCH_BOUNDS == 512,
              "EXL3 grouped prefill launch bounds must be 0, 128, 256, or 512");
static_assert(!EXL3_PF_LAUNCH_BOUNDS || EXL3_PF_M_TILE == 64,
              "EXL3 fixed launch bounds require M64");
static_assert(!EXL3_PF_LAUNCH_BOUNDS || EXL3_PF_EXACT_FULL_GRID == 1,
              "EXL3 fixed launch bounds require exact-full-grid mode");
static_assert(!EXL3_PF_LAUNCH_BOUNDS ||
              EXL3_PF_LAUNCH_BOUNDS == EXL3_PF_N_WARPS * 32,
              "EXL3 fixed launch bounds must cover one warp per N16 tile");
static_assert(EXL3_PF_PACKED_BF16_STORE == 0 ||
              EXL3_PF_PACKED_BF16_STORE == 1,
              "EXL3 grouped prefill packed BF16 store flag must be boolean");
static_assert(!EXL3_PF_PACKED_BF16_STORE || EXL3_PF_K_STEP == 64,
              "EXL3 packed BF16 stores require K64 stages");
static_assert(!EXL3_PF_PACKED_BF16_STORE || EXL3_PF_FIXED_N != 0,
              "EXL3 packed BF16 stores require a fixed N");
static_assert(!EXL3_PF_PACKED_BF16_STORE || (EXL3_PF_FIXED_N & 1) == 0,
              "EXL3 packed BF16 stores require an even N");
static_assert(!EXL3_PF_PACKED_BF16_STORE || EXL3_PF_FIXED_K != 0,
              "EXL3 packed BF16 stores require a fixed K");
static_assert(!EXL3_PF_PACKED_BF16_STORE || EXL3_PF_EXACT_FULL_GRID == 1,
              "EXL3 packed BF16 stores require an exact fixed shape");
static_assert(EXL3_PF_K2_LO_WINDOWS == 0 || EXL3_PF_K2_LO_WINDOWS == 1,
              "EXL3 grouped prefill K2 low-window flag must be boolean");
static_assert(!EXL3_PF_K2_LO_WINDOWS || EXL3_PF_FIXED_BITS == 2,
              "EXL3 K2 low windows require fixed K2");
static_assert(!EXL3_PF_K2_LO_WINDOWS || EXL3_PF_K_STEP == 64,
              "EXL3 K2 low windows require K64 stages");

#if EXL3_PF_LAUNCH_BOUNDS
#define EXL3_PF_LAUNCH_ATTR __launch_bounds__(EXL3_PF_LAUNCH_BOUNDS)
#else
#define EXL3_PF_LAUNCH_ATTR
#endif

#if EXL3_PF_EXACT_FULL_GRID
__host__ __device__ constexpr unsigned int exl3_pf_log2(unsigned int value) {
    return value <= 1 ? 0 : 1 + exl3_pf_log2(value >> 1);
}
#endif

#if EXL3_PF_ASYNC_STAGE
__device__ __forceinline__ void exl3_pf_cp_async_ca_4(
    void* dst_smem, const void* src_gmem) {
    const unsigned int dst = (unsigned int)__cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::
                 "r"(dst), "l"(src_gmem));
}

__device__ __forceinline__ void exl3_pf_cp_async_cg_16(
    void* dst_smem, const void* src_gmem) {
    const unsigned int dst = (unsigned int)__cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                 "r"(dst), "l"(src_gmem));
}
#endif

union exl3_pf_h2u32 {
    unsigned int u;
    __half2 h2;
};

__device__ __forceinline__ __half2 exl3_pf_decode2(unsigned int x0, unsigned int x1) {
    x0 *= EXL3_PF_MCG_MULT;
    x1 *= EXL3_PF_MCG_MULT;
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
    exl3_pf_h2u32 u0, u1;
    u0.u = x0;
    u1.u = x1;
    return __hadd2(__lows2half2(u0.h2, u1.h2), __highs2half2(u0.h2, u1.h2));
}

struct Exl3PfLaneGeom {
    int ia, ib, shift;
};

__device__ __forceinline__ Exl3PfLaneGeom exl3_pf_lane_geom(int lane, int bits) {
    const int b1 = (lane * 8 + 257) * bits;
    const int b0 = b1 - 16;
    const int b2 = b1 + 7 * bits;
    const int i0 = b0 >> 5;
    const int i2 = (b2 - 1) >> 5;
    return {i0 % (8 * bits), i2 % (8 * bits), (i2 + 1) * 32 - b2};
}

__device__ __forceinline__ void exl3_pf_dq8(
    const unsigned int* tile, Exl3PfLaneGeom g,
    __half2& d01, __half2& d23, __half2& d45, __half2& d67, int bits) {
    const unsigned int a = tile[g.ia];
    const unsigned int b = tile[g.ib];
    const unsigned int lo = __funnelshift_r(b, a, g.shift);
#if EXL3_PF_K2_LO_WINDOWS
    const unsigned int w7 = lo;
    const unsigned int w5 = lo >> 4;
    const unsigned int w3 = lo >> 8;
    const unsigned int w1 = lo >> 12;
#else
    const unsigned int hi = a >> g.shift;
    const unsigned int w7 = lo;
    const unsigned int w5 = __funnelshift_r(lo, hi, 2 * bits);
    const unsigned int w3 = __funnelshift_r(lo, hi, 4 * bits);
    const unsigned int w1 = __funnelshift_r(lo, hi, 6 * bits);
#endif
    d01 = exl3_pf_decode2((w1 >> bits) & 0xffffu, w1 & 0xffffu);
    d23 = exl3_pf_decode2((w3 >> bits) & 0xffffu, w3 & 0xffffu);
    d45 = exl3_pf_decode2((w5 >> bits) & 0xffffu, w5 & 0xffffu);
    d67 = exl3_pf_decode2((w7 >> bits) & 0xffffu, w7 & 0xffffu);
}

__device__ __forceinline__ unsigned int exl3_pf_bf16_pair(__half2 pair) {
    __nv_bfloat162 out;
    out.x = __float2bfloat16(__half2float(__low2half(pair)));
    out.y = __float2bfloat16(__half2float(__high2half(pair)));
    return *reinterpret_cast<unsigned int*>(&out);
}

__device__ __forceinline__ void exl3_pf_mma(
    float acc[4], unsigned int a0, unsigned int a1, unsigned int a2,
    unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
}

// Exact grid: (N/N_TILE, ceil(max_expert_rows/M_TILE), num_experts).
// Each M64 half uses one warp per N16 tile.
// Persistent production grid: one 1-D CTA per exact (expert, N_TILE strip).
// Generic persistent kernels also support undersubscription by grid-striding
// that same work. Each CTA reads its expert's device offsets once and walks
// only its live M tiles, avoiding both a host histogram synchronization and
// the old rectangular mostly-empty M-tile universe.
extern "C" __global__ void EXL3_PF_LAUNCH_ATTR EXL3_PF_KERNEL_NAME(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ trellis_tab,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, unsigned int num_experts,
    unsigned int N, unsigned int K, unsigned int bits,
    unsigned int persistent_mode) {
#if EXL3_PF_EXACT_FULL_GRID
#if EXL3_PF_LAUNCH_BOUNDS == 128
    asm volatile(
        "{ .reg .pred p; .reg .u32 n;\n\t"
        "mov.u32 n, %ntid.x; setp.ne.u32 p, n, 128; @p exit;\n\t"
        "mov.u32 n, %ntid.y; setp.ne.u32 p, n, 1; @p exit;\n\t"
        "mov.u32 n, %ntid.z; setp.ne.u32 p, n, 1; @p exit; }");
#elif EXL3_PF_LAUNCH_BOUNDS == 256
    asm volatile(
        "{ .reg .pred p; .reg .u32 n;\n\t"
        "mov.u32 n, %ntid.x; setp.ne.u32 p, n, 256; @p exit;\n\t"
        "mov.u32 n, %ntid.y; setp.ne.u32 p, n, 1; @p exit;\n\t"
        "mov.u32 n, %ntid.z; setp.ne.u32 p, n, 1; @p exit; }");
#elif EXL3_PF_LAUNCH_BOUNDS == 512
    asm volatile(
        "{ .reg .pred p; .reg .u32 n;\n\t"
        "mov.u32 n, %ntid.x; setp.ne.u32 p, n, 512; @p exit;\n\t"
        "mov.u32 n, %ntid.y; setp.ne.u32 p, n, 1; @p exit;\n\t"
        "mov.u32 n, %ntid.z; setp.ne.u32 p, n, 1; @p exit; }");
#endif
#endif
#if EXL3_PF_FIXED_N
    if (N != EXL3_PF_FIXED_N) return;
    constexpr unsigned int n_extent = EXL3_PF_FIXED_N;
#else
    const unsigned int n_extent = N;
#endif
#if EXL3_PF_FIXED_K
    if (K != EXL3_PF_FIXED_K) return;
    constexpr unsigned int k_extent = EXL3_PF_FIXED_K;
#else
    const unsigned int k_extent = K;
#endif
#if EXL3_PF_FIXED_PERSISTENT
    if (persistent_mode != 1) return;
#endif
#if EXL3_PF_FIXED_IDENTITY_ROWS
    if (sorted_token_ids != nullptr) return;
#endif
#if EXL3_PF_FIXED_BITS
    if (bits != EXL3_PF_FIXED_BITS) return;
    const unsigned int bit_width = EXL3_PF_FIXED_BITS;
#else
    if (bits != 2 && bits != 3) return;
    const unsigned int bit_width = bits;
#endif
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int n_warp = warp & (EXL3_PF_N_WARPS - 1);
    const unsigned int m_warp = warp / EXL3_PF_N_WARPS;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int group = lane >> 2;
    const unsigned int tid = lane & 3;
    __shared__ __align__(16) __nv_bfloat16 smem_A[EXL3_PF_M_TILE][EXL3_PF_K_STEP + EXL3_PF_PAD];
    __shared__ __align__(16) uint4 smem_T[EXL3_PF_K_TILES][EXL3_PF_N_WARPS * 2 * EXL3_PF_SMEM_BITS];
    const Exl3PfLaneGeom lane_geom = exl3_pf_lane_geom(lane, bit_width);
    const unsigned int n_tiles = n_extent / EXL3_PF_N_TILE;
#if EXL3_PF_EXACT_FULL_GRID
    const unsigned long long total_strips =
        (unsigned long long)num_experts * n_tiles;
    if (gridDim.y != 1 || gridDim.z != 1 ||
        (unsigned long long)gridDim.x != total_strips) return;
    const unsigned int n_tile = blockIdx.x & (n_tiles - 1);
    constexpr unsigned int n_tile_shift =
        exl3_pf_log2(EXL3_PF_FIXED_N / EXL3_PF_N_TILE);
    const unsigned int expert_id = blockIdx.x >> n_tile_shift;
    constexpr int m_first = 0;
    do {
#else
#if EXL3_PF_FIXED_PERSISTENT
    constexpr bool persistent = true;
#else
    const bool persistent = persistent_mode != 0;
#endif
    const unsigned int total_strips = num_experts * n_tiles;
    const unsigned int total_work = persistent
        ? total_strips
        : num_experts * gridDim.y * n_tiles;
    const unsigned int work0 = persistent
        ? blockIdx.x
        : (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    const unsigned int work_stride = persistent
        ? gridDim.x
        : gridDim.x * gridDim.y * gridDim.z;

    for (unsigned int work = work0; work < total_work; work += work_stride) {
        const unsigned int n_tile = work % n_tiles;
        const unsigned int expert_m = work / n_tiles;
        const unsigned int expert_id = persistent
            ? expert_m
            : expert_m / gridDim.y;
        const int m_first = persistent
            ? 0
            : (int)(expert_m % gridDim.y) * EXL3_PF_M_TILE;
#endif
        const int m_start = expert_offsets[expert_id];
        const int m_end = expert_offsets[expert_id + 1];
        if (m_start + m_first >= m_end) continue;
#if EXL3_PF_EXACT_FULL_GRID
        const int m_stop = m_end - m_start;
#else
        const int m_stop = persistent
            ? m_end - m_start
            : m_first + EXL3_PF_M_TILE;
#endif
        const unsigned int n_base = n_tile * EXL3_PF_N_TILE;
        const unsigned short* trellis = (const unsigned short*)trellis_tab[expert_id];
        if (!trellis) continue;

        for (int m_local = m_first;
             m_local < m_stop && m_start + m_local < m_end;
             m_local += EXL3_PF_M_TILE) {
            const bool active_m_warp =
                m_warp == 0 || m_start + m_local + 64 < m_end;

            // One warp owns each N16 tile in an M64 half. Wider N tiles add
            // column warps; M128 adds a second row-half using the same strip.
            float acc[4][2][4] = {};

            for (unsigned int k_base = 0; k_base < k_extent; k_base += EXL3_PF_K_STEP) {
                const unsigned int vectors_per_row = EXL3_PF_K_STEP / 8;
                const unsigned int load_vecs = EXL3_PF_M_TILE * vectors_per_row;
                for (unsigned int load_vec = threadIdx.x; load_vec < load_vecs;
                     load_vec += blockDim.x) {
                    const unsigned int row_local = load_vec / vectors_per_row;
                    const unsigned int k_col = (load_vec % vectors_per_row) * 8;
                    const unsigned int row = m_local + row_local;
                    const int sorted_row = m_start + row;
                    unsigned int* dst =
                        (unsigned int*)&smem_A[row_local][k_col];
#if EXL3_PF_ASYNC_STAGE
                    if (sorted_row < m_end) {
#if EXL3_PF_FIXED_IDENTITY_ROWS
                        const int input_row = sorted_row;
#else
                        const int input_row =
                            sorted_token_ids ? sorted_token_ids[sorted_row] : sorted_row;
#endif
                        const unsigned int* src = (const unsigned int*)(
                            A + (unsigned long long)input_row * k_extent + k_base + k_col);
                        // The +2 BF16 shared row pad leaves only 4-B alignment.
                        // Four async words preserve that bank-dispersing pad.
#pragma unroll
                        for (unsigned int word = 0; word < 4; ++word) {
                            exl3_pf_cp_async_ca_4(dst + word, src + word);
                        }
                    } else {
                        const uint4 zero = {0, 0, 0, 0};
                        dst[0] = zero.x;
                        dst[1] = zero.y;
                        dst[2] = zero.z;
                        dst[3] = zero.w;
                    }
#else
                    uint4 packed = {0, 0, 0, 0};
                    if (sorted_row < m_end) {
#if EXL3_PF_FIXED_IDENTITY_ROWS
                        const int input_row = sorted_row;
#else
                        const int input_row =
                            sorted_token_ids ? sorted_token_ids[sorted_row] : sorted_row;
#endif
                        const uint4* src = (const uint4*)(
                            A + (unsigned long long)input_row * k_extent + k_base + k_col);
                        packed = *src;
                    }
                    dst[0] = packed.x;
                    dst[1] = packed.y;
                    dst[2] = packed.z;
                    dst[3] = packed.w;
#endif
                }
                const unsigned int strip_u4 = EXL3_PF_N_WARPS * 2 * bit_width;
#pragma unroll
                for (unsigned int k_tile = 0; k_tile < EXL3_PF_K_TILES; ++k_tile) {
                    if (threadIdx.x < strip_u4) {
                        const unsigned int kb = (k_base >> 4) + k_tile;
                        const unsigned int nb = n_base >> 4;
                        const uint4* src = (const uint4*)trellis +
                            ((unsigned long long)kb * (n_extent >> 4) + nb) * (2 * bit_width) +
                            threadIdx.x;
#if EXL3_PF_ASYNC_STAGE
                        exl3_pf_cp_async_cg_16(&smem_T[k_tile][threadIdx.x], src);
#else
                        smem_T[k_tile][threadIdx.x] = *src;
#endif
                    }
                }
#if EXL3_PF_ASYNC_STAGE
                asm volatile("cp.async.commit_group;");
                asm volatile("cp.async.wait_group 0;");
#endif
                __syncthreads();

                if (active_m_warp) {
                    const unsigned short* a = (const unsigned short*)smem_A;
                    const unsigned int a_stride = EXL3_PF_K_STEP + EXL3_PF_PAD;
#pragma unroll
                    for (unsigned int k_tile = 0; k_tile < EXL3_PF_K_TILES; ++k_tile) {
                        const unsigned int* tile =
                            (const unsigned int*)smem_T[k_tile] + n_warp * (8 * bit_width);
                        __half2 d01, d23, d45, d67;
                        exl3_pf_dq8(
                            tile, lane_geom, d01, d23, d45, d67, bit_width);
                        const unsigned int b0 = exl3_pf_bf16_pair(d01);
                        const unsigned int b1 = exl3_pf_bf16_pair(d23);
                        const unsigned int b2 = exl3_pf_bf16_pair(d45);
                        const unsigned int b3 = exl3_pf_bf16_pair(d67);
#pragma unroll
                        for (unsigned int mt = 0; mt < 4; ++mt) {
                            const unsigned int r0 = m_warp * 64 + mt * 16 + group;
                            const unsigned int r1 = r0 + 8;
                            const unsigned int c0 = k_tile * 16 + tid * 2;
                            const unsigned int c1 = c0 + 8;
                            const unsigned int a0 = *(const unsigned int*)&a[r0 * a_stride + c0];
                            const unsigned int a1 = *(const unsigned int*)&a[r1 * a_stride + c0];
                            const unsigned int a2 = *(const unsigned int*)&a[r0 * a_stride + c1];
                            const unsigned int a3 = *(const unsigned int*)&a[r1 * a_stride + c1];
                            exl3_pf_mma(acc[mt][0], a0, a1, a2, a3, b0, b1);
                            exl3_pf_mma(acc[mt][1], a0, a1, a2, a3, b2, b3);
                        }
                    }
                }
                __syncthreads();
            }

            if (active_m_warp) {
#pragma unroll
                for (unsigned int mt = 0; mt < 4; ++mt) {
#pragma unroll
                    for (unsigned int nt = 0; nt < 2; ++nt) {
                        const unsigned int col0 =
                            n_base + n_warp * 16 + nt * 8 + tid * 2;
                        const unsigned int row0 =
                            m_local + m_warp * 64 + mt * 16 + group;
                        const unsigned int row1 = row0 + 8;
                        if (m_start + row0 < m_end) {
                            __nv_bfloat16* out =
                                C + (unsigned long long)(m_start + row0) * n_extent + col0;
#if EXL3_PF_PACKED_BF16_STORE
                            *reinterpret_cast<__nv_bfloat162*>(out) =
                                __floats2bfloat162_rn(acc[mt][nt][0], acc[mt][nt][1]);
#else
                            out[0] = __float2bfloat16(acc[mt][nt][0]);
                            out[1] = __float2bfloat16(acc[mt][nt][1]);
#endif
                        }
                        if (m_start + row1 < m_end) {
                            __nv_bfloat16* out =
                                C + (unsigned long long)(m_start + row1) * n_extent + col0;
#if EXL3_PF_PACKED_BF16_STORE
                            *reinterpret_cast<__nv_bfloat162*>(out) =
                                __floats2bfloat162_rn(acc[mt][nt][2], acc[mt][nt][3]);
#else
                            out[0] = __float2bfloat16(acc[mt][nt][2]);
                            out[1] = __float2bfloat16(acc[mt][nt][3]);
#endif
                        }
                    }
                }
            }
            __syncthreads();
        }
#if EXL3_PF_EXACT_FULL_GRID
    } while (false);
#else
    }
#endif
}
