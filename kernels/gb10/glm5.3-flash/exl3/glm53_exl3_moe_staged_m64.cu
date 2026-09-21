// SPDX-License-Identifier: AGPL-3.0-only
// Device implementation core: ExLlamaV3 MIT (`quant/exl3_gemm_inner.cuh`),
// pinned and verified by build.rs.
//
// 64-row-chunk variant of the staged expert-major MoE GEMMs. The donor inner
// is a 16-row design (`static_assert(TILESIZE_M == 16)`): every 16-row chunk
// re-decodes its expert's whole trellis. Here one CTA owns one N256 output
// tile for up to 64 rows (four m16 sub-tiles) and decodes each B tile once,
// issuing four MMAs per decoded fragment instead of one. Per output element
// the MMA sequence over K is unchanged, so results are bit-identical to the
// 16-row kernels; only the decode count per expert drops ~4x.

#include <cuda_fp16.h>
#include <stdint.h>

#include <util.h>
#include <util.cuh>
#include <ptx.cuh>
#include <quant/exl3_kernel_map.cuh>
#include <quant/hadamard_inner.cuh>
#include <quant/exl3_dq.cuh>

namespace atlas_m64 {

constexpr int THREADS = 256;
constexpr int TILE_K = 16;
constexpr int BITS = 2;
constexpr int CB = 2;
// Default cp.async pipeline depth; overridable per instantiation (see STAGES).
constexpr int SH_STAGES_DEFAULT = 3;
constexpr int BLOCK_N_UINT16 = 256 / 16 * BITS;                        // 32

// C[size_m, size_n] (f16, row-major) tile column blockIdx.x of width TILE_N =
// A[size_m, size_k] (f16) x trellis B. gridDim.x must be size_n / TILE_N.
// MSUB m16 sub-tiles per CTA; sub-tiles past size_m are skipped entirely.
template <int MSUB, int TILE_N, int KSUB, int SH_STAGES = SH_STAGES_DEFAULT,
          bool LATE_ISSUE = false>
__device__ __forceinline__ void gemm_inner_mx
(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    half* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n
)
{
    constexpr int TILE_M = 16 * MSUB;
    constexpr int STAGE_K = TILE_K * KSUB;
    constexpr int TILEBLOCKS_N = TILE_N / 16;
    constexpr int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (THREADS / 32);
    constexpr int SH_A_STAGE = TILE_M * STAGE_K;                       // halves
    constexpr int SH_B_BLOCKROW = TILEBLOCKS_N * 256 / 16 * BITS;      // uint16 per K16 block row
    constexpr int SH_B_STAGE = SH_B_BLOCKROW * KSUB;                   // uint16
    constexpr int A_COLS_S = STAGE_K / 8;                              // int4 per A row per stage
    constexpr int A_SWZ_MASK_S = A_COLS_S - 1;
    constexpr int A_SWZ_SHIFT_S = (A_COLS_S <= 2) ? 2 : 1;
    static_assert(FRAGS_N_PER_WARP >= 2 && FRAGS_N_PER_WARP % 2 == 0, "N tile");
    extern __shared__ half shared[];
    half* sh_a = shared;
    const int msub_live = (size_m + 15) / 16;   // uniform across the CTA
    uint16_t* sh_b = (uint16_t*) (sh_a + SH_STAGES * SH_A_STAGE);

    const int t = threadIdx.x;
    const int warp_id = t / 32;
    const int lane_id = t % 32;

    const int tiles_k = size_k / STAGE_K;
    const int blocks_n = size_n / 16;
    const int n_tile = blockIdx.x;

    // A tile: TILE_M rows x STAGE_K halves; A_ITERS predicated int4 copies per thread.
    constexpr int A_INT4 = TILE_M * A_COLS_S;
    constexpr int A_ITERS = (A_INT4 + THREADS - 1) / THREADS;
    bool a_pred[A_ITERS]; int a_gl[A_ITERS]; int a_sh[A_ITERS];
    #pragma unroll
    for (int i = 0; i < A_ITERS; ++i)
    {
        const int idx = i * THREADS + t;
        const int a_k = idx % A_COLS_S;
        const int a_m = idx / A_COLS_S;
        a_pred[i] = idx < A_INT4 && a_m < size_m;
        a_gl[i] = a_m * size_k / 8 + a_k;
        a_sh[i] = a_m * A_COLS_S + (a_k ^ ((a_m >> A_SWZ_SHIFT_S) & A_SWZ_MASK_S));
    }

    // B tile: KSUB K16 block rows of TILEBLOCKS_N N16 blocks.
    const int gl_b_stride_k = blocks_n * BLOCK_N_UINT16;
    constexpr int B_INT4_ROW = SH_B_BLOCKROW / 8;
    constexpr int B_INT4 = B_INT4_ROW * KSUB;
    constexpr int B_ITERS = (B_INT4 + THREADS - 1) / THREADS;
    const uint16_t* gl_b = B + n_tile * TILEBLOCKS_N * BLOCK_N_UINT16;
    const half* gl_a = A;

    auto issue_load = [&] (int tile)
    {
        if (tile < tiles_k)
        {
            const int stage = tile % SH_STAGES;
            #pragma unroll
            for (int i = 0; i < A_ITERS; ++i)
                if (a_pred[i])
                    cp_async((int4*) (sh_a + stage * SH_A_STAGE) + a_sh[i],
                             (const int4*) (gl_a + tile * STAGE_K) + a_gl[i]);
            #pragma unroll
            for (int i = 0; i < B_ITERS; ++i)
            {
                const int idx = i * THREADS + t;
                if (idx < B_INT4)
                {
                    const int kb = idx / B_INT4_ROW;
                    const int r = idx % B_INT4_ROW;
                    cp_async((int4*) (sh_b + stage * SH_B_STAGE + kb * SH_B_BLOCKROW) + r,
                             (const int4*) (gl_b + (tile * KSUB + kb) * gl_b_stride_k) + r);
                }
            }
        }
        cp_async_fence();
    };

    FragA frag_a[MSUB];
    FragB frag_b[FRAGS_N_PER_WARP];
    FragC frag_c[MSUB][FRAGS_N_PER_WARP];
    #pragma unroll
    for (int m = 0; m < MSUB; ++m)
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            frag_c[m][n] = {};

    #pragma unroll
    for (int i = 0; i < SH_STAGES - 1; ++i)
        issue_load(i);

    for (int tile = 0; tile < tiles_k; ++tile)
    {
        // LATE_ISSUE moves this to the BOTTOM of the body, which removes the
        // trailing __syncthreads: the issue then targets stage
        // (tile+SH_STAGES-1)%SH_STAGES, a different buffer from the tile%SH_STAGES
        // this CTA is reading, so it needs no barrier of its own, and the NEXT
        // iteration's leading barrier already orders it against the following read.
        // One barrier per K-tile instead of two. No change to loads or MMA order.
        if (!LATE_ISSUE)
            issue_load(tile + SH_STAGES - 1);
        cp_async_wait<SH_STAGES - 2>();
        __syncthreads();

        const int stage = tile % SH_STAGES;
        const half* sh1_a = sh_a + stage * SH_A_STAGE;
        const uint16_t* sh1_b = sh_b + stage * SH_B_STAGE;

        #pragma unroll
        for (int ks = 0; ks < KSUB; ++ks)
        {
            // A fragments (XOR-swizzled shared layout), one per m16 sub-tile
            {
                const int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
                const int base_c = lane_id / 16 + ks * 2;
                #pragma unroll
                for (int m = 0; m < MSUB; ++m)
                {
                    if (m >= msub_live) break;
                    const int R = r + m * 16;
                    const int c_swizzled = base_c ^ ((R >> A_SWZ_SHIFT_S) & A_SWZ_MASK_S);
                    ldsm4(frag_a[m], (int4*) sh1_a + R * A_COLS_S + c_swizzled);
                }
            }
            // B fragments: decode once per k16 block row
            #pragma unroll
            for (int n2 = 0; n2 < FRAGS_N_PER_WARP; n2 += 2)
            {
                const int sub_n2 = warp_id * FRAGS_N_PER_WARP / 2 + n2 / 2;
                const uint32_t* shb = (const uint32_t*) (sh1_b + ks * SH_B_BLOCKROW + sub_n2 * BLOCK_N_UINT16);
                dq_dispatch<BITS, CB>(shb, lane_id << 3, frag_b[n2], frag_b[n2 + 1]);
            }
            #pragma unroll
            for (int m = 0; m < MSUB; ++m)
            {
                if (m >= msub_live) break;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    ptx_mma_m16n8k16(frag_a[m], frag_b[n], frag_c[m][n]);
            }
        }
        if (LATE_ISSUE)
            issue_load(tile + SH_STAGES - 1);
        else
            __syncthreads();
    }

    // Row-major F16 write-out (donor `write_sum_gl`, c_fp32 = false)
    half* gl_c = C + n_tile * TILE_N;
    const int n0 = warp_id * FRAGS_N_PER_WARP;
    const int c = (lane_id % 4) * 2;
    #pragma unroll
    for (int m = 0; m < MSUB; ++m)
    {
        const int r0 = lane_id / 4 + m * 16;
        const int r1 = r0 + 8;
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            if (r0 < size_m)
            {
                half2* c_ptr = (half2*) (gl_c + r0 * size_n + (n0 + n) * 8 + c);
                *c_ptr = __floats2half2_rn(frag_c[m][n][0], frag_c[m][n][1]);
            }
            if (r1 < size_m)
            {
                half2* c_ptr = (half2*) (gl_c + r1 * size_n + (n0 + n) * 8 + c);
                *c_ptr = __floats2half2_rn(frag_c[m][n][2], frag_c[m][n][3]);
            }
        }
    }
}

} // namespace atlas_m64


// Same descriptor contract as `atlas_glm53_exl3_build_chunks_private`, with
// 64-row chunks. `max_chunks` remains the 16-row upper bound the plan sizes.
template <uint32_t CHUNK_ROWS>
__device__ __forceinline__ void atlas_glm53_exl3_build_chunks_private_mx(
    const int64_t* __restrict__ expert_count,
    uint32_t* __restrict__ pair_expert,
    uint32_t* __restrict__ chunk_expert,
    uint32_t* __restrict__ chunk_start,
    uint32_t* __restrict__ chunk_rows,
    uint32_t* __restrict__ chunk_count,
    uint32_t* __restrict__ status,
    uint32_t experts,
    uint32_t max_chunks,
    uint32_t pairs)
{
    __shared__ uint32_t expert_start[289];
    __shared__ uint32_t ready;
    const uint32_t thread = threadIdx.x;
    if (thread == 0)
    {
        ready = 0;
        *chunk_count = 0;
        if (*status == 0)
        {
            bool valid = experts == 288 && pairs > 0 && pairs <= 2048 * 8 &&
                         pairs % 8 == 0 && max_chunks > 0 && max_chunks <= 1312;
            uint32_t total = 0;
            uint32_t chunks = 0;
            if (valid)
            {
                for (uint32_t expert = 0; expert < experts; ++expert)
                {
                    expert_start[expert] = total;
                    const int64_t count = expert_count[expert];
                    if (count < 0 || static_cast<uint64_t>(count) > pairs - total)
                    {
                        valid = false;
                        break;
                    }
                    const uint32_t rows = static_cast<uint32_t>(count);
                    const uint32_t needed = (rows + CHUNK_ROWS - 1) / CHUNK_ROWS;
                    if (needed > max_chunks - chunks)
                    {
                        valid = false;
                        break;
                    }
                    total += rows;
                    chunks += needed;
                }
            }
            if (valid && total == pairs)
            {
                expert_start[experts] = total;
                uint32_t chunk = 0;
                for (uint32_t expert = 0; expert < experts; ++expert)
                {
                    const uint32_t end = expert_start[expert + 1];
                    for (uint32_t start = expert_start[expert]; start < end; start += CHUNK_ROWS)
                    {
                        chunk_expert[chunk] = expert;
                        chunk_start[chunk] = start;
                        chunk_rows[chunk] = min(CHUNK_ROWS, end - start);
                        ++chunk;
                    }
                }
                *chunk_count = chunks;
                ready = 1;
            }
            else
            {
                atomicExch(status, 2U);
            }
        }
    }
    __syncthreads();
    if (!ready) return;
    if (thread < experts)
    {
        for (uint32_t pair = expert_start[thread]; pair < expert_start[thread + 1]; ++pair)
            pair_expert[pair] = thread;
    }
}

#define ATLAS_MX_KERNELS_L(SUFFIX, MSUB, TILE_N, KSUB, MINB, STAGES, LATE) \
extern "C" __global__ void atlas_glm53_exl3_build_chunks_private_##SUFFIX( \
    const int64_t* __restrict__ expert_count, uint32_t* __restrict__ pair_expert, \
    uint32_t* __restrict__ chunk_expert, uint32_t* __restrict__ chunk_start, \
    uint32_t* __restrict__ chunk_rows, uint32_t* __restrict__ chunk_count, \
    uint32_t* __restrict__ status, uint32_t experts, uint32_t max_chunks, uint32_t pairs) \
{ \
    atlas_glm53_exl3_build_chunks_private_mx<16U * MSUB>(expert_count, pair_expert, chunk_expert, \
        chunk_start, chunk_rows, chunk_count, status, experts, max_chunks, pairs); \
} \
extern "C" __global__ __launch_bounds__(256, MINB) \
void atlas_glm53_exl3_staged_gate_up_##SUFFIX( \
    const half* __restrict__ state_g, const half* __restrict__ state_u, \
    half* __restrict__ intermediate_g, half* __restrict__ intermediate_u, \
    const uint16_t** __restrict__ gate_trellis, const uint16_t** __restrict__ up_trellis, \
    const uint32_t* __restrict__ chunk_expert, const uint32_t* __restrict__ chunk_start, \
    const uint32_t* __restrict__ chunk_rows, const uint32_t* __restrict__ chunk_count, \
    int hidden_dim, int intermediate_dim, int lock_stride, int* __restrict__ locks) \
{ \
    (void) lock_stride; (void) locks; \
    const uint32_t chunk = blockIdx.y; \
    if (chunk >= *chunk_count) return; \
    if (gridDim.x != static_cast<uint32_t>(intermediate_dim / TILE_N)) return; \
    const uint32_t projection = blockIdx.z; \
    const uint32_t expert = chunk_expert[chunk]; \
    const uint32_t start = chunk_start[chunk]; \
    const int rows = static_cast<int>(chunk_rows[chunk]); \
    if (rows <= 0 || rows > 16 * MSUB) return; \
    const half* input = (projection == 0 ? state_g : state_u) + \
                        static_cast<uint64_t>(start) * hidden_dim; \
    half* output = (projection == 0 ? intermediate_g : intermediate_u) + \
                   static_cast<uint64_t>(start) * intermediate_dim; \
    const uint16_t* trellis = (projection == 0 ? gate_trellis : up_trellis)[expert]; \
    atlas_m64::gemm_inner_mx<MSUB, TILE_N, KSUB, STAGES, LATE>(input, trellis, output, rows, hidden_dim, intermediate_dim); \
} \
extern "C" __global__ __launch_bounds__(256, MINB) \
void atlas_glm53_exl3_staged_down_##SUFFIX( \
    const half* __restrict__ intermediate, half* __restrict__ state, \
    const uint16_t** __restrict__ down_trellis, const uint32_t* __restrict__ chunk_expert, \
    const uint32_t* __restrict__ chunk_start, const uint32_t* __restrict__ chunk_rows, \
    const uint32_t* __restrict__ chunk_count, int lock_stride, int* __restrict__ locks) \
{ \
    (void) lock_stride; (void) locks; \
    const uint32_t chunk = blockIdx.y; \
    if (chunk >= *chunk_count) return; \
    if (gridDim.x != 4096 / TILE_N) return; \
    const uint32_t expert = chunk_expert[chunk]; \
    const uint32_t start = chunk_start[chunk]; \
    const int rows = static_cast<int>(chunk_rows[chunk]); \
    if (rows <= 0 || rows > 16 * MSUB) return; \
    atlas_m64::gemm_inner_mx<MSUB, TILE_N, KSUB, STAGES, LATE>( \
        intermediate + static_cast<uint64_t>(start) * 2048, down_trellis[expert], \
        state + static_cast<uint64_t>(start) * 4096, rows, 2048, 4096); \
}

// Back-compat: the historical 6-arg form is the early-issue (two-barrier) path.
#define ATLAS_MX_KERNELS_S(SUFFIX, MSUB, TILE_N, KSUB, MINB, STAGES) \
    ATLAS_MX_KERNELS_L(SUFFIX, MSUB, TILE_N, KSUB, MINB, STAGES, false)

#define ATLAS_MX_KERNELS(SUFFIX, MSUB, TILE_N, KSUB, MINB) \
    ATLAS_MX_KERNELS_S(SUFFIX, MSUB, TILE_N, KSUB, MINB, 3)

ATLAS_MX_KERNELS(m64, 4, 256, 1, 2)
ATLAS_MX_KERNELS(m128, 8, 128, 1, 2)
ATLAS_MX_KERNELS(m64k2, 4, 256, 2, 2)
ATLAS_MX_KERNELS(m128w, 8, 256, 1, 1)
ATLAS_MX_KERNELS(m128wk2, 8, 256, 2, 1)
// Deeper K staging: KSUB=2 (m64k2) was the largest single MoE win, so carry the
// same axis further. Each cp.async stage covers KSUB K16 block rows, so the
// trellis decode is amortised over more MMAs and the stage count drops.
// Shared per stage = 16*MSUB*16*KSUB*2 + (TILE_N/16)*32*KSUB*2 bytes:
//   m64k3 = 9216 B/stage (27,648 for 3 stages)
//   m64k4 = 12288 B/stage (36,864 for 3 stages)
// both well inside the 228 KB SM budget at MINB=2.
ATLAS_MX_KERNELS(m64k3, 4, 256, 3, 2)
ATLAS_MX_KERNELS(m64k4, 4, 256, 4, 2)
// Deeper cp.async pipeline on the winning 64k2 shape: 4 stages instead of 3, to
// hide more of the trellis decode latency. Shared = 4 * 6144 = 24,576 B.
ATLAS_MX_KERNELS_S(m64k2s4, 4, 256, 2, 2, 4)
// Deeper still on the same shape. The m64k2 stage is 6,144 B and registers -- not
// shared memory -- already pin residency at 2 CTAs/SM (119/120 regs, 0 spills), so
// shared memory is the one resource being left unused: 2 x 3 x 6,144 = 36,864 B of
// the SM's 102,400 B. Stage depth therefore costs NOTHING in occupancy up to 8:
//   stages 3 -> 36,864 B/SM (36%)   [shipped]
//   stages 4 -> 49,152 B/SM (48%)   m64k2s4
//   stages 6 -> 73,728 B/SM (72%)   m64k2s6
//   stages 8 -> 98,304 B/SM (96%)   m64k2s8  <- the limit
// Per-block dynamic shared is 49,152 B at stages 8, under the 101,376 B opt-in cap.
// NOTE the budget in the m64k3/m64k4 comment above says "228 KB SM budget": that is
// HOPPER. GB10 is 102,400 B/SM and 101,376 B max dynamic per block. Anything sized
// against 228 KB will fail to launch.
// These change NO byte counts and NO MMA order -- only how far ahead the cp.async
// pipeline runs (`cp_async_wait<SH_STAGES-2>` keeps SH_STAGES-1 loads in flight
// instead of 2). Bit-identical to m64k2 by construction.
ATLAS_MX_KERNELS_S(m64k2s6, 4, 256, 2, 2, 6)
ATLAS_MX_KERNELS_S(m64k2s8, 4, 256, 2, 2, 8)
// Wider N tile. Each CTA owns one N-tile of one 64-row chunk, and re-reads that
// chunk's A rows from global for EVERY N tile: with TILE_N=256 the gate_up grid
// is 2048/256 = 8 tiles and the down grid 4096/256 = 16, so a 64x4096 A tile is
// read 8x and a 64x2048 one 16x. At ~500 live chunks x 42 layers that is the
// dominant traffic term. TILE_N=512 halves both.
// FRAGS_N_PER_WARP = 2*(512/16)/8 = 8 (even, >=2, so the N-tile assert holds);
// shared per stage = 64*32*2 + 32*32*2*2 = 8192 B, 24,576 B for 3 stages.
ATLAS_MX_KERNELS(m64n512, 4, 512, 1, 2)
ATLAS_MX_KERNELS(m64n512k2, 4, 512, 2, 2)

// Single-barrier (late-issue) variants: `issue_load` moves to the bottom of the
// K-tile body, removing the trailing __syncthreads. 256 barriers per gate_up CTA
// become 128. Loads, MMA order and output are UNCHANGED -- bit-identical to the
// corresponding early-issue variant. Instantiated at stage depth 3 (directly
// comparable to the live m64k2) and at 8 (combined with the deepest pipeline,
// since late issue shortens the prefetch distance by one compute phase and extra
// stages are what give that back).
ATLAS_MX_KERNELS_L(m64k2li,   4, 256, 2, 2, 3, true)
ATLAS_MX_KERNELS_L(m64k2s8li, 4, 256, 2, 2, 8, true)
