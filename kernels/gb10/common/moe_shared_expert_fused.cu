// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Fused MoE Expert+Shared GEMV — shared expert as extra blockIdx.y slot.
//
// Same as moe_expert_gemv_fused.cu gate_up_2x / silu_down_2x but with
// blockIdx.y == top_k serving the shared expert using direct weight pointers.
// The shared expert blocks run concurrently with routed expert blocks within
// the same kernel launch, eliminating 2 separate kernel launches per layer
// (96 graph nodes across 48 MoE layers).
//
// Grid: gate_up (ceil(N/8), top_k+1, 2),  silu_down (ceil(N/8), top_k+1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_SHARED[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// NVFP4 per-block FP8-E4M3 scale decode. SCALE/gfx1151 `(float)__nv_fp8_e4m3`
// is NON-STANDARD (same bug fixed in moe_sorted_prefill.cu / the decode GEMVs) —
// software scl_fp8 there; NVIDIA path is the verbatim cast.
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float atlas_dec_e4m3(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#else
__device__ __forceinline__ float atlas_dec_e4m3(unsigned char b) {
    __nv_fp8_e4m3 f; *(unsigned char*)&f = b; return (float)f;
}
#endif

// ── Fused Gate+Up 2x with shared expert ──
//
// blockIdx.y < top_k: routed expert (pointer table lookup)
// blockIdx.y == top_k: shared expert (direct weight pointers)
// Grid: (ceil(N/8), top_k+1, 2)  Block: (128, 1, 1)
extern "C" __global__ void moe_expert_gate_up_shared(
    const __nv_bfloat16* __restrict__ A,
    // Routed expert tables
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert direct pointers
    const unsigned char* __restrict__ sh_gate_packed,
    const unsigned char* __restrict__ sh_gate_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;

    if (is_shared) {
        // NULL shared expert: model has no shared expert weights (e.g., Mistral).
        // Write zeros and return to prevent NULL pointer dereference.
        if (sh_gate_packed == 0) {
            __nv_bfloat16* out = (proj == 0) ? sh_gate_out : sh_up_out;
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE)
                out[n_base + i] = __float2bfloat16(0.0f);
            return;
        }
        if (proj == 0) {
            B_packed = sh_gate_packed; B_scale = sh_gate_scale;
            s2 = sh_gate_s2; C = sh_gate_out;
        } else {
            B_packed = sh_up_packed; B_scale = sh_up_scale;
            s2 = sh_up_s2; C = sh_up_out;
        }
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id]; C = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id]; C = up_out;
        }
        // EP: NULL pointer means remote expert — write zero output and return
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_SHARED[threadIdx.x];
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    // 16 K-values per iteration: uint64 weight + 2×uint4 activation
    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;
        uint4 a_lo = ((const uint4*)A)[k16 * 2];
        uint4 a_hi = ((const uint4*)A)[k16 * 2 + 1];
        const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};

        unsigned long long packed8_1 = *(const unsigned long long*)(B_packed + (unsigned long long)n1 * half_K + k16 * 8);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        float sc1 = atlas_dec_e4m3(sb1) * s2;

        unsigned long long packed8_2 = have_n2 ?
            *(const unsigned long long*)(B_packed + (unsigned long long)n2 * half_K + k16 * 8) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char bv1 = (unsigned char)(packed8_1 >> (b * 8));
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (unsigned char)(packed8_2 >> (b * 8));
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;
            __nv_bfloat16 al, ah;
            *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
            float afl = __bfloat162float(al), afh = __bfloat162float(ah);
            acc1 += afl * w1l + afh * w1h;
            acc2 += afl * w2l + afh * w2h;
        }
    }

    // Output offset: shared expert writes at [0..N], routed at [slot*N..N]
    const unsigned long long base = is_shared ? 0 : (unsigned long long)expert_slot * N;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) C[base + n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) C[base + n2] = __float2bfloat16(acc2);
    }
}

// ── Fused SiLU+Down 2x with shared expert ──
//
// Precomputes SiLU(gate)*up in shared memory once per block, eliminating
// redundant SiLU compute across all 4 thread groups and replacing global
// gate/up loads with fast shared memory reads in the GEMV inner loop.
//
// blockIdx.y < top_k: routed expert (pointer table + expert_gate_out/up_out)
// blockIdx.y == top_k: shared expert (direct pointers + sh_gate_in/up_in)
// Grid: (ceil(N/8), top_k+1, 1)  Block: (128, 1, 1)
extern "C" __global__ void moe_expert_silu_down_shared(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;

    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;

    if (is_shared) {
        // NULL shared expert: write zeros and return
        if (sh_down_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE)
                sh_down_out[n_base + i] = __float2bfloat16(0.0f);
            return;
        }
        B_packed = sh_down_packed; B_scale = sh_down_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in; u_ptr = sh_up_in;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)expert_slot * K;
        u_ptr = up_out + (unsigned long long)expert_slot * K;
        // EP: NULL pointer means remote expert — write zero output and return
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    extern __shared__ float s_act[];

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_SHARED[threadIdx.x];

    // Phase 1: Cooperatively precompute SiLU(gate)*up into shared memory.
    // DeepSeek-V4 clamps the ROUTED expert swiglu inputs to ±swiglu_limit
    // (gate<=limit, up in [-limit,limit]); the shared expert (DeepseekV4MLP)
    // is NOT clamped. swiglu_limit = 10.0 (config; hardcoded here pending a
    // config-threaded kernel arg).
    const float SWIGLU_LIMIT = 10.0f;
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        if (!is_shared) {
            gf = fminf(gf, SWIGLU_LIMIT);
            uf = fminf(fmaxf(uf, -SWIGLU_LIMIT), SWIGLU_LIMIT);
        }
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();

    // Phase 2: GEMV with 16 K-values per iteration
    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;

        unsigned long long packed8_1 = *(const unsigned long long*)(B_packed + (unsigned long long)n1 * half_K + k16 * 8);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        float sc1 = atlas_dec_e4m3(sb1) * s2;

        unsigned long long packed8_2 = have_n2 ?
            *(const unsigned long long*)(B_packed + (unsigned long long)n2 * half_K + k16 * 8) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 8; b++) {
            float al = s_act[base_k + b * 2];
            float ah = s_act[base_k + b * 2 + 1];

            unsigned char bv1 = (unsigned char)(packed8_1 >> (b * 8));
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (unsigned char)(packed8_2 >> (b * 8));
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;

            acc1 += al * w1l + ah * w1h;
            acc2 += al * w2l + ah * w2h;
        }
    }

    // Output: shared writes to sh_down_out, routed writes to C[slot*N]
    __nv_bfloat16* out = is_shared ? sh_down_out : (C + (unsigned long long)expert_slot * N);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) out[n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) out[n2] = __float2bfloat16(acc2);
    }
}

// ============================================================================
// v5 — cp.async bulk-staged M=1 serial-decode variants (ATLAS_KN_V5).
//
// Same launch contract and BIT-IDENTICAL outputs to moe_expert_gate_up_shared
// / moe_expert_silu_down_shared above: the per-row lane partition (32 lanes,
// k16 = lane + 32*i ascending), uint64 nibble decode order and FMA order are
// copied verbatim — only HOW the weight bytes arrive changes:
//
//   1. cp.async bulk staging: each block issues its ENTIRE 8-row weight slice
//      (packed + scales) as asynchronous 16B copies up front (one commit
//      group per 1024-elem K-tile), so the full slice is in flight while
//      earlier tiles are decoded/FMA'd. The serial kernels issue one
//      dependent 8B load per lane per iteration and stall through the
//      decode+FMA chain — the same pattern that pinned the batchN v2/v4
//      verify kernels at ~176 GB/s before their v5 rewrite (211 GB/s).
//   2. down v5 covers 16 rows per block (2 pipelined 8-row cp.async tiles)
//      instead of 8: grid.x = ceil(N/16) (launcher differs from v1),
//      halving block count and overlapping tile-1 arrival with tile-0
//      compute (K=inter=1024 gives each lane only TWO k16 groups per row —
//      v1 had zero load overlap).
//
// Shape limits (Rust dispatch falls back to the serial kernels outside):
//   gate_up_v5: K % 1024 == 0 && K <= 3072;   silu_down_v5: K == 1024.
// The NULL-shared-expert guard (Laguna: routed NVFP4 + NULL shared) and the
// EP remote-expert NULL guard behave exactly as in the serial kernels.
// ============================================================================

// cp.async helpers (SM80+) — same as moe_shared_expert_fused_batch2.cu v5.
__device__ __forceinline__ void s5_cp_async_16(void* dst_smem, const void* src_gmem, bool pred) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}
__device__ __forceinline__ void s5_cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

#define S5_TILE 1024                       // K elems per stage tile
#define S5_TILE_PK (S5_TILE / 2)           // 512 packed bytes per row per tile
#define S5_TILE_SC (S5_TILE / GROUP_SIZE)  // 64 scale bytes per row per tile
#define S5_GU_TILES 3                      // gate_up KMAX = 3072
#define S5_DN_ROW_TILES 2                  // down: 2 pipelined 8-row tiles

// gate_up v5 — identical launch contract to moe_expert_gate_up_shared:
// grid (ceil(N/8), top_k+1, 2), block 128, no dynamic smem. The block's whole
// 8-row weight slice (packed + scales, all K-tiles) is cp.async-staged up
// front in 3 commit groups; compute waits tile-by-tile (wait_group 2/1/0).
// Bit-identical outputs to the serial kernel.
extern "C" __global__ void moe_expert_gate_up_shared_v5(
    const __nv_bfloat16* __restrict__ A,
    // Routed expert tables
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert direct pointers
    const unsigned char* __restrict__ sh_gate_packed,
    const unsigned char* __restrict__ sh_gate_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;

    if (is_shared) {
        // NULL shared expert (Laguna/Mistral): write zeros and return —
        // unchanged from the serial kernel.
        if (sh_gate_packed == 0) {
            __nv_bfloat16* out = (proj == 0) ? sh_gate_out : sh_up_out;
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE)
                out[n_base + i] = __float2bfloat16(0.0f);
            return;
        }
        if (proj == 0) {
            B_packed = sh_gate_packed; B_scale = sh_gate_scale;
            s2 = sh_gate_s2; C = sh_gate_out;
        } else {
            B_packed = sh_up_packed; B_scale = sh_up_scale;
            s2 = sh_up_s2; C = sh_up_out;
        }
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id]; C = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id]; C = up_out;
        }
        // EP: NULL pointer means remote expert — write zero output and return
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int row_base = blockIdx.x * (N_PER_BLOCK * 2);
    const unsigned int n1 = row_base + local_out * 2;
    const unsigned int n2 = n1 + 1;
    const bool have_n1 = (n1 < N);
    const bool have_n2 = (n2 < N);
    const unsigned int r1 = local_out * 2, r2 = r1 + 1;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int num_tiles = K / S5_TILE;   // 1..3 (dispatch-guarded)

    __shared__ float s_lut[16];
    __shared__ __align__(16) unsigned char s_wq[S5_GU_TILES][8][S5_TILE_PK]; // 12KB
    __shared__ __align__(16) unsigned char s_sc[S5_GU_TILES][8][S5_TILE_SC]; // 1.5KB
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_SHARED[threadIdx.x];

    #pragma unroll
    for (unsigned int t = 0; t < S5_GU_TILES; t++) {
        if (t < num_tiles) {
            for (unsigned int i = threadIdx.x; i < 8 * (S5_TILE_PK / 16); i += BLOCK_SIZE) {
                const unsigned int r = i / (S5_TILE_PK / 16);
                const unsigned int c = i % (S5_TILE_PK / 16);
                const unsigned int n = row_base + r;
                s5_cp_async_16(&s_wq[t][r][c * 16],
                    B_packed + (unsigned long long)n * half_K + t * S5_TILE_PK + c * 16,
                    n < N);
            }
            for (unsigned int i = threadIdx.x; i < 8 * (S5_TILE_SC / 16); i += BLOCK_SIZE) {
                const unsigned int r = i / (S5_TILE_SC / 16);
                const unsigned int c = i % (S5_TILE_SC / 16);
                const unsigned int n = row_base + r;
                s5_cp_async_16(&s_sc[t][r][c * 16],
                    B_scale + (unsigned long long)n * num_groups + t * S5_TILE_SC + c * 16,
                    n < N);
            }
        }
        s5_cp_async_commit();
    }

    float acc1 = 0.0f, acc2 = 0.0f;

    #pragma unroll
    for (unsigned int t = 0; t < S5_GU_TILES; t++) {
        if (t == 0)      asm volatile("cp.async.wait_group 2;");
        else if (t == 1) asm volatile("cp.async.wait_group 1;");
        else             asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        if (t >= num_tiles) break;
        if (!have_n1) continue;
        // 2 k16 groups per lane per tile (64 groups / 32 lanes) — the same
        // ascending per-lane k16 sequence as the serial kernel's `k16 += 32`
        // walk, so the accumulation order is identical.
        #pragma unroll
        for (unsigned int j = 0; j < S5_TILE / 16 / 32; j++) {
            const unsigned int lk16 = j * threads_per_out + lane;      // in-tile
            const unsigned int k16 = t * (S5_TILE / 16) + lk16;        // global
            uint4 a_lo = ((const uint4*)A)[k16 * 2];
            uint4 a_hi = ((const uint4*)A)[k16 * 2 + 1];
            const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                            a_hi.x, a_hi.y, a_hi.z, a_hi.w};

            unsigned long long packed8_1 = *(const unsigned long long*)&s_wq[t][r1][lk16 * 8];
            unsigned char sb1 = s_sc[t][r1][lk16];
            float sc1 = atlas_dec_e4m3(sb1) * s2;

            unsigned long long packed8_2 = have_n2 ?
                *(const unsigned long long*)&s_wq[t][r2][lk16 * 8] : 0;
            unsigned char sb2 = have_n2 ? s_sc[t][r2][lk16] : 0;
            float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;

            #pragma unroll
            for (int b = 0; b < 8; b++) {
                unsigned char bv1 = (unsigned char)(packed8_1 >> (b * 8));
                float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
                unsigned char bv2 = (unsigned char)(packed8_2 >> (b * 8));
                float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;
                __nv_bfloat16 al, ah;
                *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
                *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
                float afl = __bfloat162float(al), afh = __bfloat162float(ah);
                acc1 += afl * w1l + afh * w1h;
                acc2 += afl * w2l + afh * w2h;
            }
        }
    }
    if (!have_n1) return;

    // Output offset: shared expert writes at [0..N], routed at [slot*N..N]
    const unsigned long long base = is_shared ? 0 : (unsigned long long)expert_slot * N;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) C[base + n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) C[base + n2] = __float2bfloat16(acc2);
    }
}

// silu_down v5 — 16 rows per block (2 pipelined 8-row cp.async tiles).
// Grid: (ceil(N/16), top_k+1, 1) — NOT ceil(N/8), the Rust launcher differs
// from the serial kernel. Block 128, dynamic smem K*4 (s_act, same as v1).
// Requires K == 1024 (Laguna inter). Bit-identical outputs to
// moe_expert_silu_down_shared (same SWIGLU clamp, silu order, lane
// partition k16 = lane, lane+32).
extern "C" __global__ void moe_expert_silu_down_shared_v5(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const bool is_shared = (expert_slot == top_k);
    const unsigned int row_base = blockIdx.x * (N_PER_BLOCK * 2 * S5_DN_ROW_TILES);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;

    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;

    if (is_shared) {
        // NULL shared expert: write zeros (all 16 covered rows) and return
        if (sh_down_packed == 0) {
            for (unsigned int i = threadIdx.x;
                 i < N_PER_BLOCK * 2 * S5_DN_ROW_TILES && row_base + i < N;
                 i += BLOCK_SIZE)
                sh_down_out[row_base + i] = __float2bfloat16(0.0f);
            return;
        }
        B_packed = sh_down_packed; B_scale = sh_down_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in; u_ptr = sh_up_in;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)expert_slot * K;
        u_ptr = up_out + (unsigned long long)expert_slot * K;
        // EP: NULL pointer means remote expert — write zero output and return
        if (B_packed == 0) {
            for (unsigned int i = threadIdx.x;
                 i < N_PER_BLOCK * 2 * S5_DN_ROW_TILES && row_base + i < N;
                 i += BLOCK_SIZE) {
                C[(unsigned long long)expert_slot * N + row_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int half_K = K / 2;               // 512 (K == 1024 guarded)
    const unsigned int num_groups = K / GROUP_SIZE;  // 64
    const unsigned int K16 = K / 16;                 // 64

    __shared__ float s_lut[16];
    __shared__ __align__(16) unsigned char s_wq[S5_DN_ROW_TILES][8][S5_TILE_PK]; // 8KB
    __shared__ __align__(16) unsigned char s_sc[S5_DN_ROW_TILES][8][S5_TILE_SC]; // 1KB
    extern __shared__ float s_act[];  // K floats (launcher passes K*4, as v1)

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_SHARED[threadIdx.x];

    // Stage both 8-row weight tiles up front (one commit group each) so they
    // arrive while phase 1 computes the activations.
    #pragma unroll
    for (unsigned int p = 0; p < S5_DN_ROW_TILES; p++) {
        for (unsigned int i = threadIdx.x; i < 8 * (S5_TILE_PK / 16); i += BLOCK_SIZE) {
            const unsigned int r = i / (S5_TILE_PK / 16);
            const unsigned int c = i % (S5_TILE_PK / 16);
            const unsigned int n = row_base + p * 8 + r;
            s5_cp_async_16(&s_wq[p][r][c * 16],
                B_packed + (unsigned long long)n * half_K + c * 16, n < N);
        }
        for (unsigned int i = threadIdx.x; i < 8 * (S5_TILE_SC / 16); i += BLOCK_SIZE) {
            const unsigned int r = i / (S5_TILE_SC / 16);
            const unsigned int c = i % (S5_TILE_SC / 16);
            const unsigned int n = row_base + p * 8 + r;
            s5_cp_async_16(&s_sc[p][r][c * 16],
                B_scale + (unsigned long long)n * num_groups + c * 16, n < N);
        }
        s5_cp_async_commit();
    }

    // Phase 1: Cooperatively precompute SiLU(gate)*up into shared memory —
    // copied verbatim from the serial kernel (incl. the DeepSeek-V4 routed
    // SWIGLU clamp) so s_act is bit-identical.
    const float SWIGLU_LIMIT = 10.0f;
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        if (!is_shared) {
            gf = fminf(gf, SWIGLU_LIMIT);
            uf = fminf(fmaxf(uf, -SWIGLU_LIMIT), SWIGLU_LIMIT);
        }
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();

    __nv_bfloat16* out_base = is_shared ? sh_down_out : (C + (unsigned long long)expert_slot * N);

    // Phase 2: GEMV over the 2 staged row tiles.
    #pragma unroll
    for (unsigned int p = 0; p < S5_DN_ROW_TILES; p++) {
        if (p == 0) asm volatile("cp.async.wait_group 1;");
        else        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        const unsigned int n1 = row_base + p * 8 + local_out * 2;
        const unsigned int n2 = n1 + 1;
        const bool have_n1 = (n1 < N);
        const bool have_n2 = (n2 < N);
        const unsigned int r1 = local_out * 2, r2 = r1 + 1;
        if (!have_n1) continue;

        float acc1 = 0.0f, acc2 = 0.0f;

        // Same ascending per-lane k16 walk as the serial kernel (k16 += 32).
        for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
            const unsigned int base_k = k16 * 16;

            unsigned long long packed8_1 = *(const unsigned long long*)&s_wq[p][r1][k16 * 8];
            unsigned char sb1 = s_sc[p][r1][k16];
            float sc1 = atlas_dec_e4m3(sb1) * s2;

            unsigned long long packed8_2 = have_n2 ?
                *(const unsigned long long*)&s_wq[p][r2][k16 * 8] : 0;
            unsigned char sb2 = have_n2 ? s_sc[p][r2][k16] : 0;
            float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;

            #pragma unroll
            for (int b = 0; b < 8; b++) {
                float al = s_act[base_k + b * 2];
                float ah = s_act[base_k + b * 2 + 1];

                unsigned char bv1 = (unsigned char)(packed8_1 >> (b * 8));
                float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
                unsigned char bv2 = (unsigned char)(packed8_2 >> (b * 8));
                float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;

                acc1 += al * w1l + ah * w1h;
                acc2 += al * w2l + ah * w2h;
            }
        }

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        if (lane == 0) out_base[n1] = __float2bfloat16(acc1);

        if (have_n2) {
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
            if (lane == 0) out_base[n2] = __float2bfloat16(acc2);
        }
    }
}
