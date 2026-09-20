// SPDX-License-Identifier: AGPL-3.0-only
//
// `atlas_glm53_dsa_dense_causal_fast_bf16`: the GLM-5.3 DSA prefill attention,
// specialised for the two facts the generic HDIM=512 template cannot assume.
//
// The donor is `kernels/gb10/common/prefill_paged_compute_512.cuh` (compiled
// for GLM as `atlas_glm53_dsa_dense_causal_bf16`). It is a Gemma-4 paged
// full-attention template; at the GLM prefill shape it measures 15.8 ms per
// layer, 174 ms per prefill over 11 layers, which is ~17 TFLOPS on a 275
// GFLOP/layer problem. Two structural costs, both stated in the donor's own
// header, account for the gap:
//
//  1. "Single-buffered K ... V load: warps 2-7". The template keeps K and V in
//     SEPARATE 32 KB smem tiles and loads each one per KV block. GLM's DSA is
//     MLA: `Glm53DsaSelectedAttentionKernels::launch_dense_causal` passes
//     `latent_cache_bf16.ptr` as BOTH the K and the V argument, so those two
//     tiles hold byte-identical data. Half of the KV traffic, and the whole
//     warp-specialised V-load phase, is redundant. Here V IS K: one tile, one
//     load, and the PV MMA reads its B operand out of the same buffer the QK
//     MMA read its own B operand from (the tile is [kv][dim] either way; QK
//     walks it as [position][dim] and PV as [dim][position]).
//
//  2. "PAD_KV=0 (saves 1.5 KB; bank conflicts in K/V smem reads accepted as a
//     v1 perf cost -- correctness first)". With an unpadded 512-half row the
//     stride is 1024 B = 256 words = 0 mod 32, so the eight lane-groups of a
//     warp (group = lane >> 2) read eight different rows that all land on the
//     same bank: an 8-way conflict on every Q and K fragment load, in the
//     innermost loop of both MMA phases. Dropping the V tile frees the budget
//     to pad the row stride to 520 halves = 1040 B = 260 words = 4 mod 32, so
//     the eight groups hit eight distinct banks.
//
// Smem: Q[32][520] 33,280 + K[32][520] 33,280 + P[32][40] 2,560 + m/l 256
//     = 69,376 B, inside the 99 KB per-block opt-in cap on sm_121.
//
// NUMERICS: every arithmetic operation, its order, and the bf16 rounding of P
// before the PV MMA are unchanged from the donor. Only which bytes are read
// from where changes, so the result is bit-identical. That is the acceptance
// test -- the logits dump must match the donor arm exactly.
//
// PRECONDITIONS (the caller must guarantee; the kernel re-checks and returns):
//   V_cache == K_cache, block_table == nullptr (contiguous cache),
//   num_kv_heads == 1, head_dim == 512, causal_mask_enabled == 1,
//   sliding_window == 0.

#include <cuda_bf16.h>

#define GF_BR 32U
#define GF_BC 32U
#define GF_HDIM 512U
#define GF_STRIDE 520U        // 512 + 8: 4 mod 32 words, breaks the group conflict
#define GF_PAD_P 8U
#define GF_P_STRIDE (GF_BC + GF_PAD_P)
#define GF_NTILES 16U         // (512/8)/4 column groups
#define GF_THREADS 256U

__device__ __forceinline__ void gf_cp16(void *smem_dst, const void *gmem_src) {
    unsigned s = __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem_src));
}

// Load one [32][512] KV tile from the contiguous latent cache into a padded
// [32][GF_STRIDE] smem tile. Rows past the end of the sequence are zeroed.
__device__ __forceinline__ void gf_load_kv(
        const __nv_bfloat16 *__restrict__ cache, __nv_bfloat16 *smem,
        unsigned int kv_start, unsigned int kv_len,
        unsigned int tid, unsigned int stride) {
    const unsigned int cpr = GF_HDIM / 8U;             // 64 chunks of 8 halves
    for (unsigned int idx = tid; idx < GF_BC * cpr; idx += stride) {
        const unsigned int row = idx / cpr;
        const unsigned int col = (idx % cpr) * 8U;
        const unsigned int pos = kv_start + row;
        if (pos < kv_len) {
            gf_cp16(&smem[row * GF_STRIDE + col],
                    (const void *) (cache + (unsigned long long) pos * GF_HDIM + col));
        } else {
            *((uint4 *) &smem[row * GF_STRIDE + col]) = make_uint4(0U, 0U, 0U, 0U);
        }
    }
}

extern "C" __global__ void atlas_glm53_dsa_dense_causal_fast_bf16(
        const __nv_bfloat16 *__restrict__ Q,
        const __nv_bfloat16 *__restrict__ K_cache,
        const __nv_bfloat16 *__restrict__ V_cache,
        __nv_bfloat16 *__restrict__ O,
        const int *__restrict__ block_table,
        const unsigned int q_len,
        const unsigned int kv_len,
        const unsigned int q_offset,
        const unsigned int num_q_heads,
        const unsigned int num_kv_heads,
        const unsigned int head_dim,
        const unsigned int cache_block_size,
        const unsigned int sliding_window,
        const unsigned int causal_mask_enabled,
        const float inv_sqrt_d) {
    // Specialisation guards: anything outside the GLM DSA prefill contract
    // must not silently produce a different answer.
    if (block_table != nullptr || V_cache != K_cache || num_kv_heads != 1U ||
        head_dim != GF_HDIM || causal_mask_enabled != 1U || sliding_window != 0U ||
        cache_block_size != 1U || blockDim.x != GF_THREADS) {
        return;
    }
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32U;
    const unsigned int lane_id = tid % 32U;
    if (q_head >= num_q_heads) {
        return;
    }
    const unsigned int q_start = q_block * GF_BR;
    if (q_start >= q_len) {
        return;
    }
    const unsigned int q_tile_end = min(q_start + GF_BR, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;

    extern __shared__ __align__(16) unsigned char gf_smem[];
    __nv_bfloat16 *smem_Q = reinterpret_cast<__nv_bfloat16 *>(gf_smem);
    __nv_bfloat16 *smem_K = smem_Q + GF_BR * GF_STRIDE;
    __nv_bfloat16 *smem_P = smem_K + GF_BC * GF_STRIDE;
    float *smem_ml = reinterpret_cast<float *>(smem_P + GF_BR * GF_P_STRIDE);

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3U;
    const unsigned int qk_warp_m = (warp_id & 1U) * 16U;
    const unsigned int pv_warp_m = (warp_id & 1U) * 16U;
    const unsigned int pv_n_start = (warp_id >> 1) * GF_NTILES;

    float acc_o[GF_NTILES][4];
    #pragma unroll
    for (int i = 0; i < (int) GF_NTILES; i++) {
        acc_o[i][0] = 0.f; acc_o[i][1] = 0.f; acc_o[i][2] = 0.f; acc_o[i][3] = 0.f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.f, l_r1 = 0.f;

    unsigned int num_kv_blocks = (kv_len + GF_BC - 1U) / GF_BC;
    {
        unsigned int mx = (q_offset + q_tile_end - 1U) / GF_BC;
        num_kv_blocks = min(num_kv_blocks, mx + 1U);
    }

    // === Q tile + K[0] ===
    {
        const unsigned int cpr = GF_HDIM / 8U;
        for (unsigned int idx = tid; idx < GF_BR * cpr; idx += GF_THREADS) {
            const unsigned int row = idx / cpr;
            const unsigned int col = (idx % cpr) * 8U;
            if (q_start + row < q_len) {
                gf_cp16(&smem_Q[row * GF_STRIDE + col],
                        (const void *) &Q[(q_start + row) * q_seq_stride +
                                          q_head * head_dim + col]);
            } else {
                *((uint4 *) &smem_Q[row * GF_STRIDE + col]) = make_uint4(0U, 0U, 0U, 0U);
            }
        }
        if (num_kv_blocks > 0U) {
            gf_load_kv(K_cache, smem_K, 0U, kv_len, tid, GF_THREADS);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const unsigned int kv_start = kv_block * GF_BC;
        const unsigned int kv_end = min(kv_start + GF_BC, kv_len);
        const unsigned int kv_tile_len = kv_end - kv_start;

        // === QK^T: warps 0-1, 16 Q rows each (warps 2-7 idle; the donor's
        // V load they used to overlap with no longer exists) ===
        float acc_s[4][4];
        if (warp_id < 2U) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                acc_s[i][0] = 0; acc_s[i][1] = 0; acc_s[i][2] = 0; acc_s[i][3] = 0;
            }
            const unsigned short *sQ = (const unsigned short *) smem_Q;
            const unsigned short *sK = (const unsigned short *) smem_K;
            #pragma unroll
            for (unsigned int ks = 0; ks < (GF_HDIM / 16U); ks++) {
                const unsigned int kb = ks * 16U;
                const unsigned int ar0 = qk_warp_m + group_id, ar1 = ar0 + 8U;
                const unsigned int ac0 = kb + tid_in_group * 2U, ac1 = ac0 + 8U;
                const unsigned int a0 = *(const unsigned int *) &sQ[ar0 * GF_STRIDE + ac0];
                const unsigned int a1 = *(const unsigned int *) &sQ[ar1 * GF_STRIDE + ac0];
                const unsigned int a2 = *(const unsigned int *) &sQ[ar0 * GF_STRIDE + ac1];
                const unsigned int a3 = *(const unsigned int *) &sQ[ar1 * GF_STRIDE + ac1];
                #pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    const unsigned int nc = nt * 8U + group_id;
                    const unsigned int k0 = kb + tid_in_group * 2U, k1 = k0 + 8U;
                    const unsigned int b0 = ((unsigned int) sK[nc * GF_STRIDE + k0 + 1] << 16) |
                                            (unsigned int) sK[nc * GF_STRIDE + k0];
                    const unsigned int b1 = ((unsigned int) sK[nc * GF_STRIDE + k1 + 1] << 16) |
                                            (unsigned int) sK[nc * GF_STRIDE + k1];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                                 : "=f"(acc_s[nt][0]), "=f"(acc_s[nt][1]),
                                   "=f"(acc_s[nt][2]), "=f"(acc_s[nt][3])
                                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                                   "f"(acc_s[nt][0]), "f"(acc_s[nt][1]),
                                   "f"(acc_s[nt][2]), "f"(acc_s[nt][3]));
                }
            }

            const unsigned int row0 = qk_warp_m + group_id, row1 = row0 + 8U;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                acc_s[nt][0] *= inv_sqrt_d; acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d; acc_s[nt][3] *= inv_sqrt_d;
                const unsigned int c0 = nt * 8U + tid_in_group * 2U, c1 = c0 + 1U;
                const unsigned int qr0 = q_offset + q_start + row0;
                const unsigned int qr1 = q_offset + q_start + row1;
                if (kv_start + c0 > qr0) acc_s[nt][0] = -1e30f;
                if (kv_start + c1 > qr0) acc_s[nt][1] = -1e30f;
                if (kv_start + c0 > qr1) acc_s[nt][2] = -1e30f;
                if (kv_start + c1 > qr1) acc_s[nt][3] = -1e30f;
                if (c0 >= kv_tile_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
                if (c1 >= kv_tile_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
                if (row0 >= q_tile_len) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
                if (row1 >= q_tile_len) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
            }

            float rmax0 = -1e30f, rmax1 = -1e30f;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                rmax0 = fmaxf(rmax0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
                rmax1 = fmaxf(rmax1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
            }
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

            const float mn0 = fmaxf(m_r0, rmax0);
            if (mn0 != m_r0) {
                const float eo0 = __expf(m_r0 - mn0);
                l_r0 *= eo0;
                #pragma unroll
                for (int i = 0; i < (int) GF_NTILES; i++) { acc_o[i][0] *= eo0; acc_o[i][1] *= eo0; }
                m_r0 = mn0;
            }
            const float mn1 = fmaxf(m_r1, rmax1);
            if (mn1 != m_r1) {
                const float eo1 = __expf(m_r1 - mn1);
                l_r1 *= eo1;
                #pragma unroll
                for (int i = 0; i < (int) GF_NTILES; i++) { acc_o[i][2] *= eo1; acc_o[i][3] *= eo1; }
                m_r1 = mn1;
            }

            float sum0 = 0, sum1 = 0;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const float p00 = __expf(acc_s[nt][0] - m_r0);
                const float p01 = __expf(acc_s[nt][1] - m_r0);
                const float p10 = __expf(acc_s[nt][2] - m_r1);
                const float p11 = __expf(acc_s[nt][3] - m_r1);
                sum0 += p00 + p01; sum1 += p10 + p11;
                const unsigned int c0 = nt * 8U + tid_in_group * 2U;
                smem_P[row0 * GF_P_STRIDE + c0]      = __float2bfloat16(p00);
                smem_P[row0 * GF_P_STRIDE + c0 + 1]  = __float2bfloat16(p01);
                smem_P[row1 * GF_P_STRIDE + c0]      = __float2bfloat16(p10);
                smem_P[row1 * GF_P_STRIDE + c0 + 1]  = __float2bfloat16(p11);
            }
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
            l_r0 += sum0; l_r1 += sum1;

            if (tid_in_group == 0U) {
                smem_ml[row0 * 2U] = m_r0; smem_ml[row0 * 2U + 1U] = l_r0;
                smem_ml[row1 * 2U] = m_r1; smem_ml[row1 * 2U + 1U] = l_r1;
            }
        }
        __syncthreads();

        // Warps 2-7 rescale their accumulators to the current row max.
        if (warp_id >= 2U) {
            const unsigned int r0 = pv_warp_m + group_id, r1 = r0 + 8U;
            const float cm0 = smem_ml[r0 * 2U], cm1 = smem_ml[r1 * 2U];
            if (cm0 != m_r0) {
                const float er0 = __expf(m_r0 - cm0);
                #pragma unroll
                for (int i = 0; i < (int) GF_NTILES; i++) { acc_o[i][0] *= er0; acc_o[i][1] *= er0; }
                m_r0 = cm0;
            }
            if (cm1 != m_r1) {
                const float er1 = __expf(m_r1 - cm1);
                #pragma unroll
                for (int i = 0; i < (int) GF_NTILES; i++) { acc_o[i][2] *= er1; acc_o[i][3] *= er1; }
                m_r1 = cm1;
            }
        }

        // === PV MMA (all 8 warps). B operand is the SAME tile as QK's:
        // V_cache == K_cache, and the tile is [position][dim] for both. ===
        {
            const unsigned short *sP = (const unsigned short *) smem_P;
            const unsigned short *sV = (const unsigned short *) smem_K;
            #pragma unroll
            for (unsigned int ks = 0; ks < 2U; ks++) {
                const unsigned int ko = ks * 16U;
                const unsigned int ar0 = pv_warp_m + group_id, ar1 = ar0 + 8U;
                const unsigned int ac0 = ko + tid_in_group * 2U, ac1 = ac0 + 8U;
                const unsigned int a0 = *(const unsigned int *) &sP[ar0 * GF_P_STRIDE + ac0];
                const unsigned int a1 = *(const unsigned int *) &sP[ar1 * GF_P_STRIDE + ac0];
                const unsigned int a2 = *(const unsigned int *) &sP[ar0 * GF_P_STRIDE + ac1];
                const unsigned int a3 = *(const unsigned int *) &sP[ar1 * GF_P_STRIDE + ac1];
                #pragma unroll
                for (int nt = 0; nt < (int) GF_NTILES; nt++) {
                    const unsigned int nc = (pv_n_start + nt) * 8U + group_id;
                    const unsigned int k0 = ko + tid_in_group * 2U, k1 = k0 + 8U;
                    const unsigned int b0 = ((unsigned int) sV[(k0 + 1U) * GF_STRIDE + nc] << 16) |
                                            (unsigned int) sV[k0 * GF_STRIDE + nc];
                    const unsigned int b1 = ((unsigned int) sV[(k1 + 1U) * GF_STRIDE + nc] << 16) |
                                            (unsigned int) sV[k1 * GF_STRIDE + nc];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                                 : "=f"(acc_o[nt][0]), "=f"(acc_o[nt][1]),
                                   "=f"(acc_o[nt][2]), "=f"(acc_o[nt][3])
                                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                                   "f"(acc_o[nt][0]), "f"(acc_o[nt][1]),
                                   "f"(acc_o[nt][2]), "f"(acc_o[nt][3]));
                }
            }
        }
        __syncthreads();

        if (kv_block + 1U < num_kv_blocks) {
            gf_load_kv(K_cache, smem_K, (kv_block + 1U) * GF_BC, kv_len, tid, GF_THREADS);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
        }
    }

    // === Normalise and store ===
    {
        const unsigned int r0 = pv_warp_m + group_id, r1 = r0 + 8U;
        float il0, il1;
        if (warp_id < 2U) {
            il0 = (l_r0 > 0) ? (1.f / l_r0) : 0.f;
            il1 = (l_r1 > 0) ? (1.f / l_r1) : 0.f;
        } else {
            const float lv0 = smem_ml[r0 * 2U + 1U], lv1 = smem_ml[r1 * 2U + 1U];
            il0 = (lv0 > 0) ? (1.f / lv0) : 0.f;
            il1 = (lv1 > 0) ? (1.f / lv1) : 0.f;
        }
        __nv_bfloat16 *ob = O + q_head * head_dim;
        #pragma unroll
        for (int nt = 0; nt < (int) GF_NTILES; nt++) {
            const unsigned int c0 = (pv_n_start + nt) * 8U + tid_in_group * 2U;
            const unsigned int gr0 = q_start + r0, gr1 = q_start + r1;
            if (gr0 < q_len && r0 < q_tile_len && c0 < head_dim) {
                const unsigned int lo = (unsigned int) __bfloat16_as_ushort(
                    __float2bfloat16(acc_o[nt][0] * il0));
                const unsigned int hi = (unsigned int) __bfloat16_as_ushort(
                    __float2bfloat16(acc_o[nt][1] * il0));
                *(unsigned int *) &ob[gr0 * q_seq_stride + c0] = lo | (hi << 16);
            }
            if (gr1 < q_len && r1 < q_tile_len && c0 < head_dim) {
                const unsigned int lo = (unsigned int) __bfloat16_as_ushort(
                    __float2bfloat16(acc_o[nt][2] * il1));
                const unsigned int hi = (unsigned int) __bfloat16_as_ushort(
                    __float2bfloat16(acc_o[nt][3] * il1));
                *(unsigned int *) &ob[gr1 * q_seq_stride + c0] = lo | (hi << 16);
            }
        }
    }
}
