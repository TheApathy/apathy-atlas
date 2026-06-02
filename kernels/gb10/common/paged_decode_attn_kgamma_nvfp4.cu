// SPDX-License-Identifier: AGPL-3.0-only
//
// K=γ-fused Paged Decode Attention — NVFP4 KV cache variant.
//
// FlashAttention-v2 inspired Q-tile fusion for DFlash K=γ verify
// (typically QTILE = γ+1 = 17 queries against ONE shared block_table /
// KV history). The existing `paged_decode_attn_nvfp4` kernel launches
// one CTA per (q_head, query) so K and V are loaded once *per query*
// — at QTILE=17 that's 17× redundant HBM traffic over the KV history.
//
// This kernel collapses the QTILE axis into a single CTA per q_head:
//
//   Grid:  (num_q_heads, 1, 1)
//   Block: (256, 1, 1) = 8 warps × 32 lanes
//
// Each WARP owns a slice of queries (ceil(QTILE / NUM_WARPS) per warp)
// and scans the full KV history for its queries. Online softmax stats
// + output accumulator live in per-lane registers — no shared memory
// for the accumulator, only the E2M1 LUT. This keeps the kernel within
// the 100 KB/SM smem budget on sm_120 (only the 64-byte LUT lives there).
//
// All 8 warps issue identical K/V global loads for any given position,
// so HBM traffic per position is amortized via L1/L2 reuse across warps
// (vs the legacy kernel where 17 separate CTAs miss the cache for the
// same KV history). Inside each warp, one K-load × QPER_WARP queries
// (typically 2-3) gives a direct 2-3× HBM reduction; across-warp L2
// reuse further compounds.
//
// Caller contract:
//   - num_seqs == 1 (one real sequence). num_qtile = γ+1 ≤ QTILE_MAX.
//   - Q tensor: [num_qtile, num_q_heads, head_dim] BF16, q_stride sep.
//   - O tensor: [num_qtile, num_q_heads, head_dim] BF16.
//   - block_tables: [num_qtile, max_blocks_per_seq] i32 — all rows
//     identical (verify_*.rs replicates one block_table per query row).
//   - seq_lens: [num_qtile] i32, one per query (causal cutoff).
//   - kv_indirection MUST be nullptr — tree-aware path uses legacy kernel.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32
#ifndef HDIM
#define HDIM 256
#endif
#define VEC_BF16 (HDIM / WARP_SIZE)        // 8 for HDIM=256
#define VEC_U32  (HDIM / (WARP_SIZE * 2))  // 4 for HDIM=256
#define NUM_WARPS 8
#define NVFP4_GROUP_SIZE 16
// Hard upper bound per CTA. With NUM_WARPS=8 and QPER_WARP up to 4
// (this gives QTILE_MAX = 32), per-lane register cost is
//   q_reg : QPER_WARP * VEC_BF16 = 32 floats
//   o_reg : QPER_WARP * VEC_BF16 = 32 floats
//   m,l   : QPER_WARP * 2        = 8 floats
// Plus loop-local k_vec/v_vec (16 floats). Total ~90 regs/lane — well
// under the 255-reg/lane limit. QTILE_MAX=32 covers γ ≤ 31.
#ifndef QTILE_MAX
#define QTILE_MAX 32
#endif
#define QPER_WARP_MAX ((QTILE_MAX + NUM_WARPS - 1) / NUM_WARPS)

// ---- Helpers (copied from paged_decode_attn_nvfp4.cu) ----

__device__ __forceinline__ void unpack2_bf16(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ float fp8e4m3_to_f32(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

__device__ __forceinline__ void nvfp4_dequant_local(
    const unsigned char* data_ptr,
    const unsigned char* scale_ptr,
    const float* lut,
    float* out
) {
    float gs = fp8e4m3_to_f32((__nv_fp8_storage_t)*scale_ptr);
#if VEC_BF16 == 8
    unsigned int pk = *(const unsigned int*)data_ptr;
    out[0] = lut[(pk)       & 0xF] * gs;
    out[1] = lut[(pk >> 4)  & 0xF] * gs;
    out[2] = lut[(pk >> 8)  & 0xF] * gs;
    out[3] = lut[(pk >> 12) & 0xF] * gs;
    out[4] = lut[(pk >> 16) & 0xF] * gs;
    out[5] = lut[(pk >> 20) & 0xF] * gs;
    out[6] = lut[(pk >> 24) & 0xF] * gs;
    out[7] = lut[pk >> 28]         * gs;
#elif VEC_BF16 == 4
    unsigned short pk = *(const unsigned short*)data_ptr;
    out[0] = lut[(pk)       & 0xF] * gs;
    out[1] = lut[(pk >> 4)  & 0xF] * gs;
    out[2] = lut[(pk >> 8)  & 0xF] * gs;
    out[3] = lut[pk >> 12]         * gs;
#else
    #error "Unsupported VEC_BF16"
#endif
}

// ============================================================================
// K=γ-fused NVFP4 paged decode attention
// ============================================================================

extern "C" __global__ void paged_decode_attn_kgamma_nvfp4(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    const unsigned int num_qtile,
    const int* __restrict__ kv_indirection,
    const int* __restrict__ kv_indir_base_ptr,
    const unsigned int kv_indir_stride
) {
    const unsigned int q_head  = blockIdx.x;
    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;
    if (num_qtile == 0 || num_qtile > QTILE_MAX) return;

    (void)kv_indirection; (void)kv_indir_base_ptr; (void)kv_indir_stride;
    (void)head_dim;  // implicit via HDIM template
    (void)max_blocks_per_seq;

    // ---- E2M1 dequant LUT in shared memory ----
    __shared__ float e2m1_lut[16];
    if (tid < 16) {
        const float lut_init[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
           -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
        };
        e2m1_lut[tid] = lut_init[tid];
    }
    __syncthreads();

    // ---- Address arithmetic ----
    const unsigned int gqa_ratio   = num_q_heads / num_kv_heads;
    const unsigned int kv_head     = q_head / gqa_ratio;
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;

    const unsigned int head_data_bytes  = HDIM / 2;
    const unsigned int head_scale_bytes = HDIM / NVFP4_GROUP_SIZE;
    const unsigned int token_data_stride  = num_kv_heads * head_data_bytes;
    const unsigned int token_scale_stride = num_kv_heads * head_scale_bytes;
    const unsigned int kv_data_offset  = kv_head * head_data_bytes + lane_id * (VEC_BF16 / 2);
    const unsigned int kv_scale_offset = kv_head * head_scale_bytes + (lane_id * VEC_BF16 / NVFP4_GROUP_SIZE);

    // All QTILE queries share one block_table (K=γ verify replicates).
    const int* my_block_table = block_tables;

    // ---- Determine which queries this WARP owns ----
    // Round-robin slice: warp w gets queries [w, w+NUM_WARPS, w+2*NUM_WARPS, ...].
    // This balances load: warps see distinct seq_lens that differ by only ±1,
    // so a contiguous slice would skew the last warp; round-robin balances.
    unsigned int my_qs[QPER_WARP_MAX];
    unsigned int my_count = 0;
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
        unsigned int q = warp_id + (unsigned int)slot * NUM_WARPS;
        if (q < num_qtile) {
            my_qs[slot] = q;
            my_count++;
        } else {
            my_qs[slot] = 0;  // ignored
        }
    }
    if (my_count == 0) return;

    // ---- Load THIS WARP's Q-tile into per-lane registers ----
    float q_reg[QPER_WARP_MAX][VEC_BF16];
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
        if ((unsigned int)slot < my_count) {
            unsigned int q = my_qs[slot];
            const unsigned int* q32 = (const unsigned int*)(Q
                + (unsigned long long)q * q_stride
                + (unsigned long long)q_head * HDIM
                + vec_offset_bf16);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                unpack2_bf16(q32[i], q_reg[slot][2*i], q_reg[slot][2*i + 1]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) q_reg[slot][i] = 0.0f;
        }
    }

    // Per-query softmax + output accumulators (registers).
    float m[QPER_WARP_MAX], l[QPER_WARP_MAX];
    float o_reg[QPER_WARP_MAX][VEC_BF16];
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
        m[slot] = -1e30f;
        l[slot] = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) o_reg[slot][i] = 0.0f;
    }

    // Per-query causal cutoffs.
    unsigned int my_sl[QPER_WARP_MAX];
    unsigned int max_sl = 0;
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
        my_sl[slot] = ((unsigned int)slot < my_count) ? (unsigned int)seq_lens[my_qs[slot]] : 0;
        if (my_sl[slot] > max_sl) max_sl = my_sl[slot];
    }
    if (max_sl == 0) return;

    // ---- Main loop: scan KV positions, fan over this warp's queries ----
    for (unsigned int pos = 0; pos < max_sl; pos++) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset  = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];

        const unsigned char* k_block = K_cache + (unsigned long long)physical_block * block_stride_bytes;
        const unsigned char* v_block = V_cache + (unsigned long long)physical_block * block_stride_bytes;

        const unsigned char* kd = k_block + block_offset * token_data_stride + kv_data_offset;
        const unsigned char* ks = k_block + data_section_bytes + block_offset * token_scale_stride + kv_scale_offset;
        const unsigned char* vd = v_block + block_offset * token_data_stride + kv_data_offset;
        const unsigned char* vs = v_block + data_section_bytes + block_offset * token_scale_stride + kv_scale_offset;

        float k_vec[VEC_BF16], v_vec[VEC_BF16];
        nvfp4_dequant_local(kd, ks, e2m1_lut, k_vec);
        nvfp4_dequant_local(vd, vs, e2m1_lut, v_vec);

        // For each owned query, accumulate.
        #pragma unroll
        for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
            if ((unsigned int)slot >= my_count) break;
            if (pos >= my_sl[slot]) continue;  // causal mask

            // Partial dot (this lane's HDIM slice).
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) dot += q_reg[slot][i] * k_vec[i];
            // Full warp reduction.
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);
            float score = dot * inv_sqrt_d;

            // Online softmax (register).
            float m_new = fmaxf(m[slot], score);
            float exp_old = __expf(m[slot] - m_new);
            float exp_new = __expf(score   - m_new);
            l[slot] = l[slot] * exp_old + exp_new;
            m[slot] = m_new;

            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) {
                o_reg[slot][i] = o_reg[slot][i] * exp_old + exp_new * v_vec[i];
            }
        }
    }

    // ---- Write outputs (each warp writes its own queries) ----
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
        if ((unsigned int)slot >= my_count) break;
        unsigned int q = my_qs[slot];
        float inv_l = (l[slot] > 0.0f) ? (1.0f / l[slot]) : 0.0f;
        unsigned int* o32 = (unsigned int*)(O
            + (unsigned long long)q * num_q_heads * HDIM
            + (unsigned long long)q_head * HDIM
            + vec_offset_bf16);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0 = o_reg[slot][2*i]     * inv_l;
            float v1 = o_reg[slot][2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}

// ============================================================================
// K=γ-fused NVFP4 paged decode attention — VEC variant
// ============================================================================
//
// 2-position dequant batching: in the KV scan, each iteration issues all
// FOUR NVFP4 dequants (K0, V0, K1, V1) back-to-back so the compiler can
// interleave the 4 global loads with the unpack ALU work. Same total HBM
// bytes — but per-pos amortized launch + tile-setup overhead is halved,
// and the LSU can overlap the second pair of loads with the first pair's
// unpack/multiply. With both dot products and softmax updates for each
// owned query also batched (2 positions back-to-back), the per-warp
// reduction's branch-divergence cost amortizes over 2 positions instead
// of 1.
//
// NUM_WARPS kept at 8 (same as baseline) — earlier experiment with
// NUM_WARPS=16 regressed by ~17% on counting because 16 warps × QPER_WARP=2
// performed MORE total slot-loop iterations per pos (32) than 8 × QPER_WARP=3
// (~24), AND added warp-scheduler contention on the reduction phase. The
// slot loop is not the bottleneck — the dequant + warp-reduce per pos is —
// so 2-position batching is the focused unlock.
//
// Per-lane register cost at NUM_WARPS=8, QPER_WARP_MAX=4:
//   q_reg, o_reg : 4 * 8 each = 64 floats (same as baseline)
//   k0/k1/v0/v1 inner-loop : 4 * 8 = 32 floats (was 2 * 8 = 16)
//   Total ~110 regs/lane (was ~95) — well under the 255-reg limit.
//
// Grid:  (num_q_heads, 1, 1)
// Block: (256, 1, 1) = 8 warps × 32 lanes  (same as baseline)
//
// Caller contract is identical to `paged_decode_attn_kgamma_nvfp4`.

#define NUM_WARPS_VEC 8
#define QPER_WARP_VEC_MAX ((QTILE_MAX + NUM_WARPS_VEC - 1) / NUM_WARPS_VEC)

extern "C" __global__ void paged_decode_attn_kgamma_nvfp4_vec(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    const unsigned int num_qtile,
    const int* __restrict__ kv_indirection,
    const int* __restrict__ kv_indir_base_ptr,
    const unsigned int kv_indir_stride
) {
    const unsigned int q_head  = blockIdx.x;
    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;
    if (num_qtile == 0 || num_qtile > QTILE_MAX) return;

    (void)kv_indirection; (void)kv_indir_base_ptr; (void)kv_indir_stride;
    (void)head_dim;
    (void)max_blocks_per_seq;

    // ---- E2M1 dequant LUT in shared memory ----
    __shared__ float e2m1_lut[16];
    if (tid < 16) {
        const float lut_init[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
           -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
        };
        e2m1_lut[tid] = lut_init[tid];
    }
    __syncthreads();

    // ---- Address arithmetic ----
    const unsigned int gqa_ratio       = num_q_heads / num_kv_heads;
    const unsigned int kv_head         = q_head / gqa_ratio;
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;

    const unsigned int head_data_bytes   = HDIM / 2;
    const unsigned int head_scale_bytes  = HDIM / NVFP4_GROUP_SIZE;
    const unsigned int token_data_stride  = num_kv_heads * head_data_bytes;
    const unsigned int token_scale_stride = num_kv_heads * head_scale_bytes;
    const unsigned int kv_data_offset  = kv_head * head_data_bytes + lane_id * (VEC_BF16 / 2);
    const unsigned int kv_scale_offset = kv_head * head_scale_bytes + (lane_id * VEC_BF16 / NVFP4_GROUP_SIZE);

    const int* my_block_table = block_tables;

    // ---- Determine which queries this WARP owns (round-robin over 16 warps) ----
    unsigned int my_qs[QPER_WARP_VEC_MAX];
    unsigned int my_count = 0;
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_VEC_MAX; slot++) {
        unsigned int q = warp_id + (unsigned int)slot * NUM_WARPS_VEC;
        if (q < num_qtile) {
            my_qs[slot] = q;
            my_count++;
        } else {
            my_qs[slot] = 0;
        }
    }
    if (my_count == 0) return;

    // ---- Load THIS WARP's Q-tile into per-lane registers ----
    float q_reg[QPER_WARP_VEC_MAX][VEC_BF16];
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_VEC_MAX; slot++) {
        if ((unsigned int)slot < my_count) {
            unsigned int q = my_qs[slot];
            const unsigned int* q32 = (const unsigned int*)(Q
                + (unsigned long long)q * q_stride
                + (unsigned long long)q_head * HDIM
                + vec_offset_bf16);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                unpack2_bf16(q32[i], q_reg[slot][2*i], q_reg[slot][2*i + 1]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) q_reg[slot][i] = 0.0f;
        }
    }

    // Per-query softmax + output accumulators (registers).
    float m[QPER_WARP_VEC_MAX], l[QPER_WARP_VEC_MAX];
    float o_reg[QPER_WARP_VEC_MAX][VEC_BF16];
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_VEC_MAX; slot++) {
        m[slot] = -1e30f;
        l[slot] = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) o_reg[slot][i] = 0.0f;
    }

    // Per-query causal cutoffs.
    unsigned int my_sl[QPER_WARP_VEC_MAX];
    unsigned int max_sl = 0;
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_VEC_MAX; slot++) {
        my_sl[slot] = ((unsigned int)slot < my_count) ? (unsigned int)seq_lens[my_qs[slot]] : 0;
        if (my_sl[slot] > max_sl) max_sl = my_sl[slot];
    }
    if (max_sl == 0) return;

    // ---- Main loop: scan KV positions in PAIRS (2 per iter), tail handles odd ----
    unsigned int pos = 0;
    const unsigned int paired_end = (max_sl >= 1) ? (max_sl - 1) : 0;
    for (; pos < paired_end; pos += 2) {
        const unsigned int pos0 = pos;
        const unsigned int pos1 = pos + 1;

        const unsigned int lb0 = pos0 / block_size;
        const unsigned int bo0 = pos0 % block_size;
        const unsigned int lb1 = pos1 / block_size;
        const unsigned int bo1 = pos1 % block_size;

        const unsigned int pb0 = (unsigned int)my_block_table[lb0];
        const unsigned int pb1 = (unsigned int)my_block_table[lb1];

        const unsigned char* k_blk0 = K_cache + (unsigned long long)pb0 * block_stride_bytes;
        const unsigned char* v_blk0 = V_cache + (unsigned long long)pb0 * block_stride_bytes;
        const unsigned char* k_blk1 = K_cache + (unsigned long long)pb1 * block_stride_bytes;
        const unsigned char* v_blk1 = V_cache + (unsigned long long)pb1 * block_stride_bytes;

        const unsigned char* kd0 = k_blk0 + bo0 * token_data_stride  + kv_data_offset;
        const unsigned char* ks0 = k_blk0 + data_section_bytes + bo0 * token_scale_stride + kv_scale_offset;
        const unsigned char* vd0 = v_blk0 + bo0 * token_data_stride  + kv_data_offset;
        const unsigned char* vs0 = v_blk0 + data_section_bytes + bo0 * token_scale_stride + kv_scale_offset;
        const unsigned char* kd1 = k_blk1 + bo1 * token_data_stride  + kv_data_offset;
        const unsigned char* ks1 = k_blk1 + data_section_bytes + bo1 * token_scale_stride + kv_scale_offset;
        const unsigned char* vd1 = v_blk1 + bo1 * token_data_stride  + kv_data_offset;
        const unsigned char* vs1 = v_blk1 + data_section_bytes + bo1 * token_scale_stride + kv_scale_offset;

        // Issue all 4 dequants for the pair. NVCC will schedule the loads.
        // Same total bytes as 2 separate-iter calls — but the compiler now
        // sees all 4 outputs live simultaneously and can interleave LD/ALU.
        float k0_vec[VEC_BF16], v0_vec[VEC_BF16], k1_vec[VEC_BF16], v1_vec[VEC_BF16];
        nvfp4_dequant_local(kd0, ks0, e2m1_lut, k0_vec);
        nvfp4_dequant_local(vd0, vs0, e2m1_lut, v0_vec);
        nvfp4_dequant_local(kd1, ks1, e2m1_lut, k1_vec);
        nvfp4_dequant_local(vd1, vs1, e2m1_lut, v1_vec);

        // Process both positions for each owned query.
        #pragma unroll
        for (int slot = 0; slot < QPER_WARP_VEC_MAX; slot++) {
            if ((unsigned int)slot >= my_count) break;

            // ---- pos0 ----
            if (pos0 < my_sl[slot]) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) dot += q_reg[slot][i] * k0_vec[i];
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                float score = dot * inv_sqrt_d;

                float m_new = fmaxf(m[slot], score);
                float exp_old = __expf(m[slot] - m_new);
                float exp_new = __expf(score   - m_new);
                l[slot] = l[slot] * exp_old + exp_new;
                m[slot] = m_new;

                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    o_reg[slot][i] = o_reg[slot][i] * exp_old + exp_new * v0_vec[i];
                }
            }

            // ---- pos1 ----
            if (pos1 < my_sl[slot]) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) dot += q_reg[slot][i] * k1_vec[i];
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                float score = dot * inv_sqrt_d;

                float m_new = fmaxf(m[slot], score);
                float exp_old = __expf(m[slot] - m_new);
                float exp_new = __expf(score   - m_new);
                l[slot] = l[slot] * exp_old + exp_new;
                m[slot] = m_new;

                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    o_reg[slot][i] = o_reg[slot][i] * exp_old + exp_new * v1_vec[i];
                }
            }
        }
    }

    // ---- Tail (odd) position ----
    if (pos < max_sl) {
        unsigned int lb = pos / block_size;
        unsigned int bo = pos % block_size;
        unsigned int pb = (unsigned int)my_block_table[lb];

        const unsigned char* k_blk = K_cache + (unsigned long long)pb * block_stride_bytes;
        const unsigned char* v_blk = V_cache + (unsigned long long)pb * block_stride_bytes;

        const unsigned char* kd = k_blk + bo * token_data_stride  + kv_data_offset;
        const unsigned char* ks = k_blk + data_section_bytes + bo * token_scale_stride + kv_scale_offset;
        const unsigned char* vd = v_blk + bo * token_data_stride  + kv_data_offset;
        const unsigned char* vs = v_blk + data_section_bytes + bo * token_scale_stride + kv_scale_offset;

        float k_vec[VEC_BF16], v_vec[VEC_BF16];
        nvfp4_dequant_local(kd, ks, e2m1_lut, k_vec);
        nvfp4_dequant_local(vd, vs, e2m1_lut, v_vec);

        #pragma unroll
        for (int slot = 0; slot < QPER_WARP_VEC_MAX; slot++) {
            if ((unsigned int)slot >= my_count) break;
            if (pos >= my_sl[slot]) continue;

            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) dot += q_reg[slot][i] * k_vec[i];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);
            float score = dot * inv_sqrt_d;

            float m_new = fmaxf(m[slot], score);
            float exp_old = __expf(m[slot] - m_new);
            float exp_new = __expf(score   - m_new);
            l[slot] = l[slot] * exp_old + exp_new;
            m[slot] = m_new;

            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) {
                o_reg[slot][i] = o_reg[slot][i] * exp_old + exp_new * v_vec[i];
            }
        }
    }

    // ---- Write outputs ----
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_VEC_MAX; slot++) {
        if ((unsigned int)slot >= my_count) break;
        unsigned int q = my_qs[slot];
        float inv_l = (l[slot] > 0.0f) ? (1.0f / l[slot]) : 0.0f;
        unsigned int* o32 = (unsigned int*)(O
            + (unsigned long long)q * num_q_heads * HDIM
            + (unsigned long long)q_head * HDIM
            + vec_offset_bf16);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0 = o_reg[slot][2*i]     * inv_l;
            float v1 = o_reg[slot][2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}

// ============================================================================
// K=γ-fused NVFP4 paged decode attention — SPLIT-K variant
// ============================================================================
//
// Task #96 follow-up to task #94: the single-CTA kgamma kernel above hits a
// hard occupancy ceiling because grid = (num_q_heads, 1, 1) = 4 CTAs on a
// 48-SM chip (sm_120 / GB10). Each CTA scans the *entire* KV history.
//
// This split-K variant partitions the KV history across `num_splits` CTAs
// per q_head:
//
//   Grid:  (num_q_heads, num_splits, 1)
//   Block: (256, 1, 1) = 8 warps × 32 lanes
//
// Each split CTA computes a partial online-softmax state (m, l, o) per owned
// query (same per-warp Q-tile fusion + round-robin slicing as the parent
// kernel). Partial results are written to a workspace buffer:
//
//   workspace[q, q_head, split] = { o_reg[HDIM], m, l }   // F32
//
// A separate `paged_decode_attn_kgamma_reduce_nvfp4` kernel combines the
// per-split partials using standard log-sum-exp rescaling and writes the
// final BF16 output.
//
// With num_splits = 12, the grid becomes 4×12 = 48 CTAs — one per SM —
// turning the previously-underutilized 4-CTA launch into a fully populated
// chip. Counting decode (γ=16, dominated by KV scan) is the target workload.
//
// Caller contract (unchanged from the single-CTA kernel):
//   - num_seqs == 1 (one real sequence). num_qtile = γ+1 ≤ QTILE_MAX.
//   - Q tensor: [num_qtile, num_q_heads, head_dim] BF16, q_stride sep.
//   - block_tables: [num_qtile, max_blocks_per_seq] i32 — all rows identical.
//   - seq_lens: [num_qtile] i32, one per query (causal cutoff).
//   - kv_indirection NOT supported (tree-aware path uses legacy kernel).
//   - workspace: F32 buffer of size num_qtile * num_q_heads * num_splits *
//                (HDIM + 2). Caller MUST size BufferSizes::splitk_workspace
//                to fit this (gated by ATLAS_FLASH_ATTN_KGAMMA_SPLITK=1).
//   - num_splits ≥ 1. For num_splits=1, prefer the single-CTA kernel.

extern "C" __global__ void paged_decode_attn_kgamma_nvfp4_splitk(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    float* __restrict__ workspace,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int num_splits,
    const unsigned int q_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    const unsigned int num_qtile
) {
    const unsigned int q_head   = blockIdx.x;
    const unsigned int split_id = blockIdx.y;
    const unsigned int tid      = threadIdx.x;
    const unsigned int warp_id  = tid / WARP_SIZE;
    const unsigned int lane_id  = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;
    if (split_id >= num_splits) return;
    if (num_qtile == 0 || num_qtile > QTILE_MAX) return;

    (void)head_dim;
    (void)max_blocks_per_seq;

    // ---- E2M1 dequant LUT in shared memory ----
    __shared__ float e2m1_lut[16];
    if (tid < 16) {
        const float lut_init[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
           -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
        };
        e2m1_lut[tid] = lut_init[tid];
    }
    __syncthreads();

    // ---- Address arithmetic (identical to single-CTA kernel) ----
    const unsigned int gqa_ratio       = num_q_heads / num_kv_heads;
    const unsigned int kv_head         = q_head / gqa_ratio;
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;

    const unsigned int head_data_bytes   = HDIM / 2;
    const unsigned int head_scale_bytes  = HDIM / NVFP4_GROUP_SIZE;
    const unsigned int token_data_stride  = num_kv_heads * head_data_bytes;
    const unsigned int token_scale_stride = num_kv_heads * head_scale_bytes;
    const unsigned int kv_data_offset  = kv_head * head_data_bytes + lane_id * (VEC_BF16 / 2);
    const unsigned int kv_scale_offset = kv_head * head_scale_bytes + (lane_id * VEC_BF16 / NVFP4_GROUP_SIZE);

    const int* my_block_table = block_tables;

    // ---- Determine which queries this WARP owns (same round-robin) ----
    unsigned int my_qs[QPER_WARP_MAX];
    unsigned int my_count = 0;
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
        unsigned int q = warp_id + (unsigned int)slot * NUM_WARPS;
        if (q < num_qtile) {
            my_qs[slot] = q;
            my_count++;
        } else {
            my_qs[slot] = 0;
        }
    }
    if (my_count == 0) return;

    // ---- Per-query seq lengths + global max ----
    unsigned int my_sl[QPER_WARP_MAX];
    unsigned int max_sl = 0;
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
        my_sl[slot] = ((unsigned int)slot < my_count) ? (unsigned int)seq_lens[my_qs[slot]] : 0;
        if (my_sl[slot] > max_sl) max_sl = my_sl[slot];
    }
    if (max_sl == 0) return;

    // ---- Compute this split's KV range over the GLOBAL max history.
    // Splits partition [0, max_sl). Per-query causal mask still applied
    // inside the inner loop (`pos >= my_sl[slot]` skips). A split whose
    // entire range is past a particular query's seq_len contributes the
    // identity state (m=-inf, l=0) which the reduce kernel safely ignores.
    const unsigned int split_size = (max_sl + num_splits - 1) / num_splits;
    unsigned int kv_start = split_id * split_size;
    unsigned int kv_end   = kv_start + split_size;
    if (kv_end > max_sl) kv_end = max_sl;
    if (kv_start >= max_sl) kv_start = kv_end; // empty split

    // ---- Load THIS WARP's Q-tile into per-lane registers ----
    float q_reg[QPER_WARP_MAX][VEC_BF16];
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
        if ((unsigned int)slot < my_count) {
            unsigned int q = my_qs[slot];
            const unsigned int* q32 = (const unsigned int*)(Q
                + (unsigned long long)q * q_stride
                + (unsigned long long)q_head * HDIM
                + vec_offset_bf16);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                unpack2_bf16(q32[i], q_reg[slot][2*i], q_reg[slot][2*i + 1]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) q_reg[slot][i] = 0.0f;
        }
    }

    // Per-query partial accumulators (registers).
    float m[QPER_WARP_MAX], l[QPER_WARP_MAX];
    float o_reg[QPER_WARP_MAX][VEC_BF16];
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
        m[slot] = -1e30f;
        l[slot] = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) o_reg[slot][i] = 0.0f;
    }

    // ---- Main loop: scan THIS SPLIT's KV positions ----
    for (unsigned int pos = kv_start; pos < kv_end; pos++) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset  = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];

        const unsigned char* k_block = K_cache + (unsigned long long)physical_block * block_stride_bytes;
        const unsigned char* v_block = V_cache + (unsigned long long)physical_block * block_stride_bytes;

        const unsigned char* kd = k_block + block_offset * token_data_stride + kv_data_offset;
        const unsigned char* ks = k_block + data_section_bytes + block_offset * token_scale_stride + kv_scale_offset;
        const unsigned char* vd = v_block + block_offset * token_data_stride + kv_data_offset;
        const unsigned char* vs = v_block + data_section_bytes + block_offset * token_scale_stride + kv_scale_offset;

        float k_vec[VEC_BF16], v_vec[VEC_BF16];
        nvfp4_dequant_local(kd, ks, e2m1_lut, k_vec);
        nvfp4_dequant_local(vd, vs, e2m1_lut, v_vec);

        #pragma unroll
        for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
            if ((unsigned int)slot >= my_count) break;
            if (pos >= my_sl[slot]) continue;  // causal mask per query

            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) dot += q_reg[slot][i] * k_vec[i];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);
            float score = dot * inv_sqrt_d;

            float m_new = fmaxf(m[slot], score);
            float exp_old = __expf(m[slot] - m_new);
            float exp_new = __expf(score   - m_new);
            l[slot] = l[slot] * exp_old + exp_new;
            m[slot] = m_new;

            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) {
                o_reg[slot][i] = o_reg[slot][i] * exp_old + exp_new * v_vec[i];
            }
        }
    }

    // ---- Write partial (m, l, o) per owned query to workspace ----
    // Layout: workspace[q, q_head, split] holds (HDIM + 2) floats:
    //   [o_reg[0..HDIM-1], m, l]
    // Indexed: q * (num_q_heads * num_splits) + q_head * num_splits + split_id,
    // then * (HDIM + 2) for stride.
    const unsigned int ws_stride = HDIM + 2;
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
        if ((unsigned int)slot >= my_count) break;
        unsigned int q = my_qs[slot];
        float* ws_base = workspace
            + ((unsigned long long)q * num_q_heads * num_splits
               + (unsigned long long)q_head * num_splits
               + split_id) * ws_stride;
        // o vector: each lane writes its VEC_BF16 elements
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) {
            ws_base[vec_offset_bf16 + i] = o_reg[slot][i];
        }
        // m, l written once per warp
        if (lane_id == 0) {
            ws_base[HDIM]     = m[slot];
            ws_base[HDIM + 1] = l[slot];
        }
    }
}

// ============================================================================
// K=γ split-K REDUCE: merge num_splits partials → final BF16 output per query
// Grid:  (num_q_heads, num_qtile, 1)
// Block: (32, 1, 1)  — one warp, each lane covers VEC_BF16 elements of HDIM
// ============================================================================

extern "C" __global__ void paged_decode_attn_kgamma_reduce_nvfp4(
    const float* __restrict__ workspace,    // [num_qtile, num_q_heads, num_splits, HDIM+2] F32
    __nv_bfloat16* __restrict__ O,          // [num_qtile, num_q_heads, HDIM] BF16
    const unsigned int num_q_heads,
    const unsigned int num_splits,
    const unsigned int num_qtile
) {
    const unsigned int q_head  = blockIdx.x;
    const unsigned int q       = blockIdx.y;
    const unsigned int lane_id = threadIdx.x;  // 0..31

    if (q_head >= num_q_heads) return;
    if (q >= num_qtile) return;

    const unsigned int vec_off = lane_id * VEC_BF16;
    const unsigned int ws_stride = HDIM + 2;
    const float* ws_base = workspace
        + ((unsigned long long)q * num_q_heads * num_splits
           + (unsigned long long)q_head * num_splits) * ws_stride;

    // ---- Initialize from split 0 ----
    float m_acc = ws_base[HDIM];
    float l_acc = ws_base[HDIM + 1];
    float o_acc[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        o_acc[i] = ws_base[vec_off + i];
    }

    // ---- Merge splits 1..num_splits-1 via online softmax rescaling ----
    for (unsigned int s = 1; s < num_splits; s++) {
        const float* ws = ws_base + (unsigned long long)s * ws_stride;
        float m_s = ws[HDIM];
        float l_s = ws[HDIM + 1];
        // Empty / past-causal split: m_s = -1e30, l_s = 0. Skip.
        if (l_s <= 0.0f) continue;
        // First valid split when initial split 0 was empty: adopt it directly.
        if (l_acc <= 0.0f) {
            m_acc = m_s;
            l_acc = l_s;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) {
                o_acc[i] = ws[vec_off + i];
            }
            continue;
        }
        float m_new   = fmaxf(m_acc, m_s);
        float scale_a = __expf(m_acc - m_new);
        float scale_s = __expf(m_s   - m_new);
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) {
            o_acc[i] = o_acc[i] * scale_a + ws[vec_off + i] * scale_s;
        }
        l_acc = l_acc * scale_a + l_s * scale_s;
        m_acc = m_new;
    }

    // ---- Normalize and write final BF16 output ----
    float inv_l = (l_acc > 0.0f) ? (1.0f / l_acc) : 0.0f;
    unsigned int* o32 = (unsigned int*)(O
        + (unsigned long long)q * num_q_heads * HDIM
        + (unsigned long long)q_head * HDIM
        + vec_off);
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        float v0 = o_acc[2*i]     * inv_l;
        float v1 = o_acc[2*i + 1] * inv_l;
        unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
        unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
        o32[i] = lo | (hi << 16);
    }
}

// ============================================================================
// K=γ-fused NVFP4 paged decode attention — FA2-grafted variant
// ============================================================================
//
// Closer port of FlashAttention-v2's inner-loop technique to Atlas's PTX-only
// model. The baseline kgamma kernel above streams KV one position at a time
// with direct global-memory loads per thread — no SMEM staging, no cp.async
// pipelining, no compute/load overlap. Result: HBM-bound, compute pipeline
// idles waiting on loads.
//
// FA2's `compute_attn_1rowblock` (Dao-AILab/flash-attention csrc/flash_attn/
// src/flash_fwd_kernel.h lines 250-340) gets ~2x throughput vs naive by:
//   1. Tiling K/V into kBlockN = 64-128 chunks loaded once into SMEM
//   2. Double-buffering (kStages=2): cp.async tile N+1 while computing tile N
//   3. cp.async_fence/wait barriers to overlap load with compute
//   4. All threads cooperate on the SMEM load (vectorized 16B per thread)
//
// We graft the same shape onto NVFP4 paged-cache decode:
//
//   Grid:  (num_q_heads, 1, 1)         — same as baseline
//   Block: (256, 1, 1) = 8 warps × 32  — same as baseline
//   TILE_N = 32 positions per KV tile  — covers 1-2 paged blocks (bs=16 or 32)
//
// SMEM layout per stage (double-buffered, 2 stages):
//   k_data[TILE_N][HDIM/2]   bytes (E2M1 nibbles)  → 32 * 128 =  4 KB
//   k_scale[TILE_N][HDIM/16] bytes (FP8 scales)    → 32 *  16 =  0.5 KB
//   v_data[TILE_N][HDIM/2]   bytes                 → 32 * 128 =  4 KB
//   v_scale[TILE_N][HDIM/16] bytes                 → 32 *  16 =  0.5 KB
//                                              per-stage total ≈ 9 KB
//                                              × 2 stages    ≈ 18 KB SMEM
// Plus the 64-byte E2M1 LUT. Well within sm_120's 100 KB/SM budget,
// and small enough to keep occupancy at 1-2 CTAs/SM (the baseline
// kgamma kernel already hits the 4-CTA-per-chip occupancy ceiling).
//
// Inner-loop sequence (per warp):
//   1. ISSUE cp.async loads for tile 0 (all threads cooperate)
//   2. cp.async_fence
//   3. for tile_idx = 0 ... num_tiles - 1:
//        - if tile_idx + 1 < num_tiles: ISSUE cp.async for tile_idx+1
//        - cp.async_wait<1>: wait until tile_idx loads landed
//        - __syncthreads()
//        - for each owned query, for each pos in tile_idx:
//             dequant K from SMEM, dot with Q (in regs), softmax, accum V
//        - __syncthreads() (before SMEM can be overwritten by stage swap)
//
// Per-lane register cost:
//   q_reg, o_reg : QPER_WARP_MAX * VEC_BF16 (≈ 64 floats)
//   m, l         : QPER_WARP_MAX * 2        (8 floats)
//   k_vec, v_vec : VEC_BF16 * 2             (16 floats, inner-loop)
//   ≈ 90-100 regs/lane (similar to baseline; well under 255)
//
// Caller contract is IDENTICAL to `paged_decode_attn_kgamma_nvfp4`. The
// dispatch site (qwen3_attention/trait_impl/multi_seq/attn.rs) selects this
// kernel when `ATLAS_FA2_KGAMMA=1` is set in the env.

// FA2 pipeline config — verified-optimal as of 2026-05-25 benchmark sweep.
//
// SMEM cost per block: FA2_STAGES * 2 * (FA2_KDATA_PER_TILE + FA2_KSCALE_PER_TILE)
//   = STAGES * 2 * (TILE_N*128 + TILE_N*16) = STAGES * TILE_N * 288 bytes.
//
// Tested configs (10-12 runs each, 27B Qwen3.6 DDTree, counting prompt):
//   STAGES=2 TILE=32 (18KB) : mean 45.8 ± 6.9, max 51.4 tok/s  ← KEPT
//   STAGES=4 TILE=32 (36KB) : mean 45.8 ± 2.5, max 50.6 tok/s
//   STAGES=2 TILE=64 (36KB) : mean 43.3 ± 5.9, max 50.9 tok/s
//   STAGES=4 TILE=64 (72KB) : crashes — exceeds 48KB static-SMEM limit
//
// STAGES > 2 is a no-op for latency hiding because the prefetch loop below
// only ever keeps 1 group in flight (`cp.async.wait_group 1` at line ~1256).
// Increasing STAGES would need a prologue that issues N-1 prefetches before
// the main loop + `wait_group N-1` — that's a loop restructuring, not a
// single-knob tweak. STAGES=4 only buys a stdev reduction (6.9 → 2.5) by
// avoiding inflight-prefetch contention but doesn't raise the mean.
// TILE_N=64 is a slight regression: doubling per-tile compute outpaces the
// barrier-amortization win because we still only prefetch 1 tile ahead.
#define FA2_TILE_N    32
#define FA2_STAGES    2
#define FA2_KDATA_PER_TILE  (FA2_TILE_N * (HDIM / 2))       // 32 * 128 = 4096 B
#define FA2_KSCALE_PER_TILE (FA2_TILE_N * (HDIM / NVFP4_GROUP_SIZE))  // 32 * 16 = 512 B
// Each cp.async issues 16 B per call.
#define FA2_CPASYNC_BYTES  16

// Cooperative tile loader: ONE_TILE worth of K_data + K_scale + V_data + V_scale
// for a single stage slot. Falls through to zero-fill for out-of-range positions
// (causal mask handles the result). All 256 threads stride-cooperate; each
// thread issues HDIM-aligned cp.async batches of 16 B.
__device__ __forceinline__ void fa2_issue_tile_loads(
    unsigned char* smem_k_data,        // stage slot: [FA2_TILE_N][HDIM/2]
    unsigned char* smem_k_scale,       // stage slot: [FA2_TILE_N][HDIM/NVFP4_GROUP_SIZE]
    unsigned char* smem_v_data,        // stage slot: [FA2_TILE_N][HDIM/2]
    unsigned char* smem_v_scale,       // stage slot: [FA2_TILE_N][HDIM/NVFP4_GROUP_SIZE]
    const unsigned char* K_cache,
    const unsigned char* V_cache,
    const int* block_table,
    unsigned int tile_start_pos,
    unsigned int tile_end_pos,         // exclusive; min(tile_start + TILE_N, max_sl)
    unsigned int block_size,
    unsigned int kv_head,
    unsigned int num_kv_heads,
    unsigned long long block_stride_bytes,
    unsigned long long data_section_bytes,
    unsigned int tid
) {
    const unsigned int head_data_bytes  = HDIM / 2;             // 128
    const unsigned int head_scale_bytes = HDIM / NVFP4_GROUP_SIZE;  // 16
    const unsigned int token_data_stride  = num_kv_heads * head_data_bytes;
    const unsigned int token_scale_stride = num_kv_heads * head_scale_bytes;
    const unsigned int kv_data_base  = kv_head * head_data_bytes;
    const unsigned int kv_scale_base = kv_head * head_scale_bytes;

    // ---- Data section (HDIM/2 bytes per pos = 128 B; need 8 × 16 B cp.async per pos) ----
    // Layout: each tid handles a different (pos, 16B-chunk) pair, strided across the
    // tile. Total chunks per tile: TILE_N * 8 = 256 (matches our 256 threads → 1 chunk each).
    const unsigned int CHUNKS_PER_POS_DATA = head_data_bytes / FA2_CPASYNC_BYTES;  // 8
    const unsigned int TOTAL_DATA_CHUNKS   = FA2_TILE_N * CHUNKS_PER_POS_DATA;     // 256

    for (unsigned int idx = tid; idx < TOTAL_DATA_CHUNKS; idx += 256) {
        unsigned int local_pos   = idx / CHUNKS_PER_POS_DATA;   // 0..TILE_N-1
        unsigned int chunk_in_pos = idx % CHUNKS_PER_POS_DATA;  // 0..7
        unsigned int abs_pos = tile_start_pos + local_pos;

        unsigned char* dst_k = smem_k_data + local_pos * head_data_bytes
                              + chunk_in_pos * FA2_CPASYNC_BYTES;
        unsigned char* dst_v = smem_v_data + local_pos * head_data_bytes
                              + chunk_in_pos * FA2_CPASYNC_BYTES;
        unsigned int sa_k = __cvta_generic_to_shared(dst_k);
        unsigned int sa_v = __cvta_generic_to_shared(dst_v);

        if (abs_pos < tile_end_pos) {
            unsigned int lb = abs_pos / block_size;
            unsigned int bo = abs_pos % block_size;
            unsigned int pb = (unsigned int)block_table[lb];

            const unsigned char* k_blk = K_cache + (unsigned long long)pb * block_stride_bytes;
            const unsigned char* v_blk = V_cache + (unsigned long long)pb * block_stride_bytes;

            const void* gm_k = (const void*)(k_blk + bo * token_data_stride + kv_data_base
                                             + chunk_in_pos * FA2_CPASYNC_BYTES);
            const void* gm_v = (const void*)(v_blk + bo * token_data_stride + kv_data_base
                                             + chunk_in_pos * FA2_CPASYNC_BYTES);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa_k), "l"(gm_k));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa_v), "l"(gm_v));
        } else {
            // Zero-fill OOB so causal mask + softmax stay numerically safe.
            *((uint4*)dst_k) = make_uint4(0, 0, 0, 0);
            *((uint4*)dst_v) = make_uint4(0, 0, 0, 0);
        }
    }

    // ---- Scale section (HDIM/16 bytes per pos = 16 B; 1 × 16 B cp.async per pos) ----
    // Total chunks per tile: TILE_N = 32. First 32 threads issue these.
    if (tid < FA2_TILE_N) {
        unsigned int local_pos = tid;
        unsigned int abs_pos = tile_start_pos + local_pos;

        unsigned char* dst_ks = smem_k_scale + local_pos * head_scale_bytes;
        unsigned char* dst_vs = smem_v_scale + local_pos * head_scale_bytes;
        unsigned int sa_ks = __cvta_generic_to_shared(dst_ks);
        unsigned int sa_vs = __cvta_generic_to_shared(dst_vs);

        if (abs_pos < tile_end_pos) {
            unsigned int lb = abs_pos / block_size;
            unsigned int bo = abs_pos % block_size;
            unsigned int pb = (unsigned int)block_table[lb];

            const unsigned char* k_blk = K_cache + (unsigned long long)pb * block_stride_bytes;
            const unsigned char* v_blk = V_cache + (unsigned long long)pb * block_stride_bytes;

            const void* gm_ks = (const void*)(k_blk + data_section_bytes
                                              + bo * token_scale_stride + kv_scale_base);
            const void* gm_vs = (const void*)(v_blk + data_section_bytes
                                              + bo * token_scale_stride + kv_scale_base);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa_ks), "l"(gm_ks));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa_vs), "l"(gm_vs));
        } else {
            *((uint4*)dst_ks) = make_uint4(0, 0, 0, 0);
            *((uint4*)dst_vs) = make_uint4(0, 0, 0, 0);
        }
    }
}

extern "C" __global__ void paged_decode_attn_kgamma_nvfp4_fa2(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes,
    const unsigned int num_qtile,
    const int* __restrict__ kv_indirection,
    const int* __restrict__ kv_indir_base_ptr,
    const unsigned int kv_indir_stride
) {
    const unsigned int q_head  = blockIdx.x;
    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;
    if (num_qtile == 0 || num_qtile > QTILE_MAX) return;

    (void)kv_indirection; (void)kv_indir_base_ptr; (void)kv_indir_stride;
    (void)head_dim;
    (void)max_blocks_per_seq;

    // ---- E2M1 dequant LUT in shared memory ----
    __shared__ float e2m1_lut[16];
    // ---- KV tile staging in shared memory (double-buffered) ----
    // Layout per stage: [k_data | k_scale | v_data | v_scale]
    //   k_data:  FA2_TILE_N * HDIM/2  bytes  (4096 B)
    //   k_scale: FA2_TILE_N * HDIM/16 bytes  ( 512 B)
    //   v_data:  FA2_TILE_N * HDIM/2  bytes  (4096 B)
    //   v_scale: FA2_TILE_N * HDIM/16 bytes  ( 512 B)
    // Per-stage = 9216 B, × 2 stages = 18432 B.
    __shared__ __align__(16) unsigned char kv_smem
        [FA2_STAGES][2 * (FA2_KDATA_PER_TILE + FA2_KSCALE_PER_TILE)];

    if (tid < 16) {
        const float lut_init[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
           -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
        };
        e2m1_lut[tid] = lut_init[tid];
    }
    __syncthreads();

    // ---- Per-stage SMEM section pointers (helper macros) ----
    #define K_DATA_PTR(stage)  (&kv_smem[stage][0])
    #define K_SCALE_PTR(stage) (&kv_smem[stage][FA2_KDATA_PER_TILE])
    #define V_DATA_PTR(stage)  (&kv_smem[stage][FA2_KDATA_PER_TILE + FA2_KSCALE_PER_TILE])
    #define V_SCALE_PTR(stage) (&kv_smem[stage][2*FA2_KDATA_PER_TILE + FA2_KSCALE_PER_TILE])

    // ---- Address arithmetic ----
    const unsigned int gqa_ratio       = num_q_heads / num_kv_heads;
    const unsigned int kv_head         = q_head / gqa_ratio;
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;

    // SMEM-side dequant offsets: per-lane slice for the same HDIM partition the
    // dot product uses. Each lane reads VEC_BF16 (8) BF16-equivalent values, which
    // is VEC_BF16/2 = 4 bytes of E2M1 data and lane*VEC_BF16/NVFP4_GROUP_SIZE bytes
    // of scale offset.
    const unsigned int smem_data_lane_off  = lane_id * (VEC_BF16 / 2);  // 4
    const unsigned int smem_scale_lane_off = (lane_id * VEC_BF16) / NVFP4_GROUP_SIZE;  // 0 or 1

    const int* my_block_table = block_tables;

    // ---- Determine which queries this WARP owns (same round-robin as baseline) ----
    // NOTE: do NOT early-return when my_count == 0. The cooperative cp.async
    // tile loader and the per-tile __syncthreads() require ALL warps in the
    // CTA to execute the SAME number of iterations. A warp with my_count==0
    // still participates in tile loading (256 threads cooperate) and only
    // skips its slot-loop body. With QTILE=17 and NUM_WARPS=8 every warp
    // gets at least one query — but be safe for future configs.
    unsigned int my_qs[QPER_WARP_MAX];
    unsigned int my_count = 0;
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
        unsigned int q = warp_id + (unsigned int)slot * NUM_WARPS;
        if (q < num_qtile) {
            my_qs[slot] = q;
            my_count++;
        } else {
            my_qs[slot] = 0;
        }
    }

    // ---- Load THIS WARP's Q-tile into per-lane registers ----
    float q_reg[QPER_WARP_MAX][VEC_BF16];
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
        if ((unsigned int)slot < my_count) {
            unsigned int q = my_qs[slot];
            const unsigned int* q32 = (const unsigned int*)(Q
                + (unsigned long long)q * q_stride
                + (unsigned long long)q_head * HDIM
                + vec_offset_bf16);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                unpack2_bf16(q32[i], q_reg[slot][2*i], q_reg[slot][2*i + 1]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) q_reg[slot][i] = 0.0f;
        }
    }

    // Per-query softmax + output accumulators (registers).
    float m[QPER_WARP_MAX], l[QPER_WARP_MAX];
    float o_reg[QPER_WARP_MAX][VEC_BF16];
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
        m[slot] = -1e30f;
        l[slot] = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) o_reg[slot][i] = 0.0f;
    }

    // Per-query causal cutoffs (warp-local).
    unsigned int my_sl[QPER_WARP_MAX];
    unsigned int warp_max_sl = 0;
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
        my_sl[slot] = ((unsigned int)slot < my_count) ? (unsigned int)seq_lens[my_qs[slot]] : 0;
        if (my_sl[slot] > warp_max_sl) warp_max_sl = my_sl[slot];
    }
    // CTA-wide max_sl: cp.async pipeline + __syncthreads need all warps to
    // execute the SAME number of tile iterations. Each warp owns DIFFERENT
    // queries with DIFFERENT seq_lens, so warp-local max_sl values differ.
    // Reduce via shared memory so every warp sees the same num_tiles.
    __shared__ unsigned int cta_max_sl_smem;
    if (tid == 0) cta_max_sl_smem = 0;
    __syncthreads();
    // Each warp lane 0 contributes its warp's max via atomicMax.
    if (lane_id == 0 && warp_max_sl > 0) {
        atomicMax(&cta_max_sl_smem, warp_max_sl);
    }
    __syncthreads();
    const unsigned int max_sl = cta_max_sl_smem;

    if (max_sl == 0) {
        // Every query has seq_len 0 — skip compute, jump to output zeroing.
        // CTA-wide consistent so no sync issue.
        goto write_output;
    }

    {
    // ---- FA2-style pipelined tile loop ----
    // num_tiles = ceil(max_sl / FA2_TILE_N). Issue stage-0 first, then for each
    // tile compute (stage curr) while prefetching (stage next). All warps
    // iterate the SAME num_tiles because max_sl is CTA-wide.
    const unsigned int num_tiles = (max_sl + FA2_TILE_N - 1) / FA2_TILE_N;

    // ---- Stage 0 prefetch ----
    {
        unsigned int t0_start = 0;
        unsigned int t0_end   = (t0_start + FA2_TILE_N < max_sl) ? (t0_start + FA2_TILE_N) : max_sl;
        fa2_issue_tile_loads(
            K_DATA_PTR(0), K_SCALE_PTR(0), V_DATA_PTR(0), V_SCALE_PTR(0),
            K_cache, V_cache, my_block_table,
            t0_start, t0_end, block_size, kv_head, num_kv_heads,
            block_stride_bytes, data_section_bytes, tid);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    for (unsigned int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        const unsigned int curr_stage = tile_idx % FA2_STAGES;
        const unsigned int next_tile  = tile_idx + 1;
        const unsigned int next_stage = next_tile % FA2_STAGES;

        // Prefetch next tile (if any) — overlaps with this tile's compute.
        if (next_tile < num_tiles) {
            unsigned int n_start = next_tile * FA2_TILE_N;
            unsigned int n_end   = (n_start + FA2_TILE_N < max_sl) ? (n_start + FA2_TILE_N) : max_sl;
            fa2_issue_tile_loads(
                K_DATA_PTR(next_stage), K_SCALE_PTR(next_stage),
                V_DATA_PTR(next_stage), V_SCALE_PTR(next_stage),
                K_cache, V_cache, my_block_table,
                n_start, n_end, block_size, kv_head, num_kv_heads,
                block_stride_bytes, data_section_bytes, tid);
            asm volatile("cp.async.commit_group;\n" ::);
            // Wait until only 1 group is in flight (i.e., current tile landed).
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            // Last tile: drain everything.
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        // ---- Compute on this tile from SMEM ----
        const unsigned int tile_start = tile_idx * FA2_TILE_N;
        const unsigned int tile_end_local =
            (tile_start + FA2_TILE_N < max_sl) ? FA2_TILE_N : (max_sl - tile_start);

        const unsigned char* k_data_smem  = K_DATA_PTR(curr_stage);
        const unsigned char* k_scale_smem = K_SCALE_PTR(curr_stage);
        const unsigned char* v_data_smem  = V_DATA_PTR(curr_stage);
        const unsigned char* v_scale_smem = V_SCALE_PTR(curr_stage);

        for (unsigned int local_pos = 0; local_pos < tile_end_local; local_pos++) {
            // Dequant this position's K and V slice for THIS lane.
            const unsigned char* kd =
                k_data_smem  + local_pos * (HDIM / 2) + smem_data_lane_off;
            const unsigned char* ks =
                k_scale_smem + local_pos * (HDIM / NVFP4_GROUP_SIZE) + smem_scale_lane_off;
            const unsigned char* vd =
                v_data_smem  + local_pos * (HDIM / 2) + smem_data_lane_off;
            const unsigned char* vs =
                v_scale_smem + local_pos * (HDIM / NVFP4_GROUP_SIZE) + smem_scale_lane_off;

            float k_vec[VEC_BF16], v_vec[VEC_BF16];
            nvfp4_dequant_local(kd, ks, e2m1_lut, k_vec);
            nvfp4_dequant_local(vd, vs, e2m1_lut, v_vec);

            const unsigned int abs_pos = tile_start + local_pos;

            #pragma unroll
            for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
                if ((unsigned int)slot >= my_count) break;
                if (abs_pos >= my_sl[slot]) continue;  // causal mask

                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) dot += q_reg[slot][i] * k_vec[i];
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                float score = dot * inv_sqrt_d;

                float m_new = fmaxf(m[slot], score);
                float exp_old = __expf(m[slot] - m_new);
                float exp_new = __expf(score   - m_new);
                l[slot] = l[slot] * exp_old + exp_new;
                m[slot] = m_new;

                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    o_reg[slot][i] = o_reg[slot][i] * exp_old + exp_new * v_vec[i];
                }
            }
        }

        // Barrier before stage rotates (next iteration may write into curr_stage SMEM).
        __syncthreads();
    }
    }

write_output:
    // ---- Write outputs (each warp writes its own queries) ----
    #pragma unroll
    for (int slot = 0; slot < QPER_WARP_MAX; slot++) {
        if ((unsigned int)slot >= my_count) break;
        unsigned int q = my_qs[slot];
        float inv_l = (l[slot] > 0.0f) ? (1.0f / l[slot]) : 0.0f;
        unsigned int* o32 = (unsigned int*)(O
            + (unsigned long long)q * num_q_heads * HDIM
            + (unsigned long long)q_head * HDIM
            + vec_offset_bf16);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0 = o_reg[slot][2*i]     * inv_l;
            float v1 = o_reg[slot][2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[i] = lo | (hi << 16);
        }
    }

    #undef K_DATA_PTR
    #undef K_SCALE_PTR
    #undef V_DATA_PTR
    #undef V_SCALE_PTR
}
