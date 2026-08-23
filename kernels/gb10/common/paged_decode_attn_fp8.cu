// SPDX-License-Identifier: AGPL-3.0-only

// Paged Decode Attention — FP8 E4M3 KV cache variant.
//
// Reads FP8 K/V from paged cache, dequantizes to F32 in registers, computes
// attention with BF16 query, writes BF16 output.
//
// Same algorithmic structure as paged_decode_attn.cu (BF16):
//   - One CTA per (q_head, seq) pair
//   - 8 warps split KV sequence
//   - Batched loading (BC=4) within physical blocks
//   - Online softmax with tree-based inter-warp reduction
//
// Key differences from BF16:
//   - K/V cache pointers are __nv_fp8_storage_t* (1 byte/elem vs 2)
//   - Each uint32 load yields 4 FP8 values (vs 2 BF16) → VEC_U32_FP8=2
//   - Dequant: f32_val = fp8_to_f32(byte) * scale
//   - Half the memory loads for same 8 elements per thread
//
// Grid: (num_q_heads, num_seqs, 1)  [splitk: (num_q_heads, num_splits, num_seqs)]
// Block: (256, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32
#ifndef HDIM
#define HDIM 256
#endif
#define VEC_BF16 (HDIM / WARP_SIZE)
#define VEC_U32  (HDIM / (WARP_SIZE * 2))
#define VEC_U32_FP8 (HDIM / (WARP_SIZE * 4))
#define NUM_WARPS 8
#define BC 4

// K-gamma FP8 specialization: Qwen3.8 verify currently uses M=15. Keep the
// bound explicit so an unsupported verify shape cannot silently index past
// the register tile.
#define KGAMMA_FP8_QTILE 15
#define KGAMMA_FP8_QPER_WARP ((KGAMMA_FP8_QTILE + NUM_WARPS - 1) / NUM_WARPS)

// Unpack 2 BF16 from uint32 → 2 F32 (reused for Q loading)
__device__ __forceinline__ void unpack2_bf16(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

// Unpack 4 FP8 E4M3 from uint32 → 4 F32 with dequant scale
__device__ __forceinline__ void unpack4_fp8(
    unsigned int packed, float dq_scale,
    float& v0, float& v1, float& v2, float& v3
) {
    __nv_fp8_storage_t b0 = (__nv_fp8_storage_t)(packed & 0xFF);
    __nv_fp8_storage_t b1 = (__nv_fp8_storage_t)((packed >> 8) & 0xFF);
    __nv_fp8_storage_t b2 = (__nv_fp8_storage_t)((packed >> 16) & 0xFF);
    __nv_fp8_storage_t b3 = (__nv_fp8_storage_t)((packed >> 24) & 0xFF);
    // FP8 → half_raw → float, then multiply by dequant scale
    v0 = __half2float(__nv_cvt_fp8_to_halfraw(b0, __NV_E4M3)) * dq_scale;
    v1 = __half2float(__nv_cvt_fp8_to_halfraw(b1, __NV_E4M3)) * dq_scale;
    v2 = __half2float(__nv_cvt_fp8_to_halfraw(b2, __NV_E4M3)) * dq_scale;
    v3 = __half2float(__nv_cvt_fp8_to_halfraw(b3, __NV_E4M3)) * dq_scale;
}

// Fused-query FP8 paged attention for the deterministic Qwen3.8 K-gamma
// verify shape. Each warp owns up to two queries and scans their shared KV
// history once. KV positions remain grouped in BC=4 batches, with the exact
// max-rescaled online-softmax update used by paged_decode_attn_fp8. Query dot
// products retain the same XOR warp reduction order.
//
// Contract: num_qtile == 15, identical block-table rows, no tree indirection.
extern "C" __global__ void paged_decode_attn_kgamma_fp8_bc4(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_fp8_storage_t* __restrict__ K_cache,
    const __nv_fp8_storage_t* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const float k_scale,
    const float v_scale,
    const unsigned int q_stride,
    const unsigned long long cache_stride,
    const unsigned int num_qtile
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;
    if (q_head >= num_q_heads || num_qtile != KGAMMA_FP8_QTILE || head_dim != HDIM) return;

    unsigned int my_q[KGAMMA_FP8_QPER_WARP];
    unsigned int my_count = 0;
    #pragma unroll
    for (int s = 0; s < KGAMMA_FP8_QPER_WARP; ++s) {
        unsigned int q = warp_id + s * NUM_WARPS;
        my_q[s] = q;
        if (q < num_qtile) ++my_count;
    }
    if (my_count == 0) return;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset = lane_id * VEC_BF16;
    const unsigned long long token_stride = (unsigned long long)num_kv_heads * HDIM;
    const int* block_table = block_tables; // all verify rows are identical

    float q_reg[KGAMMA_FP8_QPER_WARP][VEC_BF16];
    float out[KGAMMA_FP8_QPER_WARP][VEC_BF16];
    float m[KGAMMA_FP8_QPER_WARP], l[KGAMMA_FP8_QPER_WARP];
    unsigned int sl[KGAMMA_FP8_QPER_WARP], max_sl = 0;
    #pragma unroll
    for (int s = 0; s < KGAMMA_FP8_QPER_WARP; ++s) {
        m[s] = -1e30f; l[s] = 0.0f;
        sl[s] = s < (int)my_count ? (unsigned int)seq_lens[my_q[s]] : 0;
        max_sl = max(max_sl, sl[s]);
        #pragma unroll
        for (int i = 0; i < VEC_BF16; ++i) out[s][i] = 0.0f;
        if (s < (int)my_count) {
            const unsigned int* q32 = (const unsigned int*)(Q
                + (unsigned long long)my_q[s] * q_stride
                + (unsigned long long)q_head * HDIM + vec_offset);
            #pragma unroll
            for (int i = 0; i < VEC_U32; ++i)
                unpack2_bf16(q32[i], q_reg[s][2*i], q_reg[s][2*i+1]);
        }
    }

    unsigned int pos = 0;
    while (pos < max_sl) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)block_table[logical_block];
        unsigned int count = min(min(BC, block_size - block_offset), max_sl - pos);
        const __nv_fp8_storage_t* kb = K_cache + (unsigned long long)physical_block * cache_stride
            + (unsigned long long)block_offset * token_stride + (unsigned long long)kv_head * HDIM;
        const __nv_fp8_storage_t* vb = V_cache + (unsigned long long)physical_block * cache_stride
            + (unsigned long long)block_offset * token_stride + (unsigned long long)kv_head * HDIM;
        unsigned int kp[BC][VEC_U32_FP8], vp[BC][VEC_U32_FP8];
        #pragma unroll
        for (int b = 0; b < BC; ++b) {
            if (b < (int)count) {
                const unsigned int* k32 = (const unsigned int*)(kb + (unsigned long long)b * token_stride + vec_offset);
                const unsigned int* v32 = (const unsigned int*)(vb + (unsigned long long)b * token_stride + vec_offset);
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; ++i) { kp[b][i] = k32[i]; vp[b][i] = v32[i]; }
            }
        }

        #pragma unroll
        for (int s = 0; s < KGAMMA_FP8_QPER_WARP; ++s) {
            if (s >= (int)my_count) break;
            float scores[BC];
            #pragma unroll
            for (int b = 0; b < BC; ++b) {
                scores[b] = -1e30f;
                if (b < (int)count && pos + b < sl[s]) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VEC_U32_FP8; ++i) {
                        float a,b1,c,d; unpack4_fp8(kp[b][i], k_scale, a,b1,c,d);
                        dot += q_reg[s][4*i]*a + q_reg[s][4*i+1]*b1 + q_reg[s][4*i+2]*c + q_reg[s][4*i+3]*d;
                    }
                    #pragma unroll
                    for (int off = WARP_SIZE/2; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
                    scores[b] = dot * inv_sqrt_d;
                }
            }
            float mn = m[s];
            #pragma unroll
            for (int b = 0; b < BC; ++b) mn = fmaxf(mn, scores[b]);
            float eo = __expf(m[s] - mn); l[s] *= eo;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; ++i) out[s][i] *= eo;
            #pragma unroll
            for (int b = 0; b < BC; ++b) {
                if (scores[b] <= -1e29f) continue;
                float en = __expf(scores[b] - mn); l[s] += en;
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; ++i) {
                    float a,b1,c,d; unpack4_fp8(vp[b][i], v_scale, a,b1,c,d);
                    out[s][4*i] += en*a; out[s][4*i+1] += en*b1; out[s][4*i+2] += en*c; out[s][4*i+3] += en*d;
                }
            }
            m[s] = mn;
        }
        pos += count;
    }

    #pragma unroll
    for (int s = 0; s < KGAMMA_FP8_QPER_WARP; ++s) {
        if (s >= (int)my_count) break;
        float il = l[s] > 0.0f ? 1.0f/l[s] : 0.0f;
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)my_q[s] * num_q_heads * HDIM
            + (unsigned long long)q_head * HDIM + vec_offset);
        #pragma unroll
        for (int i = 0; i < VEC_U32; ++i) {
            unsigned int lo = __bfloat16_as_ushort(__float2bfloat16(out[s][2*i]*il));
            unsigned int hi = __bfloat16_as_ushort(__float2bfloat16(out[s][2*i+1]*il));
            o32[i] = lo | (hi << 16);
        }
    }
}

// Bit-parity candidate: two queries share one 16-warp CTA, but each query
// retains the legacy kernel's complete eight-warp reduction topology.
extern "C" __global__ void paged_decode_attn_kgamma_fp8_bc4_exact(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_fp8_storage_t* __restrict__ K_cache,
    const __nv_fp8_storage_t* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const float k_scale,
    const float v_scale,
    const unsigned int q_stride,
    const unsigned long long cache_stride,
    const unsigned int num_qtile
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int global_warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int group = global_warp / NUM_WARPS;
    const unsigned int warp_id = global_warp % NUM_WARPS;
    const unsigned int query = blockIdx.y * 2 + group;
    if (q_head >= num_q_heads || num_qtile != 15 || head_dim != HDIM) return;
    const bool active = query < num_qtile;
    const unsigned int seq_len = active ? (unsigned int)seq_lens[query] : 0u;
    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset = lane_id * VEC_BF16;
    const unsigned long long token_stride = (unsigned long long)num_kv_heads * HDIM;
    const int* table = block_tables + (unsigned long long)(active ? query : 0u) * max_blocks_per_seq;

    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)(active ? query : 0u) * q_stride
        + (unsigned long long)q_head * HDIM + vec_offset);
    float qr[VEC_BF16];
    #pragma unroll
    for (int i=0;i<VEC_U32;i++) unpack2_bf16(q32[i], qr[2*i], qr[2*i+1]);

    // Identical legacy partition for this query's local warp id.
    unsigned int chunk = (seq_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int begin = min(warp_id * chunk, seq_len);
    unsigned int end = min(begin + chunk, seq_len);
    float m = -1e30f, l = 0.0f, out[VEC_BF16];
    #pragma unroll
    for (int i=0;i<VEC_BF16;i++) out[i]=0.0f;

    unsigned int pos=begin;
    while(pos<end) {
        unsigned int logical=pos/block_size, off=pos%block_size;
        unsigned int count=min(block_size-off,end-pos);
        unsigned int physical=(unsigned int)table[logical];
        const __nv_fp8_storage_t* kb=K_cache+(unsigned long long)physical*cache_stride
            +(unsigned long long)off*token_stride+(unsigned long long)kv_head*HDIM;
        const __nv_fp8_storage_t* vb=V_cache+(unsigned long long)physical*cache_stride
            +(unsigned long long)off*token_stride+(unsigned long long)kv_head*HDIM;
        unsigned int done=0, aligned=(count/BC)*BC;
        for(;done<aligned;done+=BC) {
            unsigned int kp[BC][VEC_U32_FP8],vp[BC][VEC_U32_FP8];
            #pragma unroll
            for(int b=0;b<BC;b++) {
                const unsigned int* k=(const unsigned int*)(kb+(unsigned long long)(done+b)*token_stride+vec_offset);
                const unsigned int* v=(const unsigned int*)(vb+(unsigned long long)(done+b)*token_stride+vec_offset);
                #pragma unroll
                for(int i=0;i<VEC_U32_FP8;i++){kp[b][i]=k[i];vp[b][i]=v[i];}
            }
            float scores[BC];
            #pragma unroll
            for(int b=0;b<BC;b++) {
                float dot=0.0f;
                #pragma unroll
                for(int i=0;i<VEC_U32_FP8;i++) {
                    float a,b1,c,d; unpack4_fp8(kp[b][i],k_scale,a,b1,c,d);
                    dot+=qr[4*i]*a+qr[4*i+1]*b1+qr[4*i+2]*c+qr[4*i+3]*d;
                }
                #pragma unroll
                for(int x=WARP_SIZE/2;x>0;x>>=1)dot+=__shfl_xor_sync(0xffffffff,dot,x);
                scores[b]=dot*inv_sqrt_d;
            }
            float mn=m;
            #pragma unroll
            for(int b=0;b<BC;b++)mn=fmaxf(mn,scores[b]);
            float eo=__expf(m-mn);l*=eo;
            #pragma unroll
            for(int i=0;i<VEC_BF16;i++)out[i]*=eo;
            float ef[BC];
            #pragma unroll
            for(int b=0;b<BC;b++){ef[b]=__expf(scores[b]-mn);l+=ef[b];}
            m=mn;
            #pragma unroll
            for(int b=0;b<BC;b++){
                #pragma unroll
                for(int i=0;i<VEC_U32_FP8;i++){
                    float a,b1,c,d;unpack4_fp8(vp[b][i],v_scale,a,b1,c,d);
                    out[4*i]+=ef[b]*a;out[4*i+1]+=ef[b]*b1;out[4*i+2]+=ef[b]*c;out[4*i+3]+=ef[b]*d;
                }
            }
        }
        for(;done<count;done++) {
            const unsigned int* k=(const unsigned int*)(kb+(unsigned long long)done*token_stride+vec_offset);
            float dot=0.0f;
            #pragma unroll
            for(int i=0;i<VEC_U32_FP8;i++){
                float a,b1,c,d;unpack4_fp8(k[i],k_scale,a,b1,c,d);
                dot+=qr[4*i]*a+qr[4*i+1]*b1+qr[4*i+2]*c+qr[4*i+3]*d;
            }
            #pragma unroll
            for(int x=WARP_SIZE/2;x>0;x>>=1)dot+=__shfl_xor_sync(0xffffffff,dot,x);
            float score=dot*inv_sqrt_d,mn=fmaxf(m,score),eo=__expf(m-mn),en=__expf(score-mn);
            l=l*eo+en;
            const unsigned int* v=(const unsigned int*)(vb+(unsigned long long)done*token_stride+vec_offset);
            #pragma unroll
            for(int i=0;i<VEC_U32_FP8;i++){
                float a,b1,c,d;unpack4_fp8(v[i],v_scale,a,b1,c,d);
                out[4*i]=out[4*i]*eo+en*a;out[4*i+1]=out[4*i+1]*eo+en*b1;
                out[4*i+2]=out[4*i+2]*eo+en*c;out[4*i+3]=out[4*i+3]*eo+en*d;
            }
            m=mn;
        }
        pos+=count;
    }

    __shared__ float sm[2][NUM_WARPS],sl[2][NUM_WARPS],so[2][NUM_WARPS][HDIM];
    if(lane_id==0){sm[group][warp_id]=m;sl[group][warp_id]=l;}
    #pragma unroll
    for(int i=0;i<VEC_BF16;i++)so[group][warp_id][vec_offset+i]=out[i];
    __syncthreads();
    #pragma unroll
    for(int stride=NUM_WARPS/2;stride>0;stride>>=1){
        if(warp_id<(unsigned int)stride){
            unsigned int other=warp_id+stride;
            float lw=sl[group][other];
            if(lw>0.0f){
                float mw=sm[group][other],my_m=sm[group][warp_id],my_l=sl[group][warp_id];
                float mn=fmaxf(my_m,mw),sa=__expf(my_m-mn),sb=__expf(mw-mn);
                sl[group][warp_id]=my_l*sa+lw*sb;sm[group][warp_id]=mn;
                #pragma unroll
                for(int i=0;i<VEC_BF16;i++)so[group][warp_id][vec_offset+i]=
                    so[group][warp_id][vec_offset+i]*sa+so[group][other][vec_offset+i]*sb;
            }
        }
        __syncthreads();
    }
    if(active && warp_id==0){
        float final_l=sl[group][0];
        float il=final_l>0.0f?1.0f/final_l:0.0f;
        unsigned int* o=(unsigned int*)(O+(unsigned long long)query*num_q_heads*HDIM+(unsigned long long)q_head*HDIM+vec_offset);
        #pragma unroll
        for(int i=0;i<VEC_U32;i++){
            unsigned int lo=__bfloat16_as_ushort(__float2bfloat16(so[group][0][vec_offset+2*i]*il));
            unsigned int hi=__bfloat16_as_ushort(__float2bfloat16(so[group][0][vec_offset+2*i+1]*il));o[i]=lo|(hi<<16);
        }
    }
}

// ============================================================================
// Basic FP8 paged decode attention
// ============================================================================

extern "C" __global__ void paged_decode_attn_fp8(
    const __nv_bfloat16* __restrict__ Q,             // [num_seqs, num_q_heads, head_dim] BF16
    const __nv_fp8_storage_t* __restrict__ K_cache,  // [num_blocks, block_size, num_kv_heads, head_dim] FP8
    const __nv_fp8_storage_t* __restrict__ V_cache,  // [num_blocks, block_size, num_kv_heads, head_dim] FP8
    __nv_bfloat16* __restrict__ O,                   // [num_seqs, num_q_heads, head_dim] BF16
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const float k_scale,
    const float v_scale,
    const unsigned int q_stride,              // query.stride(0) in elements
    const unsigned long long cache_stride,    // k_cache.stride(0) in elements (block-level stride)
    // Optional tree-aware KV indirection (ATLAS_TREE_AWARE_ATTN). When
    // `kv_indirection == nullptr`, behavior is identical to the pre-tree
    // path. Otherwise positions in `[kv_indir_base..seq_lens[t])` are
    // remapped via `actual_pos = kv_indir_base + kv_indirection[seq_idx*kv_indir_stride + (pos - kv_indir_base)]`.
    //
    // CUDA graph fix: `kv_indir_base` is read from a 1×i32 device buffer
    // (`kv_indir_base_ptr`) instead of a kernel-launch immediate so the
    // captured graph reads the fresh value on each replay. Host writes the
    // current `seq.seq_len` to the buffer before launching. `nullptr` for
    // legacy / non-tree paths (kernel ignores `kv_indir_base` when
    // `kv_indirection == nullptr`).
    const int* __restrict__ kv_indirection,
    const int* __restrict__ kv_indir_base_ptr,
    const unsigned int kv_indir_stride,
    // ATLAS_TREE_KV_PACK fast path: pre-scattered ancestor KV pool. When
    // `K_pack_pool != nullptr`, positions in `[kv_indir_base..seq_lens[t])`
    // read from this contiguous scratch (one block per seq, BC=4 batched)
    // instead of walking the indirection table. Layout matches the regular
    // FP8 paged cache but with `block_size = pack_block_size`,
    // `num_blocks = num_seqs`, identity block table. Linear context
    // (`[0..kv_indir_base)`) still reads from `K_cache` via `block_tables`
    // — both contributions feed the same online-softmax accumulator.
    const __nv_fp8_storage_t* __restrict__ K_pack_pool,
    const __nv_fp8_storage_t* __restrict__ V_pack_pool,
    const unsigned int pack_block_size
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Load tree-window base position from device buffer (graph-safe).
    // `kv_indir_base_ptr` is nullptr for legacy non-tree callers; in that
    // case `kv_indir_base` is unused (gated by `kv_indirection == nullptr`).
    // `volatile` to defeat any aggressive caching of the load — the value
    // is updated each step via cuMemcpyHtoDAsync before launch.
    const unsigned int kv_indir_base = (kv_indir_base_ptr != nullptr)
        ? (unsigned int)(*((const volatile int*)kv_indir_base_ptr)) : 0u;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    // FP8: each lane covers 8 elements, but byte offset = lane_id * 8
    // (vs BF16 where byte offset = lane_id * 8 * 2)
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;  // BF16 element offset for Q/O
    const unsigned int vec_offset_fp8 = lane_id * VEC_BF16;   // FP8 element offset for K/V cache

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    const bool pack_active = (K_pack_pool != nullptr) && (V_pack_pool != nullptr);

    // Load Q (BF16, strided: Q may be a non-contiguous QKV split view)
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset_bf16);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_bf16(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }

    unsigned int chunk_size = (seq_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    // FP8 cache strides: cache_stride (block-level) passed from host to handle
    // non-contiguous layouts (e.g. vLLM BF16-allocated, uint8-viewed FP8 cache).
    // head_stride_kv (within-block) is always contiguous.
    unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;

    // Tree-aware indirection (see kernel docstring).
    const bool tree_active = (kv_indirection != nullptr);
    const int* my_indir = tree_active
        ? kv_indirection + (unsigned long long)seq_idx * kv_indir_stride
        : nullptr;

    unsigned int pos = my_start;
    while (pos < my_end) {
        // ── ATLAS_TREE_KV_PACK fast path ──
        // When `pack_active`, the tree-window slots have been pre-scattered
        // into a contiguous per-seq block in `K_pack_pool` / `V_pack_pool`
        // with `block_size = pack_block_size`, `num_blocks = num_seqs`, and
        // implicit identity block table. Read BC=4 from the pack pool —
        // no indirection lookups, no block_table walk per position.
        if (pack_active && pos >= kv_indir_base) {
            unsigned int local_off = pos - kv_indir_base;
            unsigned int remaining_total = my_end - pos;
            unsigned int batch_count = remaining_total;
            // Reads stay within the one pack block per seq (pack_block_size
            // slots). Already guaranteed by seq_lens[t] ≤ kv_indir_base + pack_block_size.
            const unsigned long long pack_block_stride =
                (unsigned long long)pack_block_size * head_stride_kv;
            const __nv_fp8_storage_t* k_pack_base = K_pack_pool
                + (unsigned long long)seq_idx * pack_block_stride
                + (unsigned long long)local_off * head_stride_kv
                + (unsigned long long)kv_head * head_dim;
            const __nv_fp8_storage_t* v_pack_base = V_pack_pool
                + (unsigned long long)seq_idx * pack_block_stride
                + (unsigned long long)local_off * head_stride_kv
                + (unsigned long long)kv_head * head_dim;

            unsigned int processed = 0;
            unsigned int aligned_count = (batch_count / BC) * BC;
            for (; processed < aligned_count; processed += BC) {
                unsigned int k_packed[BC][VEC_U32_FP8];
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    const unsigned int* k32 = (const unsigned int*)(k_pack_base
                        + (unsigned long long)(processed + b) * head_stride_kv + vec_offset_fp8);
                    #pragma unroll
                    for (int i = 0; i < VEC_U32_FP8; i++)
                        k_packed[b][i] = k32[i];
                }
                float scores[BC];
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VEC_U32_FP8; i++) {
                        float k0, k1, k2, k3;
                        unpack4_fp8(k_packed[b][i], k_scale, k0, k1, k2, k3);
                        dot += q_reg[4*i]   * k0 + q_reg[4*i+1] * k1
                             + q_reg[4*i+2] * k2 + q_reg[4*i+3] * k3;
                    }
                    #pragma unroll
                    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                        dot += __shfl_xor_sync(0xffffffff, dot, offset);
                    scores[b] = dot * inv_sqrt_d;
                }
                unsigned int v_packed[BC][VEC_U32_FP8];
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    const unsigned int* v32 = (const unsigned int*)(v_pack_base
                        + (unsigned long long)(processed + b) * head_stride_kv + vec_offset_fp8);
                    #pragma unroll
                    for (int i = 0; i < VEC_U32_FP8; i++)
                        v_packed[b][i] = v32[i];
                }
                float m_new = m;
                #pragma unroll
                for (int b = 0; b < BC; b++) m_new = fmaxf(m_new, scores[b]);
                float exp_old = __expf(m - m_new);
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) o_reg[i] *= exp_old;
                l *= exp_old;
                float exp_factors[BC];
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    exp_factors[b] = __expf(scores[b] - m_new);
                    l += exp_factors[b];
                }
                m = m_new;
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    float ef = exp_factors[b];
                    #pragma unroll
                    for (int i = 0; i < VEC_U32_FP8; i++) {
                        float v0, v1, v2, v3;
                        unpack4_fp8(v_packed[b][i], v_scale, v0, v1, v2, v3);
                        o_reg[4*i]   += ef * v0;
                        o_reg[4*i+1] += ef * v1;
                        o_reg[4*i+2] += ef * v2;
                        o_reg[4*i+3] += ef * v3;
                    }
                }
            }
            // Remainder: single-position from pack pool (still no indirection).
            for (; processed < batch_count; processed++) {
                const unsigned int* k32 = (const unsigned int*)(k_pack_base
                    + (unsigned long long)processed * head_stride_kv + vec_offset_fp8);
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++) {
                    float k0, k1, k2, k3;
                    unpack4_fp8(k32[i], k_scale, k0, k1, k2, k3);
                    dot += q_reg[4*i] * k0 + q_reg[4*i+1] * k1
                         + q_reg[4*i+2] * k2 + q_reg[4*i+3] * k3;
                }
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                float score = dot * inv_sqrt_d;
                float m_new = fmaxf(m, score);
                float exp_old = __expf(m - m_new);
                float exp_new = __expf(score - m_new);
                l = l * exp_old + exp_new;
                const unsigned int* v32 = (const unsigned int*)(v_pack_base
                    + (unsigned long long)processed * head_stride_kv + vec_offset_fp8);
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++) {
                    float v0, v1, v2, v3;
                    unpack4_fp8(v32[i], v_scale, v0, v1, v2, v3);
                    o_reg[4*i]   = o_reg[4*i]   * exp_old + exp_new * v0;
                    o_reg[4*i+1] = o_reg[4*i+1] * exp_old + exp_new * v1;
                    o_reg[4*i+2] = o_reg[4*i+2] * exp_old + exp_new * v2;
                    o_reg[4*i+3] = o_reg[4*i+3] * exp_old + exp_new * v3;
                }
                m = m_new;
            }
            pos += batch_count;
            continue;
        }

        // Tree-window single-position fallback when active and past the base
        // (only reached when pack_pool is NULL — i.e. ATLAS_TREE_AWARE_ATTN
        // without ATLAS_TREE_KV_PACK).
        if (tree_active && pos >= kv_indir_base) {
            unsigned int local_off = pos - kv_indir_base;
            int compact_idx = my_indir[local_off];
            unsigned int actual_pos = kv_indir_base + (unsigned int)compact_idx;
            unsigned int logical_block_t = actual_pos / block_size;
            unsigned int block_offset_t = actual_pos % block_size;
            unsigned int physical_block_t = (unsigned int)my_block_table[logical_block_t];
            const __nv_fp8_storage_t* k_base_t = K_cache
                + (unsigned long long)physical_block_t * cache_stride
                + (unsigned long long)block_offset_t * head_stride_kv
                + (unsigned long long)kv_head * head_dim;
            const __nv_fp8_storage_t* v_base_t = V_cache
                + (unsigned long long)physical_block_t * cache_stride
                + (unsigned long long)block_offset_t * head_stride_kv
                + (unsigned long long)kv_head * head_dim;

            const unsigned int* k32 = (const unsigned int*)(k_base_t + vec_offset_fp8);
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_U32_FP8; i++) {
                float k0, k1, k2, k3;
                unpack4_fp8(k32[i], k_scale, k0, k1, k2, k3);
                dot += q_reg[4*i] * k0 + q_reg[4*i+1] * k1
                     + q_reg[4*i+2] * k2 + q_reg[4*i+3] * k3;
            }
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);

            float score = dot * inv_sqrt_d;
            float m_new = fmaxf(m, score);
            float exp_old = __expf(m - m_new);
            float exp_new = __expf(score - m_new);
            l = l * exp_old + exp_new;

            const unsigned int* v32 = (const unsigned int*)(v_base_t + vec_offset_fp8);
            #pragma unroll
            for (int i = 0; i < VEC_U32_FP8; i++) {
                float v0, v1, v2, v3;
                unpack4_fp8(v32[i], v_scale, v0, v1, v2, v3);
                o_reg[4*i]   = o_reg[4*i]   * exp_old + exp_new * v0;
                o_reg[4*i+1] = o_reg[4*i+1] * exp_old + exp_new * v1;
                o_reg[4*i+2] = o_reg[4*i+2] * exp_old + exp_new * v2;
                o_reg[4*i+3] = o_reg[4*i+3] * exp_old + exp_new * v3;
            }
            m = m_new;
            pos += 1;
            continue;
        }

        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = remaining_in_block < remaining_total ? remaining_in_block : remaining_total;

        // Don't bleed past kv_indir_base into the indirected window
        // (slow fallback) or the packed window (pack_pool fast path).
        if ((tree_active || pack_active) && pos < kv_indir_base) {
            unsigned int remaining_to_base = kv_indir_base - pos;
            if (batch_count > remaining_to_base) batch_count = remaining_to_base;
        }

        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        const __nv_fp8_storage_t* k_block_base = K_cache + (unsigned long long)physical_block * cache_stride
                                                          + (unsigned long long)block_offset * head_stride_kv
                                                          + (unsigned long long)kv_head * head_dim;
        const __nv_fp8_storage_t* v_block_base = V_cache + (unsigned long long)physical_block * cache_stride
                                                          + (unsigned long long)block_offset * head_stride_kv
                                                          + (unsigned long long)kv_head * head_dim;

        unsigned int processed = 0;
        unsigned int aligned_count = (batch_count / BC) * BC;

        // Batched path: BC=4 positions at a time
        for (; processed < aligned_count; processed += BC) {
            // Load BC K vectors (FP8: 2 uint32 per thread = 8 FP8 elements)
            unsigned int k_packed[BC][VEC_U32_FP8];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* k32 = (const unsigned int*)(k_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset_fp8);
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++)
                    k_packed[b][i] = k32[i];
            }

            // Compute BC dot products (FP8 dequant + dot)
            float scores[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++) {
                    float k0, k1, k2, k3;
                    unpack4_fp8(k_packed[b][i], k_scale, k0, k1, k2, k3);
                    dot += q_reg[4*i]   * k0 + q_reg[4*i+1] * k1
                         + q_reg[4*i+2] * k2 + q_reg[4*i+3] * k3;
                }
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                scores[b] = dot * inv_sqrt_d;
            }

            // Load BC V vectors
            unsigned int v_packed[BC][VEC_U32_FP8];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* v32 = (const unsigned int*)(v_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset_fp8);
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++)
                    v_packed[b][i] = v32[i];
            }

            // Batched softmax update
            float m_new = m;
            #pragma unroll
            for (int b = 0; b < BC; b++)
                m_new = fmaxf(m_new, scores[b]);

            float exp_old = __expf(m - m_new);
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                o_reg[i] *= exp_old;
            l *= exp_old;

            float exp_factors[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                exp_factors[b] = __expf(scores[b] - m_new);
                l += exp_factors[b];
            }
            m = m_new;

            // V accumulate (FP8 dequant)
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float ef = exp_factors[b];
                #pragma unroll
                for (int i = 0; i < VEC_U32_FP8; i++) {
                    float v0, v1, v2, v3;
                    unpack4_fp8(v_packed[b][i], v_scale, v0, v1, v2, v3);
                    o_reg[4*i]   += ef * v0;
                    o_reg[4*i+1] += ef * v1;
                    o_reg[4*i+2] += ef * v2;
                    o_reg[4*i+3] += ef * v3;
                }
            }
        }

        // Remainder: single positions
        for (; processed < batch_count; processed++) {
            const unsigned int* k32 = (const unsigned int*)(k_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset_fp8);
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_U32_FP8; i++) {
                float k0, k1, k2, k3;
                unpack4_fp8(k32[i], k_scale, k0, k1, k2, k3);
                dot += q_reg[4*i] * k0 + q_reg[4*i+1] * k1
                     + q_reg[4*i+2] * k2 + q_reg[4*i+3] * k3;
            }
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);

            float score = dot * inv_sqrt_d;
            float m_new = fmaxf(m, score);
            float exp_old = __expf(m - m_new);
            float exp_new = __expf(score - m_new);
            l = l * exp_old + exp_new;

            const unsigned int* v32 = (const unsigned int*)(v_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset_fp8);
            #pragma unroll
            for (int i = 0; i < VEC_U32_FP8; i++) {
                float v0, v1, v2, v3;
                unpack4_fp8(v32[i], v_scale, v0, v1, v2, v3);
                o_reg[4*i]   = o_reg[4*i]   * exp_old + exp_new * v0;
                o_reg[4*i+1] = o_reg[4*i+1] * exp_old + exp_new * v1;
                o_reg[4*i+2] = o_reg[4*i+2] * exp_old + exp_new * v2;
                o_reg[4*i+3] = o_reg[4*i+3] * exp_old + exp_new * v3;
            }
            m = m_new;
        }

        pos += batch_count;
    }

    // Tree-based inter-warp reduction (identical to BF16 version)
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        smem_o[warp_id][vec_offset_bf16 + i] = o_reg[i];
    }
    __syncthreads();

    #pragma unroll
    for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.0f) {
                float mw = smem_m[other];
                float my_m = smem_m[warp_id];
                float my_l = smem_l[warp_id];
                float m_new = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - m_new);
                float scale_w = __expf(mw - m_new);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    smem_o[warp_id][vec_offset_bf16 + i] =
                        smem_o[warp_id][vec_offset_bf16 + i] * scale_me +
                        smem_o[other][vec_offset_bf16 + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        float final_l = smem_l[0];
        float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                              + (unsigned long long)q_head * head_dim + vec_offset_bf16);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0 = smem_o[0][vec_offset_bf16 + 2*i]     * inv_l;
            float v1 = smem_o[0][vec_offset_bf16 + 2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}

// ============================================================================
// Split-K FP8 variant for long sequences with few heads
// Grid: (num_q_heads, num_splits, num_seqs)
// ============================================================================

extern "C" __global__ void paged_decode_attn_splitk_fp8(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_fp8_storage_t* __restrict__ K_cache,
    const __nv_fp8_storage_t* __restrict__ V_cache,
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
    const float k_scale,
    const float v_scale,
    const unsigned int q_stride,              // query.stride(0) in elements
    const unsigned long long cache_stride,    // k_cache.stride(0) in elements (block-level stride)
    // See `paged_decode_attn_fp8` for the indirection contract.
    // `kv_indir_base_ptr` is a 1×i32 device buffer (graph-safe replacement
    // for the prior scalar). Pass `nullptr` for legacy / non-tree paths.
    const int* __restrict__ kv_indirection,
    const int* __restrict__ kv_indir_base_ptr,
    const unsigned int kv_indir_stride
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int split_id = blockIdx.y;
    const unsigned int seq_idx = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Load tree-window base position from device buffer (graph-safe, volatile).
    const unsigned int kv_indir_base = (kv_indir_base_ptr != nullptr)
        ? (unsigned int)(*((const volatile int*)kv_indir_base_ptr)) : 0u;

    unsigned int split_size = (seq_len + num_splits - 1) / num_splits;
    unsigned int kv_start = split_id * split_size;
    unsigned int kv_end = kv_start + split_size;
    if (kv_end > seq_len) kv_end = seq_len;
    if (kv_start >= seq_len) kv_start = kv_end;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;
    const unsigned int vec_offset_fp8 = lane_id * VEC_BF16;

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q (BF16, strided: Q may be a non-contiguous QKV split view)
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset_bf16);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_bf16(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }

    unsigned int local_len = kv_end - kv_start;
    unsigned int chunk_size = (local_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = kv_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > kv_end) my_end = kv_end;
    if (my_start > kv_end) my_start = kv_end;

    float m_val = -1e30f;
    float l_val = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    // cache_stride (block-level) passed from host; head_stride_kv (within-block) always contiguous.
    unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;

    // Tree-aware indirection (see paged_decode_attn_fp8 contract).
    const bool tree_active = (kv_indirection != nullptr);
    const int* my_indir = tree_active
        ? kv_indirection + (unsigned long long)seq_idx * kv_indir_stride
        : nullptr;

    for (unsigned int pos = my_start; pos < my_end; pos++) {
        unsigned int actual_pos = pos;
        if (tree_active && pos >= kv_indir_base) {
            unsigned int local_off = pos - kv_indir_base;
            int compact_idx = my_indir[local_off];
            actual_pos = kv_indir_base + (unsigned int)compact_idx;
        }
        unsigned int logical_block = actual_pos / block_size;
        unsigned int block_offset = actual_pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];

        const unsigned int* k32 = (const unsigned int*)(K_cache
            + (unsigned long long)physical_block * cache_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim + vec_offset_fp8);

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_U32_FP8; i++) {
            float k0, k1, k2, k3;
            unpack4_fp8(k32[i], k_scale, k0, k1, k2, k3);
            dot += q_reg[4*i] * k0 + q_reg[4*i+1] * k1
                 + q_reg[4*i+2] * k2 + q_reg[4*i+3] * k3;
        }
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, offset);

        float score = dot * inv_sqrt_d;
        float m_new = fmaxf(m_val, score);
        float exp_old = __expf(m_val - m_new);
        float exp_new = __expf(score - m_new);
        l_val = l_val * exp_old + exp_new;

        const unsigned int* v32 = (const unsigned int*)(V_cache
            + (unsigned long long)physical_block * cache_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim + vec_offset_fp8);

        #pragma unroll
        for (int i = 0; i < VEC_U32_FP8; i++) {
            float v0, v1, v2, v3;
            unpack4_fp8(v32[i], v_scale, v0, v1, v2, v3);
            o_reg[4*i]   = o_reg[4*i]   * exp_old + exp_new * v0;
            o_reg[4*i+1] = o_reg[4*i+1] * exp_old + exp_new * v1;
            o_reg[4*i+2] = o_reg[4*i+2] * exp_old + exp_new * v2;
            o_reg[4*i+3] = o_reg[4*i+3] * exp_old + exp_new * v3;
        }
        m_val = m_new;
    }

    // Tree merge within CTA
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m_val;
        smem_l[warp_id] = l_val;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        smem_o[warp_id][vec_offset_bf16 + i] = o_reg[i];
    }
    __syncthreads();

    #pragma unroll
    for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.0f) {
                float mw = smem_m[other];
                float my_m = smem_m[warp_id];
                float my_l = smem_l[warp_id];
                float m_new = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - m_new);
                float scale_w = __expf(mw - m_new);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    smem_o[warp_id][vec_offset_bf16 + i] =
                        smem_o[warp_id][vec_offset_bf16 + i] * scale_me +
                        smem_o[other][vec_offset_bf16 + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    // Write partial to workspace (F32)
    unsigned int ws_stride = (head_dim + 2);
    float* ws_base = workspace + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride
                   + split_id * ws_stride;

    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) {
            ws_base[vec_offset_bf16 + i] = smem_o[0][vec_offset_bf16 + i];
        }
        if (lane_id == 0) {
            ws_base[head_dim] = smem_m[0];
            ws_base[head_dim + 1] = smem_l[0];
        }
    }
}

// ============================================================================
// Reduce split-K partials into final BF16 output (FP8 variant).
//
// Identical to NVFP4 reduce — the workspace format is quantization-agnostic:
//   workspace[seq_idx, q_head, split, :] = [head_dim F32 values, m, l]
//
// Grid: (num_q_heads, num_seqs, 1)  Block: (32, 1, 1)
// ============================================================================

extern "C" __global__ void paged_decode_attn_reduce_fp8(
    const float* __restrict__ workspace,    // [num_seqs, num_q_heads, num_splits, (head_dim+2)] F32
    __nv_bfloat16* __restrict__ O,          // [num_seqs, num_q_heads, head_dim] BF16
    const int* __restrict__ seq_lens,
    const unsigned int num_q_heads,
    const unsigned int head_dim,
    const unsigned int num_splits
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int lane_id = threadIdx.x;  // 0..31

    if (q_head >= num_q_heads) return;
    if (seq_lens[seq_idx] == 0) return;

    const unsigned int vec_off = lane_id * VEC_BF16;
    const unsigned int ws_stride = head_dim + 2;
    const float* ws_base = workspace
        + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride;

    // Load split 0
    float m = ws_base[head_dim];
    float l = ws_base[head_dim + 1];
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++)
        o_reg[i] = ws_base[vec_off + i];

    // Merge splits 1..num_splits-1
    for (unsigned int s = 1; s < num_splits; s++) {
        const float* ws = ws_base + s * ws_stride;
        float ms = ws[head_dim];
        float ls = ws[head_dim + 1];

        if (ls <= 0.0f) continue;

        float m_new = fmaxf(m, ms);
        float scale_me = __expf(m - m_new);
        float scale_s = __expf(ms - m_new);

        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++)
            o_reg[i] = o_reg[i] * scale_me + ws[vec_off + i] * scale_s;

        l = l * scale_me + ls * scale_s;
        m = m_new;
    }

    // Normalize and write BF16 output
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                          + (unsigned long long)q_head * head_dim + vec_off);
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        float v0 = o_reg[2*i] * inv_l;
        float v1 = o_reg[2*i + 1] * inv_l;
        unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
        unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
        o32[i] = lo | (hi << 16);
    }
}
