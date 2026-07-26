// SPDX-License-Identifier: AGPL-3.0-only

// Fused small-op swarm for the DFlash K=γ flat verify AND the serial (M=1)
// decode path (ATLAS_FUSED_ELEMWISE=1).
//
// Kernel 1 — fused_qkv_norm_rope_cache_write_bf16:
//   Replaces, per layer, the post-QKV-GEMM elementwise chain of the
//   multi-seq verify path (crates/spark-model/.../multi_seq/):
//     - 3n scatter copy_d2d (q/k/v scratch → per-seq qkv_buf)
//     - 2n rms_norm launches (per-head q_norm / k_norm)
//     - n  rope_forward_yarn_scaled launches
//     - n  reshape_and_cache_flash launches
//     - n  gather copy_d2d (qkv_buf q → contiguous Q)
//   = 8n launches (64 at n=8) with ONE kernel. All operands stay in the
//   CONTIGUOUS GEMM output layout ([n, nq*hd] / [n, nkv*hd]); Q is
//   normed+roped in place (the contiguous buffer IS the paged-decode Q
//   input), K is normed+roped+written directly into the paged BF16 cache,
//   V is copied into the cache verbatim.
//
//   SERIAL (M=1) decode reuses this kernel unchanged at n=1 (grid
//   (nq+2*nkv, 1)): the decode GEMVs write q/k/v contiguously into
//   qkv_output, which at a single row IS the [n, nq*hd] / [n, nkv*hd]
//   layout above — replacing the 4-launch chain (q rms_norm → k rms_norm →
//   rope_forward_yarn_scaled → reshape_and_cache_flash; no scatter/gather
//   exists at M=1). See decode/fused_epilogue.rs.
//
//   BIT-EXACTNESS CONTRACT: every stage reproduces the exact FP32
//   expression order of the kernel it replaces, and rounds to BF16 at
//   every point the unfused chain went through memory:
//     norm     — rms_norm / rms_norm_vanilla (block-per-row, blockDim =
//                head_dim, identical vectorized sum + warp/blk reduction);
//                result rounded to BF16 (the unfused chain stored it).
//     rope     — rope_forward_yarn_scaled (reads the BF16 normed value,
//                angle = (float)pos * inv_freq[pair], cos/sin * factor,
//                x0*c - x1*s / x1*c + x0*s, rounded to BF16).
//     cache    — reshape_and_cache_flash (BF16 bit copy of the roped K
//                and the raw V into the paged pools; slot<0 rows skip).
//   This file inherits the directory's `--fmad=false` (KERNEL.toml), the
//   same flag the replaced kernels are compiled with.
//
// Kernel 2 — moe_weighted_sum_blend_residual_batchn:
//   moe_weighted_sum_blend_batch2 (general-N) + bf16_residual_add fused:
//   the blend result is rounded to BF16 exactly as before (and still
//   written to `output`), then re-expanded and added to the BF16 residual
//   stream exactly as bf16_residual_add does. Removes one launch and one
//   [n, hidden] BF16 read+write round-trip per layer.
//
// Graph-capture safe: pure device pointers + scalars, no host reads.

#include <cuda_bf16.h>

#define FVE_MAX_HEAD_DIM 256
#define FVE_WARP_SIZE 32

__device__ __forceinline__ void fve_unpack_bf16x2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ float fve_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// One block per (row, head). Grid: (num_q_heads + 2*num_kv_heads, n, 1).
// Block: (head_dim, 1, 1) — matches the blockDim=min(head_dim,1024) the
// unfused rms_norm launch used per head-row, so the reduction order (and
// therefore the FP32 sum) is IDENTICAL. head_dim must be even, <= 256.
//
// blockIdx.x < nq                : Q head  (norm + rope, in place in `q`)
// blockIdx.x in [nq, nq+nkv)     : K head  (norm + rope + paged cache write)
// blockIdx.x >= nq+nkv           : V head  (verbatim paged cache write)
//
// norm_offset_one: 0 → vanilla `x * rms * w` (rms_norm_vanilla — Laguna),
//                  1 → offset  `x * rms * (1 + w)` (rms_norm).
extern "C" __global__ void fused_qkv_norm_rope_cache_write_bf16(
    __nv_bfloat16* __restrict__ q,               // [n, nq*hd] in/out (in place)
    const __nv_bfloat16* __restrict__ k,         // [n, nkv*hd] (GEMM output)
    const __nv_bfloat16* __restrict__ v,         // [n, nkv*hd] (GEMM output)
    const __nv_bfloat16* __restrict__ q_norm_w,  // [hd]
    const __nv_bfloat16* __restrict__ k_norm_w,  // [hd]
    const unsigned int* __restrict__ positions,  // [n] u32
    const float* __restrict__ inv_freq,          // [rotary_dim/2] f32 table
    __nv_bfloat16* __restrict__ k_cache,         // paged [blocks, bs, nkv, hd]
    __nv_bfloat16* __restrict__ v_cache,         // paged [blocks, bs, nkv, hd]
    const long long* __restrict__ slot_mapping,  // [n] i64 (-1 = skip write)
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const unsigned int block_size,
    const float eps,
    const float attention_factor,
    const unsigned int norm_offset_one
) {
    const unsigned int role = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (tid >= head_dim) return;

    const long long slot = slot_mapping[row];

    // ── V role: verbatim BF16 copy into the paged cache (reshape_and_cache_flash) ──
    if (role >= num_q_heads + num_kv_heads) {
        if (slot < 0) return;
        const unsigned int kv_head = role - num_q_heads - num_kv_heads;
        const __nv_bfloat16* src =
            v + (unsigned long long)row * num_kv_heads * head_dim
              + (unsigned long long)kv_head * head_dim;
        const unsigned int n_elems = num_kv_heads * head_dim;
        const unsigned long long cache_stride = (unsigned long long)block_size * n_elems;
        const unsigned int block_idx = (unsigned int)(slot / (long long)block_size);
        const unsigned int block_offset = (unsigned int)(slot % (long long)block_size);
        __nv_bfloat16* dst = v_cache
            + (unsigned long long)block_idx * cache_stride
            + (unsigned long long)block_offset * n_elems
            + (unsigned long long)kv_head * head_dim;
        dst[tid] = src[tid];   // pure bit copy — identical to the unfused write
        return;
    }

    // ── Q / K role: per-head rms_norm → BF16 round → yarn-scaled rope ──
    const bool is_q = role < num_q_heads;
    const unsigned int head = is_q ? role : role - num_q_heads;
    const unsigned int nheads = is_q ? num_q_heads : num_kv_heads;
    const __nv_bfloat16* x =
        (is_q ? (const __nv_bfloat16*)q : k)
        + (unsigned long long)row * nheads * head_dim
        + (unsigned long long)head * head_dim;
    const __nv_bfloat16* w = is_q ? q_norm_w : k_norm_w;

    // Phase 1: sum of squares — verbatim port of rms_norm[_vanilla] with
    // hidden_size = head_dim and blockDim.x = head_dim (same loop trips,
    // same per-thread accumulation, same warp/block reduction shape).
    float sum_sq = 0.0f;
    const unsigned int half_size = head_dim / 2;
    const unsigned int* x32 = (const unsigned int*)x;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        fve_unpack_bf16x2(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }

    sum_sq = fve_warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = fve_warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    const float rms = rsqrtf(warp_sums[0] / (float)head_dim + eps);

    // Phase 2: apply weight, round to BF16 (the unfused chain stored the
    // normed head to memory here — reproduce that rounding), stage in smem.
    __shared__ __nv_bfloat16 normed_bf[FVE_MAX_HEAD_DIM];
    const unsigned int* w32 = (const unsigned int*)w;
    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float xv0, xv1, wv0, wv1;
        fve_unpack_bf16x2(x32[i], xv0, xv1);
        fve_unpack_bf16x2(w32[i], wv0, wv1);
        const float weff0 = norm_offset_one ? (1.0f + wv0) : wv0;
        const float weff1 = norm_offset_one ? (1.0f + wv1) : wv1;
        normed_bf[i * 2]     = __float2bfloat16(xv0 * rms * weff0);
        normed_bf[i * 2 + 1] = __float2bfloat16(xv1 * rms * weff1);
    }
    __syncthreads();

    // Phase 3: rope_forward_yarn_scaled on the BF16 normed values.
    // Pair (t, t + rotary_dim/2) for t < rotary_dim/2; passthrough beyond.
    const unsigned int pairs = rotary_dim / 2;
    __nv_bfloat16 out_val;
    bool have_out = false;
    __nv_bfloat16 out_val_hi;              // partner element t + pairs
    if (tid < pairs) {
        const float x0 = __bfloat162float(normed_bf[tid]);
        const float x1 = __bfloat162float(normed_bf[tid + pairs]);
        const float angle = (float)positions[row] * inv_freq[tid];
        const float cos_val = cosf(angle) * attention_factor;
        const float sin_val = sinf(angle) * attention_factor;
        out_val = __float2bfloat16(x0 * cos_val - x1 * sin_val);
        out_val_hi = __float2bfloat16(x1 * cos_val + x0 * sin_val);
        have_out = true;
    } else if (tid >= rotary_dim) {
        out_val = normed_bf[tid];          // passthrough channel
        have_out = true;
    }
    // threads in [pairs, rotary_dim) write nothing: their element is the
    // partner (d1) of thread tid - pairs.

    // Phase 4: write destination — Q in place, K into the paged cache.
    if (is_q) {
        __nv_bfloat16* dst = q
            + (unsigned long long)row * num_q_heads * head_dim
            + (unsigned long long)head * head_dim;
        if (have_out) dst[tid] = out_val;
        if (tid < pairs) dst[tid + pairs] = out_val_hi;
    } else {
        if (slot < 0) return;
        const unsigned int n_elems = num_kv_heads * head_dim;
        const unsigned long long cache_stride = (unsigned long long)block_size * n_elems;
        const unsigned int block_idx = (unsigned int)(slot / (long long)block_size);
        const unsigned int block_offset = (unsigned int)(slot % (long long)block_size);
        __nv_bfloat16* dst = k_cache
            + (unsigned long long)block_idx * cache_stride
            + (unsigned long long)block_offset * n_elems
            + (unsigned long long)head * head_dim;
        if (have_out) dst[tid] = out_val;
        if (tid < pairs) dst[tid + pairs] = out_val_hi;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2 — MoE weighted-sum blend + residual add (KN verify tail).
//
// Byte-identical `output` to moe_weighted_sum_blend_batch2 (general-N),
// plus the bf16_residual_add fold: hidden[j] = bf16(f32(hidden[j]) +
// f32(output[j])) — with output[j] being the freshly rounded BF16 blend,
// exactly what the separate residual kernel would have re-read.
//
// Grid: (ceil(hidden/256), num_tokens, 1)  Block: (256, 1, 1)
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void moe_weighted_sum_blend_residual_batchn(
    __nv_bfloat16* __restrict__ output,              // [n, hidden] BF16
    const __nv_bfloat16* __restrict__ expert_out,    // [n*top_k, hidden] BF16
    const float* __restrict__ expert_weights,        // [n*top_k] f32
    const __nv_bfloat16* __restrict__ shared_out,    // [n, hidden] BF16
    const __nv_bfloat16* __restrict__ input,         // [n, K] BF16 (MoE input)
    const __nv_bfloat16* __restrict__ gate_weight,   // [1, K] BF16 or NULL
    __nv_bfloat16* __restrict__ hidden_resid,        // [n, hidden] BF16 in/out
    unsigned int hidden,
    unsigned int top_k,
    unsigned int K
) {
    const unsigned int token = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / FVE_WARP_SIZE;
    const unsigned int lane = tid % FVE_WARP_SIZE;

    const __nv_bfloat16* my_input = input + (unsigned long long)token * K;
    const float* my_weights = expert_weights + token * top_k;
    const __nv_bfloat16* my_expert_out = expert_out + (unsigned long long)token * top_k * hidden;
    const __nv_bfloat16* my_shared_out = shared_out + (unsigned long long)token * hidden;
    __nv_bfloat16* my_output = output + (unsigned long long)token * hidden;
    __nv_bfloat16* my_hidden = hidden_resid + (unsigned long long)token * hidden;

    // ── Phase 1: shared-expert gate scalar (verbatim from the blend kernel;
    // NULL gate_weight → 1.0, the Laguna/Mistral ungated shared-expert case) ──
    __shared__ float s_warp_sums[8];
    __shared__ float sigmoid_val;

    if (gate_weight == 0) {
        if (tid == 0) sigmoid_val = 1.0f;
        __syncthreads();
    } else {

    float dot_acc = 0.0f;
    unsigned int K8 = K / 8;
    for (unsigned int k8 = tid; k8 < K8; k8 += 256) {
        uint4 a_data = ((const uint4*)my_input)[k8];
        uint4 w_data = (gate_weight != nullptr) ? ((const uint4*)gate_weight)[k8]
                                                : make_uint4(0u, 0u, 0u, 0u);
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int w_raw[4] = {w_data.x, w_data.y, w_data.z, w_data.w};

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            __nv_bfloat16 a_lo, a_hi, w_lo, w_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            *(unsigned short*)&w_lo = (unsigned short)(w_raw[b] & 0xFFFF);
            *(unsigned short*)&w_hi = (unsigned short)(w_raw[b] >> 16);
            dot_acc += __bfloat162float(a_lo) * __bfloat162float(w_lo);
            dot_acc += __bfloat162float(a_hi) * __bfloat162float(w_hi);
        }
    }

    #pragma unroll
    for (int offset = FVE_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        dot_acc += __shfl_down_sync(0xFFFFFFFF, dot_acc, offset);
    }
    if (lane == 0) {
        s_warp_sums[warp_id] = dot_acc;
    }
    __syncthreads();

    if (tid == 0) {
        float gate_scalar = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            gate_scalar += s_warp_sums[w];
        }
        sigmoid_val = (gate_weight != nullptr) ? (1.0f / (1.0f + __expf(-gate_scalar))) : 1.0f;
    }
    __syncthreads();

    }  // end else (gate_weight != 0)

    // ── Phase 2: weighted sum + blend (identical accumulation) + residual ──
    unsigned int j = blockIdx.x * blockDim.x + tid;
    if (j >= hidden) return;

    float acc = 0.0f;
    for (unsigned int e = 0; e < top_k; e++) {
        acc += my_weights[e] * __bfloat162float(my_expert_out[(unsigned long long)e * hidden + j]);
    }
    acc += sigmoid_val * __bfloat162float(my_shared_out[j]);
    const __nv_bfloat16 blended = __float2bfloat16(acc);
    my_output[j] = blended;    // byte-identical moe_output (diagnostics read it)

    // bf16_residual_add fold: read the rounded BF16 blend back, f32 add, round.
    const float r = __bfloat162float(my_hidden[j]);
    const float s = __bfloat162float(blended);
    my_hidden[j] = __float2bfloat16(r + s);
}
