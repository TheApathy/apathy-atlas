// SPDX-License-Identifier: AGPL-3.0-only

// MLA Absorbed Attention Kernels — batched per-head GEMV for Q absorption and V extraction.
//
// Q absorption: Q_absorbed[n, Lkv] = Q_nope[n, P] @ W_UK_T[n, P, Lkv]
//   - 32 heads in parallel, each head does [1, P=64] @ [P, Lkv=256] → [1, Lkv=256]
//   - Grid: (ceil(Lkv/4), N_heads, 1)  Block: (256, 1, 1)
//
// V extraction: v_out[n, V] = attn_latent[n, Lkv] @ W_UV[n, V, Lkv]^T
//   - Actually: v_out[n, v] = sum_l(W_UV[n, v, l] * attn_latent[n, l])
//   - 32 heads in parallel, each head does [V=128, Lkv=256] @ [Lkv=256, 1] → [V=128, 1]
//   - Grid: (ceil(V/4), N_heads, 1)  Block: (256, 1, 1)
//
// Both kernels use the same structure: batched GEMV with per-head weight pointers.
// Input is at a fixed stride per head in the input buffer.
// Output is at a fixed stride per head in the output buffer.

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32

// Batched GEMV: output[head, n] = sum_k(weight[head, n, k] * input[head, k])
// for all heads in parallel.
//
// Grid: (ceil(N_out / (N_PER_BLOCK*2)), num_heads, 1)
// Block: (256, 1, 1)
//
// input:  [num_heads, K]        BF16, contiguous per head at stride input_stride
// weight: [num_heads, N_out, K] BF16, contiguous per head at stride N_out * K
// output: [num_heads, N_out]    BF16, contiguous per head at stride output_stride
extern "C" __global__ void mla_batched_gemv(
    const __nv_bfloat16* __restrict__ input,   // [num_heads * input_stride]
    const __nv_bfloat16* __restrict__ weight,  // [num_heads * N_out * K]
    __nv_bfloat16* __restrict__ output,         // [num_heads * output_stride]
    unsigned int N_out,                         // output dimension per head
    unsigned int K,                             // input dimension per head
    unsigned int input_stride,                  // elements between consecutive heads in input
    unsigned int output_stride                  // elements between consecutive heads in output
) {
    const unsigned int head = blockIdx.y;
    const unsigned int tid = threadIdx.x;

    // Each block computes N_PER_BLOCK * 2 output elements for one head
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = tid / threads_per_out;            // 0..3
    const unsigned int lane = tid % threads_per_out;                 // 0..63

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N_out) return;
    const bool have_n2 = (n2 < N_out);

    // Pointers for this head
    const __nv_bfloat16* A = input + (unsigned long long)head * input_stride;
    const __nv_bfloat16* B = weight + (unsigned long long)head * N_out * K;
    __nv_bfloat16* C = output + (unsigned long long)head * output_stride;

    const unsigned int K4 = K / 4;
    const unsigned long long* A64 = (const unsigned long long*)A;

    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k4 = lane; k4 < K4; k4 += threads_per_out) {
        // Load 4 input values (vectorized 64-bit load)
        unsigned long long av = A64[k4];
        float a0, a1, a2, a3;
        unsigned int lo = (unsigned int)av;
        unsigned int hi = (unsigned int)(av >> 32);
        __nv_bfloat16 tmp;
        *(unsigned short*)&tmp = (unsigned short)(lo & 0xFFFF); a0 = __bfloat162float(tmp);
        *(unsigned short*)&tmp = (unsigned short)(lo >> 16);     a1 = __bfloat162float(tmp);
        *(unsigned short*)&tmp = (unsigned short)(hi & 0xFFFF); a2 = __bfloat162float(tmp);
        *(unsigned short*)&tmp = (unsigned short)(hi >> 16);     a3 = __bfloat162float(tmp);

        unsigned int base_k = k4 * 4;

        // Weight row n1
        float w10 = __bfloat162float(B[n1 * K + base_k]);
        float w11 = __bfloat162float(B[n1 * K + base_k + 1]);
        float w12 = __bfloat162float(B[n1 * K + base_k + 2]);
        float w13 = __bfloat162float(B[n1 * K + base_k + 3]);
        acc1 += a0 * w10 + a1 * w11 + a2 * w12 + a3 * w13;

        if (have_n2) {
            float w20 = __bfloat162float(B[n2 * K + base_k]);
            float w21 = __bfloat162float(B[n2 * K + base_k + 1]);
            float w22 = __bfloat162float(B[n2 * K + base_k + 2]);
            float w23 = __bfloat162float(B[n2 * K + base_k + 3]);
            acc2 += a0 * w20 + a1 * w21 + a2 * w22 + a3 * w23;
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        if (have_n2) acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    // Cross-warp reduction via shared memory
    __shared__ float s_partial[N_PER_BLOCK * 2][2]; // [out_idx][warp_idx within out]
    unsigned int warp_in_out = (tid % threads_per_out) / WARP_SIZE;
    unsigned int lane_in_warp = tid % WARP_SIZE;
    if (lane_in_warp == 0) {
        s_partial[local_out * 2][warp_in_out] = acc1;
        if (have_n2) s_partial[local_out * 2 + 1][warp_in_out] = acc2;
    }
    __syncthreads();

    // Final reduction: thread 0 of each output element
    unsigned int warps_per_out = threads_per_out / WARP_SIZE;
    if (lane_in_warp == 0 && warp_in_out == 0) {
        float sum1 = 0.0f;
        for (unsigned int w = 0; w < warps_per_out; w++) sum1 += s_partial[local_out * 2][w];
        C[n1] = __float2bfloat16(sum1);

        if (have_n2) {
            float sum2 = 0.0f;
            for (unsigned int w = 0; w < warps_per_out; w++) sum2 += s_partial[local_out * 2 + 1][w];
            C[n2] = __float2bfloat16(sum2);
        }
    }
}

// Assemble Q for absorbed MLA: copies Q_absorbed + Q_rope into contiguous [Lkv+R] per head.
// Also handles RoPE application to Q_rope and K_rope.
//
// Grid: (num_heads, 1, 1)  Block: (max(Lkv, R), 1, 1)
// NOT IMPLEMENTED YET — use D2D copies for now.

// Fused Q_rope extract + writeback: eliminates 64 D2D copies per layer.
// Extracts Q_rope from q_full[nq, hd] at offset nope per head,
// then writes to q_absorbed_buf at offset kv_lora per head (stride mla_cache_dim).
//
// Grid: (1, 1, 1)  Block: (256, 1, 1)
// Each thread handles ceil(nq * rope / 256) elements.
extern "C" __global__ void mla_q_rope_scatter(
    const __nv_bfloat16* __restrict__ q_full,      // [nq, hd]
    __nv_bfloat16* __restrict__ q_absorbed_buf,     // [nq, mla_cache_dim]
    __nv_bfloat16* __restrict__ q_rope_contiguous,  // [nq * rope] for RoPE kernel
    unsigned int nq,
    unsigned int hd,            // head_dim (512)
    unsigned int nope,          // nope head dim (448)
    unsigned int rope,          // rope head dim (64)
    unsigned int kv_lora,       // kv_lora_rank (512)
    unsigned int mla_cache_dim  // kv_lora + rope (576)
) {
    unsigned int total = nq * rope;
    for (unsigned int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        unsigned int head = idx / rope;
        unsigned int r = idx % rope;
        // Read from q_full[head * hd + nope + r]
        __nv_bfloat16 val = q_full[head * hd + nope + r];
        // Write to BOTH destinations in one pass (eliminates separate extract loop)
        q_absorbed_buf[head * mla_cache_dim + kv_lora + r] = val;
        q_rope_contiguous[head * rope + r] = val;
    }
}

// Scatter RoPE'd Q_rope back to strided q_absorbed_buf layout.
// After RoPE, q_rope_direct is [nq, rope] contiguous.
// Write to q_absorbed_buf[head * mla_cache_dim + kv_lora .. + kv_lora + rope].
extern "C" __global__ void mla_q_rope_writeback(
    const __nv_bfloat16* __restrict__ q_rope_direct,   // [nq * rope] contiguous
    __nv_bfloat16* __restrict__ q_absorbed_buf,         // [nq, mla_cache_dim]
    unsigned int nq,
    unsigned int rope,
    unsigned int kv_lora,
    unsigned int mla_cache_dim
) {
    unsigned int total = nq * rope;
    for (unsigned int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        unsigned int head = idx / rope;
        unsigned int r = idx % rope;
        q_absorbed_buf[head * mla_cache_dim + kv_lora + r] = q_rope_direct[head * rope + r];
    }
}

// ════════════════════════════════════════════════════════════════════════════
// BATCHED PREFILL VARIANTS — eliminate per-token per-head D2D copy loops
// ════════════════════════════════════════════════════════════════════════════

// Extract Q rope portions from expanded Q[N, nq, hd] into contiguous [N, nq, rope] for RoPE.
// Replaces: for t in 0..N { for h in 0..nq { copy_d2d(q_full[t,h,nope:], q_rope[t,h]) } }
// Grid: (ceil(total/256), 1, 1)  Block: (256, 1, 1)  where total = num_tokens * nq * rope
extern "C" __global__ void mla_q_rope_extract_batched(
    const __nv_bfloat16* __restrict__ q_full,     // [N, q_dim] where q_dim = nq * hd
    __nv_bfloat16* __restrict__ q_rope_out,        // [N, nq * rope] contiguous
    unsigned int num_tokens,
    unsigned int nq,
    unsigned int hd,
    unsigned int nope,
    unsigned int rope,
    unsigned int q_dim                              // nq * hd
) {
    unsigned int total = num_tokens * nq * rope;
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        unsigned int t = idx / (nq * rope);
        unsigned int rem = idx % (nq * rope);
        unsigned int head = rem / rope;
        unsigned int r = rem % rope;
        q_rope_out[t * nq * rope + head * rope + r] =
            q_full[t * q_dim + head * hd + nope + r];
    }
}

// Write back RoPE'd Q rope portions into expanded Q[N, nq, hd] at offset nope per head.
// Replaces: for t in 0..N { for h in 0..nq { copy_d2d(q_rope[t,h], q_full[t,h,nope:]) } }
// Grid: (ceil(total/256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void mla_q_rope_writeback_batched(
    const __nv_bfloat16* __restrict__ q_rope_in,  // [N, nq * rope] contiguous
    __nv_bfloat16* __restrict__ q_full,            // [N, q_dim]
    unsigned int num_tokens,
    unsigned int nq,
    unsigned int hd,
    unsigned int nope,
    unsigned int rope,
    unsigned int q_dim
) {
    unsigned int total = num_tokens * nq * rope;
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        unsigned int t = idx / (nq * rope);
        unsigned int rem = idx % (nq * rope);
        unsigned int head = rem / rope;
        unsigned int r = rem % rope;
        q_full[t * q_dim + head * hd + nope + r] =
            q_rope_in[t * nq * rope + head * rope + r];
    }
}

// Assemble K=[nope|rope] and extract V from kv_expanded for N tokens.
// K: concatenate k_nope (from kv_expanded per head) with k_rope (broadcast from single head).
// V: extract v_dim portion from kv_expanded per head.
// Replaces: for t in 0..N { for h in 0..nkv { 3 copy_d2d calls } }
// Grid: (num_tokens, 2, 1)  Block: (256, 1, 1)
//   blockIdx.y==0: assemble K [nkv * hd elements per token]
//   blockIdx.y==1: extract V [nkv * v_dim elements per token]
extern "C" __global__ void mla_kv_assemble_batched(
    const __nv_bfloat16* __restrict__ kv_expanded,  // [N, nkv * (nope + v_dim)]
    const __nv_bfloat16* __restrict__ k_rope_buf,   // [N, rope]
    __nv_bfloat16* __restrict__ k_out,               // [N, nkv * hd] where hd = nope + rope
    __nv_bfloat16* __restrict__ v_out,               // [N, nkv * v_dim]
    unsigned int nkv,
    unsigned int nope,
    unsigned int v_dim,
    unsigned int rope,
    unsigned int hd,                                 // nope + rope (= K head dim)
    unsigned int kv_expanded_stride                  // nkv * (nope + v_dim) per token
) {
    unsigned int t = blockIdx.x;  // token index

    if (blockIdx.y == 0) {
        // Assemble K: [nkv, hd] where hd = nope + rope
        unsigned int k_total = nkv * hd;
        for (unsigned int idx = threadIdx.x; idx < k_total; idx += blockDim.x) {
            unsigned int head = idx / hd;
            unsigned int dim = idx % hd;
            __nv_bfloat16 val;
            if (dim < nope) {
                // k_nope from kv_expanded[t, head, dim]
                val = kv_expanded[(unsigned long long)t * kv_expanded_stride + head * (nope + v_dim) + dim];
            } else {
                // k_rope broadcast from single-head k_rope_buf[t, dim - nope]
                val = k_rope_buf[(unsigned long long)t * rope + (dim - nope)];
            }
            k_out[(unsigned long long)t * nkv * hd + idx] = val;
        }
    } else {
        // Extract V: [nkv, v_dim]
        unsigned int v_total = nkv * v_dim;
        for (unsigned int idx = threadIdx.x; idx < v_total; idx += blockDim.x) {
            unsigned int head = idx / v_dim;
            unsigned int dim = idx % v_dim;
            // V is at offset nope within each head's (nope + v_dim) block
            v_out[(unsigned long long)t * nkv * v_dim + idx] =
                kv_expanded[(unsigned long long)t * kv_expanded_stride + head * (nope + v_dim) + nope + dim];
        }
    }
}

// Assemble compressed MLA cache entries for N tokens.
// K_cache = [kv_latent(kv_lora) | k_rope(rope)] per token
// V_cache = [kv_latent(kv_lora) | zeros(rope)] per token
// Replaces: for t in 0..N { 4 copy_d2d/memset calls }
// Grid: (num_tokens, 1, 1)  Block: (mla_cache_dim or 256, 1, 1)
extern "C" __global__ void mla_cache_assemble_batched(
    const __nv_bfloat16* __restrict__ kv_latent,    // [N, kv_lora]
    const __nv_bfloat16* __restrict__ k_rope,       // [N, rope]
    __nv_bfloat16* __restrict__ k_cache,             // [N, mla_cache_dim]
    __nv_bfloat16* __restrict__ v_cache,             // [N, mla_cache_dim]
    unsigned int kv_lora,
    unsigned int rope,
    unsigned int mla_cache_dim                       // kv_lora + rope
) {
    unsigned int t = blockIdx.x;
    unsigned long long k_off = (unsigned long long)t * mla_cache_dim;
    unsigned long long lat_off = (unsigned long long)t * kv_lora;
    unsigned long long rope_off = (unsigned long long)t * rope;

    for (unsigned int idx = threadIdx.x; idx < mla_cache_dim; idx += blockDim.x) {
        if (idx < kv_lora) {
            __nv_bfloat16 val = kv_latent[lat_off + idx];
            k_cache[k_off + idx] = val;
            v_cache[k_off + idx] = val;
        } else {
            unsigned int r = idx - kv_lora;
            // DeepSeek-V4 MLA: V == K (the kv latent is the key AND the value),
            // so V's rope tail carries the SAME rotated rope as K. Writing zeros
            // here made the paged decode read V with a zeroed rope tail while the
            // prefill inline attention used V with the real rope (k_out) — so
            // decode attention diverged from prefill at every layer and
            // generation derailed. Store V's rope = K's rope.
            __nv_bfloat16 rope_val = k_rope[rope_off + r];
            k_cache[k_off + idx] = rope_val;
            v_cache[k_off + idx] = rope_val;
        }
    }
}

// Assemble Q_final from Q_absorbed and Q_rope: [absorbed(kv_lora)|rope(rope)] per head per token.
// Q_absorbed: [N, nq * kv_lora] contiguous
// Q_rope: [N, nq * rope] contiguous
// Q_final: [N, nq * mla_cache_dim] where mla_cache_dim = kv_lora + rope
// Grid: (ceil(total/256), 1, 1) where total = N * nq * mla_cache_dim
// Block: (256, 1, 1)
extern "C" __global__ void mla_q_final_assemble_batched(
    const __nv_bfloat16* __restrict__ q_absorbed,  // [N, nq * kv_lora]
    const __nv_bfloat16* __restrict__ q_rope,      // [N, nq * rope]
    __nv_bfloat16* __restrict__ q_final,           // [N, nq * mla_cache_dim]
    unsigned int num_tokens,
    unsigned int nq,
    unsigned int kv_lora,
    unsigned int rope,
    unsigned int mla_cache_dim   // kv_lora + rope
) {
    unsigned int total = num_tokens * nq * mla_cache_dim;
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        unsigned int t = idx / (nq * mla_cache_dim);
        unsigned int rem = idx % (nq * mla_cache_dim);
        unsigned int head = rem / mla_cache_dim;
        unsigned int d = rem % mla_cache_dim;
        if (d < kv_lora) {
            q_final[idx] = q_absorbed[t * nq * kv_lora + head * kv_lora + d];
        } else {
            q_final[idx] = q_rope[t * nq * rope + head * rope + (d - kv_lora)];
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// DECODE SINGLE-TOKEN VARIANTS (existing)
// ════════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════════
// V4 M=1 DECODE GLUE FUSION (ATLAS_V4_DECODE_FUSED=1)
//
// The plain-decode attention chain launches 8 tiny glue kernels per layer
// (waterfall 2026-08-10: rope_extract 6.7µs, rope 6.1, rope_writeback 5.1,
// k_rope_extract 5.1, k_rope_writeback 5.1, cache_assemble 5.2, write_kv_cache
// 7.6 — each moving KBs, all launch/node-bound at M=1). The two kernels below
// collapse the rope group (5 launches) and the cache group (2 launches) into
// one launch each:
//
//   v4_decode_rope_fused      = mla_q_rope_extract_batched (Q) +
//                               mla_q_rope_extract_batched (K) +
//                               rope_forward_yarn_interleaved[_inv] +
//                               mla_q_rope_writeback_batched (Q) +
//                               mla_q_rope_writeback_batched (K)
//   v4_decode_cache_fused_fp8 = mla_cache_assemble_batched +
//                               reshape_and_cache_flash_fp8
//
// Numerics tier 1 (bit-identical): the extract/writeback stages are pure BF16
// data movement (copies do not change bits) and the remaining arithmetic is
// written with the exact same expressions, operand order, and conversion
// intrinsics as the incumbent kernels (see rope.cu:rope_forward_yarn_interleaved
// and reshape_and_cache.cu:reshape_and_cache_flash_fp8). No reductions anywhere
// in the chain, so there is no reassociation to worry about.
// ════════════════════════════════════════════════════════════════════════════

#include <cuda_fp8.h>

// Fused decode-step rope for the V4-Flash direct-KV chain: rotates the
// trailing `rope` interleaved channels of each Q head (and the single K head
// when nkv==1) IN PLACE, replacing the extract → rotate → writeback triple.
//
// Layout contract (matches attention_forward_v4.rs step 3): each head's rope
// channels live at [nope, nope+rope) within its `hd`-wide slice; pairs are
// interleaved (2i, 2i+1); frequency for pair i is inv_freq[i]; the YaRN
// attention-temperature mscale is folded into cos/sin exactly as the
// incumbent kernel does. `conjugate != 0` selects the negated-sin (inverse)
// rotation used by the step-5.5 attention-output de-rotation (pass nkv=0 and
// k_full may be null in that mode).
//
// Single token only (positions[0] is THE decode position).
// Grid: (ceil((nq+nkv)*rope/2 / 256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void v4_decode_rope_fused(
    __nv_bfloat16* __restrict__ q_full,          // [nq, hd] rotated in place
    __nv_bfloat16* __restrict__ k_full,          // [nkv, hd] rotated in place (nkv==0: unused)
    const unsigned int* __restrict__ positions,  // [1] absolute decode position
    const unsigned int nq,
    const unsigned int nkv,                      // 0 or 1
    const unsigned int hd,
    const unsigned int nope,
    const unsigned int rope,
    const float* __restrict__ inv_freq,          // [rope/2]
    const float mscale,
    const unsigned int conjugate                 // 0 = forward, 1 = inverse (eq.26 de-rotation)
) {
    const unsigned int pairs_per_head = rope / 2;
    const unsigned int total = (nq + nkv) * pairs_per_head;
    const unsigned int abs_pos = positions[0];
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        const unsigned int head = idx / pairs_per_head;
        const unsigned int pair_idx = idx % pairs_per_head;
        __nv_bfloat16* ptr = (head < nq)
            ? (q_full + (unsigned long long)head * hd + nope)
            : (k_full + (unsigned long long)(head - nq) * hd + nope);
        // Same expressions as rope_forward_yarn_interleaved (bit-identical).
        const float freq = inv_freq[pair_idx];
        const float angle = (float)abs_pos * freq;
        const float cos_val = cosf(angle) * mscale;
        const float sin_val = sinf(angle) * mscale;
        const unsigned int d0 = 2 * pair_idx;
        const unsigned int d1 = d0 + 1;
        float x0 = (float)ptr[d0];
        float x1 = (float)ptr[d1];
        float y0, y1;
        if (conjugate) {
            y0 = x0 * cos_val + x1 * sin_val;
            y1 = x1 * cos_val - x0 * sin_val;
        } else {
            y0 = x0 * cos_val - x1 * sin_val;
            y1 = x1 * cos_val + x0 * sin_val;
        }
        ptr[d0] = __float2bfloat16(y0);
        ptr[d1] = __float2bfloat16(y1);
    }
}

// Same vectorized BF16→FP8 pair conversion as reshape_and_cache.cu (kept
// textually identical so the fused write is bit-identical to the incumbent).
__device__ __forceinline__ __nv_fp8x2_storage_t
v4_bf16x2_to_fp8x2(unsigned int packed_bf16, float inv_scale) {
    float v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed_bf16 & 0xFFFF)));
    float v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed_bf16 >> 16)));
    float2 scaled = make_float2(v0 * inv_scale, v1 * inv_scale);
    return __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
}

// Fused decode-step MLA cache assemble + FP8 paged write. Replaces
// mla_cache_assemble_batched (build the 576-dim [latent|rope] row twice in
// BF16 scratch) + reshape_and_cache_flash_fp8 (re-read + quantize) with one
// kernel that quantizes straight from the sources.
//
// Cache layout contract (mla_paged_decode_fp8.cu): each token row in BOTH the
// K and the V pool is [kv_lora latent | rope] = 576 FP8 bytes; K and V carry
// the SAME values (V4-Flash V==K, incl. the rotated rope tail — see the
// mla_cache_assemble_batched comment for why V's rope must not be zeros) but
// are quantized with their own k_scale / v_scale. Addressing is identical to
// reshape_and_cache_flash_fp8 with num_kv_heads=1, head_dim=kv_lora+rope.
//
// Pair grouping matches the incumbent: pairs never straddle the latent/rope
// boundary because kv_lora is even (512). REQUIRES kv_lora and rope even and
// both source pointers 4-byte aligned (asserted host-side).
//
// Single token only (slot_mapping[0] is THE decode slot; slot<0 = skip).
// Grid: (1, 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void v4_decode_cache_fused_fp8(
    const __nv_bfloat16* __restrict__ kv_latent,   // [kv_lora] pre-rope latent (v_out)
    const __nv_bfloat16* __restrict__ k_rope,      // [rope] ROPED K tail (k_out + nope)
    __nv_fp8_storage_t* __restrict__ k_cache,      // paged FP8 pool
    __nv_fp8_storage_t* __restrict__ v_cache,      // paged FP8 pool
    const long long* __restrict__ slot_mapping,    // [1], int64; <0 = padding
    const unsigned int kv_lora,
    const unsigned int rope,
    const unsigned int block_size,
    const float k_scale,                           // dequant: bf16 = fp8 * k_scale
    const float v_scale,
    const unsigned long long cache_stride          // pool block stride in elements
) {
    const long long slot = slot_mapping[0];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = kv_lora + rope;

    __nv_fp8_storage_t* key_dst = k_cache + (unsigned long long)block_idx * cache_stride
                                          + (unsigned long long)block_offset * n_elems;
    __nv_fp8_storage_t* val_dst = v_cache + (unsigned long long)block_idx * cache_stride
                                          + (unsigned long long)block_offset * n_elems;

    const float inv_k_scale = 1.0f / k_scale;
    const float inv_v_scale = 1.0f / v_scale;

    const unsigned int lat_pairs = kv_lora / 2;
    const unsigned int n_pairs = n_elems / 2;
    const unsigned int* lat32 = (const unsigned int*)kv_latent;
    const unsigned int* rope32 = (const unsigned int*)k_rope;
    __nv_fp8x2_storage_t* key_dst16 = (__nv_fp8x2_storage_t*)key_dst;
    __nv_fp8x2_storage_t* val_dst16 = (__nv_fp8x2_storage_t*)val_dst;

    for (unsigned int i = threadIdx.x; i < n_pairs; i += blockDim.x) {
        const unsigned int packed = (i < lat_pairs) ? lat32[i] : rope32[i - lat_pairs];
        key_dst16[i] = v4_bf16x2_to_fp8x2(packed, inv_k_scale);
        val_dst16[i] = v4_bf16x2_to_fp8x2(packed, inv_v_scale);
    }
}

// Fused KV cache assembly: concatenate [kv_latent | k_rope] → K_cache and [kv_latent | zeros] → V_cache.
// Eliminates 4 D2D copies + 1 memset per decode step.
extern "C" __global__ void mla_cache_assemble(
    const __nv_bfloat16* __restrict__ kv_latent,  // [kv_lora]
    const __nv_bfloat16* __restrict__ k_rope,     // [rope]
    __nv_bfloat16* __restrict__ k_cache_entry,     // [mla_cache_dim]
    __nv_bfloat16* __restrict__ v_cache_entry,     // [mla_cache_dim]
    unsigned int kv_lora,
    unsigned int rope,
    unsigned int mla_cache_dim
) {
    unsigned int idx = threadIdx.x;
    // K = [latent | k_rope]
    if (idx < kv_lora) {
        k_cache_entry[idx] = kv_latent[idx];
        v_cache_entry[idx] = kv_latent[idx];
    } else if (idx < mla_cache_dim) {
        unsigned int r = idx - kv_lora;
        k_cache_entry[idx] = (r < rope) ? k_rope[r] : __float2bfloat16(0.0f);
        v_cache_entry[idx] = __float2bfloat16(0.0f); // V padding = zeros
    }
}
