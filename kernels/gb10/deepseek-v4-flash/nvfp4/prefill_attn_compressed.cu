// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4 CSA prefill attention: core attention over the concatenation of
//   [ raw sliding-window KV (causal) | compressed KV (windowed-causal) ]
// plus a per-head attention sink, in one softmax. Reference:
// modeling_deepseek_v4.py DeepseekV4Attention.forward (compressor path) +
// eager_attention_forward with s_aux=self.sinks.
//
// Q/K/V are [S, num_heads, head_dim] (num_kv_heads=1 → MQA broadcast).
// Compressed K/V are [n_comp, head_dim]. Query at position t attends to:
//   - raw keys 0..t   (standard causal)
//   - compressed entries w where (w+1)*ratio <= t+1  (window fully in the past)
//   - the per-head sink logit (no value; only enters the softmax denominator)
//
// Layout: 128 threads = 16 rows × 8 dim-lanes, each dim-lane owns 64 of the
// head_dim=512 output dims.
// Grid: (num_q_heads, ceil(S/16), batch)  Block: (128,1,1)
//
// ── K/V are staged through shared memory ──────────────────────────────────
// The original inner loop read K and V straight from global for EVERY query
// row, so the 16 rows in a block each streamed the same keys and every head
// re-streamed them again (num_kv_heads=1 → all 64 q-heads share one K/V).
// nsys on a 911-token prefill: 139 ms/call, 5.0 s total, 25.5% of prefill GPU
// time — and ~15 GB of K/V traffic per layer against 1.9 MB of actual K/V.
//
// Now each block loads a tile of KT keys into shared memory once and all 16
// rows attend against it, and because this call site passes the SAME pointer
// for K and V (MLA keeps V==K, rope in the tail) the tile is loaded once and
// used for both when the pointers match.
//
// BIT-EXACTNESS: rows no longer carry per-row loop bounds (which made the
// warp-level shuffles diverge); every row walks the whole tile union and keys
// outside a row's own window get score = -INFINITY. That is an exact no-op in
// the online softmax — m_new = max(m, -inf) = m, so eo = exp(0) = 1 and
// en = exp(-inf) = 0, leaving m, l and o_acc untouched — so each row still
// folds exactly its own keys, in ascending order, in the same order and with
// the same float ops as before. Masked keys must use -INFINITY and not a large
// negative constant: at the initial m = -1e30 a score of -1e30 would give
// en = exp(0) = 1 and fold a spurious key.

#include <cuda_bf16.h>

#define BR 16
// Keys staged per shared-memory tile. KT*MAX_HD*2 bytes per buffer, two
// buffers → 32 KB, under the 48 KB static-smem limit with no opt-in.
#define KT 16
#define MAX_HD 512

extern "C" __global__ void prefill_attn_compressed(
    const __nv_bfloat16* __restrict__ Q,       // [S, num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ K,       // [S, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ V,       // [S, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ Kc,      // [n_comp, head_dim]  (kv head 0)
    const __nv_bfloat16* __restrict__ Vc,      // [n_comp, head_dim]
    const float* __restrict__ sinks,   // [num_q_heads]  per-head sink logit (FP32: checkpoint-native; reading as bf16 hard-zeroed 7 heads)
    __nv_bfloat16* __restrict__ O,             // [S, num_q_heads, head_dim]
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int n_comp,
    const unsigned int ratio,
    const unsigned int sliding_window,   // raw arm attends only the last `sliding_window` keys (0 = full)
    const float inv_sqrt_d
) {
    __shared__ __nv_bfloat16 sK[KT * MAX_HD];
    __shared__ __nv_bfloat16 sV[KT * MAX_HD];

    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (q_head >= num_q_heads) return;

    const unsigned int q_row = q_block * BR + (tid / 8);
    const bool valid = q_row < seq_len;
    const unsigned int dim_lane = tid % 8;
    const unsigned int dim_start = dim_lane * 64;
    // NOTE: no early `return` on dim_start >= head_dim — every thread must
    // reach the __syncthreads() in the tile loop. head_dim <= MAX_HD = 8*64
    // holds for this model; a lane past the end simply does no useful work.
    const bool lane_live = dim_start < head_dim;

    const unsigned int gqa = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa;
    const unsigned int q_stride = num_q_heads * head_dim;
    const unsigned int kv_stride = num_kv_heads * head_dim;
    // Shuffle partners are the 8 lanes of ONE row (xor 1/2/4 never leaves the
    // aligned 8-group), so reduce over that group's mask rather than the whole
    // warp — the block is now uniform, but the narrow mask states the intent.
    const unsigned int lane_in_warp = tid & 31u;
    const unsigned int row_mask = 0xFFu << (lane_in_warp & ~7u);

    const __nv_bfloat16* Qr = Q + (size_t)q_row * q_stride + (size_t)q_head * head_dim;

    float m = -1e30f, l = 0.0f;
    float o_acc[64];
    for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; ++d) o_acc[d] = 0.0f;

    // Q row into registers once: it is reused by every key.
    float q_reg[64];
    for (unsigned int d = 0; d < 64; ++d) {
        q_reg[d] = (valid && lane_live && dim_start + d < head_dim)
                     ? __bfloat162float(Qr[dim_start + d])
                     : 0.0f;
    }

    // Fold one staged key into the online softmax. `in_range` false ⇒ exact
    // no-op (see BIT-EXACTNESS above). Every thread in the block executes
    // this for every staged key, so the shuffles stay convergent.
    #define ATTEND_S(SLOT, IN_RANGE)                                          \
    do {                                                                       \
        const __nv_bfloat16* Ks = sK + (size_t)(SLOT) * head_dim;             \
        const __nv_bfloat16* Vs = sV + (size_t)(SLOT) * head_dim;             \
        float dot = 0.0f;                                                     \
        if (lane_live) {                                                      \
            for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; ++d) \
                dot += q_reg[d] * __bfloat162float(Ks[dim_start + d]);        \
        }                                                                     \
        dot += __shfl_xor_sync(row_mask, dot, 1);                             \
        dot += __shfl_xor_sync(row_mask, dot, 2);                             \
        dot += __shfl_xor_sync(row_mask, dot, 4);                             \
        float score = (IN_RANGE) ? (dot * inv_sqrt_d) : -INFINITY;            \
        float m_new = fmaxf(m, score);                                        \
        float eo = __expf(m - m_new);                                         \
        float en = __expf(score - m_new);                                     \
        for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; ++d)     \
            o_acc[d] = o_acc[d] * eo + en * __bfloat162float(Vs[dim_start + d]); \
        l = l * eo + en;                                                      \
        m = m_new;                                                            \
    } while (0)

    // Stage `count` rows starting at `base` from a [*, stride] source.
    #define STAGE(SRC, STRIDE, HEAD_OFF, BASE, COUNT)                          \
    do {                                                                       \
        for (unsigned int kk = 0; kk < (COUNT); ++kk) {                        \
            const __nv_bfloat16* src =                                         \
                (SRC) + (size_t)((BASE) + kk) * (STRIDE) + (HEAD_OFF);         \
            for (unsigned int dd = tid; dd < head_dim; dd += 128)              \
                sK[kk * head_dim + dd] = src[dd];                              \
        }                                                                      \
    } while (0)

    // V==K at this call site (MLA keeps V in K's buffer); stage once when so.
    const bool kv_same = (K == V);
    const bool comp_same = (Kc == Vc);

    // ── raw keys (sliding-window causal) ──────────────────────────────────
    // Union over the block's 16 rows: row q covers [max(0,q+1-W), q]. Rows
    // walk the union and mask; the extra keys are exact no-ops.
    const unsigned int q_first = q_block * BR;
    const unsigned int q_last_excl = (q_first + BR < seq_len) ? (q_first + BR) : seq_len;
    const unsigned int kv_start = valid ? ((sliding_window > 0u && q_row + 1u > sliding_window)
                                             ? (q_row + 1u - sliding_window) : 0u)
                                        : 0u;
    const unsigned int kv_len = valid ? (q_row + 1u) : 0u;
    if (q_first < seq_len) {
        unsigned int union_lo = 0u;
        if (sliding_window > 0u && q_first + 1u > sliding_window)
            union_lo = q_first + 1u - sliding_window;
        const unsigned int union_hi = q_last_excl;  // exclusive
        for (unsigned int base = union_lo; base < union_hi; base += KT) {
            unsigned int count = union_hi - base;
            if (count > KT) count = KT;
            __syncthreads();
            STAGE(K, kv_stride, (size_t)kv_head * head_dim, base, count);
            if (!kv_same) {
                for (unsigned int kk = 0; kk < count; ++kk) {
                    const __nv_bfloat16* src =
                        V + (size_t)(base + kk) * kv_stride + (size_t)kv_head * head_dim;
                    for (unsigned int dd = tid; dd < head_dim; dd += 128)
                        sV[kk * head_dim + dd] = src[dd];
                }
            }
            __syncthreads();
            const __nv_bfloat16* Vtile = kv_same ? sK : sV;
            for (unsigned int kk = 0; kk < count; ++kk) {
                const unsigned int kp = base + kk;
                const bool in_range = valid && kp >= kv_start && kp < kv_len;
                // Re-point the V read when V aliases K.
                const __nv_bfloat16* Vs = Vtile + (size_t)kk * head_dim;
                const __nv_bfloat16* Ks = sK + (size_t)kk * head_dim;
                float dot = 0.0f;
                if (lane_live) {
                    for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; ++d)
                        dot += q_reg[d] * __bfloat162float(Ks[dim_start + d]);
                }
                dot += __shfl_xor_sync(row_mask, dot, 1);
                dot += __shfl_xor_sync(row_mask, dot, 2);
                dot += __shfl_xor_sync(row_mask, dot, 4);
                float score = in_range ? (dot * inv_sqrt_d) : -INFINITY;
                float m_new = fmaxf(m, score);
                float eo = __expf(m - m_new);
                float en = __expf(score - m_new);
                for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; ++d)
                    o_acc[d] = o_acc[d] * eo + en * __bfloat162float(Vs[dim_start + d]);
                l = l * eo + en;
                m = m_new;
            }
        }
    }

    // ── compressed keys (windowed-causal) ─────────────────────────────────
    unsigned int comp_vis = valid ? ((q_row + 1u) / ratio) : 0u;
    if (comp_vis > n_comp) comp_vis = n_comp;
    // Union: comp_vis is monotonic in q_row, so the block's max is the last
    // valid row's. Compute it without assuming which rows are valid.
    unsigned int comp_union = (q_last_excl > 0u) ? (q_last_excl / ratio) : 0u;
    if (comp_union > n_comp) comp_union = n_comp;
    for (unsigned int base = 0; base < comp_union; base += KT) {
        unsigned int count = comp_union - base;
        if (count > KT) count = KT;
        __syncthreads();
        STAGE(Kc, head_dim, 0u, base, count);
        if (!comp_same) {
            for (unsigned int kk = 0; kk < count; ++kk) {
                const __nv_bfloat16* src = Vc + (size_t)(base + kk) * head_dim;
                for (unsigned int dd = tid; dd < head_dim; dd += 128)
                    sV[kk * head_dim + dd] = src[dd];
            }
        }
        __syncthreads();
        const __nv_bfloat16* Vtile = comp_same ? sK : sV;
        for (unsigned int kk = 0; kk < count; ++kk) {
            const bool in_range = (base + kk) < comp_vis;
            const __nv_bfloat16* Ks = sK + (size_t)kk * head_dim;
            const __nv_bfloat16* Vs = Vtile + (size_t)kk * head_dim;
            float dot = 0.0f;
            if (lane_live) {
                for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; ++d)
                    dot += q_reg[d] * __bfloat162float(Ks[dim_start + d]);
            }
            dot += __shfl_xor_sync(row_mask, dot, 1);
            dot += __shfl_xor_sync(row_mask, dot, 2);
            dot += __shfl_xor_sync(row_mask, dot, 4);
            float score = in_range ? (dot * inv_sqrt_d) : -INFINITY;
            float m_new = fmaxf(m, score);
            float eo = __expf(m - m_new);
            float en = __expf(score - m_new);
            for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; ++d)
                o_acc[d] = o_acc[d] * eo + en * __bfloat162float(Vs[dim_start + d]);
            l = l * eo + en;
            m = m_new;
        }
    }

    // ── attention sink: per-head logit in the denominator only (no value) ──
    if (valid && lane_live && sinks != nullptr) {
        float sg = sinks[q_head];
        float m_new = fmaxf(m, sg);
        float eo = __expf(m - m_new);
        for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; ++d) o_acc[d] *= eo;
        l = l * eo + __expf(sg - m_new);
        m = m_new;
    }

    if (valid && lane_live) {
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        __nv_bfloat16* Or = O + (size_t)q_row * q_stride + (size_t)q_head * head_dim;
        for (unsigned int d = 0; d < 64 && dim_start + d < head_dim; ++d)
            Or[dim_start + d] = __float2bfloat16(o_acc[d] * inv_l);
    }
    #undef ATTEND_S
    #undef STAGE
}
