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
// ── lane -> dim mapping is INTERLEAVED, not contiguous ────────────────────
// A lane owning a contiguous 64-dim block put the 8 lanes of a row 64 elements
// = 128 bytes apart, which with 32 4-byte banks is the SAME bank for all eight
// — an 8-way conflict on every one of the ~128 scalar 2-byte smem loads per
// key. Profiled cost: 104.8 ms/layer, 73% of prefill wall, at 0.16 TFLOPS on
// 17 GFLOP/layer — ~50x off even the scalar FP32 peak, with global traffic
// only ~7% of the time, i.e. smem-instruction bound.
// Lane l now owns dims { l*8 + k*64 + j : k,j in [0,8) } and loads each k-chunk
// as one 16-byte vector: the 8 lanes then cover 128 contiguous bytes = all 32
// banks, conflict-free, with 8x fewer load instructions.
//
// This REORDERS the dot-product summation (same terms, different order), so it
// is not bit-identical to the contiguous mapping — validated by tool-eval-bench
// rather than by byte comparison.
//
// The masking below IS exact: rows no longer carry per-row loop bounds (which
// made the warp-level shuffles diverge); every row walks the whole tile union
// and keys outside a row's own window get score = -INFINITY. That is an exact no-op in
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
// Chunks per lane. MUST be a compile-time constant: with a runtime chunk count
// the per-lane arrays are dynamically indexed, ptxas cannot keep them in
// registers, and they land in local memory (a 768-byte stack frame was
// measured). head_dim is 512 for this model = 8 lanes x NCHUNK x 8 dims.
#define NCHUNK 8

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
    // __align__(16) is REQUIRED, not cosmetic: the inner loops read each
    // 8-dim chunk as a uint4, and a bf16 array is only 2-byte aligned by
    // default. head_dim (512) and lane_base (multiples of 8) keep every chunk
    // offset a multiple of 16 bytes, so aligning the base makes every uint4
    // access 16-byte aligned.
    __shared__ __align__(16) __nv_bfloat16 sK[KT * MAX_HD];
    __shared__ __align__(16) __nv_bfloat16 sV[KT * MAX_HD];
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (q_head >= num_q_heads) return;

    const unsigned int q_row = q_block * BR + (tid / 8);
    const bool valid = q_row < seq_len;
    const unsigned int dim_lane = tid % 8;
    // Interleaved ownership: lane l owns dims { l*8 + k*64 + j }, k,j in [0,8).
    // Chunk k starts at element `lane_base + k*64` and spans 8 bf16 = 16 bytes,
    // so the 8 lanes of a row cover 128 contiguous bytes = all 32 banks.
    const unsigned int lane_base = dim_lane * 8;
    // No early `return` for out-of-range lanes: every thread must reach the
    // __syncthreads() in the tile loops.
    const bool lane_live = lane_base < head_dim;

    const unsigned int gqa = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa;
    const unsigned int q_stride = num_q_heads * head_dim;
    const unsigned int kv_stride = num_kv_heads * head_dim;
    // Shuffle partners are the 8 lanes of ONE row (xor 1/2/4 never leaves the
    // aligned 8-group), so reduce over that group's mask rather than 0xFFFFFFFF.
    const unsigned int lane_in_warp = tid & 31u;
    const unsigned int row_mask = 0xFFu << (lane_in_warp & ~7u);

    const __nv_bfloat16* Qr = Q + (size_t)q_row * q_stride + (size_t)q_head * head_dim;

    float m = -1e30f, l = 0.0f;
    // o_acc[k*8 + j] holds dim (lane_base + k*64 + j).
    float o_acc[64];
    #pragma unroll
    for (unsigned int i = 0; i < 64; ++i) o_acc[i] = 0.0f;

    // Q row into registers once: reused by every key (it used to be re-read
    // from global memory per key).
    float q_reg[64];
    #pragma unroll
    for (unsigned int i = 0; i < 64; ++i) q_reg[i] = 0.0f;
    if (valid && lane_live) {
        for (unsigned int k = 0; k < NCHUNK; ++k) {
            const unsigned int d0 = lane_base + k * 64;
            #pragma unroll
            for (unsigned int j = 0; j < 8; ++j)
                q_reg[k * 8 + j] = __bfloat162float(Qr[d0 + j]);
        }
    }

    // Fold one staged key into the online softmax. IN_RANGE false => score is
    // -INFINITY, an exact no-op: m_new = max(m, -inf) = m, eo = exp(0) = 1,
    // en = exp(-inf) = 0. Every thread runs this for every staged key, so the
    // shuffles stay convergent.
    #define ATTEND_TILE(KS, VS, IN_RANGE)                                      \
    do {                                                                        \
        float dot = 0.0f;                                                      \
        if (lane_live) {                                                       \
            _Pragma("unroll")                                                  \
            for (unsigned int k = 0; k < NCHUNK; ++k) {                        \
                const uint4 kv4 =                                              \
                    *reinterpret_cast<const uint4*>((KS) + lane_base + k * 64);\
                const __nv_bfloat16* kb =                                      \
                    reinterpret_cast<const __nv_bfloat16*>(&kv4);              \
                _Pragma("unroll")                                              \
                for (unsigned int j = 0; j < 8; ++j)                           \
                    dot += q_reg[k * 8 + j] * __bfloat162float(kb[j]);         \
            }                                                                  \
        }                                                                      \
        dot += __shfl_xor_sync(row_mask, dot, 1);                              \
        dot += __shfl_xor_sync(row_mask, dot, 2);                              \
        dot += __shfl_xor_sync(row_mask, dot, 4);                              \
        float score = (IN_RANGE) ? (dot * inv_sqrt_d) : -INFINITY;             \
        float m_new = fmaxf(m, score);                                         \
        float eo = __expf(m - m_new);                                          \
        float en = __expf(score - m_new);                                      \
        if (lane_live) {                                                       \
            _Pragma("unroll")                                                  \
            for (unsigned int k = 0; k < NCHUNK; ++k) {                        \
                const uint4 vv4 =                                              \
                    *reinterpret_cast<const uint4*>((VS) + lane_base + k * 64);\
                const __nv_bfloat16* vb =                                      \
                    reinterpret_cast<const __nv_bfloat16*>(&vv4);              \
                _Pragma("unroll")                                              \
                for (unsigned int j = 0; j < 8; ++j)                           \
                    o_acc[k * 8 + j] = o_acc[k * 8 + j] * eo                   \
                                     + en * __bfloat162float(vb[j]);           \
            }                                                                  \
        }                                                                      \
        l = l * eo + en;                                                       \
        m = m_new;                                                             \
    } while (0)

    // Stage COUNT rows starting at BASE into sK (and sV when V does not alias).
    #define STAGE_TILE(KSRC, VSRC, STRIDE, HEAD_OFF, BASE, COUNT, SAME)        \
    do {                                                                        \
        for (unsigned int kk = 0; kk < (COUNT); ++kk) {                        \
            const __nv_bfloat16* ks =                                          \
                (KSRC) + (size_t)((BASE) + kk) * (STRIDE) + (HEAD_OFF);        \
            for (unsigned int dd = tid; dd < head_dim; dd += 128)              \
                sK[kk * head_dim + dd] = ks[dd];                               \
            if (!(SAME)) {                                                     \
                const __nv_bfloat16* vs =                                      \
                    (VSRC) + (size_t)((BASE) + kk) * (STRIDE) + (HEAD_OFF);    \
                for (unsigned int dd = tid; dd < head_dim; dd += 128)          \
                    sV[kk * head_dim + dd] = vs[dd];                           \
            }                                                                  \
        }                                                                      \
    } while (0)

    // V==K at this call site (MLA keeps V in K's buffer); stage once when so.
    const bool kv_same = (K == V);
    const bool comp_same = (Kc == Vc);

    // ── raw keys (sliding-window causal) ──────────────────────────────────
    // Union over the block's 16 rows: row q covers [max(0,q+1-W), q]. Rows walk
    // the union and mask; the extra keys are exact no-ops.
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
            STAGE_TILE(K, V, kv_stride, (size_t)kv_head * head_dim, base, count, kv_same);
            __syncthreads();
            const __nv_bfloat16* Vtile = kv_same ? sK : sV;
            for (unsigned int kk = 0; kk < count; ++kk) {
                const unsigned int kp = base + kk;
                const bool in_range = valid && kp >= kv_start && kp < kv_len;
                ATTEND_TILE(sK + (size_t)kk * head_dim,
                            Vtile + (size_t)kk * head_dim, in_range);
            }
        }
    }

    // ── compressed keys (windowed-causal) ─────────────────────────────────
    unsigned int comp_vis = valid ? ((q_row + 1u) / ratio) : 0u;
    if (comp_vis > n_comp) comp_vis = n_comp;
    // comp_vis is monotonic in q_row, so the block's union is the last row's.
    unsigned int comp_union = (q_last_excl > 0u) ? (q_last_excl / ratio) : 0u;
    if (comp_union > n_comp) comp_union = n_comp;
    for (unsigned int base = 0; base < comp_union; base += KT) {
        unsigned int count = comp_union - base;
        if (count > KT) count = KT;
        __syncthreads();
        STAGE_TILE(Kc, Vc, head_dim, 0u, base, count, comp_same);
        __syncthreads();
        const __nv_bfloat16* Vtile = comp_same ? sK : sV;
        for (unsigned int kk = 0; kk < count; ++kk) {
            const bool in_range = (base + kk) < comp_vis;
            ATTEND_TILE(sK + (size_t)kk * head_dim,
                        Vtile + (size_t)kk * head_dim, in_range);
        }
    }

    // ── attention sink: per-head logit in the denominator only (no value) ──
    if (valid && lane_live && sinks != nullptr) {
        float sg = sinks[q_head];
        float m_new = fmaxf(m, sg);
        float eo = __expf(m - m_new);
        #pragma unroll
        for (unsigned int i = 0; i < 64; ++i) o_acc[i] *= eo;
        l = l * eo + __expf(sg - m_new);
        m = m_new;
    }

    if (valid && lane_live) {
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        __nv_bfloat16* Or = O + (size_t)q_row * q_stride + (size_t)q_head * head_dim;
        for (unsigned int k = 0; k < NCHUNK; ++k) {
            const unsigned int d0 = lane_base + k * 64;
            #pragma unroll
            for (unsigned int j = 0; j < 8; ++j)
                Or[d0 + j] = __float2bfloat16(o_acc[k * 8 + j] * inv_l);
        }
    }
    #undef ATTEND_TILE
    #undef STAGE_TILE
}
