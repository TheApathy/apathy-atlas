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

// ═══════════════════════════════════════════════════════════════════════════
// TENSOR-CORE variant (design: docs/kernels/prefill-attn-tensorcore.md).
//
// Same semantics as the scalar kernel above — raw sliding-window arm,
// compressed windowed-causal arm, per-head sink, -INFINITY masking as an
// exact softmax no-op — with the two GEMMs per key tile on
// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 (fragment index math
// copied from the verified moe_w4a16_grouped_gemm.cu:148-168 block).
//
// Mapping (the corrected one): the BLOCK owns BR=16 q-rows; 4 warps split
// head_dim=512, 128 dims each = 16 m16n8 n-tiles = 64 f32 o_acc per thread
// (the known-good register footprint). One refinement over the design doc:
// with PARTIAL-S contraction (each warp contracts only its own 128-dim
// k-slice, partials summed through smem) a warp needs only its own slice of
// Q — 8 k-steps x 8 bf16 = 32 u32 registers — so Q never touches shared
// memory and the whole working set fits in ~39 KB STATIC smem (no dynamic-
// smem opt-in needed):
//     sKT  512 x (16+1)  bf16  transposed K tile (QK^T B operand)   17.0 KB
//     sV    16 x (512+8) bf16  natural V tile    (PV   B operand)   16.3 KB
//     sSp    4 x 16 x 16 f32   per-warp S partials                   4.0 KB
//     sP    16 x 18      bf16  softmaxed P       (PV   A operand)    0.6 KB
//
// The softmax is REPLICATED per warp instead of cross-warp: after the
// partial-sum every warp reconstructs the full [16 x KT] S from sSp (its
// 2 n-tiles of D-fragments cover all 16 columns across the 4 lanes of a row
// group), so row max/sum are warp-local shuffles and every warp derives
// bit-identical m/l/eo — no cross-warp softmax state at all.
//
// Tile-level online softmax processes KT=16 keys per rescale instead of the
// scalar kernel's one-key-at-a-time loop, so this is NOT bit-identical to
// the scalar kernel (same terms, different reduction order) — validated
// behaviourally: microtest cosine vs scalar, prefill_scan curve,
// tool-eval-bench >= 90/100 (same contract as the interleaved-lane rewrite).
//
// head_dim MUST be 512 (compile-time layout); the dispatcher falls back to
// the scalar kernel otherwise.
// Grid: (num_q_heads, ceil(S/16), 1)   Block: (128, 1, 1) = 4 warps
// ═══════════════════════════════════════════════════════════════════════════

#define TC_KT 16
#define TC_KT_PAD 17
#define TC_HD 512
#define TC_VPAD 520
#define TC_PPAD 18

extern "C" __global__ void prefill_attn_compressed_tc(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ Kc,
    const __nv_bfloat16* __restrict__ Vc,
    const float* __restrict__ sinks,
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int n_comp,
    const unsigned int ratio,
    const unsigned int sliding_window,
    const float inv_sqrt_d
) {
    __shared__ __align__(16) __nv_bfloat16 sKT[TC_HD * TC_KT_PAD];
    __shared__ __align__(16) __nv_bfloat16 sV[TC_KT * TC_VPAD];
    __shared__ float sSp[4][BR][TC_KT];
    __shared__ __align__(4) __nv_bfloat16 sP[BR * TC_PPAD];

    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    if (q_head >= num_q_heads || head_dim != TC_HD) return;

    const unsigned int tid_x = threadIdx.x;
    const unsigned int warp = tid_x >> 5;          // 0..3: owns dims [warp*128, +128)
    const unsigned int laneid = tid_x & 31u;
    const unsigned int g = laneid >> 2;            // fragment group_id (row g / g+8)
    const unsigned int t = laneid & 3u;            // fragment tid (col pair t*2)

    const unsigned int gqa = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa;
    const unsigned int q_stride = num_q_heads * head_dim;
    const unsigned int kv_stride = num_kv_heads * head_dim;

    const unsigned int q_first = q_block * BR;
    const unsigned int q_last_excl =
        (q_first + BR < seq_len) ? (q_first + BR) : seq_len;
    if (q_first >= seq_len) return;

    // This thread's two fragment rows and their per-row softmax state.
    const unsigned int r0 = g, r1 = g + 8;
    const unsigned int qrow0 = q_first + r0, qrow1 = q_first + r1;
    const bool v0 = qrow0 < seq_len, v1 = qrow1 < seq_len;
    // Raw-arm visibility per row (same bounds math as the scalar kernel).
    const unsigned int kvs0 =
        (sliding_window > 0u && qrow0 + 1u > sliding_window) ? (qrow0 + 1u - sliding_window) : 0u;
    const unsigned int kvl0 = qrow0 + 1u;
    const unsigned int kvs1 =
        (sliding_window > 0u && qrow1 + 1u > sliding_window) ? (qrow1 + 1u - sliding_window) : 0u;
    const unsigned int kvl1 = qrow1 + 1u;
    unsigned int cvis0 = (qrow0 + 1u) / ratio; if (cvis0 > n_comp) cvis0 = n_comp;
    unsigned int cvis1 = (qrow1 + 1u) / ratio; if (cvis1 > n_comp) cvis1 = n_comp;

    float m0 = -1e30f, l0 = 0.0f, m1 = -1e30f, l1 = 0.0f;

    // o_acc: 16 n-tiles (dims warp*128 + nt*8 + {t*2, t*2+1}) x 4 f32.
    float o_acc[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        o_acc[nt][0] = 0.0f; o_acc[nt][1] = 0.0f;
        o_acc[nt][2] = 0.0f; o_acc[nt][3] = 0.0f;
    }

    // ── Q A-fragments for this warp's 128-dim k-slice, loaded from global
    // ONCE (8 k-steps x a0..a3). Invalid rows load zeros (their scores are
    // masked to -INFINITY regardless).
    unsigned int qa[8][4];
    {
        const unsigned int col_base = warp * 128u + t * 2u;
        const __nv_bfloat16* Qr0 =
            Q + (size_t)qrow0 * q_stride + (size_t)q_head * head_dim;
        const __nv_bfloat16* Qr1 =
            Q + (size_t)qrow1 * q_stride + (size_t)q_head * head_dim;
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            const unsigned int c0 = col_base + (unsigned int)s * 16u;
            const unsigned int c1 = c0 + 8u;
            qa[s][0] = v0 ? *reinterpret_cast<const unsigned int*>(Qr0 + c0) : 0u;
            qa[s][1] = v1 ? *reinterpret_cast<const unsigned int*>(Qr1 + c0) : 0u;
            qa[s][2] = v0 ? *reinterpret_cast<const unsigned int*>(Qr0 + c1) : 0u;
            qa[s][3] = v1 ? *reinterpret_cast<const unsigned int*>(Qr1 + c1) : 0u;
        }
    }

    const bool kv_same = (K == V);
    const bool comp_same = (Kc == Vc);

    // ── One KT-key tile: stage → QK^T partials → replicated softmax → PV. ──
    // IN0/IN1 are per-(row, key) visibility lambdas via macro args.
    #define TC_TILE(KSRC, VSRC, STRIDE, HEAD_OFF, BASE, COUNT, SAME,           \
                    LO0, HI0, LO1, HI1)                                        \
    do {                                                                        \
        __syncthreads();                                                       \
        /* Stage K transposed + V natural; one global read when V aliases. */  \
        for (unsigned int idx = tid_x; idx < (COUNT) * 64u; idx += 128u) {     \
            const unsigned int kk = idx / 64u;                                 \
            const unsigned int d0 = (idx % 64u) * 8u;                          \
            const __nv_bfloat16* ks =                                          \
                (KSRC) + (size_t)((BASE) + kk) * (STRIDE) + (HEAD_OFF) + d0;   \
            const uint4 kq = *reinterpret_cast<const uint4*>(ks);              \
            const __nv_bfloat16* kb =                                          \
                reinterpret_cast<const __nv_bfloat16*>(&kq);                   \
            _Pragma("unroll")                                                  \
            for (unsigned int j = 0; j < 8; ++j)                               \
                sKT[(d0 + j) * TC_KT_PAD + kk] = kb[j];                        \
            if (SAME) {                                                        \
                *reinterpret_cast<uint4*>(&sV[kk * TC_VPAD + d0]) = kq;        \
            } else {                                                           \
                const __nv_bfloat16* vs =                                      \
                    (VSRC) + (size_t)((BASE) + kk) * (STRIDE) + (HEAD_OFF) + d0;\
                *reinterpret_cast<uint4*>(&sV[kk * TC_VPAD + d0]) =            \
                    *reinterpret_cast<const uint4*>(vs);                       \
            }                                                                  \
        }                                                                      \
        /* Zero sV tail rows on partial tiles: P is exactly 0 there, but      \
           0 * NaN (uninitialized smem) would poison the PV MMA. */           \
        for (unsigned int idx = tid_x; idx < (TC_KT - (COUNT)) * 64u;          \
             idx += 128u) {                                                    \
            const unsigned int kk = (COUNT) + idx / 64u;                       \
            const unsigned int d0 = (idx % 64u) * 8u;                          \
            *reinterpret_cast<uint4*>(&sV[kk * TC_VPAD + d0]) =                \
                make_uint4(0u, 0u, 0u, 0u);                                    \
        }                                                                      \
        __syncthreads();                                                       \
        /* QK^T partial over this warp's 128 dims: 8 k-steps x 2 n-tiles. */   \
        {                                                                      \
            float sp[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};                     \
            const unsigned short* sB = (const unsigned short*)sKT;             \
            _Pragma("unroll")                                                  \
            for (int s = 0; s < 8; s++) {                                      \
                const unsigned int kd = warp * 128u + (unsigned int)s * 16u;   \
                _Pragma("unroll")                                              \
                for (int nt = 0; nt < 2; nt++) {                               \
                    const unsigned int nc = (unsigned int)nt * 8u + g;         \
                    const unsigned int k0 = kd + t * 2u, k1 = k0 + 8u;         \
                    unsigned int b0 =                                          \
                        ((unsigned int)sB[(k0 + 1) * TC_KT_PAD + nc] << 16) |  \
                        (unsigned int)sB[k0 * TC_KT_PAD + nc];                 \
                    unsigned int b1 =                                          \
                        ((unsigned int)sB[(k1 + 1) * TC_KT_PAD + nc] << 16) |  \
                        (unsigned int)sB[k1 * TC_KT_PAD + nc];                 \
                    asm volatile(                                              \
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"\
                        : "=f"(sp[nt][0]), "=f"(sp[nt][1]),                    \
                          "=f"(sp[nt][2]), "=f"(sp[nt][3])                     \
                        : "r"(qa[s][0]), "r"(qa[s][1]),                        \
                          "r"(qa[s][2]), "r"(qa[s][3]),                        \
                          "r"(b0), "r"(b1),                                    \
                          "f"(sp[nt][0]), "f"(sp[nt][1]),                      \
                          "f"(sp[nt][2]), "f"(sp[nt][3]));                     \
                }                                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int nt = 0; nt < 2; nt++) {                                   \
                const unsigned int c0 = (unsigned int)nt * 8u + t * 2u;        \
                sSp[warp][r0][c0] = sp[nt][0];                                 \
                sSp[warp][r0][c0 + 1] = sp[nt][1];                             \
                sSp[warp][r1][c0] = sp[nt][2];                                 \
                sSp[warp][r1][c0 + 1] = sp[nt][3];                             \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        /* Replicated softmax: reconstruct full S rows, tile-level online. */  \
        {                                                                      \
            float s0[4], s1[4];                                                \
            _Pragma("unroll")                                                  \
            for (int cix = 0; cix < 4; cix++) {                                \
                const unsigned int c =                                         \
                    (cix / 2) * 8u + t * 2u + (unsigned int)(cix % 2);         \
                float a = sSp[0][r0][c] + sSp[1][r0][c] +                      \
                          sSp[2][r0][c] + sSp[3][r0][c];                       \
                float b = sSp[0][r1][c] + sSp[1][r1][c] +                      \
                          sSp[2][r1][c] + sSp[3][r1][c];                       \
                const unsigned int kp = (BASE) + c;                            \
                const bool in0 = v0 && c < (COUNT) && kp >= (LO0) && kp < (HI0);\
                const bool in1 = v1 && c < (COUNT) && kp >= (LO1) && kp < (HI1);\
                s0[cix] = in0 ? (a * inv_sqrt_d) : -INFINITY;                  \
                s1[cix] = in1 ? (b * inv_sqrt_d) : -INFINITY;                  \
            }                                                                  \
            float mx0 = fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s0[2], s0[3]));       \
            float mx1 = fmaxf(fmaxf(s1[0], s1[1]), fmaxf(s1[2], s1[3]));       \
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFFu, mx0, 1));            \
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFFu, mx0, 2));            \
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFFu, mx1, 1));            \
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFFu, mx1, 2));            \
            const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);            \
            const float eo0 = __expf(m0 - mn0), eo1 = __expf(m1 - mn1);        \
            float en0[4], en1[4], se0 = 0.0f, se1 = 0.0f;                      \
            _Pragma("unroll")                                                  \
            for (int cix = 0; cix < 4; cix++) {                                \
                en0[cix] = __expf(s0[cix] - mn0);                              \
                en1[cix] = __expf(s1[cix] - mn1);                              \
                se0 += en0[cix]; se1 += en1[cix];                              \
            }                                                                  \
            se0 += __shfl_xor_sync(0xFFFFFFFFu, se0, 1);                       \
            se0 += __shfl_xor_sync(0xFFFFFFFFu, se0, 2);                       \
            se1 += __shfl_xor_sync(0xFFFFFFFFu, se1, 1);                       \
            se1 += __shfl_xor_sync(0xFFFFFFFFu, se1, 2);                       \
            l0 = l0 * eo0 + se0; m0 = mn0;                                     \
            l1 = l1 * eo1 + se1; m1 = mn1;                                     \
            /* P to smem (warp 0 writes; all warps hold identical values). */  \
            if (warp == 0) {                                                   \
                _Pragma("unroll")                                              \
                for (int cix = 0; cix < 4; cix++) {                            \
                    const unsigned int c =                                     \
                        (cix / 2) * 8u + t * 2u + (unsigned int)(cix % 2);     \
                    sP[r0 * TC_PPAD + c] = __float2bfloat16(en0[cix]);         \
                    sP[r1 * TC_PPAD + c] = __float2bfloat16(en1[cix]);         \
                }                                                              \
            }                                                                  \
            /* Rescale this warp's o_acc by eo (rows r0 / r1). */              \
            _Pragma("unroll")                                                  \
            for (int nt = 0; nt < 16; nt++) {                                  \
                o_acc[nt][0] *= eo0; o_acc[nt][1] *= eo0;                      \
                o_acc[nt][2] *= eo1; o_acc[nt][3] *= eo1;                      \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        /* PV: A = sP [16 x 16] (one k-step), B = sV natural, 16 n-tiles. */   \
        {                                                                      \
            const unsigned short* sPa = (const unsigned short*)sP;             \
            const unsigned short* sBv = (const unsigned short*)sV;             \
            const unsigned int fc0 = t * 2u, fc1 = fc0 + 8u;                   \
            unsigned int pa0 = ((unsigned int)sPa[r0 * TC_PPAD + fc0 + 1] << 16)\
                             | (unsigned int)sPa[r0 * TC_PPAD + fc0];          \
            unsigned int pa1 = ((unsigned int)sPa[r1 * TC_PPAD + fc0 + 1] << 16)\
                             | (unsigned int)sPa[r1 * TC_PPAD + fc0];          \
            unsigned int pa2 = ((unsigned int)sPa[r0 * TC_PPAD + fc1 + 1] << 16)\
                             | (unsigned int)sPa[r0 * TC_PPAD + fc1];          \
            unsigned int pa3 = ((unsigned int)sPa[r1 * TC_PPAD + fc1 + 1] << 16)\
                             | (unsigned int)sPa[r1 * TC_PPAD + fc1];          \
            _Pragma("unroll")                                                  \
            for (int nt = 0; nt < 16; nt++) {                                  \
                const unsigned int nc = warp * 128u + (unsigned int)nt * 8u + g;\
                const unsigned int k0 = t * 2u, k1 = k0 + 8u;                  \
                unsigned int b0 =                                              \
                    ((unsigned int)sBv[(k0 + 1) * TC_VPAD + nc] << 16) |       \
                    (unsigned int)sBv[k0 * TC_VPAD + nc];                      \
                unsigned int b1 =                                              \
                    ((unsigned int)sBv[(k1 + 1) * TC_VPAD + nc] << 16) |       \
                    (unsigned int)sBv[k1 * TC_VPAD + nc];                      \
                asm volatile(                                                  \
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "     \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"   \
                    : "=f"(o_acc[nt][0]), "=f"(o_acc[nt][1]),                  \
                      "=f"(o_acc[nt][2]), "=f"(o_acc[nt][3])                   \
                    : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3),                  \
                      "r"(b0), "r"(b1),                                        \
                      "f"(o_acc[nt][0]), "f"(o_acc[nt][1]),                    \
                      "f"(o_acc[nt][2]), "f"(o_acc[nt][3]));                   \
            }                                                                  \
        }                                                                      \
    } while (0)

    // ── raw arm: block-union sliding-window causal tiles ──
    {
        unsigned int union_lo = 0u;
        if (sliding_window > 0u && q_first + 1u > sliding_window)
            union_lo = q_first + 1u - sliding_window;
        const unsigned int union_hi = q_last_excl;
        for (unsigned int base = union_lo; base < union_hi; base += TC_KT) {
            unsigned int count = union_hi - base;
            if (count > TC_KT) count = TC_KT;
            TC_TILE(K, V, kv_stride, (size_t)kv_head * head_dim, base, count,
                    kv_same, kvs0, kvl0, kvs1, kvl1);
        }
    }

    // ── compressed arm: windowed-causal tiles ──
    {
        unsigned int comp_union = (q_last_excl > 0u) ? (q_last_excl / ratio) : 0u;
        if (comp_union > n_comp) comp_union = n_comp;
        for (unsigned int base = 0; base < comp_union; base += TC_KT) {
            unsigned int count = comp_union - base;
            if (count > TC_KT) count = TC_KT;
            TC_TILE(Kc, Vc, head_dim, (size_t)0u, base, count,
                    comp_same, 0u, cvis0, 0u, cvis1);
        }
    }
    #undef TC_TILE

    // ── sink: per-head logit in the denominator only ──
    if (sinks != nullptr) {
        const float sg = sinks[q_head];
        const float mn0 = fmaxf(m0, sg), mn1 = fmaxf(m1, sg);
        const float eo0 = __expf(m0 - mn0), eo1 = __expf(m1 - mn1);
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            o_acc[nt][0] *= eo0; o_acc[nt][1] *= eo0;
            o_acc[nt][2] *= eo1; o_acc[nt][3] *= eo1;
        }
        l0 = l0 * eo0 + __expf(sg - mn0);
        l1 = l1 * eo1 + __expf(sg - mn1);
        m0 = mn0; m1 = mn1;
    }

    // ── epilogue: normalize and store this warp's 128-dim slice ──
    {
        const float il0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
        const float il1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
        __nv_bfloat16* O0 =
            O + (size_t)qrow0 * q_stride + (size_t)q_head * head_dim;
        __nv_bfloat16* O1 =
            O + (size_t)qrow1 * q_stride + (size_t)q_head * head_dim;
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            const unsigned int c = warp * 128u + (unsigned int)nt * 8u + t * 2u;
            if (v0) {
                __nv_bfloat162 p;
                p.x = __float2bfloat16(o_acc[nt][0] * il0);
                p.y = __float2bfloat16(o_acc[nt][1] * il0);
                *reinterpret_cast<__nv_bfloat162*>(O0 + c) = p;
            }
            if (v1) {
                __nv_bfloat162 p;
                p.x = __float2bfloat16(o_acc[nt][2] * il1);
                p.y = __float2bfloat16(o_acc[nt][3] * il1);
                *reinterpret_cast<__nv_bfloat162*>(O1 + c) = p;
            }
        }
    }
}
