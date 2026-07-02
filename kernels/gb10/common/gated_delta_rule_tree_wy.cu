// SPDX-License-Identifier: AGPL-3.0-only

// Atlas tree-aware WY-fused Gated Delta Rule — M8A v2.
//
// Generalizes gated_delta_rule_wy17.cu to support arbitrary tree topology
// (parent_ids[T] input, -1 = root). For a flat chain (parent_ids[t] = t-1
// with parent_ids[0] = -1), output is BIT-EQUIVALENT to wy17 — same H-root
// reads, same kd_flat block reductions, same scalar WY correction algebra.
// For tree branches (parent_ids[t] != t-1), the WY correction walks the
// ancestor chain for that token instead of assuming linear t-1.
//
// Key invariants:
//   - parent_ids[t] < t  (DAG; enforced host-side)
//   - parent_ids[0] = -1 (the bonus / root)
//   - 1 <= T <= K_MAX (T passed at runtime; K_MAX=32 for SMEM sizing)
//
// PASS 1: read H_root once, compute hk_root[t] = H_root @ k[t] for all t.
// WY:     for each t, walk ancestors via parent_ids, accumulate corrected[t]
//         using the algebra:
//             corrected[t] = (∏ g over ancestors[..]) * hk_root[t]
//                          + Σ over ancestors s of
//                              (∏ g over ancestors more recent than s)
//                              * kd[t][s] * vn[s]
//         vn[t] = (v[t] - g[t] * corrected[t]) * beta[t]
// PASS 2: per-token sequential. For chain (parent[t]=t-1), reuse rolling H
//         registers from previous iter. For branch, re-read from H_inter[parent].
//
// Grid: (num_v_heads, batch, 1)   Block: (128, 1, 1)
// SMEM (static): sk/sq dominate = 2*K_MAX*128*4 B. At K_MAX=32 ≈ 32 KB +
// kd_flat (K*(K-1)/2 floats ≈ 2 KB) + scalars ≈ 34 KB — under the 48 KB
// static-SMEM cap, so no dynamic-SMEM opt-in needed.
//
// K_MAX governs the max DDTree verify width (tree nodes incl. root). It was
// 17 (= drafter γ+1, flat-chain only). Raised to 32 to admit wide DDTree
// branch trees: the CaDDTree throughput-optimal budget on a bandwidth-bound
// device (where the O(T^2) WY correction below is the cost driver) is ~32,
// not the 256 used on HBM datacenter parts. The per-token arrays vi/hk_root/
// vn/qd are indexed by a RUNTIME t, so nvcc already spills them to (L1/L2-
// cached) local memory at K=17 — raising K_MAX only grows that footprint, it
// does not change the spill class, and this kernel is not the verify
// bottleneck (the bandwidth-bound FFN/attn GEMMs are). The compute ORDER is
// unchanged, so the linear-chain fast path stays bit-equivalent to wy17.

#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#define BLOCK_SIZE 128
#define K_MAX 32

extern "C" __global__ void gated_delta_rule_tree_wy(
    float* __restrict__ h_state,                // [batch, vh, k_dim, v_dim] FP32 — RO here
    const __nv_bfloat16* __restrict__ query,    // [batch*T, qk_stride]
    const __nv_bfloat16* __restrict__ key,      // same shape as query
    const __nv_bfloat16* __restrict__ value,    // [batch*T, v_stride]
    const float* __restrict__ gate,             // [batch*T, gb_stride]
    const float* __restrict__ beta,             // [batch*T, gb_stride]
    const int* __restrict__ parent_ids,         // [T] — -1 for root, else < t
    __nv_bfloat16* __restrict__ output,         // [batch*T, num_v_heads, v_dim]
    float* __restrict__ h_state_inter_base,     // [T, batch, vh, hv] stride per inter_stride_floats
    unsigned int inter_stride_floats,
    unsigned int num_tokens,                    // T, 1..K_MAX
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;
    const unsigned int T = num_tokens;

    float* H_root = h_state + ((b * num_v_heads + vh) * hv);
    // h_state_inter slot t = h_state_inter_base + t*inter_stride_floats + (b*nv+vh)*hv
    const unsigned int vh_off = (b * num_v_heads + vh) * hv;

    // ── SMEM ──
    __shared__ float sk[K_MAX][128];
    __shared__ float sq[K_MAX][128];
    __shared__ float sg[K_MAX];
    __shared__ float sbt[K_MAX];
    __shared__ int   sparent[K_MAX];
    __shared__ float kd_flat[K_MAX * (K_MAX - 1) / 2]; // upper triangular: kd[t][s] for s<t at [t*(t-1)/2 + s]
    __shared__ float smem_warp[4];

    // Load q,k into SMEM for all T tokens.
    if (tid < k_dim) {
        for (unsigned int t = 0; t < T; t++) {
            const __nv_bfloat16* qp = query + (b * T + t) * qk_stride + kh * k_dim;
            const __nv_bfloat16* kp = key   + (b * T + t) * qk_stride + kh * k_dim;
            sq[t][tid] = (float)qp[tid];
            sk[t][tid] = (float)kp[tid];
        }
    }
    if (tid < T) {
        // Per-token scalars at offset (b*T + t)*gb_stride + vh for gate/beta.
        float g_raw = gate[(b * T + tid) * gb_stride + vh];
        // BIT-EQUIVALENCE FIX: clamp must MATCH gated_delta_rule_wy17.cu's
        // [0, 1] clamp exactly. The previous [1e-6, 1-1e-6] clamp made the
        // tree kernel diverge from wy17 on LINEAR chains for any gate at the
        // [0,1] boundary, propagating through the entire WY recurrence and
        // collapsing drafter accept. The wider clamp does not improve
        // stability (gate is already a sigmoid output in (0,1)); it only
        // breaks the chain-equivalence invariant.
        sg[tid]  = fminf(fmaxf(g_raw, 0.0f), 1.0f);
        sbt[tid] = beta[(b * T + tid) * gb_stride + vh];
        sparent[tid] = parent_ids[tid];
    }
    __syncthreads();

    // ── Compute kd[t][s] for s<t via block reductions ──
    for (unsigned int t = 1; t < T; t++) {
        for (unsigned int s = 0; s < t; s++) {
            float p = (tid < k_dim) ? sk[t][tid] * sk[s][tid] : 0.0f;
            float r = atlas_block_reduce_sum(p, smem_warp, tid);
            if (tid == 0) kd_flat[t * (t - 1) / 2 + s] = r;
            __syncthreads();
        }
    }

    if (tid < v_dim) {
        // Load v[t] for this thread's v_dim slot.
        float vi[K_MAX];
        for (unsigned int t = 0; t < T; t++) {
            const __nv_bfloat16* vp = value + (b * T + t) * v_stride + vh * v_dim;
            vi[t] = (float)vp[tid];
        }

        // ── PASS 1: one read of H_root, compute hk_root[t] for all t ──
        float hk_root[K_MAX];
        for (unsigned int t = 0; t < T; t++) hk_root[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H_root[(j + 0) * v_dim + tid];
            float h1 = H_root[(j + 1) * v_dim + tid];
            float h2 = H_root[(j + 2) * v_dim + tid];
            float h3 = H_root[(j + 3) * v_dim + tid];
            for (unsigned int t = 0; t < T; t++) {
                hk_root[t] += h0 * sk[t][j + 0] + h1 * sk[t][j + 1]
                            + h2 * sk[t][j + 2] + h3 * sk[t][j + 3];
            }
        }

        // ── WY correction (sequential over T tokens, ancestor-aware) ──
        // For token t with ancestor chain a_0 (oldest=root-side) .. a_{L-1} (closest=parent[t]):
        //   corrected[t] = (g[a_0]*g[a_1]*...*g[a_{L-1}]) * hk_root[t]
        //                + Σ_{i=0..L-1} (g[a_{i+1}]*...*g[a_{L-1}]) * kd[t][a_i] * vn[a_i]
        // where the empty product is 1.
        // For parent_ids[t] = -1 (root): corrected[t] = hk_root[t].
        float vn[K_MAX];
        for (unsigned int t = 0; t < T; t++) {
            int p = sparent[t];
            float corrected;
            if (p < 0) {
                // Root → no chain.
                corrected = hk_root[t];
            } else if (p == (int)t - 1) {
                // ── LINEAR-CHAIN FAST PATH (bit-equivalent to wy17) ──
                // When parent[t] == t-1 the ancestor chain is the contiguous
                // run [0, 1, ..., t-1]. Reproduce gated_delta_rule_wy17.cu's
                // EXACT accumulation order so the result is bit-identical:
                //   - leading product accumulated ascending u = 0..t-1
                //   - cross terms summed ascending s = 0..t-1, each with a
                //     FRESH nested product gprod = ∏_{u=s+1..t-1} sg[u]
                // (The general ancestor-walk below visits s descending with a
                // running gprod — same math, different FP rounding. Matching
                // wy17 order is what makes the chain path bit-exact.)
                float lead_prod = 1.0f;
                for (int u = 0; u < (int)t; u++) lead_prod *= sg[u];
                corrected = lead_prod * hk_root[t];
                for (int s = 0; s < (int)t; s++) {
                    float gprod = 1.0f;
                    for (int u = s + 1; u < (int)t; u++) gprod *= sg[u];
                    corrected += gprod * kd_flat[t * (t - 1) / 2 + s] * vn[s];
                }
            } else {
                // ── GENERAL BRANCH PATH (ancestor walk) ──
                // Walk ancestors from parent back to root, oldest-first list.
                int chain[K_MAX]; int L = 0;
                int cur = p;
                while (cur >= 0 && L < (int)K_MAX) {
                    chain[L++] = cur;
                    cur = sparent[cur];
                }
                // chain[0] = parent (closest), chain[L-1] = root-child (oldest in chain).
                // Reverse semantics: ancestor a_i for i=0..L-1 with a_0=oldest=chain[L-1], a_{L-1}=closest=chain[0].
                // Leading term: ∏_{i=0..L-1} g[a_i] = ∏ g[chain[k]] for all k.
                // Accumulate oldest→closest (ascending real index) so the
                // leading product matches the wy17 ordering convention.
                float lead = 1.0f;
                for (int k = L - 1; k >= 0; k--) lead *= sg[chain[k]];
                corrected = lead * hk_root[t];
                // Cross terms: Σ_{i=0..L-1} (∏_{j=i+1..L-1} g[a_j]) * kd[t][a_i] * vn[a_i]
                // a_i = chain[L-1-i]. For ancestor at chain[ci] (ci in 0..L-1):
                //   gprod = ∏ g[chain[k]] for k in 0..ci-1   (empty product = 1)
                //   cross += gprod * kd[t][chain[ci]] * vn[chain[ci]]
                // Sum ancestors oldest→closest (ci = L-1 down to 0) so that,
                // on a contiguous chain, the per-term gprod is the same fresh
                // ∏_{u=s+1..t-1} sg[u] wy17 uses. gprod is rebuilt from
                // scratch per ancestor to avoid running-product FP drift.
                for (int ci = L - 1; ci >= 0; ci--) {
                    int s = chain[ci];
                    float kd_ts = kd_flat[t * (t - 1) / 2 + s];
                    // gprod = ∏ g[chain[k]] for k = 0 .. ci-1 (closer-than-s ancestors)
                    float gprod = 1.0f;
                    for (int k = 0; k < ci; k++) gprod *= sg[chain[k]];
                    corrected += gprod * kd_ts * vn[s];
                }
            }
            vn[t] = (vi[t] - sg[t] * corrected) * sbt[t];
        }

        // ── PASS 2: per-token sequential, write H_inter[t], compute qd[t] ──
        // For each token, read H from parent slot (root or h_state_inter[parent]),
        // compute updated state, write to h_state_inter[t], dot with q[t] for output.
        float qd[K_MAX];
        for (unsigned int t = 0; t < T; t++) qd[t] = 0.0f;

        for (unsigned int t = 0; t < T; t++) {
            int p = sparent[t];
            float* H_in;
            if (p < 0) {
                H_in = H_root;
            } else {
                H_in = h_state_inter_base + ((unsigned int)p) * inter_stride_floats + vh_off;
            }
            float* H_out = h_state_inter_base + t * inter_stride_floats + vh_off;
            #pragma unroll 4
            for (unsigned int j = 0; j < k_dim; j += 4) {
                float h0 = H_in[(j + 0) * v_dim + tid];
                float h1 = H_in[(j + 1) * v_dim + tid];
                float h2 = H_in[(j + 2) * v_dim + tid];
                float h3 = H_in[(j + 3) * v_dim + tid];
                h0 = sg[t] * h0 + sk[t][j + 0] * vn[t];
                h1 = sg[t] * h1 + sk[t][j + 1] * vn[t];
                h2 = sg[t] * h2 + sk[t][j + 2] * vn[t];
                h3 = sg[t] * h3 + sk[t][j + 3] * vn[t];
                H_out[(j + 0) * v_dim + tid] = h0;
                H_out[(j + 1) * v_dim + tid] = h1;
                H_out[(j + 2) * v_dim + tid] = h2;
                H_out[(j + 3) * v_dim + tid] = h3;
                qd[t] += h0 * sq[t][j + 0] + h1 * sq[t][j + 1]
                       + h2 * sq[t][j + 2] + h3 * sq[t][j + 3];
            }
        }

        // ── Write outputs ──
        float scale = rsqrtf((float)k_dim);
        for (unsigned int t = 0; t < T; t++) {
            output[((b * T + t) * num_v_heads + vh) * v_dim + tid] =
                __float2bfloat16(qd[t] * scale);
        }
    }
}
