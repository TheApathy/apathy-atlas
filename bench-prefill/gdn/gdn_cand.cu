// SPDX-License-Identifier: AGPL-3.0-only
// GDN candidate R1: register-resident state. Bit-exact by construction --
// every arithmetic statement, operand order and accumulation order is
// unchanged from gdn_ref.cu. Only the storage location of the 128x128 FP32
// state changes, so shared memory drops 95,216 -> 29,680 bytes.
#define KT_STRIDE 36
// SPDX-License-Identifier: AGPL-3.0-only

// Exact gate-product-cached shadow of gated_delta_rule_prefill_wy64.
//
// The parent WY32 kernel computes every gate prefix/product independently in
// all 128 threads even though those coefficients are identical across V
// columns. This shadow computes each coefficient once, in the same
// left-to-right FP32 multiplication order, and stores it in the unused
// diagonal/upper triangle of smem_kd. The lower triangle continues to hold
// k_i^T k_j. H, K, Q, WY correction, state update, output accumulation, and
// BF16 conversion remain statement-for-statement equivalent to the parent.


__device__ __forceinline__ float cand_warp_reduce(float val) {
    for (int offset = 16; offset >= 1; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__ void __launch_bounds__(128, 1)
gdn_candidate(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int seq_len,
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
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const float inv_sqrt_d = rsqrtf((float)k_dim);

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_k = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* smem_q = smem_k + C * K_DIM;
    float* smem_warp = (float*)(smem_q + C * K_DIM);
    float* smem_kd = smem_warp + 4;
    float* smem_g = smem_kd + C * C;
    float* smem_bt = smem_g + C;
    // Four parent-order warp partials for every strict-lower K-dot pair.
    // The host reserves KD_PAIR_COUNT * WARP_COUNT FP32 values after smem_bt.
    float* smem_dot_partials = smem_bt + C;
    // One invariant packed (i,j) entry per strict-lower pair. Building this
    // once avoids making every thread rescan all 496 triangular coordinates
    // after each chunk's partial-dot barrier.
    unsigned short* smem_pair_ij = (unsigned short*)
        (smem_dot_partials + KD_PAIR_COUNT * WARP_COUNT);
#ifdef GDN_CARRAYS_SMEM
    // R2: park the three per-chunk C-length per-thread arrays in the shared
    // memory freed by moving H to registers, so Hreg does not have to share
    // the 255-register budget with them. Indexed [t][tid]: thread-private,
    // no cross-thread access, so arithmetic and ordering are untouched.
    float* smem_carr = (float*)(((unsigned long long)(smem_pair_ij + KD_PAIR_COUNT) + 15ull) & ~15ull);
    float* hk_prev   = smem_carr + 0 * C * V_DIM + tid;
    float* v_new_arr = smem_carr + 1 * C * V_DIM + tid;
    float* o_out     = smem_carr + 2 * C * V_DIM + tid;
    #define CIDX(t) [(t) * V_DIM]
#else
    #define CIDX(t) [(t)]
#endif

    float* H_global = h_state
        + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    // Thread `tid` is the sole reader and writer of state column `tid`
    // (every parent access is at index j * V_DIM + tid), so the whole
    // 128x128 FP32 state moves into a per-thread register array.
    float Hreg[K_DIM];
    #pragma unroll
    for (int j = 0; j < K_DIM; j++) Hreg[j] = H_global[j * V_DIM + tid];
    for (unsigned int pair = tid; pair < KD_PAIR_COUNT; pair += V_DIM) {
        unsigned int i = 1;
        unsigned int row_start = 0;
        while (pair >= row_start + i) {
            row_start += i;
            i++;
        }
        const unsigned int j = pair - row_start;
        smem_pair_ij[pair] = (unsigned short)((i << 8) | j);
    }
    __syncthreads();

    const unsigned int wy_end = (seq_len / C) * C;
    for (unsigned int chunk_start = 0; chunk_start < wy_end; chunk_start += C) {
        for (unsigned int idx = tid; idx < C * K_DIM; idx += V_DIM) {
            const unsigned int tok = idx / K_DIM;
            const unsigned int dim = idx % K_DIM;
            const unsigned long long off =
                (unsigned long long)(chunk_start + tok) * qk_stride + kh * k_dim + dim;
            smem_k[tok * K_DIM + dim] = key[off];
            smem_q[tok * K_DIM + dim] = query[off];
        }
        if (tid < C) {
            smem_g[tid] = gate[(unsigned long long)(chunk_start + tid) * gb_stride + vh];
            smem_bt[tid] = beta[(unsigned long long)(chunk_start + tid) * gb_stride + vh];
        }

        // Compute every strict-lower K-dot pair without a CTA rendezvous per
        // pair or before the dot phase. Each lane reads only the 32 K values
        // it just wrote at its own dimension; no cross-thread K visibility is
        // needed until the warp-local products have reached registers. Each
        // warp retains the parent's exact shuffle tree, and lane zero stores
        // all four parent-order partials into disjoint locations.
        const unsigned int warp_id = tid / 32;
        const unsigned int lane_id = tid % 32;
        int pair_index = 0;
        for (int i = 1; i < C; i++) {
            for (int j = 0; j < i; j++) {
                float partial = (float)smem_k[i * K_DIM + tid]
                    * (float)smem_k[j * K_DIM + tid];
                partial = cand_warp_reduce(partial);
                if (lane_id == 0) {
                    smem_dot_partials[pair_index * WARP_COUNT + warp_id] = partial;
                }
                pair_index++;
            }
        }
        // Publishes Q/K/gate/beta as well as all dot partials. The input loads
        // and dot products need only thread-local program order before here.
        __syncthreads();

        // Preserve the parent's cross-warp association exactly:
        // ((warp0 + warp1) + warp2) + warp3. Distributed CTA threads only
        // parallelize independent pairs; they do not alter any dot reduction.
        // The one-time packed map gives each thread only 3--4 assigned pairs.
        for (unsigned int pair = tid; pair < KD_PAIR_COUNT; pair += V_DIM) {
            const unsigned int packed = smem_pair_ij[pair];
            const unsigned int i = packed >> 8;
            const unsigned int j = packed & 0xFF;
            const float* partial = &smem_dot_partials[pair * WARP_COUNT];
            smem_kd[i * C + j] =
                partial[0] + partial[1] + partial[2] + partial[3];
        }

        // Cache the exact products used below with rolling left-to-right
        // scans. Diagonal [t,t] stores product(g[0..t)); upper [s,t]
        // stores product(g[s+1..t)). Carrying each prefix forward retains
        // the parent's initialization and multiplication order while avoiding
        // overlapping from-scratch recomputation.
        if (tid == 0) {
            float g_prefix = 1.0f;
            for (unsigned int t = 0; t < C; t++) {
                smem_kd[t * C + t] = g_prefix;
                if (t + 1 < C) g_prefix *= smem_g[t];
            }
        }
        if (tid < C) {
            float g_between = 1.0f;
            for (unsigned int t = tid + 1; t < C; t++) {
                smem_kd[tid * C + t] = g_between;
                if (t + 1 < C) g_between *= smem_g[t];
            }
        }
        // One visibility barrier publishes both final K-dots and cached gate
        // coefficients. Together with the partial barrier above, this replaces
        // the parent's three full-CTA barriers for each of 496 pairs.
        __syncthreads();

#ifndef GDN_CARRAYS_SMEM
        float hk_prev[C];
#endif
        for (int t = 0; t < C; t++) hk_prev CIDX(t) = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j++) {
            const float h_j = Hreg[j];
            for (int t = 0; t < C; t++) {
                hk_prev CIDX(t) += h_j * (float)smem_k[t * K_DIM + j];
            }
        }

#ifndef GDN_CARRAYS_SMEM
        float v_new_arr[C];
#endif
        for (int t = 0; t < C; t++) {
            const float v_t = (float)value[
                (unsigned long long)(chunk_start + t) * v_stride + vh * v_dim + tid
            ];
            float hk_corr = smem_kd[t * C + t] * hk_prev CIDX(t);
            for (int s = 0; s < t; s++) {
                hk_corr += smem_kd[s * C + t] * smem_kd[t * C + s] * v_new_arr CIDX(s);
            }
            v_new_arr CIDX(t) = (v_t - smem_g[t] * hk_corr) * smem_bt[t];
        }

#ifndef GDN_CARRAYS_SMEM
        float o_out[C];
#endif
        for (int t = 0; t < C; t++) o_out CIDX(t) = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j++) {
            float h_j = Hreg[j];
            for (int t = 0; t < C; t++) {
                h_j = smem_g[t] * h_j + (float)smem_k[t * K_DIM + j] * v_new_arr CIDX(t);
                o_out CIDX(t) += h_j * (float)smem_q[t * K_DIM + j];
            }
            Hreg[j] = h_j;
        }

        for (int t = 0; t < C; t++) {
            const unsigned long long out_off =
                (unsigned long long)(chunk_start + t) * num_v_heads * v_dim
                + vh * v_dim + tid;
            output[out_off] = __float2bfloat16(o_out CIDX(t) * inv_sqrt_d);
        }
        __syncthreads();
    }

    // Remainder path is identical to the parent and does not use WY products.
    for (unsigned int t = wy_end; t < seq_len; t++) {
        if (tid < K_DIM) {
            const unsigned long long qk_off =
                (unsigned long long)t * qk_stride + kh * k_dim;
            smem_k[tid] = key[qk_off + tid];
            smem_q[tid] = query[qk_off + tid];
        }
        __syncthreads();

        const float v_i = (float)value[
            (unsigned long long)t * v_stride + vh * v_dim + tid
        ];
        const float g_t = gate[(unsigned long long)t * gb_stride + vh];
        const float bt_t = beta[(unsigned long long)t * gb_stride + vh];

        float hk = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            hk += Hreg[j+0]*(float)smem_k[j]
                + Hreg[j+1]*(float)smem_k[j+1]
                + Hreg[j+2]*(float)smem_k[j+2]
                + Hreg[j+3]*(float)smem_k[j+3];
        }
        const float vn = (v_i - g_t * hk) * bt_t;

        float q_dot = 0.0f;
        #pragma unroll
        for (int j = 0; j < K_DIM; j += 4) {
            const float h0 = g_t*Hreg[j+0] + (float)smem_k[j]*vn;
            const float h1 = g_t*Hreg[j+1] + (float)smem_k[j+1]*vn;
            const float h2 = g_t*Hreg[j+2] + (float)smem_k[j+2]*vn;
            const float h3 = g_t*Hreg[j+3] + (float)smem_k[j+3]*vn;
            Hreg[j+0]=h0; Hreg[j+1]=h1;
            Hreg[j+2]=h2; Hreg[j+3]=h3;
            q_dot += h0*(float)smem_q[j] + h1*(float)smem_q[j+1]
                + h2*(float)smem_q[j+2] + h3*(float)smem_q[j+3];
        }
        const unsigned long long out_off =
            (unsigned long long)t * num_v_heads * v_dim + vh * v_dim + tid;
        output[out_off] = __float2bfloat16(q_dot * inv_sqrt_d);
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < K_DIM; j++) H_global[j * V_DIM + tid] = Hreg[j];
}
