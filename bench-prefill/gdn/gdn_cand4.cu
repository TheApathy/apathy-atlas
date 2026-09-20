// SPDX-License-Identifier: AGPL-3.0-only
//
// GDN candidate R4: R3 (K-split, 512 threads, register-resident state) plus a
// matmul-style Gram phase replacing 496 warp-shuffle reductions.
//
// The parent kernel launches 48 CTAs of 128 threads (one thread per V column)
// for 48 V heads, which is exactly one CTA and four warps per SM on a 48-SM
// part -- about 6% of the 64-warp capacity. It is latency-bound on dependent
// FP32 chains, not bandwidth-bound, so the only way to go faster is more warps.
//
// The one unexploited dimension is K (128 wide). This kernel uses 512 threads
// arranged as 4 j-groups x 128 V columns: thread (jg, v) owns state rows
// j = jg*32 .. jg*32+31 of column v. That is 32 floats per thread, so the whole
// 128x128 FP32 state fits in registers with no spilling (unlike R1, where one
// thread held all 128 rows and spilled).
//
// Exactness: the two dot products over j (hk_prev and o_out) are no longer a
// single sequential FP32 sum over j = 0..127; each group sums its own 32 terms
// and the four partials are combined in ascending group order. This is a
// reassociation of an FP32 sum, so the kernel is NOT bit-exact and must be
// gated. Everything else -- the K-dot tree, the gate prefix scans, the WY
// correction, the state recurrence, the BF16 rounding -- is unchanged.

#define C4_K_DIM 128
#define C4_V_DIM 128
#define C4_C 32
#define C4_WARPS 4
#define C4_PAIRS 496
#define C4_JPT 32  // state rows per thread
#define C4_KS 130 // padded K/Q row stride (128 + 2) -> conflict-free Gram reads

static_assert(C4_PAIRS == C4_C * (C4_C - 1) / 2, "WY32 strict-lower pair count drift");
static_assert(C4_JPT * 4 == C4_K_DIM, "j-group split must tile K");

extern "C" __global__ void __launch_bounds__(512, 1)
gdn_candidate4(
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

    const unsigned int tid = threadIdx.x;              // 0..511
    const unsigned int v = tid & (C4_V_DIM - 1);       // V column
    const unsigned int jg = tid >> 7;                  // j group 0..3
    const unsigned int j0 = jg * C4_JPT;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const float inv_sqrt_d = rsqrtf((float)k_dim);

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_k = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* smem_q = smem_k + C4_C * C4_KS;
    float* smem_kd = (float*)(smem_q + C4_C * C4_KS);
    float* smem_g = smem_kd + C4_C * C4_C;
    float* smem_bt = smem_g + C4_C;
    unsigned short* smem_pair_ij = (unsigned short*)(smem_bt + C4_C);
    // Cross-group reduction scratch: groups 1..3 deposit their partials here and
    // group 0 folds them in ascending order. Reused for hk_prev and for o_out.
    float* smem_red = (float*)(((unsigned long long)(smem_pair_ij + C4_PAIRS) + 15ull) & ~15ull);
    float* smem_vnew = smem_red + 3 * C4_C * C4_V_DIM;

    float* H_global = h_state
        + ((unsigned long long)(b * num_v_heads + vh) * C4_K_DIM * C4_V_DIM);

    float Hreg[C4_JPT];
    #pragma unroll
    for (int jj = 0; jj < C4_JPT; jj++)
        Hreg[jj] = H_global[(unsigned long long)(j0 + jj) * C4_V_DIM + v];

    for (unsigned int pair = tid; pair < C4_PAIRS; pair += 512) {
        unsigned int i = 1, row_start = 0;
        while (pair >= row_start + i) { row_start += i; i++; }
        smem_pair_ij[pair] = (unsigned short)((i << 8) | (pair - row_start));
    }
    __syncthreads();

    const unsigned int wy_end = (seq_len / C4_C) * C4_C;
    for (unsigned int chunk_start = 0; chunk_start < wy_end; chunk_start += C4_C) {
        for (unsigned int idx = tid; idx < C4_C * C4_K_DIM; idx += 512) {
            const unsigned int tok = idx / C4_K_DIM;
            const unsigned int dim = idx % C4_K_DIM;
            const unsigned long long off =
                (unsigned long long)(chunk_start + tok) * qk_stride + kh * k_dim + dim;
            smem_k[tok * C4_KS + dim] = key[off];
            smem_q[tok * C4_KS + dim] = query[off];
        }
        if (tid < C4_C) {
            smem_g[tid] = gate[(unsigned long long)(chunk_start + tid) * gb_stride + vh];
            smem_bt[tid] = beta[(unsigned long long)(chunk_start + tid) * gb_stride + vh];
        }
        // Unlike the parent, a thread no longer writes only its own K column,
        // so K must be published before the dot phase reads it.
        __syncthreads();

        // K-dot phase: the parent computes all 496 strict-lower Gram entries with
        // 496 warp-shuffle reductions per thread (~2,480 shuffle ops), which
        // dominates this kernel. With 512 threads there is at most one pair per
        // thread, so each thread instead walks the 128-element dot product in
        // registers: 128 FMAs and no shuffles, no cross-thread rendezvous and no
        // partial buffer. Rows are padded to C4_KS so the 32 distinct i values
        // land on 32 distinct banks.
        if (tid < C4_PAIRS) {
            const unsigned int packed = smem_pair_ij[tid];
            const unsigned int i = packed >> 8, j = packed & 0xFF;
            const __nv_bfloat16* ki = smem_k + i * C4_KS;
            const __nv_bfloat16* kj = smem_k + j * C4_KS;
            float acc = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < C4_K_DIM; d++) acc += (float)ki[d] * (float)kj[d];
            smem_kd[i * C4_C + j] = acc;
        }
        if (tid == 0) {
            float g_prefix = 1.0f;
            for (unsigned int t = 0; t < C4_C; t++) {
                smem_kd[t * C4_C + t] = g_prefix;
                if (t + 1 < C4_C) g_prefix *= smem_g[t];
            }
        }
        if (tid < C4_C) {
            float g_between = 1.0f;
            for (unsigned int t = tid + 1; t < C4_C; t++) {
                smem_kd[tid * C4_C + t] = g_between;
                if (t + 1 < C4_C) g_between *= smem_g[t];
            }
        }
        __syncthreads();

        // Phase 1: each group accumulates H^T k over its own 32 state rows.
        float part[C4_C];
        #pragma unroll
        for (int t = 0; t < C4_C; t++) part[t] = 0.0f;
        #pragma unroll
        for (int jj = 0; jj < C4_JPT; jj++) {
            const float h_j = Hreg[jj];
            const unsigned int j = j0 + jj;
            #pragma unroll
            for (int t = 0; t < C4_C; t++)
                part[t] += h_j * (float)smem_k[t * C4_KS + j];
        }
        if (jg > 0) {
            #pragma unroll
            for (int t = 0; t < C4_C; t++)
                smem_red[(jg - 1) * C4_C * C4_V_DIM + t * C4_V_DIM + v] = part[t];
        }
        __syncthreads();

        // Phase 2: group 0 folds the partials and runs the WY correction, which
        // is sequential in t and identical to the parent's.
        if (jg == 0) {
            float v_new[C4_C];
            #pragma unroll
            for (int t = 0; t < C4_C; t++) {
                float hk = part[t];
                hk += smem_red[0 * C4_C * C4_V_DIM + t * C4_V_DIM + v];
                hk += smem_red[1 * C4_C * C4_V_DIM + t * C4_V_DIM + v];
                hk += smem_red[2 * C4_C * C4_V_DIM + t * C4_V_DIM + v];
                const float v_t = (float)value[
                    (unsigned long long)(chunk_start + t) * v_stride + vh * v_dim + v];
                float hk_corr = smem_kd[t * C4_C + t] * hk;
                #pragma unroll
                for (int s = 0; s < t; s++)
                    hk_corr += smem_kd[s * C4_C + t] * smem_kd[t * C4_C + s] * v_new[s];
                v_new[t] = (v_t - smem_g[t] * hk_corr) * smem_bt[t];
                smem_vnew[t * C4_V_DIM + v] = v_new[t];
            }
        }
        __syncthreads();

        // Phase 3: every group advances its own state rows and accumulates its
        // share of the output dot product.
        #pragma unroll
        for (int t = 0; t < C4_C; t++) part[t] = 0.0f;
        #pragma unroll
        for (int jj = 0; jj < C4_JPT; jj++) {
            float h_j = Hreg[jj];
            const unsigned int j = j0 + jj;
            #pragma unroll
            for (int t = 0; t < C4_C; t++) {
                h_j = smem_g[t] * h_j
                    + (float)smem_k[t * C4_KS + j] * smem_vnew[t * C4_V_DIM + v];
                part[t] += h_j * (float)smem_q[t * C4_KS + j];
            }
            Hreg[jj] = h_j;
        }
        if (jg > 0) {
            #pragma unroll
            for (int t = 0; t < C4_C; t++)
                smem_red[(jg - 1) * C4_C * C4_V_DIM + t * C4_V_DIM + v] = part[t];
        }
        __syncthreads();

        if (jg == 0) {
            #pragma unroll
            for (int t = 0; t < C4_C; t++) {
                float o = part[t];
                o += smem_red[0 * C4_C * C4_V_DIM + t * C4_V_DIM + v];
                o += smem_red[1 * C4_C * C4_V_DIM + t * C4_V_DIM + v];
                o += smem_red[2 * C4_C * C4_V_DIM + t * C4_V_DIM + v];
                output[(unsigned long long)(chunk_start + t) * num_v_heads * v_dim
                       + vh * v_dim + v] = __float2bfloat16(o * inv_sqrt_d);
            }
        }
        __syncthreads();
    }

    // Remainder tokens, same group split and same cross-group fold.
    for (unsigned int t = wy_end; t < seq_len; t++) {
        const unsigned long long qk_off =
            (unsigned long long)t * qk_stride + kh * k_dim;
        if (tid < C4_K_DIM) {
            smem_k[tid] = key[qk_off + tid];
            smem_q[tid] = query[qk_off + tid];
        }
        __syncthreads();

        const float g_t = gate[(unsigned long long)t * gb_stride + vh];
        const float bt_t = beta[(unsigned long long)t * gb_stride + vh];

        float hk = 0.0f;
        #pragma unroll
        for (int jj = 0; jj < C4_JPT; jj++)
            hk += Hreg[jj] * (float)smem_k[j0 + jj];
        if (jg > 0) smem_red[(jg - 1) * C4_V_DIM + v] = hk;
        __syncthreads();
        if (jg == 0) {
            float hk_full = hk + smem_red[0 * C4_V_DIM + v]
                + smem_red[1 * C4_V_DIM + v] + smem_red[2 * C4_V_DIM + v];
            const float v_i = (float)value[
                (unsigned long long)t * v_stride + vh * v_dim + v];
            smem_vnew[v] = (v_i - g_t * hk_full) * bt_t;
        }
        __syncthreads();

        const float vn = smem_vnew[v];
        float q_dot = 0.0f;
        #pragma unroll
        for (int jj = 0; jj < C4_JPT; jj++) {
            const float h = g_t * Hreg[jj] + (float)smem_k[j0 + jj] * vn;
            Hreg[jj] = h;
            q_dot += h * (float)smem_q[j0 + jj];
        }
        if (jg > 0) smem_red[(jg - 1) * C4_V_DIM + v] = q_dot;
        __syncthreads();
        if (jg == 0) {
            float o = q_dot + smem_red[0 * C4_V_DIM + v]
                + smem_red[1 * C4_V_DIM + v] + smem_red[2 * C4_V_DIM + v];
            output[(unsigned long long)t * num_v_heads * v_dim + vh * v_dim + v] =
                __float2bfloat16(o * inv_sqrt_d);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int jj = 0; jj < C4_JPT; jj++)
        H_global[(unsigned long long)(j0 + jj) * C4_V_DIM + v] = Hreg[jj];
}
