// SPDX-License-Identifier: AGPL-3.0-only
//
// GDN candidate R3: K-split, 512 threads, register-resident state.
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

#define C3_K_DIM 128
#define C3_V_DIM 128
#define C3_C 32
#define C3_WARPS 4
#define C3_PAIRS 496
#define C3_JPT 32  // state rows per thread

static_assert(C3_PAIRS == C3_C * (C3_C - 1) / 2, "WY32 strict-lower pair count drift");
static_assert(C3_JPT * 4 == C3_K_DIM, "j-group split must tile K");

__device__ __forceinline__ float cand3_warp_reduce(float val) {
    for (int offset = 16; offset >= 1; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__ void __launch_bounds__(512, 1)
gdn_candidate3(
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
    const unsigned int v = tid & (C3_V_DIM - 1);       // V column
    const unsigned int jg = tid >> 7;                  // j group 0..3
    const unsigned int j0 = jg * C3_JPT;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const float inv_sqrt_d = rsqrtf((float)k_dim);

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_k = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* smem_q = smem_k + C3_C * C3_K_DIM;
    float* smem_kd = (float*)(smem_q + C3_C * C3_K_DIM);
    float* smem_g = smem_kd + C3_C * C3_C;
    float* smem_bt = smem_g + C3_C;
    float* smem_dot_partials = smem_bt + C3_C;
    unsigned short* smem_pair_ij =
        (unsigned short*)(smem_dot_partials + C3_PAIRS * C3_WARPS);
    // Cross-group reduction scratch: groups 1..3 deposit their partials here and
    // group 0 folds them in ascending order. Reused for hk_prev and for o_out.
    float* smem_red = (float*)(((unsigned long long)(smem_pair_ij + C3_PAIRS) + 15ull) & ~15ull);
    float* smem_vnew = smem_red + 3 * C3_C * C3_V_DIM;

    float* H_global = h_state
        + ((unsigned long long)(b * num_v_heads + vh) * C3_K_DIM * C3_V_DIM);

    float Hreg[C3_JPT];
    #pragma unroll
    for (int jj = 0; jj < C3_JPT; jj++)
        Hreg[jj] = H_global[(unsigned long long)(j0 + jj) * C3_V_DIM + v];

    for (unsigned int pair = tid; pair < C3_PAIRS; pair += 512) {
        unsigned int i = 1, row_start = 0;
        while (pair >= row_start + i) { row_start += i; i++; }
        smem_pair_ij[pair] = (unsigned short)((i << 8) | (pair - row_start));
    }
    __syncthreads();

    const unsigned int wy_end = (seq_len / C3_C) * C3_C;
    for (unsigned int chunk_start = 0; chunk_start < wy_end; chunk_start += C3_C) {
        for (unsigned int idx = tid; idx < C3_C * C3_K_DIM; idx += 512) {
            const unsigned int tok = idx / C3_K_DIM;
            const unsigned int dim = idx % C3_K_DIM;
            const unsigned long long off =
                (unsigned long long)(chunk_start + tok) * qk_stride + kh * k_dim + dim;
            smem_k[tok * C3_K_DIM + dim] = key[off];
            smem_q[tok * C3_K_DIM + dim] = query[off];
        }
        if (tid < C3_C) {
            smem_g[tid] = gate[(unsigned long long)(chunk_start + tid) * gb_stride + vh];
            smem_bt[tid] = beta[(unsigned long long)(chunk_start + tid) * gb_stride + vh];
        }
        // Unlike the parent, a thread no longer writes only its own K column,
        // so K must be published before the dot phase reads it.
        __syncthreads();

        // K-dot phase: the parent's exact shuffle tree and warp association,
        // executed by the first 128 threads so the reduction order is preserved.
        if (tid < 128) {
            const unsigned int warp_id = tid / 32;
            const unsigned int lane_id = tid % 32;
            int pair_index = 0;
            for (int i = 1; i < C3_C; i++) {
                for (int j = 0; j < i; j++) {
                    float partial = (float)smem_k[i * C3_K_DIM + tid]
                        * (float)smem_k[j * C3_K_DIM + tid];
                    partial = cand3_warp_reduce(partial);
                    if (lane_id == 0)
                        smem_dot_partials[pair_index * C3_WARPS + warp_id] = partial;
                    pair_index++;
                }
            }
        }
        __syncthreads();

        for (unsigned int pair = tid; pair < C3_PAIRS; pair += 512) {
            const unsigned int packed = smem_pair_ij[pair];
            const unsigned int i = packed >> 8, j = packed & 0xFF;
            const float* p = &smem_dot_partials[pair * C3_WARPS];
            smem_kd[i * C3_C + j] = p[0] + p[1] + p[2] + p[3];
        }
        if (tid == 0) {
            float g_prefix = 1.0f;
            for (unsigned int t = 0; t < C3_C; t++) {
                smem_kd[t * C3_C + t] = g_prefix;
                if (t + 1 < C3_C) g_prefix *= smem_g[t];
            }
        }
        if (tid < C3_C) {
            float g_between = 1.0f;
            for (unsigned int t = tid + 1; t < C3_C; t++) {
                smem_kd[tid * C3_C + t] = g_between;
                if (t + 1 < C3_C) g_between *= smem_g[t];
            }
        }
        __syncthreads();

        // Phase 1: each group accumulates H^T k over its own 32 state rows.
        float part[C3_C];
        #pragma unroll
        for (int t = 0; t < C3_C; t++) part[t] = 0.0f;
        #pragma unroll
        for (int jj = 0; jj < C3_JPT; jj++) {
            const float h_j = Hreg[jj];
            const unsigned int j = j0 + jj;
            #pragma unroll
            for (int t = 0; t < C3_C; t++)
                part[t] += h_j * (float)smem_k[t * C3_K_DIM + j];
        }
        if (jg > 0) {
            #pragma unroll
            for (int t = 0; t < C3_C; t++)
                smem_red[(jg - 1) * C3_C * C3_V_DIM + t * C3_V_DIM + v] = part[t];
        }
        __syncthreads();

        // Phase 2: group 0 folds the partials and runs the WY correction, which
        // is sequential in t and identical to the parent's.
        if (jg == 0) {
            float v_new[C3_C];
            #pragma unroll
            for (int t = 0; t < C3_C; t++) {
                float hk = part[t];
                hk += smem_red[0 * C3_C * C3_V_DIM + t * C3_V_DIM + v];
                hk += smem_red[1 * C3_C * C3_V_DIM + t * C3_V_DIM + v];
                hk += smem_red[2 * C3_C * C3_V_DIM + t * C3_V_DIM + v];
                const float v_t = (float)value[
                    (unsigned long long)(chunk_start + t) * v_stride + vh * v_dim + v];
                float hk_corr = smem_kd[t * C3_C + t] * hk;
                #pragma unroll
                for (int s = 0; s < t; s++)
                    hk_corr += smem_kd[s * C3_C + t] * smem_kd[t * C3_C + s] * v_new[s];
                v_new[t] = (v_t - smem_g[t] * hk_corr) * smem_bt[t];
                smem_vnew[t * C3_V_DIM + v] = v_new[t];
            }
        }
        __syncthreads();

        // Phase 3: every group advances its own state rows and accumulates its
        // share of the output dot product.
        #pragma unroll
        for (int t = 0; t < C3_C; t++) part[t] = 0.0f;
        #pragma unroll
        for (int jj = 0; jj < C3_JPT; jj++) {
            float h_j = Hreg[jj];
            const unsigned int j = j0 + jj;
            #pragma unroll
            for (int t = 0; t < C3_C; t++) {
                h_j = smem_g[t] * h_j
                    + (float)smem_k[t * C3_K_DIM + j] * smem_vnew[t * C3_V_DIM + v];
                part[t] += h_j * (float)smem_q[t * C3_K_DIM + j];
            }
            Hreg[jj] = h_j;
        }
        if (jg > 0) {
            #pragma unroll
            for (int t = 0; t < C3_C; t++)
                smem_red[(jg - 1) * C3_C * C3_V_DIM + t * C3_V_DIM + v] = part[t];
        }
        __syncthreads();

        if (jg == 0) {
            #pragma unroll
            for (int t = 0; t < C3_C; t++) {
                float o = part[t];
                o += smem_red[0 * C3_C * C3_V_DIM + t * C3_V_DIM + v];
                o += smem_red[1 * C3_C * C3_V_DIM + t * C3_V_DIM + v];
                o += smem_red[2 * C3_C * C3_V_DIM + t * C3_V_DIM + v];
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
        if (tid < C3_K_DIM) {
            smem_k[tid] = key[qk_off + tid];
            smem_q[tid] = query[qk_off + tid];
        }
        __syncthreads();

        const float g_t = gate[(unsigned long long)t * gb_stride + vh];
        const float bt_t = beta[(unsigned long long)t * gb_stride + vh];

        float hk = 0.0f;
        #pragma unroll
        for (int jj = 0; jj < C3_JPT; jj++)
            hk += Hreg[jj] * (float)smem_k[j0 + jj];
        if (jg > 0) smem_red[(jg - 1) * C3_V_DIM + v] = hk;
        __syncthreads();
        if (jg == 0) {
            float hk_full = hk + smem_red[0 * C3_V_DIM + v]
                + smem_red[1 * C3_V_DIM + v] + smem_red[2 * C3_V_DIM + v];
            const float v_i = (float)value[
                (unsigned long long)t * v_stride + vh * v_dim + v];
            smem_vnew[v] = (v_i - g_t * hk_full) * bt_t;
        }
        __syncthreads();

        const float vn = smem_vnew[v];
        float q_dot = 0.0f;
        #pragma unroll
        for (int jj = 0; jj < C3_JPT; jj++) {
            const float h = g_t * Hreg[jj] + (float)smem_k[j0 + jj] * vn;
            Hreg[jj] = h;
            q_dot += h * (float)smem_q[j0 + jj];
        }
        if (jg > 0) smem_red[(jg - 1) * C3_V_DIM + v] = q_dot;
        __syncthreads();
        if (jg == 0) {
            float o = q_dot + smem_red[0 * C3_V_DIM + v]
                + smem_red[1 * C3_V_DIM + v] + smem_red[2 * C3_V_DIM + v];
            output[(unsigned long long)t * num_v_heads * v_dim + vh * v_dim + v] =
                __float2bfloat16(o * inv_sqrt_d);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int jj = 0; jj < C3_JPT; jj++)
        H_global[(unsigned long long)(j0 + jj) * C3_V_DIM + v] = Hreg[jj];
}
