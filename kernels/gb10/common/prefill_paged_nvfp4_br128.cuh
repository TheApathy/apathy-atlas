// SPDX-License-Identifier: AGPL-3.0-only

// NVFP4-only BR=128 paged Flash Attention shadow.
//
// The kernel preserves the BR=64 BC=32 QK/softmax/PV arithmetic order for
// each 64-row half, but shares one dequantized K/V tile across both halves.
// K and V deliberately alias: all QK work completes before V overwrites K,
// trading overlap for a 95,808-byte dynamic-shared footprint (including LUT).
// Dynamic shared memory is required because ptxas limits direct static shared
// allocations to 48 KiB; the launcher's >48 KiB path opts the function into
// the architecture-specific per-block limit before launch.

#define BR128 128
#define TILE_CHUNKS_Q128 (BR128 * (HDIM / 8))
#define BR128_Q_BYTES (BR128 * HDIM_PAD * sizeof(__nv_bfloat16))
#define BR128_KV_BYTES (BC * HDIM_PAD * sizeof(__nv_bfloat16))
#define BR128_P_BYTES (BR128 * (BC + PAD_P) * sizeof(__half))
#define BR128_ML_BYTES (BR128 * 2 * sizeof(float))
#define BR128_LUT_BYTES (16 * sizeof(float))
#define BR128_SHARED_BYTES \
    (BR128_Q_BYTES + BR128_KV_BYTES + BR128_P_BYTES + \
     BR128_ML_BYTES + BR128_LUT_BYTES)

static_assert(BR128_SHARED_BYTES == 95808,
              "NVFP4 BR128 shared-memory ABI changed");

extern "C" __global__ __launch_bounds__(512, 1) void PAGED_CONCAT(KERNEL_NAME, _128)(
    const __nv_bfloat16* __restrict__ Q,
    K_CACHE_TYPE K_cache,
    V_CACHE_TYPE V_cache,
    __nv_bfloat16* __restrict__ O,
#ifdef PREFILL_BATCHED
    const int* const* __restrict__ block_table_ptrs,
    const unsigned int batch_size,
#else
    const int* __restrict__ block_table,
#endif
    const unsigned int q_len,
    const unsigned int kv_len,
    const unsigned int q_offset,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_block_size,
    const unsigned int sliding_window,
    const unsigned int causal_mask_enabled
    KERNEL_EXTRA_PARAMS
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
#ifdef PREFILL_BATCHED
    const unsigned int b = blockIdx.z;
    if (b >= batch_size) return;
    const int* const __restrict__ block_table = block_table_ptrs[b];
#endif
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;
    const unsigned int q_start = q_block * BR128;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + BR128, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);
#ifdef PREFILL_BATCHED
    const unsigned long long q_batch_off = (unsigned long long)b * q_len * q_seq_stride;
#endif

    extern __shared__ __align__(16) unsigned char smem_raw128[];
    __nv_bfloat16 (*smem_Q128)[HDIM_PAD] =
        reinterpret_cast<__nv_bfloat16 (*)[HDIM_PAD]>(smem_raw128);
    __nv_bfloat16 (*smem_KV128)[HDIM_PAD] =
        reinterpret_cast<__nv_bfloat16 (*)[HDIM_PAD]>(
            smem_raw128 + BR128_Q_BYTES);
    // Phase 2c: smem_P128 FP16 — same rationale as smem_P above. Both types
    // occupy two bytes, so the shared ABI remains identical under the bisect.
#ifdef ATLAS_DISABLE_FP16_PV
    __nv_bfloat16 (*smem_P128)[BC + PAD_P] =
        reinterpret_cast<__nv_bfloat16 (*)[BC + PAD_P]>(
            smem_raw128 + BR128_Q_BYTES + BR128_KV_BYTES);
#else
    __half (*smem_P128)[BC + PAD_P] =
        reinterpret_cast<__half (*)[BC + PAD_P]>(
            smem_raw128 + BR128_Q_BYTES + BR128_KV_BYTES);
#endif
    float (*smem_ml128)[2] = reinterpret_cast<float (*)[2]>(
        smem_raw128 + BR128_Q_BYTES + BR128_KV_BYTES + BR128_P_BYTES);
    float* e2m1_lut = reinterpret_cast<float*>(
        smem_raw128 + BR128_Q_BYTES + BR128_KV_BYTES +
        BR128_P_BYTES + BR128_ML_BYTES);
    if (tid < 16) {
        const float lut[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
           -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
        };
        e2m1_lut[tid] = lut[tid];
    }
    __syncthreads();

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int qk_warp_m = warp_id * 16;       // warps 0-7, each 16 rows
    const unsigned int pv_warp_m = (warp_id & 7) * 16; // pairs (0,8)..(7,15)
    const unsigned int pv_n_start = (warp_id >> 3) * N_TILES_PER_WARP;
    const unsigned int p_smem_stride128 = BC + PAD_P;

    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f, l_r1 = 0.0f;

    const unsigned int all_kv_blocks = (kv_len + BC - 1) / BC;
    const unsigned int lower_q_tile_end = min(q_start + 64u, q_len);
    unsigned int lower_num_kv_blocks = all_kv_blocks;
    { unsigned int mx = (q_offset + lower_q_tile_end - 1) / BC;
      lower_num_kv_blocks = min(lower_num_kv_blocks, mx + 1); }
    unsigned int full_num_kv_blocks = all_kv_blocks;
    { unsigned int mx = (q_offset + q_tile_end - 1) / BC;
      full_num_kv_blocks = min(full_num_kv_blocks, mx + 1); }

    // === Merged Q(128 rows) + K[0](32 rows) load ===
    {
        const unsigned int cpr = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS_Q128; idx += 512) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            unsigned int sa = __cvta_generic_to_shared(&smem_Q128[row][col]);
            if (q_start + row < q_len) {
#ifdef PREFILL_BATCHED
                const void* gm = (const void*)&Q[q_batch_off + (q_start+row)*q_seq_stride + q_head*head_dim + col];
#else
                const void* gm = (const void*)&Q[(q_start+row)*q_seq_stride + q_head*head_dim + col];
#endif
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(gm));
            } else { *((uint4*)&smem_Q128[row][col]) = make_uint4(0,0,0,0); }
        }
        if (full_num_kv_blocks > 0) {
            LOAD_KV_TILE(K_cache, block_table, smem_KV128, 0, kv_len, kv_head, tid, blockDim.x);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < full_num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;
        const bool lower_half = (warp_id & 7) < 4;
        const unsigned int warp_num_kv_blocks = lower_half
            ? lower_num_kv_blocks : full_num_kv_blocks;
        const bool warp_tile_active = kv_block < warp_num_kv_blocks;

        // QK^T uses the aliased tile as K. Lower-half warps stop at the exact
        // BR64 parent bound; executing additional all-masked MMAs would change
        // online-softmax state and violate parent-byte equivalence.
        float acc_s[4][4];
        if (warp_id < 8 && warp_tile_active) {
            #pragma unroll
            for (int i = 0; i < 4; i++) { acc_s[i][0]=0; acc_s[i][1]=0; acc_s[i][2]=0; acc_s[i][3]=0; }

            const unsigned short* sQ = (const unsigned short*)smem_Q128;
            const unsigned short* sK = (const unsigned short*)smem_KV128;

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM/16); ks++) {
                unsigned int kb = ks*16;
                unsigned int ar0=qk_warp_m+group_id, ar1=ar0+8;
                unsigned int ac0=kb+tid_in_group*2, ac1=ac0+8;
                unsigned int a0=*(const unsigned int*)&sQ[ar0*HDIM_PAD+ac0];
                unsigned int a1=*(const unsigned int*)&sQ[ar1*HDIM_PAD+ac0];
                unsigned int a2=*(const unsigned int*)&sQ[ar0*HDIM_PAD+ac1];
                unsigned int a3=*(const unsigned int*)&sQ[ar1*HDIM_PAD+ac1];

                #pragma unroll
                for (int nt=0; nt<4; nt++) {
                    unsigned int nc=nt*8+group_id, k0=kb+tid_in_group*2, k1=k0+8;
                    unsigned int b0=((unsigned int)sK[nc*HDIM_PAD+k0+1]<<16)|(unsigned int)sK[nc*HDIM_PAD+k0];
                    unsigned int b1=((unsigned int)sK[nc*HDIM_PAD+k1+1]<<16)|(unsigned int)sK[nc*HDIM_PAD+k1];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_s[nt][0]),"=f"(acc_s[nt][1]),"=f"(acc_s[nt][2]),"=f"(acc_s[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_s[nt][0]),"f"(acc_s[nt][1]),"f"(acc_s[nt][2]),"f"(acc_s[nt][3]));
                }
            }

            // === Register-based softmax with causal mask ===
            unsigned int row0=qk_warp_m+group_id, row1=row0+8;
            #pragma unroll
            for (int nt=0; nt<4; nt++) {
                acc_s[nt][0]*=inv_sqrt_d; acc_s[nt][1]*=inv_sqrt_d;
                acc_s[nt][2]*=inv_sqrt_d; acc_s[nt][3]*=inv_sqrt_d;
                unsigned int c0=nt*8+tid_in_group*2, c1=c0+1;
                unsigned int qr0=q_offset+q_start+row0, qr1=q_offset+q_start+row1;
                // Causal mask gated for DFlash γ-block (causal_mask_enabled=0).
                if(causal_mask_enabled){
                    if(kv_start+c0>qr0) acc_s[nt][0]=-1e30f; if(kv_start+c1>qr0) acc_s[nt][1]=-1e30f;
                    if(kv_start+c0>qr1) acc_s[nt][2]=-1e30f; if(kv_start+c1>qr1) acc_s[nt][3]=-1e30f;
                }
                if(c0>=kv_tile_len){acc_s[nt][0]=-1e30f;acc_s[nt][2]=-1e30f;}
                if(c1>=kv_tile_len){acc_s[nt][1]=-1e30f;acc_s[nt][3]=-1e30f;}
                if(row0>=q_tile_len){acc_s[nt][0]=-1e30f;acc_s[nt][1]=-1e30f;}
                if(row1>=q_tile_len){acc_s[nt][2]=-1e30f;acc_s[nt][3]=-1e30f;}
                // Sliding window mask: K positions outside [Q-window+1, Q]. Only
                // evaluate after causal mask so (qr - kv_pos) is non-negative.
                if(sliding_window>0){
                    if(qr0>=kv_start+c0 && qr0-(kv_start+c0)>=sliding_window) acc_s[nt][0]=-1e30f;
                    if(qr0>=kv_start+c1 && qr0-(kv_start+c1)>=sliding_window) acc_s[nt][1]=-1e30f;
                    if(qr1>=kv_start+c0 && qr1-(kv_start+c0)>=sliding_window) acc_s[nt][2]=-1e30f;
                    if(qr1>=kv_start+c1 && qr1-(kv_start+c1)>=sliding_window) acc_s[nt][3]=-1e30f;
                }
            }

            float rmax0=-1e30f, rmax1=-1e30f;
            #pragma unroll
            for(int nt=0;nt<4;nt++){
                rmax0=fmaxf(rmax0,fmaxf(acc_s[nt][0],acc_s[nt][1]));
                rmax1=fmaxf(rmax1,fmaxf(acc_s[nt][2],acc_s[nt][3]));
            }
            rmax0=fmaxf(rmax0,__shfl_xor_sync(0xFFFFFFFF,rmax0,1));
            rmax0=fmaxf(rmax0,__shfl_xor_sync(0xFFFFFFFF,rmax0,2));
            rmax1=fmaxf(rmax1,__shfl_xor_sync(0xFFFFFFFF,rmax1,1));
            rmax1=fmaxf(rmax1,__shfl_xor_sync(0xFFFFFFFF,rmax1,2));

            float mn0=fmaxf(m_r0,rmax0);
            if (mn0 != m_r0) {
                float eo0=sw_exp(m_r0-mn0); l_r0*=eo0;
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][0]*=eo0;acc_o[i][1]*=eo0;}
                m_r0=mn0;
            }
            float mn1=fmaxf(m_r1,rmax1);
            if (mn1 != m_r1) {
                float eo1=sw_exp(m_r1-mn1); l_r1*=eo1;
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][2]*=eo1;acc_o[i][3]*=eo1;}
                m_r1=mn1;
            }

            float sum0=0, sum1=0;
            #pragma unroll
            for(int nt=0;nt<4;nt++){
                float p00=sw_exp(acc_s[nt][0]-m_r0),p01=sw_exp(acc_s[nt][1]-m_r0);
                float p10=sw_exp(acc_s[nt][2]-m_r1),p11=sw_exp(acc_s[nt][3]-m_r1);
                sum0+=p00+p01; sum1+=p10+p11;
                unsigned int c0=nt*8+tid_in_group*2;
#ifdef ATLAS_DISABLE_FP16_PV
                smem_P128[row0][c0]=__float2bfloat16_rn(p00); smem_P128[row0][c0+1]=__float2bfloat16_rn(p01);
                smem_P128[row1][c0]=__float2bfloat16_rn(p10); smem_P128[row1][c0+1]=__float2bfloat16_rn(p11);
#else
                smem_P128[row0][c0]=__float2half_rn(p00); smem_P128[row0][c0+1]=__float2half_rn(p01);
                smem_P128[row1][c0]=__float2half_rn(p10); smem_P128[row1][c0+1]=__float2half_rn(p11);
#endif
            }
            sum0+=__shfl_xor_sync(0xFFFFFFFF,sum0,1); sum0+=__shfl_xor_sync(0xFFFFFFFF,sum0,2);
            sum1+=__shfl_xor_sync(0xFFFFFFFF,sum1,1); sum1+=__shfl_xor_sync(0xFFFFFFFF,sum1,2);
            l_r0+=sum0; l_r1+=sum1;

            if(tid_in_group==0){
                smem_ml128[row0][0]=m_r0; smem_ml128[row0][1]=l_r0;
                smem_ml128[row1][0]=m_r1; smem_ml128[row1][1]=l_r1;
            }
        }

        // All QK readers and P/m-l writers must finish before V aliases K.
        __syncthreads();
        LOAD_KV_TILE(V_cache, block_table, smem_KV128,
                     kv_start, kv_len, kv_head, tid, blockDim.x);
        __syncthreads();

        // Second warp in each PV pair adopts the QK owner's online state.
        if(warp_id>=8 && warp_tile_active){
            unsigned int r0=pv_warp_m+group_id, r1=r0+8;
            float cm0=smem_ml128[r0][0], cm1=smem_ml128[r1][0];
            if (cm0 != m_r0) {
                float er0=sw_exp(m_r0-cm0);
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][0]*=er0;acc_o[i][1]*=er0;}
                m_r0=cm0;
            }
            if (cm1 != m_r1) {
                float er1=sw_exp(m_r1-cm1);
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP;i++){acc_o[i][2]*=er1;acc_o[i][3]*=er1;}
                m_r1=cm1;
            }
        }

        // === PV MMA (all 16 warps, in eight row-owner pairs) ===
        if (warp_tile_active) {
            // Phase 2c: FP16 PV MMA — see BR=32 path above for rationale.
            const unsigned short* sP=(const unsigned short*)smem_P128;
            #pragma unroll
            for(unsigned int ks=0;ks<2;ks++){
                unsigned int ko=ks*16;
                unsigned int ar0=pv_warp_m+group_id, ar1=ar0+8;
                unsigned int ac0=ko+tid_in_group*2, ac1=ac0+8;
                unsigned int a0=*(const unsigned int*)&sP[ar0*p_smem_stride128+ac0];
                unsigned int a1=*(const unsigned int*)&sP[ar1*p_smem_stride128+ac0];
                unsigned int a2=*(const unsigned int*)&sP[ar0*p_smem_stride128+ac1];
                unsigned int a3=*(const unsigned int*)&sP[ar1*p_smem_stride128+ac1];
                #pragma unroll
                for(int nt=0;nt<N_TILES_PER_WARP;nt++){
                    unsigned int nc=(pv_n_start+nt)*8+group_id, k0=ko+tid_in_group*2, k1=k0+8;
#ifdef ATLAS_DISABLE_FP16_PV
                    const unsigned short* sV=(const unsigned short*)smem_KV128;
                    unsigned int b0=((unsigned int)sV[(k0+1)*HDIM_PAD+nc]<<16)|(unsigned int)sV[k0*HDIM_PAD+nc];
                    unsigned int b1=((unsigned int)sV[(k1+1)*HDIM_PAD+nc]<<16)|(unsigned int)sV[k1*HDIM_PAD+nc];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_o[nt][0]),"=f"(acc_o[nt][1]),"=f"(acc_o[nt][2]),"=f"(acc_o[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_o[nt][0]),"f"(acc_o[nt][1]),"f"(acc_o[nt][2]),"f"(acc_o[nt][3]));
#else
                    unsigned int b0=bf16x2_to_f16x2_bits(
                        smem_KV128[k0][nc], smem_KV128[k0+1][nc]);
                    unsigned int b1=bf16x2_to_f16x2_bits(
                        smem_KV128[k1][nc], smem_KV128[k1+1][nc]);
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_o[nt][0]),"=f"(acc_o[nt][1]),"=f"(acc_o[nt][2]),"=f"(acc_o[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_o[nt][0]),"f"(acc_o[nt][1]),"f"(acc_o[nt][2]),"f"(acc_o[nt][3]));
#endif
                }
            }
        }

        // Finish every V read before overwriting the aliased tile with K[i+1].
        __syncthreads();
        if(kv_block+1<full_num_kv_blocks){
            LOAD_KV_TILE(K_cache, block_table, smem_KV128,
                         (kv_block+1)*BC, kv_len, kv_head, tid, blockDim.x);
        }
        __syncthreads();
    }

    // === Final normalization and store ===
    {
        unsigned int r0=pv_warp_m+group_id, r1=r0+8;
        float il0,il1;
        if(warp_id<8){
            il0=(l_r0>0)?(1.f/l_r0):0;
            il1=(l_r1>0)?(1.f/l_r1):0;
        } else {
            float lv0=smem_ml128[r0][1], lv1=smem_ml128[r1][1];
            il0=(lv0>0)?(1.f/lv0):0;
            il1=(lv1>0)?(1.f/lv1):0;
        }

#ifdef PREFILL_BATCHED
        __nv_bfloat16* ob=O+q_batch_off+q_head*head_dim;
#else
        __nv_bfloat16* ob=O+q_head*head_dim;
#endif
        #pragma unroll
        for(int nt=0;nt<N_TILES_PER_WARP;nt++){
            unsigned int c0=(pv_n_start+nt)*8+tid_in_group*2;
            unsigned int gr0=q_start+r0, gr1=q_start+r1;
            if(gr0<q_len&&r0<q_tile_len&&c0<head_dim){
                unsigned int lo=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0]*il0));
                unsigned int hi=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1]*il0));
                *(unsigned int*)&ob[gr0*q_seq_stride+c0]=lo|(hi<<16);
            }
            if(gr1<q_len&&r1<q_tile_len&&c0<head_dim){
                unsigned int lo=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2]*il1));
                unsigned int hi=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3]*il1));
                *(unsigned int*)&ob[gr1*q_seq_stride+c0]=lo|(hi<<16);
            }
        }
    }
}

#undef TILE_CHUNKS_Q128
#undef BR128
#undef BR128_SHARED_BYTES
#undef BR128_LUT_BYTES
#undef BR128_ML_BYTES
#undef BR128_P_BYTES
#undef BR128_KV_BYTES
#undef BR128_Q_BYTES
