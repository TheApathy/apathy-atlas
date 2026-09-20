// SPDX-License-Identifier: AGPL-3.0-only

__device__ __constant__ float OI640_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

extern "C" __global__ __launch_bounds__(128, 1)
void moe_w4a16_orig_i640_compact_prefill_gemm(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const oi640::Contract* __restrict__ contract,
    const oi640::Workspace* __restrict__ workspace,
    int* status,
    unsigned int N,
    unsigned int K
) {
    if (gridDim.x != oi640::FIXED_GRID || gridDim.y != 1 || gridDim.z != 1 ||
        blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_GEOMETRY);
        return;
    }
    const bool shape = (N == oi640::INTER && K == oi640::HIDDEN) ||
        (N == oi640::HIDDEN && K == oi640::INTER);
    if (!shape || A == nullptr || packed_ptrs == nullptr || scale_ptrs == nullptr ||
        scale2_vals == nullptr || C == nullptr || expert_offsets == nullptr ||
        sorted_token_ids == nullptr || !oi640::contract_ok(contract) ||
        !oi640::workspace_ok(workspace, contract) || status == nullptr ||
        status[0] != oi640::PLANNED) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_ABI);
        return;
    }
    const unsigned int n_tiles = (N + oi640::TILE_N - 1) / oi640::TILE_N;
    const unsigned int task_count = workspace->work_count * n_tiles;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;
    __shared__ __nv_bfloat16 smem_A[oi640::TILE_M][oi640::STEP_K + oi640::PAD_K];
    __shared__ __nv_bfloat16 smem_B[oi640::STEP_K][oi640::TILE_N + oi640::PAD_K];

    for (unsigned int task = blockIdx.x; task < task_count; task += gridDim.x) {
        const oi640::WorkItem item = workspace->items[task / n_tiles];
        const unsigned int cta_n = (task % n_tiles) * oi640::TILE_N;
        if (item.expert >= oi640::EXPERTS) {
            if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_PLAN);
            return;
        }
        const int m_start = expert_offsets[item.expert];
        const int m_end = expert_offsets[item.expert + 1];
        const int local = static_cast<int>(item.m_tile * oi640::TILE_M);
        const int M_expert = m_end - m_start;
        const auto* B = reinterpret_cast<const unsigned char*>(packed_ptrs[item.expert]);
        const auto* S = reinterpret_cast<const unsigned char*>(scale_ptrs[item.expert]);
        const float scale2 = scale2_vals[item.expert];
        if (m_start < 0 || m_end < m_start || local < 0 || local >= M_expert ||
            B == nullptr || S == nullptr || !isfinite(scale2)) {
            if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_POINTER);
            return;
        }
        const unsigned int cta_m = static_cast<unsigned int>(m_start + local);
        float acc[8][4];
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
        const unsigned int a_stride = oi640::STEP_K + oi640::PAD_K;
        const unsigned int b_stride = oi640::TILE_N + oi640::PAD_K;
#if !OI640_TRANSPOSED
        const unsigned int half_K = K / 2;
        const unsigned int groups = K / oi640::GROUP_K;
#endif

        for (unsigned int k_base = 0; k_base < K; k_base += oi640::STEP_K) {
            constexpr unsigned int a_ept = (oi640::TILE_M * oi640::STEP_K) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < a_ept; ++i) {
                const unsigned int idx = threadIdx.x * a_ept + i;
                const unsigned int row = idx / oi640::STEP_K;
                const unsigned int col = idx % oi640::STEP_K;
                if (local + static_cast<int>(row) < M_expert) {
                    const unsigned int ar = K == oi640::HIDDEN
                        ? static_cast<unsigned int>(sorted_token_ids[cta_m + row])
                        : cta_m + row;
                    smem_A[row][col] = A[static_cast<unsigned long long>(ar) * K + k_base + col];
                } else smem_A[row][col] = __float2bfloat16(0.0f);
            }
            constexpr unsigned int b_ept = (oi640::STEP_K * oi640::TILE_N) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < b_ept; ++i) {
                const unsigned int idx = threadIdx.x * b_ept + i;
                const unsigned int k = idx / oi640::TILE_N;
                const unsigned int n = idx % oi640::TILE_N;
                const unsigned int gk = k_base + k;
                const unsigned int gn = cta_n + n;
                if (gn < N) {
#if OI640_TRANSPOSED
                    const unsigned char pb =
                        B[static_cast<unsigned long long>(gk / 2) * N + gn];
                    const unsigned char sb =
                        S[static_cast<unsigned long long>(gk / oi640::GROUP_K) * N + gn];
#else
                    const unsigned char pb = B[static_cast<unsigned long long>(gn) * half_K + gk / 2];
                    const unsigned char sb = S[static_cast<unsigned long long>(gn) * groups + gk / oi640::GROUP_K];
#endif
                    const unsigned int nibble = (gk & 1) ? pb >> 4 : pb & 15;
                    __nv_fp8_e4m3 fp8;
                    *reinterpret_cast<unsigned char*>(&fp8) = sb;
                    smem_B[k][n] = __float2bfloat16(OI640_E2M1_LUT[nibble] * float(fp8) * scale2);
                } else smem_B[k][n] = __float2bfloat16(0.0f);
            }
            __syncthreads();
            const auto* sA = reinterpret_cast<const unsigned short*>(smem_A);
            const auto* sB = reinterpret_cast<const unsigned short*>(smem_B);
            #pragma unroll
            for (unsigned int k_sub = 0; k_sub < oi640::STEP_K; k_sub += 16) {
                const unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8;
                const unsigned int fc0 = k_sub + tid * 2, fc1 = fc0 + 8;
                const unsigned int a0 = (unsigned(sA[fr0*a_stride+fc0+1])<<16)|sA[fr0*a_stride+fc0];
                const unsigned int a1 = (unsigned(sA[fr1*a_stride+fc0+1])<<16)|sA[fr1*a_stride+fc0];
                const unsigned int a2 = (unsigned(sA[fr0*a_stride+fc1+1])<<16)|sA[fr0*a_stride+fc1];
                const unsigned int a3 = (unsigned(sA[fr1*a_stride+fc1+1])<<16)|sA[fr1*a_stride+fc1];
                #pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
                    const unsigned int nc = nt * 8 + group_id;
                    const unsigned int k0 = k_sub + tid * 2, k1 = k0 + 8;
                    const unsigned int b0 = (unsigned(sB[(k0+1)*b_stride+nc])<<16)|sB[k0*b_stride+nc];
                    const unsigned int b1 = (unsigned(sB[(k1+1)*b_stride+nc])<<16)|sB[k1*b_stride+nc];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]));
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const unsigned int c0 = cta_n + nt*8 + tid*2, c1 = c0 + 1;
            const unsigned int r0 = cta_m + warp_m_offset + group_id, r1 = r0 + 8;
            const bool r0v = int(warp_m_offset + group_id) + local < M_expert;
            const bool r1v = int(warp_m_offset + group_id + 8) + local < M_expert;
            if (r0v && c0 < N) C[static_cast<unsigned long long>(r0)*N+c0] = __float2bfloat16(acc[nt][0]);
            if (r0v && c1 < N) C[static_cast<unsigned long long>(r0)*N+c1] = __float2bfloat16(acc[nt][1]);
            if (r1v && c0 < N) C[static_cast<unsigned long long>(r1)*N+c0] = __float2bfloat16(acc[nt][2]);
            if (r1v && c1 < N) C[static_cast<unsigned long long>(r1)*N+c1] = __float2bfloat16(acc[nt][3]);
        }
        __syncthreads();
    }
}
