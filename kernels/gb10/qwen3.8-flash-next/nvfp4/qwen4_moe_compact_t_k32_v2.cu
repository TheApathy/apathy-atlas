// SPDX-License-Identifier: AGPL-3.0-only
//
// V2 of the transposed compact-worklist MoE GEMM (same ABI, same plan
// kernel / workspace / status contract as `qwen4_moe_compact_t_k32`, same
// 64x64 tile per work item, same BF16 operand rounding and the same
// per-thread K16 accumulation order, so results match the V1 kernel).
//
// What changed: a 3-stage cp.async pipeline for the A tile and the RAW
// packed-nibble / FP8-scale B slices, B fragments dequantized straight from
// the raw bytes into mma registers (no BF16 B round trip through shared
// memory), ldmatrix A fragments, and 3 CTAs per SM instead of 1.
#define ATLAS_QWEN4_MOE_COMPACT_CONTRACT 1
#define OI640_STEP_K 32
#define OI640_TRANSPOSED 1
#define flash_next_orig_i640_prefill flash_next_f32_transposed_compact_k32_v2
#include "moe_w4a16_orig_i640_compact_prefill.cuh"
namespace oi640 = flash_next_f32_transposed_compact_k32_v2;

namespace {
constexpr unsigned STAGES = 3;
constexpr unsigned A_STRIDE = 40;   // bf16 per smem A row (32 + 8 pad) = 80 B
constexpr unsigned B_STRIDE = 80;   // bytes per smem packed-B row (64 + 16 pad)
constexpr unsigned S_STRIDE = 80;   // bytes per smem scale row
constexpr unsigned BF_STRIDE = 72;
constexpr unsigned TN3 = 128;        // v3 tile N (64 rows x 128 cols per CTA, 8 warps)
constexpr unsigned B3_STRIDE = 144;  // bytes per raw packed-B row (128 + 16 pad)
constexpr unsigned S3_STRIDE = 144;
constexpr unsigned BF3_STRIDE = 136; // bf16 per dequantized B row (128 + 8 pad) = 272 B
  // bf16 per dequantized B row (64 + 8 pad) = 144 B

__device__ __constant__ float V2_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem, bool pred) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
__device__ __forceinline__ unsigned pack_bf16x2(float lo, float hi) {
    const __nv_bfloat16 l = __float2bfloat16(lo);
    const __nv_bfloat16 h = __float2bfloat16(hi);
    return (unsigned)(*reinterpret_cast<const unsigned short*>(&l)) |
           ((unsigned)(*reinterpret_cast<const unsigned short*>(&h)) << 16);
}
} // namespace

extern "C" __global__ __launch_bounds__(128, 3)
void qwen4_moe_compact_t_gemm_k32_v2(
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
    // Raw pipeline stages (cp.async targets) + double-buffered dequantized B.
    __shared__ __align__(16) __nv_bfloat16 sA[STAGES][oi640::TILE_M][A_STRIDE];
    __shared__ __align__(16) unsigned char sB[STAGES][16][B_STRIDE];
    __shared__ __align__(16) unsigned char sS[STAGES][2][S_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 sBf[2][oi640::STEP_K][BF_STRIDE];
    __shared__ float s_lut[16];

    const unsigned n_tiles = (N + oi640::TILE_N - 1) / oi640::TILE_N;
    const unsigned task_count = workspace->work_count * n_tiles;
    const unsigned tid = threadIdx.x;
    const unsigned warp_id = tid >> 5;
    const unsigned lane = tid & 31;
    const unsigned warp_m = warp_id * 16;
    const unsigned group_id = lane >> 2;
    const unsigned tig = lane & 3;
    const unsigned KT = K / oi640::STEP_K;
    const bool gather = (K == oi640::HIDDEN);
    if (tid < 16) s_lut[tid] = V2_E2M1_LUT[tid];
    // Dequant assignment: thread -> column n = tid & 63, packed rows 8*(tid>>6) .. +8
    // (16 k values) of the 32-deep stage.
    const unsigned dq_n = tid & 63;
    const unsigned dq_r0 = (tid >> 6) * 8;
    __syncthreads();

    for (unsigned task = blockIdx.x; task < task_count; task += gridDim.x) {
        const oi640::WorkItem item = workspace->items[task / n_tiles];
        const unsigned cta_n = (task % n_tiles) * oi640::TILE_N;
        if (item.expert >= oi640::EXPERTS) {
            if (tid == 0) oi640::fail(status, oi640::ERR_PLAN);
            return;
        }
        const int m_start = expert_offsets[item.expert];
        const int m_end = expert_offsets[item.expert + 1];
        const int local = static_cast<int>(item.m_tile * oi640::TILE_M);
        const int M_expert = m_end - m_start;
        const unsigned char* B = reinterpret_cast<const unsigned char*>(packed_ptrs[item.expert]);
        const unsigned char* S = reinterpret_cast<const unsigned char*>(scale_ptrs[item.expert]);
        const float scale2 = scale2_vals[item.expert];
        if (m_start < 0 || m_end < m_start || local < 0 || local >= M_expert ||
            B == nullptr || S == nullptr || !isfinite(scale2)) {
            if (tid == 0) oi640::fail(status, oi640::ERR_POINTER);
            return;
        }
        const unsigned cta_m = static_cast<unsigned>(m_start + local);

        const __nv_bfloat16* a_src[2];
        bool a_ok[2];
        unsigned a_row[2], a_cc[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned c = tid + i * 128;
            a_row[i] = c >> 2;
            a_cc[i] = c & 3;
            const bool ok = (local + static_cast<int>(a_row[i])) < M_expert;
            const unsigned ar = ok
                ? (gather ? static_cast<unsigned>(sorted_token_ids[cta_m + a_row[i]]) : cta_m + a_row[i])
                : 0u;
            a_ok[i] = ok;
            a_src[i] = A + static_cast<unsigned long long>(ar) * K + a_cc[i] * 8;
        }

        auto load_stage = [&](unsigned stage, unsigned kt) {
            const unsigned k_base = kt * oi640::STEP_K;
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                cp_async_16(&sA[stage][a_row[i]][a_cc[i] * 8], a_src[i] + k_base, a_ok[i]);
            }
            if (tid < 64) {
                const unsigned r = tid >> 2, cc = tid & 3;
                cp_async_16(&sB[stage][r][cc * 16],
                            B + static_cast<unsigned long long>(k_base / 2 + r) * N + cta_n + cc * 16, true);
            } else if (tid < 72) {
                const unsigned t = tid - 64;
                const unsigned r = t >> 2, cc = t & 3;
                cp_async_16(&sS[stage][r][cc * 16],
                            S + static_cast<unsigned long long>(k_base / 16 + r) * N + cta_n + cc * 16, true);
            }
        };

        float acc[8][4];
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

        #pragma unroll
        for (unsigned s = 0; s < STAGES - 1; ++s) {
            if (s < KT) load_stage(s, s);
            cp_async_commit();
        }

        for (unsigned kt = 0; kt < KT; ++kt) {
            cp_async_wait<STAGES - 2>();
            __syncthreads();
            {
                const unsigned nk = kt + STAGES - 1;
                if (nk < KT) load_stage(nk % STAGES, nk);
                cp_async_commit();
            }
            const unsigned st = kt % STAGES;
            const unsigned bf = kt & 1;
            // Cooperative dequant of the 32x64 B slice: same per-element
            // arithmetic and BF16 rounding as the V1 kernel's smem_B fill.
            {
                // packed rows dq_r0..dq_r0+7 -> k = 2*row, 2*row+1; scale row = k/16.
                #pragma unroll
                for (unsigned rr = 0; rr < 8; ++rr) {
                    const unsigned pr = dq_r0 + rr;
                    const unsigned k = pr * 2;
                    __nv_fp8_e4m3 fp8;
                    *reinterpret_cast<unsigned char*>(&fp8) = sS[st][k >> 4][dq_n];
                    const float f = static_cast<float>(fp8);
                    const unsigned char pb = sB[st][pr][dq_n];
                    sBf[bf][k][dq_n] = __float2bfloat16(s_lut[pb & 15] * f * scale2);
                    sBf[bf][k + 1][dq_n] = __float2bfloat16(s_lut[pb >> 4] * f * scale2);
                }
            }
            __syncthreads();
            #pragma unroll
            for (unsigned k_sub = 0; k_sub < oi640::STEP_K; k_sub += 16) {
                unsigned a0, a1, a2, a3;
                {
                    const unsigned row = warp_m + (lane & 15);
                    const unsigned col = k_sub + ((lane >> 4) << 3);
                    const unsigned addr = static_cast<unsigned>(
                        __cvta_generic_to_shared(&sA[st][row][col]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
                }
                #pragma unroll
                for (int np = 0; np < 4; ++np) {
                    // Two n-tiles per ldmatrix.x4.trans: matrices (k0-7,n0), (k8-15,n0), (k0-7,n1), (k8-15,n1).
                    unsigned b00, b01, b10, b11;
                    const unsigned row = k_sub + (lane & 7) + (((lane >> 3) & 1) << 3);
                    const unsigned col = (np * 2 + (lane >> 4)) * 8;
                    const unsigned addr = static_cast<unsigned>(
                        __cvta_generic_to_shared(&sBf[bf][row][col]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(b00), "=r"(b01), "=r"(b10), "=r"(b11) : "r"(addr));
                    const int nt0 = np * 2, nt1 = np * 2 + 1;
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc[nt0][0]),"=f"(acc[nt0][1]),"=f"(acc[nt0][2]),"=f"(acc[nt0][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b00),"r"(b01),
                         "f"(acc[nt0][0]),"f"(acc[nt0][1]),"f"(acc[nt0][2]),"f"(acc[nt0][3]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc[nt1][0]),"=f"(acc[nt1][1]),"=f"(acc[nt1][2]),"=f"(acc[nt1][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b10),"r"(b11),
                         "f"(acc[nt1][0]),"f"(acc[nt1][1]),"f"(acc[nt1][2]),"f"(acc[nt1][3]));
                }
            }
        }
        cp_async_wait<0>();

        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const unsigned c0 = cta_n + nt * 8 + tig * 2, c1 = c0 + 1;
            const unsigned r0 = cta_m + warp_m + group_id, r1 = r0 + 8;
            const bool r0v = static_cast<int>(warp_m + group_id) + local < M_expert;
            const bool r1v = static_cast<int>(warp_m + group_id + 8) + local < M_expert;
            if (r0v && c0 < N) C[static_cast<unsigned long long>(r0) * N + c0] = __float2bfloat16(acc[nt][0]);
            if (r0v && c1 < N) C[static_cast<unsigned long long>(r0) * N + c1] = __float2bfloat16(acc[nt][1]);
            if (r1v && c0 < N) C[static_cast<unsigned long long>(r1) * N + c0] = __float2bfloat16(acc[nt][2]);
            if (r1v && c1 < N) C[static_cast<unsigned long long>(r1) * N + c1] = __float2bfloat16(acc[nt][3]);
        }
        __syncthreads();
    }
}


// V3: identical arithmetic, 64x128 CTA tile (halves the A re-reads across n-tiles).
extern "C" __global__ __launch_bounds__(256, 2)
void qwen4_moe_compact_t_gemm_k32_v3(
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
        blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1) {
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
    // Raw pipeline stages (cp.async targets) + double-buffered dequantized B.
    __shared__ __align__(16) __nv_bfloat16 sA[STAGES][oi640::TILE_M][A_STRIDE];
    __shared__ __align__(16) unsigned char sB[STAGES][16][B3_STRIDE];
    __shared__ __align__(16) unsigned char sS[STAGES][2][S3_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 sBf[2][oi640::STEP_K][BF3_STRIDE];
    __shared__ float s_lut[16];

    const unsigned n_tiles = (N + TN3 - 1) / TN3;
    const unsigned task_count = workspace->work_count * n_tiles;
    const unsigned tid = threadIdx.x;
    const unsigned warp_id = tid >> 5;
    const unsigned lane = tid & 31;
    const unsigned warp_m = (warp_id & 3) * 16;
    const unsigned warp_n = (warp_id >> 2) * 64;
    const unsigned group_id = lane >> 2;
    const unsigned tig = lane & 3;
    const unsigned KT = K / oi640::STEP_K;
    const bool gather = (K == oi640::HIDDEN);
    if (tid < 16) s_lut[tid] = V2_E2M1_LUT[tid];
    // Dequant assignment: thread -> column n = tid & 63, packed rows 8*(tid>>6) .. +8
    // (16 k values) of the 32-deep stage.
    const unsigned dq_n = tid & 127;
    const unsigned dq_r0 = (tid >> 7) * 8;
    __syncthreads();

    for (unsigned task = blockIdx.x; task < task_count; task += gridDim.x) {
        const oi640::WorkItem item = workspace->items[task / n_tiles];
        const unsigned cta_n = (task % n_tiles) * TN3;
        if (item.expert >= oi640::EXPERTS) {
            if (tid == 0) oi640::fail(status, oi640::ERR_PLAN);
            return;
        }
        const int m_start = expert_offsets[item.expert];
        const int m_end = expert_offsets[item.expert + 1];
        const int local = static_cast<int>(item.m_tile * oi640::TILE_M);
        const int M_expert = m_end - m_start;
        const unsigned char* B = reinterpret_cast<const unsigned char*>(packed_ptrs[item.expert]);
        const unsigned char* S = reinterpret_cast<const unsigned char*>(scale_ptrs[item.expert]);
        const float scale2 = scale2_vals[item.expert];
        if (m_start < 0 || m_end < m_start || local < 0 || local >= M_expert ||
            B == nullptr || S == nullptr || !isfinite(scale2)) {
            if (tid == 0) oi640::fail(status, oi640::ERR_POINTER);
            return;
        }
        const unsigned cta_m = static_cast<unsigned>(m_start + local);

        const __nv_bfloat16* a_src[1];
        bool a_ok[1];
        unsigned a_row[1], a_cc[1];
        #pragma unroll
        for (int i = 0; i < 1; ++i) {
            const unsigned c = tid;
            a_row[i] = c >> 2;
            a_cc[i] = c & 3;
            const bool ok = (local + static_cast<int>(a_row[i])) < M_expert;
            const unsigned ar = ok
                ? (gather ? static_cast<unsigned>(sorted_token_ids[cta_m + a_row[i]]) : cta_m + a_row[i])
                : 0u;
            a_ok[i] = ok;
            a_src[i] = A + static_cast<unsigned long long>(ar) * K + a_cc[i] * 8;
        }

        auto load_stage = [&](unsigned stage, unsigned kt) {
            const unsigned k_base = kt * oi640::STEP_K;
            cp_async_16(&sA[stage][a_row[0]][a_cc[0] * 8], a_src[0] + k_base, a_ok[0]);
            if (tid < 128) {
                const unsigned r = tid >> 3, cc = tid & 7;
                cp_async_16(&sB[stage][r][cc * 16],
                            B + static_cast<unsigned long long>(k_base / 2 + r) * N + cta_n + cc * 16, true);
            } else if (tid < 144) {
                const unsigned t = tid - 128;
                const unsigned r = t >> 3, cc = t & 7;
                cp_async_16(&sS[stage][r][cc * 16],
                            S + static_cast<unsigned long long>(k_base / 16 + r) * N + cta_n + cc * 16, true);
            }
        };

        float acc[8][4];
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

        #pragma unroll
        for (unsigned s = 0; s < STAGES - 1; ++s) {
            if (s < KT) load_stage(s, s);
            cp_async_commit();
        }

        for (unsigned kt = 0; kt < KT; ++kt) {
            cp_async_wait<STAGES - 2>();
            __syncthreads();
            {
                const unsigned nk = kt + STAGES - 1;
                if (nk < KT) load_stage(nk % STAGES, nk);
                cp_async_commit();
            }
            const unsigned st = kt % STAGES;
            const unsigned bf = kt & 1;
            // Cooperative dequant of the 32x64 B slice: same per-element
            // arithmetic and BF16 rounding as the V1 kernel's smem_B fill.
            {
                // packed rows dq_r0..dq_r0+7 -> k = 2*row, 2*row+1; scale row = k/16.
                #pragma unroll
                for (unsigned rr = 0; rr < 8; ++rr) {
                    const unsigned pr = dq_r0 + rr;
                    const unsigned k = pr * 2;
                    __nv_fp8_e4m3 fp8;
                    *reinterpret_cast<unsigned char*>(&fp8) = sS[st][k >> 4][dq_n];
                    const float f = static_cast<float>(fp8);
                    const unsigned char pb = sB[st][pr][dq_n];
                    sBf[bf][k][dq_n] = __float2bfloat16(s_lut[pb & 15] * f * scale2);
                    sBf[bf][k + 1][dq_n] = __float2bfloat16(s_lut[pb >> 4] * f * scale2);
                }
            }
            __syncthreads();
            #pragma unroll
            for (unsigned k_sub = 0; k_sub < oi640::STEP_K; k_sub += 16) {
                unsigned a0, a1, a2, a3;
                {
                    const unsigned row = warp_m + (lane & 15);
                    const unsigned col = k_sub + ((lane >> 4) << 3);
                    const unsigned addr = static_cast<unsigned>(
                        __cvta_generic_to_shared(&sA[st][row][col]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
                }
                #pragma unroll
                for (int np = 0; np < 4; ++np) {
                    // Two n-tiles per ldmatrix.x4.trans: matrices (k0-7,n0), (k8-15,n0), (k0-7,n1), (k8-15,n1).
                    unsigned b00, b01, b10, b11;
                    const unsigned row = k_sub + (lane & 7) + (((lane >> 3) & 1) << 3);
                    const unsigned col = warp_n + (np * 2 + (lane >> 4)) * 8;
                    const unsigned addr = static_cast<unsigned>(
                        __cvta_generic_to_shared(&sBf[bf][row][col]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(b00), "=r"(b01), "=r"(b10), "=r"(b11) : "r"(addr));
                    const int nt0 = np * 2, nt1 = np * 2 + 1;
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc[nt0][0]),"=f"(acc[nt0][1]),"=f"(acc[nt0][2]),"=f"(acc[nt0][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b00),"r"(b01),
                         "f"(acc[nt0][0]),"f"(acc[nt0][1]),"f"(acc[nt0][2]),"f"(acc[nt0][3]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc[nt1][0]),"=f"(acc[nt1][1]),"=f"(acc[nt1][2]),"=f"(acc[nt1][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b10),"r"(b11),
                         "f"(acc[nt1][0]),"f"(acc[nt1][1]),"f"(acc[nt1][2]),"f"(acc[nt1][3]));
                }
            }
        }
        cp_async_wait<0>();

        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const unsigned c0 = cta_n + warp_n + nt * 8 + tig * 2, c1 = c0 + 1;
            const unsigned r0 = cta_m + warp_m + group_id, r1 = r0 + 8;
            const bool r0v = static_cast<int>(warp_m + group_id) + local < M_expert;
            const bool r1v = static_cast<int>(warp_m + group_id + 8) + local < M_expert;
            if (r0v && c0 < N) C[static_cast<unsigned long long>(r0) * N + c0] = __float2bfloat16(acc[nt][0]);
            if (r0v && c1 < N) C[static_cast<unsigned long long>(r0) * N + c1] = __float2bfloat16(acc[nt][1]);
            if (r1v && c0 < N) C[static_cast<unsigned long long>(r1) * N + c0] = __float2bfloat16(acc[nt][2]);
            if (r1v && c1 < N) C[static_cast<unsigned long long>(r1) * N + c1] = __float2bfloat16(acc[nt][3]);
        }
        __syncthreads();
    }
}

// Fused gate+up (+silu*up epilogue): one A tile feeds two independent accumulator sets.
extern "C" __global__ __launch_bounds__(128, 3)
void qwen4_moe_compact_t_gemm_gateup_k32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
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
    const bool shape = (N == oi640::INTER && K == oi640::HIDDEN);
    if (!shape || A == nullptr || packed_ptrs == nullptr || scale_ptrs == nullptr ||
        scale2_vals == nullptr || up_packed_ptrs == nullptr || up_scale_ptrs == nullptr || up_scale2_vals == nullptr || C == nullptr || expert_offsets == nullptr ||
        sorted_token_ids == nullptr || !oi640::contract_ok(contract) ||
        !oi640::workspace_ok(workspace, contract) || status == nullptr ||
        status[0] != oi640::PLANNED) {
        if (threadIdx.x == 0) oi640::fail(status, oi640::ERR_ABI);
        return;
    }
    // Raw pipeline stages (cp.async targets) + double-buffered dequantized B.
    __shared__ __align__(16) __nv_bfloat16 sA[STAGES][oi640::TILE_M][A_STRIDE];
    __shared__ __align__(16) unsigned char sB[STAGES][16][B_STRIDE];
    __shared__ __align__(16) unsigned char sS[STAGES][2][S_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 sBf[2][oi640::STEP_K][BF_STRIDE];
    __shared__ __align__(16) unsigned char sB2[STAGES][16][B_STRIDE];
    __shared__ __align__(16) unsigned char sS2[STAGES][2][S_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 sBf2[2][oi640::STEP_K][BF_STRIDE];
    __shared__ float s_lut[16];

    const unsigned n_tiles = (N + oi640::TILE_N - 1) / oi640::TILE_N;
    const unsigned task_count = workspace->work_count * n_tiles;
    const unsigned tid = threadIdx.x;
    const unsigned warp_id = tid >> 5;
    const unsigned lane = tid & 31;
    const unsigned warp_m = warp_id * 16;
    const unsigned group_id = lane >> 2;
    const unsigned tig = lane & 3;
    const unsigned KT = K / oi640::STEP_K;
    const bool gather = (K == oi640::HIDDEN);
    if (tid < 16) s_lut[tid] = V2_E2M1_LUT[tid];
    // Dequant assignment: thread -> column n = tid & 63, packed rows 8*(tid>>6) .. +8
    // (16 k values) of the 32-deep stage.
    const unsigned dq_n = tid & 63;
    const unsigned dq_r0 = (tid >> 6) * 8;
    __syncthreads();

    for (unsigned task = blockIdx.x; task < task_count; task += gridDim.x) {
        const oi640::WorkItem item = workspace->items[task / n_tiles];
        const unsigned cta_n = (task % n_tiles) * oi640::TILE_N;
        if (item.expert >= oi640::EXPERTS) {
            if (tid == 0) oi640::fail(status, oi640::ERR_PLAN);
            return;
        }
        const int m_start = expert_offsets[item.expert];
        const int m_end = expert_offsets[item.expert + 1];
        const int local = static_cast<int>(item.m_tile * oi640::TILE_M);
        const int M_expert = m_end - m_start;
        const unsigned char* B = reinterpret_cast<const unsigned char*>(packed_ptrs[item.expert]);
        const unsigned char* S = reinterpret_cast<const unsigned char*>(scale_ptrs[item.expert]);
        const float scale2 = scale2_vals[item.expert];
        const unsigned char* B2 = reinterpret_cast<const unsigned char*>(up_packed_ptrs[item.expert]);
        const unsigned char* S2 = reinterpret_cast<const unsigned char*>(up_scale_ptrs[item.expert]);
        const float scale2u = up_scale2_vals[item.expert];
        if (m_start < 0 || m_end < m_start || local < 0 || local >= M_expert ||
            B == nullptr || S == nullptr || !isfinite(scale2) || B2 == nullptr || S2 == nullptr || !isfinite(scale2u)) {
            if (tid == 0) oi640::fail(status, oi640::ERR_POINTER);
            return;
        }
        const unsigned cta_m = static_cast<unsigned>(m_start + local);

        const __nv_bfloat16* a_src[2];
        bool a_ok[2];
        unsigned a_row[2], a_cc[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned c = tid + i * 128;
            a_row[i] = c >> 2;
            a_cc[i] = c & 3;
            const bool ok = (local + static_cast<int>(a_row[i])) < M_expert;
            const unsigned ar = ok
                ? (gather ? static_cast<unsigned>(sorted_token_ids[cta_m + a_row[i]]) : cta_m + a_row[i])
                : 0u;
            a_ok[i] = ok;
            a_src[i] = A + static_cast<unsigned long long>(ar) * K + a_cc[i] * 8;
        }

        auto load_stage = [&](unsigned stage, unsigned kt) {
            const unsigned k_base = kt * oi640::STEP_K;
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                cp_async_16(&sA[stage][a_row[i]][a_cc[i] * 8], a_src[i] + k_base, a_ok[i]);
            }
            if (tid < 64) {
                const unsigned r = tid >> 2, cc = tid & 3;
                cp_async_16(&sB[stage][r][cc * 16],
                            B + static_cast<unsigned long long>(k_base / 2 + r) * N + cta_n + cc * 16, true);
                cp_async_16(&sB2[stage][r][cc * 16],
                            B2 + static_cast<unsigned long long>(k_base / 2 + r) * N + cta_n + cc * 16, true);
            } else if (tid < 72) {
                const unsigned t = tid - 64;
                const unsigned r = t >> 2, cc = t & 3;
                cp_async_16(&sS[stage][r][cc * 16],
                            S + static_cast<unsigned long long>(k_base / 16 + r) * N + cta_n + cc * 16, true);
                cp_async_16(&sS2[stage][r][cc * 16],
                            S2 + static_cast<unsigned long long>(k_base / 16 + r) * N + cta_n + cc * 16, true);
            }
        };

        float acc[8][4], acc2[8][4];
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) { acc[i][j] = 0.0f; acc2[i][j] = 0.0f; }

        #pragma unroll
        for (unsigned s = 0; s < STAGES - 1; ++s) {
            if (s < KT) load_stage(s, s);
            cp_async_commit();
        }

        for (unsigned kt = 0; kt < KT; ++kt) {
            cp_async_wait<STAGES - 2>();
            __syncthreads();
            {
                const unsigned nk = kt + STAGES - 1;
                if (nk < KT) load_stage(nk % STAGES, nk);
                cp_async_commit();
            }
            const unsigned st = kt % STAGES;
            const unsigned bf = kt & 1;
            // Cooperative dequant of the 32x64 B slice: same per-element
            // arithmetic and BF16 rounding as the V1 kernel's smem_B fill.
            {
                // packed rows dq_r0..dq_r0+7 -> k = 2*row, 2*row+1; scale row = k/16.
                #pragma unroll
                for (unsigned rr = 0; rr < 8; ++rr) {
                    const unsigned pr = dq_r0 + rr;
                    const unsigned k = pr * 2;
                    __nv_fp8_e4m3 fp8;
                    *reinterpret_cast<unsigned char*>(&fp8) = sS[st][k >> 4][dq_n];
                    const float f = static_cast<float>(fp8);
                    const unsigned char pb = sB[st][pr][dq_n];
                    sBf[bf][k][dq_n] = __float2bfloat16(s_lut[pb & 15] * f * scale2);
                    sBf[bf][k + 1][dq_n] = __float2bfloat16(s_lut[pb >> 4] * f * scale2);
                    __nv_fp8_e4m3 fp8u;
                    *reinterpret_cast<unsigned char*>(&fp8u) = sS2[st][k >> 4][dq_n];
                    const float fu = static_cast<float>(fp8u);
                    const unsigned char pb2 = sB2[st][pr][dq_n];
                    sBf2[bf][k][dq_n] = __float2bfloat16(s_lut[pb2 & 15] * fu * scale2u);
                    sBf2[bf][k + 1][dq_n] = __float2bfloat16(s_lut[pb2 >> 4] * fu * scale2u);
                }
            }
            __syncthreads();
            #pragma unroll
            for (unsigned k_sub = 0; k_sub < oi640::STEP_K; k_sub += 16) {
                unsigned a0, a1, a2, a3;
                {
                    const unsigned row = warp_m + (lane & 15);
                    const unsigned col = k_sub + ((lane >> 4) << 3);
                    const unsigned addr = static_cast<unsigned>(
                        __cvta_generic_to_shared(&sA[st][row][col]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
                }
                #pragma unroll
                for (int np = 0; np < 4; ++np) {
                    // Two n-tiles per ldmatrix.x4.trans: matrices (k0-7,n0), (k8-15,n0), (k0-7,n1), (k8-15,n1).
                    unsigned b00, b01, b10, b11;
                    const unsigned row = k_sub + (lane & 7) + (((lane >> 3) & 1) << 3);
                    const unsigned col = (np * 2 + (lane >> 4)) * 8;
                    const unsigned addr = static_cast<unsigned>(
                        __cvta_generic_to_shared(&sBf[bf][row][col]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(b00), "=r"(b01), "=r"(b10), "=r"(b11) : "r"(addr));
                    const int nt0 = np * 2, nt1 = np * 2 + 1;
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc[nt0][0]),"=f"(acc[nt0][1]),"=f"(acc[nt0][2]),"=f"(acc[nt0][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b00),"r"(b01),
                         "f"(acc[nt0][0]),"f"(acc[nt0][1]),"f"(acc[nt0][2]),"f"(acc[nt0][3]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc[nt1][0]),"=f"(acc[nt1][1]),"=f"(acc[nt1][2]),"=f"(acc[nt1][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b10),"r"(b11),
                         "f"(acc[nt1][0]),"f"(acc[nt1][1]),"f"(acc[nt1][2]),"f"(acc[nt1][3]));
                    unsigned c00, c01, c10, c11;
                    const unsigned addr2 = static_cast<unsigned>(
                        __cvta_generic_to_shared(&sBf2[bf][row][col]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(c00), "=r"(c01), "=r"(c10), "=r"(c11) : "r"(addr2));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc2[nt0][0]),"=f"(acc2[nt0][1]),"=f"(acc2[nt0][2]),"=f"(acc2[nt0][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(c00),"r"(c01),
                         "f"(acc2[nt0][0]),"f"(acc2[nt0][1]),"f"(acc2[nt0][2]),"f"(acc2[nt0][3]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc2[nt1][0]),"=f"(acc2[nt1][1]),"=f"(acc2[nt1][2]),"=f"(acc2[nt1][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(c10),"r"(c11),
                         "f"(acc2[nt1][0]),"f"(acc2[nt1][1]),"f"(acc2[nt1][2]),"f"(acc2[nt1][3]));
                }
            }
        }
        cp_async_wait<0>();

        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const unsigned c0 = cta_n + nt * 8 + tig * 2, c1 = c0 + 1;
            const unsigned r0 = cta_m + warp_m + group_id, r1 = r0 + 8;
            const bool r0v = static_cast<int>(warp_m + group_id) + local < M_expert;
            const bool r1v = static_cast<int>(warp_m + group_id + 8) + local < M_expert;
            // Epilogue = moe_silu_mul on the BF16-rounded gate/up values:
            // g*sigmoid(g)*u with __expf, same operation order, --fmad=false.
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const bool rv = (j < 2) ? r0v : r1v;
                const unsigned c = (j & 1) ? c1 : c0;
                const unsigned r = (j < 2) ? r0 : r1;
                if (rv && c < N) {
                    const float g = __bfloat162float(__float2bfloat16(acc[nt][j]));
                    const float u = __bfloat162float(__float2bfloat16(acc2[nt][j]));
                    const float sigmoid_g = 1.0f / (1.0f + __expf(-g));
                    const float result = g * sigmoid_g * u;
                    C[static_cast<unsigned long long>(r) * N + c] = __float2bfloat16(result);
                }
            }
        }
        __syncthreads();
    }
}

// V2 planner: identical work-item list (expert-major, tile-minor) and status
// contract as the V1 planner, but the 20480-token validation loop and the
// 1792-item reset run across the 512-thread block instead of on thread 0.
// census_hash covers the expert census only (it is consumed by the offline
// microgate harness, not by the GEMM or the runtime).
extern "C" __global__ __launch_bounds__(512, 1)
void qwen4_moe_compact_t_plan_k32_v2(
    const int* expert_offsets,
    const int* sorted_token_ids,
    const oi640::Contract* contract,
    oi640::Workspace* workspace,
    int* status
) {
    __shared__ int s_err;
    __shared__ uint32_t s_tiles[oi640::EXPERTS];
    __shared__ uint32_t s_prefix[oi640::EXPERTS];
    const unsigned tid = threadIdx.x;
    if (gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1 ||
        blockDim.x != oi640::EXPERTS || blockDim.y != 1 || blockDim.z != 1 ||
        expert_offsets == nullptr || sorted_token_ids == nullptr || contract == nullptr ||
        workspace == nullptr || status == nullptr) return;
    if (tid == 0) {
        s_err = 0;
        const uintptr_t offsets_begin = reinterpret_cast<uintptr_t>(expert_offsets);
        const uintptr_t tokens_begin = reinterpret_cast<uintptr_t>(sorted_token_ids);
        const uintptr_t contract_begin = reinterpret_cast<uintptr_t>(contract);
        const uintptr_t workspace_begin = reinterpret_cast<uintptr_t>(workspace);
        const uintptr_t status_begin = reinterpret_cast<uintptr_t>(status);
        if ((status_begin & 3) || status_begin > UINTPTR_MAX - sizeof(int)) { s_err = -1; }
        else if ((offsets_begin & 3) || (tokens_begin & 3) || (contract_begin & 15) ||
                 (workspace_begin & 15)) { s_err = oi640::ERR_ALIGNMENT; }
        else {
            const uint64_t offsets_bytes = (oi640::EXPERTS + 1ULL) * sizeof(int);
            const auto overlaps = [](uintptr_t a, uint64_t an, uintptr_t b, uint64_t bn) {
                return an > 0 && bn > 0 && a <= UINTPTR_MAX - an && b <= UINTPTR_MAX - bn &&
                    a < b + bn && b < a + an;
            };
            if (offsets_begin > UINTPTR_MAX - offsets_bytes ||
                contract_begin > UINTPTR_MAX - sizeof(oi640::Contract) ||
                workspace_begin > UINTPTR_MAX - sizeof(oi640::Workspace) ||
                overlaps(status_begin, sizeof(int), offsets_begin, offsets_bytes) ||
                overlaps(status_begin, sizeof(int), contract_begin, sizeof(oi640::Contract)) ||
                overlaps(status_begin, sizeof(int), workspace_begin, sizeof(oi640::Workspace)) ||
                overlaps(workspace_begin, sizeof(oi640::Workspace), offsets_begin, offsets_bytes) ||
                overlaps(workspace_begin, sizeof(oi640::Workspace), contract_begin,
                         sizeof(oi640::Contract))) { s_err = oi640::ERR_ALIAS; }
            else if (status[0] != oi640::PENDING || !oi640::contract_ok(contract)) { s_err = oi640::ERR_ABI; }
            else {
                const uint64_t token_bytes = static_cast<uint64_t>(contract->expanded) * sizeof(int);
                if (tokens_begin > UINTPTR_MAX - token_bytes ||
                    overlaps(status_begin, sizeof(int), tokens_begin, token_bytes) ||
                    overlaps(workspace_begin, sizeof(oi640::Workspace), tokens_begin, token_bytes) ||
                    overlaps(contract_begin, sizeof(oi640::Contract), tokens_begin, token_bytes)) {
                    s_err = oi640::ERR_ALIAS;
                }
            }
        }
    }
    __syncthreads();
    if (s_err == -1) return;
    if (s_err != 0) { if (tid == 0) status[0] = s_err; return; }
    if (tid == 0) {
        status[0] = oi640::BUILDING;
        workspace->ready_magic = 0;
        workspace->version = oi640::VERSION;
        workspace->rows = contract->rows;
        workspace->expanded = contract->expanded;
        workspace->work_count = 0;
        workspace->active_experts = 0;
        workspace->max_rows = 0;
        workspace->reserved = 0;
    }
    for (uint32_t i = tid; i < oi640::MAX_WORK_ITEMS; i += blockDim.x)
        workspace->items[i] = {0xffffffffU, 0xffffffffU};
    // Parallel token validation.
    const uint32_t expanded = contract->expanded;
    for (uint32_t i = tid; i < expanded; i += blockDim.x) {
        const int token = sorted_token_ids[i];
        if (token < 0 || static_cast<uint32_t>(token) >= contract->rows) atomicExch(&s_err, oi640::ERR_TOKEN);
    }
    // Per-expert row counts.
    {
        const int begin = expert_offsets[tid];
        const int end = expert_offsets[tid + 1];
        if (begin < 0 || end < begin || static_cast<uint32_t>(end) > expanded) atomicExch(&s_err, oi640::ERR_OFFSETS);
        const uint32_t rows = (end >= begin) ? static_cast<uint32_t>(end - begin) : 0u;
        s_tiles[tid] = (rows + oi640::TILE_M - 1) / oi640::TILE_M;
    }
    __syncthreads();
    if (tid == 0) {
        if (s_err == 0 && (expert_offsets[0] != 0 || expert_offsets[oi640::EXPERTS] != static_cast<int>(expanded)))
            s_err = oi640::ERR_OFFSETS;
        if (s_err == 0) {
            uint32_t count = 0, active = 0, max_rows = 0;
            uint64_t hash = 1469598103934665603ULL;
            for (uint32_t e = 0; e < oi640::EXPERTS; ++e) {
                const uint32_t rows = static_cast<uint32_t>(expert_offsets[e + 1] - expert_offsets[e]);
                hash = (hash ^ rows) * 1099511628211ULL;
                s_prefix[e] = count;
                if (rows == 0) continue;
                ++active;
                max_rows = max_rows > rows ? max_rows : rows;
                count += s_tiles[e];
            }
            if (count > oi640::MAX_WORK_ITEMS) s_err = oi640::ERR_CAPACITY;
            else if (count == 0) s_err = oi640::ERR_PLAN;
            else {
                workspace->work_count = count;
                workspace->active_experts = active;
                workspace->max_rows = max_rows;
                workspace->census_hash = hash;
            }
        }
    }
    __syncthreads();
    if (s_err != 0) { if (tid == 0) status[0] = s_err; return; }
    for (uint32_t t = 0; t < s_tiles[tid]; ++t)
        workspace->items[s_prefix[tid] + t] = {tid, t};
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        workspace->ready_magic = oi640::MAGIC;
        __threadfence();
        status[0] = oi640::PLANNED;
    }
}
