// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Fused MoE Expert+Shared GEMV — W3 (3-bit Lloyd-Max) routed experts.
//
// W3 clones of the exact kernel set the Laguna decode/verify path launches
// on NVFP4 routed experts (ATLAS_MOE_W3=1 swaps the handles):
//
//   moe_expert_gate_up_shared_w3          (single-token decode, forward.rs /
//   moe_expert_silu_down_shared_w3         forward_batched.rs per-token loop)
//   moe_expert_gate_up_shared_batchN_w3   (wide DFlash verify, forward_kn v1)
//   moe_expert_silu_down_shared_batchN_w3
//   moe_expert_gate_up_shared_batchN_v2_w3 (ATLAS_KN_V2 dedup gate_up)
//   moe_expert_down_dedup_batchN_w3        (ATLAS_KN_V4 dedup down; the
//                                           silu-precompute + blend kernels
//                                           read no expert weights → reused)
//
// ROUTED experts decode 3-bit Turbo3-packed codebook indices:
//     w = w3_lut[idx] * e4m3(group_scale) * scale2
// with w3_lut an 8-entry symmetric Lloyd-Max codebook in E2M1 units
// (device pointer, LAST kernel argument), packed [N, K*3/8] N-major,
// 8 values per 3-byte trio, value j at bits [3j, 3j+3).
// FP8-E4M3 per-16 group scales + per-expert scale2 are UNCHANGED NVFP4.
//
// The SHARED expert stays NVFP4 (Laguna passes its BF16-authoritative shared
// expert separately; the in-kernel shared slots keep the NVFP4 placeholder
// semantics of the parent kernels, including NULL → zero rows).
//
// Per-block math (FP32 accumulate, LUT × e4m3 × s2 dequant, warp-shuffle
// reduce, BF16 store) is byte-identical in structure to the parent NVFP4
// kernels — only the index width changes.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_W3F[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float atlas_dec_e4m3(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#else
__device__ __forceinline__ float atlas_dec_e4m3(unsigned char b) {
    __nv_fp8_e4m3 f; *(unsigned char*)&f = b; return (float)f;
}
#endif

// Dequant one 3-byte trio (8 elems) into w[0..7] with a single scale.
__device__ __forceinline__ void w3_dequant_trio(
    unsigned int bits, const float* s_lut8, float sc, float* w
) {
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        w[j] = s_lut8[(bits >> (3 * j)) & 7u] * sc;
    }
}

// Load a trio from 3 consecutive bytes (byte-addressed; any alignment).
__device__ __forceinline__ unsigned int w3_load_trio(const unsigned char* p) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16);
}

// ── Fused Gate+Up with shared expert — single-token W3 variant ──
// Grid: (ceil(N/8), top_k+1, 2)  Block: (128, 1, 1)
extern "C" __global__ void moe_expert_gate_up_shared_w3(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_packed,
    const unsigned char* __restrict__ sh_gate_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k,
    const float* __restrict__ w3_lut
) {
    const unsigned int expert_slot = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;

    if (is_shared) {
        if (sh_gate_packed == 0) {
            __nv_bfloat16* out = (proj == 0) ? sh_gate_out : sh_up_out;
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE)
                out[n_base + i] = __float2bfloat16(0.0f);
            return;
        }
        if (proj == 0) {
            B_packed = sh_gate_packed; B_scale = sh_gate_scale;
            s2 = sh_gate_s2; C = sh_gate_out;
        } else {
            B_packed = sh_up_packed; B_scale = sh_up_scale;
            s2 = sh_up_s2; C = sh_up_out;
        }
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id]; C = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id]; C = up_out;
        }
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int row3 = (K * 3) / 8;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float s_lut8[8];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_W3F[threadIdx.x];
    if (threadIdx.x >= 16 && threadIdx.x < 24) s_lut8[threadIdx.x - 16] = w3_lut[threadIdx.x - 16];
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;
        uint4 a_lo = ((const uint4*)A)[k16 * 2];
        uint4 a_hi = ((const uint4*)A)[k16 * 2 + 1];
        const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};

        const unsigned int sg = base_k / GROUP_SIZE;
        const unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        const float sc1 = atlas_dec_e4m3(sb1) * s2;
        const unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        const float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;

        float w1[16], w2[16];
        if (is_shared) {
            // NVFP4 nibbles: 8 bytes per 16 elems.
            unsigned long long p1 = *(const unsigned long long*)(B_packed + (unsigned long long)n1 * half_K + k16 * 8);
            unsigned long long p2 = have_n2 ?
                *(const unsigned long long*)(B_packed + (unsigned long long)n2 * half_K + k16 * 8) : 0;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                unsigned char bv1 = (unsigned char)(p1 >> (b * 8));
                w1[b * 2] = s_lut[bv1 & 0xF] * sc1;
                w1[b * 2 + 1] = s_lut[bv1 >> 4] * sc1;
                unsigned char bv2 = (unsigned char)(p2 >> (b * 8));
                w2[b * 2] = s_lut[bv2 & 0xF] * sc2;
                w2[b * 2 + 1] = s_lut[bv2 >> 4] * sc2;
            }
        } else {
            // W3 trios: 6 bytes per 16 elems, u16-aligned (row3 and 6 even).
            const unsigned short* q1 = (const unsigned short*)(B_packed + (unsigned long long)n1 * row3 + k16 * 6);
            const unsigned int h10 = q1[0], h11 = q1[1], h12 = q1[2];
            w3_dequant_trio((h10 | (h11 << 16)) & 0xFFFFFFu, s_lut8, sc1, &w1[0]);
            w3_dequant_trio(((h11 >> 8) | (h12 << 8)) & 0xFFFFFFu, s_lut8, sc1, &w1[8]);
            if (have_n2) {
                const unsigned short* q2 = (const unsigned short*)(B_packed + (unsigned long long)n2 * row3 + k16 * 6);
                const unsigned int h20 = q2[0], h21 = q2[1], h22 = q2[2];
                w3_dequant_trio((h20 | (h21 << 16)) & 0xFFFFFFu, s_lut8, sc2, &w2[0]);
                w3_dequant_trio(((h21 >> 8) | (h22 << 8)) & 0xFFFFFFu, s_lut8, sc2, &w2[8]);
            } else {
                #pragma unroll
                for (int j = 0; j < 16; j++) w2[j] = 0.0f;
            }
        }

        #pragma unroll
        for (int b = 0; b < 8; b++) {
            __nv_bfloat16 al, ah;
            *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
            float afl = __bfloat162float(al), afh = __bfloat162float(ah);
            acc1 += afl * w1[b * 2] + afh * w1[b * 2 + 1];
            acc2 += afl * w2[b * 2] + afh * w2[b * 2 + 1];
        }
    }

    const unsigned long long base = is_shared ? 0 : (unsigned long long)expert_slot * N;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) C[base + n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) C[base + n2] = __float2bfloat16(acc2);
    }
}

// ── Fused SiLU+Down with shared expert — single-token W3 variant ──
// Grid: (ceil(N/8), top_k+1, 1)  Block: (128, 1, 1), smem = K floats
extern "C" __global__ void moe_expert_silu_down_shared_w3(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k,
    const float* __restrict__ w3_lut
) {
    const unsigned int expert_slot = blockIdx.y;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;

    if (is_shared) {
        if (sh_down_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE)
                sh_down_out[n_base + i] = __float2bfloat16(0.0f);
            return;
        }
        B_packed = sh_down_packed; B_scale = sh_down_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in; u_ptr = sh_up_in;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)expert_slot * K;
        u_ptr = up_out + (unsigned long long)expert_slot * K;
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int row3 = (K * 3) / 8;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float s_lut8[8];
    extern __shared__ float s_act[];

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_W3F[threadIdx.x];
    if (threadIdx.x >= 16 && threadIdx.x < 24) s_lut8[threadIdx.x - 16] = w3_lut[threadIdx.x - 16];

    // SWIGLU clamp parity with moe_shared_expert_fused.cu (routed only).
    const float SWIGLU_LIMIT = 10.0f;
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        if (!is_shared) {
            gf = fminf(gf, SWIGLU_LIMIT);
            uf = fminf(fmaxf(uf, -SWIGLU_LIMIT), SWIGLU_LIMIT);
        }
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;
        const unsigned int sg = base_k / GROUP_SIZE;
        const unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        const float sc1 = atlas_dec_e4m3(sb1) * s2;
        const unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        const float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;

        float w1[16], w2[16];
        if (is_shared) {
            unsigned long long p1 = *(const unsigned long long*)(B_packed + (unsigned long long)n1 * half_K + k16 * 8);
            unsigned long long p2 = have_n2 ?
                *(const unsigned long long*)(B_packed + (unsigned long long)n2 * half_K + k16 * 8) : 0;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                unsigned char bv1 = (unsigned char)(p1 >> (b * 8));
                w1[b * 2] = s_lut[bv1 & 0xF] * sc1;
                w1[b * 2 + 1] = s_lut[bv1 >> 4] * sc1;
                unsigned char bv2 = (unsigned char)(p2 >> (b * 8));
                w2[b * 2] = s_lut[bv2 & 0xF] * sc2;
                w2[b * 2 + 1] = s_lut[bv2 >> 4] * sc2;
            }
        } else {
            const unsigned short* q1 = (const unsigned short*)(B_packed + (unsigned long long)n1 * row3 + k16 * 6);
            const unsigned int h10 = q1[0], h11 = q1[1], h12 = q1[2];
            w3_dequant_trio((h10 | (h11 << 16)) & 0xFFFFFFu, s_lut8, sc1, &w1[0]);
            w3_dequant_trio(((h11 >> 8) | (h12 << 8)) & 0xFFFFFFu, s_lut8, sc1, &w1[8]);
            if (have_n2) {
                const unsigned short* q2 = (const unsigned short*)(B_packed + (unsigned long long)n2 * row3 + k16 * 6);
                const unsigned int h20 = q2[0], h21 = q2[1], h22 = q2[2];
                w3_dequant_trio((h20 | (h21 << 16)) & 0xFFFFFFu, s_lut8, sc2, &w2[0]);
                w3_dequant_trio(((h21 >> 8) | (h22 << 8)) & 0xFFFFFFu, s_lut8, sc2, &w2[8]);
            } else {
                #pragma unroll
                for (int j = 0; j < 16; j++) w2[j] = 0.0f;
            }
        }

        #pragma unroll
        for (int b = 0; b < 8; b++) {
            float al = s_act[base_k + b * 2];
            float ah = s_act[base_k + b * 2 + 1];
            acc1 += al * w1[b * 2] + ah * w1[b * 2 + 1];
            acc2 += al * w2[b * 2] + ah * w2[b * 2 + 1];
        }
    }

    __nv_bfloat16* out = is_shared ? sh_down_out : (C + (unsigned long long)expert_slot * N);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) out[n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) out[n2] = __float2bfloat16(acc2);
    }
}

// ============================================================================
// batchN W3 variants (forward_kn wide DFlash verify). Structure identical to
// moe_shared_expert_fused_batch2.cu's batchN kernels; routed inner dequant is
// 3-bit. Grid/block contracts unchanged; w3_lut appended as last arg.
// ============================================================================
extern "C" __global__ void moe_expert_gate_up_shared_batchN_w3(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_packed,
    const unsigned char* __restrict__ sh_gate_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int num_tokens,
    const float* __restrict__ w3_lut
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y >= total_routed);

    unsigned int token, expert_slot;
    if (is_shared) {
        token = y - total_routed;
        expert_slot = 0;
    } else {
        token = y / top_k;
        expert_slot = y % top_k;
    }

    const __nv_bfloat16* A_token = A + (unsigned long long)token * K;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;

    if (is_shared) {
        if (proj == 0) {
            if (sh_gate_packed == 0) {
                const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
                __nv_bfloat16* z = sh_gate_out + (unsigned long long)token * N;
                for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N;
                     i += BLOCK_SIZE) {
                    z[n_base + i] = __float2bfloat16(0.0f);
                }
                return;
            }
            B_packed = sh_gate_packed; B_scale = sh_gate_scale;
            s2 = sh_gate_s2; C = sh_gate_out + (unsigned long long)token * N;
        } else {
            if (sh_up_packed == 0) {
                const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
                __nv_bfloat16* z = sh_up_out + (unsigned long long)token * N;
                for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N;
                     i += BLOCK_SIZE) {
                    z[n_base + i] = __float2bfloat16(0.0f);
                }
                return;
            }
            B_packed = sh_up_packed; B_scale = sh_up_scale;
            s2 = sh_up_s2; C = sh_up_out + (unsigned long long)token * N;
        }
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id];
            C = gate_out + (unsigned long long)flat_slot * N;
        } else {
            B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id];
            C = up_out + (unsigned long long)flat_slot * N;
        }
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int row3 = (K * 3) / 8;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    __shared__ float s_lut8[8];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_W3F[threadIdx.x];
    if (threadIdx.x >= 16 && threadIdx.x < 24) s_lut8[threadIdx.x - 16] = w3_lut[threadIdx.x - 16];
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        uint4 a_data = ((const uint4*)A_token)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int base_k = k8 * 8;

        const unsigned int sg = base_k / GROUP_SIZE;
        const unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        const float sc1 = atlas_dec_e4m3(sb1) * s2;
        const unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        const float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;

        float w1[8], w2[8];
        if (is_shared) {
            unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
            unsigned int packed4_2 = have_n2 ?
                *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
                w1[b * 2] = s_lut[bv1 & 0xF] * sc1;
                w1[b * 2 + 1] = s_lut[bv1 >> 4] * sc1;
                unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
                w2[b * 2] = s_lut[bv2 & 0xF] * sc2;
                w2[b * 2 + 1] = s_lut[bv2 >> 4] * sc2;
            }
        } else {
            const unsigned int t1 = w3_load_trio(B_packed + (unsigned long long)n1 * row3 + k8 * 3);
            w3_dequant_trio(t1, s_lut8, sc1, w1);
            if (have_n2) {
                const unsigned int t2 = w3_load_trio(B_packed + (unsigned long long)n2 * row3 + k8 * 3);
                w3_dequant_trio(t2, s_lut8, sc2, w2);
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) w2[j] = 0.0f;
            }
        }

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            __nv_bfloat16 al, ah;
            *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
            float afl = __bfloat162float(al), afh = __bfloat162float(ah);
            acc1 += afl * w1[b * 2] + afh * w1[b * 2 + 1];
            acc2 += afl * w2[b * 2] + afh * w2[b * 2 + 1];
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) C[n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) C[n2] = __float2bfloat16(acc2);
    }
}

extern "C" __global__ void moe_expert_silu_down_shared_batchN_w3(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int num_tokens,
    const float* __restrict__ w3_lut
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);

    unsigned int token, expert_slot;
    if (is_shared) {
        token = y - total_routed;
        expert_slot = 0;
    } else {
        token = y / top_k;
        expert_slot = y % top_k;
    }

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;

    if (is_shared) {
        if (sh_down_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            __nv_bfloat16* z = sh_down_out + (unsigned long long)token * N;
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N;
                 i += BLOCK_SIZE) {
                z[n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
        B_packed = sh_down_packed; B_scale = sh_down_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in + (unsigned long long)token * K;
        u_ptr = sh_up_in + (unsigned long long)token * K;
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)flat_slot * K;
        u_ptr = up_out + (unsigned long long)flat_slot * K;
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[(unsigned long long)(token * top_k + expert_slot) * N + n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int row3 = (K * 3) / 8;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    __shared__ float s_lut8[8];
    extern __shared__ float s_act[];

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_W3F[threadIdx.x];
    if (threadIdx.x >= 16 && threadIdx.x < 24) s_lut8[threadIdx.x - 16] = w3_lut[threadIdx.x - 16];

    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;
        const unsigned int sg = base_k / GROUP_SIZE;
        const unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        const float sc1 = atlas_dec_e4m3(sb1) * s2;
        const unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        const float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;

        float w1[8], w2[8];
        if (is_shared) {
            unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
            unsigned int packed4_2 = have_n2 ?
                *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
                w1[b * 2] = s_lut[bv1 & 0xF] * sc1;
                w1[b * 2 + 1] = s_lut[bv1 >> 4] * sc1;
                unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
                w2[b * 2] = s_lut[bv2 & 0xF] * sc2;
                w2[b * 2 + 1] = s_lut[bv2 >> 4] * sc2;
            }
        } else {
            const unsigned int t1 = w3_load_trio(B_packed + (unsigned long long)n1 * row3 + k8 * 3);
            w3_dequant_trio(t1, s_lut8, sc1, w1);
            if (have_n2) {
                const unsigned int t2 = w3_load_trio(B_packed + (unsigned long long)n2 * row3 + k8 * 3);
                w3_dequant_trio(t2, s_lut8, sc2, w2);
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) w2[j] = 0.0f;
            }
        }

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            float al = s_act[base_k + b * 2];
            float ah = s_act[base_k + b * 2 + 1];
            acc1 += al * w1[b * 2] + ah * w1[b * 2 + 1];
            acc2 += al * w2[b * 2] + ah * w2[b * 2 + 1];
        }
    }

    __nv_bfloat16* out = is_shared
        ? (sh_down_out + (unsigned long long)token * N)
        : (C + (unsigned long long)(token * top_k + expert_slot) * N);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) out[n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) out[n2] = __float2bfloat16(acc2);
    }
}

// ============================================================================
// batchN v2 W3 — expert-dedup + M-way token batch (ATLAS_KN_V2), and the v4
// dedup down (ATLAS_KN_V4). Same launch contracts as the NVFP4 v2/v4 kernels.
// W3 routed loads: 12 bytes (3 aligned u32 words = 4 trios = 32 elems) per
// k32 iteration — row3 (= K*3/8) is a multiple of 4 for K % 32 == 0, which
// both Laguna projections satisfy (K = 3072 gate/up, 1024 down).
// ============================================================================
#define V2_MAX_M 8
#define V2_BLOCK 128
#define V2_TPO (V2_BLOCK / N_PER_BLOCK)

__device__ __forceinline__ bool v2_gather_slots_w3(
    const unsigned int* __restrict__ expert_indices,
    unsigned int y, unsigned int total_routed, unsigned int num_tokens,
    unsigned int top_k, bool is_shared,
    unsigned int* s_slot, unsigned int* s_m
) {
    if (is_shared) {
        if (y != total_routed) return false;
        if (threadIdx.x == 0) {
            *s_m = num_tokens;
            for (unsigned int t = 0; t < num_tokens; t++) s_slot[t] = t;
        }
        __syncthreads();
        return true;
    }
    __shared__ int s_is_leader;
    if (threadIdx.x == 0) {
        const unsigned int expert_id = expert_indices[y];
        int leader = 1;
        for (unsigned int s = 0; s < y; s++) {
            if (expert_indices[s] == expert_id) { leader = 0; break; }
        }
        s_is_leader = leader;
        if (leader) {
            unsigned int m = 0;
            for (unsigned int s = y; s < total_routed && m < V2_MAX_M; s++) {
                if (expert_indices[s] == expert_id) s_slot[m++] = s;
            }
            *s_m = m;
        }
    }
    __syncthreads();
    return s_is_leader != 0;
}

// Decode 32 W3 elems (4 trios) from 3 aligned u32 words into f[32] with
// per-16 scales scA (elems 0-15) / scB (elems 16-31).
__device__ __forceinline__ void w3_dequant32(
    const unsigned char* row_base, unsigned int k32,
    const float* s_lut8, float scA, float scB, float* f
) {
    const unsigned int* wp = (const unsigned int*)(row_base + k32 * 12);
    const unsigned int w0 = wp[0], w1 = wp[1], w2 = wp[2];
    const unsigned int t0 = w0 & 0xFFFFFFu;
    const unsigned int t1 = (w0 >> 24) | ((w1 & 0xFFFFu) << 8);
    const unsigned int t2 = (w1 >> 16) | ((w2 & 0xFFu) << 16);
    const unsigned int t3 = w2 >> 8;
    w3_dequant_trio(t0, s_lut8, scA, &f[0]);
    w3_dequant_trio(t1, s_lut8, scA, &f[8]);
    w3_dequant_trio(t2, s_lut8, scB, &f[16]);
    w3_dequant_trio(t3, s_lut8, scB, &f[24]);
}

// Decode 32 NVFP4 elems (uint4 = 32 nibbles) into f[32] (shared-expert path).
__device__ __forceinline__ void nv4_dequant32(
    const unsigned char* row_base, unsigned int k32,
    const float* s_lut16, float scA, float scB, float* f
) {
    const uint4 w = *(const uint4*)(row_base + k32 * 16);
    const unsigned int words[4] = {w.x, w.y, w.z, w.w};
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        const float sc = (g < 2) ? scA : scB;
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            const unsigned char bv = (words[g] >> (b * 8)) & 0xFF;
            f[g * 8 + b * 2] = s_lut16[bv & 0xF] * sc;
            f[g * 8 + b * 2 + 1] = s_lut16[bv >> 4] * sc;
        }
    }
}

extern "C" __global__ void moe_expert_gate_up_shared_batchN_v2_w3(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_packed,
    const unsigned char* __restrict__ sh_gate_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int num_tokens,
    const float* __restrict__ w3_lut
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y >= total_routed);

    __shared__ unsigned int s_slot[V2_MAX_M];
    __shared__ unsigned int s_m;
    if (!v2_gather_slots_w3(expert_indices, y, total_routed, num_tokens, top_k,
                            is_shared, s_slot, &s_m)) return;
    const unsigned int M = s_m;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C_base;

    if (is_shared) {
        if (proj == 0) { B_packed = sh_gate_packed; B_scale = sh_gate_scale;
                         s2 = sh_gate_s2; C_base = sh_gate_out; }
        else           { B_packed = sh_up_packed; B_scale = sh_up_scale;
                         s2 = sh_up_s2; C_base = sh_up_out; }
    } else {
        const unsigned int expert_id = expert_indices[y];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id];
            C_base = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id];
            C_base = up_out;
        }
    }

    if (B_packed == 0) {
        const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
        for (unsigned int m = 0; m < M; m++) {
            __nv_bfloat16* z = C_base + (unsigned long long)s_slot[m] * N;
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N;
                 i += V2_BLOCK) {
                z[n_base + i] = __float2bfloat16(0.0f);
            }
        }
        return;
    }

    const unsigned int local_out = threadIdx.x / V2_TPO;
    const unsigned int lane = threadIdx.x % V2_TPO;
    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int row3 = (K * 3) / 8;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K32 = K / 32;

    __shared__ float s_lut[16];
    __shared__ float s_lut8[8];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_W3F[threadIdx.x];
    if (threadIdx.x >= 16 && threadIdx.x < 24) s_lut8[threadIdx.x - 16] = w3_lut[threadIdx.x - 16];
    __syncthreads();

    float acc1[V2_MAX_M], acc2[V2_MAX_M];
    #pragma unroll
    for (int m = 0; m < V2_MAX_M; m++) { acc1[m] = 0.0f; acc2[m] = 0.0f; }

    for (unsigned int k32 = lane; k32 < K32; k32 += V2_TPO) {
        const unsigned int sg = k32 * 2;
        const float sc1a = atlas_dec_e4m3(B_scale[(unsigned long long)n1 * num_groups + sg]) * s2;
        const float sc1b = atlas_dec_e4m3(B_scale[(unsigned long long)n1 * num_groups + sg + 1]) * s2;
        const float sc2a = have_n2 ?
            atlas_dec_e4m3(B_scale[(unsigned long long)n2 * num_groups + sg]) * s2 : 0.0f;
        const float sc2b = have_n2 ?
            atlas_dec_e4m3(B_scale[(unsigned long long)n2 * num_groups + sg + 1]) * s2 : 0.0f;

        float f1[32], f2[32];
        if (is_shared) {
            nv4_dequant32(B_packed + (unsigned long long)n1 * half_K, k32, s_lut, sc1a, sc1b, f1);
            if (have_n2) {
                nv4_dequant32(B_packed + (unsigned long long)n2 * half_K, k32, s_lut, sc2a, sc2b, f2);
            } else {
                #pragma unroll
                for (int j = 0; j < 32; j++) f2[j] = 0.0f;
            }
        } else {
            w3_dequant32(B_packed + (unsigned long long)n1 * row3, k32, s_lut8, sc1a, sc1b, f1);
            if (have_n2) {
                w3_dequant32(B_packed + (unsigned long long)n2 * row3, k32, s_lut8, sc2a, sc2b, f2);
            } else {
                #pragma unroll
                for (int j = 0; j < 32; j++) f2[j] = 0.0f;
            }
        }

        #pragma unroll
        for (int g = 0; g < 4; g++) {
            const unsigned int elem = k32 * 32 + g * 8;
            #pragma unroll
            for (int m = 0; m < V2_MAX_M; m++) {
                if (m >= (int)M) break;
                const unsigned int token = is_shared ? s_slot[m] : s_slot[m] / top_k;
                const uint4 a = *(const uint4*)(A + (unsigned long long)token * K + elem);
                const unsigned int a_raw[4] = {a.x, a.y, a.z, a.w};
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    __nv_bfloat16 al, ah;
                    *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
                    *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
                    acc1[m] += __bfloat162float(al) * f1[g * 8 + b * 2]
                             + __bfloat162float(ah) * f1[g * 8 + b * 2 + 1];
                    acc2[m] += __bfloat162float(al) * f2[g * 8 + b * 2]
                             + __bfloat162float(ah) * f2[g * 8 + b * 2 + 1];
                }
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < V2_MAX_M; m++) {
        if (m >= (int)M) break;
        float a1 = acc1[m], a2 = acc2[m];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a1 += __shfl_down_sync(0xFFFFFFFF, a1, offset);
            a2 += __shfl_down_sync(0xFFFFFFFF, a2, offset);
        }
        if (lane == 0) {
            __nv_bfloat16* out = C_base + (unsigned long long)s_slot[m] * N;
            out[n1] = __float2bfloat16(a1);
            if (have_n2) out[n2] = __float2bfloat16(a2);
        }
    }
}

extern "C" __global__ void moe_expert_down_dedup_batchN_w3(
    const __nv_bfloat16* __restrict__ act,
    const __nv_bfloat16* __restrict__ sh_act,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int num_tokens,
    const float* __restrict__ w3_lut
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);

    __shared__ unsigned int s_slot[V2_MAX_M];
    __shared__ unsigned int s_m;
    if (!v2_gather_slots_w3(expert_indices, y, total_routed, num_tokens, top_k,
                            is_shared, s_slot, &s_m)) return;
    const unsigned int M = s_m;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C_base;
    const __nv_bfloat16* act_base;

    if (is_shared) {
        B_packed = sh_down_packed; B_scale = sh_down_scale; s2 = sh_down_s2;
        C_base = sh_down_out; act_base = sh_act;
    } else {
        const unsigned int expert_id = expert_indices[y];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        C_base = C; act_base = act;
    }

    if (B_packed == 0) {
        const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
        for (unsigned int m = 0; m < M; m++) {
            __nv_bfloat16* z = C_base + (unsigned long long)s_slot[m] * N;
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N;
                 i += V2_BLOCK) z[n_base + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    const unsigned int local_out = threadIdx.x / V2_TPO;
    const unsigned int lane = threadIdx.x % V2_TPO;
    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int row3 = (K * 3) / 8;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K32 = K / 32;

    __shared__ float s_lut[16];
    __shared__ float s_lut8[8];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_W3F[threadIdx.x];
    if (threadIdx.x >= 16 && threadIdx.x < 24) s_lut8[threadIdx.x - 16] = w3_lut[threadIdx.x - 16];
    __syncthreads();

    float acc1[V2_MAX_M], acc2[V2_MAX_M];
    #pragma unroll
    for (int m = 0; m < V2_MAX_M; m++) { acc1[m] = 0.0f; acc2[m] = 0.0f; }

    for (unsigned int k32 = lane; k32 < K32; k32 += V2_TPO) {
        const unsigned int sg = k32 * 2;
        const float sc1a = atlas_dec_e4m3(B_scale[(unsigned long long)n1 * num_groups + sg]) * s2;
        const float sc1b = atlas_dec_e4m3(B_scale[(unsigned long long)n1 * num_groups + sg + 1]) * s2;
        const float sc2a = have_n2 ? atlas_dec_e4m3(B_scale[(unsigned long long)n2 * num_groups + sg]) * s2 : 0.0f;
        const float sc2b = have_n2 ? atlas_dec_e4m3(B_scale[(unsigned long long)n2 * num_groups + sg + 1]) * s2 : 0.0f;

        float f1[32], f2[32];
        if (is_shared) {
            nv4_dequant32(B_packed + (unsigned long long)n1 * half_K, k32, s_lut, sc1a, sc1b, f1);
            if (have_n2) {
                nv4_dequant32(B_packed + (unsigned long long)n2 * half_K, k32, s_lut, sc2a, sc2b, f2);
            } else {
                #pragma unroll
                for (int j = 0; j < 32; j++) f2[j] = 0.0f;
            }
        } else {
            w3_dequant32(B_packed + (unsigned long long)n1 * row3, k32, s_lut8, sc1a, sc1b, f1);
            if (have_n2) {
                w3_dequant32(B_packed + (unsigned long long)n2 * row3, k32, s_lut8, sc2a, sc2b, f2);
            } else {
                #pragma unroll
                for (int j = 0; j < 32; j++) f2[j] = 0.0f;
            }
        }

        #pragma unroll
        for (int g = 0; g < 4; g++) {
            const unsigned int elem = k32 * 32 + g * 8;
            #pragma unroll
            for (int m = 0; m < V2_MAX_M; m++) {
                if (m >= (int)M) break;
                const unsigned int arow = s_slot[m];
                const uint4 a = *(const uint4*)(act_base + (unsigned long long)arow * K + elem);
                const unsigned int a_raw[4] = {a.x, a.y, a.z, a.w};
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    __nv_bfloat16 al, ah;
                    *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
                    *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
                    acc1[m] += __bfloat162float(al) * f1[g * 8 + b * 2]
                             + __bfloat162float(ah) * f1[g * 8 + b * 2 + 1];
                    acc2[m] += __bfloat162float(al) * f2[g * 8 + b * 2]
                             + __bfloat162float(ah) * f2[g * 8 + b * 2 + 1];
                }
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < V2_MAX_M; m++) {
        if (m >= (int)M) break;
        float a1 = acc1[m], a2 = acc2[m];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a1 += __shfl_down_sync(0xFFFFFFFF, a1, offset);
            a2 += __shfl_down_sync(0xFFFFFFFF, a2, offset);
        }
        if (lane == 0) {
            __nv_bfloat16* out = C_base + (unsigned long long)s_slot[m] * N;
            out[n1] = __float2bfloat16(a1);
            if (have_n2) out[n2] = __float2bfloat16(a2);
        }
    }
}
