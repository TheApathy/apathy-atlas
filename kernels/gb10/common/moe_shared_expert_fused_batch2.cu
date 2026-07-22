// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Fused MoE Expert+Shared GEMV — K=2 multi-token batch variant.
//
// Processes 2 tokens through MoE in single kernel launches by expanding
// blockIdx.y to accommodate 2 sets of (top_k routed + 1 shared) experts.
// Weights are loaded once and applied to both tokens' inputs, halving
// weight bandwidth for the shared expert and gate projection.
//
// Token layout in blockIdx.y:
//   y ∈ [0, 2*top_k)         → routed experts (token = y/top_k, slot = y%top_k)
//   y ∈ [2*top_k, 2*top_k+2) → shared expert  (token = y - 2*top_k)
//
// Grid: gate_up_batch2  (ceil(N/8), 2*(top_k+1), 2)
//       silu_down_batch2 (ceil(N/8), 2*(top_k+1), 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_BATCH2[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// NVFP4 per-block FP8-E4M3 scale decode. SCALE/gfx1151 `(float)__nv_fp8_e4m3`
// is NON-STANDARD (same bug fixed in moe_sorted_prefill.cu / the decode GEMVs) —
// software scl_fp8 there; NVIDIA path is the verbatim cast.
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

// ── Fused Gate+Up 2x with shared expert — K=2 batch variant ──
//
// Grid: (ceil(N/8), 2*(top_k+1), 2)  Block: (128, 1, 1)
// blockIdx.y: 0..2*top_k-1 = routed (token=y/top_k, slot=y%top_k)
//             2*top_k..2*top_k+1 = shared (token=y-2*top_k)
extern "C" __global__ void moe_expert_gate_up_shared_batch2(
    const __nv_bfloat16* __restrict__ A,       // [2, H] BF16 input (2 tokens)
    // Routed expert tables
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,      // [2*top_k, inter] BF16
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,        // [2*top_k, inter] BF16
    const unsigned int* __restrict__ expert_indices,  // [2*top_k] u32
    // Shared expert direct pointers
    const unsigned char* __restrict__ sh_gate_packed,
    const unsigned char* __restrict__ sh_gate_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,   // [2, inter] BF16
    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,     // [2, inter] BF16
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int total_routed = 2 * top_k;
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y >= total_routed);

    // Determine token index and expert slot
    unsigned int token, expert_slot;
    if (is_shared) {
        token = y - total_routed;  // 0 or 1
        expert_slot = 0;           // unused for shared
    } else {
        token = y / top_k;         // 0 or 1
        expert_slot = y % top_k;   // 0..top_k-1
    }

    // Select input for this token
    const __nv_bfloat16* A_token = A + (unsigned long long)token * K;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;

    if (is_shared) {
        // NULL shared-expert weights = "no in-kernel shared expert": zero the
        // output rows and exit, exactly as the _t variant does
        // (moe_shared_expert_fused_batch2_t.cu:85-100). A model whose shared
        // expert is a DIFFERENT precision from its routed experts (Laguna:
        // NVFP4 routed + BF16 shared) passes NULL here and computes the shared
        // half separately; without this guard that dereferences NULL and the
        // kernel faults the moment a 2-sequence batch forms.
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
        // EP: NULL pointer means remote expert — write zero output and return
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
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH2[threadIdx.x];
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        uint4 a_data = ((const uint4*)A_token)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int base_k = k8 * 8;

        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        float sc1 = atlas_dec_e4m3(sb1) * s2;

        unsigned int packed4_2 = have_n2 ?
            *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;
            __nv_bfloat16 al, ah;
            *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
            float afl = __bfloat162float(al), afh = __bfloat162float(ah);
            acc1 += afl * w1l + afh * w1h;
            acc2 += afl * w2l + afh * w2h;
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

// ── Fused SiLU+Down 2x with shared expert — K=2 batch variant ──
//
// Grid: (ceil(N/8), 2*(top_k+1), 1)  Block: (128, 1, 1)
extern "C" __global__ void moe_expert_silu_down_shared_batch2(
    const __nv_bfloat16* __restrict__ gate_out,  // [2*top_k, inter] BF16
    const __nv_bfloat16* __restrict__ up_out,    // [2*top_k, inter] BF16
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,               // [2*top_k, H] BF16
    const unsigned int* __restrict__ expert_indices,  // [2*top_k] u32
    // Shared expert
    const __nv_bfloat16* __restrict__ sh_gate_in,  // [2, inter] BF16
    const __nv_bfloat16* __restrict__ sh_up_in,    // [2, inter] BF16
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,       // [2, H] BF16
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int total_routed = 2 * top_k;
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
        // NULL shared-expert down weight = no in-kernel shared expert (see the
        // gate_up kernel above). Zero this token's shared output rows and exit
        // rather than dereferencing NULL; the caller supplies the shared half
        // separately. Mirrors moe_shared_expert_fused_batch2_t.cu:188.
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
        // EP: NULL pointer means remote expert — write zero output and return
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
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    // Dynamic: K floats, sized by the launcher (issue #85 -- the old
    // static s_act[1024] overflowed for Mistral-Small-4's
    // expert_hidden_dim=2048, illegal-addressing on the first batched
    // K=2 FFN; matches the extern pattern the _t variant already uses).
    extern __shared__ float s_act[];

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH2[threadIdx.x];

    // Phase 1: Precompute SiLU(gate)*up into shared memory
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();

    // Phase 2: GEMV reading precomputed activation from shared memory
    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        float sc1 = atlas_dec_e4m3(sb1) * s2;

        unsigned int packed4_2 = have_n2 ?
            *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            float al = s_act[base_k + b * 2];
            float ah = s_act[base_k + b * 2 + 1];

            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;

            acc1 += al * w1l + ah * w1h;
            acc2 += al * w2l + ah * w2h;
        }
    }

    // Output: shared at sh_down_out[token*N], routed at C[flat_slot*N]
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
// batchN (non-transposed) variants for forward_k16 wide DFlash verify. Byte-
// identical per-block math to batch2 above; `num_tokens` replaces the hardcoded
// 2 so all num_tokens*(top_k+1) blocks launch in one grid (fills the GPU vs the
// per-token loop's serial launches). FAITHFUL → preserves speculative acceptance.
// ============================================================================
extern "C" __global__ void moe_expert_gate_up_shared_batchN(
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
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int num_tokens
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
            B_packed = sh_gate_packed; B_scale = sh_gate_scale;
            s2 = sh_gate_s2; C = sh_gate_out + (unsigned long long)token * N;
        } else {
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
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH2[threadIdx.x];
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        uint4 a_data = ((const uint4*)A_token)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int base_k = k8 * 8;

        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        float sc1 = atlas_dec_e4m3(sb1) * s2;

        unsigned int packed4_2 = have_n2 ?
            *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;
            __nv_bfloat16 al, ah;
            *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
            float afl = __bfloat162float(al), afh = __bfloat162float(ah);
            acc1 += afl * w1l + afh * w1h;
            acc2 += afl * w2l + afh * w2h;
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

extern "C" __global__ void moe_expert_silu_down_shared_batchN(
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
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int num_tokens
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
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    extern __shared__ float s_act[];

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH2[threadIdx.x];

    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        float sc1 = atlas_dec_e4m3(sb1) * s2;

        unsigned int packed4_2 = have_n2 ?
            *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            float al = s_act[base_k + b * 2];
            float ah = s_act[base_k + b * 2 + 1];

            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;

            acc1 += al * w1l + ah * w1h;
            acc2 += al * w2l + ah * w2h;
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
// batchN v2 — expert-dedup + M-way token batch + uint4 weight loads.
//
// Same launch contract as batchN (grid (ceil(N/8), num_tokens*(top_k+1), 2/1),
// same argument list) so the Rust launcher is reused verbatim; only the block
// size differs (128 — the v1 BLOCK_SIZE macro, so threads_per_out stays 32 and
// no thread duplicates a neighbor block's rows the way v1 does at blockDim 256).
//
// Bandwidth changes vs v1 (the 1.5-1.9x-off-floor GEMV):
//   1. DEDUP: the FIRST flat_slot whose expert_id matches becomes the leader
//      and computes ALL slots routed to that expert (M-way FMA per weight
//      load); later duplicate slots exit immediately. 8 tokens x top-10 of 256
//      experts => ~88 expert-weight reads collapse to ~#unique (~48).
//      The shared expert is read ONCE (leader = first shared block row) for
//      all num_tokens tokens instead of num_tokens times.
//   2. Weights load as uint4 (16B, 32 nibbles) instead of 4B words.
//   3. Nibble->float decode happens ONCE per 8-element word, then FMAs across
//      all M tokens (v1 re-decodes per token block).
//
// Numerics: same LUT/e4m3/s2 dequant and FP32 accumulate; the k-strided lane
// partition changes (k32 vs k8 stride), so results are NOT bit-identical to
// v1 — gate is cosine + text-parity vs base decode, same as the cuBLASLt
// attention ladder (6ac39db3).
//
// V2_MAX_M caps the token fan-out a leader carries in registers; the launcher
// must fall back to v1 when num_tokens > V2_MAX_M (silu_down smem = M*K*4B
// also assumes num_tokens <= 8 at Laguna K=1024 -> 32KB dynamic smem).
// ============================================================================
#define V2_MAX_M 8
#define V2_BLOCK 128
#define V2_TPO (V2_BLOCK / N_PER_BLOCK)  // 32 lanes per output pair

// Leader election + slot gathering, shared by gate_up/silu_down v2. Returns
// false if this block is a duplicate (or out-of-range shared row) and must
// exit. Fills s_m (slot count) and s_slot[] (flat slots for this expert).
__device__ __forceinline__ bool v2_gather_slots(
    const unsigned int* __restrict__ expert_indices,
    unsigned int y, unsigned int total_routed, unsigned int num_tokens,
    unsigned int top_k, bool is_shared,
    unsigned int* s_slot, unsigned int* s_m /* smem, len 1 */
) {
    if (is_shared) {
        // one leader row computes every token's shared projection
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

extern "C" __global__ void moe_expert_gate_up_shared_batchN_v2(
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
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int num_tokens
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y >= total_routed);

    __shared__ unsigned int s_slot[V2_MAX_M];
    __shared__ unsigned int s_m;
    if (!v2_gather_slots(expert_indices, y, total_routed, num_tokens, top_k,
                         is_shared, s_slot, &s_m)) return;
    const unsigned int M = s_m;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C_base;      // routed: gate_out/up_out (row = flat_slot)
    __nv_bfloat16* C_sh;        // shared: sh_*_out (row = token)

    if (is_shared) {
        if (proj == 0) { B_packed = sh_gate_packed; B_scale = sh_gate_scale;
                         s2 = sh_gate_s2; C_sh = sh_gate_out; }
        else           { B_packed = sh_up_packed; B_scale = sh_up_scale;
                         s2 = sh_up_s2; C_sh = sh_up_out; }
        C_base = C_sh;
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
        C_sh = C_base;
    }

    // NULL weights (EP remote expert, or absent shared half): zero every
    // covered row and exit — v1 semantics extended to all M slots.
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

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K32 = K / 32;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH2[threadIdx.x];
    __syncthreads();

    float acc1[V2_MAX_M], acc2[V2_MAX_M];
    #pragma unroll
    for (int m = 0; m < V2_MAX_M; m++) { acc1[m] = 0.0f; acc2[m] = 0.0f; }

    for (unsigned int k32 = lane; k32 < K32; k32 += V2_TPO) {
        const uint4 w1 = *(const uint4*)(B_packed + (unsigned long long)n1 * half_K + k32 * 16);
        const uint4 w2 = have_n2 ?
            *(const uint4*)(B_packed + (unsigned long long)n2 * half_K + k32 * 16)
            : make_uint4(0u, 0u, 0u, 0u);
        const unsigned int words1[4] = {w1.x, w1.y, w1.z, w1.w};
        const unsigned int words2[4] = {w2.x, w2.y, w2.z, w2.w};
        const unsigned int sg = k32 * 2;  // 32 elems = 2 scale groups
        const float sc1a = atlas_dec_e4m3(B_scale[(unsigned long long)n1 * num_groups + sg]) * s2;
        const float sc1b = atlas_dec_e4m3(B_scale[(unsigned long long)n1 * num_groups + sg + 1]) * s2;
        const float sc2a = have_n2 ?
            atlas_dec_e4m3(B_scale[(unsigned long long)n2 * num_groups + sg]) * s2 : 0.0f;
        const float sc2b = have_n2 ?
            atlas_dec_e4m3(B_scale[(unsigned long long)n2 * num_groups + sg + 1]) * s2 : 0.0f;

        #pragma unroll
        for (int g = 0; g < 4; g++) {  // 4 words x 8 elems = 32 hidden elems
            const float scA = (g < 2) ? sc1a : sc1b;
            const float scB = (g < 2) ? sc2a : sc2b;
            float f1[8], f2[8];
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                const unsigned char bv1 = (words1[g] >> (b * 8)) & 0xFF;
                f1[b * 2] = s_lut[bv1 & 0xF] * scA;
                f1[b * 2 + 1] = s_lut[bv1 >> 4] * scA;
                const unsigned char bv2 = (words2[g] >> (b * 8)) & 0xFF;
                f2[b * 2] = s_lut[bv2 & 0xF] * scB;
                f2[b * 2 + 1] = s_lut[bv2 >> 4] * scB;
            }
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
                    acc1[m] += __bfloat162float(al) * f1[b * 2] + __bfloat162float(ah) * f1[b * 2 + 1];
                    acc2[m] += __bfloat162float(al) * f2[b * 2] + __bfloat162float(ah) * f2[b * 2 + 1];
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

// silu_down v2: each block covers V2D_ROWS output rows (vs v1's 8) so the
// [M, K] activation staging amortizes: 24 blocks/expert at Laguna N=3072
// instead of 384, cutting act re-read traffic ~16x. Each warp walks
// V2D_PAIRS_PER_WARP row-pairs sequentially over the staged s_act.
// Grid.x must be ceil(N / V2D_ROWS) (the Rust launcher passes a v2 flag).
#define V2D_PAIRS_PER_WARP 16
#define V2D_WARPS (V2_BLOCK / WARP_SIZE)                    // 4
#define V2D_ROWS (V2D_WARPS * V2D_PAIRS_PER_WARP * 2)       // 128
extern "C" __global__ void moe_expert_silu_down_shared_batchN_v2(
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
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int num_tokens
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);

    __shared__ unsigned int s_slot[V2_MAX_M];
    __shared__ unsigned int s_m;
    if (!v2_gather_slots(expert_indices, y, total_routed, num_tokens, top_k,
                         is_shared, s_slot, &s_m)) return;
    const unsigned int M = s_m;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C_base;

    if (is_shared) {
        B_packed = sh_down_packed; B_scale = sh_down_scale; s2 = sh_down_s2;
        C_base = sh_down_out;
    } else {
        const unsigned int expert_id = expert_indices[y];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        C_base = C;
    }

    const unsigned int row_base = blockIdx.x * V2D_ROWS;
    if (B_packed == 0) {
        for (unsigned int m = 0; m < M; m++) {
            __nv_bfloat16* z = C_base + (unsigned long long)s_slot[m] * N;
            for (unsigned int i = threadIdx.x; i < V2D_ROWS && row_base + i < N;
                 i += V2_BLOCK) {
                z[row_base + i] = __float2bfloat16(0.0f);
            }
        }
        return;
    }

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K32 = K / 32;

    __shared__ float s_lut[16];
    // [M, K] silu(gate)*up, staged once per block (launcher sizes this at
    // num_tokens*K*4 bytes; 32KB at Laguna K=1024, num_tokens<=8).
    extern __shared__ float s_act[];

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH2[threadIdx.x];

    // Vectorized fill: each thread produces 8 act values from one uint4 of
    // gate + one of up.
    for (unsigned int i8 = threadIdx.x; i8 < M * K / 8; i8 += V2_BLOCK) {
        const unsigned int m = i8 / (K / 8);
        const unsigned int k8 = i8 % (K / 8);
        const unsigned int slot = s_slot[m];
        const __nv_bfloat16* g_ptr = is_shared
            ? sh_gate_in + (unsigned long long)slot * K
            : gate_out + (unsigned long long)slot * K;
        const __nv_bfloat16* u_ptr = is_shared
            ? sh_up_in + (unsigned long long)slot * K
            : up_out + (unsigned long long)slot * K;
        const uint4 gv = ((const uint4*)g_ptr)[k8];
        const uint4 uv = ((const uint4*)u_ptr)[k8];
        const unsigned int g_raw[4] = {gv.x, gv.y, gv.z, gv.w};
        const unsigned int u_raw[4] = {uv.x, uv.y, uv.z, uv.w};
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            __nv_bfloat16 gl, gh, ul, uh;
            *(unsigned short*)&gl = (unsigned short)(g_raw[b] & 0xFFFF);
            *(unsigned short*)&gh = (unsigned short)(g_raw[b] >> 16);
            *(unsigned short*)&ul = (unsigned short)(u_raw[b] & 0xFFFF);
            *(unsigned short*)&uh = (unsigned short)(u_raw[b] >> 16);
            const float gf0 = __bfloat162float(gl), gf1 = __bfloat162float(gh);
            s_act[m * K + k8 * 8 + b * 2] =
                (gf0 / (1.0f + __expf(-gf0))) * __bfloat162float(ul);
            s_act[m * K + k8 * 8 + b * 2 + 1] =
                (gf1 / (1.0f + __expf(-gf1))) * __bfloat162float(uh);
        }
    }
    __syncthreads();

    // 8 lanes per output row: 4 independent row streams per warp, width-8
    // shuffle reductions (3 hops vs 5 full-warp), 128B-contiguous weight
    // transactions per row. Each warp owns a 32-row strip and walks it in 8
    // passes of 4 rows.
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int sub = lane / 8;  // row group within warp (0..3)
    const unsigned int l8 = lane % 8;

    #pragma unroll
    for (unsigned int pass = 0; pass < 8; pass++) {
        const unsigned int n = row_base + warp * 32 + pass * 4 + sub;
        if (n >= N) break;

        float acc[V2_MAX_M];
        #pragma unroll
        for (int m = 0; m < V2_MAX_M; m++) acc[m] = 0.0f;

        for (unsigned int k32 = l8; k32 < K32; k32 += 8) {
            const uint4 w = *(const uint4*)(B_packed + (unsigned long long)n * half_K + k32 * 16);
            const unsigned int words[4] = {w.x, w.y, w.z, w.w};
            const unsigned int sg = k32 * 2;
            const float sca = atlas_dec_e4m3(B_scale[(unsigned long long)n * num_groups + sg]) * s2;
            const float scb = atlas_dec_e4m3(B_scale[(unsigned long long)n * num_groups + sg + 1]) * s2;

            #pragma unroll
            for (int g = 0; g < 4; g++) {
                const float sc = (g < 2) ? sca : scb;
                float f[8];
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    const unsigned char bv = (words[g] >> (b * 8)) & 0xFF;
                    f[b * 2] = s_lut[bv & 0xF] * sc;
                    f[b * 2 + 1] = s_lut[bv >> 4] * sc;
                }
                const unsigned int elem = k32 * 32 + g * 8;
                #pragma unroll
                for (int m = 0; m < V2_MAX_M; m++) {
                    if (m >= (int)M) break;
                    const float* am = s_act + m * K + elem;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) acc[m] += am[j] * f[j];
                }
            }
        }

        #pragma unroll
        for (int m = 0; m < V2_MAX_M; m++) {
            if (m >= (int)M) break;
            float a = acc[m];
            a += __shfl_down_sync(0xFFFFFFFF, a, 4, 8);
            a += __shfl_down_sync(0xFFFFFFFF, a, 2, 8);
            a += __shfl_down_sync(0xFFFFFFFF, a, 1, 8);
            if (l8 == 0) {
                __nv_bfloat16* out = C_base + (unsigned long long)s_slot[m] * N;
                out[n] = __float2bfloat16(a);
            }
        }
    }
}

// ── Weighted sum + sigmoid blend — K=2 batch variant ──
//
// Combines routed expert outputs with shared expert via sigmoid gate.
// blockIdx.y = token index (0 or 1).
//
// Grid: (ceil(hidden/256), 2, 1)  Block: (256, 1, 1)
extern "C" __global__ void moe_weighted_sum_blend_batch2(
    __nv_bfloat16* __restrict__ output,              // [2, hidden] BF16
    const __nv_bfloat16* __restrict__ expert_out,    // [2*top_k, hidden] BF16
    const float* __restrict__ expert_weights,         // [2*top_k] f32
    const __nv_bfloat16* __restrict__ shared_out,    // [2, hidden] BF16
    const __nv_bfloat16* __restrict__ input,         // [2, K] BF16 (MoE input)
    const __nv_bfloat16* __restrict__ gate_weight,   // [1, K] BF16 (shared gate)
    unsigned int hidden,
    unsigned int top_k,
    unsigned int K
) {
    const unsigned int token = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;

    // Per-token input pointer
    const __nv_bfloat16* my_input = input + (unsigned long long)token * K;
    const float* my_weights = expert_weights + token * top_k;
    const __nv_bfloat16* my_expert_out = expert_out + (unsigned long long)token * top_k * hidden;
    const __nv_bfloat16* my_shared_out = shared_out + (unsigned long long)token * hidden;
    __nv_bfloat16* my_output = output + (unsigned long long)token * hidden;

    // ── Phase 1: Compute gate scalar (dot product + sigmoid) ──
    // NULL gate_weight = no gate modulation → sigmoid=1.0 (shared expert always
    // on). Matches the per-token moe_weighted_sum_blend; models with an
    // ungated shared expert (Laguna, Mistral) pass a NULL pointer here.
    __shared__ float s_warp_sums[8];
    __shared__ float sigmoid_val;

    if (gate_weight == 0) {
        if (tid == 0) sigmoid_val = 1.0f;
        __syncthreads();
    } else {

    float dot_acc = 0.0f;
    unsigned int K8 = K / 8;
    for (unsigned int k8 = tid; k8 < K8; k8 += 256) {
        uint4 a_data = ((const uint4*)my_input)[k8];
        // Null shared-expert gate (e.g. Laguna has no shared_expert_gate): read 0
        // so the dot product is 0; sigmoid_val is forced to 1.0 below (ungated add,
        // matching the single-token moe_weighted_sum_blend `weight.0==0` fallback).
        uint4 w_data = (gate_weight != nullptr) ? ((const uint4*)gate_weight)[k8]
                                                : make_uint4(0u, 0u, 0u, 0u);
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int w_raw[4] = {w_data.x, w_data.y, w_data.z, w_data.w};

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            __nv_bfloat16 a_lo, a_hi, w_lo, w_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            *(unsigned short*)&w_lo = (unsigned short)(w_raw[b] & 0xFFFF);
            *(unsigned short*)&w_hi = (unsigned short)(w_raw[b] >> 16);
            dot_acc += __bfloat162float(a_lo) * __bfloat162float(w_lo);
            dot_acc += __bfloat162float(a_hi) * __bfloat162float(w_hi);
        }
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        dot_acc += __shfl_down_sync(0xFFFFFFFF, dot_acc, offset);
    }
    if (lane == 0) {
        s_warp_sums[warp_id] = dot_acc;
    }
    __syncthreads();

    if (tid == 0) {
        float gate_scalar = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            gate_scalar += s_warp_sums[w];
        }
        sigmoid_val = (gate_weight != nullptr) ? (1.0f / (1.0f + __expf(-gate_scalar))) : 1.0f;
    }
    __syncthreads();

    }  // end else (gate_weight != 0)

    // ── Phase 2: Weighted sum + blend ──
    unsigned int j = blockIdx.x * blockDim.x + tid;
    if (j >= hidden) return;

    float acc = 0.0f;
    for (unsigned int e = 0; e < top_k; e++) {
        acc += my_weights[e] * __bfloat162float(my_expert_out[(unsigned long long)e * hidden + j]);
    }
    acc += sigmoid_val * __bfloat162float(my_shared_out[j]);
    my_output[j] = __float2bfloat16(acc);
}
