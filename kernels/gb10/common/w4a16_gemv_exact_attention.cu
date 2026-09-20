// SPDX-License-Identifier: AGPL-3.0-only

// Exact dynamic-M attention projections for the ordinary gated-Q and dual-KV
// NVFP4 K1 kernels. Each row retains the K1 K8 lane assignment, ordered
// low/high FP32 updates, shuffle tree, cross-warp add, and BF16 rounding.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16
#define ASTAGE_K8_PER_WAVE 64

__device__ __constant__ float E2M1_LUT_EXACT_ATTN[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

template <int MAX_M, bool STAGE_ACTIVATIONS>
__device__ __forceinline__ void w4a16_attention_exact_accumulate_k8(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    unsigned int M,
    unsigned int K,
    unsigned int n,
    unsigned int k8,
    unsigned int lane,
    const float* s_lut,
    const uint4* s_a,
    float* acc)
{
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int base_k = k8 * 8;
    const unsigned long long weight_row = (unsigned long long)n * half_K;
    const unsigned long long scale_row = (unsigned long long)n * num_groups;
    const unsigned int packed4 = *(const unsigned int*)(
        B_packed + weight_row + k8 * 4);
    const unsigned char scale_byte =
        B_scale[scale_row + base_k / GROUP_SIZE];
    __nv_fp8_e4m3 fp8;
    *(unsigned char*)&fp8 = scale_byte;
    const float scale = (float)fp8 * scale2;

    float w_lo[4];
    float w_hi[4];
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        const unsigned char byte_val =
            (unsigned char)(packed4 >> (b * 8));
        w_lo[b] = s_lut[byte_val & 0xF] * scale;
        w_hi[b] = s_lut[byte_val >> 4] * scale;
    }

    #pragma unroll
    for (int row = 0; row < MAX_M; ++row) {
        if ((unsigned int)row < M) {
            const uint4 a_data = STAGE_ACTIVATIONS
                ? s_a[row * ASTAGE_K8_PER_WAVE + lane]
                : ((const uint4*)(A + (unsigned long long)row * K))[k8];
            const unsigned int a_raw[4] = {
                a_data.x, a_data.y, a_data.z, a_data.w
            };
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                __nv_bfloat16 a_lo, a_hi;
                *(unsigned short*)&a_lo =
                    (unsigned short)(a_raw[b] & 0xFFFF);
                *(unsigned short*)&a_hi =
                    (unsigned short)(a_raw[b] >> 16);
                acc[row] += __bfloat162float(a_lo) * w_lo[b];
                acc[row] += __bfloat162float(a_hi) * w_hi[b];
            }
        }
    }
}

template <int MAX_M, bool QG_LAYOUT, bool STAGE_ACTIVATIONS>
__device__ __forceinline__ void w4a16_attention_exact_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int out_stride,
    float* s_lut,
    float* smem,
    uint4* s_a)
{
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int warp_lane = lane % WARP_SIZE;
    const unsigned int warp_idx = lane / WARP_SIZE;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = M > 0u && M <= (unsigned int)MAX_M && n < N;

    if (threadIdx.x < 16) {
        s_lut[threadIdx.x] = E2M1_LUT_EXACT_ATTN[threadIdx.x];
    }
    __syncthreads();

    float acc[MAX_M];
    #pragma unroll
    for (int row = 0; row < MAX_M; ++row) acc[row] = 0.0f;

    const unsigned int K8 = K / 8;
    if (STAGE_ACTIVATIONS) {
        const unsigned int waves =
            (K8 + ASTAGE_K8_PER_WAVE - 1) / ASTAGE_K8_PER_WAVE;
        for (unsigned int wave = 0; wave < waves; ++wave) {
            // Every thread participates, including invalid tail-output groups,
            // so both barriers remain full-CTA and the tile cannot be
            // overwritten while another output group is still consuming it.
            for (unsigned int tile_k8 = threadIdx.x;
                 tile_k8 < (unsigned int)MAX_M * ASTAGE_K8_PER_WAVE;
                 tile_k8 += BLOCK_SIZE) {
                const unsigned int row = tile_k8 / ASTAGE_K8_PER_WAVE;
                const unsigned int local_k8 =
                    tile_k8 - row * ASTAGE_K8_PER_WAVE;
                const unsigned int global_k8 =
                    wave * ASTAGE_K8_PER_WAVE + local_k8;
                if (row < M && global_k8 < K8) {
                    const __nv_bfloat16* A_row =
                        A + (unsigned long long)row * K;
                    s_a[tile_k8] = ((const uint4*)A_row)[global_k8];
                }
            }
            __syncthreads();

            if (valid) {
                const unsigned int global_k8 =
                    wave * ASTAGE_K8_PER_WAVE + lane;
                if (global_k8 < K8) {
                    w4a16_attention_exact_accumulate_k8<MAX_M, true>(
                        A, B_packed, B_scale, scale2, M, K, n, global_k8,
                        lane, s_lut, s_a, acc);
                }
            }
            __syncthreads();
        }
    } else if (valid) {
        const unsigned int K8 = K / 8;

        // Ordinary gated-Q and dual-KV K1 kernels both own K8 chunks by
        // lane=0..63 with stride 64. Keep that association byte-for-byte.
        for (unsigned int k8 = lane; k8 < K8; k8 += 64u) {
            w4a16_attention_exact_accumulate_k8<MAX_M, false>(
                A, B_packed, B_scale, scale2, M, K, n, k8, lane,
                s_lut, s_a, acc);
        }
    }

    #pragma unroll
    for (int row = 0; row < MAX_M; ++row) {
        if ((unsigned int)row < M) {
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                acc[row] += __shfl_down_sync(0xFFFFFFFF, acc[row], offset);
            }
            if (warp_lane == 0) {
                smem[(row * N_PER_BLOCK + local_out) * 2 + warp_idx] = acc[row];
            }
        }
    }
    __syncthreads();

    if (valid && lane == 0) {
        unsigned int out_idx = n;
        if (QG_LAYOUT) {
            const unsigned int group_dim = 2 * head_dim;
            const unsigned int h = n / group_dim;
            const unsigned int idx = n % group_dim;
            const unsigned int q_total = num_heads * head_dim;
            if (idx < head_dim) {
                out_idx = h * head_dim + idx;
            } else {
                out_idx = q_total + h * head_dim + (idx - head_dim);
            }
        }
        #pragma unroll
        for (int row = 0; row < MAX_M; ++row) {
            if ((unsigned int)row < M) {
                const unsigned int base =
                    (row * N_PER_BLOCK + local_out) * 2;
                C[(unsigned long long)row * out_stride + out_idx] =
                    __float2bfloat16(smem[base] + smem[base + 1]);
            }
        }
    }
}

extern "C" __global__ void w4a16_gemv_qg_exact_m4(
    const __nv_bfloat16* A,
    const unsigned char* B_packed,
    const unsigned char* B_scale,
    const float scale2,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int out_stride)
{
    __shared__ float s_lut[16];
    __shared__ float smem[4 * N_PER_BLOCK * 2];
    w4a16_attention_exact_body<4, true, false>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        num_heads, head_dim, out_stride, s_lut, smem, nullptr);
}

extern "C" __global__ void w4a16_gemv_dual_kv_exact_m4(
    const __nv_bfloat16* A,
    const unsigned char* K_packed,
    const unsigned char* K_scale,
    const float K_scale2,
    __nv_bfloat16* K_out,
    const unsigned char* V_packed,
    const unsigned char* V_scale,
    const float V_scale2,
    __nv_bfloat16* V_out,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride)
{
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? K_packed : V_packed;
    const unsigned char* B_scale = proj == 0 ? K_scale : V_scale;
    const float scale2 = proj == 0 ? K_scale2 : V_scale2;
    __nv_bfloat16* C = proj == 0 ? K_out : V_out;
    __shared__ float s_lut[16];
    __shared__ float smem[4 * N_PER_BLOCK * 2];
    w4a16_attention_exact_body<4, false, false>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        0, 0, out_stride, s_lut, smem, nullptr);
}

extern "C" __global__ void w4a16_gemv_qg_exact_m17(
    const __nv_bfloat16* A,
    const unsigned char* B_packed,
    const unsigned char* B_scale,
    const float scale2,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int out_stride)
{
    __shared__ float s_lut[16];
    __shared__ float smem[17 * N_PER_BLOCK * 2];
    w4a16_attention_exact_body<17, true, false>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        num_heads, head_dim, out_stride, s_lut, smem, nullptr);
}

extern "C" __global__ void w4a16_gemv_dual_kv_exact_m17(
    const __nv_bfloat16* A,
    const unsigned char* K_packed,
    const unsigned char* K_scale,
    const float K_scale2,
    __nv_bfloat16* K_out,
    const unsigned char* V_packed,
    const unsigned char* V_scale,
    const float V_scale2,
    __nv_bfloat16* V_out,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride)
{
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? K_packed : V_packed;
    const unsigned char* B_scale = proj == 0 ? K_scale : V_scale;
    const float scale2 = proj == 0 ? K_scale2 : V_scale2;
    __nv_bfloat16* C = proj == 0 ? K_out : V_out;
    __shared__ float s_lut[16];
    __shared__ float smem[17 * N_PER_BLOCK * 2];
    w4a16_attention_exact_body<17, false, false>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        0, 0, out_stride, s_lut, smem, nullptr);
}

// M18..32 exact tier. This deliberately retains the non-staged K1 ownership
// and reduction order; MAX_M only adds independent row accumulators.
extern "C" __global__ void w4a16_gemv_qg_exact_m32(
    const __nv_bfloat16* A,
    const unsigned char* B_packed,
    const unsigned char* B_scale,
    const float scale2,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int out_stride)
{
    __shared__ float s_lut[16];
    __shared__ float smem[32 * N_PER_BLOCK * 2];
    w4a16_attention_exact_body<32, true, false>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        num_heads, head_dim, out_stride, s_lut, smem, nullptr);
}

extern "C" __global__ void w4a16_gemv_dual_kv_exact_m32(
    const __nv_bfloat16* A,
    const unsigned char* K_packed,
    const unsigned char* K_scale,
    const float K_scale2,
    __nv_bfloat16* K_out,
    const unsigned char* V_packed,
    const unsigned char* V_scale,
    const float V_scale2,
    __nv_bfloat16* V_out,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride)
{
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? K_packed : V_packed;
    const unsigned char* B_scale = proj == 0 ? K_scale : V_scale;
    const float scale2 = proj == 0 ? K_scale2 : V_scale2;
    __nv_bfloat16* C = proj == 0 ? K_out : V_out;
    __shared__ float s_lut[16];
    __shared__ float smem[32 * N_PER_BLOCK * 2];
    w4a16_attention_exact_body<32, false, false>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        0, 0, out_stride, s_lut, smem, nullptr);
}

// M17-only activation-staging candidates. Each 512-column BF16 input tile is
// loaded once by the CTA and reused by its four independent output groups.
// The compute lanes still visit global k8 = lane + 64*wave in the parent's
// order; staging changes only where the identical 16 input bytes are read.
extern "C" __global__ void w4a16_gemv_qg_exact_m17_astage(
    const __nv_bfloat16* A,
    const unsigned char* B_packed,
    const unsigned char* B_scale,
    const float scale2,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int out_stride)
{
    __shared__ float s_lut[16];
    __shared__ float smem[17 * N_PER_BLOCK * 2];
    __shared__ uint4 s_a[17 * ASTAGE_K8_PER_WAVE];
    w4a16_attention_exact_body<17, true, true>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        num_heads, head_dim, out_stride, s_lut, smem, s_a);
}

extern "C" __global__ void w4a16_gemv_dual_kv_exact_m17_astage(
    const __nv_bfloat16* A,
    const unsigned char* K_packed,
    const unsigned char* K_scale,
    const float K_scale2,
    __nv_bfloat16* K_out,
    const unsigned char* V_packed,
    const unsigned char* V_scale,
    const float V_scale2,
    __nv_bfloat16* V_out,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride)
{
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? K_packed : V_packed;
    const unsigned char* B_scale = proj == 0 ? K_scale : V_scale;
    const float scale2 = proj == 0 ? K_scale2 : V_scale2;
    __nv_bfloat16* C = proj == 0 ? K_out : V_out;
    __shared__ float s_lut[16];
    __shared__ float smem[17 * N_PER_BLOCK * 2];
    __shared__ uint4 s_a[17 * ASTAGE_K8_PER_WAVE];
    w4a16_attention_exact_body<17, false, true>(
        A, B_packed, B_scale, scale2, C, M, N, K,
        0, 0, out_stride, s_lut, smem, s_a);
}
