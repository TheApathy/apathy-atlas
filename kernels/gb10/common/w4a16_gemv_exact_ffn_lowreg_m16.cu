// SPDX-License-Identifier: AGPL-3.0-only

// Exact full-M16 single gate/up projections. Keeping one projection per CTA
// shortens register lifetimes while retaining ordinary K1 K8 association and
// the BF16 projection-output rounding boundary.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_EXACT_FFN_LOWREG[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ void w4a16_exact_ffn_single_m16_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K,
    float* s_lut, float* smem)
{
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int warp_lane = lane % WARP_SIZE;
    const unsigned int warp_idx = lane / WARP_SIZE;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = M == 16u && n < N;

    if (threadIdx.x < 16) {
        s_lut[threadIdx.x] = E2M1_LUT_EXACT_FFN_LOWREG[threadIdx.x];
    }
    __syncthreads();

    float acc[16];
    #pragma unroll
    for (int row = 0; row < 16; ++row) acc[row] = 0.0f;

    if (valid) {
        const unsigned int half_K = K / 2;
        const unsigned int num_groups = K / GROUP_SIZE;
        const unsigned int K8 = K / 8;
        const unsigned long long weight_row = (unsigned long long)n * half_K;
        const unsigned long long scale_row = (unsigned long long)n * num_groups;

        for (unsigned int k8 = lane; k8 < K8; k8 += 64u) {
            const unsigned int base_k = k8 * 8;
            const uint4 a_data_base = ((const uint4*)A)[k8];
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
            for (int row = 0; row < 16; ++row) {
                const __nv_bfloat16* A_row =
                    A + (unsigned long long)row * K;
                const uint4 a_data = row == 0
                    ? a_data_base
                    : ((const uint4*)A_row)[k8];
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

    #pragma unroll
    for (int row = 0; row < 16; ++row) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            acc[row] += __shfl_down_sync(0xFFFFFFFF, acc[row], offset);
        }
        if (warp_lane == 0) {
            smem[(row * N_PER_BLOCK + local_out) * 2 + warp_idx] = acc[row];
        }
    }
    __syncthreads();

    if (valid && lane == 0) {
        #pragma unroll
        for (int row = 0; row < 16; ++row) {
            const unsigned int base = (row * N_PER_BLOCK + local_out) * 2;
            C[(unsigned long long)row * N + n] =
                __float2bfloat16(smem[base] + smem[base + 1]);
        }
    }
}

#define DEFINE_W4A16_EXACT_FFN_SINGLE_M16(NAME)                            \
extern "C" __global__ __launch_bounds__(256, 4) void NAME(                 \
    const __nv_bfloat16* A, const unsigned char* B_packed,                 \
    const unsigned char* B_scale, const float scale2,                      \
    __nv_bfloat16* C, unsigned int M, unsigned int N, unsigned int K)      \
{                                                                           \
    __shared__ float s_lut[16];                                             \
    __shared__ float smem[16 * N_PER_BLOCK * 2];                            \
    w4a16_exact_ffn_single_m16_body(                                        \
        A, B_packed, B_scale, scale2, C, M, N, K, s_lut, smem);            \
}

DEFINE_W4A16_EXACT_FFN_SINGLE_M16(w4a16_gemv_gate_exact_m16_lowreg)
DEFINE_W4A16_EXACT_FFN_SINGLE_M16(w4a16_gemv_up_exact_m16_lowreg)

extern "C" __global__ void w4a16_gate_up_materialize_f32_m16(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    float* activation,
    unsigned int M,
    unsigned int N)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = M == 16u ? M * N : 0u;
    if (idx >= elements) return;
    const __nv_bfloat16 gate_bf16 = gate[idx];
    const __nv_bfloat16 up_bf16 = up[idx];
    const float gate_f32 = __bfloat162float(gate_bf16);
    const float up_f32 = __bfloat162float(up_bf16);
    activation[idx] =
        (gate_f32 / (1.0f + __expf(-gate_f32))) * up_f32;
}

#undef DEFINE_W4A16_EXACT_FFN_SINGLE_M16
