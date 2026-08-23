// SPDX-License-Identifier: AGPL-3.0-only

// M8/M17 exact-FFN fast path. The pointwise stage materializes the activation
// after the gate/up projections have already been rounded to BF16. The down
// GEMV then preserves the K1 K8 lane ownership and reduction tree while
// avoiding a repeated SiLU evaluation for every output column.

template <int MAX_M>
__device__ __forceinline__ void w4a16_dual_silu_f32_exact_body(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    float* __restrict__ activation,
    unsigned int M, unsigned int K)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = M <= (unsigned int)MAX_M ? M * K : 0u;
    if (idx >= elements) return;

    // These BF16 loads are the required projection-output rounding boundary.
    const __nv_bfloat16 gate_bf16 = gate_out[idx];
    const __nv_bfloat16 up_bf16 = up_out[idx];
    const float gate = __bfloat162float(gate_bf16);
    const float up = __bfloat162float(up_bf16);
    activation[idx] = (gate / (1.0f + __expf(-gate))) * up;
}

template <int MAX_M>
__device__ __forceinline__ void w4a16_f32_input_exact_body(
    const float* __restrict__ A,
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
    const bool valid = M > 0u && M <= (unsigned int)MAX_M && n < N;

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED_W4[threadIdx.x];
    __syncthreads();

    float acc[MAX_M];
    #pragma unroll
    for (int row = 0; row < MAX_M; ++row) acc[row] = 0.0f;

    if (valid) {
        const unsigned int half_K = K / 2;
        const unsigned int num_groups = K / GROUP_SIZE;
        const unsigned int K8 = K / 8;
        const unsigned long long weight_row = (unsigned long long)n * half_K;
        const unsigned long long scale_row = (unsigned long long)n * num_groups;

        // Software pipeline: prefetch tile k8+64's weight+scale while
        // accumulating tile k8 (bit-exact: same addresses and FMA order,
        // only the load issue is hoisted earlier).
        unsigned int packed4 = 0u;
        unsigned char scale_byte = 0u;
        if (lane < K8) {
            packed4 = *(const unsigned int*)(
                B_packed + weight_row + (unsigned long long)lane * 4);
            scale_byte = B_scale[scale_row
                + ((unsigned long long)lane * 8) / GROUP_SIZE];
        }
        for (unsigned int k8 = lane; k8 < K8; k8 += 64u) {
            const unsigned int next_k8 = k8 + 64u;
            unsigned int next_packed4 = 0u;
            unsigned char next_scale_byte = 0u;
            if (next_k8 < K8) {
                next_packed4 = *(const unsigned int*)(
                    B_packed + weight_row + (unsigned long long)next_k8 * 4);
                next_scale_byte = B_scale[scale_row
                    + ((unsigned long long)next_k8 * 8) / GROUP_SIZE];
            }
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
                    const float* A_row = A + (unsigned long long)row * K;
                    const float4 a_lo = ((const float4*)A_row)[k8 * 2];
                    const float4 a_hi = ((const float4*)A_row)[k8 * 2 + 1];
                    const float a[8] = {
                        a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                        a_hi.x, a_hi.y, a_hi.z, a_hi.w
                    };
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        acc[row] += a[b * 2] * w_lo[b];
                        acc[row] += a[b * 2 + 1] * w_hi[b];
                    }
                }
            }
            packed4 = next_packed4;
            scale_byte = next_scale_byte;
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
        #pragma unroll
        for (int row = 0; row < MAX_M; ++row) {
            if ((unsigned int)row < M) {
                const unsigned int base = (row * N_PER_BLOCK + local_out) * 2;
                C[(unsigned long long)row * N + n] =
                    __float2bfloat16(smem[base] + smem[base + 1]);
            }
        }
    }
}

#define DEFINE_W4A16_DUAL_SILU_F32_EXACT(NAME, MAX_ROWS)                    \
extern "C" __global__ void NAME(                                             \
    const __nv_bfloat16* gate_out, const __nv_bfloat16* up_out,              \
    float* activation, unsigned int M, unsigned int K)                       \
{                                                                            \
    w4a16_dual_silu_f32_exact_body<MAX_ROWS>(                                \
        gate_out, up_out, activation, M, K);                                 \
}

#define DEFINE_W4A16_F32_INPUT_EXACT(NAME, MAX_ROWS)                        \
extern "C" __global__ void NAME(                                             \
    const float* A, const unsigned char* B_packed,                           \
    const unsigned char* B_scale, const float scale2,                        \
    __nv_bfloat16* C, unsigned int M, unsigned int N, unsigned int K)        \
{                                                                            \
    __shared__ float s_lut[16];                                               \
    __shared__ float smem[(MAX_ROWS) * N_PER_BLOCK * 2];                      \
    w4a16_f32_input_exact_body<MAX_ROWS>(                                    \
        A, B_packed, B_scale, scale2, C, M, N, K, s_lut, smem);             \
}

DEFINE_W4A16_DUAL_SILU_F32_EXACT(
    w4a16_gemv_dual_silu_f32_exact_m8, 8)
DEFINE_W4A16_DUAL_SILU_F32_EXACT(
    w4a16_gemv_dual_silu_f32_exact_m17, 17)
DEFINE_W4A16_F32_INPUT_EXACT(w4a16_gemv_f32_input_exact_m8, 8)
DEFINE_W4A16_F32_INPUT_EXACT(w4a16_gemv_f32_input_exact_m17, 17)

#undef DEFINE_W4A16_DUAL_SILU_F32_EXACT
#undef DEFINE_W4A16_F32_INPUT_EXACT
