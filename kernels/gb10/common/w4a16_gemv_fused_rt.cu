// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A16 register-tiled exact FFN GEMVs (T=2 outputs per lane group).
//
// Register-tiled twins of the two kernels that carry the default M=17 verify
// FFN: `w4a16_gemv_dual_exact_materialize_f32_m17` (fused gate/up projection
// plus FP32 activation materialization) and `w4a16_gemv_f32_input_exact_m17`
// (the down projection). Together these are ~71% of per-step weight traffic,
// so they dominate whether the register-tiling transform moves end-to-end
// tokens/s at all.
//
// Same transform as w4a16_gemv_rt.cu: each 64-lane group covers T ADJACENT
// output rows, so the activation slab for a given (k8, row) is loaded once and
// feeds T independent accumulator chains. Grid shrinks to
// ceil(N / (N_PER_BLOCK * T)).
//
// Bit-exactness contract, unchanged from the baseline bodies:
//   * K8 lane ownership is unchanged: k8 = lane, stride 64, ascending.
//   * Weights are pre-scaled per element (LUT[nibble] * scale) and the two
//     accumulator updates per packed byte stay in (low, high) order. Under the
//     common/ `--fmad=false` build this is a genuine MUL then ADD and must not
//     be rewritten as fmaf().
//   * The per-output five-step shuffle tree and ordered two-warp cross-warp add
//     are replayed independently per (output, row).
//   * The fused kernel keeps its BF16 gate rounding boundary and both
//     per-projection barriers, so the SiLU sees the same rounded operands.
//   * Tail groups never return early; they enter every barrier with zero
//     accumulators while global loads/stores stay predicated.
// Nothing above depends on T, so acc[o][row] sees the identical operand
// sequence the baseline would produce for output row n0 + o.
//
// The down projection reads FP32 activations (32 B per (k8, row) against 4.5 B
// of weight), so at M=17 it moves ~121x more activation than weight bytes and
// is the most L1-bound kernel in the model — the strongest rt2 candidate here.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

// Must stay byte-identical to E2M1_LUT_FUSED_W4 in w4a16_gemv_fused.cu.
__device__ __constant__ float E2M1_LUT_FUSED_RT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ── Fused gate/up projection + FP32 activation materialization ──────────
template <int MAX_M, int T>
__device__ __forceinline__ void w4a16_dual_exact_materialize_f32_rt_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ gate_packed,
    const unsigned char* __restrict__ gate_scale,
    const float gate_scale2,
    const unsigned char* __restrict__ up_packed,
    const unsigned char* __restrict__ up_scale,
    const float up_scale2,
    float* __restrict__ activation,
    unsigned int M, unsigned int N, unsigned int K,
    float* s_lut, float* smem, __nv_bfloat16* rounded_gate)
{
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int warp_lane = lane % WARP_SIZE;
    const unsigned int warp_idx = lane / WARP_SIZE;
    const unsigned int n0 = (blockIdx.x * N_PER_BLOCK + local_out) * T;
    const bool rows_valid = M > 0u && M <= (unsigned int)MAX_M;

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED_RT[threadIdx.x];
    __syncthreads();

    // Pass 0 is gate and pass 1 is up. The barrier at the end of each pass
    // makes the BF16 gate rounding boundary visible before up is consumed.
    #pragma unroll 1
    for (int proj = 0; proj < 2; ++proj) {
        const unsigned char* B_packed = proj == 0 ? gate_packed : up_packed;
        const unsigned char* B_scale = proj == 0 ? gate_scale : up_scale;
        const float scale2 = proj == 0 ? gate_scale2 : up_scale2;

        float acc[T][MAX_M];
        #pragma unroll
        for (int o = 0; o < T; ++o) {
            #pragma unroll
            for (int row = 0; row < MAX_M; ++row) acc[o][row] = 0.0f;
        }

        const unsigned int half_K = K / 2;
        const unsigned int num_groups = K / GROUP_SIZE;
        const unsigned int K8 = K / 8;

        for (unsigned int k8 = lane; k8 < K8; k8 += 64u) {
            const unsigned int base_k = k8 * 8;

            float w_lo[T][4];
            float w_hi[T][4];
            #pragma unroll
            for (int o = 0; o < T; ++o) {
                const unsigned int n = n0 + (unsigned int)o;
                if (rows_valid && n < N) {
                    const unsigned long long weight_row =
                        (unsigned long long)n * half_K;
                    const unsigned long long scale_row =
                        (unsigned long long)n * num_groups;
                    const unsigned int packed4 = *(const unsigned int*)(
                        B_packed + weight_row + k8 * 4);
                    const unsigned char scale_byte =
                        B_scale[scale_row + base_k / GROUP_SIZE];
                    __nv_fp8_e4m3 fp8;
                    *(unsigned char*)&fp8 = scale_byte;
                    const float scale = (float)fp8 * scale2;
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        const unsigned char byte_val =
                            (unsigned char)(packed4 >> (b * 8));
                        w_lo[o][b] = s_lut[byte_val & 0xF] * scale;
                        w_hi[o][b] = s_lut[byte_val >> 4] * scale;
                    }
                } else {
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        w_lo[o][b] = 0.0f;
                        w_hi[o][b] = 0.0f;
                    }
                }
            }

            #pragma unroll
            for (int row = 0; row < MAX_M; ++row) {
                if (rows_valid && (unsigned int)row < M && n0 < N) {
                    const __nv_bfloat16* A_row =
                        A + (unsigned long long)row * K;
                    // ONE activation slab per (k8, row), reused by all T chains.
                    const uint4 a_data = ((const uint4*)A_row)[k8];
                    const unsigned int a_raw[4] = {
                        a_data.x, a_data.y, a_data.z, a_data.w
                    };
                    #pragma unroll
                    for (int o = 0; o < T; ++o) {
                        #pragma unroll
                        for (int b = 0; b < 4; ++b) {
                            __nv_bfloat16 a_lo, a_hi;
                            *(unsigned short*)&a_lo =
                                (unsigned short)(a_raw[b] & 0xFFFF);
                            *(unsigned short*)&a_hi =
                                (unsigned short)(a_raw[b] >> 16);
                            acc[o][row] += __bfloat162float(a_lo) * w_lo[o][b];
                            acc[o][row] += __bfloat162float(a_hi) * w_hi[o][b];
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int o = 0; o < T; ++o) {
            #pragma unroll
            for (int row = 0; row < MAX_M; ++row) {
                if ((unsigned int)row < M) {
                    #pragma unroll
                    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                        acc[o][row] +=
                            __shfl_down_sync(0xFFFFFFFF, acc[o][row], offset);
                    }
                    if (warp_lane == 0) {
                        smem[(((unsigned int)row * N_PER_BLOCK + local_out) *
                              (unsigned int)T + (unsigned int)o) * 2 + warp_idx] =
                            acc[o][row];
                    }
                }
            }
        }
        __syncthreads();

        if (lane == 0) {
            #pragma unroll
            for (int o = 0; o < T; ++o) {
                const unsigned int n = n0 + (unsigned int)o;
                if (!rows_valid || n >= N) continue;
                #pragma unroll
                for (int row = 0; row < MAX_M; ++row) {
                    if ((unsigned int)row < M) {
                        const unsigned int slot =
                            ((unsigned int)row * N_PER_BLOCK + local_out) *
                            (unsigned int)T + (unsigned int)o;
                        const __nv_bfloat16 rounded = __float2bfloat16(
                            smem[slot * 2] + smem[slot * 2 + 1]);
                        if (proj == 0) {
                            rounded_gate[slot] = rounded;
                        } else {
                            const float gate =
                                __bfloat162float(rounded_gate[slot]);
                            const float up = __bfloat162float(rounded);
                            activation[(unsigned long long)row * N + n] =
                                (gate / (1.0f + __expf(-gate))) * up;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

// ── Down projection, FP32 activation input ──────────────────────────────
template <int MAX_M, int T>
__device__ __forceinline__ void w4a16_f32_input_exact_rt_body(
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
    const unsigned int n0 = (blockIdx.x * N_PER_BLOCK + local_out) * T;
    const bool rows_valid = M > 0u && M <= (unsigned int)MAX_M;

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED_RT[threadIdx.x];
    __syncthreads();

    float acc[T][MAX_M];
    #pragma unroll
    for (int o = 0; o < T; ++o) {
        #pragma unroll
        for (int row = 0; row < MAX_M; ++row) acc[o][row] = 0.0f;
    }

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    for (unsigned int k8 = lane; k8 < K8; k8 += 64u) {
        const unsigned int base_k = k8 * 8;

        float w_lo[T][4];
        float w_hi[T][4];
        #pragma unroll
        for (int o = 0; o < T; ++o) {
            const unsigned int n = n0 + (unsigned int)o;
            if (rows_valid && n < N) {
                const unsigned long long weight_row =
                    (unsigned long long)n * half_K;
                const unsigned long long scale_row =
                    (unsigned long long)n * num_groups;
                const unsigned int packed4 = *(const unsigned int*)(
                    B_packed + weight_row + k8 * 4);
                const unsigned char scale_byte =
                    B_scale[scale_row + base_k / GROUP_SIZE];
                __nv_fp8_e4m3 fp8;
                *(unsigned char*)&fp8 = scale_byte;
                const float scale = (float)fp8 * scale2;
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const unsigned char byte_val =
                        (unsigned char)(packed4 >> (b * 8));
                    w_lo[o][b] = s_lut[byte_val & 0xF] * scale;
                    w_hi[o][b] = s_lut[byte_val >> 4] * scale;
                }
            } else {
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    w_lo[o][b] = 0.0f;
                    w_hi[o][b] = 0.0f;
                }
            }
        }

        #pragma unroll
        for (int row = 0; row < MAX_M; ++row) {
            if (rows_valid && (unsigned int)row < M && n0 < N) {
                const float* A_row = A + (unsigned long long)row * K;
                // ONE FP32 activation slab per (k8, row) for all T chains.
                const float4 a_lo = ((const float4*)A_row)[k8 * 2];
                const float4 a_hi = ((const float4*)A_row)[k8 * 2 + 1];
                const float a[8] = {
                    a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                    a_hi.x, a_hi.y, a_hi.z, a_hi.w
                };
                #pragma unroll
                for (int o = 0; o < T; ++o) {
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        acc[o][row] += a[b * 2] * w_lo[o][b];
                        acc[o][row] += a[b * 2 + 1] * w_hi[o][b];
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int o = 0; o < T; ++o) {
        #pragma unroll
        for (int row = 0; row < MAX_M; ++row) {
            if ((unsigned int)row < M) {
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                    acc[o][row] +=
                        __shfl_down_sync(0xFFFFFFFF, acc[o][row], offset);
                }
                if (warp_lane == 0) {
                    smem[(((unsigned int)row * N_PER_BLOCK + local_out) *
                          (unsigned int)T + (unsigned int)o) * 2 + warp_idx] =
                        acc[o][row];
                }
            }
        }
    }
    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int o = 0; o < T; ++o) {
            const unsigned int n = n0 + (unsigned int)o;
            if (!rows_valid || n >= N) continue;
            #pragma unroll
            for (int row = 0; row < MAX_M; ++row) {
                if ((unsigned int)row < M) {
                    const unsigned int base =
                        (((unsigned int)row * N_PER_BLOCK + local_out) *
                         (unsigned int)T + (unsigned int)o) * 2;
                    C[(unsigned long long)row * N + n] =
                        __float2bfloat16(smem[base] + smem[base + 1]);
                }
            }
        }
    }
}

#define DEFINE_W4A16_DUAL_MATERIALIZE_F32_RT(NAME, MAX_ROWS, TILE)            \
extern "C" __global__ void NAME(                                             \
    const __nv_bfloat16* A,                                                   \
    const unsigned char* gate_packed, const unsigned char* gate_scale,        \
    const float gate_scale2,                                                  \
    const unsigned char* up_packed, const unsigned char* up_scale,            \
    const float up_scale2, float* activation,                                 \
    unsigned int M, unsigned int N, unsigned int K)                           \
{                                                                            \
    __shared__ float s_lut[16];                                               \
    __shared__ float smem[(MAX_ROWS) * N_PER_BLOCK * (TILE) * 2];             \
    __shared__ __nv_bfloat16 rounded_gate[(MAX_ROWS) * N_PER_BLOCK * (TILE)]; \
    w4a16_dual_exact_materialize_f32_rt_body<MAX_ROWS, TILE>(                 \
        A, gate_packed, gate_scale, gate_scale2,                              \
        up_packed, up_scale, up_scale2, activation,                           \
        M, N, K, s_lut, smem, rounded_gate);                                  \
}

#define DEFINE_W4A16_F32_INPUT_EXACT_RT(NAME, MAX_ROWS, TILE)                 \
extern "C" __global__ void NAME(                                             \
    const float* A, const unsigned char* B_packed,                            \
    const unsigned char* B_scale, const float scale2,                         \
    __nv_bfloat16* C, unsigned int M, unsigned int N, unsigned int K)         \
{                                                                            \
    __shared__ float s_lut[16];                                               \
    __shared__ float smem[(MAX_ROWS) * N_PER_BLOCK * (TILE) * 2];             \
    w4a16_f32_input_exact_rt_body<MAX_ROWS, TILE>(                            \
        A, B_packed, B_scale, scale2, C, M, N, K, s_lut, smem);               \
}

DEFINE_W4A16_DUAL_MATERIALIZE_F32_RT(
    w4a16_gemv_dual_exact_materialize_f32_rt2_m17, 17, 2)
DEFINE_W4A16_F32_INPUT_EXACT_RT(w4a16_gemv_f32_input_exact_rt2_m8, 8, 2)
DEFINE_W4A16_F32_INPUT_EXACT_RT(w4a16_gemv_f32_input_exact_rt2_m17, 17, 2)

#undef DEFINE_W4A16_DUAL_MATERIALIZE_F32_RT
#undef DEFINE_W4A16_F32_INPUT_EXACT_RT
