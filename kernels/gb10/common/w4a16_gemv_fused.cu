// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A16 GEMV Fused — dual projection + silu-input variants.
//
// Reduces shared expert kernels from 4 to 2 per layer (saves 96 launches total):
//   Before: gate (1) + up (1) + silu_mul (1) + down (1) = 4 per layer × 48 = 192
//   After:  gate_up_dual (1) + silu_down (1) = 2 per layer × 48 = 96
//
// w4a16_gemv_dual: blockIdx.z selects projection 0 vs 1.
//   Both projections share the same BF16 input A[1, K].
//   Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
//
// w4a16_gemv_silu_input: reads gate_out + up_out BF16 vectors, computes
//   silu(gate)*up inline as activation, then GEMV with NVFP4 down weights.
//   Eliminates separate silu_mul kernel entirely.
//   Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
//
// rms_norm_residual_w4a16_gemv: fuses rms_norm_residual + w4a16_gemv into
//   a single launch. SSM qkvz path on Qwen3.6-27B (5120 → 12288).
//   - Pass 1 (cooperative across the whole block): read raw BF16 input,
//     compute sum-of-squares, write raw input to `residual` buffer.
//   - Pass 2 (cooperative): compute normalized BF16 `output` = x * rms * (1+w).
//   - Pass 3 (per-output GEMV): each output column streams W4A16 weights once
//     and accumulates against the freshly-normalized BF16 vector held in smem.
//   Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_FUSED_W4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// PR #519: stage the data-dependent NVFP4 LUT in shared memory per CUDA
// warp. The SCALE/HIP symlinked builds retain their established constant path.
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
#define ATLAS_WARP_LUT_STAGED 0
#else
#define ATLAS_WARP_LUT_STAGED 1
#endif

__device__ __forceinline__ void stage_e2m1_lut_fused_warp(
    float* s_lut, unsigned int lane)
{
#if ATLAS_WARP_LUT_STAGED
    if (lane < 16u) s_lut[lane] = E2M1_LUT_FUSED_W4[lane];
    __syncwarp();
#else
    (void)s_lut;
    (void)lane;
#endif
}

// One base-kernel lane's K8 partial for the dual-projection GEMV.
//
// `orig_lane` is the 64-thread kernel's lane (0..63): k8 = orig_lane, stride
// threads_per_out = BLOCK_SIZE / N_PER_BLOCK = 64.
//
// SSOT: shared by `w4a16_gemv_dual` and `w4a16_gemv_dual_sw` so the SW path
// cannot drift off the base association. `lut` is the base kernel's
// shared-memory E2M1 table or `E2M1_LUT_FUSED_W4` itself — same 16 values,
// so the arithmetic is bit-identical either way.
__device__ __forceinline__ float w4a16_dual_partial(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    const float* lut,
    unsigned int n, unsigned int half_K, unsigned int num_groups,
    unsigned int K8, unsigned int orig_lane)
{
    float acc = 0.0f;

    for (unsigned int k8 = orig_lane; k8 < K8; k8 += 64u) {
        const unsigned int base_k = k8 * 8;

        uint4 a_data = ((const uint4*)A)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};

        unsigned int packed4 = *(const unsigned int*)(
            B_packed + (unsigned long long)n * half_K + k8 * 4);

        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[
            (unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = lut[byte_val & 0xF] * scale;
            float w_hi = lut[byte_val >> 4] * scale;

            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo) * w_lo;
            acc += __bfloat162float(a_hi) * w_hi;
        }
    }
    return acc;
}

// One base-kernel lane's K8 partial for the SiLU-fused-input down GEMV.
// Same `orig_lane` contract as `w4a16_dual_partial`.
__device__ __forceinline__ float w4a16_silu_partial(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    const float* lut,
    unsigned int n, unsigned int half_K, unsigned int num_groups,
    unsigned int K8, unsigned int orig_lane)
{
    float acc = 0.0f;

    for (unsigned int k8 = orig_lane; k8 < K8; k8 += 64u) {
        const unsigned int base_k = k8 * 8;

        uint4 g_data = ((const uint4*)gate_out)[k8];
        uint4 u_data = ((const uint4*)up_out)[k8];

        unsigned int packed4 = *(const unsigned int*)(
            B_packed + (unsigned long long)n * half_K + k8 * 4);

        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[
            (unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        const unsigned int g_raw[4] = {g_data.x, g_data.y, g_data.z, g_data.w};
        const unsigned int u_raw[4] = {u_data.x, u_data.y, u_data.z, u_data.w};

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = lut[byte_val & 0xF] * scale;
            float w_hi = lut[byte_val >> 4] * scale;

            __nv_bfloat16 g_lo, g_hi;
            *(unsigned short*)&g_lo = (unsigned short)(g_raw[b] & 0xFFFF);
            *(unsigned short*)&g_hi = (unsigned short)(g_raw[b] >> 16);
            float gf_lo = __bfloat162float(g_lo);
            float gf_hi = __bfloat162float(g_hi);

            __nv_bfloat16 u_lo, u_hi;
            *(unsigned short*)&u_lo = (unsigned short)(u_raw[b] & 0xFFFF);
            *(unsigned short*)&u_hi = (unsigned short)(u_raw[b] >> 16);

            // SiLU(gate) * up = (gate / (1 + exp(-gate))) * up
            float a_lo = (gf_lo / (1.0f + __expf(-gf_lo))) * __bfloat162float(u_lo);
            float a_hi = (gf_hi / (1.0f + __expf(-gf_hi))) * __bfloat162float(u_hi);

            acc += a_lo * w_lo;
            acc += a_hi * w_hi;
        }
    }
    return acc;
}

// ── W4A16 GEMV Dual Projection ──
//
// blockIdx.z = 0: first projection (gate), blockIdx.z = 1: second (up).
// Both read same shared BF16 input A[1, K] with different NVFP4 weights.
// Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_dual(
    const __nv_bfloat16* __restrict__ A,           // [1, K] shared input
    const unsigned char* __restrict__ B1_packed,    // [N, K/2] proj 0 weights
    const unsigned char* __restrict__ B1_scale,     // [N, K/GROUP_SIZE] proj 0
    const float scale2_1,
    __nv_bfloat16* __restrict__ C1,                 // [1, N] proj 0 output
    const unsigned char* __restrict__ B2_packed,    // [N, K/2] proj 1 weights
    const unsigned char* __restrict__ B2_scale,     // [N, K/GROUP_SIZE] proj 1
    const float scale2_2,
    __nv_bfloat16* __restrict__ C2,                 // [1, N] proj 1 output
    unsigned int N,
    unsigned int K
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? B1_packed : B2_packed;
    const unsigned char* B_scale = proj == 0 ? B1_scale : B2_scale;
    float scale2 = proj == 0 ? scale2_1 : scale2_2;
    __nv_bfloat16* C = proj == 0 ? C1 : C2;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED_W4[threadIdx.x];
    __syncthreads();

    float acc = w4a16_dual_partial(A, B_packed, B_scale, scale2, s_lut,
                                   n, half_K, num_groups, K8, lane);

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

// ── W4A16 GEMV with SiLU-fused Input ──
//
// Reads gate_out[K] and up_out[K] BF16, computes silu(gate)*up inline
// as the activation, then GEMV with NVFP4 down weights.
// Eliminates the separate silu_mul kernel entirely.
// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_silu_input(
    const __nv_bfloat16* __restrict__ gate_out,    // [1, K] gate proj output
    const __nv_bfloat16* __restrict__ up_out,      // [1, K] up proj output
    const unsigned char* __restrict__ B_packed,     // [N, K/2] down weights
    const unsigned char* __restrict__ B_scale,      // [N, K/GROUP_SIZE]
    const float scale2,
    __nv_bfloat16* __restrict__ C,                  // [1, N] output
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED_W4[threadIdx.x];
    __syncthreads();

    float acc = w4a16_silu_partial(gate_out, up_out, B_packed, B_scale, scale2,
                                   s_lut, n, half_K, num_groups, K8, lane);

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

// ════════════════════════════════════════════════════════════════════
// Exact dynamic-M dense-FFN kernels.
//
// Each row reproduces the ordinary K1 kernels above: k8=lane with stride 64,
// the same ordered low/high FP32 updates, the same five-step warp reduction,
// the same ordered cross-warp add, and one final BF16 conversion. Weights are
// decoded once per K8 chunk and reused across independent row accumulators.
// The smallest fitting M4/M8/M17/M32 family keeps register pressure bounded.
// ════════════════════════════════════════════════════════════════════

template <int MAX_M>
__device__ __forceinline__ void w4a16_gemv_dual_exact_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B1_packed,
    const unsigned char* __restrict__ B1_scale,
    const float scale2_1,
    __nv_bfloat16* __restrict__ C1,
    const unsigned char* __restrict__ B2_packed,
    const unsigned char* __restrict__ B2_scale,
    const float scale2_2,
    __nv_bfloat16* __restrict__ C2,
    unsigned int M, unsigned int N, unsigned int K,
    float* s_lut, float* smem)
{
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? B1_packed : B2_packed;
    const unsigned char* B_scale = proj == 0 ? B1_scale : B2_scale;
    const float scale2 = proj == 0 ? scale2_1 : scale2_2;
    __nv_bfloat16* C = proj == 0 ? C1 : C2;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int warp_lane = lane % WARP_SIZE;
    const unsigned int warp_idx = lane / WARP_SIZE;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool rows_valid = M > 0 && M <= (unsigned int)MAX_M;
    const bool valid = rows_valid && n < N;

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED_W4[threadIdx.x];
    __syncthreads();

    float acc[MAX_M];
    #pragma unroll
    for (int row = 0; row < MAX_M; ++row) acc[row] = 0.0f;

    if (valid) {
        const unsigned int half_K = K / 2;
        const unsigned int num_groups = K / GROUP_SIZE;
        const unsigned int K8 = K / 8;
        const unsigned long long weight_row =
            (unsigned long long)n * half_K;
        const unsigned long long scale_row =
            (unsigned long long)n * num_groups;

        // Software pipeline: prefetch tile k8+64's weight+scale while
        // accumulating tile k8, hiding the weight-stream DRAM latency behind
        // dequant+FMA work. Bit-exact: identical addresses and identical
        // dequant/FMA order; only the load issue is hoisted earlier.
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
                    const __nv_bfloat16* A_row =
                        A + (unsigned long long)row * K;
                    const uint4 a_data = ((const uint4*)A_row)[k8];
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
                smem[(row * N_PER_BLOCK + local_out) * 2 + warp_idx] =
                    acc[row];
            }
        }
    }
    __syncthreads();

    if (valid && lane == 0) {
        #pragma unroll
        for (int row = 0; row < MAX_M; ++row) {
            if ((unsigned int)row < M) {
                const unsigned int base =
                    (row * N_PER_BLOCK + local_out) * 2;
                C[(unsigned long long)row * N + n] =
                    __float2bfloat16(smem[base] + smem[base + 1]);
            }
        }
    }
}

template <int MAX_M>
__device__ __forceinline__ void w4a16_gemv_silu_input_exact_body(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
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
    const bool rows_valid = M > 0 && M <= (unsigned int)MAX_M;
    const bool valid = rows_valid && n < N;

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED_W4[threadIdx.x];
    __syncthreads();

    float acc[MAX_M];
    #pragma unroll
    for (int row = 0; row < MAX_M; ++row) acc[row] = 0.0f;

    if (valid) {
        const unsigned int half_K = K / 2;
        const unsigned int num_groups = K / GROUP_SIZE;
        const unsigned int K8 = K / 8;
        const unsigned long long weight_row =
            (unsigned long long)n * half_K;
        const unsigned long long scale_row =
            (unsigned long long)n * num_groups;

        for (unsigned int k8 = lane; k8 < K8; k8 += 64u) {
            const unsigned int base_k = k8 * 8;
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
                    const __nv_bfloat16* gate_row =
                        gate_out + (unsigned long long)row * K;
                    const __nv_bfloat16* up_row =
                        up_out + (unsigned long long)row * K;
                    const uint4 g_data = ((const uint4*)gate_row)[k8];
                    const uint4 u_data = ((const uint4*)up_row)[k8];
                    const unsigned int g_raw[4] = {
                        g_data.x, g_data.y, g_data.z, g_data.w
                    };
                    const unsigned int u_raw[4] = {
                        u_data.x, u_data.y, u_data.z, u_data.w
                    };

                    #pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        __nv_bfloat16 g_lo, g_hi, u_lo, u_hi;
                        *(unsigned short*)&g_lo =
                            (unsigned short)(g_raw[b] & 0xFFFF);
                        *(unsigned short*)&g_hi =
                            (unsigned short)(g_raw[b] >> 16);
                        *(unsigned short*)&u_lo =
                            (unsigned short)(u_raw[b] & 0xFFFF);
                        *(unsigned short*)&u_hi =
                            (unsigned short)(u_raw[b] >> 16);
                        const float gf_lo = __bfloat162float(g_lo);
                        const float gf_hi = __bfloat162float(g_hi);
                        const float a_lo =
                            (gf_lo / (1.0f + __expf(-gf_lo))) *
                            __bfloat162float(u_lo);
                        const float a_hi =
                            (gf_hi / (1.0f + __expf(-gf_hi))) *
                            __bfloat162float(u_hi);
                        acc[row] += a_lo * w_lo[b];
                        acc[row] += a_hi * w_hi[b];
                    }
                }
            }
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
                smem[(row * N_PER_BLOCK + local_out) * 2 + warp_idx] =
                    acc[row];
            }
        }
    }
    __syncthreads();

    if (valid && lane == 0) {
        #pragma unroll
        for (int row = 0; row < MAX_M; ++row) {
            if ((unsigned int)row < M) {
                const unsigned int base =
                    (row * N_PER_BLOCK + local_out) * 2;
                C[(unsigned long long)row * N + n] =
                    __float2bfloat16(smem[base] + smem[base + 1]);
            }
        }
    }
}

#define DEFINE_W4A16_GEMV_DUAL_EXACT(NAME, MAX_ROWS)                        \
extern "C" __global__ void NAME(                                             \
    const __nv_bfloat16* A,                                                  \
    const unsigned char* B1_packed, const unsigned char* B1_scale,           \
    const float scale2_1, __nv_bfloat16* C1,                                 \
    const unsigned char* B2_packed, const unsigned char* B2_scale,           \
    const float scale2_2, __nv_bfloat16* C2,                                 \
    unsigned int M, unsigned int N, unsigned int K)                           \
{                                                                            \
    __shared__ float s_lut[16];                                               \
    __shared__ float smem[(MAX_ROWS) * N_PER_BLOCK * 2];                      \
    w4a16_gemv_dual_exact_body<MAX_ROWS>(                                    \
        A, B1_packed, B1_scale, scale2_1, C1, B2_packed, B2_scale,           \
        scale2_2, C2, M, N, K, s_lut, smem);                                 \
}

#define DEFINE_W4A16_GEMV_SILU_INPUT_EXACT(NAME, MAX_ROWS)                  \
extern "C" __global__ void NAME(                                             \
    const __nv_bfloat16* gate_out, const __nv_bfloat16* up_out,              \
    const unsigned char* B_packed, const unsigned char* B_scale,             \
    const float scale2, __nv_bfloat16* C,                                    \
    unsigned int M, unsigned int N, unsigned int K)                           \
{                                                                            \
    __shared__ float s_lut[16];                                               \
    __shared__ float smem[(MAX_ROWS) * N_PER_BLOCK * 2];                      \
    w4a16_gemv_silu_input_exact_body<MAX_ROWS>(                              \
        gate_out, up_out, B_packed, B_scale, scale2, C,                      \
        M, N, K, s_lut, smem);                                               \
}

DEFINE_W4A16_GEMV_DUAL_EXACT(w4a16_gemv_dual_exact_m4, 4)
DEFINE_W4A16_GEMV_DUAL_EXACT(w4a16_gemv_dual_exact_m8, 8)
DEFINE_W4A16_GEMV_DUAL_EXACT(w4a16_gemv_dual_exact_m17, 17)
DEFINE_W4A16_GEMV_DUAL_EXACT(w4a16_gemv_dual_exact_m32, 32)

DEFINE_W4A16_GEMV_SILU_INPUT_EXACT(
    w4a16_gemv_silu_input_exact_m4, 4)
DEFINE_W4A16_GEMV_SILU_INPUT_EXACT(
    w4a16_gemv_silu_input_exact_m8, 8)
DEFINE_W4A16_GEMV_SILU_INPUT_EXACT(
    w4a16_gemv_silu_input_exact_m17, 17)
DEFINE_W4A16_GEMV_SILU_INPUT_EXACT(
    w4a16_gemv_silu_input_exact_m32, 32)

#undef DEFINE_W4A16_GEMV_DUAL_EXACT
#undef DEFINE_W4A16_GEMV_SILU_INPUT_EXACT

// M8/M17 materialized-FP32 entries; keep this include revision in the owning
// translation unit so the kernel cache cannot retain the earlier M8-only PTX.
#include "w4a16_gemv_exact_f32_m8.cuh"
#include "w4a16_gemv_exact_f32_m17_fused.cuh"

// ════════════════════════════════════════════════════════════════════
// SINGLE-WARP-PER-OUTPUT variants (lossless; default ON, kill with
// ATLAS_NO_GEMV_SW=1).
//
// Ported from upstream `w4a16_gemv_dual_sw` / `w4a16_gemv_silu_input_sw`
// (upstream-latest/kernels/gb10/common/w4a16_gemv_fused.cu:265-385),
// re-derived against THIS tree's base kernels — ours are the K8 sequential
// `acc += a*w` form, upstream's dual has since been re-associated into a
// 2-chunk K16 pipeline. Bit-parity here is against OUR bases, which is what
// keeps the committed token stream unchanged. (Upstream's own
// `silu_input_sw` is derived the same way, against its still-sequential K8
// silu base.)
//
// 32 threads (1 warp) per output instead of 64, 8 outputs per 256-thread
// block instead of 4. The cross-warp __syncthreads() + smem round-trip is
// replaced by one FP32 add of two warp-shuffle reductions:
//   acc_a[lane] == base acc[lane]       (k8 = lane,      stride 64)
//   acc_b[lane] == base acc[lane + 32]  (k8 = lane + 32, stride 64)
// reduced in the same 5-step tree, so reduced_a + reduced_b is the base
// kernel's smem[0] + smem[1], operand-for-operand.
//
// Grid: (ceil(N/8), 1, z)  Block: (256, 1, 1) — z as for the base kernel.
// ════════════════════════════════════════════════════════════════════

#define N_PER_BLOCK_SW 8

extern "C" __global__ void w4a16_gemv_dual_sw(
    const __nv_bfloat16* __restrict__ A,           // [1, K] shared input
    const unsigned char* __restrict__ B1_packed,    // [N, K/2] proj 0 weights
    const unsigned char* __restrict__ B1_scale,     // [N, K/GROUP_SIZE] proj 0
    const float scale2_1,
    __nv_bfloat16* __restrict__ C1,                 // [1, N] proj 0 output
    const unsigned char* __restrict__ B2_packed,    // [N, K/2] proj 1 weights
    const unsigned char* __restrict__ B2_scale,     // [N, K/GROUP_SIZE] proj 1
    const float scale2_2,
    __nv_bfloat16* __restrict__ C2,                 // [1, N] proj 1 output
    unsigned int N,
    unsigned int K
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? B1_packed : B2_packed;
    const unsigned char* B_scale = proj == 0 ? B1_scale : B2_scale;
    float scale2 = proj == 0 ? scale2_1 : scale2_2;
    __nv_bfloat16* C = proj == 0 ? C1 : C2;

    const unsigned int local_out = threadIdx.x / WARP_SIZE;  // 0..7
    const unsigned int lane = threadIdx.x % WARP_SIZE;       // 0..31
    const unsigned int n = blockIdx.x * N_PER_BLOCK_SW + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[N_PER_BLOCK_SW][16];
    stage_e2m1_lut_fused_warp(s_lut[local_out], lane);
#if ATLAS_WARP_LUT_STAGED
    const float* __restrict__ warp_lut = s_lut[local_out];
#else
    const float* __restrict__ warp_lut = E2M1_LUT_FUSED_W4;
#endif

    float acc_a = w4a16_dual_partial(A, B_packed, B_scale, scale2,
                                     warp_lut, n, half_K, num_groups,
                                     K8, lane);
    float acc_b = w4a16_dual_partial(A, B_packed, B_scale, scale2,
                                     warp_lut, n, half_K, num_groups,
                                     K8, lane + 32u);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xFFFFFFFF, acc_a, offset);
        acc_b += __shfl_down_sync(0xFFFFFFFF, acc_b, offset);
    }

    if (lane == 0) {
        C[n] = __float2bfloat16(acc_a + acc_b);
    }
}

extern "C" __global__ void w4a16_gemv_silu_input_sw(
    const __nv_bfloat16* __restrict__ gate_out,    // [1, K] gate proj output
    const __nv_bfloat16* __restrict__ up_out,      // [1, K] up proj output
    const unsigned char* __restrict__ B_packed,     // [N, K/2] down weights
    const unsigned char* __restrict__ B_scale,      // [N, K/GROUP_SIZE]
    const float scale2,
    __nv_bfloat16* __restrict__ C,                  // [1, N] output
    unsigned int N,
    unsigned int K
) {
    const unsigned int local_out = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int n = blockIdx.x * N_PER_BLOCK_SW + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[N_PER_BLOCK_SW][16];
    stage_e2m1_lut_fused_warp(s_lut[local_out], lane);
#if ATLAS_WARP_LUT_STAGED
    const float* __restrict__ warp_lut = s_lut[local_out];
#else
    const float* __restrict__ warp_lut = E2M1_LUT_FUSED_W4;
#endif

    float acc_a = w4a16_silu_partial(gate_out, up_out, B_packed, B_scale,
                                     scale2, warp_lut, n, half_K,
                                     num_groups, K8, lane);
    float acc_b = w4a16_silu_partial(gate_out, up_out, B_packed, B_scale,
                                     scale2, warp_lut, n, half_K,
                                     num_groups, K8, lane + 32u);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xFFFFFFFF, acc_a, offset);
        acc_b += __shfl_down_sync(0xFFFFFFFF, acc_b, offset);
    }

    if (lane == 0) {
        C[n] = __float2bfloat16(acc_a + acc_b);
    }
}

// ── Fused RMS Norm + Residual Save + W4A16 GEMV ──
//
// Eliminates the dedicated `rms_norm_residual` launch in front of the SSM
// QKVZ projection on the Qwen3.6-27B decode path. The previous flow was:
//
//     rms_norm_residual(hidden, gamma) -> {normed, residual}
//     w4a16_gemv(normed, W_qkvz) -> qkvz_out
//
// Two launches, two grid setups, two L1 prologues, two writes of the
// normalized hidden state to DRAM and back. This kernel performs both in a
// single CTA: the whole block cooperates to compute the RMS sum, write the
// residual copy, materialize the normed BF16 vector into shared memory, and
// then each of the 4 outputs per CTA streams the W4A16 weights once and FMAs
// against the smem-resident normalized vector.
//
// Layout assumptions (validated against the SSM qkvz call sites):
//   - hidden_size K is a multiple of 8 (5120 ✓), N a multiple of 4 (12288 ✓).
//   - K * 2 bytes must fit in dynamic shared memory (5120 * 2 = 10 KiB ✓).
//   - The block writes `normed_out` (BF16, [K]) for downstream consumers (the
//     SSM ba_gates projection re-reads `normed`).
//
// Grid: (ceil(N/4), 1, 1)   Block: (256, 1, 1)
// Dynamic shared memory: hidden_size * sizeof(__nv_bfloat16) + 16*4 + 8*4 + 8*4 bytes.
__device__ __forceinline__ void _fused_unpack_bf16x2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int _fused_pack_bf16x2(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

extern "C" __global__ void rms_norm_residual_w4a16_gemv(
    const __nv_bfloat16* __restrict__ input,        // [1, K] BF16 raw hidden
    const __nv_bfloat16* __restrict__ gamma,        // [K] BF16 rms weight (1+w offset)
    __nv_bfloat16* __restrict__ normed_out,         // [1, K] BF16 — written for downstream ba_gates
    __nv_bfloat16* __restrict__ residual_out,       // [1, K] BF16 — raw input copy
    const unsigned char* __restrict__ B_packed,     // [N, K/2] NVFP4 packed weights
    const unsigned char* __restrict__ B_scale,      // [N, K/GROUP_SIZE] FP8-E4M3 scales
    const float scale2,                             // per-tensor second-level scale
    __nv_bfloat16* __restrict__ C,                  // [1, N] GEMV output
    unsigned int N,
    unsigned int K,
    float eps
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int tid = threadIdx.x;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    // Dynamic shared memory layout:
    //   s_normed:  __nv_bfloat16[K]   — fresh normed vector (shared by all 4 outputs)
    //   s_lut:     float[16]          — E2M1 dequant LUT
    //   s_warp:    float[BLOCK_SIZE / WARP_SIZE] — per-warp sum-sq partial sums
    //   s_redux:   float[N_PER_BLOCK * 2]        — cross-warp GEMV reduction scratch
    extern __shared__ unsigned char s_raw[];
    __nv_bfloat16* s_normed = reinterpret_cast<__nv_bfloat16*>(s_raw);
    float* s_lut   = reinterpret_cast<float*>(s_normed + K);
    float* s_warp  = s_lut + 16;
    float* s_redux = s_warp + (BLOCK_SIZE / WARP_SIZE);

    if (tid < 16) s_lut[tid] = E2M1_LUT_FUSED_W4[tid];

    // ── Pass 1: vectorized 4-wide BF16 reads → write residual + compute sum-of-squares ──
    // Each thread handles a strided slice of the K-dim. Vectorize as uint64
    // (= 4 BF16 values) for 64-bit loads.
    const unsigned long long* in64  = reinterpret_cast<const unsigned long long*>(input);
    unsigned long long* res64       = reinterpret_cast<unsigned long long*>(residual_out);
    const unsigned int quad_K = K / 4;

    float sum_sq = 0.0f;
    for (unsigned int q = tid; q < quad_K; q += BLOCK_SIZE) {
        unsigned long long packed = in64[q];
        res64[q] = packed;  // zero-extra-bandwidth residual copy

        float f0, f1, f2, f3;
        _fused_unpack_bf16x2((unsigned int)packed, f0, f1);
        _fused_unpack_bf16x2((unsigned int)(packed >> 32), f2, f3);
        sum_sq += f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3;
    }

    // Warp reduce (xor-tree), then warp-0 reduces the per-warp partials.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int wlane = tid % WARP_SIZE;
    if (wlane == 0) s_warp[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float v = (wlane < (BLOCK_SIZE / WARP_SIZE)) ? s_warp[wlane] : 0.0f;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
        }
        if (wlane == 0) s_warp[0] = v;
    }
    __syncthreads();

    const float rms = rsqrtf(s_warp[0] / (float)K + eps);

    // ── Pass 2: apply norm * (1 + gamma), write normed to DRAM and smem ──
    const unsigned long long* g64 = reinterpret_cast<const unsigned long long*>(gamma);
    unsigned long long* normed64  = reinterpret_cast<unsigned long long*>(normed_out);
    unsigned long long* s_normed64 = reinterpret_cast<unsigned long long*>(s_normed);

    for (unsigned int q = tid; q < quad_K; q += BLOCK_SIZE) {
        unsigned long long x_packed = in64[q];
        unsigned long long w_packed = g64[q];
        float xv0, xv1, xv2, xv3;
        _fused_unpack_bf16x2((unsigned int)x_packed, xv0, xv1);
        _fused_unpack_bf16x2((unsigned int)(x_packed >> 32), xv2, xv3);
        float wv0, wv1, wv2, wv3;
        _fused_unpack_bf16x2((unsigned int)w_packed, wv0, wv1);
        _fused_unpack_bf16x2((unsigned int)(w_packed >> 32), wv2, wv3);

        unsigned int lo = _fused_pack_bf16x2(xv0 * rms * (1.0f + wv0),
                                             xv1 * rms * (1.0f + wv1));
        unsigned int hi = _fused_pack_bf16x2(xv2 * rms * (1.0f + wv2),
                                             xv3 * rms * (1.0f + wv3));
        unsigned long long out_packed = ((unsigned long long)hi << 32) | (unsigned long long)lo;
        normed64[q] = out_packed;
        s_normed64[q] = out_packed;
    }
    __syncthreads();

    // ── Pass 3: W4A16 GEMV consuming the smem-resident normed vector ──
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    float acc = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        // 8 BF16 activations from smem as uint4 (128-bit load).
        uint4 a_data = ((const uint4*)s_normed)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};

        unsigned int packed4 = *(const unsigned int*)(
            B_packed + (unsigned long long)n * half_K + k8 * 4);

        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[
            (unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo) * w_lo;
            acc += __bfloat162float(a_hi) * w_hi;
        }
    }

    // 64 threads per output = 2 warps → smem cross-warp reduction.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    if (wlane == 0) {
        s_redux[local_out * 2 + (lane / WARP_SIZE)] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = s_redux[local_out * 2] + s_redux[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

// ── Fused RMS Norm + Residual Save + W4A16 batch-3 GEMV ──
//
// K=3 speculative-verify counterpart of `rms_norm_residual_w4a16_gemv`.
// Processes 3 tokens in one CTA: each output column FMAs a single weight
// stream against 3 smem-resident normalized vectors. Mirrors the geometry
// of `w4a16_gemv_batch3` (1 CTA per N-tile of 4 outputs, 64 threads/output).
//
// Per-token RMS norm + residual is done cooperatively in 3 successive passes
// (one per token). Each token's normalized vector is materialized into its
// own slot in shared memory; the GEMV phase then performs 3 separate
// accumulations against the shared weight load.
//
// Grid: (ceil(N/4), 1, 1)   Block: (256, 1, 1)
// Dynamic smem: 3*K*sizeof(BF16) + 16*4 + (BLOCK_SIZE/WARP_SIZE)*4 + N_PER_BLOCK*6*4
extern "C" __global__ void rms_norm_residual_w4a16_gemv_batch3(
    const __nv_bfloat16* __restrict__ input,        // [3, K] BF16 raw hidden
    const __nv_bfloat16* __restrict__ gamma,        // [K] BF16 rms weight (1+w offset)
    __nv_bfloat16* __restrict__ normed_out,         // [3, K] BF16 — written for downstream ba_gates
    __nv_bfloat16* __restrict__ residual_out,       // [3, K] BF16 — raw input copy
    const unsigned char* __restrict__ B_packed,     // [N, K/2] NVFP4 packed weights
    const unsigned char* __restrict__ B_scale,      // [N, K/GROUP_SIZE] FP8-E4M3 scales
    const float scale2,                             // per-tensor second-level scale
    __nv_bfloat16* __restrict__ C,                  // [3, N] GEMV output
    unsigned int N,
    unsigned int K,
    float eps
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int tid = threadIdx.x;
    const unsigned int wlane = tid % WARP_SIZE;
    const unsigned int warp_id = tid / WARP_SIZE;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;
    const unsigned int quad_K = K / 4;

    extern __shared__ unsigned char s_raw[];
    __nv_bfloat16* s_normed = reinterpret_cast<__nv_bfloat16*>(s_raw);  // 3*K
    float* s_lut   = reinterpret_cast<float*>(s_normed + 3 * K);
    float* s_warp  = s_lut + 16;
    float* s_redux = s_warp + (BLOCK_SIZE / WARP_SIZE);

    if (tid < 16) s_lut[tid] = E2M1_LUT_FUSED_W4[tid];

    const unsigned long long* g64 = reinterpret_cast<const unsigned long long*>(gamma);

    // ── Pass A: per-token rms_norm_residual ──
    // Loop over 3 tokens. Each token: copy residual, compute sum_sq, reduce,
    // write normed to DRAM and to its smem slot.
    #pragma unroll
    for (int t = 0; t < 3; t++) {
        const unsigned long long* in64 =
            reinterpret_cast<const unsigned long long*>(input + (size_t)t * K);
        unsigned long long* res64 =
            reinterpret_cast<unsigned long long*>(residual_out + (size_t)t * K);
        unsigned long long* normed64 =
            reinterpret_cast<unsigned long long*>(normed_out + (size_t)t * K);
        unsigned long long* s_token64 =
            reinterpret_cast<unsigned long long*>(s_normed + (size_t)t * K);

        float sum_sq = 0.0f;
        for (unsigned int q = tid; q < quad_K; q += BLOCK_SIZE) {
            unsigned long long packed = in64[q];
            res64[q] = packed;
            float f0, f1, f2, f3;
            _fused_unpack_bf16x2((unsigned int)packed, f0, f1);
            _fused_unpack_bf16x2((unsigned int)(packed >> 32), f2, f3);
            sum_sq += f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3;
        }

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
        }
        if (wlane == 0) s_warp[warp_id] = sum_sq;
        __syncthreads();

        if (warp_id == 0) {
            float v = (wlane < (BLOCK_SIZE / WARP_SIZE)) ? s_warp[wlane] : 0.0f;
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
            }
            if (wlane == 0) s_warp[0] = v;
        }
        __syncthreads();

        const float rms = rsqrtf(s_warp[0] / (float)K + eps);

        for (unsigned int q = tid; q < quad_K; q += BLOCK_SIZE) {
            unsigned long long x_packed = in64[q];
            unsigned long long w_packed = g64[q];
            float xv0, xv1, xv2, xv3;
            _fused_unpack_bf16x2((unsigned int)x_packed, xv0, xv1);
            _fused_unpack_bf16x2((unsigned int)(x_packed >> 32), xv2, xv3);
            float wv0, wv1, wv2, wv3;
            _fused_unpack_bf16x2((unsigned int)w_packed, wv0, wv1);
            _fused_unpack_bf16x2((unsigned int)(w_packed >> 32), wv2, wv3);
            unsigned int lo = _fused_pack_bf16x2(xv0 * rms * (1.0f + wv0),
                                                 xv1 * rms * (1.0f + wv1));
            unsigned int hi = _fused_pack_bf16x2(xv2 * rms * (1.0f + wv2),
                                                 xv3 * rms * (1.0f + wv3));
            unsigned long long out_packed =
                ((unsigned long long)hi << 32) | (unsigned long long)lo;
            normed64[q] = out_packed;
            s_token64[q] = out_packed;
        }
        __syncthreads();
    }

    // ── Pass B: W4A16 batch-3 GEMV against smem-resident normalized vectors ──
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const __nv_bfloat16* s_a0 = s_normed;
    const __nv_bfloat16* s_a1 = s_normed + K;
    const __nv_bfloat16* s_a2 = s_normed + 2 * K;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)s_a0)[k8];
        uint4 a1_data = ((const uint4*)s_a1)[k8];
        uint4 a2_data = ((const uint4*)s_a2)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int a2_raw[4] = {a2_data.x, a2_data.y, a2_data.z, a2_data.w};

        unsigned int packed4 = *(const unsigned int*)(
            B_packed + (unsigned long long)n * half_K + k8 * 4);

        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[
            (unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo + __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo + __bfloat162float(a1_hi) * w_hi;

            __nv_bfloat16 a2_lo, a2_hi;
            *(unsigned short*)&a2_lo = (unsigned short)(a2_raw[b] & 0xFFFF);
            *(unsigned short*)&a2_hi = (unsigned short)(a2_raw[b] >> 16);
            acc2 += __bfloat162float(a2_lo) * w_lo + __bfloat162float(a2_hi) * w_hi;
        }
    }

    // Cross-warp reduction (2 warps per output × 3 accumulators).
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }
    if (wlane == 0) {
        unsigned int wi = lane / WARP_SIZE;
        s_redux[local_out * 6 + wi * 3]     = acc0;
        s_redux[local_out * 6 + wi * 3 + 1] = acc1;
        s_redux[local_out * 6 + wi * 3 + 2] = acc2;
    }
    __syncthreads();

    if (lane == 0) {
        float r0 = s_redux[local_out * 6]     + s_redux[local_out * 6 + 3];
        float r1 = s_redux[local_out * 6 + 1] + s_redux[local_out * 6 + 4];
        float r2 = s_redux[local_out * 6 + 2] + s_redux[local_out * 6 + 5];
        C[n]         = __float2bfloat16(r0);
        C[N + n]     = __float2bfloat16(r1);
        C[2 * N + n] = __float2bfloat16(r2);
    }
}
