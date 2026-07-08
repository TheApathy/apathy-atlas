// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W3A16 GEMV — fused 3-bit weight dequant + BF16 GEMV for M=1 decode.
//
// Mixed-precision byte-reduction lane: selected FFN layers' weights drop
// from NVFP4 (4-bit E2M1) to a 3-bit format, cutting packed weight bytes
// 25% on a weight-bandwidth-bound decode. Gated by ATLAS_FFN_W3_LAYERS +
// sidecar presence; quality-gated by the ABBA eval (NOT md5 — the weights
// differ from W4 by construction).
//
// W3 FORMAT (v1 — must match local/tools/repack_w3.py):
//   * 8-level global LUT (e2m1-subset magnitudes {0,1,2,4}, sign in bit 2):
//       W3_LUT[8] = {0, 1, 2, 4, -0, -1, -2, -4}
//   * Packing: 8 weights -> 3 bytes, little-endian 24-bit word:
//       u24 = sum_i code_i << (3*i);  bytes (u24&0xFF, u24>>8, u24>>16)
//     B_packed3: [N, 3*K/8] u8 row-major.
//   * Scales: SAME per-16 FP8-E4M3 group-scale scheme as NVFP4:
//       B_scale [N, K/16] u8, per-tensor f32 scale2 (sidecar value = 1.5x
//       the W4 scale2; Lmax 6 -> 4 rescale).
//   * Dequant (exact contract, mirrored in Python + Rust host tests):
//       sv = (float)e4m3(scale_byte) * scale2;   w = W3_LUT[code] * sv;
//
// Kernel bodies are structural clones of kernels/gb10/common/w4a16_gemv.cu
// and w4a16_gemv_fused.cu: same grid/block geometry, same per-k
// accumulation order (ascending k, acc += a_lo*w_lo; acc += a_hi*w_hi per
// byte-pair), same warp-shuffle + smem reduction, same BF16 round-trips.
// ONLY the weight unpack differs: 3-bit codes from 6-byte/16-weight reads
// instead of 4-bit nibbles from 8-byte/16-weight reads.
//
// All kernels iterate 16 K-values per step (= 1 scale group = 2 octets =
// 6 packed bytes), so every scale lookup covers exactly one iteration.
// Row byte stride 3*K/8 is even for K % 16 == 0, so the three u16 loads
// per iteration are 2-byte aligned.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float W3_LUT[8] = {
    0.0f, 1.0f, 2.0f, 4.0f,
    -0.0f, -1.0f, -2.0f, -4.0f
};

// Load 6 packed bytes (2 octets = 16 weights) at 2-byte alignment and
// return the 48-bit code word (codes for k = base_k .. base_k+15).
__device__ __forceinline__ unsigned long long w3_load6(const unsigned char* p) {
    const unsigned short* p16 = (const unsigned short*)p;
    unsigned long long lo = p16[0];
    unsigned long long mid = p16[1];
    unsigned long long hi = p16[2];
    return lo | (mid << 16) | (hi << 32);
}

// ── W3A16 GEMV: C[n] = sum_k A[k] * dequant3(B[n, k]) ──
//
// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1) — same as w4a16_gemv.
extern "C" __global__ void w3a16_gemv(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const unsigned char* __restrict__ B_packed,  // [N, 3*K/8] u8
    const unsigned char* __restrict__ B_scale,   // [N, K/16] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [1, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int row_bytes = (K >> 3) * 3;   // 3*K/8
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[8];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 8) s_lut[threadIdx.x] = W3_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    // 16 K-values per iteration: 2x uint4 activations + 6 weight bytes +
    // exactly 1 scale lookup (GROUP_SIZE == 16).
    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        uint4 a_lo = ((const uint4*)A)[k16 * 2];
        uint4 a_hi = ((const uint4*)A)[k16 * 2 + 1];
        const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};

        unsigned long long codes48 =
            w3_load6(B_packed + (unsigned long long)n * row_bytes + k16 * 6);

        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + k16];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        #pragma unroll
        for (int b = 0; b < 8; b++) {
            // Values 2b (lo) and 2b+1 (hi) — same pairing/order as the W4
            // kernel's byte loop, so the accumulation order is identical.
            float w_lo = s_lut[(codes48 >> (6 * b)) & 7] * scale;
            float w_hi = s_lut[(codes48 >> (6 * b + 3)) & 7] * scale;

            __nv_bfloat16 a_lo_bf, a_hi_bf;
            *(unsigned short*)&a_lo_bf = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi_bf = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo_bf) * w_lo;
            acc += __bfloat162float(a_hi_bf) * w_hi;
        }
    }

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

// ── W3A16 dual GEMV: gate + up sharing one BF16 input, one launch ──
//
// blockIdx.z = 0: proj 0 (gate), 1: proj 1 (up).
// Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1) — same as w4a16_gemv_dual.
extern "C" __global__ void w3a16_gemv_dual(
    const __nv_bfloat16* __restrict__ A,            // [1, K] shared input
    const unsigned char* __restrict__ B1_packed,     // [N, 3K/8] proj 0
    const unsigned char* __restrict__ B1_scale,      // [N, K/16] proj 0
    const float scale2_1,
    __nv_bfloat16* __restrict__ C1,                  // [1, N] proj 0 out
    const unsigned char* __restrict__ B2_packed,     // [N, 3K/8] proj 1
    const unsigned char* __restrict__ B2_scale,      // [N, K/16] proj 1
    const float scale2_2,
    __nv_bfloat16* __restrict__ C2,                  // [1, N] proj 1 out
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

    const unsigned int row_bytes = (K >> 3) * 3;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[8];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 8) s_lut[threadIdx.x] = W3_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        uint4 a_lo = ((const uint4*)A)[k16 * 2];
        uint4 a_hi = ((const uint4*)A)[k16 * 2 + 1];
        const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};

        unsigned long long codes48 =
            w3_load6(B_packed + (unsigned long long)n * row_bytes + k16 * 6);

        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + k16];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        #pragma unroll
        for (int b = 0; b < 8; b++) {
            float w_lo = s_lut[(codes48 >> (6 * b)) & 7] * scale;
            float w_hi = s_lut[(codes48 >> (6 * b + 3)) & 7] * scale;

            __nv_bfloat16 a_lo_bf, a_hi_bf;
            *(unsigned short*)&a_lo_bf = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi_bf = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo_bf) * w_lo;
            acc += __bfloat162float(a_hi_bf) * w_hi;
        }
    }

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

// ── W3A16 GEMV with SiLU-fused input (down_proj) ──
//
// Reads gate_out[K] + up_out[K] BF16, computes silu(gate)*up inline, then
// GEMV against the W3 down weights. Grid: (ceil(N/4), 1, 1)  Block: (256,
// 1, 1) — same as w4a16_gemv_silu_input (which iterates 8 K/step; this
// one iterates 16 K/step to keep the packed reads byte-aligned — the
// activation math per element is unchanged).
extern "C" __global__ void w3a16_gemv_silu_input(
    const __nv_bfloat16* __restrict__ gate_out,     // [1, K]
    const __nv_bfloat16* __restrict__ up_out,       // [1, K]
    const unsigned char* __restrict__ B_packed,      // [N, 3K/8]
    const unsigned char* __restrict__ B_scale,       // [N, K/16]
    const float scale2,
    __nv_bfloat16* __restrict__ C,                   // [1, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int row_bytes = (K >> 3) * 3;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[8];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 8) s_lut[threadIdx.x] = W3_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        uint4 g0 = ((const uint4*)gate_out)[k16 * 2];
        uint4 g1 = ((const uint4*)gate_out)[k16 * 2 + 1];
        uint4 u0 = ((const uint4*)up_out)[k16 * 2];
        uint4 u1 = ((const uint4*)up_out)[k16 * 2 + 1];
        const unsigned int g_raw[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
        const unsigned int u_raw[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};

        unsigned long long codes48 =
            w3_load6(B_packed + (unsigned long long)n * row_bytes + k16 * 6);

        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + k16];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        #pragma unroll
        for (int b = 0; b < 8; b++) {
            float w_lo = s_lut[(codes48 >> (6 * b)) & 7] * scale;
            float w_hi = s_lut[(codes48 >> (6 * b + 3)) & 7] * scale;

            __nv_bfloat16 g_lo, g_hi;
            *(unsigned short*)&g_lo = (unsigned short)(g_raw[b] & 0xFFFF);
            *(unsigned short*)&g_hi = (unsigned short)(g_raw[b] >> 16);
            float gf_lo = __bfloat162float(g_lo);
            float gf_hi = __bfloat162float(g_hi);

            __nv_bfloat16 u_lo, u_hi;
            *(unsigned short*)&u_lo = (unsigned short)(u_raw[b] & 0xFFFF);
            *(unsigned short*)&u_hi = (unsigned short)(u_raw[b] >> 16);

            // SiLU(gate) * up — identical expression to w4a16_gemv_silu_input.
            float a_lo = (gf_lo / (1.0f + __expf(-gf_lo))) * __bfloat162float(u_lo);
            float a_hi = (gf_hi / (1.0f + __expf(-gf_hi))) * __bfloat162float(u_hi);

            acc += a_lo * w_lo;
            acc += a_hi * w_hi;
        }
    }

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
