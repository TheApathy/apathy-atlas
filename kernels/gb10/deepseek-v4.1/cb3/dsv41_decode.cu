// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1 decode-size (M = 1) kernels: the FP8 dense GEMV, and a BIT-IDENTICAL split of
// `dsv41_hc_mixes` (see the end of the file).
//
// DeepSeek-V4.1 dense projections at DECODE (M = 1): an FP8 GEMV that reads the checkpoint's
// e4m3 weight and its UE8M0 32x32 block scales directly.
//
// Why: the prefill path dequantizes each FP8 weight to a transient bf16 copy and runs a
// 16-row cuBLASLt GEMM. At M = 1 that moves ~5 bytes per weight (1 read fp8, 2 written, 2
// read) for a matrix-vector product whose floor is 1 byte. Over the 7.2 GB of per-layer
// dense weights that is the difference between ~30 ms and ~150 ms per token.
//
// NUMERICS. e4m3 * 2^(s-127) is exact in fp32 (it is exactly what dsv41_dequant_fp8_ue8m0
// writes to bf16), and its product with a bf16 activation is exact in fp32. So the result
// differs from dequant + cuBLASLt only in fp32 ACCUMULATION ORDER, then rounds once to bf16.
// This is also the class of the reference's decode path: `v41_ref.dense` sends an FP8Weight
// at M <= 16 to `fp8_linear`'s Triton kernel (fp8 weight, bf16 activation, fp32 accumulate).
// A 16-weight partial sum lies inside one 32-wide scale block and is scaled by the power of
// two afterwards, which is exact.
//
// Grouped form (for wo_a): output row n belongs to group n / n_per_group, whose activation
// starts at x + group * x_group_stride. A plain linear passes n_per_group = N, stride 0.
//
// Grid (ceil(N / DD_WARPS)), Block (32 * DD_WARPS). One warp per output row; K % 16 == 0.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define DD_WARPS 8

__device__ __forceinline__ void dd_fp8x16(const uint4 v, float* f) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const __nv_fp8x2_storage_t p = (__nv_fp8x2_storage_t)((w[i] >> (16 * j)) & 0xffffu);
            const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(p, __NV_E4M3);
            const float2 ff = __half22float2(*reinterpret_cast<const __half2*>(&hr));
            f[4 * i + 2 * j] = ff.x;
            f[4 * i + 2 * j + 1] = ff.y;
        }
    }
}

__device__ __forceinline__ void dd_bf16x8(const uint4 v, float* f) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i] = __uint_as_float(w[i] << 16);
        f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

// R activation rows share every weight load. Row r's arithmetic is EXACTLY the one-row kernel's
// (same lane -> chunk mapping, same order, same reduction tree), so a row's result does not depend
// on how many rows share the call: M = 1..8 are bit-identical per row (chunk-invariant by
// construction). x rows are `ldx` elements apart, out rows `ldo` apart.
template <int R>
__device__ __forceinline__ void dd_gemv(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w,
                                        const uint8_t* __restrict__ scale, __nv_bfloat16* __restrict__ out,
                                        const unsigned N, const unsigned K, const unsigned n_per_group,
                                        const unsigned x_group_stride, const unsigned m, const unsigned ldx,
                                        const unsigned ldo) {
    const unsigned warp = threadIdx.x >> 5, L = threadIdx.x & 31;
    const unsigned n = blockIdx.x * DD_WARPS + warp;
    if (n >= N) return;
    const __nv_bfloat16* xg = x + (size_t)(n / n_per_group) * x_group_stride;
    const uint4* wr = reinterpret_cast<const uint4*>(w + (size_t)n * K);
    const uint8_t* sr = scale + (size_t)(n / 32) * ((K + 31) / 32);
    const unsigned chunks = K / 16;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
#pragma unroll 4
    for (unsigned c = L; c < chunks; c += 32) {
        const uint4 wv = __ldg(wr + c);
        const float s = ldexpf(1.0f, (int)__ldg(sr + (c >> 1)) - 127);
        float wf[16];
        dd_fp8x16(wv, wf);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if ((unsigned)r >= m) break;
            const __nv_bfloat16* xr = xg + (size_t)r * ldx;
            const uint4 xa = __ldg(reinterpret_cast<const uint4*>(xr + 16 * c));
            const uint4 xb = __ldg(reinterpret_cast<const uint4*>(xr + 16 * c + 8));
            float xf[16];
            dd_bf16x8(xa, xf);
            dd_bf16x8(xb, xf + 8);
            float p = 0.0f;
#pragma unroll
            for (int q = 0; q < 16; ++q) p = __fadd_rn(p, __fmul_rn(wf[q], xf[q]));
            acc[r] = __fadd_rn(acc[r], __fmul_rn(p, s));
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if ((unsigned)r >= m) break;
        float a = acc[r];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (L == 0) out[(size_t)r * ldo + n] = __float2bfloat16(a);
    }
}

extern "C" __global__ void __launch_bounds__(32 * DD_WARPS) dsv41_fp8_gemv_m1(
    const __nv_bfloat16* __restrict__ x,   // activation(s), bf16
    const uint8_t* __restrict__ w,         // e4m3 [N, K]
    const uint8_t* __restrict__ scale,     // ue8m0 [ceil(N/32), ceil(K/32)]
    __nv_bfloat16* __restrict__ out,       // bf16 [N]
    const unsigned N, const unsigned K,
    const unsigned n_per_group, const unsigned x_group_stride) {
    dd_gemv<1>(x, w, scale, out, N, K, n_per_group, x_group_stride, 1, 0, 0);
}

// M = 1..8 rows (speculative verify / draft blocks). Grid and block as the one-row kernel.
extern "C" __global__ void __launch_bounds__(32 * DD_WARPS) dsv41_fp8_gemv_m8(
    const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ scale,
    __nv_bfloat16* __restrict__ out, const unsigned N, const unsigned K, const unsigned n_per_group,
    const unsigned x_group_stride, const unsigned m, const unsigned ldx, const unsigned ldo) {
    dd_gemv<8>(x, w, scale, out, N, K, n_per_group, x_group_stride, m, ldx, ldo);
}

// ── hc_mixes at decode: the same arithmetic as dsv41_fwd.cu::dsv41_hc_mixes, spread over blocks ──
// dsv41_hc_mixes runs ONE block per token; at T = 1 that is one block reading the 1.97 MB fp32
// hc_fn — measured ~165 us per call, 2 calls per layer, ~13 ms per decode token. Here block m
// (0..23) computes mix m and block 24 computes the sum of squares, each with EXACTLY the
// per-thread sequential order (thread j: k = j, j + 256, ...) and the same shared-memory tree as
// the original, so every value is bit-identical; `dsv41_hc_mix_finish` then runs the original's
// epilogue verbatim. KEEP THE TWO IN SYNC: any change to dsv41_hc_mixes must be mirrored here.

#define HCD_BLOCK 256
#define HCD_HC 4
#define HCD_NMIX 24

__device__ __forceinline__ float hcd_block_sum(float v, float* red) {
    const unsigned tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (unsigned s = HCD_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

// Grid (HCD_NMIX + 1, T), Block 256. raw[t, m] = block-reduced dot, raw[t, 24] = sum of squares.
extern "C" __global__ void __launch_bounds__(HCD_BLOCK) dsv41_hc_mix_dot(
    const __nv_bfloat16* __restrict__ h, const float* __restrict__ fn, float* __restrict__ raw,
    const unsigned D) {
    __shared__ float red[HCD_BLOCK];
    const unsigned m = blockIdx.x, t = blockIdx.y;
    const unsigned R = HCD_HC * D;
    const __nv_bfloat16* x = h + (size_t)t * R;
    float acc = 0.f;
    if (m < HCD_NMIX) {
        const float* f = fn + (size_t)m * R;
        for (unsigned k = threadIdx.x; k < R; k += HCD_BLOCK) acc += __bfloat162float(x[k]) * f[k];
    } else {
        for (unsigned k = threadIdx.x; k < R; k += HCD_BLOCK) {
            const float v = __bfloat162float(x[k]);
            acc += v * v;
        }
    }
    const float r = hcd_block_sum(acc, red);
    if (threadIdx.x == 0) raw[t * (HCD_NMIX + 1) + m] = r;
}

// Grid (T), Block 32. Lane (i, j) = (lane / 4, lane % 4) owns comb[i][j] for lanes 0..15; every
// row / column sum gathers the four values by shuffle and adds them in the ORIGINAL sequential
// order (0.f + v0 + v1 + v2 + v3), and every division is the same division, so each value is
// bit-identical to the original single-thread epilogue -- just not serialised over 16 entries.
extern "C" __global__ void dsv41_hc_mix_finish(const float* __restrict__ raw,
                                               const float* __restrict__ scale,
                                               const float* __restrict__ base,
                                               float* __restrict__ pre, float* __restrict__ post,
                                               float* __restrict__ comb, const unsigned D,
                                               const unsigned iters, const float eps,
                                               const float hc_eps) {
    const unsigned t = blockIdx.x, lane = threadIdx.x & 31;
    const unsigned R = HCD_HC * D;
    const unsigned HC = HCD_HC;
    const float* rw = raw + t * (HCD_NMIX + 1);
    const float rs = rsqrtf(rw[HCD_NMIX] / (float)R + eps);
    if (lane < HC) {
        const float vp = (rw[lane] * rs) * scale[0] + base[lane];
        pre[t * HC + lane] = 1.f / (1.f + expf(-vp)) + hc_eps;
    } else if (lane < 2 * HC) {
        const unsigned i = lane - HC;
        const float vq = (rw[HC + i] * rs) * scale[1] + base[HC + i];
        post[t * HC + i] = 2.f * (1.f / (1.f + expf(-vq)));
    }
    const unsigned i = (lane >> 2) & 3u, j = lane & 3u;  // lanes 16..31 mirror 0..15, unused
    const unsigned m = 2 * HC + i * HC + j;
    float c = (rw[m] * rs) * scale[2] + base[m];
    auto row = [&](float v, unsigned q) { return __shfl_sync(0xffffffffu, v, (lane & ~3u) + q); };
    auto col = [&](float v, unsigned q) { return __shfl_sync(0xffffffffu, v, (lane & 16u) + q * 4 + j); };
    float mx = -INFINITY;
#pragma unroll
    for (unsigned q = 0; q < 4; ++q) mx = fmaxf(mx, row(c, q));
    c = expf(c - mx);
    float z = 0.f;
#pragma unroll
    for (unsigned q = 0; q < 4; ++q) z += row(c, q);
    c = c / z + hc_eps;
    for (unsigned it = 0; it < iters; ++it) {
        if (it > 0) {
            float zr = 0.f;
#pragma unroll
            for (unsigned q = 0; q < 4; ++q) zr += row(c, q);
            zr += hc_eps;
            c /= zr;
        }
        float zc = 0.f;
#pragma unroll
        for (unsigned q = 0; q < 4; ++q) zc += col(c, q);
        zc += hc_eps;
        c /= zc;
    }
    if (lane < 16) comb[t * HC * HC + i * HC + j] = c;
}
