// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1 DSpark (MTP block) routed MoE at decode size: 128 FP4 experts, top-3, for the
// draft block (T <= 8 rows). The same structure as cb3_moe_decode.cu (GPU routing, grouped GEMV,
// no host sync, no bf16 expert copy), for the other expert format.
//
// Expert format (Python `tools/fp4_moe.py`, integrate's `mtp::Fp4Linear`):
//   w1, w3  packed e2m1 [2304, 2560] bytes (K = 5120; low nibble = even k, high = odd k)
//   w2      packed e2m1 [5120, 1152] bytes (K = 2304)
//   s1..s3  UE8M0, one byte per 32 consecutive k of a row: [N, K / 32]
// One 16-byte load is exactly one 32-weight scale group of one row.
//
// NUMERICS. e2m1 x 2^(s-127) is exact in fp32; its product with a bf16 activation is exact; a
// group's 32 products are summed in order and scaled once, then groups accumulate in fp32 in the
// lane's chunk order and reduce by the fixed butterfly. Rounding points as the reference
// (`fp4_moe.py`): h = bf16(silu(min(g, L)) * clamp(u) * w), out = bf16(sum over picks, fp32).
// The Python kernel casts activations to fp16 before its dot; bf16 -> fp16 is exact for every
// activation above fp16's subnormal range, so the difference is order-level only. Draft numerics
// cannot change the spec OUTPUT (verification decides every token); they only move acceptance.
//
// Routing (`model.py::moe` for n_experts == 128): scores = sqrt(softplus(y @ gate^T)) fp32,
// keys = scores + gate.bias (text bias only; no residency mask: all 128 resident), top-3 with
// ties to the lower id, weights = scores / (sum + 1e-20) * route_scale.
//
// Layouts (per call, T <= 8):  y bf16 [T, 5120] ; groups i32 [1 + 24 * GI] ; row_w f32 [T*3] ;
// h bf16 [T*3, 2304] ; down f32 [T*3, 5120] ; out bf16 [T, 5120]. Experts are passed as a
// device table of addresses `ptrs` u64 [128][6] = {w1, s1, w3, s3, w2, s2} per expert (the
// checkpoint's tensors stay where the weight store put them; no arena copy).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <stdint.h>

#define MTP_E 128
#define MTP_K 3
#define MTP_MAXT 8
#define MTP_MAXR 8
#define MTP_GI (2 + MTP_MAXR)
#define MTP_WARPS 8

__device__ __forceinline__ float mtp_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// 16 packed bytes -> 32 floats in k order (byte j: k = 2j from the low nibble, 2j + 1 high).
__device__ __forceinline__ void mtp_fp4x32(const uint4 v, float* w) {
    const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)((u[i] >> (8 * b)) & 0xffu), __NV_E2M1);
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
            w[8 * i + 2 * b] = f.x;
            w[8 * i + 2 * b + 1] = f.y;
        }
    }
}

__device__ __forceinline__ void mtp_bf16x8(const uint4 v, float* f) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i] = __uint_as_float(w[i] << 16);
        f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

// One FP4 row (K = 32 * groups) against up to R activation rows xr[r] (bf16, K wide).
// Lane L handles groups c = L, L + 32, ...: all its weight and scale loads are issued first.
template <int R, int K>
__device__ __forceinline__ void mtp_fp4_row(const uint8_t* __restrict__ w_row, const uint8_t* __restrict__ s_row,
                                            const __nv_bfloat16* const* xr, int nr, float* acc) {
    constexpr int G = K / 32;
    constexpr int PER = (G + 31) / 32;
    const int L = threadIdx.x & 31;
    uint4 wv[PER];
    uint32_t sv[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int c = L + 32 * i;
        const bool live = c < G;
        wv[i] = live ? __ldg(reinterpret_cast<const uint4*>(w_row) + c) : make_uint4(0, 0, 0, 0);
        sv[i] = live ? (uint32_t)__ldg(s_row + c) : 127u;
    }
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int c = L + 32 * i;
        if (c >= G) break;
        float w[32];
        mtp_fp4x32(wv[i], w);
        const float s = ldexpf(1.0f, (int)sv[i] - 127);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r >= nr) break;
            float x[32];
#pragma unroll
            for (int q = 0; q < 4; ++q) mtp_bf16x8(__ldg(reinterpret_cast<const uint4*>(xr[r] + 32 * c) + q), x + 8 * q);
            float p = 0.0f;
#pragma unroll
            for (int q = 0; q < 32; ++q) p = __fadd_rn(p, __fmul_rn(w[q], x[q]));
            acc[r] = __fadd_rn(acc[r], __fmul_rn(p, s));
        }
    }
}

// ── router logits: logits[t, e] = sum_k f32(y[t,k]) * f32(gate[e,k]) (bf16 weight, exact widening)
// Grid (128 / MTP_WARPS, T), Block (32 * MTP_WARPS).
extern "C" __global__ void __launch_bounds__(32 * MTP_WARPS) dsv41_mtp_router_logits(
    const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ gate, float* __restrict__ logits) {
    constexpr int K = 5120;
    const int t = blockIdx.y, warp = threadIdx.x >> 5, L = threadIdx.x & 31;
    const int e = blockIdx.x * MTP_WARPS + warp;
    if (e >= MTP_E) return;
    const uint2* w = reinterpret_cast<const uint2*>(gate + (size_t)e * K);
    const uint2* x = reinterpret_cast<const uint2*>(y + (size_t)t * K);
    float acc = 0.0f;
#pragma unroll 8
    for (int i = L; i < K / 4; i += 32) {
        const uint2 wb = __ldg(w + i), xv = __ldg(x + i);
        acc = __fadd_rn(acc, __fmul_rn(__uint_as_float(wb.x << 16), __uint_as_float(xv.x << 16)));
        acc = __fadd_rn(acc, __fmul_rn(__uint_as_float(wb.x & 0xffff0000u), __uint_as_float(xv.x & 0xffff0000u)));
        acc = __fadd_rn(acc, __fmul_rn(__uint_as_float(wb.y << 16), __uint_as_float(xv.y << 16)));
        acc = __fadd_rn(acc, __fmul_rn(__uint_as_float(wb.y & 0xffff0000u), __uint_as_float(xv.y & 0xffff0000u)));
    }
    acc = mtp_warp_sum(acc);
    if (L == 0) logits[t * MTP_E + e] = acc;
}

// ── routing: top-3 of 128, grouped by expert. Grid (1), Block (32 * T).
extern "C" __global__ void dsv41_mtp_route(const float* __restrict__ logits, const float* __restrict__ bias,
                                           int* __restrict__ groups, float* __restrict__ row_w,
                                           int* __restrict__ sel, const int T, const float route_scale) {
    __shared__ int s_e[MTP_MAXT * MTP_K];
    const int t = threadIdx.x >> 5, L = threadIdx.x & 31;
    if (t < T) {
        float sc[4], key[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int e = L + 32 * i;
            const float x = logits[t * MTP_E + e];
            sc[i] = sqrtf(x > 20.0f ? x : log1pf(expf(x)));
            key[i] = sc[i] + bias[e];
        }
        int picked[MTP_K];
        float ps[MTP_K];
        for (int k = 0; k < MTP_K; ++k) {
            float bk = -INFINITY;
            int bi = 0x7fffffff;
#pragma unroll
            for (int i = 0; i < 4; ++i)
                if (key[i] > bk) { bk = key[i]; bi = L + 32 * i; }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ok = __shfl_xor_sync(0xffffffffu, bk, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ok > bk || (ok == bk && oi < bi)) { bk = ok; bi = oi; }
            }
            picked[k] = bi;
            float s = 0.0f;
            if ((bi & 31) == L) {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    if (L + 32 * i == bi) { s = sc[i]; key[i] = -INFINITY; }
            }
            ps[k] = __shfl_sync(0xffffffffu, s, bi & 31);
        }
        if (L == 0) {
            float sum = 0.0f;
            for (int k = 0; k < MTP_K; ++k) sum += ps[k];
            const float den = sum + 1e-20f;
            for (int k = 0; k < MTP_K; ++k) {
                row_w[t * MTP_K + k] = ps[k] / den * route_scale;
                sel[t * MTP_K + k] = picked[k];
                s_e[t * MTP_K + k] = picked[k];
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int n = 0;
        for (int p = 0; p < T * MTP_K; ++p) {
            int g = 0;
            while (g < n && groups[1 + g * MTP_GI] != s_e[p]) ++g;
            int* G = groups + 1 + g * MTP_GI;
            if (g == n) { G[0] = s_e[p]; G[1] = 0; ++n; }
            G[2 + G[1]] = p;
            G[1] += 1;
        }
        groups[0] = n;
    }
}

// ── gate + up + SwiGLU. Grid (2304 / MTP_WARPS, max_groups), Block (32 * MTP_WARPS).
template <int R>
__device__ __forceinline__ void mtp_gateup(const unsigned long long* __restrict__ ptrs,
                                           const __nv_bfloat16* __restrict__ y, const int* __restrict__ groups,
                                           const float* __restrict__ row_w, __nv_bfloat16* __restrict__ h, float limit) {
    constexpr int K = 5120, N = 2304;
    const int g = blockIdx.y;
    if (g >= groups[0]) return;
    const int warp = threadIdx.x >> 5, L = threadIdx.x & 31;
    const int n = blockIdx.x * MTP_WARPS + warp;
    if (n >= N) return;
    const int* G = groups + 1 + g * MTP_GI;
    const unsigned long long e = (unsigned long long)G[0];
    const int nr = min(G[1], R);
    const __nv_bfloat16* xr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) xr[r] = y + (size_t)(G[2 + (r < nr ? r : 0)] / MTP_K) * K;
    float ag[R], au[R];
#pragma unroll
    for (int r = 0; r < R; ++r) ag[r] = au[r] = 0.0f;
    const size_t rw = (size_t)n * (K / 2), rs = (size_t)n * (K / 32);
    const unsigned long long* P = ptrs + e * 6;
    mtp_fp4_row<R, K>(reinterpret_cast<const uint8_t*>(P[0]) + rw, reinterpret_cast<const uint8_t*>(P[1]) + rs, xr, nr, ag);
    mtp_fp4_row<R, K>(reinterpret_cast<const uint8_t*>(P[2]) + rw, reinterpret_cast<const uint8_t*>(P[3]) + rs, xr, nr, au);
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (r >= nr) break;
        const float gs = mtp_warp_sum(ag[r]), us = mtp_warp_sum(au[r]);
        if (L == 0) {
            const int row = G[2 + r];
            const float gv = fminf(gs, limit);
            const float uv = fminf(fmaxf(us, -limit), limit);
            const float sig = 1.0f / (1.0f + expf(-gv));
            h[(size_t)row * N + n] = __float2bfloat16(gv * sig * uv * row_w[row]);
        }
    }
}

// ── down. Grid (5120 / MTP_WARPS, max_groups), Block (32 * MTP_WARPS). K = 2304 (72 groups).
template <int R>
__device__ __forceinline__ void mtp_down(const unsigned long long* __restrict__ ptrs,
                                         const __nv_bfloat16* __restrict__ h, const int* __restrict__ groups,
                                         float* __restrict__ down) {
    constexpr int K = 2304, N = 5120;
    const int g = blockIdx.y;
    if (g >= groups[0]) return;
    const int warp = threadIdx.x >> 5, L = threadIdx.x & 31;
    const int n = blockIdx.x * MTP_WARPS + warp;
    if (n >= N) return;
    const int* G = groups + 1 + g * MTP_GI;
    const unsigned long long e = (unsigned long long)G[0];
    const int nr = min(G[1], R);
    const __nv_bfloat16* xr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) xr[r] = h + (size_t)G[2 + (r < nr ? r : 0)] * K;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    const unsigned long long* P = ptrs + e * 6;
    mtp_fp4_row<R, K>(reinterpret_cast<const uint8_t*>(P[4]) + (size_t)n * (K / 2),
                      reinterpret_cast<const uint8_t*>(P[5]) + (size_t)n * (K / 32), xr, nr, acc);
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (r >= nr) break;
        const float v = mtp_warp_sum(acc[r]);
        if (L == 0) down[(size_t)G[2 + r] * N + n] = v;
    }
}

#define MTP_ENTRIES(R)                                                                                  \
    extern "C" __global__ void __launch_bounds__(32 * MTP_WARPS) dsv41_mtp_gateup_r##R(                 \
        const unsigned long long* ptrs, const __nv_bfloat16* y, const int* groups, const float* row_w,  \
        __nv_bfloat16* h, float limit) {                                                                \
        mtp_gateup<R>(ptrs, y, groups, row_w, h, limit);                                                \
    }                                                                                                   \
    extern "C" __global__ void __launch_bounds__(32 * MTP_WARPS) dsv41_mtp_down_r##R(                   \
        const unsigned long long* ptrs, const __nv_bfloat16* h, const int* groups, float* down) {       \
        mtp_down<R>(ptrs, h, groups, down);                                                             \
    }
MTP_ENTRIES(1)
MTP_ENTRIES(8)

// ── sum over the 3 picks. Grid (ceil(5120 / 256), T), Block (256).
extern "C" __global__ void dsv41_mtp_sum(const float* __restrict__ down, __nv_bfloat16* __restrict__ out) {
    constexpr int N = 5120;
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < MTP_K; ++k) acc += down[(size_t)(t * MTP_K + k) * N + c];
    out[(size_t)t * N + c] = __float2bfloat16(acc);
}
