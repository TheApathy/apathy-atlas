// SPDX-License-Identifier: AGPL-3.0-only

// DeepSeek-V4.1 forward glue: everything between the GEMMs that no lane-specific kernel owns.
//
// Every kernel here mirrors ONE function of the Python reference
// (dsv41-prefill-work/engine/model.py + tools/v41_ref.py + tools/hc_kernels.py) and says which.
// The math is done in fp32 and rounded once, where the reference rounds. None of these is a
// performance kernel; they are the numerics contract. Everything is per-row, so the result
// of a row never depends on how many rows are in the call (chunk invariance by construction).
//
// Layouts: the hyper-connection stream `h` is bf16 [T, HC, D] (HC = 4, D = 5120), pre/post
// are fp32 [T, HC], comb is fp32 [T, HC, HC] row-major (comb[i][j]).

#include <cuda_bf16.h>
#include <stdint.h>

#define DSV41_BLOCK 256
#define DSV41_HC 4
#define DSV41_NMIX 24  // (2 + HC) * HC

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }

// e4m3fn byte -> float (exact).
__device__ __forceinline__ float e4m3_to_f32(uint8_t b) {
    const uint32_t s = b >> 7, e = (b >> 3) & 0xF, m = b & 0x7;
    float v;
    if (e == 0) {
        v = ldexpf((float)m, -9);  // subnormal: m/8 * 2^-6
    } else if (e == 0xF && m == 0x7) {
        v = __int_as_float(0x7fc00000);  // NaN (e4m3fn has no inf)
    } else {
        v = ldexpf(1.0f + (float)m / 8.0f, (int)e - 7);
    }
    return s ? -v : v;
}

// Block reduction of one float over DSV41_BLOCK threads; result valid in every thread.
__device__ __forceinline__ float block_sum(float v, float* red) {
    const unsigned tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (unsigned s = DSV41_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

// ── dequant: v41_ref.dequant_fp8_block ────────────────────────────────────────
// fp8 e4m3 [N, K] with UE8M0 scale bytes [ceil(N/32), ceil(K/32)] -> bf16 [N, K].
// (w.float() * 2^(s-127)).to(bf16): exact for every finite e4m3 value.
// Grid: (ceil(K/256), N)  Block: 256.
extern "C" __global__ void dsv41_dequant_fp8_ue8m0(
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ scale,
    __nv_bfloat16* __restrict__ out, const unsigned N, const unsigned K) {
    const unsigned n = blockIdx.y;
    const unsigned k = blockIdx.x * DSV41_BLOCK + threadIdx.x;
    if (n >= N || k >= K) return;
    const unsigned sk = (K + 31) / 32;
    const float s = exp2f((float)scale[(size_t)(n / 32) * sk + k / 32] - 127.0f);
    out[(size_t)n * K + k] = __float2bfloat16(e4m3_to_f32(w[(size_t)n * K + k]) * s);
}

// ── casts ─────────────────────────────────────────────────────────────────────
extern "C" __global__ void dsv41_bf16_to_f32(const __nv_bfloat16* __restrict__ x,
                                             float* __restrict__ y, const unsigned long long n) {
    const unsigned long long i = (unsigned long long)blockIdx.x * DSV41_BLOCK + threadIdx.x;
    if (i < n) y[i] = bf(x[i]);
}

extern "C" __global__ void dsv41_f32_to_bf16(const float* __restrict__ x,
                                             __nv_bfloat16* __restrict__ y, const unsigned long long n) {
    const unsigned long long i = (unsigned long long)blockIdx.x * DSV41_BLOCK + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(x[i]);
}

// out = bf16(f32(a) + f32(b)).  model.moe: (routed.float() + shared.float()).to(bf16).
extern "C" __global__ void dsv41_add_bf16(const __nv_bfloat16* __restrict__ a,
                                          const __nv_bfloat16* __restrict__ b,
                                          __nv_bfloat16* __restrict__ out, const unsigned long long n) {
    const unsigned long long i = (unsigned long long)blockIdx.x * DSV41_BLOCK + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(bf(a[i]) + bf(b[i]));
}

// ── embedding: model.forward `self.W.embed[ids]` ──────────────────────────────
// Grid: (T)  Block: 256.
extern "C" __global__ void dsv41_embed(const __nv_bfloat16* __restrict__ table,
                                       const uint32_t* __restrict__ ids,
                                       __nv_bfloat16* __restrict__ out, const unsigned D) {
    const unsigned t = blockIdx.x;
    const __nv_bfloat16* row = table + (size_t)ids[t] * D;
    for (unsigned d = threadIdx.x; d < D; d += DSV41_BLOCK) out[(size_t)t * D + d] = row[d];
}

// ── hc_expand: `h = embeds.unsqueeze(1).repeat(1, hc, 1)`, pre_mix = one-hot(0) ──
// Grid: (T)  Block: 256.
extern "C" __global__ void dsv41_hc_expand(const __nv_bfloat16* __restrict__ x,
                                           __nv_bfloat16* __restrict__ h,
                                           float* __restrict__ pre_mix, const unsigned D) {
    const unsigned t = blockIdx.x;
    for (unsigned d = threadIdx.x; d < D; d += DSV41_BLOCK) {
        const __nv_bfloat16 v = x[(size_t)t * D + d];
#pragma unroll
        for (unsigned i = 0; i < DSV41_HC; ++i) h[((size_t)t * DSV41_HC + i) * D + d] = v;
    }
    if (threadIdx.x < DSV41_HC) pre_mix[t * DSV41_HC + threadIdx.x] = threadIdx.x == 0 ? 1.0f : 0.0f;
}

// ── rmsnorm: v41_ref.rmsnorm / hc_kernels.rmsnorm ─────────────────────────────
// y = bf16(w * (x * rsqrt(mean(x^2) + eps))), fp32 math. x, y bf16 [R, N]; w bf16 [N].
// Grid: (R)  Block: 256.
extern "C" __global__ void dsv41_rmsnorm(const __nv_bfloat16* __restrict__ x,
                                         const __nv_bfloat16* __restrict__ w,
                                         __nv_bfloat16* __restrict__ y, const unsigned N,
                                         const float eps) {
    __shared__ float red[DSV41_BLOCK];
    const size_t r = blockIdx.x;
    const __nv_bfloat16* xr = x + r * N;
    float ss = 0.f;
    for (unsigned k = threadIdx.x; k < N; k += DSV41_BLOCK) {
        const float v = bf(xr[k]);
        ss += v * v;
    }
    const float rs = rsqrtf(block_sum(ss, red) / (float)N + eps);
    for (unsigned k = threadIdx.x; k < N; k += DSV41_BLOCK)
        y[r * N + k] = __float2bfloat16(bf(w[k]) * (bf(xr[k]) * rs));
}

// ── hc_mixes: v41_ref.hc_mixes + hc_split_sinkhorn ────────────────────────────
// x = h[t] flattened to [HC*D] (fp32), mixes[m] = (x . fn[m]) * rsqrt(mean(x^2) + eps), then
//   pre  = sigmoid(mixes[:HC]   * s0 + base[:HC])      + hc_eps
//   post = 2 * sigmoid(mixes[HC:2HC] * s1 + base[HC:2HC])
//   comb = softmax_j(mixes[2HC:] * s2 + base[2HC:]) + hc_eps; comb /= colsum + hc_eps;
//          (iters - 1) x { comb /= rowsum + hc_eps; comb /= colsum + hc_eps }
// NO final exact column projection: that is a V4-0731 kernel's deviation, not the reference.
// One block per token. Grid: (T)  Block: 256.
extern "C" __global__ void dsv41_hc_mixes(const __nv_bfloat16* __restrict__ h,
                                          const float* __restrict__ fn,     // [24, HC*D]
                                          const float* __restrict__ scale,  // [3]
                                          const float* __restrict__ base,   // [24]
                                          float* __restrict__ pre, float* __restrict__ post,
                                          float* __restrict__ comb, const unsigned D,
                                          const unsigned iters, const float eps,
                                          const float hc_eps) {
    __shared__ float red[DSV41_BLOCK];
    __shared__ float mix[DSV41_NMIX];
    const unsigned t = blockIdx.x;
    const unsigned R = DSV41_HC * D;
    const __nv_bfloat16* x = h + (size_t)t * R;
    float acc[DSV41_NMIX];
#pragma unroll
    for (int m = 0; m < DSV41_NMIX; ++m) acc[m] = 0.f;
    float ss = 0.f;
    for (unsigned k = threadIdx.x; k < R; k += DSV41_BLOCK) {
        const float v = bf(x[k]);
        ss += v * v;
#pragma unroll
        for (int m = 0; m < DSV41_NMIX; ++m) acc[m] += v * fn[(size_t)m * R + k];
    }
    const float rs = rsqrtf(block_sum(ss, red) / (float)R + eps);
    for (int m = 0; m < DSV41_NMIX; ++m) {
        const float r = block_sum(acc[m], red);
        if (threadIdx.x == 0) mix[m] = r * rs;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    const unsigned HC = DSV41_HC;
    for (unsigned i = 0; i < HC; ++i) {
        const float vp = mix[i] * scale[0] + base[i];
        pre[t * HC + i] = 1.f / (1.f + expf(-vp)) + hc_eps;
        const float vq = mix[HC + i] * scale[1] + base[HC + i];
        post[t * HC + i] = 2.f * (1.f / (1.f + expf(-vq)));
    }
    float c[DSV41_HC * DSV41_HC];
    for (unsigned i = 0; i < HC; ++i) {
        float mx = -INFINITY;
        for (unsigned j = 0; j < HC; ++j) {
            c[i * HC + j] = mix[2 * HC + i * HC + j] * scale[2] + base[2 * HC + i * HC + j];
            mx = fmaxf(mx, c[i * HC + j]);
        }
        float z = 0.f;
        for (unsigned j = 0; j < HC; ++j) {
            c[i * HC + j] = expf(c[i * HC + j] - mx);
            z += c[i * HC + j];
        }
        for (unsigned j = 0; j < HC; ++j) c[i * HC + j] = c[i * HC + j] / z + hc_eps;
    }
    for (unsigned it = 0; it < iters; ++it) {
        if (it > 0) {
            for (unsigned i = 0; i < HC; ++i) {
                float z = 0.f;
                for (unsigned j = 0; j < HC; ++j) z += c[i * HC + j];
                z += hc_eps;
                for (unsigned j = 0; j < HC; ++j) c[i * HC + j] /= z;
            }
        }
        for (unsigned j = 0; j < HC; ++j) {
            float z = 0.f;
            for (unsigned i = 0; i < HC; ++i) z += c[i * HC + j];
            z += hc_eps;
            for (unsigned i = 0; i < HC; ++i) c[i * HC + j] /= z;
        }
    }
    for (unsigned q = 0; q < HC * HC; ++q) comb[t * HC * HC + q] = c[q];
}

// ── hc_mixes with ONE batched reduction tree (prefill) ─────────────────────────
// dsv41_hc_mixes runs 25 sequential block_sum()s per token (8 barriers each). Here all 25 sums
// (24 mix dots + the sum of squares) go through ONE tree with block_sum's exact pairings:
// shared-memory steps s = 128, 64, 32 (3 barriers), then warp shuffles s = 16..1, where lane l
// adds lane l+s's current value -- the same red[tid] += red[tid + s] as block_sum. The per-thread
// accumulation is dsv41_hc_mixes' (k stride, fma chain), so the result is BIT-IDENTICAL.
#define DSV41_NSUM (DSV41_NMIX + 1)

// Mixes of one token row x = h[t] (HC*D bf16). On return (after a barrier) mix_out[m] holds
// dsv41_hc_mixes' mix[m] (= dot * rsqrt(mean(x^2) + eps)) in every thread's view.
__device__ __forceinline__ void hc_mix_dots(const __nv_bfloat16* __restrict__ x,
                                            const float* __restrict__ fn, const unsigned R,
                                            const float eps, float (*red)[DSV41_BLOCK],
                                            float* mix_out) {
    const unsigned tid = threadIdx.x;
    float acc[DSV41_NMIX];
#pragma unroll
    for (int m = 0; m < DSV41_NMIX; ++m) acc[m] = 0.f;
    float ss = 0.f;
    for (unsigned k = tid; k < R; k += DSV41_BLOCK) {
        const float v = bf(x[k]);
        ss += v * v;
#pragma unroll
        for (int m = 0; m < DSV41_NMIX; ++m) acc[m] += v * fn[(size_t)m * R + k];
    }
#pragma unroll
    for (int m = 0; m < DSV41_NMIX; ++m) red[m][tid] = acc[m];
    red[DSV41_NMIX][tid] = ss;
    __syncthreads();
    for (unsigned s = DSV41_BLOCK / 2; s >= 32; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int m = 0; m < DSV41_NSUM; ++m) red[m][tid] += red[m][tid + s];
        }
        __syncthreads();
    }
    const unsigned warp = tid / 32, lane = tid % 32;
    for (unsigned m = warp; m < DSV41_NSUM; m += DSV41_BLOCK / 32) {
        float v = red[m][lane];
#pragma unroll
        for (unsigned s = 16; s > 0; s >>= 1) v += __shfl_down_sync(0xffffffffu, v, s);
        if (lane == 0) red[m][0] = v;
    }
    __syncthreads();
    if (tid == 0) {
        const float rs = rsqrtf(red[DSV41_NMIX][0] / (float)R + eps);
        for (int m = 0; m < DSV41_NMIX; ++m) mix_out[m] = red[m][0] * rs;
    }
    __syncthreads();
}

// The Sinkhorn epilogue of token t from its mixes, run by ONE WARP (all 32 lanes): decode's
// dsv41_hc_mix_finish arithmetic (dsv41_decode.cu). Lane (i, j) = (lane / 4, lane % 4) owns
// comb[i][j]; every row / column sum gathers its four values by shuffle and adds them in the
// original sequential order (0.f + v0 + v1 + v2 + v3), every division is the same division, so
// each value is bit-identical to dsv41_hc_mixes' single-thread epilogue.
__device__ __forceinline__ void hc_mix_epilogue_warp(const float* mix, const unsigned t,
                                                     const float* __restrict__ scale,
                                                     const float* __restrict__ base, float* pre,
                                                     float* post, float* comb,
                                                     const unsigned iters, const float hc_eps) {
    const unsigned lane = threadIdx.x & 31;
    const unsigned HC = DSV41_HC;
    if (lane < HC) {
        const float vp = mix[lane] * scale[0] + base[lane];
        pre[t * HC + lane] = 1.f / (1.f + expf(-vp)) + hc_eps;
    } else if (lane < 2 * HC) {
        const unsigned i = lane - HC;
        const float vq = mix[HC + i] * scale[1] + base[HC + i];
        post[t * HC + i] = 2.f * (1.f / (1.f + expf(-vq)));
    }
    const unsigned i = (lane >> 2) & 3u, j = lane & 3u;  // lanes 16..31 mirror 0..15, unused
    const unsigned m = 2 * HC + i * HC + j;
    float c = mix[m] * scale[2] + base[m];
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

// Grid: (T)  Block: 256. Bit-identical to dsv41_hc_mixes.
extern "C" __global__ void dsv41_hc_mixes_v2(const __nv_bfloat16* __restrict__ h,
                                             const float* __restrict__ fn,
                                             const float* __restrict__ scale,
                                             const float* __restrict__ base,
                                             float* __restrict__ pre, float* __restrict__ post,
                                             float* __restrict__ comb, const unsigned D,
                                             const unsigned iters, const float eps,
                                             const float hc_eps) {
    __shared__ float red[DSV41_NSUM][DSV41_BLOCK];
    __shared__ float mix[DSV41_NMIX];
    const unsigned t = blockIdx.x;
    const unsigned R = DSV41_HC * D;
    hc_mix_dots(h + (size_t)t * R, fn, R, eps, red, mix);
    if (threadIdx.x < 32) hc_mix_epilogue_warp(mix, t, scale, base, pre, post, comb, iters, hc_eps);
}

// ── fused mHC stream passes (prefill), one token per block ────────────────────
//   [hc_post(y = ya (+ yb))] -> hc_mixes -> hc_pre(side_pre) -> rmsnorm, in place on h / into x.
// Each stage is the separate kernel's per-thread arithmetic (post/pre/norm) or dsv41_hc_mixes_v2's
// identical tree, so the result is BIT-IDENTICAL to the unfused sequence. `side_pre` may alias
// `pre_o` (BlockControl::OwnPre): it is read after the epilogue wrote it, as in the unfused order.
// Grid: (T)  Block: 256.
extern "C" __global__ void dsv41_hc_fused_v2(
    const __nv_bfloat16* __restrict__ ya, const __nv_bfloat16* __restrict__ yb,
    const float* __restrict__ post_in, const float* __restrict__ comb_in,
    __nv_bfloat16* h,
    const float* __restrict__ fn, const float* __restrict__ scale, const float* __restrict__ base,
    float* pre_o, float* __restrict__ post_o, float* __restrict__ comb_o,
    const float* side_pre, const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* x, const unsigned D, const unsigned iters,
    const float eps, const float hc_eps) {
    __shared__ float red[DSV41_NSUM][DSV41_BLOCK];
    __shared__ float mix[DSV41_NMIX];
    const unsigned t = blockIdx.x;
    const unsigned R = DSV41_HC * D;
    __nv_bfloat16* ht = h + (size_t)t * R;
    if (ya != nullptr) {
        const float* po = post_in + t * DSV41_HC;
        const float* cb = comb_in + t * DSV41_HC * DSV41_HC;
        for (unsigned d = threadIdx.x; d < D; d += DSV41_BLOCK) {
            const size_t yi = (size_t)t * D + d;
            const float yv = yb != nullptr ? bf(__float2bfloat16(bf(ya[yi]) + bf(yb[yi]))) : bf(ya[yi]);
            float r[DSV41_HC];
#pragma unroll
            for (unsigned i = 0; i < DSV41_HC; ++i) r[i] = bf(ht[i * D + d]);
#pragma unroll
            for (unsigned j = 0; j < DSV41_HC; ++j) {
                float acc = po[j] * yv;
#pragma unroll
                for (unsigned i = 0; i < DSV41_HC; ++i) acc += cb[i * DSV41_HC + j] * r[i];
                ht[j * D + d] = __float2bfloat16(acc);
            }
        }
        __syncthreads();
    }
    hc_mix_dots(ht, fn, R, eps, red, mix);
    if (threadIdx.x < 32) hc_mix_epilogue_warp(mix, t, scale, base, pre_o, post_o, comb_o, iters, hc_eps);
    __syncthreads();
    const float* p = side_pre + t * DSV41_HC;
    float ssq = 0.f;
    for (unsigned d = threadIdx.x; d < D; d += DSV41_BLOCK) {
        float acc = 0.f;
#pragma unroll
        for (unsigned i = 0; i < DSV41_HC; ++i) acc += p[i] * bf(ht[i * D + d]);
        const __nv_bfloat16 xv = __float2bfloat16(acc);
        x[(size_t)t * D + d] = xv;
        const float v = bf(xv);
        ssq += v * v;
    }
    const float rs = rsqrtf(block_sum(ssq, red[0]) / (float)D + eps);
    for (unsigned d = threadIdx.x; d < D; d += DSV41_BLOCK)
        x[(size_t)t * D + d] = __float2bfloat16(bf(norm_w[d]) * (bf(x[(size_t)t * D + d]) * rs));
}

// ── fused mHC v3: the token's h row held in shared memory across the stages ──
// dsv41_hc_fused_v2 reads h from global memory three times (post, mixes, pre). v3 keeps the row
// ([HC, D] bf16 = 40 KB at D = 5120) in shared memory: post writes it to both, mixes and pre read
// it from shared. The 25 sums still go through block_sum's exact tree, in 5 batches of 5 (a
// smaller scratch that keeps the block under 48 KB of static shared memory; each sum's pairings
// are unchanged). Every value is BIT-IDENTICAL to v2 / the separate kernels. D <= DSV41_DMAX.
#define DSV41_DMAX 5120
#define DSV41_SUM_BATCH 5
extern "C" __global__ void __launch_bounds__(DSV41_BLOCK) dsv41_hc_fused_v3(
    const __nv_bfloat16* __restrict__ ya, const __nv_bfloat16* __restrict__ yb,
    const float* __restrict__ post_in, const float* __restrict__ comb_in,
    __nv_bfloat16* h,
    const float* __restrict__ fn, const float* __restrict__ scale, const float* __restrict__ base,
    float* pre_o, float* __restrict__ post_o, float* __restrict__ comb_o,
    const float* side_pre, const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* x, const unsigned D, const unsigned iters,
    const float eps, const float hc_eps) {
    __shared__ __nv_bfloat16 hs[DSV41_HC * DSV41_DMAX];
    __shared__ float red[DSV41_SUM_BATCH][DSV41_BLOCK];
    __shared__ float sums[DSV41_NSUM];
    __shared__ float mix[DSV41_NMIX];
    const unsigned t = blockIdx.x, tid = threadIdx.x;
    const unsigned R = DSV41_HC * D;
    __nv_bfloat16* ht = h + (size_t)t * R;
    if (ya != nullptr) {
        const float* po = post_in + t * DSV41_HC;
        const float* cb = comb_in + t * DSV41_HC * DSV41_HC;
        for (unsigned d = tid; d < D; d += DSV41_BLOCK) {
            const size_t yi = (size_t)t * D + d;
            const float yv = yb != nullptr ? bf(__float2bfloat16(bf(ya[yi]) + bf(yb[yi]))) : bf(ya[yi]);
            float r[DSV41_HC];
#pragma unroll
            for (unsigned i = 0; i < DSV41_HC; ++i) r[i] = bf(ht[i * D + d]);
#pragma unroll
            for (unsigned j = 0; j < DSV41_HC; ++j) {
                float acc = po[j] * yv;
#pragma unroll
                for (unsigned i = 0; i < DSV41_HC; ++i) acc += cb[i * DSV41_HC + j] * r[i];
                const __nv_bfloat16 o = __float2bfloat16(acc);
                ht[j * D + d] = o;
                hs[j * D + d] = o;
            }
        }
    } else {
        for (unsigned k = tid; k < R; k += DSV41_BLOCK) hs[k] = ht[k];
    }
    __syncthreads();
    // mixes: dsv41_hc_mixes' per-thread accumulation, reading the row from shared memory
    float acc[DSV41_NMIX];
#pragma unroll
    for (int m = 0; m < DSV41_NMIX; ++m) acc[m] = 0.f;
    float ss = 0.f;
    for (unsigned k = tid; k < R; k += DSV41_BLOCK) {
        const float v = bf(hs[k]);
        ss += v * v;
#pragma unroll
        for (int m = 0; m < DSV41_NMIX; ++m) acc[m] += v * fn[(size_t)m * R + k];
    }
    // 25 sums (24 dots + ss) through block_sum's tree, DSV41_SUM_BATCH at a time
    const unsigned warp = tid / 32, lane = tid % 32;
#pragma unroll
    for (int b0 = 0; b0 < DSV41_NSUM; b0 += DSV41_SUM_BATCH) {
#pragma unroll
        for (int j = 0; j < DSV41_SUM_BATCH; ++j) {
            const int m = b0 + j;
            red[j][tid] = m < DSV41_NMIX ? acc[m < DSV41_NMIX ? m : 0] : ss;
        }
        __syncthreads();
        for (unsigned s = DSV41_BLOCK / 2; s >= 32; s >>= 1) {
            if (tid < s) {
#pragma unroll
                for (int j = 0; j < DSV41_SUM_BATCH; ++j) red[j][tid] += red[j][tid + s];
            }
            __syncthreads();
        }
        if (warp < DSV41_SUM_BATCH) {
            float v = red[warp][lane];
#pragma unroll
            for (unsigned s = 16; s > 0; s >>= 1) v += __shfl_down_sync(0xffffffffu, v, s);
            if (lane == 0) sums[b0 + warp] = v;
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float rs = rsqrtf(sums[DSV41_NMIX] / (float)R + eps);
        for (int m = 0; m < DSV41_NMIX; ++m) mix[m] = sums[m] * rs;
    }
    __syncthreads();
    if (tid < 32) hc_mix_epilogue_warp(mix, t, scale, base, pre_o, post_o, comb_o, iters, hc_eps);
    __syncthreads();
    // pre (side_pre, may alias pre_o: read after the epilogue, as unfused) + rmsnorm
    const float* p = side_pre + t * DSV41_HC;
    float ssq = 0.f;
    for (unsigned d = tid; d < D; d += DSV41_BLOCK) {
        float a2 = 0.f;
#pragma unroll
        for (unsigned i = 0; i < DSV41_HC; ++i) a2 += p[i] * bf(hs[i * D + d]);
        const __nv_bfloat16 xv = __float2bfloat16(a2);
        x[(size_t)t * D + d] = xv;
        const float v = bf(xv);
        ssq += v * v;
    }
    const float rs2 = rsqrtf(block_sum(ssq, red[0]) / (float)D + eps);
    for (unsigned d = tid; d < D; d += DSV41_BLOCK)
        x[(size_t)t * D + d] = __float2bfloat16(bf(norm_w[d]) * (bf(x[(size_t)t * D + d]) * rs2));
}

// ── add + hc_post: dsv41_add_bf16 then dsv41_hc_post, one pass (the FFN end of a block) ──
// Grid: (T)  Block: 256.
extern "C" __global__ void dsv41_hc_add_post(const __nv_bfloat16* __restrict__ ya,
                                             const __nv_bfloat16* __restrict__ yb,
                                             __nv_bfloat16* h, const float* __restrict__ post,
                                             const float* __restrict__ comb, const unsigned D) {
    const unsigned t = blockIdx.x;
    const float* po = post + t * DSV41_HC;
    const float* cb = comb + t * DSV41_HC * DSV41_HC;
    for (unsigned d = threadIdx.x; d < D; d += DSV41_BLOCK) {
        const size_t yi = (size_t)t * D + d;
        const float yv = bf(__float2bfloat16(bf(ya[yi]) + bf(yb[yi])));
        float r[DSV41_HC];
#pragma unroll
        for (unsigned i = 0; i < DSV41_HC; ++i) r[i] = bf(h[((size_t)t * DSV41_HC + i) * D + d]);
#pragma unroll
        for (unsigned j = 0; j < DSV41_HC; ++j) {
            float acc = po[j] * yv;
#pragma unroll
            for (unsigned i = 0; i < DSV41_HC; ++i) acc += cb[i * DSV41_HC + j] * r[i];
            h[((size_t)t * DSV41_HC + j) * D + d] = __float2bfloat16(acc);
        }
    }
}

// ── hc_pre: v41_ref.hc_pre  y[t] = bf16(sum_i pre[t,i] * h[t,i]) ───────────────
// Grid: (T)  Block: 256.
extern "C" __global__ void dsv41_hc_pre(const __nv_bfloat16* __restrict__ h,
                                        const float* __restrict__ pre,
                                        __nv_bfloat16* __restrict__ y, const unsigned D) {
    const unsigned t = blockIdx.x;
    const float* p = pre + t * DSV41_HC;
    for (unsigned d = threadIdx.x; d < D; d += DSV41_BLOCK) {
        float acc = 0.f;
#pragma unroll
        for (unsigned i = 0; i < DSV41_HC; ++i) acc += p[i] * bf(h[((size_t)t * DSV41_HC + i) * D + d]);
        y[(size_t)t * D + d] = __float2bfloat16(acc);
    }
}

// ── hc_post: v41_ref.hc_post ──────────────────────────────────────────────────
// out[t,j] = bf16(post[t,j] * y[t] + sum_i comb[t,i,j] * res[t,i]). `out` MAY alias `res`:
// each thread reads all HC residual values of its column before writing any.
// Grid: (T)  Block: 256.
extern "C" __global__ void dsv41_hc_post(const __nv_bfloat16* __restrict__ y,
                                         const __nv_bfloat16* res, const float* __restrict__ post,
                                         const float* __restrict__ comb, __nv_bfloat16* out,
                                         const unsigned D) {
    const unsigned t = blockIdx.x;
    const float* po = post + t * DSV41_HC;
    const float* cb = comb + t * DSV41_HC * DSV41_HC;
    for (unsigned d = threadIdx.x; d < D; d += DSV41_BLOCK) {
        const float yv = bf(y[(size_t)t * D + d]);
        float r[DSV41_HC];
#pragma unroll
        for (unsigned i = 0; i < DSV41_HC; ++i) r[i] = bf(res[((size_t)t * DSV41_HC + i) * D + d]);
#pragma unroll
        for (unsigned j = 0; j < DSV41_HC; ++j) {
            float acc = po[j] * yv;
#pragma unroll
            for (unsigned i = 0; i < DSV41_HC; ++i) acc += cb[i * DSV41_HC + j] * r[i];
            out[((size_t)t * DSV41_HC + j) * D + d] = __float2bfloat16(acc);
        }
    }
}

// ── swiglu: v41_ref.expert_ffn (shared expert), limit > 0 ─────────────────────
// out = bf16(silu(min(gate, L)) * clamp(up, -L, L)); gate/up are the bf16 GEMM outputs.
extern "C" __global__ void dsv41_swiglu(const __nv_bfloat16* __restrict__ gate,
                                        const __nv_bfloat16* __restrict__ up,
                                        __nv_bfloat16* __restrict__ out,
                                        const unsigned long long n, const float limit) {
    const unsigned long long i = (unsigned long long)blockIdx.x * DSV41_BLOCK + threadIdx.x;
    if (i >= n) return;
    float g = bf(gate[i]);
    float u = bf(up[i]);
    if (limit > 0.f) {
        u = fminf(fmaxf(u, -limit), limit);
        g = fminf(g, limit);
    }
    out[i] = __float2bfloat16(g / (1.f + expf(-g)) * u);
}

// swiglu over the concatenated w1|w3 GEMM output: gu bf16 [rows, 2 * inter], gate in columns
// [0, inter), up in [inter, 2 * inter). Same arithmetic as dsv41_swiglu, element for element.
extern "C" __global__ void dsv41_swiglu_cat(const __nv_bfloat16* __restrict__ gu,
                                            __nv_bfloat16* __restrict__ out,
                                            const unsigned long long rows, const unsigned long long inter,
                                            const float limit) {
    const unsigned long long i = (unsigned long long)blockIdx.x * DSV41_BLOCK + threadIdx.x;
    if (i >= rows * inter) return;
    const unsigned long long r = i / inter, c = i % inter;
    float g = bf(gu[r * 2 * inter + c]);
    float u = bf(gu[r * 2 * inter + inter + c]);
    if (limit > 0.f) {
        u = fminf(fmaxf(u, -limit), limit);
        g = fminf(g, limit);
    }
    out[i] = __float2bfloat16(g / (1.f + expf(-g)) * u);
}

// ── RoPE on the tail: v41_ref.apply_rotary (adjacent pairs as complex) ────────
// x bf16 [T, H, Dh], rotate the LAST 2*P dims in place. Row t uses table row pos[t]
// (cos/sin fp32 [*, P]); inverse = conj. Grid: (T, H)  Block: P.
extern "C" __global__ void dsv41_rope_tail(__nv_bfloat16* __restrict__ x,
                                           const int* __restrict__ pos,
                                           const float* __restrict__ cos_t,
                                           const float* __restrict__ sin_t, const unsigned H,
                                           const unsigned Dh, const unsigned P,
                                           const int inverse) {
    const unsigned t = blockIdx.x, hh = blockIdx.y, p = threadIdx.x;
    if (p >= P) return;
    __nv_bfloat16* v = x + ((size_t)t * H + hh) * Dh + (Dh - 2 * P) + 2 * p;
    const size_t row = (size_t)pos[t] * P + p;
    const float c = cos_t[row];
    const float s = inverse ? -sin_t[row] : sin_t[row];
    const float a = bf(v[0]), b = bf(v[1]);
    v[0] = __float2bfloat16(a * c - b * s);
    v[1] = __float2bfloat16(a * s + b * c);
}

// ── engram: rows -> bf16 with the dead-head mask ──────────────────────────────
// model.forward: rows.masked_fill(dead_heads.unsqueeze(-1), 0), then v41_ref.engram_forward's
// rows.reshape(T, -1).to(bf16). rows f32 [T, 24, 256]; dead u8 [T, 24] or NULL.
extern "C" __global__ void dsv41_engram_rows_bf16(const float* __restrict__ rows,
                                                  const uint8_t* __restrict__ dead,
                                                  __nv_bfloat16* __restrict__ out,
                                                  const unsigned long long n) {
    const unsigned long long i = (unsigned long long)blockIdx.x * DSV41_BLOCK + threadIdx.x;
    if (i >= n) return;
    const bool masked = dead != nullptr && dead[i / 256] != 0;
    out[i] = __float2bfloat16(masked ? 0.f : rows[i]);
}

// ── engram: v41_ref.engram_forward gate + residual ────────────────────────────
// kv bf16 [T, HC*D + D] = [key (HC x D) | value (D)]; weight fp32 [HC, D] = q_weight * k_weight.
// rstd = rsqrt(mean(h^2)+eps) * rsqrt(mean(key^2)+eps);
// dot = sum((h * weight) * key) * rstd * D^-0.5;  gate = sigmoid(sign(dot) sqrt(max(|dot|,1e-6)));
// h_out[t,i] = bf16(h[t,i] + gate * value). Grid: (T, HC)  Block: 256.
extern "C" __global__ void dsv41_engram_gate(__nv_bfloat16* __restrict__ h,
                                             const __nv_bfloat16* __restrict__ kv,
                                             const float* __restrict__ weight, const unsigned D,
                                             const float eps) {
    __shared__ float red[DSV41_BLOCK];
    const unsigned t = blockIdx.x, i = blockIdx.y;
    __nv_bfloat16* hr = h + ((size_t)t * DSV41_HC + i) * D;
    const size_t W = (size_t)DSV41_HC * D + D;
    const __nv_bfloat16* key = kv + (size_t)t * W + (size_t)i * D;
    const __nv_bfloat16* val = kv + (size_t)t * W + (size_t)DSV41_HC * D;
    const float* wr = weight + (size_t)i * D;
    float sh = 0.f, sk = 0.f, dt = 0.f;
    for (unsigned d = threadIdx.x; d < D; d += DSV41_BLOCK) {
        const float hv = bf(hr[d]), kk = bf(key[d]);
        sh += hv * hv;
        sk += kk * kk;
        dt += (hv * wr[d]) * kk;
    }
    const float msh = block_sum(sh, red) / (float)D;
    const float msk = block_sum(sk, red) / (float)D;
    const float dot0 = block_sum(dt, red);
    const float rstd = rsqrtf(msh + eps) * rsqrtf(msk + eps);
    const float dot = dot0 * rstd * rsqrtf((float)D);
    const float sq = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    const float g = 1.f / (1.f + expf(-copysignf(sq, dot)));
    for (unsigned d = threadIdx.x; d < D; d += DSV41_BLOCK)
        hr[d] = __float2bfloat16(bf(hr[d]) + g * bf(val[d]));
}

// ── engram weight product: q_weight.float() * k_weight.float() ────────────────
extern "C" __global__ void dsv41_mul_bf16_to_f32(const __nv_bfloat16* __restrict__ a,
                                                 const __nv_bfloat16* __restrict__ b,
                                                 float* __restrict__ out,
                                                 const unsigned long long n) {
    const unsigned long long i = (unsigned long long)blockIdx.x * DSV41_BLOCK + threadIdx.x;
    if (i < n) out[i] = bf(a[i]) * bf(b[i]);
}

// ── DSpark seed: engine/model.py `h.float().mean(dim=1)` at L37-39, then `.to(bf16)` ──
// out[t * ld + off + d] = bf16(((h0 + h1) + h2 + h3) / hc). Grid: (T)  Block: 256.
extern "C" __global__ void dsv41_hc_mean_bf16(const __nv_bfloat16* __restrict__ h,
                                              __nv_bfloat16* __restrict__ out, const unsigned D,
                                              const unsigned ld, const unsigned off) {
    const unsigned t = blockIdx.x;
    for (unsigned d = threadIdx.x; d < D; d += DSV41_BLOCK) {
        float acc = 0.f;
#pragma unroll
        for (unsigned i = 0; i < DSV41_HC; ++i) acc += bf(h[((size_t)t * DSV41_HC + i) * D + d]);
        out[(size_t)t * ld + off + d] = __float2bfloat16(acc / (float)DSV41_HC);
    }
}
