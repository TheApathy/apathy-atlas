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

// ── hc_mixes, DSV41_HC_TB tokens per block (prefill) ─────────────────────────
// dsv41_hc_mixes reads the whole fp32 fn [24, HC*D] (1.97 MB) once PER TOKEN, which makes it
// L2-bound at prefill T. Here one block serves DSV41_HC_TB tokens and each fn element is read
// once for all of them. Per token the arithmetic is dsv41_hc_mixes' exactly: thread k-stride
// order, the same fma chain, the same block_sum tree, the same epilogue -> bit-identical.
// Grid: (ceil(T / TB))  Block: 256.
#define DSV41_HC_TB 4
extern "C" __global__ void dsv41_hc_mixes_tb(const __nv_bfloat16* __restrict__ h,
                                             const float* __restrict__ fn,     // [24, HC*D]
                                             const float* __restrict__ scale,  // [3]
                                             const float* __restrict__ base,   // [24]
                                             float* __restrict__ pre, float* __restrict__ post,
                                             float* __restrict__ comb, const unsigned D,
                                             const unsigned T, const unsigned iters,
                                             const float eps, const float hc_eps) {
    __shared__ float red[DSV41_BLOCK];
    __shared__ float mix[DSV41_HC_TB][DSV41_NMIX];
    const unsigned t0 = blockIdx.x * DSV41_HC_TB;
    const unsigned nt = min((unsigned)DSV41_HC_TB, T - t0);
    const unsigned R = DSV41_HC * D;
    float acc[DSV41_HC_TB][DSV41_NMIX];
    float ss[DSV41_HC_TB];
#pragma unroll
    for (int b = 0; b < DSV41_HC_TB; ++b) {
        ss[b] = 0.f;
#pragma unroll
        for (int m = 0; m < DSV41_NMIX; ++m) acc[b][m] = 0.f;
    }
    for (unsigned k = threadIdx.x; k < R; k += DSV41_BLOCK) {
        float v[DSV41_HC_TB];
#pragma unroll
        for (int b = 0; b < DSV41_HC_TB; ++b) {
            v[b] = (unsigned)b < nt ? bf(h[(size_t)(t0 + b) * R + k]) : 0.f;
            ss[b] += v[b] * v[b];
        }
#pragma unroll
        for (int m = 0; m < DSV41_NMIX; ++m) {
            const float f = fn[(size_t)m * R + k];
#pragma unroll
            for (int b = 0; b < DSV41_HC_TB; ++b) acc[b][m] += v[b] * f;
        }
    }
#pragma unroll
    for (int b = 0; b < DSV41_HC_TB; ++b) {
        const float rs = rsqrtf(block_sum(ss[b], red) / (float)R + eps);
        for (int m = 0; m < DSV41_NMIX; ++m) {
            const float r = block_sum(acc[b][m], red);
            if (threadIdx.x == 0) mix[b][m] = r * rs;
        }
    }
    __syncthreads();
    if (threadIdx.x >= nt) return;
    // Thread b runs token t0 + b's epilogue: dsv41_hc_mixes' thread-0 code, verbatim.
    const unsigned t = t0 + threadIdx.x;
    const float* mx_ = mix[threadIdx.x];
    const unsigned HC = DSV41_HC;
    for (unsigned i = 0; i < HC; ++i) {
        const float vp = mx_[i] * scale[0] + base[i];
        pre[t * HC + i] = 1.f / (1.f + expf(-vp)) + hc_eps;
        const float vq = mx_[HC + i] * scale[1] + base[HC + i];
        post[t * HC + i] = 2.f * (1.f / (1.f + expf(-vq)));
    }
    float c[DSV41_HC * DSV41_HC];
    for (unsigned i = 0; i < HC; ++i) {
        float mx = -INFINITY;
        for (unsigned j = 0; j < HC; ++j) {
            c[i * HC + j] = mx_[2 * HC + i * HC + j] * scale[2] + base[2 * HC + i * HC + j];
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
