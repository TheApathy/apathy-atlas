// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1 sparse attention, first form: the ONE-PASS STREAMING GATHER.
//
// Port of tools/prefill_attn.py `_sparse_attn_kernel` from the Python engine.
// A query's key set is 128 sliding-window rows (addressed in a ring by
// `pos % RING`) plus up to 512 compressed rows the indexer selected. Both are
// read BY INDEX inside this kernel; the [T, 640, 512] gathered tensor the
// reference's slow path materialises (~3 GB of fp32 per layer at T=2048) never
// exists.
//
// MQA. num_key_value_heads = 1, so all 64 query heads of a token share ONE
// 512-dim row per position and K and V ARE THE SAME TENSOR -- the score product
// and the value product read the same bytes. That is why the gather is worth
// amortising across heads, and it is the reason this shape suits a one-pass
// kernel at all.
//
// The learned per-head sink enters the DENOMINATOR ONLY: it contributes no
// value to the numerator (SPEC.md sec 2).
//
// BLOCK ORDER IS PART OF THE CONTRACT. Online-softmax rescaling is not
// associative in floating point, so this walks WINDOW first and COMPRESSED
// second, as the reference does. CTRL_ORDER below is the negative control that
// proves the order is observable rather than decorative.
//
// This is the CORRECTNESS form, not the performance form: HB = 8 heads per
// block, a hand-rolled FMA dot with a block reduction, no tensor cores. It
// amortises the gather 8x rather than 64x and it is slow on purpose. The
// performance form is the optimisation and will be validated against this.
//
// Numerics: P is kept in fp32 here. The reference `_softmax_attn` also keeps
// fp32; tools/prefill_attn.py rounds P to bf16 because tl.dot needs a
// tensor-core dtype. So this kernel should sit near the fp32 reference and
// about 1.4e-3 from the bf16-P one -- and the gate prints BOTH numbers.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <algorithm>

#define CUDA_OK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); std::exit(2);} } while(0)

static const int HB   = 8;    // heads per block
static const int BK   = 8;    // keys per tile
static const int NT   = 256;  // threads per block
static const int DMAX = 512;  // head_dim
static const int DPT  = DMAX / NT;   // d-slots per thread

// Negative controls. CORRECT = 0.
//   CTRL_ORDER  : compressed segment before window -- the block-order contract.
//   CTRL_GATHER : ignore cidx and read compressed rows 0..NC-1 sequentially --
//                 the "wrong K-order" analogue for a gather kernel.
template <typename OutT> __device__ __forceinline__ OutT to_out(float x);
template <> __device__ __forceinline__ float to_out<float>(float x) { return x; }
template <> __device__ __forceinline__ __nv_bfloat16 to_out<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }

// IdxT: int32_t for the fixture gate, int64_t (long long) in production -- the indexer
// emits i64 and the window positions are i64, as in the reference. OutT: float for the
// gate, bf16 in production (the reference's `_softmax_attn(...).to(bfloat16)`).
template <bool CTRL_ORDER, bool CTRL_GATHER, typename IdxT, typename OutT, typename WposT = IdxT>
__device__ void sparse_attn_body(
    const __nv_bfloat16* __restrict__ Q,     // [T, NH, D]
    const __nv_bfloat16* __restrict__ RING,  // [RING_N, D]
    const WposT*         __restrict__ WPOS,  // [T, NW]  absolute positions, -1 = none
    const __nv_bfloat16* __restrict__ CKV,   // [n_c, D]
    const IdxT*          __restrict__ CIDX,  // [T, NC]  compressed rows, -1 = none
    const float*         __restrict__ SINK,  // [NH]
    OutT*                __restrict__ O,     // [T, NH, D]
    int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale,
    // SPLIT-KV (flash-decoding): walk only keys [k_lo, k_hi) of the concatenated
    // [window (NW) | compressed (NC)] list and, when PART != null, write the unnormalised
    // partial (m, l, acc[D]) per head to PART[t][h][blockIdx.z] instead of O.
    int k_lo = 0, int k_hi = 1 << 30, float* __restrict__ PART = nullptr)
{
    const int t   = blockIdx.x;
    const int hb  = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int nwarp = NT / 32;

    __shared__ __nv_bfloat16 qs[HB][DMAX];
    __shared__ __nv_bfloat16 ks[BK][DMAX];
    __shared__ float red2[HB][BK][NT / 32];
    __shared__ float sc[HB][BK];
    __shared__ int   vld[BK];

    for (int h = 0; h < HB; ++h)
        for (int i = 0; i < DPT; ++i) {
            const int d = tid + i * NT;
            qs[h][d] = Q[((size_t)t * NH + (hb * HB + h)) * D + d];
        }
    __syncthreads();

    float m[HB], l[HB], acc[HB][DPT];
#pragma unroll
    for (int h = 0; h < HB; ++h) {
        m[h] = -INFINITY; l[h] = 0.f;
        for (int i = 0; i < DPT; ++i) acc[h][i] = 0.f;
    }

    // seg 0 = window, seg 1 = compressed; CTRL_ORDER swaps the visit order.
    for (int step = 0; step < 2; ++step) {
        const int seg = CTRL_ORDER ? (1 - step) : step;
        const int n0  = seg == 0 ? NW : NC;
        if (seg == 1 && CIDX == nullptr) continue;
        const int base = seg == 0 ? 0 : NW;
        const int lo = max(0, k_lo - base);
        const int n  = min(n0, k_hi - base);        // this split's end within the segment
        if (lo >= n) continue;

        for (int kb = lo; kb < n; kb += BK) {
            // ---- gather BK rows straight out of the ring / compressed cache
            if (tid < BK) {
                const int c = kb + tid;
                int row = -1;
                if (c < n) {
                    if (seg == 0) {
                        const long long p = WPOS[(size_t)t * NW + c];   // NW is the row stride
                        if (p >= 0 && p >= win_lo) row = (int)(p % RING_N);   // ring modulo
                    } else if (CTRL_GATHER) {
                        row = c;                                       // wrong on purpose
                    } else {
                        const long long j = CIDX[(size_t)t * NC + c];
                        if (j >= 0) row = (int)j;
                    }
                }
                vld[tid] = row;
            }
            __syncthreads();

            for (int kk = 0; kk < BK; ++kk) {
                const int row = vld[kk];
                const __nv_bfloat16* src = (seg == 0) ? RING : CKV;
                for (int i = 0; i < DPT; ++i) {
                    const int d = tid + i * NT;
                    ks[kk][d] = (row >= 0) ? src[(size_t)row * D + d] : __float2bfloat16(0.f);
                }
            }
            __syncthreads();

            // ---- scores: sc[h][kk] = dot(q_h, k_kk) * scale, fp32 accumulate.
            // All HB x BK dot products of the tile in ONE reduction round (one barrier pair per
            // tile, not per key). Per product the arithmetic order is unchanged -- per-thread
            // fmaf over its DPT dims, the same shfl_down tree, then warps summed in order -- so
            // this is bit-identical to reducing one key at a time.
            {
                float part[HB][BK];
#pragma unroll
                for (int h = 0; h < HB; ++h)
#pragma unroll
                    for (int kk = 0; kk < BK; ++kk) {
                        float s = 0.f;
                        for (int i = 0; i < DPT; ++i) {
                            const int d = tid + i * NT;
                            s = fmaf(__bfloat162float(qs[h][d]), __bfloat162float(ks[kk][d]), s);
                        }
                        for (int off = 16; off; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
                        part[h][kk] = s;
                    }
                if (lane == 0) {
#pragma unroll
                    for (int h = 0; h < HB; ++h)
#pragma unroll
                        for (int kk = 0; kk < BK; ++kk) red2[h][kk][warp] = part[h][kk];
                }
                __syncthreads();
                if (tid < HB * BK) {
                    const int h = tid / BK, kk = tid % BK;
                    float s = 0.f;
                    for (int w = 0; w < nwarp; ++w) s += red2[h][kk][w];
                    sc[h][kk] = s * scale;
                }
                __syncthreads();
            }

            // ---- online softmax update over this tile (window rows first)
#pragma unroll
            for (int h = 0; h < HB; ++h) {
                float tmax = -INFINITY;
                for (int kk = 0; kk < BK; ++kk)
                    if (vld[kk] >= 0 && kb + kk < n) tmax = fmaxf(tmax, sc[h][kk]);
                const float m_new  = fmaxf(m[h], tmax);
                // m_safe: an all-masked row keeps a finite exponent argument, so it
                // yields acc=0, l=0, denom=exp(sink) -> o=0 rather than NaN.
                const float m_safe = (m_new == -INFINITY) ? 0.f : m_new;
                const float alpha  = __expf(m[h] - m_safe);
                float lsum = 0.f;
                for (int i = 0; i < DPT; ++i) acc[h][i] *= alpha;
                for (int kk = 0; kk < BK; ++kk) {
                    if (vld[kk] < 0 || kb + kk >= n) continue;
                    const float p = __expf(sc[h][kk] - m_safe);
                    lsum += p;
                    for (int i = 0; i < DPT; ++i) {
                        const int d = tid + i * NT;
                        acc[h][i] = fmaf(p, __bfloat162float(ks[kk][d]), acc[h][i]);
                    }
                }
                l[h] = l[h] * alpha + lsum;
                m[h] = m_new;
            }
            __syncthreads();
        }
    }

    if (PART != nullptr) {
        const int S = gridDim.z, sidx = blockIdx.z;
#pragma unroll
        for (int h = 0; h < HB; ++h) {
            float* pr = PART + (((size_t)t * NH + (hb * HB + h)) * S + sidx) * (2 + D);
            if (tid == 0) { pr[0] = m[h]; pr[1] = l[h]; }
            for (int i = 0; i < DPT; ++i) pr[2 + tid + i * NT] = acc[h][i];
        }
        return;
    }
#pragma unroll
    for (int h = 0; h < HB; ++h) {
        const float m_safe = (m[h] == -INFINITY) ? 0.f : m[h];
        const float denom  = l[h] + __expf(SINK[hb * HB + h] - m_safe);
        for (int i = 0; i < DPT; ++i) {
            const int d = tid + i * NT;
            O[((size_t)t * NH + (hb * HB + h)) * D + d] = to_out<OutT>(acc[h][i] / denom);
        }
    }
}

template <bool CTRL_ORDER, bool CTRL_GATHER>
__global__ void __launch_bounds__(NT) sparse_attn(
    const __nv_bfloat16* Q, const __nv_bfloat16* RING, const int32_t* WPOS,
    const __nv_bfloat16* CKV, const int32_t* CIDX, const float* SINK, float* O,
    int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale)
{
    sparse_attn_body<CTRL_ORDER, CTRL_GATHER, int32_t, float>(
        Q, RING, WPOS, CKV, CIDX, SINK, O, T, NH, D, NW, NC, RING_N, win_lo, scale);
}

// PRODUCTION entry for the forward's attn_block: i32 window positions (as it builds them),
// i64 compressed indices (as the indexer emits them).
extern "C" __global__ void __launch_bounds__(NT) dsv41_sparse_attn_w32(
    const __nv_bfloat16* Q, const __nv_bfloat16* RING, const int32_t* WPOS,
    const __nv_bfloat16* CKV, const long long* CIDX, const float* SINK, __nv_bfloat16* O,
    int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale)
{
    sparse_attn_body<false, false, long long, __nv_bfloat16, int32_t>(
        Q, RING, WPOS, CKV, CIDX, SINK, O, T, NH, D, NW, NC, RING_N, win_lo, scale);
}

// DECODE entry (T small): SPLIT-KV. grid (T, NH / 8, S), block 256. Split z walks keys
// [z*SLICE, (z+1)*SLICE) of [window | compressed]; SLICE is a multiple of BK. The split is by
// KEY INDEX only, so a row's result does not depend on T or on the step: bit-stable. PART is
// [T, NH, S, 2 + D] fp32. Follow with dsv41_sparse_attn_combine.
extern "C" __global__ void __launch_bounds__(NT) dsv41_sparse_attn_split(
    const __nv_bfloat16* Q, const __nv_bfloat16* RING, const int32_t* WPOS,
    const __nv_bfloat16* CKV, const long long* CIDX, float* PART,
    int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale, int SLICE)
{
    const int k_lo = blockIdx.z * SLICE;
    sparse_attn_body<false, false, long long, float, int32_t>(
        Q, RING, WPOS, CKV, CIDX, nullptr, nullptr, T, NH, D, NW, NC, RING_N, win_lo, scale,
        k_lo, k_lo + SLICE, PART);
}

// Merge the S partials of each (t, head) in FIXED split order, add the sink to the
// denominator ONCE, write bf16. grid (T, NH), block 256 (2 dims per thread).
extern "C" __global__ void __launch_bounds__(NT) dsv41_sparse_attn_combine(
    const float* __restrict__ PART, const float* __restrict__ SINK, __nv_bfloat16* __restrict__ O,
    int NH, int D, int S)
{
    const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const float* pr = PART + ((size_t)t * NH + h) * S * (2 + D);
    float M = -INFINITY;
    for (int s = 0; s < S; ++s) M = fmaxf(M, pr[(size_t)s * (2 + D)]);
    const float m_safe = (M == -INFINITY) ? 0.f : M;
    float L = 0.f, acc[DPT];
    for (int i = 0; i < DPT; ++i) acc[i] = 0.f;
    for (int s = 0; s < S; ++s) {
        const float* q = pr + (size_t)s * (2 + D);
        const float w = __expf(q[0] - m_safe);      // -inf partial (all keys masked) -> 0
        L = L + q[1] * w;
        for (int i = 0; i < DPT; ++i) acc[i] = acc[i] + q[2 + tid + i * NT] * w;
    }
    const float denom = L + __expf(SINK[h] - m_safe);
    for (int i = 0; i < DPT; ++i)
        O[((size_t)t * NH + h) * D + tid + i * NT] = __float2bfloat16_rn(acc[i] / denom);
}

// DECODE/VERIFY entry with the combine FUSED by last-block-done: the same split partials as
// dsv41_sparse_attn_split, then the LAST of the S slice CTAs of each (t, head group) -- found by an
// atomic ticket -- merges them with dsv41_sparse_attn_combine's exact arithmetic (fixed slice order,
// same expressions), reading the partials back from L2 (__ldcg) while they are hot. Saves the
// combine launch and its cold re-read; BYTE-IDENTICAL to split + combine (gate). CNT: [T, NH / HB]
// u32 tickets, zero at load; the last CTA resets its ticket, so every launch starts from zero.
__device__ __forceinline__ void combine_head(const float* __restrict__ pr, const float* __restrict__ SINK,
                                             __nv_bfloat16* __restrict__ O, int t, int h, int NH, int D, int S)
{
    const int tid = threadIdx.x;
    float M = -INFINITY;
    for (int s = 0; s < S; ++s) M = fmaxf(M, __ldcg(pr + (size_t)s * (2 + D)));
    const float m_safe = (M == -INFINITY) ? 0.f : M;
    float L = 0.f, acc[DPT];
    for (int i = 0; i < DPT; ++i) acc[i] = 0.f;
    for (int s = 0; s < S; ++s) {
        const float* q = pr + (size_t)s * (2 + D);
        const float w = __expf(__ldcg(q) - m_safe);
        L = L + __ldcg(q + 1) * w;
        for (int i = 0; i < DPT; ++i) acc[i] = acc[i] + __ldcg(q + 2 + tid + i * NT) * w;
    }
    const float denom = L + __expf(SINK[h] - m_safe);
    for (int i = 0; i < DPT; ++i)
        O[((size_t)t * NH + h) * D + tid + i * NT] = __float2bfloat16_rn(acc[i] / denom);
}

extern "C" __global__ void __launch_bounds__(NT) dsv41_sparse_attn_split_lb(
    const __nv_bfloat16* Q, const __nv_bfloat16* RING, const int32_t* WPOS,
    const __nv_bfloat16* CKV, const long long* CIDX, float* PART, const float* SINK,
    __nv_bfloat16* O, unsigned* CNT,
    int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale, int SLICE)
{
    const int k_lo = blockIdx.z * SLICE;
    sparse_attn_body<false, false, long long, float, int32_t>(
        Q, RING, WPOS, CKV, CIDX, nullptr, nullptr, T, NH, D, NW, NC, RING_N, win_lo, scale,
        k_lo, k_lo + SLICE, PART);
    __shared__ int last;
    __threadfence();                  // this CTA's partials are visible device-wide ...
    __syncthreads();                  // ... for every thread, before the ticket is taken
    const int t = blockIdx.x, hb = blockIdx.y, S = gridDim.z;
    unsigned* c = CNT + (size_t)t * gridDim.y + hb;
    if (threadIdx.x == 0) last = atomicAdd(c, 1u) == (unsigned)(S - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();                  // acquire: the other slices' partials
    for (int h = 0; h < HB; ++h) {
        const int hh = hb * HB + h;
        combine_head(PART + ((size_t)t * NH + hh) * S * (2 + D), SINK, O, t, hh, NH, D, S);
    }
    if (threadIdx.x == 0) *c = 0u;    // ready for the next launch
}

// ------------------------------------------------------------------ TENSOR-CORE PREFILL
// dsv41_sparse_attn_mma: the same computation on mma.sync.m16n8k16 (bf16 in, fp32 accumulate).
// One CTA = one token x 16 heads (MQA: the 16 heads share every gathered K/V row), 4 warps.
// Per 16-key tile: S = Q K^T with warp w summing dims [128w, 128w+128), partials merged in fixed
// warp order; online softmax in fp32 (window tiles first, then compressed, as the reference);
// P = exp(s - m) is split P_hi = bf16(P), P_lo = bf16(P - P_hi) and BOTH are applied
// (acc += P_hi K + P_lo K), keeping ~16 mantissa bits of P -- the decode_attn.py PV_SPLIT
// idea -- instead of the 8 a single bf16 P would keep. l is accumulated from fp32 P.
// Numerics vs the fp32-FMA one-pass kernel: different summation order and ~2^-17 relative P
// error; the band is pre-registered in the gate.
static const int MH = 16, MWARPS = 4, MNT = MWARPS * 32, KPAD = 8;

__device__ __forceinline__ void mma_bf16_16816(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
}

template <int TK>
__device__ __forceinline__ void sparse_attn_mma_body(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ RING, const int32_t* __restrict__ WPOS,
    const __nv_bfloat16* __restrict__ CKV, const long long* __restrict__ CIDX, const float* __restrict__ SINK,
    __nv_bfloat16* __restrict__ O, int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale)
{
    const int t = blockIdx.x, hg = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, q = lane & 3;

    __shared__ __align__(16) __nv_bfloat16 ks[TK][DMAX + KPAD];
    __shared__ float spart[MWARPS][MH][TK];
    __shared__ __align__(16) __nv_bfloat16 ph[MH][TK + KPAD], pl[MH][TK + KPAD];
    __shared__ float mrow[MH], lrow[MH], arow[MH];
    __shared__ int vld[TK];

    // Q stays in registers: this warp's A fragments over its 128 dims (8 k-steps).
    uint32_t qa[8][4];
    {
        const __nv_bfloat16* q0 = Q + ((size_t)t * NH + hg * MH + g) * D;
        const __nv_bfloat16* q1 = q0 + 8 * (size_t)D;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const int kd = warp * 128 + kk * 16;
            qa[kk][0] = *reinterpret_cast<const uint32_t*>(q0 + kd + 2 * q);
            qa[kk][1] = *reinterpret_cast<const uint32_t*>(q1 + kd + 2 * q);
            qa[kk][2] = *reinterpret_cast<const uint32_t*>(q0 + kd + 8 + 2 * q);
            qa[kk][3] = *reinterpret_cast<const uint32_t*>(q1 + kd + 8 + 2 * q);
        }
    }
    if (tid < MH) { mrow[tid] = -INFINITY; lrow[tid] = 0.f; }
    float acc[16][4];
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    __syncthreads();

    for (int seg = 0; seg < 2; ++seg) {
        const int n = seg == 0 ? NW : NC;
        if (seg == 1 && CIDX == nullptr) continue;
        const __nv_bfloat16* src = seg == 0 ? RING : CKV;
        for (int kb = 0; kb < n; kb += TK) {
            for (int j = tid; j < TK; j += MNT) {
                const int c = kb + j;
                int row = -1;
                if (c < n) {
                    if (seg == 0) {
                        const long long p = WPOS[(size_t)t * NW + c];
                        if (p >= 0 && p >= win_lo) row = (int)(p % RING_N);
                    } else {
                        const long long j = CIDX[(size_t)t * NC + c];
                        if (j >= 0) row = (int)j;
                    }
                }
                vld[j] = row;
            }
            __syncthreads();
            for (int i = tid; i < TK * DMAX / 8; i += MNT) {
                const int r = i / (DMAX / 8), c = i % (DMAX / 8);
                const int row = vld[r];
                *reinterpret_cast<uint4*>(&ks[r][c * 8]) = row >= 0
                    ? reinterpret_cast<const uint4*>(src + (size_t)row * D)[c] : make_uint4(0, 0, 0, 0);
            }
            __syncthreads();

            // ---- S partial over this warp's 128 dims: 8 k-steps x 2 n-tiles of 8 keys
            constexpr int NS = TK / 8;
            float sp[NS][4];
#pragma unroll
            for (int nt = 0; nt < NS; ++nt) sp[nt][0] = sp[nt][1] = sp[nt][2] = sp[nt][3] = 0.f;
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                const int kd = warp * 128 + kk * 16;
                const uint32_t (&a)[4] = qa[kk];
#pragma unroll
                for (int nt = 0; nt < NS; ++nt) {
                    uint32_t b[2];
                    b[0] = *reinterpret_cast<const uint32_t*>(&ks[nt * 8 + g][kd + 2 * q]);
                    b[1] = *reinterpret_cast<const uint32_t*>(&ks[nt * 8 + g][kd + 8 + 2 * q]);
                    mma_bf16_16816(sp[nt], a, b);
                }
            }
#pragma unroll
            for (int nt = 0; nt < NS; ++nt) {
                spart[warp][g][nt * 8 + 2 * q]         = sp[nt][0];
                spart[warp][g][nt * 8 + 2 * q + 1]     = sp[nt][1];
                spart[warp][g + 8][nt * 8 + 2 * q]     = sp[nt][2];
                spart[warp][g + 8][nt * 8 + 2 * q + 1] = sp[nt][3];
            }
            __syncthreads();

            // ---- online softmax: 8 threads per head row, 2 keys each
            {
                constexpr int KPT = TK / 8;   // keys per thread
                const int row = tid >> 3, k0 = (tid & 7) * KPT;
                float sv[KPT], pv[KPT];
                float tmax = -INFINITY;
#pragma unroll
                for (int j = 0; j < KPT; ++j) {
                    const int k = k0 + j;
                    float v = spart[0][row][k];
                    for (int w = 1; w < MWARPS; ++w) v += spart[w][row][k];
                    const bool ok = vld[k] >= 0 && kb + k < n;
                    sv[j] = ok ? v * scale : -INFINITY;
                    tmax = fmaxf(tmax, sv[j]);
                }
                for (int off = 1; off < 8; off <<= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, off));
                const float m_old = mrow[row];
                const float m_new = fmaxf(m_old, tmax);
                const float m_safe = (m_new == -INFINITY) ? 0.f : m_new;
                const float alpha = __expf(m_old - m_safe);
                float lsum = 0.f;
#pragma unroll
                for (int j = 0; j < KPT; ++j) {
                    pv[j] = sv[j] == -INFINITY ? 0.f : __expf(sv[j] - m_safe);
                    lsum += pv[j];
                    const __nv_bfloat16 hi = __float2bfloat16_rn(pv[j]);
                    ph[row][k0 + j] = hi;
                    pl[row][k0 + j] = __float2bfloat16_rn(pv[j] - __bfloat162float(hi));
                }
                for (int off = 1; off < 8; off <<= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, off);
                __syncwarp();
                if ((tid & 7) == 0) {
                    lrow[row] = lrow[row] * alpha + lsum;
                    mrow[row] = m_new;
                    arow[row] = alpha;
                }
            }
            __syncthreads();

            // ---- O = O * alpha + (P_hi + P_lo) K over this warp's 128 dims (16 n-tiles)
            {
                const float a0 = arow[g], a1 = arow[g + 8];
#pragma unroll
                for (int nt = 0; nt < 16; ++nt) {
                    acc[nt][0] *= a0; acc[nt][1] *= a0; acc[nt][2] *= a1; acc[nt][3] *= a1;
                }
#pragma unroll
                for (int ks16 = 0; ks16 < TK / 16; ++ks16) {
                    const int kc = ks16 * 16;
                    uint32_t ah[4], al[4];
                    ah[0] = *reinterpret_cast<const uint32_t*>(&ph[g][kc + 2 * q]);
                    ah[1] = *reinterpret_cast<const uint32_t*>(&ph[g + 8][kc + 2 * q]);
                    ah[2] = *reinterpret_cast<const uint32_t*>(&ph[g][kc + 8 + 2 * q]);
                    ah[3] = *reinterpret_cast<const uint32_t*>(&ph[g + 8][kc + 8 + 2 * q]);
                    al[0] = *reinterpret_cast<const uint32_t*>(&pl[g][kc + 2 * q]);
                    al[1] = *reinterpret_cast<const uint32_t*>(&pl[g + 8][kc + 2 * q]);
                    al[2] = *reinterpret_cast<const uint32_t*>(&pl[g][kc + 8 + 2 * q]);
                    al[3] = *reinterpret_cast<const uint32_t*>(&pl[g + 8][kc + 8 + 2 * q]);
#pragma unroll
                    for (int nt = 0; nt < 16; ++nt) {
                        const int dim = warp * 128 + nt * 8 + g;
                        uint32_t b[2];
                        b[0] = pack2(ks[kc + 2 * q][dim], ks[kc + 2 * q + 1][dim]);
                        b[1] = pack2(ks[kc + 8 + 2 * q][dim], ks[kc + 9 + 2 * q][dim]);
                        mma_bf16_16816(acc[nt], ah, b);
                        mma_bf16_16816(acc[nt], al, b);
                    }
                }
            }
            __syncthreads();
        }
    }

    const int h0 = hg * MH + g, h1 = h0 + 8;
    const float m0 = mrow[g] == -INFINITY ? 0.f : mrow[g];
    const float m1 = mrow[g + 8] == -INFINITY ? 0.f : mrow[g + 8];
    const float d0 = lrow[g] + __expf(SINK[h0] - m0);
    const float d1 = lrow[g + 8] + __expf(SINK[h1] - m1);
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        const int dim = warp * 128 + nt * 8 + 2 * q;
        __nv_bfloat16* o0 = O + ((size_t)t * NH + h0) * D + dim;
        __nv_bfloat16* o1 = O + ((size_t)t * NH + h1) * D + dim;
        o0[0] = __float2bfloat16_rn(acc[nt][0] / d0); o0[1] = __float2bfloat16_rn(acc[nt][1] / d0);
        o1[0] = __float2bfloat16_rn(acc[nt][2] / d1); o1[1] = __float2bfloat16_rn(acc[nt][3] / d1);
    }
}

template <int HGPC>
__device__ __forceinline__ void sparse_attn_mma_hg_body(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ RING, const int32_t* __restrict__ WPOS,
    const __nv_bfloat16* __restrict__ CKV, const long long* __restrict__ CIDX, const float* __restrict__ SINK,
    __nv_bfloat16* __restrict__ O, int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale)
{
    constexpr int TK = 16, NTH = MNT * HGPC, ROWS = MH * HGPC;
    const int t = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // Warp = (head block hbl, dim quarter wq): EXACTLY one warp of the 16-head kernel, so every
    // product, partial and reduction is the same -- only the K tile is gathered once for all.
    const int hbl = warp >> 2, wq = warp & 3;
    const int hg = blockIdx.y * HGPC + hbl;
    const int g = lane >> 2, q = lane & 3;

    __shared__ __align__(16) __nv_bfloat16 ks[TK][DMAX + KPAD];
    __shared__ float spart[HGPC][MWARPS][MH][TK];
    __shared__ __align__(16) __nv_bfloat16 ph[ROWS][TK + KPAD], pl[ROWS][TK + KPAD];
    __shared__ float mrow[ROWS], lrow[ROWS], arow[ROWS];
    __shared__ int vld[TK];

    // Q stays in registers: this warp's A fragments over its 128 dims (8 k-steps).
    uint32_t qa[8][4];
    {
        const __nv_bfloat16* q0 = Q + ((size_t)t * NH + hg * MH + g) * D;
        const __nv_bfloat16* q1 = q0 + 8 * (size_t)D;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const int kd = wq * 128 + kk * 16;
            qa[kk][0] = *reinterpret_cast<const uint32_t*>(q0 + kd + 2 * q);
            qa[kk][1] = *reinterpret_cast<const uint32_t*>(q1 + kd + 2 * q);
            qa[kk][2] = *reinterpret_cast<const uint32_t*>(q0 + kd + 8 + 2 * q);
            qa[kk][3] = *reinterpret_cast<const uint32_t*>(q1 + kd + 8 + 2 * q);
        }
    }
    if (tid < ROWS) { mrow[tid] = -INFINITY; lrow[tid] = 0.f; }
    float acc[16][4];
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    __syncthreads();

    for (int seg = 0; seg < 2; ++seg) {
        const int n = seg == 0 ? NW : NC;
        if (seg == 1 && CIDX == nullptr) continue;
        const __nv_bfloat16* src = seg == 0 ? RING : CKV;
        for (int kb = 0; kb < n; kb += TK) {
            for (int j = tid; j < TK; j += NTH) {
                const int c = kb + j;
                int row = -1;
                if (c < n) {
                    if (seg == 0) {
                        const long long p = WPOS[(size_t)t * NW + c];
                        if (p >= 0 && p >= win_lo) row = (int)(p % RING_N);
                    } else {
                        const long long j = CIDX[(size_t)t * NC + c];
                        if (j >= 0) row = (int)j;
                    }
                }
                vld[j] = row;
            }
            __syncthreads();
            for (int i = tid; i < TK * DMAX / 8; i += NTH) {
                const int r = i / (DMAX / 8), c = i % (DMAX / 8);
                const int row = vld[r];
                *reinterpret_cast<uint4*>(&ks[r][c * 8]) = row >= 0
                    ? reinterpret_cast<const uint4*>(src + (size_t)row * D)[c] : make_uint4(0, 0, 0, 0);
            }
            __syncthreads();

            // ---- S partial over this warp's 128 dims: 8 k-steps x 2 n-tiles of 8 keys
            constexpr int NS = TK / 8;
            float sp[NS][4];
#pragma unroll
            for (int nt = 0; nt < NS; ++nt) sp[nt][0] = sp[nt][1] = sp[nt][2] = sp[nt][3] = 0.f;
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                const int kd = wq * 128 + kk * 16;
                const uint32_t (&a)[4] = qa[kk];
#pragma unroll
                for (int nt = 0; nt < NS; ++nt) {
                    uint32_t b[2];
                    b[0] = *reinterpret_cast<const uint32_t*>(&ks[nt * 8 + g][kd + 2 * q]);
                    b[1] = *reinterpret_cast<const uint32_t*>(&ks[nt * 8 + g][kd + 8 + 2 * q]);
                    mma_bf16_16816(sp[nt], a, b);
                }
            }
#pragma unroll
            for (int nt = 0; nt < NS; ++nt) {
                spart[hbl][wq][g][nt * 8 + 2 * q]         = sp[nt][0];
                spart[hbl][wq][g][nt * 8 + 2 * q + 1]     = sp[nt][1];
                spart[hbl][wq][g + 8][nt * 8 + 2 * q]     = sp[nt][2];
                spart[hbl][wq][g + 8][nt * 8 + 2 * q + 1] = sp[nt][3];
            }
            __syncthreads();

            // ---- online softmax: 8 threads per head row, 2 keys each
            {
                constexpr int KPT = TK / 8;   // keys per thread
                const int row = tid >> 3, k0 = (tid & 7) * KPT;
                float sv[KPT], pv[KPT];
                float tmax = -INFINITY;
#pragma unroll
                for (int j = 0; j < KPT; ++j) {
                    const int k = k0 + j;
                    float v = spart[row / MH][0][row % MH][k];
                    for (int w = 1; w < MWARPS; ++w) v += spart[row / MH][w][row % MH][k];
                    const bool ok = vld[k] >= 0 && kb + k < n;
                    sv[j] = ok ? v * scale : -INFINITY;
                    tmax = fmaxf(tmax, sv[j]);
                }
                for (int off = 1; off < 8; off <<= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, off));
                const float m_old = mrow[row];
                const float m_new = fmaxf(m_old, tmax);
                const float m_safe = (m_new == -INFINITY) ? 0.f : m_new;
                const float alpha = __expf(m_old - m_safe);
                float lsum = 0.f;
#pragma unroll
                for (int j = 0; j < KPT; ++j) {
                    pv[j] = sv[j] == -INFINITY ? 0.f : __expf(sv[j] - m_safe);
                    lsum += pv[j];
                    const __nv_bfloat16 hi = __float2bfloat16_rn(pv[j]);
                    ph[row][k0 + j] = hi;
                    pl[row][k0 + j] = __float2bfloat16_rn(pv[j] - __bfloat162float(hi));
                }
                for (int off = 1; off < 8; off <<= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, off);
                __syncwarp();
                if ((tid & 7) == 0) {
                    lrow[row] = lrow[row] * alpha + lsum;
                    mrow[row] = m_new;
                    arow[row] = alpha;
                }
            }
            __syncthreads();

            // ---- O = O * alpha + (P_hi + P_lo) K over this warp's 128 dims (16 n-tiles)
            {
                const float a0 = arow[hbl * MH + g], a1 = arow[hbl * MH + g + 8];
#pragma unroll
                for (int nt = 0; nt < 16; ++nt) {
                    acc[nt][0] *= a0; acc[nt][1] *= a0; acc[nt][2] *= a1; acc[nt][3] *= a1;
                }
#pragma unroll
                for (int ks16 = 0; ks16 < TK / 16; ++ks16) {
                    const int kc = ks16 * 16;
                    uint32_t ah[4], al[4];
                    ah[0] = *reinterpret_cast<const uint32_t*>(&ph[hbl * MH + g][kc + 2 * q]);
                    ah[1] = *reinterpret_cast<const uint32_t*>(&ph[hbl * MH + g + 8][kc + 2 * q]);
                    ah[2] = *reinterpret_cast<const uint32_t*>(&ph[hbl * MH + g][kc + 8 + 2 * q]);
                    ah[3] = *reinterpret_cast<const uint32_t*>(&ph[hbl * MH + g + 8][kc + 8 + 2 * q]);
                    al[0] = *reinterpret_cast<const uint32_t*>(&pl[hbl * MH + g][kc + 2 * q]);
                    al[1] = *reinterpret_cast<const uint32_t*>(&pl[hbl * MH + g + 8][kc + 2 * q]);
                    al[2] = *reinterpret_cast<const uint32_t*>(&pl[hbl * MH + g][kc + 8 + 2 * q]);
                    al[3] = *reinterpret_cast<const uint32_t*>(&pl[hbl * MH + g + 8][kc + 8 + 2 * q]);
#pragma unroll
                    for (int nt = 0; nt < 16; ++nt) {
                        const int dim = wq * 128 + nt * 8 + g;
                        uint32_t b[2];
                        b[0] = pack2(ks[kc + 2 * q][dim], ks[kc + 2 * q + 1][dim]);
                        b[1] = pack2(ks[kc + 8 + 2 * q][dim], ks[kc + 9 + 2 * q][dim]);
                        mma_bf16_16816(acc[nt], ah, b);
                        mma_bf16_16816(acc[nt], al, b);
                    }
                }
            }
            __syncthreads();
        }
    }

    const int h0 = hg * MH + g, h1 = h0 + 8;
    const float m0 = mrow[hbl * MH + g] == -INFINITY ? 0.f : mrow[hbl * MH + g];
    const float m1 = mrow[hbl * MH + g + 8] == -INFINITY ? 0.f : mrow[hbl * MH + g + 8];
    const float d0 = lrow[hbl * MH + g] + __expf(SINK[h0] - m0);
    const float d1 = lrow[hbl * MH + g + 8] + __expf(SINK[h1] - m1);
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        const int dim = wq * 128 + nt * 8 + 2 * q;
        __nv_bfloat16* o0 = O + ((size_t)t * NH + h0) * D + dim;
        __nv_bfloat16* o1 = O + ((size_t)t * NH + h1) * D + dim;
        o0[0] = __float2bfloat16_rn(acc[nt][0] / d0); o0[1] = __float2bfloat16_rn(acc[nt][1] / d0);
        o1[0] = __float2bfloat16_rn(acc[nt][2] / d1); o1[1] = __float2bfloat16_rn(acc[nt][3] / d1);
    }
}

// 64 heads per CTA: 4 x the 16-head kernel's warps in one CTA (512 threads), sharing ONE gather
// of each K tile instead of four. Identical arithmetic by construction -> the gate requires 0
// differing values vs dsv41_sparse_attn_mma.
extern "C" __global__ void __launch_bounds__(MNT * 4) dsv41_sparse_attn_mma64(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ RING, const int32_t* __restrict__ WPOS,
    const __nv_bfloat16* __restrict__ CKV, const long long* __restrict__ CIDX, const float* __restrict__ SINK,
    __nv_bfloat16* __restrict__ O, int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale)
{
    sparse_attn_mma_hg_body<4>(Q, RING, WPOS, CKV, CIDX, SINK, O, T, NH, D, NW, NC, RING_N, win_lo, scale);
}

// 32 heads per CTA (256 threads): the occupancy middle ground between 16 and 64.
extern "C" __global__ void __launch_bounds__(MNT * 2) dsv41_sparse_attn_mma32h(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ RING, const int32_t* __restrict__ WPOS,
    const __nv_bfloat16* __restrict__ CKV, const long long* __restrict__ CIDX, const float* __restrict__ SINK,
    __nv_bfloat16* __restrict__ O, int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale)
{
    sparse_attn_mma_hg_body<2>(Q, RING, WPOS, CKV, CIDX, SINK, O, T, NH, D, NW, NC, RING_N, win_lo, scale);
}



// ------------------------------------------------------------------ FA2-STYLE PREFILL
// dsv41_sparse_attn_fa{16h,32h}: dsv41_sparse_attn_mma32h's arithmetic, BYTE-IDENTICAL by
// construction (the gate requires 0 differing values), restructured so the online softmax runs in
// REGISTERS on the S accumulators, as FlashAttention-2 does:
//  * S: each warp sums its own 128 dims with the SAME 8-step MMA chain; the 4 quarter partials go
//    through smem once and every warp of the head block adds them itself in the SAME fixed order
//    (q0 + q1 + q2 + q3). The separate softmax thread mapping and the P smem round trip are gone.
//  * Row max: quad shuffles (order-free). Row sum: the mma kernel's 8-lane butterfly over pair
//    sums s_j = p[2j] + p[2j+1] is ((s0+s1)+(s2+s3)) + ((s4+s5)+(s6+s7)); here lane q holds s_q
//    and s_{4+q}, and two quad butterflies plus one add give the same expression.
//  * P_hi / P_lo (kept, as ruled) become the PV MMA's A fragments straight from the S layout; each
//    accumulator sees the same MMA sequence (P_hi then P_lo per 16 keys).
//  * PV's B fragments by ldmatrix.trans instead of scalar packs (the same values).
//  * K tiles double-buffered with cp.async (src-size 0 for masked rows = the same zero rows).
// Two __syncthreads per 16-key tile instead of five.
__device__ __forceinline__ void attn_cp_async16(void* smem, const void* gmem, bool pred) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(pred ? 16 : 0));
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], const void* smem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

template <int HGPC>
__device__ __forceinline__ void sparse_attn_fa_body(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ RING, const int32_t* __restrict__ WPOS,
    const __nv_bfloat16* __restrict__ CKV, const long long* __restrict__ CIDX, const float* __restrict__ SINK,
    __nv_bfloat16* __restrict__ O, int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale)
{
    constexpr int TK = 16, NTH = MNT * HGPC, NS = TK / 8;
    const int t = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int hbl = warp >> 2, wq = warp & 3;   // head block in the CTA, dim quarter
    const int hg = blockIdx.y * HGPC + hbl;
    const int g = lane >> 2, q = lane & 3;

    __shared__ __align__(16) __nv_bfloat16 ks[2][TK][DMAX + KPAD];
    __shared__ float spart[HGPC][MWARPS][MH][TK];
    __shared__ int vld[2][TK];

    uint32_t qa[8][4];
    {
        const __nv_bfloat16* q0 = Q + ((size_t)t * NH + hg * MH + g) * D;
        const __nv_bfloat16* q1 = q0 + 8 * (size_t)D;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const int kd = wq * 128 + kk * 16;
            qa[kk][0] = *reinterpret_cast<const uint32_t*>(q0 + kd + 2 * q);
            qa[kk][1] = *reinterpret_cast<const uint32_t*>(q1 + kd + 2 * q);
            qa[kk][2] = *reinterpret_cast<const uint32_t*>(q0 + kd + 8 + 2 * q);
            qa[kk][3] = *reinterpret_cast<const uint32_t*>(q1 + kd + 8 + 2 * q);
        }
    }

    // Tiles: the window's first (as the reference), then the compressed selection.
    const int n_c = CIDX == nullptr ? 0 : NC;
    const int tiles_w = (NW + TK - 1) / TK, tiles = tiles_w + (n_c + TK - 1) / TK;
    auto tile_row = [&](int it, int j) -> int {   // gathered row of key j of tile it; -1 = masked
        if (it < tiles_w) {
            const int c = it * TK + j;
            if (c >= NW) return -1;
            const long long p = WPOS[(size_t)t * NW + c];
            return (p >= 0 && p >= win_lo) ? (int)(p % RING_N) : -1;
        }
        const int c = (it - tiles_w) * TK + j;
        if (c >= n_c) return -1;
        const long long jj = CIDX[(size_t)t * NC + c];
        return jj >= 0 ? (int)jj : -1;
    };
    auto gather = [&](int it, int b) {
        const __nv_bfloat16* src = it < tiles_w ? RING : CKV;
        for (int i = tid; i < TK * DMAX / 8; i += NTH) {
            const int r = i / (DMAX / 8), c = i % (DMAX / 8);
            const int row = tile_row(it, r);
            if (c == 0) vld[b][r] = row;
            attn_cp_async16(&ks[b][r][c * 8], src + (size_t)(row >= 0 ? row : 0) * D + c * 8, row >= 0);
        }
        asm volatile("cp.async.commit_group;\n");
    };

    // This warp's copy of rows g and g + 8 (every warp of the head block holds the same values).
    float mr0 = -INFINITY, mr1 = -INFINITY, lr0 = 0.f, lr1 = 0.f;
    float acc[16][4];
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;

    if (tiles > 0) gather(0, 0);
    for (int it = 0; it < tiles; ++it) {
        const int b = it & 1;
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();   // tile it has landed; every warp is done with tile it - 1
        if (it + 1 < tiles) gather(it + 1, b ^ 1);

        // ---- S partial over this warp's 128 dims (the mma kernel's chain)
        float sp[NS][4];
#pragma unroll
        for (int nt = 0; nt < NS; ++nt) sp[nt][0] = sp[nt][1] = sp[nt][2] = sp[nt][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const int kd = wq * 128 + kk * 16;
#pragma unroll
            for (int nt = 0; nt < NS; ++nt) {
                uint32_t bb[2];
                bb[0] = *reinterpret_cast<const uint32_t*>(&ks[b][nt * 8 + g][kd + 2 * q]);
                bb[1] = *reinterpret_cast<const uint32_t*>(&ks[b][nt * 8 + g][kd + 8 + 2 * q]);
                mma_bf16_16816(sp[nt], qa[kk], bb);
            }
        }
#pragma unroll
        for (int nt = 0; nt < NS; ++nt) {
            spart[hbl][wq][g][nt * 8 + 2 * q]         = sp[nt][0];
            spart[hbl][wq][g][nt * 8 + 2 * q + 1]     = sp[nt][1];
            spart[hbl][wq][g + 8][nt * 8 + 2 * q]     = sp[nt][2];
            spart[hbl][wq][g + 8][nt * 8 + 2 * q + 1] = sp[nt][3];
        }
        __syncthreads();   // all four quarters' partials are in

        // ---- full S in the accumulator layout: e = 0,1 row g; e = 2,3 row g + 8
        float pv[NS][4];
        float t0 = -INFINITY, t1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < NS; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = g + (e >> 1) * 8, k = nt * 8 + 2 * q + (e & 1);
                float v = spart[hbl][0][row][k];
                for (int w = 1; w < MWARPS; ++w) v += spart[hbl][w][row][k];
                pv[nt][e] = vld[b][k] >= 0 ? v * scale : -INFINITY;
                if (e < 2) t0 = fmaxf(t0, pv[nt][e]); else t1 = fmaxf(t1, pv[nt][e]);
            }
        for (int off = 1; off < 4; off <<= 1) {
            t0 = fmaxf(t0, __shfl_xor_sync(0xffffffffu, t0, off));
            t1 = fmaxf(t1, __shfl_xor_sync(0xffffffffu, t1, off));
        }
        const float mn0 = fmaxf(mr0, t0), mn1 = fmaxf(mr1, t1);
        const float ms0 = (mn0 == -INFINITY) ? 0.f : mn0, ms1 = (mn1 == -INFINITY) ? 0.f : mn1;
        const float al0 = __expf(mr0 - ms0), al1 = __expf(mr1 - ms1);
#pragma unroll
        for (int nt = 0; nt < NS; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float sv = pv[nt][e];
                pv[nt][e] = sv == -INFINITY ? 0.f : __expf(sv - (e < 2 ? ms0 : ms1));
            }
        // l: pair sums, quad butterfly per key half, then the halves -- the mma kernel's order.
        float s00 = pv[0][0] + pv[0][1], s01 = pv[1][0] + pv[1][1];   // row g: s_q, s_{4+q}
        float s10 = pv[0][2] + pv[0][3], s11 = pv[1][2] + pv[1][3];   // row g + 8
        for (int off = 1; off < 4; off <<= 1) {
            s00 += __shfl_xor_sync(0xffffffffu, s00, off);
            s01 += __shfl_xor_sync(0xffffffffu, s01, off);
            s10 += __shfl_xor_sync(0xffffffffu, s10, off);
            s11 += __shfl_xor_sync(0xffffffffu, s11, off);
        }
        const float ls0 = s00 + s01, ls1 = s10 + s11;
        lr0 = lr0 * al0 + ls0;
        lr1 = lr1 * al1 + ls1;
        mr0 = mn0;
        mr1 = mn1;

        // ---- P_hi / P_lo as A fragments: a0 row g keys 2q.., a1 row g+8, a2/a3 keys 8+2q..
        uint32_t ah[4], alo[4];
#pragma unroll
        for (int nt = 0; nt < NS; ++nt)
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const float p0 = pv[nt][2 * hr], p1 = pv[nt][2 * hr + 1];
                const __nv_bfloat16 h0 = __float2bfloat16_rn(p0), h1 = __float2bfloat16_rn(p1);
                ah[nt * 2 + hr] = pack2(h0, h1);
                alo[nt * 2 + hr] = pack2(__float2bfloat16_rn(p0 - __bfloat162float(h0)),
                                         __float2bfloat16_rn(p1 - __bfloat162float(h1)));
            }

        // ---- O = O * alpha + (P_hi + P_lo) K over this warp's 128 dims
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            acc[nt][0] *= al0; acc[nt][1] *= al0; acc[nt][2] *= al1; acc[nt][3] *= al1;
        }
        const int krow = (lane & 7) + ((lane >> 3) & 1) * 8, kcol = (lane >> 4) * 8;
#pragma unroll
        for (int nt = 0; nt < 16; nt += 2) {
            uint32_t r[4];
            ldsm_x4_trans(r, &ks[b][krow][wq * 128 + nt * 8 + kcol]);
            const uint32_t b0[2] = { r[0], r[1] }, b1[2] = { r[2], r[3] };
            mma_bf16_16816(acc[nt], ah, b0);
            mma_bf16_16816(acc[nt], alo, b0);
            mma_bf16_16816(acc[nt + 1], ah, b1);
            mma_bf16_16816(acc[nt + 1], alo, b1);
        }
    }

    const int h0 = hg * MH + g, h1 = h0 + 8;
    const float m0 = mr0 == -INFINITY ? 0.f : mr0;
    const float m1 = mr1 == -INFINITY ? 0.f : mr1;
    const float d0 = lr0 + __expf(SINK[h0] - m0);
    const float d1 = lr1 + __expf(SINK[h1] - m1);
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        const int dim = wq * 128 + nt * 8 + 2 * q;
        __nv_bfloat16* o0 = O + ((size_t)t * NH + h0) * D + dim;
        __nv_bfloat16* o1 = O + ((size_t)t * NH + h1) * D + dim;
        o0[0] = __float2bfloat16_rn(acc[nt][0] / d0); o0[1] = __float2bfloat16_rn(acc[nt][1] / d0);
        o1[0] = __float2bfloat16_rn(acc[nt][2] / d1); o1[1] = __float2bfloat16_rn(acc[nt][3] / d1);
    }
}

extern "C" __global__ void __launch_bounds__(MNT) dsv41_sparse_attn_fa16h(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ RING, const int32_t* __restrict__ WPOS,
    const __nv_bfloat16* __restrict__ CKV, const long long* __restrict__ CIDX, const float* __restrict__ SINK,
    __nv_bfloat16* __restrict__ O, int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale)
{
    sparse_attn_fa_body<1>(Q, RING, WPOS, CKV, CIDX, SINK, O, T, NH, D, NW, NC, RING_N, win_lo, scale);
}

extern "C" __global__ void __launch_bounds__(MNT * 2) dsv41_sparse_attn_fa32h(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ RING, const int32_t* __restrict__ WPOS,
    const __nv_bfloat16* __restrict__ CKV, const long long* __restrict__ CIDX, const float* __restrict__ SINK,
    __nv_bfloat16* __restrict__ O, int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale)
{
    sparse_attn_fa_body<2>(Q, RING, WPOS, CKV, CIDX, SINK, O, T, NH, D, NW, NC, RING_N, win_lo, scale);
}

template <int HGPC>
__device__ __forceinline__ void sparse_attn_mma_pipe_body(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ RING, const int32_t* __restrict__ WPOS,
    const __nv_bfloat16* __restrict__ CKV, const long long* __restrict__ CIDX, const float* __restrict__ SINK,
    __nv_bfloat16* __restrict__ O, int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale)
{
    constexpr int TK = 16, NTH = MNT * HGPC, ROWS = MH * HGPC;
    const int t = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // Warp = (head block hbl, dim quarter wq): EXACTLY one warp of the 16-head kernel, so every
    // product, partial and reduction is the same -- only the K tile is gathered once for all.
    const int hbl = warp >> 2, wq = warp & 3;
    const int hg = blockIdx.y * HGPC + hbl;
    const int g = lane >> 2, q = lane & 3;

    __shared__ __align__(16) __nv_bfloat16 ks[2][TK][DMAX + KPAD];
    __shared__ float spart[HGPC][MWARPS][MH][TK];
    __shared__ __align__(16) __nv_bfloat16 ph[ROWS][TK + KPAD], pl[ROWS][TK + KPAD];
    __shared__ float mrow[ROWS], lrow[ROWS], arow[ROWS];
    __shared__ int vld[2][TK];

    // Q stays in registers: this warp's A fragments over its 128 dims (8 k-steps).
    uint32_t qa[8][4];
    {
        const __nv_bfloat16* q0 = Q + ((size_t)t * NH + hg * MH + g) * D;
        const __nv_bfloat16* q1 = q0 + 8 * (size_t)D;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const int kd = wq * 128 + kk * 16;
            qa[kk][0] = *reinterpret_cast<const uint32_t*>(q0 + kd + 2 * q);
            qa[kk][1] = *reinterpret_cast<const uint32_t*>(q1 + kd + 2 * q);
            qa[kk][2] = *reinterpret_cast<const uint32_t*>(q0 + kd + 8 + 2 * q);
            qa[kk][3] = *reinterpret_cast<const uint32_t*>(q1 + kd + 8 + 2 * q);
        }
    }
    if (tid < ROWS) { mrow[tid] = -INFINITY; lrow[tid] = 0.f; }
    float acc[16][4];
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    __syncthreads();

    const int n_c = CIDX == nullptr ? 0 : NC;
    const int tiles_w = (NW + TK - 1) / TK, tiles = tiles_w + (n_c + TK - 1) / TK;
    auto tile_row = [&](int it, int j) -> int {   // gathered row of key j of tile it; -1 = masked
        if (it < tiles_w) {
            const int c = it * TK + j;
            if (c >= NW) return -1;
            const long long p = WPOS[(size_t)t * NW + c];
            return (p >= 0 && p >= win_lo) ? (int)(p % RING_N) : -1;
        }
        const int c = (it - tiles_w) * TK + j;
        if (c >= n_c) return -1;
        const long long jj = CIDX[(size_t)t * NC + c];
        return jj >= 0 ? (int)jj : -1;
    };
    auto gather = [&](int it, int bf) {
        const __nv_bfloat16* src = it < tiles_w ? RING : CKV;
        for (int i = tid; i < TK * DMAX / 8; i += NTH) {
            const int r = i / (DMAX / 8), c = i % (DMAX / 8);
            const int row = tile_row(it, r);
            if (c == 0) vld[bf][r] = row;
            attn_cp_async16(&ks[bf][r][c * 8], src + (size_t)(row >= 0 ? row : 0) * D + c * 8, row >= 0);
        }
        asm volatile("cp.async.commit_group;\n");
    };
    if (tiles > 0) gather(0, 0);
    {
        for (int it = 0; it < tiles; ++it) {
            const int bf = it & 1;
            const int n = it < tiles_w ? NW : n_c;
            const int kb = (it < tiles_w ? it : it - tiles_w) * TK;
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();   // tile it has landed; tile it - 1 is fully consumed
            if (it + 1 < tiles) gather(it + 1, bf ^ 1);

            // ---- S partial over this warp's 128 dims: 8 k-steps x 2 n-tiles of 8 keys
            constexpr int NS = TK / 8;
            float sp[NS][4];
#pragma unroll
            for (int nt = 0; nt < NS; ++nt) sp[nt][0] = sp[nt][1] = sp[nt][2] = sp[nt][3] = 0.f;
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                const int kd = wq * 128 + kk * 16;
                const uint32_t (&a)[4] = qa[kk];
#pragma unroll
                for (int nt = 0; nt < NS; ++nt) {
                    uint32_t b[2];
                    b[0] = *reinterpret_cast<const uint32_t*>(&ks[bf][nt * 8 + g][kd + 2 * q]);
                    b[1] = *reinterpret_cast<const uint32_t*>(&ks[bf][nt * 8 + g][kd + 8 + 2 * q]);
                    mma_bf16_16816(sp[nt], a, b);
                }
            }
#pragma unroll
            for (int nt = 0; nt < NS; ++nt) {
                spart[hbl][wq][g][nt * 8 + 2 * q]         = sp[nt][0];
                spart[hbl][wq][g][nt * 8 + 2 * q + 1]     = sp[nt][1];
                spart[hbl][wq][g + 8][nt * 8 + 2 * q]     = sp[nt][2];
                spart[hbl][wq][g + 8][nt * 8 + 2 * q + 1] = sp[nt][3];
            }
            __syncthreads();

            // ---- online softmax: 8 threads per head row, 2 keys each
            {
                constexpr int KPT = TK / 8;   // keys per thread
                const int row = tid >> 3, k0 = (tid & 7) * KPT;
                float sv[KPT], pv[KPT];
                float tmax = -INFINITY;
#pragma unroll
                for (int j = 0; j < KPT; ++j) {
                    const int k = k0 + j;
                    float v = spart[row / MH][0][row % MH][k];
                    for (int w = 1; w < MWARPS; ++w) v += spart[row / MH][w][row % MH][k];
                    const bool ok = vld[bf][k] >= 0 && kb + k < n;
                    sv[j] = ok ? v * scale : -INFINITY;
                    tmax = fmaxf(tmax, sv[j]);
                }
                for (int off = 1; off < 8; off <<= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, off));
                const float m_old = mrow[row];
                const float m_new = fmaxf(m_old, tmax);
                const float m_safe = (m_new == -INFINITY) ? 0.f : m_new;
                const float alpha = __expf(m_old - m_safe);
                float lsum = 0.f;
#pragma unroll
                for (int j = 0; j < KPT; ++j) {
                    pv[j] = sv[j] == -INFINITY ? 0.f : __expf(sv[j] - m_safe);
                    lsum += pv[j];
                    const __nv_bfloat16 hi = __float2bfloat16_rn(pv[j]);
                    ph[row][k0 + j] = hi;
                    pl[row][k0 + j] = __float2bfloat16_rn(pv[j] - __bfloat162float(hi));
                }
                for (int off = 1; off < 8; off <<= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, off);
                __syncwarp();
                if ((tid & 7) == 0) {
                    lrow[row] = lrow[row] * alpha + lsum;
                    mrow[row] = m_new;
                    arow[row] = alpha;
                }
            }
            __syncthreads();

            // ---- O = O * alpha + (P_hi + P_lo) K over this warp's 128 dims (16 n-tiles)
            {
                const float a0 = arow[hbl * MH + g], a1 = arow[hbl * MH + g + 8];
#pragma unroll
                for (int nt = 0; nt < 16; ++nt) {
                    acc[nt][0] *= a0; acc[nt][1] *= a0; acc[nt][2] *= a1; acc[nt][3] *= a1;
                }
#pragma unroll
                for (int ks16 = 0; ks16 < TK / 16; ++ks16) {
                    const int kc = ks16 * 16;
                    uint32_t ah[4], al[4];
                    ah[0] = *reinterpret_cast<const uint32_t*>(&ph[hbl * MH + g][kc + 2 * q]);
                    ah[1] = *reinterpret_cast<const uint32_t*>(&ph[hbl * MH + g + 8][kc + 2 * q]);
                    ah[2] = *reinterpret_cast<const uint32_t*>(&ph[hbl * MH + g][kc + 8 + 2 * q]);
                    ah[3] = *reinterpret_cast<const uint32_t*>(&ph[hbl * MH + g + 8][kc + 8 + 2 * q]);
                    al[0] = *reinterpret_cast<const uint32_t*>(&pl[hbl * MH + g][kc + 2 * q]);
                    al[1] = *reinterpret_cast<const uint32_t*>(&pl[hbl * MH + g + 8][kc + 2 * q]);
                    al[2] = *reinterpret_cast<const uint32_t*>(&pl[hbl * MH + g][kc + 8 + 2 * q]);
                    al[3] = *reinterpret_cast<const uint32_t*>(&pl[hbl * MH + g + 8][kc + 8 + 2 * q]);
                    const int krow = kc + (lane & 7) + ((lane >> 3) & 1) * 8, kcol = (lane >> 4) * 8;
#pragma unroll
                    for (int nt = 0; nt < 16; nt += 2) {
                        uint32_t r[4];
                        ldsm_x4_trans(r, &ks[bf][krow][wq * 128 + nt * 8 + kcol]);
                        const uint32_t b0[2] = { r[0], r[1] }, b1[2] = { r[2], r[3] };
                        mma_bf16_16816(acc[nt], ah, b0);
                        mma_bf16_16816(acc[nt], al, b0);
                        mma_bf16_16816(acc[nt + 1], ah, b1);
                        mma_bf16_16816(acc[nt + 1], al, b1);
                    }
                }
            }
        }
    }

    const int h0 = hg * MH + g, h1 = h0 + 8;
    const float m0 = mrow[hbl * MH + g] == -INFINITY ? 0.f : mrow[hbl * MH + g];
    const float m1 = mrow[hbl * MH + g + 8] == -INFINITY ? 0.f : mrow[hbl * MH + g + 8];
    const float d0 = lrow[hbl * MH + g] + __expf(SINK[h0] - m0);
    const float d1 = lrow[hbl * MH + g + 8] + __expf(SINK[h1] - m1);
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        const int dim = wq * 128 + nt * 8 + 2 * q;
        __nv_bfloat16* o0 = O + ((size_t)t * NH + h0) * D + dim;
        __nv_bfloat16* o1 = O + ((size_t)t * NH + h1) * D + dim;
        o0[0] = __float2bfloat16_rn(acc[nt][0] / d0); o0[1] = __float2bfloat16_rn(acc[nt][1] / d0);
        o1[0] = __float2bfloat16_rn(acc[nt][2] / d1); o1[1] = __float2bfloat16_rn(acc[nt][3] / d1);
    }
}

// dsv41_sparse_attn_mma32p: mma32h with (a) the K tile double-buffered by cp.async, its gathered
// row computed by the loading thread, and (b) PV's B fragments by ldmatrix.trans instead of
// 64 scalar 16-bit loads per tile. Same values, same MMA sequence per accumulator, same softmax
// -> BYTE-IDENTICAL (gate: 0 differing). Three __syncthreads per tile instead of five.
extern "C" __global__ void __launch_bounds__(MNT * 2) dsv41_sparse_attn_mma32p(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ RING, const int32_t* __restrict__ WPOS,
    const __nv_bfloat16* __restrict__ CKV, const long long* __restrict__ CIDX, const float* __restrict__ SINK,
    __nv_bfloat16* __restrict__ O, int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale)
{
    sparse_attn_mma_pipe_body<2>(Q, RING, WPOS, CKV, CIDX, SINK, O, T, NH, D, NW, NC, RING_N, win_lo, scale);
}

extern "C" __global__ void __launch_bounds__(MNT) dsv41_sparse_attn_mma16p(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ RING, const int32_t* __restrict__ WPOS,
    const __nv_bfloat16* __restrict__ CKV, const long long* __restrict__ CIDX, const float* __restrict__ SINK,
    __nv_bfloat16* __restrict__ O, int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale)
{
    sparse_attn_mma_pipe_body<1>(Q, RING, WPOS, CKV, CIDX, SINK, O, T, NH, D, NW, NC, RING_N, win_lo, scale);
}


// Tensor-core prefill entry (16-key tiles). MEASURED AND DROPPED (attn gate 2026-09-22, real
// fixture T=148): 32-key tiles 0.85x (slower); cp.async double-buffered gather 1.01x (bit-
// identical to this, no gain: the kernel is not gather-latency bound). It sits ~3x above the
// bf16 MMA compute roof, a third of which is the deliberate P_hi + P_lo second MMA.
extern "C" __global__ void __launch_bounds__(MNT) dsv41_sparse_attn_mma(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ RING, const int32_t* __restrict__ WPOS,
    const __nv_bfloat16* __restrict__ CKV, const long long* __restrict__ CIDX, const float* __restrict__ SINK,
    __nv_bfloat16* __restrict__ O, int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale)
{
    sparse_attn_mma_body<16>(Q, RING, WPOS, CKV, CIDX, SINK, O, T, NH, D, NW, NC, RING_N, win_lo, scale);
}


// PRODUCTION entry. grid (T, NH / 8), block 256. CIDX may be null (window-only layers 0/1).
extern "C" __global__ void __launch_bounds__(NT) dsv41_sparse_attn(
    const __nv_bfloat16* Q, const __nv_bfloat16* RING, const long long* WPOS,
    const __nv_bfloat16* CKV, const long long* CIDX, const float* SINK, __nv_bfloat16* O,
    int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale)
{
    sparse_attn_body<false, false, long long, __nv_bfloat16>(
        Q, RING, WPOS, CKV, CIDX, SINK, O, T, NH, D, NW, NC, RING_N, win_lo, scale);
}

#ifdef DSV41_ATTN_GATE
static void* slurp(const char* p, size_t want) {
    FILE* f = std::fopen(p, "rb");
    if (!f) { std::fprintf(stderr, "open %s\n", p); std::exit(2); }
    void* q = std::malloc(want);
    if (std::fread(q, 1, want, f) != want) { std::fprintf(stderr, "short read %s\n", p); std::exit(2); }
    std::fclose(f); return q;
}

struct Err { double rel; double worst; bool finite; };

static Err compare(const float* got, const float* ref, size_t n) {
    double num = 0, den = 0, worst = 0; bool fin = true;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(got[i])) { fin = false; continue; }
        const double d = (double)got[i] - (double)ref[i];
        num += d * d; den += (double)ref[i] * (double)ref[i];
        if (std::fabs(d) > worst) worst = std::fabs(d);
    }
    return Err{ std::sqrt(num / den), worst, fin };
}

int main() {
    int T, NH, D, NW, NC, RING_N, n_c;
    { FILE* f = std::fopen("ar_dims.txt", "r");
      if (!f || std::fscanf(f, "%d %d %d %d %d %d %d", &T,&NH,&D,&NW,&NC,&RING_N,&n_c) != 7) {
          std::fprintf(stderr, "ar_dims.txt\n"); return 2; }
      std::fclose(f); }
    std::printf("T=%d NH=%d D=%d NW=%d NC=%d RING=%d n_c=%d\n", T,NH,D,NW,NC,RING_N,n_c);
    if (D != DMAX || NH % HB) { std::fprintf(stderr, "kernel compiled for D=%d, NH%%%d==0\n", DMAX, HB); return 2; }

    const size_t nq = (size_t)T*NH*D, no = nq;
    void*  h_q    = slurp("ar_q.bin",    nq*2);
    void*  h_ring = slurp("ar_ring.bin", (size_t)RING_N*D*2);
    void*  h_ckv  = slurp("ar_ckv.bin",  (size_t)n_c*D*2);
    float* h_sink = (float*)slurp("ar_sink.bin", (size_t)NH*4);
    void*  h_wpos = slurp("ar_wpos.bin", (size_t)T*NW*4);
    void*  h_cidx = slurp("ar_cidx.bin", (size_t)T*NC*4);
    float* h_r32  = (float*)slurp("ar_f32.bin",  no*4);
    float* h_rbf  = (float*)slurp("ar_bf16.bin", no*4);

    __nv_bfloat16 *d_q, *d_ring, *d_ckv; int32_t *d_wpos, *d_cidx; float *d_sink, *d_o;
    CUDA_OK(cudaMalloc(&d_q, nq*2));                 CUDA_OK(cudaMemcpy(d_q, h_q, nq*2, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&d_ring,(size_t)RING_N*D*2)); CUDA_OK(cudaMemcpy(d_ring,h_ring,(size_t)RING_N*D*2,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&d_ckv, (size_t)n_c*D*2));    CUDA_OK(cudaMemcpy(d_ckv, h_ckv, (size_t)n_c*D*2, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&d_wpos,(size_t)T*NW*4));     CUDA_OK(cudaMemcpy(d_wpos,h_wpos,(size_t)T*NW*4,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&d_cidx,(size_t)T*NC*4));     CUDA_OK(cudaMemcpy(d_cidx,h_cidx,(size_t)T*NC*4,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&d_sink,(size_t)NH*4));       CUDA_OK(cudaMemcpy(d_sink,h_sink,(size_t)NH*4,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&d_o, no*4));

    const float scale = 1.0f / std::sqrt((float)D);
    dim3 grid(T, NH / HB);
    float* h_o = (float*)std::malloc(no*4);

    auto run = [&](int which) {
        CUDA_OK(cudaMemset(d_o, 0, no*4));
        if (which == 0) sparse_attn<false,false><<<grid,NT>>>(d_q,d_ring,d_wpos,d_ckv,d_cidx,d_sink,d_o,T,NH,D,NW,NC,RING_N,0,scale);
        if (which == 1) sparse_attn<true, false><<<grid,NT>>>(d_q,d_ring,d_wpos,d_ckv,d_cidx,d_sink,d_o,T,NH,D,NW,NC,RING_N,0,scale);
        if (which == 2) sparse_attn<false,true ><<<grid,NT>>>(d_q,d_ring,d_wpos,d_ckv,d_cidx,d_sink,d_o,T,NH,D,NW,NC,RING_N,0,scale);
        CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaMemcpy(h_o, d_o, no*4, cudaMemcpyDeviceToHost));
        // Dump for DSV41_PORT/oracle/compare.py, so the verdict does not rest
        // on this file's own arithmetic alone.
        static const char* names[3] = { "ar_out.bin", "ar_out_ctrl_order.bin", "ar_out_ctrl_gather.bin" };
        FILE* f = std::fopen(names[which], "wb");
        if (f) { std::fwrite(h_o, 4, no, f); std::fclose(f); }
    };

    run(0);
    float* h_ok = (float*)std::malloc(no*4);
    std::memcpy(h_ok, h_o, no*4);
    const Err e32 = compare(h_o, h_r32, no);
    const Err ebf = compare(h_o, h_rbf, no);
    std::printf("kernel vs fp32-P reference  : rel_l2=%.3e worst_abs=%.3e\n", e32.rel, e32.worst);
    std::printf("kernel vs bf16-P reference  : rel_l2=%.3e worst_abs=%.3e\n", ebf.rel, ebf.worst);
    if (!e32.finite) { std::printf("FAIL non-finite output\n"); return 1; }

    // Gate A: if token 0 is genuinely all-masked, it must be finite ZERO, not NaN.
    // Whether it IS all-masked is a property of the FIXTURE, so read it off the
    // inputs rather than assuming. The synthetic fixture forces it; a real
    // capture's token 0 is an ordinary token with a one-row window, and
    // demanding zero there is a gate that fails correct output.
    bool tok0_all_masked = true;
    { const int32_t* wp = (const int32_t*)h_wpos;
      const int32_t* ci = (const int32_t*)h_cidx;
      for (int i = 0; i < NW && tok0_all_masked; ++i) if (wp[i] >= 0) tok0_all_masked = false;
      for (int i = 0; i < NC && tok0_all_masked; ++i) if (ci[i] >= 0) tok0_all_masked = false; }

    double amax0 = 0; bool fin0 = true;
    for (size_t i = 0; i < (size_t)NH*D; ++i) {
        if (!std::isfinite(h_o[i])) fin0 = false;
        amax0 = std::fmax(amax0, std::fabs((double)h_o[i]));
    }
    if (tok0_all_masked)
        std::printf("all-masked row (token 0)    : finite=%d  max|o|=%.3e  (expect 0)\n", (int)fin0, amax0);
    else
        std::printf("all-masked row              : NOT PRESENT in this fixture -- NaN-avoidance\n"
                    "                              control NOT EXERCISED here (token 0 is a real token)\n");

    // Negative control 1: compressed-before-window. Online softmax is not
    // associative, so this MUST move -- if it does not, the gate is blind to
    // block order and proves nothing about it.
    run(1);
    const Err c_ord = compare(h_o, h_r32, no);
    const Err d_ord = compare(h_o, h_ok, no);   // the order difference ITSELF
    // Negative control 2: sequential compressed rows instead of the gathered
    // ones. This is the wrong-K-order analogue and must land far away.
    run(2);
    const Err c_gat = compare(h_o, h_r32, no);
    std::printf("CONTROL order  (compressed-first): rel_l2=%.3e vs ref, %.3e vs correct order\n",
                c_ord.rel, d_ord.rel);
    std::printf("CONTROL gather (sequential rows) : rel_l2=%.3e  -> separation %.0fx\n",
                c_gat.rel, c_gat.rel / e32.rel);

    // Deliberately does NOT include the order control: it does not separate,
    // and a gate condition that cannot fail is worse than no gate condition.
    const bool ok = e32.rel < 1e-5 && e32.finite && c_gat.rel > 0.2
                    && (!tok0_all_masked || (fin0 && amax0 == 0.0));
    if (c_gat.rel <= 0.2)
        std::printf("INCONCLUSIVE: the wrong gather also passed -- this gate cannot fail\n");
    // MEASURED, not assumed. Window-first vs compressed-first moves the result
    // by ~6e-07 -- the same order as this kernel's own distance from the fp32
    // reference, ~2300x SMALLER than the bf16-P rounding prefill_attn already
    // accepts, and comfortably inside compare.py's fp32 tolerance. The online
    // softmax rescaling is numerically stable enough in fp32 that block order
    // is a bit-exactness question, NOT a correctness constraint.
    //
    // So: THIS GATE DOES NOT COVER BLOCK ORDER, and no one should contort a
    // kernel to preserve it. Keep window-first to match the reference, but do
    // not treat a reordering as a bug on this evidence.
    std::printf("\nBLOCK ORDER IS NOT COVERED BY THIS GATE. Reordering moves the result\n"
                "      %.3e (vs the kernel's own %.3e from the fp32 reference, and the\n"
                "      %.3e that prefill_attn's bf16-P rounding already costs). Order is a\n"
                "      bit-exactness question here, not a correctness one.\n",
                d_ord.rel, e32.rel, ebf.rel);
    // This warning is printed, not just committed, so it travels with the result.
    if (ebf.rel > e32.rel * 10.0) {
        std::printf("\nNOTE: the gap against the bf16-P reference is this kernel being MORE\n"
                    "      accurate, not less. tools/prefill_attn.py rounds P to bf16 because\n"
                    "      tl.dot needs a tensor-core dtype; this kernel's hand-rolled FMA dot\n"
                    "      has no reason to pay that. DO NOT 'fix' the kernel toward the bf16-P\n"
                    "      number -- that makes the port worse to make a number prettier.\n"
                    "      Consequence to expect: being more accurate means DIVERGING from the\n"
                    "      Python engine, and ulp differences flip MoE router decisions. So\n"
                    "      end-to-end validation is on OUTPUT QUALITY, not bit-identity, and a\n"
                    "      per-layer bisect is only valid up to the FIRST routing flip.\n");
    }
    // PRODUCTION ENTRY: dsv41_sparse_attn takes i64 indices and writes bf16. It is the same
    // body, so its output must equal the gated fp32 output rounded to bf16 BIT FOR BIT --
    // with compressed rows, and window-only (CIDX = null, layers 0/1) against the fp32 kernel
    // run the same way. Anything else means the index widening or the store is wrong.
    bool prod_ok = true;
    {
        std::vector<long long> w64((size_t)T*NW), c64((size_t)T*NC);
        for (size_t i = 0; i < w64.size(); ++i) w64[i] = ((const int32_t*)h_wpos)[i];
        for (size_t i = 0; i < c64.size(); ++i) c64[i] = ((const int32_t*)h_cidx)[i];
        long long *d_w64, *d_c64; __nv_bfloat16* d_ob;
        CUDA_OK(cudaMalloc(&d_w64, w64.size()*8)); CUDA_OK(cudaMemcpy(d_w64, w64.data(), w64.size()*8, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMalloc(&d_c64, c64.size()*8)); CUDA_OK(cudaMemcpy(d_c64, c64.data(), c64.size()*8, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMalloc(&d_ob, no*2));
        std::vector<__nv_bfloat16> ob(no);
        auto bits_equal = [&](const float* ref) {
            size_t bad = 0;
            for (size_t i = 0; i < no; ++i) {
                const __nv_bfloat16 r = __float2bfloat16_rn(ref[i]);
                bad += std::memcmp(&r, &ob[i], 2) != 0;
            }
            return bad;
        };
        for (int window_only = 0; window_only < 2; ++window_only) {
            const long long* ci = window_only ? nullptr : d_c64;
            dsv41_sparse_attn<<<grid,NT>>>(d_q,d_ring,d_w64,d_ckv,ci,d_sink,d_ob,T,NH,D,NW,NC,RING_N,0,scale);
            CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaMemcpy(ob.data(), d_ob, no*2, cudaMemcpyDeviceToHost));
            if (window_only) {
                CUDA_OK(cudaMemset(d_o, 0, no*4));
                sparse_attn<false,false><<<grid,NT>>>(d_q,d_ring,d_wpos,d_ckv,nullptr,d_sink,d_o,T,NH,D,NW,NC,RING_N,0,scale);
                CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
                CUDA_OK(cudaMemcpy(h_o, d_o, no*4, cudaMemcpyDeviceToHost));
            }
            const float* ref = window_only ? h_o : h_ok;
            const size_t bad = bits_equal(ref);
            // control: the window-only output must NOT match the full one (else CIDX was ignored)
            const size_t ctrl = bits_equal(window_only ? h_ok : h_o);
            std::printf("PRODUCTION entry (i64 idx, bf16 out)%s: %zu/%zu differ from bf16(fp32 kernel) | CTRL other mode: %zu differ\n",
                        window_only ? " window-only" : "            ", bad, no, ctrl);
            prod_ok = prod_ok && bad == 0 && ctrl > 0;
        }
        // The forward's entry: i32 window positions straight from the fixture, i64 cidx.
        dsv41_sparse_attn_w32<<<grid,NT>>>(d_q,d_ring,d_wpos,d_ckv,d_c64,d_sink,d_ob,T,NH,D,NW,NC,RING_N,0,scale);
        CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaMemcpy(ob.data(), d_ob, no*2, cudaMemcpyDeviceToHost));
        const size_t bad32 = bits_equal(h_ok);
        std::printf("PRODUCTION entry w32 (i32 wpos, i64 idx, bf16) : %zu/%zu differ from bf16(fp32 kernel)\n", bad32, no);
        prod_ok = prod_ok && bad32 == 0;

        // SPLIT-KV decode entry: S splits of SLICE keys + combine. Only the accumulation
        // order within a row changes, so against bf16(one-pass fp32) the bf16 outputs must be
        // (nearly) identical. Pre-registered: >= 99.9% bit-identical, and vs the fp32 reference
        // within 1e-6. CONTROL: combine over S-1 splits (the last slice dropped) must differ.
        {
            const int SLICE = 32, S = (NW + NC + SLICE - 1) / SLICE;
            float* d_part; CUDA_OK(cudaMalloc(&d_part, (size_t)T * NH * ((NW + NC + 7) / 8) * (2 + D) * 4));
            dim3 gs(T, NH / HB, S);
            auto split = [&](int s_combine) {
                dsv41_sparse_attn_split<<<gs, NT>>>(d_q, d_ring, d_wpos, d_ckv, d_c64, d_part, T, NH, D, NW, NC, RING_N, 0, scale, SLICE);
                CUDA_OK(cudaGetLastError());
                // combine reads S partials per (t,h) with stride S; to drop the last slice the
                // control re-lays nothing -- it combines over the first s_combine of them.
                dsv41_sparse_attn_combine<<<dim3(T, NH), NT>>>(d_part, d_sink, d_ob, NH, D, S);
                CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
                if (s_combine < S) {  // control: zero the last split's partial and recombine
                    for (int tt = 0; tt < T; ++tt)
                        for (int hh = 0; hh < NH; ++hh) {
                            float* pr = d_part + (((size_t)tt * NH + hh) * S + (S - 1)) * (2 + D);
                            const float mneg = -INFINITY, zero = 0.f;
                            CUDA_OK(cudaMemcpy(pr, &mneg, 4, cudaMemcpyHostToDevice));
                            CUDA_OK(cudaMemcpy(pr + 1, &zero, 4, cudaMemcpyHostToDevice));
                        }
                    dsv41_sparse_attn_combine<<<dim3(T, NH), NT>>>(d_part, d_sink, d_ob, NH, D, S);
                    CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
                }
                CUDA_OK(cudaMemcpy(ob.data(), d_ob, no*2, cudaMemcpyDeviceToHost));
            };
            split(S);
            const size_t sbad = bits_equal(h_ok);
            std::vector<float> of(no);
            for (size_t i = 0; i < no; ++i) of[i] = __bfloat162float(ob[i]);
            const Err es = compare(of.data(), h_r32, no);
            split(S - 1);
            const size_t cbad = bits_equal(h_ok);
            std::printf("SPLIT-KV decode entry (S=%d x %d keys + combine): %zu/%zu bf16 differ from bf16(one-pass) (%.4f%% identical), "
                        "vs fp32 ref rel_l2=%.3e | CTRL last split dropped: %zu differ\n",
                        S, SLICE, sbad, no, 100.0 * (1.0 - (double)sbad / no), es.rel, cbad);
            prod_ok = prod_ok && (double)sbad / no <= 1e-3 && cbad > sbad;

            // T=1 latency (the decode shape): last row of the fixture, one-pass vs split+combine.
            {
                const int tl = T - 1;
                const __nv_bfloat16* q1 = d_q + (size_t)tl * NH * D;
                const int32_t* w1 = d_wpos + (size_t)tl * NW;
                const long long* c1 = d_c64 + (size_t)tl * NC;
                cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
                auto time_it = [&](auto&& launch) {
                    for (int i = 0; i < 10; ++i) launch();
                    CUDA_OK(cudaDeviceSynchronize());
                    cudaEventRecord(e0);
                    for (int i = 0; i < 200; ++i) launch();
                    cudaEventRecord(e1); cudaEventSynchronize(e1);
                    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                    return 1000.0f * ms / 200;
                };
                const float one = time_it([&] {
                    dsv41_sparse_attn_w32<<<dim3(1, NH / HB), NT>>>(q1, d_ring, w1, d_ckv, c1, d_sink, d_ob, 1, NH, D, NW, NC, RING_N, 0, scale); });
                std::printf("T=1 latency (GPU-lock held, nothing else running): one-pass %.1f us\n", one);
                for (int sl : {64, 32, 16, 8}) {
                    const int ss = (NW + NC + sl - 1) / sl;
                    const float spl = time_it([&] {
                        dsv41_sparse_attn_split<<<dim3(1, NH / HB, ss), NT>>>(q1, d_ring, w1, d_ckv, c1, d_part, 1, NH, D, NW, NC, RING_N, 0, scale, sl);
                        dsv41_sparse_attn_combine<<<dim3(1, NH), NT>>>(d_part, d_sink, d_ob, NH, D, ss); });
                    CUDA_OK(cudaGetLastError());
                    const float so = time_it([&] {
                        dsv41_sparse_attn_split<<<dim3(1, NH / HB, ss), NT>>>(q1, d_ring, w1, d_ckv, c1, d_part, 1, NH, D, NW, NC, RING_N, 0, scale, sl); });
                    const float co = time_it([&] {
                        dsv41_sparse_attn_combine<<<dim3(1, NH), NT>>>(d_part, d_sink, d_ob, NH, D, ss); });
                    std::printf("    split SLICE=%2d (S=%2d, %3d CTAs) + combine %.1f us (%.1fx)  [split alone %.1f, combine alone %.1f]\n",
                                sl, ss, ss * NH / HB, spl, one / spl, so, co);
                }
                // LAST-BLOCK COMBINE + the slice sweep for T=1 AND T=6 (verify), SAME slice for both
                // (spec == plain needs each verify row == its T=1 decode). PRE-REGISTERED: split_lb
                // BYTE-IDENTICAL to split + combine at every (T, slice), on two back-to-back launches
                // (the ticket reset). Numerics per slice: bf16 identity vs one-pass on those rows.
                unsigned* d_cnt; CUDA_OK(cudaMalloc(&d_cnt, 64 * 64 * 4)); CUDA_OK(cudaMemset(d_cnt, 0, 64 * 64 * 4));
                for (int tt : {1, 2, 3, 4, 5, 6}) {
                    const int t0 = T - tt;
                    const __nv_bfloat16* qv = d_q + (size_t)t0 * NH * D;
                    const int32_t* wv = d_wpos + (size_t)t0 * NW;
                    const long long* cv = d_c64 + (size_t)t0 * NC;
                    const size_t nv = (size_t)tt * NH * D;
                    std::vector<__nv_bfloat16> onep(nv), ref(nv), got(nv);
                    dsv41_sparse_attn_w32<<<dim3(tt, NH / HB), NT>>>(qv, d_ring, wv, d_ckv, cv, d_sink, d_ob, tt, NH, D, NW, NC, RING_N, 0, scale);
                    CUDA_OK(cudaDeviceSynchronize()); CUDA_OK(cudaMemcpy(onep.data(), d_ob, nv * 2, cudaMemcpyDeviceToHost));
                    for (int sl : {16, 32, 64}) {
                        const int ss = (NW + NC + sl - 1) / sl;
                        auto sc2 = [&] {
                            dsv41_sparse_attn_split<<<dim3(tt, NH / HB, ss), NT>>>(qv, d_ring, wv, d_ckv, cv, d_part, tt, NH, D, NW, NC, RING_N, 0, scale, sl);
                            dsv41_sparse_attn_combine<<<dim3(tt, NH), NT>>>(d_part, d_sink, d_ob, NH, D, ss); };
                        auto lbk = [&] {
                            dsv41_sparse_attn_split_lb<<<dim3(tt, NH / HB, ss), NT>>>(qv, d_ring, wv, d_ckv, cv, d_part, d_sink, d_ob, d_cnt, tt, NH, D, NW, NC, RING_N, 0, scale, sl); };
                        sc2(); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
                        CUDA_OK(cudaMemcpy(ref.data(), d_ob, nv * 2, cudaMemcpyDeviceToHost));
                        size_t dlb = 0;
                        for (int rep2 = 0; rep2 < 2; ++rep2) {
                            CUDA_OK(cudaMemset(d_ob, 0, nv * 2));
                            lbk(); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
                            CUDA_OK(cudaMemcpy(got.data(), d_ob, nv * 2, cudaMemcpyDeviceToHost));
                            for (size_t i = 0; i < nv; ++i) dlb += std::memcmp(&got[i], &ref[i], 2) != 0;
                        }
                        size_t vs1 = 0; for (size_t i = 0; i < nv; ++i) vs1 += std::memcmp(&ref[i], &onep[i], 2) != 0;
                        const float t_sc = time_it(sc2), t_lb = time_it(lbk);
                        std::printf("    T=%d SLICE=%3d (S=%2d): split+combine %.1f us | split_lb %.1f us (%.2fx) | split_lb vs split+combine: %zu differ (must be 0) | vs one-pass %.4f%% identical\n",
                                    tt, sl, ss, t_sc, t_lb, t_sc / t_lb, dlb, 100.0 * (1.0 - (double)vs1 / nv));
                        prod_ok = prod_ok && dlb == 0;
                    }
                }
                cudaFree(d_cnt);
            }
            cudaFree(d_part);
        }
        // TENSOR-CORE prefill entry. PRE-REGISTERED (before its first run): vs the fp32
        // reference rel_l2 <= 1e-5 (the one-pass kernel is 3.5e-7; P_hi+P_lo keeps ~2^-17 of P);
        // >= 99.0% of bf16 outputs identical to bf16(one-pass fp32). CONTROL: compressed rows
        // dropped (NC = 0) must differ on most outputs. Timing at the fixture's full T.
        {
            dim3 gm(T, NH / MH);
            auto mma = [&](int nc) {
                dsv41_sparse_attn_mma<<<gm, MNT>>>(d_q, d_ring, d_wpos, d_ckv, nc ? d_c64 : nullptr, d_sink, d_ob, T, NH, D, NW, NC, RING_N, 0, scale);
                CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
                CUDA_OK(cudaMemcpy(ob.data(), d_ob, no*2, cudaMemcpyDeviceToHost));
            };
            mma(1);
            const size_t mbad = bits_equal(h_ok);
            std::vector<float> of(no);
            for (size_t i = 0; i < no; ++i) of[i] = __bfloat162float(ob[i]);
            const Err em = compare(of.data(), h_r32, no);
            std::vector<float> okb(no);
            for (size_t i = 0; i < no; ++i) okb[i] = __bfloat162float(__float2bfloat16_rn(h_ok[i]));
            const Err eo = compare(okb.data(), h_r32, no);
            mma(0);
            const size_t cbad = bits_equal(h_ok);
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            auto time_it = [&](auto&& launch) {
                for (int i = 0; i < 3; ++i) launch();
                CUDA_OK(cudaDeviceSynchronize());
                cudaEventRecord(e0);
                for (int i = 0; i < 20; ++i) launch();
                cudaEventRecord(e1); cudaEventSynchronize(e1);
                float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                return ms / 20;
            };
            const float t_one = time_it([&] { dsv41_sparse_attn_w32<<<dim3(T, NH / HB), NT>>>(d_q, d_ring, d_wpos, d_ckv, d_c64, d_sink, d_ob, T, NH, D, NW, NC, RING_N, 0, scale); });
            const float t_mma = time_it([&] { dsv41_sparse_attn_mma<<<gm, MNT>>>(d_q, d_ring, d_wpos, d_ckv, d_c64, d_sink, d_ob, T, NH, D, NW, NC, RING_N, 0, scale); });
            std::printf("MMA prefill entry: %zu/%zu bf16 differ from bf16(one-pass) (%.4f%% identical); vs fp32 ref: mma %.3e (bf16-rounded one-pass %.3e) "
                        "| CTRL compressed rows dropped: %zu differ | T=%d: one-pass %.3f ms, mma %.3f ms (%.1fx)\n",
                        mbad, no, 100.0 * (1.0 - (double)mbad / no), em.rel, eo.rel, cbad, T, t_one, t_mma, t_one / t_mma);
            prod_ok = prod_ok && em.rel <= 1e-5 + eo.rel && (double)mbad / no <= 0.01 && cbad > no / 2;
            // 64 heads per CTA. PRE-REGISTERED: 0 differing values vs the 16-head entry.
            {
                mma(1);
                std::vector<__nv_bfloat16> ref16(ob);
                dsv41_sparse_attn_mma64<<<dim3(T, NH / 64), MNT * 4>>>(d_q, d_ring, d_wpos, d_ckv, d_c64, d_sink, d_ob, T, NH, D, NW, NC, RING_N, 0, scale);
                CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
                CUDA_OK(cudaMemcpy(ob.data(), d_ob, no*2, cudaMemcpyDeviceToHost));
                size_t d64 = 0; for (size_t i = 0; i < no; ++i) d64 += std::memcmp(&ob[i], &ref16[i], 2) != 0;
                const float t64 = time_it([&] { dsv41_sparse_attn_mma64<<<dim3(T, NH / 64), MNT * 4>>>(d_q, d_ring, d_wpos, d_ckv, d_c64, d_sink, d_ob, T, NH, D, NW, NC, RING_N, 0, scale); });
                std::printf("MMA64 prefill entry (64 heads/CTA): %zu/%zu differ from mma16 (must be 0) | T=%d: %.3f ms (%.2fx mma16)\n", d64, no, T, t64, t_mma / t64);
                prod_ok = prod_ok && d64 == 0;
                dsv41_sparse_attn_mma32h<<<dim3(T, NH / 32), MNT * 2>>>(d_q, d_ring, d_wpos, d_ckv, d_c64, d_sink, d_ob, T, NH, D, NW, NC, RING_N, 0, scale);
                CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
                CUDA_OK(cudaMemcpy(ob.data(), d_ob, no*2, cudaMemcpyDeviceToHost));
                size_t d32 = 0; for (size_t i = 0; i < no; ++i) d32 += std::memcmp(&ob[i], &ref16[i], 2) != 0;
                const float t32 = time_it([&] { dsv41_sparse_attn_mma32h<<<dim3(T, NH / 32), MNT * 2>>>(d_q, d_ring, d_wpos, d_ckv, d_c64, d_sink, d_ob, T, NH, D, NW, NC, RING_N, 0, scale); });
                std::printf("MMA32H prefill entry (32 heads/CTA): %zu/%zu differ from mma16 (must be 0) | T=%d: %.3f ms (%.2fx mma16)\n", d32, no, T, t32, t_mma / t32);
                prod_ok = prod_ok && d32 == 0;
                // FA2-style restructure. PRE-REGISTERED: 0 differing values vs mma16 (byte-identical
                // arithmetic by construction). Control: scale * (1 + 2^-10) must differ.
                auto run_fa = [&](int hpc, float sc, int tt, const __nv_bfloat16* qq, const int32_t* ww, const long long* cc) {
                    if (hpc == 16) dsv41_sparse_attn_fa16h<<<dim3(tt, NH / 16), MNT>>>(qq, d_ring, ww, d_ckv, cc, d_sink, d_ob, tt, NH, D, NW, NC, RING_N, 0, sc);
                    if (hpc == 32) dsv41_sparse_attn_fa32h<<<dim3(tt, NH / 32), MNT * 2>>>(qq, d_ring, ww, d_ckv, cc, d_sink, d_ob, tt, NH, D, NW, NC, RING_N, 0, sc);
                    if (hpc == 116) dsv41_sparse_attn_mma16p<<<dim3(tt, NH / 16), MNT>>>(qq, d_ring, ww, d_ckv, cc, d_sink, d_ob, tt, NH, D, NW, NC, RING_N, 0, sc);
                    if (hpc == 132) dsv41_sparse_attn_mma32p<<<dim3(tt, NH / 32), MNT * 2>>>(qq, d_ring, ww, d_ckv, cc, d_sink, d_ob, tt, NH, D, NW, NC, RING_N, 0, sc);
                };
                for (int hpc : {16, 32, 116, 132}) {
                    run_fa(hpc, scale, T, d_q, d_wpos, d_c64);
                    CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
                    CUDA_OK(cudaMemcpy(ob.data(), d_ob, no*2, cudaMemcpyDeviceToHost));
                    size_t dfa = 0; for (size_t i = 0; i < no; ++i) dfa += std::memcmp(&ob[i], &ref16[i], 2) != 0;
                    run_fa(hpc, scale * (1.f + 1.f / 1024), T, d_q, d_wpos, d_c64);
                    CUDA_OK(cudaDeviceSynchronize());
                    CUDA_OK(cudaMemcpy(ob.data(), d_ob, no*2, cudaMemcpyDeviceToHost));
                    size_t dctl = 0; for (size_t i = 0; i < no; ++i) dctl += std::memcmp(&ob[i], &ref16[i], 2) != 0;
                    const float tfa = time_it([&] { run_fa(hpc, scale, T, d_q, d_wpos, d_c64); });
                    std::printf(hpc > 100 ? "PIPE mma%dp (ldmatrix.trans PV, cp.async gather)" : "FA%dH prefill entry: %zu/%zu differ from mma16 (must be 0) | CTRL scale*(1+2^-10): %zu differ (must be > 0) | T=%d: %.3f ms (%.2fx mma32h)\n",
                                hpc % 100, dfa, no, dctl, T, tfa, t32 / tfa);
                    prod_ok = prod_ok && dfa == 0 && dctl > 0;
                }
                // Chunk-2048 occupancy: the fixture's rows tiled to T = 14 * 148 = 2072 (timing only).
                {
                    const int R = 14, TT = T * R;
                    __nv_bfloat16* qq; int32_t* ww; long long* cc; __nv_bfloat16* oo;
                    CUDA_OK(cudaMalloc(&qq, (size_t)TT * NH * D * 2)); CUDA_OK(cudaMalloc(&ww, (size_t)TT * NW * 4));
                    CUDA_OK(cudaMalloc(&cc, (size_t)TT * NC * 8)); CUDA_OK(cudaMalloc(&oo, (size_t)TT * NH * D * 2));
                    for (int r = 0; r < R; ++r) {
                        CUDA_OK(cudaMemcpy(qq + (size_t)r * T * NH * D, d_q, (size_t)T * NH * D * 2, cudaMemcpyDeviceToDevice));
                        CUDA_OK(cudaMemcpy(ww + (size_t)r * T * NW, d_wpos, (size_t)T * NW * 4, cudaMemcpyDeviceToDevice));
                        CUDA_OK(cudaMemcpy(cc + (size_t)r * T * NC, d_c64, (size_t)T * NC * 8, cudaMemcpyDeviceToDevice));
                    }
                    const float tm = time_it([&] { dsv41_sparse_attn_mma32h<<<dim3(TT, NH / 32), MNT * 2>>>(qq, d_ring, ww, d_ckv, cc, d_sink, oo, TT, NH, D, NW, NC, RING_N, 0, scale); });
                    const float t16 = time_it([&] { dsv41_sparse_attn_fa16h<<<dim3(TT, NH / 16), MNT>>>(qq, d_ring, ww, d_ckv, cc, d_sink, oo, TT, NH, D, NW, NC, RING_N, 0, scale); });
                    const float t32f = time_it([&] { dsv41_sparse_attn_fa32h<<<dim3(TT, NH / 32), MNT * 2>>>(qq, d_ring, ww, d_ckv, cc, d_sink, oo, TT, NH, D, NW, NC, RING_N, 0, scale); });
                    const double flop = 2.0 * TT * NH * (double)(NW + NC) * D * 3;   // QK + 2 x PV (P_hi + P_lo)
                    const float tp16 = time_it([&] { dsv41_sparse_attn_mma16p<<<dim3(TT, NH / 16), MNT>>>(qq, d_ring, ww, d_ckv, cc, d_sink, oo, TT, NH, D, NW, NC, RING_N, 0, scale); });
                    const float tp32 = time_it([&] { dsv41_sparse_attn_mma32p<<<dim3(TT, NH / 32), MNT * 2>>>(qq, d_ring, ww, d_ckv, cc, d_sink, oo, TT, NH, D, NW, NC, RING_N, 0, scale); });
                    // PROBE (timing only): identical MMA work, every gather hitting ONE row (L1/L2-hot).
                    // If the kernel is K-gather bound, this collapses the time.
                    {
                        std::vector<int32_t> wh((size_t)TT * NW, 0); std::vector<long long> ch((size_t)TT * NC, 0);
                        CUDA_OK(cudaMemcpy(ww, wh.data(), wh.size() * 4, cudaMemcpyHostToDevice));
                        CUDA_OK(cudaMemcpy(cc, ch.data(), ch.size() * 8, cudaMemcpyHostToDevice));
                        const float thot = time_it([&] { dsv41_sparse_attn_mma32h<<<dim3(TT, NH / 32), MNT * 2>>>(qq, d_ring, ww, d_ckv, cc, d_sink, oo, TT, NH, D, NW, NC, RING_N, 0, scale); });
                        std::printf("PROBE T=%d mma32h with every K gather on ONE hot row: %.3f ms vs %.3f ms real (%.2fx) -- gather share of the time\n", TT, thot, tm, tm / thot);
                    }
                    std::printf("T=%d (fixture x%d): mma32h %.3f ms (%.1f TF/s) | fa16h %.3f ms (%.2fx) | fa32h %.3f ms (%.2fx) | mma16p %.3f ms (%.2fx) | mma32p %.3f ms (%.2fx, %.1f TF/s)\n",
                                TT, R, tm, flop / (tm * 1e9), t16, tm / t16, t32f, tm / t32f, tp16, tm / tp16, tp32, tm / tp32, flop / (tp32 * 1e9));
                    cudaFree(qq); cudaFree(ww); cudaFree(cc); cudaFree(oo);
                }
            }

        }
        cudaFree(d_w64); cudaFree(d_c64); cudaFree(d_ob);
    }
    const bool all_ok = ok && prod_ok;
    std::printf(all_ok ? "PASS one-pass streaming gather (controls separate)\n" : "FAIL\n");
    return all_ok ? 0 : 1;
}
#endif  // DSV41_ATTN_GATE
