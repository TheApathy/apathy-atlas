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
template <bool CTRL_ORDER, bool CTRL_GATHER, typename IdxT, typename OutT>
__device__ void sparse_attn_body(
    const __nv_bfloat16* __restrict__ Q,     // [T, NH, D]
    const __nv_bfloat16* __restrict__ RING,  // [RING_N, D]
    const IdxT*          __restrict__ WPOS,  // [T, NW]  absolute positions, -1 = none
    const __nv_bfloat16* __restrict__ CKV,   // [n_c, D]
    const IdxT*          __restrict__ CIDX,  // [T, NC]  compressed rows, -1 = none
    const float*         __restrict__ SINK,  // [NH]
    OutT*                __restrict__ O,     // [T, NH, D]
    int T, int NH, int D, int NW, int NC, int RING_N, int win_lo, float scale)
{
    const int t   = blockIdx.x;
    const int hb  = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int nwarp = NT / 32;

    __shared__ __nv_bfloat16 qs[HB][DMAX];
    __shared__ __nv_bfloat16 ks[BK][DMAX];
    __shared__ float red[HB][NT / 32];
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
        const int n   = seg == 0 ? NW : NC;
        if (seg == 1 && CIDX == nullptr) continue;

        for (int kb = 0; kb < n; kb += BK) {
            // ---- gather BK rows straight out of the ring / compressed cache
            if (tid < BK) {
                const int c = kb + tid;
                int row = -1;
                if (c < n) {
                    if (seg == 0) {
                        const long long p = WPOS[(size_t)t * NW + c];
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

            // ---- scores: sc[h][kk] = dot(q_h, k_kk) * scale, fp32 accumulate
            for (int kk = 0; kk < BK; ++kk) {
                float part[HB];
#pragma unroll
                for (int h = 0; h < HB; ++h) {
                    float s = 0.f;
                    for (int i = 0; i < DPT; ++i) {
                        const int d = tid + i * NT;
                        s = fmaf(__bfloat162float(qs[h][d]), __bfloat162float(ks[kk][d]), s);
                    }
                    part[h] = s;
                }
#pragma unroll
                for (int h = 0; h < HB; ++h) {
                    float s = part[h];
                    for (int off = 16; off; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
                    if (lane == 0) red[h][warp] = s;
                }
                __syncthreads();
                if (tid < HB) {
                    float s = 0.f;
                    for (int w = 0; w < nwarp; ++w) s += red[tid][w];
                    sc[tid][kk] = s * scale;
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
        cudaFree(d_w64); cudaFree(d_c64); cudaFree(d_ob);
    }
    const bool all_ok = ok && prod_ok;
    std::printf(all_ok ? "PASS one-pass streaming gather (controls separate)\n" : "FAIL\n");
    return all_ok ? 0 : 1;
}
#endif  // DSV41_ATTN_GATE
