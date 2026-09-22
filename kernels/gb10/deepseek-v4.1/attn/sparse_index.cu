// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1 sparse index selection: the INDEXER (SPEC.md sec 4).
//
// Three kernels, run on the 8 index-source layers {2,8,14,20,24,28,32,36}:
//
//   dsv41_index_score        score[t, n] = sum_h relu(bf16(q[t,h,:] . ik[n,:])) * w[t,h]
//                            -inf where n >= compress_lens[t] or (cand given and !cand[t,n])
//   dsv41_select_candidates  layer 20 only: amax over blocks of 8 columns, force-keep the
//                            block holding compress_lens-1, keep the top 2048 finite blocks
//   dsv41_index_topk         the top min(512, n_c) finite columns per row, ASCENDING, padded
//                            to EXACTLY 512 with -1
//
// NUMERICS THAT DECIDE SELECTION
// * The dot is fp32-accumulated on tensor cores (bf16 x bf16 -> fp32, WMMA 16x16x16, k in
//   ascending 16-steps from a zero accumulator) and then ROUNDED TO BF16 BEFORE THE RELU.
//   The torch path's einsum returned bf16 and tools/indexer_kernel.py reproduces that with
//   `s.to(bf16).to(f32)`. Skipping the rounding changes which rows are selected near ties.
//   The MMA form is deliberate: it is the instruction the production Triton kernel's
//   tl.dot lowers to, so the fp32 dot (and hence which side of a bf16 rounding boundary it
//   lands on) can match it, where a scalar FMA loop could not.
// * The head sum is fp32 in a FIXED order (HEAD_SUM below), so a score depends on its own
//   query only: row-count- and chunk-invariant.
//
// SELECTION CONTRACT
// * Output width is ALWAYS 512 (index_topk), -1 padded. Not a buffer size: a varying N flips
//   the attention GEMM's kernel choice downstream (SPEC.md sec 4).
// * The reference is `topk(score, min(512, n_c)).sort()` then `where(idx < compress_lens,
//   idx, -1)`. This kernel instead selects only FINITE columns. The two agree whenever a row
//   has fewer than k finite columns only because of compress_lens, which holds for this
//   checkpoint: candidates keep 2048 blocks x 8 = 16384 >= 512 columns, and every visible
//   block is kept while fewer than 2048 are visible. (If a config ever had
//   candidate_topk_blocks * candidate_block_size < index_topk, the reference would keep
//   arbitrary -inf non-candidate columns below compress_lens; this kernel would emit -1.)
// * Exact ties at the k-th value resolve to the LOWEST column. torch.topk's tie order is
//   unspecified, so this is the one place a port may legitimately differ; the gate reports
//   boundary ties rather than assuming there are none.

#include <cstdint>
#include <cuda_bf16.h>
#include <mma.h>

namespace dsv41_index {

constexpr int H = 32;          // index_n_heads
constexpr int DH = 128;        // index_head_dim
constexpr int BN = 128;        // keys per score block
constexpr int SCORE_THREADS = 128;
constexpr int TOPK = 512;      // index_topk
constexpr int SEL_THREADS = 512;

// Head-sum order. 0 = sequential h = 0..31. 1 = adjacent pairwise tree. 2 = butterfly
// (h + h^16, then ^8, ^4, ^2, ^1). Production is HEAD_SUM_ORDER; the others exist so the
// gate can MEASURE which one the reference used instead of asserting it.
// MEASURED (index_gate.log, runD = the production Triton indexer): pairwise is bit-exact on
// 99.05-99.65% of scores, sequential 95.8-98.2%, butterfly 97.7-99.1%. None is 100%: the
// residue is <= 4.8e-07 abs (1-2 ulp of the head sum). The DOT is exact -- the bf16-round
// flips a scalar/fp64 dot produced (worst 8.7e-03) are gone under WMMA.
#ifndef HEAD_SUM_ORDER
#define HEAD_SUM_ORDER 1
#endif

__device__ __forceinline__ float round_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

template <int ORDER>
__device__ __forceinline__ float head_sum(float (&v)[H]) {
    if (ORDER == 0) {
        float s = 0.f;
#pragma unroll
        for (int h = 0; h < H; ++h) s += v[h];
        return s;
    } else if (ORDER == 1) {
#pragma unroll
        for (int w = 1; w < H; w <<= 1)
#pragma unroll
            for (int h = 0; h < H; h += 2 * w) v[h] = v[h] + v[h + w];
        return v[0];
    } else {
#pragma unroll
        for (int w = H / 2; w >= 1; w >>= 1)
#pragma unroll
            for (int h = 0; h < w; ++h) v[h] = v[h] + v[h + w];
        return v[0];
    }
}

// Shared memory is declared by the __global__ entry, once, so the probe's several
// instantiations share one allocation. 40 KB: under the 48 KB static limit.
#define DSV41_INDEX_SCORE_SMEM                                                   \
    __shared__ __align__(32) __nv_bfloat16 sq[H * DH];                           \
    __shared__ __align__(32) unsigned char sbuf[BN * DH * 2];

// One block = one token x BN keys. 4 warps; warp w owns key columns [32w, 32w+32).
template <bool ROUND, int ORDER>
__device__ void index_score_body(
    const __nv_bfloat16* __restrict__ Q,     // [T, H, DH]
    const __nv_bfloat16* __restrict__ IK,    // [n_keys, DH]
    const float* __restrict__ W,             // [T, H]
    const uint8_t* __restrict__ CAND,        // [T, cand_ld] or nullptr
    float* __restrict__ OUT,                 // [T, n_pad]
    int n_keys, int n_pad, long long pos0, int ratio, int cand_ld,
    __nv_bfloat16* sq,                       // smem [H * DH]            8 KB
    unsigned char* sbuf)                     // smem [BN * DH * 2]      32 KB: keys, then scores
{
    using namespace nvcuda;
    __nv_bfloat16* sk = reinterpret_cast<__nv_bfloat16*>(sbuf);
    float* ss = reinterpret_cast<float*>(sbuf);                   // [H][BN] fp32 = 16 KB

    const int t = blockIdx.x;
    const int n0 = blockIdx.y * BN;
    const int tid = threadIdx.x;
    const int warp = tid / 32;

    const uint4* qsrc = reinterpret_cast<const uint4*>(Q + (size_t)t * H * DH);
    uint4* qdst = reinterpret_cast<uint4*>(sq);
    for (int i = tid; i < H * DH / 8; i += SCORE_THREADS) qdst[i] = qsrc[i];
    uint4* kdst = reinterpret_cast<uint4*>(sk);
    for (int i = tid; i < BN * DH / 8; i += SCORE_THREADS) {
        const int n = n0 + i / (DH / 8);
        kdst[i] = n < n_keys ? reinterpret_cast<const uint4*>(IK + (size_t)n * DH)[i % (DH / 8)]
                             : make_uint4(0, 0, 0, 0);
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < 2; ++n) wmma::fill_fragment(acc[m][n], 0.f);
#pragma unroll
    for (int k = 0; k < DH; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[2];
#pragma unroll
        for (int m = 0; m < 2; ++m) wmma::load_matrix_sync(a[m], sq + (m * 16) * DH + k, DH);
#pragma unroll
        for (int n = 0; n < 2; ++n) wmma::load_matrix_sync(b[n], sk + (warp * 32 + n * 16) * DH + k, DH);
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int n = 0; n < 2; ++n) wmma::mma_sync(acc[m][n], a[m], b[n], acc[m][n]);
    }
    __syncthreads();   // keys are dead; reuse the buffer for the [H][BN] dot products
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < 2; ++n)
            wmma::store_matrix_sync(ss + (m * 16) * BN + warp * 32 + n * 16, acc[m][n], BN,
                                    wmma::mem_row_major);
    __syncthreads();

    const int n = n0 + tid;
    if (n >= n_pad) return;
    const long long lens = (pos0 + t + 1) / ratio;       // compress_lens, ABSOLUTE position
    bool valid = n < lens;
    if (CAND != nullptr) valid = valid && n < cand_ld && CAND[(size_t)t * cand_ld + n] != 0;
    float v[H];
#pragma unroll
    for (int h = 0; h < H; ++h) {
        float s = ss[h * BN + tid];
        if (ROUND) s = round_bf16(s);
        v[h] = fmaxf(s, 0.f) * W[(size_t)t * H + h];
    }
    const float score = head_sum<ORDER>(v);
    OUT[(size_t)t * n_pad + n] = valid ? score : -INFINITY;
}

// ------------------------------------------------------------------ block radix select
// Orderable key: larger float -> larger key. -inf and NaN are "not selectable".
__device__ __forceinline__ uint32_t key_of(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ bool selectable(float f) { return f != -INFINITY && !isnan(f); }

// Inclusive block scan over SEL_THREADS ints; returns inclusive prefix, *total = block sum.
__device__ int block_scan(int x, int* scratch, int* total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) scratch[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = lane < SEL_THREADS / 32 ? scratch[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += y;
        }
        if (lane < SEL_THREADS / 32) scratch[lane] = w;
    }
    __syncthreads();
    const int r = x + (warp > 0 ? scratch[warp - 1] : 0);
    *total = scratch[SEL_THREADS / 32 - 1];
    __syncthreads();
    return r;
}

// Select the k largest selectable values of row[0..n). Returns the threshold key and how many
// entries EQUAL to it are taken (lowest columns first). k is clamped to the selectable count.
struct Threshold { uint32_t key; int take_eq; int k; };

__device__ Threshold radix_threshold(const float* row, int n, int kmax, int* hist, int* scratch) {
    int cnt = 0;
    for (int i = threadIdx.x; i < n; i += SEL_THREADS) cnt += selectable(row[i]);
    int total;
    block_scan(cnt, scratch, &total);
    const int k = min(kmax, total);
    __shared__ uint32_t s_prefix;
    __shared__ int s_remaining;
    if (threadIdx.x == 0) { s_prefix = 0; s_remaining = k; }
    uint32_t mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += SEL_THREADS) hist[i] = 0;
        __syncthreads();
        const uint32_t prefix = s_prefix;
        if (k > 0)
            for (int i = threadIdx.x; i < n; i += SEL_THREADS) {
                const float f = row[i];
                if (!selectable(f)) continue;
                const uint32_t key = key_of(f);
                if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255], 1);
            }
        __syncthreads();
        if (threadIdx.x == 0 && k > 0) {
            int rem = s_remaining, d = 255;
            for (; d > 0; --d) {
                if (hist[d] >= rem) break;
                rem -= hist[d];
            }
            s_remaining = rem;
            s_prefix = prefix | ((uint32_t)d << shift);
        }
        mask |= 255u << shift;
        __syncthreads();
    }
    return Threshold{ s_prefix, s_remaining, k };
}

// Walk the row in ASCENDING column order and call emit(col, slot) for each selected column.
template <typename Emit>
__device__ void emit_selected(const float* row, int n, Threshold th, int* scratch, Emit emit) {
    if (th.k == 0) return;
    int base_out = 0, base_eq = 0;
    for (int c0 = 0; c0 < n; c0 += SEL_THREADS) {
        const int c = c0 + threadIdx.x;
        const float f = c < n ? row[c] : -INFINITY;
        const bool ok = selectable(f);
        const uint32_t key = ok ? key_of(f) : 0;
        const bool gt = ok && key > th.key;
        const bool eq = ok && key == th.key;
        int eq_total;
        const int eq_incl = block_scan(eq ? 1 : 0, scratch, &eq_total);
        const bool take = gt || (eq && base_eq + eq_incl - 1 < th.take_eq);
        int out_total;
        const int out_incl = block_scan(take ? 1 : 0, scratch, &out_total);
        if (take) emit(c, base_out + out_incl - 1);
        base_out += out_total;
        base_eq += eq_total;
    }
}

}  // namespace dsv41_index

using namespace dsv41_index;

extern "C" __global__ void __launch_bounds__(SCORE_THREADS)
dsv41_index_score(const __nv_bfloat16* Q, const __nv_bfloat16* IK, const float* W,
                  const uint8_t* CAND, float* OUT, int n_keys, int n_pad, long long pos0,
                  int ratio, int cand_ld)
{
    DSV41_INDEX_SCORE_SMEM
    index_score_body<true, HEAD_SUM_ORDER>(Q, IK, W, CAND, OUT, n_keys, n_pad, pos0, ratio, cand_ld, sq, sbuf);
}

// One block per row. OUT_IDX: [T, 512] int64, ascending selected columns then -1.
// kmax = min(512, n_c) -- the caller passes it so a short bucket cannot ask for more.
extern "C" __global__ void __launch_bounds__(SEL_THREADS)
dsv41_index_topk(const float* SCORE, long long* OUT_IDX, int n_pad, int kmax)
{
    __shared__ int hist[256];
    __shared__ int scratch[SEL_THREADS / 32];
    const float* row = SCORE + (size_t)blockIdx.x * n_pad;
    long long* out = OUT_IDX + (size_t)blockIdx.x * TOPK;
    for (int i = threadIdx.x; i < TOPK; i += SEL_THREADS) out[i] = -1;
    const Threshold th = radix_threshold(row, n_pad, min(kmax, TOPK), hist, scratch);
    __syncthreads();
    emit_selected(row, n_pad, th, scratch, [&](int c, int slot) { out[slot] = c; });
}

// Layer 20 only. BLOCK_SCORE: [T, n_pad/8] fp32 scratch. CAND: [T, n_pad] uint8.
extern "C" __global__ void __launch_bounds__(SEL_THREADS)
dsv41_select_candidates(const float* SCORE, float* BLOCK_SCORE, uint8_t* CAND, int n_pad,
                        long long pos0, int ratio, int topk_blocks, int block_size)
{
    __shared__ int hist[256];
    __shared__ int scratch[SEL_THREADS / 32];
    const int t = blockIdx.x;
    const int nb = (n_pad + block_size - 1) / block_size;
    const float* row = SCORE + (size_t)t * n_pad;
    float* brow = BLOCK_SCORE + (size_t)t * nb;
    uint8_t* crow = CAND + (size_t)t * n_pad;
    const long long lens = (pos0 + t + 1) / ratio;
    const long long last = lens >= 1 ? (lens - 1) / block_size : -1;   // floor, as torch's //
    for (int b = threadIdx.x; b < nb; b += SEL_THREADS) {
        float m = -INFINITY;
        for (int i = 0; i < block_size; ++i) {
            const int c = b * block_size + i;
            if (c < n_pad) m = fmaxf(m, row[c]);
        }
        brow[b] = b == last ? INFINITY : m;
    }
    for (int c = threadIdx.x; c < n_pad; c += SEL_THREADS) crow[c] = 0;
    __syncthreads();
    const Threshold th = radix_threshold(brow, nb, topk_blocks, hist, scratch);
    __syncthreads();
    emit_selected(brow, nb, th, scratch, [&](int b, int) {
        for (int i = 0; i < block_size; ++i) {
            const int c = b * block_size + i;
            if (c < n_pad) crow[c] = 1;
        }
    });
}

#ifdef DSV41_INDEX_GATE
// Negative-control and order-probe entry points. NOT for production.
extern "C" __global__ void __launch_bounds__(SCORE_THREADS)
dsv41_index_score_probe(const __nv_bfloat16* Q, const __nv_bfloat16* IK, const float* W,
                        const uint8_t* CAND, float* OUT, int n_keys, int n_pad, long long pos0,
                        int ratio, int cand_ld, int variant)
{
    DSV41_INDEX_SCORE_SMEM
    switch (variant) {
    case 0: index_score_body<true, 0>(Q, IK, W, CAND, OUT, n_keys, n_pad, pos0, ratio, cand_ld, sq, sbuf); break;
    case 1: index_score_body<true, 1>(Q, IK, W, CAND, OUT, n_keys, n_pad, pos0, ratio, cand_ld, sq, sbuf); break;
    case 2: index_score_body<true, 2>(Q, IK, W, CAND, OUT, n_keys, n_pad, pos0, ratio, cand_ld, sq, sbuf); break;
    default: index_score_body<false, HEAD_SUM_ORDER>(Q, IK, W, CAND, OUT, n_keys, n_pad, pos0, ratio, cand_ld, sq, sbuf); break;
    }
}
#endif
