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
//
// CANDIDATE MASK = ONE BIT PER 8-COLUMN BLOCK: [T, cand_ld / 256] u32, bit (n >> 3) of row t.
// The selection keeps whole blocks, so a per-column byte mask carried the same information 64x
// larger (2.1 GB at max_seq 1M x chunk 2048; 33 MB as bits).
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
    const uint32_t* __restrict__ CAND,       // [T, cand_ld / 256] block bits, or nullptr
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
    if (CAND != nullptr)
        valid = valid && n < cand_ld && ((CAND[(size_t)t * (cand_ld / 256) + (n >> 8)] >> ((n >> 3) & 31)) & 1u);
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
                  const uint32_t* CAND, float* OUT, int n_keys, int n_pad, long long pos0,
                  int ratio, int cand_ld)
{
    DSV41_INDEX_SCORE_SMEM
    index_score_body<true, HEAD_SUM_ORDER>(Q, IK, W, CAND, OUT, n_keys, n_pad, pos0, ratio, cand_ld, sq, sbuf);
}

// One row's top-k over score columns [0, n): ascending selected columns, then -1.
__device__ void topk_row(const float* row, long long* out, int n, int kmax, int* hist, int* scratch)
{
    for (int i = threadIdx.x; i < TOPK; i += SEL_THREADS) out[i] = -1;
    const Threshold th = radix_threshold(row, n, min(kmax, TOPK), hist, scratch);
    __syncthreads();
    emit_selected(row, n, th, scratch, [&](int c, int slot) { out[slot] = c; });
}

// One row's candidate mask over score columns [0, n). brow holds ceil(n / block_size) floats.
__device__ void candidates_row(const float* row, float* brow, uint32_t* crow, int n, long long lens,
                               int topk_blocks, int block_size, int* hist, int* scratch)
{
    const int nb = (n + block_size - 1) / block_size;
    const long long last = lens >= 1 ? (lens - 1) / block_size : -1;   // floor, as torch's //
    for (int b = threadIdx.x; b < nb; b += SEL_THREADS) {
        float m = -INFINITY;
        for (int i = 0; i < block_size; ++i) {
            const int c = b * block_size + i;
            if (c < n) m = fmaxf(m, row[c]);
        }
        brow[b] = b == last ? INFINITY : m;
    }
    for (int w = threadIdx.x; w < n / 256; w += SEL_THREADS) crow[w] = 0u;
    __syncthreads();
    const Threshold th = radix_threshold(brow, nb, topk_blocks, hist, scratch);
    __syncthreads();
    emit_selected(brow, nb, th, scratch, [&](int b, int) {
        for (int c = b * block_size; c < min(n, (b + 1) * block_size); c += 8)   // its 8-col bits
            atomicOr(&crow[c >> 8], 1u << ((c >> 3) & 31));
    });
}

// One block per row. OUT_IDX: [T, 512] int64, ascending selected columns then -1.
// kmax = min(512, n_c) -- the caller passes it so a short bucket cannot ask for more.
extern "C" __global__ void __launch_bounds__(SEL_THREADS)
dsv41_index_topk(const float* SCORE, long long* OUT_IDX, int n_pad, int kmax)
{
    __shared__ int hist[256];
    __shared__ int scratch[SEL_THREADS / 32];
    topk_row(SCORE + (size_t)blockIdx.x * n_pad, OUT_IDX + (size_t)blockIdx.x * TOPK, n_pad, kmax, hist, scratch);
}

// Layer 20 only. BLOCK_SCORE: [T, n_pad/block_size] fp32 scratch. CAND: [T, n_pad/256] u32 bits.
// block_size must be a multiple of 8 (the bit granularity).
extern "C" __global__ void __launch_bounds__(SEL_THREADS)
dsv41_select_candidates(const float* SCORE, float* BLOCK_SCORE, uint32_t* CAND, int n_pad,
                        long long pos0, int ratio, int topk_blocks, int block_size)
{
    __shared__ int hist[256];
    __shared__ int scratch[SEL_THREADS / 32];
    const int t = blockIdx.x;
    const int nb = (n_pad + block_size - 1) / block_size;
    candidates_row(SCORE + (size_t)t * n_pad, BLOCK_SCORE + (size_t)t * nb, CAND + (size_t)t * (n_pad / 256), n_pad,
                   (pos0 + t + 1) / ratio, topk_blocks, block_size, hist, scratch);
}

#ifdef DSV41_INDEX_GATE
// Negative-control and order-probe entry points. NOT for production.
extern "C" __global__ void __launch_bounds__(SCORE_THREADS)
dsv41_index_score_probe(const __nv_bfloat16* Q, const __nv_bfloat16* IK, const float* W,
                        const uint32_t* CAND, float* OUT, int n_keys, int n_pad, long long pos0,
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

// ------------------------------------------------------------------ compressor + index glue

// Ratio-2 gated combine (engine/model.py `_compressed`, r = 2):
//   latent[g, c] = kv[2g, c] * w0 + kv[2g+1, c] * w1,  (w0, w1) = softmax(sc[2g, c], sc[2g+1, c])
// fp32 in, bf16 out (`latent.to(bfloat16)` before comp_norm). expf, not __expf: torch's softmax.
__device__ __forceinline__ void combine2_at(const float* __restrict__ KV, const float* __restrict__ SC,
                                            __nv_bfloat16* __restrict__ OUT, int n_pairs, int d)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n_pairs * d) return;
    const long long g = i / d, c = i % d;
    const float a = SC[(2 * g) * d + c], b = SC[(2 * g + 1) * d + c];
    const float m = fmaxf(a, b);
    const float ea = expf(a - m), eb = expf(b - m);
    const float s = ea + eb;
    const float lat = KV[(2 * g) * d + c] * (ea / s) + KV[(2 * g + 1) * d + c] * (eb / s);
    OUT[i] = __float2bfloat16_rn(lat);
}

extern "C" __global__ void dsv41_compress_combine2(const float* __restrict__ KV,
                                                  const float* __restrict__ SC,
                                                  __nv_bfloat16* __restrict__ OUT,
                                                  int n_pairs, int d)
{
    combine2_at(KV, SC, OUT, n_pairs, d);
}

// wts = float(bf16 x @ weights_proj^T) * scale   (scale = 128^-0.5 * 32^-0.5)
extern "C" __global__ void dsv41_index_wts(const __nv_bfloat16* __restrict__ RAW, float* __restrict__ OUT,
                                          float scale, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) OUT[i] = __bfloat162float(RAW[i]) * scale;
}

// out[i] = start + i * stride: absolute RoPE positions (tokens: stride 1; compressed rows: r).
extern "C" __global__ void dsv41_iota_i32(int* __restrict__ OUT, int start, int stride, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) OUT[i] = start + i * stride;
}

// C[M, N] = A[M, K] @ B[N, K]^T, TRUE fp32 (no tensor cores), for the ratio-2 compressor.
// Every output is ONE sequential fmaf chain over k = 0..K-1, so a row's result does not depend on
// M, on the tile it lands in, or on how the prompt was chunked -- invariance by construction,
// without the reference's 16-row tiling (which, through cuBLASLt's fp32 path, cost 24 ms per
// call: 32 x 16-row GEMMs re-reading a 10.5 MB weight). 64x64 tile, 256 threads x (4x4) outputs.
// Requires K % 16 == 0.
extern "C" __global__ void __launch_bounds__(256) dsv41_gemm_f32_nt(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K)
{
    __shared__ float as[16][64 + 4];
    __shared__ float bs[16][64 + 4];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    const int tm = (tid / 16) * 4, tn = (tid % 16) * 4;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = tid; i < 64 * 16; i += 256) {
            const int r = i / 16, k = i % 16;
            as[k][r] = (m0 + r < M) ? A[(size_t)(m0 + r) * K + k0 + k] : 0.f;
            bs[k][r] = (n0 + r < N) ? B[(size_t)(n0 + r) * K + k0 + k] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) { a[i] = as[k][tm + i]; b[i] = bs[k][tn + i]; }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (m0 + tm + i < M && n0 + tn + j < N) C[(size_t)(m0 + tm + i) * N + n0 + tn + j] = acc[i][j];
}

// Small-N bf16 projection for the indexer's head weights: out[m, n] = bf16(sum_k x[m,k] w[n,k]),
// fp32 accumulate. One block per row m, warp w owns outputs n = w, w+8, ...; each lane walks
// k = 2*lane + 64*i in a fixed order, then a fixed shuffle tree -> deterministic and row-invariant
// by construction (replaces 32 cuBLASLt calls of 16 rows at N=32). K % 64 == 0.
extern "C" __global__ void __launch_bounds__(256) dsv41_gemm_bf16_smalln(
    const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ OUT,
    int N, int K)
{
    const int m = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(X + (size_t)m * K);
    for (int n = warp; n < N; n += 8) {
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(W + (size_t)n * K);
        float s = 0.f;
        for (int i = lane; i < K / 2; i += 32) {
            const float2 xv = __bfloat1622float2(x2[i]);
            const float2 wv = __bfloat1622float2(w2[i]);
            s = fmaf(xv.x, wv.x, s);
            s = fmaf(xv.y, wv.y, s);
        }
        for (int off = 16; off; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
        if (lane == 0) OUT[(size_t)m * N + n] = __float2bfloat16_rn(s);
    }
}

// Prefill-M form of dsv41_gemm_f32_nt, BIT-IDENTICAL to it by construction: every output is the
// SAME single fmaf(a, b, acc) chain over k = 0..K-1 from 0.f; only the tiling changes. 128x64
// tile, 256 threads x (8 m x 4 n) outputs, float4 smem reads (3 per 32 FMAs vs 8 per 16),
// global loads for tile k+1 held in registers while tile k computes. K % 16 == 0, K % 4 == 0.
namespace dsv41_gemm2 {
constexpr int G_BM = 128, G_BN = 64, G_BK = 16, G_PAD = 4;
}
extern "C" __global__ void __launch_bounds__(256) dsv41_gemm_f32_nt_v2(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K)
{
    using dsv41_gemm2::G_BM; using dsv41_gemm2::G_BN; using dsv41_gemm2::G_BK; using dsv41_gemm2::G_PAD;
    __shared__ __align__(16) float as[2][G_BK][G_BM + G_PAD];
    __shared__ __align__(16) float bs[2][G_BK][G_BN + G_PAD];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * G_BM, n0 = blockIdx.x * G_BN;
    const int tm = (tid / 16) * 8, tn = (tid % 16) * 4;
    // Loaders: A 128 x 16 = 512 float4 (2 per thread), B 64 x 16 = 256 float4 (1 per thread).
    float4 ra[2], rb;
    auto fetch = [&](int k0) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int i = tid + u * 256, r = i / 4, c = (i % 4) * 4;
            ra[u] = m0 + r < M ? *reinterpret_cast<const float4*>(A + (size_t)(m0 + r) * K + k0 + c) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        const int r = tid / 4, c = (tid % 4) * 4;
        rb = n0 + r < N ? *reinterpret_cast<const float4*>(B + (size_t)(n0 + r) * K + k0 + c) : make_float4(0.f, 0.f, 0.f, 0.f);
    };
    auto stash = [&](int buf) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int i = tid + u * 256, r = i / 4, c = (i % 4) * 4;
            as[buf][c][r] = ra[u].x; as[buf][c + 1][r] = ra[u].y; as[buf][c + 2][r] = ra[u].z; as[buf][c + 3][r] = ra[u].w;
        }
        const int r = tid / 4, c = (tid % 4) * 4;
        bs[buf][c][r] = rb.x; bs[buf][c + 1][r] = rb.y; bs[buf][c + 2][r] = rb.z; bs[buf][c + 3][r] = rb.w;
    };
    float acc[8][4] = {};
    fetch(0);
    stash(0);
    __syncthreads();
    const int nk = K / G_BK;
    for (int kt = 0; kt < nk; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < nk) fetch((kt + 1) * G_BK);
#pragma unroll
        for (int k = 0; k < G_BK; ++k) {
            const float4 a0 = *reinterpret_cast<const float4*>(&as[buf][k][tm]);
            const float4 a1 = *reinterpret_cast<const float4*>(&as[buf][k][tm + 4]);
            const float4 b4 = *reinterpret_cast<const float4*>(&bs[buf][k][tn]);
            const float a[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w };
            const float b[4] = { b4.x, b4.y, b4.z, b4.w };
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        if (kt + 1 < nk) stash(buf ^ 1);   // its readers finished before the previous sync
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int m = m0 + tm + i;
        if (m >= M) continue;
        if (n0 + tn + 3 < N) {
            *reinterpret_cast<float4*>(C + (size_t)m * N + n0 + tn) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        } else {
            for (int j = 0; j < 4; ++j)
                if (n0 + tn + j < N) C[(size_t)m * N + n0 + tn + j] = acc[i][j];
        }
    }
}

// Small-M (<= 16) form of dsv41_gemm_f32_nt, BIT-IDENTICAL to it by construction: every output is
// the SAME single fmaf(a, b, acc) chain over k = 0..K-1 from 0.f. dsv41_gemm_f32_nt tiles N by 64,
// so at decode/verify M its grid is N/64 = 8 CTAs streaming a 10.5 MB fp32 weight (~800 us per
// call at T=6, dsv41-decode nsys). This one tiles N by 8 (64 CTAs) and never splits K, which would
// change the order. Double-buffered smem, K % 128 == 0.
namespace dsv41_gemv {
constexpr int NB = 8, KC = 128, MMAX = 16;
}
extern "C" __global__ void __launch_bounds__(256) dsv41_gemv_f32_nt(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K)
{
    using namespace dsv41_gemv;
    __shared__ float ws[2][NB][KC + 1];     // +1: the 8 rows would share a bank
    __shared__ float xs[2][MMAX][KC + 1];
    const int tid = threadIdx.x, n0 = blockIdx.x * NB;
    auto load = [&](int k0, int b) {
        for (int i = tid; i < NB * KC; i += 256) {
            const int r = i / KC, c = i % KC;
            ws[b][r][c] = n0 + r < N ? B[(size_t)(n0 + r) * K + k0 + c] : 0.f;
        }
        for (int i = tid; i < M * KC; i += 256) {
            const int r = i / KC, c = i % KC;
            xs[b][r][c] = A[(size_t)r * K + k0 + c];
        }
    };
    const int n = tid % NB, m = tid / NB;
    const bool act = m < M && n0 + n < N;
    float acc = 0.f;
    load(0, 0);
    __syncthreads();
    for (int k0 = 0, it = 0; k0 < K; k0 += KC, ++it) {
        const int b = it & 1;
        if (k0 + KC < K) load(k0 + KC, b ^ 1);   // the other buffer: its readers passed the last sync
        if (act)
#pragma unroll 8
            for (int k = 0; k < KC; ++k) acc = fmaf(xs[b][m][k], ws[b][n][k], acc);
        __syncthreads();
    }
    if (act) C[(size_t)m * N + n0 + n] = acc;
}

// ------------------------------------------------------------------ shape-static decode core
// ATLAS_DSV41_CORE_STATIC: Decode/Verify passes (T = gridDim.x or the `t` argument, T <= 8)
// read the pass START from device memory (*DSTART), so one captured CUDA graph replays at any
// position. Everything that varied per step is derived here from it:
//   ratio-2 pending parity p = start & 1 (every pass is contiguous from 0, so len % 2 IS it),
//   compressed rows visible n_c = (start + T) / ratio, score width score_width(n_c),
//   first published row j0 = (start - p) / 2 (ratio 2) or start (ratio 1), rows published
//   (T + p) / 2 or T.
// The launch geometry is FIXED: the score is launched over a static width `ld` (the largest
// the sequence can reach); blocks beyond score_width(n_c) exit at once and the selection loops
// stop there, so the work matches the dynamic path. Columns past n_c are -inf and never
// selectable, so the outputs are the dynamic path's, bit for bit (gated, not assumed).

namespace dsv41_index {
constexpr int KEY_BLOCK = 512;   // index.rs KEY_BLOCK: score_width granularity

__device__ __forceinline__ int score_width_of(long long n_c) {
    const long long b = (n_c + KEY_BLOCK - 1) / KEY_BLOCK;
    return (int)(b < 1 ? 1 : b) * KEY_BLOCK;
}
}  // namespace dsv41_index

extern "C" __global__ void __launch_bounds__(SCORE_THREADS)
dsv41_index_score_dev(const __nv_bfloat16* Q, const __nv_bfloat16* IK, const float* W,
                      const uint32_t* CAND, float* OUT, int ld, const int* DSTART, int ratio, int cand_ld)
{
    const long long start = DSTART[0];
    const int T = gridDim.x;
    const long long n_c = (start + T) / ratio;
    if ((int)blockIdx.y * BN >= score_width_of(n_c)) return;
    DSV41_INDEX_SCORE_SMEM
    index_score_body<true, HEAD_SUM_ORDER>(Q, IK, W, CAND, OUT, (int)n_c, ld, start, ratio, cand_ld, sq, sbuf);
}

extern "C" __global__ void __launch_bounds__(SEL_THREADS)
dsv41_index_topk_dev(const float* SCORE, long long* OUT_IDX, int ld, const int* DSTART, int ratio)
{
    __shared__ int hist[256];
    __shared__ int scratch[SEL_THREADS / 32];
    const long long n_c = (DSTART[0] + (long long)gridDim.x) / ratio;
    // kmax = TOPK: the dynamic path's min(TOPK, n_c) is implied, because radix_threshold clamps k
    // to the finite count, and at most n_c columns are finite.
    topk_row(SCORE + (size_t)blockIdx.x * ld, OUT_IDX + (size_t)blockIdx.x * TOPK, score_width_of(n_c), TOPK, hist, scratch);
}

extern "C" __global__ void __launch_bounds__(SEL_THREADS)
dsv41_select_candidates_dev(const float* SCORE, float* BLOCK_SCORE, uint32_t* CAND, int ld,
                            const int* DSTART, int ratio, int topk_blocks, int block_size)
{
    __shared__ int hist[256];
    __shared__ int scratch[SEL_THREADS / 32];
    const int t = blockIdx.x;
    const long long start = DSTART[0];
    const int n = score_width_of((start + (long long)gridDim.x) / ratio);
    const int nb_ld = (ld + block_size - 1) / block_size;
    candidates_row(SCORE + (size_t)t * ld, BLOCK_SCORE + (size_t)t * nb_ld, CAND + (size_t)t * (ld / 256), n,
                   (start + t + 1) / ratio, topk_blocks, block_size, hist, scratch);
}

// out[i] = base + i * stride, base = start (tokens, ratio-1 rows) or start & ~1 (= j0 * 2, the
// first ratio-2 row this pass publishes).
extern "C" __global__ void dsv41_iota_dev(int* __restrict__ OUT, const int* DSTART, int even_floor, int stride, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = even_floor ? (DSTART[0] & ~1) : DSTART[0];
    if (i < n) OUT[i] = base + i * stride;
}

// Ratio-2 compressor, static form. The projections of the pass's T rows always sit at slots
// 1..T of KV/SC; a pending row (p = 1) is copied to slot 0 and the pairs start at slot 1 - p.
extern "C" __global__ void dsv41_comp2_pending_in(const float* __restrict__ PENDING, float* __restrict__ KV,
                                                  float* __restrict__ SC, int d, const int* DSTART)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((DSTART[0] & 1) == 0 || i >= d) return;
    KV[i] = PENDING[i];
    SC[i] = PENDING[d + i];
}

extern "C" __global__ void dsv41_compress_combine2_dev(const float* __restrict__ KV,
                                                      const float* __restrict__ SC,
                                                      __nv_bfloat16* __restrict__ OUT,
                                                      int n_pairs, int d, const int* DSTART)
{
    const int off = 1 - (DSTART[0] & 1);
    combine2_at(KV + (size_t)off * d, SC + (size_t)off * d, OUT, n_pairs, d);
}

// The row left unpaired when T + p is odd is always the pass's LAST row, slot T.
extern "C" __global__ void dsv41_comp2_pending_out(const float* __restrict__ KV, const float* __restrict__ SC,
                                                   float* __restrict__ PENDING, int d, int t, const int* DSTART)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (((t + (DSTART[0] & 1)) & 1) == 0 || i >= d) return;
    PENDING[i] = KV[(size_t)t * d + i];
    PENDING[d + i] = SC[(size_t)t * d + i];
}

// Publish the pass's compressed rows: SRC rows 0..nj-1 -> DST rows j0..j0+nj-1 (16-byte units).
// grid.x = the most rows the pass can publish; blocks past nj do nothing.
extern "C" __global__ void dsv41_publish_rows(const uint4* __restrict__ SRC, uint4* __restrict__ DST,
                                              int row_u4, int ratio, int t, const int* DSTART)
{
    const int start = DSTART[0];
    const int p = ratio == 2 ? (start & 1) : 0;
    const int j0 = ratio == 2 ? (start - p) / 2 : start;
    const int nj = ratio == 2 ? (t + p) / 2 : t;
    const int r = blockIdx.x;
    if (r >= nj) return;
    for (int i = threadIdx.x; i < row_u4; i += blockDim.x)
        DST[(size_t)(j0 + r) * row_u4 + i] = SRC[(size_t)r * row_u4 + i];
}
