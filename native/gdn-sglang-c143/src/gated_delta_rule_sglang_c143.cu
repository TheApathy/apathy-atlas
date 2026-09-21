// SPDX-License-Identifier: AGPL-3.0-only

// Isolated default-unrouted SM121 shadow. The CUDA decomposition is retained from
// Atlas clean-upstream commit 6d79e14cc49bbc0c21a9b68901f5d78a522a2caa
// (`gated_delta_rule_fla.cu` SHA256 454590abfb614ba9a87601626a8ab6085041db8673882d8de2fa8e5d2508aa6e).
// Its chunkwise equations were re-audited against pinned SGLang c14312a66420b75ca9a11bf1817c4db1fa26b097.
// This is approximate FLA arithmetic and is never a bit-exact replacement for Atlas WY32.


// Atlas GDN prefill — FLA-style MULTI-KERNEL decomposition (the path to beat
// vLLM). The single-fused chunk64 kernel was boxed in (serial-per-chunk in one
// CTA → 0.38-0.69x). FLA's speed comes from splitting into passes where the BIG
// matmuls are PARALLEL over all chunks (full 48-SM occupancy) and only a small
// state-passing is serial:
//   recompute_w_u (THIS, parallel over chunks×heads, H-independent):
//       solve (I+L)U = βV, (I+L)W = β·exp(gc)·K  via forward-substitution.
//       L[i][l] = β_i·exp(gc_i-gc_l)·<k_l,k_i> (l<i), Gram built on tensor cores.
//   chunk_delta_h (serial over chunks): S_{c+1}=exp(gc_last)S_c + K̃ᵀU - K̃ᵀ(WS).
//   chunk_fwd_o (parallel over chunks): O = Q̃·S + tril(Q̃·Kᵀ)·U.
// Math validated == recurrent SSOT by the CPU oracle
// (crates/spark-runtime/tests/gdn_chunk64_oracle.rs :: fla_decomposed_ref).
//
// This isolated c143 port accepts FP32 log-decay directly, matching the pinned
// SGLang API. The older Atlas FLA source accepted linear decay and applied
// logf here; removing that round-trip is the only equation-level source delta.
// GB10 sm_121: mma.sync.m16n8k16 BF16 (no wgmma/TMA, ldmatrix broken).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>   // nvcuda::wmma fragment types (tc_vblock variant; mma_gram itself
                   // stays raw mma.sync PTX — ldmatrix is broken on sm_121).
#include <cstdint>
#include <cstdio>
#include <limits>

#ifndef ATLAS_GDN_C143_SOURCE_SHA256
#error "build must inject ATLAS_GDN_C143_SOURCE_SHA256"
#endif

#define K_DIM 128
#define V_DIM 128
#define CHUNK 64
// Floor for the linear gate before log-space cumsum. Deep-layer gates can
// underflow to exactly 0.0 (or tiny negatives), and log(0)=-inf → exp(gc_i-gc_l)
// = exp(-inf - -inf) = NaN. The per-token recurrence (h=g·h) tolerates g≈0; the
// chunked log-space form does not. 1e-30 ⇒ log≈-69 (effectively full decay) and
// is a no-op for any normal gate. (GATE-B used g∈[0.8,0.999], never hitting this.)
#define GATE_FLOOR 1e-30f

// Per-stream prefill geometry. VARLEN (ragged co-dispatch batch) reads
// cu_seqlens[b]/cu_chunks[b]; uniform/single-stream (is_varlen=0) reduces to
// b*seq_len / b*num_chunks (byte-identical to the pre-varlen path). Requires
// b, seq_len, num_chunks, cu_seqlens, cu_chunks, is_varlen in scope. `tokoff` is
// the token offset into the [Σ seqlen_b] packed inputs; `choff` the chunk offset
// into the [Σ nchunks_b] scratch; in varlen `num_chunks` (grid x) = MAX nchunks.
struct GdnGeom { unsigned int seqlen, nchunks, choff; unsigned long long tokoff; };
#define GDN_GEOM(g)                                                            \
    GdnGeom g;                                                                 \
    (void)cu_chunks;                                                          \
    if (is_varlen) {                                                           \
        unsigned int _s0 = (unsigned int)cu_seqlens[b];                       \
        g.seqlen  = (unsigned int)cu_seqlens[b + 1] - _s0;                    \
        g.tokoff  = (unsigned long long)_s0;                                  \
        unsigned int _co = 0; /* choff = Σ_{i<b} ceil(len_i/64), in-kernel */ \
        for (unsigned int _i = 0; _i < b; _i++)                               \
            _co += ((unsigned int)(cu_seqlens[_i + 1] - cu_seqlens[_i])       \
                    + CHUNK - 1) / CHUNK;                                      \
        g.choff   = _co;                                                       \
        g.nchunks = (g.seqlen + CHUNK - 1) / CHUNK;                           \
    } else {                                                                  \
        g.seqlen  = seq_len;                                                   \
        g.tokoff  = (unsigned long long)b * seq_len;                          \
        g.choff   = b * num_chunks;                                            \
        g.nchunks = num_chunks;                                                \
    }

// GB10 sm_121 has cp.async.cg (NO TMA). 16-byte async global→shared copy + group
// commit/wait — used to double-buffer the per-chunk W/U/K loads in chunk_delta_h
// so the serial state spine overlaps the next chunk's load with the current
// chunk's compute (it was global-load-LATENCY bound at 4 warps, not FLOP bound).
__device__ __forceinline__ void cp_async16(void* dst_smem, const void* src_gmem) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src_gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// C[m][n] = Σ_k A[m][k]·B[n][k], M=64, K=K_DIM, N=NTC*8. A/B row-major bf16 smem;
// 128 threads = 4 warps (16 M-rows each). NSTRIDE = C row-stride. (SSOT helper.)
template <int NTC, int NSTRIDE, bool OutBf16>
__device__ __forceinline__ void mma_gram(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, void* __restrict__ C
) {
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31;
    const unsigned grp = lane >> 2;
    const unsigned q = lane & 3;
    const unsigned warp_m = warp * 16;
    const unsigned short* sA = (const unsigned short*)A;
    const unsigned short* sB = (const unsigned short*)B;
    float acc[NTC][4];
    #pragma unroll
    for (int nt = 0; nt < NTC; nt++) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f; }
    #pragma unroll
    for (unsigned ks = 0; ks < K_DIM; ks += 16) {
        unsigned fr0 = warp_m + grp, fr1 = fr0 + 8;
        unsigned fc0 = ks + q * 2, fc1 = fc0 + 8;
        unsigned a0 = *(const unsigned*)&sA[fr0 * K_DIM + fc0];
        unsigned a1 = *(const unsigned*)&sA[fr1 * K_DIM + fc0];
        unsigned a2 = *(const unsigned*)&sA[fr0 * K_DIM + fc1];
        unsigned a3 = *(const unsigned*)&sA[fr1 * K_DIM + fc1];
        #pragma unroll
        for (int nt = 0; nt < NTC; nt++) {
            unsigned nc = nt * 8 + grp;
            unsigned k0 = ks + q * 2, k1 = k0 + 8;
            unsigned b0 = ((unsigned)sB[nc * K_DIM + k0 + 1] << 16) | (unsigned)sB[nc * K_DIM + k0];
            unsigned b1 = ((unsigned)sB[nc * K_DIM + k1 + 1] << 16) | (unsigned)sB[nc * K_DIM + k1];
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3]));
        }
    }
    #pragma unroll
    for (int nt = 0; nt < NTC; nt++) {
        unsigned n0 = nt * 8 + q * 2, n1 = n0 + 1;
        unsigned m0 = warp_m + grp, m1 = m0 + 8;
        if (OutBf16) {
            __nv_bfloat16* Cb = (__nv_bfloat16*)C;
            Cb[m0 * NSTRIDE + n0] = __float2bfloat16(acc[nt][0]);
            Cb[m0 * NSTRIDE + n1] = __float2bfloat16(acc[nt][1]);
            Cb[m1 * NSTRIDE + n0] = __float2bfloat16(acc[nt][2]);
            Cb[m1 * NSTRIDE + n1] = __float2bfloat16(acc[nt][3]);
        } else {
            float* Cf = (float*)C;
            Cf[m0 * NSTRIDE + n0] = acc[nt][0];
            Cf[m0 * NSTRIDE + n1] = acc[nt][1];
            Cf[m1 * NSTRIDE + n0] = acc[nt][2];
            Cf[m1 * NSTRIDE + n1] = acc[nt][3];
        }
    }
}

// ── KERNEL 1: recompute_w_u ──────────────────────────────────────────────
// Grid: (NT, num_v_heads, batch)  Block: (128,1,1).  One CTA per (chunk, head).
// Outputs (f32, layout [(b*NT+c)*nv+vh][CHUNK][·]):
//   U_out: [.. ][CHUNK][V_DIM]   = T·(βV)
//   W_out: [.. ][CHUNK][K_DIM]   = T·(β·exp(gc)·K)
// where T=(I+L)⁻¹ applied by forward-substitution (parallel over the V/K cols).
// smem: sk_bf(16K) + kk(16K f32) + L(16K f32) + gc(256) ≈ 48.25KB.
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_recompute_wu(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ W_out,   // bf16 — per-chunk intermediate, fed to TC matmuls in #2/#3
    __nv_bfloat16* __restrict__ U_out,
    float* __restrict__ gc_out,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,   // NT = ceil(seq_len/CHUNK)
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gate_stride,
    unsigned int beta_stride,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    const unsigned int c = blockIdx.x;     // chunk index (grid x = MAX num_chunks)
    const unsigned int vh = blockIdx.y;
    const unsigned int b = blockIdx.z;
    if (vh >= num_v_heads || b >= batch_size) return;
    GDN_GEOM(g);
    if (c >= g.nchunks) return;            // per-stream chunk bound

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const unsigned int cs = c * CHUNK;
    const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
    const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

    // Per-stream input offset (cu_seqlens[b] in varlen, else b*seq_len; b=0 → +0).
    key   += g.tokoff * qk_stride;
    value += g.tokoff * v_stride;
    gate  += g.tokoff * gate_stride;
    beta  += g.tokoff * beta_stride;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sk = (__nv_bfloat16*)smem_raw;       // [CHUNK*K_DIM] bf16
    float* kk = (float*)(sk + CHUNK * K_DIM);           // [CHUNK*CHUNK] f32 Gram
    float* L = kk + CHUNK * CHUNK;                      // [CHUNK*CHUNK] f32 decay-weighted strict-lower
    float* gc = L + CHUNK * CHUNK;                      // [CHUNK]

    for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += 128) {
        unsigned int i = idx / k_dim, j = idx % k_dim;
        sk[i * K_DIM + j] = (i < ce)
            ? key[(unsigned long long)(cs + i) * qk_stride + kh * k_dim + j]
            : __float2bfloat16(0.0f);
    }
    if (tid == 0) {
        float acc = 0.0f;
        for (unsigned int i = 0; i < ce; i++) {
            acc += gate[(unsigned long long)(cs + i) * gate_stride + vh];
            gc[i] = acc;
            gc_out[base * CHUNK + i] = acc;
        }
    }
    __syncthreads();

    mma_gram<8, CHUNK, false>(sk, sk, kk);   // kk[l][i] = <k_l,k_i>
    __syncthreads();

    // L[i][l] = β_i·exp(gc_i-gc_l)·<k_l,k_i>  for l<i ; 0 otherwise.  (kk symmetric)
    for (unsigned int p = tid; p < CHUNK * CHUNK; p += 128) {
        unsigned int i = p / CHUNK, l = p % CHUNK;
        if (i < ce && l < i) {
            float bi = beta[(unsigned long long)(cs + i) * beta_stride + vh];
            L[i * CHUNK + l] = bi * expf(gc[i] - gc[l]) * kk[l * CHUNK + i];
        } else {
            L[i * CHUNK + l] = 0.0f;
        }
    }
    __syncthreads();

    // Pass 1: U[:,v] forward-sub (one thread per v-element).  U_i = β_i·V_i - Σ_{l<i} L[i][l]·U_l
    if (tid < v_dim) {
        float u[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float bi = beta[(unsigned long long)(cs + i) * beta_stride + vh];
            float ui = bi * (float)value[(unsigned long long)(cs + i) * v_stride + vh * v_dim + tid];
            for (unsigned int l = 0; l < i; l++) ui -= L[i * CHUNK + l] * u[l];
            u[i] = ui;
            U_out[base * CHUNK * V_DIM + i * v_dim + tid] = __float2bfloat16(ui);
        }
    }
    // Pass 2: W[:,k] forward-sub (one thread per k-element).  W_i = β_i·exp(gc_i)·K_i - Σ_{l<i} L[i][l]·W_l
    if (tid < k_dim) {
        float w[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float bi = beta[(unsigned long long)(cs + i) * beta_stride + vh];
            float wi = bi * expf(gc[i]) * (float)sk[i * K_DIM + tid];
            for (unsigned int l = 0; l < i; l++) wi -= L[i * CHUNK + l] * w[l];
            w[i] = wi;
            W_out[base * CHUNK * K_DIM + i * k_dim + tid] = __float2bfloat16(wi);
        }
    }
}

// chunk_delta_h double-buffer: per-buffer smem holds {W,K,U} bf16 for one chunk.
#define CDH_BUFSZ (CHUNK * (2 * K_DIM + V_DIM))   // 24576 bf16 = 48KB

// Prefetch chunk c's W/U/K into buffer slot p via cp.async, and load its gc
// plus decays on tid 0. K is loaded per-row and bounded to i<ce
// (rows cs+i ≥ seq_len would be OOB); W/U load the full block (in-bounds, and the
// i≥ce tail is never read since the compute loops are bounded by ce).
__device__ __forceinline__ void cdh_prefetch(
    __nv_bfloat16* buf, float* gcb, float* decb, unsigned int p,
    const __nv_bfloat16* __restrict__ W_in, const __nv_bfloat16* __restrict__ U_in,
    const __nv_bfloat16* __restrict__ key, const float* __restrict__ gate,
    const float* __restrict__ gc_in,
    unsigned int c, unsigned int b, unsigned int vh, unsigned int seq_len,
    unsigned int num_chunks, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int kh, unsigned int qk_stride, unsigned int gb_stride,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    const unsigned int tid = threadIdx.x;
    GDN_GEOM(g);
    const unsigned int cs = c * CHUNK;
    const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
    const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);
    key += g.tokoff * qk_stride;   // per-stream token offset (cu_seqlens / uniform b*seq_len)
    __nv_bfloat16* Wp = buf + (unsigned long long)p * CDH_BUFSZ;
    __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
    __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
    const unsigned int nthr = blockDim.x;   // 128 (scalar/TC) or 256 (k-split)
    const __nv_bfloat16* Wsrc = W_in + base * CHUNK * K_DIM;
    for (unsigned int e = tid * 8; e < CHUNK * K_DIM; e += nthr * 8) cp_async16(&Wp[e], &Wsrc[e]);
    const __nv_bfloat16* Usrc = U_in + base * CHUNK * V_DIM;
    for (unsigned int e = tid * 8; e < CHUNK * V_DIM; e += nthr * 8) cp_async16(&Up[e], &Usrc[e]);
    for (unsigned int j = tid; j < CHUNK * 16; j += nthr) {
        unsigned int i = j >> 4, c16 = (j & 15) * 8;
        if (i < ce)
            cp_async16(&Kp[i * K_DIM + c16],
                       key + (unsigned long long)(cs + i) * qk_stride + kh * k_dim + c16);
    }
    if (tid == 0) {
        float dl = gc_in[base * CHUNK + ce - 1];
        decb[p * (CHUNK + 1)] = expf(dl);
        for (unsigned int i = 0; i < ce; i++) {
            float g = gc_in[base * CHUNK + i];
            gcb[p * CHUNK + i] = g;
            decb[p * (CHUNK + 1) + 1 + i] = expf(dl - g);
        }
    }
    cp_commit();
}

// ── KERNEL 2: chunk_delta_h ──────────────────────────────────────────────
// The SERIAL state-passing spine — PRECISION-CRITICAL, so S stays f32 and its
// matmuls are fp32-FFMA (NOT bf16-TC: bf16-S drift fails token-equality).
// Grid: (num_v_heads, batch). One CTA per head, serial over chunks. 128 threads
// = v-columns; thread tid owns the WHOLE state column S[:,tid] RESIDENT IN
// REGISTERS (Sreg[K_DIM]) across all chunks — loaded once, updated in-register
// per chunk, stored once. This kills the per-chunk smem read/write of the 64KB
// f32 state, and the freed smem is spent on a DOUBLE BUFFER so the per-chunk
// W/U/K loads (the real bottleneck: global-load latency unhidden at 4 warps)
// are cp.async-prefetched for chunk c+1 while chunk c computes.
// (V-tiling for occupancy REGRESSED — 2 CTAs/head redundantly reload W + re-run
// the serial loop. bf16-TC for the matmuls is precision-SAFE (oracle probe @28k:
// +0.05% output drift) but architecturally BLOCKED on GB10: the 64KB f32 state
// must persist across chunks — in registers (128/thread → collides with TC's
// ~128 fragment-accumulator regs → spills) or in smem (64KB → no room for bf16
// TC operands + f32 outputs under the 99KB cap; V-tiling S to fit = the redundant
// reload that regressed). So scalar register-S + cp.async double-buffer is the
// achieved optimum here; TC needs a ground-up smem-state-tiling rewrite.)
// Per chunk c (entry S_c): store bf16(S_c) → S_out; uc = U_c - W_c·S_c → uc_out;
// S_{c+1} = exp(gc_last)·S_c + Σ_i exp(gc_last-gc_i)·uc_i·k_i.
// smem: 2×{W(16K)+K(16K)+U(16K)} bf16 + gc[2][CHUNK] + decay[2][CHUNK+1].
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_chunk_delta_h(
    float* __restrict__ h_state,          // [nv][K][V] per (b,vh): entry state IN, final state OUT
    const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in,
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate,
    const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out,    // [(b*NT+c)*nv+vh][K][V] per-chunk ENTRY states
    __nv_bfloat16* __restrict__ uc_out,   // [(b*NT+c)*nv+vh][C][V] corrected values
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int gb_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* buf = (__nv_bfloat16*)smem_raw;          // buf[2][CDH_BUFSZ]
    float* gcb = (float*)(buf + 2 * CDH_BUFSZ);             // gcb[2][CHUNK]
    float* decb = gcb + 2 * CHUNK;                          // decb[2][CHUNK+1], [0]=exp(gc_last)

    // State column S[:,tid] resident in registers for this thread's whole lifetime.
    float* H = h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sreg[K_DIM];
    #pragma unroll
    for (unsigned int k = 0; k < K_DIM; k++) Sreg[k] = H[k * V_DIM + tid];

    // Prologue: kick off chunk 0's loads.
    cdh_prefetch(buf, gcb, decb, 0, W_in, U_in, key, gate, gc_in, 0, b, vh, seq_len,
                 num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                 nullptr, nullptr, 0);  // scalar chunk_delta_h: uniform only

    for (unsigned int c = 0; c < num_chunks; c++) {
        const unsigned int cur = c & 1u;
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (seq_len - cs) < CHUNK ? (seq_len - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(b * num_chunks + c) * num_v_heads + vh);

        if (c + 1 < num_chunks) {
            cdh_prefetch(buf, gcb, decb, (c + 1) & 1u, W_in, U_in, key, gate, gc_in, c + 1, b, vh, seq_len,
                         num_chunks, num_v_heads, k_dim, kh, qk_stride, gb_stride,
                 nullptr, nullptr, 0);  // scalar chunk_delta_h: uniform only
            cp_wait<1>();   // chunk c's loads (older group) complete; keep c+1's in flight
        } else {
            cp_wait<0>();
        }
        __syncthreads();    // make buf[cur] visible CTA-wide

        __nv_bfloat16* Wp = buf + (unsigned long long)cur * CDH_BUFSZ;
        __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;
        __nv_bfloat16* Up = Kp + CHUNK * K_DIM;
        const float* dec = decb + cur * (CHUNK + 1);

        // Store entry state S_c for the output pass. The recurrent master stays f32 in Sreg/H;
        // chunk_fwd_o consumes S_c as a bf16 MMA operand, so f32 scratch only adds traffic.
        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++)
            S_out[base * K_DIM * V_DIM + k * V_DIM + tid] = __float2bfloat16(Sreg[k]);

        // uc_i = U_i - W_i·S   (W·S contracts over k against the register state column)
        float duc[CHUNK];
        const float edl = dec[0];
        for (unsigned int i = 0; i < ce; i++) {
            float ws = 0.0f;
            #pragma unroll
            for (unsigned int k = 0; k < K_DIM; k++)
                ws += (float)Wp[i * K_DIM + k] * Sreg[k];
            float uci = (float)Up[i * V_DIM + tid] - ws;
            uc_out[base * CHUNK * V_DIM + i * v_dim + tid] = __float2bfloat16(uci);
            duc[i] = dec[1 + i] * uci;   // decayed corrected-value, once per i
        }
        // S_{c+1} = edl·S + Σ_i duc_i·k_i   (in-register update, no smem state traffic)
        #pragma unroll
        for (unsigned int k = 0; k < K_DIM; k++) {
            float hv = edl * Sreg[k];
            for (unsigned int i = 0; i < ce; i++)
                hv += duc[i] * (float)Kp[i * K_DIM + k];   // pure MAC inner loop
            Sreg[k] = hv;
        }
        __syncthreads();   // before buf[cur] is overwritten by the chunk-(c+2) prefetch
    }

    #pragma unroll
    for (unsigned int k = 0; k < K_DIM; k++) H[k * V_DIM + tid] = Sreg[k];
}

// ── KERNEL 2-TC: chunk_delta_h_tc ────────────────────────────────────────
// State-tiling tensor-core variant of the serial spine (A/B candidate vs the
// scalar register-S kernel above). Same math, same outputs. register-S stays the
// f32 MASTER state (no 64KB smem state); each chunk a bf16 SNAPSHOT Sᵀ[v][k]=
// bf16(S[k][v]) is staged to smem PURELY as an mma operand — the f32 master in
// registers is undamaged, so accumulation precision is unchanged (the snapshot
// is only a per-chunk read, like a bf16 GEMM input; oracle probe @28k = +0.05%).
//   Phase A (TC):  ws[i][v] = Σ_k W[i][k]·Sᵀ[v][k]  via mma_gram → uc=U-ws → duc.
//   Phase B (scalar, increment-1):  S[k][v] = edl·S[k][v] + Σ_i duc_i·K[i][k].
// smem: Sᵀ(32K) + W(16K) + ws(32K f32) + U(16K) = 96.25KB (K reuses Sᵀ for phase B).
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_chunk_fwd_o(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate,
    const float* __restrict__ gc_in,
    const __nv_bfloat16* __restrict__ S_in, // [(b*NT+c)*nv+vh][K][V] entry states (from #2)
    const __nv_bfloat16* __restrict__ uc_in,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int q_stride,
    unsigned int k_stride,
    unsigned int gb_stride,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    const unsigned int c = blockIdx.x;
    const unsigned int vh = blockIdx.y;
    const unsigned int b = blockIdx.z;
    if (vh >= num_v_heads || b >= batch_size) return;
    GDN_GEOM(g);
    if (c >= g.nchunks) return;            // per-stream chunk bound
    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const float inv_sqrt_d = rsqrtf((float)k_dim);
    const unsigned int cs = c * CHUNK;
    const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
    const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);
    const unsigned long long out_base = (g.tokoff * num_v_heads + vh) * v_dim;
    // Per-stream input offset (cu_seqlens[b] in varlen, else b*seq_len; b=0 → +0).
    query += g.tokoff * q_stride;
    key   += g.tokoff * k_stride;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sq = (__nv_bfloat16*)smem_raw;          // [CHUNK*K_DIM]
    __nv_bfloat16* sk = sq + CHUNK * K_DIM;                // [CHUNK*K_DIM]
    float* kq = (float*)(sk + CHUNK * K_DIM);              // [CHUNK*CHUNK]
    __nv_bfloat16* ucb = (__nv_bfloat16*)(kq + CHUNK * CHUNK); // [CHUNK*V_DIM]
    __nv_bfloat16* Sb = ucb + CHUNK * V_DIM;               // [K_DIM*V_DIM] bf16 (S_c)
    float* gc = (float*)(Sb + K_DIM * V_DIM);              // [CHUNK]
    float* egc = gc + CHUNK;                               // [CHUNK] exp(gc)

    for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += 128) {
        unsigned int i = idx / k_dim, j = idx % k_dim;
        if (i < ce) {
            unsigned long long qoff = (unsigned long long)(cs + i) * q_stride + kh * k_dim + j;
            unsigned long long koff = (unsigned long long)(cs + i) * k_stride + kh * k_dim + j;
            sq[i * K_DIM + j] = query[qoff];
            sk[i * K_DIM + j] = key[koff];
        } else {
            sq[i * K_DIM + j] = __float2bfloat16(0.0f);
            sk[i * K_DIM + j] = __float2bfloat16(0.0f);
        }
    }
    for (unsigned int idx = tid; idx < CHUNK * v_dim; idx += 128) {
        unsigned int i = idx / v_dim, v = idx % v_dim;
        ucb[i * V_DIM + v] = (i < ce) ? uc_in[base * CHUNK * V_DIM + i * v_dim + v] : __float2bfloat16(0.0f);
    }
    // S_c read TRANSPOSED → Sbᵀ[v][k] = S_c[k][v], so mma_gram(q, Sbᵀ) = <q_i,S_c[:,v]>.
    for (unsigned int idx = tid; idx < K_DIM * V_DIM; idx += 128) {
        unsigned int v = idx / K_DIM, k = idx % K_DIM;
        Sb[idx] = S_in[base * K_DIM * V_DIM + k * V_DIM + v];
    }
    for (unsigned int i = tid; i < ce; i += 128) {
        float g = gc_in[base * CHUNK + i];
        gc[i] = g;
        egc[i] = expf(g);
    }
    __syncthreads();

    mma_gram<8, CHUNK, false>(sq, sk, kq);   // kq[i][l] = <q_i, k_l>
    __syncthreads();

    // Fold the intra-chunk decay into the Gram ONCE: kq[i][l] ← exp(gc_i-gc_l)·<q_i,k_l>
    // (was: expf(gc_i-gc_l) recomputed per v-column = v_dim× redundant transcendentals).
    for (unsigned int p = tid; p < CHUNK * CHUNK; p += 128) {
        unsigned int i = p / CHUNK, l = p % CHUNK;
        if (i < ce && l <= i) kq[p] = expf(gc[i] - gc[l]) * kq[p];
    }
    __syncthreads();   // sk is free past mma1 → reuse its region for the o1 = q·Sᵀ result

    // o1[i][v] = <q_i, S_c[:,v]>  on tensor cores (bf16 out → terminal, precision-safe).
    __nv_bfloat16* o1 = sk;                   // [CHUNK*V_DIM] bf16, reuses sk's 16KB
    mma_gram<16, V_DIM, true>(sq, Sb, o1);
    __syncthreads();

    if (tid < v_dim) {
        for (unsigned int i = 0; i < ce; i++) {
            float t1 = egc[i] * (float)o1[i * V_DIM + tid];
            float t2 = 0.0f;
            for (unsigned int l = 0; l <= i; l++)
                t2 += kq[i * CHUNK + l] * (float)ucb[l * V_DIM + tid];   // pure MAC inner loop
            output[out_base + (unsigned long long)(cs + i) * num_v_heads * v_dim + tid] =
                __float2bfloat16((t1 + t2) * inv_sqrt_d);
        }
    }
}

// ── Isolated host ABI ────────────────────────────────────────────────────
// Workspace is caller-owned and stream-ordered. Nothing allocates, frees or
// synchronizes here; this library is deliberately absent from Atlas routing.
//
// v1 consumes compact Q/K/V and compact FP32 log-decay/beta. v2 preserves the
// same arithmetic but admits Atlas production's single strided QKV buffer and
// interleaved [alpha(48), beta(48)] FP32 rows. The conversion kernel writes
// compact log(alpha) into the v2-only tail of the caller workspace. Both ABIs
// then enter the exact same three-kernel core.
extern "C" __global__ void gated_delta_rule_alpha_to_log_v2(
    const float* __restrict__ gate_beta, float* __restrict__ log_gate,
    unsigned int seq_len, unsigned int gate_beta_stride) {
    const unsigned long long idx =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long count = (unsigned long long)seq_len * 48;
    if (idx >= count) return;
    const unsigned int token = (unsigned int)(idx / 48);
    const unsigned int head = (unsigned int)(idx % 48);
    const float alpha = gate_beta[(unsigned long long)token * gate_beta_stride + head];
    log_gate[idx] = (isfinite(alpha) && alpha >= 0.0f)
        ? logf(fmaxf(alpha, GATE_FLOOR))
        : nanf("");
}

// ── v3: pinned-c143 program geometry and tensor-core dataflow ─────────────
//
// v1/v2 deliberately remain the scalar numerical oracle.  The runtime gate
// showed that their one-program-per-head recurrence and output passes are
// about 4x slower than pinned c143.  c143 instead uses four independent BV=32
// state programs per head, two BV=64 output programs per chunk/head, and
// tensor-core products for both recurrence contractions and A@V.  v3 mirrors
// that exact decomposition while retaining Atlas's external [H,K,V] FP32
// state and production-strided v2 input ABI.

template <int KRED, int NTC, int ASTRIDE, int BSTRIDE, int CSTRIDE, bool OutBf16>
__device__ __forceinline__ void c143_v3_mma_row_row(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    void* __restrict__ C) {
    static_assert((KRED % 16) == 0, "mma reduction must be a multiple of 16");
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31;
    const unsigned grp = lane >> 2;
    const unsigned q = lane & 3;
    const unsigned warp_m = warp * 16;
    const unsigned short* sA = reinterpret_cast<const unsigned short*>(A);
    const unsigned short* sB = reinterpret_cast<const unsigned short*>(B);
    float acc[NTC][4];
#pragma unroll
    for (int nt = 0; nt < NTC; ++nt) {
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
    }
#pragma unroll
    for (unsigned ks = 0; ks < KRED; ks += 16) {
        const unsigned fr0 = warp_m + grp;
        const unsigned fr1 = fr0 + 8;
        const unsigned fc0 = ks + q * 2;
        const unsigned fc1 = fc0 + 8;
        const unsigned a0 = *reinterpret_cast<const unsigned*>(&sA[fr0 * ASTRIDE + fc0]);
        const unsigned a1 = *reinterpret_cast<const unsigned*>(&sA[fr1 * ASTRIDE + fc0]);
        const unsigned a2 = *reinterpret_cast<const unsigned*>(&sA[fr0 * ASTRIDE + fc1]);
        const unsigned a3 = *reinterpret_cast<const unsigned*>(&sA[fr1 * ASTRIDE + fc1]);
#pragma unroll
        for (int nt = 0; nt < NTC; ++nt) {
            const unsigned nc = nt * 8 + grp;
            const unsigned k0 = ks + q * 2;
            const unsigned k1 = k0 + 8;
            const unsigned b0 =
                (static_cast<unsigned>(sB[nc * BSTRIDE + k0 + 1]) << 16) |
                static_cast<unsigned>(sB[nc * BSTRIDE + k0]);
            const unsigned b1 =
                (static_cast<unsigned>(sB[nc * BSTRIDE + k1 + 1]) << 16) |
                static_cast<unsigned>(sB[nc * BSTRIDE + k1]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]),
                  "=f"(acc[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]),
                  "f"(acc[nt][3]));
        }
    }
#pragma unroll
    for (int nt = 0; nt < NTC; ++nt) {
        const unsigned n0 = nt * 8 + q * 2;
        const unsigned n1 = n0 + 1;
        const unsigned m0 = warp_m + grp;
        const unsigned m1 = m0 + 8;
        if constexpr (OutBf16) {
            __nv_bfloat16* out = static_cast<__nv_bfloat16*>(C);
            out[m0 * CSTRIDE + n0] = __float2bfloat16(acc[nt][0]);
            out[m0 * CSTRIDE + n1] = __float2bfloat16(acc[nt][1]);
            out[m1 * CSTRIDE + n0] = __float2bfloat16(acc[nt][2]);
            out[m1 * CSTRIDE + n1] = __float2bfloat16(acc[nt][3]);
        } else {
            float* out = static_cast<float*>(C);
            out[m0 * CSTRIDE + n0] = acc[nt][0];
            out[m0 * CSTRIDE + n1] = acc[nt][1];
            out[m1 * CSTRIDE + n0] = acc[nt][2];
            out[m1 * CSTRIDE + n1] = acc[nt][3];
        }
    }
}

// QK-specialized sibling of c143_v3_mma_row_row<128,8,...>. It deliberately
// keeps the same accumulator initialization, K-step, fragment loads and HMMA
// order. The sole layout change is the epilogue: every warp retains its FP32
// accumulators across a full-CTA barrier, proving all K reads complete before
// the dead K tile is overwritten by the decayed/causal BF16 matrix.
__device__ __forceinline__ void c143_v3_mma_qk_causal(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ causal,
    const float* __restrict__ gc,
    unsigned long long base,
    unsigned int ce) {
    constexpr unsigned NTC = 8;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31;
    const unsigned grp = lane >> 2;
    const unsigned q = lane & 3;
    const unsigned warp_m = warp * 16;
    const unsigned short* s_query =
        reinterpret_cast<const unsigned short*>(query);
    const unsigned short* s_key =
        reinterpret_cast<const unsigned short*>(key);
    float acc[NTC][4];
#pragma unroll
    for (unsigned nt = 0; nt < NTC; ++nt) {
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
    }
#pragma unroll
    for (unsigned ks = 0; ks < K_DIM; ks += 16) {
        const unsigned fr0 = warp_m + grp;
        const unsigned fr1 = fr0 + 8;
        const unsigned fc0 = ks + q * 2;
        const unsigned fc1 = fc0 + 8;
        const unsigned a0 =
            *reinterpret_cast<const unsigned*>(&s_query[fr0 * K_DIM + fc0]);
        const unsigned a1 =
            *reinterpret_cast<const unsigned*>(&s_query[fr1 * K_DIM + fc0]);
        const unsigned a2 =
            *reinterpret_cast<const unsigned*>(&s_query[fr0 * K_DIM + fc1]);
        const unsigned a3 =
            *reinterpret_cast<const unsigned*>(&s_query[fr1 * K_DIM + fc1]);
#pragma unroll
        for (unsigned nt = 0; nt < NTC; ++nt) {
            const unsigned nc = nt * 8 + grp;
            const unsigned k0 = ks + q * 2;
            const unsigned k1 = k0 + 8;
            const unsigned b0 =
                (static_cast<unsigned>(s_key[nc * K_DIM + k0 + 1]) << 16) |
                static_cast<unsigned>(s_key[nc * K_DIM + k0]);
            const unsigned b1 =
                (static_cast<unsigned>(s_key[nc * K_DIM + k1 + 1]) << 16) |
                static_cast<unsigned>(s_key[nc * K_DIM + k1]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]),
                  "=f"(acc[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]),
                  "f"(acc[nt][3]));
        }
    }

    // `causal` aliases `key`. No thread may publish its epilogue until every
    // warp has consumed the complete K tile.
    __syncthreads();
#pragma unroll
    for (unsigned nt = 0; nt < NTC; ++nt) {
        const unsigned n0 = nt * 8 + q * 2;
        const unsigned n1 = n0 + 1;
        const unsigned m0 = warp_m + grp;
        const unsigned m1 = m0 + 8;
        float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (m0 < ce && n0 <= m0) {
            values[0] = expf(gc[base * CHUNK + m0] - gc[base * CHUNK + n0]) *
                        acc[nt][0];
        }
        if (m0 < ce && n1 <= m0) {
            values[1] = expf(gc[base * CHUNK + m0] - gc[base * CHUNK + n1]) *
                        acc[nt][1];
        }
        if (m1 < ce && n0 <= m1) {
            values[2] = expf(gc[base * CHUNK + m1] - gc[base * CHUNK + n0]) *
                        acc[nt][2];
        }
        if (m1 < ce && n1 <= m1) {
            values[3] = expf(gc[base * CHUNK + m1] - gc[base * CHUNK + n1]) *
                        acc[nt][3];
        }
        causal[m0 * CHUNK + n0] = __float2bfloat16(values[0]);
        causal[m0 * CHUNK + n1] = __float2bfloat16(values[1]);
        causal[m1 * CHUNK + n0] = __float2bfloat16(values[2]);
        causal[m1 * CHUNK + n1] = __float2bfloat16(values[3]);
    }
}

// One program per (chunk,value-head), exactly the donor's (NT,B*H) geometry.
// It fuses local gate cumsum with KKT and materializes the solved 64x64 A in
// BF16.  Keeping A explicit permits the following W/U pass to be pure MMA and
// removes the v2 kernel's two 64-element per-thread local arrays.
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_c143_v3_kkt_solve(
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ log_gate,
    const float* __restrict__ beta,
    float* __restrict__ gc_out,
    __nv_bfloat16* __restrict__ a_out,
    unsigned int seq_len,
    unsigned int key_stride,
    unsigned int gate_stride,
    unsigned int beta_stride) {
    const unsigned c = blockIdx.x;
    const unsigned vh = blockIdx.y;
    const unsigned tid = threadIdx.x;
    const unsigned nt = (seq_len + CHUNK - 1) / CHUNK;
    if (c >= nt || vh >= 48) return;
    const unsigned cs = c * CHUNK;
    const unsigned ce = min(CHUNK, seq_len - cs);
    const unsigned kh = vh / 3;
    const unsigned long long base =
        (static_cast<unsigned long long>(c) * 48 + vh);

    extern __shared__ char raw[];
    __nv_bfloat16* sk = reinterpret_cast<__nv_bfloat16*>(raw);
    static_assert(CHUNK * K_DIM * sizeof(__nv_bfloat16) ==
                  CHUNK * CHUNK * sizeof(float),
                  "dead key tile must exactly hold the strict-lower matrix");
    float* lower = reinterpret_cast<float*>(sk);
    float* gram = reinterpret_cast<float*>(sk + CHUNK * K_DIM);
    float* inverse = gram + CHUNK * CHUNK;
    float* gc = inverse + CHUNK * CHUNK;

    for (unsigned idx = tid; idx < CHUNK * K_DIM; idx += blockDim.x) {
        const unsigned i = idx / K_DIM;
        const unsigned k = idx % K_DIM;
        sk[idx] = i < ce
            ? key[static_cast<unsigned long long>(cs + i) * key_stride + kh * K_DIM + k]
            : __float2bfloat16(0.0f);
    }
    if (tid == 0) {
        float sum = 0.0f;
        for (unsigned i = 0; i < ce; ++i) {
            sum += log_gate[static_cast<unsigned long long>(cs + i) * gate_stride + vh];
            gc[i] = sum;
            gc_out[base * CHUNK + i] = sum;
        }
        for (unsigned i = ce; i < CHUNK; ++i) gc[i] = 0.0f;
    }
    __syncthreads();

    mma_gram<8, CHUNK, false>(sk, sk, gram);
    __syncthreads();
    // The key tile is dead after Gram completion, so its disjoint 16 KiB can
    // hold L without racing the still-read-only transposed Gram cells.
    for (unsigned idx = tid; idx < CHUNK * CHUNK; idx += blockDim.x) {
        const unsigned i = idx / CHUNK;
        const unsigned j = idx % CHUNK;
        float value = 0.0f;
        if (i < ce && j < i) {
            const float bi = beta[static_cast<unsigned long long>(cs + i) * beta_stride + vh];
            value = bi * expf(gc[i] - gc[j]) * gram[j * CHUNK + i];
        }
        lower[idx] = value;
        inverse[idx] = 0.0f;
    }
    __syncthreads();

    // Parallel columns of (I+L)^-1. This preserves the donor's explicit-A
    // dataflow; only the tiny triangular solve is scalar FP32.
    for (unsigned i = 0; i < ce; ++i) {
        if (tid < CHUNK) {
            const unsigned j = tid;
            if (j == i) {
                inverse[i * CHUNK + j] = 1.0f;
            } else if (j < i) {
                float sum = 0.0f;
                for (unsigned l = j; l < i; ++l) {
                    sum += lower[i * CHUNK + l] * inverse[l * CHUNK + j];
                }
                inverse[i * CHUNK + j] = -sum;
            }
        }
        __syncthreads();
    }
    for (unsigned idx = tid; idx < CHUNK * CHUNK; idx += blockDim.x) {
        a_out[base * CHUNK * CHUNK + idx] = __float2bfloat16(inverse[idx]);
    }
}

// One program per (chunk,value-head), donor BK=BV=64, 4-warps geometry.
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_c143_v3_recompute_wu(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ gc,
    const __nv_bfloat16* __restrict__ a,
    __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ u,
    unsigned int seq_len,
    unsigned int key_stride,
    unsigned int value_stride,
    unsigned int beta_stride) {
    const unsigned c = blockIdx.x;
    const unsigned vh = blockIdx.y;
    const unsigned tid = threadIdx.x;
    const unsigned nt = (seq_len + CHUNK - 1) / CHUNK;
    if (c >= nt || vh >= 48) return;
    const unsigned cs = c * CHUNK;
    const unsigned ce = min(CHUNK, seq_len - cs);
    const unsigned kh = vh / 3;
    const unsigned long long base =
        (static_cast<unsigned long long>(c) * 48 + vh);
    extern __shared__ char raw[];
    __nv_bfloat16* sa = reinterpret_cast<__nv_bfloat16*>(raw);       // 64x64
    __nv_bfloat16* sb = sa + CHUNK * CHUNK;                          // 128x64
    for (unsigned idx = tid; idx < CHUNK * CHUNK; idx += blockDim.x) {
        sa[idx] = a[base * CHUNK * CHUNK + idx];
    }
    for (unsigned idx = tid; idx < K_DIM * CHUNK; idx += blockDim.x) {
        const unsigned k = idx / CHUNK;
        const unsigned i = idx % CHUNK;
        float x = 0.0f;
        if (i < ce) {
            const float bi = beta[static_cast<unsigned long long>(cs + i) * beta_stride + vh];
            x = bi * expf(gc[base * CHUNK + i]) *
                static_cast<float>(key[static_cast<unsigned long long>(cs + i) * key_stride + kh * K_DIM + k]);
        }
        sb[idx] = __float2bfloat16(x);
    }
    __syncthreads();
    c143_v3_mma_row_row<64, 16, 64, 64, 128, true>(
        sa, sb, w + base * CHUNK * K_DIM);
    __syncthreads();
    for (unsigned idx = tid; idx < V_DIM * CHUNK; idx += blockDim.x) {
        const unsigned v = idx / CHUNK;
        const unsigned i = idx % CHUNK;
        float x = 0.0f;
        if (i < ce) {
            const float bi = beta[static_cast<unsigned long long>(cs + i) * beta_stride + vh];
            x = bi * static_cast<float>(value[
                static_cast<unsigned long long>(cs + i) * value_stride + vh * V_DIM + v]);
        }
        sb[idx] = __float2bfloat16(x);
    }
    __syncthreads();
    c143_v3_mma_row_row<64, 16, 64, 64, 128, true>(
        sa, sb, u + base * CHUNK * V_DIM);
}

// Four BV=32 programs per value head, matching c143's grid(V/BV,N*H).
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_c143_v3_delta_h(
    float* __restrict__ state_kv,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ u,
    const float* __restrict__ gc,
    __nv_bfloat16* __restrict__ h_entries,
    __nv_bfloat16* __restrict__ v_new,
    unsigned int seq_len,
    unsigned int key_stride) {
    constexpr unsigned BV = 32;
    const unsigned vb = blockIdx.x;
    const unsigned vh = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (vb >= 4 || vh >= 48) return;
    const unsigned nt = (seq_len + CHUNK - 1) / CHUNK;
    const unsigned kh = vh / 3;
    const unsigned v0 = vb * BV;
    float state[32];
#pragma unroll
    for (unsigned slot = 0; slot < 32; ++slot) {
        const unsigned idx = tid + slot * blockDim.x;
        const unsigned lv = idx / K_DIM;
        const unsigned k = idx % K_DIM;
        state[slot] = state_kv[(static_cast<unsigned long long>(vh) * K_DIM + k) * V_DIM + v0 + lv];
    }

    extern __shared__ char raw[];
    __nv_bfloat16* ma = reinterpret_cast<__nv_bfloat16*>(raw);       // max 128x64
    __nv_bfloat16* mb = ma + CHUNK * K_DIM;                          // max 32x128
    float* mc = reinterpret_cast<float*>(mb + BV * K_DIM);           // 64x32

    for (unsigned c = 0; c < nt; ++c) {
        const unsigned cs = c * CHUNK;
        const unsigned ce = min(CHUNK, seq_len - cs);
        const unsigned long long base =
            (static_cast<unsigned long long>(c) * 48 + vh);
#pragma unroll
        for (unsigned slot = 0; slot < 32; ++slot) {
            const unsigned idx = tid + slot * blockDim.x;
            const unsigned lv = idx / K_DIM;
            const unsigned k = idx % K_DIM;
            const __nv_bfloat16 snapshot = __float2bfloat16(state[slot]);
            mb[lv * K_DIM + k] = snapshot;
            h_entries[base * V_DIM * K_DIM + (v0 + lv) * K_DIM + k] = snapshot;
        }
        for (unsigned idx = tid; idx < CHUNK * K_DIM; idx += blockDim.x) {
            ma[idx] = w[base * CHUNK * K_DIM + idx];
        }
        __syncthreads();
        c143_v3_mma_row_row<128, 4, 128, 128, BV, false>(ma, mb, mc);
        __syncthreads();

        // Reuse mb as [BV,64] decayed corrected values.
        const float g_last = gc[base * CHUNK + ce - 1];
        for (unsigned idx = tid; idx < CHUNK * BV; idx += blockDim.x) {
            const unsigned i = idx / BV;
            const unsigned lv = idx % BV;
            float corrected = 0.0f;
            if (i < ce) {
                corrected = static_cast<float>(u[base * CHUNK * V_DIM + i * V_DIM + v0 + lv]) - mc[idx];
                v_new[base * CHUNK * V_DIM + i * V_DIM + v0 + lv] =
                    __float2bfloat16(corrected);
                corrected *= expf(g_last - gc[base * CHUNK + i]);
            }
            mb[lv * CHUNK + i] = __float2bfloat16(corrected);
        }
        // Reuse ma as K^T [128,64].
        for (unsigned idx = tid; idx < K_DIM * CHUNK; idx += blockDim.x) {
            const unsigned k = idx / CHUNK;
            const unsigned i = idx % CHUNK;
            ma[idx] = i < ce
                ? key[static_cast<unsigned long long>(cs + i) * key_stride + kh * K_DIM + k]
                : __float2bfloat16(0.0f);
        }
        __syncthreads();
        const float end_decay = expf(g_last);
        c143_v3_mma_row_row<64, 4, 64, 64, BV, false>(ma, mb, mc);
        __syncthreads();
#pragma unroll
        for (unsigned slot = 0; slot < 32; ++slot) {
            const unsigned idx = tid + slot * blockDim.x;
            const unsigned lv = idx / K_DIM;
            const unsigned k = idx % K_DIM;
            if (k < 64) state[slot] = end_decay * state[slot] + mc[k * BV + lv];
        }
        __syncthreads();
        c143_v3_mma_row_row<64, 4, 64, 64, BV, false>(ma + 64 * CHUNK, mb, mc);
        __syncthreads();
#pragma unroll
        for (unsigned slot = 0; slot < 32; ++slot) {
            const unsigned idx = tid + slot * blockDim.x;
            const unsigned lv = idx / K_DIM;
            const unsigned k = idx % K_DIM;
            if (k >= 64) state[slot] = end_decay * state[slot] + mc[(k - 64) * BV + lv];
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned slot = 0; slot < 32; ++slot) {
        const unsigned idx = tid + slot * blockDim.x;
        const unsigned lv = idx / K_DIM;
        const unsigned k = idx % K_DIM;
        state_kv[(static_cast<unsigned long long>(vh) * K_DIM + k) * V_DIM + v0 + lv] = state[slot];
    }
}

// Two BV=64 programs per (chunk,value-head), matching pinned c143's
// grid(V/BV,NT,B*H). q@h, q@k and causal A@v_new are all tensor-core dots.
extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_c143_v3_fwd_o(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gc,
    const __nv_bfloat16* __restrict__ h_entries,
    const __nv_bfloat16* __restrict__ v_new,
    __nv_bfloat16* __restrict__ output,
    unsigned int seq_len,
    unsigned int query_stride,
    unsigned int key_stride) {
    constexpr unsigned BV = 64;
    const unsigned vb = blockIdx.x;
    const unsigned c = blockIdx.y;
    const unsigned vh = blockIdx.z;
    const unsigned tid = threadIdx.x;
    const unsigned nt = (seq_len + CHUNK - 1) / CHUNK;
    if (vb >= 2 || c >= nt || vh >= 48) return;
    const unsigned cs = c * CHUNK;
    const unsigned ce = min(CHUNK, seq_len - cs);
    const unsigned kh = vh / 3;
    const unsigned v0 = vb * BV;
    const unsigned long long base =
        (static_cast<unsigned long long>(c) * 48 + vh);
    extern __shared__ char raw[];
    __nv_bfloat16* sq = reinterpret_cast<__nv_bfloat16*>(raw);       // Q, then FP32 A@V
    __nv_bfloat16* sh = sq + CHUNK * K_DIM;                          // H, then K, then causal+V^T
    float* o_state = reinterpret_cast<float*>(sh + CHUNK * K_DIM);   // persistent 64x64
    for (unsigned idx = tid; idx < CHUNK * K_DIM; idx += blockDim.x) {
        const unsigned i = idx / K_DIM;
        const unsigned k = idx % K_DIM;
        sq[idx] = i < ce
            ? query[static_cast<unsigned long long>(cs + i) * query_stride + kh * K_DIM + k]
            : __float2bfloat16(0.0f);
        sh[idx] = h_entries[base * V_DIM * K_DIM + (v0 + i) * K_DIM + k];
    }
    __syncthreads();
    c143_v3_mma_row_row<128, 8, 128, 128, BV, false>(sq, sh, o_state);
    __syncthreads();

    // H is dead after q@h. Reuse its 16 KiB tile for K; the specialized QK
    // epilogue holds all FP32 accumulators across a CTA barrier before
    // overwriting this same region with the 8 KiB causal BF16 matrix.
    for (unsigned idx = tid; idx < CHUNK * K_DIM; idx += blockDim.x) {
        const unsigned i = idx / K_DIM;
        const unsigned k = idx % K_DIM;
        sh[idx] = i < ce
            ? key[static_cast<unsigned long long>(cs + i) * key_stride + kh * K_DIM + k]
            : __float2bfloat16(0.0f);
    }
    __syncthreads();
    __nv_bfloat16* causal = sh;                    // 64x64; aliases dead K
    c143_v3_mma_qk_causal(sq, sh, causal, gc, base, ce);

    // Q is now dead. Its 16 KiB tile becomes the FP32 A@V result. The upper
    // half of the former K tile is disjoint from causal and holds V^T.
    __nv_bfloat16* vt = causal + CHUNK * CHUNK;    // 64x64
    float* av = reinterpret_cast<float*>(sq);      // 64x64
    for (unsigned idx = tid; idx < CHUNK * CHUNK; idx += blockDim.x) {
        const unsigned i = idx / CHUNK;
        const unsigned j = idx % CHUNK;
        vt[idx] = j < ce
            ? v_new[base * CHUNK * V_DIM + j * V_DIM + v0 + i]
            : __float2bfloat16(0.0f);
    }
    __syncthreads();
    c143_v3_mma_row_row<64, 8, 64, 64, BV, false>(causal, vt, av);
    __syncthreads();
    const float scale = rsqrtf(static_cast<float>(K_DIM));
    for (unsigned idx = tid; idx < CHUNK * BV; idx += blockDim.x) {
        const unsigned i = idx / BV;
        const unsigned lv = idx % BV;
        if (i < ce) {
            const float value =
                (expf(gc[base * CHUNK + i]) * o_state[idx] + av[idx]) * scale;
            output[(static_cast<unsigned long long>(cs + i) * 48 + vh) * V_DIM + v0 + lv] =
                __float2bfloat16(value);
        }
    }
}

namespace {
constexpr unsigned int C143_KEY_HEADS = 16;
constexpr unsigned int C143_VALUE_HEADS = 48;
constexpr unsigned int C143_DIM = 128;
constexpr unsigned int C143_Q_WIDTH = C143_KEY_HEADS * C143_DIM;
constexpr unsigned int C143_K_WIDTH = C143_KEY_HEADS * C143_DIM;
constexpr unsigned int C143_V_WIDTH = C143_VALUE_HEADS * C143_DIM;
constexpr unsigned int C143_QKV_WIDTH = C143_Q_WIDTH + C143_K_WIDTH + C143_V_WIDTH;
constexpr unsigned int C143_GATE_BETA_WIDTH = C143_VALUE_HEADS * 2;
constexpr unsigned int C143_WU_SMEM = 49408;
constexpr unsigned int C143_DH_SMEM = 99336;
constexpr unsigned int C143_OUT_SMEM = 98816;
constexpr unsigned int C143_V3_KKT_SMEM = 49408;
constexpr unsigned int C143_V3_WU_SMEM = 24576;
constexpr unsigned int C143_V3_DH_SMEM = 32768;
constexpr unsigned int C143_V3_OUT_SMEM = 49152;

thread_local char c143_error[512] = "ok";

int c143_fail(cudaError_t status, const char* operation) {
    std::snprintf(c143_error, sizeof(c143_error), "%s: %s", operation,
                  cudaGetErrorString(status));
    return static_cast<int>(status);
}

int c143_arg_fail(int code, const char* message) {
    std::snprintf(c143_error, sizeof(c143_error), "%s", message);
    return code;
}

bool checked_mul(size_t a, size_t b, size_t* out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    *out = a * b;
    return true;
}

bool checked_add(size_t a, size_t b, size_t* out) {
    if (b > std::numeric_limits<size_t>::max() - a) return false;
    *out = a + b;
    return true;
}

bool c143_workspace_v1(unsigned int seq_len, size_t* out) {
    if (seq_len == 0) return false;
    const size_t nt = (static_cast<size_t>(seq_len) + CHUNK - 1) / CHUNK;
    size_t base = 0;
    if (!checked_mul(nt, C143_VALUE_HEADS, &base)) return false;
    const size_t per_head =
        CHUNK * K_DIM * sizeof(__nv_bfloat16) +  // W
        CHUNK * V_DIM * sizeof(__nv_bfloat16) +  // U
        CHUNK * sizeof(float) +                  // gc
        K_DIM * V_DIM * sizeof(__nv_bfloat16) +  // S
        CHUNK * V_DIM * sizeof(__nv_bfloat16);   // uc
    return checked_mul(base, per_head, out);
}

bool c143_workspace_v2(unsigned int seq_len, size_t* out) {
    size_t base = 0;
    size_t log_bytes = 0;
    if (!c143_workspace_v1(seq_len, &base) ||
        !checked_mul(static_cast<size_t>(seq_len), C143_VALUE_HEADS * sizeof(float),
                     &log_bytes)) {
        return false;
    }
    return checked_add(base, log_bytes, out);
}

struct C143V3Layout {
    size_t w_offset;
    size_t u_offset;
    size_t gc_offset;
    size_t phase_offset;
    size_t h_offset;
    size_t v_new_offset;
    size_t log_offset;
    size_t total_bytes;
};

bool c143_workspace_v3(unsigned int seq_len, C143V3Layout* layout) {
    if (seq_len == 0 || layout == nullptr) return false;
    const size_t nt = (static_cast<size_t>(seq_len) + CHUNK - 1) / CHUNK;
    // fwd_o mirrors Triton's (BV,NT,B*H) grid, whose CUDA y dimension is 16-bit.
    // This still admits the full 1M-token target (NT=16384) and rejects an
    // otherwise representable but unlaunchable grid before the alpha kernel.
    if (nt > 65535) return false;
    size_t programs = 0;
    if (!checked_mul(nt, C143_VALUE_HEADS, &programs)) return false;
    size_t w_bytes = 0;
    size_t u_bytes = 0;
    size_t gc_bytes = 0;
    size_t a_bytes = 0;
    size_t h_bytes = 0;
    size_t v_new_bytes = 0;
    size_t log_bytes = 0;
    if (!checked_mul(programs, CHUNK * K_DIM * sizeof(__nv_bfloat16), &w_bytes) ||
        !checked_mul(programs, CHUNK * V_DIM * sizeof(__nv_bfloat16), &u_bytes) ||
        !checked_mul(programs, CHUNK * sizeof(float), &gc_bytes) ||
        !checked_mul(programs, CHUNK * CHUNK * sizeof(__nv_bfloat16), &a_bytes) ||
        !checked_mul(programs, V_DIM * K_DIM * sizeof(__nv_bfloat16), &h_bytes) ||
        !checked_mul(programs, CHUNK * V_DIM * sizeof(__nv_bfloat16), &v_new_bytes) ||
        !checked_mul(static_cast<size_t>(seq_len), C143_VALUE_HEADS * sizeof(float),
                     &log_bytes)) {
        return false;
    }
    size_t h_and_v = 0;
    if (!checked_add(h_bytes, v_new_bytes, &h_and_v)) return false;
    const size_t phase_bytes = a_bytes > h_and_v ? a_bytes : h_and_v;
    size_t cursor = 0;
    layout->w_offset = cursor;
    if (!checked_add(cursor, w_bytes, &cursor)) return false;
    layout->u_offset = cursor;
    if (!checked_add(cursor, u_bytes, &cursor)) return false;
    layout->gc_offset = cursor;
    if (!checked_add(cursor, gc_bytes, &cursor)) return false;
    layout->phase_offset = cursor;
    layout->h_offset = cursor;
    if (!checked_add(cursor, h_bytes, &layout->v_new_offset)) return false;
    if (!checked_add(layout->phase_offset, phase_bytes, &cursor)) return false;
    layout->log_offset = cursor;
    if (!checked_add(cursor, log_bytes, &layout->total_bytes)) return false;
    return true;
}

bool aligned(const void* ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

int c143_launch_core(
    float* state_kv, const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const float* log_gate, const float* beta,
    __nv_bfloat16* output, void* workspace, unsigned int seq_len,
    unsigned int q_stride, unsigned int k_stride, unsigned int v_stride,
    unsigned int log_gate_stride, unsigned int beta_stride, void* stream_raw) {
    const unsigned int nt = (seq_len + CHUNK - 1) / CHUNK;
    char* cursor = static_cast<char*>(workspace);
    const size_t base = static_cast<size_t>(nt) * C143_VALUE_HEADS;
    __nv_bfloat16* w = reinterpret_cast<__nv_bfloat16*>(cursor);
    cursor += base * CHUNK * K_DIM * sizeof(__nv_bfloat16);
    __nv_bfloat16* u = reinterpret_cast<__nv_bfloat16*>(cursor);
    cursor += base * CHUNK * V_DIM * sizeof(__nv_bfloat16);
    float* gc = reinterpret_cast<float*>(cursor);
    cursor += base * CHUNK * sizeof(float);
    __nv_bfloat16* s = reinterpret_cast<__nv_bfloat16*>(cursor);
    cursor += base * K_DIM * V_DIM * sizeof(__nv_bfloat16);
    __nv_bfloat16* uc = reinterpret_cast<__nv_bfloat16*>(cursor);

    cudaError_t status = cudaFuncSetAttribute(
        gated_delta_rule_recompute_wu,
        cudaFuncAttributeMaxDynamicSharedMemorySize, C143_WU_SMEM);
    if (status != cudaSuccess) return c143_fail(status, "set recompute_wu smem");
    status = cudaFuncSetAttribute(gated_delta_rule_chunk_delta_h,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  C143_DH_SMEM);
    if (status != cudaSuccess) return c143_fail(status, "set delta_h smem");
    status = cudaFuncSetAttribute(gated_delta_rule_chunk_fwd_o,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  C143_OUT_SMEM);
    if (status != cudaSuccess) return c143_fail(status, "set fwd_o smem");

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    gated_delta_rule_recompute_wu<<<dim3(nt, C143_VALUE_HEADS, 1), 128,
                                      C143_WU_SMEM, stream>>>(
        key, value, log_gate, beta, w, u, gc, 1, seq_len, nt, C143_KEY_HEADS,
        C143_VALUE_HEADS, C143_DIM, C143_DIM, k_stride, v_stride,
        log_gate_stride, beta_stride, nullptr, nullptr, 0);
    status = cudaGetLastError();
    if (status != cudaSuccess) return c143_fail(status, "launch recompute_wu");
    gated_delta_rule_chunk_delta_h<<<dim3(C143_VALUE_HEADS, 1, 1), 128,
                                      C143_DH_SMEM, stream>>>(
        state_kv, w, u, key, log_gate, gc, s, uc, 1, seq_len, nt,
        C143_KEY_HEADS, C143_VALUE_HEADS, C143_DIM, C143_DIM, k_stride,
        log_gate_stride);
    status = cudaGetLastError();
    if (status != cudaSuccess) return c143_fail(status, "launch delta_h");
    gated_delta_rule_chunk_fwd_o<<<dim3(nt, C143_VALUE_HEADS, 1), 128,
                                    C143_OUT_SMEM, stream>>>(
        query, key, log_gate, gc, s, uc, output, 1, seq_len, nt,
        C143_KEY_HEADS, C143_VALUE_HEADS, C143_DIM, C143_DIM, q_stride,
        k_stride, log_gate_stride, nullptr, nullptr, 0);
    status = cudaGetLastError();
    if (status != cudaSuccess) return c143_fail(status, "launch fwd_o");
    std::snprintf(c143_error, sizeof(c143_error), "ok");
    return 0;
}

int c143_launch_v3_core(
    float* state_kv, const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const float* gate_beta,
    __nv_bfloat16* output, void* workspace, unsigned int seq_len,
    unsigned int query_stride, unsigned int key_stride,
    unsigned int value_stride, unsigned int gate_beta_stride,
    const C143V3Layout& layout, void* stream_raw) {
    char* base = static_cast<char*>(workspace);
    __nv_bfloat16* w = reinterpret_cast<__nv_bfloat16*>(base + layout.w_offset);
    __nv_bfloat16* u = reinterpret_cast<__nv_bfloat16*>(base + layout.u_offset);
    float* gc = reinterpret_cast<float*>(base + layout.gc_offset);
    __nv_bfloat16* phase = reinterpret_cast<__nv_bfloat16*>(base + layout.phase_offset);
    __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(base + layout.h_offset);
    __nv_bfloat16* v_new = reinterpret_cast<__nv_bfloat16*>(base + layout.v_new_offset);
    float* log_gate = reinterpret_cast<float*>(base + layout.log_offset);
    const float* beta = gate_beta + C143_VALUE_HEADS;
    const unsigned int nt = (seq_len + CHUNK - 1) / CHUNK;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);

    cudaError_t status = cudaFuncSetAttribute(
        gated_delta_rule_c143_v3_kkt_solve,
        cudaFuncAttributeMaxDynamicSharedMemorySize, C143_V3_KKT_SMEM);
    if (status != cudaSuccess) return c143_fail(status, "set v3 kkt_solve smem");
    status = cudaFuncSetAttribute(gated_delta_rule_c143_v3_fwd_o,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  C143_V3_OUT_SMEM);
    if (status != cudaSuccess) return c143_fail(status, "set v3 fwd_o smem");

    const unsigned long long count =
        static_cast<unsigned long long>(seq_len) * C143_VALUE_HEADS;
    const unsigned int blocks = static_cast<unsigned int>((count + 255) / 256);
    gated_delta_rule_alpha_to_log_v2<<<blocks, 256, 0, stream>>>(
        gate_beta, log_gate, seq_len, gate_beta_stride);
    status = cudaGetLastError();
    if (status != cudaSuccess) return c143_fail(status, "launch v3 alpha_to_log");
    gated_delta_rule_c143_v3_kkt_solve<<<dim3(nt, C143_VALUE_HEADS, 1), 128,
                                            C143_V3_KKT_SMEM, stream>>>(
        key, log_gate, beta, gc, phase, seq_len, key_stride,
        C143_VALUE_HEADS, gate_beta_stride);
    status = cudaGetLastError();
    if (status != cudaSuccess) return c143_fail(status, "launch v3 kkt_solve");
    gated_delta_rule_c143_v3_recompute_wu<<<dim3(nt, C143_VALUE_HEADS, 1), 128,
                                               C143_V3_WU_SMEM, stream>>>(
        key, value, beta, gc, phase, w, u, seq_len, key_stride, value_stride,
        gate_beta_stride);
    status = cudaGetLastError();
    if (status != cudaSuccess) return c143_fail(status, "launch v3 recompute_wu");
    gated_delta_rule_c143_v3_delta_h<<<dim3(4, C143_VALUE_HEADS, 1), 128,
                                          C143_V3_DH_SMEM, stream>>>(
        state_kv, key, w, u, gc, h, v_new, seq_len, key_stride);
    status = cudaGetLastError();
    if (status != cudaSuccess) return c143_fail(status, "launch v3 delta_h");
    gated_delta_rule_c143_v3_fwd_o<<<dim3(2, nt, C143_VALUE_HEADS), 128,
                                        C143_V3_OUT_SMEM, stream>>>(
        query, key, gc, h, v_new, output, seq_len, query_stride, key_stride);
    status = cudaGetLastError();
    if (status != cudaSuccess) return c143_fail(status, "launch v3 fwd_o");
    std::snprintf(c143_error, sizeof(c143_error), "ok");
    return 0;
}
}  // namespace

extern "C" const char* atlas_gdn_c143_last_error() { return c143_error; }

extern "C" const char* atlas_gdn_c143_abi_identity() {
    return "atlas-gdn-c143-abi-v3:qwen38-c1-b1-workspace-v3:"
           ATLAS_GDN_C143_SOURCE_SHA256;
}

extern "C" size_t atlas_gdn_c143_workspace_size(unsigned int seq_len) {
    size_t required = 0;
    return c143_workspace_v1(seq_len, &required) ? required : 0;
}

extern "C" size_t atlas_gdn_c143_workspace_size_v2(unsigned int seq_len) {
    size_t required = 0;
    return c143_workspace_v2(seq_len, &required) ? required : 0;
}

extern "C" size_t atlas_gdn_c143_workspace_size_v3(unsigned int seq_len) {
    C143V3Layout layout{};
    return c143_workspace_v3(seq_len, &layout) ? layout.total_bytes : 0;
}

extern "C" int atlas_gdn_c143_launch(
    float* state_kv, const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const float* log_gate, const float* beta,
    __nv_bfloat16* output, void* workspace, size_t workspace_bytes,
    unsigned int seq_len, void* stream_raw) {
    if (!state_kv || !query || !key || !value || !log_gate || !beta ||
        !output || !workspace || seq_len == 0) {
        return c143_arg_fail(-1, "invalid null/zero argument");
    }
    size_t required = 0;
    if (!c143_workspace_v1(seq_len, &required)) {
        return c143_arg_fail(-2, "workspace size overflow");
    }
    if (workspace_bytes < required) {
        std::snprintf(c143_error, sizeof(c143_error),
                      "workspace too small: have=%zu need=%zu", workspace_bytes,
                      required);
        return -3;
    }
    if (!aligned(state_kv, alignof(float)) ||
        !aligned(log_gate, alignof(float)) || !aligned(beta, alignof(float)) ||
        !aligned(query, alignof(__nv_bfloat16)) ||
        !aligned(key, alignof(__nv_bfloat16)) ||
        !aligned(value, alignof(__nv_bfloat16)) ||
        !aligned(output, alignof(__nv_bfloat16)) || !aligned(workspace, 16)) {
        return c143_arg_fail(-4, "misaligned pointer");
    }
    return c143_launch_core(
        state_kv, query, key, value, log_gate, beta, output, workspace, seq_len,
        C143_Q_WIDTH, C143_K_WIDTH, C143_V_WIDTH, C143_VALUE_HEADS,
        C143_VALUE_HEADS, stream_raw);
}

extern "C" int atlas_gdn_c143_launch_v2(
    float* state_kv, const __nv_bfloat16* qkv_base,
    size_t query_base_offset, size_t key_base_offset, size_t value_base_offset,
    unsigned int query_row_stride, unsigned int key_row_stride,
    unsigned int value_row_stride, const float* gate_beta,
    unsigned int gate_beta_row_stride, __nv_bfloat16* output, void* workspace,
    size_t workspace_bytes, unsigned int seq_len, void* stream_raw) {
    if (!state_kv || !qkv_base || !gate_beta || !output || !workspace ||
        seq_len == 0) {
        return c143_arg_fail(-1, "v2 invalid null/zero argument");
    }
    size_t required_v1 = 0;
    size_t required_v2 = 0;
    if (!c143_workspace_v1(seq_len, &required_v1) ||
        !c143_workspace_v2(seq_len, &required_v2)) {
        return c143_arg_fail(-2, "v2 workspace size overflow");
    }
    if (workspace_bytes < required_v2) {
        std::snprintf(c143_error, sizeof(c143_error),
                      "v2 workspace too small: have=%zu need=%zu",
                      workspace_bytes, required_v2);
        return -3;
    }
    if (query_base_offset != 0 || key_base_offset != C143_Q_WIDTH ||
        value_base_offset != C143_Q_WIDTH + C143_K_WIDTH ||
        query_row_stride != C143_QKV_WIDTH ||
        key_row_stride != C143_QKV_WIDTH ||
        value_row_stride != C143_QKV_WIDTH ||
        gate_beta_row_stride != C143_GATE_BETA_WIDTH) {
        return c143_arg_fail(-4, "v2 requires exact Qwen3.8 C1 production layout");
    }
    if (!aligned(state_kv, alignof(float)) ||
        !aligned(qkv_base, alignof(__nv_bfloat16)) ||
        !aligned(gate_beta, alignof(float)) ||
        !aligned(output, alignof(__nv_bfloat16)) || !aligned(workspace, 16)) {
        return c143_arg_fail(-5, "v2 misaligned pointer");
    }

    const __nv_bfloat16* query = qkv_base + query_base_offset;
    const __nv_bfloat16* key = qkv_base + key_base_offset;
    const __nv_bfloat16* value = qkv_base + value_base_offset;
    float* log_gate = reinterpret_cast<float*>(
        static_cast<char*>(workspace) + required_v1);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const unsigned long long count =
        static_cast<unsigned long long>(seq_len) * C143_VALUE_HEADS;
    const unsigned int blocks = static_cast<unsigned int>((count + 255) / 256);
    gated_delta_rule_alpha_to_log_v2<<<blocks, 256, 0, stream>>>(
        gate_beta, log_gate, seq_len, gate_beta_row_stride);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return c143_fail(status, "launch alpha_to_log_v2");

    return c143_launch_core(
        state_kv, query, key, value, log_gate,
        gate_beta + C143_VALUE_HEADS, output, workspace, seq_len,
        query_row_stride, key_row_stride, value_row_stride, C143_VALUE_HEADS,
        gate_beta_row_stride, stream_raw);
}

extern "C" int atlas_gdn_c143_launch_v3(
    float* state_kv, const __nv_bfloat16* qkv_base,
    size_t query_base_offset, size_t key_base_offset, size_t value_base_offset,
    unsigned int query_row_stride, unsigned int key_row_stride,
    unsigned int value_row_stride, const float* gate_beta,
    unsigned int gate_beta_row_stride, __nv_bfloat16* output, void* workspace,
    size_t workspace_bytes, unsigned int seq_len, void* stream_raw) {
    if (!state_kv || !qkv_base || !gate_beta || !output || !workspace ||
        !stream_raw || seq_len == 0) {
        return c143_arg_fail(-1, "v3 invalid null/zero argument");
    }
    C143V3Layout layout{};
    if (!c143_workspace_v3(seq_len, &layout)) {
        return c143_arg_fail(-2, "v3 workspace size overflow");
    }
    if (workspace_bytes < layout.total_bytes) {
        std::snprintf(c143_error, sizeof(c143_error),
                      "v3 workspace too small: have=%zu need=%zu",
                      workspace_bytes, layout.total_bytes);
        return -3;
    }
    if (query_base_offset != 0 || key_base_offset != C143_Q_WIDTH ||
        value_base_offset != C143_Q_WIDTH + C143_K_WIDTH ||
        query_row_stride != C143_QKV_WIDTH ||
        key_row_stride != C143_QKV_WIDTH ||
        value_row_stride != C143_QKV_WIDTH ||
        gate_beta_row_stride != C143_GATE_BETA_WIDTH) {
        return c143_arg_fail(-4, "v3 requires exact Qwen3.8 C1 production layout");
    }
    if (!aligned(state_kv, alignof(float)) ||
        !aligned(qkv_base, alignof(__nv_bfloat16)) ||
        !aligned(gate_beta, alignof(float)) ||
        !aligned(output, alignof(__nv_bfloat16)) || !aligned(workspace, 16)) {
        return c143_arg_fail(-5, "v3 misaligned pointer");
    }
    const __nv_bfloat16* query = qkv_base + query_base_offset;
    const __nv_bfloat16* key = qkv_base + key_base_offset;
    const __nv_bfloat16* value = qkv_base + value_base_offset;
    return c143_launch_v3_core(
        state_kv, query, key, value, gate_beta, output, workspace, seq_len,
        query_row_stride, key_row_stride, value_row_stride,
        gate_beta_row_stride, layout, stream_raw);
}
