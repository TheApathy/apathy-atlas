// SPDX-License-Identifier: AGPL-3.0-only
//
// The V4.1 MoE GEMM, first form: CB3 -> bf16 reconstruct, then cuBLASLt.
//
// This is the piece `deepseek_v41.rs`'s hard stop calls "the MoE GEMM that
// consumes the decoded e2m1 tiles". The shape here is one expert's up
// projection, y = x @ W1^T, with x [M, 5120] bf16 and W1 [2304, 5120] CB3.
//
// Reconstruct-then-cuBLASLt rather than a fused decode+MMA loop, because that
// is what actually won on this box twice: Flash-Next 227 -> 1068 tok/s and GLM
// EXL3 516 -> 744, both "reconstruct + cuBLAS", and cuBLASLt measured 3.5-4.8x
// our own kernels at these shapes. A fused kernel is the OPTIMISATION, and it
// will be validated against this.
//
// Accuracy note. The reconstruct is exact (CB3 decodes to e2m1 grid values
// times a power-of-two UE8M0 scale, all representable in bf16 without
// rounding), so every bit of error in the result comes from the GEMM's own
// bf16 x bf16 -> fp32 accumulation, not from the format. That is why the
// tolerance below is a bf16 GEMM tolerance and not a "quantisation" one.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>

__constant__ float kFp4[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                               -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

#define CUDA_OK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); std::exit(2);} } while(0)
#define LT_OK(x) do { cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ \
    std::fprintf(stderr,"%s:%d cublasLt status %d\n",__FILE__,__LINE__,(int)s_); std::exit(2);} } while(0)

/// CB3 planes -> bf16 [N, K], row-major. See CB3_FORMAT.md for the K-order;
/// this is the same arithmetic as cb3_reconstruct.cu, which is bit-exact
/// against `unpack_cb3_v2`.
/// `WRONG` is the NEGATIVE CONTROL: it flips the r sub-position, producing a
/// permutation of the same multiset along K. That is the ambiguity
/// CB3_FORMAT.md called genuinely undecidable by reading, and it is what a
/// plausible-but-wrong decoder looks like — finite, correctly shaped,
/// meaningless. A gate nobody has watched fail is not evidence.
template <bool WRONG>
__global__ void cb3_to_bf16(const uint8_t* __restrict__ lo, const uint8_t* __restrict__ hi,
                            const uint8_t* __restrict__ cb, const uint8_t* __restrict__ scale,
                            __nv_bfloat16* __restrict__ out, int N, int K) {
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= (long long)N * K) return;
    const int row = (int)(tid / K), k = (int)(tid % K);
    const int block = k / 512, off = k % 512;
    const int g = off / 32, lane = (off % 32) / 2;
    const int r = WRONG ? ((off & 1) ^ 1) : (off & 1);
    const int lo_byte = block * 128 + (g / 2) * 16 + lane;
    const int hi_byte = block * 64 + ((g / 2) / 2) * 16 + lane;
    const uint32_t l2 = (lo[(long long)row * (K / 4) + lo_byte] >> (4 * (g % 2) + 2 * r)) & 3u;
    const uint32_t h1 = (hi[(long long)row * (K / 8) + hi_byte] >> (((g / 2) % 2) * 4 + (g % 2) * 2 + r)) & 1u;
    const uint8_t code = cb[(long long)row * 8 + (l2 | (h1 << 2))] & 0x0Fu;
    out[tid] = __float2bfloat16(kFp4[code] * exp2f((float)scale[(long long)row * (K / 32) + k / 32] - 127.0f));
}

static void* slurp(const char* p, size_t want) {
    FILE* f = std::fopen(p, "rb");
    if (!f) { std::fprintf(stderr, "open %s\n", p); std::exit(2); }
    void* q = std::malloc(want);
    if (std::fread(q, 1, want, f) != want) { std::fprintf(stderr, "short read %s\n", p); std::exit(2); }
    std::fclose(f); return q;
}

int main() {
    int M = 0, N = 0, K = 0;
    { FILE* f = std::fopen("gx_dims.txt", "r");
      if (!f || std::fscanf(f, "%d %d %d", &M, &N, &K) != 3) { std::fprintf(stderr, "gx_dims.txt\n"); return 2; }
      std::fclose(f); }
    std::printf("M=%d N=%d K=%d\n", M, N, K);

    uint8_t* h_lo = (uint8_t*)slurp("gx_lo.bin", (size_t)N * (K / 4));
    uint8_t* h_hi = (uint8_t*)slurp("gx_hi.bin", (size_t)N * (K / 8));
    uint8_t* h_cb = (uint8_t*)slurp("gx_cb.bin", (size_t)N * 8);
    uint8_t* h_s  = (uint8_t*)slurp("gx_s.bin",  (size_t)N * (K / 32));
    uint16_t* h_x = (uint16_t*)slurp("gx_x.bin", (size_t)M * K * 2);
    float* h_wref = (float*)slurp("gx_w.bin", (size_t)N * K * 4);
    float* h_yref = (float*)slurp("gx_y.bin", (size_t)M * N * 4);

    uint8_t *d_lo,*d_hi,*d_cb,*d_s; __nv_bfloat16 *d_w,*d_x; float* d_y;
    CUDA_OK(cudaMalloc(&d_lo,(size_t)N*(K/4)));  CUDA_OK(cudaMalloc(&d_hi,(size_t)N*(K/8)));
    CUDA_OK(cudaMalloc(&d_cb,(size_t)N*8));      CUDA_OK(cudaMalloc(&d_s,(size_t)N*(K/32)));
    CUDA_OK(cudaMalloc(&d_w,(size_t)N*K*2));     CUDA_OK(cudaMalloc(&d_x,(size_t)M*K*2));
    CUDA_OK(cudaMalloc(&d_y,(size_t)M*N*4));
    CUDA_OK(cudaMemcpy(d_lo,h_lo,(size_t)N*(K/4),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_hi,h_hi,(size_t)N*(K/8),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_cb,h_cb,(size_t)N*8,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_s,h_s,(size_t)N*(K/32),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_x,h_x,(size_t)M*K*2,cudaMemcpyHostToDevice));

    const long long tot = (long long)N * K;
    cb3_to_bf16<false><<<(int)((tot+255)/256),256>>>(d_lo,d_hi,d_cb,d_s,d_w,N,K);
    CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());

    // Gate 1: the reconstruct itself, against the unpacker's weights.
    {
        __nv_bfloat16* h_w = (__nv_bfloat16*)std::malloc((size_t)tot*2);
        CUDA_OK(cudaMemcpy(h_w,d_w,(size_t)tot*2,cudaMemcpyDeviceToHost));
        long long bad = 0;
        for (long long i=0;i<tot;++i) if (__bfloat162float(h_w[i]) != h_wref[i]) ++bad;
        std::printf("reconstruct: %lld weights, %lld differ from unpack_cb3_v2\n", tot, bad);
        if (bad) { std::printf("FAIL reconstruct\n"); return 1; }
        std::free(h_w);
    }

    // Gate 2: the GEMM. y[M,N] = x[M,K] @ W[N,K]^T, fp32 accumulate.
    // cuBLASLt is column-major; computing y^T = W * x^T in its terms gives a
    // row-major [M,N] result without a transpose of the output.
    cublasLtHandle_t lt; LT_OK(cublasLtCreate(&lt));
    cublasLtMatmulDesc_t op; LT_OK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    LT_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
    LT_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
    cublasLtMatrixLayout_t lw, lx, ly;
    LT_OK(cublasLtMatrixLayoutCreate(&lw, CUDA_R_16BF, K, N, K));   // W^T: K x N, ld K
    LT_OK(cublasLtMatrixLayoutCreate(&lx, CUDA_R_16BF, K, M, K));   // x^T: K x M, ld K
    LT_OK(cublasLtMatrixLayoutCreate(&ly, CUDA_R_32F,  N, M, N));   // y^T: N x M, ld N
    const float alpha = 1.0f, beta = 0.0f;
    LT_OK(cublasLtMatmul(lt, op, &alpha, d_w, lw, d_x, lx, &beta, d_y, ly, d_y, ly,
                         nullptr, nullptr, 0, 0));
    CUDA_OK(cudaDeviceSynchronize());

    float* h_y = (float*)std::malloc((size_t)M*N*4);
    CUDA_OK(cudaMemcpy(h_y,d_y,(size_t)M*N*4,cudaMemcpyDeviceToHost));
    double num=0, den=0, worst=0;
    for (long long i=0;i<(long long)M*N;++i) {
        double d = (double)h_y[i]-(double)h_yref[i];
        num += d*d; den += (double)h_yref[i]*(double)h_yref[i];
        if (std::fabs(d) > worst) worst = std::fabs(d);
    }
    const double rel = std::sqrt(num/den);
    std::printf("gemm: rel_l2=%.3e worst_abs=%.3e over %d x %d\n", rel, worst, M, N);

    // NEGATIVE CONTROL: same pipeline, r sub-position flipped. Must be far away.
    cb3_to_bf16<true><<<(int)((tot+255)/256),256>>>(d_lo,d_hi,d_cb,d_s,d_w,N,K);
    CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
    LT_OK(cublasLtMatmul(lt, op, &alpha, d_w, lw, d_x, lx, &beta, d_y, ly, d_y, ly,
                         nullptr, nullptr, 0, 0));
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(h_y,d_y,(size_t)M*N*4,cudaMemcpyDeviceToHost));
    double cnum=0;
    for (long long i=0;i<(long long)M*N;++i) {
        double d=(double)h_y[i]-(double)h_yref[i]; cnum += d*d;
    }
    const double crel = std::sqrt(cnum/den);
    std::printf("control (r flipped): rel_l2=%.3e  -> separation %.0fx\n", crel, crel/rel);
    // bf16 inputs with fp32 accumulate over K=5120: ~1e-3 is the floor. A wrong
    // K-order lands at ~1.3 (measured, korder_control.py), so this threshold
    // separates the two by three orders of magnitude.
    // The format contributes ZERO error: CB3 values are e2m1 (one mantissa bit)
    // times a power-of-two scale, so bf16 holds them exactly and only fp32
    // accumulation order remains. Measured 2.6e-07, not the 1e-3 a genuine
    // bf16 rounding path would give. The control lands near 1.0.
    const bool ok = rel < 1e-5 && crel > 0.2;
    if (!ok && crel <= 0.2)
        std::printf("INCONCLUSIVE: the wrong K-order also passed — this gate cannot fail\n");
    std::printf(ok ? "PASS cb3 -> bf16 -> cuBLASLt (control separates)\n" : "FAIL\n");
    return ok ? 0 : 1;
}
