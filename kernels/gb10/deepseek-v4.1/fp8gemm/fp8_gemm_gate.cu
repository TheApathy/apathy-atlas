// SPDX-License-Identifier: AGPL-3.0-only
//
// Gate for cb3/dsv41_fp8_gemm.cu against the CURRENT dense FP8 path (ops.rs linear_fp8 at M>16:
// dequant to a bf16 copy, then ONE cuBLASLt bf16 GEMM, fp32 compute, split-K forbidden), on
// real layer-2 weights (make_fixture.py).
//
//   nvcc -O3 -std=c++17 -arch=sm_121a --fmad=false fp8_gemm_gate.cu -lcublasLt -o fp8_gemm_gate
//   ./fp8_gemm_gate fx/wq_b fx/wo_b fx/w1 fx/wq_a
//
// PRE-REGISTERED (written before the first run):
//   numerics: >= 99.0% of bf16 outputs identical to the current path; rel_l2 vs it <= 1e-3
//             (a different fp32 accumulation order flips bf16 roundings, nothing more);
//   chunk invariance: rows 0..19 computed at M=20 BYTE-IDENTICAL to the same rows at M=512;
//   control: every scale +1 (all weights x2) must move rel_l2 to ~1 -- the gate can fail;
//   speed: reported against dequant+GEMM and GEMM alone at M=512; no speed claim is gated.

#define DSV41_FP8GEMM_GATE
#include "../cb3/dsv41_fp8_gemm.cu"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cublasLt.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)
#define LK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "%s:%d cublasLt %d\n", __FILE__, __LINE__, (int)s_); std::exit(2); } } while (0)

static std::vector<char> slurp(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "open %s\n", p.c_str()); std::exit(2); }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> v(n);
    if (std::fread(v.data(), 1, n, f) != (size_t)n) std::exit(2);
    std::fclose(f);
    return v;
}
template <class T> static T* up(const std::vector<char>& v) { void* d; CK(cudaMalloc(&d, v.size())); CK(cudaMemcpy(d, v.data(), v.size(), cudaMemcpyHostToDevice)); return (T*)d; }

// The current path's dequant: bf16(fp8 * 2^(s-127)), exact.
__global__ void dequant(const uint8_t* W, const uint8_t* S, int s_ld, __nv_bfloat16* out, int N, int K) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)N * K) return;
    const int n = i / K, k = i % K;
    __nv_fp8_e4m3 f; f.__x = W[i];
    out[i] = __float2bfloat16_rn(float(f) * exp2f((float)S[(size_t)(n / 32) * s_ld + k / 32] - 127.f));
}

struct Lt {
    cublasLtHandle_t h; void* ws; size_t wss = 32 << 20;
    Lt() { LK(cublasLtCreate(&h)); CK(cudaMalloc(&ws, wss)); }
    // Row-major out[m,n] = a[m,k] @ w[n,k]^T, bf16 in/out, fp32 compute, NO split-K (typed_ex).
    void gemm(const void* a, const void* w, void* out, int m, int n, int k) {
        cublasLtMatmulDesc_t d; LK(cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        LK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
        LK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
        cublasLtMatrixLayout_t la, lb, lc;
        LK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, k, n, k));
        LK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, k, m, k));
        LK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, n, m, n));
        cublasLtMatmulPreference_t p; LK(cublasLtMatmulPreferenceCreate(&p));
        LK(cublasLtMatmulPreferenceSetAttribute(p, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wss, sizeof wss));
        uint32_t none = 0;
        LK(cublasLtMatmulPreferenceSetAttribute(p, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &none, sizeof none));
        cublasLtMatmulHeuristicResult_t r; int got = 0;
        LK(cublasLtMatmulAlgoGetHeuristic(h, d, la, lb, lc, lc, p, 1, &r, &got));
        const float one = 1.f, zero = 0.f;
        LK(cublasLtMatmul(h, d, &one, w, la, a, lb, &zero, out, lc, out, lc, &r.algo, ws, wss, 0));
        cublasLtMatmulPreferenceDestroy(p); cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(d);
    }
};

static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
    Lt lt;
    int fails = 0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto time_it = [&](auto&& f) { for (int i = 0; i < 3; ++i) f(); CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
        for (int i = 0; i < 20; ++i) f(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); return ms / 20; };
    for (int ai = 1; ai < argc; ++ai) {
        const std::string dir = argv[ai];
        int M, N, K, SN, SK;
        { FILE* f = std::fopen((dir + "/dims.txt").c_str(), "r"); if (!f || std::fscanf(f, "%d %d %d %d %d", &M, &N, &K, &SN, &SK) != 5) return 2; std::fclose(f); }
        auto hs = slurp(dir + "/s.bin");
        __nv_bfloat16* A = up<__nv_bfloat16>(slurp(dir + "/a.bin"));
        uint8_t* W = up<uint8_t>(slurp(dir + "/w.bin"));
        uint8_t* S = up<uint8_t>(hs);
        std::vector<char> hs2(hs); for (auto& c : hs2) c = (char)((uint8_t)c + 1);
        uint8_t* S2 = up<uint8_t>(hs2);
        __nv_bfloat16 *Wb, *Cref, *Cf;
        CK(cudaMalloc(&Wb, (size_t)N * K * 2)); CK(cudaMalloc(&Cref, (size_t)M * N * 2)); CK(cudaMalloc(&Cf, (size_t)M * N * 2));
        auto current = [&] { dequant<<<(unsigned)(((size_t)N * K + 255) / 256), 256>>>(W, S, SK, Wb, N, K); lt.gemm(A, Wb, Cref, M, N, K); };
        auto fused = [&](const uint8_t* s, int m, __nv_bfloat16* c) {
            dsv41_fp8_gemm_nt<<<dim3(N / dsv41_fp8gemm::BN, (m + dsv41_fp8gemm::BM - 1) / dsv41_fp8gemm::BM), dsv41_fp8gemm::THREADS>>>(A, K, W, s, SK, c, N, m, N, K);
        };
        current(); fused(S, M, Cf);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<uint16_t> r((size_t)M * N), f((size_t)M * N);
        CK(cudaMemcpy(r.data(), Cref, r.size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(f.data(), Cf, f.size() * 2, cudaMemcpyDeviceToHost));
        auto cmp = [&](const std::vector<uint16_t>& x, const std::vector<uint16_t>& y, size_t n, double& rel) {
            size_t same = 0; double num = 0, den = 0;
            for (size_t i = 0; i < n; ++i) { same += x[i] == y[i]; const double a = bf2f(x[i]), b = bf2f(y[i]); num += (a - b) * (a - b); den += b * b; }
            rel = std::sqrt(num / den); return (double)same / n; };
        double rel; const double same = cmp(f, r, f.size(), rel);
        // chunk invariance: first 20 rows at M=20
        CK(cudaMemset(Cf, 0, (size_t)M * N * 2)); fused(S, 20, Cf); CK(cudaDeviceSynchronize());
        std::vector<uint16_t> f20((size_t)20 * N); CK(cudaMemcpy(f20.data(), Cf, f20.size() * 2, cudaMemcpyDeviceToHost));
        const bool inv = std::memcmp(f20.data(), f.data(), f20.size() * 2) == 0;
        // control: every scale +1
        fused(S2, M, Cf); CK(cudaDeviceSynchronize());
        std::vector<uint16_t> fc((size_t)M * N); CK(cudaMemcpy(fc.data(), Cf, fc.size() * 2, cudaMemcpyDeviceToHost));
        double crel; cmp(fc, r, fc.size(), crel);
        const float t_cur = time_it(current);
        const float t_gemm = time_it([&] { lt.gemm(A, Wb, Cref, M, N, K); });
        const float t_fused = time_it([&] { fused(S, M, Cf); });
        const double tf = 2.0 * M * N * K / (t_fused * 1e9);
        // v2 (on-the-fly B, 3 stages): PRE-REGISTERED byte-identical to v1.
        auto fused2 = [&](int m, __nv_bfloat16* c) {
            dsv41_fp8_gemm_nt_v2<<<dim3(N / dsv41_fp8gemm::BN, (m + dsv41_fp8gemm::BM - 1) / dsv41_fp8gemm::BM), dsv41_fp8gemm::THREADS>>>(A, K, W, S, SK, c, N, m, N, K);
        };
        fused2(M, Cf); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<uint16_t> f2((size_t)M * N); CK(cudaMemcpy(f2.data(), Cf, f2.size() * 2, cudaMemcpyDeviceToHost));
        const bool v2same = std::memcmp(f2.data(), f.data(), f.size() * 2) == 0;
        // PREFILL TAILS (M <= 16 now take this kernel too): for M in {1, 4, 16, 20}, v2's rows
        // must be BYTE-IDENTICAL to the same rows at M=512 and every row >= M must keep its
        // 0xFFFF sentinel (no out-of-bounds store). Control: launch M+1 rows, the sentinel check
        // must see row M written.
        auto tail_ok = [&](int ms, int launch_m) {
            CK(cudaMemset(Cf, 0xFF, (size_t)M * N * 2)); fused2(launch_m, Cf); CK(cudaDeviceSynchronize());
            std::vector<uint16_t> t((size_t)M * N); CK(cudaMemcpy(t.data(), Cf, t.size() * 2, cudaMemcpyDeviceToHost));
            const bool rows = std::memcmp(t.data(), f2.data(), (size_t)ms * N * 2) == 0;
            bool untouched = true;
            for (size_t i = (size_t)ms * N; i < t.size() && untouched; ++i) untouched = t[i] == 0xFFFF;
            return rows && untouched; };
        bool tails = true;
        for (int ms : {1, 4, 16, 20}) tails = tails && tail_ok(ms, ms);
        const bool tail_ctrl = !tail_ok(4, 5);
        std::printf("           v2 tails M=1/4/16/20: rows byte-identical to M=512 and no store past M: %s | CTRL (M+1 launched) caught: %s\n",
                    tails ? "yes" : "NO", tail_ctrl ? "yes" : "NO");
        const float t_v2 = time_it([&] { fused2(M, Cf); });
        std::printf("           v2 (on-the-fly B, 3 stages): byte-identical to v1: %s, %.3f ms (%.1f TF/s, %.2fx current)\n",
                    v2same ? "yes" : "NO", t_v2, 2.0 * M * N * K / (t_v2 * 1e9), t_cur / t_v2);
        const float t_nc = time_it([&] { dsv41_fp8_gemm_probe_noconv<<<dim3(N / dsv41_fp8gemm::BN, (M + dsv41_fp8gemm::BM - 1) / dsv41_fp8gemm::BM), dsv41_fp8gemm::THREADS>>>(A, K, W, S, SK, Cf, N, M, N, K); });
        std::printf("           PROBE v2 without the fp8->bf16 conversion (wrong values, timing only): %.3f ms (%.1f TF/s)\n", t_nc, 2.0 * M * N * K / (t_nc * 1e9));
        const bool ok = same >= 0.99 && rel <= 1e-3 && inv && crel > 0.5 && v2same && tails && tail_ctrl;
        fails += !ok;
        std::printf("%-10s M=%d N=%5d K=%5d | identical %.4f%% rel %.2e | M=20 rows byte-identical: %s | CTRL scale+1 rel %.2e | "
                    "current (dequant+gemm) %.3f ms, gemm alone %.3f ms, FUSED %.3f ms (%.1f TF/s, %.2fx current) %s\n",
                    dir.c_str(), M, N, K, 100 * same, rel, inv ? "yes" : "NO", crel, t_cur, t_gemm, t_fused, tf, t_cur / t_fused, ok ? "PASS" : "FAIL");
        cudaFree(A); cudaFree(W); cudaFree(S); cudaFree(S2); cudaFree(Wb); cudaFree(Cref); cudaFree(Cf);
    }
    std::printf("GATE %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
