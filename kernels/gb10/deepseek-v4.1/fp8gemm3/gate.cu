// SPDX-License-Identifier: AGPL-3.0-only
//
// v3 fused FP8 GEMM sweep + gate against the CURRENT dense path (dequant to bf16, then one
// no-split-K cuBLASLt bf16 GEMM, fp32 compute) on real layer-2 weights (attention2's fixtures).
//
//   nvcc -O3 -std=c++17 -arch=sm_121a --fmad=false gate.cu -lcublasLt -o gate
//   ./gate <fixture dir>...        (each at M = 512 and M = 2048; A rows tiled from the 512)
//
// PRE-REGISTERED (before the first run): every config BYTE-IDENTICAL to the current path at both
// M (100.0000%), rows 0..19 at M=20 byte-identical to the big-M rows, scale+1 control rel ~1.
// Speed target: the best config within 15% of "gemm alone" (cuBLAS on the pre-dequantized bf16
// weight) on the large-N/large-K shapes (wq_b, wo_b, w1, w2) at M=2048.

#define DSV41_FP8GEMM_GATE
#include "../cb3/dsv41_fp8_gemm.cu"
#include "fp8_gemm_v3.cuh"

DSV41_FP8GEMM5_ENTRY(v5_e3bf, 256, 128, 32, 3, 4, 2, false)
DSV41_FP8GEMM5_ENTRY(v5_e1bf, 128, 256, 32, 3, 2, 4, false)
DSV41_FP8GEMM5_ENTRY(v5_e4, 256, 128, 32, 3, 4, 2, true)
DSV41_FP8GEMM6_ENTRY(v6_f1, 256, 128, 32, 3, 4, 2)

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

typedef void (*Kern)(const __nv_bfloat16*, int, const uint8_t*, const uint8_t*, int, __nv_bfloat16*, int, int, int, int);
struct V3 { const char* name; Kern k; int bm, bn, threads, smem; bool bf16b; bool probe = false; };
#define V5E(n, BM, BN, BK, ST, WM, WN, F) V3{#n, n, BM, BN, 32 * WM * WN, dsv41_fp8gemm3::Cfg5<BM, BN, BK, ST, WM, WN, F>::SMEM, !F}
#define V5P(n, BM, BN, BK, ST, WM, WN, P) V3{#n, n, BM, BN, 32 * WM * WN, dsv41_fp8gemm3::Cfg5<BM, BN, BK, ST, WM, WN, true, P>::SMEM, false, true}
#define V6E(n, BM, BN, BK, ST, WM, WN) V3{#n, n, BM, BN, 32 * WM * WN, dsv41_fp8gemm3::Cfg6<BM, BN, BK, ST, WM, WN>::SMEM, false}
#define V7E(n, k, BM, BN, BK, ST, WM, WN) V3{#n, k, BM, BN, 32 * WM * WN, dsv41_fp8gemm7::Cfg6<BM, BN, BK, ST, WM, WN>::SMEM, false}
static const V3 kV3[] = {
    V5E(v5_e3bf, 256, 128, 32, 3, 4, 2, false), V5E(v5_e1bf, 128, 256, 32, 3, 2, 4, false),
    V5E(v5_e4, 256, 128, 32, 3, 4, 2, true), V6E(v6_f1, 256, 128, 32, 3, 4, 2),
    V7E(v7_m256, dsv41_fp8_gemm_nt_v7_m256, 256, 128, 32, 3, 4, 2), V7E(v7_n256, dsv41_fp8_gemm_nt_v7_n256, 128, 256, 32, 3, 2, 4),
};

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
        if (std::getenv("GATE_ALGO")) {
            int id = -1, tile = -1, stages = -1, splitk = -1, swz = -1, cta = -1, inner = -1; size_t w = 0;
            cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, &w);
            cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile, &w);
            cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof stages, &w);
            cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof splitk, &w);
            cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof swz, &w);
            cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cta, sizeof cta, &w);
            cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner, sizeof inner, &w);
            std::printf("  cublasLt algo m=%d n=%d k=%d: id %d tile %d stages %d splitk %d swizzle %d cluster %d inner %d\n", m, n, k, id, tile, stages, splitk, swz, cta, inner);
        }
        const float one = 1.f, zero = 0.f;
        LK(cublasLtMatmul(h, d, &one, w, la, a, lb, &zero, out, lc, out, lc, &r.algo, ws, wss, 0));
        cublasLtMatmulPreferenceDestroy(p); cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(d);
    }
};

int main(int argc, char** argv) {
    for (const auto& v : kV3) CK(cudaFuncSetAttribute(v.k, cudaFuncAttributeMaxDynamicSharedMemorySize, v.smem));
    Lt lt;
    int fails = 0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto time_it = [&](auto&& f) { for (int i = 0; i < 3; ++i) f(); CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
        for (int i = 0; i < 20; ++i) f(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); return ms / 20; };
    if (std::getenv("GATE_ALGO")) {   // one cuBLAS call per shape (for the algo line / nsys kernel names)
        for (int ai = 1; ai < argc; ++ai) {
            int M0, N, K, SN, SK; FILE* f = std::fopen((std::string(argv[ai]) + "/dims.txt").c_str(), "r");
            if (!f || std::fscanf(f, "%d %d %d %d %d", &M0, &N, &K, &SN, &SK) != 5) return 2; std::fclose(f);
            for (int M : {512, 2048}) {
                void *a, *w, *c; CK(cudaMalloc(&a, (size_t)M * K * 2)); CK(cudaMalloc(&w, (size_t)N * K * 2)); CK(cudaMalloc(&c, (size_t)M * N * 2));
                lt.gemm(a, w, c, M, N, K); CK(cudaDeviceSynchronize()); cudaFree(a); cudaFree(w); cudaFree(c);
            }
        }
        return 0;
    }
    for (int ai = 1; ai < argc; ++ai) {
        const std::string dir = argv[ai];
        int M0, N, K, SN, SK;
        { FILE* f = std::fopen((dir + "/dims.txt").c_str(), "r"); if (!f || std::fscanf(f, "%d %d %d %d %d", &M0, &N, &K, &SN, &SK) != 5) return 2; std::fclose(f); }
        auto ha = slurp(dir + "/a.bin");
        auto hs = slurp(dir + "/s.bin");
        uint8_t* W = up<uint8_t>(slurp(dir + "/w.bin"));
        uint8_t* S = up<uint8_t>(hs);
        std::vector<char> hs2(hs); for (auto& c : hs2) c = (char)((uint8_t)c + 1);
        uint8_t* S2 = up<uint8_t>(hs2);
        __nv_bfloat16* Wb; CK(cudaMalloc(&Wb, (size_t)N * K * 2));
        for (int M : {512, 2048}) {
            std::vector<char> hA((size_t)M * K * 2);
            for (int r = 0; r < M; ++r) std::memcpy(&hA[(size_t)r * K * 2], &ha[(size_t)(r % M0) * K * 2], (size_t)K * 2);
            __nv_bfloat16* A = up<__nv_bfloat16>(hA);
            __nv_bfloat16 *Cref, *Cf; CK(cudaMalloc(&Cref, (size_t)M * N * 2)); CK(cudaMalloc(&Cf, (size_t)M * N * 2));
            auto current = [&] { dequant<<<(unsigned)(((size_t)N * K + 255) / 256), 256>>>(W, S, SK, Wb, N, K); lt.gemm(A, Wb, Cref, M, N, K); };
            current(); CK(cudaDeviceSynchronize());
            std::vector<uint16_t> r((size_t)M * N); CK(cudaMemcpy(r.data(), Cref, r.size() * 2, cudaMemcpyDeviceToHost));
            const float t_cur = time_it(current);
            const float t_gemm = time_it([&] { lt.gemm(A, Wb, Cref, M, N, K); });
            const float t_v2 = time_it([&] {
                dsv41_fp8_gemm_nt_v2<<<dim3(N / dsv41_fp8gemm::BN, (M + dsv41_fp8gemm::BM - 1) / dsv41_fp8gemm::BM), dsv41_fp8gemm::THREADS>>>(A, K, W, S, SK, Cf, N, M, N, K); });
            const double fl = 2.0 * M * N * K;
            std::printf("%-9s M=%4d N=%5d K=%5d | current %.3f ms | gemm alone %.3f ms (%.1f TF/s) | v2 %.3f ms (%.1f TF/s)\n",
                        dir.substr(dir.rfind('/') + 1).c_str(), M, N, K, t_cur, t_gemm, fl / (t_gemm * 1e9), t_v2, fl / (t_v2 * 1e9));
            for (const auto& v : kV3) {
                if (N % v.bn) { std::printf("    %-6s skipped (N %% %d)\n", v.name, v.bn); continue; }
                auto run = [&](const uint8_t* s, int m, __nv_bfloat16* c) {
                    v.k<<<dim3((N / v.bn) * ((m + v.bm - 1) / v.bm)), v.threads, v.smem>>>(A, K, v.bf16b ? (const uint8_t*)Wb : W, s, SK, c, N, m, N, K); };
                CK(cudaMemset(Cf, 0, (size_t)M * N * 2));
                run(S, M, Cf); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                std::vector<uint16_t> f((size_t)M * N); CK(cudaMemcpy(f.data(), Cf, f.size() * 2, cudaMemcpyDeviceToHost));
                const bool ident = std::memcmp(f.data(), r.data(), f.size() * 2) == 0;
                size_t same = 0; for (size_t i = 0; i < f.size(); ++i) same += f[i] == r[i];
                CK(cudaMemset(Cf, 0, (size_t)M * N * 2)); run(S, 20, Cf); CK(cudaDeviceSynchronize());
                std::vector<uint16_t> f20((size_t)20 * N); CK(cudaMemcpy(f20.data(), Cf, f20.size() * 2, cudaMemcpyDeviceToHost));
                const bool inv = std::memcmp(f20.data(), r.data(), f20.size() * 2) == 0;
                run(S2, M, Cf); CK(cudaDeviceSynchronize());
                std::vector<uint16_t> fc((size_t)M * N); CK(cudaMemcpy(fc.data(), Cf, fc.size() * 2, cudaMemcpyDeviceToHost));
                size_t csame = 0; for (size_t i = 0; i < fc.size(); ++i) csame += fc[i] == r[i];
                const bool ctrl = v.bf16b || csame < fc.size() / 2;   // the bf16 control reads no scales
                // Tails (M = 1/4/16/20): rows byte-identical to the big-M rows, nothing stored past
                // row M (0xFFFF sentinel). CONTROL: launching M+1 rows must be caught.
                auto tail_ok = [&](int ms, int launch_m) {
                    CK(cudaMemset(Cf, 0xFF, (size_t)M * N * 2)); run(S, launch_m, Cf); CK(cudaDeviceSynchronize());
                    std::vector<uint16_t> t((size_t)M * N); CK(cudaMemcpy(t.data(), Cf, t.size() * 2, cudaMemcpyDeviceToHost));
                    bool ok = std::memcmp(t.data(), r.data(), (size_t)ms * N * 2) == 0;
                    for (size_t i = (size_t)ms * N; i < t.size() && ok; ++i) ok = t[i] == 0xFFFF;
                    return ok; };
                bool tails = true;
                for (int ms : {1, 4, 16, 20}) tails = tails && tail_ok(ms, ms);
                const bool tail_ctrl = !tail_ok(4, 5);
                const float t = time_it([&] { run(S, M, Cf); });
                const bool ok = v.probe || (ident && inv && ctrl && tails && tail_ctrl);
                if (!tails || !tail_ctrl) std::printf("    %-6s TAILS %s CTRL %s\n", v.name, tails ? "ok" : "FAIL", tail_ctrl ? "caught" : "MISSED");   // probes compute wrong values by design
                fails += !ok;
                std::printf("    %-6s %.3f ms %5.1f TF/s  %.2fx gemm-alone %.2fx current | identical %.4f%% M=20 rows %s CTRL scale+1 caught %s %s\n",
                            v.name, t, fl / (t * 1e9), t_gemm / t, t_cur / t, 100.0 * same / f.size(), inv ? "yes" : "NO", ctrl ? "yes" : "NO", ok ? "PASS" : "FAIL");
            }
            cudaFree(A); cudaFree(Cref); cudaFree(Cf);
        }
        cudaFree(W); cudaFree(S); cudaFree(S2); cudaFree(Wb);
    }
    std::printf("GATE %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
