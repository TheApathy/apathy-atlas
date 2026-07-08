// SPDX-License-Identifier: AGPL-3.0-only
//
// DECISIVE microbench for the column-sparse NVFP4 GEMV (PROJECT-150 "ghost").
//
// Question: does `w4a16_gemv_sparse_cols` (reads only surviving weight columns
// in a precomputed keep-list) beat the production dense `w4a16_gemv` at the
// MEASURED real activation sparsity (~30% kept), on single-stream decode?
//
// This is the same-author kernel as the just-REFUTED W3 path (w3a16 clones were
// 5x slower than optimized w4a16). Treat skeptically: the sparse kernel is a
// STRUCTURAL half-width clone of dense (reads 4 packed bytes/iter vs dense's 8),
// and its skipped weight reads are STRIDED 4-byte gathers — DRAM sectors are
// 32 B, so skipping 70% of chunks may NOT drop DRAM traffic by 70% (overfetch).
//
// Target shape = Qwen3.6-27B down_proj GEMV (M=1 decode):
//   N = hidden       = 5120
//   K = intermediate = 17408   (K/8 = 2176 k8-chunks per row)
//   NVFP4 packed weight [N, K/2] + FP8-E4M3 scales [N, K/16] + scalar scale2.
//
// We construct the activation so EXACTLY a target fraction of k8-chunks survive
// the thresholder, then sweep kept-fraction ∈ {0.10,0.20,0.30,0.40,0.50,0.60,
// 0.80,0.90,1.00}. 0.30 is the MEASURED real case (70% zero).
//
// Reports per kept-fraction: sparse-GEMV us, keep-build us, sparse-total us,
// dense us, speedup, and effective GB/s (ideal weight bytes at that keep
// fraction / time) so we can see if sparse hits the byte-skip floor or is
// killed by strided overfetch.
//
// BUILD + RUN (GPU window, engine idle):
//   nvcc -arch=sm_121f -O3 --use_fast_math -Xptxas -O3 \
//     -I kernels/gb10/common \
//     local/bench/bench_sparse_cols.cu \
//     -o /tmp/bench_sparse_cols && /tmp/bench_sparse_cols
//
// NOTE: uses runtime API, no server. Pulls the PRODUCTION kernels verbatim via
// #include so we bench exactly the code the engine would ship.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Production kernels, verbatim. Both live in kernels/gb10/common.
#include "w4a16_gemv.cu"            // dense reference: w4a16_gemv(...)
#include "w4a16_gemv_sparse_cols.cu"// sparse: w4a16_gemv_sparse_cols + ffn_build_keep_chunks

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e_)); exit(1); } } while(0)

int main() {
    const unsigned int N = 5120;     // hidden  (output dim)
    const unsigned int K = 17408;    // intermediate (contraction dim)
    const unsigned int half_K = K / 2;
    const unsigned int groups = K / 16;
    const unsigned int K8 = K / 8;   // 2176 k8-chunks per row

    size_t bp_bytes = (size_t)N * half_K;   // packed weight bytes
    size_t bs_bytes = (size_t)N * groups;   // fp8 scale bytes

    // ---- Host inputs (deterministic LCG) ------------------------------
    __nv_bfloat16* hA = (__nv_bfloat16*)malloc((size_t)K * sizeof(__nv_bfloat16));
    uint8_t* hB = (uint8_t*)malloc(bp_bytes);
    uint8_t* hS = (uint8_t*)malloc(bs_bytes);

    uint64_t s = 0x1234567;
    auto rnd = [&]() { s = s * 6364136223846793005ULL + 1442695040888963407ULL;
                       return (uint32_t)(s >> 33); };
    for (size_t i = 0; i < bp_bytes; i++) hB[i] = (uint8_t)rnd();
    for (size_t i = 0; i < bs_bytes; i++) hS[i] = (uint8_t)(0x38 + (rnd() & 0x7));
    const float scale2 = 1.0f / 448.0f;

    // ---- Device buffers -----------------------------------------------
    __nv_bfloat16 *dA, *dC;
    uint8_t *dB, *dS;
    unsigned int *dKeepIdx, *dKeepLen;
    CK(cudaMalloc(&dA, (size_t)K * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dB, bp_bytes));
    CK(cudaMalloc(&dS, bs_bytes));
    CK(cudaMalloc(&dC, (size_t)N * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dKeepIdx, (size_t)K8 * sizeof(unsigned int)));
    CK(cudaMalloc(&dKeepLen, sizeof(unsigned int)));
    CK(cudaMemcpy(dB, hB, bp_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dS, hS, bs_bytes, cudaMemcpyHostToDevice));

    // Build an activation whose thresholder keeps EXACTLY `keep` chunks.
    // Strategy: put a large spike (=1.0) in the first activation of each KEPT
    // chunk, and a tiny value (=1e-4) elsewhere. rowmax=1.0, cut=tau*1.0. With
    // tau=0.01, kept chunks (cmax=1.0>=0.01) survive; others (1e-4<0.01) drop.
    // We choose WHICH chunks are kept by a strided pattern so kept columns are
    // spread across K (realistic: survivors aren't the first contiguous block —
    // that would let the L2 prefetcher hide the strided-read penalty).
    const float tau = 0.01f;
    auto build_activation = [&](unsigned int keep) {
        for (size_t i = 0; i < K; i++) hA[i] = __float2bfloat16(1e-4f);
        if (keep == 0) return;
        // stride so kept chunks are evenly spread across [0, K8)
        // pick chunk indices round((j+0.5)*K8/keep) for j in [0,keep)
        for (unsigned int j = 0; j < keep; j++) {
            unsigned int c = (unsigned int)(((double)j + 0.5) * (double)K8 / (double)keep);
            if (c >= K8) c = K8 - 1;
            hA[(size_t)c * 8] = __float2bfloat16(1.0f); // spike in first slot
        }
        CK(cudaMemcpy(dA, hA, (size_t)K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    };

    dim3 grid((N + 3) / 4, 1, 1);
    dim3 block(256, 1, 1);

    auto launch_dense = [&]() {
        w4a16_gemv<<<grid, block>>>(dA, dB, dS, scale2, dC, N, K);
    };
    auto launch_build = [&]() {
        ffn_build_keep_chunks<<<dim3(1,1,1), block>>>(dA, tau, dKeepIdx, dKeepLen, K);
    };
    auto launch_sparse = [&](unsigned int keep_len) {
        w4a16_gemv_sparse_cols<<<grid, block>>>(dA, dB, dS, scale2,
                                                dKeepIdx, keep_len, dC, N, K);
    };

    const int WARM = 20, ITERS = 200;
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    auto time_kernel = [&](auto&& fn) -> double {
        for (int i = 0; i < WARM; i++) fn();
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int i = 0; i < ITERS; i++) fn();
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        return (double)ms * 1e3 / ITERS; // us
    };

    // ---- DENSE baseline (keep-independent; activation doesn't change cost) ---
    build_activation(K8); // full activation
    double us_dense = time_kernel(launch_dense);
    // dense weight bytes: whole packed + whole scales (per row × N already in bp/bs)
    double dense_wbytes = (double)bp_bytes + (double)bs_bytes;
    double bw_dense = dense_wbytes / (us_dense * 1e-6) / 1e9;

    printf("=== down_proj  N=%u  K=%u  (K8=%u chunks/row)  NVFP4 ===\n", N, K, K8);
    printf("dense w4a16_gemv        : %8.3f us   %6.1f GB/s   (roofline 273 GB/s = %.1f%%)\n\n",
           us_dense, bw_dense, 100.0 * bw_dense / 273.0);

    printf("kept   keep_len  build_us  gemv_us  total_us  vs_dense   ideal_GBs  eff_of_dense\n");
    printf("-----  --------  --------  -------  --------  --------  ---------  ------------\n");

    double kept_fracs[] = {0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 0.90, 1.00};
    for (double kf : kept_fracs) {
        unsigned int keep = (unsigned int)(kf * K8 + 0.5);
        if (keep < 1) keep = 1;
        if (keep > K8) keep = K8;
        build_activation(keep);

        // Verify the thresholder actually produced `keep` survivors.
        launch_build(); CK(cudaDeviceSynchronize());
        unsigned int keep_len = 0;
        CK(cudaMemcpy(&keep_len, dKeepLen, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        double us_build  = time_kernel(launch_build);
        double us_gemv   = time_kernel([&]() { launch_sparse(keep_len); });
        double us_total  = us_build + us_gemv;

        // Ideal weight bytes at this keep fraction: only surviving chunks'
        // packed reads (4 B/chunk × N) + their scale reads. This is the DRAM
        // FLOOR if strided reads were free of sector overfetch.
        double frac = (double)keep_len / (double)K8;
        double ideal_wbytes = frac * dense_wbytes;
        double bw_ideal = ideal_wbytes / (us_gemv * 1e-6) / 1e9;

        printf("%.2f   %8u  %8.2f  %7.2f  %8.2f  %6.3fx   %8.1f   %6.3fx\n",
               kf, keep_len, us_build, us_gemv, us_total,
               us_dense / us_total,           // speedup incl. index-build overhead
               bw_ideal,                      // effective GB/s vs ideal skipped bytes
               us_dense / us_gemv);           // gemv-only speedup (no build)
    }

    printf("\nColumns:\n");
    printf("  vs_dense     = dense_us / (build_us + gemv_us)   >1 = sparse wins incl. index build\n");
    printf("  eff_of_dense = dense_us / gemv_us                >1 = sparse GEMV alone beats dense\n");
    printf("  ideal_GBs    = (kept-fraction × dense weight bytes) / gemv_us.\n");
    printf("                 If this exceeds ~273 the kernel is NOT actually moving that few\n");
    printf("                 bytes (strided sector overfetch) — DRAM traffic didn't drop by\n");
    printf("                 the skip fraction. If it tracks dense's GB/s, byte-skip is real.\n");

    return 0;
}
