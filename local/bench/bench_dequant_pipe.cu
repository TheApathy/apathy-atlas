// SPDX-License-Identifier: AGPL-3.0-only
//
// Standalone microbench + byte-exactness check for the DEQUANT-IN-REGISTERS
// fused gate/up kernel (ATLAS_DEQUANT_PIPE) vs the SMEM-staged baseline.
//
// Target shape = DFlash K=γ+1=17 verify FFN gate/up on Qwen3.6-27B:
//   M=17, N=inter=17408, K=hidden=5120, ldb=N (tightly packed T-weights).
//
// This file is NOT part of the engine build (no `extern "C" __global__`
// entry points of its own that the loader picks up — it #includes the
// production translation unit so both kernels are the SAME code the engine
// ships). It is compiled + run ONLY as a manual bench in a GPU window.
//
// BUILD + RUN (next GPU window, engine must be idle):
//   nvcc -arch=sm_121f -O3 --use_fast_math -Xptxas -O3 --fmad=false \
//     -I kernels/gb10/qwen3.6-27b/nvfp4 \
//     local/bench/bench_dequant_pipe.cu \
//     -o /tmp/bench_dequant_pipe && /tmp/bench_dequant_pipe
//
// PASS criteria:
//   * "BYTE-EXACT: PASS  (0 mismatched BF16 elements)"  ← the md5 gate proxy
//   * speedup > 1.0 on the mean kernel time
// A single mismatched element FAILS the bit-exactness constitution — do NOT
// ship ATLAS_DEQUANT_PIPE=1 in that case; report the divergence instead.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Pull in the production kernels verbatim (both baseline + _pipe live here).
#include "w4a16_gemm.cu"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e_)); exit(1); } } while(0)

int main() {
    const unsigned int M = 17, N = 17408, K = 5120, ldb = N;
    const unsigned int half_K = K / 2;
    const unsigned int groups = K / 16;

    // ---- Host-side deterministic inputs -------------------------------
    // A: [M, K] BF16. Weights: gate + up, each [N, K/2] packed nibbles +
    // [N, K/16] FP8 E4M3 scales + one FP32 scale2.
    size_t a_elems = (size_t)M * K;
    size_t bp_bytes = (size_t)N * half_K;   // per weight
    size_t bs_bytes = (size_t)N * groups;   // per weight
    size_t c_elems = (size_t)M * N;

    __nv_bfloat16* hA = (__nv_bfloat16*)malloc(a_elems * sizeof(__nv_bfloat16));
    uint8_t* hBg = (uint8_t*)malloc(bp_bytes);
    uint8_t* hBu = (uint8_t*)malloc(bp_bytes);
    uint8_t* hSg = (uint8_t*)malloc(bs_bytes);
    uint8_t* hSu = (uint8_t*)malloc(bs_bytes);

    // Deterministic LCG so the two runs see identical inputs.
    uint64_t s = 0x1234567;
    auto rnd = [&]() { s = s * 6364136223846793005ULL + 1442695040888963407ULL;
                       return (uint32_t)(s >> 33); };
    for (size_t i = 0; i < a_elems; i++) {
        float v = ((int)(rnd() & 0xFF) - 128) / 256.0f;    // ~[-0.5, 0.5)
        hA[i] = __float2bfloat16(v);
    }
    for (size_t i = 0; i < bp_bytes; i++) { hBg[i] = (uint8_t)rnd(); hBu[i] = (uint8_t)rnd(); }
    // FP8 E4M3 scale bytes: keep them in a sane exponent range so dequant
    // doesn't saturate to inf everywhere (values near 1.0). 0x38..0x40 ≈
    // 0.5..2.0 in e4m3.
    for (size_t i = 0; i < bs_bytes; i++) {
        hSg[i] = (uint8_t)(0x38 + (rnd() & 0x7));
        hSu[i] = (uint8_t)(0x38 + (rnd() & 0x7));
    }
    const float scale2_g = 1.0f / 448.0f;
    const float scale2_u = 1.0f / 448.0f;

    // ---- Device buffers -----------------------------------------------
    __nv_bfloat16 *dA, *dCbase, *dCpipe;
    uint8_t *dBg, *dBu, *dSg, *dSu;
    CK(cudaMalloc(&dA, a_elems * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dBg, bp_bytes)); CK(cudaMalloc(&dBu, bp_bytes));
    CK(cudaMalloc(&dSg, bs_bytes)); CK(cudaMalloc(&dSu, bs_bytes));
    CK(cudaMalloc(&dCbase, c_elems * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dCpipe, c_elems * sizeof(__nv_bfloat16)));
    CK(cudaMemcpy(dA, hA, a_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBg, hBg, bp_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBu, hBu, bp_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSg, hSg, bs_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSu, hSu, bs_bytes, cudaMemcpyHostToDevice));

    // Grid/block identical to the Rust launcher `w4a16_gemm_n64_m32_gateup_silu`:
    //   grid (ceil(N/64), ceil(M/32), 1)  block (128,1,1)
    dim3 grid((N + 63) / 64, (M + 31) / 32, 1);
    dim3 block(128, 1, 1);

    auto launch_base = [&](__nv_bfloat16* C) {
        w4a16_gemm_t_m32_n64_gateup_silu<<<grid, block>>>(
            dA, dBg, dSg, scale2_g, dBu, dSu, scale2_u, C, M, N, K, ldb);
    };
    auto launch_pipe = [&](__nv_bfloat16* C) {
        w4a16_gemm_t_m32_n64_gateup_silu_pipe<<<grid, block>>>(
            dA, dBg, dSg, scale2_g, dBu, dSu, scale2_u, C, M, N, K, ldb);
    };

    // ---- Correctness: byte-exact compare ------------------------------
    CK(cudaMemset(dCbase, 0, c_elems * sizeof(__nv_bfloat16)));
    CK(cudaMemset(dCpipe, 0, c_elems * sizeof(__nv_bfloat16)));
    launch_base(dCbase); CK(cudaDeviceSynchronize());
    launch_pipe(dCpipe); CK(cudaDeviceSynchronize());

    uint16_t* hCb = (uint16_t*)malloc(c_elems * sizeof(uint16_t));
    uint16_t* hCp = (uint16_t*)malloc(c_elems * sizeof(uint16_t));
    CK(cudaMemcpy(hCb, dCbase, c_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hCp, dCpipe, c_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    size_t mismatch = 0, first = (size_t)-1;
    for (size_t i = 0; i < c_elems; i++) {
        if (hCb[i] != hCp[i]) { if (first == (size_t)-1) first = i; mismatch++; }
    }
    printf("BYTE-EXACT: %s  (%zu mismatched BF16 elements of %zu)\n",
           mismatch == 0 ? "PASS" : "FAIL", mismatch, c_elems);
    if (mismatch) {
        printf("  first mismatch at elem %zu: base=0x%04x pipe=0x%04x\n",
               first, hCb[first], hCp[first]);
    }

    // ---- Timing --------------------------------------------------------
    const int WARM = 20, ITERS = 200;
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    for (int i = 0; i < WARM; i++) launch_base(dCbase);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    for (int i = 0; i < ITERS; i++) launch_base(dCbase);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms_base = 0; CK(cudaEventElapsedTime(&ms_base, e0, e1));

    for (int i = 0; i < WARM; i++) launch_pipe(dCpipe);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    for (int i = 0; i < ITERS; i++) launch_pipe(dCpipe);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms_pipe = 0; CK(cudaEventElapsedTime(&ms_pipe, e0, e1));

    double us_base = ms_base * 1e3 / ITERS;
    double us_pipe = ms_pipe * 1e3 / ITERS;
    // Weight bytes moved per launch (gate + up): packed + fp8 scales.
    double wbytes = 2.0 * ((double)bp_bytes + (double)bs_bytes);
    double bw_base = wbytes / (us_base * 1e-6) / 1e9;   // GB/s
    double bw_pipe = wbytes / (us_pipe * 1e-6) / 1e9;
    printf("baseline gateup_silu     : %8.2f us   %6.1f GB/s\n", us_base, bw_base);
    printf("pipe (dequant-in-regs)   : %8.2f us   %6.1f GB/s\n", us_pipe, bw_pipe);
    printf("speedup                  : %6.3fx   (roofline: 273 GB/s LPDDR5X)\n",
           us_base / us_pipe);
    printf("roofline%% base=%.1f%%  pipe=%.1f%%\n",
           100.0 * bw_base / 273.0, 100.0 * bw_pipe / 273.0);

    return mismatch == 0 ? 0 : 2;
}
