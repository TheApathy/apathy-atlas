// Bit-exactness + timing harness for the W4A16 prefill projection kernels.
#include "../src/kernels/gb10/qwen3.8-27b/nvfp4/w4a16_gemm.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
static uint32_t rng = 12345u;
static uint32_t nxt() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16); }
int main(int argc, char** argv) {
  int shapes[][3] = {{2048,12288,5120},{2048,1024,5120},{2048,5120,6144},{2048,16384,5120},{2047,1024,5120}};
  int fails = 0;
  for (auto& s : shapes) {
    int M = s[0], N = s[1], K = s[2];
    std::vector<uint16_t> hA((size_t)M * K);
    for (auto& v : hA) v = f2bf(((int)(nxt() % 2001) - 1000) / 250.0f);
    std::vector<uint8_t> hB((size_t)N * K / 2), hS((size_t)N * K / 16);
    for (auto& v : hB) v = nxt() & 0xFF;
    for (auto& v : hS) { uint8_t b; do { b = nxt() & 0x7F; } while ((b & 0x78) == 0x78 || (b & 0x78) < 0x28); v = b; }  // e4m3 finite, moderate
    float scale2 = 0.0123f;
    uint16_t *dA, *dC0, *dC1, *dC2; uint8_t *dB, *dS;
    cudaMalloc(&dA, hA.size() * 2); cudaMalloc(&dB, hB.size()); cudaMalloc(&dS, hS.size());
    cudaMalloc(&dC0, (size_t)M * N * 2); cudaMalloc(&dC1, (size_t)M * N * 2); cudaMalloc(&dC2, (size_t)M * N * 2);
    cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dS, hS.data(), hS.size(), cudaMemcpyHostToDevice);
    cudaMemset(dC0, 0x11, (size_t)M*N*2); cudaMemset(dC1, 0x22, (size_t)M*N*2); cudaMemset(dC2, 0x33, (size_t)M*N*2);
    auto t = [&](const char* name, auto launch, uint16_t* dC) {
      launch(dC); cudaDeviceSynchronize();
      cudaError_t e = cudaGetLastError(); if (e) { printf("%s error %s\n", name, cudaGetErrorString(e)); exit(1); }
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
      cudaEventRecord(e0); for (int i = 0; i < 5; i++) launch(dC); cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 5;
      printf("  %-28s %8.3f ms  %6.1f TFLOP/s\n", name, ms, 2.0 * M * N * K / ms / 1e9);
    };
    printf("M=%d N=%d K=%d\n", M, N, K);
    t("w4a16_gemm (baseline)", [&](uint16_t* C){ w4a16_gemm<<<dim3((N+63)/64,(M+63)/64), 128>>>((__nv_bfloat16*)dA, dB, dS, scale2, (__nv_bfloat16*)C, M, N, K); }, dC0);
    t("w4a16_gemm_pipe", [&](uint16_t* C){ w4a16_gemm_pipe<<<dim3((N+63)/64,(M+63)/64), 128>>>((__nv_bfloat16*)dA, dB, dS, scale2, (__nv_bfloat16*)C, M, N, K); }, dC1);
    t("w4a16_gemm_pipe_m128n128", [&](uint16_t* C){ w4a16_gemm_pipe_m128n128<<<dim3(N/128,(M+127)/128), 256>>>((__nv_bfloat16*)dA, dB, dS, scale2, (__nv_bfloat16*)C, M, N, K); }, dC2);
    std::vector<uint16_t> h0((size_t)M*N), h1((size_t)M*N), h2((size_t)M*N);
    cudaMemcpy(h0.data(), dC0, h0.size()*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h1.data(), dC1, h1.size()*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h2.data(), dC2, h2.size()*2, cudaMemcpyDeviceToHost);
    size_t d1 = 0, d2 = 0, nz = 0;
    for (size_t i = 0; i < h0.size(); i++) { d1 += h0[i] != h1[i]; d2 += h0[i] != h2[i]; nz += h0[i] != 0; }
    printf("  mismatches vs baseline: pipe=%zu m128n128=%zu (nonzero outputs %zu/%zu)\n", d1, d2, nz, h0.size());
    if (d2) fails++;
    cudaFree(dA); cudaFree(dB); cudaFree(dS); cudaFree(dC0); cudaFree(dC1); cudaFree(dC2);
  }
  // Transposed-layout FP8-MMA family: w4a16_gemm_t vs w4a16_gemm_t_m128 vs w4a16_gemm_t_m128n128_w8,
  // and fp8_gemm_t_m128 vs fp8_gemm_t_m128n128_w8 (SSM QKVZ / out_proj, PROJ_FAST attention class).
  for (auto& s : shapes) {
    int M = s[0], N = s[1], K = s[2];
    std::vector<uint16_t> hA((size_t)M * K); for (auto& v : hA) v = f2bf(((int)(nxt() % 2001) - 1000) / 250.0f);
    std::vector<uint8_t> hB((size_t)N * K / 2), hS((size_t)N * K / 16);
    for (auto& v : hB) v = nxt() & 0xFF;
    for (auto& v : hS) { uint8_t b; do { b = nxt() & 0x7F; } while ((b & 0x78) == 0x78 || (b & 0x78) < 0x28); v = b; }
    float scale2 = 0.0123f;
    uint16_t *dA, *dC0, *dC1, *dC2; uint8_t *dB, *dS;
    cudaMalloc(&dA, hA.size()*2); cudaMalloc(&dB, hB.size()); cudaMalloc(&dS, hS.size());
    cudaMalloc(&dC0, (size_t)M*N*2); cudaMalloc(&dC1, (size_t)M*N*2); cudaMalloc(&dC2, (size_t)M*N*2);
    cudaMemcpy(dA, hA.data(), hA.size()*2, cudaMemcpyHostToDevice); cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice); cudaMemcpy(dS, hS.data(), hS.size(), cudaMemcpyHostToDevice);
    cudaMemset(dC0, 0x11, (size_t)M*N*2); cudaMemset(dC1, 0x22, (size_t)M*N*2); cudaMemset(dC2, 0x33, (size_t)M*N*2);
    auto t = [&](const char* name, auto launch, uint16_t* dC) {
      launch(dC); cudaDeviceSynchronize(); cudaError_t e = cudaGetLastError(); if (e) { printf("%s error %s\n", name, cudaGetErrorString(e)); exit(1); }
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0); for (int i = 0; i < 5; i++) launch(dC); cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 5; printf("  %-28s %8.3f ms  %6.1f TFLOP/s\n", name, ms, 2.0*M*N*K/ms/1e9); };
    printf("[transposed/FP8] M=%d N=%d K=%d\n", M, N, K);
    t("w4a16_gemm_t", [&](uint16_t* C){ w4a16_gemm_t<<<dim3((N+127)/128,(M+63)/64), 128>>>((__nv_bfloat16*)dA, dB, dS, scale2, (__nv_bfloat16*)C, M, N, K); }, dC0);
    t("w4a16_gemm_t_m128", [&](uint16_t* C){ w4a16_gemm_t_m128<<<dim3((N+127)/128,(M+127)/128), 128>>>((__nv_bfloat16*)dA, dB, dS, scale2, (__nv_bfloat16*)C, M, N, K); }, dC1);
    t("w4a16_gemm_t_m128n128_w8", [&](uint16_t* C){ w4a16_gemm_t_m128n128_w8<<<dim3(N/128,(M+127)/128), 256>>>((__nv_bfloat16*)dA, dB, dS, scale2, (__nv_bfloat16*)C, M, N, K); }, dC2);
    std::vector<uint16_t> h0((size_t)M*N), h1((size_t)M*N), h2((size_t)M*N);
    cudaMemcpy(h0.data(), dC0, h0.size()*2, cudaMemcpyDeviceToHost); cudaMemcpy(h1.data(), dC1, h1.size()*2, cudaMemcpyDeviceToHost); cudaMemcpy(h2.data(), dC2, h2.size()*2, cudaMemcpyDeviceToHost);
    size_t d1 = 0, d2 = 0; for (size_t i = 0; i < h0.size(); i++) { d1 += h0[i] != h1[i]; d2 += h0[i] != h2[i]; }
    printf("  mismatches t vs t_m128: %zu, t vs w8: %zu\n", d1, d2); if (d1 | d2) fails++;
    std::vector<uint8_t> hW((size_t)N * K); for (auto& v : hW) { uint8_t b = nxt() & 0xFF; if ((b & 0x7F) == 0x7F) b &= 0x7E; v = b; }
    uint8_t* dW; cudaMalloc(&dW, hW.size()); cudaMemcpy(dW, hW.data(), hW.size(), cudaMemcpyHostToDevice);
    cudaMemset(dC1, 0x66, (size_t)M*N*2); cudaMemset(dC2, 0x77, (size_t)M*N*2);
    t("fp8_gemm_t_m128", [&](uint16_t* C){ fp8_gemm_t_m128<<<dim3((N+127)/128,(M+127)/128), 128>>>((__nv_bfloat16*)dA, dW, (__nv_bfloat16*)C, M, N, K); }, dC1);
    t("fp8_gemm_t_m128n128_w8", [&](uint16_t* C){ fp8_gemm_t_m128n128_w8<<<dim3(N/128,(M+127)/128), 256>>>((__nv_bfloat16*)dA, dW, (__nv_bfloat16*)C, M, N, K); }, dC2);
    cudaMemcpy(h1.data(), dC1, h1.size()*2, cudaMemcpyDeviceToHost); cudaMemcpy(h2.data(), dC2, h2.size()*2, cudaMemcpyDeviceToHost);
    size_t d3 = 0; for (size_t i = 0; i < h1.size(); i++) d3 += h1[i] != h2[i];
    printf("  mismatches fp8 m128 vs w8: %zu\n", d3); if (d3) fails++;
    cudaFree(dW); cudaFree(dA); cudaFree(dB); cudaFree(dS); cudaFree(dC0); cudaFree(dC1); cudaFree(dC2);
  }
  printf(fails ? "FAIL\n" : "ALL-EXACT\n");
  return fails;
}
