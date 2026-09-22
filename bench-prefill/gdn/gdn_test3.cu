// Timing + numeric-deviation harness: production WY32 kernel vs the K-split R3 kernel.
#include <cuda_bf16.h>
#include "gdn_ref.cu"
#include "gdn_cand3.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include <cmath>
static uint32_t rng = 777u;
static uint32_t nxt() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float urand() { return (nxt() % 100000) / 100000.0f; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16); }
static float bf2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
int main(int argc, char** argv) {
  const int L = argc > 1 ? atoi(argv[1]) : 2048, NK = 16, NV = 48, KD = 128, VD = 128;
  const int conv_dim = NK*KD*2 + NV*VD;
  const int gb_stride = NV * 2;
  size_t hs = (size_t)NV * KD * VD;
  std::vector<float> h0(hs); for (auto& x : h0) x = (urand() - 0.5f) * 0.1f;
  std::vector<uint16_t> qkv((size_t)L * conv_dim);
  for (size_t t = 0; t < (size_t)L; t++) for (int c = 0; c < conv_dim; c++) {
    float x = (urand() - 0.5f) * 2.0f; if (c < NK*KD*2) x *= 0.0884f;
    qkv[t * conv_dim + c] = f2bf(x);
  }
  std::vector<float> gb((size_t)L * gb_stride);
  for (size_t t = 0; t < (size_t)L; t++) for (int vv = 0; vv < NV; vv++) {
    gb[t*gb_stride+vv] = 0.9f + 0.1f*urand(); gb[t*gb_stride+NV+vv] = urand(); }
  float *dH0, *dH1, *dG; uint16_t *dQKV, *dO0, *dO1;
  cudaMalloc(&dH0, hs*4); cudaMalloc(&dH1, hs*4); cudaMalloc(&dG, gb.size()*4);
  cudaMalloc(&dQKV, qkv.size()*2); cudaMalloc(&dO0, (size_t)L*NV*VD*2); cudaMalloc(&dO1, (size_t)L*NV*VD*2);
  cudaMemcpy(dQKV, qkv.data(), qkv.size()*2, cudaMemcpyHostToDevice);
  cudaMemcpy(dG, gb.data(), gb.size()*4, cudaMemcpyHostToDevice);
  const __nv_bfloat16* q = (const __nv_bfloat16*)dQKV;
  const __nv_bfloat16* k = q + NK*KD; const __nv_bfloat16* v = q + 2*NK*KD;
  size_t C_ = 32, pairs = 496;
  size_t smem_ref = KD*VD*4 + C_*KD*2*2 + 16 + C_*C_*4 + C_*4*2 + pairs*4*4 + pairs*2;
  smem_ref = (smem_ref + 255)/256*256;
  size_t base3 = C_*KD*2*2 + C_*C_*4 + C_*4*2 + pairs*4*4 + pairs*2;
  base3 = (base3 + 15)/16*16;
  size_t smem_c3 = base3 + 3*C_*VD*4 + C_*VD*4;
  smem_c3 = (smem_c3 + 255)/256*256;
  printf("smem: ref=%zu cand3=%zu (optin limit 101376)\n", smem_ref, smem_c3);
  cudaFuncSetAttribute(gated_delta_rule_prefill_wy32_gatecache, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_ref);
  cudaFuncSetAttribute(gdn_candidate3, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_c3);
  auto run = [&](int which, float* dH, uint16_t* dO) {
    cudaMemcpy(dH, h0.data(), hs*4, cudaMemcpyHostToDevice);
    if (which == 0) gated_delta_rule_prefill_wy32_gatecache<<<dim3(NV,1), 128, smem_ref>>>(
        dH, q, k, v, dG, dG + NV, (__nv_bfloat16*)dO, 1, L, NK, NV, KD, VD, conv_dim, conv_dim, gb_stride);
    else gdn_candidate3<<<dim3(NV,1), 512, smem_c3>>>(
        dH, q, k, v, dG, dG + NV, (__nv_bfloat16*)dO, 1, L, NK, NV, KD, VD, conv_dim, conv_dim, gb_stride);
  };
  for (int which = 0; which < 2; which++) {
    float* dH = which ? dH1 : dH0; uint16_t* dO = which ? dO1 : dO0;
    run(which, dH, dO); cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e) { printf("kernel %d error %s\n", which, cudaGetErrorString(e)); return 1; }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float best = 1e9;
    for (int it = 0; it < 5; it++) { cudaEventRecord(e0); run(which, dH, dO); cudaEventRecord(e1);
      cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms; }
    printf("%-10s L=%d: %.3f ms (incl. H upload)\n", which ? "cand3" : "reference", L, best);
  }
  std::vector<float> hA(hs), hB(hs);
  std::vector<uint16_t> oA((size_t)L*NV*VD), oB((size_t)L*NV*VD);
  cudaMemcpy(hA.data(), dH0, hs*4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hB.data(), dH1, hs*4, cudaMemcpyDeviceToHost);
  cudaMemcpy(oA.data(), dO0, oA.size()*2, cudaMemcpyDeviceToHost);
  cudaMemcpy(oB.data(), dO1, oB.size()*2, cudaMemcpyDeviceToHost);
  size_t dh = 0, dbits = 0, ulp1 = 0;
  double sa = 0, sd = 0, maxrel = 0;
  for (size_t i = 0; i < hs; i++) { if (memcmp(&hA[i], &hB[i], 4) != 0) dh++;
    double d = fabs((double)hA[i] - hB[i]); sd += d*d; sa += (double)hA[i]*hA[i];
    double r = fabs(hA[i]) > 1e-20 ? d/fabs(hA[i]) : 0; if (r > maxrel) maxrel = r; }
  double soa = 0, sod = 0, omaxabs = 0;
  for (size_t i = 0; i < oA.size(); i++) { if (oA[i] != oB[i]) { dbits++;
      int diff = (int)oA[i] - (int)oB[i]; if (diff == 1 || diff == -1) ulp1++; }
    double a = bf2f(oA[i]), bb = bf2f(oB[i]); soa += a*a; sod += (a-bb)*(a-bb);
    if (fabs(a-bb) > omaxabs) omaxabs = fabs(a-bb); }
  printf("h_state : %zu/%zu differ, rel-L2 %.3e, max rel %.3e\n", dh, hs, sqrt(sd/(sa+1e-30)), maxrel);
  printf("output  : %zu/%zu bf16 words differ (%.4f%%), of which %zu are 1 ulp; rel-L2 %.3e, max abs %.3e\n",
         dbits, oA.size(), 100.0*dbits/oA.size(), ulp1, sqrt(sod/(soa+1e-30)), omaxabs);
  printf("%s\n", (dh|dbits) ? "NOT-BIT-EXACT (expected: j-reduction is regrouped)" : "BIT-EXACT");
  return 0;
}
