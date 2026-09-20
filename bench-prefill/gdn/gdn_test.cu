// Bit-exactness + timing harness for the GDN WY32 prefill kernel (reference vs candidate).
#include "gdn_ref.cu"
#include "gdn_cand.cu"
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
int main(int argc, char** argv) {
  const int L = argc > 1 ? atoi(argv[1]) : 2048, NK = 16, NV = 48, KD = 128, VD = 128;
  const int conv_dim = NK*KD*2 + NV*VD;  // 10240
  const int gb_stride = NV * 2;
  size_t hs = (size_t)NV * KD * VD;
  std::vector<float> h0(hs); for (auto& v : h0) v = (urand() - 0.5f) * 0.1f;
  std::vector<uint16_t> qkv((size_t)L * conv_dim);
  for (size_t t = 0; t < (size_t)L; t++) for (int c = 0; c < conv_dim; c++) {
    float v = (urand() - 0.5f) * 2.0f; if (c < NK*KD*2) v *= 0.0884f;  // l2-normalised-ish q,k
    qkv[t * conv_dim + c] = f2bf(v);
  }
  std::vector<float> gb((size_t)L * gb_stride);
  for (size_t t = 0; t < (size_t)L; t++) for (int v = 0; v < NV; v++) { gb[t*gb_stride+v] = 0.9f + 0.1f*urand(); gb[t*gb_stride+NV+v] = urand(); }
  float *dH0, *dH1, *dG; uint16_t *dQKV, *dO0, *dO1;
  cudaMalloc(&dH0, hs*4); cudaMalloc(&dH1, hs*4); cudaMalloc(&dG, gb.size()*4);
  cudaMalloc(&dQKV, qkv.size()*2); cudaMalloc(&dO0, (size_t)L*NV*VD*2); cudaMalloc(&dO1, (size_t)L*NV*VD*2);
  cudaMemcpy(dQKV, qkv.data(), qkv.size()*2, cudaMemcpyHostToDevice);
  cudaMemcpy(dG, gb.data(), gb.size()*4, cudaMemcpyHostToDevice);
  const __nv_bfloat16* q = (const __nv_bfloat16*)dQKV; const __nv_bfloat16* k = q + NK*KD; const __nv_bfloat16* v = q + 2*NK*KD;
  size_t C_ = 32, pairs = 496;
  size_t smem_ref = KD*VD*4 + C_*KD*2*2 + 16 + C_*C_*4 + C_*4*2 + pairs*4*4 + pairs*2; smem_ref = (smem_ref + 255)/256*256;
  size_t smem_cand = smem_ref;  // candidate may override below
  if (argc > 2) smem_cand = atoi(argv[2]);
  cudaFuncSetAttribute(gated_delta_rule_prefill_wy32_gatecache, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_ref);
  cudaFuncSetAttribute(gdn_candidate, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_cand);
  auto run = [&](int which, float* dH, uint16_t* dO) {
    cudaMemcpy(dH, h0.data(), hs*4, cudaMemcpyHostToDevice);
    if (which == 0) gated_delta_rule_prefill_wy32_gatecache<<<dim3(NV,1), 128, smem_ref>>>(dH, q, k, v, dG, dG + NV, (__nv_bfloat16*)dO, 1, L, NK, NV, KD, VD, conv_dim, conv_dim, gb_stride);
    else gdn_candidate<<<dim3(NV,1), 128, smem_cand>>>(dH, q, k, v, dG, dG + NV, (__nv_bfloat16*)dO, 1, L, NK, NV, KD, VD, conv_dim, conv_dim, gb_stride);
  };
  for (int which = 0; which < 2; which++) {
    float* dH = which ? dH1 : dH0; uint16_t* dO = which ? dO1 : dO0;
    run(which, dH, dO); cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError(); if (e) { printf("kernel %d error %s\n", which, cudaGetErrorString(e)); return 1; }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float best = 1e9;
    for (int it = 0; it < 3; it++) { cudaEventRecord(e0); run(which, dH, dO); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms; }
    fflush(stdout); printf("%s L=%d: %.3f ms (incl. H upload)\n", which ? "candidate" : "reference", L, best);
  }
  std::vector<float> hA(hs), hB(hs); std::vector<uint16_t> oA((size_t)L*NV*VD), oB((size_t)L*NV*VD);
  cudaMemcpy(hA.data(), dH0, hs*4, cudaMemcpyDeviceToHost); cudaMemcpy(hB.data(), dH1, hs*4, cudaMemcpyDeviceToHost);
  cudaMemcpy(oA.data(), dO0, oA.size()*2, cudaMemcpyDeviceToHost); cudaMemcpy(oB.data(), dO1, oB.size()*2, cudaMemcpyDeviceToHost);
  size_t dh = 0, dO = 0; for (size_t i = 0; i < hs; i++) dh += memcmp(&hA[i], &hB[i], 4) != 0; for (size_t i = 0; i < oA.size(); i++) dO += oA[i] != oB[i];
  fflush(stdout); printf("mismatch: h_state=%zu/%zu output=%zu/%zu -> %s\n", dh, hs, dO, oA.size(), (dh|dO) ? "FAIL" : "BIT-EXACT");
  return (dh | dO) ? 1 : 0;
}
