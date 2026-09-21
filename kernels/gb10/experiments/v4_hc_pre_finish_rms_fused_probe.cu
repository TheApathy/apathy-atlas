// SPDX-License-Identifier: AGPL-3.0-only

// Standalone, GPU-runnable qualification probe against DeepSeek-V4's vanilla
// RMSNorm (`x * rms * weight`). The implementation also has a target-specific,
// strict-default-off Atlas production wrapper.

#include <cuda_runtime.h>

#include <cmath>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef V4_HC_PROBE_BUILD_ID
#error "V4_HC_PROBE_BUILD_ID must bind this probe to its build receipt"
#endif
#ifndef V4_HC_PROBE_MIN_SPEEDUP_TEXT
#error "V4_HC_PROBE_MIN_SPEEDUP_TEXT must bind the canonical binary64 threshold"
#endif
#ifndef V4_HC_PROBE_MIN_SPEEDUP_HEX
#error "V4_HC_PROBE_MIN_SPEEDUP_HEX must bind the exact binary64 threshold value"
#endif

// BEGIN HC probe kernel contract
#include "../deepseek-v4-flash/nvfp4/hyper_connection.cu"
#include "../common/rms_norm_vanilla.cu"
#include "v4_hc_pre_finish_rms_fused.cu"

namespace probe {
constexpr unsigned kTokens = 2410;
constexpr unsigned kHidden = 4096;
constexpr unsigned kHc = 4;
constexpr unsigned kFinishShards = 16;
constexpr unsigned kRmsThreads = 1024;
constexpr unsigned kMix = 24;
constexpr int kParityCases = 6;
constexpr int kPoisonCases = 2;
constexpr int kGeometryCases = 10;
constexpr int kTimingSamples = 6;
constexpr double kMinSpeedup = V4_HC_PROBE_MIN_SPEEDUP_HEX;
static_assert(kParityCases == 6, "trained-epsilon parity census changed");
static_assert(kGeometryCases == 10, "malformed geometry census changed");
static_assert(kMinSpeedup > 1.0 && kMinSpeedup <= 100.0,
              "receipt-bound speedup threshold is outside the admitted range");

static void launch_baseline(const float* streams, const float* mix,
                            const float* scale, const float* base,
                            const __nv_bfloat16* weight, __nv_bfloat16* hidden,
                            __nv_bfloat16* normed, float* post, float* comb,
                            unsigned iters, float norm_eps, float hc_eps,
                            float rms_eps) {
  hc_pre_finish<<<dim3(kTokens, kFinishShards, 1), dim3(256, 1, 1)>>>(
      streams, mix, scale, base, hidden, post, comb, kHidden, kHc, iters,
      norm_eps, hc_eps);
  rms_norm_vanilla<<<dim3(kTokens, 1, 1), dim3(kRmsThreads, 1, 1)>>>(
      hidden, weight, normed, kHidden, rms_eps);
}

static void launch_candidate(const float* streams, const float* mix,
                             const float* scale, const float* base,
                             const __nv_bfloat16* weight,
                             __nv_bfloat16* hidden, __nv_bfloat16* normed,
                             float* post, float* comb, unsigned iters,
                             float norm_eps, float hc_eps, float rms_eps) {
  v4_hc_pre_finish_rms_fused<<<dim3(kTokens, 1, 1), dim3(kRmsThreads, 1, 1)>>>(
      streams, mix, scale, base, weight, hidden, normed, post, comb, kTokens,
      kHidden, kHc, iters, norm_eps, hc_eps, rms_eps);
}
// END HC probe kernel contract

[[noreturn]] static void die(const char* message) {
  std::fprintf(stderr, "%s\n", message);
  std::exit(1);
}

static void cuda_ok(cudaError_t error, const char* where) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(error));
    std::exit(1);
  }
}

static uint64_t fnv(uint64_t hash, const void* data, size_t bytes) {
  const auto* p = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < bytes; ++i) hash = (hash ^ p[i]) * 1099511628211ull;
  return hash;
}

template <class T> struct Guarded {
  static constexpr size_t kGuard = 256;
  unsigned char* allocation = nullptr;
  T* data = nullptr;
  size_t count = 0;
  explicit Guarded(size_t n) : count(n) {
    cuda_ok(cudaMalloc(&allocation, bytes() + 2 * kGuard), "cudaMalloc");
    data = reinterpret_cast<T*>(allocation + kGuard);
    cuda_ok(cudaMemset(allocation, 0xa7, kGuard), "prefix guard");
    cuda_ok(cudaMemset(allocation + kGuard + bytes(), 0x7a, kGuard), "suffix guard");
  }
  ~Guarded() { cudaFree(allocation); }
  size_t bytes() const { return count * sizeof(T); }
  void fill(unsigned char value) { cuda_ok(cudaMemset(data, value, bytes()), "fill"); }
  void upload(const std::vector<T>& host) {
    if (host.size() != count) die("host/device extent mismatch");
    cuda_ok(cudaMemcpy(data, host.data(), bytes(), cudaMemcpyHostToDevice), "upload");
  }
  std::vector<unsigned char> download() const {
    std::vector<unsigned char> out(bytes());
    cuda_ok(cudaMemcpy(out.data(), data, bytes(), cudaMemcpyDeviceToHost), "download");
    return out;
  }
  bool guards_clean() const {
    std::vector<unsigned char> lo(kGuard), hi(kGuard);
    cuda_ok(cudaMemcpy(lo.data(), allocation, kGuard, cudaMemcpyDeviceToHost), "prefix_guard");
    cuda_ok(cudaMemcpy(hi.data(), allocation + kGuard + bytes(), kGuard,
                       cudaMemcpyDeviceToHost), "suffix_guard");
    for (auto x : lo) if (x != 0xa7) return false;
    for (auto x : hi) if (x != 0x7a) return false;
    return true;
  }
};

// BEGIN HC probe deterministic cases
// Contract: kParityCases = 6, including the text-checkpoint and actual Vision
// runtime epsilon tuples independently from the mHC stabilization epsilon.
struct CaseSpec {
  unsigned sinkhorn_iters;
  float hc_norm_eps, hc_eps, rms_eps;
  unsigned stream_salt, mix_salt, weight_salt;
};
constexpr CaseSpec kCases[kParityCases] = {
    {1, 1.0e-6f, 1.0e-6f, 1.0e-6f, 11, 37, 71},
    {2, 2.0e-6f, 3.0e-6f, 4.0e-6f, 19, 43, 83},
    {3, 5.0e-7f, 2.0e-5f, 8.0e-6f, 29, 59, 97},
    {4, 9.0e-6f, 7.0e-7f, 3.0e-5f, 31, 67, 109},
    {20, 1.0e-6f, 1.0e-6f, 1.0e-20f, 41, 73, 127},
    {20, 1.0e-20f, 1.0e-6f, 1.0e-20f, 47, 79, 131},
};

static uint16_t bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + bias) >> 16);
}

static void make_case(const CaseSpec& c, std::vector<float>& streams,
                      std::vector<float>& mix, std::vector<__nv_bfloat16>& weight) {
  for (size_t i = 0; i < streams.size(); ++i) {
    int v = static_cast<int>((i * 131u + c.stream_salt * 17u) % 509u) - 254;
    float x = static_cast<float>(v) * (1.0f / 1024.0f);
    if ((i + c.stream_salt) % 65537u == 0) x = (i & 1) ? -0.0f : 0.0f; // signed-zero
    streams[i] = x;
  }
  for (unsigned t = 0; t < kTokens; ++t) {
    for (unsigned m = 0; m < kMix; ++m) {
      int v = static_cast<int>((t * 23u + m * 41u + c.mix_salt) % 193u) - 96;
      mix[static_cast<size_t>(t) * (kMix + 1) + m] = v * (1.0f / 128.0f);
    }
    mix[static_cast<size_t>(t) * (kMix + 1) + kMix] =
        1100.0f + static_cast<float>((t + c.mix_salt) % 257u);
  }
  auto* raw = reinterpret_cast<uint16_t*>(weight.data());
  for (unsigned i = 0; i < kHidden; ++i) {
    int v = static_cast<int>((i * 29u + c.weight_salt) % 101u) - 50;
    raw[i] = bf16(1.0f + v * (1.0f / 512.0f));
  }
}
// END HC probe deterministic cases

static size_t mismatch_bytes(const std::vector<unsigned char>& a,
                             const std::vector<unsigned char>& b) {
  if (a.size() != b.size()) return a.size() + b.size();
  size_t n = 0;
  for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
  return n;
}

struct Outputs {
  Guarded<__nv_bfloat16> hidden{static_cast<size_t>(kTokens) * kHidden};
  Guarded<__nv_bfloat16> normed{static_cast<size_t>(kTokens) * kHidden};
  Guarded<float> post{static_cast<size_t>(kTokens) * kHc};
  Guarded<float> comb{static_cast<size_t>(kTokens) * kHc * kHc};
  void fill(unsigned char x) { hidden.fill(x); normed.fill(x); post.fill(x); comb.fill(x); }
  bool guards_clean() const {
    return hidden.guards_clean() && normed.guards_clean() &&
           post.guards_clean() && comb.guards_clean();
  }
};

// BEGIN HC probe exact byte parity
constexpr unsigned char kPoisonA = 0x5a;
constexpr unsigned char kPoisonB = 0xc3;
// Byte-exact memcmp is applied to all full output buffers under both opposite
// poisons, so hidden_mismatches/normed_mismatches/post_mismatches/
// comb_mismatches expose unwritten bytes.
// poison_a, poison_b, and guards_clean are reported only after both trials.
// END HC probe exact byte parity

// BEGIN HC probe malformed geometry
// Contract: kGeometryCases = 10.
static void malformed(const float* streams, const float* mix, const float* scale,
                      const float* base, const __nv_bfloat16* weight, Outputs& out,
                      const CaseSpec& c, dim3 grid, dim3 block, unsigned hidden,
                      unsigned hc, bool null_input, const char* label) {
  out.fill(0x3c);
  const auto h0 = out.hidden.download(), n0 = out.normed.download();
  const auto p0 = out.post.download(), c0 = out.comb.download();
  v4_hc_pre_finish_rms_fused<<<grid, block>>>(
      null_input ? static_cast<const float*>(0) : streams, mix, scale, base,
      weight, out.hidden.data, out.normed.data, out.post.data, out.comb.data,
      kTokens, hidden, hc, c.sinkhorn_iters, c.hc_norm_eps, c.hc_eps,
      c.rms_eps);
  cuda_ok(cudaGetLastError(), label);
  cuda_ok(cudaDeviceSynchronize(), label);
  const bool expect_unchanged = out.hidden.download() == h0 &&
                                out.normed.download() == n0 &&
                                out.post.download() == p0 && out.comb.download() == c0;
  const bool prefix_guard = out.hidden.guards_clean() && out.normed.guards_clean();
  const bool suffix_guard = out.post.guards_clean() && out.comb.guards_clean();
  if (!expect_unchanged || !prefix_guard || !suffix_guard) die("malformed geometry wrote output");
}
// block-x block-y block-z grid-x-small grid-x-large grid-y grid-z wrong-hidden wrong-hc nullptr
// END HC probe malformed geometry

// BEGIN HC probe ABBA timing contract
static float timed(bool baseline, const float* streams, const float* mix,
                   const float* scale, const float* base, const __nv_bfloat16* weight,
                   Outputs& out, const CaseSpec& c) {
  cudaEvent_t begin, end;
  cuda_ok(cudaEventCreate(&begin), "event create"); cuda_ok(cudaEventCreate(&end), "event create");
  cuda_ok(cudaEventRecord(begin), "cudaEventRecord");
  if (baseline) launch_baseline(streams, mix, scale, base, weight, out.hidden.data,
                                out.normed.data, out.post.data, out.comb.data,
                                c.sinkhorn_iters, c.hc_norm_eps, c.hc_eps, c.rms_eps);
  else launch_candidate(streams, mix, scale, base, weight, out.hidden.data,
                        out.normed.data, out.post.data, out.comb.data,
                        c.sinkhorn_iters, c.hc_norm_eps, c.hc_eps, c.rms_eps);
  cuda_ok(cudaEventRecord(end), "cudaEventRecord"); cuda_ok(cudaEventSynchronize(end), "event sync");
  float ms = 0; cuda_ok(cudaEventElapsedTime(&ms, begin, end), "cudaEventElapsedTime");
  cudaEventDestroy(begin); cudaEventDestroy(end); return ms;
}
// The production-shape order is ABBA per sample; baseline_ms/candidate_ms and
// speedup aggregate the same kTimingSamples and exclude allocation and copies.
// END HC probe ABBA timing contract

static void validate_compiled_threshold(int argc) {
  if (argc != 1) {
    std::fprintf(stderr, "usage: v4-hc-pre-finish-rms-fused-probe\n");
    std::exit(2);
  }
  errno = 0;
  char* end = nullptr;
  const char* text = V4_HC_PROBE_MIN_SPEEDUP_TEXT;
  const double parsed = std::strtod(text, &end);
  if (errno != 0 || end == text || *end != '\0' || !std::isfinite(parsed) ||
      parsed != kMinSpeedup) {
    std::fprintf(stderr, "receipt-bound binary64 threshold mismatch\n");
    std::exit(2);
  }
}

static int run(int argc) {
  validate_compiled_threshold(argc); // Must precede every CUDA call.
  const double minimum = kMinSpeedup;
  int driver = 0, runtime = 0, device = 0; cudaDeviceProp prop{};
  cuda_ok(cudaDriverGetVersion(&driver), "driver"); cuda_ok(cudaRuntimeGetVersion(&runtime), "runtime");
  cuda_ok(cudaGetDevice(&device), "device"); cuda_ok(cudaGetDeviceProperties(&prop, device), "properties");
  Guarded<float> streams(static_cast<size_t>(kTokens) * kHc * kHidden);
  Guarded<float> mix(static_cast<size_t>(kTokens) * (kMix + 1));
  Guarded<float> scale(3), base(kMix); Guarded<__nv_bfloat16> weight(kHidden);
  Outputs incumbent, candidate;
  std::vector<float> hs(streams.count), hm(mix.count), hscale{0.75f, -0.5f, 1.125f}, hbase(kMix);
  std::vector<__nv_bfloat16> hw(kHidden); for (unsigned i=0;i<kMix;++i) hbase[i]=(static_cast<int>(i)-12)*0.0078125f;
  scale.upload(hscale); base.upload(hbase);
  uint64_t ih=1469598103934665603ull, mh=ih, wh=ih;
  size_t hmismatch=0, nm=0, pm=0, cm=0;
  for (int ci=0; ci<kParityCases; ++ci) {
    make_case(kCases[ci], hs, hm, hw); streams.upload(hs); mix.upload(hm); weight.upload(hw);
    ih=fnv(ih,hs.data(),hs.size()*sizeof(float)); mh=fnv(mh,hm.data(),hm.size()*sizeof(float)); wh=fnv(wh,hw.data(),hw.size()*sizeof(__nv_bfloat16));
    for (int poison=0; poison<kPoisonCases; ++poison) {
      incumbent.fill(poison ? kPoisonB : kPoisonA); candidate.fill(poison ? kPoisonA : kPoisonB);
      launch_baseline(streams.data,mix.data,scale.data,base.data,weight.data,incumbent.hidden.data,incumbent.normed.data,incumbent.post.data,incumbent.comb.data,kCases[ci].sinkhorn_iters,kCases[ci].hc_norm_eps,kCases[ci].hc_eps,kCases[ci].rms_eps);
      launch_candidate(streams.data,mix.data,scale.data,base.data,weight.data,candidate.hidden.data,candidate.normed.data,candidate.post.data,candidate.comb.data,kCases[ci].sinkhorn_iters,kCases[ci].hc_norm_eps,kCases[ci].hc_eps,kCases[ci].rms_eps);
      cuda_ok(cudaGetLastError(),"parity launch"); cuda_ok(cudaDeviceSynchronize(),"parity sync");
      auto ihidden=incumbent.hidden.download(), chidden=candidate.hidden.download();
      auto in=incumbent.normed.download(), cn=candidate.normed.download(); auto ip=incumbent.post.download(), cp=candidate.post.download(); auto ic=incumbent.comb.download(), cc=candidate.comb.download();
      hmismatch += mismatch_bytes(ihidden,chidden); nm += mismatch_bytes(in,cn); pm += mismatch_bytes(ip,cp); cm += mismatch_bytes(ic,cc);
      if (std::memcmp(ihidden.data(),chidden.data(),ihidden.size()) || std::memcmp(in.data(),cn.data(),in.size()) || !incumbent.guards_clean() || !candidate.guards_clean()) die("byte parity or guard failure");
    }
  }
  if (hmismatch||nm||pm||cm) die("byte parity failure");
  const auto& c=kCases[0];
  malformed(streams.data,mix.data,scale.data,base.data,weight.data,candidate,c,dim3(kTokens),dim3(512),kHidden,kHc,false,"block-x");
  malformed(streams.data,mix.data,scale.data,base.data,weight.data,candidate,c,dim3(kTokens),dim3(512,2),kHidden,kHc,false,"block-y");
  malformed(streams.data,mix.data,scale.data,base.data,weight.data,candidate,c,dim3(kTokens),dim3(512,1,2),kHidden,kHc,false,"block-z");
  malformed(streams.data,mix.data,scale.data,base.data,weight.data,candidate,c,dim3(kTokens-1),dim3(kRmsThreads),kHidden,kHc,false,"grid-x-small");
  malformed(streams.data,mix.data,scale.data,base.data,weight.data,candidate,c,dim3(kTokens+1),dim3(kRmsThreads),kHidden,kHc,false,"grid-x-large");
  malformed(streams.data,mix.data,scale.data,base.data,weight.data,candidate,c,dim3(kTokens,2),dim3(kRmsThreads),kHidden,kHc,false,"grid-y");
  malformed(streams.data,mix.data,scale.data,base.data,weight.data,candidate,c,dim3(kTokens,1,2),dim3(kRmsThreads),kHidden,kHc,false,"grid-z");
  malformed(streams.data,mix.data,scale.data,base.data,weight.data,candidate,c,dim3(kTokens),dim3(kRmsThreads),kHidden-2,kHc,false,"wrong-hidden");
  malformed(streams.data,mix.data,scale.data,base.data,weight.data,candidate,c,dim3(kTokens),dim3(kRmsThreads),kHidden,kHc-1,false,"wrong-hc");
  malformed(streams.data,mix.data,scale.data,base.data,weight.data,candidate,c,dim3(kTokens),dim3(kRmsThreads),kHidden,kHc,true,"nullptr");
  for(int i=0;i<2;++i){timed(true,streams.data,mix.data,scale.data,base.data,weight.data,candidate,c);timed(false,streams.data,mix.data,scale.data,base.data,weight.data,candidate,c);}
  double bm=0, fm=0; for(int i=0;i<kTimingSamples;++i){bm+=timed(true,streams.data,mix.data,scale.data,base.data,weight.data,candidate,c);fm+=timed(false,streams.data,mix.data,scale.data,base.data,weight.data,candidate,c);fm+=timed(false,streams.data,mix.data,scale.data,base.data,weight.data,candidate,c);bm+=timed(true,streams.data,mix.data,scale.data,base.data,weight.data,candidate,c);}
  bm/=2*kTimingSamples; fm/=2*kTimingSamples; const double speedup=bm/fm;
  if(!std::isfinite(speedup)||speedup<minimum) die("pre-registered speedup threshold not met");
  char uuid[33]; for(int i=0;i<16;++i) std::sprintf(uuid+2*i,"%02x",static_cast<unsigned char>(prop.uuid.bytes[i])); uuid[32]=0;
  // BEGIN HC probe bounded output contract
  std::printf("build_id=%s\n", V4_HC_PROBE_BUILD_ID);
  std::printf("device_uuid=%s driver=%d runtime=%d\n",uuid,driver,runtime);
  std::printf("input_hash=%016llx mix_hash=%016llx weight_hash=%016llx\n",(unsigned long long)ih,(unsigned long long)mh,(unsigned long long)wh);
  std::printf("tokens=2410 hidden=4096 hc=4 parity_cases=6 poison_cases=2 geometry_cases=10 hidden_bytes=19742720 normed_bytes=19742720 post_bytes=38560 comb_bytes=154240\n");
  std::printf("threshold min_speedup=%s\n",V4_HC_PROBE_MIN_SPEEDUP_TEXT);
  std::printf("baseline_ms=%.6f candidate_ms=%.6f speedup=%.9f abba_samples=%d\n",bm,fm,speedup,kTimingSamples);
  std::printf("hidden_mismatches=0 normed_mismatches=0 post_mismatches=0 comb_mismatches=0 guards=clean poison_a=clean poison_b=clean\n");
  std::printf("result=PASS\n");
  // END HC probe bounded output contract
  return 0;
}
}  // namespace probe

int main(int argc, char**) { return probe::run(argc); }
