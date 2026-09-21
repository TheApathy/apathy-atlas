// SPDX-License-Identifier: AGPL-3.0-only

// Standalone, GPU-runnable admission probe. It is deliberately absent from
// the Atlas registry and serving path.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef V4_TC2_WARP0_PROBE_BUILD_ID
#error "V4_TC2_WARP0_PROBE_BUILD_ID must bind this probe to its build receipt"
#endif
#ifndef V4_TC2_WARP0_PROBE_MIN_SPEEDUP
#error "V4_TC2_WARP0_PROBE_MIN_SPEEDUP must bind the binary64 threshold"
#endif
#ifndef V4_TC2_WARP0_PROBE_MIN_SPEEDUP_TEXT
#error "V4_TC2_WARP0_PROBE_MIN_SPEEDUP_TEXT must bind the canonical threshold text"
#endif

// BEGIN TC2 warp0 probe kernel contract
// Rename only the incumbent TC2 symbols that collide with the standalone
// candidate, then compile both exact source files into this admission binary.
#define tc2_ldm_x4 v4_probe_incumbent_ldm_x4
#define tc2_ldm_x4_trans v4_probe_incumbent_ldm_x4_trans
#define tc2_pack_bf16 v4_probe_incumbent_pack_bf16
#define prefill_attn_compressed_tc2 v4_probe_incumbent_tc2
#include "../deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu"
#undef prefill_attn_compressed_tc2
#undef tc2_pack_bf16
#undef tc2_ldm_x4_trans
#undef tc2_ldm_x4
#include "v4_prefill_attn_compressed_tc2_warp0.cu"

namespace probe {
constexpr unsigned kHeadDim = 512;
constexpr unsigned kKvHeads = 1;
constexpr int kParityCases = 9;
constexpr int kPoisonCases = 2;
constexpr int kTimingSamples = 6;
constexpr int kMalformedCases = 36;
constexpr double kMinSpeedup = V4_TC2_WARP0_PROBE_MIN_SPEEDUP;
static_assert(kMinSpeedup > 1.0 && kMinSpeedup <= 100.0,
              "build-bound threshold must require a finite win");

static void launch(bool incumbent, const __nv_bfloat16* q,
                   const __nv_bfloat16* k, const __nv_bfloat16* v,
                   const __nv_bfloat16* kc, const __nv_bfloat16* vc,
                   const float* sinks, __nv_bfloat16* output,
                   unsigned seq_len, unsigned heads, unsigned n_comp,
                   unsigned ratio, unsigned sliding_window) {
  dim3 grid(heads, (seq_len + 15u) / 16u, 1);
  dim3 block(128, 1, 1);
  const float scale = 0.04419417382415922f;  // 1/sqrt(512)
  if (incumbent) {
    v4_probe_incumbent_tc2<<<grid, block>>>(
        q, k, v, kc, vc, sinks, output, seq_len, heads, kKvHeads, kHeadDim,
        n_comp, ratio, sliding_window, scale);
  } else {
    v4_prefill_attn_compressed_tc2_warp0<<<grid, block>>>(
        q, k, v, kc, vc, sinks, output, seq_len, heads, kKvHeads, kHeadDim,
        n_comp, ratio, sliding_window, scale);
  }
}

static void launch_candidate_raw(
    dim3 grid, dim3 block, const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, const __nv_bfloat16* kc,
    const __nv_bfloat16* vc, const float* sinks, __nv_bfloat16* output,
    unsigned seq_len, unsigned heads, unsigned kv_heads, unsigned head_dim,
    unsigned n_comp, unsigned ratio, unsigned sliding_window, float scale) {
  v4_prefill_attn_compressed_tc2_warp0<<<grid, block>>>(
      q, k, v, kc, vc, sinks, output, seq_len, heads, kv_heads, head_dim,
      n_comp, ratio, sliding_window, scale);
}
// END TC2 warp0 probe kernel contract

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
    if ((reinterpret_cast<uintptr_t>(data) & 255u) != 0u)
      die("guarded payload lost cudaMalloc alignment");
    cuda_ok(cudaMemset(allocation, 0xa7, kGuard), "prefix guard");
    cuda_ok(cudaMemset(allocation + kGuard + bytes(), 0x7a, kGuard),
            "suffix guard");
  }
  ~Guarded() { cudaFree(allocation); }
  size_t bytes() const { return count * sizeof(T); }
  void fill(unsigned char value) {
    cuda_ok(cudaMemset(data, value, bytes()), "fill");
  }
  void upload(const std::vector<T>& host) {
    if (host.size() != count) die("host/device extent mismatch");
    cuda_ok(cudaMemcpy(data, host.data(), bytes(), cudaMemcpyHostToDevice),
            "upload");
  }
  std::vector<unsigned char> download() const {
    std::vector<unsigned char> out(bytes());
    cuda_ok(cudaMemcpy(out.data(), data, bytes(), cudaMemcpyDeviceToHost),
            "download");
    return out;
  }
  bool payload_equals(const std::vector<T>& expected) const {
    if (expected.size() != count) return false;
    const auto actual = download();
    return std::memcmp(actual.data(), expected.data(), bytes()) == 0;
  }
  bool all_payload(unsigned char expected) const {
    const auto actual = download();
    return std::all_of(actual.begin(), actual.end(),
                       [expected](unsigned char value) { return value == expected; });
  }
  bool prefix_guard() const { return guard_is(0, 0xa7); }
  bool suffix_guard() const { return guard_is(kGuard + bytes(), 0x7a); }
  bool guards_clean() const { return prefix_guard() && suffix_guard(); }

private:
  bool guard_is(size_t offset, unsigned char expected) const {
    std::vector<unsigned char> host(kGuard);
    cuda_ok(cudaMemcpy(host.data(), allocation + offset, kGuard,
                       cudaMemcpyDeviceToHost), "download guard");
    return std::all_of(host.begin(), host.end(),
                       [expected](unsigned char x) { return x == expected; });
  }
};

// BEGIN TC2 warp0 deterministic cases
// Contract: kParityCases = 9. This matrix crosses independent raw_alias and
// compressed_alias paths, sinks/null sinks, partial 16-row tails, and
// raw/compressed/sliding boundaries. Four T=2410 rows cover CSA mixes and the
// exact dense K=V=Kc=Vc topology with and without sinks.
struct CaseSpec {
  unsigned seq_len, n_comp, ratio, sliding_window;
  bool raw_alias, compressed_alias, with_sinks, all_kv_alias;
  unsigned salt;
};
constexpr CaseSpec kCases[kParityCases] = {
    {31, 7, 4, 0, true, true, true, false, 11},
    {33, 8, 4, 16, false, false, true, false, 23},
    {47, 0, 4, 17, true, false, false, false, 37},
    {65, 16, 4, 32, false, true, true, false, 53},
    {129, 1, 128, 128, true, true, true, false, 71},
    {2410, 602, 4, 128, true, true, true, false, 89},
    {2410, 18, 128, 128, true, true, true, false, 107},
    // Exact dense production pointer topology: K=V=Kc=Vc. The compressed
    // extent is zero, so the four-way read-only alias is explicitly legal.
    {2410, 0, 1, 128, true, true, true, true, 149},
    {2410, 0, 1, 128, true, true, false, true, 163},
};

static uint16_t bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + bias) >> 16);
}

static std::vector<__nv_bfloat16> make_bf16(size_t count, unsigned salt) {
  std::vector<__nv_bfloat16> out(count);
  auto* raw = reinterpret_cast<uint16_t*>(out.data());
  for (size_t i = 0; i < count; ++i) {
    const int v = static_cast<int>((i * 131u + salt * 29u) % 509u) - 254;
    float value = static_cast<float>(v) * (1.0f / 512.0f);
    if ((i + salt) % 65537u == 0) value = (i & 1u) ? -0.0f : 0.0f;
    raw[i] = bf16(value);
  }
  return out;
}

static std::vector<float> make_sinks(unsigned heads, unsigned salt) {
  std::vector<float> out(heads);
  for (unsigned i = 0; i < heads; ++i) {
    const int v = static_cast<int>((i * 17u + salt) % 41u) - 20;
    out[i] = static_cast<float>(v) * (1.0f / 16.0f);
  }
  return out;
}
// END TC2 warp0 deterministic cases

static size_t mismatch_bytes(const std::vector<unsigned char>& a,
                             const std::vector<unsigned char>& b) {
  if (a.size() != b.size()) return a.size() + b.size();
  size_t mismatches = 0;
  for (size_t i = 0; i < a.size(); ++i) mismatches += a[i] != b[i];
  return mismatches;
}

struct Fixture {
  CaseSpec spec;
  unsigned heads;
  Guarded<__nv_bfloat16> q;
  Guarded<__nv_bfloat16> k;
  Guarded<__nv_bfloat16> v;
  Guarded<__nv_bfloat16> kc;
  Guarded<__nv_bfloat16> vc;
  Guarded<float> sinks;
  Guarded<__nv_bfloat16> incumbent;
  Guarded<__nv_bfloat16> candidate;
  std::vector<__nv_bfloat16> host_q, host_k, host_v, host_kc, host_vc;
  std::vector<float> host_sinks;

  Fixture(CaseSpec value, unsigned head_count, uint64_t& input_hash)
      : spec(value), heads(head_count),
        q(static_cast<size_t>(value.seq_len) * head_count * kHeadDim),
        k(static_cast<size_t>(value.seq_len) * kHeadDim),
        v(static_cast<size_t>(value.seq_len) * kHeadDim),
        kc(std::max<size_t>(1, static_cast<size_t>(value.n_comp) * kHeadDim)),
        vc(std::max<size_t>(1, static_cast<size_t>(value.n_comp) * kHeadDim)),
        sinks(head_count),
        incumbent(static_cast<size_t>(value.seq_len) * head_count * kHeadDim),
        candidate(static_cast<size_t>(value.seq_len) * head_count * kHeadDim) {
    host_q = make_bf16(q.count, value.salt);
    host_k = make_bf16(k.count, value.salt + 1);
    host_v = make_bf16(v.count, value.salt + 2);
    host_kc = make_bf16(kc.count, value.salt + 3);
    host_vc = make_bf16(vc.count, value.salt + 4);
    host_sinks = make_sinks(heads, value.salt + 5);
    reset_inputs();
    for (const auto* bytes : {&host_q, &host_k, &host_v, &host_kc, &host_vc})
      input_hash = fnv(input_hash, bytes->data(), bytes->size() * sizeof((*bytes)[0]));
    input_hash = fnv(input_hash, host_sinks.data(), host_sinks.size() * sizeof(float));
  }

  const __nv_bfloat16* raw_k() const { return k.data; }
  const __nv_bfloat16* raw_v() const { return spec.raw_alias ? k.data : v.data; }
  const __nv_bfloat16* comp_k() const {
    return spec.all_kv_alias ? k.data : kc.data;
  }
  const __nv_bfloat16* comp_v() const {
    if (spec.all_kv_alias) return k.data;
    return spec.compressed_alias ? kc.data : vc.data;
  }
  const float* sink_ptr() const { return spec.with_sinks ? sinks.data : nullptr; }
  void reset_inputs() {
    q.upload(host_q); k.upload(host_k); v.upload(host_v); kc.upload(host_kc);
    vc.upload(host_vc); sinks.upload(host_sinks);
  }
  bool inputs_immutable() const {
    return q.payload_equals(host_q) && k.payload_equals(host_k) &&
           v.payload_equals(host_v) && kc.payload_equals(host_kc) &&
           vc.payload_equals(host_vc) && sinks.payload_equals(host_sinks);
  }
  bool input_guards_clean() const {
    return q.guards_clean() && k.guards_clean() && v.guards_clean() &&
           kc.guards_clean() && vc.guards_clean() && sinks.guards_clean();
  }
  bool output_guards_clean() const {
    return incumbent.guards_clean() && candidate.guards_clean();
  }
  bool all_guards_clean() const {
    return input_guards_clean() && output_guards_clean();
  }
};

// BEGIN TC2 warp0 exact byte parity
constexpr unsigned char kPoisonA = 0x5a;
constexpr unsigned char kPoisonB = 0xc3;
// Contract: kPoisonCases = 2. Full-output memcmp and mismatch_bytes run under
// both opposite poisons. prefix_guard/suffix_guard, poison_a, and poison_b are
// reported only after every byte and both guard regions pass.
static size_t exact_parity(Fixture& f) {
  size_t mismatches = 0;
  for (int poison = 0; poison < kPoisonCases; ++poison) {
    f.incumbent.fill(poison ? kPoisonB : kPoisonA);
    f.candidate.fill(poison ? kPoisonA : kPoisonB);
    f.reset_inputs();
    launch(true, f.q.data, f.raw_k(), f.raw_v(), f.comp_k(), f.comp_v(),
           f.sink_ptr(), f.incumbent.data, f.spec.seq_len, f.heads,
           f.spec.n_comp, f.spec.ratio, f.spec.sliding_window);
    cuda_ok(cudaGetLastError(), "incumbent parity launch");
    cuda_ok(cudaDeviceSynchronize(), "incumbent parity sync");
    if (!f.inputs_immutable() || !f.all_guards_clean())
      die("incumbent mutated input or guard");
    const auto incumbent = f.incumbent.download();

    f.reset_inputs();
    launch(false, f.q.data, f.raw_k(), f.raw_v(), f.comp_k(), f.comp_v(),
           f.sink_ptr(), f.candidate.data, f.spec.seq_len, f.heads,
           f.spec.n_comp, f.spec.ratio, f.spec.sliding_window);
    cuda_ok(cudaGetLastError(), "candidate parity launch");
    cuda_ok(cudaDeviceSynchronize(), "candidate parity sync");
    if (!f.inputs_immutable() || !f.all_guards_clean())
      die("candidate mutated input or guard");
    const auto candidate = f.candidate.download();
    mismatches += mismatch_bytes(incumbent, candidate);
    if (std::memcmp(incumbent.data(), candidate.data(), incumbent.size()) != 0 ||
        !f.inputs_immutable() || !f.all_guards_clean())
      die("exact output parity or guard failure");
  }
  return mismatches;
}
// END TC2 warp0 exact byte parity

// BEGIN TC2 warp0 malformed ABI no-write
struct MalformedLaunch {
  dim3 grid, block;
  const __nv_bfloat16 *q, *k, *v, *kc, *vc;
  const float* sinks;
  __nv_bfloat16* output;
  unsigned seq_len, heads, kv_heads, head_dim, n_comp, ratio, window;
  float scale;
};

template <class T> static T* byte_offset(T* pointer, size_t bytes) {
  return reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(pointer) + bytes);
}

static int malformed_no_write(uint64_t& input_hash) {
  CaseSpec spec{31, 7, 4, 16, false, false, true, false, 211};
  Fixture f(spec, 4, input_hash);
  auto valid = [&]() {
    return MalformedLaunch{
        dim3(f.heads, (f.spec.seq_len + 15u) / 16u, 1), dim3(128, 1, 1),
        f.q.data, f.raw_k(), f.raw_v(), f.comp_k(), f.comp_v(), f.sinks.data,
        f.candidate.data, f.spec.seq_len, f.heads, kKvHeads, kHeadDim,
        f.spec.n_comp, f.spec.ratio, f.spec.sliding_window,
        0.04419417382415922f};
  };
  int malformed_cases = 0;
  auto run = [&](const char* label, MalformedLaunch test) {
    f.reset_inputs();
    f.candidate.fill(0xa5);
    launch_candidate_raw(test.grid, test.block, test.q, test.k, test.v, test.kc,
                         test.vc, test.sinks, test.output, test.seq_len,
                         test.heads, test.kv_heads, test.head_dim, test.n_comp,
                         test.ratio, test.window, test.scale);
    cuda_ok(cudaGetLastError(), label);
    cuda_ok(cudaDeviceSynchronize(), "malformed cudaDeviceSynchronize");
    if (!f.candidate.all_payload(0xa5) || !f.inputs_immutable() ||
        !f.all_guards_clean())
      die(label);
    ++malformed_cases;
  };

  auto test = valid(); test.q = nullptr; run("null-q", test);
  test = valid(); test.k = nullptr; run("null-k", test);
  test = valid(); test.v = nullptr; run("null-v", test);
  test = valid(); test.kc = nullptr; run("null-kc", test);
  test = valid(); test.vc = nullptr; run("null-vc", test);
  test = valid(); test.output = nullptr; run("null-o", test);
  test = valid(); test.q = byte_offset(f.q.data, 2); run("misaligned-q", test);
  test = valid(); test.k = byte_offset(f.k.data, 2); run("misaligned-k", test);
  test = valid(); test.v = byte_offset(f.v.data, 2); run("misaligned-v", test);
  test = valid(); test.kc = byte_offset(f.kc.data, 2); run("misaligned-kc", test);
  test = valid(); test.vc = byte_offset(f.vc.data, 2); run("misaligned-vc", test);
  test = valid(); test.sinks = byte_offset(f.sinks.data, 2); run("misaligned-sinks", test);
  test = valid(); test.output = byte_offset(f.candidate.data, 2); run("misaligned-o", test);
  test = valid(); test.output = f.q.data; run("o-alias-q", test);
  test = valid(); test.output = f.k.data; run("o-alias-k", test);
  test = valid(); test.output = f.v.data; run("o-alias-v", test);
  test = valid(); test.output = f.kc.data; run("o-alias-kc", test);
  test = valid(); test.output = f.vc.data; run("o-alias-vc", test);
  test = valid(); test.output = reinterpret_cast<__nv_bfloat16*>(f.sinks.data); run("o-alias-sinks", test);
  test = valid(); test.output = byte_offset(f.q.data, 16); run("o-overlap-q-offset", test);
  test = valid(); test.output = byte_offset(f.k.data, 16); run("o-overlap-k-offset", test);
  test = valid(); test.scale = NAN; run("scale-nan", test);
  test = valid(); test.scale = INFINITY; run("scale-pos-inf", test);
  test = valid(); test.scale = -INFINITY; run("scale-neg-inf", test);
  test = valid(); test.scale = 0.0f; run("scale-zero", test);
  test = valid(); test.scale = -1.0f; run("scale-negative", test);
  test = valid(); test.head_dim = 511; run("head-dim", test);
  test = valid(); test.kv_heads = 0; run("kv-heads-zero", test);
  test = valid(); test.kv_heads = 3; run("head-ratio", test);
  test = valid(); test.ratio = 0; run("ratio-zero", test);
  test = valid(); test.block = dim3(64, 1, 1); run("block-x", test);
  test = valid(); test.block = dim3(128, 2, 1); run("block-y", test);
  test = valid(); test.block = dim3(128, 1, 2); run("block-z", test);
  test = valid(); test.grid = dim3(f.heads + 1, 2, 1); run("grid-x", test);
  test = valid(); test.grid = dim3(f.heads, 3, 1); run("grid-y", test);
  test = valid(); test.grid = dim3(f.heads, 2, 2); run("grid-z", test);

  if (malformed_cases != kMalformedCases) die("malformed_cases census");
  return malformed_cases;
}
// END TC2 warp0 malformed ABI no-write

// BEGIN TC2 warp0 ABBA timing contract
constexpr unsigned kTimingTokens = 2410;
constexpr unsigned kTimingHeads = 64;
constexpr unsigned kTimingNComp = 602;
constexpr unsigned kTimingRatio = 4;
constexpr unsigned kTimingWindow = 128;
constexpr unsigned kDenseNComp = 0;
constexpr unsigned kDenseRatio = 1;

static float timed(bool incumbent, Fixture& f) {
  Guarded<__nv_bfloat16>& output = incumbent ? f.incumbent : f.candidate;
  f.reset_inputs();
  output.fill(0);
  cudaEvent_t begin, end;
  cuda_ok(cudaEventCreate(&begin), "event create");
  cuda_ok(cudaEventCreate(&end), "event create");
  cuda_ok(cudaEventRecord(begin), "cudaEventRecord");
  launch(incumbent, f.q.data, f.raw_k(), f.raw_v(), f.comp_k(), f.comp_v(),
         f.sink_ptr(), output.data, f.spec.seq_len, f.heads, f.spec.n_comp,
         f.spec.ratio, f.spec.sliding_window);
  cuda_ok(cudaEventRecord(end), "cudaEventRecord");
  cuda_ok(cudaEventSynchronize(end), "event sync");
  cuda_ok(cudaGetLastError(), "timing launch");
  float milliseconds = 0;
  cuda_ok(cudaEventElapsedTime(&milliseconds, begin, end),
          "cudaEventElapsedTime");
  cudaEventDestroy(begin); cudaEventDestroy(end);
  if (!f.inputs_immutable() || !f.all_guards_clean())
    die("timing input or guard mutation");
  return milliseconds;
}
struct TimingResult { double baseline_ms, candidate_ms, speedup; };

static TimingResult measure_abba(Fixture& timing, const char* route) {
  for (int i = 0; i < 2; ++i) { timed(true, timing); timed(false, timing); }
  double baseline_ms = 0.0, candidate_ms = 0.0;
  // Each CSA or dense production route is independently ABBA; results are
  // never aggregated across routes. Dense uses K=V=Kc=Vc exactly as dispatch.
  for (int i = 0; i < kTimingSamples; ++i) {
    baseline_ms += timed(true, timing);
    candidate_ms += timed(false, timing);
    candidate_ms += timed(false, timing);
    baseline_ms += timed(true, timing);
  }
  baseline_ms /= 2 * kTimingSamples;
  candidate_ms /= 2 * kTimingSamples;
  const double speedup = baseline_ms / candidate_ms;
  if (!std::isfinite(baseline_ms) || !std::isfinite(candidate_ms) ||
      !std::isfinite(speedup) || baseline_ms <= 0.0 || candidate_ms <= 0.0 ||
      speedup < kMinSpeedup)
    die(route);
  return {baseline_ms, candidate_ms, speedup};
}
// Production-shape order is ABBA for each of kTimingSamples; csa and dense
// baseline_ms, candidate_ms, and speedup use separate exact-shape fixtures.
// END TC2 warp0 ABBA timing contract

static int run() {
  int driver = 0, runtime = 0, device = 0;
  cudaDeviceProp prop{};
  cuda_ok(cudaDriverGetVersion(&driver), "driver");
  cuda_ok(cudaRuntimeGetVersion(&runtime), "runtime");
  cuda_ok(cudaGetDevice(&device), "device");
  cuda_ok(cudaGetDeviceProperties(&prop, device), "properties");

  uint64_t input_hash = 1469598103934665603ull;
  size_t output_mismatches = 0;
  for (const auto& spec : kCases) {
    Fixture fixture(spec, spec.seq_len == kTimingTokens ? kTimingHeads : 4,
                    input_hash);
    output_mismatches += exact_parity(fixture);
  }
  if (output_mismatches != 0) die("exact byte parity failed");
  const int malformed_cases = malformed_no_write(input_hash);

  TimingResult csa{};
  {
    CaseSpec csa_spec{kTimingTokens, kTimingNComp, kTimingRatio,
                      kTimingWindow, true, true, true, false, 229};
    Fixture timing(csa_spec, kTimingHeads, input_hash);
    csa = measure_abba(timing, "CSA pre-registration threshold not met");
  }
  TimingResult dense{};
  {
    CaseSpec dense_spec{kTimingTokens, kDenseNComp, kDenseRatio,
                        kTimingWindow, true, true, true, true, 241};
    Fixture timing(dense_spec, kTimingHeads, input_hash);
    dense = measure_abba(timing, "dense pre-registration threshold not met");
  }

  char uuid[33];
  for (int i = 0; i < 16; ++i)
    std::sprintf(uuid + 2 * i, "%02x", static_cast<unsigned char>(prop.uuid.bytes[i]));
  uuid[32] = 0;
  // BEGIN TC2 warp0 bounded output contract
  std::printf("build_id=%s\n", V4_TC2_WARP0_PROBE_BUILD_ID);
  std::printf("device_uuid=%s driver=%d runtime=%d\n", uuid, driver, runtime);
  std::printf("input_hash=%016llx\n", (unsigned long long)input_hash);
  std::printf("parity_cases=9 poison_cases=2 malformed_cases=%d\n", malformed_cases);
  std::printf("csa_timing_tokens=2410 heads=64 head_dim=512 n_comp=602 ratio=4 window=128\n");
  std::printf("dense_timing_tokens=2410 heads=64 head_dim=512 n_comp=0 ratio=1 window=128 kv_alias=K=V=Kc=Vc\n");
  std::printf("threshold min_speedup=%s\n", V4_TC2_WARP0_PROBE_MIN_SPEEDUP_TEXT);
  std::printf("csa_baseline_ms=%.6f csa_candidate_ms=%.6f csa_speedup=%.9f abba_samples=%d\n",
              csa.baseline_ms, csa.candidate_ms, csa.speedup, kTimingSamples);
  std::printf("dense_baseline_ms=%.6f dense_candidate_ms=%.6f dense_speedup=%.9f abba_samples=%d\n",
              dense.baseline_ms, dense.candidate_ms, dense.speedup, kTimingSamples);
  std::printf("output_mismatches=0 inputs=immutable input_guards=clean output_guards=clean poison_a=clean poison_b=clean\n");
  std::printf("result=PASS\n");
  // END TC2 warp0 bounded output contract
  return 0;
}
}  // namespace probe

int main() { return probe::run(); }
