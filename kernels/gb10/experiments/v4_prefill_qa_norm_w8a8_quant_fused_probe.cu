// SPDX-License-Identifier: AGPL-3.0-only

// Standalone GPU promotion probe for the isolated V4 q_a RMSNorm + W8A8
// activation quantizer. It is deliberately absent from serving and registry.

#include <cuda_runtime.h>

#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef V4_QA_PROBE_BUILD_ID
#error "V4_QA_PROBE_BUILD_ID must bind this probe to its build receipt"
#endif

// BEGIN q_a probe kernel contract
#include "../common/rms_norm_vanilla.cu"
#include "../common/w8a8_gemm_pipelined.cu"
#include "v4_prefill_qa_norm_w8a8_quant_fused.cu"

namespace probe {
constexpr unsigned kHidden = 1024;
constexpr unsigned kRmsThreads = 1024;
constexpr unsigned kQuantThreads = 256;
constexpr int kParityCases = 4;
constexpr int kPoisonCases = 2;
constexpr int kGeometryCases = 15;
constexpr int kTimingSamples = 8;
static_assert(kGeometryCases == 15, "malformed ABI census changed");

static void launch_baseline(const __nv_bfloat16* input,
                            const __nv_bfloat16* weight,
                            __nv_bfloat16* normed, unsigned char* output_fp8,
                            float* row_scale, unsigned tokens, float eps) {
  rms_norm_vanilla<<<dim3(tokens, 1, 1), dim3(kRmsThreads, 1, 1)>>>(
      input, weight, normed, kHidden, eps);
  quantize_a_fp8_rows<<<dim3(tokens, 1, 1), dim3(kQuantThreads, 1, 1)>>>(
      normed, output_fp8, row_scale, tokens, kHidden);
}

static void launch_candidate(const __nv_bfloat16* input,
                             const __nv_bfloat16* weight,
                             unsigned char* output_fp8, float* row_scale,
                             unsigned tokens, float eps) {
  v4_prefill_qa_norm_w8a8_quant_fused<<<dim3(tokens, 1, 1), dim3(kRmsThreads, 1, 1)>>>(
      input, weight, output_fp8, row_scale, tokens, kHidden, eps);
}
// END q_a probe kernel contract

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
  for (size_t i = 0; i < bytes; ++i) {
    hash = (hash ^ p[i]) * 1099511628211ull;
  }
  return hash;
}

template <class T>
struct Guarded {
  static constexpr size_t kGuard = 256;
  unsigned char* allocation = nullptr;
  T* data = nullptr;
  size_t count = 0;

  explicit Guarded(size_t elements) : count(elements) {
    cuda_ok(cudaMalloc(&allocation, bytes() + 2 * kGuard), "cudaMalloc");
    data = reinterpret_cast<T*>(allocation + kGuard);
    cuda_ok(cudaMemset(allocation, 0xa7, kGuard), "prefix guard");
    cuda_ok(cudaMemset(allocation + kGuard + bytes(), 0x7a, kGuard),
            "suffix guard");
  }
  ~Guarded() { cudaFree(allocation); }
  Guarded(const Guarded&) = delete;
  Guarded& operator=(const Guarded&) = delete;
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
    std::vector<unsigned char> host(bytes());
    cuda_ok(cudaMemcpy(host.data(), data, bytes(), cudaMemcpyDeviceToHost),
            "download");
    return host;
  }
  bool guards_clean() const {
    std::vector<unsigned char> prefix(kGuard), suffix(kGuard);
    cuda_ok(cudaMemcpy(prefix.data(), allocation, kGuard, cudaMemcpyDeviceToHost),
            "prefix_guard");
    cuda_ok(cudaMemcpy(suffix.data(), allocation + kGuard + bytes(), kGuard,
                       cudaMemcpyDeviceToHost),
            "suffix_guard");
    for (unsigned char value : prefix)
      if (value != 0xa7) return false;
    for (unsigned char value : suffix)
      if (value != 0x7a) return false;
    return true;
  }
};

static uint16_t bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + bias) >> 16);
}

// BEGIN q_a probe deterministic cases
enum Pattern { kZeros, kVaried, kExtremes };
struct CaseSpec {
  unsigned tokens;
  Pattern pattern;
  unsigned input_salt;
  unsigned weight_salt;
  float rms_eps;
};
constexpr CaseSpec kCases[kParityCases] = {
    {1, kZeros, 0, 11, 1.0e-6f},
    {7, kVaried, 17, 29, 2.0e-6f},
    {128, kExtremes, 31, 43, 5.0e-7f},
    {2410, kVaried, 59, 71, 3.0e-5f},
};

static void make_case(const CaseSpec& spec,
                      std::vector<__nv_bfloat16>& input,
                      std::vector<__nv_bfloat16>& weight) {
  auto* input_bits = reinterpret_cast<uint16_t*>(input.data());
  auto* weight_bits = reinterpret_cast<uint16_t*>(weight.data());
  constexpr uint16_t extremes[] = {
      0x7f7f, 0xff7f, 0x0001, 0x8001, 0x0000, 0x8000, 0x43e0, 0xc3e0};
  for (size_t i = 0; i < input.size(); ++i) {
    if (spec.pattern == kZeros) {
      input_bits[i] = (i & 1u) ? 0x8000u : 0x0000u;  // signed-zero
    } else if (spec.pattern == kExtremes) {
      input_bits[i] = extremes[(i + spec.input_salt) % 8];  // max-finite
    } else {
      const int value = static_cast<int>((i * 131u + spec.input_salt) % 509u) - 254;
      input_bits[i] = bf16(value * (1.0f / 128.0f));
      if ((i + spec.input_salt) % 4093u == 0)
        input_bits[i] = (i & 1u) ? 0x8000u : 0x0000u;  // signed-zero
    }
  }
  for (unsigned i = 0; i < kHidden; ++i) {
    const int value = static_cast<int>((i * 37u + spec.weight_salt) % 257u) - 128;
    weight_bits[i] = bf16(1.0f + value * (1.0f / 512.0f));
    if ((i + spec.weight_salt) % 251u == 0) weight_bits[i] = 0;
  }
}
// END q_a probe deterministic cases

static size_t mismatch_bytes(const std::vector<unsigned char>& lhs,
                             const std::vector<unsigned char>& rhs) {
  if (lhs.size() != rhs.size()) return lhs.size() + rhs.size();
  size_t mismatches = 0;
  for (size_t i = 0; i < lhs.size(); ++i) mismatches += lhs[i] != rhs[i];
  return mismatches;
}

struct Outputs {
  Guarded<unsigned char> fp8;
  Guarded<float> scale;
  explicit Outputs(unsigned tokens)
      : fp8(static_cast<size_t>(tokens) * kHidden), scale(tokens) {}
  void fill(unsigned char value) {
    fp8.fill(value);
    scale.fill(value);
  }
  bool guards_clean() const { return fp8.guards_clean() && scale.guards_clean(); }
};

// BEGIN q_a probe exact byte parity
constexpr unsigned char kPoisonA = 0x5a;
constexpr unsigned char kPoisonB = 0xc3;
// Full memcmp over FP8 bytes and FP32 scale bits under both opposite poisons;
// fp8_mismatches/scale_mismatches and poison_a/poison_b prove full writes.
static void run_parity_case(const CaseSpec& spec, uint64_t& input_hash,
                            uint64_t& weight_hash, size_t& fp8_mismatches,
                            size_t& scale_mismatches) {
  std::vector<__nv_bfloat16> host_input(
      static_cast<size_t>(spec.tokens) * kHidden);
  std::vector<__nv_bfloat16> host_weight(kHidden);
  make_case(spec, host_input, host_weight);
  input_hash = fnv(input_hash, host_input.data(), host_input.size() * sizeof(host_input[0]));
  weight_hash = fnv(weight_hash, host_weight.data(), host_weight.size() * sizeof(host_weight[0]));

  Guarded<__nv_bfloat16> input(host_input.size()), weight(kHidden);
  Guarded<__nv_bfloat16> normed(host_input.size());
  Outputs incumbent(spec.tokens), candidate(spec.tokens);
  input.upload(host_input);
  weight.upload(host_weight);
  for (int poison = 0; poison < kPoisonCases; ++poison) {
    normed.fill(poison ? kPoisonB : kPoisonA);
    incumbent.fill(poison ? kPoisonB : kPoisonA);
    candidate.fill(poison ? kPoisonA : kPoisonB);
    launch_baseline(input.data, weight.data, normed.data, incumbent.fp8.data,
                    incumbent.scale.data, spec.tokens, spec.rms_eps);
    launch_candidate(input.data, weight.data, candidate.fp8.data,
                     candidate.scale.data, spec.tokens, spec.rms_eps);
    cuda_ok(cudaGetLastError(), "parity launch");
    cuda_ok(cudaDeviceSynchronize(), "parity sync");
    const auto incumbent_fp8 = incumbent.fp8.download();
    const auto candidate_fp8 = candidate.fp8.download();
    const auto incumbent_scale = incumbent.scale.download();
    const auto candidate_scale = candidate.scale.download();
    fp8_mismatches += mismatch_bytes(incumbent_fp8, candidate_fp8);
    scale_mismatches += mismatch_bytes(incumbent_scale, candidate_scale);
    if (std::memcmp(incumbent_fp8.data(), candidate_fp8.data(),
                    incumbent_fp8.size()) != 0 ||
        std::memcmp(incumbent_scale.data(), candidate_scale.data(),
                    incumbent_scale.size()) != 0 ||
        !input.guards_clean() || !weight.guards_clean() ||
        !normed.guards_clean() || !incumbent.guards_clean() ||
        !candidate.guards_clean()) {
      die("exact byte parity or guards_clean failure");
    }
  }
}
// END q_a probe exact byte parity

// BEGIN q_a probe malformed ABI
enum class NullSlot { kNone, kInput, kWeight, kOutput, kScale };
static void malformed(const Guarded<__nv_bfloat16>& input,
                      const Guarded<__nv_bfloat16>& weight, Outputs& output,
                      dim3 grid, dim3 block, unsigned tokens, unsigned hidden,
                      float eps, NullSlot null_slot, const char* label) {
  output.fill(0x3c);
  const auto fp8_before = output.fp8.download();
  const auto scale_before = output.scale.download();
  const auto* input_ptr =
      null_slot == NullSlot::kInput ? static_cast<const __nv_bfloat16*>(nullptr) : input.data;
  const auto* weight_ptr =
      null_slot == NullSlot::kWeight ? static_cast<const __nv_bfloat16*>(nullptr) : weight.data;
  auto* fp8_ptr = null_slot == NullSlot::kOutput ? nullptr : output.fp8.data;
  auto* scale_ptr = null_slot == NullSlot::kScale ? nullptr : output.scale.data;
  v4_prefill_qa_norm_w8a8_quant_fused<<<grid, block>>>(
      input_ptr, weight_ptr, fp8_ptr, scale_ptr, tokens, hidden, eps);
  cuda_ok(cudaGetLastError(), label);
  cuda_ok(cudaDeviceSynchronize(), label);
  const bool expect_unchanged = output.fp8.download() == fp8_before &&
                                output.scale.download() == scale_before;
  const bool prefix_guard = output.fp8.guards_clean();
  const bool suffix_guard = output.scale.guards_clean();
  if (!expect_unchanged || !prefix_guard || !suffix_guard)
    die("malformed ABI wrote output");
}

static void run_malformed(const Guarded<__nv_bfloat16>& input,
                          const Guarded<__nv_bfloat16>& weight,
                          Outputs& output) {
  constexpr unsigned tokens = 2410;
  malformed(input, weight, output, dim3(tokens), dim3(512), tokens, kHidden,
            1.0e-6f, NullSlot::kNone, "block-x");
  malformed(input, weight, output, dim3(tokens), dim3(512, 2), tokens, kHidden,
            1.0e-6f, NullSlot::kNone, "block-y");
  malformed(input, weight, output, dim3(tokens), dim3(512, 1, 2), tokens,
            kHidden, 1.0e-6f, NullSlot::kNone, "block-z");
  malformed(input, weight, output, dim3(tokens - 1), dim3(kRmsThreads), tokens,
            kHidden, 1.0e-6f, NullSlot::kNone, "grid-x-small");
  malformed(input, weight, output, dim3(tokens + 1), dim3(kRmsThreads), tokens,
            kHidden, 1.0e-6f, NullSlot::kNone, "grid-x-large");
  malformed(input, weight, output, dim3(tokens, 2), dim3(kRmsThreads), tokens,
            kHidden, 1.0e-6f, NullSlot::kNone, "grid-y");
  malformed(input, weight, output, dim3(tokens, 1, 2), dim3(kRmsThreads),
            tokens, kHidden, 1.0e-6f, NullSlot::kNone, "grid-z");
  malformed(input, weight, output, dim3(1), dim3(kRmsThreads), 0, kHidden,
            1.0e-6f, NullSlot::kNone, "zero-tokens");
  malformed(input, weight, output, dim3(tokens), dim3(kRmsThreads), tokens,
            kHidden - 2, 1.0e-6f, NullSlot::kNone, "wrong-hidden");
  malformed(input, weight, output, dim3(tokens), dim3(kRmsThreads), tokens,
            kHidden, 0.0f, NullSlot::kNone, "eps-zero");
  malformed(input, weight, output, dim3(tokens), dim3(kRmsThreads), tokens,
            kHidden, NAN, NullSlot::kNone, "eps-nan");
  malformed(input, weight, output, dim3(tokens), dim3(kRmsThreads), tokens,
            kHidden, 1.0e-6f, NullSlot::kInput, "null-input");
  malformed(input, weight, output, dim3(tokens), dim3(kRmsThreads), tokens,
            kHidden, 1.0e-6f, NullSlot::kWeight, "null-weight");
  malformed(input, weight, output, dim3(tokens), dim3(kRmsThreads), tokens,
            kHidden, 1.0e-6f, NullSlot::kOutput, "null-output");
  malformed(input, weight, output, dim3(tokens), dim3(kRmsThreads), tokens,
            kHidden, 1.0e-6f, NullSlot::kScale, "null-scale");
}
// Contract: kGeometryCases = 15.
// END q_a probe malformed ABI

// BEGIN q_a probe ABBA timing contract
static float timed(bool baseline, Guarded<__nv_bfloat16>& input,
                   Guarded<__nv_bfloat16>& weight,
                   Guarded<__nv_bfloat16>& normed, Outputs& output,
                   const std::vector<__nv_bfloat16>& host_input,
                   const std::vector<__nv_bfloat16>& host_weight, float eps) {
  // Every input/output reset is before cudaEventRecord(begin), hence excluded.
  input.upload(host_input);
  weight.upload(host_weight);
  normed.fill(0x91);
  output.fill(0x4d);
  cudaEvent_t begin, end;
  cuda_ok(cudaEventCreate(&begin), "event create");
  cuda_ok(cudaEventCreate(&end), "event create");
  cuda_ok(cudaEventRecord(begin), "cudaEventRecord(begin)");
  if (baseline) {
    launch_baseline(input.data, weight.data, normed.data, output.fp8.data,
                    output.scale.data, 2410, eps);
  } else {
    launch_candidate(input.data, weight.data, output.fp8.data,
                     output.scale.data, 2410, eps);
  }
  cuda_ok(cudaEventRecord(end), "cudaEventRecord(end)");
  cuda_ok(cudaEventSynchronize(end), "event sync");
  float milliseconds = 0.0f;
  cuda_ok(cudaEventElapsedTime(&milliseconds, begin, end),
          "cudaEventElapsedTime");
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return milliseconds;
}
// Production-shape samples execute ABBA; baseline_ms/candidate_ms and speedup
// aggregate the same kTimingSamples after warmups, excluding all resets.
// END q_a probe ABBA timing contract

static double threshold(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <min_end_to_end_speedup>\n", argv[0]);
    std::exit(2);
  }
  const char* input = argv[1];
  bool digit = false, dot = false, exponent = false, exponent_digit = false;
  for (size_t i = 0; input[i] != '\0'; ++i) {
    const unsigned char ch = static_cast<unsigned char>(input[i]);
    if (std::isdigit(ch)) {
      digit = true;
      if (exponent) exponent_digit = true;
    } else if (ch == '.' && !dot && !exponent) {
      dot = true;
    } else if ((ch == 'e' || ch == 'E') && digit && !exponent) {
      exponent = true;
    } else if ((ch == '+' || ch == '-') && exponent && !exponent_digit &&
               i > 0 && (input[i - 1] == 'e' || input[i - 1] == 'E')) {
      continue;
    } else {
      std::fprintf(stderr, "invalid explicit numeric threshold\n");
      std::exit(2);
    }
  }
  if (!digit || (exponent && !exponent_digit)) {
    std::fprintf(stderr, "invalid explicit numeric threshold\n");
    std::exit(2);
  }
  errno = 0;
  char* end = nullptr;
  const double value = std::strtod(input, &end);
  if (errno != 0 || end == input || *end != '\0' || !std::isfinite(value) ||
      value <= 1.0 || value > 100.0) {
    std::fprintf(stderr, "invalid explicit numeric threshold\n");
    std::exit(2);
  }
  return value;
}

static int run(int argc, char** argv) {
  const double minimum = threshold(argc, argv);  // Before every CUDA call.
  int driver = 0, runtime = 0, device = 0;
  cudaDeviceProp properties{};
  cuda_ok(cudaDriverGetVersion(&driver), "driver");
  cuda_ok(cudaRuntimeGetVersion(&runtime), "runtime");
  cuda_ok(cudaGetDevice(&device), "device");
  cuda_ok(cudaGetDeviceProperties(&properties, device), "properties");

  uint64_t input_hash = 1469598103934665603ull;
  uint64_t weight_hash = 1469598103934665603ull;
  size_t fp8_mismatches = 0, scale_mismatches = 0;
  for (const CaseSpec& spec : kCases) {
    run_parity_case(spec, input_hash, weight_hash, fp8_mismatches,
                    scale_mismatches);
  }
  if (fp8_mismatches != 0 || scale_mismatches != 0)
    die("exact byte parity failure");

  const CaseSpec& production = kCases[3];
  std::vector<__nv_bfloat16> host_input(
      static_cast<size_t>(production.tokens) * kHidden);
  std::vector<__nv_bfloat16> host_weight(kHidden);
  make_case(production, host_input, host_weight);
  Guarded<__nv_bfloat16> input(host_input.size()), weight(kHidden);
  Guarded<__nv_bfloat16> normed(host_input.size());
  Outputs output(production.tokens);
  input.upload(host_input);
  weight.upload(host_weight);
  run_malformed(input, weight, output);

  for (int warmup = 0; warmup < 2; ++warmup) {
    timed(true, input, weight, normed, output, host_input, host_weight,
          production.rms_eps);
    timed(false, input, weight, normed, output, host_input, host_weight,
          production.rms_eps);
  }
  double baseline_ms = 0.0, candidate_ms = 0.0;
  for (int sample = 0; sample < kTimingSamples; ++sample) {
    baseline_ms += timed(true, input, weight, normed, output, host_input,
                         host_weight, production.rms_eps);
    candidate_ms += timed(false, input, weight, normed, output, host_input,
                          host_weight, production.rms_eps);
    candidate_ms += timed(false, input, weight, normed, output, host_input,
                          host_weight, production.rms_eps);
    baseline_ms += timed(true, input, weight, normed, output, host_input,
                         host_weight, production.rms_eps);
  }
  baseline_ms /= 2 * kTimingSamples;
  candidate_ms /= 2 * kTimingSamples;
  const double speedup = baseline_ms / candidate_ms;
  if (!std::isfinite(speedup) || baseline_ms <= 0.0 || candidate_ms <= 0.0 ||
      speedup < minimum)
    die("pre-registered speedup threshold not met");

  char uuid[33];
  for (int i = 0; i < 16; ++i)
    std::sprintf(uuid + 2 * i, "%02x",
                 static_cast<unsigned char>(properties.uuid.bytes[i]));
  uuid[32] = '\0';

  // BEGIN q_a probe bounded output contract
  std::printf("build_id=%s\n", V4_QA_PROBE_BUILD_ID);
  std::printf("device_uuid=%s driver=%d runtime=%d\n", uuid, driver, runtime);
  std::printf("input_hash=%016llx weight_hash=%016llx\n",
              static_cast<unsigned long long>(input_hash),
              static_cast<unsigned long long>(weight_hash));
  std::printf("shapes=1,7,128,2410 hidden=1024 parity_cases=4 poison_cases=2 geometry_cases=15 fp8_bytes=2467840 scale_bytes=9640 normed_bytes=4935680\n");
  std::printf("threshold min_speedup=%.17g\n", minimum);
  std::printf("baseline_ms=%.6f candidate_ms=%.6f speedup=%.9f abba_samples=%d\n",
              baseline_ms, candidate_ms, speedup, kTimingSamples);
  std::printf("fp8_mismatches=0 scale_mismatches=0 guards=clean poison_a=clean poison_b=clean\n");
  std::printf("result=PASS\n");
  // END q_a probe bounded output contract
  return 0;
}
}  // namespace probe

int main(int argc, char** argv) { return probe::run(argc, argv); }
