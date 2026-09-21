// SPDX-License-Identifier: AGPL-3.0-only

// Standalone downstream parity and promotion probe for the isolated DeepSeek
// V4 q_a norm-to-W8A8 experiment. This translation unit is deliberately
// absent from the kernel registry and serving dispatch.

#include <cuda_runtime.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef V4_QA_PIPELINE_PROBE_BUILD_ID
#error "V4_QA_PIPELINE_PROBE_BUILD_ID must bind this probe to its receipt"
#endif

// BEGIN q_a W8A8 pipeline kernel contract
#include "../common/rms_norm_vanilla.cu"
#include "../common/w8a8_gemm_pipelined.cu"
#include "v4_prefill_qa_norm_w8a8_quant_fused.cu"

namespace pipeline_probe {
constexpr unsigned kProductionM = 2410;
constexpr unsigned kN = 32768;
constexpr unsigned kK = 1024;
constexpr unsigned kRmsThreads = 1024;
constexpr unsigned kQuantThreads = 256;
constexpr int kParityCases = 3;
constexpr int kPoisonCases = 2;
constexpr int kWarmupRounds = 2;
constexpr int kTimingSamples = 4;

static unsigned div_up(unsigned value, unsigned divisor) {
  return (value + divisor - 1) / divisor;
}

static void launch_baseline(const __nv_bfloat16* input,
                            const __nv_bfloat16* norm_weight,
                            __nv_bfloat16* normed, unsigned char* activation,
                            float* row_scale, const unsigned char* weight_fp8,
                            const float* block_scale, __nv_bfloat16* output,
                            unsigned m, float eps) {
  rms_norm_vanilla<<<dim3(m, 1, 1), dim3(kRmsThreads, 1, 1)>>>(
      input, norm_weight, normed, kK, eps);
  quantize_a_fp8_rows<<<dim3(m, 1, 1), dim3(kQuantThreads, 1, 1)>>>(
      normed, activation, row_scale, m, kK);
  w8a8_gemm_pipelined<<<dim3(kN / 32, div_up(m, 128), 1), dim3(256, 1, 1)>>>(
      activation, row_scale, weight_fp8, block_scale, output, m, kN, kK);
}

static void launch_candidate(const __nv_bfloat16* input,
                             const __nv_bfloat16* norm_weight,
                             unsigned char* activation, float* row_scale,
                             const unsigned char* weight_fp8,
                             const float* block_scale, __nv_bfloat16* output,
                             unsigned m, float eps) {
  v4_prefill_qa_norm_w8a8_quant_fused<<<dim3(m, 1, 1), dim3(kRmsThreads, 1, 1)>>>(
      input, norm_weight, activation, row_scale, m, kK, eps);
  w8a8_gemm_pipelined<<<dim3(kN / 32, div_up(m, 128), 1), dim3(256, 1, 1)>>>(
      activation, row_scale, weight_fp8, block_scale, output, m, kN, kK);
}
// END q_a W8A8 pipeline kernel contract

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
  const auto* cursor = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < bytes; ++i) {
    hash = (hash ^ cursor[i]) * 1099511628211ull;
  }
  return hash;
}

template <class T>
struct Guarded {
  static constexpr size_t kGuardBytes = 256;
  unsigned char* allocation = nullptr;
  T* data = nullptr;
  size_t count = 0;

  explicit Guarded(size_t elements) : count(elements) {
    cuda_ok(cudaMalloc(&allocation, bytes() + 2 * kGuardBytes), "cudaMalloc");
    data = reinterpret_cast<T*>(allocation + kGuardBytes);
    cuda_ok(cudaMemset(allocation, 0xa7, kGuardBytes), "prefix guard");
    cuda_ok(cudaMemset(allocation + kGuardBytes + bytes(), 0x7a, kGuardBytes),
            "suffix guard");
  }
  ~Guarded() { cudaFree(allocation); }
  Guarded(const Guarded&) = delete;
  Guarded& operator=(const Guarded&) = delete;

  size_t bytes() const { return count * sizeof(T); }
  void fill(unsigned char poison) {
    cuda_ok(cudaMemset(data, poison, bytes()), "output fill");
  }
  void upload(const std::vector<T>& host) {
    if (host.size() != count) die("host/device extent mismatch");
    cuda_ok(cudaMemcpy(data, host.data(), bytes(), cudaMemcpyHostToDevice),
            "input upload");
  }
  std::vector<unsigned char> download() const {
    std::vector<unsigned char> host(bytes());
    cuda_ok(cudaMemcpy(host.data(), data, bytes(), cudaMemcpyDeviceToHost),
            "output download");
    return host;
  }
  bool guards_clean() const {
    std::vector<unsigned char> prefix(kGuardBytes), suffix(kGuardBytes);
    cuda_ok(cudaMemcpy(prefix.data(), allocation, kGuardBytes,
                       cudaMemcpyDeviceToHost),
            "prefix guard download");
    cuda_ok(cudaMemcpy(suffix.data(), allocation + kGuardBytes + bytes(),
                       kGuardBytes, cudaMemcpyDeviceToHost),
            "suffix guard download");
    for (unsigned char value : prefix)
      if (value != 0xa7) return false;
    for (unsigned char value : suffix)
      if (value != 0x7a) return false;
    return true;
  }
};

static uint16_t bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + bias) >> 16);
}

// BEGIN q_a W8A8 pipeline deterministic inputs
enum Pattern { kZeros, kExtremes, kVaried };
struct CaseSpec {
  unsigned m;
  Pattern pattern;
  unsigned salt;
  float eps;
};
constexpr CaseSpec kCases[kParityCases] = {
    {1, kZeros, 11, 1.0e-6f},
    {7, kExtremes, 29, 5.0e-7f},
    {kProductionM, kVaried, 47, 3.0e-5f},
};

static void make_activations(const CaseSpec& spec,
                             std::vector<__nv_bfloat16>& input,
                             std::vector<__nv_bfloat16>& norm_weight) {
  auto* input_bits = reinterpret_cast<uint16_t*>(input.data());
  auto* norm_bits = reinterpret_cast<uint16_t*>(norm_weight.data());
  constexpr uint16_t extremes[] = {
      0x7f7f,  // max-finite
      0xff7f, 0x0080, 0x8080, 0x0001, 0x8001,
      0x0000,  // signed-zero pair
      0x8000,
  };
  for (size_t i = 0; i < input.size(); ++i) {
    if (spec.pattern == kZeros) {
      input_bits[i] = (i & 1u) ? 0x8000u : 0x0000u;  // signed-zero
    } else if (spec.pattern == kExtremes) {
      input_bits[i] = extremes[(i + spec.salt) % 8];
    } else {
      const int value = static_cast<int>((i * 131u + spec.salt) % 509u) - 254;
      input_bits[i] = bf16(value * (1.0f / 128.0f));
      if ((i + spec.salt) % 4093u == 0)
        input_bits[i] = (i & 1u) ? 0x8000u : 0x0000u;  // signed-zero
    }
  }
  for (unsigned i = 0; i < kK; ++i) {
    const int value = static_cast<int>((i * 37u + spec.salt) % 257u) - 128;
    norm_bits[i] = bf16(1.0f + value * (1.0f / 512.0f));
    if ((i + spec.salt) % 251u == 0) norm_bits[i] = 0;
  }
}

static void make_w8a8_inputs(std::vector<unsigned char>& weight_fp8,
                             std::vector<float>& block_scale) {
  // Raw, finite E4M3 bytes with both signs; deliberately avoid the NaN codes.
  constexpr unsigned char E4M3[] = {
      0x00, 0x80, 0x20, 0xa0, 0x30, 0xb0, 0x38, 0xb8,
      0x40, 0xc0, 0x48, 0xc8, 0x50, 0xd0, 0x58, 0xd8,
  };
  for (size_t i = 0; i < weight_fp8.size(); ++i) {
    weight_fp8[i] = E4M3[(i * 17u + i / kK * 13u + 5u) % 16u];
  }
  for (size_t i = 0; i < block_scale.size(); ++i) {
    block_scale[i] = 0.0009765625f * static_cast<float>(1u + (i * 29u) % 31u);
  }
}
// weight_hash and scale_hash bind the deterministic E4M3/block_scale bytes.
// END q_a W8A8 pipeline deterministic inputs

struct Outputs {
  Guarded<unsigned char> a8;
  Guarded<float> row_scale;
  Guarded<__nv_bfloat16> gemm;
  Outputs(unsigned m)
      : a8(static_cast<size_t>(m) * kK), row_scale(m),
        gemm(static_cast<size_t>(m) * kN) {}
  void fill(unsigned char poison) {
    a8.fill(poison);
    row_scale.fill(poison);
    gemm.fill(poison);
  }
  bool guards_clean() const {
    return a8.guards_clean() && row_scale.guards_clean() &&
           gemm.guards_clean();
  }
};

static size_t mismatch_bytes(const std::vector<unsigned char>& lhs,
                             const std::vector<unsigned char>& rhs) {
  if (lhs.size() != rhs.size()) return lhs.size() + rhs.size();
  size_t mismatches = 0;
  for (size_t i = 0; i < lhs.size(); ++i) mismatches += lhs[i] != rhs[i];
  return mismatches;
}

// BEGIN q_a W8A8 pipeline exact parity
constexpr unsigned char kPoisonA = 0x5a;
constexpr unsigned char kPoisonB = 0xc3;
static void run_parity_case(const CaseSpec& spec,
                            const std::vector<unsigned char>& host_weight_fp8,
                            const std::vector<float>& host_block_scale,
                            uint64_t& input_hash, uint64_t& norm_weight_hash,
                            size_t& a8_mismatches,
                            size_t& row_scale_mismatches,
                            size_t& gemm_mismatches) {
  std::vector<__nv_bfloat16> host_input(static_cast<size_t>(spec.m) * kK);
  std::vector<__nv_bfloat16> host_norm_weight(kK);
  make_activations(spec, host_input, host_norm_weight);
  input_hash = fnv(input_hash, host_input.data(),
                   host_input.size() * sizeof(host_input[0]));
  norm_weight_hash = fnv(norm_weight_hash, host_norm_weight.data(),
                         host_norm_weight.size() * sizeof(host_norm_weight[0]));

  Guarded<__nv_bfloat16> input(host_input.size());
  Guarded<__nv_bfloat16> norm_weight(kK);
  Guarded<unsigned char> weight_fp8(host_weight_fp8.size());
  Guarded<float> block_scale(host_block_scale.size());
  Guarded<__nv_bfloat16> normed(host_input.size());
  Outputs incumbent(spec.m), candidate(spec.m);
  input.upload(host_input);
  norm_weight.upload(host_norm_weight);
  weight_fp8.upload(host_weight_fp8);
  block_scale.upload(host_block_scale);

  for (int poison = 0; poison < kPoisonCases; ++poison) {
    const unsigned char poison_a = poison == 0 ? kPoisonA : kPoisonB;
    const unsigned char poison_b = poison == 0 ? kPoisonB : kPoisonA;
    normed.fill(poison_a);
    incumbent.fill(poison_a);
    candidate.fill(poison_b);
    launch_baseline(input.data, norm_weight.data, normed.data, incumbent.a8.data,
                    incumbent.row_scale.data, weight_fp8.data, block_scale.data,
                    incumbent.gemm.data, spec.m, spec.eps);
    launch_candidate(input.data, norm_weight.data, candidate.a8.data,
                     candidate.row_scale.data, weight_fp8.data,
                     block_scale.data, candidate.gemm.data, spec.m, spec.eps);
    cuda_ok(cudaGetLastError(), "pipeline parity launch");
    cuda_ok(cudaDeviceSynchronize(), "pipeline parity sync");

    const auto incumbent_a8 = incumbent.a8.download();
    const auto candidate_a8 = candidate.a8.download();
    const auto incumbent_scale = incumbent.row_scale.download();
    const auto candidate_scale = candidate.row_scale.download();
    const auto incumbent_gemm = incumbent.gemm.download();
    const auto candidate_gemm = candidate.gemm.download();
    a8_mismatches += mismatch_bytes(incumbent_a8, candidate_a8);
    row_scale_mismatches += mismatch_bytes(incumbent_scale, candidate_scale);
    gemm_mismatches += mismatch_bytes(incumbent_gemm, candidate_gemm);
    const bool guards_clean =
        input.guards_clean() && norm_weight.guards_clean() &&
        weight_fp8.guards_clean() && block_scale.guards_clean() &&
        normed.guards_clean() && incumbent.guards_clean() &&
        candidate.guards_clean();
    if (std::memcmp(incumbent_a8.data(), candidate_a8.data(),
                    incumbent_a8.size()) != 0 ||
        std::memcmp(incumbent_scale.data(), candidate_scale.data(),
                    incumbent_scale.size()) != 0 ||
        std::memcmp(incumbent_gemm.data(), candidate_gemm.data(),
                    incumbent_gemm.size()) != 0 ||
        !guards_clean) {
      die("pipeline exact-byte parity or poisoned guard failure");
    }
  }
}
// Full memcmp covers exact A8 bytes, row_scale bits, and every BF16 GEMM byte;
// opposite poison_a/poison_b values make any mutually unwritten byte disagree.
// END q_a W8A8 pipeline exact parity

// BEGIN q_a W8A8 pipeline ABBA timing
// The promotion protocol fixes kWarmupRounds = 2 and kTimingSamples = 4.
static float timed(bool baseline, Guarded<__nv_bfloat16>& input,
                   Guarded<__nv_bfloat16>& norm_weight,
                   Guarded<unsigned char>& weight_fp8,
                   Guarded<float>& block_scale,
                   Guarded<__nv_bfloat16>& normed, Outputs& outputs,
                   const std::vector<__nv_bfloat16>& host_input,
                   const std::vector<__nv_bfloat16>& host_norm_weight,
                   const std::vector<unsigned char>& host_weight_fp8,
                   const std::vector<float>& host_block_scale, float eps) {
  input.upload(host_input);
  norm_weight.upload(host_norm_weight);
  weight_fp8.upload(host_weight_fp8);
  block_scale.upload(host_block_scale);
  normed.fill(0x91);
  outputs.fill(0x4d);
  cudaEvent_t begin, end;
  cuda_ok(cudaEventCreate(&begin), "event create");
  cuda_ok(cudaEventCreate(&end), "event create");
  cuda_ok(cudaEventRecord(begin), "cudaEventRecord(begin)");
  if (baseline) {
    launch_baseline(input.data, norm_weight.data, normed.data, outputs.a8.data,
                    outputs.row_scale.data, weight_fp8.data, block_scale.data,
                    outputs.gemm.data, kProductionM, eps);
  } else {
    launch_candidate(input.data, norm_weight.data, outputs.a8.data,
                     outputs.row_scale.data, weight_fp8.data, block_scale.data,
                     outputs.gemm.data, kProductionM, eps);
  }
  cuda_ok(cudaEventRecord(end), "cudaEventRecord(end)");
  cuda_ok(cudaEventSynchronize(end), "event sync");
  float milliseconds = 0.0f;
  cuda_ok(cudaEventElapsedTime(&milliseconds, begin, end), "elapsed time");
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return milliseconds;
}
// Exact production-shape order per sample: baseline, candidate, candidate, baseline.
// baseline_ms/candidate_ms and speedup average identical kTimingSamples arms.
// END q_a W8A8 pipeline ABBA timing

static double threshold(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <min_end_to_end_speedup>\n", argv[0]);
    std::exit(2);
  }
  const char* text = argv[1];
  bool digit = false, dot = false, exponent = false, exponent_digit = false;
  for (size_t i = 0; text[i] != '\0'; ++i) {
    const unsigned char ch = static_cast<unsigned char>(text[i]);
    if (ch >= '0' && ch <= '9') {
      digit = true;
      if (exponent) exponent_digit = true;
    } else if (ch == '.' && !dot && !exponent) {
      dot = true;
    } else if ((ch == 'e' || ch == 'E') && digit && !exponent) {
      exponent = true;
    } else if ((ch == '+' || ch == '-') && exponent && !exponent_digit &&
               i > 0 && (text[i - 1] == 'e' || text[i - 1] == 'E')) {
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
  const double value = std::strtod(text, &end);
  if (errno != 0 || end == text || *end != '\0' || !std::isfinite(value) ||
      value <= 1.0 || value > 100.0) {
    std::fprintf(stderr, "invalid explicit numeric threshold\n");
    std::exit(2);
  }
  return value;
}

static int run(int argc, char** argv) {
  const double minimum = threshold(argc, argv);  // Must precede every CUDA call.

  int driver = 0, runtime = 0, device = 0;
  cudaDeviceProp properties{};
  cuda_ok(cudaDriverGetVersion(&driver), "driver");
  cuda_ok(cudaRuntimeGetVersion(&runtime), "runtime");
  cuda_ok(cudaGetDevice(&device), "device");
  cuda_ok(cudaGetDeviceProperties(&properties, device), "properties");

  std::vector<unsigned char> host_weight_fp8(static_cast<size_t>(kN) * kK);
  std::vector<float> host_block_scale(
      static_cast<size_t>(kN / 128) * (kK / 128));
  make_w8a8_inputs(host_weight_fp8, host_block_scale);
  uint64_t input_hash = 1469598103934665603ull;
  uint64_t norm_weight_hash = 1469598103934665603ull;
  const uint64_t weight_hash =
      fnv(1469598103934665603ull, host_weight_fp8.data(),
          host_weight_fp8.size());
  const uint64_t scale_hash =
      fnv(1469598103934665603ull, host_block_scale.data(),
          host_block_scale.size() * sizeof(host_block_scale[0]));
  size_t a8_mismatches = 0, row_scale_mismatches = 0, gemm_mismatches = 0;
  for (const CaseSpec& spec : kCases) {
    run_parity_case(spec, host_weight_fp8, host_block_scale, input_hash,
                    norm_weight_hash, a8_mismatches, row_scale_mismatches,
                    gemm_mismatches);
  }
  if (a8_mismatches != 0 || row_scale_mismatches != 0 ||
      gemm_mismatches != 0) {
    die("pipeline exact-byte parity failed");
  }

  const CaseSpec& production = kCases[kParityCases - 1];
  std::vector<__nv_bfloat16> host_input(
      static_cast<size_t>(kProductionM) * kK);
  std::vector<__nv_bfloat16> host_norm_weight(kK);
  make_activations(production, host_input, host_norm_weight);
  Guarded<__nv_bfloat16> input(host_input.size());
  Guarded<__nv_bfloat16> norm_weight(kK);
  Guarded<unsigned char> weight_fp8(host_weight_fp8.size());
  Guarded<float> block_scale(host_block_scale.size());
  Guarded<__nv_bfloat16> normed(host_input.size());
  Outputs outputs(kProductionM);
  for (int warmup = 0; warmup < kWarmupRounds; ++warmup) {
    timed(true, input, norm_weight, weight_fp8, block_scale, normed, outputs,
          host_input, host_norm_weight, host_weight_fp8, host_block_scale,
          production.eps);
    timed(false, input, norm_weight, weight_fp8, block_scale, normed, outputs,
          host_input, host_norm_weight, host_weight_fp8, host_block_scale,
          production.eps);
  }
  double baseline_ms = 0.0, candidate_ms = 0.0;
  for (int sample = 0; sample < kTimingSamples; ++sample) {
    baseline_ms += timed(true, input, norm_weight, weight_fp8, block_scale,
                         normed, outputs, host_input, host_norm_weight,
                         host_weight_fp8, host_block_scale, production.eps);
    candidate_ms += timed(false, input, norm_weight, weight_fp8, block_scale,
                          normed, outputs, host_input, host_norm_weight,
                          host_weight_fp8, host_block_scale, production.eps);
    candidate_ms += timed(false, input, norm_weight, weight_fp8, block_scale,
                          normed, outputs, host_input, host_norm_weight,
                          host_weight_fp8, host_block_scale, production.eps);
    baseline_ms += timed(true, input, norm_weight, weight_fp8, block_scale,
                         normed, outputs, host_input, host_norm_weight,
                         host_weight_fp8, host_block_scale, production.eps);
  }
  baseline_ms /= 2 * kTimingSamples;
  candidate_ms /= 2 * kTimingSamples;
  const double speedup = baseline_ms / candidate_ms;
  if (!std::isfinite(speedup) || baseline_ms <= 0.0 || candidate_ms <= 0.0 ||
      speedup < minimum) {
    die("pre-registered pipeline speedup threshold not met");
  }

  char uuid[33];
  for (int i = 0; i < 16; ++i)
    std::sprintf(uuid + 2 * i, "%02x",
                 static_cast<unsigned char>(properties.uuid.bytes[i]));
  uuid[32] = '\0';
  std::printf("build_id=%s\n", V4_QA_PIPELINE_PROBE_BUILD_ID);
  std::printf("device_uuid=%s driver=%d runtime=%d\n", uuid, driver, runtime);
  std::printf(
      "input_hash=%016llx norm_weight_hash=%016llx weight_hash=%016llx scale_hash=%016llx\n",
      static_cast<unsigned long long>(input_hash),
      static_cast<unsigned long long>(norm_weight_hash),
      static_cast<unsigned long long>(weight_hash),
      static_cast<unsigned long long>(scale_hash));
  std::printf(
      "shape m=2410 n=32768 k=1024 parity_shapes=1,7,2410 parity_cases=3 poison_cases=2 output_bytes=157941760\n");
  std::printf(
      "a8_mismatches=0 row_scale_mismatches=0 gemm_mismatches=0 guards=clean poison_a=clean poison_b=clean\n");
  std::printf("threshold min_speedup=%.17g\n", minimum);
  std::printf(
      "baseline_ms=%.6f candidate_ms=%.6f speedup=%.9f abba_samples=%d\n",
      baseline_ms, candidate_ms, speedup, kTimingSamples);
  std::printf("result=PASS\n");
  return 0;
}
}  // namespace pipeline_probe

int main(int argc, char** argv) { return pipeline_probe::run(argc, argv); }
