// SPDX-License-Identifier: AGPL-3.0-only

// Standalone GPU promotion probe for the isolated V4 inverse-RoPE + full-row
// W8A8 quantizer experiment. It remains outside registry and serving.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#ifndef V4_INV_QUANT_PROBE_BUILD_ID
#error "V4_INV_QUANT_PROBE_BUILD_ID must bind the immutable build receipt"
#endif

// BEGIN V4 inverse quant probe kernel contract
#include "v4_prefill_rope_fused.cu"
#include "../common/w8a8_gemm_pipelined.cu"
#include "v4_prefill_inverse_rope_w8a8_quant_fused.cu"

namespace probe {

constexpr unsigned kProductionTokens = 2410;
constexpr unsigned kNq = 64;
constexpr unsigned kHeadDim = 512;
constexpr unsigned kNopeDim = 448;
constexpr unsigned kRopeDim = 64;
constexpr unsigned kRopePairs = 32;
constexpr unsigned kRowWidth = 32768;
constexpr unsigned kQuantThreads = 256;
constexpr unsigned kParityCases = 4;
constexpr unsigned kPoisonCases = 2;
constexpr unsigned kMalformedCases = 22;
constexpr unsigned kTimingRounds = 8;

static_assert(kNq * kHeadDim == kRowWidth, "exact V4 attention row");
static_assert(kRopeDim / 2 == kRopePairs, "one inverse pair per warp lane");
static_assert(kProductionTokens == 2410, "exact production prefill case");

static void launch_baseline(__nv_bfloat16* input,
                            const unsigned int* positions,
                            const float* frequencies,
                            unsigned char* output_fp8,
                            float* row_scale,
                            unsigned tokens,
                            float mscale) {
    v4_prefill_rope_fused_inverse<<<dim3(tokens, kNq, 1), dim3(kRopePairs, 1, 1)>>>(
        input, positions, frequencies, tokens, kNq, 0, kHeadDim, kNopeDim,
        kRopeDim, mscale);
    quantize_a_fp8_rows<<<dim3(tokens, 1, 1), dim3(kQuantThreads, 1, 1)>>>(
        input, output_fp8, row_scale, tokens, kRowWidth);
}

static void launch_candidate(const __nv_bfloat16* input,
                             const unsigned int* positions,
                             const float* frequencies,
                             unsigned char* output_fp8,
                             float* row_scale,
                             unsigned tokens,
                             float mscale) {
    v4_prefill_inverse_rope_w8a8_quant_fused<<<dim3(tokens, 1, 1), dim3(kQuantThreads, 1, 1)>>>(
        input, positions, frequencies, output_fp8, row_scale, tokens, kNq,
        kHeadDim, kNopeDim, kRopeDim, kRowWidth, kQuantThreads, mscale);
}
// END V4 inverse quant probe kernel contract

// BEGIN V4 inverse quant applicability
// Any future host integration must decide w8a8_inplace eligibility before launch
// and require diag_this == false. Diagnostics currently consume materialized inverse-rotated BF16 attn_out
// before wo_a, so those paths must bypass the candidate.
// END V4 inverse quant applicability

[[noreturn]] static void fail(const char* message) {
    std::fprintf(stderr, "FAIL %s\n", message);
    std::exit(1);
}

static void cuda_ok(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(1);
    }
}

static uint64_t fnv1a(uint64_t hash, const void* data, size_t bytes) {
    const auto* raw = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; ++i) {
        hash = (hash ^ raw[i]) * 1099511628211ULL;
    }
    return hash;
}

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(size_t count) : bytes_(count * sizeof(T)) {
        cuda_ok(cudaMalloc(&pointer_, bytes_), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return static_cast<T*>(pointer_); }
    size_t bytes() const { return bytes_; }
    void upload(const void* source) const {
        cuda_ok(cudaMemcpy(pointer_, source, bytes_, cudaMemcpyHostToDevice), "upload");
    }

  private:
    void* pointer_ = nullptr;
    size_t bytes_ = 0;
};

template <typename T>
class Guarded {
  public:
    static constexpr size_t kGuardBytes = 256;
    Guarded(size_t count, unsigned char prefix, unsigned char suffix)
        : bytes_(count * sizeof(T)), prefix_(prefix), suffix_(suffix) {
        cuda_ok(cudaMalloc(&allocation_, bytes_ + 2 * kGuardBytes), "guarded cudaMalloc");
        data_ = reinterpret_cast<T*>(allocation_ + kGuardBytes);
        reset_guards();
    }
    ~Guarded() { cudaFree(allocation_); }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    T* data() const { return data_; }
    size_t bytes() const { return bytes_; }
    void upload(const void* source) const {
        cuda_ok(cudaMemcpy(data_, source, bytes_, cudaMemcpyHostToDevice), "guarded upload");
    }
    void fill(unsigned char value) const {
        cuda_ok(cudaMemset(data_, value, bytes_), "guarded fill");
    }
    void reset_guards() const {
        cuda_ok(cudaMemset(allocation_, prefix_, kGuardBytes), "prefix guard");
        cuda_ok(cudaMemset(allocation_ + kGuardBytes + bytes_, suffix_, kGuardBytes),
                "suffix guard");
    }
    std::vector<unsigned char> download() const {
        std::vector<unsigned char> host(bytes_);
        cuda_ok(cudaMemcpy(host.data(), data_, bytes_, cudaMemcpyDeviceToHost), "download");
        return host;
    }
    bool guards_clean() const {
        std::vector<unsigned char> prefix(kGuardBytes), suffix(kGuardBytes);
        cuda_ok(cudaMemcpy(prefix.data(), allocation_, kGuardBytes, cudaMemcpyDeviceToHost),
                "read prefix guard");
        cuda_ok(cudaMemcpy(suffix.data(), allocation_ + kGuardBytes + bytes_, kGuardBytes,
                           cudaMemcpyDeviceToHost),
                "read suffix guard");
        return std::all_of(prefix.begin(), prefix.end(), [&](unsigned char value) {
                   return value == prefix_;
               }) &&
               std::all_of(suffix.begin(), suffix.end(), [&](unsigned char value) {
                   return value == suffix_;
               });
    }

  private:
    unsigned char* allocation_ = nullptr;
    T* data_ = nullptr;
    size_t bytes_ = 0;
    unsigned char prefix_ = 0;
    unsigned char suffix_ = 0;
};

struct Outputs {
    Guarded<unsigned char> fp8;
    Guarded<float> scale;
    Outputs(unsigned tokens, unsigned char prefix, unsigned char suffix)
        : fp8(static_cast<size_t>(tokens) * kRowWidth, prefix, suffix),
          scale(tokens, suffix, prefix) {}
    void fill(unsigned char poison) const {
        fp8.fill(poison);
        scale.fill(poison);
    }
    bool guards_clean() const { return fp8.guards_clean() && scale.guards_clean(); }
};

static size_t mismatch_bytes(const std::vector<unsigned char>& expected,
                             const std::vector<unsigned char>& actual) {
    if (expected.size() != actual.size()) fail("mismatch extent");
    size_t mismatches = 0;
    for (size_t i = 0; i < expected.size(); ++i) mismatches += expected[i] != actual[i];
    return mismatches;
}

static uint16_t bf16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t bias = 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + bias) >> 16);
}

// BEGIN V4 inverse quant deterministic cases
struct CaseSpec {
    unsigned tokens;
    unsigned salt;
    float theta;
    float mscale;
};

constexpr CaseSpec kCases[kParityCases] = {
    {1, 3, 10000.0f, 1.0f},
    {7, 11, 160000.0f, 0.70710677f},
    {128, 29, 87500.0f, 1.125f},
    {2410, 47, 160000.0f, 0.875f},
};

static void make_case(const CaseSpec& spec,
                      std::vector<__nv_bfloat16>& input,
                      std::vector<unsigned int>& positions,
                      std::vector<float>& frequencies) {
    auto* bits = reinterpret_cast<uint16_t*>(input.data());
    for (size_t i = 0; i < input.size(); ++i) {
        const int raw = static_cast<int>((i * 131ULL + spec.salt * 977ULL) % 8191ULL) - 4095;
        bits[i] = bf16_bits(static_cast<float>(raw) / 257.0f);
        if ((i + spec.salt) % 65521 == 0) bits[i] = (i & 1) ? 0x8000u : 0x0000u;
    }
    // Each case has distinct positions and distinct frequencies.
    for (size_t token = 0; token < positions.size(); ++token) {
        positions[token] = spec.salt * 17u + static_cast<unsigned>(token) * (2u * spec.salt + 1u);
    }
    for (unsigned pair = 0; pair < kRopePairs; ++pair) {
        frequencies[pair] = std::pow(
            spec.theta, -2.0f * static_cast<float>(pair) / static_cast<float>(kRopeDim));
    }
}
// END V4 inverse quant deterministic cases

struct Evidence {
    uint64_t input_hash = 1469598103934665603ULL;
    uint64_t position_hash = 1469598103934665603ULL;
    uint64_t frequency_hash = 1469598103934665603ULL;
    size_t fp8_mismatches = 0;
    size_t scale_mismatches = 0;
    bool candidate_input_unchanged = true;
};

// BEGIN V4 inverse quant exact byte parity
constexpr unsigned char kPoisonA = 0x5A;
constexpr unsigned char kPoisonB = 0xC3;
// full FP8 and FP32 scale-bit memcmp under poison_a/poison_b proves full writes.
static void run_parity_case(const CaseSpec& spec, Evidence& evidence) {
    const size_t elements = static_cast<size_t>(spec.tokens) * kRowWidth;
    std::vector<__nv_bfloat16> host_input(elements);
    std::vector<unsigned int> host_positions(spec.tokens);
    std::vector<float> host_frequencies(kRopePairs);
    make_case(spec, host_input, host_positions, host_frequencies);
    evidence.input_hash = fnv1a(evidence.input_hash, host_input.data(), elements * 2);
    evidence.position_hash = fnv1a(evidence.position_hash, host_positions.data(),
                                   host_positions.size() * sizeof(unsigned));
    evidence.frequency_hash = fnv1a(evidence.frequency_hash, host_frequencies.data(),
                                    host_frequencies.size() * sizeof(float));

    Guarded<__nv_bfloat16> baseline_input(elements, 0xA5, 0x5A);
    Guarded<__nv_bfloat16> candidate_input(elements, 0x3C, 0xC3);
    DeviceBuffer<unsigned int> positions(spec.tokens);
    DeviceBuffer<float> frequencies(kRopePairs);
    Outputs baseline(spec.tokens, 0x96, 0x69);
    Outputs candidate(spec.tokens, 0x2D, 0xD2);
    positions.upload(host_positions.data());
    frequencies.upload(host_frequencies.data());

    for (unsigned poison = 0; poison < kPoisonCases; ++poison) {
        baseline_input.upload(host_input.data());
        candidate_input.upload(host_input.data());
        baseline.fill(poison == 0 ? kPoisonA : kPoisonB);
        candidate.fill(poison == 0 ? kPoisonB : kPoisonA);
        const auto candidate_input_before = candidate_input.download();
        launch_baseline(baseline_input.data(), positions.get(), frequencies.get(),
                        baseline.fp8.data(), baseline.scale.data(), spec.tokens, spec.mscale);
        launch_candidate(candidate_input.data(), positions.get(), frequencies.get(),
                         candidate.fp8.data(), candidate.scale.data(), spec.tokens, spec.mscale);
        cuda_ok(cudaGetLastError(), "parity launch");
        cuda_ok(cudaDeviceSynchronize(), "parity synchronize");

        const auto expected_fp8 = baseline.fp8.download();
        const auto actual_fp8 = candidate.fp8.download();
        const auto expected_scale = baseline.scale.download();
        const auto actual_scale = candidate.scale.download();
        const auto candidate_input_after = candidate_input.download();
        evidence.fp8_mismatches += mismatch_bytes(expected_fp8, actual_fp8);
        evidence.scale_mismatches += mismatch_bytes(expected_scale, actual_scale);
        evidence.candidate_input_unchanged &= candidate_input_before == candidate_input_after;
        if (std::memcmp(expected_fp8.data(), actual_fp8.data(), expected_fp8.size()) != 0 ||
            std::memcmp(expected_scale.data(), actual_scale.data(), expected_scale.size()) != 0 ||
            !(candidate_input_before == candidate_input_after) ||
            !baseline_input.guards_clean() || !candidate_input.guards_clean() ||
            !baseline.guards_clean() || !candidate.guards_clean()) {
            fail("exact parity, candidate_input_unchanged, or guards_clean");
        }
    }
}
// END V4 inverse quant exact byte parity

// BEGIN V4 inverse quant malformed ABI
enum class NullSlot { kNone, kInput, kPositions, kFrequency, kOutput, kScale };

static void malformed(const char* label,
                      Guarded<__nv_bfloat16>& input,
                      DeviceBuffer<unsigned int>& positions,
                      DeviceBuffer<float>& frequencies,
                      Outputs& output,
                      NullSlot null_slot,
                      bool alias_output,
                      unsigned tokens,
                      unsigned nq,
                      unsigned head_dim,
                      unsigned nope_dim,
                      unsigned rotary_dim,
                      unsigned row_width,
                      unsigned quant_block,
                      float mscale,
                      dim3 grid,
                      dim3 block) {
    input.fill(0x51);
    output.fill(0xA6);
    const auto input_before = input.download();
    const auto fp8_before = output.fp8.download();
    const auto scale_before = output.scale.download();
    const auto* input_ptr = null_slot == NullSlot::kInput ? nullptr : input.data();
    const auto* position_ptr = null_slot == NullSlot::kPositions ? nullptr : positions.get();
    const auto* frequency_ptr = null_slot == NullSlot::kFrequency ? nullptr : frequencies.get();
    auto* output_ptr = null_slot == NullSlot::kOutput ? nullptr : output.fp8.data();
    if (alias_output) output_ptr = reinterpret_cast<unsigned char*>(input.data());
    auto* scale_ptr = null_slot == NullSlot::kScale ? nullptr : output.scale.data();
    v4_prefill_inverse_rope_w8a8_quant_fused<<<grid, block>>>(
        input_ptr, position_ptr, frequency_ptr, output_ptr, scale_ptr, tokens, nq,
        head_dim, nope_dim, rotary_dim, row_width, quant_block, mscale);
    cuda_ok(cudaGetLastError(), label);
    cuda_ok(cudaDeviceSynchronize(), label);
    const auto input_after = input.download();
    const auto fp8_after = output.fp8.download();
    const auto scale_after = output.scale.download();
    if (!(input_before == input_after) || !(fp8_before == fp8_after) ||
        !(scale_before == scale_after) || !input.guards_clean() || !output.guards_clean()) {
        fail("malformed ABI wrote output");
    }
}

static unsigned run_malformed_cases() {
    // Contract: kMalformedCases = 22.
    constexpr unsigned tokens = 2;
    const size_t elements = static_cast<size_t>(tokens) * kRowWidth;
    Guarded<__nv_bfloat16> input(elements, 0x47, 0x74);
    DeviceBuffer<unsigned int> positions(tokens);
    DeviceBuffer<float> frequencies(kRopePairs);
    Outputs output(tokens, 0x18, 0x81);
    std::vector<unsigned int> host_positions(tokens, 7);
    std::vector<float> host_frequencies(kRopePairs, 0.125f);
    positions.upload(host_positions.data());
    frequencies.upload(host_frequencies.data());
    const dim3 valid_grid(tokens, 1, 1);
    const dim3 valid_block(kQuantThreads, 1, 1);
    unsigned no_write = 0;
    auto run = [&](const char* label, NullSlot null_slot, bool alias_output,
                   unsigned token_arg, unsigned nq, unsigned head_dim,
                   unsigned nope_dim, unsigned rotary_dim, unsigned row_width,
                   unsigned quant_block, float mscale, dim3 grid, dim3 block) {
        malformed(label, input, positions, frequencies, output, null_slot, alias_output,
                  token_arg, nq, head_dim, nope_dim, rotary_dim, row_width,
                  quant_block, mscale, grid, block);
        ++no_write;
    };
    const auto valid = [&](const char* label, NullSlot slot = NullSlot::kNone,
                           bool alias = false) {
        run(label, slot, alias, tokens, kNq, kHeadDim, kNopeDim, kRopeDim,
            kRowWidth, kQuantThreads, 1.0f, valid_grid, valid_block);
    };
    valid("null-input", NullSlot::kInput);
    valid("null-positions", NullSlot::kPositions);
    valid("null-inv-freq", NullSlot::kFrequency);
    valid("null-output", NullSlot::kOutput);
    valid("null-scale", NullSlot::kScale);
    valid("input-output-alias", NullSlot::kNone, true);
    run("zero-tokens", NullSlot::kNone, false, 0, kNq, kHeadDim, kNopeDim,
        kRopeDim, kRowWidth, kQuantThreads, 1.0f, valid_grid, valid_block);
    run("wrong-q-heads", NullSlot::kNone, false, tokens, kNq - 1, kHeadDim,
        kNopeDim, kRopeDim, kRowWidth, kQuantThreads, 1.0f, valid_grid, valid_block);
    run("wrong-head-dim", NullSlot::kNone, false, tokens, kNq, kHeadDim - 1,
        kNopeDim, kRopeDim, kRowWidth, kQuantThreads, 1.0f, valid_grid, valid_block);
    run("wrong-nope-dim", NullSlot::kNone, false, tokens, kNq, kHeadDim,
        kNopeDim - 1, kRopeDim, kRowWidth, kQuantThreads, 1.0f, valid_grid, valid_block);
    run("wrong-rotary-dim", NullSlot::kNone, false, tokens, kNq, kHeadDim,
        kNopeDim, kRopeDim - 2, kRowWidth, kQuantThreads, 1.0f, valid_grid, valid_block);
    run("wrong-row-width", NullSlot::kNone, false, tokens, kNq, kHeadDim,
        kNopeDim, kRopeDim, kRowWidth - 1, kQuantThreads, 1.0f, valid_grid, valid_block);
    run("wrong-quant-block", NullSlot::kNone, false, tokens, kNq, kHeadDim,
        kNopeDim, kRopeDim, kRowWidth, kQuantThreads / 2, 1.0f, valid_grid, valid_block);
    run("mscale-nan", NullSlot::kNone, false, tokens, kNq, kHeadDim, kNopeDim,
        kRopeDim, kRowWidth, kQuantThreads, std::numeric_limits<float>::quiet_NaN(),
        valid_grid, valid_block);
    run("mscale-inf", NullSlot::kNone, false, tokens, kNq, kHeadDim, kNopeDim,
        kRopeDim, kRowWidth, kQuantThreads, std::numeric_limits<float>::infinity(),
        valid_grid, valid_block);
    run("block-x", NullSlot::kNone, false, tokens, kNq, kHeadDim, kNopeDim,
        kRopeDim, kRowWidth, kQuantThreads, 1.0f, valid_grid, dim3(128, 1, 1));
    run("block-y", NullSlot::kNone, false, tokens, kNq, kHeadDim, kNopeDim,
        kRopeDim, kRowWidth, kQuantThreads, 1.0f, valid_grid, dim3(256, 2, 1));
    run("block-z", NullSlot::kNone, false, tokens, kNq, kHeadDim, kNopeDim,
        kRopeDim, kRowWidth, kQuantThreads, 1.0f, valid_grid, dim3(256, 1, 2));
    run("grid-x-small", NullSlot::kNone, false, tokens, kNq, kHeadDim, kNopeDim,
        kRopeDim, kRowWidth, kQuantThreads, 1.0f, dim3(tokens - 1, 1, 1), valid_block);
    run("grid-x-large", NullSlot::kNone, false, tokens, kNq, kHeadDim, kNopeDim,
        kRopeDim, kRowWidth, kQuantThreads, 1.0f, dim3(tokens + 1, 1, 1), valid_block);
    run("grid-y", NullSlot::kNone, false, tokens, kNq, kHeadDim, kNopeDim,
        kRopeDim, kRowWidth, kQuantThreads, 1.0f, dim3(tokens, 2, 1), valid_block);
    run("grid-z", NullSlot::kNone, false, tokens, kNq, kHeadDim, kNopeDim,
        kRopeDim, kRowWidth, kQuantThreads, 1.0f, dim3(tokens, 1, 2), valid_block);
    if (no_write != kMalformedCases) fail("malformed ABI census");
    return no_write;
}
// END V4 inverse quant malformed ABI

struct Timing {
    double baseline_ms;
    double candidate_ms;
    double speedup;
};

// BEGIN V4 inverse quant ABBA timing
template <typename Reset, typename Launch>
static float event_time(cudaEvent_t begin, cudaEvent_t end, Reset reset, Launch launch) {
    reset();
    cuda_ok(cudaEventRecord(begin), "cudaEventRecord(begin)");
    launch();
    cuda_ok(cudaEventRecord(end), "cudaEventRecord(end)");
    cuda_ok(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
    float milliseconds = 0.0f;
    cuda_ok(cudaEventElapsedTime(&milliseconds, begin, end), "cudaEventElapsedTime");
    return milliseconds;
}

static Timing run_timing() {
    // Contract: kTimingRounds = 8.
    const CaseSpec& spec = kCases[kParityCases - 1];
    const size_t elements = static_cast<size_t>(spec.tokens) * kRowWidth;
    std::vector<__nv_bfloat16> host_input(elements);
    std::vector<unsigned int> host_positions(spec.tokens);
    std::vector<float> host_frequencies(kRopePairs);
    make_case(spec, host_input, host_positions, host_frequencies);
    DeviceBuffer<__nv_bfloat16> seed(elements);
    DeviceBuffer<__nv_bfloat16> work(elements);
    DeviceBuffer<unsigned int> positions(spec.tokens);
    DeviceBuffer<float> frequencies(kRopePairs);
    DeviceBuffer<unsigned char> output_fp8(elements);
    DeviceBuffer<float> row_scale(spec.tokens);
    seed.upload(host_input.data());
    positions.upload(host_positions.data());
    frequencies.upload(host_frequencies.data());
    auto reset = [&] {
        cuda_ok(cudaMemcpyAsync(work.get(), seed.get(), work.bytes(), cudaMemcpyDeviceToDevice),
                "timing D2D reset");
        cuda_ok(cudaMemsetAsync(output_fp8.get(), 0xA7, output_fp8.bytes()),
                "timing FP8 memset");
        cuda_ok(cudaMemsetAsync(row_scale.get(), 0x7A, row_scale.bytes()),
                "timing scale memset");
    };
    auto baseline = [&] {
        launch_baseline(work.get(), positions.get(), frequencies.get(), output_fp8.get(),
                        row_scale.get(), spec.tokens, spec.mscale);
    };
    auto candidate = [&] {
        launch_candidate(work.get(), positions.get(), frequencies.get(), output_fp8.get(),
                         row_scale.get(), spec.tokens, spec.mscale);
    };
    // Warmup both paths with the same out-of-event resets.
    for (int warmup = 0; warmup < 2; ++warmup) {
        reset();
        baseline();
        reset();
        candidate();
    }
    cuda_ok(cudaDeviceSynchronize(), "timing warmup");
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    cuda_ok(cudaEventCreate(&begin), "event create begin");
    cuda_ok(cudaEventCreate(&end), "event create end");
    double baseline_ms = 0.0;
    double candidate_ms = 0.0;
    // Every round is baseline, candidate, candidate, baseline. All D2D/memset
    // reset work happens before cudaEventRecord(begin), outside the measurement.
    for (unsigned round = 0; round < kTimingRounds; ++round) {
        baseline_ms += event_time(begin, end, reset, baseline);
        candidate_ms += event_time(begin, end, reset, candidate);
        candidate_ms += event_time(begin, end, reset, candidate);
        baseline_ms += event_time(begin, end, reset, baseline);
    }
    cuda_ok(cudaEventDestroy(begin), "event destroy begin");
    cuda_ok(cudaEventDestroy(end), "event destroy end");
    baseline_ms /= 2.0 * kTimingRounds;
    candidate_ms /= 2.0 * kTimingRounds;
    if (!(baseline_ms > 0.0) || !(candidate_ms > 0.0)) fail("invalid timing");
    return {baseline_ms, candidate_ms, baseline_ms / candidate_ms};
}
// END V4 inverse quant ABBA timing

static double parse_threshold(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <min_speedup>\n", argv[0]);
        std::exit(2);
    }
    const char* input = argv[1];
    bool digit = false;
    bool dot = false;
    bool exponent = false;
    bool exponent_digit = false;
    for (size_t i = 0; input[i] != '\0'; ++i) {
        const unsigned char character = static_cast<unsigned char>(input[i]);
        if (std::isdigit(character)) {
            digit = true;
            if (exponent) exponent_digit = true;
        } else if (character == '.' && !dot && !exponent) {
            dot = true;
        } else if ((character == 'e' || character == 'E') && digit && !exponent) {
            exponent = true;
        } else if ((character == '+' || character == '-') && exponent &&
                   !exponent_digit && i > 0 &&
                   (input[i - 1] == 'e' || input[i - 1] == 'E')) {
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

static bool valid_build_id(const char* value) {
    if (std::strlen(value) != 64) return false;
    for (size_t i = 0; i < 64; ++i) {
        if (!((value[i] >= '0' && value[i] <= '9') ||
              (value[i] >= 'a' && value[i] <= 'f'))) {
            return false;
        }
    }
    return true;
}

static int run(int argc, char** argv) {
    const double minimum = parse_threshold(argc, argv);
    if (!valid_build_id(V4_INV_QUANT_PROBE_BUILD_ID)) fail("invalid build id");
    int driver = 0;
    int runtime = 0;
    int device = 0;
    cudaDeviceProp properties{};
    cuda_ok(cudaDriverGetVersion(&driver), "driver version");
    cuda_ok(cudaRuntimeGetVersion(&runtime), "runtime version");
    cuda_ok(cudaGetDevice(&device), "device");
    cuda_ok(cudaGetDeviceProperties(&properties, device), "device properties");

    Evidence evidence;
    for (const CaseSpec& spec : kCases) run_parity_case(spec, evidence);
    if (evidence.fp8_mismatches != 0 || evidence.scale_mismatches != 0 ||
        !evidence.candidate_input_unchanged) {
        fail("exact byte parity");
    }
    const unsigned no_write = run_malformed_cases();
    const Timing timing = run_timing();
    if (!std::isfinite(timing.speedup) || timing.speedup < minimum) {
        fail("pre-registered speedup threshold not met");
    }

    char uuid[33];
    for (int i = 0; i < 16; ++i) {
        std::sprintf(uuid + 2 * i, "%02x",
                     static_cast<unsigned char>(properties.uuid.bytes[i]));
    }
    uuid[32] = '\0';
    std::printf("build_id=%s\n", V4_INV_QUANT_PROBE_BUILD_ID);
    std::printf("device_uuid=%s driver=%d runtime=%d\n", uuid, driver, runtime);
    std::printf("input_hash=%016llx position_hash=%016llx frequency_hash=%016llx\n",
                static_cast<unsigned long long>(evidence.input_hash),
                static_cast<unsigned long long>(evidence.position_hash),
                static_cast<unsigned long long>(evidence.frequency_hash));
    std::printf("shapes=1,7,128,2410 row=32768 parity_cases=4 poison_cases=2 candidate_input_unchanged=yes\n");
    std::printf("threshold min_speedup=%.17g\n", minimum);
    std::printf("timing baseline_ms=%.6f candidate_ms=%.6f speedup=%.9f abba_rounds=8\n",
                timing.baseline_ms, timing.candidate_ms, timing.speedup);
    std::printf("fp8_mismatches=0 scale_mismatches=0 malformed_cases=22 no_write=%u guards=clean poison_a=clean poison_b=clean\n",
                no_write);
    std::printf("result=PASS\n");
    return 0;
}

}  // namespace probe

int main(int argc, char** argv) { return probe::run(argc, argv); }
