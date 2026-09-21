// SPDX-License-Identifier: AGPL-3.0-only

// Standalone GPU promotion probe. This translation unit deliberately includes
// the exact incumbent kernels and the isolated candidate; it has no registry or
// serving reachability.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../deepseek-v4-flash/nvfp4/mla_absorbed.cu"
#include "../common/rope.cu"
#include "v4_prefill_rope_fused.cu"

#ifndef V4_ROPE_PROBE_BUILD_ID
#error "V4_ROPE_PROBE_BUILD_ID is required"
#endif

namespace probe {

constexpr unsigned int kTokens = 2410;
constexpr unsigned int kNq = 64;
constexpr unsigned int kNkv = 1;
constexpr unsigned int kHeadDim = 512;
constexpr unsigned int kNopeDim = 448;
constexpr unsigned int kRopeDim = 64;
constexpr int kParityCases = 3;
constexpr int kPoisonCases = 38;
constexpr int kAbbaRounds = 6;
constexpr size_t kRedzoneBytes = 256;
constexpr int kRedzoneBuffers = 6;

constexpr size_t kQElements = static_cast<size_t>(kTokens) * kNq * kHeadDim;
constexpr size_t kKElements = static_cast<size_t>(kTokens) * kNkv * kHeadDim;
constexpr size_t kQBytes = kQElements * sizeof(__nv_bfloat16);
constexpr size_t kKBytes = kKElements * sizeof(__nv_bfloat16);
constexpr size_t kQTempElements = static_cast<size_t>(kTokens) * kNq * kRopeDim;
constexpr size_t kKTempElements = static_cast<size_t>(kTokens) * kNkv * kRopeDim;

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "FAIL %s\n", message);
    std::exit(1);
}

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", operation, cudaGetErrorString(result));
        std::exit(2);
    }
}

double parse_speedup_threshold(const char* value) {
    if (value == nullptr || value[0] == '\0') {
        std::fprintf(stderr, "invalid explicit numeric threshold\n");
        std::exit(2);
    }
    bool digit = false;
    bool dot = false;
    bool exponent = false;
    bool exponent_digit = false;
    for (size_t i = 0; value[i] != '\0'; ++i) {
        const unsigned char ch = static_cast<unsigned char>(value[i]);
        if (std::isdigit(ch)) {
            digit = true;
            if (exponent) exponent_digit = true;
        } else if (ch == '.' && !dot && !exponent) {
            dot = true;
        } else if ((ch == 'e' || ch == 'E') && digit && !exponent) {
            exponent = true;
        } else if ((ch == '+' || ch == '-') && exponent && !exponent_digit &&
                   i > 0 && (value[i - 1] == 'e' || value[i - 1] == 'E')) {
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
    const double min_speedup = std::strtod(value, &end);
    if (errno != 0 || end == value || *end != '\0' || !std::isfinite(min_speedup) ||
        min_speedup <= 1.0 || min_speedup > 100.0) {
        std::fprintf(stderr, "invalid explicit numeric threshold\n");
        std::exit(2);
    }
    return min_speedup;
}

bool valid_build_id(const char* value) {
    if (std::strlen(value) != 64) return false;
    for (size_t i = 0; i < 64; ++i) {
        if (!((value[i] >= '0' && value[i] <= '9') || (value[i] >= 'a' && value[i] <= 'f'))) {
            return false;
        }
    }
    return true;
}

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        check(cudaMalloc(&ptr_, count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return static_cast<T*>(ptr_); }
    size_t bytes() const { return count_ * sizeof(T); }

  private:
    void* ptr_ = nullptr;
    size_t count_;
};

class GuardedBf16Buffer {
  public:
    GuardedBf16Buffer(size_t count, unsigned char prefix_poison,
                      unsigned char suffix_poison)
        : storage_(kRedzoneBytes + count * sizeof(__nv_bfloat16) + kRedzoneBytes),
          payload_bytes_(count * sizeof(__nv_bfloat16)),
          prefix_poison_(prefix_poison),
          suffix_poison_(suffix_poison) {
        static_assert(kRedzoneBytes % 256 == 0);
        // cudaMalloc is 256-byte aligned and the 256-byte redzone means the
        // payload pointer preserves cudaMalloc's 256-byte alignment.
        if ((reinterpret_cast<uintptr_t>(get()) & 255U) != 0) fail("payload-alignment");
        reset_redzones();
    }

    __nv_bfloat16* get() const {
        return reinterpret_cast<__nv_bfloat16*>(storage_.get() + kRedzoneBytes);
    }

    void poison_payload(unsigned char poison) {
        check(cudaMemset(get(), poison, payload_bytes_), "poison guarded payload");
    }

    bool redzones_clean() const {
        std::array<unsigned char, kRedzoneBytes> prefix{};
        std::array<unsigned char, kRedzoneBytes> suffix{};
        check(cudaMemcpy(prefix.data(), storage_.get(), prefix.size(), cudaMemcpyDeviceToHost),
              "read prefix redzone");
        const auto* suffix_ptr = reinterpret_cast<const unsigned char*>(get()) + payload_bytes_;
        check(cudaMemcpy(suffix.data(), suffix_ptr, suffix.size(), cudaMemcpyDeviceToHost),
              "read suffix redzone");
        return std::all_of(prefix.begin(), prefix.end(), [&](unsigned char value) {
                   return value == prefix_poison_;
               }) &&
               std::all_of(suffix.begin(), suffix.end(), [&](unsigned char value) {
                   return value == suffix_poison_;
               });
    }

  private:
    void reset_redzones() {
        check(cudaMemset(storage_.get(), prefix_poison_, kRedzoneBytes),
              "initialize prefix redzone");
        auto* suffix_ptr = reinterpret_cast<unsigned char*>(get()) + payload_bytes_;
        check(cudaMemset(suffix_ptr, suffix_poison_, kRedzoneBytes),
              "initialize suffix redzone");
    }

    DeviceBuffer<unsigned char> storage_;
    size_t payload_bytes_;
    unsigned char prefix_poison_;
    unsigned char suffix_poison_;
};

void require_redzones(const GuardedBf16Buffer& buffer, const char* label) {
    if (!buffer.redzones_clean()) fail(label);
}

uint64_t fnv1a(uint64_t hash, const void* data, size_t bytes) {
    const auto* raw = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; ++i) hash = (hash ^ raw[i]) * 1099511628211ULL;
    return hash;
}

void fill_values(std::vector<__nv_bfloat16>& values, unsigned int salt) {
    for (size_t i = 0; i < values.size(); ++i) {
        const int signed_value = static_cast<int>((i * 131 + salt * 977) % 8191) - 4095;
        values[i] = __float2bfloat16(static_cast<float>(signed_value) / 257.0f);
    }
}

void fill_tables(
    std::vector<unsigned int>& positions,
    std::vector<float>& frequencies,
    unsigned int position_salt,
    unsigned int frequency_salt) {
    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i] = static_cast<unsigned int>((i * (position_salt * 2 + 1) + 17 * position_salt) % 131071);
    }
    for (size_t i = 0; i < frequencies.size(); ++i) {
        const float exponent = static_cast<float>(i) / static_cast<float>(frequencies.size());
        frequencies[i] = std::pow(10000.0f + 7500.0f * frequency_salt, -exponent);
    }
}

void incumbent_forward(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    __nv_bfloat16* q_tmp,
    __nv_bfloat16* k_tmp,
    const unsigned int* positions,
    const float* frequencies,
    float mscale) {
    const dim3 block256(256, 1, 1);
    const unsigned int q_total = kTokens * kNq * kRopeDim;
    const unsigned int k_total = kTokens * kNkv * kRopeDim;
    mla_q_rope_extract_batched<<<dim3((q_total + 255) / 256, 1, 1), block256>>>(
        q, q_tmp, kTokens, kNq, kHeadDim, kNopeDim, kRopeDim, kNq * kHeadDim);
    mla_q_rope_extract_batched<<<dim3((k_total + 255) / 256, 1, 1), block256>>>(
        k, k_tmp, kTokens, kNkv, kHeadDim, kNopeDim, kRopeDim, kNkv * kHeadDim);
    rope_forward_yarn_interleaved<<<dim3(kNq + kNkv, (kTokens + 3) / 4, 1), dim3(128, 1, 1)>>>(
        q_tmp, k_tmp, positions, kTokens, kNq, kNkv, kRopeDim, kRopeDim, frequencies, mscale);
    mla_q_rope_writeback_batched<<<dim3((q_total + 255) / 256, 1, 1), block256>>>(
        q_tmp, q, kTokens, kNq, kHeadDim, kNopeDim, kRopeDim, kNq * kHeadDim);
    mla_q_rope_writeback_batched<<<dim3((k_total + 255) / 256, 1, 1), block256>>>(
        k_tmp, k, kTokens, kNkv, kHeadDim, kNopeDim, kRopeDim, kNkv * kHeadDim);
}

void incumbent_inverse(
    __nv_bfloat16* q,
    __nv_bfloat16* q_tmp,
    const unsigned int* positions,
    const float* frequencies,
    float mscale) {
    const unsigned int total = kTokens * kNq * kRopeDim;
    mla_q_rope_extract_batched<<<dim3((total + 255) / 256, 1, 1), dim3(256, 1, 1)>>>(
        q, q_tmp, kTokens, kNq, kHeadDim, kNopeDim, kRopeDim, kNq * kHeadDim);
    rope_forward_yarn_interleaved_inv<<<dim3(kNq, (kTokens + 3) / 4, 1), dim3(128, 1, 1)>>>(
        q_tmp, q_tmp, positions, kTokens, kNq, 0, kRopeDim, kRopeDim, frequencies, mscale);
    mla_q_rope_writeback_batched<<<dim3((total + 255) / 256, 1, 1), dim3(256, 1, 1)>>>(
        q_tmp, q, kTokens, kNq, kHeadDim, kNopeDim, kRopeDim, kNq * kHeadDim);
}

void candidate_forward(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    const unsigned int* positions,
    const float* frequencies,
    float mscale) {
    v4_prefill_rope_fused_forward<<<dim3(kTokens, kNq + kNkv, 1), dim3(32, 1, 1)>>>(
        q, k, positions, frequencies, kTokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim, mscale);
}

void candidate_inverse(
    __nv_bfloat16* q,
    const unsigned int* positions,
    const float* frequencies,
    float mscale) {
    v4_prefill_rope_fused_inverse<<<dim3(kTokens, kNq, 1), dim3(32, 1, 1)>>>(
        q, positions, frequencies, kTokens, kNq, 0, kHeadDim, kNopeDim, kRopeDim, mscale);
}

bool equal_bytes(const void* lhs, const void* rhs, size_t bytes) {
    return std::memcmp(lhs, rhs, bytes) == 0;
}

bool all_poison(const std::vector<__nv_bfloat16>& values, unsigned char poison) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
    return std::all_of(bytes, bytes + values.size() * sizeof(__nv_bfloat16),
                       [poison](unsigned char value) { return value == poison; });
}

template <typename Setup, typename Launch>
float event_time(cudaEvent_t start, cudaEvent_t finish, Setup setup, Launch launch) {
    setup();
    check(cudaEventRecord(start), "cudaEventRecord(start)");
    launch();
    check(cudaEventRecord(finish), "cudaEventRecord(finish)");
    check(cudaEventSynchronize(finish), "cudaEventSynchronize(finish)");
    float milliseconds = 0.0f;
    check(cudaEventElapsedTime(&milliseconds, start, finish), "cudaEventElapsedTime");
    return milliseconds;
}

struct Timing {
    float baseline_ms;
    float candidate_ms;
    float speedup;
};

template <typename BaselineSetup, typename Baseline, typename CandidateSetup,
          typename Candidate>
Timing time_abba(BaselineSetup baseline_setup, Baseline baseline,
                 CandidateSetup candidate_setup, Candidate candidate) {
    // BEGIN V4 RoPE ABBA timing
    // Every round is deliberately ordered: baseline, candidate, candidate, baseline.
    baseline_setup();
    baseline();
    candidate_setup();
    candidate();
    check(cudaDeviceSynchronize(), "ABBA warmup");
    cudaEvent_t start = nullptr;
    cudaEvent_t finish = nullptr;
    check(cudaEventCreate(&start), "cudaEventCreate(start)");
    check(cudaEventCreate(&finish), "cudaEventCreate(finish)");
    float baseline_ms = 0.0f;
    float candidate_ms = 0.0f;
    for (int round = 0; round < kAbbaRounds; ++round) {
        baseline_ms += event_time(start, finish, baseline_setup, baseline);
        candidate_ms += event_time(start, finish, candidate_setup, candidate);
        candidate_ms += event_time(start, finish, candidate_setup, candidate);
        baseline_ms += event_time(start, finish, baseline_setup, baseline);
    }
    check(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check(cudaEventDestroy(finish), "cudaEventDestroy(finish)");
    baseline_ms /= 2.0f * kAbbaRounds;
    candidate_ms /= 2.0f * kAbbaRounds;
    if (!(baseline_ms > 0.0f) || !(candidate_ms > 0.0f)) fail("invalid-ABBA-timing");
    return {baseline_ms, candidate_ms, baseline_ms / candidate_ms};
    // END V4 RoPE ABBA timing
}

}  // namespace probe

int main(int argc, char** argv) {
    using namespace probe;
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <min_speedup>\n", argv[0]);
        return 2;
    }
    const double min_speedup = parse_speedup_threshold(argv[1]);
    if (!valid_build_id(V4_ROPE_PROBE_BUILD_ID)) fail("invalid-build-id");

    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    int driver = 0;
    int runtime = 0;
    check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");

    GuardedBf16Buffer q_baseline(kQElements, 0x11, 0x12);
    GuardedBf16Buffer q_candidate(kQElements, 0x21, 0x22);
    GuardedBf16Buffer k_baseline(kKElements, 0x31, 0x32);
    GuardedBf16Buffer k_candidate(kKElements, 0x41, 0x42);
    GuardedBf16Buffer inverse_baseline(kQElements, 0x51, 0x52);
    GuardedBf16Buffer inverse_candidate(kQElements, 0x61, 0x62);
    DeviceBuffer<__nv_bfloat16> q_tmp(kQTempElements);
    DeviceBuffer<__nv_bfloat16> k_tmp(kKTempElements);
    DeviceBuffer<unsigned int> d_positions(kTokens);
    DeviceBuffer<float> d_frequencies(kRopeDim / 2);

    std::vector<__nv_bfloat16> q_input(kQElements);
    std::vector<__nv_bfloat16> k_input(kKElements);
    std::vector<__nv_bfloat16> expected(kQElements);
    std::vector<__nv_bfloat16> actual(kQElements);
    std::vector<__nv_bfloat16> expected_k(kKElements);
    std::vector<__nv_bfloat16> actual_k(kKElements);
    std::vector<unsigned int> positions(kTokens);
    std::vector<float> frequencies(kRopeDim / 2);
    const float mscales[kParityCases] = {1.0f, 0.70710677f, 1.125f};
    uint64_t input_hash = 1469598103934665603ULL;
    uint64_t position_hash = 1469598103934665603ULL;
    uint64_t frequency_hash = 1469598103934665603ULL;
    int forward_mismatches = 0;
    int inverse_mismatches = 0;

    for (int test_case = 0; test_case < kParityCases; ++test_case) {
        const unsigned int position_salt = static_cast<unsigned int>(test_case + 1);
        const unsigned int frequency_salt = static_cast<unsigned int>(test_case + 2);
        const float mscale = mscales[test_case];
        fill_values(q_input, 11 + position_salt);
        fill_values(k_input, 29 + frequency_salt);
        fill_tables(positions, frequencies, position_salt, frequency_salt);
        input_hash = fnv1a(input_hash, q_input.data(), kQBytes);
        input_hash = fnv1a(input_hash, k_input.data(), kKBytes);
        position_hash = fnv1a(position_hash, positions.data(), positions.size() * sizeof(unsigned int));
        frequency_hash = fnv1a(frequency_hash, frequencies.data(), frequencies.size() * sizeof(float));
        frequency_hash = fnv1a(frequency_hash, &mscale, sizeof(mscale));

        check(cudaMemcpy(q_baseline.get(), q_input.data(), kQBytes, cudaMemcpyHostToDevice), "copy Q baseline");
        check(cudaMemcpy(q_candidate.get(), q_input.data(), kQBytes, cudaMemcpyHostToDevice), "copy Q candidate");
        check(cudaMemcpy(k_baseline.get(), k_input.data(), kKBytes, cudaMemcpyHostToDevice), "copy K baseline");
        check(cudaMemcpy(k_candidate.get(), k_input.data(), kKBytes, cudaMemcpyHostToDevice), "copy K candidate");
        check(cudaMemcpy(d_positions.get(), positions.data(), d_positions.bytes(), cudaMemcpyHostToDevice), "copy positions");
        check(cudaMemcpy(d_frequencies.get(), frequencies.data(), d_frequencies.bytes(), cudaMemcpyHostToDevice), "copy frequencies");
        incumbent_forward(q_baseline.get(), k_baseline.get(), q_tmp.get(), k_tmp.get(), d_positions.get(), d_frequencies.get(), mscale);
        candidate_forward(q_candidate.get(), k_candidate.get(), d_positions.get(), d_frequencies.get(), mscale);
        check(cudaDeviceSynchronize(), "forward parity synchronize");
        require_redzones(q_baseline, "forward baseline redzone");
        require_redzones(q_candidate, "forward candidate Q redzone");
        require_redzones(k_baseline, "forward baseline K redzone");
        require_redzones(k_candidate, "forward candidate K redzone");
        check(cudaMemcpy(expected.data(), q_baseline.get(), kQBytes, cudaMemcpyDeviceToHost), "read baseline Q");
        check(cudaMemcpy(actual.data(), q_candidate.get(), kQBytes, cudaMemcpyDeviceToHost), "read candidate Q");
        check(cudaMemcpy(expected_k.data(), k_baseline.get(), kKBytes, cudaMemcpyDeviceToHost), "read baseline K");
        check(cudaMemcpy(actual_k.data(), k_candidate.get(), kKBytes, cudaMemcpyDeviceToHost), "read candidate K");
        forward_mismatches += !equal_bytes(expected.data(), actual.data(), kQBytes);
        forward_mismatches += !equal_bytes(expected_k.data(), actual_k.data(), kKBytes);

        check(cudaMemcpy(inverse_baseline.get(), q_input.data(), kQBytes, cudaMemcpyHostToDevice), "copy inverse baseline");
        check(cudaMemcpy(inverse_candidate.get(), q_input.data(), kQBytes, cudaMemcpyHostToDevice), "copy inverse candidate");
        incumbent_inverse(inverse_baseline.get(), q_tmp.get(), d_positions.get(), d_frequencies.get(), mscale);
        candidate_inverse(inverse_candidate.get(), d_positions.get(), d_frequencies.get(), mscale);
        check(cudaDeviceSynchronize(), "inverse parity synchronize");
        require_redzones(inverse_baseline, "inverse baseline redzone");
        require_redzones(inverse_candidate, "inverse candidate redzone");
        check(cudaMemcpy(expected.data(), inverse_baseline.get(), kQBytes, cudaMemcpyDeviceToHost), "read inverse baseline");
        check(cudaMemcpy(actual.data(), inverse_candidate.get(), kQBytes, cudaMemcpyDeviceToHost), "read inverse candidate");
        inverse_mismatches += !equal_bytes(expected.data(), actual.data(), kQBytes);
    }
    if (forward_mismatches != 0 || inverse_mismatches != 0) fail("byte-parity");

    int guards_unchanged = 0;
    // BEGIN V4 RoPE forward poison guard
    auto run_forward_guard = [&](__nv_bfloat16* q, __nv_bfloat16* k,
                                 const unsigned int* guard_positions,
                                 const float* guard_frequencies,
                                 unsigned int num_tokens, unsigned int num_q_heads,
                                 unsigned int num_kv_heads, unsigned int head_dim,
                                 unsigned int nope_dim, unsigned int rotary_dim,
                                 dim3 grid, dim3 block, const char* label,
                                 float guard_mscale = 1.0f) {
        q_candidate.poison_payload(0xA5);
        k_candidate.poison_payload(0x5A);
        v4_prefill_rope_fused_forward<<<grid, block>>>(
            q, k, guard_positions, guard_frequencies, num_tokens, num_q_heads,
            num_kv_heads, head_dim, nope_dim, rotary_dim, guard_mscale);
        check(cudaGetLastError(), label);
        check(cudaDeviceSynchronize(), label);
        require_redzones(q_candidate, label);
        require_redzones(k_candidate, label);
        check(cudaMemcpy(actual.data(), q_candidate.get(), kQBytes, cudaMemcpyDeviceToHost),
              "read forward poison Q");
        check(cudaMemcpy(actual_k.data(), k_candidate.get(), kKBytes, cudaMemcpyDeviceToHost),
              "read forward poison K");
        if (!all_poison(actual, 0xA5) || !all_poison(actual_k, 0x5A)) fail(label);
        ++guards_unchanged;
    };
    const dim3 forward_grid(kTokens, kNq + kNkv, 1);
    const dim3 inverse_grid(kTokens, kNq, 1);
    const dim3 valid_block(32, 1, 1);
    run_forward_guard(nullptr, k_candidate.get(), d_positions.get(), d_frequencies.get(),
                      kTokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
                      forward_grid, valid_block, "forward-null-q");
    run_forward_guard(q_candidate.get(), nullptr, d_positions.get(), d_frequencies.get(),
                      kTokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
                      forward_grid, valid_block, "forward-null-k");
    run_forward_guard(q_candidate.get(), k_candidate.get(), nullptr, d_frequencies.get(),
                      kTokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
                      forward_grid, valid_block, "forward-null-positions");
    run_forward_guard(q_candidate.get(), k_candidate.get(), d_positions.get(), nullptr,
                      kTokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
                      forward_grid, valid_block, "forward-null-inv-freq");
    run_forward_guard(q_candidate.get(), q_candidate.get(), d_positions.get(),
                      d_frequencies.get(), kTokens, kNq, kNkv, kHeadDim, kNopeDim,
                      kRopeDim, forward_grid, valid_block, "forward-q-k-alias");
    run_forward_guard(q_candidate.get(), k_candidate.get(), d_positions.get(),
                      d_frequencies.get(), kTokens, kNq, kNkv, kHeadDim, kNopeDim,
                      kRopeDim, forward_grid, valid_block, "forward-mscale-nan",
                      std::numeric_limits<float>::quiet_NaN());
    run_forward_guard(q_candidate.get(), k_candidate.get(), d_positions.get(),
                      d_frequencies.get(), kTokens, kNq, kNkv, kHeadDim, kNopeDim,
                      kRopeDim, forward_grid, valid_block, "forward-mscale-pos-inf",
                      std::numeric_limits<float>::infinity());
    run_forward_guard(q_candidate.get(), k_candidate.get(), d_positions.get(),
                      d_frequencies.get(), kTokens, kNq, kNkv, kHeadDim, kNopeDim,
                      kRopeDim, forward_grid, valid_block, "forward-mscale-neg-inf",
                      -std::numeric_limits<float>::infinity());
    run_forward_guard(q_candidate.get(), k_candidate.get(), d_positions.get(),
                      d_frequencies.get(), 0, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
                      forward_grid, valid_block, "forward-zero-tokens");
    run_forward_guard(q_candidate.get(), k_candidate.get(), d_positions.get(),
                      d_frequencies.get(), kTokens, kNq - 1, kNkv, kHeadDim, kNopeDim,
                      kRopeDim, forward_grid, valid_block, "forward-q-heads");
    run_forward_guard(q_candidate.get(), k_candidate.get(), d_positions.get(),
                      d_frequencies.get(), kTokens, kNq, kNkv + 1, kHeadDim, kNopeDim,
                      kRopeDim, forward_grid, valid_block, "forward-kv-heads");
    run_forward_guard(q_candidate.get(), k_candidate.get(), d_positions.get(),
                      d_frequencies.get(), kTokens, kNq, kNkv, kHeadDim - 1, kNopeDim,
                      kRopeDim, forward_grid, valid_block, "forward-head-dim");
    run_forward_guard(q_candidate.get(), k_candidate.get(), d_positions.get(),
                      d_frequencies.get(), kTokens, kNq, kNkv, kHeadDim, kNopeDim - 1,
                      kRopeDim, forward_grid, valid_block, "forward-nope-dim");
    run_forward_guard(q_candidate.get(), k_candidate.get(), d_positions.get(),
                      d_frequencies.get(), kTokens, kNq, kNkv, kHeadDim, kNopeDim,
                      kRopeDim - 2, forward_grid, valid_block, "forward-rotary-dim");
    run_forward_guard(q_candidate.get(), k_candidate.get(), d_positions.get(),
                      d_frequencies.get(), kTokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
                      forward_grid, dim3(16, 1, 1), "forward-block-x");
    // Y/Z must pair with X=16 so the launch stays within __launch_bounds__(32).
    run_forward_guard(q_candidate.get(), k_candidate.get(), d_positions.get(),
                      d_frequencies.get(), kTokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
                      forward_grid, dim3(16, 2, 1), "forward-block-y");
    run_forward_guard(q_candidate.get(), k_candidate.get(), d_positions.get(),
                      d_frequencies.get(), kTokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
                      forward_grid, dim3(16, 1, 2), "forward-block-z");
    run_forward_guard(q_candidate.get(), k_candidate.get(), d_positions.get(),
                      d_frequencies.get(), kTokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
                      dim3(kTokens + 1, kNq + kNkv, 1), valid_block, "forward-grid-x");
    run_forward_guard(q_candidate.get(), k_candidate.get(), d_positions.get(),
                      d_frequencies.get(), kTokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
                      dim3(kTokens, kNq + kNkv + 1, 1), valid_block, "forward-grid-y");
    run_forward_guard(q_candidate.get(), k_candidate.get(), d_positions.get(),
                      d_frequencies.get(), kTokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
                      dim3(kTokens, kNq + kNkv, 2), valid_block, "forward-grid-z");
    // END V4 RoPE forward poison guard

    // BEGIN V4 RoPE inverse poison guard
    auto run_inverse_guard = [&](__nv_bfloat16* q,
                                 const unsigned int* guard_positions,
                                 const float* guard_frequencies,
                                 unsigned int num_tokens, unsigned int num_q_heads,
                                 unsigned int num_kv_heads, unsigned int head_dim,
                                 unsigned int nope_dim, unsigned int rotary_dim,
                                 dim3 grid, dim3 block, const char* label,
                                 float guard_mscale = 1.0f) {
        inverse_candidate.poison_payload(0x3C);
        v4_prefill_rope_fused_inverse<<<grid, block>>>(
            q, guard_positions, guard_frequencies, num_tokens, num_q_heads, num_kv_heads,
            head_dim, nope_dim, rotary_dim, guard_mscale);
        check(cudaGetLastError(), label);
        check(cudaDeviceSynchronize(), label);
        require_redzones(inverse_candidate, label);
        check(cudaMemcpy(actual.data(), inverse_candidate.get(), kQBytes, cudaMemcpyDeviceToHost),
              "read inverse poison Q");
        if (!all_poison(actual, 0x3C)) fail(label);
        ++guards_unchanged;
    };
    run_inverse_guard(nullptr, d_positions.get(), d_frequencies.get(), kTokens, kNq, 0,
                      kHeadDim, kNopeDim, kRopeDim, inverse_grid, valid_block,
                      "inverse-null-q");
    run_inverse_guard(inverse_candidate.get(), nullptr, d_frequencies.get(), kTokens, kNq, 0,
                      kHeadDim, kNopeDim, kRopeDim, inverse_grid, valid_block,
                      "inverse-null-positions");
    run_inverse_guard(inverse_candidate.get(), d_positions.get(), nullptr, kTokens, kNq, 0,
                      kHeadDim, kNopeDim, kRopeDim, inverse_grid, valid_block,
                      "inverse-null-inv-freq");
    run_inverse_guard(inverse_candidate.get(), d_positions.get(), d_frequencies.get(), kTokens,
                      kNq, 0, kHeadDim, kNopeDim, kRopeDim, inverse_grid, valid_block,
                      "inverse-mscale-nan", std::numeric_limits<float>::quiet_NaN());
    run_inverse_guard(inverse_candidate.get(), d_positions.get(), d_frequencies.get(), kTokens,
                      kNq, 0, kHeadDim, kNopeDim, kRopeDim, inverse_grid, valid_block,
                      "inverse-mscale-pos-inf", std::numeric_limits<float>::infinity());
    run_inverse_guard(inverse_candidate.get(), d_positions.get(), d_frequencies.get(), kTokens,
                      kNq, 0, kHeadDim, kNopeDim, kRopeDim, inverse_grid, valid_block,
                      "inverse-mscale-neg-inf", -std::numeric_limits<float>::infinity());
    run_inverse_guard(inverse_candidate.get(), d_positions.get(), d_frequencies.get(), 0,
                      kNq, 0, kHeadDim, kNopeDim, kRopeDim, inverse_grid, valid_block,
                      "inverse-zero-tokens");
    run_inverse_guard(inverse_candidate.get(), d_positions.get(), d_frequencies.get(), kTokens,
                      kNq - 1, 0, kHeadDim, kNopeDim, kRopeDim, inverse_grid, valid_block,
                      "inverse-q-heads");
    run_inverse_guard(inverse_candidate.get(), d_positions.get(), d_frequencies.get(), kTokens,
                      kNq, 1, kHeadDim, kNopeDim, kRopeDim, inverse_grid, valid_block,
                      "inverse-kv-heads");
    run_inverse_guard(inverse_candidate.get(), d_positions.get(), d_frequencies.get(), kTokens,
                      kNq, 0, kHeadDim - 1, kNopeDim, kRopeDim, inverse_grid, valid_block,
                      "inverse-head-dim");
    run_inverse_guard(inverse_candidate.get(), d_positions.get(), d_frequencies.get(), kTokens,
                      kNq, 0, kHeadDim, kNopeDim - 1, kRopeDim, inverse_grid, valid_block,
                      "inverse-nope-dim");
    run_inverse_guard(inverse_candidate.get(), d_positions.get(), d_frequencies.get(), kTokens,
                      kNq, 0, kHeadDim, kNopeDim, kRopeDim - 2, inverse_grid, valid_block,
                      "inverse-rotary-dim");
    run_inverse_guard(inverse_candidate.get(), d_positions.get(), d_frequencies.get(), kTokens,
                      kNq, 0, kHeadDim, kNopeDim, kRopeDim, inverse_grid, dim3(16, 1, 1),
                      "inverse-block-x");
    run_inverse_guard(inverse_candidate.get(), d_positions.get(), d_frequencies.get(), kTokens,
                      kNq, 0, kHeadDim, kNopeDim, kRopeDim, inverse_grid, dim3(16, 2, 1),
                      "inverse-block-y");
    run_inverse_guard(inverse_candidate.get(), d_positions.get(), d_frequencies.get(), kTokens,
                      kNq, 0, kHeadDim, kNopeDim, kRopeDim, inverse_grid, dim3(16, 1, 2),
                      "inverse-block-z");
    run_inverse_guard(inverse_candidate.get(), d_positions.get(), d_frequencies.get(), kTokens,
                      kNq, 0, kHeadDim, kNopeDim, kRopeDim,
                      dim3(kTokens + 1, kNq, 1), valid_block, "inverse-grid-x");
    run_inverse_guard(inverse_candidate.get(), d_positions.get(), d_frequencies.get(), kTokens,
                      kNq, 0, kHeadDim, kNopeDim, kRopeDim,
                      dim3(kTokens, kNq + 1, 1), valid_block, "inverse-grid-y");
    run_inverse_guard(inverse_candidate.get(), d_positions.get(), d_frequencies.get(), kTokens,
                      kNq, 0, kHeadDim, kNopeDim, kRopeDim,
                      dim3(kTokens, kNq, 2), valid_block, "inverse-grid-z");
    // END V4 RoPE inverse poison guard
    if (guards_unchanged != kPoisonCases) fail("guard-poison");

    const float timing_mscale = mscales[kParityCases - 1];
    auto baseline_reset = [&]() {
        check(cudaMemcpy(q_baseline.get(), q_input.data(), kQBytes, cudaMemcpyHostToDevice), "timing reset Q baseline");
        check(cudaMemcpy(k_baseline.get(), k_input.data(), kKBytes, cudaMemcpyHostToDevice), "timing reset K baseline");
        check(cudaMemcpy(inverse_baseline.get(), q_input.data(), kQBytes, cudaMemcpyHostToDevice), "timing reset inverse baseline");
    };
    auto candidate_reset = [&]() {
        check(cudaMemcpy(q_candidate.get(), q_input.data(), kQBytes, cudaMemcpyHostToDevice), "timing reset Q candidate");
        check(cudaMemcpy(k_candidate.get(), k_input.data(), kKBytes, cudaMemcpyHostToDevice), "timing reset K candidate");
        check(cudaMemcpy(inverse_candidate.get(), q_input.data(), kQBytes, cudaMemcpyHostToDevice), "timing reset inverse candidate");
    };
    auto baseline = [&]() {
        incumbent_forward(q_baseline.get(), k_baseline.get(), q_tmp.get(), k_tmp.get(), d_positions.get(), d_frequencies.get(), timing_mscale);
        incumbent_inverse(inverse_baseline.get(), q_tmp.get(), d_positions.get(), d_frequencies.get(), timing_mscale);
    };
    auto candidate = [&]() {
        candidate_forward(q_candidate.get(), k_candidate.get(), d_positions.get(), d_frequencies.get(), timing_mscale);
        candidate_inverse(inverse_candidate.get(), d_positions.get(), d_frequencies.get(), timing_mscale);
    };
    const Timing timing = time_abba(baseline_reset, baseline, candidate_reset, candidate);
    require_redzones(q_baseline, "timing baseline Q redzone");
    require_redzones(q_candidate, "timing candidate Q redzone");
    require_redzones(k_baseline, "timing baseline K redzone");
    require_redzones(k_candidate, "timing candidate K redzone");
    require_redzones(inverse_baseline, "timing baseline inverse redzone");
    require_redzones(inverse_candidate, "timing candidate inverse redzone");
    if (!std::isfinite(timing.speedup) || timing.speedup < min_speedup) fail("speedup-threshold");

    const auto* uuid = reinterpret_cast<const unsigned char*>(&properties.uuid);
    std::printf("build_id=%s\n", V4_ROPE_PROBE_BUILD_ID);
    std::printf("device_uuid=");
    for (int i = 0; i < 16; ++i) std::printf("%02x", uuid[i]);
    std::printf(" driver=%d runtime=%d\n", driver, runtime);
    std::printf("input_hash=%016llx position_hash=%016llx frequency_hash=%016llx\n",
                static_cast<unsigned long long>(input_hash),
                static_cast<unsigned long long>(position_hash),
                static_cast<unsigned long long>(frequency_hash));
    std::printf("parity cases=%d forward_mismatches=%d inverse_mismatches=%d\n",
                kParityCases, forward_mismatches, inverse_mismatches);
    std::printf(
        "guards poison_cases=%d unchanged=%d redzone_buffers=%d prefix=clean suffix=clean\n",
        kPoisonCases, guards_unchanged, kRedzoneBuffers);
    std::printf("timing baseline_ms=%.9g candidate_ms=%.9g speedup=%.9g abba_rounds=%d\n",
                timing.baseline_ms, timing.candidate_ms, timing.speedup, kAbbaRounds);
    std::printf("threshold min_speedup=%s\n", argv[1]);
    std::printf("result=PASS\n");
    return 0;
}
