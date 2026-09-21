// SPDX-License-Identifier: AGPL-3.0-only

// Standalone GPU promotion probe for the isolated V4 Q-B norm + RoPE
// experiment. It is deliberately absent from Atlas's registry and serving path.

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

#include "../deepseek-v4-flash/nvfp4/v4_prefill_qb_norm_rope_fused.cu"

#ifndef V4_QB_NORM_ROPE_PROBE_BUILD_ID
#error "V4_QB_NORM_ROPE_PROBE_BUILD_ID must bind this probe to its receipt"
#endif

namespace probe {

constexpr unsigned kProductionTokens = 2410;
constexpr unsigned kNq = 64;
constexpr unsigned kNkv = 1;
constexpr unsigned kHeadDim = 512;
constexpr unsigned kNopeDim = 448;
constexpr unsigned kRopeDim = 64;
constexpr unsigned kRopePairs = 32;
constexpr unsigned kThreads = 512;
constexpr unsigned kTokenCases[] = {1, 7, 128, kProductionTokens};
constexpr int kParityCases = 4;
constexpr int kAbbaRounds = 6;

[[noreturn]] static void fail(const char* message) {
    std::fprintf(stderr, "FAIL %s\n", message);
    std::exit(1);
}

static void cuda_ok(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", operation, cudaGetErrorString(result));
        std::exit(1);
    }
}

static uint64_t fnv1a(uint64_t hash, const void* data, size_t bytes) {
    const auto* raw = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; ++i) hash = (hash ^ raw[i]) * 1099511628211ULL;
    return hash;
}

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        cuda_ok(cudaMalloc(&pointer_, count_ * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return static_cast<T*>(pointer_); }
    size_t bytes() const { return count_ * sizeof(T); }
    void upload(const void* source) {
        cuda_ok(cudaMemcpy(pointer_, source, bytes(), cudaMemcpyHostToDevice), "upload");
    }

  private:
    void* pointer_ = nullptr;
    size_t count_ = 0;
};

template <typename T>
class Guarded {
  public:
    static constexpr size_t kGuardBytes = 256;
    explicit Guarded(size_t count) : count_(count) {
        cuda_ok(cudaMalloc(&allocation_, bytes() + 2 * kGuardBytes), "guarded cudaMalloc");
        data_ = reinterpret_cast<T*>(allocation_ + kGuardBytes);
    }
    ~Guarded() { cudaFree(allocation_); }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    T* data() const { return data_; }
    size_t bytes() const { return count_ * sizeof(T); }
    void upload(const void* source) {
        cuda_ok(cudaMemcpy(data_, source, bytes(), cudaMemcpyHostToDevice), "guarded upload");
    }
    void set_guards(unsigned char prefix, unsigned char suffix) {
        prefix_ = prefix;
        suffix_ = suffix;
        cuda_ok(cudaMemset(allocation_, prefix_, kGuardBytes), "prefix guard");
        cuda_ok(cudaMemset(allocation_ + kGuardBytes + bytes(), suffix_, kGuardBytes),
                "suffix guard");
    }
    std::vector<unsigned char> download_data() const {
        std::vector<unsigned char> result(bytes());
        cuda_ok(cudaMemcpy(result.data(), data_, bytes(), cudaMemcpyDeviceToHost),
                "download data");
        return result;
    }
    std::vector<unsigned char> download_all() const {
        std::vector<unsigned char> result(bytes() + 2 * kGuardBytes);
        cuda_ok(cudaMemcpy(result.data(), allocation_, result.size(), cudaMemcpyDeviceToHost),
                "download guarded allocation");
        return result;
    }
    bool guards_clean() const {
        std::vector<unsigned char> prefix(kGuardBytes);
        std::vector<unsigned char> suffix(kGuardBytes);
        cuda_ok(cudaMemcpy(prefix.data(), allocation_, kGuardBytes, cudaMemcpyDeviceToHost),
                "read prefix guard");
        cuda_ok(cudaMemcpy(suffix.data(), allocation_ + kGuardBytes + bytes(), kGuardBytes,
                           cudaMemcpyDeviceToHost),
                "read suffix guard");
        return std::all_of(prefix.begin(), prefix.end(),
                           [&](unsigned char value) { return value == prefix_; }) &&
               std::all_of(suffix.begin(), suffix.end(),
                           [&](unsigned char value) { return value == suffix_; });
    }

  private:
    unsigned char* allocation_ = nullptr;
    T* data_ = nullptr;
    size_t count_ = 0;
    unsigned char prefix_ = 0;
    unsigned char suffix_ = 0;
};

// BEGIN V4 Q-B exact local oracle
__device__ __forceinline__ void probe_unpack(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(
        __ushort_as_bfloat16(static_cast<unsigned short>(packed & 0xFFFFU)));
    v1 = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(packed >> 16)));
}

__device__ __forceinline__ unsigned int probe_pack(float v0, float v1) {
    const unsigned int lo = __bfloat16_as_ushort(__float2bfloat16(v0));
    const unsigned int hi = __bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

__device__ __forceinline__ float probe_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
    }
    return value;
}

extern "C" __global__ __launch_bounds__(512) void probe_incumbent_qb_rms(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int head_dim,
    float eps) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const __nv_bfloat16* const x = input + static_cast<size_t>(row) * head_dim;
    __nv_bfloat16* const out = output + static_cast<size_t>(row) * head_dim;
    const unsigned int* const x32 = reinterpret_cast<const unsigned int*>(x);
    const unsigned int* const weight32 = reinterpret_cast<const unsigned int*>(weight);
    unsigned int packed = 0;
    float x0 = 0.0f;
    float x1 = 0.0f;
    float sum_sq = 0.0f;
    if (tid < head_dim / 2) {
        packed = x32[tid];
        probe_unpack(packed, x0, x1);
        sum_sq += x0 * x0 + x1 * x1;
    }
    sum_sq = probe_warp_sum(sum_sq);
    __shared__ float warp_sums[32];
    const unsigned int lane = tid & 31;
    const unsigned int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        float value = lane < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0f;
        value = probe_warp_sum(value);
        if (lane == 0) warp_sums[0] = value;
    }
    __syncthreads();
    const float rms = rsqrtf(warp_sums[0] / static_cast<float>(kHeadDim) + eps);
    if (tid < head_dim / 2) {
        // Reload like incumbent rms_norm's apply pass; weight is the zero-filled
        // norm_unit_w buffer, so this remains a pure normalize.
        probe_unpack(x32[tid], x0, x1);
        float w0;
        float w1;
        probe_unpack(weight32[tid], w0, w1);
        reinterpret_cast<unsigned int*>(out)[tid] =
            probe_pack(x0 * rms * (1.0f + w0), x1 * rms * (1.0f + w1));
    }
}

extern "C" __global__ __launch_bounds__(32) void probe_incumbent_direct_rope(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ inv_freq,
    unsigned int tokens,
    unsigned int num_q_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int nope_dim,
    float mscale) {
    const unsigned int token = blockIdx.x;
    const unsigned int head_slot = blockIdx.y;
    const unsigned int pair = threadIdx.x;
    const bool is_q = head_slot < num_q_heads;
    const unsigned int head = is_q ? head_slot : head_slot - num_q_heads;
    const unsigned int tensor_heads = is_q ? num_q_heads : num_kv_heads;
    __nv_bfloat16* const base = is_q ? q : k;
    __nv_bfloat16* const pointer =
        base + (static_cast<size_t>(token) * tensor_heads + head) * head_dim + nope_dim;
    const float x0 = static_cast<float>(pointer[2 * pair]);
    const float x1 = static_cast<float>(pointer[2 * pair + 1]);
    const float angle = static_cast<float>(positions[token]) * inv_freq[pair];
    const float cos_value = cosf(angle) * mscale;
    const float sin_value = sinf(angle) * mscale;
    pointer[2 * pair] = __float2bfloat16(x0 * cos_value - x1 * sin_value);
    pointer[2 * pair + 1] = __float2bfloat16(x1 * cos_value + x0 * sin_value);
    (void)tokens;
}
// END V4 Q-B exact local oracle

// BEGIN V4 Q-B probe kernel contract
static void launch_baseline(__nv_bfloat16* q, __nv_bfloat16* k,
                            const __nv_bfloat16* zero_weight,
                            const unsigned int* positions, const float* frequencies,
                            unsigned int tokens, float eps, float mscale) {
    probe_incumbent_qb_rms<<<dim3(tokens * kNq, 1, 1), dim3(kThreads, 1, 1)>>>(
        q, zero_weight, q, kHeadDim, eps);
    probe_incumbent_direct_rope<<<dim3(tokens, kNq + kNkv, 1),
                                  dim3(kRopePairs, 1, 1)>>>(
        q, k, positions, frequencies, tokens, kNq, kNkv, kHeadDim, kNopeDim, mscale);
}

static void launch_candidate(__nv_bfloat16* q, __nv_bfloat16* k,
                             const __nv_bfloat16* zero_weight,
                             const unsigned int* positions, const float* frequencies,
                             unsigned int tokens, float eps, float mscale) {
    v4_prefill_qb_norm_rope_fused<<<dim3(tokens, kNq, 1), dim3(kThreads, 1, 1)>>>(
        q, k, zero_weight, positions, frequencies, tokens, kNq, kNkv, kHeadDim,
        kNopeDim, kRopeDim, eps, mscale);
}
// END V4 Q-B probe kernel contract

static void fill_bf16(std::vector<__nv_bfloat16>& values, unsigned salt) {
    for (size_t i = 0; i < values.size(); ++i) {
        const int value = static_cast<int>((i * 131ULL + salt * 977ULL) % 8191ULL) - 4095;
        values[i] = __float2bfloat16(static_cast<float>(value) / 257.0f);
    }
}

static void fill_tables(std::vector<unsigned int>& positions, std::vector<float>& frequencies,
                        unsigned salt) {
    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i] = static_cast<unsigned int>(17 * salt + i * (2 * salt + 1));
    }
    for (size_t pair = 0; pair < frequencies.size(); ++pair) {
        frequencies[pair] =
            std::pow(10000.0f + 7500.0f * salt,
                     -static_cast<float>(pair) / static_cast<float>(frequencies.size()));
    }
}

static size_t mismatch_bytes(const std::vector<unsigned char>& expected,
                             const std::vector<unsigned char>& actual) {
    if (expected.size() != actual.size()) fail("mismatch extent");
    size_t mismatches = 0;
    for (size_t i = 0; i < expected.size(); ++i) mismatches += expected[i] != actual[i];
    return mismatches;
}

struct Evidence {
    uint64_t input_hash = 1469598103934665603ULL;
    uint64_t position_hash = 1469598103934665603ULL;
    uint64_t frequency_hash = 1469598103934665603ULL;
    uint64_t weight_hash = 1469598103934665603ULL;
    size_t q_mismatch_bytes = 0;
    size_t k_mismatch_bytes = 0;
};

// BEGIN V4 Q-B exact byte parity
static void run_parity_case(unsigned int tokens, unsigned int case_index, Evidence& evidence) {
    const size_t q_elements = static_cast<size_t>(tokens) * kNq * kHeadDim;
    const size_t k_elements = static_cast<size_t>(tokens) * kNkv * kHeadDim;
    std::vector<__nv_bfloat16> q_input(q_elements);
    std::vector<__nv_bfloat16> k_input(k_elements);
    std::vector<__nv_bfloat16> zero_weight(kHeadDim);
    std::vector<unsigned int> positions(tokens);
    std::vector<float> frequencies(kRopePairs);
    fill_bf16(q_input, 11 + case_index * 7);
    fill_bf16(k_input, 29 + case_index * 13);
    std::memset(zero_weight.data(), 0, zero_weight.size() * sizeof(__nv_bfloat16));
    // Every case uses distinct positions and distinct frequencies.
    fill_tables(positions, frequencies, case_index + 1);
    const auto* weight_bytes = reinterpret_cast<const unsigned char*>(zero_weight.data());
    if (!std::all_of(weight_bytes,
                     weight_bytes + zero_weight.size() * sizeof(__nv_bfloat16),
                     [](unsigned char value) { return value == 0; })) {
        fail("zero_weight invariant");
    }

    DeviceBuffer<__nv_bfloat16> weight(kHeadDim);
    DeviceBuffer<unsigned int> device_positions(tokens);
    DeviceBuffer<float> device_frequencies(kRopePairs);
    Guarded<__nv_bfloat16> baseline_q(q_elements);
    Guarded<__nv_bfloat16> baseline_k(k_elements);
    Guarded<__nv_bfloat16> candidate_q(q_elements);
    Guarded<__nv_bfloat16> candidate_k(k_elements);
    weight.upload(zero_weight.data());
    device_positions.upload(positions.data());
    device_frequencies.upload(frequencies.data());
    baseline_q.upload(q_input.data());
    baseline_k.upload(k_input.data());
    candidate_q.upload(q_input.data());
    candidate_k.upload(k_input.data());
    baseline_q.set_guards(0xA5, 0x5A);
    baseline_k.set_guards(0x96, 0x69);
    candidate_q.set_guards(0xC3, 0x3C);
    candidate_k.set_guards(0xD2, 0x2D);

    const float eps = 1.0e-6f * static_cast<float>(case_index + 1);
    constexpr float mscales[kParityCases] = {1.0f, 0.70710677f, 1.125f, 0.875f};
    launch_baseline(baseline_q.data(), baseline_k.data(), weight.get(), device_positions.get(),
                    device_frequencies.get(), tokens, eps, mscales[case_index]);
    launch_candidate(candidate_q.data(), candidate_k.data(), weight.get(), device_positions.get(),
                     device_frequencies.get(), tokens, eps, mscales[case_index]);
    cuda_ok(cudaGetLastError(), "parity launch");
    cuda_ok(cudaDeviceSynchronize(), "parity synchronize");

    // full Q and K byte memcmp, plus opposite poison prefix_guard/suffix_guard regions.
    const auto expected_q = baseline_q.download_data();
    const auto actual_q = candidate_q.download_data();
    const auto expected_k = baseline_k.download_data();
    const auto actual_k = candidate_k.download_data();
    evidence.q_mismatch_bytes += mismatch_bytes(expected_q, actual_q);
    evidence.k_mismatch_bytes += mismatch_bytes(expected_k, actual_k);
    if (std::memcmp(expected_q.data(), actual_q.data(), expected_q.size()) != 0 ||
        std::memcmp(expected_k.data(), actual_k.data(), expected_k.size()) != 0 ||
        !baseline_q.guards_clean() || !baseline_k.guards_clean() ||
        !candidate_q.guards_clean() || !candidate_k.guards_clean()) {
        fail("byte parity or poison guard");
    }

    evidence.input_hash = fnv1a(evidence.input_hash, q_input.data(), q_input.size() * 2);
    evidence.input_hash = fnv1a(evidence.input_hash, k_input.data(), k_input.size() * 2);
    evidence.position_hash =
        fnv1a(evidence.position_hash, positions.data(), positions.size() * sizeof(unsigned));
    evidence.frequency_hash =
        fnv1a(evidence.frequency_hash, frequencies.data(), frequencies.size() * sizeof(float));
    evidence.weight_hash =
        fnv1a(evidence.weight_hash, zero_weight.data(), zero_weight.size() * 2);
}
// kTokenCases drives representative 1/7/128-token cases and exact N=2410.
// END V4 Q-B exact byte parity

// BEGIN V4 Q-B malformed ABI
constexpr int kMalformedCases = 22;
static int run_malformed_cases() {
    constexpr unsigned tokens = 2;
    const size_t q_elements = static_cast<size_t>(tokens) * kNq * kHeadDim;
    const size_t k_elements = static_cast<size_t>(tokens) * kNkv * kHeadDim;
    std::vector<__nv_bfloat16> q_input(q_elements);
    std::vector<__nv_bfloat16> k_input(k_elements);
    std::vector<__nv_bfloat16> zero_weight(kHeadDim);
    std::vector<unsigned int> positions(tokens);
    std::vector<float> frequencies(kRopePairs);
    fill_bf16(q_input, 101);
    fill_bf16(k_input, 103);
    std::memset(zero_weight.data(), 0, zero_weight.size() * 2);
    fill_tables(positions, frequencies, 11);
    Guarded<__nv_bfloat16> q(q_elements);
    Guarded<__nv_bfloat16> k(k_elements);
    DeviceBuffer<__nv_bfloat16> weight(kHeadDim);
    DeviceBuffer<unsigned int> device_positions(tokens);
    DeviceBuffer<float> device_frequencies(kRopePairs);
    weight.upload(zero_weight.data());
    device_positions.upload(positions.data());
    device_frequencies.upload(frequencies.data());
    int no_write = 0;

    auto malformed = [&](const char* label, __nv_bfloat16* q_argument,
                         __nv_bfloat16* k_argument, const __nv_bfloat16* weight_argument,
                         const unsigned int* position_argument, const float* frequency_argument,
                         unsigned int token_argument, unsigned int nq_argument,
                         unsigned int nkv_argument, unsigned int head_argument,
                         unsigned int nope_argument, unsigned int rope_argument, float eps,
                         float mscale, dim3 grid, dim3 block) {
        q.upload(q_input.data());
        k.upload(k_input.data());
        q.set_guards(0x4B, 0xB4);
        k.set_guards(0x6D, 0xD6);
        const auto before = q.download_all();
        const auto before_k = k.download_all();
        v4_prefill_qb_norm_rope_fused<<<grid, block>>>(
            q_argument, k_argument, weight_argument, position_argument, frequency_argument,
            token_argument, nq_argument, nkv_argument, head_argument, nope_argument,
            rope_argument, eps, mscale);
        cuda_ok(cudaGetLastError(), label);
        cuda_ok(cudaDeviceSynchronize(), label);
        const auto after = q.download_all();
        const auto after_k = k.download_all();
        if (!(before == after) || !(before_k == after_k) || !q.guards_clean() ||
            !k.guards_clean()) {
            fail("malformed ABI wrote output");
        }
        ++no_write;
    };
    const dim3 valid_grid(tokens, kNq, 1);
    const dim3 valid_block(kThreads, 1, 1);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    malformed("null-q", nullptr, k.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
              1.0e-6f, 1.0f, valid_grid, valid_block);
    malformed("null-k", q.data(), nullptr, weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
              1.0e-6f, 1.0f, valid_grid, valid_block);
    malformed("null-weight", q.data(), k.data(), nullptr, device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
              1.0e-6f, 1.0f, valid_grid, valid_block);
    malformed("null-positions", q.data(), k.data(), weight.get(), nullptr,
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
              1.0e-6f, 1.0f, valid_grid, valid_block);
    malformed("null-inv-freq", q.data(), k.data(), weight.get(), device_positions.get(),
              nullptr, tokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim, 1.0e-6f, 1.0f,
              valid_grid, valid_block);
    malformed("q-k-alias", q.data(), q.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
              1.0e-6f, 1.0f, valid_grid, valid_block);
    malformed("zero-tokens", q.data(), k.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), 0, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
              1.0e-6f, 1.0f, valid_grid, valid_block);
    malformed("q-heads", q.data(), k.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq - 1, kNkv, kHeadDim, kNopeDim,
              kRopeDim, 1.0e-6f, 1.0f, valid_grid, valid_block);
    malformed("kv-heads", q.data(), k.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv + 1, kHeadDim, kNopeDim,
              kRopeDim, 1.0e-6f, 1.0f, valid_grid, valid_block);
    malformed("head-dim", q.data(), k.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim - 2, kNopeDim,
              kRopeDim, 1.0e-6f, 1.0f, valid_grid, valid_block);
    malformed("nope-dim", q.data(), k.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim, kNopeDim - 2,
              kRopeDim, 1.0e-6f, 1.0f, valid_grid, valid_block);
    malformed("rotary-dim", q.data(), k.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim, kNopeDim,
              kRopeDim - 2, 1.0e-6f, 1.0f, valid_grid, valid_block);
    malformed("eps-zero", q.data(), k.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
              0.0f, 1.0f, valid_grid, valid_block);
    malformed("eps-nan", q.data(), k.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
              nan, 1.0f, valid_grid, valid_block);
    malformed("eps-inf", q.data(), k.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
              std::numeric_limits<float>::infinity(), 1.0f, valid_grid, valid_block);
    malformed("mscale-nan", q.data(), k.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
              1.0e-6f, nan, valid_grid, valid_block);
    malformed("block-x", q.data(), k.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
              1.0e-6f, 1.0f, valid_grid, dim3(256, 1, 1));
    malformed("block-y", q.data(), k.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
              1.0e-6f, 1.0f, valid_grid, dim3(256, 2, 1));
    malformed("block-z", q.data(), k.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
              1.0e-6f, 1.0f, valid_grid, dim3(256, 1, 2));
    malformed("grid-x", q.data(), k.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
              1.0e-6f, 1.0f, dim3(tokens + 1, kNq, 1), valid_block);
    malformed("grid-y", q.data(), k.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
              1.0e-6f, 1.0f, dim3(tokens, kNq + 1, 1), valid_block);
    malformed("grid-z", q.data(), k.data(), weight.get(), device_positions.get(),
              device_frequencies.get(), tokens, kNq, kNkv, kHeadDim, kNopeDim, kRopeDim,
              1.0e-6f, 1.0f, dim3(tokens, kNq, 2), valid_block);
    if (no_write != kMalformedCases) fail("malformed ABI census");
    return no_write;
}
// END V4 Q-B malformed ABI

struct Timing {
    float baseline_ms;
    float candidate_ms;
    float speedup;
};

// BEGIN V4 Q-B ABBA timing
template <typename Setup, typename Launch>
static float event_time(cudaEvent_t start, cudaEvent_t finish, Setup setup, Launch launch) {
    setup();
    cuda_ok(cudaEventRecord(start), "cudaEventRecord(start)");
    launch();
    cuda_ok(cudaEventRecord(finish), "cudaEventRecord(finish)");
    cuda_ok(cudaEventSynchronize(finish), "cudaEventSynchronize(finish)");
    float milliseconds = 0.0f;
    cuda_ok(cudaEventElapsedTime(&milliseconds, start, finish), "cudaEventElapsedTime");
    return milliseconds;
}

static Timing run_timing() {
    const size_t q_elements = static_cast<size_t>(kProductionTokens) * kNq * kHeadDim;
    const size_t k_elements = static_cast<size_t>(kProductionTokens) * kNkv * kHeadDim;
    std::vector<__nv_bfloat16> q_host(q_elements);
    std::vector<__nv_bfloat16> k_host(k_elements);
    std::vector<__nv_bfloat16> zero_weight(kHeadDim);
    std::vector<unsigned int> positions(kProductionTokens);
    std::vector<float> frequencies(kRopePairs);
    fill_bf16(q_host, 211);
    fill_bf16(k_host, 223);
    std::memset(zero_weight.data(), 0, zero_weight.size() * 2);
    fill_tables(positions, frequencies, 17);
    DeviceBuffer<__nv_bfloat16> q_seed(q_elements);
    DeviceBuffer<__nv_bfloat16> k_seed(k_elements);
    DeviceBuffer<__nv_bfloat16> q_work(q_elements);
    DeviceBuffer<__nv_bfloat16> k_work(k_elements);
    DeviceBuffer<__nv_bfloat16> weight(kHeadDim);
    DeviceBuffer<unsigned int> device_positions(kProductionTokens);
    DeviceBuffer<float> device_frequencies(kRopePairs);
    q_seed.upload(q_host.data());
    k_seed.upload(k_host.data());
    weight.upload(zero_weight.data());
    device_positions.upload(positions.data());
    device_frequencies.upload(frequencies.data());
    auto baseline_reset = [&] {
        cuda_ok(cudaMemcpyAsync(q_work.get(), q_seed.get(), q_work.bytes(), cudaMemcpyDeviceToDevice),
                "timing reset Q baseline");
        cuda_ok(cudaMemcpyAsync(k_work.get(), k_seed.get(), k_work.bytes(), cudaMemcpyDeviceToDevice),
                "timing reset K baseline");
    };
    auto candidate_reset = [&] {
        cuda_ok(cudaMemcpyAsync(q_work.get(), q_seed.get(), q_work.bytes(), cudaMemcpyDeviceToDevice),
                "timing reset Q candidate");
        cuda_ok(cudaMemcpyAsync(k_work.get(), k_seed.get(), k_work.bytes(), cudaMemcpyDeviceToDevice),
                "timing reset K candidate");
    };
    auto baseline = [&] {
        launch_baseline(q_work.get(), k_work.get(), weight.get(), device_positions.get(),
                        device_frequencies.get(), kProductionTokens, 1.0e-6f, 1.125f);
    };
    auto candidate = [&] {
        launch_candidate(q_work.get(), k_work.get(), weight.get(), device_positions.get(),
                         device_frequencies.get(), kProductionTokens, 1.0e-6f, 1.125f);
    };
    baseline_reset();
    baseline();
    candidate_reset();
    candidate();
    cuda_ok(cudaDeviceSynchronize(), "ABBA warmup");
    cudaEvent_t start = nullptr;
    cudaEvent_t finish = nullptr;
    cuda_ok(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_ok(cudaEventCreate(&finish), "cudaEventCreate(finish)");
    float baseline_ms = 0.0f;
    float candidate_ms = 0.0f;
    // Every round is baseline, candidate, candidate, baseline; setup/reset is
    // deliberately before the start event and excluded from kernel timing.
    for (int round = 0; round < kAbbaRounds; ++round) {
        baseline_ms += event_time(start, finish, baseline_reset, baseline);
        candidate_ms += event_time(start, finish, candidate_reset, candidate);
        candidate_ms += event_time(start, finish, candidate_reset, candidate);
        baseline_ms += event_time(start, finish, baseline_reset, baseline);
    }
    cuda_ok(cudaEventDestroy(start), "cudaEventDestroy(start)");
    cuda_ok(cudaEventDestroy(finish), "cudaEventDestroy(finish)");
    baseline_ms /= 2.0f * kAbbaRounds;
    candidate_ms /= 2.0f * kAbbaRounds;
    if (!(baseline_ms > 0.0f) || !(candidate_ms > 0.0f)) fail("invalid timing");
    return {baseline_ms, candidate_ms, baseline_ms / candidate_ms};
}
// END V4 Q-B ABBA timing

static float parse_speedup_threshold(const char* value) {
    if (value == nullptr || value[0] == '\0') {
        std::fprintf(stderr, "invalid explicit numeric threshold\n");
        std::exit(2);
    }
    bool digit = false;
    bool dot = false;
    bool exponent = false;
    bool exponent_digit = false;
    for (size_t i = 0; value[i] != '\0'; ++i) {
        const unsigned char character = static_cast<unsigned char>(value[i]);
        if (std::isdigit(character)) {
            digit = true;
            if (exponent) exponent_digit = true;
        } else if (character == '.' && !dot && !exponent) {
            dot = true;
        } else if ((character == 'e' || character == 'E') && digit && !exponent) {
            exponent = true;
        } else if ((character == '+' || character == '-') && exponent && !exponent_digit &&
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
    const float min_speedup = std::strtof(value, &end);
    if (errno != 0 || end == value || *end != '\0' || !std::isfinite(min_speedup) ||
        min_speedup <= 1.0f || min_speedup > 100.0f) {
        std::fprintf(stderr, "invalid explicit numeric threshold\n");
        std::exit(2);
    }
    return min_speedup;
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
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <min_speedup>\n", argv[0]);
        return 2;
    }
    const float min_speedup = parse_speedup_threshold(argv[1]);
    if (!valid_build_id(V4_QB_NORM_ROPE_PROBE_BUILD_ID)) fail("invalid build id");

    int device = 0;
    cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    cuda_ok(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    int driver = 0;
    int runtime = 0;
    cuda_ok(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    cuda_ok(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");

    Evidence evidence;
    for (unsigned int case_index = 0; case_index < kParityCases; ++case_index) {
        run_parity_case(kTokenCases[case_index], case_index, evidence);
    }
    if (evidence.q_mismatch_bytes != 0 || evidence.k_mismatch_bytes != 0) {
        fail("byte parity");
    }
    const int no_write = run_malformed_cases();
    const Timing timing = run_timing();
    if (!std::isfinite(timing.speedup) || timing.speedup < min_speedup) {
        fail("pre-registered speedup threshold not met");
    }

    char uuid[33];
    for (int i = 0; i < 16; ++i) {
        std::sprintf(uuid + 2 * i, "%02x", static_cast<unsigned char>(properties.uuid.bytes[i]));
    }
    uuid[32] = '\0';
    std::printf("build_id=%s\n", V4_QB_NORM_ROPE_PROBE_BUILD_ID);
    std::printf("device_uuid=%s driver=%d runtime=%d\n", uuid, driver, runtime);
    std::printf("input_hash=%016llx position_hash=%016llx frequency_hash=%016llx weight_hash=%016llx\n",
                static_cast<unsigned long long>(evidence.input_hash),
                static_cast<unsigned long long>(evidence.position_hash),
                static_cast<unsigned long long>(evidence.frequency_hash),
                static_cast<unsigned long long>(evidence.weight_hash));
    std::printf("parity token_cases=4 exact_tokens=2410 q_mismatch_bytes=0 k_mismatch_bytes=0\n");
    std::printf("guards malformed_cases=22 no_write=%d prefix=clean suffix=clean\n", no_write);
    std::printf("timing baseline_ms=%.6f candidate_ms=%.6f speedup=%.9f abba_rounds=6\n",
                timing.baseline_ms, timing.candidate_ms, timing.speedup);
    std::printf("threshold min_speedup=%s\n", argv[1]);
    std::printf("result=PASS\n");
    return 0;
}

}  // namespace probe

int main(int argc, char** argv) { return probe::run(argc, argv); }
