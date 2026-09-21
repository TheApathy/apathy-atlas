// SPDX-License-Identifier: AGPL-3.0-only
//
// Standalone GB10 promotion probe for the compile-only N128 fused W2A8
// gate/up -> H128/SwiGLU/down-H128 -> A8 experiment. This translation unit is
// intentionally outside every model registry and serving build.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef W2A8_FUSED_GU_PROBE_BUILD_ID
#error "W2A8_FUSED_GU_PROBE_BUILD_ID must be supplied by the receipt-producing build"
#endif
#ifndef W2A8_FUSED_GU_PROBE_MIN_SPEEDUP
#error "W2A8_FUSED_GU_PROBE_MIN_SPEEDUP must be supplied by the receipt-producing build"
#endif
#ifndef W2A8_FUSED_GU_PROBE_MIN_SPEEDUP_TEXT
#error "W2A8_FUSED_GU_PROBE_MIN_SPEEDUP_TEXT must be supplied by the receipt-producing build"
#endif

// BEGIN fused probe kernel contract
#define W2A8_FIXED_N 2048
#define W2A8_FIXED_K 4096
#define W2A8_KERNEL_NAME exl3_w2a8_grouped_prefill_n128_gu
#include "exl3_w2a8_grouped_prefill_n128.cu"
#undef W2A8_KERNEL_NAME

#define W2A8_KERNEL_NAME exl3_w2a8_fused_gu_down_emit_n128
#include "exl3_w2a8_fused_gu_down_emit_n128.cu"
#undef W2A8_KERNEL_NAME

#include "exl3_w2a8_h128_emit.cu"
// The incumbent comparison is exactly three launches: one N128 gate grouped
// kernel, one N128 up grouped kernel, then the established H128/SwiGLU/down-A8
// emitter. The candidate is one fused launch over the same N128 expert tiles.
// END fused probe kernel contract

namespace probe {

constexpr unsigned kN = 2048;
constexpr unsigned kK = 4096;
constexpr unsigned kNTile = 128;
constexpr unsigned kThreads = 256;
constexpr unsigned kProductionRows = 14460;
constexpr unsigned kMaxExperts = 256;
constexpr unsigned kExpertsWith57Rows = 124;
constexpr unsigned kExpertsWith56Rows = 132;
static_assert(kExpertsWith57Rows + kExpertsWith56Rows == kMaxExperts);
static_assert(
    kExpertsWith57Rows * 57 + kExpertsWith56Rows * 56 == kProductionRows);
constexpr std::size_t kGuardBytes = 256;
constexpr unsigned char kGuard = 0xcd;
constexpr unsigned char kPoisonA = 0xa5;
constexpr unsigned char kPoisonB = 0x5a;
constexpr int kWarmups = 2;
constexpr int kTimingIterations = 8;
constexpr double kMinSpeedup = W2A8_FUSED_GU_PROBE_MIN_SPEEDUP;
static_assert(kMinSpeedup > 1.0 && kMinSpeedup <= 100.0);

[[noreturn]] void fail(const char* check_name) {
    std::fprintf(stderr, "FAIL check=%s\n", check_name);
    std::exit(1);
}

[[noreturn]] void cuda_fail(const char* operation, cudaError_t error) {
    std::fprintf(
        stderr, "FAIL cuda=%s error=%s\n", operation,
        cudaGetErrorString(error));
    std::exit(2);
}

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) cuda_fail(operation, error);
}

struct Buffer {
    unsigned char* base = nullptr;
    unsigned char* data = nullptr;
    std::size_t bytes = 0;

    explicit Buffer(std::size_t size) : bytes(size) {
        check(cudaMalloc(&base, bytes + 2 * kGuardBytes), "cudaMalloc");
        data = base + kGuardBytes;
        check(
            cudaMemset(base, kGuard, bytes + 2 * kGuardBytes),
            "guard initialization");
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() {
        if (base != nullptr) cudaFree(base);
    }

    template <typename T>
    void upload(const std::vector<T>& values) {
        if (values.size() * sizeof(T) != bytes) fail("upload-size");
        check(
            cudaMemcpy(data, values.data(), bytes, cudaMemcpyHostToDevice),
            "upload");
    }

    void fill(unsigned char value) {
        check(cudaMemset(data, value, bytes), "buffer fill");
    }

    std::vector<unsigned char> download() const {
        std::vector<unsigned char> result(bytes);
        check(
            cudaMemcpy(
                result.data(), data, result.size(), cudaMemcpyDeviceToHost),
            "download");
        return result;
    }

    bool guards_clean() const {
        std::vector<unsigned char> guards(2 * kGuardBytes);
        check(
            cudaMemcpy(
                guards.data(), base, kGuardBytes, cudaMemcpyDeviceToHost),
            "prefix guard download");
        check(
            cudaMemcpy(
                guards.data() + kGuardBytes, data + bytes, kGuardBytes,
                cudaMemcpyDeviceToHost),
            "suffix guard download");
        for (unsigned char value : guards) {
            if (value != kGuard) return false;
        }
        return true;
    }
};

std::uint64_t next_random(std::uint64_t& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 0x2545f4914f6cdd1dull;
}

std::uint64_t hash_bytes(
    std::uint64_t hash, const void* pointer, std::size_t bytes) {
    const auto* values = static_cast<const unsigned char*>(pointer);
    for (std::size_t index = 0; index < bytes; ++index) {
        hash ^= values[index];
        hash *= 0x100000001b3ull;
    }
    return hash;
}

template <typename T>
std::uint64_t hash_vector(
    std::uint64_t hash, const std::vector<T>& values) {
    return hash_bytes(hash, values.data(), values.size() * sizeof(T));
}

bool valid_build_id(const char* value) {
    if (std::strlen(value) != 64) return false;
    for (unsigned index = 0; index < 64; ++index) {
        const char character = value[index];
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

// Keep every synthetic activation byte in finite E4M3 space. Signed zeros,
// subnormals, normal signs/exponents, and the +/-448 finite endpoints occur.
std::vector<unsigned char> make_activation(unsigned rows, unsigned salt) {
    std::uint64_t state = 0x9e3779b97f4a7c15ull ^ salt;
    std::vector<unsigned char> values(
        static_cast<std::size_t>(rows) * kK);
    constexpr unsigned char edges[] = {
        0x00, 0x80, 0x01, 0x81, 0x08, 0x88, 0x38, 0xb8, 0x7e, 0xfe,
    };
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index % 257 < sizeof(edges)) {
            values[index] = edges[index % 257];
        } else {
            unsigned char value =
                static_cast<unsigned char>(next_random(state) >> 56);
            if ((value & 0x7f) > 0x7e) value = (value & 0x80) | 0x7e;
            values[index] = value;
        }
    }
    return values;
}

std::vector<float> make_scales(unsigned rows, unsigned salt) {
    std::vector<float> values(
        static_cast<std::size_t>(rows) * (kK / 128));
    for (std::size_t index = 0; index < values.size(); ++index) {
        const int exponent = static_cast<int>((index * 7 + salt) % 17) - 14;
        values[index] = std::ldexp(1.0f, exponent);
    }
    return values;
}

std::vector<std::uint16_t> make_trellis(unsigned salt) {
    const std::size_t values_per_expert =
        static_cast<std::size_t>(kK / 16) * (kN / 16) * 32;
    std::uint64_t state = 0xd1b54a32d192ed03ull ^ salt;
    std::vector<std::uint16_t> values(
        static_cast<std::size_t>(kMaxExperts) * values_per_expert);
    for (std::uint16_t& value : values) {
        value = static_cast<std::uint16_t>(next_random(state) >> 48);
    }
    return values;
}

// BEGIN fused GU nondegenerate signs
// Each sign table uses an independent table_salt. Expert and H128 chunk both
// participate so every gate_svh, up_svh, and down_suh table is nondegenerate.
std::vector<std::uint16_t> make_signs(unsigned table_salt) {
    std::vector<std::uint16_t> values(
        static_cast<std::size_t>(kMaxExperts) * kN);
    for (unsigned expert = 0; expert < kMaxExperts; ++expert) {
        for (unsigned index = 0; index < kN; ++index) {
            const unsigned chunk = index / 128;
            values[static_cast<std::size_t>(expert) * kN + index] =
                (((index >> table_salt) ^ expert ^ chunk ^ table_salt) & 1u) != 0
                    ? 0xbc00
                    : 0x3c00;
        }
    }
    return values;
}

void validate_sign_table(const std::vector<std::uint16_t>& values) {
    for (unsigned expert = 0; expert < kMaxExperts; ++expert) {
        for (unsigned chunk = 0; chunk < kN / 128; ++chunk) {
            bool positive = false;
            bool negative = false;
            const std::size_t begin =
                static_cast<std::size_t>(expert) * kN + chunk * 128;
            for (unsigned index = 0; index < 128; ++index) {
                positive |= values[begin + index] == 0x3c00;
                negative |= values[begin + index] == 0xbc00;
            }
            if (!positive || !negative) fail("degenerate-sign-chunk");
        }
    }
}
// END fused GU nondegenerate signs

std::vector<unsigned long long> pointer_table(
    const Buffer& storage, std::size_t bytes_per_expert) {
    std::vector<unsigned long long> pointers(kMaxExperts);
    for (unsigned expert = 0; expert < kMaxExperts; ++expert) {
        pointers[expert] = static_cast<unsigned long long>(
            reinterpret_cast<std::uintptr_t>(storage.data) +
            expert * bytes_per_expert);
    }
    return pointers;
}

// BEGIN fused GU routing cases
// `expert_offsets` define the sorted expert-major rows consumed by both paths;
// `sorted_expert_ids` is derived from them for the incumbent emitter. Repeated
// (duplicate) offsets are intentional `empty_expert` cases. Negative and
// `out_of_range` offsets are malformed canaries, never fallback routes.
struct RowCase {
    const char* label;
    std::vector<int> offsets;
};

std::vector<RowCase> row_cases() {
    return {
        {"single-row", {0, 1}},
        {"before-warp", {0, 63}},
        {"warp", {0, 64}},
        {"after-warp", {0, 65}},
        {"before-tile", {0, 127}},
        {"tile", {0, 128}},
        {"after-tile", {0, 129}},
        {"interior-empty", {0, 0, 63, 63, 129}},
    };
}

std::vector<int> balanced_offsets() {
    std::vector<int> offsets(kMaxExperts + 1, 0);
    for (unsigned expert = 0; expert < kMaxExperts; ++expert) {
        const int rows = expert < kExpertsWith57Rows ? 57 : 56;
        offsets[expert + 1] = offsets[expert] + rows;
    }
    if (offsets.back() != static_cast<int>(kProductionRows)) {
        fail("production-offset-total");
    }
    return offsets;
}

std::vector<int> expert_ids(
    const std::vector<int>& offsets, unsigned rows_capacity) {
    if (offsets.size() < 2 || offsets.front() != 0 ||
        offsets.back() < 0 ||
        offsets.back() > static_cast<int>(rows_capacity)) {
        fail("host-offset-shape");
    }
    std::vector<int> result(rows_capacity, -1);
    for (std::size_t expert = 0; expert + 1 < offsets.size(); ++expert) {
        if (offsets[expert] > offsets[expert + 1]) fail("host-offset-order");
        for (int row = offsets[expert]; row < offsets[expert + 1]; ++row) {
            result[row] = static_cast<int>(expert);
        }
    }
    return result;
}
// END fused GU routing cases

struct Fixture {
    unsigned rows_capacity;
    std::vector<unsigned char> gate_activation_h;
    std::vector<unsigned char> up_activation_h;
    std::vector<float> gate_scale_h;
    std::vector<float> up_scale_h;
    std::vector<std::uint16_t> gate_trellis_h;
    std::vector<std::uint16_t> up_trellis_h;
    std::vector<std::uint16_t> gate_svh_h;
    std::vector<std::uint16_t> up_svh_h;
    std::vector<std::uint16_t> down_suh_h;

    Buffer gate_activation;
    Buffer up_activation;
    Buffer gate_scale;
    Buffer up_scale;
    Buffer gate_trellis;
    Buffer up_trellis;
    Buffer gate_trellis_tab;
    Buffer up_trellis_tab;
    Buffer gate_svh;
    Buffer up_svh;
    Buffer down_suh;
    Buffer gate_svh_tab;
    Buffer up_svh_tab;
    Buffer down_suh_tab;
    Buffer offsets;
    Buffer experts;
    Buffer gate_bf16;
    Buffer up_bf16;
    Buffer reference_fp8;
    Buffer reference_scale;
    Buffer candidate_fp8;
    Buffer candidate_scale;

    explicit Fixture(unsigned rows)
        : rows_capacity(rows),
          gate_activation_h(make_activation(rows, 11)),
          up_activation_h(make_activation(rows, 73)),
          gate_scale_h(make_scales(rows, 3)),
          up_scale_h(make_scales(rows, 9)),
          gate_trellis_h(make_trellis(17)),
          up_trellis_h(make_trellis(91)),
          gate_svh_h(make_signs(0)),
          up_svh_h(make_signs(1)),
          down_suh_h(make_signs(2)),
          gate_activation(gate_activation_h.size()),
          up_activation(up_activation_h.size()),
          gate_scale(gate_scale_h.size() * sizeof(float)),
          up_scale(up_scale_h.size() * sizeof(float)),
          gate_trellis(gate_trellis_h.size() * sizeof(std::uint16_t)),
          up_trellis(up_trellis_h.size() * sizeof(std::uint16_t)),
          gate_trellis_tab(kMaxExperts * sizeof(unsigned long long)),
          up_trellis_tab(kMaxExperts * sizeof(unsigned long long)),
          gate_svh(gate_svh_h.size() * sizeof(std::uint16_t)),
          up_svh(up_svh_h.size() * sizeof(std::uint16_t)),
          down_suh(down_suh_h.size() * sizeof(std::uint16_t)),
          gate_svh_tab(kMaxExperts * sizeof(unsigned long long)),
          up_svh_tab(kMaxExperts * sizeof(unsigned long long)),
          down_suh_tab(kMaxExperts * sizeof(unsigned long long)),
          offsets((kMaxExperts + 1) * sizeof(int)),
          experts(static_cast<std::size_t>(rows) * sizeof(int)),
          gate_bf16(static_cast<std::size_t>(rows) * kN * 2),
          up_bf16(static_cast<std::size_t>(rows) * kN * 2),
          reference_fp8(static_cast<std::size_t>(rows) * kN),
          reference_scale(
              static_cast<std::size_t>(rows) * (kN / 128) * sizeof(float)),
          candidate_fp8(static_cast<std::size_t>(rows) * kN),
          candidate_scale(
              static_cast<std::size_t>(rows) * (kN / 128) * sizeof(float)) {
        validate_sign_table(gate_svh_h);
        validate_sign_table(up_svh_h);
        validate_sign_table(down_suh_h);
        if (gate_svh_h == up_svh_h || gate_svh_h == down_suh_h ||
            up_svh_h == down_suh_h) {
            fail("duplicate-sign-tables");
        }
        gate_activation.upload(gate_activation_h);
        up_activation.upload(up_activation_h);
        gate_scale.upload(gate_scale_h);
        up_scale.upload(up_scale_h);
        gate_trellis.upload(gate_trellis_h);
        up_trellis.upload(up_trellis_h);
        gate_svh.upload(gate_svh_h);
        up_svh.upload(up_svh_h);
        down_suh.upload(down_suh_h);

        const std::size_t trellis_bytes_per_expert =
            gate_trellis.bytes / kMaxExperts;
        const std::size_t sign_bytes_per_expert = kN * sizeof(std::uint16_t);
        gate_trellis_tab.upload(
            pointer_table(gate_trellis, trellis_bytes_per_expert));
        up_trellis_tab.upload(
            pointer_table(up_trellis, trellis_bytes_per_expert));
        gate_svh_tab.upload(pointer_table(gate_svh, sign_bytes_per_expert));
        up_svh_tab.upload(pointer_table(up_svh, sign_bytes_per_expert));
        down_suh_tab.upload(pointer_table(down_suh, sign_bytes_per_expert));
    }

    void upload_routing(const std::vector<int>& case_offsets) {
        std::vector<int> padded_offsets(kMaxExperts + 1, case_offsets.back());
        std::memcpy(
            padded_offsets.data(), case_offsets.data(),
            case_offsets.size() * sizeof(int));
        offsets.upload(padded_offsets);
        experts.upload(expert_ids(case_offsets, rows_capacity));
    }
};

void launch_reference_leg(
    const unsigned char* activation, const float* scale,
    const unsigned long long* trellis, __nv_bfloat16* output,
    const int* offsets, unsigned num_experts) {
    exl3_w2a8_grouped_prefill_n128_gu<<<
        dim3(num_experts * (kN / kNTile), 1, 1),
        dim3(kThreads, 1, 1)>>>(
        activation, scale, trellis, output, offsets, num_experts, kN, kK, 2,
        1);
    check(cudaGetLastError(), "reference grouped launch");
}

// Exactly three launches form the incumbent operator boundary.
void launch_reference(Fixture& fixture, unsigned rows, unsigned num_experts) {
    launch_reference_leg(
        fixture.gate_activation.data,
        reinterpret_cast<float*>(fixture.gate_scale.data),
        reinterpret_cast<unsigned long long*>(fixture.gate_trellis_tab.data),
        reinterpret_cast<__nv_bfloat16*>(fixture.gate_bf16.data),
        reinterpret_cast<int*>(fixture.offsets.data), num_experts);
    launch_reference_leg(
        fixture.up_activation.data,
        reinterpret_cast<float*>(fixture.up_scale.data),
        reinterpret_cast<unsigned long long*>(fixture.up_trellis_tab.data),
        reinterpret_cast<__nv_bfloat16*>(fixture.up_bf16.data),
        reinterpret_cast<int*>(fixture.offsets.data), num_experts);
    exl3_w2a8_h128_post_silu_pre_emit_h2048<<<
        dim3(rows, 2, 1), dim3(256, 1, 1)>>>(
        reinterpret_cast<__nv_bfloat16*>(fixture.gate_bf16.data),
        reinterpret_cast<__nv_bfloat16*>(fixture.up_bf16.data),
        reinterpret_cast<int*>(fixture.experts.data),
        reinterpret_cast<unsigned long long*>(fixture.gate_svh_tab.data),
        reinterpret_cast<unsigned long long*>(fixture.up_svh_tab.data),
        reinterpret_cast<unsigned long long*>(fixture.down_suh_tab.data),
        fixture.reference_fp8.data,
        reinterpret_cast<float*>(fixture.reference_scale.data), kN, rows);
    check(cudaGetLastError(), "reference emitter launch");
}

void launch_candidate_raw(
    Fixture& fixture, const unsigned char* gate_activation,
    const float* gate_scale, const unsigned char* up_activation,
    const float* up_scale, const unsigned long long* gate_trellis,
    const unsigned long long* up_trellis,
    const unsigned long long* gate_svh,
    const unsigned long long* up_svh,
    const unsigned long long* down_suh, const int* offsets,
    unsigned num_experts, unsigned total_rows, unsigned n, unsigned k,
    unsigned bits, unsigned persistent_mode, dim3 grid, dim3 block) {
    exl3_w2a8_fused_gu_down_emit_n128<<<grid, block>>>(
        gate_activation, gate_scale, up_activation, up_scale, gate_trellis,
        up_trellis, gate_svh, up_svh, down_suh,
        fixture.candidate_fp8.data,
        reinterpret_cast<float*>(fixture.candidate_scale.data), offsets,
        num_experts, total_rows, n, k, bits, persistent_mode);
    check(cudaGetLastError(), "fused candidate launch");
}

void launch_candidate(Fixture& fixture, unsigned total_rows) {
    launch_candidate_raw(
        fixture, fixture.gate_activation.data,
        reinterpret_cast<float*>(fixture.gate_scale.data),
        fixture.up_activation.data,
        reinterpret_cast<float*>(fixture.up_scale.data),
        reinterpret_cast<unsigned long long*>(fixture.gate_trellis_tab.data),
        reinterpret_cast<unsigned long long*>(fixture.up_trellis_tab.data),
        reinterpret_cast<unsigned long long*>(fixture.gate_svh_tab.data),
        reinterpret_cast<unsigned long long*>(fixture.up_svh_tab.data),
        reinterpret_cast<unsigned long long*>(fixture.down_suh_tab.data),
        reinterpret_cast<int*>(fixture.offsets.data), kMaxExperts, total_rows,
        kN, kK, 2, 1, dim3(kMaxExperts * (kN / kNTile), 1, 1),
        dim3(kThreads, 1, 1));
}

bool all_value(
    const std::vector<unsigned char>& bytes, std::size_t begin,
    unsigned char value) {
    for (std::size_t index = begin; index < bytes.size(); ++index) {
        if (bytes[index] != value) return false;
    }
    return true;
}

void verify_guards_and_tail(
    const Buffer& buffer, std::size_t active_bytes, unsigned char poison,
    const char* check_name) {
    if (!buffer.guards_clean()) fail(check_name);
    const auto bytes = buffer.download();
    if (!all_value(bytes, active_bytes, poison)) fail(check_name);
}

// BEGIN fused GU exact byte contract
struct PoisonPair {
    unsigned char reference_poison;
    unsigned char candidate_poison;
};

constexpr PoisonPair kPoisonPairs[] = {
    {kPoisonA, kPoisonB},
    {kPoisonB, kPoisonA},
};

std::size_t byte_mismatches(
    const Buffer& left, const Buffer& right, std::size_t bytes) {
    const auto left_h = left.download();
    const auto right_h = right.download();
    if (std::memcmp(left_h.data(), right_h.data(), bytes) == 0) return 0;
    std::size_t mismatches = 0;
    for (std::size_t index = 0; index < bytes; ++index) {
        mismatches += left_h[index] != right_h[index];
    }
    return mismatches;
}

struct ParityResult {
    std::size_t fp8_mismatches = 0;
    std::size_t scale_mismatches = 0;
    unsigned cases = 0;
    std::uint64_t routing_hash = 0xcbf29ce484222325ull;
};

std::vector<unsigned char> active_prefix(
    const Buffer& buffer, std::size_t active_bytes) {
    auto bytes = buffer.download();
    if (active_bytes == 0 || active_bytes > bytes.size()) {
        fail("invalid-active-extent");
    }
    bytes.resize(active_bytes);
    return bytes;
}

void verify_active_stable(
    const std::vector<unsigned char>& first,
    const std::vector<unsigned char>& second, const char* check_name) {
    if (first.size() != second.size() ||
        std::memcmp(first.data(), second.data(), first.size()) != 0) {
        fail(check_name);
    }
}

void verify_scale_values(const Buffer& scale_buffer, std::size_t rows) {
    const std::size_t scale_count = rows * (kN / 128);
    const auto scale_bytes = scale_buffer.download();
    const auto* scales = reinterpret_cast<const float*>(scale_bytes.data());
    for (std::size_t index = 0; index < scale_count; ++index) {
        if (!(scales[index] >= 1.0e-12f) || !std::isfinite(scales[index])) {
            fail("invalid-output-scale");
        }
    }
}

void run_parity_case(
    Fixture& fixture, ParityResult& result, const std::vector<int>& offsets) {
    const unsigned rows = static_cast<unsigned>(offsets.back());
    const unsigned experts = static_cast<unsigned>(offsets.size() - 1);
    fixture.upload_routing(offsets);
    result.routing_hash = hash_vector(result.routing_hash, offsets);
    result.routing_hash = hash_vector(
        result.routing_hash, expert_ids(offsets, fixture.rows_capacity));

    std::vector<unsigned char> reference_fp8_first;
    std::vector<unsigned char> candidate_fp8_first;
    std::vector<unsigned char> reference_scale_first;
    std::vector<unsigned char> candidate_scale_first;
    for (unsigned pass = 0; pass < 2; ++pass) {
        const PoisonPair poisons = kPoisonPairs[pass];
        fixture.gate_bf16.fill(poisons.reference_poison);
        fixture.up_bf16.fill(poisons.reference_poison);
        fixture.reference_fp8.fill(poisons.reference_poison);
        fixture.reference_scale.fill(poisons.reference_poison);
        fixture.candidate_fp8.fill(poisons.candidate_poison);
        fixture.candidate_scale.fill(poisons.candidate_poison);

        launch_reference(fixture, rows, experts);
        launch_candidate(fixture, rows);
        check(cudaDeviceSynchronize(), "parity synchronize");

        const std::size_t down_fp8_bytes =
            static_cast<std::size_t>(rows) * kN;
        const std::size_t down_scale_bytes =
            static_cast<std::size_t>(rows) * (kN / 128) * sizeof(float);
        result.fp8_mismatches += byte_mismatches(
            fixture.reference_fp8, fixture.candidate_fp8, down_fp8_bytes);
        result.scale_mismatches += byte_mismatches(
            fixture.reference_scale, fixture.candidate_scale,
            down_scale_bytes);
        ++result.cases;

        const auto reference_fp8 =
            active_prefix(fixture.reference_fp8, down_fp8_bytes);
        const auto candidate_fp8 =
            active_prefix(fixture.candidate_fp8, down_fp8_bytes);
        const auto reference_scale =
            active_prefix(fixture.reference_scale, down_scale_bytes);
        const auto candidate_scale =
            active_prefix(fixture.candidate_scale, down_scale_bytes);
        if (pass == 0) {
            reference_fp8_first = reference_fp8;
            candidate_fp8_first = candidate_fp8;
            reference_scale_first = reference_scale;
            candidate_scale_first = candidate_scale;
        } else {
            verify_active_stable(
                reference_fp8_first, reference_fp8,
                "reference-fp8-active-unwritten");
            verify_active_stable(
                candidate_fp8_first, candidate_fp8,
                "candidate-fp8-active-unwritten");
            verify_active_stable(
                reference_scale_first, reference_scale,
                "reference-scale-active-unwritten");
            verify_active_stable(
                candidate_scale_first, candidate_scale,
                "candidate-scale-active-unwritten");
        }
        verify_scale_values(fixture.reference_scale, rows);
        verify_scale_values(fixture.candidate_scale, rows);
        verify_guards_and_tail(
            fixture.reference_fp8, down_fp8_bytes,
            poisons.reference_poison, "reference-fp8-tail-or-guard");
        verify_guards_and_tail(
            fixture.candidate_fp8, down_fp8_bytes,
            poisons.candidate_poison, "candidate-fp8-tail-or-guard");
        verify_guards_and_tail(
            fixture.reference_scale, down_scale_bytes,
            poisons.reference_poison, "reference-scale-tail-or-guard");
        verify_guards_and_tail(
            fixture.candidate_scale, down_scale_bytes,
            poisons.candidate_poison, "candidate-scale-tail-or-guard");
        verify_guards_and_tail(
            fixture.gate_bf16, down_fp8_bytes * 2,
            poisons.reference_poison,
            "gate-intermediate-tail-or-guard");
        verify_guards_and_tail(
            fixture.up_bf16, down_fp8_bytes * 2,
            poisons.reference_poison,
            "up-intermediate-tail-or-guard");
    }
}

void run_boundary_parity(Fixture& fixture, ParityResult& result) {
    for (const RowCase& row_case : row_cases()) {
        run_parity_case(fixture, result, row_case.offsets);
    }
}

// This full-row route is deliberately synthetic and balanced. It proves exact
// layout parity but is unrepresentative of production routing skew.
void run_synthetic_balanced_parity(Fixture& fixture, ParityResult& result) {
    const std::vector<int> balanced = balanced_offsets();
    constexpr unsigned parity_rows = kProductionRows;
    if (balanced.back() != static_cast<int>(parity_rows)) {
        fail("production-parity-rows");
    }
    run_parity_case(fixture, result, balanced);
}
// END fused GU exact byte contract

// BEGIN fused GU guard and poison contract
// Every valid case starts reference/candidate outputs with opposite 0xa5/0x5a
// values, then swaps them (`poison_independent`); incumbent BF16 scratch follows
// the reference poison. `prefix_guard` and `suffix_guard` remain 0xcd, and
// `inactive_rows` stay poisoned. Malformed canaries also run under both poisons.
void verify_fully_poisoned(
    const Buffer& buffer, unsigned char poison, const char* name) {
    if (!buffer.guards_clean() || !all_value(buffer.download(), 0, poison)) {
        fail(name);
    }
}

template <typename Launch>
void invalid_canary(Fixture& fixture, const char* name, Launch launch) {
    for (unsigned char poison : {kPoisonA, kPoisonB}) {
        fixture.candidate_fp8.fill(poison);
        fixture.candidate_scale.fill(poison);
        launch();
        check(cudaGetLastError(), name);
        check(cudaDeviceSynchronize(), name);
        verify_fully_poisoned(fixture.candidate_fp8, poison, name);
        verify_fully_poisoned(fixture.candidate_scale, poison, name);
    }
}

void run_malformed_canaries(Fixture& fixture) {
    const unsigned total_rows = fixture.rows_capacity;
    const std::vector<int> offsets = {0, static_cast<int>(total_rows)};
    fixture.upload_routing(offsets);
    const dim3 valid_grid(kMaxExperts * (kN / kNTile), 1, 1);
    const dim3 valid_block(kThreads, 1, 1);
    auto raw = [&](unsigned total_rows, unsigned n, unsigned k, unsigned bits,
                   unsigned persistent, unsigned experts, dim3 grid,
                   dim3 block, bool null_inputs) {
        launch_candidate_raw(
            fixture,
            null_inputs ? nullptr : fixture.gate_activation.data,
            null_inputs ? nullptr
                        : reinterpret_cast<float*>(fixture.gate_scale.data),
            null_inputs ? nullptr : fixture.up_activation.data,
            null_inputs ? nullptr
                        : reinterpret_cast<float*>(fixture.up_scale.data),
            null_inputs
                ? nullptr
                : reinterpret_cast<unsigned long long*>(
                      fixture.gate_trellis_tab.data),
            null_inputs
                ? nullptr
                : reinterpret_cast<unsigned long long*>(
                      fixture.up_trellis_tab.data),
            null_inputs
                ? nullptr
                : reinterpret_cast<unsigned long long*>(
                      fixture.gate_svh_tab.data),
            null_inputs
                ? nullptr
                : reinterpret_cast<unsigned long long*>(fixture.up_svh_tab.data),
            null_inputs
                ? nullptr
                : reinterpret_cast<unsigned long long*>(fixture.down_suh_tab.data),
            null_inputs ? nullptr : reinterpret_cast<int*>(fixture.offsets.data),
            experts, total_rows, n, k, bits, persistent, grid, block);
    };

    invalid_canary(fixture, "wrong-block-x", [&] {
        raw(total_rows, kN, kK, 2, 1, kMaxExperts, valid_grid,
            dim3(128, 1, 1), false);
    });
    invalid_canary(fixture, "wrong-block-y", [&] {
        raw(total_rows, kN, kK, 2, 1, kMaxExperts, valid_grid,
            dim3(256, 2, 1), false);
    });
    invalid_canary(fixture, "wrong-block-z", [&] {
        raw(total_rows, kN, kK, 2, 1, kMaxExperts, valid_grid,
            dim3(256, 1, 2), false);
    });
    invalid_canary(fixture, "wrong-grid-x", [&] {
        raw(total_rows, kN, kK, 2, 1, kMaxExperts,
            dim3(valid_grid.x + 1, 1, 1), valid_block, false);
    });
    invalid_canary(fixture, "wrong-grid-y", [&] {
        raw(total_rows, kN, kK, 2, 1, kMaxExperts,
            dim3(valid_grid.x, 2, 1), valid_block, false);
    });
    invalid_canary(fixture, "wrong-grid-z", [&] {
        raw(total_rows, kN, kK, 2, 1, kMaxExperts,
            dim3(valid_grid.x, 1, 2), valid_block, false);
    });
    invalid_canary(fixture, "wrong-n", [&] {
        raw(total_rows, kN - 1, kK, 2, 1, kMaxExperts, valid_grid,
            valid_block, false);
    });
    invalid_canary(fixture, "wrong-k", [&] {
        raw(total_rows, kN, kK - 1, 2, 1, kMaxExperts, valid_grid,
            valid_block, false);
    });
    invalid_canary(fixture, "wrong-bits", [&] {
        raw(total_rows, kN, kK, 3, 1, kMaxExperts, valid_grid, valid_block,
            false);
    });
    invalid_canary(fixture, "wrong-runtime-persistent", [&] {
        raw(total_rows, kN, kK, 2, 0, kMaxExperts, valid_grid, valid_block,
            false);
    });
    invalid_canary(fixture, "zero-experts", [&] {
        raw(total_rows, kN, kK, 2, 1, 0, valid_grid, valid_block, false);
    });
    invalid_canary(fixture, "null-inputs", [&] {
        raw(total_rows, kN, kK, 2, 1, kMaxExperts, valid_grid, valid_block,
            true);
    });

    std::vector<int> padded(kMaxExperts + 1, static_cast<int>(total_rows));
    padded[0] = 0;
    padded[1] = -1;
    fixture.offsets.upload(padded);
    invalid_canary(fixture, "out-of-range-offset", [&] {
        raw(total_rows, kN, kK, 2, 1, kMaxExperts, valid_grid, valid_block,
            false);
    });
    padded[0] = 0;
    padded[1] = static_cast<int>(fixture.rows_capacity + 1);
    padded[2] = static_cast<int>(total_rows);
    fixture.offsets.upload(padded);
    invalid_canary(fixture, "oversized-positive-offset", [&] {
        raw(total_rows, kN, kK, 2, 1, kMaxExperts, valid_grid, valid_block,
            false);
    });
    padded.assign(kMaxExperts + 1, static_cast<int>(total_rows));
    padded[0] = 0;
    padded[1] = static_cast<int>(fixture.rows_capacity + 1);
    padded[2] = 1;
    padded[3] = 2;
    fixture.offsets.upload(padded);
    invalid_canary(fixture, "nonmonotonic-offset", [&] {
        raw(total_rows, kN, kK, 2, 1, kMaxExperts, valid_grid, valid_block,
            false);
    });
    padded.assign(kMaxExperts + 1, static_cast<int>(total_rows));
    padded[0] = 1;
    fixture.offsets.upload(padded);
    invalid_canary(fixture, "prefix-gap-offset", [&] {
        raw(total_rows, kN, kK, 2, 1, kMaxExperts, valid_grid, valid_block,
            false);
    });
    padded.assign(kMaxExperts + 1, static_cast<int>(total_rows));
    padded[0] = 0;
    padded[kMaxExperts] = static_cast<int>(total_rows - 1);
    fixture.offsets.upload(padded);
    invalid_canary(fixture, "final-gap-offset", [&] {
        raw(total_rows, kN, kK, 2, 1, kMaxExperts, valid_grid, valid_block,
            false);
    });

    padded = balanced_offsets();
    padded[kMaxExperts - 8] = padded[kMaxExperts - 9] - 1;
    fixture.offsets.upload(padded);
    invalid_canary(fixture, "late-nonmonotonic-offset", [&] {
        raw(total_rows, kN, kK, 2, 1, kMaxExperts, valid_grid, valid_block,
            false);
    });
    padded = balanced_offsets();
    padded[kMaxExperts - 4] = static_cast<int>(total_rows + 1);
    fixture.offsets.upload(padded);
    invalid_canary(fixture, "late-out-of-range-offset", [&] {
        raw(total_rows, kN, kK, 2, 1, kMaxExperts, valid_grid, valid_block,
            false);
    });
    padded = balanced_offsets();
    fixture.offsets.upload(padded);
    invalid_canary(fixture, "too-many-experts", [&] {
        raw(total_rows, kN, kK, 2, 1, kMaxExperts + 1,
            dim3((kMaxExperts + 1) * (kN / kNTile), 1, 1), valid_block,
            false);
    });
}
// END fused GU guard and poison contract

// BEGIN fused GU ABBA timing contract
template <typename Launch>
float time_launch(Launch launch) {
    cudaEvent_t start = nullptr;
    cudaEvent_t finish = nullptr;
    check(cudaEventCreate(&start), "timing start event");
    check(cudaEventCreate(&finish), "timing finish event");
    check(cudaEventRecord(start), "timing start record");
    for (int iteration = 0; iteration < kTimingIterations; ++iteration) {
        launch();
    }
    check(cudaEventRecord(finish), "timing finish record");
    check(cudaEventSynchronize(finish), "timing finish synchronize");
    float milliseconds = 0.0f;
    check(
        cudaEventElapsedTime(&milliseconds, start, finish),
        "timing elapsed");
    check(cudaEventDestroy(start), "timing start destroy");
    check(cudaEventDestroy(finish), "timing finish destroy");
    return milliseconds / kTimingIterations;
}

struct TimingResult {
    float baseline_ms;
    float candidate_ms;
    float speedup;
};

TimingResult run_abba_timing(Fixture& fixture) {
    // Synthetic balanced routing is intentionally unrepresentative of
    // production expert skew; the receipt reports that limitation verbatim.
    const std::vector<int> offsets = balanced_offsets();
    fixture.upload_routing(offsets);
    constexpr unsigned rows = kProductionRows;
    constexpr unsigned experts = kMaxExperts;
    auto reference = [&] { launch_reference(fixture, rows, experts); };
    auto candidate = [&] { launch_candidate(fixture, rows); };
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        reference();
        candidate();
    }
    check(cudaDeviceSynchronize(), "ABBA warmup synchronize");

    const float a0 = time_launch(reference);
    const float b0 = time_launch(candidate);
    const float b1 = time_launch(candidate);
    const float a1 = time_launch(reference);
    const float baseline_ms = 0.5f * (a0 + a1);
    const float candidate_ms = 0.5f * (b0 + b1);
    if (!(baseline_ms > 0.0f) || !(candidate_ms > 0.0f) ||
        !std::isfinite(baseline_ms) || !std::isfinite(candidate_ms)) {
        fail("invalid-ABBA-timing");
    }
    return {baseline_ms, candidate_ms, baseline_ms / candidate_ms};
}
// END fused GU ABBA timing contract

std::uint64_t input_hash(const Fixture& fixture) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    hash = hash_vector(hash, fixture.gate_activation_h);
    hash = hash_vector(hash, fixture.up_activation_h);
    hash = hash_vector(hash, fixture.gate_scale_h);
    return hash_vector(hash, fixture.up_scale_h);
}

std::uint64_t tables_hash(const Fixture& fixture) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    hash = hash_vector(hash, fixture.gate_trellis_h);
    hash = hash_vector(hash, fixture.up_trellis_h);
    hash = hash_vector(hash, fixture.gate_svh_h);
    hash = hash_vector(hash, fixture.up_svh_h);
    return hash_vector(hash, fixture.down_suh_h);
}

}  // namespace probe

// BEGIN fused GU bounded output contract
int main(int argc, char** argv) {
    if (argc != 1) {
        std::fprintf(stderr, "usage: %s\n", argv[0]);
        return 2;
    }
    const char* build_id = W2A8_FUSED_GU_PROBE_BUILD_ID;
    if (!probe::valid_build_id(build_id)) {
        std::fprintf(stderr, "invalid fused probe build id\n");
        return 2;
    }

    probe::ParityResult parity;
    probe::Fixture production_fixture(probe::kProductionRows);
    probe::run_boundary_parity(production_fixture, parity);
    probe::run_malformed_canaries(production_fixture);
    probe::run_synthetic_balanced_parity(production_fixture, parity);
    if (parity.fp8_mismatches != 0 || parity.scale_mismatches != 0) {
        probe::fail("byte-parity");
    }
    const probe::TimingResult timing =
        probe::run_abba_timing(production_fixture);
    if (timing.speedup < probe::kMinSpeedup) {
        probe::fail("speedup-threshold");
    }

    int device = 0;
    int driver = 0;
    int runtime = 0;
    cudaDeviceProp properties{};
    probe::check(cudaGetDevice(&device), "cudaGetDevice");
    probe::check(
        cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties");
    probe::check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    probe::check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");

    std::printf("build_id=%s\n", build_id);
    std::printf("device_uuid=");
    for (unsigned char byte : properties.uuid.bytes) {
        std::printf("%02x", static_cast<unsigned>(byte));
    }
    std::printf(" driver=%d runtime=%d\n", driver, runtime);
    std::printf(
        "input_hash=%016llx routing_hash=%016llx tables_hash=%016llx\n",
        static_cast<unsigned long long>(probe::input_hash(production_fixture)),
        static_cast<unsigned long long>(parity.routing_hash),
        static_cast<unsigned long long>(probe::tables_hash(production_fixture)));
    std::printf(
        "rows=14460 fp8_bytes=29614080 scale_bytes=925440 "
        "experts=256 parity_rows=14460 boundary_cases=8 "
        "empty_expert_cases=1 geometry_cases=20\n");
    std::printf(
        "routing=synthetic_balanced representative=0\n");
    std::printf(
        "threshold min_speedup=%s\n",
        W2A8_FUSED_GU_PROBE_MIN_SPEEDUP_TEXT);
    std::printf(
        "baseline_ms=%.9g candidate_ms=%.9g speedup=%.9g abba_samples=4\n",
        timing.baseline_ms, timing.candidate_ms, timing.speedup);
    std::printf(
        "fp8_mismatches=0 scale_mismatches=0 guards=clean "
        "poison_a=clean poison_b=clean active_extents=written\n");
    std::printf("result=PASS\n");
    return 0;
}
// END fused GU bounded output contract
