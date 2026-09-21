// SPDX-License-Identifier: AGPL-3.0-only
//
// Immutable GB10 promotion probe for the exact DeepSeek-V4 W2A8 routed-MoE
// core. The incumbent is the production N64 five-launch chain. The candidate
// composes the locked fused N128 gate/up-to-down-A8 kernel with the locked N256
// down kernel. This host harness is linked against the production wrappers as
// separate translation units; it is not part of any serving registry.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef W2A8_COMPOSED_PROBE_BUILD_ID
#error "W2A8_COMPOSED_PROBE_BUILD_ID must be receipt-bound"
#endif
#ifndef W2A8_COMPOSED_PROBE_MIN_SPEEDUP
#error "W2A8_COMPOSED_PROBE_MIN_SPEEDUP must be receipt-bound"
#endif
#ifndef W2A8_COMPOSED_PROBE_MIN_SPEEDUP_TEXT
#error "W2A8_COMPOSED_PROBE_MIN_SPEEDUP_TEXT must be receipt-bound"
#endif
#ifndef W2A8_PACKED_E4M3_CANDIDATE
#error "W2A8_PACKED_E4M3_CANDIDATE must be receipt-bound"
#endif
#ifndef W2A8_COMPOSED_PROBE_ENFORCE_SPEEDUP
#error "W2A8_COMPOSED_PROBE_ENFORCE_SPEEDUP must be receipt-bound"
#endif

extern "C" __global__ void exl3_w2a8_h128_pre_dual_emit_h4096(
    const __nv_bfloat16*, const int*, const int*, const unsigned long long*,
    const unsigned long long*, unsigned char*, float*, unsigned char*, float*,
    unsigned int, unsigned int);
extern "C" __global__ void exl3_w2a8_grouped_prefill_k2_gu(
    const unsigned char*, const float*, const unsigned long long*,
    __nv_bfloat16*, const int*, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int);
extern "C" __global__ void exl3_w2a8_h128_post_silu_pre_emit_h2048(
    const __nv_bfloat16*, const __nv_bfloat16*, const int*,
    const unsigned long long*, const unsigned long long*,
    const unsigned long long*, unsigned char*, float*, unsigned int,
    unsigned int);
extern "C" __global__ void exl3_w2a8_grouped_prefill_k2_down(
    const unsigned char*, const float*, const unsigned long long*,
    __nv_bfloat16*, const int*, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int);
extern "C" __global__ void exl3_w2a8_fused_gu_down_emit_n128(
    const unsigned char*, const float*, const unsigned char*, const float*,
    const unsigned long long*, const unsigned long long*,
    const unsigned long long*, const unsigned long long*,
    const unsigned long long*, unsigned char*, float*, const int*,
    unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
    unsigned int);
extern "C" __global__ void exl3_w2a8_grouped_prefill_n256_k2_down(
    const unsigned char*, const float*, const unsigned long long*,
    __nv_bfloat16*, const int*, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int);
extern "C" __global__ void exl3_h128_post_rows(
    __nv_bfloat16*, const int*, const unsigned long long*, unsigned int);

namespace probe {

constexpr unsigned kTokens = 2410;
constexpr unsigned kTopK = 6;
constexpr unsigned kRows = 14460;
constexpr unsigned kExperts = 256;
constexpr unsigned kHidden = 4096;
constexpr unsigned kIntermediate = 2048;
constexpr unsigned kGroup = 128;
constexpr std::size_t kGuardBytes = 256;
constexpr unsigned char kGuard = 0xcd;
constexpr unsigned char kPoisonA = 0xa5;
constexpr unsigned char kPoisonB = 0x5a;
constexpr double kMinSpeedup = W2A8_COMPOSED_PROBE_MIN_SPEEDUP;
constexpr int kWarmups = 2;
constexpr int kTimingIterations = 2;
static_assert(kRows == kTokens * kTopK);
static_assert(kMinSpeedup > 1.0 && kMinSpeedup <= 100.0);
static_assert(W2A8_PACKED_E4M3_CANDIDATE == 0 ||
              W2A8_PACKED_E4M3_CANDIDATE == 1);
static_assert(W2A8_COMPOSED_PROBE_ENFORCE_SPEEDUP == 0 ||
              W2A8_COMPOSED_PROBE_ENFORCE_SPEEDUP == 1);

[[noreturn]] void fail(const char* check) {
    std::fprintf(stderr, "FAIL check=%s\n", check);
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
            "initialize guards");
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() {
        if (base != nullptr) cudaFree(base);
    }

    template <typename T>
    T* as() const {
        return reinterpret_cast<T*>(data);
    }

    void fill(unsigned char value) {
        check(cudaMemset(data, value, bytes), "fill buffer");
    }

    template <typename T>
    void upload(const std::vector<T>& values) {
        if (values.size() * sizeof(T) != bytes) fail("upload-size");
        check(
            cudaMemcpy(data, values.data(), bytes, cudaMemcpyHostToDevice),
            "upload buffer");
    }

    bool guards_clean() const {
        std::vector<unsigned char> guards(2 * kGuardBytes);
        check(
            cudaMemcpy(
                guards.data(), base, kGuardBytes, cudaMemcpyDeviceToHost),
            "download prefix guard");
        check(
            cudaMemcpy(
                guards.data() + kGuardBytes, data + bytes, kGuardBytes,
                cudaMemcpyDeviceToHost),
            "download suffix guard");
        return std::all_of(
            guards.begin(), guards.end(),
            [](unsigned char value) { return value == kGuard; });
    }
};

__global__ void fill_u16_kernel(
    unsigned short* values, std::size_t count, unsigned int salt) {
    const std::size_t stride = (std::size_t)blockDim.x * gridDim.x;
    for (std::size_t index = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        unsigned int value = (unsigned int)index ^ (salt * 0x9e3779b9u);
        value ^= value >> 16;
        value *= 0x7feb352du;
        value ^= value >> 15;
        values[index] = (unsigned short)(value ^ (value >> 16));
    }
}

__global__ void fill_bf16_kernel(
    __nv_bfloat16* values, std::size_t count, unsigned int salt) {
    const std::size_t stride = (std::size_t)blockDim.x * gridDim.x;
    for (std::size_t index = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        const int centered = (int)((index * 37 + salt * 101) % 2047) - 1023;
        values[index] = __float2bfloat16((float)centered / 257.0f);
    }
}

__global__ void fill_sign_kernel(
    unsigned short* values, std::size_t count, unsigned int salt) {
    const std::size_t stride = (std::size_t)blockDim.x * gridDim.x;
    for (std::size_t index = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        values[index] = (((index >> 3) ^ index ^ salt) & 1u) ? 0xbc00 : 0x3c00;
    }
}

__global__ void compare_kernel(
    const unsigned char* left, const unsigned char* right, std::size_t bytes,
    unsigned int* mismatch) {
    const std::size_t stride = (std::size_t)blockDim.x * gridDim.x;
    for (std::size_t index = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < bytes; index += stride) {
        if (left[index] != right[index]) atomicExch(mismatch, 1u);
    }
}

__global__ void all_byte_kernel(
    const unsigned char* values, std::size_t bytes, unsigned char expected,
    unsigned int* mismatch) {
    const std::size_t stride = (std::size_t)blockDim.x * gridDim.x;
    for (std::size_t index = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < bytes; index += stride) {
        if (values[index] != expected) atomicExch(mismatch, 1u);
    }
}

__global__ void hash_kernel(
    const unsigned char* values, std::size_t bytes,
    unsigned long long* result) {
    const std::size_t global = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = (std::size_t)blockDim.x * gridDim.x;
    unsigned long long hash = 0xcbf29ce484222325ull ^ global;
    for (std::size_t index = global; index < bytes; index += stride) {
        hash ^= values[index];
        hash *= 0x100000001b3ull;
    }
    atomicXor(result, hash);
}

unsigned grid_for(std::size_t count) {
    return (unsigned)std::min<std::size_t>((count + 255) / 256, 65535);
}

void fill_u16(Buffer& buffer, unsigned salt) {
    const std::size_t count = buffer.bytes / sizeof(unsigned short);
    fill_u16_kernel<<<grid_for(count), 256>>>(buffer.as<unsigned short>(), count, salt);
    check(cudaGetLastError(), "fill u16 launch");
}

void fill_bf16(Buffer& buffer, unsigned salt) {
    const std::size_t count = buffer.bytes / sizeof(__nv_bfloat16);
    fill_bf16_kernel<<<grid_for(count), 256>>>(buffer.as<__nv_bfloat16>(), count, salt);
    check(cudaGetLastError(), "fill bf16 launch");
}

void fill_signs(Buffer& buffer, unsigned salt) {
    const std::size_t count = buffer.bytes / sizeof(unsigned short);
    fill_sign_kernel<<<grid_for(count), 256>>>(buffer.as<unsigned short>(), count, salt);
    check(cudaGetLastError(), "fill signs launch");
}

void compare_exact(const Buffer& left, const Buffer& right, const char* label) {
    if (left.bytes != right.bytes) fail("compare-size");
    unsigned int* mismatch = nullptr;
    check(cudaMalloc(&mismatch, sizeof(*mismatch)), "allocate compare flag");
    check(cudaMemset(mismatch, 0, sizeof(*mismatch)), "clear compare flag");
    compare_kernel<<<grid_for(left.bytes), 256>>>(
        left.data, right.data, left.bytes, mismatch);
    check(cudaGetLastError(), "compare launch");
    unsigned int host = 0;
    check(
        cudaMemcpy(&host, mismatch, sizeof(host), cudaMemcpyDeviceToHost),
        "download compare flag");
    cudaFree(mismatch);
    if (host != 0) fail(label);
}

void assert_all_byte(
    const Buffer& buffer, unsigned char expected, const char* label) {
    unsigned int* mismatch = nullptr;
    check(cudaMalloc(&mismatch, sizeof(*mismatch)), "allocate byte flag");
    check(cudaMemset(mismatch, 0, sizeof(*mismatch)), "clear byte flag");
    all_byte_kernel<<<grid_for(buffer.bytes), 256>>>(
        buffer.data, buffer.bytes, expected, mismatch);
    check(cudaGetLastError(), "all-byte launch");
    unsigned int host = 0;
    check(
        cudaMemcpy(&host, mismatch, sizeof(host), cudaMemcpyDeviceToHost),
        "download byte flag");
    cudaFree(mismatch);
    if (host != 0) fail(label);
}

unsigned long long hash_buffer(const Buffer& buffer) {
    unsigned long long* device = nullptr;
    check(cudaMalloc(&device, sizeof(*device)), "allocate hash");
    check(cudaMemset(device, 0, sizeof(*device)), "clear hash");
    hash_kernel<<<256, 256>>>(buffer.data, buffer.bytes, device);
    check(cudaGetLastError(), "hash launch");
    unsigned long long host = 0;
    check(
        cudaMemcpy(&host, device, sizeof(host), cudaMemcpyDeviceToHost),
        "download hash");
    cudaFree(device);
    return host;
}

void snapshot(const Buffer& live, Buffer& copy) {
    if (live.bytes != copy.bytes) fail("snapshot-size");
    check(
        cudaMemcpy(copy.data, live.data, live.bytes, cudaMemcpyDeviceToDevice),
        "snapshot input");
}

constexpr std::size_t fp8_bytes(unsigned rows, unsigned columns) {
    return (std::size_t)rows * columns;
}

constexpr std::size_t scale_bytes(unsigned rows, unsigned columns) {
    return (std::size_t)rows * (columns / kGroup) * sizeof(float);
}

constexpr std::size_t bf16_bytes(unsigned rows, unsigned columns) {
    return (std::size_t)rows * columns * sizeof(__nv_bfloat16);
}

constexpr std::size_t trellis_bytes_per_expert(unsigned n, unsigned k) {
    return (std::size_t)(k / 16) * (n / 16) * 32 * sizeof(unsigned short);
}

struct ChainBuffers {
    Buffer gate_fp8{fp8_bytes(kRows, kHidden)};
    Buffer gate_scale{scale_bytes(kRows, kHidden)};
    Buffer up_fp8{fp8_bytes(kRows, kHidden)};
    Buffer up_scale{scale_bytes(kRows, kHidden)};
    Buffer gate_bf16{bf16_bytes(kRows, kIntermediate)};
    Buffer up_bf16{bf16_bytes(kRows, kIntermediate)};
    Buffer down_fp8{fp8_bytes(kRows, kIntermediate)};
    Buffer down_scale{scale_bytes(kRows, kIntermediate)};
    Buffer raw_down{bf16_bytes(kRows, kHidden)};
    Buffer final_down{bf16_bytes(kRows, kHidden)};

    void reset(unsigned char poison) {
        gate_fp8.fill(poison);
        gate_scale.fill(poison);
        up_fp8.fill(poison);
        up_scale.fill(poison);
        gate_bf16.fill(poison);
        up_bf16.fill(poison);
        down_fp8.fill(poison);
        down_scale.fill(poison);
        raw_down.fill(poison);
        final_down.fill(poison);
    }

    bool guards_clean() const {
        return gate_fp8.guards_clean() && gate_scale.guards_clean() &&
            up_fp8.guards_clean() && up_scale.guards_clean() &&
            gate_bf16.guards_clean() && up_bf16.guards_clean() &&
            down_fp8.guards_clean() && down_scale.guards_clean() &&
            raw_down.guards_clean() && final_down.guards_clean();
    }
};

// BEGIN production arena alias parity
struct ProductionArena {
    // These three allocations match the minimum capacities and destructive
    // lifetimes used by `try_run_exl3_w2a8_prefill`.
    Buffer expert_gate{bf16_bytes(kRows, kIntermediate)};
    Buffer expert_up{
        fp8_bytes(kRows, kHidden) + scale_bytes(kRows, kHidden)};
    Buffer expert_down{bf16_bytes(kRows, kHidden)};

    void reset(unsigned char poison) {
        expert_gate.fill(poison);
        expert_up.fill(poison);
        expert_down.fill(poison);
    }

    bool guards_clean() const {
        return expert_gate.guards_clean() && expert_up.guards_clean() &&
            expert_down.guards_clean();
    }
};

struct Fixture {
    static constexpr std::size_t kTrellisPerExpert =
        trellis_bytes_per_expert(kIntermediate, kHidden);
    static_assert(
        kTrellisPerExpert == trellis_bytes_per_expert(kHidden, kIntermediate));

    Buffer input{bf16_bytes(kTokens, kHidden)};
    Buffer input_copy{input.bytes};
    Buffer gate_trellis{kExperts * kTrellisPerExpert};
    Buffer gate_trellis_copy{gate_trellis.bytes};
    Buffer up_trellis{kExperts * kTrellisPerExpert};
    Buffer up_trellis_copy{up_trellis.bytes};
    Buffer down_trellis{kExperts * kTrellisPerExpert};
    Buffer down_trellis_copy{down_trellis.bytes};
    Buffer gate_suh{kExperts * kHidden * sizeof(unsigned short)};
    Buffer gate_suh_copy{gate_suh.bytes};
    Buffer up_suh{kExperts * kHidden * sizeof(unsigned short)};
    Buffer up_suh_copy{up_suh.bytes};
    Buffer gate_svh{kExperts * kIntermediate * sizeof(unsigned short)};
    Buffer gate_svh_copy{gate_svh.bytes};
    Buffer up_svh{kExperts * kIntermediate * sizeof(unsigned short)};
    Buffer up_svh_copy{up_svh.bytes};
    Buffer down_suh{kExperts * kIntermediate * sizeof(unsigned short)};
    Buffer down_suh_copy{down_suh.bytes};
    Buffer down_svh{kExperts * kHidden * sizeof(unsigned short)};
    Buffer down_svh_copy{down_svh.bytes};
    Buffer gate_trellis_tab{kExperts * sizeof(unsigned long long)};
    Buffer gate_trellis_tab_copy{gate_trellis_tab.bytes};
    Buffer up_trellis_tab{kExperts * sizeof(unsigned long long)};
    Buffer up_trellis_tab_copy{up_trellis_tab.bytes};
    Buffer down_trellis_tab{kExperts * sizeof(unsigned long long)};
    Buffer down_trellis_tab_copy{down_trellis_tab.bytes};
    Buffer gate_suh_tab{kExperts * sizeof(unsigned long long)};
    Buffer gate_suh_tab_copy{gate_suh_tab.bytes};
    Buffer up_suh_tab{kExperts * sizeof(unsigned long long)};
    Buffer up_suh_tab_copy{up_suh_tab.bytes};
    Buffer gate_svh_tab{kExperts * sizeof(unsigned long long)};
    Buffer gate_svh_tab_copy{gate_svh_tab.bytes};
    Buffer up_svh_tab{kExperts * sizeof(unsigned long long)};
    Buffer up_svh_tab_copy{up_svh_tab.bytes};
    Buffer down_suh_tab{kExperts * sizeof(unsigned long long)};
    Buffer down_suh_tab_copy{down_suh_tab.bytes};
    Buffer down_svh_tab{kExperts * sizeof(unsigned long long)};
    Buffer down_svh_tab_copy{down_svh_tab.bytes};
    Buffer offsets{(kExperts + 1) * sizeof(int)};
    Buffer offsets_copy{offsets.bytes};
    Buffer sorted_tokens{kRows * sizeof(int)};
    Buffer sorted_tokens_copy{sorted_tokens.bytes};
    Buffer sorted_experts{kRows * sizeof(int)};
    Buffer sorted_experts_copy{sorted_experts.bytes};
    ChainBuffers baseline;
    ChainBuffers candidate;

    Fixture() {
        fill_bf16(input, 1);
        fill_u16(gate_trellis, 11);
        fill_u16(up_trellis, 29);
        fill_u16(down_trellis, 47);
        fill_signs(gate_suh, 3);
        fill_signs(up_suh, 5);
        fill_signs(gate_svh, 7);
        fill_signs(up_svh, 11);
        fill_signs(down_suh, 13);
        fill_signs(down_svh, 17);
        upload_table(gate_trellis_tab, gate_trellis, kTrellisPerExpert);
        upload_table(up_trellis_tab, up_trellis, kTrellisPerExpert);
        upload_table(down_trellis_tab, down_trellis, kTrellisPerExpert);
        upload_table(gate_suh_tab, gate_suh, kHidden * sizeof(unsigned short));
        upload_table(up_suh_tab, up_suh, kHidden * sizeof(unsigned short));
        upload_table(gate_svh_tab, gate_svh, kIntermediate * sizeof(unsigned short));
        upload_table(up_svh_tab, up_svh, kIntermediate * sizeof(unsigned short));
        upload_table(down_suh_tab, down_suh, kIntermediate * sizeof(unsigned short));
        upload_table(down_svh_tab, down_svh, kHidden * sizeof(unsigned short));
        check(cudaDeviceSynchronize(), "fixture initialization");
        snapshot_static_inputs();
    }

    static void upload_table(Buffer& table, const Buffer& values, std::size_t stride) {
        std::vector<unsigned long long> pointers(kExperts);
        for (unsigned expert = 0; expert < kExperts; ++expert) {
            pointers[expert] = (unsigned long long)(
                reinterpret_cast<std::uintptr_t>(values.data) + expert * stride);
        }
        table.upload(pointers);
    }

    void snapshot_static_inputs() {
        snapshot(input, input_copy);
        snapshot(gate_trellis, gate_trellis_copy);
        snapshot(up_trellis, up_trellis_copy);
        snapshot(down_trellis, down_trellis_copy);
        snapshot(gate_suh, gate_suh_copy);
        snapshot(up_suh, up_suh_copy);
        snapshot(gate_svh, gate_svh_copy);
        snapshot(up_svh, up_svh_copy);
        snapshot(down_suh, down_suh_copy);
        snapshot(down_svh, down_svh_copy);
        snapshot(gate_trellis_tab, gate_trellis_tab_copy);
        snapshot(up_trellis_tab, up_trellis_tab_copy);
        snapshot(down_trellis_tab, down_trellis_tab_copy);
        snapshot(gate_suh_tab, gate_suh_tab_copy);
        snapshot(up_suh_tab, up_suh_tab_copy);
        snapshot(gate_svh_tab, gate_svh_tab_copy);
        snapshot(up_svh_tab, up_svh_tab_copy);
        snapshot(down_suh_tab, down_suh_tab_copy);
        snapshot(down_svh_tab, down_svh_tab_copy);
    }

    // BEGIN production and adversarial routes
    void load_route(const char* name) {
        std::vector<unsigned> counts(kExperts, 0);
        if (std::strcmp(name, "balanced") == 0) {
            for (unsigned expert = 0; expert < kExperts; ++expert)
                counts[expert] = expert < 124 ? 57 : 56;
        } else if (std::strcmp(name, "empty-expert") == 0) {
            for (unsigned expert = 1; expert < kExperts; ++expert)
                counts[expert] = expert <= 180 ? 57 : 56;
        } else if (std::strcmp(name, "skewed") == 0) {
            counts[0] = 512;
            counts[1] = 0;
            for (unsigned expert = 2; expert < kExperts; ++expert)
                counts[expert] = expert < 234 ? 55 : 54;
        } else if (std::strcmp(name, "m64-boundaries") == 0) {
            constexpr unsigned boundary_counts[] = {63, 64, 65, 127, 128, 129};
            for (unsigned expert = 0; expert < 6; ++expert)
                counts[expert] = boundary_counts[expert];
            for (unsigned expert = 6; expert < kExperts; ++expert)
                counts[expert] = expert < 140 ? 56 : 55;
        } else if (std::strcmp(name, "checkpoint-like-synthetic-v1") == 0) {
            // Deterministic heavy-head qualification traffic only. This is not
            // represented as a histogram captured from a model checkpoint.
            counts[0] = 512;
            counts[1] = 384;
            counts[2] = 256;
            counts[3] = 192;
            counts[4] = 129;
            counts[5] = 128;
            counts[6] = 127;
            counts[7] = 65;
            counts[8] = 64;
            counts[9] = 63;
            counts[10] = 0;
            counts[11] = 0;
            for (unsigned expert = 12; expert < kExperts; ++expert)
                counts[expert] = expert < 108 ? 52 : 51;
        } else {
            fail("unknown-route");
        }
        std::vector<int> route_offsets(kExperts + 1, 0);
        for (unsigned expert = 0; expert < kExperts; ++expert)
            route_offsets[expert + 1] = route_offsets[expert] + (int)counts[expert];
        if (route_offsets.back() != static_cast<int>(kRows)) fail("route-total");
        std::vector<int> experts(kRows);
        for (unsigned expert = 0; expert < kExperts; ++expert)
            std::fill(
                experts.begin() + route_offsets[expert],
                experts.begin() + route_offsets[expert + 1], (int)expert);
        std::vector<int> tokens(kRows);
        for (unsigned row = 0; row < kRows; ++row)
            tokens[row] = (int)((row * 131u + 17u) % kTokens);
        offsets.upload(route_offsets);
        sorted_experts.upload(experts);
        sorted_tokens.upload(tokens);
        snapshot(offsets, offsets_copy);
        snapshot(sorted_experts, sorted_experts_copy);
        snapshot(sorted_tokens, sorted_tokens_copy);
    }
    // END production and adversarial routes

    void verify_immutable_inputs() const {
        compare_exact(input, input_copy, "immutable-input");
        compare_exact(gate_trellis, gate_trellis_copy, "immutable-gate-trellis");
        compare_exact(up_trellis, up_trellis_copy, "immutable-up-trellis");
        compare_exact(down_trellis, down_trellis_copy, "immutable-down-trellis");
        compare_exact(gate_suh, gate_suh_copy, "immutable-gate-suh");
        compare_exact(up_suh, up_suh_copy, "immutable-up-suh");
        compare_exact(gate_svh, gate_svh_copy, "immutable-gate-svh");
        compare_exact(up_svh, up_svh_copy, "immutable-up-svh");
        compare_exact(down_suh, down_suh_copy, "immutable-down-suh");
        compare_exact(down_svh, down_svh_copy, "immutable-down-svh");
        compare_exact(gate_trellis_tab, gate_trellis_tab_copy, "immutable-gate-tab");
        compare_exact(up_trellis_tab, up_trellis_tab_copy, "immutable-up-tab");
        compare_exact(down_trellis_tab, down_trellis_tab_copy, "immutable-down-tab");
        compare_exact(gate_suh_tab, gate_suh_tab_copy, "immutable-gate-suh-tab");
        compare_exact(up_suh_tab, up_suh_tab_copy, "immutable-up-suh-tab");
        compare_exact(gate_svh_tab, gate_svh_tab_copy, "immutable-gate-svh-tab");
        compare_exact(up_svh_tab, up_svh_tab_copy, "immutable-up-svh-tab");
        compare_exact(down_suh_tab, down_suh_tab_copy, "immutable-down-suh-tab");
        compare_exact(down_svh_tab, down_svh_tab_copy, "immutable-down-svh-tab");
        compare_exact(offsets, offsets_copy, "immutable-offsets");
        compare_exact(sorted_experts, sorted_experts_copy, "immutable-experts");
        compare_exact(sorted_tokens, sorted_tokens_copy, "immutable-tokens");
    }

    // BEGIN runtime input content hash
    unsigned long long input_hash() const {
        unsigned long long result = 0xcbf29ce484222325ull;
        const Buffer* content_inputs[] = {
            &input, &gate_trellis, &up_trellis, &down_trellis,
            &gate_suh, &up_suh, &gate_svh, &up_svh, &down_suh, &down_svh,
            &offsets, &sorted_tokens, &sorted_experts,
        };
        for (const Buffer* content : content_inputs) {
            result ^= hash_buffer(*content);
            result *= 0x100000001b3ull;
        }
        return result;
    }
    // END runtime input content hash

    bool guards_clean() const {
        const Buffer* inputs[] = {
            &input, &input_copy, &gate_trellis, &gate_trellis_copy,
            &up_trellis, &up_trellis_copy, &down_trellis,
            &down_trellis_copy, &gate_suh, &gate_suh_copy, &up_suh,
            &up_suh_copy, &gate_svh, &gate_svh_copy, &up_svh,
            &up_svh_copy, &down_suh, &down_suh_copy, &down_svh,
            &down_svh_copy, &gate_trellis_tab, &gate_trellis_tab_copy,
            &up_trellis_tab, &up_trellis_tab_copy, &down_trellis_tab,
            &down_trellis_tab_copy, &gate_suh_tab, &gate_suh_tab_copy,
            &up_suh_tab, &up_suh_tab_copy, &gate_svh_tab,
            &gate_svh_tab_copy, &up_svh_tab, &up_svh_tab_copy,
            &down_suh_tab, &down_suh_tab_copy, &down_svh_tab,
            &down_svh_tab_copy, &offsets, &offsets_copy, &sorted_tokens,
            &sorted_tokens_copy, &sorted_experts, &sorted_experts_copy,
        };
        for (const Buffer* input_buffer : inputs)
            if (!input_buffer->guards_clean()) return false;
        return baseline.guards_clean() && candidate.guards_clean();
    }
};

// BEGIN five-launch incumbent
void launch_baseline_core(Fixture& f, ChainBuffers& output) {
    exl3_w2a8_h128_pre_dual_emit_h4096<<<dim3(kRows, 4, 1), 256>>>(
        f.input.as<__nv_bfloat16>(), f.sorted_tokens.as<int>(),
        f.sorted_experts.as<int>(), f.gate_suh_tab.as<unsigned long long>(),
        f.up_suh_tab.as<unsigned long long>(), output.gate_fp8.data,
        output.gate_scale.as<float>(), output.up_fp8.data,
        output.up_scale.as<float>(), kHidden, kRows);
    exl3_w2a8_grouped_prefill_k2_gu<<<dim3(kExperts * (kIntermediate / 64)), 128>>>(
        output.gate_fp8.data, output.gate_scale.as<float>(),
        f.gate_trellis_tab.as<unsigned long long>(), output.gate_bf16.as<__nv_bfloat16>(),
        f.offsets.as<int>(), kExperts, kIntermediate, kHidden, 2, 1);
    exl3_w2a8_grouped_prefill_k2_gu<<<dim3(kExperts * (kIntermediate / 64)), 128>>>(
        output.up_fp8.data, output.up_scale.as<float>(),
        f.up_trellis_tab.as<unsigned long long>(), output.up_bf16.as<__nv_bfloat16>(),
        f.offsets.as<int>(), kExperts, kIntermediate, kHidden, 2, 1);
    exl3_w2a8_h128_post_silu_pre_emit_h2048<<<dim3(kRows, 2, 1), 256>>>(
        output.gate_bf16.as<__nv_bfloat16>(), output.up_bf16.as<__nv_bfloat16>(),
        f.sorted_experts.as<int>(), f.gate_svh_tab.as<unsigned long long>(),
        f.up_svh_tab.as<unsigned long long>(), f.down_suh_tab.as<unsigned long long>(),
        output.down_fp8.data, output.down_scale.as<float>(), kIntermediate, kRows);
    exl3_w2a8_grouped_prefill_k2_down<<<dim3(kExperts * (kHidden / 64)), 128>>>(
        output.down_fp8.data, output.down_scale.as<float>(),
        f.down_trellis_tab.as<unsigned long long>(), output.raw_down.as<__nv_bfloat16>(),
        f.offsets.as<int>(), kExperts, kHidden, kIntermediate, 2, 1);
    check(cudaGetLastError(), "five-launch incumbent");
}
// END five-launch incumbent

// BEGIN three-launch candidate
void launch_candidate_core(Fixture& f, ChainBuffers& output) {
    exl3_w2a8_h128_pre_dual_emit_h4096<<<dim3(kRows, 4, 1), 256>>>(
        f.input.as<__nv_bfloat16>(), f.sorted_tokens.as<int>(),
        f.sorted_experts.as<int>(), f.gate_suh_tab.as<unsigned long long>(),
        f.up_suh_tab.as<unsigned long long>(), output.gate_fp8.data,
        output.gate_scale.as<float>(), output.up_fp8.data,
        output.up_scale.as<float>(), kHidden, kRows);
    exl3_w2a8_fused_gu_down_emit_n128<<<dim3(kExperts * (kIntermediate / 128)), 256>>>(
        output.gate_fp8.data, output.gate_scale.as<float>(), output.up_fp8.data,
        output.up_scale.as<float>(), f.gate_trellis_tab.as<unsigned long long>(),
        f.up_trellis_tab.as<unsigned long long>(), f.gate_svh_tab.as<unsigned long long>(),
        f.up_svh_tab.as<unsigned long long>(), f.down_suh_tab.as<unsigned long long>(),
        output.down_fp8.data, output.down_scale.as<float>(), f.offsets.as<int>(),
        kExperts, kRows, kIntermediate, kHidden, 2, 1);
    exl3_w2a8_grouped_prefill_n256_k2_down<<<dim3(kExperts * (kHidden / 256)), 512>>>(
        output.down_fp8.data, output.down_scale.as<float>(),
        f.down_trellis_tab.as<unsigned long long>(), output.raw_down.as<__nv_bfloat16>(),
        f.offsets.as<int>(), kExperts, kRows, kHidden, kIntermediate, 2, 1);
    check(cudaGetLastError(), "three-launch candidate");
}
// END three-launch candidate

void launch_baseline_alias(Fixture& f, ProductionArena& arena) {
    unsigned char* gate_fp8 = arena.expert_down.data;
    float* gate_scale = reinterpret_cast<float*>(
        arena.expert_down.data + fp8_bytes(kRows, kHidden));
    unsigned char* up_fp8 = arena.expert_up.data;
    float* up_scale = reinterpret_cast<float*>(
        arena.expert_up.data + fp8_bytes(kRows, kHidden));
    exl3_w2a8_h128_pre_dual_emit_h4096<<<dim3(kRows, 4, 1), 256>>>(
        f.input.as<__nv_bfloat16>(), f.sorted_tokens.as<int>(),
        f.sorted_experts.as<int>(), f.gate_suh_tab.as<unsigned long long>(),
        f.up_suh_tab.as<unsigned long long>(), gate_fp8, gate_scale, up_fp8,
        up_scale, kHidden, kRows);
    exl3_w2a8_grouped_prefill_k2_gu<<<dim3(kExperts * (kIntermediate / 64)), 128>>>(
        gate_fp8, gate_scale, f.gate_trellis_tab.as<unsigned long long>(),
        arena.expert_gate.as<__nv_bfloat16>(), f.offsets.as<int>(), kExperts,
        kIntermediate, kHidden, 2, 1);
    exl3_w2a8_grouped_prefill_k2_gu<<<dim3(kExperts * (kIntermediate / 64)), 128>>>(
        up_fp8, up_scale, f.up_trellis_tab.as<unsigned long long>(),
        arena.expert_down.as<__nv_bfloat16>(), f.offsets.as<int>(), kExperts,
        kIntermediate, kHidden, 2, 1);
    unsigned char* down_fp8 = arena.expert_up.data;
    float* down_scale = reinterpret_cast<float*>(
        arena.expert_up.data + fp8_bytes(kRows, kIntermediate));
    exl3_w2a8_h128_post_silu_pre_emit_h2048<<<dim3(kRows, 2, 1), 256>>>(
        arena.expert_gate.as<__nv_bfloat16>(),
        arena.expert_down.as<__nv_bfloat16>(), f.sorted_experts.as<int>(),
        f.gate_svh_tab.as<unsigned long long>(),
        f.up_svh_tab.as<unsigned long long>(),
        f.down_suh_tab.as<unsigned long long>(), down_fp8, down_scale,
        kIntermediate, kRows);
    exl3_w2a8_grouped_prefill_k2_down<<<dim3(kExperts * (kHidden / 64)), 128>>>(
        down_fp8, down_scale, f.down_trellis_tab.as<unsigned long long>(),
        arena.expert_down.as<__nv_bfloat16>(), f.offsets.as<int>(), kExperts,
        kHidden, kIntermediate, 2, 1);
    check(cudaGetLastError(), "five-launch production alias");
}

void launch_candidate_alias(Fixture& f, ProductionArena& arena) {
    unsigned char* gate_fp8 = arena.expert_down.data;
    float* gate_scale = reinterpret_cast<float*>(
        arena.expert_down.data + fp8_bytes(kRows, kHidden));
    unsigned char* up_fp8 = arena.expert_up.data;
    float* up_scale = reinterpret_cast<float*>(
        arena.expert_up.data + fp8_bytes(kRows, kHidden));
    exl3_w2a8_h128_pre_dual_emit_h4096<<<dim3(kRows, 4, 1), 256>>>(
        f.input.as<__nv_bfloat16>(), f.sorted_tokens.as<int>(),
        f.sorted_experts.as<int>(), f.gate_suh_tab.as<unsigned long long>(),
        f.up_suh_tab.as<unsigned long long>(), gate_fp8, gate_scale, up_fp8,
        up_scale, kHidden, kRows);
    unsigned char* down_fp8 = arena.expert_gate.data;
    float* down_scale = reinterpret_cast<float*>(
        arena.expert_gate.data + fp8_bytes(kRows, kIntermediate));
    exl3_w2a8_fused_gu_down_emit_n128<<<dim3(kExperts * (kIntermediate / 128)), 256>>>(
        gate_fp8, gate_scale, up_fp8, up_scale,
        f.gate_trellis_tab.as<unsigned long long>(),
        f.up_trellis_tab.as<unsigned long long>(),
        f.gate_svh_tab.as<unsigned long long>(),
        f.up_svh_tab.as<unsigned long long>(),
        f.down_suh_tab.as<unsigned long long>(), down_fp8, down_scale,
        f.offsets.as<int>(), kExperts, kRows, kIntermediate, kHidden, 2, 1);
    exl3_w2a8_grouped_prefill_n256_k2_down<<<dim3(kExperts * (kHidden / 256)), 512>>>(
        down_fp8, down_scale, f.down_trellis_tab.as<unsigned long long>(),
        arena.expert_down.as<__nv_bfloat16>(), f.offsets.as<int>(), kExperts,
        kRows, kHidden, kIntermediate, 2, 1);
    check(cudaGetLastError(), "three-launch production alias");
}

void launch_final(Fixture& f, Buffer& output) {
    exl3_h128_post_rows<<<dim3(kRows, kHidden / 1024, 1), 256>>>(
        output.as<__nv_bfloat16>(), f.sorted_experts.as<int>(),
        f.down_svh_tab.as<unsigned long long>(), kHidden);
    check(cudaGetLastError(), "final H128/SVH");
}

// BEGIN composed exact-byte parity
unsigned long long run_parity(
    Fixture& f, const char* route, unsigned char baseline_poison,
    unsigned char candidate_poison) {
    f.load_route(route);
    f.baseline.reset(baseline_poison == kPoisonA ? kPoisonA : kPoisonB);
    f.candidate.reset(candidate_poison == kPoisonA ? kPoisonA : kPoisonB);
    launch_baseline_core(f, f.baseline);
    launch_candidate_core(f, f.candidate);
    check(cudaDeviceSynchronize(), "core parity synchronize");
    compare_exact(f.baseline.down_fp8, f.candidate.down_fp8, "down-fp8-parity");
    compare_exact(f.baseline.down_scale, f.candidate.down_scale, "down-scale-parity");
    compare_exact(f.baseline.raw_down, f.candidate.raw_down, "raw-down-parity");
    snapshot(f.baseline.raw_down, f.baseline.final_down);
    snapshot(f.candidate.raw_down, f.candidate.final_down);
    launch_final(f, f.baseline.final_down);
    launch_final(f, f.candidate.final_down);
    check(cudaDeviceSynchronize(), "final parity synchronize");
    compare_exact(f.baseline.final_down, f.candidate.final_down, "final-down-parity");
    f.verify_immutable_inputs();
    if (!f.guards_clean()) fail("parity-guards");
    return hash_buffer(f.candidate.down_fp8) ^
        hash_buffer(f.candidate.down_scale) ^
        hash_buffer(f.candidate.raw_down) ^
        hash_buffer(f.candidate.final_down);
}
// END composed exact-byte parity

unsigned long long run_alias_parity(Fixture& f) {
    f.load_route("m64-boundaries");
    f.baseline.reset(kPoisonA);
    f.candidate.reset(kPoisonB);
    launch_baseline_core(f, f.baseline);
    launch_candidate_core(f, f.candidate);
    check(cudaDeviceSynchronize(), "alias reference synchronize");
    snapshot(f.baseline.raw_down, f.baseline.final_down);
    snapshot(f.candidate.raw_down, f.candidate.final_down);
    launch_final(f, f.baseline.final_down);
    launch_final(f, f.candidate.final_down);
    check(cudaDeviceSynchronize(), "alias reference final synchronize");

    ProductionArena baseline_alias;
    ProductionArena candidate_alias;
    baseline_alias.reset(kPoisonB);
    candidate_alias.reset(kPoisonA);
    launch_baseline_alias(f, baseline_alias);
    check(cudaDeviceSynchronize(), "baseline alias raw synchronize");
    compare_exact(
        baseline_alias.expert_down, f.baseline.raw_down, "baseline-alias-raw");
    launch_final(f, baseline_alias.expert_down);
    check(cudaDeviceSynchronize(), "baseline alias final synchronize");
    compare_exact(
        baseline_alias.expert_down, f.baseline.final_down, "baseline-alias-final");

    launch_candidate_alias(f, candidate_alias);
    check(cudaDeviceSynchronize(), "candidate alias raw synchronize");
    compare_exact(
        candidate_alias.expert_down, f.candidate.raw_down, "candidate-alias-raw");
    launch_final(f, candidate_alias.expert_down);
    check(cudaDeviceSynchronize(), "candidate alias final synchronize");
    compare_exact(
        candidate_alias.expert_down, f.candidate.final_down, "candidate-alias-final");
    f.verify_immutable_inputs();
    if (!baseline_alias.guards_clean() || !candidate_alias.guards_clean() ||
        !f.guards_clean()) fail("production-alias-guards");
    return hash_buffer(baseline_alias.expert_down) ^
        hash_buffer(candidate_alias.expert_down);
}
// END production arena alias parity

// BEGIN guards inputs and malformed no-write
void launch_fused_malformed(
    Fixture& f, dim3 grid, dim3 block, unsigned n, unsigned k, unsigned bits,
    unsigned mode, unsigned experts, unsigned rows, const int* offsets,
    const char* label) {
    f.candidate.down_fp8.fill(kPoisonA);
    f.candidate.down_scale.fill(kPoisonA);
    exl3_w2a8_fused_gu_down_emit_n128<<<grid, block>>>(
        f.candidate.gate_fp8.data, f.candidate.gate_scale.as<float>(),
        f.candidate.up_fp8.data, f.candidate.up_scale.as<float>(),
        f.gate_trellis_tab.as<unsigned long long>(), f.up_trellis_tab.as<unsigned long long>(),
        f.gate_svh_tab.as<unsigned long long>(), f.up_svh_tab.as<unsigned long long>(),
        f.down_suh_tab.as<unsigned long long>(), f.candidate.down_fp8.data,
        f.candidate.down_scale.as<float>(), offsets, experts, rows, n, k, bits, mode);
    check(cudaGetLastError(), label);
    check(cudaDeviceSynchronize(), label);
    assert_all_byte(f.candidate.down_fp8, kPoisonA, label);
    assert_all_byte(f.candidate.down_scale, kPoisonA, label);
}

void launch_down_malformed(
    Fixture& f, dim3 grid, dim3 block, unsigned n, unsigned k, unsigned bits,
    unsigned mode, unsigned experts, unsigned rows, const int* offsets,
    const char* label) {
    f.candidate.raw_down.fill(kPoisonB);
    exl3_w2a8_grouped_prefill_n256_k2_down<<<grid, block>>>(
        f.candidate.down_fp8.data, f.candidate.down_scale.as<float>(),
        f.down_trellis_tab.as<unsigned long long>(), f.candidate.raw_down.as<__nv_bfloat16>(),
        offsets, experts, rows, n, k, bits, mode);
    check(cudaGetLastError(), label);
    check(cudaDeviceSynchronize(), label);
    assert_all_byte(f.candidate.raw_down, kPoisonB, label);
}

unsigned run_malformed(Fixture& f) {
    f.load_route("balanced");
    f.candidate.reset(kPoisonA);
    exl3_w2a8_h128_pre_dual_emit_h4096<<<dim3(kRows, 4, 1), 256>>>(
        f.input.as<__nv_bfloat16>(), f.sorted_tokens.as<int>(),
        f.sorted_experts.as<int>(), f.gate_suh_tab.as<unsigned long long>(),
        f.up_suh_tab.as<unsigned long long>(), f.candidate.gate_fp8.data,
        f.candidate.gate_scale.as<float>(), f.candidate.up_fp8.data,
        f.candidate.up_scale.as<float>(), kHidden, kRows);
    check(cudaDeviceSynchronize(), "malformed setup");
    const dim3 fused_grid(kExperts * (kIntermediate / 128));
    const dim3 down_grid(kExperts * (kHidden / 256));
    const int* offsets = f.offsets.as<int>();
    launch_fused_malformed(f, fused_grid, 255, kIntermediate, kHidden, 2, 1,
        kExperts, kRows, offsets, "wrong-fused-block");
    launch_fused_malformed(f, fused_grid.x - 1, 256, kIntermediate, kHidden, 2, 1,
        kExperts, kRows, offsets, "wrong-fused-grid");
    launch_fused_malformed(f, fused_grid, 256, kIntermediate - 1, kHidden, 2, 1,
        kExperts, kRows, offsets, "wrong-fused-N");
    launch_fused_malformed(f, fused_grid, 256, kIntermediate, kHidden - 1, 2, 1,
        kExperts, kRows, offsets, "wrong-fused-K");
    launch_fused_malformed(f, fused_grid, 256, kIntermediate, kHidden, 3, 1,
        kExperts, kRows, offsets, "wrong-fused-bits");
    launch_fused_malformed(f, fused_grid, 256, kIntermediate, kHidden, 2, 0,
        kExperts, kRows, offsets, "wrong-fused-mode");
    std::vector<int> bad_offsets(kExperts + 1, 0);
    check(cudaMemcpy(bad_offsets.data(), offsets, f.offsets.bytes, cudaMemcpyDeviceToHost),
        "download malformed offsets");
    bad_offsets[0] = 1;
    f.offsets.upload(bad_offsets);
    launch_fused_malformed(f, fused_grid, 256, kIntermediate, kHidden, 2, 1,
        kExperts, kRows, offsets, "wrong-fused-offsets");
    f.load_route("balanced");
    check(cudaMemcpy(bad_offsets.data(), offsets, f.offsets.bytes, cudaMemcpyDeviceToHost),
        "download late fused offsets");
    bad_offsets[kExperts - 1] = static_cast<int>(kRows + 1);
    f.offsets.upload(bad_offsets);
    launch_fused_malformed(f, fused_grid, 256, kIntermediate, kHidden, 2, 1,
        kExperts, kRows, offsets, "wrong-fused-late-offsets");
    f.load_route("balanced");
    launch_down_malformed(f, down_grid, 256, kHidden, kIntermediate, 2, 1,
        kExperts, kRows, offsets, "wrong-down-block");
    launch_down_malformed(f, down_grid.x - 1, 512, kHidden, kIntermediate, 2, 1,
        kExperts, kRows, offsets, "wrong-down-grid");
    launch_down_malformed(f, down_grid, 512, kHidden - 1, kIntermediate, 2, 1,
        kExperts, kRows, offsets, "wrong-down-N");
    launch_down_malformed(f, down_grid, 512, kHidden, kIntermediate - 1, 2, 1,
        kExperts, kRows, offsets, "wrong-down-K");
    launch_down_malformed(f, down_grid, 512, kHidden, kIntermediate, 3, 1,
        kExperts, kRows, offsets, "wrong-down-bits");
    launch_down_malformed(f, down_grid, 512, kHidden, kIntermediate, 2, 0,
        kExperts, kRows, offsets, "wrong-down-mode");
    check(cudaMemcpy(bad_offsets.data(), offsets, f.offsets.bytes, cudaMemcpyDeviceToHost),
        "download down offsets");
    bad_offsets[0] = 1;
    f.offsets.upload(bad_offsets);
    launch_down_malformed(f, down_grid, 512, kHidden, kIntermediate, 2, 1,
        kExperts, kRows, offsets, "wrong-down-offsets");
    f.load_route("balanced");
    check(cudaMemcpy(bad_offsets.data(), offsets, f.offsets.bytes, cudaMemcpyDeviceToHost),
        "download late down offsets");
    bad_offsets[kExperts - 1] = static_cast<int>(kRows + 1);
    f.offsets.upload(bad_offsets);
    launch_down_malformed(f, down_grid, 512, kHidden, kIntermediate, 2, 1,
        kExperts, kRows, offsets, "wrong-down-late-offsets");
    f.load_route("balanced");
    f.verify_immutable_inputs();
    if (!f.guards_clean()) fail("malformed-guards_clean");
    return 16;
}
// END guards inputs and malformed no-write

float time_baseline(Fixture& f, ProductionArena& arena) {
    cudaEvent_t start = nullptr, stop = nullptr;
    check(cudaEventCreate(&start), "create baseline start");
    check(cudaEventCreate(&stop), "create baseline stop");
    check(cudaEventRecord(start), "record baseline start");
    for (int iteration = 0; iteration < kTimingIterations; ++iteration) {
        launch_baseline_alias(f, arena);
        launch_final(f, arena.expert_down);
    }
    check(cudaEventRecord(stop), "record baseline stop");
    check(cudaEventSynchronize(stop), "synchronize baseline stop");
    float elapsed = 0.0f;
    check(cudaEventElapsedTime(&elapsed, start, stop), "elapsed baseline");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsed / kTimingIterations;
}

float time_candidate(Fixture& f, ProductionArena& arena) {
    cudaEvent_t start = nullptr, stop = nullptr;
    check(cudaEventCreate(&start), "create candidate start");
    check(cudaEventCreate(&stop), "create candidate stop");
    check(cudaEventRecord(start), "record candidate start");
    for (int iteration = 0; iteration < kTimingIterations; ++iteration) {
        launch_candidate_alias(f, arena);
        launch_final(f, arena.expert_down);
    }
    check(cudaEventRecord(stop), "record candidate stop");
    check(cudaEventSynchronize(stop), "synchronize candidate stop");
    float elapsed = 0.0f;
    check(cudaEventElapsedTime(&elapsed, start, stop), "elapsed candidate");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsed / kTimingIterations;
}

struct Timing {
    double baseline;
    double candidate;
    double speedup;
};

// BEGIN whole-chain ABBA timing
Timing run_timing(Fixture& f) {
    f.load_route("checkpoint-like-synthetic-v1");
    ProductionArena arena;
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        launch_baseline_alias(f, arena);
        launch_final(f, arena.expert_down);
        launch_candidate_alias(f, arena);
        launch_final(f, arena.expert_down);
    }
    check(cudaDeviceSynchronize(), "timing warmups");
    const double baseline_ms0 = time_baseline(f, arena);
    const double candidate_ms0 = time_candidate(f, arena);
    const double candidate_ms1 = time_candidate(f, arena);
    const double baseline_ms1 = time_baseline(f, arena);
    const double baseline = 0.5 * (baseline_ms0 + baseline_ms1);
    const double candidate = 0.5 * (candidate_ms0 + candidate_ms1);
    const double speedup = baseline / candidate;
    if (!(baseline > 0.0) || !(candidate > 0.0) || !std::isfinite(speedup))
        fail("timing-invalid");
    if (W2A8_COMPOSED_PROBE_ENFORCE_SPEEDUP && speedup < kMinSpeedup)
        fail("timing-threshold");
    const char* threshold_text = W2A8_COMPOSED_PROBE_MIN_SPEEDUP_TEXT;
    if (std::strtod(threshold_text, nullptr) != kMinSpeedup) fail("threshold-text");
    return {baseline, candidate, speedup};
}
// END whole-chain ABBA timing

bool valid_build_id(const char* value) {
    if (std::strlen(value) != 64) return false;
    for (unsigned index = 0; index < 64; ++index) {
        const char character = value[index];
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) return false;
    }
    return true;
}

std::string device_uuid(const cudaDeviceProp& properties) {
    char text[33];
    for (unsigned index = 0; index < 16; ++index)
        std::snprintf(text + 2 * index, 3, "%02x", (unsigned char)properties.uuid.bytes[index]);
    text[32] = '\0';
    return text;
}

}  // namespace probe

int main() {
    using namespace probe;
    if (!valid_build_id(W2A8_COMPOSED_PROBE_BUILD_ID)) fail("build-id");
    int device = 0;
    check(cudaGetDevice(&device), "get device");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "device properties");
    int driver = 0, runtime = 0;
    check(cudaDriverGetVersion(&driver), "driver version");
    check(cudaRuntimeGetVersion(&runtime), "runtime version");

    Fixture fixture;
    unsigned long long output_hash = 0xcbf29ce484222325ull;
    const auto mix_hash = [&](unsigned long long parity_hash) {
        output_hash ^= parity_hash;
        output_hash *= 0x100000001b3ull;
    };
    mix_hash(run_parity(fixture, "balanced", kPoisonA, kPoisonB));
    mix_hash(run_parity(fixture, "balanced", kPoisonB, kPoisonA));
    mix_hash(run_parity(fixture, "empty-expert", kPoisonA, kPoisonB));
    mix_hash(run_parity(fixture, "skewed", kPoisonB, kPoisonA));
    mix_hash(run_parity(fixture, "m64-boundaries", kPoisonA, kPoisonB));
    mix_hash(run_parity(fixture, "m64-boundaries", kPoisonB, kPoisonA));
    mix_hash(run_parity(
        fixture, "checkpoint-like-synthetic-v1", kPoisonA, kPoisonB));
    mix_hash(run_parity(
        fixture, "checkpoint-like-synthetic-v1", kPoisonB, kPoisonA));
    mix_hash(run_alias_parity(fixture));
    const unsigned malformed_cases = run_malformed(fixture);
    const Timing timing = run_timing(fixture);
    fixture.verify_immutable_inputs();
    if (!fixture.guards_clean()) fail("final-guards");
    const unsigned long long input_hash = fixture.input_hash();

    std::printf("build_id=%s\n", W2A8_COMPOSED_PROBE_BUILD_ID);
    std::printf(
        "variant=%s packed_e4m3=%d\n",
        W2A8_PACKED_E4M3_CANDIDATE ? "candidate" : "incumbent",
        W2A8_PACKED_E4M3_CANDIDATE);
    std::printf(
        "device_uuid=%s driver=%d runtime=%d\n",
        device_uuid(properties).c_str(), driver, runtime);
    std::printf(
        "tokens=%u rows=%u topk=%u experts=%u hidden=%u intermediate=%u\n",
        kTokens, kRows, kTopK, kExperts, kHidden, kIntermediate);
    std::printf(
        "routes=balanced,empty-expert,skewed,m64-boundaries,"
        "checkpoint-like-synthetic-v1 timing_route=checkpoint-like-synthetic-v1 "
        "offsets=257 poison_passes=8\n");
    std::printf(
        "threshold min_speedup=%s binary64=%.17g enforcement=%s\n",
        W2A8_COMPOSED_PROBE_MIN_SPEEDUP_TEXT, kMinSpeedup,
        W2A8_COMPOSED_PROBE_ENFORCE_SPEEDUP ? "enforced" : "parity-only");
    std::printf(
        "baseline_ms=%.9g candidate_ms=%.9g speedup=%.17g abba_samples=4 timed=whole-chain\n",
        timing.baseline, timing.candidate, timing.speedup);
    std::printf("intermediate_fp8=exact intermediate_scale=exact\n");
    std::printf("raw_bf16=exact final_bf16=exact production_alias=exact\n");
    std::printf(
        "guards=clean inputs=immutable malformed_cases=%u all_offsets=validated\n",
        malformed_cases);
    std::printf("input_hash=%016llx\n", input_hash);
    std::printf("output_hash=%016llx\n", output_hash);
    std::printf("result=PASS\n");
    return 0;
}
