// SPDX-License-Identifier: AGPL-3.0-only
//
// Standalone GB10 promotion probe for the isolated EXL3 W2A8 component.
// Build it offline with scripts/check-exl3-prefill-w2a8-probe-build.sh, then
// run it on GB10 with explicit cosine, max-error, and end-to-end-speed gates.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifndef W2A8_PROBE_GU
#error "W2A8_PROBE_GU must be explicit"
#endif
#ifndef W2A8_PROBE_N_TILE
#error "W2A8_PROBE_N_TILE must be explicit"
#endif
#if W2A8_PROBE_N_TILE != 64 && W2A8_PROBE_N_TILE != 128 && \
    W2A8_PROBE_N_TILE != 256
#error "W2A8_PROBE_N_TILE must be 64, 128, or 256"
#endif
#if W2A8_PROBE_N_TILE == 256
#define W2A8_PROBE_THREADS 512
#elif W2A8_PROBE_N_TILE == 128
#define W2A8_PROBE_THREADS 256
#else
#define W2A8_PROBE_THREADS 128
#endif
#ifndef W2A8_BUILD_ID
#error "W2A8_BUILD_ID must be supplied by the receipt-producing build"
#endif
#if W2A8_PROBE_GU
#define W2A8_FIXED_N 2048
#define W2A8_FIXED_K 4096
#if W2A8_PROBE_N_TILE == 256
#define W2A8_KERNEL_NAME exl3_w2a8_grouped_prefill_n256_gu
#elif W2A8_PROBE_N_TILE == 128
#define W2A8_KERNEL_NAME exl3_w2a8_grouped_prefill_n128_gu
#else
#define W2A8_KERNEL_NAME exl3_w2a8_grouped_prefill_gu
#endif
#define W2A8_BASELINE_KERNEL exl3_grouped_prefill_k64_k2_gu
#include "../common/exl3_grouped_prefill_k64_k2_gu.cu"
#else
#define W2A8_FIXED_N 4096
#define W2A8_FIXED_K 2048
#if W2A8_PROBE_N_TILE == 256
#define W2A8_KERNEL_NAME exl3_w2a8_grouped_prefill_n256_down
#elif W2A8_PROBE_N_TILE == 128
#define W2A8_KERNEL_NAME exl3_w2a8_grouped_prefill_n128_down
#else
#define W2A8_KERNEL_NAME exl3_w2a8_grouped_prefill_down
#endif
#define W2A8_BASELINE_KERNEL exl3_grouped_prefill_k64_k2_down
#include "../common/exl3_grouped_prefill_k64_k2_down.cu"
#endif
#include "../common/per_token_group_quant_fp8.cu"
#if W2A8_PROBE_N_TILE == 256
#include "exl3_w2a8_grouped_prefill_n256.cu"
#elif W2A8_PROBE_N_TILE == 128
#include "exl3_w2a8_grouped_prefill_n128.cu"
#else
#include "exl3_w2a8_grouped_prefill.cu"
#endif
static void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
        std::exit(2);
    }
}
static std::uint16_t f32_to_bf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (std::uint16_t)(bits >> 16);
}
static float bf16_to_f32(std::uint16_t value) {
    const std::uint32_t bits = (std::uint32_t)value << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
static std::uint64_t next_random(std::uint64_t& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 0x2545f4914f6cdd1dull;
}
template <typename T>
static T* device_alloc(std::size_t count) {
    T* pointer = nullptr;
    cuda_check(cudaMalloc(&pointer, count * sizeof(T)), "cudaMalloc");
    return pointer;
}
template <typename Launch>
static float time_launch(Launch launch) {
    constexpr int warmups = 2;
    constexpr int iterations = 5;
    for (int iteration = 0; iteration < warmups; ++iteration) launch();
    cuda_check(cudaDeviceSynchronize(), "warmup synchronize");
    cudaEvent_t start;
    cudaEvent_t end;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&end), "cudaEventCreate(end)");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int iteration = 0; iteration < iterations; ++iteration) launch();
    cuda_check(cudaEventRecord(end), "cudaEventRecord(end)");
    cuda_check(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
    float elapsed_ms = 0.0f;
    cuda_check(
        cudaEventElapsedTime(&elapsed_ms, start, end),
        "cudaEventElapsedTime");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy(start)");
    cuda_check(cudaEventDestroy(end), "cudaEventDestroy(end)");
    return elapsed_ms / iterations;
}
static void launch_baseline(
    const __nv_bfloat16* input, const unsigned long long* trellis,
    __nv_bfloat16* output, const int* offsets, unsigned int num_experts,
    dim3 block = dim3(128, 1, 1)) {
    W2A8_BASELINE_KERNEL<<<
        dim3(num_experts * (W2A8_FIXED_N / 64), 1, 1), block>>>(
        input, trellis, output, offsets, nullptr, num_experts, W2A8_FIXED_N,
        W2A8_FIXED_K, 2, 1);
    cuda_check(cudaGetLastError(), "baseline launch");
}
static void launch_w2a8_raw(
    const unsigned char* input, const float* scales,
    const unsigned long long* trellis, __nv_bfloat16* output,
    const int* offsets, unsigned int num_experts, unsigned int total_rows,
    dim3 grid, dim3 block,
    unsigned int n, unsigned int k, unsigned int bits,
    unsigned int persistent_mode) {
#if W2A8_PROBE_N_TILE == 256
    W2A8_KERNEL_NAME<<<grid, block>>>(
        input, scales, trellis, output, offsets, num_experts, total_rows, n, k,
        bits, persistent_mode);
#else
    (void)total_rows;
    W2A8_KERNEL_NAME<<<grid, block>>>(
        input, scales, trellis, output, offsets, num_experts, n, k, bits,
        persistent_mode);
#endif
    cuda_check(cudaGetLastError(), "W2A8 launch");
}
static void launch_w2a8(
    const unsigned char* input, const float* scales,
    const unsigned long long* trellis, __nv_bfloat16* output,
    const int* offsets, unsigned int num_experts, unsigned int total_rows) {
    launch_w2a8_raw(
        input, scales, trellis, output, offsets, num_experts, total_rows,
        dim3(num_experts * (W2A8_FIXED_N / W2A8_PROBE_N_TILE), 1, 1),
        dim3(W2A8_PROBE_THREADS, 1, 1),
        W2A8_FIXED_N, W2A8_FIXED_K, 2, 1);
}

static std::uint64_t fnv1a(const std::vector<std::uint16_t>& values) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (std::uint16_t value : values) {
        hash ^= value & 0xffu;
        hash *= 0x100000001b3ull;
        hash ^= value >> 8;
        hash *= 0x100000001b3ull;
    }
    return hash;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(
            stderr,
            "usage: %s <min_cosine> <max_abs_error> "
            "<min_end_to_end_speedup>\n",
            argv[0]);
        return 2;
    }
    char* cosine_end = nullptr;
    char* error_end = nullptr;
    char* speedup_end = nullptr;
    const float min_cosine = std::strtof(argv[1], &cosine_end);
    const float max_abs_error = std::strtof(argv[2], &error_end);
    const float min_end_to_end_speedup = std::strtof(argv[3], &speedup_end);
    if (*argv[1] == '\0' || *cosine_end != '\0' || !std::isfinite(min_cosine) ||
        min_cosine < 0.99f || min_cosine > 1.0f || *argv[2] == '\0' ||
        *error_end != '\0' || !std::isfinite(max_abs_error) ||
        max_abs_error <= 0.0f || max_abs_error > 1.0f || *argv[3] == '\0' ||
        *speedup_end != '\0' || !std::isfinite(min_end_to_end_speedup) ||
        min_end_to_end_speedup <= 1.0f || min_end_to_end_speedup > 100.0f) {
        std::fprintf(stderr, "invalid explicit numeric threshold\n");
        return 2;
    }
    const char* dump_path = std::getenv("W2A8_PROBE_DUMP");
    std::FILE* dump = nullptr;
    if (dump_path != nullptr && *dump_path != '\0') {
        dump = std::fopen(dump_path, "wbx");
        if (dump == nullptr) {
            std::perror("exclusive W2A8 probe dump");
            return 2;
        }
    }
    std::printf("build_id=%s\n", W2A8_BUILD_ID);

    int device = 0;
    int driver_version = 0;
    int runtime_version = 0;
    cudaDeviceProp device_properties{};
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cuda_check(
        cudaGetDeviceProperties(&device_properties, device),
        "cudaGetDeviceProperties");
    cuda_check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    cuda_check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    std::printf(
        "device=%d name=%s cc=%d.%d driver=%d runtime=%d uuid=", device,
        device_properties.name, device_properties.major, device_properties.minor,
        driver_version, runtime_version);
    for (unsigned char byte : device_properties.uuid.bytes) {
        std::printf("%02x", (unsigned int)byte);
    }
    std::printf(
        "\nthresholds cosine=%.9g max_abs=%.9g end_to_end_speedup=%.9g "
        "N=%d K=%d n_tile=%d block=%d seed=%016llx\n",
        min_cosine, max_abs_error, min_end_to_end_speedup, W2A8_FIXED_N,
        W2A8_FIXED_K, W2A8_PROBE_N_TILE, W2A8_PROBE_THREADS,
        0x5732413820260827ull);

    constexpr int max_rows = 129;
#if W2A8_PROBE_N_TILE == 256
    constexpr unsigned int rows_capacity = max_rows;
#endif
    constexpr int max_experts = 4;
    constexpr std::size_t guard_bytes = 256;
    const std::size_t input_elements =
        (std::size_t)max_rows * W2A8_FIXED_K;
    const std::size_t output_elements =
        (std::size_t)max_rows * W2A8_FIXED_N;
    const std::size_t output_bytes = output_elements * sizeof(std::uint16_t);
    const std::size_t guarded_output_bytes = output_bytes + 2 * guard_bytes;
    const std::size_t trellis_elements_per_expert =
        (std::size_t)(W2A8_FIXED_K / 16) * (W2A8_FIXED_N / 16) * 32;

    std::uint64_t random_state = 0x5732413820260827ull;
    std::vector<std::uint16_t> input(input_elements);
    for (int row = 0; row < max_rows; ++row) {
        for (int k = 0; k < W2A8_FIXED_K; ++k) {
            const int group = k / 128;
            float value = (k & 1) == 0 ? 0.0f : -0.0f;
            if (group % 7 != 0) {
                const float unit = (float)(next_random(random_state) >> 40) /
                    (float)(1u << 24);
                const float amplitude =
                    0.00390625f * (float)(1u << (group % 8));
                value = (unit * 2.0f - 1.0f) * amplitude;
            }
            input[(std::size_t)row * W2A8_FIXED_K + k] =
                f32_to_bf16(value);
        }
    }
    std::vector<std::uint16_t> trellis(
        (std::size_t)max_experts * trellis_elements_per_expert);
    for (std::uint16_t& value : trellis) {
        value = (std::uint16_t)(next_random(random_state) >> 48);
    }

    __nv_bfloat16* d_input = device_alloc<__nv_bfloat16>(input_elements);
    unsigned char* d_a8 = device_alloc<unsigned char>(input_elements);
    float* d_scales = device_alloc<float>(
        (std::size_t)max_rows * (W2A8_FIXED_K / 128));
    std::uint16_t* d_trellis = device_alloc<std::uint16_t>(trellis.size());
    unsigned long long* d_trellis_table =
        device_alloc<unsigned long long>(max_experts);
    int* d_offsets = device_alloc<int>(max_experts + 1);
    unsigned char* d_baseline_storage =
        device_alloc<unsigned char>(guarded_output_bytes);
    unsigned char* d_w2a8_storage =
        device_alloc<unsigned char>(guarded_output_bytes);
    __nv_bfloat16* d_baseline = reinterpret_cast<__nv_bfloat16*>(
        d_baseline_storage + guard_bytes);
    __nv_bfloat16* d_w2a8 = reinterpret_cast<__nv_bfloat16*>(
        d_w2a8_storage + guard_bytes);

    cuda_check(
        cudaMemcpy(
            d_input, input.data(), input.size() * sizeof(input[0]),
            cudaMemcpyHostToDevice),
        "copy input");
    cuda_check(
        cudaMemcpy(
            d_trellis, trellis.data(), trellis.size() * sizeof(trellis[0]),
            cudaMemcpyHostToDevice),
        "copy trellis");
    std::vector<unsigned long long> trellis_pointers(max_experts);
    for (int expert = 0; expert < max_experts; ++expert) {
        trellis_pointers[expert] = (unsigned long long)(std::uintptr_t)(
            d_trellis + (std::size_t)expert * trellis_elements_per_expert);
    }
    cuda_check(
        cudaMemcpy(
            d_trellis_table, trellis_pointers.data(),
            trellis_pointers.size() * sizeof(trellis_pointers[0]),
            cudaMemcpyHostToDevice),
        "copy trellis table");

    auto quantize = [&](int rows) {
        per_token_group_quant_fp8<<<
            dim3(rows, W2A8_FIXED_K / 128, 1), dim3(128, 1, 1)>>>(
            d_input, d_a8, d_scales, rows, W2A8_FIXED_K);
        cuda_check(cudaGetLastError(), "activation quantization launch");
    };

    bool passed = true;
    auto run_case = [&](const std::vector<int>& offsets, const char* label) {
        const int rows = offsets.back();
        const unsigned int num_experts = (unsigned int)offsets.size() - 1;
        cuda_check(
            cudaMemcpy(
                d_offsets, offsets.data(), offsets.size() * sizeof(offsets[0]),
                cudaMemcpyHostToDevice),
            "copy offsets");
        const std::size_t active_elements =
            (std::size_t)rows * W2A8_FIXED_N;
        const std::size_t active_bytes =
            active_elements * sizeof(std::uint16_t);
        const dim3 valid_grid(
            num_experts * (W2A8_FIXED_N / W2A8_PROBE_N_TILE), 1, 1);

        auto invalid_canary = [&](const char* name, auto launch) {
            cuda_check(
                cudaMemset(d_w2a8_storage, 0xa5, guarded_output_bytes),
                "poison invalid output");
            launch();
            cuda_check(
                cudaDeviceSynchronize(), "invalid canary synchronize");
            std::vector<unsigned char> bytes(guarded_output_bytes);
            cuda_check(
                cudaMemcpy(
                    bytes.data(), d_w2a8_storage, bytes.size(),
                    cudaMemcpyDeviceToHost),
                "copy invalid canary");
            bool clean = true;
            for (unsigned char value : bytes) {
                if (value != 0xa5) clean = false;
            }
            passed &= clean;
            std::printf(
                "case=%s %s canary=%s\n", label, name,
                clean ? "clean" : "FAIL");
        };

        // The wrong-block canary must remain byte-for-byte poisoned.
        invalid_canary("wrong-block-x", [&]() {
            launch_w2a8_raw(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows, valid_grid,
                dim3(W2A8_PROBE_THREADS / 2, 1, 1),
                W2A8_FIXED_N, W2A8_FIXED_K, 2, 1);
        });
        invalid_canary("wrong-block-y", [&]() {
            launch_w2a8_raw(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows, valid_grid,
                dim3(W2A8_PROBE_THREADS, 2, 1),
                W2A8_FIXED_N, W2A8_FIXED_K, 2, 1);
        });
        invalid_canary("wrong-block-z", [&]() {
            launch_w2a8_raw(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows, valid_grid,
                dim3(W2A8_PROBE_THREADS, 1, 2),
                W2A8_FIXED_N, W2A8_FIXED_K, 2, 1);
        });
        // The wrong-grid canary binds exact expert x selected-N launch geometry.
        invalid_canary("wrong-grid-x", [&]() {
            launch_w2a8_raw(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows, dim3(valid_grid.x + 1, 1, 1),
                dim3(W2A8_PROBE_THREADS, 1, 1), W2A8_FIXED_N,
                W2A8_FIXED_K, 2, 1);
        });
        invalid_canary("wrong-grid-y", [&]() {
            launch_w2a8_raw(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows, dim3(valid_grid.x, 2, 1),
                dim3(W2A8_PROBE_THREADS, 1, 1), W2A8_FIXED_N,
                W2A8_FIXED_K, 2, 1);
        });
        invalid_canary("wrong-grid-z", [&]() {
            launch_w2a8_raw(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows, dim3(valid_grid.x, 1, 2),
                dim3(W2A8_PROBE_THREADS, 1, 1), W2A8_FIXED_N,
                W2A8_FIXED_K, 2, 1);
        });
        // The wrong-runtime canary set binds shape, bit-width, and persistence.
        invalid_canary("wrong-runtime-N", [&]() {
            launch_w2a8_raw(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows, valid_grid,
                dim3(W2A8_PROBE_THREADS, 1, 1),
                W2A8_FIXED_N - W2A8_PROBE_N_TILE, W2A8_FIXED_K, 2, 1);
        });
        invalid_canary("wrong-runtime-K", [&]() {
            launch_w2a8_raw(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows, valid_grid,
                dim3(W2A8_PROBE_THREADS, 1, 1),
                W2A8_FIXED_N, W2A8_FIXED_K - 128, 2, 1);
        });
        invalid_canary("wrong-runtime-bits", [&]() {
            launch_w2a8_raw(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows, valid_grid,
                dim3(W2A8_PROBE_THREADS, 1, 1),
                W2A8_FIXED_N, W2A8_FIXED_K, 3, 1);
        });
        invalid_canary("wrong-runtime-persistent", [&]() {
            launch_w2a8_raw(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows, valid_grid,
                dim3(W2A8_PROBE_THREADS, 1, 1),
                W2A8_FIXED_N, W2A8_FIXED_K, 2, 0);
        });

#if W2A8_PROBE_N_TILE == 256
        std::vector<int> negative_offsets(offsets.size(), 0);
        negative_offsets[0] = -1;
        cuda_check(
            cudaMemcpy(d_offsets, negative_offsets.data(),
                       negative_offsets.size() * sizeof(negative_offsets[0]),
                       cudaMemcpyHostToDevice),
            "copy negative-offset canary");
        invalid_canary("negative-offset", [&]() {
            launch_w2a8_raw(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows_capacity, valid_grid,
                dim3(W2A8_PROBE_THREADS, 1, 1), W2A8_FIXED_N,
                W2A8_FIXED_K, 2, 1);
        });

        std::vector<int> oversized_offsets(offsets.size(), rows_capacity + 1);
        oversized_offsets[0] = 0;
        cuda_check(
            cudaMemcpy(d_offsets, oversized_offsets.data(),
                       oversized_offsets.size() * sizeof(oversized_offsets[0]),
                       cudaMemcpyHostToDevice),
            "copy oversized-positive-offset canary");
        invalid_canary("oversized-positive-offset", [&]() {
            launch_w2a8_raw(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows_capacity, valid_grid,
                dim3(W2A8_PROBE_THREADS, 1, 1), W2A8_FIXED_N,
                W2A8_FIXED_K, 2, 1);
        });
        if (offsets.size() >= 5) {
            std::vector<int> nonmonotonic_offsets(offsets.size(), rows);
            nonmonotonic_offsets[0] = 0;
            nonmonotonic_offsets[1] = rows_capacity + 1;
            nonmonotonic_offsets[2] = 1;
            nonmonotonic_offsets[3] = 2;
            cuda_check(
                cudaMemcpy(d_offsets, nonmonotonic_offsets.data(),
                           nonmonotonic_offsets.size() *
                               sizeof(nonmonotonic_offsets[0]),
                           cudaMemcpyHostToDevice),
                "copy nonmonotonic-offset canary");
            invalid_canary("nonmonotonic-offset", [&]() {
                launch_w2a8_raw(
                    d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                    num_experts, rows_capacity, valid_grid,
                    dim3(W2A8_PROBE_THREADS, 1, 1), W2A8_FIXED_N,
                    W2A8_FIXED_K, 2, 1);
            });
        }
        std::vector<int> prefix_gap_offsets = offsets;
        prefix_gap_offsets[0] = 1;
        cuda_check(
            cudaMemcpy(d_offsets, prefix_gap_offsets.data(),
                       prefix_gap_offsets.size() * sizeof(prefix_gap_offsets[0]),
                       cudaMemcpyHostToDevice),
            "copy prefix-gap-offset canary");
        invalid_canary("prefix-gap-offset", [&]() {
            launch_w2a8_raw(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows, valid_grid,
                dim3(W2A8_PROBE_THREADS, 1, 1), W2A8_FIXED_N,
                W2A8_FIXED_K, 2, 1);
        });
        std::vector<int> final_gap_offsets = offsets;
        final_gap_offsets.back() = rows - 1;
        cuda_check(
            cudaMemcpy(d_offsets, final_gap_offsets.data(),
                       final_gap_offsets.size() * sizeof(final_gap_offsets[0]),
                       cudaMemcpyHostToDevice),
            "copy final-gap-offset canary");
        invalid_canary("final-gap-offset", [&]() {
            launch_w2a8_raw(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows, valid_grid,
                dim3(W2A8_PROBE_THREADS, 1, 1), W2A8_FIXED_N,
                W2A8_FIXED_K, 2, 1);
        });
        invalid_canary("null-inputs", [&]() {
            launch_w2a8_raw(
                nullptr, nullptr, nullptr, nullptr, nullptr, num_experts,
                rows_capacity, valid_grid, dim3(W2A8_PROBE_THREADS, 1, 1),
                W2A8_FIXED_N, W2A8_FIXED_K, 2, 1);
        });
        cuda_check(
            cudaMemcpy(d_offsets, offsets.data(),
                       offsets.size() * sizeof(offsets[0]),
                       cudaMemcpyHostToDevice),
            "restore valid offsets");
#endif

        quantize(rows);
        const std::size_t scale_count =
            (std::size_t)rows * (W2A8_FIXED_K / 128);
        std::vector<unsigned char> activation_fp8((std::size_t)rows * W2A8_FIXED_K);
        std::vector<float> activation_scales(scale_count);
        cuda_check(
            cudaMemcpy(activation_fp8.data(), d_a8, activation_fp8.size(),
                       cudaMemcpyDeviceToHost),
            "copy activation FP8 telemetry");
        cuda_check(
            cudaMemcpy(activation_scales.data(), d_scales,
                       activation_scales.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "copy activation scale telemetry");
        std::size_t saturated_values = 0;
        std::size_t floor_scale_groups = 0;
        for (unsigned char value : activation_fp8) {
            saturated_values += (value & 0x7f) == 0x7e;
            passed &= (value & 0x7f) != 0x7f;
        }
        for (float scale : activation_scales) {
            floor_scale_groups += scale == 1.0e-12f;
            passed &= std::isfinite(scale) && scale >= 1.0e-12f;
        }
        cuda_check(
            cudaMemset(d_baseline_storage, 0x3c, guarded_output_bytes),
            "poison baseline output");
        launch_baseline(
            d_input, d_trellis_table, d_baseline, d_offsets, num_experts);
        cuda_check(cudaDeviceSynchronize(), "baseline synchronize");
        std::vector<unsigned char> baseline_bytes(guarded_output_bytes);
        cuda_check(
            cudaMemcpy(
                baseline_bytes.data(), d_baseline_storage,
                baseline_bytes.size(), cudaMemcpyDeviceToHost),
            "copy guarded baseline");

        auto guarded_candidate = [&](unsigned char poison) {
            cuda_check(
                cudaMemset(d_w2a8_storage, poison, guarded_output_bytes),
                "poison guarded W2A8 output");
            launch_w2a8(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows);
            cuda_check(
                cudaDeviceSynchronize(), "guarded W2A8 synchronize");
            std::vector<unsigned char> result(guarded_output_bytes);
            cuda_check(
                cudaMemcpy(
                    result.data(), d_w2a8_storage, result.size(),
                    cudaMemcpyDeviceToHost),
                "copy guarded W2A8");
            for (std::size_t index = 0; index < guard_bytes; ++index) {
                if (result[index] != poison) passed = false;
            }
            for (std::size_t index = guard_bytes + active_bytes;
                 index < result.size(); ++index) {
                if (result[index] != poison) passed = false;
            }
            return result;
        };
        std::vector<unsigned char> w2a8_a5 = guarded_candidate(0xa5);
        std::vector<unsigned char> w2a8_5a = guarded_candidate(0x5a);
        for (std::size_t index = 0; index < active_bytes; ++index) {
            if (w2a8_a5[guard_bytes + index] !=
                w2a8_5a[guard_bytes + index]) {
                passed = false;
            }
        }
        for (std::size_t index = 0; index < guard_bytes; ++index) {
            if (baseline_bytes[index] != 0x3c) passed = false;
        }
        for (std::size_t index = guard_bytes + active_bytes;
             index < baseline_bytes.size(); ++index) {
            if (baseline_bytes[index] != 0x3c) passed = false;
        }

        std::vector<std::uint16_t> baseline(active_elements);
        std::vector<std::uint16_t> w2a8(active_elements);
        std::memcpy(
            baseline.data(), baseline_bytes.data() + guard_bytes, active_bytes);
        std::memcpy(
            w2a8.data(), w2a8_a5.data() + guard_bytes, active_bytes);
        if (dump != nullptr) {
            const std::uint32_t header[] = {
                (std::uint32_t)rows,
                (std::uint32_t)num_experts,
                (std::uint32_t)active_bytes,
            };
            if (std::fwrite(header, sizeof(header), 1, dump) != 1 ||
                std::fwrite(w2a8.data(), active_bytes, 1, dump) != 1) {
                std::perror("write W2A8 probe dump");
                std::exit(2);
            }
        }
        double dot = 0.0;
        double norm0 = 0.0;
        double norm1 = 0.0;
        float max_abs = 0.0f;
        for (std::size_t index = 0; index < active_elements; ++index) {
            const float lhs = bf16_to_f32(baseline[index]);
            const float rhs = bf16_to_f32(w2a8[index]);
            passed &= std::isfinite(lhs) && std::isfinite(rhs);
            dot += (double)lhs * rhs;
            norm0 += (double)lhs * lhs;
            norm1 += (double)rhs * rhs;
            max_abs = std::fmax(max_abs, std::fabs(lhs - rhs));
        }
        const double cosine = dot / std::sqrt(norm0 * norm1);
        if (!std::isfinite(cosine) || cosine < min_cosine) passed = false;
        if (max_abs > max_abs_error) passed = false;

        const float baseline_ms0 = time_launch([&]() {
            launch_baseline(
                d_input, d_trellis_table, d_baseline, d_offsets, num_experts);
        });
        const float end_to_end_ms0 = time_launch([&]() {
            quantize(rows);
            launch_w2a8(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows);
        });
        const float end_to_end_ms1 = time_launch([&]() {
            quantize(rows);
            launch_w2a8(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows);
        });
        const float baseline_ms1 = time_launch([&]() {
            launch_baseline(
                d_input, d_trellis_table, d_baseline, d_offsets, num_experts);
        });
        const float quant_ms = time_launch([&]() { quantize(rows); });
        const float kernel_ms = time_launch([&]() {
            launch_w2a8(
                d_a8, d_scales, d_trellis_table, d_w2a8, d_offsets,
                num_experts, rows);
        });
        const float baseline_ms = 0.5f * (baseline_ms0 + baseline_ms1);
        const float end_to_end_ms =
            0.5f * (end_to_end_ms0 + end_to_end_ms1);
        const float speedup = baseline_ms / end_to_end_ms;
        if (!std::isfinite(baseline_ms) || baseline_ms <= 0.0f ||
            !std::isfinite(end_to_end_ms) || end_to_end_ms <= 0.0f ||
            !std::isfinite(quant_ms) || quant_ms <= 0.0f ||
            !std::isfinite(kernel_ms) || kernel_ms <= 0.0f ||
            !std::isfinite(speedup) ||
            speedup < min_end_to_end_speedup) {
            passed = false;
        }
        std::printf(
            "case=%s rows=%d experts=%u cosine=%.9f max_abs=%.9g "
            "baseline_ms=%.6f quant_ms=%.6f w2a8_kernel_ms=%.6f "
            "end_to_end_ms=%.6f speedup=%.6f baseline_hash=%016llx "
            "w2a8_hash=%016llx a8_saturated=%zu/%zu floor_scales=%zu/%zu\n",
            label, rows, num_experts, cosine, max_abs, baseline_ms, quant_ms,
            kernel_ms, end_to_end_ms, speedup,
            (unsigned long long)fnv1a(baseline),
            (unsigned long long)fnv1a(w2a8), saturated_values,
            activation_fp8.size(), floor_scale_groups, scale_count);
    };

    for (int rows : {1, 63, 64, 65, 127, 128, 129}) {
        run_case({0, rows}, "single");
    }
    run_case({0, 0, 63, 63, 129}, "multi-with-empty");
    run_case({0, 63, 129, 129, 129}, "multi-trailing-empty");
    std::printf(
        "input_hash=%016llx trellis_hash=%016llx result=%s\n",
        (unsigned long long)fnv1a(input),
        (unsigned long long)fnv1a(trellis), passed ? "PASS" : "FAIL");
    if (dump != nullptr && std::fclose(dump) != 0) {
        std::perror("close W2A8 probe dump");
        return 2;
    }

    cuda_check(cudaFree(d_w2a8_storage), "cudaFree W2A8 output");
    cuda_check(cudaFree(d_baseline_storage), "cudaFree baseline output");
    cuda_check(cudaFree(d_offsets), "cudaFree offsets");
    cuda_check(cudaFree(d_trellis_table), "cudaFree trellis table");
    cuda_check(cudaFree(d_trellis), "cudaFree trellis");
    cuda_check(cudaFree(d_scales), "cudaFree scales");
    cuda_check(cudaFree(d_a8), "cudaFree A8");
    cuda_check(cudaFree(d_input), "cudaFree input");
    return passed ? 0 : 1;
}
