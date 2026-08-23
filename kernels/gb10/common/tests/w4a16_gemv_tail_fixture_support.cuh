// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

inline void cuda_check(cudaError_t status, const char* expr) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expr) + ": " + cudaGetErrorString(status));
    }
}

#define CUDA_CHECK(EXPR) cuda_check((EXPR), #EXPR)

constexpr unsigned int FIXTURE_K = 2048;
constexpr unsigned int FIXTURE_MAX_N = 8;
constexpr unsigned int GUARD_ELEMS = 32;
constexpr unsigned short CANARY = 0xA5A5;

template <typename T>
struct Guarded {
    T* base = nullptr;
    T* data = nullptr;
    size_t count;

    explicit Guarded(size_t elements) : count(elements) {
        CUDA_CHECK(cudaMalloc(&base, (count + 2 * GUARD_ELEMS) * sizeof(T)));
        CUDA_CHECK(cudaMemset(base, 0xA5, (count + 2 * GUARD_ELEMS) * sizeof(T)));
        data = base + GUARD_ELEMS;
    }
    ~Guarded() { cudaFree(base); }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;

    std::vector<unsigned short> copy_bf16() const {
        static_assert(sizeof(T) == sizeof(unsigned short));
        std::vector<unsigned short> host(count + 2 * GUARD_ELEMS);
        CUDA_CHECK(cudaMemcpy(host.data(), base, host.size() * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }

    void verify_canaries(const std::vector<bool>& written, const std::string& label) const {
        const auto host = copy_bf16();
        for (size_t i = 0; i < GUARD_ELEMS; ++i) {
            if (host[i] != CANARY || host[GUARD_ELEMS + count + i] != CANARY) {
                throw std::runtime_error(label + ": outer canary changed");
            }
        }
        for (size_t i = 0; i < count; ++i) {
            if (!written[i] && host[GUARD_ELEMS + i] != CANARY) {
                throw std::runtime_error(label + ": unwritten output changed at " + std::to_string(i));
            }
        }
    }

    std::vector<unsigned short> normalize(unsigned int rows, unsigned int stride,
                                           unsigned int n) const {
        const auto host = copy_bf16();
        std::vector<unsigned short> out;
        for (unsigned int row = 0; row < rows; ++row) {
            for (unsigned int col = 0; col < n; ++col) {
                out.push_back(host[GUARD_ELEMS + row * stride + col]);
            }
        }
        return out;
    }
};

struct FixtureData {
    __nv_bfloat16* a = nullptr;
    unsigned char* packed = nullptr;
    unsigned char* scales = nullptr;

    FixtureData() {
        std::vector<__nv_bfloat16> host_a(3 * FIXTURE_K);
        for (size_t i = 0; i < host_a.size(); ++i) {
            const float value = static_cast<float>(static_cast<int>(i % 29) - 14) / 16.0f;
            host_a[i] = __float2bfloat16(value);
        }
        std::vector<unsigned char> host_packed(FIXTURE_MAX_N * FIXTURE_K / 2);
        for (size_t i = 0; i < host_packed.size(); ++i) {
            const unsigned char lo = static_cast<unsigned char>(1 + i % 7);
            const unsigned char hi = static_cast<unsigned char>(1 + (i / 7) % 7);
            host_packed[i] = static_cast<unsigned char>(lo | (hi << 4));
        }
        std::vector<unsigned char> host_scales(FIXTURE_MAX_N * FIXTURE_K / 16, 0x38);
        CUDA_CHECK(cudaMalloc(&a, host_a.size() * sizeof(*a)));
        CUDA_CHECK(cudaMalloc(&packed, host_packed.size()));
        CUDA_CHECK(cudaMalloc(&scales, host_scales.size()));
        CUDA_CHECK(cudaMemcpy(a, host_a.data(), host_a.size() * sizeof(*a), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(packed, host_packed.data(), host_packed.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(scales, host_scales.data(), host_scales.size(), cudaMemcpyHostToDevice));
    }
    ~FixtureData() {
        cudaFree(a);
        cudaFree(packed);
        cudaFree(scales);
    }
};

enum class Kind {
    Batch2, Qg, Qkvz, QgBatch2, DualBatch2, Batch3, QgBatch3, DualBatch3,
    V1, V3, Batch3Logits, V4, QgBatch3Strided, DualBatch3Strided, DualBatch3Tuned,
};

struct CaseSpec {
    const char* name;
    Kind kind;
    unsigned int width;
    unsigned int rows;
    unsigned int projections;
    bool strided;
};

inline const CaseSpec CASES[] = {
    {"batch2", Kind::Batch2, 4, 2, 1, false},
    {"qg", Kind::Qg, 4, 1, 1, false},
    {"qkvz", Kind::Qkvz, 4, 1, 1, false},
    {"qg_batch2", Kind::QgBatch2, 4, 2, 1, false},
    {"dual_batch2", Kind::DualBatch2, 4, 2, 2, false},
    {"batch3", Kind::Batch3, 4, 3, 1, false},
    {"qg_batch3", Kind::QgBatch3, 4, 3, 1, false},
    {"dual_batch3", Kind::DualBatch3, 4, 3, 2, false},
    {"v1", Kind::V1, 2, 1, 1, false},
    {"v3", Kind::V3, 8, 1, 1, false},
    {"batch3_logits", Kind::Batch3Logits, 8, 3, 1, false},
    {"v4", Kind::V4, 2, 1, 1, false},
    {"qg_batch3_strided", Kind::QgBatch3Strided, 4, 3, 1, true},
    {"dual_batch3_strided", Kind::DualBatch3Strided, 4, 3, 2, true},
    {"dual_batch3_tuned", Kind::DualBatch3Tuned, 4, 3, 2, false},
};

inline std::map<std::string, std::vector<unsigned short>> read_oracle(const char* path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open oracle");
    std::map<std::string, std::vector<unsigned short>> result;
    std::string name;
    size_t count;
    while (input >> name >> count) {
        auto& values = result[name];
        values.resize(count);
        for (auto& value : values) {
            unsigned int parsed;
            input >> std::hex >> parsed >> std::dec;
            value = static_cast<unsigned short>(parsed);
        }
    }
    return result;
}

inline void write_record(std::ofstream& output, const CaseSpec& spec,
                         const std::vector<unsigned short>& values) {
    output << spec.name << ' ' << values.size();
    for (const auto value : values) {
        output << ' ' << std::hex << std::setw(4) << std::setfill('0') << value << std::dec;
    }
    output << '\n';
}

