// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

namespace oi640_gate {
constexpr size_t GUARD = 4096;
constexpr unsigned char CANARY = 0xa5;

template <typename T> struct Guarded {
    unsigned char* allocation = nullptr;
    T* data = nullptr;
    size_t count = 0;
    bool readonly = false;
    std::vector<T> initial;
    Guarded() = default;
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    ~Guarded() { if (allocation != nullptr) cudaFree(allocation); }
    void allocate(size_t n, bool ro = false) {
        count = n; readonly = ro;
        CUDA_OK(cudaMalloc(&allocation, GUARD + count * sizeof(T) + GUARD));
        CUDA_OK(cudaMemset(allocation, CANARY, GUARD + count * sizeof(T) + GUARD));
        data = reinterpret_cast<T*>(allocation + GUARD);
    }
    void upload(const std::vector<T>& host, bool remember = true) {
        if (host.size() != count) std::abort();
        CUDA_OK(cudaMemcpy(data, host.data(), count * sizeof(T), cudaMemcpyHostToDevice));
        if (readonly && remember) initial = host;
    }
    void fill(unsigned char value) { CUDA_OK(cudaMemset(data, value, count * sizeof(T))); }
    void fill_async(unsigned char value, cudaStream_t stream) {
        CUDA_OK(cudaMemsetAsync(data, value, count * sizeof(T), stream));
    }
    std::vector<T> download() const {
        std::vector<T> out(count);
        CUDA_OK(cudaMemcpy(out.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }
    bool safe() const {
        std::vector<unsigned char> edges(GUARD * 2);
        CUDA_OK(cudaMemcpy(edges.data(), allocation, GUARD, cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(edges.data() + GUARD, allocation + GUARD + count * sizeof(T),
                           GUARD, cudaMemcpyDeviceToHost));
        const bool canaries = std::all_of(edges.begin(), edges.end(),
            [](unsigned char v) { return v == CANARY; });
        if (!readonly) return canaries;
        const auto current = download();
        return canaries && current.size() == initial.size() &&
            std::memcmp(current.data(), initial.data(), current.size() * sizeof(T)) == 0;
    }
};

uint32_t random_word() {
    static uint32_t state = 0x8d739a51U;
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}

std::vector<unsigned char> random_bytes(size_t count, uint32_t salt) {
    std::vector<unsigned char> result(count);
    for (size_t i = 0; i < count; ++i)
        result[i] = static_cast<unsigned char>((random_word() + salt + i * 17) & 255);
    return result;
}

std::vector<unsigned char> finite_scale_bytes(size_t count, uint32_t salt) {
    auto result = random_bytes(count, salt);
    for (auto& value : result) value = static_cast<unsigned char>(1 + value % 125);
    return result;
}

struct Weights {
    static constexpr size_t PACKED_ONE = 819200;
    static constexpr size_t SCALE_ONE = 102400;
    Guarded<unsigned char> gate_packed, gate_scale, up_packed, up_scale, down_packed, down_scale;
    Guarded<unsigned long long> gate_pt, gate_st, up_pt, up_st, down_pt, down_st;
    Guarded<float> gate_s2, up_s2, down_s2;

    Weights() {
        const size_t packed = PACKED_ONE * oi640::EXPERTS;
        const size_t scales = SCALE_ONE * oi640::EXPERTS;
        gate_packed.allocate(packed, true); gate_scale.allocate(scales, true);
        up_packed.allocate(packed, true); up_scale.allocate(scales, true);
        down_packed.allocate(packed, true); down_scale.allocate(scales, true);
        gate_packed.upload(random_bytes(packed, 11)); gate_scale.upload(finite_scale_bytes(scales, 17));
        up_packed.upload(random_bytes(packed, 23)); up_scale.upload(finite_scale_bytes(scales, 29));
        down_packed.upload(random_bytes(packed, 31)); down_scale.upload(finite_scale_bytes(scales, 37));
        make_table(gate_pt, gate_packed.data, PACKED_ONE); make_table(gate_st, gate_scale.data, SCALE_ONE);
        make_table(up_pt, up_packed.data, PACKED_ONE); make_table(up_st, up_scale.data, SCALE_ONE);
        make_table(down_pt, down_packed.data, PACKED_ONE); make_table(down_st, down_scale.data, SCALE_ONE);
        make_scale2(gate_s2, 0.0078125f); make_scale2(up_s2, 0.009765625f);
        make_scale2(down_s2, 0.01171875f);
    }
    static void make_table(Guarded<unsigned long long>& table, unsigned char* base, size_t stride) {
        table.allocate(oi640::EXPERTS, true);
        std::vector<unsigned long long> host(oi640::EXPERTS);
        for (size_t i = 0; i < host.size(); ++i)
            host[i] = reinterpret_cast<unsigned long long>(base + i * stride);
        table.upload(host);
    }
    static void make_scale2(Guarded<float>& values, float base) {
        values.allocate(oi640::EXPERTS, true);
        std::vector<float> host(oi640::EXPERTS);
        for (size_t i = 0; i < host.size(); ++i) host[i] = base + float(i % 7) / 4096.0f;
        values.upload(host);
    }
    bool safe() const {
        return gate_packed.safe() && gate_scale.safe() && up_packed.safe() && up_scale.safe() &&
            down_packed.safe() && down_scale.safe() && gate_pt.safe() && gate_st.safe() &&
            up_pt.safe() && up_st.safe() && down_pt.safe() && down_st.safe() &&
            gate_s2.safe() && up_s2.safe() && down_s2.safe();
    }
};

} // namespace oi640_gate
