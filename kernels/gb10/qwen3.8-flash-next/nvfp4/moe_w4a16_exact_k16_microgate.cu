// SPDX-License-Identifier: AGPL-3.0-only

// GPU-ready raw gate for moe_w4a16_exact_k16.cu.  This is intentionally not
// part of Atlas's kernel manifest or production route.  Build/static checks
// are safe on a CPU host; executing this binary requires an explicitly
// reserved sm_121 device.

#include "moe_w4a16_exact_k16.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

// These are injected from the build command after hashing the exact inputs.
// An ad-hoc/unsealed build still compiles for source reuse, but this gate will
// fail rather than emit PASS with unbound source provenance.
#ifndef FLASH_NEXT_EXACT_SOURCE_SHA256
#define FLASH_NEXT_EXACT_SOURCE_SHA256 "UNBOUND"
#endif
#ifndef FLASH_NEXT_MICROGATE_SOURCE_SHA256
#define FLASH_NEXT_MICROGATE_SOURCE_SHA256 "UNBOUND"
#endif
#ifndef FLASH_NEXT_PARENT_BATCH_SHA256
#define FLASH_NEXT_PARENT_BATCH_SHA256 "UNBOUND"
#endif
#ifndef FLASH_NEXT_PARENT_SERIAL_SHA256
#define FLASH_NEXT_PARENT_SERIAL_SHA256 "UNBOUND"
#endif
#ifndef FLASH_NEXT_PARENT_BLEND_SHA256
#define FLASH_NEXT_PARENT_BLEND_SHA256 "UNBOUND"
#endif

#define CUDA_OK(expr) do { \
    cudaError_t e_ = (expr); \
    if (e_ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA failure %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
        std::exit(2); \
    } \
} while (0)

// Frozen ordinary parent (arbitrary-token batch3 ABI).
extern "C" __global__ void moe_expert_gate_up_shared_batch3(
    const __nv_bfloat16*, const unsigned long long*, const unsigned long long*, const float*,
    __nv_bfloat16*, const unsigned long long*, const unsigned long long*, const float*,
    __nv_bfloat16*, const unsigned int*, const unsigned char*, const unsigned char*, float,
    __nv_bfloat16*, const unsigned char*, const unsigned char*, float, __nv_bfloat16*,
    unsigned int, unsigned int, unsigned int, unsigned int);
extern "C" __global__ void moe_expert_silu_down_shared_batch3(
    const __nv_bfloat16*, const __nv_bfloat16*, const unsigned long long*,
    const unsigned long long*, const float*, __nv_bfloat16*, const unsigned int*,
    const __nv_bfloat16*, const __nv_bfloat16*, const unsigned char*,
    const unsigned char*, float, __nv_bfloat16*, unsigned int, unsigned int,
    unsigned int, unsigned int);
extern "C" __global__ void moe_weighted_sum_blend_batch3(
    __nv_bfloat16*, const __nv_bfloat16*, const float*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, unsigned int, unsigned int,
    unsigned int);

// Frozen ordinary serial-K1 ABI.
extern "C" __global__ void moe_expert_gate_up_shared(
    const __nv_bfloat16*, const unsigned long long*, const unsigned long long*, const float*,
    __nv_bfloat16*, const unsigned long long*, const unsigned long long*, const float*,
    __nv_bfloat16*, const unsigned int*, const unsigned char*, const unsigned char*, float,
    __nv_bfloat16*, const unsigned char*, const unsigned char*, float, __nv_bfloat16*,
    unsigned int, unsigned int, unsigned int);
extern "C" __global__ void moe_expert_silu_down_shared(
    const __nv_bfloat16*, const __nv_bfloat16*, const unsigned long long*,
    const unsigned long long*, const float*, __nv_bfloat16*, const unsigned int*,
    const __nv_bfloat16*, const __nv_bfloat16*, const unsigned char*,
    const unsigned char*, float, __nv_bfloat16*, unsigned int, unsigned int,
    unsigned int);
extern "C" __global__ void moe_weighted_sum_blend(
    __nv_bfloat16*, const __nv_bfloat16*, const float*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, unsigned int, unsigned int,
    unsigned int);

namespace {

using namespace flash_next_exact_k16;
constexpr size_t GUARD = 4096;
constexpr unsigned char GUARD_BYTE = 0xa5;
constexpr unsigned char OUTPUT_BYTE = 0xcd;
constexpr unsigned int WEIGHT_VARIANTS = 8;

bool digest_bound(const char* digest) {
    if (std::strlen(digest) != 64) return false;
    return std::all_of(digest, digest + 64, [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

std::string executable_sha256() {
    // Resolve `self` in this process, not in the sha256sum child (where
    // `/proc/self/exe` would incorrectly name sha256sum itself).
    char command[96]{};
    const int written = std::snprintf(
        command, sizeof(command), "/usr/bin/sha256sum /proc/%ld/exe",
        static_cast<long>(getpid()));
    if (written <= 0 || static_cast<size_t>(written) >= sizeof(command)) return {};
    FILE* pipe = popen(command, "r");
    if (pipe == nullptr) return {};
    char digest[65]{};
    const int scanned = std::fscanf(pipe, "%64[0-9a-f]", digest);
    const int closed = pclose(pipe);
    if (scanned != 1 || closed != 0 || !digest_bound(digest)) return {};
    return digest;
}

template <typename T>
struct Guarded {
    unsigned char* allocation = nullptr;
    T* data = nullptr;
    size_t count = 0;
    std::vector<T> initial;
    bool readonly = false;

    Guarded() = default;
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    Guarded(Guarded&& other) noexcept { *this = std::move(other); }
    Guarded& operator=(Guarded&& other) noexcept {
        allocation = other.allocation; data = other.data; count = other.count;
        initial = std::move(other.initial); readonly = other.readonly;
        other.allocation = nullptr; other.data = nullptr; other.count = 0;
        return *this;
    }
    ~Guarded() { if (allocation != nullptr) cudaFree(allocation); }

    void allocate(size_t n, bool is_readonly = false) {
        count = n; readonly = is_readonly;
        CUDA_OK(cudaMalloc(&allocation, GUARD + n * sizeof(T) + GUARD));
        CUDA_OK(cudaMemset(allocation, GUARD_BYTE, GUARD + n * sizeof(T) + GUARD));
        data = reinterpret_cast<T*>(allocation + GUARD);
    }
    void upload(const std::vector<T>& host, bool remember = true) {
        if (host.size() != count) std::abort();
        CUDA_OK(cudaMemcpy(data, host.data(), count * sizeof(T), cudaMemcpyHostToDevice));
        if (remember) initial = host;
    }
    void fill(unsigned char byte) { CUDA_OK(cudaMemset(data, byte, count * sizeof(T))); }
    std::vector<T> download() const {
        std::vector<T> out(count);
        CUDA_OK(cudaMemcpy(out.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }
    bool canary_ok(const char* name) const {
        std::vector<unsigned char> lo(GUARD), hi(GUARD);
        CUDA_OK(cudaMemcpy(lo.data(), allocation, GUARD, cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(hi.data(), allocation + GUARD + count * sizeof(T), GUARD,
                           cudaMemcpyDeviceToHost));
        const bool ok = std::all_of(lo.begin(), lo.end(), [](unsigned char v) { return v == GUARD_BYTE; }) &&
                        std::all_of(hi.begin(), hi.end(), [](unsigned char v) { return v == GUARD_BYTE; });
        if (!ok) std::fprintf(stderr, "canary drift: %s\n", name);
        return ok;
    }
    bool immutable_ok(const char* name) const {
        if (!readonly) return true;
        const auto got = download();
        const bool ok = got.size() == initial.size() &&
            std::memcmp(got.data(), initial.data(), got.size() * sizeof(T)) == 0;
        if (!ok) std::fprintf(stderr, "immutable input drift: %s\n", name);
        return ok;
    }
};

uint32_t rng_state = 0x62ad91f3u;
uint32_t next_u32() {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}

std::vector<unsigned char> packed_values(size_t n, unsigned int salt) {
    rng_state ^= salt * 0x9e3779b9u;
    std::vector<unsigned char> v(n);
    for (auto& x : v) x = static_cast<unsigned char>(next_u32());
    return v;
}

std::vector<unsigned char> scale_values(size_t n, unsigned int salt) {
    static constexpr unsigned char finite_fp8[] = {0x28, 0x30, 0x34, 0x38, 0x3c, 0x40};
    std::vector<unsigned char> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = finite_fp8[(i * 7 + salt * 3) % 6];
    return v;
}

std::vector<__nv_bfloat16> bf16_values(size_t n, unsigned int salt) {
    std::vector<__nv_bfloat16> v(n);
    for (size_t i = 0; i < n; ++i) {
        const int raw = static_cast<int>((i * 37 + salt * 101) % 257) - 128;
        const float x = static_cast<float>(raw) / 192.0f;
        v[i] = __float2bfloat16(x);
    }
    return v;
}

struct WeightBank {
    std::vector<Guarded<unsigned char>> gate_packed, gate_scale;
    std::vector<Guarded<unsigned char>> up_packed, up_scale;
    std::vector<Guarded<unsigned char>> down_packed, down_scale;
    Guarded<unsigned long long> gate_packed_table, gate_scale_table;
    Guarded<unsigned long long> up_packed_table, up_scale_table;
    Guarded<unsigned long long> down_packed_table, down_scale_table;
    Guarded<float> gate_s2, up_s2, down_s2;
    Guarded<unsigned char> sh_gate_packed, sh_gate_scale;
    Guarded<unsigned char> sh_up_packed, sh_up_scale;
    Guarded<unsigned char> sh_down_packed, sh_down_scale;
    float sh_gate_s2 = 0.75f, sh_up_s2 = 0.625f, sh_down_s2 = 0.875f;

    WeightBank() {
        constexpr size_t GP = static_cast<size_t>(INTER) * HIDDEN / 2;
        constexpr size_t GPS = static_cast<size_t>(INTER) * HIDDEN / GROUP;
        constexpr size_t DP = static_cast<size_t>(HIDDEN) * INTER / 2;
        constexpr size_t DPS = static_cast<size_t>(HIDDEN) * INTER / GROUP;
        gate_packed.reserve(WEIGHT_VARIANTS); gate_scale.reserve(WEIGHT_VARIANTS);
        up_packed.reserve(WEIGHT_VARIANTS); up_scale.reserve(WEIGHT_VARIANTS);
        down_packed.reserve(WEIGHT_VARIANTS); down_scale.reserve(WEIGHT_VARIANTS);
        for (unsigned int i = 0; i < WEIGHT_VARIANTS; ++i) {
            gate_packed.emplace_back(); gate_packed.back().allocate(GP, true);
            gate_packed.back().upload(packed_values(GP, 11 + i));
            gate_scale.emplace_back(); gate_scale.back().allocate(GPS, true);
            gate_scale.back().upload(scale_values(GPS, 17 + i));
            up_packed.emplace_back(); up_packed.back().allocate(GP, true);
            up_packed.back().upload(packed_values(GP, 23 + i));
            up_scale.emplace_back(); up_scale.back().allocate(GPS, true);
            up_scale.back().upload(scale_values(GPS, 29 + i));
            down_packed.emplace_back(); down_packed.back().allocate(DP, true);
            down_packed.back().upload(packed_values(DP, 31 + i));
            down_scale.emplace_back(); down_scale.back().allocate(DPS, true);
            down_scale.back().upload(scale_values(DPS, 37 + i));
        }
        std::vector<unsigned long long> gp(EXPERTS), gs(EXPERTS), up(EXPERTS), us(EXPERTS),
                                        dp(EXPERTS), ds(EXPERTS);
        std::vector<float> g2(EXPERTS), u2(EXPERTS), d2(EXPERTS);
        for (unsigned int e = 0; e < EXPERTS; ++e) {
            const unsigned int v = (e * 13 + e / 7) % WEIGHT_VARIANTS;
            gp[e] = reinterpret_cast<unsigned long long>(gate_packed[v].data);
            gs[e] = reinterpret_cast<unsigned long long>(gate_scale[v].data);
            up[e] = reinterpret_cast<unsigned long long>(up_packed[v].data);
            us[e] = reinterpret_cast<unsigned long long>(up_scale[v].data);
            dp[e] = reinterpret_cast<unsigned long long>(down_packed[v].data);
            ds[e] = reinterpret_cast<unsigned long long>(down_scale[v].data);
            g2[e] = 0.5f + 0.03125f * static_cast<float>(v);
            u2[e] = 0.625f + 0.03125f * static_cast<float>(v);
            d2[e] = 0.75f + 0.03125f * static_cast<float>(v);
        }
        gate_packed_table.allocate(EXPERTS, true); gate_packed_table.upload(gp);
        gate_scale_table.allocate(EXPERTS, true); gate_scale_table.upload(gs);
        up_packed_table.allocate(EXPERTS, true); up_packed_table.upload(up);
        up_scale_table.allocate(EXPERTS, true); up_scale_table.upload(us);
        down_packed_table.allocate(EXPERTS, true); down_packed_table.upload(dp);
        down_scale_table.allocate(EXPERTS, true); down_scale_table.upload(ds);
        gate_s2.allocate(EXPERTS, true); gate_s2.upload(g2);
        up_s2.allocate(EXPERTS, true); up_s2.upload(u2);
        down_s2.allocate(EXPERTS, true); down_s2.upload(d2);
        sh_gate_packed.allocate(GP, true); sh_gate_packed.upload(packed_values(GP, 41));
        sh_gate_scale.allocate(GPS, true); sh_gate_scale.upload(scale_values(GPS, 43));
        sh_up_packed.allocate(GP, true); sh_up_packed.upload(packed_values(GP, 47));
        sh_up_scale.allocate(GPS, true); sh_up_scale.upload(scale_values(GPS, 53));
        sh_down_packed.allocate(DP, true); sh_down_packed.upload(packed_values(DP, 59));
        sh_down_scale.allocate(DPS, true); sh_down_scale.upload(scale_values(DPS, 61));
    }

    bool validate() const {
        bool ok = true;
        for (unsigned int i = 0; i < WEIGHT_VARIANTS; ++i) {
            ok &= gate_packed[i].canary_ok("gate_packed") && gate_packed[i].immutable_ok("gate_packed");
            ok &= gate_scale[i].canary_ok("gate_scale") && gate_scale[i].immutable_ok("gate_scale");
            ok &= up_packed[i].canary_ok("up_packed") && up_packed[i].immutable_ok("up_packed");
            ok &= up_scale[i].canary_ok("up_scale") && up_scale[i].immutable_ok("up_scale");
            ok &= down_packed[i].canary_ok("down_packed") && down_packed[i].immutable_ok("down_packed");
            ok &= down_scale[i].canary_ok("down_scale") && down_scale[i].immutable_ok("down_scale");
        }
        ok &= gate_packed_table.canary_ok("gate_packed_table") && gate_packed_table.immutable_ok("gate_packed_table");
        ok &= gate_scale_table.canary_ok("gate_scale_table") && gate_scale_table.immutable_ok("gate_scale_table");
        ok &= up_packed_table.canary_ok("up_packed_table") && up_packed_table.immutable_ok("up_packed_table");
        ok &= up_scale_table.canary_ok("up_scale_table") && up_scale_table.immutable_ok("up_scale_table");
        ok &= down_packed_table.canary_ok("down_packed_table") && down_packed_table.immutable_ok("down_packed_table");
        ok &= down_scale_table.canary_ok("down_scale_table") && down_scale_table.immutable_ok("down_scale_table");
        ok &= gate_s2.canary_ok("gate_s2") && gate_s2.immutable_ok("gate_s2");
        ok &= up_s2.canary_ok("up_s2") && up_s2.immutable_ok("up_s2");
        ok &= down_s2.canary_ok("down_s2") && down_s2.immutable_ok("down_s2");
        ok &= sh_gate_packed.canary_ok("sh_gate_packed") && sh_gate_packed.immutable_ok("sh_gate_packed");
        ok &= sh_gate_scale.canary_ok("sh_gate_scale") && sh_gate_scale.immutable_ok("sh_gate_scale");
        ok &= sh_up_packed.canary_ok("sh_up_packed") && sh_up_packed.immutable_ok("sh_up_packed");
        ok &= sh_up_scale.canary_ok("sh_up_scale") && sh_up_scale.immutable_ok("sh_up_scale");
        ok &= sh_down_packed.canary_ok("sh_down_packed") && sh_down_packed.immutable_ok("sh_down_packed");
        ok &= sh_down_scale.canary_ok("sh_down_scale") && sh_down_scale.immutable_ok("sh_down_scale");
        return ok;
    }
};

struct Outputs {
    Guarded<__nv_bfloat16> gate, up, shared_gate, shared_up, down, shared_down, final;
    Outputs() {
        gate.allocate(static_cast<size_t>(ROWS) * TOP_K * INTER);
        up.allocate(static_cast<size_t>(ROWS) * TOP_K * INTER);
        shared_gate.allocate(static_cast<size_t>(ROWS) * INTER);
        shared_up.allocate(static_cast<size_t>(ROWS) * INTER);
        down.allocate(static_cast<size_t>(ROWS) * TOP_K * HIDDEN);
        shared_down.allocate(static_cast<size_t>(ROWS) * HIDDEN);
        final.allocate(static_cast<size_t>(ROWS) * HIDDEN);
        reset();
    }
    void reset() {
        gate.fill(OUTPUT_BYTE); up.fill(OUTPUT_BYTE); shared_gate.fill(OUTPUT_BYTE);
        shared_up.fill(OUTPUT_BYTE); down.fill(OUTPUT_BYTE); shared_down.fill(OUTPUT_BYTE);
        final.fill(OUTPUT_BYTE);
    }
    bool canaries() const {
        return gate.canary_ok("out.gate") && up.canary_ok("out.up") &&
            shared_gate.canary_ok("out.shared_gate") && shared_up.canary_ok("out.shared_up") &&
            down.canary_ok("out.down") && shared_down.canary_ok("out.shared_down") &&
            final.canary_ok("out.final");
    }
};

struct Fixture {
    WeightBank weights;
    Guarded<__nv_bfloat16> input, shared_gate_weight;
    Guarded<unsigned int> indices;
    Guarded<float> expert_weights;
    Guarded<Contract> contract;
    Guarded<int> status;
    Outputs candidate, parent, serial, candidate_repeat;
    cudaStream_t stream = nullptr;

    Fixture() {
        input.allocate(static_cast<size_t>(ROWS) * HIDDEN, true);
        input.upload(bf16_values(input.count, 71));
        shared_gate_weight.allocate(HIDDEN, true);
        shared_gate_weight.upload(bf16_values(HIDDEN, 73));
        std::vector<unsigned int> ids(ROWS * TOP_K);
        std::vector<float> ew(ROWS * TOP_K);
        for (unsigned int r = 0; r < ROWS; ++r) {
            float total = 0.0f;
            for (unsigned int e = 0; e < TOP_K; ++e) {
                ids[r * TOP_K + e] = (r * 37 + e * 17 + (r % 3) * e) % EXPERTS;
                ew[r * TOP_K + e] = 1.0f + static_cast<float>((r * 11 + e * 7) % 19);
                total += ew[r * TOP_K + e];
            }
            for (unsigned int e = 0; e < TOP_K; ++e) ew[r * TOP_K + e] /= total;
        }
        indices.allocate(ids.size(), true); indices.upload(ids);
        expert_weights.allocate(ew.size(), true); expert_weights.upload(ew);
        Contract c{ABI_MAGIC, ABI_VERSION, ROWS, HIDDEN, INTER, EXPERTS, TOP_K, 0,
            static_cast<unsigned long long>(ROWS) * HIDDEN,
            static_cast<unsigned long long>(ROWS) * TOP_K * INTER,
            static_cast<unsigned long long>(ROWS) * TOP_K * HIDDEN,
            static_cast<unsigned long long>(ROWS) * INTER,
            static_cast<unsigned long long>(ROWS) * HIDDEN,
            static_cast<unsigned long long>(ROWS) * HIDDEN,
            static_cast<unsigned long long>(ROWS) * TOP_K,
            static_cast<unsigned long long>(ROWS) * TOP_K,
            EXPERTS, EXPERTS, 0};
        contract.allocate(1, true); contract.upload({c});
        status.allocate(1); status.upload({PENDING}, false);
        CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~Fixture() { if (stream != nullptr) cudaStreamDestroy(stream); }

    void arm_status() {
        CUDA_OK(cudaMemsetAsync(status.data, 0xff, sizeof(int), stream));
    }

    void preflight(Outputs& o, const Contract* c, const void* workspace,
                   __nv_bfloat16* final_override = nullptr, int* status_override = nullptr) {
        moe_w4a16_exact_k16_preflight<<<1, 1, 0, stream>>>(
            input.data, weights.gate_packed_table.data, weights.gate_scale_table.data,
            weights.gate_s2.data, o.gate.data, weights.up_packed_table.data,
            weights.up_scale_table.data, weights.up_s2.data, o.up.data, indices.data,
            weights.sh_gate_packed.data, weights.sh_gate_scale.data, o.shared_gate.data,
            weights.sh_up_packed.data, weights.sh_up_scale.data, o.shared_up.data,
            weights.down_packed_table.data, weights.down_scale_table.data, weights.down_s2.data,
            o.down.data, weights.sh_down_packed.data, weights.sh_down_scale.data,
            o.shared_down.data, expert_weights.data, shared_gate_weight.data,
            final_override == nullptr ? o.final.data : final_override, c, workspace,
            status_override == nullptr ? status.data : status_override);
    }

    void launch_candidate(Outputs& o) {
        arm_status();
        preflight(o, contract.data, nullptr);
        constexpr unsigned int GT = 2 * (ROWS * TOP_K + ROWS / WARPS) * (INTER / OUTPUTS_PER_CTA);
        constexpr unsigned int DT = (ROWS * TOP_K + ROWS / WARPS) * (HIDDEN / OUTPUTS_PER_CTA);
        moe_w4a16_exact_k16_gate_up<<<GT, 128, 0, stream>>>(
            input.data, weights.gate_packed_table.data, weights.gate_scale_table.data,
            weights.gate_s2.data, o.gate.data, weights.up_packed_table.data,
            weights.up_scale_table.data, weights.up_s2.data, o.up.data, indices.data,
            weights.sh_gate_packed.data, weights.sh_gate_scale.data, weights.sh_gate_s2,
            o.shared_gate.data, weights.sh_up_packed.data, weights.sh_up_scale.data,
            weights.sh_up_s2, o.shared_up.data, contract.data, status.data);
        moe_w4a16_exact_k16_silu_down<<<DT, 128, INTER * sizeof(float), stream>>>(
            o.gate.data, o.up.data, weights.down_packed_table.data, weights.down_scale_table.data,
            weights.down_s2.data, o.down.data, indices.data, o.shared_gate.data,
            o.shared_up.data, weights.sh_down_packed.data, weights.sh_down_scale.data,
            weights.sh_down_s2, o.shared_down.data, contract.data, status.data);
        moe_w4a16_exact_k16_weighted_sum_blend<<<dim3(HIDDEN / 256, ROWS, 1), 256, 0, stream>>>(
            o.final.data, o.down.data, expert_weights.data, o.shared_down.data, input.data,
            shared_gate_weight.data, contract.data, status.data);
    }

    void launch_parent(Outputs& o) {
        moe_expert_gate_up_shared_batch3<<<dim3(INTER / 8, ROWS * (TOP_K + 1), 2), 128, 0, stream>>>(
            input.data, weights.gate_packed_table.data, weights.gate_scale_table.data,
            weights.gate_s2.data, o.gate.data, weights.up_packed_table.data,
            weights.up_scale_table.data, weights.up_s2.data, o.up.data, indices.data,
            weights.sh_gate_packed.data, weights.sh_gate_scale.data, weights.sh_gate_s2,
            o.shared_gate.data, weights.sh_up_packed.data, weights.sh_up_scale.data,
            weights.sh_up_s2, o.shared_up.data, INTER, HIDDEN, TOP_K, ROWS);
        moe_expert_silu_down_shared_batch3<<<dim3(HIDDEN / 8, ROWS * (TOP_K + 1), 1),
            128, INTER * sizeof(float), stream>>>(
            o.gate.data, o.up.data, weights.down_packed_table.data, weights.down_scale_table.data,
            weights.down_s2.data, o.down.data, indices.data, o.shared_gate.data, o.shared_up.data,
            weights.sh_down_packed.data, weights.sh_down_scale.data, weights.sh_down_s2,
            o.shared_down.data, HIDDEN, INTER, TOP_K, ROWS);
        moe_weighted_sum_blend_batch3<<<dim3(HIDDEN / 256, ROWS, 1), 256, 0, stream>>>(
            o.final.data, o.down.data, expert_weights.data, o.shared_down.data, input.data,
            shared_gate_weight.data, HIDDEN, TOP_K, HIDDEN);
    }

    void launch_serial(Outputs& o) {
        for (unsigned int r = 0; r < ROWS; ++r) {
            const size_t ri = static_cast<size_t>(r) * TOP_K * INTER;
            const size_t rh = static_cast<size_t>(r) * TOP_K * HIDDEN;
            moe_expert_gate_up_shared<<<dim3(INTER / 8, TOP_K + 1, 2), 128, 0, stream>>>(
                input.data + static_cast<size_t>(r) * HIDDEN,
                weights.gate_packed_table.data, weights.gate_scale_table.data, weights.gate_s2.data,
                o.gate.data + ri, weights.up_packed_table.data, weights.up_scale_table.data,
                weights.up_s2.data, o.up.data + ri, indices.data + r * TOP_K,
                weights.sh_gate_packed.data, weights.sh_gate_scale.data, weights.sh_gate_s2,
                o.shared_gate.data + static_cast<size_t>(r) * INTER,
                weights.sh_up_packed.data, weights.sh_up_scale.data, weights.sh_up_s2,
                o.shared_up.data + static_cast<size_t>(r) * INTER, INTER, HIDDEN, TOP_K);
            moe_expert_silu_down_shared<<<dim3(HIDDEN / 8, TOP_K + 1, 1), 128,
                INTER * sizeof(float), stream>>>(
                o.gate.data + ri, o.up.data + ri, weights.down_packed_table.data,
                weights.down_scale_table.data, weights.down_s2.data, o.down.data + rh,
                indices.data + r * TOP_K, o.shared_gate.data + static_cast<size_t>(r) * INTER,
                o.shared_up.data + static_cast<size_t>(r) * INTER,
                weights.sh_down_packed.data, weights.sh_down_scale.data, weights.sh_down_s2,
                o.shared_down.data + static_cast<size_t>(r) * HIDDEN, HIDDEN, INTER, TOP_K);
            moe_weighted_sum_blend<<<HIDDEN / 256, 256, 0, stream>>>(
                o.final.data + static_cast<size_t>(r) * HIDDEN, o.down.data + rh,
                expert_weights.data + r * TOP_K, o.shared_down.data + static_cast<size_t>(r) * HIDDEN,
                input.data + static_cast<size_t>(r) * HIDDEN, shared_gate_weight.data,
                HIDDEN, TOP_K, HIDDEN);
        }
    }

    bool readonly_and_canaries() const {
        return weights.validate() && input.canary_ok("input") && input.immutable_ok("input") &&
            indices.canary_ok("indices") && indices.immutable_ok("indices") &&
            expert_weights.canary_ok("expert_weights") && expert_weights.immutable_ok("expert_weights") &&
            shared_gate_weight.canary_ok("shared_gate_weight") && shared_gate_weight.immutable_ok("shared_gate_weight") &&
            contract.canary_ok("contract") && contract.immutable_ok("contract") &&
            status.canary_ok("status") && candidate.canaries() && parent.canaries() &&
            serial.canaries() && candidate_repeat.canaries();
    }
};

template <typename T>
bool exact(const Guarded<T>& a, const Guarded<T>& b, const char* label) {
    const auto av = a.download(), bv = b.download();
    if (av.size() == bv.size() && std::memcmp(av.data(), bv.data(), av.size() * sizeof(T)) == 0) return true;
    size_t at = 0;
    while (at < av.size() && std::memcmp(&av[at], &bv[at], sizeof(T)) == 0) ++at;
    std::fprintf(stderr, "exact mismatch %s at element %zu\n", label, at);
    return false;
}

bool outputs_exact(const Outputs& a, const Outputs& b, const char* prefix) {
    bool ok = true;
    ok &= exact(a.gate, b.gate, (std::string(prefix) + ".gate").c_str());
    ok &= exact(a.up, b.up, (std::string(prefix) + ".up").c_str());
    ok &= exact(a.shared_gate, b.shared_gate, (std::string(prefix) + ".shared_gate").c_str());
    ok &= exact(a.shared_up, b.shared_up, (std::string(prefix) + ".shared_up").c_str());
    ok &= exact(a.down, b.down, (std::string(prefix) + ".down").c_str());
    ok &= exact(a.shared_down, b.shared_down, (std::string(prefix) + ".shared_down").c_str());
    ok &= exact(a.final, b.final, (std::string(prefix) + ".final").c_str());
    return ok;
}

template <typename T>
bool payload_is_byte(const Guarded<T>& buffer, unsigned char wanted, const char* label) {
    const auto values = buffer.download();
    const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
    const bool ok = std::all_of(bytes, bytes + values.size() * sizeof(T),
                                [wanted](unsigned char value) { return value == wanted; });
    if (!ok) std::fprintf(stderr, "payload changed on rejected launch: %s\n", label);
    return ok;
}

bool outputs_untouched(const Outputs& o) {
    return payload_is_byte(o.gate, OUTPUT_BYTE, "gate") &&
        payload_is_byte(o.up, OUTPUT_BYTE, "up") &&
        payload_is_byte(o.shared_gate, OUTPUT_BYTE, "shared_gate") &&
        payload_is_byte(o.shared_up, OUTPUT_BYTE, "shared_up") &&
        payload_is_byte(o.down, OUTPUT_BYTE, "down") &&
        payload_is_byte(o.shared_down, OUTPUT_BYTE, "shared_down") &&
        payload_is_byte(o.final, OUTPUT_BYTE, "final");
}

int read_status(const Fixture& f) {
    return f.status.download()[0];
}

bool invalid_contract_gates(Fixture& f) {
    bool ok = true;
    const Contract good = f.contract.download()[0];
    auto check = [&](const char* name, Contract bad, const void* workspace,
                     __nv_bfloat16* output_override, int wanted) {
        f.candidate.reset();
        f.arm_status();
        f.contract.upload({bad}, false);
        f.preflight(f.candidate, f.contract.data, workspace, output_override);
        CUDA_OK(cudaStreamSynchronize(f.stream));
        const int got = read_status(f);
        const bool pass = got == wanted && outputs_untouched(f.candidate);
        if (!pass) std::fprintf(stderr, "invalid gate failed %s: status=%d wanted=%d\n", name, got, wanted);
        ok &= pass;
        f.contract.upload({good}, false);
        return pass;
    };
    Contract bad = good; bad.rows = 15;
    check("geometry", bad, nullptr, nullptr, ERR_GEOMETRY);
    bad = good; bad.routed_hidden_elems--;
    check("capacity", bad, nullptr, nullptr, ERR_CAPACITY);
    bad = good; bad.workspace_bytes = 16;
    check("workspace", bad, reinterpret_cast<void*>(0x10), nullptr, ERR_WORKSPACE);
    check("alias", good, nullptr, f.candidate.gate.data, ERR_ALIAS);
    check("range_overflow", good, nullptr,
          reinterpret_cast<__nv_bfloat16*>(UINTPTR_MAX - 15), ERR_RANGE_OVERFLOW);

    auto ids = f.indices.download();
    ids[37] = EXPERTS;
    f.indices.upload(ids, false);
    check("route", good, nullptr, nullptr, ERR_ROUTE);
    ids[37] = (3 * 37 + 7 * 17 + (3 % 3) * 7) % EXPERTS;
    f.indices.upload(ids, false);
    f.indices.initial = ids;
    return ok;
}

bool hostile_status_ranges_gate_without_writes(Fixture& f) {
    bool ok = true;
    const auto input_before = f.input.download();
    auto check_unchanged = [&](const char* name, int* bad_status) {
        f.candidate.reset();
        f.arm_status();
        f.preflight(f.candidate, f.contract.data, nullptr, nullptr, bad_status);
        CUDA_OK(cudaStreamSynchronize(f.stream));
        const bool pass = outputs_untouched(f.candidate) &&
            std::memcmp(input_before.data(), f.input.download().data(),
                        input_before.size() * sizeof(__nv_bfloat16)) == 0 &&
            read_status(f) == PENDING;
        if (!pass) std::fprintf(stderr, "unsafe status range changed protected state: %s\n", name);
        ok &= pass;
    };
    check_unchanged("status_alias_output", reinterpret_cast<int*>(f.candidate.final.data));
    check_unchanged("status_alias_input", reinterpret_cast<int*>(f.input.data));
    check_unchanged("status_range_overflow", reinterpret_cast<int*>(UINTPTR_MAX - 3));
    return ok;
}

float percentile(std::vector<float> v, double p) {
    std::sort(v.begin(), v.end());
    const size_t i = static_cast<size_t>(std::ceil(p * static_cast<double>(v.size()))) - 1;
    return v[std::min(i, v.size() - 1)];
}

float median(std::vector<float> v) { return percentile(std::move(v), 0.5); }

bool timing_gate(Fixture& f) {
    for (int i = 0; i < 3; ++i) { f.launch_parent(f.parent); f.launch_candidate(f.candidate); }
    CUDA_OK(cudaStreamSynchronize(f.stream));
    std::vector<float> parent, candidate, paired;
    cudaEvent_t begin, end;
    CUDA_OK(cudaEventCreate(&begin)); CUDA_OK(cudaEventCreate(&end));
    auto measure = [&](auto&& launch) {
        CUDA_OK(cudaEventRecord(begin, f.stream));
        launch();
        CUDA_OK(cudaEventRecord(end, f.stream));
        CUDA_OK(cudaEventSynchronize(end));
        float ms = 0.0f; CUDA_OK(cudaEventElapsedTime(&ms, begin, end)); return ms;
    };
    for (int r = 0; r < 21; ++r) {
        float p = 0.0f, c = 0.0f;
        if ((r & 1) == 0) {
            p = measure([&] { f.launch_parent(f.parent); });
            c = measure([&] { f.launch_candidate(f.candidate); });
        } else {
            c = measure([&] { f.launch_candidate(f.candidate); });
            p = measure([&] { f.launch_parent(f.parent); });
        }
        parent.push_back(p); candidate.push_back(c); paired.push_back(p - c);
    }
    CUDA_OK(cudaEventDestroy(begin)); CUDA_OK(cudaEventDestroy(end));
    const float pm = median(parent), cm = median(candidate);
    const float pp90 = percentile(parent, 0.9), cp90 = percentile(candidate, 0.9);
    const float dm = median(paired);
    std::vector<float> deviations;
    for (float d : paired) deviations.push_back(std::fabs(d - dm));
    const float mad = median(deviations);
    const float frame_saving = dm * 48.0f;
    const bool pass = cm < pm && cp90 < pp90 && dm - 3.0f * mad > 0.0f && frame_saving >= 0.5f;
    std::printf("TIMING parent_median_ms=%.6f candidate_median_ms=%.6f parent_p90_ms=%.6f "
                "candidate_p90_ms=%.6f paired_median_ms=%.6f mad_ms=%.6f x48_ms=%.6f pass=%s\n",
                pm, cm, pp90, cp90, dm, mad, frame_saving, pass ? "true" : "false");
    return pass;
}

} // namespace

int main(int argc, char** argv) {
    const bool timing = argc == 2 && std::string(argv[1]) == "--timing";
    if (argc > 2 || (argc == 2 && !timing)) {
        std::fprintf(stderr, "usage: %s [--timing]\n", argv[0]); return 2;
    }
    int device = 0;
    CUDA_OK(cudaGetDevice(&device));
    cudaDeviceProp prop{}; CUDA_OK(cudaGetDeviceProperties(&prop, device));
    if (prop.major != 12 || prop.minor != 1) {
        std::fprintf(stderr, "requires exact sm_121, got sm_%d%d\n", prop.major, prop.minor); return 2;
    }
    const std::string binary_sha256 = executable_sha256();
    const bool provenance_bound = !binary_sha256.empty() &&
        digest_bound(FLASH_NEXT_EXACT_SOURCE_SHA256) &&
        digest_bound(FLASH_NEXT_MICROGATE_SOURCE_SHA256) &&
        digest_bound(FLASH_NEXT_PARENT_BATCH_SHA256) &&
        digest_bound(FLASH_NEXT_PARENT_SERIAL_SHA256) &&
        digest_bound(FLASH_NEXT_PARENT_BLEND_SHA256);
    if (!provenance_bound) {
        std::fprintf(stderr, "unbound source/binary provenance; refusing raw qualification\n");
        return 2;
    }
    Fixture f;
    f.launch_candidate(f.candidate);
    f.launch_candidate(f.candidate_repeat);
    f.launch_parent(f.parent);
    f.launch_serial(f.serial);
    CUDA_OK(cudaStreamSynchronize(f.stream));
    CUDA_OK(cudaGetLastError());
    bool ok = read_status(f) == READY;
    ok &= outputs_exact(f.candidate, f.parent, "candidate_parent");
    ok &= outputs_exact(f.candidate, f.serial, "candidate_serial_k1");
    ok &= outputs_exact(f.candidate, f.candidate_repeat, "candidate_determinism");
    ok &= f.readonly_and_canaries();
    ok &= invalid_contract_gates(f);
    ok &= hostile_status_ranges_gate_without_writes(f);
    ok &= f.readonly_and_canaries();
    if (timing && ok) ok &= timing_gate(f);
    std::printf("FLASH_NEXT_EXACT_K16_MOE rows=16 hidden=2560 intermediate=640 experts=512 "
                "top_k=10 parent=exact serial_k1=exact deterministic=true immutable=true "
                "canaries=true selected_packed_bytes=%llu selected_scale_bytes=%llu "
                "config_sha256=%s index_sha256=%s routed_first_shard_sha256=%s "
                "routed_final_shard_sha256=%s exact_source_sha256=%s microgate_source_sha256=%s "
                "parent_batch_sha256=%s parent_serial_sha256=%s parent_blend_sha256=%s "
                "binary_sha256=%s invalid_gates=6 hostile_status_gates=3 timing=%s pass=%s\n",
                SELECTED_PACKED_BYTES, SELECTED_SCALE_BYTES, CONFIG_SHA256, INDEX_SHA256,
                ROUTED_FIRST_SHARD_SHA256, ROUTED_FINAL_SHARD_SHA256,
                FLASH_NEXT_EXACT_SOURCE_SHA256, FLASH_NEXT_MICROGATE_SOURCE_SHA256,
                FLASH_NEXT_PARENT_BATCH_SHA256, FLASH_NEXT_PARENT_SERIAL_SHA256,
                FLASH_NEXT_PARENT_BLEND_SHA256, binary_sha256.c_str(),
                timing ? "measured" : "unrun", ok ? "true" : "false");
    return ok ? 0 : 1;
}
