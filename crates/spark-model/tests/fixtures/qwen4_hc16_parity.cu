// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../../../../kernels/gb10/common/qwen4_hyper.cu"

#define CUDA_OK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        std::exit(2); \
    } \
} while (0)

static constexpr unsigned ROWS = 16, H = 2560, HC = 4, R = H * HC, GUARD = 64;

struct DeviceWords {
    unsigned short* base = nullptr;
    unsigned short* data = nullptr;
    size_t words = 0;
    explicit DeviceWords(size_t count) : words(count) {
        CUDA_OK(cudaMalloc(&base, (count + 2 * GUARD) * 2));
        data = base + GUARD;
    }
    ~DeviceWords() { cudaFree(base); }
};

static bool guarded_equal(const std::vector<unsigned short>& expected, DeviceWords& actual,
                          unsigned short canary, const char* label) {
    std::vector<unsigned short> got(actual.words + 2 * GUARD);
    CUDA_OK(cudaMemcpy(got.data(), actual.base, got.size() * 2, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (unsigned i = 0; i < GUARD; ++i)
        ok &= got[i] == canary && got[GUARD + actual.words + i] == canary;
    for (size_t i = 0; i < expected.size(); ++i) ok &= got[GUARD + i] == expected[i];
    if (!ok) std::fprintf(stderr, "%s data or guard mismatch\n", label);
    return ok;
}

static void upload_guarded(const std::vector<unsigned short>& data, DeviceWords& device,
                           unsigned short canary) {
    std::vector<unsigned short> guarded(data.size() + 2 * GUARD, canary);
    std::copy(data.begin(), data.end(), guarded.begin() + GUARD);
    CUDA_OK(cudaMemcpy(device.base, guarded.data(), guarded.size() * 2, cudaMemcpyHostToDevice));
}

int main() {
    const unsigned short canary = 0x5a17;
    std::vector<unsigned short> residual((size_t)ROWS * R);
    std::vector<unsigned short> hidden((size_t)ROWS * R);
    std::vector<unsigned short> packed((size_t)ROWS * H);
    for (size_t i = 0; i < residual.size(); ++i)
        residual[i] = (unsigned short)(((i & 1) << 15) | 0x3e80 | (i & 0x3f));
    for (size_t i = 0; i < hidden.size(); ++i)
        hidden[i] = (unsigned short)((((i >> 1) & 1) << 15) | 0x3f00 | (i & 0x1f));
    for (unsigned row = 0; row < ROWS; ++row)
        for (unsigned stream = 0; stream < HC; ++stream)
            residual[(size_t)row * R + R - HC + stream] =
                (unsigned short)(0x3e00 + row * HC + stream);
    for (unsigned row = 0; row < ROWS; ++row)
        std::copy_n(residual.begin() + (size_t)row * R, H,
                    packed.begin() + (size_t)row * H);

    DeviceWords d_residual(residual.size()), d_pack(packed.size());
    DeviceWords d_hidden_ref(hidden.size()), d_hidden_candidate(hidden.size());
    upload_guarded(residual, d_residual, canary);
    upload_guarded(std::vector<unsigned short>(packed.size(), 0x6b2d), d_pack, canary);
    upload_guarded(hidden, d_hidden_ref, canary);
    upload_guarded(hidden, d_hidden_candidate, canary);

    qwen4_hc_pack_mixed<<<(ROWS * H + 255) / 256, 256>>>(
        (__nv_bfloat16*)d_residual.data, (__nv_bfloat16*)d_pack.data, ROWS, H, HC);
    CUDA_OK(cudaDeviceSynchronize());
    bool ok = guarded_equal(packed, d_pack, canary, "pack");
    ok &= guarded_equal(residual, d_residual, canary, "residual after pack");

    for (unsigned row = 0; row < ROWS; ++row) {
        qwen4_hc_inject<<<dim3(1, (R + 255) / 256, 1), 256>>>(
            (__nv_bfloat16*)d_hidden_ref.data + (size_t)row * R,
            (__nv_bfloat16*)d_pack.data + (size_t)row * H,
            (__nv_bfloat16*)d_residual.data + (size_t)row * R + R - HC, H, HC);
    }
    qwen4_hc_inject_saved<<<dim3(ROWS, (R + 255) / 256, 1), 256>>>(
        (__nv_bfloat16*)d_hidden_candidate.data, (__nv_bfloat16*)d_pack.data,
        (__nv_bfloat16*)d_residual.data, H, HC);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<unsigned short> reference(hidden.size());
    CUDA_OK(cudaMemcpy(reference.data(), d_hidden_ref.data, reference.size() * 2,
                       cudaMemcpyDeviceToHost));
    ok &= guarded_equal(reference, d_hidden_candidate, canary, "injection");
    ok &= guarded_equal(packed, d_pack, canary, "packed after injection");
    ok &= guarded_equal(residual, d_residual, canary, "residual after injection");
    std::printf("F38 HC16 raw parity: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
