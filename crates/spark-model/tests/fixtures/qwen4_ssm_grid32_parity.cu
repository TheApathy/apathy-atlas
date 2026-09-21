// SPDX-License-Identifier: AGPL-3.0-only

// Standalone raw parity gate. This fixture is intentionally outside every
// kernel manifest and compares the incumbent serial M32 launches with F39's
// single two-dimensional launch.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../../../../kernels/gb10/common/w4a16_gemv_rt.cu"

#define CUDA_OK(call) do { const cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    std::exit(2); } } while (0)

template <typename T> struct Dev {
    T* p = nullptr;
    explicit Dev(size_t count) { CUDA_OK(cudaMalloc(&p, count * sizeof(T))); }
    ~Dev() { if (p) cudaFree(p); }
};

static bool shape(unsigned rows, unsigned n, unsigned k, unsigned salt) {
    const size_t a_count = static_cast<size_t>(rows) * k;
    const size_t packed_count = static_cast<size_t>(n) * k / 2;
    const size_t scale_count = static_cast<size_t>(n) * k / 16;
    const size_t c_count = static_cast<size_t>(rows) * n;
    std::vector<unsigned short> a(a_count);
    std::vector<unsigned char> packed(packed_count), scales(scale_count);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<unsigned short>(0x3c00u + ((i * 17u + salt) & 0x01ffu));
    for (size_t i = 0; i < packed.size(); ++i)
        packed[i] = static_cast<unsigned char>((i * 29u + salt * 7u) & 0xffu);
    for (size_t i = 0; i < scales.size(); ++i)
        scales[i] = static_cast<unsigned char>(0x18u + ((i * 13u + salt) % 0x40u));

    Dev<unsigned short> d_a(a_count), d_ref(c_count), d_grid(c_count);
    Dev<unsigned char> d_packed(packed_count), d_scales(scale_count);
    CUDA_OK(cudaMemcpy(d_a.p, a.data(), a_count * 2, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_packed.p, packed.data(), packed_count, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_scales.p, scales.data(), scale_count, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(d_ref.p, 0xa5, c_count * 2));
    CUDA_OK(cudaMemset(d_grid.p, 0xa5, c_count * 2));

    const unsigned grid_x = (n + 7u) / 8u;
    for (unsigned start = 0; start < rows; start += 32) {
        w4a16_gemv_batch_logits_exact_rt2_m32<<<grid_x, 256>>>(
            reinterpret_cast<const __nv_bfloat16*>(d_a.p) + static_cast<size_t>(start) * k,
            d_packed.p, d_scales.p, 0.0078125f,
            reinterpret_cast<__nv_bfloat16*>(d_ref.p) + static_cast<size_t>(start) * n,
            32, n, k);
    }
    w4a16_gemv_batch_logits_exact_rt2_m32_grid<<<dim3(grid_x, rows / 32, 1), 256>>>(
        reinterpret_cast<const __nv_bfloat16*>(d_a.p), d_packed.p, d_scales.p,
        0.0078125f, reinterpret_cast<__nv_bfloat16*>(d_grid.p), rows, n, k);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<unsigned short> reference(c_count), candidate(c_count), a_after(a_count);
    std::vector<unsigned char> packed_after(packed_count), scales_after(scale_count);
    CUDA_OK(cudaMemcpy(reference.data(), d_ref.p, c_count * 2, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(candidate.data(), d_grid.p, c_count * 2, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(a_after.data(), d_a.p, a_count * 2, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(packed_after.data(), d_packed.p, packed_count, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(scales_after.data(), d_scales.p, scale_count, cudaMemcpyDeviceToHost));
    const bool ok = reference == candidate && a == a_after && packed == packed_after
        && scales == scales_after;
    std::printf("rows=%u n=%u k=%u exact=%s\n", rows, n, k, ok ? "true" : "false");
    return ok;
}

int main() {
    bool ok = true;
    ok &= shape(64, 16384, 2560, 1);
    ok &= shape(64, 2560, 6144, 2);
    ok &= shape(2048, 64, 128, 3);
    std::printf("F39 SSM GRID32 raw parity: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
