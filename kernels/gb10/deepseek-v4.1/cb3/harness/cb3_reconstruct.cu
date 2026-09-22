// SPDX-License-Identifier: AGPL-3.0-only
//
// CB3 -> bf16 weight reconstruction, and the check that proves it.
//
// WHY RECONSTRUCT AT ALL. Both prefill wins measured on this box came from
// reconstructing the weight and handing the GEMM to cuBLASLt rather than
// fusing decode into a hand-written MMA loop: Flash-Next 227 -> 1068 tok/s
// ("dequant + cuBLASLt"), GLM EXL3 516 -> 744 ("reconstruct + cuBLAS"). At
// these shapes cuBLASLt measured 3.5-4.8x our own kernels. So the port's first
// GEMM is cuBLASLt's, and this kernel's only job is to hand it a bf16 tile.
//
// WHY THIS VERSION IS DELIBERATELY SLOW. One thread per weight, straight from
// the K-order settled in CB3_FORMAT.md. It exists to be OBVIOUSLY right, so it
// can be the reference the PTX-decoder version (cb3_decode.cuh, 4 bytes per
// instruction) is validated against. Optimising before there is something to
// disagree with is how a wrong K-order ships as "fast".
//
// THE K-ORDER, restated so this file is readable alone. K is cut into blocks
// of 512 (a 256 tail where 512 does not divide). Within a block, for scale
// group g (0..15), lane (0..15) and sub-position r (0..1):
//
//     k        = block*512 + g*32 + lane*2 + r
//     lo byte  = block*128 + (g/2)*16 + lane      shift = 4*(g%2) + 2*r
//     hi byte  = block*64  + ((g/2)/2)*16 + lane  bit   = ((g/2)%2)*4 + (g%2)*2 + r
//
// index = (lo >> shift) & 3 | ((hi >> bit) & 1) << 2   -> 0..7, into the ROW's
// 8-entry codebook, whose entries are e2m1 codes; the value is then scaled by
// UE8M0 exp2(s - 127) for the group k/32.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

// The e2m1 grid, indexed by the 4-bit code a codebook entry holds.
__constant__ float kFp4[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                               -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

#define CUDA_OK(x)                                                                        \
    do {                                                                                  \
        cudaError_t e_ = (x);                                                             \
        if (e_ != cudaSuccess) {                                                          \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));\
            std::exit(2);                                                                 \
        }                                                                                 \
    } while (0)

/// One thread per weight. `out` is row-major [N, K] float.
__global__ void cb3_reconstruct_ref(const uint8_t* __restrict__ lo,
                                    const uint8_t* __restrict__ hi,
                                    const uint8_t* __restrict__ cb,
                                    const uint8_t* __restrict__ scale,
                                    float* __restrict__ out, int N, int K) {
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= (long long)N * K) return;
    const int row = (int)(tid / K);
    const int k = (int)(tid % K);

    // Position within the 512-block. The tail block is 256 wide; its groups are
    // numbered from the same base, which is why `g` is derived from the offset
    // inside the block rather than from k directly.
    const int block = k / 512;
    const int off = k % 512;
    const int g = off / 32;
    const int lane = (off % 32) / 2;
    const int r = off & 1;

    const int lo_byte = block * 128 + (g / 2) * 16 + lane;
    const int hi_byte = block * 64 + ((g / 2) / 2) * 16 + lane;
    const int shift = 4 * (g % 2) + 2 * r;
    const int bit = ((g / 2) % 2) * 4 + (g % 2) * 2 + r;

    const uint32_t l2 = (lo[(long long)row * (K / 4) + lo_byte] >> shift) & 0x3u;
    const uint32_t h1 = (hi[(long long)row * (K / 8) + hi_byte] >> bit) & 0x1u;
    const uint32_t idx = l2 | (h1 << 2);                       // 0..7

    const uint8_t code = cb[(long long)row * 8 + idx] & 0x0Fu; // e2m1 code
    const uint8_t e8 = scale[(long long)row * (K / 32) + k / 32];
    out[tid] = kFp4[code] * exp2f((float)e8 - 127.0f);
}

static void* slurp(const char* path, size_t want) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "open %s\n", path); std::exit(2); }
    void* p = std::malloc(want);
    size_t got = std::fread(p, 1, want, f);
    if (got != want) { std::fprintf(stderr, "%s: %zu of %zu bytes\n", path, got, want); std::exit(2); }
    std::fclose(f);
    return p;
}

int main() {
    int N = 0, K = 0;
    { FILE* f = std::fopen("fx_dims.txt", "r");
      if (!f || std::fscanf(f, "%d %d", &N, &K) != 2) { std::fprintf(stderr, "fx_dims.txt\n"); return 2; }
      std::fclose(f); }
    std::printf("fixture N=%d K=%d\n", N, K);

    uint8_t* h_lo = (uint8_t*)slurp("fx_lo.bin", (size_t)N * (K / 4));
    uint8_t* h_hi = (uint8_t*)slurp("fx_hi.bin", (size_t)N * (K / 8));
    uint8_t* h_cb = (uint8_t*)slurp("fx_cb.bin", (size_t)N * 8);
    uint8_t* h_s  = (uint8_t*)slurp("fx_s.bin",  (size_t)N * (K / 32));
    float*   h_ref= (float*)  slurp("fx_ref.bin",(size_t)N * K * sizeof(float));

    uint8_t *d_lo, *d_hi, *d_cb, *d_s; float* d_out;
    CUDA_OK(cudaMalloc(&d_lo, (size_t)N * (K / 4)));
    CUDA_OK(cudaMalloc(&d_hi, (size_t)N * (K / 8)));
    CUDA_OK(cudaMalloc(&d_cb, (size_t)N * 8));
    CUDA_OK(cudaMalloc(&d_s,  (size_t)N * (K / 32)));
    CUDA_OK(cudaMalloc(&d_out,(size_t)N * K * sizeof(float)));
    CUDA_OK(cudaMemcpy(d_lo, h_lo, (size_t)N * (K / 4), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_hi, h_hi, (size_t)N * (K / 8), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_cb, h_cb, (size_t)N * 8, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_s,  h_s,  (size_t)N * (K / 32), cudaMemcpyHostToDevice));

    const long long total = (long long)N * K;
    cb3_reconstruct_ref<<<(int)((total + 255) / 256), 256>>>(d_lo, d_hi, d_cb, d_s, d_out, N, K);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    float* h_out = (float*)std::malloc((size_t)total * sizeof(float));
    CUDA_OK(cudaMemcpy(h_out, d_out, (size_t)total * sizeof(float), cudaMemcpyDeviceToHost));

    long long mismatched = 0; double worst = 0.0; long long worst_at = -1; long long nz = 0;
    for (long long i = 0; i < total; ++i) {
        if (h_ref[i] != 0.0f) ++nz;
        double d = std::fabs((double)h_out[i] - (double)h_ref[i]);
        if (d > worst) { worst = d; worst_at = i; }
        if (h_out[i] != h_ref[i]) ++mismatched;
    }
    std::printf("weights=%lld nonzero_ref=%lld mismatched=%lld worst_abs=%.3e", total, nz, mismatched, worst);
    if (worst_at >= 0 && mismatched)
        std::printf(" at row=%lld k=%lld got=%g want=%g",
                    worst_at / K, worst_at % K, h_out[worst_at], h_ref[worst_at]);
    std::printf("\n");

    // A kernel that wrote a constant, or all zeros, would "match" a degenerate
    // fixture. Refuse to call that a pass.
    if (nz * 4 < total) { std::printf("FIXTURE TOO SPARSE to be evidence\n"); return 3; }
    std::printf(mismatched == 0 ? "PASS bit-exact vs unpack_cb3_v2\n" : "FAIL\n");
    return mismatched == 0 ? 0 : 1;
}
