// SPDX-License-Identifier: AGPL-3.0-only
//
// CB3 -> bf16 reconstruction for DeepSeek-V4.1 routed experts.
//
// This is the PRODUCTION form of `cb3_to_bf16<false>` from
// `harness/cb3_gemm.cu`, which measured rel_l2 2.58e-07 against the Python
// reference with a wrong-K-order control at 1.41 (a 5.5M x separation). The
// arithmetic below is transcribed unchanged; only the harness scaffolding
// (fixture I/O, cuBLASLt call, the `WRONG` control template parameter) is
// gone. The control STAYS in the harness: this file must not carry a
// deliberately-wrong branch into the serve path.
//
// Why reconstruct-then-cuBLASLt rather than a fused decode+MMA loop: it is
// what won on this box twice (Flash-Next 227 -> 1068 tok/s, GLM EXL3
// 516 -> 744), and cuBLASLt measured 3.5-4.8x our own kernels at these
// shapes. Fusion is the optimisation and will be validated against this.
//
// The format costs NOTHING numerically. A CB3 value is an e2m1 grid point
// (one mantissa bit) times a power-of-two UE8M0 scale, so bf16 represents it
// exactly. Every bit of error downstream comes from the GEMM's own fp32
// accumulation order, not from the quantisation.
//
// PLANE GEOMETRY, per output row of N rows and K columns:
//   lo    : K/4  bytes  (two bits per weight, 4 weights per byte)
//   hi    : K/8  bytes  (one bit  per weight, 8 weights per byte)
//   cb    : 8    bytes  (the row's 8-entry e2m1 codebook, low nibble each)
//   scale : K/32 bytes  (UE8M0, one per 32-weight group)
// K is cut into 512-blocks (a 256 tail where 512 does not divide K); see
// `CB3_FORMAT.md` for the derivation and its negative control.

#include <cstdint>
#include <cuda_bf16.h>

// The e2m1 value grid, indexed by the codebook byte's low nibble.
__constant__ float kAtlasCb3Fp4[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

/// One expert plane: CB3 -> bf16 `[N, K]` row-major.
///
/// Grid is flat over N*K; `out` must hold N*K `__nv_bfloat16`. Launch with a
/// block of 256 and `grid.x = ceil(N*K / 256)` — the same shape the harness
/// used, so the measured result transfers.
extern "C" __global__ void cb3_reconstruct_bf16(
    const uint8_t* __restrict__ lo,
    const uint8_t* __restrict__ hi,
    const uint8_t* __restrict__ cb,
    const uint8_t* __restrict__ scale,
    __nv_bfloat16* __restrict__ out,
    int N,
    int K) {
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= (long long)N * K) return;

    const int row = (int)(tid / K);
    const int k = (int)(tid % K);

    // The K-order. Settled empirically in CB3_FORMAT.md against a control that
    // flips `r`; that control is a permutation of the same multiset along K,
    // so it is finite, correctly shaped and meaningless — exactly what a
    // plausible-but-wrong decoder looks like.
    const int block = k / 512;
    const int off = k % 512;
    const int g = off / 32;
    const int lane = (off % 32) / 2;
    const int r = off & 1;

    const int lo_byte = block * 128 + (g / 2) * 16 + lane;
    const int hi_byte = block * 64 + ((g / 2) / 2) * 16 + lane;

    const uint32_t l2 =
        (lo[(long long)row * (K / 4) + lo_byte] >> (4 * (g % 2) + 2 * r)) & 3u;
    const uint32_t h1 =
        (hi[(long long)row * (K / 8) + hi_byte] >> (((g / 2) % 2) * 4 + (g % 2) * 2 + r)) & 1u;

    const uint8_t code = cb[(long long)row * 8 + (l2 | (h1 << 2))] & 0x0Fu;
    const float exponent = (float)scale[(long long)row * (K / 32) + k / 32] - 127.0f;
    out[tid] = __float2bfloat16(kAtlasCb3Fp4[code] * exp2f(exponent));
}
