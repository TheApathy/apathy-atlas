// SPDX-License-Identifier: AGPL-3.0-only
// Bench-only candidate for the pinned official FP32 CUDA angle expression.
// This is NOT an exactness claim. The retained official Q/K hashes decide.
// theta=10000, head_dim=64; cos[32] then sin[32] per patch.
#include <math.h>
extern "C" __global__ void dsv_p1_angles(float* out, unsigned gh, unsigned gw) {
    const unsigned i = blockIdx.x * 256 + threadIdx.x;
    if (i >= gh * gw * 32) return;
    const unsigned row = i / 32, col = i % 32, frequency = col % 16;
    const float exponent = __fdiv_rn(float(frequency * 2), 32.0f);
    const float inv = __fdiv_rn(1.0f, powf(10000.0f, exponent));
    const float position = float(col < 16 ? row / gw : row % gw);
    const float angle = __fmul_rn(position, inv);
    out[row * 64 + col] = cosf(angle);
    out[row * 64 + 32 + col] = sinf(angle);
}
