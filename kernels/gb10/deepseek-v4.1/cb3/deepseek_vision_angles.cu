// SPDX-License-Identifier: AGPL-3.0-only
// DeepSeek Vision theta10000/head64 CUDA FP32 angle expression, retained from
// bench/deepseek-vision-p1/angles.cu (same-QKV three-grid reference-hash gate).
// Formula provenance: deepseek-ai/DeepSeek-V4-Flash-Vision-Exp (MIT),
// 6821d6ad3681a4b137b066b76094fa82ebd0a380, inference/vision.py.
// Requires no fast math: -O3 --fmad=false --ftz=false --prec-div=true
// --prec-sqrt=true. Keep precise pow/divide/multiply/cos/sin boundaries.
#include <math.h>
#ifdef __FAST_MATH__
#error "DeepSeek vision angles require the precise FP32 compilation contract"
#endif

// Host Geometry admits positive gh/gw, gh*gw <= 3456, and padded output <=384.
// Each thread owns one frequency pair; output is cos[32],sin[32] per patch.
extern "C" __global__ void deepseek_vision_angles(float* out, unsigned gh, unsigned gw) {
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
