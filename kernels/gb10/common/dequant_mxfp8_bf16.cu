// SPDX-License-Identifier: AGPL-3.0-only

// OCP MXFP8 E4M3FN values with rowwise E8M0 group-32 scales -> BF16.
// ModelOpt stores weight[N,K] and scale[N,K/32] without a global scale.

__device__ __forceinline__ float atlas_mxfp8_e4m3_to_f32(unsigned char bits) {
    const unsigned int sign = (bits >> 7) & 1u;
    const unsigned int exponent = (bits >> 3) & 0x0fu;
    const unsigned int mantissa = bits & 0x07u;

    float value;
    if ((exponent == 0u && mantissa == 0u) ||
        (exponent == 0x0fu && mantissa == 0x07u)) {
        // Match Atlas's established safe weight behavior: signed zero and the
        // two E4M3FN NaN encodings become zero.
        value = 0.0f;
    } else if (exponent == 0u) {
        value = static_cast<float>(mantissa) * 0.001953125f;
    } else {
        const unsigned int f32_exponent = (exponent + 120u) << 23;
        const unsigned int f32_mantissa = mantissa << 20;
        value = __uint_as_float(f32_exponent | f32_mantissa);
    }
    return sign != 0u ? -value : value;
}

__device__ __forceinline__ float atlas_e8m0_to_f32(unsigned char exponent) {
    // Scale 255 is rejected by the host before launch. Retain a defensive zero
    // here so a malformed launch cannot introduce NaNs.
    if (exponent == 255u) return 0.0f;
    if (exponent == 0u) return __uint_as_float(1u << 22);  // 2^-127
    return __uint_as_float(static_cast<unsigned int>(exponent) << 23);
}

extern "C" __global__ void dequant_mxfp8_bf16(
    const unsigned char* __restrict__ fp8_weight,
    const unsigned char* __restrict__ e8m0_scale,
    unsigned short* __restrict__ bf16_out,
    unsigned int n_rows,
    unsigned int n_cols,
    unsigned int scale_cols
) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n_rows || col >= n_cols) return;

    const unsigned long long offset =
        static_cast<unsigned long long>(row) * n_cols + col;
    const unsigned long long scale_offset =
        static_cast<unsigned long long>(row) * scale_cols + col / 32u;
    const float value = atlas_mxfp8_e4m3_to_f32(fp8_weight[offset]);
    const float scale = atlas_e8m0_to_f32(e8m0_scale[scale_offset]);
    float dequantized;
    // Preserve subnormal products when scale exponent is zero; target flags
    // otherwise enable FTZ. This matches the scalar reference.
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(dequantized) : "f"(value), "f"(scale));
    bf16_out[offset] = static_cast<unsigned short>(__float_as_uint(dequantized) >> 16);
}
