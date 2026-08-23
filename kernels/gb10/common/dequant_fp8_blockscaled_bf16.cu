// SPDX-License-Identifier: AGPL-3.0-only

// FP8 E4M3 + multiplier grid -> BF16, entirely on the GPU.
//
// Supported grids are selected by the Rust loader and expressed as
// (block_n, block_k, sk): native block scales, per-row [N]/[N,1], and scalar
// scales all use the same indexing equation.
//
// Numeric contract: this matches the existing CPU reference exactly. In
// particular, Atlas's f32_to_bf16 helper truncates the low 16 bits. Do not use
// __float2bfloat16 here: CUDA implements it with round-to-nearest-even, which
// changes weights and would make a cold-load optimization alter model output.

__device__ __forceinline__ float atlas_fp8_e4m3_to_f32(unsigned char bits) {
    const unsigned int sign = (bits >> 7) & 1u;
    const unsigned int exp = (bits >> 3) & 0x0fu;
    const unsigned int mantissa = bits & 0x07u;

    float value;
    if ((exp == 0u && mantissa == 0u) || (exp == 0x0fu && mantissa == 0x07u)) {
        // Atlas maps the two E4M3 NaN encodings to zero for weight safety.
        value = 0.0f;
    } else if (exp == 0u) {
        value = static_cast<float>(mantissa) * 0.001953125f;
    } else {
        const unsigned int f32_exp = (exp + 120u) << 23;
        const unsigned int f32_mantissa = mantissa << 20;
        value = __uint_as_float(f32_exp | f32_mantissa);
    }
    return sign != 0u ? -value : value;
}

extern "C" __global__ void dequant_fp8_blockscaled_bf16(
    const unsigned char* __restrict__ fp8_in,
    const void* __restrict__ scale_grid,
    unsigned short* __restrict__ bf16_out,
    unsigned int n_rows,
    unsigned int n_cols,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int scale_cols,
    unsigned int scale_is_fp32
) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n_rows || col >= n_cols) return;

    const unsigned int scale_row = row / block_n;
    const unsigned int scale_col = col / block_k;
    const unsigned long long scale_offset =
        static_cast<unsigned long long>(scale_row) * scale_cols + scale_col;

    float scale;
    if (scale_is_fp32 != 0u) {
        scale = static_cast<const float*>(scale_grid)[scale_offset];
    } else {
        const unsigned int raw = static_cast<const unsigned short*>(scale_grid)[scale_offset];
        scale = __uint_as_float(raw << 16);
    }

    const unsigned long long offset =
        static_cast<unsigned long long>(row) * n_cols + col;
    const float fp8_value = atlas_fp8_e4m3_to_f32(fp8_in[offset]);
    float value;
    // The target's general CUDA flags enable FTZ. The CPU reference does not,
    // so spell this multiplication in PTX without `.ftz` to retain subnormal
    // products as well as the common normal-range weights.
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(value) : "f"(fp8_value), "f"(scale));
    bf16_out[offset] = static_cast<unsigned short>(__float_as_uint(value) >> 16);
}
