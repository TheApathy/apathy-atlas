// SPDX-License-Identifier: AGPL-3.0-only

// Isolated, default-unrouted activation quantizer for the FlashInfer/CUTLASS
// NVFP4 contract. Packed output is logical row-major [M,K/2]. Scale output is
// E4M3 in padded 128x4 layout and therefore occupies ceil(M/128)*K/16 bytes.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

__device__ __forceinline__ float rcp_approx_ftz(float value) {
    float result;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ float mul_rn(float lhs, float rhs) {
    float result;
    asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(lhs), "f"(rhs));
    return result;
}

__device__ __forceinline__ unsigned int e2m1_eight(const float* values, float scale) {
    unsigned int result;
    const float v0 = mul_rn(values[0], scale);
    const float v1 = mul_rn(values[1], scale);
    const float v2 = mul_rn(values[2], scale);
    const float v3 = mul_rn(values[3], scale);
    const float v4 = mul_rn(values[4], scale);
    const float v5 = mul_rn(values[5], scale);
    const float v6 = mul_rn(values[6], scale);
    const float v7 = mul_rn(values[7], scale);
    asm volatile(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n"
        "}"
        : "=r"(result)
        : "f"(v0), "f"(v1), "f"(v2), "f"(v3), "f"(v4), "f"(v5), "f"(v6), "f"(v7));
    return result;
}

__device__ __forceinline__ unsigned long long scale_offset_128x4(
    unsigned int row,
    unsigned int group,
    unsigned int groups
) {
    const unsigned int group_blocks = (groups + 3u) / 4u;
    return (((((static_cast<unsigned long long>(row / 128u) * group_blocks + group / 4u) * 32u
                + row % 32u) * 4u + (row % 128u) / 32u) * 4u) + group % 4u);
}

// These two helpers intentionally mirror the legacy Atlas software
// conversions byte-for-byte. Do not replace them with SM121 cvt instructions:
// their half-way and signed-zero behavior is part of this shadow's contract.
__device__ __forceinline__ unsigned char atlas_float_to_fp8_e4m3(float value) {
    unsigned int bits = __float_as_uint(value);
    const unsigned int sign = (bits >> 31) & 1u;
    if ((bits & 0x7fffffffu) == 0u) return static_cast<unsigned char>(sign << 7);

    float absolute = fabsf(value);
    if (absolute > 448.0f) absolute = 448.0f;
    bits = __float_as_uint(absolute);
    const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127;
    const unsigned int mantissa = bits & 0x7fffffu;
    if (exponent < -9) return static_cast<unsigned char>(sign << 7);
    if (exponent < -6) {
        int encoded = static_cast<int>(absolute * 512.0f + 0.5f);
        if (encoded > 7) encoded = 7;
        if (encoded < 0) encoded = 0;
        return static_cast<unsigned char>((sign << 7) | encoded);
    }

    int encoded_exponent = exponent + 7;
    unsigned int encoded_mantissa;
    if (encoded_exponent > 15) {
        encoded_exponent = 15;
        encoded_mantissa = 6;
    } else {
        encoded_mantissa = (mantissa + (1u << 19)) >> 20;
        if (encoded_mantissa > 7) {
            encoded_mantissa = 0;
            ++encoded_exponent;
            if (encoded_exponent > 15) {
                encoded_exponent = 15;
                encoded_mantissa = 6;
            }
        }
    }
    return static_cast<unsigned char>((sign << 7) | (encoded_exponent << 3) |
                                      encoded_mantissa);
}

__device__ __forceinline__ unsigned int atlas_quantize_e2m1(float value) {
    const float absolute = fabsf(value);
    const unsigned int sign = value < 0.0f ? 8u : 0u;
    unsigned int index;
    if (absolute <= 0.25f) index = 0;
    else if (absolute <= 0.75f) index = 1;
    else if (absolute <= 1.25f) index = 2;
    else if (absolute <= 1.75f) index = 3;
    else if (absolute <= 2.5f) index = 4;
    else if (absolute <= 3.5f) index = 5;
    else if (absolute <= 5.0f) index = 6;
    else index = 7;
    return sign | index;
}

}  // namespace

extern "C" __global__ void quantize_bf16_to_nvfp4_cutlass_128x4(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    const float* __restrict__ global_scale,
    unsigned int rows,
    unsigned int cols
) {
    const unsigned int padded_rows = ((rows + 127u) / 128u) * 128u;
    const unsigned int groups = cols / 16u;

    for (unsigned int row = blockIdx.x; row < padded_rows; row += gridDim.x) {
        if (row >= rows) {
            for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
                scale_out[scale_offset_128x4(row, group, groups)] = 0;
            }
            continue;
        }

        const __nv_bfloat16* row_input = input + static_cast<unsigned long long>(row) * cols;
        unsigned char* row_packed = packed_out + static_cast<unsigned long long>(row) * (cols / 2u);
        const float scale = global_scale[0];
        const float reciprocal_six = rcp_approx_ftz(6.0f);
        const float reciprocal_scale = rcp_approx_ftz(scale);

        for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
            const unsigned int base = group * 16u;
            float values[16];
            float maximum = 0.0f;
#pragma unroll
            for (unsigned int index = 0; index < 16u; ++index) {
                const float value = __bfloat162float(row_input[base + index]);
                values[index] = value;
                maximum = fmaxf(maximum, fabsf(value));
            }

            const float scaled_max = mul_rn(maximum, reciprocal_six);
            const float sf_value = mul_rn(scale, scaled_max);
            __nv_fp8_e4m3 encoded_scale = __nv_fp8_e4m3(sf_value);
            scale_out[scale_offset_128x4(row, group, groups)] = encoded_scale.__x;

            const float decoded_scale = static_cast<float>(encoded_scale);
            const float effective = mul_rn(decoded_scale, reciprocal_scale);
            const float output_scale = maximum != 0.0f ? rcp_approx_ftz(effective) : 0.0f;
            reinterpret_cast<unsigned int*>(row_packed + group * 8u)[0] =
                e2m1_eight(values, output_scale);
            reinterpret_cast<unsigned int*>(row_packed + group * 8u)[1] =
                e2m1_eight(values + 8, output_scale);
        }
    }
}

extern "C" __global__ void quantize_bf16_to_nvfp4_atlas_128x4(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    float scale2,
    unsigned int rows,
    unsigned int cols
) {
    const unsigned int padded_rows = ((rows + 127u) / 128u) * 128u;
    const unsigned int groups = cols / 16u;
    for (unsigned int row = blockIdx.x; row < padded_rows; row += gridDim.x) {
        if (row >= rows) {
            for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
                scale_out[scale_offset_128x4(row, group, groups)] = 0;
            }
            continue;
        }

        const __nv_bfloat16* row_input = input + static_cast<unsigned long long>(row) * cols;
        unsigned char* row_packed = packed_out + static_cast<unsigned long long>(row) * (cols / 2u);
        const float inverse_scale2 = scale2 > 0.0f ? 1.0f / scale2 : 0.0f;
        for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
            const unsigned int base = group * 16u;
            float group_max = 0.0f;
#pragma unroll
            for (int index = 0; index < 16; ++index) {
                const float value = fabsf(__bfloat162float(row_input[base + index]));
                if (value > group_max) group_max = value;
            }

            const float fp8_float =
                group_max > 0.0f ? group_max * inverse_scale2 / 6.0f : 0.0f;
            const unsigned char fp8_byte = atlas_float_to_fp8_e4m3(fp8_float);
            scale_out[scale_offset_128x4(row, group, groups)] = fp8_byte;

            const unsigned int fp8_sign = (fp8_byte >> 7) & 1u;
            const unsigned int fp8_exp = (fp8_byte >> 3) & 0xfu;
            const unsigned int fp8_mantissa = fp8_byte & 7u;
            float decoded;
            if (fp8_exp == 0) {
                decoded = static_cast<float>(fp8_mantissa) * 0.001953125f;
            } else if (fp8_exp == 15 && fp8_mantissa == 7) {
                decoded = 0.0f;
            } else {
                const unsigned int f32_bits =
                    ((fp8_exp + 120u) << 23) | (fp8_mantissa << 20);
                decoded = __uint_as_float(f32_bits);
            }
            if (fp8_sign) decoded = -decoded;
            const float effective_scale = decoded * scale2;
            const float inverse_effective = effective_scale > 0.0f ? 1.0f / effective_scale : 0.0f;
#pragma unroll
            for (int index = 0; index < 16; index += 2) {
                const float v0 = __bfloat162float(row_input[base + index]) * inverse_effective;
                const float v1 = __bfloat162float(row_input[base + index + 1]) * inverse_effective;
                const unsigned int n0 = atlas_quantize_e2m1(v0);
                const unsigned int n1 = atlas_quantize_e2m1(v1);
                row_packed[group * 8u + index / 2] =
                    static_cast<unsigned char>((n1 << 4) | (n0 & 0xfu));
            }
        }
    }
}

// Device-resident dynamic-scale arm for the Qwen3.8 SSM QKVZ projection.
//
// This deliberately remains ABI-separated from the checkpoint-static arm
// above.  `global_max` is produced by the existing
// `quantize_nvfp4::nvfp4_global_absmax` kernel on the same CUDA stream.  Every
// CTA derives the identical second-level activation scale locally so no CTA
// waits for a device-published scalar.  Only CTA0/thread0 publishes
// `scale2_out`; the following one-thread kernel forms the separate
// FlashInfer/CUTLASS alpha operand after the quantizer has finished.
//
// `status_out` is stream-resident.  A non-finite activation is not silently
// accepted merely because the legacy absmax reduction ignores NaNs: the
// quantizer records it and emits deterministic zero bytes for that group.  The
// alpha kernel traps before publishing a consumable alpha when status is
// nonzero, making a subsequent same-stream C-ABI launch fail closed without a
// D2H copy or host synchronization.
enum Nvfp4DynamicStatus : unsigned int {
    NVFP4_DYNAMIC_BAD_GLOBAL_MAX = 1u << 0,
    NVFP4_DYNAMIC_NONFINITE_INPUT = 1u << 1,
    NVFP4_DYNAMIC_BAD_SCALE2 = 1u << 2,
    NVFP4_DYNAMIC_BAD_ALPHA = 1u << 3,
};

extern "C" __global__ void quantize_bf16_to_nvfp4_atlas_128x4_from_absmax(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    const float* __restrict__ global_max,
    float* __restrict__ scale2_out,
    unsigned int* __restrict__ status_out,
    unsigned int rows,
    unsigned int cols
) {
    const float maximum = global_max[0];
    const bool valid_maximum = isfinite(maximum) && maximum >= 0.0f;
    // Match the current host derivation exactly, including the all-zero
    // special case.  __fdiv_rn prevents the bundle-wide --use_fast_math from
    // replacing the division by 6*448 with an approximate reciprocal.
    const float scale2 = valid_maximum
        ? (maximum > 0.0f ? __fdiv_rn(maximum, 2688.0f) : 1.0f)
        : 1.0f;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        scale2_out[0] = scale2;
        if (!valid_maximum) atomicOr(status_out, NVFP4_DYNAMIC_BAD_GLOBAL_MAX);
        if (!isfinite(scale2)) atomicOr(status_out, NVFP4_DYNAMIC_BAD_SCALE2);
    }

    const unsigned int padded_rows = ((rows + 127u) / 128u) * 128u;
    const unsigned int groups = cols / 16u;
    for (unsigned int row = blockIdx.x; row < padded_rows; row += gridDim.x) {
        if (row >= rows) {
            for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
                scale_out[scale_offset_128x4(row, group, groups)] = 0;
            }
            continue;
        }

        const __nv_bfloat16* row_input =
            input + static_cast<unsigned long long>(row) * cols;
        unsigned char* row_packed =
            packed_out + static_cast<unsigned long long>(row) * (cols / 2u);
        const float inverse_scale2 = scale2 > 0.0f ? 1.0f / scale2 : 0.0f;

        for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
            const unsigned int base = group * 16u;
            float values[16];
            float group_max = 0.0f;
            bool finite_group = valid_maximum;
#pragma unroll
            for (int index = 0; index < 16; ++index) {
                const float value = __bfloat162float(row_input[base + index]);
                values[index] = value;
                if (!isfinite(value)) {
                    finite_group = false;
                } else {
                    const float absolute = fabsf(value);
                    if (absolute > group_max) group_max = absolute;
                }
            }

            if (!finite_group) {
                atomicOr(status_out, NVFP4_DYNAMIC_NONFINITE_INPUT);
                scale_out[scale_offset_128x4(row, group, groups)] = 0;
#pragma unroll
                for (int index = 0; index < 8; ++index) {
                    row_packed[group * 8u + index] = 0;
                }
                continue;
            }

            // Keep this arithmetic and conversion order byte-identical to
            // quantize_bf16_to_nvfp4_atlas_128x4 above.  Only the physical
            // scale address differs from the legacy row-major host arm.
            const float fp8_float =
                group_max > 0.0f ? group_max * inverse_scale2 / 6.0f : 0.0f;
            const unsigned char fp8_byte = atlas_float_to_fp8_e4m3(fp8_float);
            scale_out[scale_offset_128x4(row, group, groups)] = fp8_byte;

            const unsigned int fp8_sign = (fp8_byte >> 7) & 1u;
            const unsigned int fp8_exp = (fp8_byte >> 3) & 0xfu;
            const unsigned int fp8_mantissa = fp8_byte & 7u;
            float decoded;
            if (fp8_exp == 0) {
                decoded = static_cast<float>(fp8_mantissa) * 0.001953125f;
            } else if (fp8_exp == 15 && fp8_mantissa == 7) {
                decoded = 0.0f;
            } else {
                const unsigned int f32_bits =
                    ((fp8_exp + 120u) << 23) | (fp8_mantissa << 20);
                decoded = __uint_as_float(f32_bits);
            }
            if (fp8_sign) decoded = -decoded;
            const float effective_scale = decoded * scale2;
            const float inverse_effective =
                effective_scale > 0.0f ? 1.0f / effective_scale : 0.0f;
#pragma unroll
            for (int index = 0; index < 16; index += 2) {
                const float v0 = values[index] * inverse_effective;
                const float v1 = values[index + 1] * inverse_effective;
                const unsigned int n0 = atlas_quantize_e2m1(v0);
                const unsigned int n1 = atlas_quantize_e2m1(v1);
                row_packed[group * 8u + index / 2] =
                    static_cast<unsigned char>((n1 << 4) | (n0 & 0xfu));
            }
        }
    }
}

extern "C" __global__ void nvfp4_dynamic_combined_alpha(
    const float* __restrict__ scale2_a,
    float weight_scale_2,
    unsigned int* __restrict__ status,
    float* __restrict__ combined_alpha_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float activation_scale = scale2_a[0];
    // Reuse the inline `mul.rn.f32` helper rather than `__fmul_rn`: with the
    // bundle's --use_fast_math flag CUDA lowers the latter to FMUL.FTZ, which
    // would not match a host IEEE multiplication for subnormal scale2 values.
    const float combined = mul_rn(activation_scale, weight_scale_2);
    if (status[0] != 0u || !isfinite(activation_scale) || activation_scale < 0.0f ||
        !isfinite(weight_scale_2) || !(weight_scale_2 > 0.0f) || !isfinite(combined) ||
        combined < 0.0f) {
        atomicOr(status, NVFP4_DYNAMIC_BAD_ALPHA);
        asm volatile("trap;");
        return;
    }
    combined_alpha_out[0] = combined;
}
