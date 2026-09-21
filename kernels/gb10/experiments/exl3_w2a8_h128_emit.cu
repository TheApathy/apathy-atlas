// SPDX-License-Identifier: AGPL-3.0-only
//
// Exact-shape H128 -> E4M3 activation emitters for the isolated EXL3 W2A8
// grouped-prefill component. DeepSeek serving exposes these symbols only
// through the explicit, default-off ATLAS_EXL3_PREFILL_W2A8 experiment;
// standalone numeric and occupancy gates must pass on GB10 before promotion.

#define EXL3_FIXED_BITS 2
#define EXL3_HROW_DSV4_FIXED 1
#include "../common/exl3_gemv.cu"
#include <cuda_fp8.h>

#define W2A8_EMIT_GROUP_K 128
#define W2A8_EMIT_E4M3_MAX 448.0f
#define W2A8_EMIT_MIN_SCALE 1.0e-12f

__device__ __forceinline__ void w2a8_emit_bf16_group4(
    float value0, float value1, float value2, float value3,
    unsigned char* __restrict__ output_fp8,
    float* __restrict__ output_scale,
    unsigned int row, unsigned int K, unsigned int chunk,
    unsigned int warp, unsigned int lane,
    float smem_abs[EXL3_HROW_WARPS][W2A8_EMIT_GROUP_K]) {
    // These explicit round/re-expand pairs are the incumbent producer store
    // followed by the standalone quantizer load, without global materialization.
    const __nv_bfloat16 boundary0 = __float2bfloat16(value0);
    const __nv_bfloat16 boundary1 = __float2bfloat16(value1);
    const __nv_bfloat16 boundary2 = __float2bfloat16(value2);
    const __nv_bfloat16 boundary3 = __float2bfloat16(value3);
    const float rounded0 = __bfloat162float(boundary0);
    const float rounded1 = __bfloat162float(boundary1);
    const float rounded2 = __bfloat162float(boundary2);
    const float rounded3 = __bfloat162float(boundary3);

    // BEGIN standalone reduction topology
    smem_abs[warp][4 * lane + 0] = fabsf(rounded0);
    smem_abs[warp][4 * lane + 1] = fabsf(rounded1);
    smem_abs[warp][4 * lane + 2] = fabsf(rounded2);
    smem_abs[warp][4 * lane + 3] = fabsf(rounded3);
    __syncwarp();

    // Replay the standalone 128-thread block as four independent 32-lane
    // reductions, including its order-sensitive fmaxf behavior for NaNs.
    float warp_max0 = smem_abs[warp][lane + 0];
    float warp_max1 = smem_abs[warp][lane + 32];
    float warp_max2 = smem_abs[warp][lane + 64];
    float warp_max3 = smem_abs[warp][lane + 96];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_max0 = fmaxf(
            warp_max0, __shfl_down_sync(0xffffffffu, warp_max0, offset));
        warp_max1 = fmaxf(
            warp_max1, __shfl_down_sync(0xffffffffu, warp_max1, offset));
        warp_max2 = fmaxf(
            warp_max2, __shfl_down_sync(0xffffffffu, warp_max2, offset));
        warp_max3 = fmaxf(
            warp_max3, __shfl_down_sync(0xffffffffu, warp_max3, offset));
    }

    float scale = 0.0f;
    if (lane == 0) {
        float global_max = 0.0f;
        global_max = fmaxf(global_max, warp_max0);
        global_max = fmaxf(global_max, warp_max1);
        global_max = fmaxf(global_max, warp_max2);
        global_max = fmaxf(global_max, warp_max3);
        scale = global_max / W2A8_EMIT_E4M3_MAX;
        if (scale < W2A8_EMIT_MIN_SCALE) scale = W2A8_EMIT_MIN_SCALE;
        output_scale[(unsigned long long)row * (K / W2A8_EMIT_GROUP_K) + chunk] = scale;
    }
    scale = __shfl_sync(0xffffffffu, scale, 0);
    // END standalone reduction topology

    float quant0 = rounded0 / scale;
    float quant1 = rounded1 / scale;
    float quant2 = rounded2 / scale;
    float quant3 = rounded3 / scale;
    quant0 = fmaxf(fminf(quant0, W2A8_EMIT_E4M3_MAX), -W2A8_EMIT_E4M3_MAX);
    quant1 = fmaxf(fminf(quant1, W2A8_EMIT_E4M3_MAX), -W2A8_EMIT_E4M3_MAX);
    quant2 = fmaxf(fminf(quant2, W2A8_EMIT_E4M3_MAX), -W2A8_EMIT_E4M3_MAX);
    quant3 = fmaxf(fminf(quant3, W2A8_EMIT_E4M3_MAX), -W2A8_EMIT_E4M3_MAX);
    uchar4 packed;
    packed.x = (unsigned char)__nv_cvt_float_to_fp8(
        quant0, __NV_SATFINITE, __NV_E4M3);
    packed.y = (unsigned char)__nv_cvt_float_to_fp8(
        quant1, __NV_SATFINITE, __NV_E4M3);
    packed.z = (unsigned char)__nv_cvt_float_to_fp8(
        quant2, __NV_SATFINITE, __NV_E4M3);
    packed.w = (unsigned char)__nv_cvt_float_to_fp8(
        quant3, __NV_SATFINITE, __NV_E4M3);
    uchar4* output = reinterpret_cast<uchar4*>(output_fp8 +
        (unsigned long long)row * K + chunk * W2A8_EMIT_GROUP_K + 4 * lane);
    *output = packed;
    __syncwarp();
}

extern "C" __global__ void exl3_w2a8_h128_pre_dual_emit_h4096(
    const __nv_bfloat16* __restrict__ input,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ sorted_expert_ids,
    const unsigned long long* __restrict__ gate_suh_tab,
    const unsigned long long* __restrict__ up_suh_tab,
    unsigned char* __restrict__ gate_fp8,
    float* __restrict__ gate_scale,
    unsigned char* __restrict__ up_fp8,
    float* __restrict__ up_scale,
    unsigned int K, unsigned int rows) {
    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1) return;
    if (K != 4096 || rows != gridDim.x || gridDim.y != 4 || gridDim.z != 1) return;

    const unsigned int row = blockIdx.x;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int chunk = blockIdx.y * EXL3_HROW_WARPS + warp;
    const int expert = sorted_expert_ids[row];
    const long long token = sorted_token_ids
        ? (long long)sorted_token_ids[row] : (long long)row;
    const __nv_bfloat16* source = input + (unsigned long long)token * K +
        chunk * W2A8_EMIT_GROUP_K + 4 * lane;
    const float input0 = __bfloat162float(source[0]);
    const float input1 = __bfloat162float(source[1]);
    const float input2 = __bfloat162float(source[2]);
    const float input3 = __bfloat162float(source[3]);
    __shared__ float smem_abs[EXL3_HROW_WARPS][W2A8_EMIT_GROUP_K];

    float h0, h1, h2, h3;
    const __half* gate_suh =
        (const __half*)gate_suh_tab[expert] + chunk * W2A8_EMIT_GROUP_K;
    exl3_h128_pre_values4(
        input0, input1, input2, input3, gate_suh, lane, h0, h1, h2, h3);
    w2a8_emit_bf16_group4(
        h0 * EXL3_RSQRT128, h1 * EXL3_RSQRT128,
        h2 * EXL3_RSQRT128, h3 * EXL3_RSQRT128,
        gate_fp8, gate_scale, row, K, chunk, warp, lane, smem_abs);

    const __half* up_suh =
        (const __half*)up_suh_tab[expert] + chunk * W2A8_EMIT_GROUP_K;
    exl3_h128_pre_values4(
        input0, input1, input2, input3, up_suh, lane, h0, h1, h2, h3);
    w2a8_emit_bf16_group4(
        h0 * EXL3_RSQRT128, h1 * EXL3_RSQRT128,
        h2 * EXL3_RSQRT128, h3 * EXL3_RSQRT128,
        up_fp8, up_scale, row, K, chunk, warp, lane, smem_abs);
}

extern "C" __global__ void exl3_w2a8_h128_post_silu_pre_emit_h2048(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    const int* __restrict__ sorted_expert_ids,
    const unsigned long long* __restrict__ gate_svh_tab,
    const unsigned long long* __restrict__ up_svh_tab,
    const unsigned long long* __restrict__ down_suh_tab,
    unsigned char* __restrict__ down_fp8,
    float* __restrict__ down_scale,
    unsigned int N, unsigned int rows) {
    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1) return;
    if (N != 2048 || rows != gridDim.x || gridDim.y != 2 || gridDim.z != 1) return;

    const unsigned int row = blockIdx.x;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int chunk = blockIdx.y * EXL3_HROW_WARPS + warp;
    const int expert = sorted_expert_ids[row];
    const __half* gate_svh =
        (const __half*)gate_svh_tab[expert] + chunk * W2A8_EMIT_GROUP_K;
    const __half* up_svh =
        (const __half*)up_svh_tab[expert] + chunk * W2A8_EMIT_GROUP_K;
    const __half* down_suh =
        (const __half*)down_suh_tab[expert] + chunk * W2A8_EMIT_GROUP_K;
    const __nv_bfloat16* gate_row = gate + (unsigned long long)row * N +
        chunk * W2A8_EMIT_GROUP_K + 4 * lane;
    const __nv_bfloat16* up_row = up + (unsigned long long)row * N +
        chunk * W2A8_EMIT_GROUP_K + 4 * lane;

    float g0 = __bfloat162float(gate_row[0]);
    float g1 = __bfloat162float(gate_row[1]);
    float g2 = __bfloat162float(gate_row[2]);
    float g3 = __bfloat162float(gate_row[3]);
    float u0 = __bfloat162float(up_row[0]);
    float u1 = __bfloat162float(up_row[1]);
    float u2 = __bfloat162float(up_row[2]);
    float u3 = __bfloat162float(up_row[3]);
    exl3_had128(g0, g1, g2, g3, lane);
    exl3_had128(u0, u1, u2, u3, lane);

    g0 = __bfloat162float(__float2bfloat16(
        g0 * EXL3_RSQRT128 * __half2float(gate_svh[4 * lane + 0])));
    g1 = __bfloat162float(__float2bfloat16(
        g1 * EXL3_RSQRT128 * __half2float(gate_svh[4 * lane + 1])));
    g2 = __bfloat162float(__float2bfloat16(
        g2 * EXL3_RSQRT128 * __half2float(gate_svh[4 * lane + 2])));
    g3 = __bfloat162float(__float2bfloat16(
        g3 * EXL3_RSQRT128 * __half2float(gate_svh[4 * lane + 3])));
    u0 = __bfloat162float(__float2bfloat16(
        u0 * EXL3_RSQRT128 * __half2float(up_svh[4 * lane + 0])));
    u1 = __bfloat162float(__float2bfloat16(
        u1 * EXL3_RSQRT128 * __half2float(up_svh[4 * lane + 1])));
    u2 = __bfloat162float(__float2bfloat16(
        u2 * EXL3_RSQRT128 * __half2float(up_svh[4 * lane + 2])));
    u3 = __bfloat162float(__float2bfloat16(
        u3 * EXL3_RSQRT128 * __half2float(up_svh[4 * lane + 3])));

    g0 = fminf(g0, EXL3_SWIGLU_LIMIT);
    g1 = fminf(g1, EXL3_SWIGLU_LIMIT);
    g2 = fminf(g2, EXL3_SWIGLU_LIMIT);
    g3 = fminf(g3, EXL3_SWIGLU_LIMIT);
    u0 = fminf(fmaxf(u0, -EXL3_SWIGLU_LIMIT), EXL3_SWIGLU_LIMIT);
    u1 = fminf(fmaxf(u1, -EXL3_SWIGLU_LIMIT), EXL3_SWIGLU_LIMIT);
    u2 = fminf(fmaxf(u2, -EXL3_SWIGLU_LIMIT), EXL3_SWIGLU_LIMIT);
    u3 = fminf(fmaxf(u3, -EXL3_SWIGLU_LIMIT), EXL3_SWIGLU_LIMIT);
    const __nv_bfloat16 activation0 =
        __float2bfloat16(g0 * (1.0f / (1.0f + __expf(-g0))) * u0);
    const __nv_bfloat16 activation1 =
        __float2bfloat16(g1 * (1.0f / (1.0f + __expf(-g1))) * u1);
    const __nv_bfloat16 activation2 =
        __float2bfloat16(g2 * (1.0f / (1.0f + __expf(-g2))) * u2);
    const __nv_bfloat16 activation3 =
        __float2bfloat16(g3 * (1.0f / (1.0f + __expf(-g3))) * u3);
    float d0 = __bfloat162float(activation0) * __half2float(down_suh[4 * lane + 0]);
    float d1 = __bfloat162float(activation1) * __half2float(down_suh[4 * lane + 1]);
    float d2 = __bfloat162float(activation2) * __half2float(down_suh[4 * lane + 2]);
    float d3 = __bfloat162float(activation3) * __half2float(down_suh[4 * lane + 3]);
    exl3_had128(d0, d1, d2, d3, lane);
    __shared__ float smem_abs[EXL3_HROW_WARPS][W2A8_EMIT_GROUP_K];
    w2a8_emit_bf16_group4(
        d0 * EXL3_RSQRT128, d1 * EXL3_RSQRT128,
        d2 * EXL3_RSQRT128, d3 * EXL3_RSQRT128,
        down_fp8, down_scale, row, N, chunk, warp, lane, smem_abs);
}
