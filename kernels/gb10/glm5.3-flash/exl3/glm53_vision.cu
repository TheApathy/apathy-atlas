// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace {

__device__ __forceinline__ float bf16f(const __nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ float f16f(const __half x) {
    return __half2float(x);
}

__device__ __forceinline__ __nv_bfloat16 bf16(const float x) {
    return __float2bfloat16_rn(x);
}

__device__ __forceinline__ float block_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    return scratch[0];
}

}  // namespace

extern "C" __global__ void atlas_glm53_vision_patch_embed(
    const float* __restrict__ pixels,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned rows) {
    constexpr unsigned K = 3 * 2 * 14 * 14;
    constexpr unsigned N = 1024;
    const unsigned row = blockIdx.x;
    const unsigned col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= rows || col >= N) return;
    float sum = f16f(bias[col]);
    const float* x = pixels + static_cast<unsigned long long>(row) * K;
    const __half* w = weight + static_cast<unsigned long long>(col) * K;
    for (unsigned k = 0; k < K; ++k) sum = fmaf(x[k], f16f(w[k]), sum);
    output[static_cast<unsigned long long>(row) * N + col] = bf16(sum);
}

extern "C" __global__ void atlas_glm53_vision_rmsnorm(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned rows,
    unsigned width,
    float epsilon) {
    extern __shared__ float scratch[];
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    const unsigned long long base = static_cast<unsigned long long>(row) * width;
    float sum = 0.0f;
    for (unsigned col = threadIdx.x; col < width; col += blockDim.x) {
        const float x = bf16f(input[base + col]);
        sum = fmaf(x, x, sum);
    }
    const float inv = rsqrtf(block_sum(sum, scratch) / static_cast<float>(width) + epsilon);
    for (unsigned col = threadIdx.x; col < width; col += blockDim.x)
        output[base + col] = bf16(bf16f(input[base + col]) * inv * bf16f(weight[col]));
}

extern "C" __global__ void atlas_glm53_vision_qkv_prepare(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const __half* __restrict__ q_bias,
    const __half* __restrict__ k_bias,
    const __half* __restrict__ v_bias,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    unsigned rows,
    unsigned grid_width,
    float epsilon) {
    __shared__ float qv[64];
    __shared__ float kv[64];
    __shared__ float qr[64];
    __shared__ float kr[64];
    __shared__ float reduce[64];
    const unsigned row = blockIdx.x;
    const unsigned head = blockIdx.y;
    const unsigned d = threadIdx.x;
    if (row >= rows || head >= 16 || d >= 64) return;
    const unsigned idx = (row * 16 + head) * 64 + d;
    qv[d] = bf16f(q_in[idx]) + f16f(q_bias[head * 64 + d]);
    kv[d] = bf16f(k_in[idx]) + f16f(k_bias[head * 64 + d]);
    v_out[idx] = bf16(bf16f(v_in[idx]) + f16f(v_bias[head * 64 + d]));
    __syncthreads();
    const float q_inv = rsqrtf(block_sum(qv[d] * qv[d], reduce) * (1.0f / 64.0f) + epsilon);
    qr[d] = qv[d] * q_inv * bf16f(q_weight[d]);
    __syncthreads();
    const float k_inv = rsqrtf(block_sum(kv[d] * kv[d], reduce) * (1.0f / 64.0f) + epsilon);
    kr[d] = kv[d] * k_inv * bf16f(k_weight[d]);
    __syncthreads();

    // Processor rows are 2x2 merge-major. GLM builds 32 rotary frequencies by
    // concatenating 16 height and 16 width frequencies, then applies NeoX RoPE
    // to the full 64-d head (first 32 paired with the second 32).
    if (d < 32) {
        const unsigned groups_w = grid_width / 2;
        const unsigned group = row / 4;
        const unsigned inner = row & 3;
        const unsigned h = (group / groups_w) * 2 + inner / 2;
        const unsigned w = (group % groups_w) * 2 + inner % 2;
        const unsigned pos = d < 16 ? h : w;
        const unsigned freq = d & 15;
        const float inv_freq = powf(10000.0f, -2.0f * static_cast<float>(freq) / 32.0f);
        float s, c;
        sincosf(static_cast<float>(pos) * inv_freq, &s, &c);
        const float q1 = qr[d], q2 = qr[d + 32];
        const float k1 = kr[d], k2 = kr[d + 32];
        q_out[(row * 16 + head) * 64 + d] = bf16(q1 * c - q2 * s);
        q_out[(row * 16 + head) * 64 + d + 32] = bf16(q2 * c + q1 * s);
        k_out[(row * 16 + head) * 64 + d] = bf16(k1 * c - k2 * s);
        k_out[(row * 16 + head) * 64 + d + 32] = bf16(k2 * c + k1 * s);
    }
}

extern "C" __global__ void atlas_glm53_vision_attention(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ output,
    unsigned rows) {
    __shared__ float reduce[64];
    __shared__ float alpha;
    __shared__ float beta;
    __shared__ float denom;
    __shared__ float running_max;
    const unsigned query = blockIdx.x;
    const unsigned head = blockIdx.y;
    const unsigned d = threadIdx.x;
    if (query >= rows || head >= 16 || d >= 64) return;
    const unsigned qbase = (query * 16 + head) * 64;
    float acc = 0.0f;
    if (d == 0) {
        denom = 0.0f;
        running_max = -CUDART_INF_F;
    }
    __syncthreads();
    for (unsigned key = 0; key < rows; ++key) {
        const unsigned kbase = (key * 16 + head) * 64;
        const float dot = block_sum(bf16f(q[qbase + d]) * bf16f(k[kbase + d]), reduce);
        if (d == 0) {
            const float score = dot * 0.125f;
            const float next_max = fmaxf(running_max, score);
            alpha = expf(running_max - next_max);
            beta = expf(score - next_max);
            denom = denom * alpha + beta;
            running_max = next_max;
        }
        __syncthreads();
        acc = acc * alpha + bf16f(v[kbase + d]) * beta;
        __syncthreads();
    }
    output[qbase + d] = bf16(acc / denom);
}

extern "C" __global__ void atlas_glm53_vision_bias_residual(
    const __nv_bfloat16* __restrict__ projected,
    const __half* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ output,
    unsigned rows,
    unsigned width) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long count = static_cast<unsigned long long>(rows) * width;
    if (index < count) {
        const unsigned col = static_cast<unsigned>(index % width);
        output[index] = bf16(bf16f(projected[index]) + f16f(bias[col]) + bf16f(residual[index]));
    }
}

extern "C" __global__ void atlas_glm53_vision_swiglu_bias(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    const __half* __restrict__ gate_bias,
    const __half* __restrict__ up_bias,
    __nv_bfloat16* __restrict__ output,
    unsigned rows,
    unsigned width,
    float limit,
    unsigned has_bias) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long count = static_cast<unsigned long long>(rows) * width;
    if (index < count) {
        const unsigned col = static_cast<unsigned>(index % width);
        float g = bf16f(gate[index]) + (has_bias ? f16f(gate_bias[col]) : 0.0f);
        float u = bf16f(up[index]) + (has_bias ? f16f(up_bias[col]) : 0.0f);
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
        const __nv_bfloat16 silu = bf16(g / (1.0f + expf(-g)));
        const __nv_bfloat16 up_narrow = bf16(u);
        output[index] = bf16(bf16f(silu) * bf16f(up_narrow));
    }
}

extern "C" __global__ void atlas_glm53_vision_post_downsample(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __half* __restrict__ conv_weight,
    const __half* __restrict__ conv_bias,
    __nv_bfloat16* __restrict__ output,
    unsigned merged_rows,
    float epsilon) {
    __shared__ float reduce[256];
    __shared__ float inv[4];
    const unsigned merged = blockIdx.x;
    const unsigned out_col = blockIdx.y * blockDim.x + threadIdx.x;
    if (merged >= merged_rows) return;
    for (unsigned patch = 0; patch < 4; ++patch) {
        float sum = 0.0f;
        const unsigned long long base = (static_cast<unsigned long long>(merged) * 4 + patch) * 1024;
        for (unsigned col = threadIdx.x; col < 1024; col += blockDim.x) {
            const float x = bf16f(input[base + col]);
            sum = fmaf(x, x, sum);
        }
        const float total = block_sum(sum, reduce);
        if (threadIdx.x == 0) inv[patch] = rsqrtf(total * (1.0f / 1024.0f) + epsilon);
        __syncthreads();
    }
    if (out_col >= 4096) return;
    float sum = f16f(conv_bias[out_col]);
    const __half* w = conv_weight + static_cast<unsigned long long>(out_col) * 1024 * 4;
    for (unsigned in_col = 0; in_col < 1024; ++in_col) {
        const float nw = bf16f(norm_weight[in_col]);
        for (unsigned patch = 0; patch < 4; ++patch) {
            const unsigned long long base = (static_cast<unsigned long long>(merged) * 4 + patch) * 1024;
            // Conv2d OIHW stores [kh,kw] in row-major; merge-major inner rows
            // use that same [00,01,10,11] order.
            sum = fmaf(bf16f(input[base + in_col]) * inv[patch] * nw,
                       f16f(w[in_col * 4 + patch]), sum);
        }
    }
    output[static_cast<unsigned long long>(merged) * 4096 + out_col] = bf16(sum);
}

extern "C" __global__ void atlas_glm53_vision_layernorm_gelu(
    const __nv_bfloat16* __restrict__ input,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned rows,
    unsigned width,
    float epsilon) {
    extern __shared__ float scratch[];
    __shared__ float mean;
    __shared__ float inv;
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    const unsigned long long base = static_cast<unsigned long long>(row) * width;
    float sum = 0.0f;
    for (unsigned col = threadIdx.x; col < width; col += blockDim.x) sum += bf16f(input[base + col]);
    const float total = block_sum(sum, scratch);
    if (threadIdx.x == 0) mean = total / static_cast<float>(width);
    __syncthreads();
    float square = 0.0f;
    for (unsigned col = threadIdx.x; col < width; col += blockDim.x) {
        const float delta = bf16f(input[base + col]) - mean;
        square = fmaf(delta, delta, square);
    }
    const float square_total = block_sum(square, scratch);
    if (threadIdx.x == 0) inv = rsqrtf(square_total / static_cast<float>(width) + epsilon);
    __syncthreads();
    for (unsigned col = threadIdx.x; col < width; col += blockDim.x) {
        const float x = (bf16f(input[base + col]) - mean) * inv * f16f(weight[col]) + f16f(bias[col]);
        const float gelu = 0.5f * x * (1.0f + erff(x * CUDART_SQRT_HALF_F));
        output[base + col] = bf16(gelu);
    }
}
