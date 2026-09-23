// SPDX-License-Identifier: AGPL-3.0-only
// Architecture equations reimplemented from DeepSeek's MIT reference:
// DeepSeek-V4-Flash-Vision-Exp@6821d6ad3681a4b137b066b76094fa82ebd0a380.
#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void deepseek_vision_rms_norm(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, unsigned hidden, float eps) {
    __shared__ float reduction[256];
    const unsigned row = blockIdx.x, tid = threadIdx.x;
    float sum = 0.0f;
    for (unsigned c = tid; c < hidden; c += 256) {
        float x = __bfloat162float(input[(unsigned long long)row * hidden + c]);
        sum += x * x;
    }
    reduction[tid] = sum;
    __syncthreads();
    for (unsigned step = 128; step; step >>= 1) {
        if (tid < step) reduction[tid] += reduction[tid + step];
        __syncthreads();
    }
    const float inv = rsqrtf(reduction[0] / hidden + eps);
    for (unsigned c = tid; c < hidden; c += 256) {
        float x = __bfloat162float(input[(unsigned long long)row * hidden + c]);
        output[(unsigned long long)row * hidden + c] =
            __float2bfloat16_rn((x * inv) * __bfloat162float(weight[c]));
    }
}

extern "C" __global__ void deepseek_vision_add(
    __nv_bfloat16* dst, const __nv_bfloat16* src, unsigned n) {
    unsigned i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16_rn(__bfloat162float(dst[i]) + __bfloat162float(src[i]));
}

extern "C" __global__ void deepseek_vision_swiglu(
    const __nv_bfloat16* input, __nv_bfloat16* output, unsigned rows, unsigned width) {
    unsigned i = blockIdx.x * 256 + threadIdx.x;
    if (i >= rows * width) return;
    unsigned row = i / width, col = i % width;
    float gate = __bfloat162float(input[(unsigned long long)row * 2 * width + col]);
    float up = __bfloat162float(input[(unsigned long long)row * 2 * width + width + col]);
    // PyTorch's unfused F.silu(gate) * up materializes a BF16 SiLU result.
    float activated = __bfloat162float(__float2bfloat16_rn(gate / (1.0f + expf(-gate))));
    output[i] = __float2bfloat16_rn(activated * up);
}

extern "C" __global__ void deepseek_vision_gelu(__nv_bfloat16* input, unsigned n) {
    unsigned i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) {
        float x = __bfloat162float(input[i]);
        input[i] = __float2bfloat16_rn(0.5f * x * (1.0f + erff(x * 0.7071067811865475244f)));
    }
}

extern "C" __global__ void deepseek_vision_rope(
    const __nv_bfloat16* qkv, __nv_bfloat16* query, __nv_bfloat16* key,
    __nv_bfloat16* value, const float* angles, unsigned patches, unsigned heads, unsigned dim) {
    unsigned i = blockIdx.x * 256 + threadIdx.x;
    const unsigned half = dim / 2, hidden = heads * dim;
    if (i >= patches * heads * half) return;
    const unsigned col = i % half, head = (i / half) % heads, row = i / (half * heads);
    const unsigned long long src = (unsigned long long)row * 3 * hidden + head * dim + col;
    const unsigned long long dst = ((unsigned long long)head * patches + row) * dim + col;
    const float co = angles[(unsigned long long)row * dim + col];
    const float si = angles[(unsigned long long)row * dim + half + col];
    float q1 = __bfloat162float(qkv[src]), q2 = __bfloat162float(qkv[src + half]);
    float k1 = __bfloat162float(qkv[src + hidden]), k2 = __bfloat162float(qkv[src + hidden + half]);
    // Explicit products preserve the reference's FP32 multiply boundaries.
    query[dst] = __float2bfloat16_rn(__fmul_rn(q1, co) - __fmul_rn(q2, si));
    query[dst + half] = __float2bfloat16_rn(__fmul_rn(q2, co) + __fmul_rn(q1, si));
    key[dst] = __float2bfloat16_rn(__fmul_rn(k1, co) - __fmul_rn(k2, si));
    key[dst + half] = __float2bfloat16_rn(__fmul_rn(k2, co) + __fmul_rn(k1, si));
    const unsigned long long vdst = (unsigned long long)head * dim * patches + col * patches + row;
    value[vdst] = qkv[src + 2 * hidden];
    value[vdst + (unsigned long long)half * patches] = qkv[src + 2 * hidden + half];
}

extern "C" __global__ void deepseek_vision_softmax(
    const float* scores, float* probs, unsigned columns, float scale) {
    __shared__ float reduction[256];
    const unsigned tid = threadIdx.x;
    const unsigned long long row = (unsigned long long)blockIdx.x * columns;
    float maximum = -INFINITY;
    for (unsigned c = tid; c < columns; c += 256) maximum = fmaxf(maximum, scores[row + c] * scale);
    reduction[tid] = maximum;
    __syncthreads();
    for (unsigned s = 128; s; s >>= 1) {
        if (tid < s) reduction[tid] = fmaxf(reduction[tid], reduction[tid + s]);
        __syncthreads();
    }
    maximum = reduction[0];
    // All threads must read maximum before reusing the reduction buffer.
    __syncthreads();
    float sum = 0.0f;
    for (unsigned c = tid; c < columns; c += 256) sum += expf(scores[row + c] * scale - maximum);
    reduction[tid] = sum;
    __syncthreads();
    for (unsigned s = 128; s; s >>= 1) {
        if (tid < s) reduction[tid] += reduction[tid + s];
        __syncthreads();
    }
    const float inv = 1.0f / reduction[0];
    for (unsigned c = tid; c < columns; c += 256)
        probs[row + c] = expf(scores[row + c] * scale - maximum) * inv;
}

extern "C" __global__ void deepseek_vision_unfold(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned gh, unsigned gw, unsigned hidden, unsigned ratio) {
    unsigned i = blockIdx.x * 256 + threadIdx.x;
    const unsigned oh = (gh + ratio - 1) / ratio, ow = (gw + ratio - 1) / ratio;
    const unsigned width = hidden * ratio * ratio;
    if (i >= oh * ow * width) return;
    const unsigned row = i / width, c = (i % width) / (ratio * ratio);
    const unsigned dy = (i % (ratio * ratio)) / ratio, dx = i % ratio;
    const unsigned y = (row / ow) * ratio + dy, x = (row % ow) * ratio + dx;
    output[i] = y < gh && x < gw
        ? input[((unsigned long long)y * gw + x) * hidden + c] : __float2bfloat16(0.0f);
}
