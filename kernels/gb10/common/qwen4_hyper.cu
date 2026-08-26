// SPDX-License-Identifier: AGPL-3.0-only

// Qwen4-Exp four-stream gated residual primitives.
// The projection GEMVs stay on Atlas's tuned dense/NVFP4 kernels; these
// kernels implement the architecture-specific normalization and joins.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __constant__ float QWEN4_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Dequantize the 16 selected PLE records into their concatenated 2560-wide
// embedding. Each record is 80 packed E2M1 bytes followed by ten E4M3
// group-16 scales; scale2 is one f32 per physical source shard.
extern "C" __global__ void qwen4_ple_dequant_rows(
    const unsigned char* __restrict__ records,
    const float* __restrict__ scale2,
    __nv_bfloat16* __restrict__ output) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2560u) return;
    const unsigned int head = i / 160u;
    const unsigned int dim = i - head * 160u;
    const unsigned char* record = records + head * 90u;
    const unsigned char packed = record[dim >> 1];
    const unsigned int code = (dim & 1u) ? (packed >> 4) : (packed & 0x0fu);
    __nv_fp8_e4m3 fp8_scale;
    *(unsigned char*)&fp8_scale = record[80u + dim / 16u];
    const float value = QWEN4_E2M1_LUT[code] * (float)fp8_scale * scale2[head];
    output[i] = __float2bfloat16(value);
}

__device__ __forceinline__ float warp_sum(float v);

// Complete one-token PLE join. The projection GEMVs are dispatched through
// Atlas's tuned NVFP4 kernels; this kernel fuses the three grouped RMS norms,
// signed-sqrt gate, dilated depthwise short convolution, state update, and
// residual injection. `conv_state` is [4H, 9], oldest sample first.
extern "C" __global__ void qwen4_ple_fuse_decode(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ norm_key,
    const __nv_bfloat16* __restrict__ norm_query,
    const __nv_bfloat16* __restrict__ norm_conv,
    const __nv_bfloat16* __restrict__ conv_weight,
    __nv_bfloat16* __restrict__ conv_state,
    unsigned int hidden_size,
    float eps,
    unsigned int reset_state) {
    const unsigned int stream_id = blockIdx.x;
    const unsigned int base = stream_id * hidden_size;
    float q2 = 0.0f, k2 = 0.0f;
    for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        const float q = __bfloat162float(hyper[base + h]);
        const float k = __bfloat162float(key[base + h]);
        q2 += q * q;
        k2 += k * k;
    }
    q2 = warp_sum(q2);
    k2 = warp_sum(k2);
    __shared__ float q_parts[32], k_parts[32], shared[4];
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    if (lane == 0) {
        q_parts[warp] = q2;
        k_parts[warp] = k2;
    }
    __syncthreads();
    if (warp == 0) {
        const unsigned int warps = (blockDim.x + 31) / 32;
        float q = lane < warps ? q_parts[lane] : 0.0f;
        float k = lane < warps ? k_parts[lane] : 0.0f;
        q = warp_sum(q);
        k = warp_sum(k);
        if (lane == 0) {
            shared[0] = rsqrtf(q / hidden_size + eps);
            shared[1] = rsqrtf(k / hidden_size + eps);
        }
    }
    __syncthreads();

    float dot = 0.0f;
    for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        const unsigned int i = base + h;
        const float q = __bfloat162float(hyper[i]) * shared[0]
                      * (1.0f + __bfloat162float(norm_query[i]));
        const float k = __bfloat162float(key[i]) * shared[1]
                      * (1.0f + __bfloat162float(norm_key[i]));
        dot += q * k;
    }
    dot = warp_sum(dot);
    if (lane == 0) q_parts[warp] = dot;
    __syncthreads();
    if (warp == 0) {
        const unsigned int warps = (blockDim.x + 31) / 32;
        float v = lane < warps ? q_parts[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) {
            v /= sqrtf((float)hidden_size);
            const float signed_root = copysignf(sqrtf(fmaxf(fabsf(v), 1.0e-6f)), v);
            shared[2] = 1.0f / (1.0f + expf(-signed_root));
        }
    }
    __syncthreads();

    // The gate is scalar per stream, so the RMS of gate*value is obtained
    // directly without materializing a 4H intermediate.
    float gv2 = 0.0f;
    for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        const float gv = shared[2] * __bfloat162float(value[h]);
        gv2 += gv * gv;
    }
    gv2 = warp_sum(gv2);
    if (lane == 0) q_parts[warp] = gv2;
    __syncthreads();
    if (warp == 0) {
        const unsigned int warps = (blockDim.x + 31) / 32;
        float v = lane < warps ? q_parts[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) shared[3] = rsqrtf(v / hidden_size + eps);
    }
    __syncthreads();

    for (unsigned int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        const unsigned int i = base + h;
        const float gv = shared[2] * __bfloat162float(value[h]);
        const float x = gv * shared[3] * (1.0f + __bfloat162float(norm_conv[i]));
        __nv_bfloat16* state = conv_state + i * 9u;
        const float s0 = reset_state ? 0.0f : __bfloat162float(state[0]);
        const float s3 = reset_state ? 0.0f : __bfloat162float(state[3]);
        const float s6 = reset_state ? 0.0f : __bfloat162float(state[6]);
        const __nv_bfloat16* w = conv_weight + i * 4u;
        float c = s0 * __bfloat162float(w[0])
                + s3 * __bfloat162float(w[1])
                + s6 * __bfloat162float(w[2])
                + x  * __bfloat162float(w[3]);
        c = c / (1.0f + expf(-c));
        if (!reset_state) {
            #pragma unroll
            for (unsigned int j = 0; j < 8; ++j) state[j] = state[j + 1];
        } else {
            #pragma unroll
            for (unsigned int j = 0; j < 8; ++j) state[j] = __float2bfloat16(0.0f);
        }
        state[8] = __float2bfloat16(x);
        hyper[i] = __float2bfloat16(__bfloat162float(hyper[i]) + gv + c);
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(0xffffffff, v, d);
    return v;
}

extern "C" __global__ void qwen4_hc_group_norm(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    unsigned int hc_count,
    float eps) {
    const unsigned int token = blockIdx.x;
    const unsigned int stream_id = blockIdx.y;
    if (stream_id >= hc_count) return;
    const unsigned int base = (token * hc_count + stream_id) * hidden_size;
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float x = __bfloat162float(input[base + i]);
        sum += x * x;
    }
    sum = warp_sum(sum);
    __shared__ float partial[32];
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float v = lane < (blockDim.x + 31) / 32 ? partial[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) partial[0] = rsqrtf(v / hidden_size + eps);
    }
    __syncthreads();
    const float inv = partial[0];
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float x = __bfloat162float(input[base + i]);
        const float w = __bfloat162float(weight[stream_id * hidden_size + i]);
        output[base + i] = __float2bfloat16(x * inv * (1.0f + w));
    }
}

extern "C" __global__ void qwen4_hc_silu_div(
    __nv_bfloat16* values, unsigned int numel, float divisor) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    float x = __bfloat162float(values[i]) / divisor;
    values[i] = __float2bfloat16(x / (1.0f + expf(-x)));
}

extern "C" __global__ void qwen4_hc_mix(
    const __nv_bfloat16* __restrict__ normed,
    __nv_bfloat16* __restrict__ mix_logits,
    __nv_bfloat16* __restrict__ inject_logits,
    __nv_bfloat16* __restrict__ mixed,
    unsigned int hidden_size,
    unsigned int hc_count) {
    const unsigned int token = blockIdx.x;
    const unsigned int h = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned int r = hidden_size * hc_count;
    if (h < hidden_size) {
        float acc = 0.0f;
        for (unsigned int s = 0; s < hc_count; ++s) {
            const unsigned int i = token * r + s * hidden_size + h;
            const float g0 = __bfloat162float(mix_logits[i]);
            const float g = 1.0f / (1.0f + expf(-g0));
            acc += g * __bfloat162float(normed[i]);
        }
        mixed[token * hidden_size + h] = __float2bfloat16(acc / hc_count);
    }
    if (blockIdx.y == 0 && h < hc_count) {
        const unsigned int i = token * hc_count + h;
        const float x = __bfloat162float(inject_logits[i]) / hc_count;
        inject_logits[i] = __float2bfloat16(2.0f / (1.0f + expf(-x)));
    }
}

extern "C" __global__ void qwen4_hc_inject(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ core,
    const __nv_bfloat16* __restrict__ inject,
    unsigned int hidden_size,
    unsigned int hc_count) {
    const unsigned int token = blockIdx.x;
    const unsigned int i = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned int r = hidden_size * hc_count;
    if (i >= r) return;
    const unsigned int s = i / hidden_size;
    const unsigned int h = i - s * hidden_size;
    const float old = __bfloat162float(hyper[token * r + i]);
    const float add = __bfloat162float(core[token * hidden_size + h]);
    const float scale = __bfloat162float(inject[token * hc_count + s]);
    hyper[token * r + i] = __float2bfloat16(old + add * scale);
}

// Preserve the per-token injection scales in the dead tail of each normalized
// hyper row. This avoids thousands of tiny D2D copies during long prefill.
extern "C" __global__ void qwen4_hc_save_inject(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ inject,
    unsigned int hidden_size,
    unsigned int hc_count,
    unsigned int num_tokens) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = num_tokens * hc_count;
    if (i >= total) return;
    const unsigned int token = i / hc_count;
    const unsigned int stream_id = i - token * hc_count;
    const unsigned int r = hidden_size * hc_count;
    residual[token * r + (r - hc_count) + stream_id] = inject[i];
}

// Batched injection reading the scales saved by qwen4_hc_save_inject.
extern "C" __global__ void qwen4_hc_inject_saved(
    __nv_bfloat16* __restrict__ hyper,
    const __nv_bfloat16* __restrict__ core,
    const __nv_bfloat16* __restrict__ residual,
    unsigned int hidden_size,
    unsigned int hc_count) {
    const unsigned int token = blockIdx.x;
    const unsigned int i = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned int r = hidden_size * hc_count;
    if (i >= r) return;
    const unsigned int stream_id = i / hidden_size;
    const unsigned int h = i - stream_id * hidden_size;
    const float old = __bfloat162float(hyper[token * r + i]);
    const float add = __bfloat162float(core[token * hidden_size + h]);
    const float scale = __bfloat162float(
        residual[token * r + (r - hc_count) + stream_id]);
    hyper[token * r + i] = __float2bfloat16(old + add * scale);
}

// Stage/pack exact small-M mixed rows through the otherwise-dead first H
// elements of each residual row. This permits an arbitrary-length prefill to
// use byte-exact M<=32 projections while presenting a contiguous [M,H] input
// to the tuned attention/SSM cores.
extern "C" __global__ void qwen4_hc_stage_mixed(
    const __nv_bfloat16* __restrict__ mixed,
    __nv_bfloat16* __restrict__ residual,
    unsigned int token_start,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int hc_count) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = num_tokens * hidden_size;
    if (i >= total) return;
    const unsigned int token = i / hidden_size;
    const unsigned int h = i - token * hidden_size;
    const unsigned int r = hidden_size * hc_count;
    residual[(token_start + token) * r + h] = mixed[i];
}

extern "C" __global__ void qwen4_hc_pack_mixed(
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ mixed,
    unsigned int num_tokens,
    unsigned int hidden_size,
    unsigned int hc_count) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = num_tokens * hidden_size;
    if (i >= total) return;
    const unsigned int token = i / hidden_size;
    const unsigned int h = i - token * hidden_size;
    const unsigned int r = hidden_size * hc_count;
    mixed[i] = residual[token * r + h];
}

extern "C" __global__ void qwen4_hc_expand_embedding(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    unsigned int hc_count) {
    const unsigned int token = blockIdx.x;
    const unsigned int i = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned int r = hidden_size * hc_count;
    if (i < r) output[token * r + i] = input[token * hidden_size + (i % hidden_size)];
}

template <typename InputT>
__device__ __forceinline__ float qwen4_read(InputT x) { return (float)x; }
template <>
__device__ __forceinline__ float qwen4_read(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename InputT>
__device__ void qwen4_gated_rms_sigmoid_body(
    const InputT* input, const __nv_bfloat16* gate,
    const __nv_bfloat16* weight, __nv_bfloat16* output,
    unsigned int hidden_size, float eps, unsigned int gate_stride) {
    const unsigned int token = blockIdx.x;
    const InputT* x = input + token * hidden_size;
    const __nv_bfloat16* z = gate + token * gate_stride;
    __nv_bfloat16* out = output + token * hidden_size;
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float v = qwen4_read(x[i]);
        sum += v * v;
    }
    sum = warp_sum(sum);
    __shared__ float parts[32];
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    if (lane == 0) parts[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float v = lane < (blockDim.x + 31) / 32 ? parts[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) parts[0] = rsqrtf(v / hidden_size + eps);
    }
    __syncthreads();
    const float inv = parts[0];
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float v = qwen4_read(x[i]);
        const float w = __bfloat162float(weight[i]);
        const float g = __bfloat162float(z[i]);
        out[i] = __float2bfloat16(v * inv * w / (1.0f + expf(-g)));
    }
}

extern "C" __global__ void qwen4_gated_rms_norm_sigmoid(
    const __nv_bfloat16* input, const __nv_bfloat16* gate,
    const __nv_bfloat16* weight, __nv_bfloat16* output,
    unsigned int hidden_size, float eps, unsigned int gate_stride,
    unsigned int group_size) {
    (void)group_size;
    qwen4_gated_rms_sigmoid_body(input, gate, weight, output, hidden_size, eps, gate_stride);
}

extern "C" __global__ void qwen4_gated_rms_norm_sigmoid_f32(
    const float* input, const __nv_bfloat16* gate,
    const __nv_bfloat16* weight, __nv_bfloat16* output,
    unsigned int hidden_size, float eps, unsigned int gate_stride,
    unsigned int group_size) {
    (void)group_size;
    qwen4_gated_rms_sigmoid_body(input, gate, weight, output, hidden_size, eps, gate_stride);
}
