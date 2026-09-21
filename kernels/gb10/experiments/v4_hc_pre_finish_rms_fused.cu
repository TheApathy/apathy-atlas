// SPDX-License-Identifier: AGPL-3.0-only

// DeepSeek-V4 exact-shape candidate: fuse the prefill-width hc_pre_finish
// collapse with its immediately following RMS norm. The collapse is rounded
// to BF16 in registers, stored to hidden_out, and widened again before the RMS
// sum, preserving the incumbent materialization boundary and hidden-state
// output. DeepSeek-V4 vanilla RMSNorm weights use `out = x * rms * weight`,
// never the Qwen offset convention `(1 + weight)`. A 1024-thread block keeps
// rms_norm_vanilla's production H=4096 packed-word ownership and reduction
// tree. A target-specific production wrapper registers this implementation;
// host dispatch remains strict default-off and exact-shape guarded.

#include <cuda_bf16.h>

#define V4_HC_H 4096u
#define V4_HC_MULT 4u
#define V4_HC_MIX ((2u + V4_HC_MULT) * V4_HC_MULT)
#define V4_HC_BLOCK 1024u
#define V4_HC_HALF (V4_HC_H / 2u)

static_assert(V4_HC_MIX == 24u, "DeepSeek-V4 hc mix width changed");
static_assert(V4_HC_HALF == 2u * V4_HC_BLOCK,
              "two packed BF16 words per thread are required");

__device__ __forceinline__ float v4_hc_widen_bf16(const unsigned short bits) {
    return __bfloat162float(__ushort_as_bfloat16(bits));
}

__device__ __forceinline__ unsigned int v4_hc_pack_bf16(const float x0,
                                                         const float x1) {
    const unsigned int lo =
        static_cast<unsigned int>(__bfloat16_as_ushort(__float2bfloat16(x0)));
    const unsigned int hi =
        static_cast<unsigned int>(__bfloat16_as_ushort(__float2bfloat16(x1)));
    return lo | (hi << 16);
}

__device__ __forceinline__ void v4_hc_unpack_bf16(const unsigned int packed,
                                                   float& x0,
                                                   float& x1) {
    x0 = v4_hc_widen_bf16(static_cast<unsigned short>(packed));
    x1 = v4_hc_widen_bf16(static_cast<unsigned short>(packed >> 16));
}

__device__ __forceinline__ float v4_hc_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float v4_hc_collapse_dim(
    const float* __restrict__ streams,
    const float* __restrict__ pre,
    const unsigned int dim) {
    float value = 0.0f;
#pragma unroll
    for (unsigned int stream = 0; stream < V4_HC_MULT; ++stream) {
        value += pre[stream] * streams[stream * V4_HC_H + dim];
    }
    return value;
}

__device__ __forceinline__ unsigned int v4_hc_collapse_pair(
    const float* __restrict__ streams,
    const float* __restrict__ pre,
    const unsigned int pair) {
    const unsigned int dim = pair * 2u;
    return v4_hc_pack_bf16(v4_hc_collapse_dim(streams, pre, dim),
                           v4_hc_collapse_dim(streams, pre, dim + 1u));
}

extern "C" __global__ __launch_bounds__(1024, 1)
void v4_hc_pre_finish_rms_fused(
    const float* __restrict__ streams,
    const float* __restrict__ mix_in,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ hidden_out,
    __nv_bfloat16* __restrict__ normed_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    const unsigned int num_tokens,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int sinkhorn_iters,
    const float hc_norm_eps,
    const float hc_eps,
    const float rms_eps) {
    if (num_tokens == 0 || streams == nullptr || mix_in == nullptr ||
        hc_scale == nullptr || hc_base == nullptr || norm_weight == nullptr ||
        hidden_out == nullptr || normed_out == nullptr || post_out == nullptr ||
        comb_out == nullptr ||
        hidden_size != V4_HC_H || hc_mult != V4_HC_MULT ||
        blockDim.x != V4_HC_BLOCK || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.x != num_tokens || gridDim.y != 1 || gridDim.z != 1) {
        return;
    }

    const unsigned int token = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const float* const x = streams + static_cast<size_t>(token) * V4_HC_MULT * V4_HC_H;
    const float* const mi = mix_in + static_cast<size_t>(token) * (V4_HC_MIX + 1u);

    __shared__ float s_pre[V4_HC_MULT];
    __shared__ float warp_sums[32];

    if (tid == 0) {
        const float hc_rsqrt =
            rsqrtf(mi[V4_HC_MIX] / static_cast<float>(V4_HC_MULT * V4_HC_H) +
                   hc_norm_eps);
        float s_mix[V4_HC_MIX];
#pragma unroll
        for (unsigned int m = 0; m < V4_HC_MIX; ++m) {
            s_mix[m] = mi[m] * hc_rsqrt;
        }

#pragma unroll
        for (unsigned int i = 0; i < V4_HC_MULT; ++i) {
            const float pre = s_mix[i] * hc_scale[0] + hc_base[i];
            s_pre[i] = 1.0f / (1.0f + expf(-pre)) + hc_eps;
            const float post =
                s_mix[V4_HC_MULT + i] * hc_scale[1] + hc_base[V4_HC_MULT + i];
            post_out[static_cast<size_t>(token) * V4_HC_MULT + i] =
                2.0f * (1.0f / (1.0f + expf(-post)));
        }

        float comb[V4_HC_MULT * V4_HC_MULT];
#pragma unroll
        for (unsigned int i = 0; i < V4_HC_MULT; ++i) {
#pragma unroll
            for (unsigned int j = 0; j < V4_HC_MULT; ++j) {
                const unsigned int index = i * V4_HC_MULT + j;
                comb[index] = s_mix[2u * V4_HC_MULT + index] * hc_scale[2] +
                              hc_base[2u * V4_HC_MULT + index];
            }
        }
#pragma unroll
        for (unsigned int i = 0; i < V4_HC_MULT; ++i) {
            float maximum = -1.0e30f;
#pragma unroll
            for (unsigned int j = 0; j < V4_HC_MULT; ++j) {
                maximum = fmaxf(maximum, comb[i * V4_HC_MULT + j]);
            }
            float sum = 0.0f;
#pragma unroll
            for (unsigned int j = 0; j < V4_HC_MULT; ++j) {
                const float value = expf(comb[i * V4_HC_MULT + j] - maximum);
                comb[i * V4_HC_MULT + j] = value;
                sum += value;
            }
#pragma unroll
            for (unsigned int j = 0; j < V4_HC_MULT; ++j) {
                comb[i * V4_HC_MULT + j] =
                    comb[i * V4_HC_MULT + j] / sum + hc_eps;
            }
        }
#pragma unroll
        for (unsigned int j = 0; j < V4_HC_MULT; ++j) {
            float column = hc_eps;
#pragma unroll
            for (unsigned int i = 0; i < V4_HC_MULT; ++i) {
                column += comb[i * V4_HC_MULT + j];
            }
#pragma unroll
            for (unsigned int i = 0; i < V4_HC_MULT; ++i) {
                comb[i * V4_HC_MULT + j] /= column;
            }
        }
        for (unsigned int iteration = 0; iteration + 1u < sinkhorn_iters;
             ++iteration) {
#pragma unroll
            for (unsigned int i = 0; i < V4_HC_MULT; ++i) {
                float row = hc_eps;
#pragma unroll
                for (unsigned int j = 0; j < V4_HC_MULT; ++j) {
                    row += comb[i * V4_HC_MULT + j];
                }
#pragma unroll
                for (unsigned int j = 0; j < V4_HC_MULT; ++j) {
                    comb[i * V4_HC_MULT + j] /= row;
                }
            }
#pragma unroll
            for (unsigned int j = 0; j < V4_HC_MULT; ++j) {
                float column = hc_eps;
#pragma unroll
                for (unsigned int i = 0; i < V4_HC_MULT; ++i) {
                    column += comb[i * V4_HC_MULT + j];
                }
#pragma unroll
                for (unsigned int i = 0; i < V4_HC_MULT; ++i) {
                    comb[i * V4_HC_MULT + j] /= column;
                }
            }
        }
#pragma unroll
        for (unsigned int j = 0; j < V4_HC_MULT; ++j) {
            float column = 0.0f;
#pragma unroll
            for (unsigned int i = 0; i < V4_HC_MULT; ++i) {
                column += comb[i * V4_HC_MULT + j];
            }
            const float inverse = column > 0.0f ? 1.0f / column : 0.0f;
#pragma unroll
            for (unsigned int i = 0; i < V4_HC_MULT; ++i) {
                comb[i * V4_HC_MULT + j] *= inverse;
            }
        }
#pragma unroll
        for (unsigned int i = 0; i < V4_HC_MULT; ++i) {
#pragma unroll
            for (unsigned int j = 0; j < V4_HC_MULT; ++j) {
                comb_out[static_cast<size_t>(token) * V4_HC_MULT * V4_HC_MULT +
                         i * V4_HC_MULT + j] = comb[i * V4_HC_MULT + j];
            }
        }
    }
    __syncthreads();

    const unsigned int pair0 = tid;
    const unsigned int pair1 = tid + V4_HC_BLOCK;
    const unsigned int collapsed0 = v4_hc_collapse_pair(x, s_pre, pair0);
    const unsigned int collapsed1 = v4_hc_collapse_pair(x, s_pre, pair1);
    unsigned int* const hidden = reinterpret_cast<unsigned int*>(
        hidden_out + static_cast<size_t>(token) * V4_HC_H);
    hidden[pair0] = collapsed0;
    hidden[pair1] = collapsed1;
    float x00, x01, x10, x11;
    v4_hc_unpack_bf16(collapsed0, x00, x01);
    v4_hc_unpack_bf16(collapsed1, x10, x11);

    // Same two packed-word accumulation and 1024-thread reduction tree as
    // incumbent rms_norm_vanilla at H=4096.
    float sum_sq = 0.0f;
    sum_sq += x00 * x00 + x01 * x01;
    sum_sq += x10 * x10 + x11 * x11;
    sum_sq = v4_hc_warp_sum(sum_sq);
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    if (lane == 0) {
        warp_sums[warp] = sum_sq;
    }
    __syncthreads();
    if (warp == 0) {
        float value = warp_sums[lane];
        value = v4_hc_warp_sum(value);
        if (lane == 0) {
            warp_sums[0] = value;
        }
    }
    __syncthreads();
    const float rms =
        rsqrtf(warp_sums[0] / static_cast<float>(V4_HC_H) + rms_eps);

    const unsigned int* const weights =
        reinterpret_cast<const unsigned int*>(norm_weight);
    unsigned int* const output = reinterpret_cast<unsigned int*>(
        normed_out + static_cast<size_t>(token) * V4_HC_H);
    float w00, w01, w10, w11;
    v4_hc_unpack_bf16(weights[pair0], w00, w01);
    v4_hc_unpack_bf16(weights[pair1], w10, w11);
    output[pair0] = v4_hc_pack_bf16(x00 * rms * w00, x01 * rms * w01);
    output[pair1] = v4_hc_pack_bf16(x10 * rms * w10, x11 * rms * w11);
}
