// SPDX-License-Identifier: AGPL-3.0-only
//
// `atlas_glm53_kda_conv_f32_stage_fused`: the KDA short-conv stage, reading the
// EXL3 combined projection directly.
//
// The donor pair is `atlas_glm53_kda_split_qkv` + `atlas_glm53_kda_conv_f32_stage`
// (`kernels/gb10/glm5.3-flash/iq3/glm53_kda.cu`, `..._kda_conv.cu`). The EXL3
// fused qkv projection lands as row-major `[tokens][3][8192]`; the split kernel
// rewrites it into three dense `[tokens][8192]` planes purely because the conv
// contract wants three pointers. At 2047 tokens that is a 100 MB read plus a
// 100 MB write per layer, 0.82 ms/layer x 34 layers = 28 ms per prefill, and it
// already runs at its memory roofline -- the only way to make it faster is not
// to do it. Here the conv indexes the combined buffer itself:
//
//     donor : input[(batch*tokens + t)*8192 + channel]          (split plane)
//     fused : combined[((batch*tokens + t)*3 + stream)*8192 + channel]
//
// `blockIdx.z` already selects the stream, so this is an index change and
// nothing else. The conv arithmetic, its f32 accumulation, the SiLU and the
// bf16 store are untouched, so the outputs are bit-identical to the donor pair.
//
// Second change: the donor is `__launch_bounds__(GLM53_KDA_CONV_THREADS, 1)` and
// launches grid (32, batch, 3) = 96 blocks of 256 threads. One block per SM over
// 48 SMs means those 96 blocks run as TWO waves while each thread serially walks
// all 2047 tokens. Raising the minimum to 4 lets the whole grid be resident at
// once; the kernel holds 8 live floats and no shared memory, so the register
// budget is not the constraint.

#include <cuda_bf16.h>

#define GF_CONV_THREADS 256U
#define GF_CONV_CHANNELS 8192U
#define GF_CONV_STREAMS 3U
#define GF_CONV_KERNEL 4U
#define GF_CONV_MAX_BATCH 64U
#define GF_CONV_MAX_QUERIES 8192U

extern "C" __global__ void __launch_bounds__(GF_CONV_THREADS, 4)
atlas_glm53_kda_conv_f32_stage_fused(
        const __nv_bfloat16 * __restrict__ combined,
        const float * __restrict__ q_weight,
        const float * __restrict__ k_weight,
        const float * __restrict__ v_weight,
        const float * __restrict__ persistent_state,
        float * __restrict__ staged_state,
        __nv_bfloat16 * __restrict__ q_output,
        __nv_bfloat16 * __restrict__ k_output,
        __nv_bfloat16 * __restrict__ v_output,
        unsigned int batch, unsigned int query_count) {
    if (combined == nullptr || q_weight == nullptr || k_weight == nullptr ||
        v_weight == nullptr || persistent_state == nullptr ||
        staged_state == nullptr || q_output == nullptr || k_output == nullptr ||
        v_output == nullptr ||
        batch == 0U || batch > GF_CONV_MAX_BATCH ||
        query_count == 0U || query_count > GF_CONV_MAX_QUERIES ||
        blockDim.x != GF_CONV_THREADS || blockDim.y != 1U || blockDim.z != 1U ||
        gridDim.x != 32U || gridDim.y != batch ||
        gridDim.z != GF_CONV_STREAMS) {
        return;
    }

    const unsigned int stream_index = blockIdx.z;
    const unsigned int batch_index = blockIdx.y;
    const unsigned int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= GF_CONV_CHANNELS) {
        return;
    }
    const float *weight = stream_index == 0U ? q_weight :
        (stream_index == 1U ? k_weight : v_weight);
    __nv_bfloat16 *output = stream_index == 0U ? q_output :
        (stream_index == 1U ? k_output : v_output);

    const unsigned long long state_base =
        (((unsigned long long) batch_index * GF_CONV_STREAMS + stream_index) *
             GF_CONV_CHANNELS + channel) * GF_CONV_KERNEL;
    float s0 = persistent_state[state_base + 0ULL];
    float s1 = persistent_state[state_base + 1ULL];
    float s2 = persistent_state[state_base + 2ULL];
    float s3 = persistent_state[state_base + 3ULL];
    const unsigned long long weight_base =
        (unsigned long long) channel * GF_CONV_KERNEL;
    const float w0 = weight[weight_base + 0ULL];
    const float w1 = weight[weight_base + 1ULL];
    const float w2 = weight[weight_base + 2ULL];
    const float w3 = weight[weight_base + 3ULL];

    for (unsigned int token = 0U; token < query_count; ++token) {
        const unsigned long long row =
            (unsigned long long) batch_index * query_count + token;
        // The only difference from the donor: read the interleaved
        // [tokens][3][8192] projection in place instead of a split plane.
        const float newest = (float) combined[
            (row * GF_CONV_STREAMS + stream_index) * GF_CONV_CHANNELS + channel];
        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = newest;
        const float correlation = s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3;
        const float activated = correlation / (1.0f + __expf(-correlation));
        output[row * GF_CONV_CHANNELS + channel] = __float2bfloat16(activated);
    }
    staged_state[state_base + 0ULL] = s0;
    staged_state[state_base + 1ULL] = s1;
    staged_state[state_base + 2ULL] = s2;
    staged_state[state_base + 3ULL] = s3;
}
