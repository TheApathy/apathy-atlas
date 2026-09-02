// SPDX-License-Identifier: AGPL-3.0-only
// Accuracy-first GLM-5.3 MoE routing for the exact one-Spark target.

#include <cuda_bf16.h>
#include <float.h>
#include <math.h>

#define GLM53_HIDDEN 4096U
#define GLM53_EXPERTS 288U
#define GLM53_TOP_K 8U
#define GLM53_THREADS 256U

extern "C" __global__ void __launch_bounds__(GLM53_THREADS, 1)
atlas_glm53_router_logits(
        const __nv_bfloat16 * __restrict__ input,
        const float * __restrict__ router,
        float * __restrict__ logits,
        unsigned int tokens, unsigned int hidden, unsigned int experts) {
    if (tokens == 0 || hidden != GLM53_HIDDEN || experts != GLM53_EXPERTS) {
        return;
    }
    const unsigned long long total =
        (unsigned long long) tokens * (unsigned long long) experts;
    const unsigned long long item = blockIdx.x;
    if (item >= total) {
        return;
    }
    const unsigned int token = (unsigned int) (item / experts);
    const unsigned int expert = (unsigned int) (item % experts);
    const unsigned long long input_base = (unsigned long long) token * hidden;
    const unsigned long long router_base = (unsigned long long) expert * hidden;

    float sum = 0.0f;
    for (unsigned int k = threadIdx.x; k < hidden; k += GLM53_THREADS) {
        sum = fmaf(
            __bfloat162float(input[input_base + k]),
            router[router_base + k],
            sum);
    }

    __shared__ float partial[GLM53_THREADS];
    partial[threadIdx.x] = sum;
    __syncthreads();
    #pragma unroll
    for (unsigned int stride = GLM53_THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        logits[item] = partial[0];
    }
}

extern "C" __global__ void __launch_bounds__(GLM53_THREADS, 1)
atlas_glm53_topk_sigmoid_f32(
        const float * __restrict__ logits,
        const float * __restrict__ bias,
        unsigned int * __restrict__ expert_indices,
        float * __restrict__ expert_weights,
        // Bring-up diagnostics. Both may be null. `probs_out` is sigmoid(logits)
        // BEFORE the bias; `biased_out` is the score that actually selects.
        float * __restrict__ probs_out,
        float * __restrict__ biased_out,
        unsigned int tokens, unsigned int experts, unsigned int top_k,
        float scaling_factor) {
    if (tokens == 0 || experts != GLM53_EXPERTS || top_k != GLM53_TOP_K ||
        scaling_factor != 2.5f || blockIdx.x >= tokens) {
        return;
    }

    __shared__ float sigmoid_scores[GLM53_EXPERTS];
    __shared__ float selection_scores[GLM53_EXPERTS];
    __shared__ float top_values[GLM53_TOP_K];
    __shared__ unsigned int top_indices[GLM53_TOP_K];
    __shared__ float warp_values[GLM53_THREADS / 32];
    __shared__ unsigned int warp_indices[GLM53_THREADS / 32];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31U;
    const unsigned int warp = tid >> 5;
    const unsigned long long base = (unsigned long long) blockIdx.x * experts;
    for (unsigned int expert = tid; expert < experts; expert += GLM53_THREADS) {
        const float score = 1.0f / (1.0f + expf(-logits[base + expert]));
        const float biased = score + bias[expert];
        sigmoid_scores[expert] = score;
        selection_scores[expert] = biased;
        // Snapshot HERE, not after selection: the loop below destroys
        // selection_scores[] by writing -FLT_MAX into each expert as it is
        // picked, so a later dump would show exactly the eight experts we care
        // about sitting at -FLT_MAX. GLM selects on the BIASED score but emits
        // the UNBIASED sigmoid as the weight, so both must be recoverable or a
        // reader cannot tell which ranking a number came from.
        if (probs_out != nullptr) {
            probs_out[base + expert] = score;
        }
        if (biased_out != nullptr) {
            biased_out[base + expert] = biased;
        }
    }
    __syncthreads();

    #pragma unroll
    for (unsigned int selected = 0; selected < GLM53_TOP_K; ++selected) {
        float local_value = -FLT_MAX;
        unsigned int local_index = 0xffffffffU;
        for (unsigned int expert = tid; expert < experts; expert += GLM53_THREADS) {
            const float value = selection_scores[expert];
            if (value > local_value ||
                (value == local_value && expert < local_index)) {
                local_value = value;
                local_index = expert;
            }
        }
        #pragma unroll
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            const float other_value =
                __shfl_down_sync(0xffffffffU, local_value, offset);
            const unsigned int other_index =
                __shfl_down_sync(0xffffffffU, local_index, offset);
            if (other_value > local_value ||
                (other_value == local_value && other_index < local_index)) {
                local_value = other_value;
                local_index = other_index;
            }
        }
        if (lane == 0) {
            warp_values[warp] = local_value;
            warp_indices[warp] = local_index;
        }
        __syncthreads();
        if (tid == 0) {
            float best_value = warp_values[0];
            unsigned int best_index = warp_indices[0];
            #pragma unroll
            for (unsigned int other = 1; other < GLM53_THREADS / 32; ++other) {
                if (warp_values[other] > best_value ||
                    (warp_values[other] == best_value &&
                     warp_indices[other] < best_index)) {
                    best_value = warp_values[other];
                    best_index = warp_indices[other];
                }
            }
            top_indices[selected] = best_index;
            top_values[selected] = sigmoid_scores[best_index];
            selection_scores[best_index] = -FLT_MAX;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (unsigned int selected = 0; selected < GLM53_TOP_K; ++selected) {
            sum += top_values[selected];
        }
        const float denominator = sum + 1.0e-20f;
        const unsigned long long output_base =
            (unsigned long long) blockIdx.x * GLM53_TOP_K;
        #pragma unroll
        for (unsigned int selected = 0; selected < GLM53_TOP_K; ++selected) {
            expert_indices[output_base + selected] = top_indices[selected];
            expert_weights[output_base + selected] =
                (top_values[selected] / denominator) * scaling_factor;
        }
    }
}
