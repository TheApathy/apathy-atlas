// SPDX-License-Identifier: AGPL-3.0-only

// Actual DeepSeek-V4 Flash Vision: image sentinel IDs select bias_vl even in
// the first three hash layers. Text keeps the existing Atlas hash/top-k math,
// reduction order, tie rule, normalization threshold, and BF16 logits ABI.
// Host validates shapes/IDs; device traps on invalid IDs/nonfinite scores
// before dereferencing an invalid table or dispatching a bogus expert.

#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void deepseek_visual_route(
    const __nv_bfloat16* __restrict__ gate_logits,
    const long* __restrict__ tid2eid,
    const unsigned int* __restrict__ token_ids,
    const float* __restrict__ text_bias,
    const float* __restrict__ visual_bias,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    unsigned int vocab_size,
    float scaling_factor
) {
    constexpr unsigned int EXPERTS = 256;
    constexpr unsigned int TOP_K = 6;
    const unsigned int token = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int tok = token_ids[token];
    if (tok >= vocab_size + 5u) {
        asm volatile("trap;");
        return;
    }
    const bool image = tok >= vocab_size;
    const __nv_bfloat16* gate = gate_logits + (size_t)token * EXPERTS;
    unsigned int* indices = expert_indices + (size_t)token * TOP_K;
    float* weights = expert_weights + (size_t)token * TOP_K;

    // Critically, the image predicate is tested BEFORE indexing tid2eid.
    if (tid2eid != nullptr && !image) {
        if (tid != 0) return;
        const long* row = tid2eid + (size_t)tok * TOP_K;
        float selected[TOP_K];
        unsigned int ids[TOP_K];
        float sum = 0.0f;
        for (unsigned int k = 0; k < TOP_K; ++k) {
            const long expert = row[k];
            if (expert < 0 || expert >= EXPERTS) {
                asm volatile("trap;");
                return;
            }
            ids[k] = (unsigned int)expert;
            const float logit = __bfloat162float(gate[ids[k]]);
            selected[k] = sqrtf(logf(1.0f + __expf(logit)));
            if (!isfinite(logit) || !isfinite(selected[k])) {
                asm volatile("trap;");
                return;
            }
            sum += selected[k];
        }
        if (sum > 1e-20f) {
            for (unsigned int k = 0; k < TOP_K; ++k) selected[k] /= sum;
        }
        for (unsigned int k = 0; k < TOP_K; ++k) {
            indices[k] = ids[k];
            weights[k] = selected[k] * scaling_factor;
        }
        return;
    }

    __shared__ float score[EXPERTS];
    __shared__ float selection[EXPERTS];
    __shared__ float selected[TOP_K];
    __shared__ unsigned int selected_ids[TOP_K];
    __shared__ float warp_val[8];
    __shared__ unsigned int warp_idx[8];

    const float logit = __bfloat162float(gate[tid]);
    const float raw = sqrtf(logf(1.0f + __expf(logit)));
    const float bias = image ? visual_bias[tid] : text_bias[tid];
    score[tid] = raw;
    selection[tid] = raw + bias;
    if (!isfinite(logit) || !isfinite(raw) || !isfinite(selection[tid])) {
        asm volatile("trap;");
        return;
    }
    __syncthreads();

    // This is the existing moe_topk_sqrtsoftplus reduction, including strict
    // > comparisons (the existing shuffle-tree tie order) and -1e30 invalidation.
    for (unsigned int k = 0; k < TOP_K; ++k) {
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        if (selection[tid] > local_max) {
            local_max = selection[tid];
            local_idx = tid;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_down_sync(0xffffffff, local_max, offset);
            const unsigned int idx = __shfl_down_sync(0xffffffff, local_idx, offset);
            if (other > local_max) {
                local_max = other;
                local_idx = idx;
            }
        }
        if (tid % 32 == 0) {
            warp_val[tid / 32] = local_max;
            warp_idx[tid / 32] = local_idx;
        }
        __syncthreads();
        if (tid == 0) {
            float best = warp_val[0];
            unsigned int index = warp_idx[0];
            for (unsigned int w = 1; w < 8; ++w) {
                if (warp_val[w] > best) {
                    best = warp_val[w];
                    index = warp_idx[w];
                }
            }
            if (!(best > -1e30f)) {
                asm volatile("trap;");
                return;
            }
            selected_ids[k] = index;
            selection[index] = -1e30f;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < TOP_K; ++k) {
            selected[k] = score[selected_ids[k]];
            sum += selected[k];
        }
        if (sum > 1e-20f) {
            for (unsigned int k = 0; k < TOP_K; ++k) selected[k] /= sum;
        }
        for (unsigned int k = 0; k < TOP_K; ++k) {
            indices[k] = selected_ids[k];
            weights[k] = selected[k] * scaling_factor;
        }
    }
}
