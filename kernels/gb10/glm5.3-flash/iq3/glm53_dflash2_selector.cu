// SPDX-License-Identifier: AGPL-3.0-only
// Accuracy-first greedy selector for incoai/GLM-5.3-Flash-DFlash2.

#include <cuda_bf16.h>
#include <math.h>

#define GLM53_DF2_RANK 256U
#define GLM53_DF2_TOP_K 16U
#define GLM53_DF2_VOCAB 154880U
#define GLM53_DF2_MASK 154856U
#define GLM53_DF2_MAX_TOKENS 8U
#define GLM53_DF2_THREADS 32U

extern "C" __global__ void __launch_bounds__(GLM53_DF2_THREADS, 1)
atlas_glm53_dflash2_selector_walk(
        const float * __restrict__ unary,
        const unsigned int * __restrict__ candidates,
        const __nv_bfloat16 * __restrict__ hidden,
        const __nv_bfloat16 * __restrict__ predecessor_codebook,
        const __nv_bfloat16 * __restrict__ successor_codebook,
        const unsigned int * __restrict__ anchors,
        const unsigned int * __restrict__ producer_status,
        unsigned int * __restrict__ path,
        unsigned int * __restrict__ status,
        unsigned int batch,
        unsigned int tokens,
        unsigned int rank,
        unsigned int top_k,
        unsigned int physical_vocab,
        unsigned int logical_vocab,
        unsigned int mask_token_id) {
    if (batch == 0U || tokens == 0U || tokens > GLM53_DF2_MAX_TOKENS ||
        rank != GLM53_DF2_RANK || top_k != GLM53_DF2_TOP_K ||
        physical_vocab != GLM53_DF2_VOCAB ||
        logical_vocab != GLM53_DF2_VOCAB ||
        mask_token_id != GLM53_DF2_MASK || blockIdx.x >= batch) {
        return;
    }

    __shared__ float scores[GLM53_DF2_TOP_K];
    __shared__ unsigned int selected[GLM53_DF2_MAX_TOKENS];
    __shared__ unsigned int predecessor;
    __shared__ unsigned int invalid;
    __shared__ unsigned int abort_walk;

    const unsigned int lane = threadIdx.x;
    const unsigned int batch_index = blockIdx.x;
    const unsigned long long row_base =
        (unsigned long long) batch_index * tokens;
    if (lane == 0U) {
        invalid = 0U;
        abort_walk = 0U;
        predecessor = anchors[batch_index];
        if (predecessor >= physical_vocab) {
            invalid = 1U;
        }
    }
    __syncthreads();

    // Preserve asynchronous failure propagation: no stale candidate or unary
    // value may be read when any top-k row in this batch was rejected.
    if (lane < tokens && producer_status[row_base + lane] != 0U) {
        atomicOr(&invalid, 1U);
    }
    __syncthreads();
    if (invalid != 0U) {
        if (lane == 0U) {
            status[batch_index] = 1U;
        }
        return;
    }

    // Validate the full candidate domain before computing or publishing any
    // path element. The top-k producer promises sixteen unique logical tokens.
    if (lane < GLM53_DF2_TOP_K) {
        for (unsigned int token = 0U; token < tokens; ++token) {
            const unsigned long long slot =
                (row_base + token) * GLM53_DF2_TOP_K + lane;
            const unsigned int candidate = candidates[slot];
            if (candidate >= logical_vocab) {
                atomicOr(&invalid, 1U);
            }
            for (unsigned int prior = 0U; prior < lane; ++prior) {
                if (candidate == candidates[slot - lane + prior]) {
                    atomicOr(&invalid, 1U);
                }
            }
            const float unary_value = unary[slot];
            const __nv_bfloat16 unary_bf16 = __float2bfloat16_rn(unary_value);
            if (!isfinite(unary_value) ||
                __bfloat162float(unary_bf16) != unary_value) {
                atomicOr(&invalid, 1U);
            }
        }
    }
    __syncthreads();
    if (invalid != 0U) {
        if (lane == 0U) {
            status[batch_index] = 1U;
        }
        return;
    }

    // Positions are sequential: the winner at t becomes the predecessor at
    // t+1. Candidate lanes are parallel only within one position.
    for (unsigned int token = 0U; token < tokens; ++token) {
        if (lane < GLM53_DF2_TOP_K) {
            const unsigned long long slot =
                (row_base + token) * GLM53_DF2_TOP_K + lane;
            const unsigned int candidate = candidates[slot];
            const unsigned long long hidden_base =
                (row_base + token) * GLM53_DF2_RANK;
            const unsigned long long pred_base =
                (unsigned long long) predecessor * GLM53_DF2_RANK;
            const unsigned long long succ_base =
                (unsigned long long) candidate * GLM53_DF2_RANK;
            float pair_accumulator = 0.0f;
            #pragma unroll
            for (unsigned int component = 0U;
                 component < GLM53_DF2_RANK; ++component) {
                // PyTorch BF16 semantics materialize this gate product before
                // the pairwise einsum; a fused three-factor product is wrong.
                const __nv_bfloat16 gate = __float2bfloat16_rn(
                    __bfloat162float(predecessor_codebook[pred_base + component]) *
                    __bfloat162float(hidden[hidden_base + component]));
                pair_accumulator = fmaf(
                    __bfloat162float(gate),
                    __bfloat162float(successor_codebook[succ_base + component]),
                    pair_accumulator);
            }
            const __nv_bfloat16 pairwise =
                __float2bfloat16_rn(pair_accumulator);
            const __nv_bfloat16 score = __float2bfloat16_rn(
                unary[slot] + __bfloat162float(pairwise));
            const float widened_score = __bfloat162float(score);
            if (!isfinite(pair_accumulator) || !isfinite(widened_score)) {
                atomicOr(&invalid, 1U);
            }
            scores[lane] = widened_score;
        }
        __syncthreads();

        if (lane == 0U) {
            if (invalid != 0U) {
                abort_walk = 1U;
            } else {
                float best_score = scores[0];
                unsigned int best_slot = 0U;
                #pragma unroll
                for (unsigned int slot = 1U;
                     slot < GLM53_DF2_TOP_K; ++slot) {
                    // Strict greater-than retains the lowest candidate slot
                    // on ties, matching torch.argmax.
                    if (scores[slot] > best_score) {
                        best_score = scores[slot];
                        best_slot = slot;
                    }
                }
                predecessor = candidates[
                    (row_base + token) * GLM53_DF2_TOP_K + best_slot];
                selected[token] = predecessor;
            }
        }
        __syncthreads();
        if (abort_walk != 0U) {
            break;
        }
    }

    if (lane == 0U) {
        status[batch_index] = invalid == 0U ? 0U : 1U;
        if (invalid == 0U) {
            for (unsigned int token = 0U; token < tokens; ++token) {
                path[row_base + token] = selected[token];
            }
        }
    }
}
