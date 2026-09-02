// SPDX-License-Identifier: AGPL-3.0-only
// Direct Q6_K token-row gather for the pinned GLM-5.3 IQ3 embedding.
// Q6 index/arithmetic order follows ggml-org/llama.cpp dequantize_q6_K at
// ca3d5a3e10d53f7ea672cb9b6178faca3e2807bc.

#include <cuda_bf16.h>
#include "../../qwen3.6-27b/nvfp4/q4k_vendor/mmq.cuh"

static_assert(QK_K == 256, "Q6_K block geometry drift");
static_assert(sizeof(block_q6_K) == 210, "Q6_K block extent drift");

namespace {
constexpr unsigned int kVocab = 154880;
constexpr unsigned int kHidden = 4096;
constexpr unsigned int kMaxTokens = 65520;
constexpr unsigned int kBlocksPerRow = kHidden / QK_K;
}  // namespace

extern "C" __global__ void __launch_bounds__(256, 1)
atlas_q6_k_token_embedding_bf16(
        const block_q6_K * __restrict__ source,
        const unsigned int * __restrict__ token_ids,
        __nv_bfloat16 * __restrict__ output,
        unsigned int token_count) {
    if (token_count == 0 || token_count > kMaxTokens) {
        return;
    }
    const unsigned int token_slot = blockIdx.x;
    if (token_slot >= token_count) {
        return;
    }
    // Host admission validates every ID before this kernel can be launched.
    const unsigned int token = token_ids[token_slot];
    if (token >= kVocab) {
        return;
    }
    for (unsigned int column = threadIdx.x; column < kHidden; column += blockDim.x) {
        const int within = (int) (column % QK_K);
        const int half = within / 128;
        const int segment = (within % 128) / 32;
        const int lane = within % 32;
        const unsigned long long block_index =
            (unsigned long long) token * kBlocksPerRow + column / QK_K;
        const block_q6_K * block = source + block_index;
        const int low_index = 64 * half + lane + 32 * (segment & 1);
        const int low = (block->ql[low_index] >> (4 * (segment / 2))) & 0x0f;
        const int high = (block->qh[32 * half + lane] >> (2 * segment)) & 0x03;
        const int quant = (low | (high << 4)) - 32;
        const int scale_index = 8 * half + lane / 16 + 2 * segment;
        float value = __half2float(block->d);
        value *= (float) block->scales[scale_index];
        value *= (float) quant;
        output[(unsigned long long) token_slot * kHidden + column] =
            __float2bfloat16_rn(value);
    }
}
