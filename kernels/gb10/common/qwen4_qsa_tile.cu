// SPDX-License-Identifier: AGPL-3.0-only
//
// Tile-batched QSA prefill pooling: pools each complete 4-token group straight
// from the projected raw keys (same values the ring copy would hold), in the
// same row order and with the same FP32 sum * 0.25 -> BF16 rounding as
// qwen4_qsa_pool_prefill. first_positions[g] = positions[4g+3] - 3.
// Grid: (num_groups), Block: (INDEX_DIM).
#include <cuda_bf16.h>
#ifndef QSA_TILE_INDEX_DIM
#define QSA_TILE_INDEX_DIM 128
#endif
#define QSA_TILE_RATIO 4

extern "C" __global__ void qwen4_qsa_pool_prefill_from_raw(
    const __nv_bfloat16* __restrict__ raw_keys,
    __nv_bfloat16* __restrict__ pooled_keys,
    unsigned int* __restrict__ first_positions,
    const unsigned int* __restrict__ positions,
    unsigned int num_groups) {
    const unsigned int group = blockIdx.x;
    const unsigned int d = threadIdx.x;
    if (group >= num_groups || d >= QSA_TILE_INDEX_DIM) return;
    float sum = 0.0f;
#pragma unroll
    for (int row = 0; row < QSA_TILE_RATIO; ++row) {
        sum += __bfloat162float(
            raw_keys[((unsigned long long)group * QSA_TILE_RATIO + row) * QSA_TILE_INDEX_DIM + d]);
    }
    pooled_keys[(unsigned long long)group * QSA_TILE_INDEX_DIM + d] = __float2bfloat16_rn(sum * 0.25f);
    if (d == 0) first_positions[group] = positions[group * QSA_TILE_RATIO + QSA_TILE_RATIO - 1] - (QSA_TILE_RATIO - 1);
}
