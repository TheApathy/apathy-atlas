// SPDX-License-Identifier: AGPL-3.0-only

// Opt-in actual Vision experiment: preserve hc_post FP32 arithmetic/storage,
// but round its final value through BF16 RNE, as official Block.hc_post does.
// Kept separate so hyper_connection.cu and the default path stay unchanged.
#include <cuda_bf16.h>
#include <stddef.h>
#define HC_BLOCK 256
#define HC_MAX_MULT 4

extern "C" __global__ void deepseek_vision_hc_post_bf16(
    const __nv_bfloat16* __restrict__ block_out, // [T, H]
    const float* __restrict__ residual,          // [T, hc, H] FP32 highway (mHC)
    const float* __restrict__ post,              // [T, hc]
    const float* __restrict__ comb,              // [T, hc, hc]
    float* __restrict__ out,                     // [T, hc, H] FP32 highway (mHC)
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;

    const __nv_bfloat16* x = block_out + (size_t)t * H;
    const float* res = residual + (size_t)t * hc * H;
    const float* p = post + (size_t)t * hc;
    const float* c = comb + (size_t)t * hc * hc;
    float* o = out + (size_t)t * hc * H;

    for (unsigned int d = blockIdx.y * HC_BLOCK + tid; d < H; d += HC_BLOCK * gridDim.y) {
        float xd = (float)x[d];
        float rv[HC_MAX_MULT];
        for (unsigned int i = 0; i < hc; ++i) rv[i] = res[i * H + d];
        for (unsigned int j = 0; j < hc; ++j) {
            float acc = p[j] * xd;
            for (unsigned int i = 0; i < hc; ++i) acc += c[i * hc + j] * rv[i];
            o[j * H + d] = __bfloat162float(__float2bfloat16_rn(acc));
        }
    }
}
