// SPDX-License-Identifier: AGPL-3.0-only

// Atlas MoE element-wise SiLU activation + multiply.
//
// output[i] = silu(gate[i]) * up[i]
// where silu(x) = x * sigmoid(x)
//
// Grid: (ceil(total_elements / 256), 1, 1)  Block: (256, 1, 1)
//
// Used after grouped gate+up GEMMs to fuse activation before down GEMM.

#include <cuda_bf16.h>

extern "C" __global__ void moe_silu_mul(
    const __nv_bfloat16* __restrict__ gate,   // [total_expanded, inter_size]
    const __nv_bfloat16* __restrict__ up,     // [total_expanded, inter_size]
    __nv_bfloat16* __restrict__ output,        // [total_expanded, inter_size]
    unsigned int total_elements
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    // DeepSeek-V4 routed-expert swiglu clamp: gate<=limit, up in [-limit,limit]
    // (swiglu_limit = 10.0, config). This is the ROUTED grouped-silu path; the
    // shared expert uses its own ungated kernel.
    const float SWIGLU_LIMIT = 10.0f;
    g = fminf(g, SWIGLU_LIMIT);
    u = fminf(fmaxf(u, -SWIGLU_LIMIT), SWIGLU_LIMIT);
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    float result = g * sigmoid_g * u;
    output[idx] = __float2bfloat16(result);
}

// Unclamped SwiGLU: plain silu(gate)*up with NO swiglu_limit. The clamped
// variant above is DeepSeek-V4-specific (swiglu_limit=10.0 from its config);
// models without a swiglu_limit (Laguna DFlash drafter's dense Qwen3 MLP)
// MUST use this one — the ±10 clamp silently butchers rows with large
// activations (anchor rows hit ~14% clipped elements → drafter acceptance
// collapse; see docs/12-drafter-parity-hunt.md).
extern "C" __global__ void silu_mul_noclamp(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    unsigned int total_elements
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    output[idx] = __float2bfloat16(g * sigmoid_g * u);
}
