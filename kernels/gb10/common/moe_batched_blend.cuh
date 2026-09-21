// SPDX-License-Identifier: AGPL-3.0-only

#ifndef ATLAS_MOE_BATCHED_BLEND_CUH
#define ATLAS_MOE_BATCHED_BLEND_CUH

#include <cuda_bf16.h>

// Exact 256-thread shared-expert gate reduction used by both the standalone
// blend and the fixed DeepSeek-V4 EXL3 tail. Callers provide eight floats of
// shared scratch and must reject every non-256-thread launch before entering.
// Volatile keeps the cross-thread gate reload after its publication barrier;
// a restricted nonvolatile pointer allowed that load to move before the store.
__device__ __forceinline__ float atlas_moe_shared_gate_scalar_256(
    const __nv_bfloat16* __restrict__ normed,
    const __nv_bfloat16* __restrict__ gate_weight,
    unsigned int hidden_size,
    volatile float* warp_partials) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31;

    float local_dot = 0.0f;
    if (gate_weight != 0) {
        for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
            const float n = __bfloat162float(normed[i]);
            const float g = __bfloat162float(gate_weight[i]);
            local_dot += n * g;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_dot += __shfl_down_sync(0xFFFFFFFF, local_dot, offset);
    }
    if (lane == 0) warp_partials[warp_id] = local_dot;
    __syncthreads();

    if (tid == 0) {
        float gate_scalar = 1.0f;
        if (gate_weight != 0) {
            float total = 0.0f;
            #pragma unroll
            for (unsigned int w = 0; w < 8; ++w) {
                total += warp_partials[w];
            }
            gate_scalar = 1.0f / (1.0f + __expf(-total));
        }
        warp_partials[0] = gate_scalar;
    }
    __syncthreads();
    return warp_partials[0];
}

__device__ __forceinline__ float atlas_round_bf16_to_f32(float value) {
    return __bfloat162float(__float2bfloat16(value));
}

__device__ __forceinline__ __nv_bfloat16 atlas_moe_blend_from_routed_bf16(
    float routed_bf16,
    __nv_bfloat16 shared,
    float gate_scalar) {
    const float s = __bfloat162float(shared);
    return __float2bfloat16(routed_bf16 + gate_scalar * s);
}

#endif  // ATLAS_MOE_BATCHED_BLEND_CUH
