// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>

// Rectangular RoPE: compact noise Q uses the suffix positions while target
// tail K and noise K use their own absolute positions. Grid is
// [Q heads + target-K heads + noise-K heads, ceil(max(C,T)/2), batch].
extern "C" __global__ void atlas_glm53_dflash2_rope_rect(
    __nv_bfloat16* __restrict__ q_noise,
    __nv_bfloat16* __restrict__ target_k,
    __nv_bfloat16* __restrict__ noise_k,
    const unsigned int noise_tokens,
    const unsigned int target_tokens,
    const unsigned int target_abs_start,
    const unsigned int noise_abs_start,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const float theta
) {
    const unsigned int branch_head = blockIdx.x;
    const unsigned int batch = blockIdx.z;
    const unsigned int pair = threadIdx.x & 63u;
    const unsigned int local_pos = blockIdx.y * 2u + (threadIdx.x >> 6);
    const bool is_q = branch_head < num_q_heads;
    const bool is_target = !is_q && branch_head < num_q_heads + num_kv_heads;
    const unsigned int head = is_q
        ? branch_head
        : is_target ? branch_head - num_q_heads : branch_head - num_q_heads - num_kv_heads;
    const unsigned int tokens = is_target ? target_tokens : noise_tokens;
    if (local_pos >= tokens || head >= (is_q ? num_q_heads : num_kv_heads)) return;

    const unsigned int absolute_pos = is_target
        ? target_abs_start + local_pos
        : noise_abs_start + local_pos;
    const double exponent = (double)(2u * pair) / (double)head_dim;
    const float frequency = (float)(1.0 / pow((double)theta, exponent));
    const float angle = (float)absolute_pos * frequency;
    float sine, cosine;
    sincosf(angle, &sine, &cosine);

    __nv_bfloat16* row;
    if (is_q) {
        row = q_noise
            + (((unsigned long long)batch * noise_tokens + local_pos) * num_q_heads + head)
                * head_dim;
    } else if (is_target) {
        row = target_k
            + (((unsigned long long)batch * target_tokens + local_pos) * num_kv_heads + head)
                * head_dim;
    } else {
        row = noise_k
            + (((unsigned long long)batch * noise_tokens + local_pos) * num_kv_heads + head)
                * head_dim;
    }
    const float first = __bfloat162float(row[pair]);
    const float second = __bfloat162float(row[pair + head_dim / 2u]);
    row[pair] = __float2bfloat16_rn(first * cosine - second * sine);
    row[pair + head_dim / 2u] = __float2bfloat16_rn(second * cosine + first * sine);
}

// The GLM target uses HDIM=256, so its common paged module cannot serve the
// DFlash2 drafter. Instantiate the proven rectangular paged implementation at
// the checkpoint's exact H128 geometry under a distinct module/symbol.
#define HDIM 128

#define LOAD_KV_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM / 8; \
        const unsigned long long _ps = \
            (unsigned long long)cache_block_size * num_kv_heads * head_dim; \
        const unsigned long long _rs = (unsigned long long)num_kv_heads * head_dim; \
        for (unsigned int _i = t; _i < TILE_CHUNKS; _i += stride) { \
            const unsigned int _row = _i / _cpr; \
            const unsigned int _col = (_i % _cpr) * 8; \
            const unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                const unsigned int _logical = _pos / cache_block_size; \
                const unsigned int _offset = _pos % cache_block_size; \
                const unsigned int _physical = (unsigned int)(bt)[_logical]; \
                const void* _source = (const void*)( \
                    (cache) + _physical * _ps + _offset * _rs + (kvh) * head_dim + _col); \
                atlas_cp16(&(smem)[_row][_col], _source); \
            } else { \
                *((uint4*)&(smem)[_row][_col]) = make_uint4(0, 0, 0, 0); \
            } \
        } \
    } while (0)

#define KERNEL_NAME atlas_glm53_dflash2_prefill_paged_h128
#define K_CACHE_TYPE const __nv_bfloat16* __restrict__
#define V_CACHE_TYPE const __nv_bfloat16* __restrict__
#define KERNEL_EXTRA_PARAMS , const float inv_sqrt_d
#define KERNEL_PREAMBLE

// This tree's common/prefill_paged_compute*.cuh predate upstream's portable
// cp.async helpers; GLM's tile-load macros call atlas_cp16, so define it here.
#ifndef GLM53_ATLAS_CP16_DEFINED
#define GLM53_ATLAS_CP16_DEFINED
__device__ __forceinline__ void atlas_cp16(void* smem_dst, const void* gmem_src) {
    unsigned _s = __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(_s), "l"(gmem_src));
}
#endif
#include "../../common/prefill_paged_compute.cuh"
