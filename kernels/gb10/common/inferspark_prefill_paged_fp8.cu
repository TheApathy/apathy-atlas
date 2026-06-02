// SPDX-License-Identifier: AGPL-3.0-only

// Paged Prefill Flash Attention — FP8 E4M3 KV cache variant.
//
// Reads FP8 K/V from paged cache, dequantizes to BF16 in shared memory,
// then runs Flash Attention with contiguous BF16 Q.
//
// Grid: (num_q_heads, ceil(q_len/BR), 1)  Block: (128 or 256, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ __nv_bfloat16 fp8_to_bf16(__nv_fp8_storage_t b, float scale) {
    float v = __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)) * scale;
    return __float2bfloat16(v);
}

// FP8 tile loader: manual load + dequant + store to smem.
// cache_stride is in FP8 elements (1 byte each).
#define LOAD_KV_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM / 8; \
        for (unsigned int _i = t; _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const __nv_fp8_storage_t* _base = (const __nv_fp8_storage_t*)(cache) \
                    + (unsigned long long)_pb * fp8_cache_stride \
                    + (unsigned long long)_bo * num_kv_heads * head_dim \
                    + (unsigned long long)(kvh) * head_dim + _col; \
                __nv_bfloat16 _v[8]; \
                for (int _j = 0; _j < 8; _j++) \
                    _v[_j] = fp8_to_bf16(_base[_j], dq_scale); \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
        /* No cp.async used; emit dummy commit so shared header's wait works */ \
    } while(0)

#define KERNEL_NAME inferspark_prefill_paged_fp8
#define K_CACHE_TYPE const void* __restrict__
#define V_CACHE_TYPE const void* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
    , const float k_scale \
    , const float v_scale \
    , const unsigned long long fp8_cache_stride
#define KERNEL_PREAMBLE /* per-K/V scales now selected inside LOAD_KV_TILE below */

// K and V need DIFFERENT dequant scales under FP8 KV cache. The shared
// `prefill_paged_compute.cuh` header calls LOAD_KV_TILE for K tiles with
// the K_cache pointer and for V tiles with V_cache. We override the macro
// so it inspects the pointer at the call site and selects k_scale for
// K-side loads, v_scale for V-side. This was the FIXME above (now fixed).

#undef LOAD_KV_TILE
#define LOAD_KV_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const float _sc = ((const void*)(cache) == (const void*)K_cache) ? k_scale : v_scale; \
        const unsigned int _cpr = HDIM / 8; \
        for (unsigned int _i = t; _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const __nv_fp8_storage_t* _base = (const __nv_fp8_storage_t*)(cache) \
                    + (unsigned long long)_pb * fp8_cache_stride \
                    + (unsigned long long)_bo * num_kv_heads * head_dim \
                    + (unsigned long long)(kvh) * head_dim + _col; \
                __nv_bfloat16 _v[8]; \
                for (int _j = 0; _j < 8; _j++) \
                    _v[_j] = fp8_to_bf16(_base[_j], _sc); \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#undef KERNEL_PREAMBLE
#define KERNEL_PREAMBLE /* nothing */

#include "prefill_paged_compute.cuh"
