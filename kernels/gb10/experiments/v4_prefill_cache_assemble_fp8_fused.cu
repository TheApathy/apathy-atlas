// SPDX-License-Identifier: AGPL-3.0-only

// Isolated, unregistered DeepSeek-V4 prefill experiment.
//
// Fuses mla_cache_assemble_batched + reshape_and_cache_flash_fp8 for the exact
// N=2410, NKv=1, [512 latent | 64 RoPE] workload. Both paged pools receive the
// same BF16 source row (V4 has V==K), but retain their independent per-tensor
// FP8 scales. Pair grouping and conversion match reshape_and_cache_flash_fp8.
//
// This file has no registry or serving-path reachability.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>

#define V4_CACHE_TOKENS 2410u
#define V4_CACHE_KV_LORA 512u
#define V4_CACHE_ROPE 64u
#define V4_CACHE_DIM 576u
#define V4_CACHE_BLOCK_SIZE 16u
#define V4_CACHE_THREADS 256u

__device__ __forceinline__ __nv_fp8x2_storage_t
v4_cache_bf16x2_to_fp8x2(unsigned int packed_bf16, float inv_scale) {
    const float v0 = __bfloat162float(
        __ushort_as_bfloat16(static_cast<unsigned short>(packed_bf16 & 0xffffu)));
    const float v1 = __bfloat162float(
        __ushort_as_bfloat16(static_cast<unsigned short>(packed_bf16 >> 16)));
    const float2 scaled = make_float2(v0 * inv_scale, v1 * inv_scale);
    return __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
}

// ABI:
//   kv_latent   [2410, 512] BF16, 4-byte aligned
//   k_rope      [2410,  64] BF16, 4-byte aligned and already rotated
//   k/v_cache   [num_blocks, 16, 1, 576] FP8, 2-byte aligned
//   slot_mapping[2410] i64; negative and out-of-pool slots are skipped
// Grid: (2410, 1, 1). Block: (256, 1, 1).
extern "C" __global__ __launch_bounds__(V4_CACHE_THREADS)
void v4_prefill_cache_assemble_fp8_fused(
    const __nv_bfloat16* __restrict__ kv_latent,
    const __nv_bfloat16* __restrict__ k_rope,
    __nv_fp8_storage_t* __restrict__ k_cache,
    __nv_fp8_storage_t* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_tokens,
    const unsigned int num_blocks,
    const unsigned int block_size,
    const float k_scale,
    const float v_scale,
    const unsigned long long cache_stride) {
    if (kv_latent == nullptr || k_rope == nullptr || k_cache == nullptr ||
        v_cache == nullptr || slot_mapping == nullptr || k_cache == v_cache) {
        return;
    }
    if (num_tokens != V4_CACHE_TOKENS || num_blocks == 0 ||
        block_size != V4_CACHE_BLOCK_SIZE ||
        cache_stride != V4_CACHE_BLOCK_SIZE * V4_CACHE_DIM) {
        return;
    }
    if (!isfinite(k_scale) || !isfinite(v_scale) || k_scale <= 0.0f || v_scale <= 0.0f) {
        return;
    }
    if (blockDim.x != V4_CACHE_THREADS || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.x != V4_CACHE_TOKENS || gridDim.y != 1 || gridDim.z != 1) {
        return;
    }
    if ((reinterpret_cast<unsigned long long>(kv_latent) & 3ull) != 0 ||
        (reinterpret_cast<unsigned long long>(k_rope) & 3ull) != 0 ||
        (reinterpret_cast<unsigned long long>(k_cache) & 1ull) != 0 ||
        (reinterpret_cast<unsigned long long>(v_cache) & 1ull) != 0 ||
        (reinterpret_cast<unsigned long long>(slot_mapping) & 7ull) != 0) {
        return;
    }

    const unsigned int token = blockIdx.x;
    const long long slot = slot_mapping[token];
    const unsigned long long max_slots =
        static_cast<unsigned long long>(num_blocks) * V4_CACHE_BLOCK_SIZE;
    if (slot < 0 || static_cast<unsigned long long>(slot) >= max_slots) {
        return;
    }

    const unsigned int block_idx =
        static_cast<unsigned int>(slot / V4_CACHE_BLOCK_SIZE);
    const unsigned int block_offset =
        static_cast<unsigned int>(slot % V4_CACHE_BLOCK_SIZE);
    const unsigned long long cache_offset =
        static_cast<unsigned long long>(block_idx) * cache_stride +
        static_cast<unsigned long long>(block_offset) * V4_CACHE_DIM;

    __nv_fp8_storage_t* const key_dst = k_cache + cache_offset;
    __nv_fp8_storage_t* const val_dst = v_cache + cache_offset;
    __nv_fp8x2_storage_t* const key_dst16 =
        reinterpret_cast<__nv_fp8x2_storage_t*>(key_dst);
    __nv_fp8x2_storage_t* const val_dst16 =
        reinterpret_cast<__nv_fp8x2_storage_t*>(val_dst);

    const unsigned int* const latent32 = reinterpret_cast<const unsigned int*>(
        kv_latent + static_cast<unsigned long long>(token) * V4_CACHE_KV_LORA);
    const unsigned int* const rope32 = reinterpret_cast<const unsigned int*>(
        k_rope + static_cast<unsigned long long>(token) * V4_CACHE_ROPE);
    const float inv_k_scale = 1.0f / k_scale;
    const float inv_v_scale = 1.0f / v_scale;
    constexpr unsigned int latent_pairs = V4_CACHE_KV_LORA / 2;
    constexpr unsigned int total_pairs = V4_CACHE_DIM / 2;

    for (unsigned int pair = threadIdx.x; pair < total_pairs; pair += V4_CACHE_THREADS) {
        const unsigned int packed =
            pair < latent_pairs ? latent32[pair] : rope32[pair - latent_pairs];
        key_dst16[pair] = v4_cache_bf16x2_to_fp8x2(packed, inv_k_scale);
        val_dst16[pair] = v4_cache_bf16x2_to_fp8x2(packed, inv_v_scale);
    }
}
