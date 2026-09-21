// SPDX-License-Identifier: AGPL-3.0-only
//
// Correctness-first Qwen4 QSA kernels.  The cache is keyed by Atlas's main
// paged-KV physical blocks: every 16-token KV page owns four compressed index
// keys and a four-row raw-key ring.  This keeps prefix sharing and block reuse
// aligned without a second host allocator.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "qwen4_qsa_positions.cuh"

namespace {

constexpr int INDEX_HEADS = 4;
constexpr int INDEX_DIM = 128;
constexpr int COMPRESS_RATIO = 4;
constexpr int TOKEN_TOPK = 2048;
constexpr int BLOCK_TOPK = TOKEN_TOPK / COMPRESS_RATIO;
constexpr int OUTPUT_WIDTH = TOKEN_TOPK + COMPRESS_RATIO - 1;
constexpr int NVFP4_GROUP_SIZE = 16;

__device__ __forceinline__ float fp8e4m3_to_f32(__nv_fp8_storage_t value) {
    return __half2float(__nv_cvt_fp8_to_halfraw(value, __NV_E4M3));
}

__device__ __forceinline__ float load_nvfp4(
    const unsigned char* __restrict__ block,
    unsigned long long element,
    unsigned long long data_section_bytes) {
    const unsigned char packed = block[element >> 1];
    const unsigned int nibble = (element & 1ull) ? (packed >> 4) : (packed & 0xfu);
    const float lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
       -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
    };
    const auto scale = (__nv_fp8_storage_t)block[
        data_section_bytes + element / NVFP4_GROUP_SIZE];
    return lut[nibble] * fp8e4m3_to_f32(scale);
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ bool less_pair(float av, int ai, float bv, int bi) {
    return av < bv || (av == bv && ai > bi);
}

__device__ void heap_sift_down(float* values, int* indices, int root, int size) {
    while (true) {
        int child = root * 2 + 1;
        if (child >= size) return;
        if (child + 1 < size &&
            less_pair(values[child + 1], indices[child + 1], values[child], indices[child])) {
            ++child;
        }
        if (!less_pair(values[child], indices[child], values[root], indices[root])) return;
        float tv = values[root]; values[root] = values[child]; values[child] = tv;
        int ti = indices[root]; indices[root] = indices[child]; indices[child] = ti;
        root = child;
    }
}

}  // namespace

// projected_qk is [4*128 query | 1*128 raw key].  Store the raw key in the
// physical page's four-row ring and, at each group boundary, average the four
// rows into pooled_key.  first_position is consumed by the ordinary RoPE
// kernel; its current reconstruction is valid only for linear positions.
extern "C" __global__ void qwen4_qsa_stage_pool(
    const __nv_bfloat16* __restrict__ projected_qk,
    __nv_bfloat16* __restrict__ raw_ring,
    __nv_bfloat16* __restrict__ pooled_key,
    unsigned int* __restrict__ first_position,
    const long long* __restrict__ slot_ptr,
    const unsigned int* __restrict__ position_ptr,
    unsigned int block_size) {
    const int d = threadIdx.x;
    if (d >= INDEX_DIM) return;
    const long long slot = *slot_ptr;
    const unsigned int position = *position_ptr;
    qwen4_qsa_positions::PhysicalSlot physical;
    if (!qwen4_qsa_positions::physical_slot(slot, block_size, physical)) return;
    const unsigned long long physical_block = physical.block;
    const unsigned int ring_row = physical.ring_row;
    const unsigned long long ring_base =
        (physical_block * COMPRESS_RATIO + ring_row) * INDEX_DIM;
    raw_ring[ring_base + d] = projected_qk[INDEX_HEADS * INDEX_DIM + d];
    __syncthreads();
    if (ring_row == COMPRESS_RATIO - 1) {
        float sum = 0.0f;
#pragma unroll
        for (int r = 0; r < COMPRESS_RATIO; ++r) {
            sum += __bfloat162float(raw_ring[(physical_block * COMPRESS_RATIO + r) * INDEX_DIM + d]);
        }
        pooled_key[d] = __float2bfloat16_rn(sum * 0.25f);
    }
    // This legacy 1D reconstruction does not implement first-member T/H/W.
    // Physical ownership alone does not qualify image QSA rotary semantics.
    if (d == 0) *first_position = position - ring_row;
}

extern "C" __global__ void qwen4_qsa_store_compressed(
    const __nv_bfloat16* __restrict__ pooled_key,
    __nv_bfloat16* __restrict__ compressed_cache,
    const long long* __restrict__ slot_ptr,
    const unsigned int* __restrict__ position_ptr,
    unsigned int block_size) {
    const int d = threadIdx.x;
    if (d >= INDEX_DIM) return;
    const long long slot = *slot_ptr;
    (void)position_ptr;  // Retained launch ABI; rotary position does not gate stores.
    qwen4_qsa_positions::PhysicalSlot physical;
    if (!qwen4_qsa_positions::physical_slot(slot, block_size, physical) ||
        physical.ring_row != COMPRESS_RATIO - 1) return;
    const unsigned long long physical_block = physical.block;
    const unsigned int group_in_block = physical.group;
    const unsigned long long dst =
        (physical_block * (block_size / COMPRESS_RATIO) + group_in_block) * INDEX_DIM;
    compressed_cache[dst + d] = pooled_key[d];
}

// Batched prefill index construction. Prefill does not consume query scores;
// only the fifth (key) projection head is needed to populate QSA history.
// Split staging/pooling into separate launches so every raw row is globally
// visible before a four-token group is reduced.
extern "C" __global__ void qwen4_qsa_stage_prefill_raw(
    const __nv_bfloat16* __restrict__ raw_keys,
    __nv_bfloat16* __restrict__ raw_ring,
    const long long* __restrict__ slots,
    const unsigned int* __restrict__ positions,
    unsigned int num_tokens,
    unsigned int block_size) {
    const unsigned int token = blockIdx.x;
    const unsigned int d = threadIdx.x;
    if (token >= num_tokens || d >= INDEX_DIM || block_size == 0) return;
    const long long slot = slots[token];
    (void)positions;  // Retained launch ABI; ring ownership uses physical slots.
    qwen4_qsa_positions::PhysicalSlot physical;
    if (!qwen4_qsa_positions::physical_slot(slot, block_size, physical)) return;
    const unsigned long long physical_block = physical.block;
    const unsigned int ring_row = physical.ring_row;
    raw_ring[(physical_block * COMPRESS_RATIO + ring_row) * INDEX_DIM + d] =
        raw_keys[(unsigned long long)token * INDEX_DIM + d];
}

extern "C" __global__ void qwen4_qsa_pool_prefill(
    const __nv_bfloat16* __restrict__ raw_ring,
    __nv_bfloat16* __restrict__ pooled_keys,
    unsigned int* __restrict__ first_positions,
    const long long* __restrict__ slots,
    const unsigned int* __restrict__ positions,
    unsigned int first_endpoint,
    unsigned int num_groups,
    unsigned int block_size) {
    const unsigned int group = blockIdx.x;
    const unsigned int d = threadIdx.x;
    if (group >= num_groups || d >= INDEX_DIM || block_size == 0) return;
    const unsigned int token = first_endpoint + group * COMPRESS_RATIO;
    const long long slot = slots[token];
    if (slot < 0) return;
    const unsigned long long physical_block = (unsigned long long)slot / block_size;
    float sum = 0.0f;
#pragma unroll
    for (int row = 0; row < COMPRESS_RATIO; ++row) {
        sum += __bfloat162float(
            raw_ring[(physical_block * COMPRESS_RATIO + row) * INDEX_DIM + d]);
    }
    pooled_keys[(unsigned long long)group * INDEX_DIM + d] =
        __float2bfloat16_rn(sum * 0.25f);
    if (d == 0) first_positions[group] = positions[token] - (COMPRESS_RATIO - 1);
}

extern "C" __global__ void qwen4_qsa_store_prefill_compressed(
    const __nv_bfloat16* __restrict__ pooled_keys,
    __nv_bfloat16* __restrict__ compressed_cache,
    const long long* __restrict__ slots,
    unsigned int first_endpoint,
    unsigned int num_groups,
    unsigned int block_size) {
    const unsigned int group = blockIdx.x;
    const unsigned int d = threadIdx.x;
    if (group >= num_groups || d >= INDEX_DIM || block_size < COMPRESS_RATIO) return;
    const unsigned int token = first_endpoint + group * COMPRESS_RATIO;
    const long long slot = slots[token];
    if (slot < 0) return;
    const unsigned long long physical_block = (unsigned long long)slot / block_size;
    const unsigned int group_in_block = ((unsigned int)slot % block_size) / COMPRESS_RATIO;
    const unsigned long long dst =
        (physical_block * (block_size / COMPRESS_RATIO) + group_in_block) * INDEX_DIM;
    compressed_cache[dst + d] = pooled_keys[(unsigned long long)group * INDEX_DIM + d];
}

// Eight warps score eight compressed micro-blocks per CTA.  The trained score
// is sum_h relu(q_h dot k) / sqrt(128), exactly matching Qwen's reference.
extern "C" __global__ void qwen4_qsa_score(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ compressed_cache,
    const int* __restrict__ block_table,
    float* __restrict__ logits,
    unsigned int visible_groups,
    unsigned int block_size) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const unsigned int group = blockIdx.x * 8u + (unsigned int)warp;
    if (group >= visible_groups) return;
    const unsigned int groups_per_page = block_size / COMPRESS_RATIO;
    const unsigned int logical_page = group / groups_per_page;
    const unsigned int group_in_page = group % groups_per_page;
    const int physical_page = block_table[logical_page];
    if (physical_page < 0) {
        if (lane == 0) logits[group] = -CUDART_INF_F;
        return;
    }
    const unsigned long long key_base =
        ((unsigned long long)physical_page * groups_per_page + group_in_page) * INDEX_DIM;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < INDEX_HEADS; ++h) {
        float partial = 0.0f;
#pragma unroll
        for (int d = lane; d < INDEX_DIM; d += 32) {
            partial += __bfloat162float(query[h * INDEX_DIM + d]) *
                       __bfloat162float(compressed_cache[key_base + d]);
        }
        partial = warp_sum(partial);
        if (lane == 0) score += fmaxf(partial, 0.0f);
    }
    if (lane == 0) logits[group] = score * 0.08838834764831845f;  // 1/sqrt(128)
}

// Exact, deterministic correctness path.  A single thread maintains a
// 512-entry min-heap, then sorts the selected logical groups by position so
// sparse attention has a stable reduction order.  This is intentionally easy
// to audit; a parallel radix selector can replace it without changing the ABI.
extern "C" __global__ void qwen4_qsa_select_expand(
    const float* __restrict__ logits,
    int* __restrict__ token_indices,
    unsigned int visible_groups,
    unsigned int position,
    unsigned int sequence_length) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    __shared__ float heap_values[BLOCK_TOPK];
    __shared__ int heap_indices[BLOCK_TOPK];
    for (int i = 0; i < OUTPUT_WIDTH; ++i) token_indices[i] = -1;

    const int count = visible_groups < BLOCK_TOPK ? (int)visible_groups : BLOCK_TOPK;
    if (visible_groups <= BLOCK_TOPK) {
        for (int i = 0; i < count; ++i) heap_indices[i] = i;
    } else {
        for (int i = 0; i < BLOCK_TOPK; ++i) {
            heap_values[i] = logits[i];
            heap_indices[i] = i;
        }
        for (int i = BLOCK_TOPK / 2 - 1; i >= 0; --i) {
            heap_sift_down(heap_values, heap_indices, i, BLOCK_TOPK);
        }
        for (unsigned int i = BLOCK_TOPK; i < visible_groups; ++i) {
            const float value = logits[i];
            if (!less_pair(value, (int)i, heap_values[0], heap_indices[0])) {
                heap_values[0] = value;
                heap_indices[0] = (int)i;
                heap_sift_down(heap_values, heap_indices, 0, BLOCK_TOPK);
            }
        }
        // Stable positional order for the subsequent softmax reduction.
        for (int i = 1; i < BLOCK_TOPK; ++i) {
            int key = heap_indices[i];
            int j = i - 1;
            while (j >= 0 && heap_indices[j] > key) {
                heap_indices[j + 1] = heap_indices[j];
                --j;
            }
            heap_indices[j + 1] = key;
        }
    }

    int out = 0;
    for (int rank = 0; rank < count; ++rank) {
        const int group = heap_indices[rank];
#pragma unroll
        for (int offset = 0; offset < COMPRESS_RATIO; ++offset) {
            const int token = group * COMPRESS_RATIO + offset;
            if ((unsigned int)token < sequence_length) token_indices[out++] = token;
        }
    }
    const unsigned int tail_start = ((position + 1) / COMPRESS_RATIO) * COMPRESS_RATIO;
    for (unsigned int token = tail_start;
         token <= position && token < sequence_length && out < OUTPUT_WIDTH;
         ++token) {
        token_indices[out++] = (int)token;
    }
}

// Graph-capture variant of qwen4_qsa_score.  The launch topology is selected
// by the host before capture, while the advancing sequence length is read from
// Atlas's persistent device metadata on every replay.
extern "C" __global__ void qwen4_qsa_score_device_meta(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ compressed_cache,
    const int* __restrict__ block_table,
    float* __restrict__ logits,
    const unsigned int* __restrict__ sequence_length_ptr,
    unsigned int block_size) {
    if (block_size < COMPRESS_RATIO) return;
    const unsigned int visible_groups = *sequence_length_ptr / COMPRESS_RATIO;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const unsigned int group = blockIdx.x * 8u + (unsigned int)warp;
    if (group >= visible_groups) return;
    const unsigned int groups_per_page = block_size / COMPRESS_RATIO;
    const unsigned int logical_page = group / groups_per_page;
    const unsigned int group_in_page = group % groups_per_page;
    const int physical_page = block_table[logical_page];
    if (physical_page < 0) {
        if (lane == 0) logits[group] = -CUDART_INF_F;
        return;
    }
    const unsigned long long key_base =
        ((unsigned long long)physical_page * groups_per_page + group_in_page) * INDEX_DIM;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < INDEX_HEADS; ++h) {
        float partial = 0.0f;
#pragma unroll
        for (int d = lane; d < INDEX_DIM; d += 32) {
            partial += __bfloat162float(query[h * INDEX_DIM + d]) *
                       __bfloat162float(compressed_cache[key_base + d]);
        }
        partial = warp_sum(partial);
        if (lane == 0) score += fmaxf(partial, 0.0f);
    }
    if (lane == 0) logits[group] = score * 0.08838834764831845f;
}

// Graph-capture variant of qwen4_qsa_select_expand.  Keep the incumbent
// deterministic heap and positional reduction order byte-for-byte equivalent;
// only the three frame-varying scalar arguments move behind stable pointers.
extern "C" __global__ void qwen4_qsa_select_expand_device_meta(
    const float* __restrict__ logits,
    int* __restrict__ token_indices,
    const unsigned int* __restrict__ position_ptr,
    const unsigned int* __restrict__ sequence_length_ptr) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    (void)position_ptr;  // Retained launch ABI; this is not a rotary query index.
    const unsigned int sequence_length = *sequence_length_ptr;
    const unsigned int visible_groups = sequence_length / COMPRESS_RATIO;
    __shared__ float heap_values[BLOCK_TOPK];
    __shared__ int heap_indices[BLOCK_TOPK];
    for (int i = 0; i < OUTPUT_WIDTH; ++i) token_indices[i] = -1;
    unsigned int position;
    if (!qwen4_qsa_positions::physical_query(sequence_length, position)) return;

    const int count = visible_groups < BLOCK_TOPK ? (int)visible_groups : BLOCK_TOPK;
    if (visible_groups <= BLOCK_TOPK) {
        for (int i = 0; i < count; ++i) heap_indices[i] = i;
    } else {
        for (int i = 0; i < BLOCK_TOPK; ++i) {
            heap_values[i] = logits[i];
            heap_indices[i] = i;
        }
        for (int i = BLOCK_TOPK / 2 - 1; i >= 0; --i) {
            heap_sift_down(heap_values, heap_indices, i, BLOCK_TOPK);
        }
        for (unsigned int i = BLOCK_TOPK; i < visible_groups; ++i) {
            const float value = logits[i];
            if (!less_pair(value, (int)i, heap_values[0], heap_indices[0])) {
                heap_values[0] = value;
                heap_indices[0] = (int)i;
                heap_sift_down(heap_values, heap_indices, 0, BLOCK_TOPK);
            }
        }
        for (int i = 1; i < BLOCK_TOPK; ++i) {
            int key = heap_indices[i];
            int j = i - 1;
            while (j >= 0 && heap_indices[j] > key) {
                heap_indices[j + 1] = heap_indices[j];
                --j;
            }
            heap_indices[j + 1] = key;
        }
    }

    int out = 0;
    for (int rank = 0; rank < count; ++rank) {
        const int group = heap_indices[rank];
#pragma unroll
        for (int offset = 0; offset < COMPRESS_RATIO; ++offset) {
            const int token = group * COMPRESS_RATIO + offset;
            if ((unsigned int)token < sequence_length) token_indices[out++] = token;
        }
    }
    const unsigned int tail_start = ((position + 1) / COMPRESS_RATIO) * COMPRESS_RATIO;
    for (unsigned int token = tail_start;
         token <= position && token < sequence_length && out < OUTPUT_WIDTH;
         ++token) {
        token_indices[out++] = (int)token;
    }
}

// Split-token sparse attention. Eight CTAs cooperate per query head so the
// 24-head Qwen4 geometry exposes 192 CTAs instead of underfilling GB10 with
// only 24. Each CTA writes a softmax partial; the reduction kernel combines
// them with the exact log-sum-exp identity.
extern "C" __global__ void qwen4_qsa_sparse_attention_bf16_partial(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_elems,
    unsigned long long v_block_stride_elems,
    unsigned int block_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int split = blockIdx.y;
    const unsigned int d = threadIdx.x;
    if (q_head >= num_query_heads || d >= head_dim) return;
    const unsigned int group_size = num_query_heads / num_kv_heads;
    const unsigned int kv_head = q_head / group_size;
    const float qv = __bfloat162float(query[(unsigned long long)q_head * head_dim + d]);
    float acc = 0.0f;
    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    __shared__ float reduce[256];
    __shared__ float alpha_shared;
    __shared__ float beta_shared;

    for (int rank = (int)split; rank < OUTPUT_WIDTH; rank += (int)num_splits) {
        const int logical_token = token_indices[rank];
        if (logical_token < 0) break;
        const unsigned int logical_page = (unsigned int)logical_token / block_size;
        const unsigned int token_in_page = (unsigned int)logical_token % block_size;
        const int physical_page = block_table[logical_page];
        if (physical_page < 0) continue;
        const unsigned long long k_index =
            (unsigned long long)physical_page * k_block_stride_elems +
            ((unsigned long long)token_in_page * num_kv_heads + kv_head) * head_dim + d;
        reduce[d] = qv * __bfloat162float(k_cache[k_index]);
        __syncthreads();
        for (unsigned int stride = head_dim >> 1; stride > 0; stride >>= 1) {
            if (d < stride) reduce[d] += reduce[d + stride];
            __syncthreads();
        }
        if (d == 0) {
            const float score = reduce[0] * softmax_scale;
            const float next_max = fmaxf(row_max, score);
            const float alpha = row_max == -CUDART_INF_F ? 0.0f : __expf(row_max - next_max);
            const float beta = __expf(score - next_max);
            row_sum = row_sum * alpha + beta;
            row_max = next_max;
            alpha_shared = alpha;
            beta_shared = beta;
        }
        __syncthreads();
        const unsigned long long v_index =
            (unsigned long long)physical_page * v_block_stride_elems +
            ((unsigned long long)token_in_page * num_kv_heads + kv_head) * head_dim + d;
        acc = acc * alpha_shared + beta_shared * __bfloat162float(v_cache[v_index]);
        __syncthreads();
    }
    const unsigned long long part = (unsigned long long)q_head * num_splits + split;
    partial_output[part * head_dim + d] = acc;
    if (d == 0) {
        partial_max[part] = row_max;
        partial_sum[part] = row_sum;
    }
}

// Same sparse-token contract as the BF16 path, but consumes Atlas's packed
// NVFP4 paged-KV layout in place. Keeping the 1M-token history compressed is
// load-bearing on a single 128 GiB GB10.
extern "C" __global__ void qwen4_qsa_sparse_attention_nvfp4_partial(
    const __nv_bfloat16* __restrict__ query,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    const int* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    unsigned long long k_block_stride_bytes,
    unsigned long long v_block_stride_bytes,
    unsigned long long data_section_bytes,
    unsigned int block_size,
    unsigned int num_query_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float softmax_scale,
    unsigned int num_splits) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int split = blockIdx.y;
    const unsigned int d = threadIdx.x;
    if (q_head >= num_query_heads || d >= head_dim) return;
    const unsigned int group_size = num_query_heads / num_kv_heads;
    const unsigned int kv_head = q_head / group_size;
    const float qv = __bfloat162float(query[(unsigned long long)q_head * head_dim + d]);
    float acc = 0.0f;
    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    __shared__ float reduce[256];
    __shared__ float alpha_shared;
    __shared__ float beta_shared;

    for (int rank = (int)split; rank < OUTPUT_WIDTH; rank += (int)num_splits) {
        const int logical_token = token_indices[rank];
        if (logical_token < 0) break;
        const unsigned int logical_page = (unsigned int)logical_token / block_size;
        const unsigned int token_in_page = (unsigned int)logical_token % block_size;
        const int physical_page = block_table[logical_page];
        if (physical_page < 0) continue;
        const unsigned long long element =
            ((unsigned long long)token_in_page * num_kv_heads + kv_head) * head_dim + d;
        const unsigned char* k_block =
            k_cache + (unsigned long long)physical_page * k_block_stride_bytes;
        reduce[d] = qv * load_nvfp4(k_block, element, data_section_bytes);
        __syncthreads();
        for (unsigned int stride = head_dim >> 1; stride > 0; stride >>= 1) {
            if (d < stride) reduce[d] += reduce[d + stride];
            __syncthreads();
        }
        if (d == 0) {
            const float score = reduce[0] * softmax_scale;
            const float next_max = fmaxf(row_max, score);
            const float alpha = row_max == -CUDART_INF_F ? 0.0f : __expf(row_max - next_max);
            const float beta = __expf(score - next_max);
            row_sum = row_sum * alpha + beta;
            row_max = next_max;
            alpha_shared = alpha;
            beta_shared = beta;
        }
        __syncthreads();
        const unsigned char* v_block =
            v_cache + (unsigned long long)physical_page * v_block_stride_bytes;
        acc = acc * alpha_shared + beta_shared * load_nvfp4(v_block, element, data_section_bytes);
        __syncthreads();
    }
    const unsigned long long part = (unsigned long long)q_head * num_splits + split;
    partial_output[part * head_dim + d] = acc;
    if (d == 0) {
        partial_max[part] = row_max;
        partial_sum[part] = row_sum;
    }
}

extern "C" __global__ void qwen4_qsa_sparse_attention_bf16_reduce(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_query_heads,
    unsigned int head_dim,
    unsigned int num_splits) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int d = threadIdx.x;
    if (q_head >= num_query_heads || d >= head_dim) return;
    const unsigned long long base = (unsigned long long)q_head * num_splits;
    float global_max = -CUDART_INF_F;
    for (unsigned int split = 0; split < num_splits; ++split)
        global_max = fmaxf(global_max, partial_max[base + split]);
    float denom = 0.0f;
    float acc = 0.0f;
    for (unsigned int split = 0; split < num_splits; ++split) {
        const float scale = __expf(partial_max[base + split] - global_max);
        denom += partial_sum[base + split] * scale;
        acc += partial_output[(base + split) * head_dim + d] * scale;
    }
    output[(unsigned long long)q_head * head_dim + d] =
        __float2bfloat16_rn(denom > 0.0f ? acc / denom : 0.0f);
}
