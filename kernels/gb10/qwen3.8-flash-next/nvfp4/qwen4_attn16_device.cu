// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

// Address-strided M16 form of the incumbent interleaved MRoPE kernel.
// Arithmetic below intentionally matches common/rope_mrope_interleaved.cu.
extern "C" __global__ void rope_forward_mrope_interleaved_strided(
    __nv_bfloat16* __restrict__ qkv,
    const unsigned int* pos_t,
    const unsigned int* pos_h,
    const unsigned int* pos_w,
    const unsigned int rows,
    const unsigned int row_stride_bf16,
    const unsigned int k_offset_bf16,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const float theta
) {
    const unsigned int head_idx = blockIdx.x;
    const unsigned int seq_block = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const bool is_q = head_idx < num_q_heads;
    const unsigned int head = is_q ? head_idx : head_idx - num_q_heads;
    if (!is_q && head >= num_kv_heads) return;

    const unsigned int pairs_per_pos = rotary_dim / 2;
    const unsigned int pos_per_block = 128 / pairs_per_pos;
    if (pos_per_block == 0) return;
    const unsigned int local_pos = tid / pairs_per_pos;
    const unsigned int pair_idx = tid % pairs_per_pos;
    const unsigned int seq_pos = seq_block * pos_per_block + local_pos;
    if (seq_pos >= rows || local_pos >= pos_per_block) return;

    const unsigned int section = pair_idx % 3;
    unsigned int abs_pos;
    if (section == 0) abs_pos = pos_t[seq_pos];
    else if (section == 1) abs_pos = pos_h[seq_pos];
    else abs_pos = pos_w[seq_pos];

    const double freq_exp_d = (double)(2 * pair_idx) / (double)rotary_dim;
    const float freq = (float)(1.0 / pow((double)theta, freq_exp_d));
    const float angle = (float)abs_pos * freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    __nv_bfloat16* ptr = qkv
        + (uint64_t)seq_pos * row_stride_bf16
        + (is_q ? 0 : k_offset_bf16)
        + (uint64_t)head * head_dim;
    const unsigned int half_rot = rotary_dim / 2;
    const unsigned int d0 = pair_idx;
    const unsigned int d1 = pair_idx + half_rot;
    const float x0 = (float)ptr[d0];
    const float x1 = (float)ptr[d1];
    const float y0 = x0 * cos_val - x1 * sin_val;
    const float y1 = x1 * cos_val + x0 * sin_val;
    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}

extern "C" __global__ void qwen4_attn16_expand_meta(
    const int* __restrict__ source_block_table,
    int* __restrict__ expanded_tables,
    unsigned int* __restrict__ row_lengths,
    unsigned int* __restrict__ current_sequence_length,
    const unsigned int block_count,
    const unsigned int tile_start
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int table_words = 16 * block_count;
    if (idx < table_words) {
        expanded_tables[idx] = source_block_table[idx % block_count];
    }
    if (idx < 16) row_lengths[idx] = tile_start + idx + 1;
    if (idx == 0) current_sequence_length[0] = tile_start + 16;
}

extern "C" __global__ void qwen4_attn32_expand_meta(
    const int* __restrict__ source_block_table,
    int* __restrict__ expanded_tables,
    unsigned int* __restrict__ row_lengths,
    unsigned int* __restrict__ current_sequence_length,
    const unsigned int block_count,
    const unsigned int tile_start
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int table_words = 32 * block_count;
    if (idx < table_words) {
        expanded_tables[idx] = source_block_table[idx % block_count];
    }
    if (idx < 32) row_lengths[idx] = tile_start + idx + 1;
    if (idx == 0) current_sequence_length[0] = tile_start + 32;
}
