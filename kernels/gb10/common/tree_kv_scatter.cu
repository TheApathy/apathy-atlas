// SPDX-License-Identifier: AGPL-3.0-only
//
// Tree-aware KV scatter — packs ancestor KV from a paged cache into a
// contiguous "scratch pool" so the existing paged_decode_attn kernels can
// process tree verification using their fast BC=4 batched path, without
// per-position indirection fallback.
//
// Used by `ATLAS_TREE_KV_PACK=1` (gated additionally by
// `ATLAS_TREE_AWARE_ATTN=1`). For each query row `t` in the K=γ verify
// batch, we materialize its ancestor chain (in compact-slot order) into a
// contiguous scratch block of size `[max_chain_len, num_kv_heads, head_dim]`.
// Positions `< kv_indir_base` (prior linear context) are copied verbatim
// from the original cache positions (they map identity), positions
// `>= kv_indir_base` use the per-row `kv_indirection` table to resolve to
// the true ancestor compact slot.
//
// Two flavors:
//   - FP8 (1 byte/elem, no per-block scale section):
//       per-token bytes  = num_kv_heads * head_dim
//       block_stride     = block_size * num_kv_heads * head_dim   (= cache_stride bytes)
//   - NVFP4 (4-bit data + per-group FP8 scale section):
//       per-token data bytes  = num_kv_heads * head_dim / 2
//       per-token scale bytes = num_kv_heads * head_dim / 16
//       block layout: [data | scale] (data_section_bytes then scale_section_bytes)
//
// The scratch layout MIRRORS the cache layout — same stride formula — so
// the consuming kernel can be invoked with `block_size = max_chain_len`,
// `num_blocks = num_seqs`, and an identity block_table.
//
// All kernels: grid (num_seqs, max_chain_len, 1), block (256, 1, 1).
// Each CTA copies one (seq, slot) record of `num_kv_heads * head_dim`
// elements (split across 256 threads). For typical Qwen3.6 27B configs
// (num_kv_heads=4, head_dim=128 => 512 elements/token, FP8 => 512 bytes
// or NVFP4 => 256 data bytes + 32 scale bytes), a single CTA finishes in
// a couple of cycles. Total bytes moved per layer per token = 2*512 = 1 KB
// (K + V); at 273 GB/s that's < 4 ns per launch — utterly negligible vs the
// >40 ms decode step cost.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// ============================================================================
// FP8 scatter (or any "byte-per-elem with no scale section" layout)
// ============================================================================
//
// Copies one ancestor (K AND V) per CTA from the paged cache into the
// contiguous packed scratch. Falls through to identity for positions in
// the prior-linear-context region (slot < kv_indir_base / block_size frame).
//
// k_pool / v_pool       — original paged caches, FP8 byte layout
// k_scratch / v_scratch — contiguous output, identical byte layout but with
//                          block_size = max_chain_len, num_blocks = num_seqs
// block_table           — caller's real block table, [num_seqs * max_blocks_per_seq] i32
// kv_indirection        — per-row ancestor chain (compact slot indices), [num_seqs * kv_indir_stride] i32
//                          Indexed by `chain_idx = (slot - kv_indir_base_in_slots)`. For our
//                          scatter, we always call from a tree window where every slot is in
//                          the indirected region (caller passes kv_indir_base = 0 in slot space).
// max_chain_len         — output block_size (= ddtree_kgamma stride)
// num_seqs              — output num_blocks
// num_kv_heads          — model num_kv_heads
// head_dim              — model head_dim
// cache_block_size      — input block_size
// kv_indir_stride       — row stride of kv_indirection
// abs_base_ptr          — pointer to a 1×i32 device buffer holding the absolute
//                         position where the tree window begins (= seq.seq_len).
//                         CUDA graph fix: keeping the value in a device buffer
//                         (instead of a kernel-launch immediate) lets a captured
//                         graph see the fresh `seq.seq_len` on each replay. Host
//                         writes the current value before launch.
//                         Slots 0..chain_len are appended on top of the prior context;
//                         for our use we re-stamp the indirection to be absolute compact
//                         positions, so the kernel just multiplies by per-token stride.
//
// `kv_indirection[t * kv_indir_stride + slot]` returns the compact slot index of the
// `slot`-th ancestor of query row `t`. The absolute KV cache position is
// `abs_base + compact_slot`, then we resolve via the caller's block_table.

extern "C" __global__ void tree_kv_scatter_fp8(
    const unsigned char* __restrict__ k_pool,
    const unsigned char* __restrict__ v_pool,
    const int* __restrict__ block_table,
    const int* __restrict__ kv_indirection,
    unsigned char* __restrict__ k_scratch,
    unsigned char* __restrict__ v_scratch,
    unsigned int num_seqs,
    unsigned int max_chain_len,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int cache_block_size,
    unsigned int max_blocks_per_seq,
    unsigned int kv_indir_stride,
    const int* __restrict__ abs_base_ptr
) {
    const unsigned int seq = blockIdx.x;
    const unsigned int slot = blockIdx.y;
    if (seq >= num_seqs || slot >= max_chain_len) return;

    // Load tree-window base position from device buffer (graph-safe, volatile).
    const unsigned int abs_base = (unsigned int)(*((const volatile int*)abs_base_ptr));

    // Resolve ancestor compact slot for (seq, slot).
    const int compact = kv_indirection[seq * kv_indir_stride + slot];
    const unsigned int abs_pos = abs_base + (unsigned int)compact;
    const unsigned int logical_block = abs_pos / cache_block_size;
    const unsigned int block_offset = abs_pos % cache_block_size;
    const int* my_bt = block_table + seq * max_blocks_per_seq;
    const unsigned int physical_block = (unsigned int)my_bt[logical_block];

    // FP8 cache_stride in BYTES (1 byte/elem).
    const unsigned long long token_stride = (unsigned long long)num_kv_heads * head_dim;
    const unsigned long long block_stride = (unsigned long long)cache_block_size * token_stride;

    // Source pointers (paged cache).
    const unsigned char* k_src = k_pool
        + (unsigned long long)physical_block * block_stride
        + (unsigned long long)block_offset * token_stride;
    const unsigned char* v_src = v_pool
        + (unsigned long long)physical_block * block_stride
        + (unsigned long long)block_offset * token_stride;

    // Dest pointers (contiguous scratch). Scratch block_stride uses
    // max_chain_len, not cache_block_size.
    const unsigned long long out_block_stride = (unsigned long long)max_chain_len * token_stride;
    unsigned char* k_dst = k_scratch
        + (unsigned long long)seq * out_block_stride
        + (unsigned long long)slot * token_stride;
    unsigned char* v_dst = v_scratch
        + (unsigned long long)seq * out_block_stride
        + (unsigned long long)slot * token_stride;

    // 256 threads copy `token_stride` bytes per call (K then V). Typical
    // token_stride = 512 bytes; we vectorize 16 bytes per thread when
    // aligned, else fall back to byte copy.
    const unsigned int tid = threadIdx.x;
    const unsigned int nthreads = blockDim.x;

    // Vectorized uint4 copy (16 bytes/thread).
    const bool aligned_in = ((((uintptr_t)k_src) | ((uintptr_t)v_src)) & 0xFu) == 0
                         && ((((uintptr_t)k_dst) | ((uintptr_t)v_dst)) & 0xFu) == 0
                         && (token_stride & 0xFu) == 0;
    if (aligned_in) {
        const unsigned int n4 = (unsigned int)(token_stride >> 4);
        const uint4* k_src4 = (const uint4*)k_src;
        const uint4* v_src4 = (const uint4*)v_src;
        uint4* k_dst4 = (uint4*)k_dst;
        uint4* v_dst4 = (uint4*)v_dst;
        for (unsigned int i = tid; i < n4; i += nthreads) {
            k_dst4[i] = k_src4[i];
            v_dst4[i] = v_src4[i];
        }
    } else {
        for (unsigned long long i = tid; i < token_stride; i += nthreads) {
            k_dst[i] = k_src[i];
            v_dst[i] = v_src[i];
        }
    }
}

// ============================================================================
// NVFP4 scatter (data section + per-group FP8 scale section)
// ============================================================================
//
// NVFP4 cache layout per physical block (K or V independently):
//   [block_size * num_kv_heads * head_dim/2 data bytes]
//   [block_size * num_kv_heads * head_dim/16 scale bytes]
//
// Within data: token-major (`pos * num_kv_heads * head_dim/2 + kv_head * head_dim/2 + lane_byte`).
// Within scale: token-major (`pos * num_kv_heads * head_dim/16 + kv_head * head_dim/16 + group_byte`).
//
// We copy both sections for each (seq, slot) ancestor into a scratch block of
// max_chain_len tokens. Scratch layout uses the SAME formula but with
// max_chain_len in place of block_size — so the consumer kernel sees an
// indistinguishable NVFP4 block.

extern "C" __global__ void tree_kv_scatter_nvfp4(
    const unsigned char* __restrict__ k_pool,
    const unsigned char* __restrict__ v_pool,
    const int* __restrict__ block_table,
    const int* __restrict__ kv_indirection,
    unsigned char* __restrict__ k_scratch,
    unsigned char* __restrict__ v_scratch,
    unsigned int num_seqs,
    unsigned int max_chain_len,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int cache_block_size,
    unsigned int max_blocks_per_seq,
    unsigned int kv_indir_stride,
    const int* __restrict__ abs_base_ptr,
    unsigned long long cache_block_stride_bytes,
    unsigned long long cache_data_section_bytes,
    unsigned long long scratch_block_stride_bytes,
    unsigned long long scratch_data_section_bytes
) {
    const unsigned int seq = blockIdx.x;
    const unsigned int slot = blockIdx.y;
    if (seq >= num_seqs || slot >= max_chain_len) return;

    // Load tree-window base position from device buffer (graph-safe, volatile).
    const unsigned int abs_base = (unsigned int)(*((const volatile int*)abs_base_ptr));

    const int compact = kv_indirection[seq * kv_indir_stride + slot];
    const unsigned int abs_pos = abs_base + (unsigned int)compact;
    const unsigned int logical_block = abs_pos / cache_block_size;
    const unsigned int block_offset = abs_pos % cache_block_size;
    const int* my_bt = block_table + seq * max_blocks_per_seq;
    const unsigned int physical_block = (unsigned int)my_bt[logical_block];

    // Per-token bytes within data + scale sections.
    const unsigned long long data_stride  = (unsigned long long)num_kv_heads * (head_dim / 2);
    const unsigned long long scale_stride = (unsigned long long)num_kv_heads * (head_dim / 16);

    // ── Data section ──
    const unsigned char* k_src_data = k_pool
        + (unsigned long long)physical_block * cache_block_stride_bytes
        + (unsigned long long)block_offset * data_stride;
    const unsigned char* v_src_data = v_pool
        + (unsigned long long)physical_block * cache_block_stride_bytes
        + (unsigned long long)block_offset * data_stride;
    unsigned char* k_dst_data = k_scratch
        + (unsigned long long)seq * scratch_block_stride_bytes
        + (unsigned long long)slot * data_stride;
    unsigned char* v_dst_data = v_scratch
        + (unsigned long long)seq * scratch_block_stride_bytes
        + (unsigned long long)slot * data_stride;

    // ── Scale section ──
    const unsigned char* k_src_scale = k_pool
        + (unsigned long long)physical_block * cache_block_stride_bytes
        + cache_data_section_bytes
        + (unsigned long long)block_offset * scale_stride;
    const unsigned char* v_src_scale = v_pool
        + (unsigned long long)physical_block * cache_block_stride_bytes
        + cache_data_section_bytes
        + (unsigned long long)block_offset * scale_stride;
    unsigned char* k_dst_scale = k_scratch
        + (unsigned long long)seq * scratch_block_stride_bytes
        + scratch_data_section_bytes
        + (unsigned long long)slot * scale_stride;
    unsigned char* v_dst_scale = v_scratch
        + (unsigned long long)seq * scratch_block_stride_bytes
        + scratch_data_section_bytes
        + (unsigned long long)slot * scale_stride;

    const unsigned int tid = threadIdx.x;
    const unsigned int nthreads = blockDim.x;

    // Copy data (vectorize if aligned).
    const bool data_aligned = ((((uintptr_t)k_src_data) | ((uintptr_t)v_src_data)) & 0xFu) == 0
                           && ((((uintptr_t)k_dst_data) | ((uintptr_t)v_dst_data)) & 0xFu) == 0
                           && (data_stride & 0xFu) == 0;
    if (data_aligned) {
        const unsigned int n4 = (unsigned int)(data_stride >> 4);
        const uint4* k_src4 = (const uint4*)k_src_data;
        const uint4* v_src4 = (const uint4*)v_src_data;
        uint4* k_dst4 = (uint4*)k_dst_data;
        uint4* v_dst4 = (uint4*)v_dst_data;
        for (unsigned int i = tid; i < n4; i += nthreads) {
            k_dst4[i] = k_src4[i];
            v_dst4[i] = v_src4[i];
        }
    } else {
        for (unsigned long long i = tid; i < data_stride; i += nthreads) {
            k_dst_data[i] = k_src_data[i];
            v_dst_data[i] = v_src_data[i];
        }
    }

    // Copy scales (byte path: only ~32 bytes typical, alignment not worth it).
    for (unsigned long long i = tid; i < scale_stride; i += nthreads) {
        k_dst_scale[i] = k_src_scale[i];
        v_dst_scale[i] = v_src_scale[i];
    }
}
