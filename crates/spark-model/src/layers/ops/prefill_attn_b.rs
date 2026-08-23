// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Write K/V to paged NVFP4 cache (E2M1 data + per-group FP8 scales).
///
/// Kernel: `reshape_and_cache_flash_nvfp4(key, value, k_cache, v_cache,
///          slot_mapping, num_kv_heads, head_dim, block_size,
///          key_stride, value_stride, block_stride_bytes, data_section_bytes)`
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
pub fn reshape_and_cache_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    key: DevicePtr,
    value: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    slot_mapping: DevicePtr,
    num_tokens: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    key_stride: u32,
    value_stride: u32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(slot_mapping)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_u32(key_stride)
        .arg_u32(value_stride)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}

/// Compute max absolute value of a BF16 buffer into a device-side f32.
///
/// Used for FP8 KV cache online scale calibration: accumulates max |K| and
/// max |V| during warmup tokens. The output f32 is updated via atomicMax,
/// so the caller must initialize it to 0.0 before the first call.
///
/// Kernel: `bf16_absmax(data, out_max, n_elems)`
/// Grid: (ceil(n_elems / (256*2)), 1, 1)  Block: (256, 1, 1)
pub fn bf16_absmax(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    data: DevicePtr,
    out_max: DevicePtr,
    n_elems: u32,
    stream: u64,
) -> Result<()> {
    // Each thread handles multiple pairs; use enough blocks to cover the buffer.
    // 256 threads per block, each reads ~8 pairs in the inner loop.
    let grid_x = (n_elems as u64).div_ceil(256 * 2).min(256) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(data)
        .arg_ptr(out_max)
        .arg_u32(n_elems)
        .launch(stream)
}

/// FlashAttention-v2 inspired K=γ-fused paged-decode attention (NVFP4 KV).
///
/// Collapses the QTILE = γ+1 axis into a single CTA per q_head: each warp
/// owns a slice of queries, K and V vectors are loaded once and reused
/// across that warp's queries. Caller MUST guarantee:
///   - `num_qtile <= QTILE_MAX (32)` (kernel compile-time bound)
///   - All `num_qtile` rows of `block_tables` are identical (K=γ verify)
///   - `kv_indirection == NULL` (tree-aware path uses legacy kernel)
///   - `head_dim == 256` (kernel compiled with HDIM=256)
///
/// Grid: `(num_q_heads, 1, 1)`  Block: `(256, 1, 1)`
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_kgamma_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    q_stride: u32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    num_qtile: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_qtile.div_ceil(2), 1])
        .block([512, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(q_stride)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .arg_u32(num_qtile)
        // Tree-aware indirection slots: must be NULL per contract.
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_u32(0)
        .launch(stream)
}

/// Qwen3.8 M=15 fused-query attention over per-tensor FP8 K/V cache.
/// The kernel retains the established FP8 BC=4 online-softmax update.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_kgamma_fp8_bc4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    k_scale: f32,
    v_scale: f32,
    q_stride: u32,
    cache_stride: u64,
    num_qtile: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(num_qtile == 15, "FP8 Kgamma BC4 requires M=15");
    anyhow::ensure!(head_dim == 128, "FP8 Kgamma BC4 requires HDIM=128");
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_qtile.div_ceil(2), 1])
        .block([512, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u32(q_stride)
        .arg_u64(cache_stride)
        .arg_u32(num_qtile)
        .launch(stream)
}

/// Flat K-gamma fused paged-decode attention for BF16 KV, HDIM=256.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_kgamma_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    q_stride: u32,
    cache_stride: u64,
    num_qtile: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_qtile.div_ceil(2), 1])
        .block([512, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(q_stride)
        .arg_u64(cache_stride)
        .arg_u32(num_qtile)
        .launch(stream)
}

/// VEC variant of the K=γ-fused NVFP4 paged-decode attention.
///
/// Same caller contract as `paged_decode_attn_kgamma_nvfp4`. Processes
/// KV positions in pairs with batched dequant — the inner loop issues
/// all 4 NVFP4 dequants (K0, V0, K1, V1) back-to-back so the compiler
/// can interleave the loads with unpack ALU. NUM_WARPS=8 (same as
/// baseline). Gated by `ATLAS_KGAMMA_VECDEQUANT=1` at the dispatch site.
///
/// Grid: `(num_q_heads, 1, 1)`  Block: `(256, 1, 1)`
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_kgamma_nvfp4_vec(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    q_stride: u32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    num_qtile: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_qtile.div_ceil(2), 1])
        .block([512, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(q_stride)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .arg_u32(num_qtile)
        // Tree-aware indirection slots: must be NULL per contract.
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_u32(0)
        .launch(stream)
}

/// Split-K variant of the K=γ-fused NVFP4 paged-decode attention.
///
/// Partitions the KV history across `num_splits` CTAs per q_head, lifting
/// the grid from `(num_q_heads, 1, 1) = 4 CTAs` (single-CTA kgamma) to
/// `(num_q_heads, num_splits, 1) = 48 CTAs` (with num_splits=12 on a
/// 48-SM GB10), restoring SM occupancy for the γ=16 verify path.
///
/// Writes per-(qtile, q_head, split) partial `(o[HDIM], m, l)` to
/// `workspace`. The caller MUST follow this kernel with
/// `paged_decode_attn_kgamma_reduce_nvfp4` to combine partials and
/// produce the final BF16 output.
///
/// Caller contract is the same as the single-CTA kgamma kernel plus:
///   - `workspace` is F32 of size
///     `num_qtile * num_q_heads * num_splits * (head_dim + 2)`.
///   - `num_splits ≥ 2` (prefer single-CTA kernel for num_splits=1).
///
/// Grid: `(num_q_heads, num_splits, 1)`  Block: `(256, 1, 1)`
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_kgamma_nvfp4_splitk(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    workspace: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    num_splits: u32,
    q_stride: u32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    num_qtile: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_splits, 1])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(workspace)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(num_splits)
        .arg_u32(q_stride)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .arg_u32(num_qtile)
        .launch(stream)
}

/// FA2-grafted variant of the K=γ-fused NVFP4 paged-decode attention.
///
/// Same caller contract as `paged_decode_attn_kgamma_nvfp4`. The kernel
/// stages `FA2_TILE_N=32` KV positions into shared memory via cp.async,
/// double-buffers across `FA2_STAGES=2` SMEM slots, and overlaps the
/// load of tile N+1 with the compute on tile N — mirroring FA2's
/// `compute_attn_1rowblock` inner-loop shape but adapted to NVFP4-packed
/// paged-cache layout. Gated by `ATLAS_FA2_KGAMMA=1` at the dispatch site.
///
/// Grid: `(num_q_heads, 1, 1)`  Block: `(256, 1, 1)`
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_kgamma_nvfp4_fa2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    q_stride: u32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    num_qtile: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_qtile.div_ceil(2), 1])
        .block([512, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(q_stride)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .arg_u32(num_qtile)
        // Tree-aware indirection slots: must be NULL per contract.
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_u32(0)
        .launch(stream)
}

/// Reduce kernel for the split-K kgamma path: merges `num_splits` partial
/// (m, l, o) tuples per (qtile, q_head) into the final BF16 output via
/// standard log-sum-exp rescaling.
///
/// Grid: `(num_q_heads, num_qtile, 1)`  Block: `(32, 1, 1)`
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_kgamma_reduce_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    workspace: DevicePtr,
    output: DevicePtr,
    num_q_heads: u32,
    num_splits: u32,
    num_qtile: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_qtile, 1])
        .block([32, 1, 1])
        .arg_ptr(workspace)
        .arg_ptr(output)
        .arg_u32(num_q_heads)
        .arg_u32(num_splits)
        .arg_u32(num_qtile)
        .launch(stream)
}

/// Paged decode attention (NVFP4 KV cache, single/multi sequence).
///
/// Kernel: `paged_decode_attn_nvfp4(Q, K_cache, V_cache, O, block_tables,
///          seq_lens, max_blocks_per_seq, num_q_heads, num_kv_heads,
///          head_dim, block_size, inv_sqrt_d, q_stride,
///          block_stride_bytes, data_section_bytes)`
/// Grid: (num_q_heads, num_seqs, 1)  Block: (256, 1, 1)
pub fn paged_decode_attn_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_seqs: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    q_stride: u32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    // ATLAS_TREE_AWARE_ATTN: optional KV indirection. Pass DevicePtr::NULL
    // for `kv_indirection` and `kv_indir_base_ptr` plus `0` for the stride
    // to take the legacy chain-mode path (kernel behavior unchanged).
    //
    // `kv_indir_base_ptr` is a 1×i32 device buffer (graph-safe replacement
    // for the prior scalar) — see `paged_decode_attn_fp8` for the contract.
    kv_indirection: DevicePtr,
    kv_indir_base_ptr: DevicePtr,
    kv_indir_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_seqs, 1])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(q_stride)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .arg_ptr(kv_indirection)
        .arg_ptr(kv_indir_base_ptr)
        .arg_u32(kv_indir_stride)
        .launch(stream)
}

/// Split-K paged decode attention (NVFP4 KV cache).
///
/// Partitions the KV sequence across `num_splits` CTAs per (q_head, seq).
/// Each CTA computes partial softmax + weighted output, written to `workspace`.
///
/// Grid: (num_q_heads, num_splits, num_seqs)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_splitk_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    workspace: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    num_splits: u32,
    q_stride: u32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    num_seqs: u32,
    kv_indirection: DevicePtr,
    kv_indir_base_ptr: DevicePtr,
    kv_indir_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_splits, num_seqs])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(workspace)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(num_splits)
        .arg_u32(q_stride)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .arg_ptr(kv_indirection)
        .arg_ptr(kv_indir_base_ptr)
        .arg_u32(kv_indir_stride)
        .launch(stream)
}

/// Reduce split-K partials into final BF16 output.
///
/// Grid: (num_q_heads, num_seqs, 1)  Block: (32, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_reduce_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    workspace: DevicePtr,
    output: DevicePtr,
    seq_lens: DevicePtr,
    num_q_heads: u32,
    head_dim: u32,
    num_splits: u32,
    num_seqs: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_seqs, 1])
        .block([32, 1, 1])
        .arg_ptr(workspace)
        .arg_ptr(output)
        .arg_ptr(seq_lens)
        .arg_u32(num_q_heads)
        .arg_u32(head_dim)
        .arg_u32(num_splits)
        .launch(stream)
}

/// Split-K paged decode attention (FP8 KV cache).
///
/// Partitions the KV sequence across `num_splits` CTAs per (q_head, seq).
/// Each CTA computes partial softmax + weighted output, written to `workspace`.
///
/// Grid: (num_q_heads, num_splits, num_seqs)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_splitk_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    workspace: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    num_splits: u32,
    k_scale: f32,
    v_scale: f32,
    q_stride: u32,
    cache_stride: u64,
    num_seqs: u32,
    kv_indirection: DevicePtr,
    kv_indir_base_ptr: DevicePtr,
    kv_indir_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_splits, num_seqs])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(workspace)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(num_splits)
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u32(q_stride)
        .arg_u64(cache_stride)
        .arg_ptr(kv_indirection)
        .arg_ptr(kv_indir_base_ptr)
        .arg_u32(kv_indir_stride)
        .launch(stream)
}

/// Reduce split-K partials into final BF16 output (FP8 variant).
///
/// Grid: (num_q_heads, num_seqs, 1)  Block: (32, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_reduce_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    workspace: DevicePtr,
    output: DevicePtr,
    seq_lens: DevicePtr,
    num_q_heads: u32,
    head_dim: u32,
    num_splits: u32,
    num_seqs: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_seqs, 1])
        .block([32, 1, 1])
        .arg_ptr(workspace)
        .arg_ptr(output)
        .arg_ptr(seq_lens)
        .arg_u32(num_q_heads)
        .arg_u32(head_dim)
        .arg_u32(num_splits)
        .launch(stream)
}

// ── SSM / Convolution ──────────────────────────────────────────────
