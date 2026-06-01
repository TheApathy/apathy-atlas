// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `super::super::decode.rs` for file-size budget.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};
use spark_runtime::kv_dequant::{
    NVFP4_E2M1_LUT, TURBO4_LUT, dequant_4bit_block_to_bf16, dequant_fp8_to_bf16,
    dequant_turbo3_block_to_bf16, dequant_turbo8_block_to_bf16,
};

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

/// Cached env-var lookup for `ATLAS_PAGED_DECODE_SPLITK`. Read once on first
/// access; subsequent calls are a relaxed atomic load.
fn paged_decode_splitk_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(2); // 2 = uninit, 0 = off, 1 = on
    let s = STATE.load(Ordering::Relaxed);
    if s != 2 {
        return s == 1;
    }
    let on = std::env::var("ATLAS_PAGED_DECODE_SPLITK")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    STATE.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

/// Compute `num_splits` for paged-decode split-K.
///
/// Two regimes:
///   • Legacy (default, `ATLAS_PAGED_DECODE_SPLITK` unset / 0): fill SMs only
///     when `current_ctas < NUM_SMS`. Once num_q_heads * num_seqs ≥ NUM_SMS the
///     dispatch falls back to the single-CTA kernel, which serializes the
///     entire KV history through 8 warps. For K=3 verify on aeon-27b
///     (24×3 = 72 CTAs ≥ 48 SMs → num_splits = 1), each CTA scans the full KV
///     history; this is the production setting.
///   • Aggressive (`ATLAS_PAGED_DECODE_SPLITK=1`, EXPERIMENTAL): additionally
///     split KV when `seq_len` is long. Empirically REGRESSES throughput on
///     FP8 KV at ctx=4K (measured 9 → 4 tok/s on aeon-27b K=3) because the
///     `paged_decode_attn_splitk_fp8` kernel processes one KV position at a
///     time (no BC=4 batching), while the non-splitk `paged_decode_attn_fp8`
///     kernel batches BC=4 K/V loads. Per-CTA throughput is ~4× lower in
///     split-K, and we add a reduce kernel launch on top. Each split processes
///     ~`SPLIT_TILE` KV positions; bounded by `MAX_SPLITS_CAP` so the workspace
///     allocated in `BufferSizes::splitk_workspace` always fits.
///
/// The aggressive regime is kept for completeness — it can be useful as a
/// scaffold for a future BC=4-aware split-K kernel that combines work-stealing
/// with batched loads — but the env var must remain OFF by default.
fn compute_num_splits(num_q_heads: u32, num_seqs: u32, max_seq_len_host: u32) -> u32 {
    use atlas_core::device::sm121::NUM_SMS;
    const SPLIT_TILE: u32 = 512;
    const MAX_SPLITS_CAP: u32 = 64;

    let current_ctas = num_q_heads * num_seqs;
    let legacy = if current_ctas >= NUM_SMS {
        1u32
    } else {
        NUM_SMS / current_ctas
    };

    if !paged_decode_splitk_enabled() {
        return legacy;
    }

    let seq_target = ((max_seq_len_host + SPLIT_TILE - 1) / SPLIT_TILE)
        .max(1)
        .min(MAX_SPLITS_CAP);
    // Keep at least `legacy` splits — never regress on the fill-SMs heuristic.
    // The kernel handles num_splits=1 directly via the non-splitk fallback, so
    // only invoke split-K when we have something to gain (>= 2 splits).
    seq_target.max(legacy)
}

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn run_paged_decode(
        &self,
        gpu: &dyn GpuBackend,
        q: DevicePtr,
        kv_cache: &PagedKvCache,
        output: DevicePtr,
        block_table: DevicePtr,
        seq_lens: DevicePtr,
        max_blocks_per_seq: u32,
        num_seqs: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        q_stride: u32,
        workspace: DevicePtr,
        // ATLAS_TREE_AWARE_ATTN: optional KV indirection from ForwardContext.
        // Pass `(NULL, NULL, 0)` for the legacy chain-mode path.
        // CUDA graph fix: `kv_indir_base_ptr` is a 1×i32 device buffer so
        // captured graphs see the fresh value on each replay.
        kv_indirection: DevicePtr,
        kv_indir_base_ptr: DevicePtr,
        kv_indir_stride: u32,
        // Host-side upper bound on `seq_lens[..num_seqs]`. Used to scale
        // `num_splits` when `ATLAS_PAGED_DECODE_SPLITK=1`. Pass the current
        // (post-write) KV length; over-estimating is safe — it only changes
        // the partition factor, never correctness.
        max_seq_len_host: u32,
        stream: u64,
    ) -> Result<()> {
        match self.kv_dtype {
            KvCacheDtype::Nvfp4 => {
                let num_splits = compute_num_splits(num_q_heads, num_seqs, max_seq_len_host);

                if num_splits > 1 {
                    let splitk_k = self
                        .paged_decode_splitk_k
                        .expect("split-K kernel required for NVFP4");
                    let reduce_k = self
                        .paged_decode_reduce_k
                        .expect("reduce kernel required for NVFP4");
                    ops::paged_decode_attn_splitk_nvfp4(
                        gpu,
                        splitk_k,
                        q,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        workspace,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        inv_sqrt_d,
                        num_splits,
                        q_stride,
                        kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                        kv_cache.nvfp4_data_bytes() as u64,
                        num_seqs,
                        kv_indirection,
                        kv_indir_base_ptr,
                        kv_indir_stride,
                        stream,
                    )?;
                    ops::paged_decode_attn_reduce_nvfp4(
                        gpu,
                        reduce_k,
                        workspace,
                        output,
                        seq_lens,
                        num_q_heads,
                        head_dim,
                        num_splits,
                        num_seqs,
                        stream,
                    )
                } else {
                    ops::paged_decode_attn_nvfp4(
                        gpu,
                        self.paged_decode_k,
                        q,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        output,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_seqs,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        inv_sqrt_d,
                        q_stride,
                        kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                        kv_cache.nvfp4_data_bytes() as u64,
                        kv_indirection,
                        kv_indir_base_ptr,
                        kv_indir_stride,
                        stream,
                    )
                }
            }
            // Turbo4/3: same 4-bit interface as NVFP4 (block_stride + data_section layout).
            KvCacheDtype::Turbo4 | KvCacheDtype::Turbo3 => {
                let kernel = if head_dim > 256 && self.paged_decode_512_k.0 != 0 {
                    self.paged_decode_512_k
                } else {
                    self.paged_decode_k
                };
                let data_bytes = match self.kv_dtype {
                    KvCacheDtype::Turbo3 => kv_cache.turbo3_data_bytes() as u64,
                    _ => kv_cache.turbo4_data_bytes() as u64,
                };
                ops::paged_decode_attn_nvfp4(
                    gpu,
                    kernel,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    data_bytes,
                    kv_indirection,
                    kv_indir_base_ptr,
                    kv_indir_stride,
                    stream,
                )
            }
            // Turbo8: WHT + FP8 — 1 byte per element + per-group FP8 scales.
            KvCacheDtype::Turbo8 => {
                let kernel = if head_dim > 256 && self.paged_decode_512_k.0 != 0 {
                    self.paged_decode_512_k
                } else {
                    self.paged_decode_k
                };
                ops::paged_decode_attn_nvfp4(
                    gpu,
                    kernel,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.turbo8_data_bytes() as u64,
                    kv_indirection,
                    kv_indir_base_ptr,
                    kv_indir_stride,
                    stream,
                )
            }
            KvCacheDtype::Bf16 => {
                // BF16 paged decode — no Split-K (not implemented for BF16 yet)
                // Use HDIM=512 kernel for Gemma-4 full-attention layers (head_dim > 256)
                let kernel = if head_dim > 256 && self.paged_decode_512_k.0 != 0 {
                    self.paged_decode_512_k
                } else {
                    self.paged_decode_k
                };
                // Gemma-4 sliding layers attend only to the last `window_size`
                // KV positions; full layers (and all non-Gemma-4 models) pass 0.
                let sliding = self.sliding_window.unwrap_or(0);
                ops::paged_decode_attn_bf16(
                    gpu,
                    kernel,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    sliding,
                    stream,
                )
            }
            _ => {
                // FP8 paged decode
                let num_splits = compute_num_splits(num_q_heads, num_seqs, max_seq_len_host);

                let (k_scale, v_scale) = self.effective_fp8_scales();

                if num_splits > 1 {
                    let splitk_k = self
                        .paged_decode_splitk_k
                        .expect("split-K kernel required for FP8");
                    let reduce_k = self
                        .paged_decode_reduce_k
                        .expect("reduce kernel required for FP8");
                    ops::paged_decode_attn_splitk_fp8(
                        gpu,
                        splitk_k,
                        q,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        workspace,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        inv_sqrt_d,
                        num_splits,
                        k_scale,
                        v_scale,
                        q_stride,
                        kv_cache.cache_stride() as u64,
                        num_seqs,
                        kv_indirection,
                        kv_indir_base_ptr,
                        kv_indir_stride,
                        stream,
                    )?;
                    ops::paged_decode_attn_reduce_fp8(
                        gpu,
                        reduce_k,
                        workspace,
                        output,
                        seq_lens,
                        num_q_heads,
                        head_dim,
                        num_splits,
                        num_seqs,
                        stream,
                    )
                } else {
                    // Use HDIM=512 kernel for Gemma-4 full-attention layers
                    let fp8_kernel = if head_dim > 256 && self.paged_decode_512_k.0 != 0 {
                        self.paged_decode_512_k
                    } else {
                        self.paged_decode_k
                    };
                    ops::paged_decode_attn_fp8(
                        gpu,
                        fp8_kernel,
                        q,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        output,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_seqs,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        inv_sqrt_d,
                        k_scale,
                        v_scale,
                        q_stride,
                        kv_cache.cache_stride() as u64,
                        kv_indirection,
                        kv_indir_base_ptr,
                        kv_indir_stride,
                        // No tree pack for the standard single-seq decode path.
                        spark_runtime::gpu::DevicePtr::NULL,
                        spark_runtime::gpu::DevicePtr::NULL,
                        0,
                        stream,
                    )
                }
            }
        }
    }
}
