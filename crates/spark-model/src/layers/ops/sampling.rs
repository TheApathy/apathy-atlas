// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// GPU-side argmax over BF16 logits.
///
/// Finds the index of the maximum value, writes a single u32 to `out`.
///
/// Kernel: `argmax_bf16(logits, out, n)`
/// Grid: (1, 1, 1)  Block: (1024, 1, 1)
pub fn argmax_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    out: DevicePtr,
    vocab_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(out)
        .arg_u32(vocab_size)
        .launch(stream)
}

/// GPU-side argmax + embedding lookup — eliminates D2H sync in MTP propose.
///
/// Reads the argmax result from `argmax_out`, looks up the embedding row
/// from `embed_table`, and writes it to `embed_out`. Also copies the token
/// ID to `token_id_out` for deferred CPU readback.
pub fn embed_from_argmax(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    argmax_out: DevicePtr,
    embed_table: DevicePtr,
    embed_out: DevicePtr,
    token_id_out: DevicePtr,
    hidden_size: u32,
    stream: u64,
) -> Result<()> {
    let grid_x = hidden_size.div_ceil(256);
    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(argmax_out)
        .arg_ptr(embed_table)
        .arg_ptr(embed_out)
        .arg_ptr(token_id_out)
        .arg_u32(hidden_size)
        .launch(stream)
}

/// GPU-side top-K over BF16 logits, batched across rows.
///
/// For each of `num_rows` rows, finds the top-`k` BF16 logits and writes
/// the resulting `(token_id, logit_value)` pairs sorted by logit descending.
///
/// Used by the DDTree (M4B v2) propose path to seed branch candidates per
/// MASK-position drafter output. K=8 is the common case; the kernel caps at
/// MAX_TOP_K=16 compile-time (silently truncates beyond).
///
/// Output layout (caller-allocated, both buffers row-major):
///   - `top_indices`: `[num_rows, k]` u32  — token IDs
///   - `top_logits` : `[num_rows, k]` f32  — raw BF16-→f32 logit values
///
/// Kernel: `topk_bf16(logits, top_indices, top_logits, num_rows, vocab, k)`
/// Grid: (num_rows, 1, 1)  Block: (1024, 1, 1)
pub fn topk_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    top_indices: DevicePtr,
    top_logits: DevicePtr,
    num_rows: u32,
    vocab: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_rows.max(1), 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(top_indices)
        .arg_ptr(top_logits)
        .arg_u32(num_rows)
        .arg_u32(vocab)
        .arg_u32(k)
        .launch(stream)
}

/// Batched embedding: gather N rows from embedding table in one launch.
///
/// Replaces N individual D2D copies with a single kernel.
/// `token_ids_dev` must point to `[num_tokens]` u32 on device.
///
/// Kernel: `batched_embed(token_ids, embed_table, output, hidden_size)`
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
pub fn batched_embed(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    token_ids_dev: DevicePtr,
    embed_table: DevicePtr,
    output: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(token_ids_dev)
        .arg_ptr(embed_table)
        .arg_ptr(output)
        .arg_u32(hidden_size)
        .launch(stream)
}

// ── MoE routing ──────────────────────────────────────────────────
