// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash 2 `GroupedDynamicCausalConv` + candidate-selector ops.
//!
//! Conv math (reference: z-lab/dflash `dflash/model.py`):
//!   `out[b,l,g,s] = sum_offset (base[stage][offset][g,s] + dyn[b,l,stage,offset,g]) * x[b,l-offset,g,s]`
//! with causal zero padding. `prepare` runs stage 0 and exports the stage-1
//! dynamic rows for `finish` (which runs stage 1 on the sublayer output).

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::*;

/// `dflash2_conv_prepare` — stage 0 conv of `hidden` plus export of the
/// stage-1 dynamic rows.
///
/// Args (device BF16 unless noted): hidden `[n_attn, hidden]`, dynamic
/// `[n_attn, 2*kernel*groups]`, base `[2, kernel, hidden]`, out
/// `[n_attn, hidden]`, dyn1_out `[n_attn, kernel*groups]`, n_attn, groups.
pub fn dflash2_conv_prepare(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    dynamic: DevicePtr,
    base: DevicePtr,
    out: DevicePtr,
    dyn1_out: DevicePtr,
    n_attn: u32,
    groups: u32,
    stream: u64,
) -> Result<()> {
    let elems = n_attn as u64 * groups as u64 * 16; // GROUP_SIZE = 16
    let grid = div_ceil(elems as u32, 256);
    KernelLaunch::new(gpu, kernel)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(dynamic)
        .arg_ptr(base)
        .arg_ptr(out)
        .arg_ptr(dyn1_out)
        .arg_u32(n_attn)
        .arg_u32(groups)
        .launch(stream)
}

/// `dflash2_conv_finish` — stage 1 conv of the sublayer output.
pub fn dflash2_conv_finish(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    dynamic: DevicePtr,
    base: DevicePtr,
    out: DevicePtr,
    n_attn: u32,
    groups: u32,
    stream: u64,
) -> Result<()> {
    let elems = n_attn as u64 * groups as u64 * 16;
    let grid = div_ceil(elems as u32, 256);
    KernelLaunch::new(gpu, kernel)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(dynamic)
        .arg_ptr(base)
        .arg_ptr(out)
        .arg_u32(n_attn)
        .arg_u32(groups)
        .launch(stream)
}

/// `dflash2_selector_walk` — DFlash 2 candidate-selector greedy walk (T=0).
///
/// Reference: `CandidateSelector.select` (z-lab/dflash `dflash/model.py`),
/// greedy branch: per position, `scores[k] = unary[k] + (pred*hidden)·succ[k]`,
/// `index = argmax_k`, `predecessor = candidates[index]`.
///
/// Args:
///   unary      `[γ, TOP_K]` f32 (topk logits)
///   candidates `[γ, TOP_K]` u32 (topk token ids)
///   hidden_proj `[γ, rank]` BF16 (hidden_projection(draft hidden))
///   pred_codebook `[V, rank]` BF16
///   succ_codebook `[V, rank]` BF16
///   path       `[γ]` u32 out (selected draft token ids)
///   anchor_id  u32 — the last verified token (row 0's predecessor)
///   gamma      i32
///   rank       i32
///
/// One block of 16 threads (TOP_K), serialized across positions.
pub fn dflash2_selector_walk(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    unary: DevicePtr,
    candidates: DevicePtr,
    hidden_proj: DevicePtr,
    pred_codebook: DevicePtr,
    succ_codebook: DevicePtr,
    path: DevicePtr,
    anchor_id: u32,
    gamma: i32,
    rank: i32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([16, 1, 1])
        .arg_ptr(unary)
        .arg_ptr(candidates)
        .arg_ptr(hidden_proj)
        .arg_ptr(pred_codebook)
        .arg_ptr(succ_codebook)
        .arg_ptr(path)
        .arg_u32(anchor_id)
        .arg_i32(gamma)
        .arg_i32(rank)
        .launch(stream)
}
