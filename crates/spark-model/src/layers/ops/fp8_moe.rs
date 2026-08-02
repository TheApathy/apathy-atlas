// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Full-sequence causal depthwise conv1d + SiLU activation.
///
/// Kernel: `causal_conv1d_fwd(input, weight, bias, output, batch, dim, seq_len, d_conv)`
/// Grid: (dim, batch, 1)  Block: (min(seq_len, 1024), 1, 1)
///
/// Input: [batch, dim, seq_len] BF16 (channel-first)
/// Weight: [dim, d_conv] BF16
/// Output: [batch, dim, seq_len] BF16
#[allow(clippy::too_many_arguments)]
pub fn conv1d_fwd(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    batch: u32,
    dim: u32,
    seq_len: u32,
    d_conv: u32,
    stream: u64,
) -> Result<()> {
    let block_x = std::cmp::min(seq_len, 1024);
    KernelLaunch::new(gpu, kernel)
        .grid([dim, batch, 1])
        .block([block_x, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(DevicePtr::NULL) // bias (none for this model)
        .arg_ptr(output)
        .arg_u32(batch)
        .arg_u32(dim)
        .arg_u32(seq_len)
        .arg_u32(d_conv)
        .launch(stream)
}

/// BF16 concatenation: out[0..N] = a[0..N], out[N..2N] = b[0..N].
///
/// Kernel: `bf16_concat(a, b, out, N)`
/// Grid: (ceil(N/256), 1, 1)  Block: (256, 1, 1)
pub fn bf16_concat(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    output: DevicePtr,
    n: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(output)
        .arg_u32(n)
        .launch(stream)
}

// ── FP8 MoE batch2/batch3 dispatch ──────────────────────────────────

/// FP8 fused gate+up GEMV for batch=2 (MTP K=2 verify).
/// Grid: (ceil(N/8), 2*(top_k+1), 2)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_fp8_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gp_w: DevicePtr,
    gp_s: DevicePtr,
    gate_out: DevicePtr,
    up_w: DevicePtr,
    up_s: DevicePtr,
    up_out: DevicePtr,
    indices: DevicePtr,
    sh_gate: &Fp8Weight,
    sh_gate_out: DevicePtr,
    sh_up: &Fp8Weight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 2 * (top_k + 1), 2])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gp_w)
        .arg_ptr(gp_s)
        .arg_ptr(gate_out)
        .arg_ptr(up_w)
        .arg_ptr(up_s)
        .arg_ptr(up_out)
        .arg_ptr(indices)
        .arg_ptr(sh_gate.weight)
        .arg_ptr(sh_gate.row_scale)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up.weight)
        .arg_ptr(sh_up.row_scale)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// FP8 fused SiLU+down GEMV for batch=2.
/// Grid: (ceil(N/8), 2*(top_k+1), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_fp8_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    dp_w: DevicePtr,
    dp_s: DevicePtr,
    output: DevicePtr,
    indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down: &Fp8Weight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 2 * (top_k + 1), 1])
        .block([128, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(dp_w)
        .arg_ptr(dp_s)
        .arg_ptr(output)
        .arg_ptr(indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down.weight)
        .arg_ptr(sh_down.row_scale)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// FP8 fused gate+up GEMV for batch=3 (MTP K=3 verify).
/// Grid: (ceil(N/8), 3*(top_k+1), 2)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_fp8_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gp_w: DevicePtr,
    gp_s: DevicePtr,
    gate_out: DevicePtr,
    up_w: DevicePtr,
    up_s: DevicePtr,
    up_out: DevicePtr,
    indices: DevicePtr,
    sh_gate: &Fp8Weight,
    sh_gate_out: DevicePtr,
    sh_up: &Fp8Weight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 3 * (top_k + 1), 2])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gp_w)
        .arg_ptr(gp_s)
        .arg_ptr(gate_out)
        .arg_ptr(up_w)
        .arg_ptr(up_s)
        .arg_ptr(up_out)
        .arg_ptr(indices)
        .arg_ptr(sh_gate.weight)
        .arg_ptr(sh_gate.row_scale)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up.weight)
        .arg_ptr(sh_up.row_scale)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// FP8 fused SiLU+down GEMV for batch=3.
/// Grid: (ceil(N/8), 3*(top_k+1), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_fp8_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    dp_w: DevicePtr,
    dp_s: DevicePtr,
    output: DevicePtr,
    indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down: &Fp8Weight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 3 * (top_k + 1), 1])
        .block([128, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(dp_w)
        .arg_ptr(dp_s)
        .arg_ptr(output)
        .arg_ptr(indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down.weight)
        .arg_ptr(sh_down.row_scale)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// Batched per-expert uint8 transpose for MoE down_proj relayout.
///
/// Reads per-expert source pointers from `src_ptrs` and writes per-expert
/// transposed `[cols, rows]` blocks via `dst_ptrs`. Both tables hold one
/// device pointer per global expert; NULL entries (EP-remote experts)
/// cause the kernel to exit early at block level.
///
/// Grid: (ceil(cols/32), ceil(rows/32), num_experts)  Block: (32, 8)
#[allow(clippy::too_many_arguments)]
pub fn moe_transpose_u8_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src_ptrs: DevicePtr,
    dst_ptrs: DevicePtr,
    rows: u32,
    cols: u32,
    num_experts: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(cols, 32), div_ceil(rows, 32), num_experts])
        .block([32, 8, 1])
        .arg_ptr(src_ptrs)
        .arg_ptr(dst_ptrs)
        .arg_u32(rows)
        .arg_u32(cols)
        .launch(stream)
}

// ─────────────────────────────────────────────────────────────────────
// Phase 8a — Transposed-layout decode MoE kernel bindings.
//
// Match the kernels in `kernels/gb10/common/moe_shared_expert_fused*_t.cu`.
// Same arg order as the non-transposed counterparts; grid changes to
// `ceil(N/128)` (each thread = one output position, lanes coalesced).
// Pointer-table args point at TRANSPOSED weights `[K/2, N]` (NVFP4) or
// `[K, N]` (FP8). Shared-expert direct-pointer args likewise. Callers
// MUST pass the transposed buffers — there is no runtime layout flag.
// ─────────────────────────────────────────────────────────────────────

// Block size for the transposed-layout decode kernels. 32 = one warp per
// block — more blocks per silu_down/gate_up call → higher SM occupancy on
// GB10 (only 25 SMs, so larger blocks under-utilise). Each thread owns one
// output regardless of block size.
pub(crate) const T_BLOCK: u32 = 32;

/// NVFP4 fused gate+up GEMV (transposed weight). Single-token decode.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_packed_t_ptrs: DevicePtr,
    gate_scale_t_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_out: DevicePtr,
    up_packed_t_ptrs: DevicePtr,
    up_scale_t_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_t: &QuantizedWeight,
    sh_gate_out: DevicePtr,
    sh_up_t: &QuantizedWeight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, T_BLOCK), top_k + 1, 2])
        .block([T_BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_packed_t_ptrs)
        .arg_ptr(gate_scale_t_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(gate_out)
        .arg_ptr(up_packed_t_ptrs)
        .arg_ptr(up_scale_t_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_t.weight)
        .arg_ptr(sh_gate_t.weight_scale)
        .arg_f32(sh_gate_t.weight_scale_2)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up_t.weight)
        .arg_ptr(sh_up_t.weight_scale)
        .arg_f32(sh_up_t.weight_scale_2)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// Outputs each thread owns (adjacent in `n`) in the split-K decode kernels.
///
/// Measured on GB10 at V4 decode shapes (`moe_unified_t_v4_microtest`): the
/// 1-byte-per-lane load in the kernels above requests 32 bytes per warp per K
/// iteration and is pinned near 130 GB/s no matter how many warps are resident
/// — 4× the warps moved it only 128 → 136 GB/s. Widening to 64 bytes (VEC=2)
/// lifts that ceiling but costs half the warps; 128 bytes (VEC=4) costs three
/// quarters and lands slower at production top_k=6. VEC=2 with the warps put
/// back by split-K measured 197 GB/s, or 32.5 → 20.6 ms/token of MoE.
pub(crate) const T_SPLIT_VEC: u32 = 2;

/// Bytes of f32 partial scratch the split-K decode path needs, given the split
/// factor and the two projection widths. Mirrors `moe_splitk_partials` in
/// `spark-runtime`'s buffer sizing; the dispatch checks the buffer against it.
pub fn moe_splitk_partial_bytes(split: u32, inter: u32, h: u32, top_k: u32) -> usize {
    let slots = (top_k + 1) as usize;
    split as usize * slots * 4 * (2 * inter as usize + h as usize)
}

/// Byte offset of the down-projection region within the split-K partial buffer.
pub fn moe_splitk_down_offset(split: u32, inter: u32, top_k: u32) -> usize {
    2 * split as usize * (top_k + 1) as usize * inter as usize * 4
}

/// Rows of f32 partials the multi-row (`_m`) split-K decode writes: the routed
/// slots in flat order, then one shared-expert row per token.
pub fn moe_splitk_m_rows(top_k: u32, num_tokens: u32) -> u32 {
    num_tokens * top_k + num_tokens
}

/// `moe_splitk_partial_bytes` for the multi-row path. Same layout, `rows`
/// instead of `top_k + 1` — the m=1 case is byte-identical to the single-row
/// sizing, so one buffer serves both.
pub fn moe_splitk_m_partial_bytes(
    split: u32,
    inter: u32,
    h: u32,
    top_k: u32,
    num_tokens: u32,
) -> usize {
    let rows = moe_splitk_m_rows(top_k, num_tokens) as usize;
    split as usize * rows * 4 * (2 * inter as usize + h as usize)
}

/// Byte offset of the down region in the multi-row partial buffer.
pub fn moe_splitk_m_down_offset(split: u32, inter: u32, top_k: u32, num_tokens: u32) -> usize {
    2 * split as usize * moe_splitk_m_rows(top_k, num_tokens) as usize * inter as usize * 4
}

/// Multi-row split-K fused gate+up GEMV — one launch covers `num_tokens`
/// candidate rows, with slots routed to the same expert collapsed onto a single
/// leader block (see the `_m` kernels in `moe_shared_expert_fused_t.cu`).
///
/// `kernel` must be an `_m{MROW}v{T_SPLIT_VEC}s{split}` entry point whose MROW
/// is >= `num_tokens`; a smaller MROW silently drops rows, so the caller checks.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_t_splitk_m(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    finalize: KernelHandle,
    input: DevicePtr,
    gate_packed_t_ptrs: DevicePtr,
    gate_scale_t_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_out: DevicePtr,
    up_packed_t_ptrs: DevicePtr,
    up_scale_t_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_t: &QuantizedWeight,
    sh_gate_out: DevicePtr,
    sh_up_t: &QuantizedWeight,
    sh_up_out: DevicePtr,
    partials: DevicePtr,
    split: u32,
    n: u32,
    k: u32,
    top_k: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    let total_routed = num_tokens * top_k;
    KernelLaunch::new(gpu, kernel)
        // grid.y is total_routed + 1, not + num_tokens: one block-set computes
        // the shared projection for every row.
        .grid([n / (T_BLOCK * T_SPLIT_VEC), total_routed + 1, 2 * split])
        .block([T_BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_packed_t_ptrs)
        .arg_ptr(gate_scale_t_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(gate_out)
        .arg_ptr(up_packed_t_ptrs)
        .arg_ptr(up_scale_t_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_t.weight)
        .arg_ptr(sh_gate_t.weight_scale)
        .arg_f32(sh_gate_t.weight_scale_2)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up_t.weight)
        .arg_ptr(sh_up_t.weight_scale)
        .arg_f32(sh_up_t.weight_scale_2)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .arg_u32(num_tokens)
        .arg_ptr(partials)
        .launch(stream)?;
    KernelLaunch::new(gpu, finalize)
        .grid([
            div_ceil(n, T_BLOCK),
            moe_splitk_m_rows(top_k, num_tokens),
            2,
        ])
        .block([T_BLOCK, 1, 1])
        .arg_ptr(partials)
        .arg_ptr(gate_out)
        .arg_ptr(sh_gate_out)
        .arg_ptr(up_out)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(total_routed)
        .arg_u32(num_tokens)
        .arg_u32(split)
        .launch(stream)
}

/// Multi-row split-K fused SiLU+down GEMV + finalize (see the gate+up twin).
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_t_splitk_m(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    finalize: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    packed_t_ptrs: DevicePtr,
    scale_t_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down_t: &QuantizedWeight,
    sh_down_out: DevicePtr,
    partials: DevicePtr,
    split: u32,
    n: u32,
    k: u32,
    top_k: u32,
    num_tokens: u32,
    mrow: u32,
    stream: u64,
) -> Result<()> {
    // One `s_act` k-slice per row the leader may gather — MROW, not
    // num_tokens: the kernel indexes slices by the compile-time MROW stride.
    let smem_bytes = mrow * (k as usize * std::mem::size_of::<f32>()) as u32 / split;
    let total_routed = num_tokens * top_k;
    KernelLaunch::new(gpu, kernel)
        .grid([n / (T_BLOCK * T_SPLIT_VEC), total_routed + 1, split])
        .block([T_BLOCK, 1, 1])
        .shared_mem(smem_bytes)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(packed_t_ptrs)
        .arg_ptr(scale_t_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down_t.weight)
        .arg_ptr(sh_down_t.weight_scale)
        .arg_f32(sh_down_t.weight_scale_2)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .arg_u32(num_tokens)
        .arg_ptr(partials)
        .launch(stream)?;
    KernelLaunch::new(gpu, finalize)
        .grid([
            div_ceil(n, T_BLOCK),
            moe_splitk_m_rows(top_k, num_tokens),
            1,
        ])
        .block([T_BLOCK, 1, 1])
        .arg_ptr(partials)
        .arg_ptr(output)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(total_routed)
        .arg_u32(num_tokens)
        .arg_u32(split)
        .launch(stream)
}

/// Split-K fused gate+up GEMV (transposed weight) + partial finalize.
///
/// `kernel` must be a `_v{T_SPLIT_VEC}s{split}` entry point and `finalize` the
/// matching `moe_gate_up_partial_finalize`; the split factor is baked into the
/// kernel at compile time and only the finalize reads it at runtime.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_t_splitk(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    finalize: KernelHandle,
    input: DevicePtr,
    gate_packed_t_ptrs: DevicePtr,
    gate_scale_t_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_out: DevicePtr,
    up_packed_t_ptrs: DevicePtr,
    up_scale_t_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_t: &QuantizedWeight,
    sh_gate_out: DevicePtr,
    sh_up_t: &QuantizedWeight,
    sh_up_out: DevicePtr,
    partials: DevicePtr,
    split: u32,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n / (T_BLOCK * T_SPLIT_VEC), top_k + 1, 2 * split])
        .block([T_BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_packed_t_ptrs)
        .arg_ptr(gate_scale_t_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(gate_out)
        .arg_ptr(up_packed_t_ptrs)
        .arg_ptr(up_scale_t_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_t.weight)
        .arg_ptr(sh_gate_t.weight_scale)
        .arg_f32(sh_gate_t.weight_scale_2)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up_t.weight)
        .arg_ptr(sh_up_t.weight_scale)
        .arg_f32(sh_up_t.weight_scale_2)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .arg_ptr(partials)
        .launch(stream)?;
    KernelLaunch::new(gpu, finalize)
        .grid([div_ceil(n, T_BLOCK), top_k + 1, 2])
        .block([T_BLOCK, 1, 1])
        .arg_ptr(partials)
        .arg_ptr(gate_out)
        .arg_ptr(sh_gate_out)
        .arg_ptr(up_out)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(top_k)
        .arg_u32(split)
        .launch(stream)
}

/// Split-K fused SiLU+down GEMV (transposed weight) + partial finalize.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_t_splitk(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    finalize: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    packed_t_ptrs: DevicePtr,
    scale_t_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down_t: &QuantizedWeight,
    sh_down_out: DevicePtr,
    partials: DevicePtr,
    split: u32,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    // `s_act` now covers only this block's k slice, so the per-block shared
    // memory drops by `split` — at K=2048 that is 8 KB → 2 KB, which is what
    // was capping blocks-per-SM on the down projection.
    let smem_bytes = (k as usize * std::mem::size_of::<f32>()) as u32 / split;
    KernelLaunch::new(gpu, kernel)
        .grid([n / (T_BLOCK * T_SPLIT_VEC), top_k + 1, split])
        .block([T_BLOCK, 1, 1])
        .shared_mem(smem_bytes)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(packed_t_ptrs)
        .arg_ptr(scale_t_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down_t.weight)
        .arg_ptr(sh_down_t.weight_scale)
        .arg_f32(sh_down_t.weight_scale_2)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .arg_ptr(partials)
        .launch(stream)?;
    KernelLaunch::new(gpu, finalize)
        .grid([div_ceil(n, T_BLOCK), top_k + 1, 1])
        .block([T_BLOCK, 1, 1])
        .arg_ptr(partials)
        .arg_ptr(output)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(top_k)
        .arg_u32(split)
        .launch(stream)
}

/// NVFP4 fused SiLU+down GEMV (transposed weight). Single-token decode.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    packed_t_ptrs: DevicePtr,
    scale_t_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down_t: &QuantizedWeight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    let smem_bytes = (k as usize * std::mem::size_of::<f32>()) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, T_BLOCK), top_k + 1, 1])
        .block([T_BLOCK, 1, 1])
        .shared_mem(smem_bytes)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(packed_t_ptrs)
        .arg_ptr(scale_t_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down_t.weight)
        .arg_ptr(sh_down_t.weight_scale)
        .arg_f32(sh_down_t.weight_scale_2)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}
