// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Fused SiLU activation: output = SiLU(gate) * up.
///
/// Kernel: `silu_mul_separate(gate, up, output, n)`
/// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
pub fn silu_mul(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    output: DevicePtr,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(output)
        .arg_u32(num_elements)
        .launch(stream)
}

/// L2 normalization (in-place): `data[i] = data[i] / sqrt(sum(data^2) + eps)`.
///
/// Applied per head: data is [num_heads, head_dim], each head normalized independently.
/// Required for Gated Delta Net Q/K normalization (use_qk_l2norm_in_kernel=True).
///
/// Kernel: `l2_norm_bf16(data, head_dim, eps)`
/// Grid: (num_heads, 1, 1)  Block: (min(head_dim, 1024), 1, 1)
pub fn l2_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    data: DevicePtr,
    num_heads: u32,
    head_dim: u32,
    eps: f32,
    num_tokens: u32,
    stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_heads, num_tokens, 1])
        .block([head_dim.min(1024), 1, 1])
        .arg_ptr(data)
        .arg_u32(head_dim)
        .arg_f32(eps)
        .arg_u32(stride)
        .launch(stream)
}

/// Element-wise sigmoid gate: `output[i] = input[i] * sigmoid(gate[i])`.
///
/// Used for gated attention in Qwen3: attn_output = attn_output * sigmoid(q_gate).
///
/// Kernel: `sigmoid_gate_mul(input, gate, output, n)`
/// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
pub fn sigmoid_gate_mul(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    output: DevicePtr,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(output)
        .arg_u32(num_elements)
        .launch(stream)
}

/// BF16 residual add: `residual[i] += src[i]` (in-place).
///
/// Kernel: `bf16_residual_add(residual, src, n)`
/// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
pub fn residual_add(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    residual: DevicePtr,
    src: DevicePtr,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(residual)
        .arg_ptr(src)
        .arg_u32(num_elements)
        .launch(stream)
}

/// Threads per row for [`direction_project`]. Must be a power of two: the
/// block reduction halves the stride each round.
const DIRECTION_PROJECT_BLOCK: u32 = 256;

/// Directional projection on the residual stream, in place:
///
/// ```text
/// h' = h - alpha * (h . d_hat) * d_hat
/// ```
///
/// Removes the component of each row that lies along the unit direction
/// `d_hat`. This is the runtime form of the rank-1 weight edit
/// `dW = -alpha * d_hat (d_hat^T W)`, so a behavioural modification can ship
/// as `hidden_size` floats applied at serve time rather than as a
/// redistributed checkpoint.
///
/// `d_hat` MUST be L2-normalised — the kernel does not normalise it, and an
/// unnormalised direction scales the subtraction by `||d||^2` instead of
/// `||d||`. Callers should normalise once at load time.
///
/// The operation is self-limiting: a row orthogonal to `d_hat` has zero
/// subtracted and is returned bit-identical.
///
/// Kernel: `bf16_direction_project(hidden, d_hat, alpha, hidden_size)`
/// Grid: (rows, 1, 1)  Block: (256, 1, 1)  Shared: 256 * 4 bytes
pub fn direction_project(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    d_hat: DevicePtr,
    alpha: f32,
    rows: u32,
    hidden_size: u32,
    stream: u64,
) -> Result<()> {
    if rows == 0 || hidden_size == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([DIRECTION_PROJECT_BLOCK, 1, 1])
        .shared_mem(DIRECTION_PROJECT_BLOCK * 4)
        .arg_ptr(hidden)
        .arg_ptr(d_hat)
        .arg_f32(alpha)
        .arg_u32(hidden_size)
        .launch(stream)
}

/// BF16 → FP32 conversion: `dst[i] = (float)src[i]`.
///
/// Kernel: `bf16_to_f32(src, dst, n)`
/// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
pub fn bf16_to_f32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(num_elements)
        .launch(stream)
}

/// BF16 scaled accumulate: `output[i] += scale * src[i]`.
///
/// Kernel: `bf16_scaled_add(output, src, scale, n)`
/// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
pub fn scaled_add(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    src: DevicePtr,
    scale: f32,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(src)
        .arg_f32(scale)
        .arg_u32(num_elements)
        .launch(stream)
}

/// Sigmoid-gated blend: output = output + sigmoid_gate * src.
///
/// Kernel: `bf16_sigmoid_blend(output, src, sigmoid_gate, n)`
/// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
pub fn sigmoid_blend(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    src: DevicePtr,
    sigmoid_gate: f32,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(src)
        .arg_f32(sigmoid_gate)
        .arg_u32(num_elements)
        .launch(stream)
}

/// GPU-side token recommit for DeepLoop multi-pass (no inter-pass D2H).
///
/// Caller must D2D-copy `draft_tokens_dev[0..gamma_eff*4]` to `staged`
/// (= topk_tokens_dev) before this call. This kernel then writes:
///   `dst[0..n_attn] = [0]*eff_ctx + [last_token] + staged[0..gamma_eff]`
/// entirely on-stream, so async drafter passes need no host barrier.
///
/// Kernel: `dflash_token_recommit(dst, staged, last_token, eff_ctx, n_attn)`
/// Grid: (ceil(n_attn / 256), 1, 1)  Block: (256, 1, 1)
pub fn token_recommit(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    dst: DevicePtr,
    staged: DevicePtr,
    last_token: u32,
    eff_ctx: u32,
    n_attn: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_attn, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(dst)
        .arg_ptr(staged)
        .arg_i32(last_token as i32)
        .arg_u32(eff_ctx)
        .arg_u32(n_attn)
        .launch(stream)
}

// ── SSM Preprocessing ─────────────────────────────────────────────
