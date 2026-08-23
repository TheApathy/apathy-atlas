// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// RoPE: apply rotary position embeddings to Q and K in-place.
///
/// Kernel: `rope_forward(Q, K, positions, seq_len, num_q_heads,
///          num_kv_heads, head_dim, rotary_dim, theta)`
/// Grid: (num_q_heads + num_kv_heads, ceil(seq_len/4), 1)
/// Block: (128, 1, 1)
///
/// `positions` must be a device pointer to a `u32[seq_len]` array.
pub fn rope(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    positions: DevicePtr,
    seq_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
    stream: u64,
) -> Result<()> {
    assert!(
        rotary_dim > 0,
        "rope: rotary_dim=0, nq={num_q_heads} nkv={num_kv_heads} hd={head_dim}"
    );
    let half_rot = (rotary_dim / 2).max(1);
    let pos_per_block = (128 / half_rot).max(1);
    let seq_blocks = div_ceil(seq_len, pos_per_block);
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads + num_kv_heads, seq_blocks, 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(positions)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_f32(theta)
        .launch(stream)
}

/// Proportional RoPE (Gemma-4 full-attention layers).
///
/// Rotation pairs are (i, i + head_dim/2) for i in [0, rope_angles).
/// Frequency denominator is `head_dim` (not rotary_dim).
#[allow(clippy::too_many_arguments)]
pub fn rope_proportional(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    positions: DevicePtr,
    seq_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rope_angles: u32,
    theta: f32,
    stream: u64,
) -> Result<()> {
    assert!(rope_angles > 0, "rope_proportional: rope_angles=0");
    let pairs_per_pos = rope_angles.max(1);
    let pos_per_block = (128 / pairs_per_pos).max(1);
    let seq_blocks = div_ceil(seq_len, pos_per_block);
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads + num_kv_heads, seq_blocks, 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(positions)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rope_angles)
        .arg_f32(theta)
        .launch(stream)
}

/// MRoPE (interleaved multi-modal rotary) for Qwen3.6.
///
/// Applies rotary embedding using three separate position-ID streams
/// (`pos_t`, `pos_h`, `pos_w`). Each rotary pair is owned by one of the
/// three sections, chosen by `pair_idx % 3` (round-robin). For text-only
/// serving pass the same pointer for all three streams — the result is
/// bit-identical to scalar RoPE.
///
/// Kernel: `rope_forward_mrope_interleaved`
#[allow(clippy::too_many_arguments)]
pub fn rope_mrope_interleaved(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    pos_t: DevicePtr,
    pos_h: DevicePtr,
    pos_w: DevicePtr,
    seq_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
    stream: u64,
) -> Result<()> {
    assert!(rotary_dim > 0, "rope_mrope_interleaved: rotary_dim=0");
    let half_rot = (rotary_dim / 2).max(1);
    let pos_per_block = (128 / half_rot).max(1);
    let seq_blocks = div_ceil(seq_len, pos_per_block);
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads + num_kv_heads, seq_blocks, 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(pos_t)
        .arg_ptr(pos_h)
        .arg_ptr(pos_w)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_f32(theta)
        .launch(stream)
}

/// Batched per-token-strided RoPE for the multi-sequence K=3 verify path.
///
/// Replaces N back-to-back `rope_forward` launches (each with `seq_len=1`)
/// with a single launch that handles all `num_tokens` slots inside a
/// per-token-strided `qkv_buf`. Output is bit-identical to the sequential
/// version when called with the same inputs.
///
/// Kernel: `rope_forward_strided_b3(qkv_base, positions, num_tokens,
///          qkv_stride_bf16, k_offset_bf16, num_q_heads, num_kv_heads,
///          head_dim, rotary_dim, theta)`
/// Grid: (num_q_heads + num_kv_heads, num_tokens, 1)
/// Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn rope_strided_b3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    qkv_base: DevicePtr,
    positions: DevicePtr,
    num_tokens: u32,
    qkv_stride_bf16: u32,
    k_offset_bf16: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
    stream: u64,
) -> Result<()> {
    assert!(rotary_dim > 0, "rope_strided_b3: rotary_dim=0");
    assert!(num_tokens > 0, "rope_strided_b3: num_tokens=0");
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads + num_kv_heads, num_tokens, 1])
        .block([128, 1, 1])
        .arg_ptr(qkv_base)
        .arg_ptr(positions)
        .arg_u32(num_tokens)
        .arg_u32(qkv_stride_bf16)
        .arg_u32(k_offset_bf16)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_f32(theta)
        .launch(stream)
}

/// RoPE with precomputed YaRN inv_freq table (Mistral Small 4).
/// The kernel reads frequencies from the table instead of computing from theta.
#[allow(clippy::too_many_arguments)]
pub fn rope_yarn(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    positions: DevicePtr,
    seq_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    inv_freq: DevicePtr,
    theta: f32,
    stream: u64,
) -> Result<()> {
    assert!(
        rotary_dim > 0,
        "rope: rotary_dim=0, nq={num_q_heads} nkv={num_kv_heads} hd={head_dim}"
    );
    let half_rot = (rotary_dim / 2).max(1);
    let pos_per_block = (128 / half_rot).max(1);
    let seq_blocks = div_ceil(seq_len, pos_per_block);
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads + num_kv_heads, seq_blocks, 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(positions)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_ptr(inv_freq)
        .arg_f32(theta)
        .launch(stream)
}

/// DFlash YaRN with HF-compatible cos/sin amplitude scaling.
///
/// `attention_factor` is 1.0 for vanilla RoPE. For YaRN, Transformers uses
/// an explicit config value when present and otherwise derives
/// `1 + 0.1 * ln(factor)`. A separate kernel symbol keeps the ABI of the
/// target-model `rope_forward_yarn` path unchanged.
#[allow(clippy::too_many_arguments)]
pub fn rope_yarn_scaled(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    positions: DevicePtr,
    seq_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    inv_freq: DevicePtr,
    theta: f32,
    attention_factor: f32,
    stream: u64,
) -> Result<()> {
    assert!(
        rotary_dim > 0,
        "rope: rotary_dim=0, nq={num_q_heads} nkv={num_kv_heads} hd={head_dim}"
    );
    let half_rot = (rotary_dim / 2).max(1);
    let pos_per_block = (128 / half_rot).max(1);
    let seq_blocks = div_ceil(seq_len, pos_per_block);
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads + num_kv_heads, seq_blocks, 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(positions)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_ptr(inv_freq)
        .arg_f32(theta)
        .arg_f32(attention_factor)
        .launch(stream)
}
