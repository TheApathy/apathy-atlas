// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for the materialized-FP32 M8/M17 exact dense-FFN path.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::{ExactFfnTier, W4A16_EXACT_FFN_OUTS_PER_BLOCK, W4a16ExactFfnKernels};
use crate::weight_map::QuantizedWeight;

const POINTWISE_THREADS: u32 = 256;

fn validate_materialized_tier(rows: u32, n: u32, k: u32) -> Result<ExactFfnTier> {
    let tier = super::validate_exact_ffn_shape(rows, n, k)?;
    ensure!(
        matches!(tier, ExactFfnTier::M8 | ExactFfnTier::M17),
        "materialized exact FFN supports only M8/M17 tiers, got rows={rows}"
    );
    Ok(tier)
}

fn dual_silu_kernel(kernels: W4a16ExactFfnKernels, tier: ExactFfnTier) -> KernelHandle {
    match tier {
        ExactFfnTier::M8 => kernels.dual_silu_f32_m8,
        ExactFfnTier::M17 => kernels.dual_silu_f32_m17,
        _ => KernelHandle(0),
    }
}

fn f32_input_kernel(kernels: W4a16ExactFfnKernels, tier: ExactFfnTier) -> KernelHandle {
    match tier {
        ExactFfnTier::M8 => kernels.f32_input_m8,
        ExactFfnTier::M17 => kernels.f32_input_m17,
        _ => KernelHandle(0),
    }
}

/// Exact M17 dual projection with BF16 rounding and direct FP32 SiLU output.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_materialize_f32_exact_m17(
    gpu: &dyn GpuBackend,
    kernels: W4a16ExactFfnKernels,
    input: DevicePtr,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    activation: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    w4a16_gemv_dual_materialize_f32_exact_m17_with(
        gpu,
        kernels,
        input,
        gate_weight,
        up_weight,
        activation,
        rows,
        n,
        k,
        stream,
        crate::layers::ops::w4a16_gemv_rt2_enabled(),
    )
}

/// Explicit-rt2 form of the fused M17 gate/up materialization, for parity runs.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_materialize_f32_exact_m17_with(
    gpu: &dyn GpuBackend,
    kernels: W4a16ExactFfnKernels,
    input: DevicePtr,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    activation: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
    use_rt2: bool,
) -> Result<()> {
    let tier = validate_materialized_tier(rows, n, k)?;
    ensure!(
        tier == ExactFfnTier::M17,
        "fused materialized exact FFN supports only M17, got rows={rows}"
    );
    ensure!(
        kernels.dual_materialize_f32_m17.0 != 0,
        "missing exact FFN kernel w4a16_gemv_dual_exact_materialize_f32_m17"
    );

    // Register-tiled substitution: kernel handle and grid divisor only. The
    // per-output arithmetic, the BF16 gate rounding boundary and the SiLU are
    // unchanged, so this is a bandwidth swap and not a routing decision.
    let rt2 = kernels.rt2_dual_materialize_f32_m17;
    let (kernel, outs_per_block) = if use_rt2 && rt2.0 != 0 {
        (rt2, W4A16_EXACT_FFN_OUTS_PER_BLOCK * 2)
    } else {
        (
            kernels.dual_materialize_f32_m17,
            W4A16_EXACT_FFN_OUTS_PER_BLOCK,
        )
    };

    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, outs_per_block), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_weight.weight)
        .arg_ptr(gate_weight.weight_scale)
        .arg_f32(gate_weight.weight_scale_2)
        .arg_ptr(up_weight.weight)
        .arg_ptr(up_weight.weight_scale)
        .arg_f32(up_weight.weight_scale_2)
        .arg_ptr(activation)
        .arg_u32(rows)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Materialize exact `SiLU(gate_bf16) * up_bf16` into row-major FP32.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_silu_f32_exact(
    gpu: &dyn GpuBackend,
    kernels: W4a16ExactFfnKernels,
    gate_output: DevicePtr,
    up_output: DevicePtr,
    activation: DevicePtr,
    rows: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let tier = validate_materialized_tier(rows, k, k)?;
    let kernel = dual_silu_kernel(kernels, tier);
    ensure!(kernel.0 != 0, "missing exact FFN materializer for {tier:?}");
    let elements = rows
        .checked_mul(k)
        .ok_or_else(|| anyhow::anyhow!("materialized exact FFN shape overflow: {rows}x{k}"))?;

    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(elements, POINTWISE_THREADS), 1, 1])
        .block([POINTWISE_THREADS, 1, 1])
        .arg_ptr(gate_output)
        .arg_ptr(up_output)
        .arg_ptr(activation)
        .arg_u32(rows)
        .arg_u32(k)
        .launch(stream)
}

/// Exact M8/M17 down projection from row-major FP32 materialized activation.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_f32_input_exact(
    gpu: &dyn GpuBackend,
    kernels: W4a16ExactFfnKernels,
    input: DevicePtr,
    down_weight: &QuantizedWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    w4a16_gemv_f32_input_exact_with(
        gpu,
        kernels,
        input,
        down_weight,
        output,
        rows,
        n,
        k,
        stream,
        crate::layers::ops::w4a16_gemv_rt2_enabled(),
    )
}

/// Same launch with the register-tiled choice passed in rather than read from
/// the environment, so a parity harness can drive both families in one process.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_f32_input_exact_with(
    gpu: &dyn GpuBackend,
    kernels: W4a16ExactFfnKernels,
    input: DevicePtr,
    down_weight: &QuantizedWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
    use_rt2: bool,
) -> Result<()> {
    let tier = validate_materialized_tier(rows, n, k)?;
    let kernel = f32_input_kernel(kernels, tier);
    ensure!(
        kernel.0 != 0,
        "missing exact FFN FP32-input kernel for {tier:?}"
    );

    let rt2 = kernels.rt2_f32_input_for_tier(tier);
    let (kernel, outs_per_block) = if use_rt2 && rt2.0 != 0 {
        (rt2, W4A16_EXACT_FFN_OUTS_PER_BLOCK * 2)
    } else {
        (kernel, W4A16_EXACT_FFN_OUTS_PER_BLOCK)
    };

    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, outs_per_block), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(down_weight.weight)
        .arg_ptr(down_weight.weight_scale)
        .arg_f32(down_weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
