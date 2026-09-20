// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail};
use spark_model::layers::ops;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{K, Kernels, Plan, SCALE2};

#[allow(clippy::too_many_arguments)]
pub(super) fn launch(
    route: bool,
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: Kernels,
    plan: Plan,
    gate: DevicePtr,
    up: DevicePtr,
    temporary: DevicePtr,
    packed: DevicePtr,
    scales: DevicePtr,
) -> Result<()> {
    if route {
        ops::quantize_silu_mul_bf16_to_nvfp4_atlas_128x4(
            gpu,
            kernels.fused,
            gate,
            up,
            packed,
            scales,
            SCALE2,
            plan.m,
            K,
            stream,
        )
    } else {
        ops::silu_mul(
            gpu,
            kernels.silu,
            gate,
            up,
            temporary,
            u32::try_from(plan.elements)?,
            stream,
        )?;
        ops::quantize_bf16_to_nvfp4_atlas_128x4(
            gpu,
            kernels.parent,
            temporary,
            packed,
            scales,
            SCALE2,
            plan.m,
            K,
            stream,
        )
    }
}

pub(super) fn equal(label: &str, left: &[u8], right: &[u8]) -> Result<()> {
    if left == right {
        return Ok(());
    }
    let index = left
        .iter()
        .zip(right)
        .position(|(lhs, rhs)| lhs != rhs)
        .unwrap_or(left.len().min(right.len()));
    bail!(
        "{label} mismatch at byte {index}, lengths {}/{}",
        left.len(),
        right.len()
    )
}

pub(super) fn hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    })
}

pub(super) fn median(values: &[f64]) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[ordered.len() / 2]
}

pub(super) fn p90(values: &[f64]) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[(ordered.len() * 9).div_ceil(10) - 1]
}
