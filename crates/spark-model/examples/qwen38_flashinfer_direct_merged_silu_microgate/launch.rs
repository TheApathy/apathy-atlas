// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail};
use spark_model::layers::ops;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelLaunch};

use super::{K, Kernels, Plan, SCALE2};

#[allow(clippy::too_many_arguments)]
pub(super) fn launch(
    candidate: bool,
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: Kernels,
    plan: Plan,
    merged: DevicePtr,
    gate: DevicePtr,
    up: DevicePtr,
    packed: DevicePtr,
    scales: DevicePtr,
) -> Result<()> {
    if candidate {
        ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
            gpu,
            kernels.candidate,
            merged,
            packed,
            scales,
            SCALE2,
            plan.m,
            K,
            stream,
        )
    } else {
        KernelLaunch::new(gpu, kernels.split)
            .grid([plan.m, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(merged)
            .arg_ptr(gate)
            .arg_ptr(up)
            .arg_u32(plan.m)
            .launch(stream)?;
        ops::quantize_silu_mul_bf16_to_nvfp4_atlas_128x4(
            gpu,
            kernels.parent,
            gate,
            up,
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

pub(super) fn exact_split(merged: &[u8], gate: &[u8], up: &[u8], plan: Plan) -> Result<()> {
    let side = K as usize * 2;
    for row in 0..plan.m as usize {
        let merged_row = &merged[row * side * 2..(row + 1) * side * 2];
        equal(
            "split gate",
            &merged_row[..side],
            &gate[row * side..(row + 1) * side],
        )?;
        equal(
            "split up",
            &merged_row[side..],
            &up[row * side..(row + 1) * side],
        )?;
    }
    Ok(())
}

pub(super) fn split_payloads(merged: &[u8], plan: Plan) -> Result<(Vec<u8>, Vec<u8>)> {
    let side = K as usize * 2;
    let mut gate = Vec::new();
    let mut up = Vec::new();
    gate.try_reserve_exact(plan.elements * 2)?;
    up.try_reserve_exact(plan.elements * 2)?;
    for row in merged.chunks_exact(side * 2) {
        gate.extend_from_slice(&row[..side]);
        up.extend_from_slice(&row[side..]);
    }
    Ok((gate, up))
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
