// SPDX-License-Identifier: AGPL-3.0-only
//! Provenance-bound transform-cache slots for official packed Qwen4 MTP experts.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::weights::{WeightDtype, WeightTensor};

use super::{DenseWeight, QuantizedWeight, quantize_to_nvfp4, quantize_to_nvfp4_cached};
use address::checked_source_address;

#[path = "qwen4_mtp_packed_cache_address.rs"]
mod address;

const GATE_UP_NAME: &str = "mtp.layers.0.mlp.experts.gate_up_proj";
const DOWN_NAME: &str = "mtp.layers.0.mlp.experts.down_proj";
const TARGET_QUANT: &str = "nvfp4-e2m1-group16-global-absmax-v1";
const TARGET_KERNELS: &str = "quantize_nvfp4/nvfp4_global_absmax+quantize_bf16_to_nvfp4";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PackedMtpProjection {
    Gate,
    Up,
    Down,
}

impl PackedMtpProjection {
    const fn name(self) -> &'static str {
        match self {
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        }
    }
}

fn checked_slice_identity(
    source_name: &str,
    source_shape: &[usize],
    source_dtype: WeightDtype,
    expert: usize,
    projection: PackedMtpProjection,
    offset: usize,
    n: usize,
    k: usize,
) -> Result<(usize, usize)> {
    ensure!(
        source_dtype == WeightDtype::BF16,
        "packed MTP cache source must be BF16"
    );
    ensure!(
        source_shape.len() == 3,
        "packed MTP cache source must be rank 3"
    );
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(16),
        "invalid packed MTP cache slice geometry"
    );
    let experts = source_shape[0];
    ensure!(
        expert < experts,
        "packed MTP cache expert index is out of range"
    );
    let elements = n
        .checked_mul(k)
        .ok_or_else(|| anyhow::anyhow!("packed MTP cache slice element overflow"))?;
    ensure!(
        elements <= u32::MAX as usize,
        "packed MTP cache slice exceeds kernel ABI"
    );
    let slice_bytes = elements
        .checked_mul(WeightDtype::BF16.byte_size())
        .ok_or_else(|| anyhow::anyhow!("packed MTP cache slice byte overflow"))?;
    let expected_offset = match projection {
        PackedMtpProjection::Gate | PackedMtpProjection::Up => {
            ensure!(
                source_name == GATE_UP_NAME,
                "packed MTP gate/up source identity mismatch"
            );
            let rows = n
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("packed MTP gate/up row overflow"))?;
            ensure!(
                source_shape == [experts, rows, k],
                "packed MTP gate/up shape mismatch"
            );
            let base =
                expert
                    .checked_mul(slice_bytes.checked_mul(2).ok_or_else(|| {
                        anyhow::anyhow!("packed MTP gate/up expert stride overflow")
                    })?)
                    .ok_or_else(|| anyhow::anyhow!("packed MTP gate/up offset overflow"))?;
            if projection == PackedMtpProjection::Up {
                base.checked_add(slice_bytes)
                    .ok_or_else(|| anyhow::anyhow!("packed MTP up offset overflow"))?
            } else {
                base
            }
        }
        PackedMtpProjection::Down => {
            ensure!(
                source_name == DOWN_NAME,
                "packed MTP down source identity mismatch"
            );
            ensure!(
                source_shape == [experts, n, k],
                "packed MTP down shape mismatch"
            );
            expert
                .checked_mul(slice_bytes)
                .ok_or_else(|| anyhow::anyhow!("packed MTP down offset overflow"))?
        }
    };
    ensure!(
        offset == expected_offset,
        "packed MTP cache slice offset mismatch"
    );
    let source_bytes = source_shape
        .iter()
        .try_fold(WeightDtype::BF16.byte_size(), |bytes, &dim| {
            bytes.checked_mul(dim)
        })
        .ok_or_else(|| anyhow::anyhow!("packed MTP source byte extent overflow"))?;
    ensure!(
        offset
            .checked_add(slice_bytes)
            .is_some_and(|end| end <= source_bytes),
        "packed MTP cache slice exceeds source"
    );
    Ok((experts, slice_bytes))
}

pub(crate) fn packed_mtp_nvfp4_cache_slot(
    source_name: &str,
    source_shape: &[usize],
    source_dtype: WeightDtype,
    expert: usize,
    projection: PackedMtpProjection,
    offset: usize,
    n: usize,
    k: usize,
) -> Result<String> {
    let (experts, slice_bytes) = checked_slice_identity(
        source_name,
        source_shape,
        source_dtype,
        expert,
        projection,
        offset,
        n,
        k,
    )?;
    Ok(format!(
        "qwen4-mtp-packed-cache-v1;source={source_name};source_shape={experts}x{}x{};\
         source_dtype=BF16;expert={expert};projection={};offset={offset};slice_shape={n}x{k};\
         slice_bytes={slice_bytes};target_quant={TARGET_QUANT};\
         target_model={};target_build_quant={};target_kernels={TARGET_KERNELS}",
        source_shape[1],
        source_shape[2],
        projection.name(),
        option_env!("ATLAS_TARGET_MODEL").unwrap_or("unspecified"),
        option_env!("ATLAS_TARGET_QUANT").unwrap_or("unspecified"),
    ))
}

fn packed_mtp_cache_slot_for_source(
    source_name: &str,
    source: &WeightTensor,
    expert: usize,
    projection: PackedMtpProjection,
    offset: usize,
    n: usize,
    k: usize,
) -> Result<(String, DenseWeight)> {
    let slot = packed_mtp_nvfp4_cache_slot(
        source_name,
        &source.shape,
        source.dtype,
        expert,
        projection,
        offset,
        n,
        k,
    )?;
    let weight = checked_source_address(source, offset, n, k)?;
    Ok((slot, DenseWeight { weight }))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn quantize_packed_mtp_slice(
    cache_official_mtp: bool,
    source_name: &str,
    source: &WeightTensor,
    expert: usize,
    projection: PackedMtpProjection,
    offset: usize,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    absmax_kernel: KernelHandle,
    quantize_kernel: KernelHandle,
    stream: u64,
) -> Result<QuantizedWeight> {
    if cache_official_mtp {
        checked_source_address(source, offset, n, k)?;
    }
    if !cache_official_mtp || crate::weight_loader::transform_cache::get().is_none() {
        let dense = DenseWeight {
            weight: source.ptr.offset(offset),
        };
        return quantize_to_nvfp4(&dense, n, k, gpu, absmax_kernel, quantize_kernel, stream);
    }
    let (slot, dense) =
        packed_mtp_cache_slot_for_source(source_name, source, expert, projection, offset, n, k)?;
    let slot =
        crate::weight_loader::transform_cache::bind_packed_mtp_slot_provenance(slot, source, gpu)?;
    quantize_to_nvfp4_cached(
        &dense,
        n,
        k,
        gpu,
        absmax_kernel,
        quantize_kernel,
        stream,
        &slot,
    )
}

#[cfg(test)]
#[path = "qwen4_mtp_packed_cache_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "qwen4_mtp_packed_cache_effect_tests.rs"]
mod effect_tests;
