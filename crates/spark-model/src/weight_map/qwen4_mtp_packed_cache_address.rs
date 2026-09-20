// SPDX-License-Identifier: AGPL-3.0-only
//! Checked device-span construction for packed Qwen4 MTP slices.

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::{WeightDtype, WeightTensor};

pub(super) fn checked_source_address(
    source: &WeightTensor,
    offset: usize,
    n: usize,
    k: usize,
) -> Result<DevicePtr> {
    if source.ptr.is_null() {
        bail!("packed MTP cache source pointer is null");
    }
    let slice_bytes = n
        .checked_mul(k)
        .and_then(|elements| elements.checked_mul(WeightDtype::BF16.byte_size()))
        .ok_or_else(|| anyhow::anyhow!("packed MTP cache slice address overflow"))?;
    let offset = u64::try_from(offset)
        .map_err(|_| anyhow::anyhow!("packed MTP cache slice offset exceeds device ABI"))?;
    let slice_bytes = u64::try_from(slice_bytes)
        .map_err(|_| anyhow::anyhow!("packed MTP cache slice extent exceeds device ABI"))?;
    let address = source
        .ptr
        .0
        .checked_add(offset)
        .and_then(|address| address.checked_add(slice_bytes).map(|_| address))
        .ok_or_else(|| anyhow::anyhow!("packed MTP cache source device address overflows"))?;
    Ok(DevicePtr(address))
}
