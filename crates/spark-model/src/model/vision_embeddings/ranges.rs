// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

/// Validate complete half-open ranges before any address-offset construction.
pub(super) fn disjoint(
    source: DevicePtr,
    source_bytes: usize,
    destination: DevicePtr,
    destination_bytes: usize,
) -> Result<()> {
    ensure!(
        source != DevicePtr::NULL && destination != DevicePtr::NULL,
        "null vision embedding buffer"
    );
    let source_end = source
        .0
        .checked_add(u64::try_from(source_bytes)?)
        .ok_or_else(|| anyhow::anyhow!("vision source address overflow"))?;
    let destination_end = destination
        .0
        .checked_add(u64::try_from(destination_bytes)?)
        .ok_or_else(|| anyhow::anyhow!("vision destination address overflow"))?;
    ensure!(
        source_end <= destination.0 || destination_end <= source.0,
        "vision source and destination ranges overlap"
    );
    Ok(())
}
