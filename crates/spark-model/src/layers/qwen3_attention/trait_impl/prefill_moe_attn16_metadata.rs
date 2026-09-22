// SPDX-License-Identifier: AGPL-3.0-only

use super::DeviceKernels;
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, HostToDeviceCopy};

#[allow(clippy::too_many_arguments)]
pub(in super::super) fn expand_or_copy_metadata(
    device: Option<DeviceKernels>,
    gpu: &dyn GpuBackend,
    source_device: DevicePtr,
    expanded: DevicePtr,
    row_lengths: DevicePtr,
    final_length: DevicePtr,
    block_table: &[u32],
    tile_start: usize,
    rows: u32,
    stream: u64,
) -> Result<()> {
    if let Some(kernels) = device {
        return kernels.expand_metadata(
            gpu,
            source_device,
            expanded,
            row_lengths,
            final_length,
            block_table.len(),
            tile_start,
            rows,
            stream,
        );
    }
    let mut tables = Vec::with_capacity(rows as usize * block_table.len());
    for _ in 0..rows {
        tables.extend_from_slice(block_table);
    }
    let lengths: Vec<u32> = (1..=rows as usize)
        .map(|row| u32::try_from(tile_start + row))
        .collect::<std::result::Result<_, _>>()?;
    let table_bytes =
        unsafe { std::slice::from_raw_parts(tables.as_ptr().cast::<u8>(), tables.len() * 4) };
    let length_bytes =
        unsafe { std::slice::from_raw_parts(lengths.as_ptr().cast::<u8>(), lengths.len() * 4) };
    let final_len = u32::try_from(tile_start + rows as usize)?.to_ne_bytes();
    gpu.copy_h2d_group_on_stream(
        &[
            HostToDeviceCopy::new(table_bytes, expanded),
            HostToDeviceCopy::new(length_bytes, row_lengths),
            HostToDeviceCopy::new(&final_len, final_length),
        ],
        stream,
    )
}
