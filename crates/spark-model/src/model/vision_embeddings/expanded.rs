// SPDX-License-Identifier: AGPL-3.0-only

//! Preserve FlashNext's BF16 hyper-stream expansion while changing source ownership.

use super::{VisionEmbeddingState, ranges};
use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

impl VisionEmbeddingState {
    pub fn splice_expanded<F>(
        &mut self,
        tokens: &[u32],
        pad: u32,
        start: usize,
        len: usize,
        dst: DevicePtr,
        hidden: usize,
        hc_count: usize,
        stream: u64,
        mut expand: F,
    ) -> Result<()>
    where
        F: FnMut(DevicePtr, DevicePtr) -> Result<()>,
    {
        let plan = self.plan(tokens, pad, start, len)?;
        if plan.copies.is_empty() {
            return Ok(());
        }
        let (layout, _) = self
            .published
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing vision aggregate"))?;
        ensure!(
            hidden == layout.geometry.hidden && hc_count > 0,
            "vision hyper-stream geometry does not match hidden width"
        );
        let source_stride = hidden
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("vision row size overflow"))?;
        let destination_stride = source_stride
            .checked_mul(hc_count)
            .ok_or_else(|| anyhow::anyhow!("vision hyper-stream row overflow"))?;
        let destination_bytes = len
            .checked_mul(destination_stride)
            .ok_or_else(|| anyhow::anyhow!("vision hyper-stream buffer overflow"))?;
        ranges::disjoint(self.buffer, self.capacity, dst, destination_bytes)?;
        for run in &plan.copies {
            ensure!(
                run.source_row + run.rows <= layout.rows && run.dest_row + run.rows <= len,
                "vision hyper-stream splice exceeds valid rows"
            );
        }
        self.streams.insert(stream);
        for run in &plan.copies {
            for row in 0..run.rows {
                if let Err(error) = expand(
                    self.buffer.offset((run.source_row + row) * source_stride),
                    dst.offset((run.dest_row + row) * destination_stride),
                ) {
                    self.invalidate();
                    return Err(error);
                }
            }
        }
        Ok(())
    }
}
