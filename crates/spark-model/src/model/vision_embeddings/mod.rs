// SPDX-License-Identifier: AGPL-3.0-only

//! C1-only published final-merger aggregate. Encoder scratch is never a consumer source.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::collections::BTreeSet;

mod bridge;
mod cache;
mod layout;
mod plan;
mod ranges;
mod rotary;
#[cfg(test)]
pub(crate) mod test_gpu;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_admission;
#[cfg(test)]
mod tests_failures;
#[cfg(test)]
mod tests_identity;

pub(crate) const MAX_AGGREGATE_BYTES: usize = 256 * 1024 * 1024;

pub(crate) struct VisionEmbeddingState {
    buffer: DevicePtr,
    capacity: usize,
    max_rows: usize,
    max_sequences: usize,
    published: Option<(layout::ImageLayout, Option<cache::ExactCacheKey>)>,
    streams: BTreeSet<u64>,
}

impl VisionEmbeddingState {
    pub fn new(max_rows: usize, max_sequences: usize) -> Self {
        Self {
            buffer: DevicePtr::NULL,
            capacity: 0,
            max_rows,
            max_sequences,
            published: None,
            streams: BTreeSet::new(),
        }
    }

    pub fn is_ready(&self) -> bool {
        self.published.is_some()
    }

    pub fn invalidate(&mut self) {
        self.published = None;
    }

    /// Returns true only on a valid cache hit. Errors leave no published rows.
    pub fn prepare<F>(
        &mut self,
        gpu: &dyn GpuBackend,
        images: &[(Vec<f32>, usize, usize)],
        geometry: layout::Geometry,
        cache: bool,
        variant: u8,
        stream: u64,
        mut encode: F,
    ) -> Result<bool>
    where
        F: FnMut(&[f32], usize, usize) -> Result<(DevicePtr, usize)>,
    {
        let old = self.published.take();
        if images.is_empty() {
            return Ok(false);
        }
        ensure!(
            self.max_sequences == 1,
            "Qwen image preparation requires max-concurrent-sequences=1"
        );
        let layout = layout::ImageLayout::validate(images, geometry, self.max_rows)?;
        let fingerprint = cache.then(|| layout::cache_key(images, geometry, variant));
        if let Some((previous, previous_key)) = old {
            if previous == layout
                && fingerprint.is_some_and(|hash| {
                    previous_key
                        .as_ref()
                        .is_some_and(|key| key.matches(images, variant, hash))
                })
            {
                ensure!(
                    self.buffer != DevicePtr::NULL && self.capacity >= layout.bytes,
                    "invalid cached vision allocation"
                );
                self.published = Some((previous, previous_key));
                return Ok(true);
            }
        }
        // Previous consumers can use a different stream. Drain before reuse/free.
        for &prior in &self.streams {
            gpu.synchronize(prior)?;
        }
        self.streams.clear();
        self.streams.insert(stream);
        gpu.synchronize(stream)?;
        if self.capacity < layout.bytes {
            let replacement = gpu.alloc(layout.bytes)?;
            ensure!(replacement != DevicePtr::NULL, "null vision allocation");
            if self.buffer != DevicePtr::NULL {
                if let Err(error) = gpu.free(self.buffer) {
                    let _ = gpu.free(replacement);
                    return Err(error);
                }
            }
            self.buffer = replacement;
            self.capacity = layout.bytes;
        }
        let mut offset = 0usize;
        for ((pixels, gh, gw), &(mh, mw)) in images.iter().zip(&layout.grids) {
            let rows = mh * mw;
            let (source, encoded_rows) = encode(pixels, *gh, *gw)?;
            ensure!(
                encoded_rows == rows * (geometry.deepstack + 1),
                "encoder output row count mismatch"
            );
            let source_bytes = encoded_rows
                .checked_mul(geometry.hidden)
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| anyhow::anyhow!("vision source size overflow"))?;
            ranges::disjoint(source, source_bytes, self.buffer, self.capacity)?;
            let bytes = rows * geometry.hidden * 2;
            ensure!(
                offset + bytes <= layout.bytes,
                "vision aggregate write out of bounds"
            );
            gpu.copy_d2d_async(source, self.buffer.offset(offset), bytes, stream)?;
            offset += bytes;
        }
        ensure!(offset == layout.bytes, "incomplete vision aggregate");
        gpu.synchronize(stream)?;
        let key = fingerprint.and_then(|hash| {
            cache::ExactCacheKey::capture(images, variant, hash, cache::MAX_CACHE_INPUT_BYTES)
        });
        self.published = Some((layout, key));
        Ok(false)
    }

    pub fn plan(
        &self,
        tokens: &[u32],
        pad: u32,
        start: usize,
        len: usize,
    ) -> Result<plan::PromptPlan> {
        ensure!(
            tokens.len() <= self.max_rows,
            "vision prompt exceeds configured sequence capacity"
        );
        match &self.published {
            Some((layout, _)) => layout.plan(tokens, pad, start, len),
            None => plan::plan(tokens, pad, &[], start, len),
        }
    }

    pub fn splice(
        &mut self,
        gpu: &dyn GpuBackend,
        tokens: &[u32],
        pad: u32,
        start: usize,
        len: usize,
        dst: DevicePtr,
        hidden: usize,
        element_bytes: usize,
        stream: u64,
    ) -> Result<()> {
        let plan = self.plan(tokens, pad, start, len)?;
        if plan.copies.is_empty() {
            return Ok(());
        }
        let (layout, _) = self
            .published
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing vision aggregate"))?;
        ensure!(
            hidden == layout.geometry.hidden && element_bytes == 2,
            "vision embedding destination must have matching BF16 hidden width"
        );
        ensure!(
            dst != DevicePtr::NULL && self.buffer != DevicePtr::NULL,
            "null vision embedding buffer"
        );
        let row_bytes = hidden
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("vision row byte overflow"))?;
        let dst_bytes = len
            .checked_mul(row_bytes)
            .ok_or_else(|| anyhow::anyhow!("vision destination size overflow"))?;
        ranges::disjoint(self.buffer, self.capacity, dst, dst_bytes)?;
        for run in &plan.copies {
            ensure!(
                run.source_row + run.rows <= layout.rows && run.dest_row + run.rows <= len,
                "vision splice range exceeds valid rows"
            );
        }
        self.streams.insert(stream);
        for run in &plan.copies {
            if let Err(error) = gpu.copy_d2d_async(
                self.buffer.offset(run.source_row * row_bytes),
                dst.offset(run.dest_row * row_bytes),
                run.rows * row_bytes,
                stream,
            ) {
                self.invalidate();
                return Err(error);
            }
        }
        Ok(())
    }

    /// Called by the model's Drop while its GPU backend is still alive.
    pub fn release(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        self.invalidate();
        for &stream in &self.streams {
            gpu.synchronize(stream)?;
        }
        if self.buffer != DevicePtr::NULL {
            gpu.free(self.buffer)?;
        }
        self.buffer = DevicePtr::NULL;
        self.capacity = 0;
        self.streams.clear();
        Ok(())
    }
}
