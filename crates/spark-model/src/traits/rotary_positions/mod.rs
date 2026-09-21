// SPDX-License-Identifier: AGPL-3.0-only

//! Sequence-owned rotary coordinates, separate from physical token addresses.
//!
//! Image spans are immutable after admission. Commit/rollback changes physical
//! sequence length, never this map. Text-only sequences retain identity mapping.

use anyhow::{Context, Result, ensure};
use std::sync::Arc;

const MAX_IMAGE_SPANS: usize = 8;
const MAX_METADATA_ROWS: usize = 1_048_576;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageSpan {
    pub start: usize,
    pub height: usize,
    pub width: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RotaryPositions {
    max_seq_len: usize,
    prompt_len: usize,
    spans: Arc<[ImageSpan]>,
}

impl RotaryPositions {
    pub fn identity() -> Self {
        Self {
            max_seq_len: u32::MAX as usize + 1,
            prompt_len: 0,
            spans: Arc::from([]),
        }
    }

    pub fn from_image_spans(
        prompt_len: usize,
        max_seq_len: usize,
        spans: &[ImageSpan],
    ) -> Result<Self> {
        ensure!(prompt_len <= max_seq_len, "rotary prompt exceeds capacity");
        ensure!(max_seq_len > 0, "rotary capacity must be positive");
        ensure!(
            max_seq_len <= u32::MAX as usize,
            "rotary capacity exceeds u32 ABI"
        );
        ensure!(
            spans.len() <= MAX_IMAGE_SPANS,
            "rotary image count exceeds eight-image admission"
        );
        let mut previous_end = 0;
        for span in spans {
            ensure!(span.height > 0 && span.width > 0, "empty rotary image grid");
            let rows = span
                .height
                .checked_mul(span.width)
                .context("rotary image grid overflow")?;
            let end = span
                .start
                .checked_add(rows)
                .context("rotary image extent overflow")?;
            ensure!(
                span.start >= previous_end && end <= prompt_len,
                "rotary image spans overlap, are unordered, or exceed prompt"
            );
            previous_end = end;
        }
        Ok(Self {
            max_seq_len,
            prompt_len,
            spans: Arc::from(spans),
        })
    }

    pub fn is_identity(&self) -> bool {
        self.spans.is_empty()
    }

    pub fn position(&self, physical: usize) -> Result<[u32; 3]> {
        ensure!(
            physical < self.max_seq_len,
            "rotary position exceeds capacity"
        );
        let mut removed = 0usize;
        for span in self.spans.iter() {
            if physical < span.start {
                break;
            }
            // Geometry and ordered extents were checked once at admission.
            let rows = span.height * span.width;
            if physical < span.start + rows {
                let base = span.start - removed;
                let row = physical - span.start;
                return Ok([
                    u32::try_from(base)?,
                    u32::try_from(base + row / span.width)?,
                    u32::try_from(base + row % span.width)?,
                ]);
            }
            removed += rows - span.height.max(span.width);
        }
        Ok([u32::try_from(physical - removed).context("rotary position exceeds u32 ABI")?; 3])
    }

    pub fn tail_scalar(&self, physical: usize) -> Result<u32> {
        ensure!(
            self.is_identity() || physical >= self.prompt_len,
            "image prompt replay requires three rotary axes"
        );
        Ok(self.position(physical)?[0])
    }

    pub fn range(&self, start: usize, len: usize) -> Result<Vec<[u32; 3]>> {
        let end = start.checked_add(len).context("rotary range overflow")?;
        ensure!(end <= self.max_seq_len, "rotary range exceeds capacity");
        ensure!(
            len <= MAX_METADATA_ROWS,
            "rotary metadata range exceeds one million rows"
        );
        (start..end).map(|p| self.position(p)).collect()
    }

    pub fn verify_tail(&self, physical: usize, depths: &[usize]) -> Result<Vec<u32>> {
        self.tail_scalar(physical)?;
        ensure!(
            depths.len() <= MAX_METADATA_ROWS,
            "rotary verify row count exceeds limit"
        );
        depths
            .iter()
            .map(|depth| {
                self.tail_scalar(
                    physical
                        .checked_add(*depth)
                        .context("rotary depth overflow")?,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
