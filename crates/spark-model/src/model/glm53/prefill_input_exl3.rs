// SPDX-License-Identifier: AGPL-3.0-only

//! Validated mixed input for a single-sequence, at-most-eight-row target walk.
//! This borrows token data and allocation descriptors, never allocation ownership.
//! The caller retains source/destination owners until its stream has drained and
//! must poison a partly executed walk before cleanup. No state publication here.

use crate::weight_loader::GLM53_MAX_CONTEXT_TOKENS;
use anyhow::{Context, Result, ensure};
use std::ops::Range;

const ROW_BYTES: usize = 4096 * 2;
const VOCAB: usize = 154_880;
const IMAGE_PAD: u32 = 154_854;
const MAX_ROWS: usize = 8;

#[derive(Clone, Copy, Debug)]
pub(super) struct InputRegion {
    pub address: u64,
    pub bytes: usize,
}

impl InputRegion {
    fn end(self) -> Result<u64> {
        ensure!(
            self.address != 0 && self.address.is_multiple_of(2) && self.bytes > 0,
            "GLM prefill input region is null, empty, or not BF16 aligned"
        );
        self.address
            .checked_add(u64::try_from(self.bytes)?)
            .context("GLM prefill input region address overflow")
    }

    fn slice(self, row: usize, rows: usize) -> Result<Self> {
        let offset = row
            .checked_mul(ROW_BYTES)
            .context("GLM input row offset overflow")?;
        let bytes = rows
            .checked_mul(ROW_BYTES)
            .context("GLM input row bytes overflow")?;
        ensure!(
            offset
                .checked_add(bytes)
                .is_some_and(|end| end <= self.bytes),
            "GLM prefill input slice exceeds its complete region"
        );
        let result = Self {
            address: self
                .address
                .checked_add(u64::try_from(offset)?)
                .context("GLM input slice address overflow")?,
            bytes,
        };
        result.end()?;
        Ok(result)
    }

    fn disjoint(self, other: Self) -> Result<()> {
        ensure!(
            self.end()? <= other.address || other.end()? <= self.address,
            "GLM prefill destination aliases a complete source allocation"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PreparedRows {
    pub region: InputRegion,
    pub rows: usize,
}

impl PreparedRows {
    pub(super) fn validate(self) -> Result<()> {
        ensure!(
            self.rows > 0 && self.rows <= GLM53_MAX_CONTEXT_TOKENS as usize,
            "GLM prepared rows exceed the model context bound"
        );
        let bytes = self
            .rows
            .checked_mul(ROW_BYTES)
            .context("GLM prepared byte overflow")?;
        ensure!(
            self.region.bytes == bytes,
            "GLM prepared BF16 H4096 extent drift"
        );
        self.region.end()?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DraftContext {
    pub position: u32,
    pub capacity: u32,
}

pub(super) struct InputPlan<'a> {
    tokens: &'a [u32],
    vision: Option<PreparedRows>,
    embedding: InputRegion,
    position: u32,
}

impl<'a> InputPlan<'a> {
    pub fn rows(&self) -> usize {
        self.tokens.len()
    }

    pub(super) fn token_slice(&self, range: Range<usize>) -> Result<&'a [u32]> {
        ensure!(
            range.start < range.end && range.end <= self.tokens.len(),
            "GLM layer-major input range exceeds the admitted prompt"
        );
        Ok(&self.tokens[range])
    }

    /// Whole-prompt admission, before reset, embedding copies, or target work.
    /// `draft` must describe the active runtime, not an invented context limit.
    pub fn new(
        tokens: &'a [u32],
        vision: Option<PreparedRows>,
        embedding: InputRegion,
        position: u32,
        capacity: u32,
        draft: Option<DraftContext>,
    ) -> Result<Self> {
        ensure!(
            !tokens.is_empty() && capacity > 0 && capacity <= GLM53_MAX_CONTEXT_TOKENS,
            "GLM mixed prefill prompt/capacity is invalid"
        );
        let end = position
            .checked_add(u32::try_from(tokens.len())?)
            .context("GLM mixed prefill position overflow")?;
        ensure!(end <= capacity, "GLM mixed prefill exceeds target context");
        if let Some(draft) = draft {
            ensure!(
                draft.position == position && draft.capacity > 0 && end <= draft.capacity,
                "GLM mixed prefill target/draft context disagrees or exceeds capacity"
            );
        }
        ensure!(
            embedding.bytes == VOCAB * ROW_BYTES,
            "GLM embedding table extent drift"
        );
        embedding.end()?;
        let mut pads = 0usize;
        for &token in tokens {
            ensure!(
                (token as usize) < VOCAB,
                "GLM prefill token exceeds vocabulary"
            );
            pads += usize::from(token == IMAGE_PAD);
        }
        if let Some(vision) = vision {
            vision.validate()?;
            ensure!(
                vision.rows > 0 && vision.rows == pads,
                "GLM prepared row count does not equal prompt image pads"
            );
        } else {
            ensure!(pads == 0, "GLM image pads have no prepared owner");
        }
        Ok(Self {
            tokens,
            vision,
            embedding,
            position,
        })
    }

    /// Builds at most eight copy descriptors; no prompt-sized prefix vector or
    /// additional device allocation. Absolute pad ordinal preserves image order.
    pub fn chunk(&self, range: Range<usize>, destination: InputRegion) -> Result<CopyBatch> {
        ensure!(
            range.start < range.end && range.end <= self.tokens.len(),
            "GLM input chunk range exceeds the admitted prompt"
        );
        let rows = range.end - range.start;
        ensure!(
            rows <= MAX_ROWS && destination.bytes == rows * ROW_BYTES,
            "GLM input chunk must contain exact 1..=8 BF16 H4096 rows"
        );
        destination.end()?;
        destination.disjoint(self.embedding)?;
        if let Some(vision) = self.vision {
            destination.disjoint(vision.region)?;
        }
        let position = self
            .position
            .checked_add(u32::try_from(range.start)?)
            .context("GLM input chunk absolute position overflow")?;
        let mut vision_row = self.tokens[..range.start]
            .iter()
            .filter(|&&token| token == IMAGE_PAD)
            .count();
        let mut copies = Vec::with_capacity(MAX_ROWS);
        let mut row = range.start;
        while row < range.end {
            let (source, count) = if self.tokens[row] == IMAGE_PAD {
                let mut end = row + 1;
                while end < range.end && self.tokens[end] == IMAGE_PAD {
                    end += 1;
                }
                let count = end - row;
                let vision = self.vision.context("GLM admitted image owner is missing")?;
                let source = vision.region.slice(vision_row, count)?;
                vision_row += count;
                (source, count)
            } else {
                (self.embedding.slice(self.tokens[row] as usize, 1)?, 1)
            };
            let dest = destination.slice(row - range.start, count)?;
            copies.push(InputCopy {
                source: source.address,
                destination: dest.address,
                bytes: source.bytes,
            });
            row += count;
        }
        Ok(CopyBatch {
            copies,
            position,
            rows,
        })
    }

    /// Overwrite only image-pad spans after the caller has embedded the full
    /// token chunk. This supports layer-major rows without allocating one copy
    /// descriptor per ordinary token.
    pub(super) fn overwrite_vision(
        &self,
        range: Range<usize>,
        destination: InputRegion,
        io: &mut impl InputCopyIo,
        stream: u64,
    ) -> Result<()> {
        let (vision, mut vision_row) = self.validate_vision_overwrite(&range, destination)?;
        let mut row = range.start;
        while row < range.end {
            if self.tokens[row] != IMAGE_PAD {
                row += 1;
                continue;
            }
            let start = row;
            while row < range.end && self.tokens[row] == IMAGE_PAD {
                row += 1;
            }
            let count = row - start;
            let source = vision.region.slice(vision_row, count)?;
            let target = destination.slice(start - range.start, count)?;
            io.copy(source.address, target.address, source.bytes, stream)
                .context("GLM layer-major vision overwrite failed; prefix may be in flight")?;
            vision_row += count;
        }
        Ok(())
    }

    pub(super) fn preflight_vision_overwrite(
        &self,
        range: Range<usize>,
        destination: InputRegion,
    ) -> Result<()> {
        self.validate_vision_overwrite(&range, destination)
            .map(|_| ())
    }

    fn validate_vision_overwrite(
        &self,
        range: &Range<usize>,
        destination: InputRegion,
    ) -> Result<(PreparedRows, usize)> {
        ensure!(
            range.start < range.end && range.end <= self.tokens.len(),
            "GLM layer-major vision range exceeds the admitted prompt"
        );
        let rows = range.end - range.start;
        ensure!(
            rows <= GLM53_MAX_CONTEXT_TOKENS as usize && destination.bytes == rows * ROW_BYTES,
            "GLM layer-major vision destination extent drift"
        );
        destination.end()?;
        destination.disjoint(self.embedding)?;
        let vision = self
            .vision
            .context("GLM layer-major vision overwrite has no prepared owner")?;
        destination.disjoint(vision.region)?;
        let vision_row = self.tokens[..range.start]
            .iter()
            .filter(|&&token| token == IMAGE_PAD)
            .count();
        Ok((vision, vision_row))
    }
}

pub(super) trait InputCopyIo {
    fn copy(&mut self, source: u64, destination: u64, bytes: usize, stream: u64) -> Result<()>;
}

struct InputCopy {
    source: u64,
    destination: u64,
    bytes: usize,
}

pub(super) struct CopyBatch {
    copies: Vec<InputCopy>,
    position: u32,
    rows: usize,
}

impl CopyBatch {
    pub fn start_position(&self) -> u32 {
        self.position
    }
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// A T1 walk consumes the original single row, without a redundant staging copy.
    pub fn single_source(&self) -> Result<InputRegion> {
        ensure!(
            self.rows == 1 && self.copies.len() == 1,
            "GLM T1 source requires exactly one admitted row"
        );
        let copy = &self.copies[0];
        ensure!(copy.bytes == ROW_BYTES, "GLM T1 source byte extent drift");
        Ok(InputRegion {
            address: copy.source,
            bytes: copy.bytes,
        })
    }

    /// Enqueue only: not a completion/commit receipt. A failure may leave an
    /// asynchronous prefix, so the owner must poison and drain before freeing.
    pub fn enqueue(&self, io: &mut impl InputCopyIo, stream: u64) -> Result<()> {
        for (index, copy) in self.copies.iter().enumerate() {
            io.copy(copy.source, copy.destination, copy.bytes, stream)
                .with_context(|| {
                    format!("GLM input copy {index} failed; prefix may be in flight")
                })?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "prefill_input_exl3_tests.rs"]
mod tests;
