// SPDX-License-Identifier: AGPL-3.0-only

//! Checked slot-major capture storage, independent of target/drafter execution.
//! Binding validates caller-owned device spans; this module allocates no GPU memory.

use std::ffi::OsStr;

use anyhow::{Context, Result, bail, ensure};

use crate::layers::Glm53TargetGeometry;
use crate::model::glm53::GLM53_CAPTURE_LAYERS;

const MAX_CAPTURE_ROWS: u32 = 2048;
const MAX_BANK_BYTES: usize = 80 * 1024 * 1024;
const LEGACY_CAPTURE_ROWS: u32 = 8;
const CAPTURE_HIDDEN: usize = 4096;
pub(super) const TILED_CAPTURE_ROWS: u32 = 128;
pub(super) const TILED_CAPTURE_STAGING_BYTES: usize =
    TILED_CAPTURE_ROWS as usize * 5 * CAPTURE_HIDDEN * 2;
const MAX_CAPTURE_STAGING_BYTES: usize = 6 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CaptureTransferMode {
    Copies8,
    Gather128,
}

impl CaptureTransferMode {
    pub(crate) fn parse(value: Option<&OsStr>) -> Result<Self> {
        match value.and_then(OsStr::to_str) {
            None | Some("0") => Ok(Self::Copies8),
            Some("1") => Ok(Self::Gather128),
            _ => bail!("ATLAS_GLM53_DFLASH2_CAPTURE_TILE128 must be absent, 0, or 1"),
        }
    }

    pub(crate) const fn rows(self) -> u32 {
        match self {
            Self::Copies8 => LEGACY_CAPTURE_ROWS,
            Self::Gather128 => TILED_CAPTURE_ROWS,
        }
    }

    pub(crate) fn staging_bytes(self, row_bytes: usize) -> Result<usize> {
        let bytes = product(
            self.rows() as usize,
            product(GLM53_CAPTURE_LAYERS.len(), row_bytes)?,
        )?;
        ensure!(
            bytes <= MAX_CAPTURE_STAGING_BYTES,
            "capture staging exceeds 6 MiB"
        );
        Ok(bytes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceSpan {
    pub address: u64,
    pub bytes: usize,
}

impl DeviceSpan {
    fn end(self) -> Result<u64> {
        self.address
            .checked_add(u64::try_from(self.bytes).context("capture span exceeds u64")?)
            .context("capture device span overflows address space")
    }

    pub(super) fn validate(self, required: usize) -> Result<()> {
        ensure!(
            self.address != 0 && self.address % 256 == 0,
            "capture device span must be nonnull and 256-byte aligned"
        );
        ensure!(
            required != 0 && self.bytes >= required,
            "capture device span is too short"
        );
        self.end()?;
        Ok(())
    }

    fn slice(self, offset: usize, bytes: usize) -> Result<Self> {
        let end = offset
            .checked_add(bytes)
            .context("capture slice extent overflow")?;
        ensure!(
            bytes != 0 && end <= self.bytes,
            "capture slice exceeds device span"
        );
        let address = self
            .address
            .checked_add(u64::try_from(offset).context("capture offset exceeds u64")?)
            .context("capture slice address overflow")?;
        let result = Self { address, bytes };
        result.end()?;
        Ok(result)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PrefillCapturePlan {
    rows: u32,
    start: u32,
    end: u32,
    context_limit: u32,
    row_bytes: usize,
    slot_stride: usize,
    bank_bytes: usize,
}

impl PrefillCapturePlan {
    /// `context_limit` must be supplied from the installed drafter's actual arena.
    /// This helper cannot enable contexts beyond the current 2047-row contract.
    pub(crate) fn new(
        capacity_rows: u32,
        rows: u32,
        start_position: u32,
        context_tokens: u32,
        context_limit: u32,
        target_capacity: u32,
    ) -> Result<Self> {
        ensure!(
            (1..=MAX_CAPTURE_ROWS).contains(&capacity_rows),
            "capture capacity must be 1..=2048"
        );
        ensure!(
            rows != 0 && rows <= capacity_rows,
            "capture rows exceed bank capacity"
        );
        ensure!(
            context_limit != 0 && context_limit < MAX_CAPTURE_ROWS,
            "capture requires an installed drafter context limit in 1..=2047"
        );
        ensure!(
            start_position == context_tokens,
            "target and drafter capture positions differ"
        );
        let end = start_position
            .checked_add(rows)
            .context("capture position overflow")?;
        ensure!(
            end <= context_limit && end <= target_capacity,
            "capture exceeds target/drafter context"
        );
        let row_bytes = product(Glm53TargetGeometry::exact(rows).hidden_size as usize, 2)?;
        let slot_stride = product(capacity_rows as usize, row_bytes)?;
        let bank_bytes = product(GLM53_CAPTURE_LAYERS.len(), slot_stride)?;
        ensure!(bank_bytes <= MAX_BANK_BYTES, "capture bank exceeds 80 MiB");
        Ok(Self {
            rows,
            start: start_position,
            end,
            context_limit,
            row_bytes,
            slot_stride,
            bank_bytes,
        })
    }

    pub(crate) fn rows(&self) -> u32 {
        self.rows
    }
    pub(crate) fn end_position(&self) -> u32 {
        self.end
    }
    pub(super) fn start_position(&self) -> u32 {
        self.start
    }
    pub(crate) fn row_bytes(&self) -> usize {
        self.row_bytes
    }
    pub(crate) fn slot_stride_bytes(&self) -> usize {
        self.slot_stride
    }
    pub(crate) fn bank_bytes(&self) -> usize {
        self.bank_bytes
    }

    pub(crate) fn bind(
        self,
        bank: DeviceSpan,
        staging: DeviceSpan,
        projected: DeviceSpan,
        mode: CaptureTransferMode,
    ) -> Result<BoundCapturePlan> {
        bank.validate(self.bank_bytes)?;
        staging.validate(mode.staging_bytes(self.row_bytes)?)?;
        projected.validate(product(self.context_limit as usize, self.row_bytes())?)?;
        for (left, right) in [(bank, staging), (bank, projected), (staging, projected)] {
            ensure!(
                left.end()? <= right.address || right.end()? <= left.address,
                "capture bank, staging and projected spans must not overlap"
            );
        }
        Ok(BoundCapturePlan {
            plan: self,
            bank,
            staging,
            projected,
            mode,
        })
    }
}

/// Fields stay private: execution can only use an already-validated binding.
#[derive(Clone, Debug)]
pub(crate) struct BoundCapturePlan {
    plan: PrefillCapturePlan,
    bank: DeviceSpan,
    staging: DeviceSpan,
    projected: DeviceSpan,
    mode: CaptureTransferMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CopyStep {
    pub source: u64,
    pub destination: u64,
    pub bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GatherStep {
    pub source: DeviceSpan,
    pub destination: DeviceSpan,
    pub first_row: u32,
    pub rows: u32,
    pub row_bytes: usize,
    pub slot_stride_bytes: usize,
}

impl GatherStep {
    pub(crate) fn addresses(&self, row: u32, slot: usize, column: usize) -> Result<(u64, u64)> {
        ensure!(row < self.rows, "capture gather row is out of range");
        ensure!(
            slot < GLM53_CAPTURE_LAYERS.len(),
            "capture gather tap is out of range"
        );
        let hidden = self.row_bytes / 2;
        ensure!(column < hidden, "capture gather column is out of range");
        let absolute_row = self
            .first_row
            .checked_add(row)
            .context("capture gather row overflow")?;
        let column_offset = product(column, 2)?;
        let source_offset = product(slot, self.slot_stride_bytes)?
            .checked_add(product(absolute_row as usize, self.row_bytes)?)
            .and_then(|base| base.checked_add(column_offset))
            .context("capture gather source offset overflow")?;
        let output_row = product(row as usize, GLM53_CAPTURE_LAYERS.len())?
            .checked_add(slot)
            .context("capture gather destination row overflow")?;
        let destination_offset = product(output_row, self.row_bytes)?
            .checked_add(column_offset)
            .context("capture gather destination offset overflow")?;
        Ok((
            self.source.slice(source_offset, 2)?.address,
            self.destination.slice(destination_offset, 2)?.address,
        ))
    }
}

#[derive(Debug)]
pub(crate) enum CaptureTransfer {
    Copies(Vec<CopyStep>),
    Gather(GatherStep),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProjectionStep {
    pub input: DeviceSpan,
    pub output: DeviceSpan,
    pub rows: u32,
}

#[derive(Debug)]
pub(crate) struct CaptureSlice {
    pub transfer: CaptureTransfer,
    pub projection: ProjectionStep,
}

impl BoundCapturePlan {
    pub(super) fn plan(&self) -> &PrefillCapturePlan {
        &self.plan
    }

    pub(super) const fn transfer_rows(&self) -> u32 {
        self.mode.rows()
    }

    pub(crate) fn tap_destination(&self, layer: u32, slot: usize) -> Result<DeviceSpan> {
        ensure!(
            GLM53_CAPTURE_LAYERS.get(slot) == Some(&layer),
            "capture layer/slot does not match the canonical five taps"
        );
        self.bank.slice(
            product(slot, self.plan.slot_stride_bytes())?,
            product(self.plan.rows as usize, self.plan.row_bytes)?,
        )
    }

    /// Token-major then canonical tap-major into the existing eight-row staging.
    pub(crate) fn slice(&self, first: u32, rows: u32) -> Result<CaptureSlice> {
        ensure!(
            (1..=self.mode.rows()).contains(&rows),
            "capture tile rows exceed mode"
        );
        let last = first
            .checked_add(rows)
            .context("capture gather row overflow")?;
        ensure!(
            last <= self.plan.rows,
            "capture gather exceeds completed rows"
        );
        let taps = GLM53_CAPTURE_LAYERS.len();
        let row_bytes = self.plan.row_bytes;
        let count = product(rows as usize, taps)?;
        let transfer_bytes = product(count, row_bytes)?;
        let transfer = match self.mode {
            CaptureTransferMode::Copies8 => {
                let mut copies = Vec::with_capacity(count);
                for row in first..last {
                    for slot in 0..taps {
                        let source_offset = product(slot, self.plan.slot_stride)?
                            .checked_add(product(row as usize, row_bytes)?)
                            .context("capture bank offset overflow")?;
                        let source = self.bank.slice(source_offset, row_bytes)?;
                        let destination = self
                            .staging
                            .slice(product(copies.len(), row_bytes)?, row_bytes)?;
                        copies.push(CopyStep {
                            source: source.address,
                            destination: destination.address,
                            bytes: row_bytes,
                        });
                    }
                }
                CaptureTransfer::Copies(copies)
            }
            CaptureTransferMode::Gather128 => CaptureTransfer::Gather(GatherStep {
                source: self.bank,
                destination: self.staging.slice(0, transfer_bytes)?,
                first_row: first,
                rows,
                row_bytes,
                slot_stride_bytes: self.plan.slot_stride,
            }),
        };
        let output_row = self
            .plan
            .start
            .checked_add(first)
            .context("capture output position overflow")?;
        let projection = ProjectionStep {
            input: self.staging.slice(0, transfer_bytes)?,
            output: self.projected.slice(
                product(output_row as usize, row_bytes)?,
                product(rows as usize, row_bytes)?,
            )?,
            rows,
        };
        Ok(CaptureSlice {
            transfer,
            projection,
        })
    }
}

fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .context("capture byte extent overflow")
}
