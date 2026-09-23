// SPDX-License-Identifier: AGPL-3.0-only

//! Checked slot-major capture storage, independent of target/drafter execution.
//! Binding validates caller-owned device spans; this module allocates no GPU memory.

use anyhow::{Context, Result, ensure};

use crate::layers::Glm53TargetGeometry;
use crate::model::glm53::GLM53_CAPTURE_LAYERS;

const MAX_CAPTURE_ROWS: u32 = 2048;
const MAX_BANK_BYTES: usize = 80 * 1024 * 1024;
pub(super) const GATHER_ROWS: u32 = 8;

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
    ) -> Result<BoundCapturePlan> {
        bank.validate(self.bank_bytes)?;
        staging.validate(product(
            GATHER_ROWS as usize,
            product(GLM53_CAPTURE_LAYERS.len(), self.row_bytes)?,
        )?)?;
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CopyStep {
    pub source: u64,
    pub destination: u64,
    pub bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProjectionStep {
    pub input: DeviceSpan,
    pub output: DeviceSpan,
    pub rows: u32,
}

#[derive(Debug)]
pub(crate) struct CaptureSlice {
    pub copies: Vec<CopyStep>,
    pub projection: ProjectionStep,
}

impl BoundCapturePlan {
    pub(super) fn plan(&self) -> &PrefillCapturePlan {
        &self.plan
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
            (1..=GATHER_ROWS).contains(&rows),
            "capture gather requires 1..=8 rows"
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
        let output_row = self
            .plan
            .start
            .checked_add(first)
            .context("capture output position overflow")?;
        let projection = ProjectionStep {
            input: self.staging.slice(0, product(count, row_bytes)?)?,
            output: self.projected.slice(
                product(output_row as usize, row_bytes)?,
                product(rows as usize, row_bytes)?,
            )?,
            rows,
        };
        Ok(CaptureSlice { copies, projection })
    }
}

fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .context("capture byte extent overflow")
}
