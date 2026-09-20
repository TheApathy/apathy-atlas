// SPDX-License-Identifier: AGPL-3.0-only
//! Checked, metadata-free pairs plus an optional final singleton.
use super::projection_contract::checked_spans;
use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemvChunk {
    pub row: u32,
    pub rows: u32,
    pub input: u64,
    pub output: u64,
}

pub trait GemvIo {
    fn launch(&mut self, chunk: GemvChunk) -> Result<()>;
}

pub struct GemvPlan {
    rows: u32,
    hidden: u32,
    width: u32,
    input: u64,
    weight: u64,
    output: u64,
    handles: [u64; 2],
}

impl GemvPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rows: u32,
        max_rows: u32,
        hidden: u32,
        width: u32,
        input: (u64, usize),
        weight: (u64, usize),
        output: (u64, usize),
        handles: [u64; 2],
    ) -> Result<Self> {
        checked_spans(rows, max_rows, hidden, width, input, weight, output)?;
        // The two kernels load uint4 groups of eight BF16 values and assign
        // four complete output groups per block. Do not tighten other families.
        ensure!(
            input.0 % 16 == 0 && weight.0 % 16 == 0 && hidden % 8 == 0 && width % 4 == 0,
            "GEMV projection requires aligned vector rows and complete output groups"
        );
        ensure!(
            handles.iter().all(|handle| *handle != 0),
            "GEMV projection kernel handle is null"
        );
        Ok(Self {
            rows,
            hidden,
            width,
            input: input.0,
            weight: weight.0,
            output: output.0,
            handles,
        })
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn hidden(&self) -> u32 {
        self.hidden
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn weight(&self) -> u64 {
        self.weight
    }
    pub fn handles(&self) -> [u64; 2] {
        self.handles
    }

    pub fn chunks(&self) -> impl Iterator<Item = GemvChunk> + '_ {
        // Admission checked complete extents and pointer ends. Each offset is
        // strictly inside those immutable spans, including odd final rows.
        (0..self.rows()).step_by(2).map(|row| GemvChunk {
            row,
            rows: (self.rows() - row).min(2),
            input: self.input + u64::from(row) * u64::from(self.hidden) * 2,
            output: self.output + u64::from(row) * u64::from(self.width) * 2,
        })
    }

    /// Submission only. The real outer Attempt/KvPrefix owns pending work and
    /// publishes cursors only after every layer and a successful stream fence.
    pub fn execute(&self, io: &mut dyn GemvIo) -> Result<()> {
        for chunk in self.chunks() {
            io.launch(chunk)?;
        }
        Ok(())
    }
}
