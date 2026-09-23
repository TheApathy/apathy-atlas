// SPDX-License-Identifier: AGPL-3.0-only
//! Immutable, metadata-free diagnostic launch plans. No completion authority.
use super::gemv_contract::{HIDDEN, OUTPUTS, Span};
use anyhow::{Context, Result, ensure};

pub const MAX_ROWS: u32 = 2047;
pub const MAX_INPUT_ROWS: u32 = 2048;
pub const MAX_KERNEL_ROWS: u32 = 16;
pub const INPUT_ROW_BYTES: usize = HIDDEN as usize * 2;
pub const OUTPUT_ROW_BYTES: usize = OUTPUTS as usize * 2;
pub const WEIGHT_BYTES: usize = HIDDEN as usize * OUTPUTS as usize * 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    PairReference,
    BatchMDirect,
    BatchMPartitioned,
}
impl Schedule {
    pub fn name(self) -> &'static str {
        match self {
            Self::PairReference => "pair-reference",
            Self::BatchMDirect => "batchm-direct",
            Self::BatchMPartitioned => "batchm-partitioned",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchMKind {
    Single,
    Pair,
    Wide,
}
#[derive(Clone, Copy, Debug)]
pub struct BatchMLaunch {
    pub source_row: u32,
    pub output_row: u32,
    pub rows: u32,
    pub kind: BatchMKind,
    pub input: Span,
    pub weight: Span,
    pub output: Span,
}
pub trait BatchMIo {
    fn launch(&mut self, launch: BatchMLaunch) -> Result<()>;
}
pub struct BatchMPlan {
    rows: u32,
    source_row: u32,
    input: Span,
    weight: Span,
    output: Span,
}
pub fn validate_kernel_rows(rows: u32) -> Result<()> {
    ensure!(
        (1..=MAX_KERNEL_ROWS).contains(&rows),
        "batchM kernel requires 1..=16 rows"
    );
    Ok(())
}
impl BatchMPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rows: u32,
        source_row: u32,
        input_rows: u32,
        input: Span,
        weight: Span,
        output: Span,
        handles: [u64; 3],
    ) -> Result<Self> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows),
            "batchM diagnostic row limit"
        );
        ensure!(
            (1..=MAX_INPUT_ROWS).contains(&input_rows),
            "batchM input row limit"
        );
        ensure!(
            source_row
                .checked_add(rows)
                .is_some_and(|end| end <= input_rows),
            "batchM selected source rows exceed input"
        );
        ensure!(
            handles.iter().all(|h| *h != 0),
            "batchM kernel handle is null"
        );
        let spans = [input, weight, output];
        let required = [
            input_rows as usize * INPUT_ROW_BYTES,
            WEIGHT_BYTES,
            rows as usize * OUTPUT_ROW_BYTES,
        ];
        let mut ends = [0; 3];
        for (index, span) in spans.iter().enumerate() {
            let alignment = if index == 2 { 2 } else { 16 };
            ensure!(
                span.ptr != 0 && span.ptr % alignment == 0,
                "batchM owner alignment"
            );
            ensure!(
                span.bytes >= required[index] && span.bytes <= isize::MAX as usize,
                "batchM owner capacity"
            );
            ends[index] = span
                .ptr
                .checked_add(u64::try_from(span.bytes)?)
                .context("batchM owner end overflow")?;
        }
        for left in 0..3 {
            for right in left + 1..3 {
                ensure!(
                    spans[left].ptr >= ends[right] || spans[right].ptr >= ends[left],
                    "batchM owners overlap"
                );
            }
        }
        Ok(Self {
            rows,
            source_row,
            input,
            weight: Span {
                bytes: WEIGHT_BYTES,
                ..weight
            },
            output,
        })
    }
    pub fn launches(&self, schedule: Schedule) -> Result<Vec<BatchMLaunch>> {
        if schedule == Schedule::BatchMDirect {
            validate_kernel_rows(self.rows)?;
        }
        let mut launches = Vec::new();
        launches.try_reserve_exact(self.rows.div_ceil(2) as usize)?;
        let mut row = 0;
        while row < self.rows {
            let remaining = self.rows - row;
            let wide = schedule == Schedule::BatchMDirect
                || (schedule == Schedule::BatchMPartitioned && remaining > 8);
            let rows = remaining.min(if wide { MAX_KERNEL_ROWS } else { 2 });
            validate_kernel_rows(rows)?;
            let source_row = self.source_row + row;
            launches.push(BatchMLaunch {
                source_row,
                output_row: row,
                rows,
                kind: if wide {
                    BatchMKind::Wide
                } else if rows == 2 {
                    BatchMKind::Pair
                } else {
                    BatchMKind::Single
                },
                input: Span {
                    ptr: self.input.ptr + u64::from(source_row) * INPUT_ROW_BYTES as u64,
                    bytes: rows as usize * INPUT_ROW_BYTES,
                },
                weight: self.weight,
                output: Span {
                    ptr: self.output.ptr + u64::from(row) * OUTPUT_ROW_BYTES as u64,
                    bytes: rows as usize * OUTPUT_ROW_BYTES,
                },
            });
            row += rows;
        }
        Ok(launches)
    }
    /// Build/validate the whole schedule before submission; stop on the first
    /// error or unwind. Only the outer owned session may fence/free resources.
    pub fn execute(&self, schedule: Schedule, io: &mut dyn BatchMIo) -> Result<()> {
        for launch in self.launches(schedule)? {
            io.launch(launch)?;
        }
        Ok(())
    }
}
