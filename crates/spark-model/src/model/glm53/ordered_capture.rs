// SPDX-License-Identifier: AGPL-3.0-only

//! Consuming chronological capture plan. The enclosing transaction owns failure.

use anyhow::{Context, Result, ensure};

pub(super) trait OrderedCaptureIo {
    fn project_row(
        &mut self,
        capture_row: u32,
        destination_position: u32,
        stream: u64,
    ) -> Result<()>;
    fn fence(&mut self, stream: u64) -> Result<()>;
}

#[must_use = "capture advancement requires successful execution and completion"]
pub(super) struct OrderedCapturePlan {
    start: u32,
    rows: u32,
    end: u32,
}

impl OrderedCapturePlan {
    pub(super) fn new(
        start: u32,
        rows: u32,
        target_position: u32,
        context_tokens: u32,
        capacity: u32,
    ) -> Result<Self> {
        ensure!(
            (2..=8).contains(&rows),
            "ordered capture requires 2..=8 rows"
        );
        let end = start
            .checked_add(rows)
            .context("ordered capture end overflow")?;
        ensure!(
            context_tokens == start && target_position == end && end <= capacity,
            "ordered capture target/context/capacity mismatch"
        );
        Ok(Self { start, rows, end })
    }

    pub(super) fn execute(self, io: &mut impl OrderedCaptureIo, stream: u64) -> Result<u32> {
        for row in 0..self.rows {
            io.project_row(row, self.start + row, stream)?;
        }
        io.fence(stream)?;
        Ok(self.end)
    }
}
