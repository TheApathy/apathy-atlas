// SPDX-License-Identifier: AGPL-3.0-only
//! Pure admission for the pinned two-row BF16 projection experiment.
use anyhow::{Context, Result, ensure};

pub const HIDDEN: u32 = 4096;
pub const OUTPUTS: u32 = 1024;
const INPUT_ROW_BYTES: usize = HIDDEN as usize * 2;
const OUTPUT_ROW_BYTES: usize = OUTPUTS as usize * 2;
const WEIGHT_BYTES: usize = HIDDEN as usize * OUTPUTS as usize * 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub ptr: u64,
    pub bytes: usize,
}

/// Immutable device-address binding, not ownership or a completion receipt.
/// The session must retain these owners and initialized metadata through drain.
pub struct GemvPlan {
    rows: u32,
    input: Span,
    weight: Span,
    output: Span,
    slots: Span,
    table: Span,
}

impl GemvPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rows: u32,
        row: u32,
        input: Span,
        weight: Span,
        output: Span,
        slots: Span,
        table: Span,
    ) -> Result<Self> {
        ensure!(
            matches!(rows, 1 | 2) && row.checked_add(rows).is_some_and(|end| end <= 2),
            "GEMV experiment requires a nonempty span within the actual two input rows"
        );
        let owners = [input, weight, output, slots, table];
        let requirements = [
            (2 * INPUT_ROW_BYTES, 16),
            (WEIGHT_BYTES, 16),
            (2 * OUTPUT_ROW_BYTES, 2),
            (8, 4),
            (8, 8),
        ];
        let mut ends = [0; 5];
        for (index, (span, (bytes, alignment))) in owners.iter().zip(requirements).enumerate() {
            ends[index] = checked_owner(*span, bytes, alignment)?;
        }
        // Validate complete owner extents, not only the currently selected row:
        // metadata or a later call must never alias another live allocation.
        for left in 0..owners.len() {
            for right in left + 1..owners.len() {
                ensure!(
                    owners[left].ptr >= ends[right] || owners[right].ptr >= ends[left],
                    "GEMV input/weight/output/metadata owners overlap"
                );
            }
        }
        Ok(Self {
            rows,
            input: slice(input, row, rows, INPUT_ROW_BYTES)?,
            weight: Span {
                ptr: weight.ptr,
                bytes: WEIGHT_BYTES,
            },
            output: slice(output, row, rows, OUTPUT_ROW_BYTES)?,
            slots: slice(slots, row, rows, 4)?,
            table: Span {
                ptr: table.ptr,
                bytes: 8,
            },
        })
    }
    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn input(&self) -> Span {
        self.input
    }
    pub fn weight(&self) -> Span {
        self.weight
    }
    pub fn output(&self) -> Span {
        self.output
    }
    pub fn slots(&self) -> Span {
        self.slots
    }
    pub fn table(&self) -> Span {
        self.table
    }
}

fn checked_owner(span: Span, required: usize, alignment: u64) -> Result<u64> {
    ensure!(
        span.ptr != 0 && span.ptr % alignment == 0,
        "GEMV owner pointer alignment"
    );
    ensure!(
        span.bytes >= required && span.bytes <= isize::MAX as usize,
        "GEMV owner capacity is insufficient or exceeds isize"
    );
    span.ptr
        .checked_add(u64::try_from(span.bytes)?)
        .context("GEMV owner address end overflow")
}

fn slice(owner: Span, row: u32, rows: u32, stride: usize) -> Result<Span> {
    let offset = usize::try_from(row)?
        .checked_mul(stride)
        .context("GEMV row offset overflow")?;
    let bytes = usize::try_from(rows)?
        .checked_mul(stride)
        .context("GEMV row extent overflow")?;
    ensure!(
        offset
            .checked_add(bytes)
            .is_some_and(|end| end <= owner.bytes),
        "GEMV selected rows exceed owner"
    );
    let ptr = owner
        .ptr
        .checked_add(u64::try_from(offset)?)
        .context("GEMV selected row pointer overflow")?;
    Ok(Span { ptr, bytes })
}

/// Caller moves these bytes into session-owned uploads, then fences before use.
/// Only slot zero exists, and it names this actual, already allocated weight.
pub fn metadata(weight: Span) -> Result<([u8; 8], [u8; 8])> {
    checked_owner(weight, WEIGHT_BYTES, 16)?;
    Ok(([0; 8], weight.ptr.to_le_bytes()))
}

pub fn validate_handles(handles: [u64; 3]) -> Result<()> {
    ensure!(
        handles.iter().all(|handle| *handle != 0),
        "GEMV experiment kernel handle is null"
    );
    Ok(())
}
