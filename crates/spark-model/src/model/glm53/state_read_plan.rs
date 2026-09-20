// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded, aligned physical coverage of a logical diagnostic state slice.
//! The real parent allocation, not a typed subview, authorizes alignment bytes.

use anyhow::{Context, Result, ensure};
use std::ops::Range;

pub(super) struct StateReadPlan {
    source: u64,
    physical_bytes: usize,
    logical_range: Range<usize>,
}

impl StateReadPlan {
    pub(super) fn new(
        parent_address: u64,
        parent_bytes: usize,
        region_address: u64,
        region_bytes: usize,
        offset: usize,
        bytes: usize,
    ) -> Result<Self> {
        ensure!(
            parent_address != 0
                && region_address != 0
                && parent_bytes > 0
                && parent_bytes <= isize::MAX as usize
                && region_bytes > 0
                && region_bytes <= isize::MAX as usize
                && bytes > 0,
            "state read requires nonempty bounded allocations and request"
        );
        let parent_end = parent_address
            .checked_add(u64::try_from(parent_bytes)?)
            .context("state read parent end overflow")?;
        let region_end = region_address
            .checked_add(u64::try_from(region_bytes)?)
            .context("state read region end overflow")?;
        ensure!(
            region_address >= parent_address && region_end <= parent_end,
            "state read region exceeds its actual allocation"
        );
        let requested_end = offset
            .checked_add(bytes)
            .context("state read offset/length overflow")?;
        ensure!(
            requested_end <= region_bytes,
            "state read exceeds logical region"
        );
        let logical_start = region_address
            .checked_add(u64::try_from(offset)?)
            .context("state read logical start overflow")?;
        let logical_end = region_address
            .checked_add(u64::try_from(requested_end)?)
            .context("state read logical end overflow")?;
        let source = logical_start & !1;
        let physical_end = logical_end
            .checked_add(1)
            .context("state read covering end overflow")?
            & !1;
        ensure!(
            source >= parent_address && physical_end <= parent_end,
            "state read aligned cover exceeds its actual allocation"
        );
        let physical_bytes = usize::try_from(physical_end - source)?;
        ensure!(
            (2..=65_536).contains(&physical_bytes) && physical_bytes % 2 == 0,
            "state read physical cover exceeds 64 KiB"
        );
        let first = usize::try_from(logical_start - source)?;
        let last = first
            .checked_add(bytes)
            .context("state read crop end overflow")?;
        ensure!(
            last <= physical_bytes,
            "state read crop exceeds physical cover"
        );
        Ok(Self {
            source,
            physical_bytes,
            logical_range: first..last,
        })
    }

    pub(super) fn source(&self) -> u64 {
        self.source
    }
    pub(super) fn physical_bytes(&self) -> usize {
        self.physical_bytes
    }
    pub(super) fn logical_range(&self) -> Range<usize> {
        self.logical_range.clone()
    }
}
