// SPDX-License-Identifier: AGPL-3.0-only

//! Absolute and resident state transitions for the bounded DFlash ring.

use super::{Result, RingCopyPlan, RingPlanError, RingSpanPlan, plan_ring_copy, plan_write_chunk};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RingState {
    pub max_context: usize,
    pub capacity: usize,
    pub absolute_len: usize,
    pub resident_len: usize,
}

impl RingState {
    pub(crate) fn new(max_context: usize, capacity: usize) -> Result<Self> {
        Self::from_lengths(max_context, capacity, 0, 0)
    }

    pub(crate) fn from_lengths(
        max_context: usize,
        capacity: usize,
        absolute_len: usize,
        resident_len: usize,
    ) -> Result<Self> {
        if max_context == 0 || capacity == 0 || capacity > max_context {
            return Err(RingPlanError("ring capacity is incompatible with context"));
        }
        if absolute_len > max_context || resident_len != absolute_len.min(capacity) {
            return Err(RingPlanError(
                "absolute and resident lengths are inconsistent",
            ));
        }
        Ok(Self {
            max_context,
            capacity,
            absolute_len,
            resident_len,
        })
    }

    pub(crate) fn resident_start(&self) -> usize {
        self.absolute_len - self.resident_len
    }

    pub(crate) fn slot_for(&self, absolute_position: usize) -> Result<usize> {
        if absolute_position < self.resident_start() || absolute_position >= self.absolute_len {
            return Err(RingPlanError("absolute position is not resident"));
        }
        Ok(absolute_position % self.capacity)
    }

    /// Gather a fully resident logical range in chronological order.
    pub(crate) fn plan_gather(&self, absolute_start: usize, rows: usize) -> Result<RingCopyPlan> {
        if rows == 0 {
            return Err(RingPlanError("gather row count must be non-zero"));
        }
        let absolute_end = absolute_start
            .checked_add(rows)
            .ok_or(RingPlanError("absolute gather end overflowed"))?;
        if absolute_start < self.resident_start() || absolute_end > self.absolute_len {
            return Err(RingPlanError("gather range is not fully resident"));
        }
        plan_ring_copy(
            self.resident_start(),
            self.absolute_len,
            self.capacity,
            absolute_start,
            absolute_end,
        )
    }

    /// Plan bootstrap plus accepted target rows and advance absolute context.
    pub(crate) fn plan_accepted_append(&self, rows: usize) -> Result<AppendPlan> {
        let write = plan_write_chunk(self.absolute_len, rows, self.max_context, self.capacity)?;
        let absolute_len = self
            .absolute_len
            .checked_add(rows)
            .ok_or(RingPlanError("absolute append end overflowed"))?;
        let next = Self::from_lengths(
            self.max_context,
            self.capacity,
            absolute_len,
            absolute_len.min(self.capacity),
        )?;
        Ok(AppendPlan { write, next })
    }

    /// Plan an append only when the caller's absolute cursor is exact.
    pub(crate) fn plan_append_at(&self, absolute_start: usize, rows: usize) -> Result<AppendPlan> {
        if absolute_start != self.absolute_len {
            return Err(RingPlanError("append start does not match absolute cursor"));
        }
        self.plan_accepted_append(rows)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AppendPlan {
    pub write: RingSpanPlan,
    pub next: RingState,
}
