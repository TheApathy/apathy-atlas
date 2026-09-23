// SPDX-License-Identifier: AGPL-3.0-only

//! Pure, checked indexing for the bounded native-V3 target-hidden ring.
//!
//! Logical positions remain absolute. Only physical storage wraps. GPU callers
//! must consume these plans instead of applying absolute byte offsets directly.

use std::error::Error;
use std::fmt::{Display, Formatter};

pub(crate) const NATIVE_V3_RING_SLOTS: usize = 4096;
pub(crate) const NATIVE_V3_CAPTURE_TAPS: usize = 8;
pub(crate) const NATIVE_V3_HIDDEN_SIZE: usize = 2560;
pub(crate) const MAX_ABSOLUTE_CONTEXT: usize = 1_048_576;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RingPlanError(&'static str);

impl Display for RingPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl Error for RingPlanError {}

type Result<T> = std::result::Result<T, RingPlanError>;

#[path = "ring_window_layout.rs"]
mod layout;
pub(crate) use layout::{accumulator_capture_offset, plan_accumulator_layout};
pub(crate) type AccumulatorLayout = layout::AccumulatorLayout;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RingSpan {
    pub physical_slot: usize,
    pub linear_slot: usize,
    pub slot_count: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RingSpanPlan {
    spans: [RingSpan; 2],
    len: usize,
}

impl RingSpanPlan {
    pub(crate) fn spans(&self) -> &[RingSpan] {
        &self.spans[..self.len]
    }

    pub(crate) fn physical_slot_for(&self, linear_slot: usize) -> Result<usize> {
        for span in self.spans() {
            let span_end = span
                .linear_slot
                .checked_add(span.slot_count)
                .ok_or(RingPlanError("ring span end overflowed"))?;
            if (span.linear_slot..span_end).contains(&linear_slot) {
                return span
                    .physical_slot
                    .checked_add(linear_slot - span.linear_slot)
                    .ok_or(RingPlanError("physical ring slot overflowed"));
            }
        }
        Err(RingPlanError("linear slot is outside the ring plan"))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RingCopySpan {
    pub src_slot: usize,
    pub dst_slot: usize,
    pub slot_count: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RingCopyPlan {
    spans: [RingCopySpan; 2],
    len: usize,
}

impl RingCopyPlan {
    pub(crate) fn spans(&self) -> &[RingCopySpan] {
        &self.spans[..self.len]
    }
}

pub(crate) fn plan_ring_copy(
    cache_start: usize,
    cache_end: usize,
    window: usize,
    needed_start: usize,
    needed_end: usize,
) -> Result<RingCopyPlan> {
    if window == 0 || cache_start > cache_end || needed_start > needed_end {
        return Err(RingPlanError("ring copy range or window is invalid"));
    }
    if cache_end - cache_start > window || needed_end - needed_start > window {
        return Err(RingPlanError("ring copy range exceeds capacity"));
    }
    let copy_start = needed_start.max(cache_start);
    let copy_end = needed_end.min(cache_end);
    if copy_start >= copy_end {
        return Ok(RingCopyPlan::default());
    }
    let total = copy_end - copy_start;
    let src_slot = copy_start % window;
    let first_count = total.min(window - src_slot);
    let second_count = total - first_count;
    Ok(RingCopyPlan {
        spans: [
            RingCopySpan {
                src_slot,
                dst_slot: copy_start - needed_start,
                slot_count: first_count,
            },
            RingCopySpan {
                src_slot: 0,
                dst_slot: copy_start - needed_start + first_count,
                slot_count: second_count,
            },
        ],
        len: if second_count == 0 { 1 } else { 2 },
    })
}

fn plan_spans(absolute_start: usize, rows: usize, capacity: usize) -> Result<RingSpanPlan> {
    if capacity == 0 || rows == 0 {
        return Err(RingPlanError(
            "ring capacity and row count must be non-zero",
        ));
    }
    // A chunk wider than the ring keeps only its last `capacity` rows: the
    // earlier ones would be overwritten by later rows of the same chunk, and
    // the ring holds a window, so nothing may read them. The plan skips them
    // (`linear_slot` starts at `dropped`) instead of writing them twice.
    let dropped = rows.saturating_sub(capacity);
    let kept = rows - dropped;
    let physical_slot = absolute_start
        .checked_add(dropped)
        .ok_or(RingPlanError("absolute chunk start overflowed"))?
        % capacity;
    let first_count = kept.min(capacity - physical_slot);
    let second_count = kept - first_count;
    Ok(RingSpanPlan {
        spans: [
            RingSpan {
                physical_slot,
                linear_slot: dropped,
                slot_count: first_count,
            },
            RingSpan {
                physical_slot: 0,
                linear_slot: dropped + first_count,
                slot_count: second_count,
            },
        ],
        len: if second_count == 0 { 1 } else { 2 },
    })
}

/// Plan an arbitrary bounded write without advancing sequence state.
pub(crate) fn plan_write_chunk(
    absolute_start: usize,
    rows: usize,
    max_context: usize,
    capacity: usize,
) -> Result<RingSpanPlan> {
    if capacity == 0 || capacity > max_context {
        return Err(RingPlanError("ring capacity is incompatible with context"));
    }
    let absolute_end = absolute_start
        .checked_add(rows)
        .ok_or(RingPlanError("absolute chunk end overflowed"))?;
    if absolute_end > max_context {
        return Err(RingPlanError("chunk exceeds absolute context limit"));
    }
    plan_spans(absolute_start, rows, capacity)
}

#[path = "ring_window_state.rs"]
mod state;
pub(crate) use state::RingState;

#[cfg(test)]
#[path = "ring_window_tests.rs"]
mod tests;
