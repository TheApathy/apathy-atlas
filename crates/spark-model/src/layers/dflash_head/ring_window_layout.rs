// SPDX-License-Identifier: AGPL-3.0-only

//! Allocation and per-capture byte geometry for the DFlash ring.

use super::{Result, RingPlanError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AccumulatorLayout {
    pub capacity: usize,
    pub slot_bytes: usize,
    pub allocation_bytes: usize,
}

pub(crate) fn plan_accumulator_layout(
    max_context: usize,
    requested_capacity: usize,
    capture_taps: usize,
    hidden_size: usize,
) -> Result<AccumulatorLayout> {
    if max_context == 0 || requested_capacity == 0 {
        return Err(RingPlanError("ring context and capacity must be non-zero"));
    }
    let capacity = max_context.min(requested_capacity);
    let slot_bytes = capture_taps
        .checked_mul(hidden_size)
        .and_then(|elements| elements.checked_mul(size_of::<u16>()))
        .filter(|bytes| *bytes != 0)
        .ok_or(RingPlanError("ring slot byte size is zero or overflowed"))?;
    let allocation_bytes = capacity
        .checked_mul(slot_bytes)
        .ok_or(RingPlanError("ring allocation byte size overflowed"))?;
    Ok(AccumulatorLayout {
        capacity,
        slot_bytes,
        allocation_bytes,
    })
}

pub(crate) fn accumulator_capture_offset(
    absolute_position: usize,
    capture_slot: usize,
    capacity: usize,
    slot_bytes: usize,
    capture_bytes: usize,
) -> Result<usize> {
    if capacity == 0 || slot_bytes == 0 || capture_bytes == 0 {
        return Err(RingPlanError("capture geometry must be non-zero"));
    }
    let capture_offset = capture_slot
        .checked_mul(capture_bytes)
        .ok_or(RingPlanError("capture offset overflowed"))?;
    let capture_end = capture_offset
        .checked_add(capture_bytes)
        .ok_or(RingPlanError("capture end overflowed"))?;
    if capture_end > slot_bytes {
        return Err(RingPlanError("capture exceeds one ring slot"));
    }
    (absolute_position % capacity)
        .checked_mul(slot_bytes)
        .and_then(|base| base.checked_add(capture_offset))
        .ok_or(RingPlanError("physical capture offset overflowed"))
}
