// SPDX-License-Identifier: AGPL-3.0-only

//! Pure indexing contract for DFlash's absolute-position circular caches.
//!
//! Keeping this arithmetic independent from GPU I/O makes wrap behavior
//! testable before a ring-native attention kernel consumes the same layout.

use anyhow::{Result, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AccumulatorLayout {
    pub capacity: usize,
    pub slot_bytes: usize,
    pub allocation_bytes: usize,
}

/// Size the target-hidden accumulator by the drafter's retained local window,
/// not by the target model's full sequence ceiling.
pub(super) fn plan_accumulator_layout(
    max_seq_len: usize,
    ctx_window: usize,
    capture_layers: usize,
    target_hidden_size: usize,
) -> Result<AccumulatorLayout> {
    let capacity = max_seq_len.min(ctx_window);
    if capacity == 0 {
        bail!("DFlash context accumulator capacity must be non-zero");
    }
    let slot_bytes = capture_layers
        .checked_mul(target_hidden_size)
        .and_then(|n| n.checked_mul(size_of::<u16>()))
        .ok_or_else(|| anyhow::anyhow!("DFlash context slot byte size overflow"))?;
    if slot_bytes == 0 {
        bail!("DFlash context accumulator slot must be non-empty");
    }
    let allocation_bytes = capacity
        .checked_mul(slot_bytes)
        .ok_or_else(|| anyhow::anyhow!("DFlash context allocation size overflow"))?;
    Ok(AccumulatorLayout {
        capacity,
        slot_bytes,
        allocation_bytes,
    })
}

/// Resolve one captured layer slice within the circular target-hidden
/// accumulator. `absolute_position` remains logical while the returned byte
/// offset is always within the bounded physical allocation.
pub(crate) fn accumulator_capture_offset(
    absolute_position: usize,
    capture_slot: usize,
    capacity: usize,
    slot_bytes: usize,
    capture_bytes: usize,
) -> Result<usize> {
    if capacity == 0 {
        bail!("DFlash context accumulator capacity must be non-zero");
    }
    if slot_bytes == 0 || capture_bytes == 0 {
        bail!("DFlash context accumulator capture sizes must be non-zero");
    }

    let capture_offset = capture_slot
        .checked_mul(capture_bytes)
        .ok_or_else(|| anyhow::anyhow!("DFlash capture slot byte offset overflow"))?;
    let capture_end = capture_offset
        .checked_add(capture_bytes)
        .ok_or_else(|| anyhow::anyhow!("DFlash capture slot byte end overflow"))?;
    if capture_end > slot_bytes {
        bail!(
            "DFlash capture slice {capture_offset}..{capture_end} exceeds slot size {slot_bytes}"
        );
    }

    let physical_slot = absolute_position % capacity;
    physical_slot
        .checked_mul(slot_bytes)
        .and_then(|slot_base| slot_base.checked_add(capture_offset))
        .ok_or_else(|| anyhow::anyhow!("DFlash context accumulator byte offset overflow"))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct RingCopySpan {
    pub src_slot: usize,
    pub dst_slot: usize,
    pub slot_count: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct RingCopyPlan {
    spans: [RingCopySpan; 2],
    len: usize,
}

impl RingCopyPlan {
    pub fn spans(&self) -> &[RingCopySpan] {
        &self.spans[..self.len]
    }
}

/// Intersect an absolute requested range with the valid cache range, then
/// express the result as at most two contiguous ring spans.
pub(super) fn plan_ring_copy(
    cache_start: usize,
    cache_end: usize,
    window: usize,
    needed_start: usize,
    needed_end: usize,
) -> Result<RingCopyPlan> {
    if window == 0 {
        bail!("DFlash ring window must be non-zero");
    }
    if cache_start > cache_end {
        bail!("DFlash cache range is reversed: {cache_start}..{cache_end}");
    }
    if needed_start > needed_end {
        bail!("DFlash requested range is reversed: {needed_start}..{needed_end}");
    }
    if cache_end - cache_start > window {
        bail!(
            "DFlash cache range {} exceeds ring window {window}",
            cache_end - cache_start
        );
    }
    if needed_end - needed_start > window {
        bail!(
            "DFlash requested range {} exceeds ring window {window}",
            needed_end - needed_start
        );
    }

    let copy_start = needed_start.max(cache_start);
    let copy_end = needed_end.min(cache_end);
    if copy_start >= copy_end {
        return Ok(RingCopyPlan::default());
    }

    let total = copy_end - copy_start;
    let src_slot = copy_start % window;
    let first_count = total.min(window - src_slot);
    let first = RingCopySpan {
        src_slot,
        dst_slot: copy_start - needed_start,
        slot_count: first_count,
    };
    let second_count = total - first_count;
    let second = RingCopySpan {
        src_slot: 0,
        dst_slot: first.dst_slot + first_count,
        slot_count: second_count,
    };

    Ok(RingCopyPlan {
        spans: [first, second],
        len: if second_count == 0 { 1 } else { 2 },
    })
}

#[cfg(test)]
mod tests {
    use super::{
        RingCopySpan, accumulator_capture_offset, plan_accumulator_layout, plan_ring_copy,
    };

    #[test]
    fn accumulator_capture_offset_wraps_at_capacity() {
        let capacity = 4096;
        let capture_bytes = 8192 * size_of::<u16>();
        let slot_bytes = 5 * capture_bytes;

        assert_eq!(
            accumulator_capture_offset(4095, 0, capacity, slot_bytes, capture_bytes).unwrap(),
            4095 * slot_bytes
        );
        assert_eq!(
            accumulator_capture_offset(4096, 0, capacity, slot_bytes, capture_bytes).unwrap(),
            0
        );
        assert_eq!(
            accumulator_capture_offset(4096, 4, capacity, slot_bytes, capture_bytes).unwrap(),
            4 * capture_bytes
        );
    }

    #[test]
    fn invalid_or_overflowing_capture_offset_fails_closed() {
        assert!(accumulator_capture_offset(0, 0, 0, 10, 2).is_err());
        assert!(accumulator_capture_offset(0, 0, 1, 0, 2).is_err());
        assert!(accumulator_capture_offset(0, 0, 1, 2, 0).is_err());
        assert!(accumulator_capture_offset(0, 1, 1, 2, 2).is_err());
        assert!(accumulator_capture_offset(0, usize::MAX, 1, 2, 2).is_err());
        assert!(accumulator_capture_offset(usize::MAX - 1, 0, usize::MAX, 3, 1).is_err());
    }

    #[test]
    fn one_million_context_retains_only_the_local_draft_window() {
        let layout = plan_accumulator_layout(1_048_576, 4096, 5, 8192).unwrap();
        assert_eq!(layout.capacity, 4096);
        assert_eq!(layout.slot_bytes, 81_920);
        assert_eq!(layout.allocation_bytes, 335_544_320);
    }

    #[test]
    fn invalid_or_overflowing_accumulator_layout_fails_closed() {
        assert!(plan_accumulator_layout(0, 4096, 5, 8192).is_err());
        assert!(plan_accumulator_layout(4096, 4096, 0, 8192).is_err());
        assert!(plan_accumulator_layout(usize::MAX, usize::MAX, usize::MAX, 2).is_err());
    }

    #[test]
    fn empty_intersection_has_no_spans() {
        assert!(plan_ring_copy(8, 12, 8, 2, 6).unwrap().spans().is_empty());
    }

    #[test]
    fn unwrapped_overlap_preserves_requested_destination_offset() {
        let plan = plan_ring_copy(10, 14, 8, 8, 16).unwrap();
        assert_eq!(
            plan.spans(),
            &[RingCopySpan {
                src_slot: 2,
                dst_slot: 2,
                slot_count: 4,
            }]
        );
    }

    #[test]
    fn wrapped_window_splits_at_physical_slot_zero() {
        let plan = plan_ring_copy(14, 20, 8, 14, 20).unwrap();
        assert_eq!(
            plan.spans(),
            &[
                RingCopySpan {
                    src_slot: 6,
                    dst_slot: 0,
                    slot_count: 2,
                },
                RingCopySpan {
                    src_slot: 0,
                    dst_slot: 2,
                    slot_count: 4,
                },
            ]
        );
    }

    #[test]
    fn full_window_at_exact_boundary_is_one_span() {
        let plan = plan_ring_copy(16, 24, 8, 16, 24).unwrap();
        assert_eq!(
            plan.spans(),
            &[RingCopySpan {
                src_slot: 0,
                dst_slot: 0,
                slot_count: 8,
            }]
        );
    }

    #[test]
    fn invalid_ranges_and_oversized_windows_fail_closed() {
        assert!(plan_ring_copy(0, 1, 0, 0, 1).is_err());
        assert!(plan_ring_copy(2, 1, 8, 0, 1).is_err());
        assert!(plan_ring_copy(0, 1, 8, 2, 1).is_err());
        assert!(plan_ring_copy(0, 9, 8, 0, 8).is_err());
        assert!(plan_ring_copy(0, 8, 8, 0, 9).is_err());
    }
}
