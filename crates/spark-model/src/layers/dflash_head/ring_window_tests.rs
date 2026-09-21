// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn native_state(absolute_len: usize) -> RingState {
    RingState::from_lengths(
        MAX_ABSOLUTE_CONTEXT,
        NATIVE_V3_RING_SLOTS,
        absolute_len,
        absolute_len.min(NATIVE_V3_RING_SLOTS),
    )
    .unwrap()
}

#[test]
fn native_v3_layout_is_160_mib_at_one_million_context() {
    let layout: AccumulatorLayout = plan_accumulator_layout(
        MAX_ABSOLUTE_CONTEXT,
        NATIVE_V3_RING_SLOTS,
        NATIVE_V3_CAPTURE_TAPS,
        NATIVE_V3_HIDDEN_SIZE,
    )
    .unwrap();
    assert_eq!(layout.capacity, 4096);
    assert_eq!(layout.slot_bytes, 40_960);
    assert_eq!(layout.allocation_bytes, 167_772_160);
}

#[test]
fn layout_refuses_zero_and_overflowing_geometry() {
    assert!(plan_accumulator_layout(0, 4096, 8, 2560).is_err());
    assert!(plan_accumulator_layout(4096, 0, 8, 2560).is_err());
    assert!(plan_accumulator_layout(4096, 4096, 0, 2560).is_err());
    assert!(plan_accumulator_layout(usize::MAX, usize::MAX, usize::MAX, 2).is_err());
}

#[test]
fn single_positions_wrap_at_4096_and_8192() {
    let first = native_state(4096);
    assert_eq!(first.slot_for(4095).unwrap(), 4095);
    assert!(first.slot_for(4096).is_err());

    let second = native_state(8192);
    assert!(second.slot_for(4095).is_err());
    assert_eq!(second.slot_for(4096).unwrap(), 0);
    assert_eq!(second.slot_for(8191).unwrap(), 4095);
}

#[test]
fn arbitrary_write_chunk_splits_once_across_wrap() {
    let plan = plan_write_chunk(4094, 5, MAX_ABSOLUTE_CONTEXT, 4096).unwrap();
    assert_eq!(
        plan.spans(),
        &[
            RingSpan {
                physical_slot: 4094,
                linear_slot: 0,
                slot_count: 2,
            },
            RingSpan {
                physical_slot: 0,
                linear_slot: 2,
                slot_count: 3,
            },
        ]
    );
    let boundary = plan_write_chunk(4096, 4096, MAX_ABSOLUTE_CONTEXT, 4096).unwrap();
    assert_eq!(boundary.spans().len(), 1);
    assert_eq!(boundary.spans()[0].slot_count, 4096);
}

#[test]
fn impossible_write_capacity_and_absolute_overflow_fail_closed() {
    assert!(plan_write_chunk(0, 1, 10, 0).is_err());
    assert!(plan_write_chunk(0, 1, 10, 11).is_err());
    assert!(plan_write_chunk(0, 4097, MAX_ABSOLUTE_CONTEXT, 4096).is_err());
    assert!(plan_write_chunk(MAX_ABSOLUTE_CONTEXT, 1, MAX_ABSOLUTE_CONTEXT, 4096).is_err());
    assert!(plan_write_chunk(usize::MAX, 1, usize::MAX, 4096).is_err());
}

#[test]
fn chronological_gather_has_at_most_two_ordered_spans() {
    let state = native_state(4100);
    let plan = state.plan_gather(4094, 6).unwrap();
    assert_eq!(
        plan.spans(),
        &[
            RingCopySpan {
                src_slot: 4094,
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
    assert_eq!(
        plan.spans()
            .iter()
            .map(|span| span.slot_count)
            .sum::<usize>(),
        6
    );
}

#[test]
fn compatibility_copy_intersects_and_preserves_destination_offset() {
    let plan = plan_ring_copy(10, 14, 8, 8, 16).unwrap();
    assert_eq!(
        plan.spans(),
        &[RingCopySpan {
            src_slot: 2,
            dst_slot: 2,
            slot_count: 4,
        }]
    );
    assert!(plan_ring_copy(8, 12, 8, 2, 6).unwrap().spans().is_empty());
    assert!(plan_ring_copy(2, 1, 8, 0, 1).is_err());
    assert!(plan_ring_copy(0, 9, 8, 0, 8).is_err());
}

#[test]
fn gather_rejects_evicted_future_empty_and_oversized_ranges() {
    let state = native_state(8192);
    assert!(state.plan_gather(4095, 1).is_err());
    assert!(state.plan_gather(8192, 1).is_err());
    assert!(state.plan_gather(4096, 0).is_err());
    assert!(state.plan_gather(4096, 4097).is_err());
    assert!(state.plan_gather(usize::MAX, 2).is_err());
}

#[test]
fn accepted_rows_advance_absolute_len_but_bound_resident_len() {
    let state = native_state(4095);
    let append = state.plan_accepted_append(3).unwrap();
    assert_eq!(append.write.spans()[0].physical_slot, 4095);
    assert_eq!(append.write.spans()[0].slot_count, 1);
    assert_eq!(append.write.spans()[1].slot_count, 2);
    assert_eq!(append.next.absolute_len, 4098);
    assert_eq!(append.next.resident_len, 4096);
    assert_eq!(append.next.resident_start(), 2);
}

#[test]
fn append_at_requires_exact_absolute_cursor() {
    let state = native_state(4096);
    assert!(state.plan_append_at(4095, 1).is_err());
    assert!(state.plan_append_at(4097, 1).is_err());
    let append = state.plan_append_at(4096, 1).unwrap();
    assert_eq!(append.write.physical_slot_for(0).unwrap(), 0);
    assert_eq!(append.next.absolute_len, 4097);
}

#[test]
fn sparse_accepted_sources_keep_chronological_wrapped_destinations() {
    let state = native_state(8191);
    let append = state.plan_append_at(8191, 3).unwrap();
    let source_rows = [0usize, 7, 12];
    let mapped: Vec<_> = source_rows
        .into_iter()
        .enumerate()
        .map(|(linear, source)| (source, append.write.physical_slot_for(linear).unwrap()))
        .collect();
    assert_eq!(mapped, [(0, 4095), (7, 0), (12, 1)]);
    assert!(append.write.physical_slot_for(3).is_err());
    assert_eq!(append.next.absolute_len, 8194);
    assert_eq!(append.next.resident_start(), 4098);
}

#[test]
fn exact_one_million_boundary_is_allowed_but_not_crossed() {
    let state = native_state(MAX_ABSOLUTE_CONTEXT - 1);
    let append = state.plan_accepted_append(1).unwrap();
    assert_eq!(append.write.spans()[0].physical_slot, 4095);
    assert_eq!(append.next.absolute_len, MAX_ABSOLUTE_CONTEXT);
    assert_eq!(append.next.resident_len, 4096);
    assert!(append.next.plan_accepted_append(1).is_err());
}

#[test]
fn decimal_one_million_keeps_only_the_latest_4096_rows() {
    let state = native_state(1_000_000);
    assert_eq!(state.absolute_len, 1_000_000);
    assert_eq!(state.resident_len, 4096);
    assert_eq!(state.resident_start(), 995_904);
    assert_eq!(state.slot_for(999_999).unwrap(), 575);
    assert!(state.slot_for(995_903).is_err());
}

#[test]
fn inconsistent_state_is_impossible() {
    assert!(RingState::new(0, 4096).is_err());
    assert!(RingState::new(4096, 4097).is_err());
    assert!(RingState::from_lengths(8192, 4096, 4097, 4095).is_err());
    assert!(RingState::from_lengths(8192, 4096, 8193, 4096).is_err());
}

#[test]
fn capture_offsets_use_eight_taps_inside_each_wrapped_slot() {
    let capture_bytes = NATIVE_V3_HIDDEN_SIZE * size_of::<u16>();
    let slot_bytes = NATIVE_V3_CAPTURE_TAPS * capture_bytes;
    assert_eq!(
        accumulator_capture_offset(4096, 7, 4096, slot_bytes, capture_bytes).unwrap(),
        7 * capture_bytes
    );
    assert_eq!(
        accumulator_capture_offset(8191, 0, 4096, slot_bytes, capture_bytes).unwrap(),
        4095 * slot_bytes
    );
    assert!(accumulator_capture_offset(0, 8, 4096, slot_bytes, capture_bytes).is_err());
    assert!(accumulator_capture_offset(0, 0, 0, slot_bytes, capture_bytes).is_err());
}

#[test]
fn compatibility_copy_wraps_and_fills_the_window_exactly() {
    let wrapped = plan_ring_copy(14, 20, 8, 14, 20).unwrap();
    assert_eq!(
        wrapped.spans(),
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
    let exact = plan_ring_copy(16, 24, 8, 16, 24).unwrap();
    assert_eq!(
        exact.spans(),
        &[RingCopySpan {
            src_slot: 0,
            dst_slot: 0,
            slot_count: 8,
        }]
    );
}

#[test]
fn compatibility_copy_rejects_zero_window_and_oversized_requests() {
    assert!(plan_ring_copy(0, 1, 0, 0, 1).is_err());
    assert!(plan_ring_copy(0, 1, 8, 2, 1).is_err());
    assert!(plan_ring_copy(0, 8, 8, 0, 9).is_err());
}
