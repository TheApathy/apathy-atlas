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

/// Replay `plan`'s spans into a simulated ring whose slots record the
/// absolute position written there, the same way the capture loop in
/// `impl_b3.rs` derives `absolute_position = chunk_start + linear_slot`.
fn replay(ring: &mut [Option<usize>], chunk_start: usize, plan: &RingSpanPlan) {
    for span in plan.spans() {
        for local in 0..span.slot_count {
            ring[span.physical_slot + local] = Some(chunk_start + span.linear_slot + local);
        }
    }
}

/// Fill a ring from absolute 0 in chunks of `chunk` rows; return the ring and
/// the final state.
fn fill_in_chunks(total: usize, chunk: usize, capacity: usize) -> (Vec<Option<usize>>, RingState) {
    let mut ring = vec![None; capacity];
    let mut state = RingState::new(MAX_ABSOLUTE_CONTEXT, capacity).unwrap();
    let mut start = 0;
    while start < total {
        let rows = chunk.min(total - start);
        let append = state.plan_append_at(start, rows).unwrap();
        replay(&mut ring, start, &append.write);
        state = append.next;
        start += rows;
    }
    (ring, state)
}

#[test]
fn chunk_wider_than_ring_keeps_only_its_last_capacity_rows() {
    let plan = plan_write_chunk(0, 8192, MAX_ABSOLUTE_CONTEXT, 4096).unwrap();
    assert_eq!(
        plan.spans(),
        &[RingSpan {
            physical_slot: 0,
            linear_slot: 4096,
            slot_count: 4096,
        }]
    );
    // Dropped rows are not addressable; kept rows land at absolute % capacity.
    assert!(plan.physical_slot_for(4095).is_err());
    assert_eq!(plan.physical_slot_for(4096).unwrap(), 0);
    assert_eq!(plan.physical_slot_for(8191).unwrap(), 4095);
}

#[test]
fn wide_chunk_tail_wraps_at_its_absolute_position() {
    // Start 1000, 6000 rows: keep absolute [2904, 7000) = slots 2904..4095
    // then 0..2903.
    let plan = plan_write_chunk(1000, 6000, MAX_ABSOLUTE_CONTEXT, 4096).unwrap();
    assert_eq!(
        plan.spans(),
        &[
            RingSpan {
                physical_slot: 2904,
                linear_slot: 1904,
                slot_count: 1192,
            },
            RingSpan {
                physical_slot: 0,
                linear_slot: 3096,
                slot_count: 2904,
            },
        ]
    );
    let kept: usize = plan.spans().iter().map(|s| s.slot_count).sum();
    assert_eq!(kept, 4096);
    for linear in 1904..6000 {
        assert_eq!(plan.physical_slot_for(linear).unwrap(), (1000 + linear) % 4096);
    }
}

#[test]
fn one_wide_chunk_leaves_the_ring_identical_to_capacity_sized_chunks() {
    for total in [4097, 6000, 7000, 8192, 12288] {
        let (wide, wide_state) = fill_in_chunks(total, 8192, 4096);
        let (narrow, narrow_state) = fill_in_chunks(total, 4096, 4096);
        assert_eq!(wide, narrow, "total={total}");
        assert_eq!(wide_state, narrow_state, "total={total}");
        // Every slot is resident and holds the position `slot_for` promises.
        for pos in wide_state.resident_start()..wide_state.absolute_len {
            assert_eq!(wide[wide_state.slot_for(pos).unwrap()], Some(pos));
        }
    }
    // A wide chunk after a narrow one (the second turn of a chat).
    let (mut ring, state) = fill_in_chunks(300, 4096, 4096);
    let append = state.plan_append_at(300, 8000).unwrap();
    replay(&mut ring, 300, &append.write);
    let (reference, reference_state) = fill_in_chunks(8300, 4096, 4096);
    assert_eq!(ring, reference);
    assert_eq!(append.next, reference_state);
}

#[test]
fn head_keeping_plan_is_caught_by_the_ring_replay() {
    // Control: a plan that keeps the FIRST capacity rows (the obvious wrong
    // fix) must fail the same comparison the real plan passes.
    let wrong = RingSpanPlan {
        spans: [
            RingSpan {
                physical_slot: 0,
                linear_slot: 0,
                slot_count: 4096,
            },
            RingSpan::default(),
        ],
        len: 1,
    };
    let mut ring = vec![None; 4096];
    replay(&mut ring, 0, &wrong);
    let (reference, _) = fill_in_chunks(8192, 4096, 4096);
    assert_ne!(ring, reference);
}

/// Chunked prefill protocol: every capture layer of a chunk plans with
/// `plan_append_at(chunk_start)`, then the chunk's end advances the ring.
fn chunked_prefill(total: usize, chunk: usize, advance_every_chunk: bool) -> Result<RingState> {
    let mut state = RingState::new(MAX_ABSOLUTE_CONTEXT, 4096)?;
    let mut start = 0;
    while start < total {
        let rows = chunk.min(total - start);
        for _capture_layer in 0..8 {
            state.plan_append_at(start, rows)?;
        }
        let last = start + rows == total;
        if advance_every_chunk || last {
            if let Some(next) = state.advanced_after_chunk(start, rows)? {
                state = next;
            }
            // The last-chunk finalizer repeats the advance: a no-op.
            assert_eq!(state.advanced_after_chunk(start, rows)?, None);
        }
        start += rows;
    }
    Ok(state)
}

#[test]
fn multi_chunk_prefill_needs_the_ring_advanced_after_every_chunk() {
    for (total, chunk) in [(7000, 4096), (10000, 8192), (12288, 2048), (4096, 4096)] {
        let state = chunked_prefill(total, chunk, true).unwrap();
        assert_eq!(state.absolute_len, total);
        assert_eq!(state.resident_len, total.min(4096));
    }
    // Control: advancing only after the last chunk (the old behaviour) fails
    // chunk 2's capture with the error seen in production.
    let err = chunked_prefill(7000, 4096, false).unwrap_err();
    assert_eq!(err.to_string(), "append start does not match absolute cursor");
    // A single chunk never needed the per-chunk advance.
    assert!(chunked_prefill(3000, 4096, false).is_ok());
}

#[test]
fn prefill_forward_advances_the_ring_after_every_chunk() {
    let forward = include_str!("../../model/trait_impl/prefill_b/forward_layers.rs");
    let body = &forward[forward.find("fn prefill_b_forward_layers(").unwrap()..];
    assert!(
        body.contains("self.update_dflash_ctx_len_after_prefill(seq, effective_seq_len_start, proc_count)?"),
        "prefill_b_forward_layers must advance the DFlash ring after each chunk"
    );
    let prefill = include_str!("../../model/impl_b3.rs");
    assert!(prefill.contains(".advanced_after_chunk(chunk_start, proc_count)?"));
}
