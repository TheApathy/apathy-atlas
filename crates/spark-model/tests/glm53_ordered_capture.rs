// SPDX-License-Identifier: AGPL-3.0-only

//! RED: real row-copy fixtures for the consuming ordered capture contract.
//! Failure/panic can leave partial device writes; outer PolicyTarget owns drain
//! and poison. This helper may neither publish a cursor nor invent rollback.

#[path = "../src/model/glm53/ordered_capture.rs"]
#[allow(dead_code)]
mod ordered_capture;

use anyhow::{Result, bail};
use ordered_capture::{OrderedCaptureIo, OrderedCapturePlan};
use std::panic::{AssertUnwindSafe, catch_unwind};

const SENTINEL: [u32; 5] = [u32::MAX; 5];

#[derive(Debug, PartialEq, Eq)]
enum Event {
    Project(u32, u32, u64),
    Fence(u64),
}

struct CopyIo {
    captures: [[u32; 5]; 8],
    projected: Vec<[u32; 5]>,
    events: Vec<Event>,
    original_context: u32,
    fail_row: Option<u32>,
    panic_row: Option<u32>,
    fail_fence: bool,
    panic_fence: bool,
    completed: bool,
}

impl CopyIo {
    fn new(start: u32, capacity: u32) -> Self {
        Self {
            captures: std::array::from_fn(|row| {
                std::array::from_fn(|tap| (row * 100 + tap + 1) as u32)
            }),
            projected: vec![SENTINEL; capacity as usize],
            events: vec![],
            original_context: start,
            fail_row: None,
            panic_row: None,
            fail_fence: false,
            panic_fence: false,
            completed: false,
        }
    }
}

impl OrderedCaptureIo for CopyIo {
    fn project_row(
        &mut self,
        capture_row: u32,
        destination_position: u32,
        stream: u64,
    ) -> Result<()> {
        self.events
            .push(Event::Project(capture_row, destination_position, stream));
        // Simulate a submission that can fail after writing, not a no-effect
        // fake. Every rejected-tail row has a distinct tag that must stay out.
        self.projected[destination_position as usize] = self.captures[capture_row as usize];
        if self.panic_row == Some(capture_row) {
            panic!("injected projection panic");
        }
        if self.fail_row == Some(capture_row) {
            bail!("injected projection failure");
        }
        Ok(())
    }

    fn fence(&mut self, stream: u64) -> Result<()> {
        self.events.push(Event::Fence(stream));
        if self.panic_fence {
            panic!("injected fence panic");
        }
        if self.fail_fence {
            bail!("injected fence failure");
        }
        self.completed = true;
        Ok(())
    }
}

fn expected_events(start: u32, rows: u32, stream: u64, fence: bool) -> Vec<Event> {
    let mut events = (0..rows)
        .map(|row| Event::Project(row, start + row, stream))
        .collect::<Vec<_>>();
    if fence {
        events.push(Event::Fence(stream));
    }
    events
}

#[test]
fn partial_and_full_lengths_pool_alignments_and_streams_project_each_row_once_in_order() {
    for rows in 2..=8 {
        for start in (0..32).chain([63, 64, 127, 128, 255, 256, 1023, 2047 - rows]) {
            for stream in [0, 17, u64::MAX] {
                let end = start + rows;
                let plan = OrderedCapturePlan::new(start, rows, end, start, 2047).unwrap();
                let mut io = CopyIo::new(start, 2047);
                let returned = plan.execute(&mut io, stream).unwrap();
                assert_eq!(returned, end);
                assert!(
                    io.completed,
                    "end may be returned only after successful fence"
                );
                assert_eq!(
                    io.original_context, start,
                    "I/O has no publication operation"
                );
                assert_eq!(io.events, expected_events(start, rows, stream, true));
                for (position, value) in io.projected.iter().enumerate() {
                    let expected = if (start as usize..end as usize).contains(&position) {
                        io.captures[position - start as usize]
                    } else {
                        SENTINEL
                    };
                    assert_eq!(*value, expected, "rows={rows} start={start} at={position}");
                }
            }
        }
    }
}

#[test]
fn plan_admission_checks_target_end_context_start_capacity_and_capture_geometry() {
    for rows in 2..=8 {
        assert!(OrderedCapturePlan::new(10, rows, 10 + rows, 10, 10 + rows).is_ok());
        for (target, context, capacity) in [
            (10, 10, 2047),
            (9 + rows, 10, 2047),
            (11 + rows, 10, 2047),
            (10 + rows, 9, 2047),
            (10 + rows, 11, 2047),
            (10 + rows, 10, 9 + rows),
            (10 + rows, 10, 0),
        ] {
            assert!(OrderedCapturePlan::new(10, rows, target, context, capacity).is_err());
        }
    }
    for rows in [0, 1, 9, u32::MAX] {
        assert!(OrderedCapturePlan::new(0, rows, rows, 0, u32::MAX).is_err());
    }
}

#[test]
fn checked_end_rejects_wrap_and_does_not_duplicate_the_runtime_capacity_constant() {
    for rows in 2..=8 {
        assert!(
            OrderedCapturePlan::new(u32::MAX - rows, rows, u32::MAX, u32::MAX - rows, u32::MAX)
                .is_ok()
        );
        assert!(
            OrderedCapturePlan::new(u32::MAX - rows + 1, rows, 0, u32::MAX - rows + 1, u32::MAX)
                .is_err()
        );
        assert!(OrderedCapturePlan::new(u32::MAX, rows, rows - 1, u32::MAX, u32::MAX).is_err());
    }
}

#[test]
fn each_row_failure_stops_without_repeating_later_rows_fencing_or_publishing() {
    for rows in 2..=8 {
        for fail_row in 0..rows {
            let start = 15;
            let plan = OrderedCapturePlan::new(start, rows, start + rows, start, 64).unwrap();
            let mut io = CopyIo::new(start, 64);
            io.fail_row = Some(fail_row);
            let mut published = start;
            let result = plan.execute(&mut io, 29);
            if let Ok(end) = result.as_ref() {
                published = *end;
            }
            assert_eq!(
                result.unwrap_err().to_string(),
                "injected projection failure"
            );
            assert_eq!(published, start);
            assert_eq!(io.events, expected_events(start, fail_row + 1, 29, false));
            assert!(!io.completed);
            assert_eq!(
                io.projected[(start + fail_row) as usize],
                io.captures[fail_row as usize]
            );
            assert_eq!(
                io.projected[(start + fail_row + 1) as usize],
                SENTINEL,
                "failed submission cannot grant authority to a later capture"
            );
        }
    }
}

#[test]
fn final_fence_failure_returns_no_end_and_retains_partial_outputs_for_outer_owner() {
    for rows in 2..=8 {
        let plan = OrderedCapturePlan::new(31, rows, 31 + rows, 31, 64).unwrap();
        let mut io = CopyIo::new(31, 64);
        io.fail_fence = true;
        assert_eq!(
            plan.execute(&mut io, 0).unwrap_err().to_string(),
            "injected fence failure"
        );
        assert_eq!(io.events, expected_events(31, rows, 0, true));
        assert!(!io.completed);
        assert_eq!(io.original_context, 31);
        assert_eq!(
            io.projected[(30 + rows) as usize],
            io.captures[(rows - 1) as usize]
        );
    }
}

#[test]
fn projection_and_fence_panics_propagate_without_hidden_cleanup_or_publication() {
    for rows in 2..=8 {
        for panic_at in 0..=rows {
            let plan = OrderedCapturePlan::new(16, rows, 16 + rows, 16, 64).unwrap();
            let mut io = CopyIo::new(16, 64);
            io.panic_row = (panic_at < rows).then_some(panic_at);
            io.panic_fence = panic_at == rows;
            let result = catch_unwind(AssertUnwindSafe(|| plan.execute(&mut io, 91)));
            assert!(result.is_err());
            assert!(!io.completed);
            assert_eq!(io.original_context, 16);
            assert_eq!(
                io.events,
                expected_events(16, (panic_at + 1).min(rows), 91, panic_at == rows)
            );
        }
    }
}

#[test]
fn plan_is_consumed_and_cannot_be_cloned_into_a_second_capture_publication() {
    let _: fn(OrderedCapturePlan, &mut CopyIo, u64) -> Result<u32> = OrderedCapturePlan::execute;
    // Ambiguous inference if the concrete production plan ever implements
    // Clone; no extra dependency or source-string approximation is needed.
    trait AmbiguousIfClone<A> {
        fn check() {}
    }
    impl<T: ?Sized> AmbiguousIfClone<()> for T {}
    impl<T: ?Sized + Clone> AmbiguousIfClone<u8> for T {}
    let _ = <OrderedCapturePlan as AmbiguousIfClone<_>>::check;
}
