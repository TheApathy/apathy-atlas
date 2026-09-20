// SPDX-License-Identifier: AGPL-3.0-only
//! CPU admission/submission tests, not CUDA arithmetic or completion evidence.
#[allow(dead_code)]
#[path = "../examples/glm53_projection_probe/batchm_contract.rs"]
mod batchm_contract;
#[allow(dead_code)]
#[path = "../examples/glm53_projection_probe/gemv_contract.rs"]
mod gemv_contract;
use anyhow::{Result, bail};
use batchm_contract::{
    BatchMIo, BatchMKind, BatchMLaunch, BatchMPlan, Schedule, validate_kernel_rows,
};
use gemv_contract::Span;

const INPUT_ROW: usize = 4096 * 2;
const OUTPUT_ROW: usize = 1024 * 2;
const SHAPES: [u32; 14] = [1, 2, 3, 8, 9, 15, 16, 17, 31, 32, 33, 256, 1024, 2047];
fn owners() -> [Span; 3] {
    [
        Span {
            ptr: 0x0100_0000,
            bytes: 2048 * INPUT_ROW,
        },
        Span {
            ptr: 0x0300_0000,
            bytes: 1024 * 4096 * 2,
        },
        Span {
            ptr: 0x0400_0000,
            bytes: 2047 * OUTPUT_ROW,
        },
    ]
}
fn plan(rows: u32, source_row: u32, p: [Span; 3], handles: [u64; 3]) -> Result<BatchMPlan> {
    BatchMPlan::new(rows, source_row, 2048, p[0], p[1], p[2], handles)
}
fn check_cover(plan: &BatchMPlan, mode: Schedule, source_row: u32, rows: u32) {
    let p = owners();
    let mut cursor = 0;
    for launch in plan.launches(mode).unwrap() {
        assert_eq!(launch.output_row, cursor);
        assert_eq!(launch.source_row, source_row + cursor);
        assert_eq!(
            launch.input.ptr,
            p[0].ptr + u64::from(source_row + cursor) * INPUT_ROW as u64
        );
        assert_eq!(
            launch.output.ptr,
            p[2].ptr + u64::from(cursor) * OUTPUT_ROW as u64
        );
        assert_eq!(launch.input.bytes, launch.rows as usize * INPUT_ROW);
        assert_eq!(launch.output.bytes, launch.rows as usize * OUTPUT_ROW);
        assert_eq!(launch.weight, p[1]);
        validate_kernel_rows(launch.rows).unwrap();
        match launch.kind {
            BatchMKind::Single => assert_eq!(launch.rows, 1),
            BatchMKind::Pair => assert_eq!(launch.rows, 2),
            BatchMKind::Wide => assert!(launch.rows <= 16),
        }
        cursor += launch.rows;
    }
    assert_eq!(cursor, rows);
}

#[test]
fn direct_kernel_exercises_every_requested_small_shape_not_the_pair_fallback() {
    for rows in [1, 2, 3, 8, 9, 15, 16] {
        for start in [0, 1, 7, 17] {
            let plan = plan(rows, start, owners(), [1, 2, 3]).unwrap();
            let launches = plan.launches(Schedule::BatchMDirect).unwrap();
            assert_eq!(launches.len(), 1);
            assert_eq!(launches[0].kind, BatchMKind::Wide);
            assert_eq!(launches[0].rows, rows);
            check_cover(&plan, Schedule::BatchMDirect, start, rows);
        }
    }
}

#[test]
fn full_partitions_and_reference_have_no_holes_overlap_or_source_offset_loss() {
    for rows in SHAPES {
        for start in [0, 1] {
            let plan = plan(rows, start, owners(), [1, 2, 3]).unwrap();
            check_cover(&plan, Schedule::PairReference, start, rows);
            check_cover(&plan, Schedule::BatchMPartitioned, start, rows);
            assert!(
                plan.launches(Schedule::PairReference)
                    .unwrap()
                    .iter()
                    .all(|v| v.rows <= 2)
            );
        }
    }
}

#[test]
fn partition_policy_keeps_short_pair_tails_and_uses_only_bounded_wide_calls() {
    for (rows, expected) in [
        (3, vec![2, 1]),
        (8, vec![2, 2, 2, 2]),
        (9, vec![9]),
        (17, vec![16, 1]),
        (31, vec![16, 15]),
        (32, vec![16, 16]),
        (33, vec![16, 16, 1]),
    ] {
        let launches = plan(rows, 0, owners(), [1, 2, 3])
            .unwrap()
            .launches(Schedule::BatchMPartitioned)
            .unwrap();
        assert_eq!(
            launches.iter().map(|v| v.rows).collect::<Vec<_>>(),
            expected
        );
        for launch in launches {
            assert_eq!(launch.kind == BatchMKind::Wide, launch.rows > 8);
        }
    }
    let launches = plan(2047, 1, owners(), [1, 2, 3])
        .unwrap()
        .launches(Schedule::BatchMPartitioned)
        .unwrap();
    assert_eq!(launches.len(), 128);
    assert!(launches[..127].iter().all(|v| v.rows == 16));
    assert_eq!(launches[127].rows, 15);
}

#[test]
fn hard_kernel_m16_and_overall_row_caps_are_distinct() {
    for rows in 1..=16 {
        validate_kernel_rows(rows).unwrap();
    }
    for rows in [0, 17, 2047, u32::MAX] {
        assert!(validate_kernel_rows(rows).is_err());
    }
    for rows in [17, 256, 2047] {
        assert!(
            plan(rows, 0, owners(), [1, 2, 3])
                .unwrap()
                .launches(Schedule::BatchMDirect)
                .is_err()
        );
    }
    for (rows, start) in [(0, 0), (2048, 0), (2047, 2), (1, 2048), (1, u32::MAX)] {
        assert!(plan(rows, start, owners(), [1, 2, 3]).is_err());
    }
    let p = owners();
    for input_rows in [0, 2049, u32::MAX] {
        assert!(BatchMPlan::new(1, 0, input_rows, p[0], p[1], p[2], [1, 2, 3]).is_err());
    }
}

#[test]
fn every_complete_owner_and_handle_is_admitted_before_any_submission() {
    for index in 0..3 {
        let mut p = owners();
        p[index].bytes = [2048 * INPUT_ROW, 1024 * 4096 * 2, OUTPUT_ROW][index] - 1;
        assert!(plan(1, 0, p, [1, 2, 3]).is_err(), "short {index}");
        let mut p = owners();
        p[index].ptr = 0;
        assert!(plan(1, 0, p, [1, 2, 3]).is_err(), "null {index}");
        let mut p = owners();
        p[index].ptr += [2, 2, 1][index];
        assert!(plan(1, 0, p, [1, 2, 3]).is_err(), "alignment {index}");
        let mut handles = [1, 2, 3];
        handles[index] = 0;
        assert!(plan(1, 0, owners(), handles).is_err(), "handle {index}");
    }
}

#[test]
fn owner_ends_checked_before_slicing_and_all_alias_pairs_rejected() {
    for index in 0..3 {
        let mut p = owners();
        p[index].ptr = u64::MAX - 15;
        assert!(plan(1, 0, p, [1, 2, 3]).is_err(), "wrapped {index}");
        let mut p = owners();
        p[index].bytes = usize::MAX;
        assert!(plan(1, 0, p, [1, 2, 3]).is_err(), "isize {index}");
        for other in index + 1..3 {
            let mut p = owners();
            p[other].ptr = p[index].ptr + 16;
            assert!(plan(1, 0, p, [1, 2, 3]).is_err(), "alias {index}/{other}");
        }
    }
}

#[derive(Default)]
struct RecordingIo {
    attempted: Vec<u32>,
    fail_at: Option<usize>,
    panic_at: Option<usize>,
}
impl BatchMIo for RecordingIo {
    fn launch(&mut self, launch: BatchMLaunch) -> Result<()> {
        let index = self.attempted.len();
        self.attempted.push(launch.output_row);
        assert_ne!(self.panic_at, Some(index), "injected enqueue panic");
        if self.fail_at == Some(index) {
            bail!("injected enqueue error");
        }
        Ok(())
    }
}

#[test]
fn submission_stops_on_first_error_and_never_retries_or_claims_completion() {
    let plan = plan(33, 1, owners(), [1, 2, 3]).unwrap();
    for failed in 0..3 {
        let mut io = RecordingIo {
            fail_at: Some(failed),
            ..Default::default()
        };
        assert!(plan.execute(Schedule::BatchMPartitioned, &mut io).is_err());
        assert_eq!(io.attempted, [0, 16, 32][..=failed]);
    }
    let mut io = RecordingIo::default();
    assert!(plan.execute(Schedule::BatchMDirect, &mut io).is_err());
    assert!(
        io.attempted.is_empty(),
        "invalid whole schedule reached I/O"
    );
    plan.execute(Schedule::BatchMPartitioned, &mut io).unwrap();
    assert_eq!(io.attempted, [0, 16, 32]);
}

#[test]
fn enqueue_panic_propagates_without_submitting_later_rows() {
    let plan = plan(33, 0, owners(), [1, 2, 3]).unwrap();
    let mut io = RecordingIo {
        panic_at: Some(1),
        ..Default::default()
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        plan.execute(Schedule::BatchMPartitioned, &mut io)
    }));
    assert!(result.is_err());
    assert_eq!(io.attempted, [0, 16]);
}
