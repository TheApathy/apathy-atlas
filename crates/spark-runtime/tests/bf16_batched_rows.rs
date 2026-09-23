// SPDX-License-Identifier: AGPL-3.0-only
//! Real production admission/lifecycle with recorded external operations.
#[allow(dead_code)]
#[path = "../src/cublaslt/diagnostic_contract.rs"]
mod diagnostic_contract;
#[path = "bf16_serial_rows/recording.rs"]
mod recording;
#[path = "../src/cublaslt/serial_rows_contract.rs"]
mod serial_rows_contract;
#[path = "../src/cublaslt/serial_rows_driver.rs"]
mod serial_rows_driver;
#[path = "../src/cublaslt/strided_rows_contract.rs"]
mod strided_rows_contract;

use recording::{Event, Recording, request, workspace};
use serial_rows_contract::Orientation;
use serial_rows_driver::{run_batched_rows, run_serial_rows};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn ready() -> Recording {
    Recording {
        batch_caps: Some([1, 2, 2, 2, 2]),
        ..Default::default()
    }
}
fn closed(io: &Recording) {
    assert_eq!(
        io.destroyed(),
        io.created.iter().copied().rev().collect::<Vec<_>>()
    );
    assert!(io.live.is_empty());
}

#[test]
fn batch_selects_one_original_algorithm_and_one_matmul_for_each_width_and_orientation() {
    for rows in 2..=8 {
        for orientation in [Orientation::Nk, Orientation::Kn] {
            let req = request(rows, orientation);
            let mut io = ready();
            run_batched_rows(&mut io, req, workspace()).unwrap();
            assert_eq!(io.queried(), 1);
            assert_eq!(io.matmuls(), 1);
            let (set, prepared_request, call, words) = match io.events[9] {
                Event::PrepareBatch(set, req, call, words) => (set, req, call, words),
                ref other => panic!("missing batch preparation: {other:?}"),
            };
            assert_eq!(prepared_request, req);
            assert_eq!(
                (call.act, call.weight, call.out),
                (req.act.address, req.weight.address, req.out.address)
            );
            assert_eq!(words, [0x100; 8]);
            assert_eq!(io.events[10], Event::Matmul(set, call, words));
            closed(&io);
            let mut off = ready();
            run_serial_rows(&mut off, req, workspace()).unwrap();
            assert_eq!(off.queried(), rows as usize);
            assert_eq!(off.matmuls(), rows as usize);
            assert!(
                !off.events
                    .iter()
                    .any(|event| matches!(event, Event::PrepareBatch(..)))
            );
            closed(&off);
        }
    }
}

#[test]
fn unsupported_or_bad_alignment_never_enqueues_or_falls_back() {
    for caps in [
        None,
        Some([0, 2, 2, 2, 2]),
        Some([1, 0, 2, 2, 2]),
        Some([1, 3, 2, 2, 2]),
        Some([1, 2, 512, 2, 2]),
        Some([1, 2, 2, 256, 2]),
        Some([1, 2, 2, 2, 256]),
    ] {
        let mut io = Recording {
            batch_caps: caps,
            ..Default::default()
        };
        assert!(run_batched_rows(&mut io, request(8, Orientation::Nk), workspace()).is_err());
        assert_eq!(io.queried(), 1);
        assert_eq!(io.matmuls(), 0);
        closed(&io);
    }
}

#[test]
fn failures_and_panics_at_every_external_stage_cleanup_without_retry() {
    for index in 0..=10 {
        let mut io = Recording {
            fail_at: Some(index),
            ..ready()
        };
        assert!(run_batched_rows(&mut io, request(8, Orientation::Nk), workspace()).is_err());
        assert!(
            io.events[index + 1..]
                .iter()
                .all(|e| matches!(e, Event::Destroy(..)))
        );
        assert!(io.queried() <= 1 && io.matmuls() <= 1);
        closed(&io);
        let mut io = Recording {
            panic_at: Some(index),
            ..ready()
        };
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _ = run_batched_rows(&mut io, request(8, Orientation::Nk), workspace());
            }))
            .is_err()
        );
        assert!(
            io.events[index + 1..]
                .iter()
                .all(|e| matches!(e, Event::Destroy(..)))
        );
        closed(&io);
    }
}

#[test]
fn existing_workspace_admission_and_cleanup_errors_are_preserved() {
    let mut io = ready();
    let mut ws = workspace();
    ws.bytes -= 1;
    assert!(run_batched_rows(&mut io, request(8, Orientation::Nk), ws).is_err());
    assert!(io.events.is_empty());
    let mut io = Recording {
        destroy_errors: vec![5, 3],
        ..ready()
    };
    let error = run_batched_rows(&mut io, request(8, Orientation::Nk), workspace()).unwrap_err();
    assert!(error.primary.is_none());
    assert_eq!(error.cleanup.len(), 2);
    assert_eq!(io.matmuls(), 1);
    assert_eq!(io.destroyed().len(), 5);
}

#[test]
fn native_ffi_checks_capabilities_layouts_and_algorithm_without_a_new_heuristic() {
    let ffi = include_str!("../src/cublaslt/serial_rows_batch_ffi.rs");
    let admit = ffi.find("admit_strided_m1(").unwrap();
    let set = ffi.find("cublasLtMatrixLayoutSetAttribute(").unwrap();
    let check = ffi.rfind("cublasLtMatmulAlgoCheck(").unwrap();
    // The first setter text is the declaration; the final invocation follows admission.
    assert!(set < admit && admit < ffi.rfind("cublasLtMatrixLayoutSetAttribute(").unwrap());
    assert!(admit < check);
    let compact = ffi.split_whitespace().collect::<String>();
    assert!(compact.contains("checked.admit(1,call.workspace_bytes)"));
    assert!(!ffi.contains("AlgoGetHeuristic"));
    assert!(!ffi.contains("cublasLtMatmul("));
}
