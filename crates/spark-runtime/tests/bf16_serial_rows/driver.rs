// SPDX-License-Identifier: AGPL-3.0-only
use super::recording::{Event, HeuristicOverride, Recording, WORKSPACE_BYTES, request, workspace};
use super::serial_rows_contract::{ByteSpan, MatrixLayout, Orientation, RowCall};
use super::serial_rows_driver::{DescriptorSet, Failure, ResourceKind, run_serial_rows};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn set() -> DescriptorSet<u64> {
    DescriptorSet {
        desc: 1,
        a: 2,
        b: 3,
        c: 4,
        d: 4,
        pref: 5,
    }
}
fn cleanup(io: &Recording) {
    assert_eq!(
        io.destroyed(),
        io.created.iter().copied().rev().collect::<Vec<_>>()
    );
    assert!(io.live.is_empty());
}
fn expected_setup(orientation: Orientation) -> Vec<Event> {
    let (ta, a) = match orientation {
        Orientation::Nk => (
            1,
            MatrixLayout {
                dtype: 14,
                rows: 128,
                cols: 64,
                ld: 128,
            },
        ),
        Orientation::Kn => (
            0,
            MatrixLayout {
                dtype: 14,
                rows: 64,
                cols: 128,
                ld: 64,
            },
        ),
    };
    vec![
        Event::CreateDesc(68, 0),
        Event::SetDesc(1, 3, ta),
        Event::SetDesc(1, 4, 0),
        Event::CreateLayout(a),
        Event::CreateLayout(MatrixLayout {
            dtype: 14,
            rows: 128,
            cols: 1,
            ld: 128,
        }),
        Event::CreateLayout(MatrixLayout {
            dtype: 14,
            rows: 64,
            cols: 1,
            ld: 64,
        }),
        Event::CreatePref,
        Event::SetPref(5, 1, WORKSPACE_BYTES),
    ]
}

#[test]
fn exact_original_attributes_order_m1_calls_and_one_resource_set_for_every_width() {
    for rows in 2..=8 {
        for orientation in [Orientation::Nk, Orientation::Kn] {
            let req = request(rows, orientation);
            let mut io = Recording::default();
            run_serial_rows(&mut io, req, workspace()).unwrap();
            assert_eq!(&io.events[..8], expected_setup(orientation));
            assert_eq!(io.created.len(), 5);
            for row in 0..rows as usize {
                assert_eq!(io.events[8 + row * 2], Event::Heuristic(set(), 1));
                assert_eq!(
                    io.events[9 + row * 2],
                    Event::Matmul(
                        set(),
                        RowCall {
                            act: req.act.address + row as u64 * 256,
                            weight: req.weight.address,
                            out: req.out.address + row as u64 * 128,
                            workspace: workspace().address,
                            workspace_bytes: WORKSPACE_BYTES,
                            stream: 17,
                            alpha_bits: 1.0f32.to_bits(),
                            beta_bits: 0.0f32.to_bits(),
                        },
                        [0x100 + row as u64; 8]
                    )
                );
            }
            assert_eq!(io.events.len(), 8 + rows as usize * 2 + 5);
            assert_eq!(io.queried(), rows as usize);
            assert_eq!(io.matmuls(), rows as usize);
            cleanup(&io);
        }
    }
}

#[test]
fn repeated_calls_get_fresh_resources_and_never_reuse_a_prior_row_algorithm() {
    let req = request(3, Orientation::Nk);
    let mut io = Recording::default();
    run_serial_rows(&mut io, req, workspace()).unwrap();
    run_serial_rows(&mut io, req, workspace()).unwrap();
    assert_eq!(io.created.len(), 10);
    assert_eq!(io.queried(), 6);
    assert_eq!(io.matmuls(), 6);
    let algorithms: Vec<_> = io
        .events
        .iter()
        .filter_map(|e| match e {
            Event::Matmul(_, _, a) => Some(a[0]),
            _ => None,
        })
        .collect();
    assert_eq!(algorithms, [0x100, 0x101, 0x102, 0x103, 0x104, 0x105]);
    assert_eq!(
        &io.destroyed()[..5],
        &io.created[..5].iter().copied().rev().collect::<Vec<_>>()
    );
    assert_eq!(
        &io.destroyed()[5..],
        &io.created[5..].iter().copied().rev().collect::<Vec<_>>()
    );
    assert!(io.live.is_empty());
}

#[test]
fn driver_rejects_invalid_request_or_workspace_before_any_io() {
    for fault in 0..8 {
        let mut req = request(8, Orientation::Nk);
        let mut ws = workspace();
        match fault {
            0 => req.rows = 1,
            1 => req.n = 0,
            2 => req.act.address = 0,
            3 => req.weight.address += 1,
            4 => req.out.bytes -= 1,
            5 => req.out.address = req.act.address,
            6 => ws.address = 0,
            _ => ws.bytes -= 1,
        }
        let mut io = Recording::default();
        let error = run_serial_rows(&mut io, req, ws).unwrap_err();
        assert!(matches!(error.primary, Some(Failure::Admission(_))));
        assert!(error.cleanup.is_empty());
        assert!(io.events.is_empty());
    }
}

#[test]
fn each_setup_failure_stops_before_query_and_cleans_only_acquired_resources() {
    for fail_at in 0..8 {
        let mut io = Recording {
            fail_at: Some(fail_at),
            ..Default::default()
        };
        let error = run_serial_rows(&mut io, request(8, Orientation::Nk), workspace()).unwrap_err();
        assert!(matches!(error.primary, Some(Failure::Io(ref e)) if e == &format!("io-{fail_at}")));
        assert!(error.cleanup.is_empty());
        assert_eq!(io.queried(), 0);
        assert_eq!(io.matmuls(), 0);
        cleanup(&io);
    }
}

#[test]
fn query_failure_at_each_row_stops_all_later_work_and_cleans_resources() {
    for row in 0..8 {
        let fail_at = 8 + row * 2;
        let mut io = Recording {
            fail_at: Some(fail_at),
            ..Default::default()
        };
        let error = run_serial_rows(&mut io, request(8, Orientation::Kn), workspace()).unwrap_err();
        assert!(matches!(error.primary, Some(Failure::Io(ref e)) if e == &format!("io-{fail_at}")));
        assert_eq!(io.queried(), row + 1);
        assert_eq!(io.matmuls(), row);
        cleanup(&io);
    }
}

#[test]
fn matmul_failure_at_each_row_never_retries_or_submits_another_row() {
    for row in 0..8 {
        let fail_at = 9 + row * 2;
        let mut io = Recording {
            fail_at: Some(fail_at),
            ..Default::default()
        };
        let error = run_serial_rows(&mut io, request(8, Orientation::Nk), workspace()).unwrap_err();
        assert!(matches!(error.primary, Some(Failure::Io(ref e)) if e == &format!("io-{fail_at}")));
        assert_eq!(io.queried(), row + 1);
        assert_eq!(io.matmuls(), row + 1);
        cleanup(&io);
    }
}

#[test]
fn actual_heuristic_admission_rejects_every_bad_result_before_that_row_matmul() {
    for query in [0, 3, 7] {
        for (returned, state, bytes, waves) in [
            (0, 0, 0, 1.0),
            (2, 0, 0, 1.0),
            (-1, 0, 0, 1.0),
            (1, 1, 0, 1.0),
            (1, 0, WORKSPACE_BYTES + 1, 1.0),
            (1, 0, 0, f32::NAN),
            (1, 0, 0, f32::INFINITY),
            (1, 0, 0, -1.0),
        ] {
            let mut io = Recording {
                heuristic_override: Some(HeuristicOverride {
                    query,
                    returned,
                    state,
                    bytes,
                    waves,
                }),
                ..Default::default()
            };
            let error =
                run_serial_rows(&mut io, request(8, Orientation::Nk), workspace()).unwrap_err();
            assert!(matches!(error.primary, Some(Failure::Admission(_))));
            assert_eq!(io.queried(), query + 1);
            assert_eq!(io.matmuls(), query);
            cleanup(&io);
        }
    }
}

#[test]
fn maximum_workspace_and_zero_waves_remain_admitted_like_original_ssot() {
    let mut io = Recording {
        heuristic_override: Some(HeuristicOverride {
            query: 1,
            returned: 1,
            state: 0,
            bytes: WORKSPACE_BYTES,
            waves: 0.0,
        }),
        ..Default::default()
    };
    run_serial_rows(&mut io, request(2, Orientation::Nk), workspace()).unwrap();
    assert_eq!(io.matmuls(), 2);
    cleanup(&io);
}

#[test]
fn destroy_errors_are_reported_with_handles_and_do_not_skip_other_cleanup() {
    let mut io = Recording {
        destroy_errors: vec![5, 3],
        ..Default::default()
    };
    let error = run_serial_rows(&mut io, request(2, Orientation::Nk), workspace()).unwrap_err();
    assert!(error.primary.is_none());
    assert_eq!(
        error
            .cleanup
            .iter()
            .map(|f| (f.kind, f.handle, f.error.as_str()))
            .collect::<Vec<_>>(),
        [
            (ResourceKind::Preference, 5, "destroy-5"),
            (ResourceKind::Layout, 3, "destroy-3"),
        ]
    );
    assert_eq!(
        io.destroyed(),
        io.created.iter().copied().rev().collect::<Vec<_>>()
    );
    assert_eq!(
        io.live,
        [(ResourceKind::Layout, 3), (ResourceKind::Preference, 5)]
    );
}

#[test]
fn query_error_and_cleanup_error_both_survive_without_false_success() {
    let mut io = Recording {
        fail_at: Some(8),
        destroy_errors: vec![4],
        ..Default::default()
    };
    let error = run_serial_rows(&mut io, request(2, Orientation::Nk), workspace()).unwrap_err();
    assert!(matches!(error.primary, Some(Failure::Io(ref e)) if e == "io-8"));
    assert_eq!(error.cleanup.len(), 1);
    assert_eq!(error.cleanup[0].handle, 4);
    assert_eq!(error.cleanup[0].error, "destroy-4");
    assert_eq!(io.destroyed().len(), 5);
    assert_eq!(io.matmuls(), 0);
}

#[test]
fn panic_at_each_setup_query_or_launch_cleans_acquired_resources_then_resumes() {
    for panic_at in 0..24 {
        let mut io = Recording {
            panic_at: Some(panic_at),
            ..Default::default()
        };
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = run_serial_rows(&mut io, request(8, Orientation::Nk), workspace());
        }))
        .expect_err("the driver must not swallow the injected panic");
        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&"injected row-driver panic")
        );
        cleanup(&io);
        let tail = &io.events[panic_at + 1..];
        assert!(tail.iter().all(|e| matches!(e, Event::Destroy(..))));
    }
}

#[test]
fn workspace_alias_rejection_occurs_before_descriptor_creation() {
    let req = request(2, Orientation::Nk);
    let mut io = Recording::default();
    let workspace = ByteSpan {
        address: req.weight.address,
        bytes: WORKSPACE_BYTES,
    };
    let error = run_serial_rows(&mut io, req, workspace).unwrap_err();
    assert!(matches!(error.primary, Some(Failure::Admission(_))));
    assert!(io.events.is_empty());
}
