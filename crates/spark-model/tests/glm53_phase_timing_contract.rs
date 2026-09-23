// SPDX-License-Identifier: AGPL-3.0-only

//! RED contract: diagnostic host-wall accounting, never a GPU stopwatch.

#[path = "../src/model/glm53/phase_timing.rs"]
#[allow(dead_code)]
mod phase_timing;

use anyhow::{Result, bail};
use phase_timing::{Boundary, Clock, Phase, PhaseRecorder, TimingSetting};
use std::{
    cell::Cell,
    ffi::{OsStr, OsString},
    panic::{AssertUnwindSafe, catch_unwind},
};

#[derive(Default)]
struct TestClock {
    time: Cell<u64>,
    reads: Cell<usize>,
}

impl TestClock {
    fn advance(&self, ns: u64) {
        self.time.set(self.time.get() + ns);
    }
}

impl Clock for TestClock {
    fn now_ns(&self) -> u64 {
        self.reads.set(self.reads.get() + 1);
        self.time.get()
    }
}

fn setting(value: &str) -> TimingSetting {
    TimingSetting::parse(Some(OsStr::new(value))).unwrap()
}

#[test]
fn strict_os_flag_is_pinned_without_mutating_process_environment() {
    assert!(!TimingSetting::parse(None).unwrap().enabled());
    assert!(!setting("0").enabled());
    assert!(setting("1").enabled());
    for value in [
        "", "true", "false", "on", "off", "01", "2", "-1", " 1", "1\n",
    ] {
        assert!(
            TimingSetting::parse(Some(OsStr::new(value))).is_err(),
            "{value:?}"
        );
    }
    let mut input = OsString::from("0");
    let admitted = TimingSetting::parse(Some(&input)).unwrap();
    input.clear();
    input.push("1");
    assert!(
        !admitted.enabled(),
        "startup setting must not borrow mutable input"
    );
    assert!(TimingSetting::parse(Some(&input)).unwrap().enabled());
}

#[cfg(unix)]
#[test]
fn non_utf8_flag_is_rejected_not_treated_as_absent() {
    use std::os::unix::ffi::OsStrExt;
    assert!(TimingSetting::parse(Some(OsStr::from_bytes(&[0xff]))).is_err());
}

#[test]
fn disabled_callback_results_are_preserved_with_zero_clock_reads_or_receipts() {
    let clock = TestClock::default();
    let recorder = PhaseRecorder::new(setting("0"), &clock);
    let calls = Cell::new(0);
    let value = recorder
        .measure(Phase::Proposal, || {
            calls.set(calls.get() + 1);
            Ok(73)
        })
        .unwrap();
    assert_eq!(value, 73);
    let error = recorder
        .measure(Phase::Readback, || -> Result<()> {
            calls.set(calls.get() + 1);
            bail!("original copy failure")
        })
        .unwrap_err();
    assert_eq!(error.to_string(), "original copy failure");
    assert_eq!(calls.get(), 2);
    assert_eq!(clock.reads.get(), 0);
    assert!(recorder.receipt().is_none());
    assert_eq!(
        clock.reads.get(),
        0,
        "reading an absent receipt must not sample time"
    );
}

#[test]
fn nested_durations_remain_inclusive_and_unattributed_wall_is_not_cpu_policy() {
    let clock = TestClock::default();
    let recorder = PhaseRecorder::new(setting("1"), &clock);
    recorder
        .measure(Phase::VerifyTotal, || {
            clock.advance(5);
            recorder.measure(Phase::StageTotal, || {
                recorder.measure(Phase::SnapshotSaveEnqueue, || {
                    clock.advance(5);
                    Ok(())
                })?;
                recorder.measure(Phase::WideStage, || {
                    clock.advance(25);
                    Ok(())
                })
            })?;
            recorder.measure(Phase::Readback, || {
                clock.advance(10);
                Ok(())
            })?;
            clock.advance(5);
            recorder.measure(Phase::CommitTotal, || {
                recorder.measure(Phase::FullCommitBody, || {
                    clock.advance(20);
                    Ok(())
                })?;
                recorder.measure(Phase::FinishFence, || {
                    clock.advance(25);
                    Ok(())
                })
            })?;
            clock.advance(5);
            Ok(())
        })
        .unwrap();
    let receipt = recorder.receipt().unwrap();
    assert!(receipt.host_wall_only);
    assert!(receipt.inclusive);
    assert!(receipt.may_include_prior_capture);
    assert!(!receipt.performance_eligible);
    assert!(receipt.valid);
    for (phase, ns) in [
        (Phase::VerifyTotal, 100),
        (Phase::StageTotal, 30),
        (Phase::SnapshotSaveEnqueue, 5),
        (Phase::WideStage, 25),
        (Phase::Readback, 10),
        (Phase::CommitTotal, 45),
        (Phase::FullCommitBody, 20),
        (Phase::FinishFence, 25),
    ] {
        let measured = receipt.phase(phase);
        assert_eq!(
            (
                measured.calls,
                measured.elapsed_ns,
                measured.errors,
                measured.incomplete
            ),
            (1, ns, 0, 0)
        );
    }
    // Only disjoint outer buckets may be subtracted from VerifyTotal.
    // Adding WideStage/FullCommitBody/FinishFence again would double count.
    let unattributed = 100 - (30 + 10 + 45);
    assert_eq!(unattributed, 15);
    assert_eq!(receipt.phase(Phase::PolicyPick).calls, 0);
    assert_eq!(clock.reads.get(), 16);
}

#[test]
fn phase_labels_pin_existing_fences_and_async_capture_carry_in() {
    assert_eq!(
        Phase::SnapshotSaveEnqueue.boundary(),
        Boundary::SubmissionOnly
    );
    for phase in [
        Phase::Proposal,
        Phase::StageTotal,
        Phase::WideStage,
        Phase::Readback,
        Phase::CommitTotal,
        Phase::PartialRestore,
        Phase::FinishFence,
        Phase::Abort,
        Phase::FailureDrain,
    ] {
        assert_eq!(
            phase.boundary(),
            Boundary::ExistingCompletionFence,
            "{phase:?}"
        );
    }
    for phase in [
        Phase::FullCommitBody,
        Phase::PartialReplay,
        Phase::VerifyTotal,
    ] {
        assert_eq!(
            phase.boundary(),
            Boundary::EnqueueAndExistingFences,
            "{phase:?}"
        );
    }
    for phase in [
        Phase::PolicyCheckpoint,
        Phase::PolicyPick,
        Phase::PolicyAdvance,
        Phase::PolicyRestore,
    ] {
        assert_eq!(phase.boundary(), Boundary::HostCallback, "{phase:?}");
    }
}

#[test]
fn error_duration_is_kept_but_error_value_is_not_replaced() {
    let clock = TestClock::default();
    let recorder = PhaseRecorder::new(setting("1"), &clock);
    let error = recorder
        .measure(Phase::PartialReplay, || -> Result<()> {
            clock.advance(17);
            bail!("irreversible replay failed")
        })
        .unwrap_err();
    assert_eq!(error.to_string(), "irreversible replay failed");
    let receipt = recorder.receipt().unwrap();
    let phase = receipt.phase(Phase::PartialReplay);
    assert_eq!(
        (
            phase.calls,
            phase.elapsed_ns,
            phase.errors,
            phase.incomplete
        ),
        (1, 17, 1, 0)
    );
    assert!(
        receipt.valid,
        "complete error timing is distinct from successful execution"
    );
    assert!(!receipt.performance_eligible);
}

#[test]
fn panic_propagates_and_drop_marks_incomplete_without_clock_or_io() {
    let clock = TestClock::default();
    let recorder = PhaseRecorder::new(setting("1"), &clock);
    let result = catch_unwind(AssertUnwindSafe(|| {
        recorder.measure(Phase::VerifyTotal, || -> Result<()> {
            recorder.measure(Phase::PolicyPick, || -> Result<()> {
                panic!("policy panic")
            })
        })
    }));
    assert_eq!(
        *result.unwrap_err().downcast::<&str>().unwrap(),
        "policy panic"
    );
    assert_eq!(clock.reads.get(), 2, "unwinding must not call the clock");
    let receipt = recorder.receipt().unwrap();
    assert!(!receipt.valid);
    for phase in [Phase::VerifyTotal, Phase::PolicyPick] {
        let row = receipt.phase(phase);
        assert_eq!(
            (row.calls, row.elapsed_ns, row.errors, row.incomplete),
            (1, 0, 0, 1)
        );
    }
}

#[test]
fn invalid_clock_order_invalidates_only_timing_not_transaction_result() {
    let clock = TestClock::default();
    clock.time.set(9);
    let recorder = PhaseRecorder::new(setting("1"), &clock);
    assert_eq!(
        recorder
            .measure(Phase::Proposal, || {
                clock.time.set(3);
                Ok(81)
            })
            .unwrap(),
        81
    );
    let receipt = recorder.receipt().unwrap();
    assert!(!receipt.valid);
    assert_eq!(receipt.phase(Phase::Proposal).elapsed_ns, 0);
}
