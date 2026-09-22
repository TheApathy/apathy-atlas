// SPDX-License-Identifier: AGPL-3.0-only

//! RED: forwarding timers exercise the real transaction and shipping sampler.
//! This CPU fixture proves callback equivalence, not asynchronous GPU ownership.

#[path = "support/glm53_verify_policy_fixture.rs"]
#[allow(dead_code)]
mod fixture;
#[path = "../src/model/glm53/phase_timing.rs"]
#[allow(dead_code)]
mod phase_timing;
#[path = "../src/model/glm53/phase_timing_wrappers.rs"]
#[allow(dead_code)]
mod phase_timing_wrappers;
#[path = "../src/model/glm53/verify_policy_transaction.rs"]
#[allow(dead_code)]
mod verify_policy_transaction;

use anyhow::{Result, bail};
use phase_timing::{Clock, Phase, PhaseRecorder, TimingReceipt, TimingSetting};
use phase_timing_wrappers::{TimedCommit, TimedLogits, TimedPolicy};
use std::{
    cell::Cell,
    ffi::OsStr,
    panic::{AssertUnwindSafe, catch_unwind},
};
use verify_policy_transaction::{
    LogitsIo, PolicyAdvance, VerifyCommitIo, VerifyOutcome, VerifyPolicy, VerifyRequest,
    run_verify_policy_transaction,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Case {
    Full,
    Partial,
    FirstReject,
    Terminal,
    OrdinaryReplay,
    CopyError,
    ShortCopy,
    CheckpointError,
    PickError,
    AdvanceError,
    RestoreError,
    AbortError,
    CommitError,
    WrongPosition,
    PanicCopy,
    PanicPick,
    PanicCommit,
}

struct SourceAdapter<'a>(&'a mut fixture::Source, Case);
impl LogitsIo for SourceAdapter<'_> {
    fn copy_logits(&mut self, out: &mut [u8]) -> Result<usize> {
        let result = self.0.copy_logits(out);
        if self.1 == Case::PanicCopy {
            panic!("injected copy panic");
        }
        result
    }
}

struct PolicyAdapter<'a>(&'a mut fixture::Policy, Case);
impl VerifyPolicy for PolicyAdapter<'_> {
    fn checkpoint(&mut self) -> Result<()> {
        self.0.checkpoint()?;
        if self.1 == Case::CheckpointError {
            bail!("injected checkpoint failure");
        }
        Ok(())
    }
    fn pick(&mut self, row: usize, logits: &[u8]) -> Result<u32> {
        let result = self.0.pick(row, logits);
        if self.1 == Case::PanicPick {
            panic!("injected pick panic");
        }
        result
    }
    fn advance(&mut self, token: u32) -> Result<PolicyAdvance> {
        let result = self.0.advance(token)?;
        if self.1 == Case::OrdinaryReplay {
            return Ok(PolicyAdvance::OrdinaryReplay);
        }
        Ok(result)
    }
    fn restore(&mut self) -> Result<()> {
        self.0.restore()
    }
}

struct TargetAdapter<'a>(&'a mut fixture::Target, Case);
impl VerifyCommitIo for TargetAdapter<'_> {
    fn commit_prefix(&mut self, request: &VerifyRequest, rows: usize) -> Result<usize> {
        let result = self.0.commit_prefix(request, rows);
        if self.1 == Case::PanicCommit {
            panic!("injected commit panic");
        }
        result
    }
    fn abort_staged(&mut self) -> Result<()> {
        self.0.abort_staged()
    }
    fn poison(&mut self) {
        self.0.poison();
    }
}

#[derive(Default)]
struct CounterClock(Cell<u64>);
impl Clock for CounterClock {
    fn now_ns(&self) -> u64 {
        let next = self.0.get() + 10;
        self.0.set(next);
        next
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Committed {
        accepted: usize,
        emitted: Vec<u32>,
        terminal: bool,
    },
    Restored {
        start: usize,
        anchor: u32,
    },
    Error(String),
    Panic(String),
}

#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    outcome: Outcome,
    events: Vec<&'static str>,
    position: usize,
    commits: Vec<usize>,
    aborts: usize,
    poisoned: bool,
    history: Vec<u32>,
    counters: [u32; 3],
    observed_history: Vec<Vec<u32>>,
    observed_rows: Vec<Vec<u8>>,
    overwritten_device: Vec<u8>,
    host: Vec<u32>,
    seq_len: usize,
    kv_valid: usize,
}

fn execute(case: Case, timing: Option<&str>) -> (Snapshot, Option<TimingReceipt>, u64) {
    let rows = vec![
        vec![0., 9., 1., 2.],
        vec![0., 1., 9., 2.],
        vec![0., 1., 2., 9.],
        vec![0., 9., 1., 2.],
    ];
    let inputs = match case {
        Case::Partial => [0, 1, 3, 2],
        Case::FirstReject => [0, 3, 2, 1],
        _ => [0, 1, 2, 3],
    };
    let request = VerifyRequest::new(4, 32, 4, &inputs).unwrap();
    let (mut source, mut policy, mut target, events) = fixture::fixture(&rows);
    source.fail = case == Case::CopyError;
    source.short = case == Case::ShortCopy;
    policy.terminal = (case == Case::Terminal).then_some(2);
    policy.fail_pick = matches!(case, Case::PickError | Case::AbortError).then_some(1);
    policy.fail_advance = case == Case::AdvanceError;
    policy.fail_restore = case == Case::RestoreError;
    target.fail_abort = case == Case::AbortError;
    target.fail_commit = case == Case::CommitError;
    target.wrong_position = case == Case::WrongPosition;
    // Retain the real sampler/history path, not a second timing-only selector.
    policy.history = vec![3];
    policy.params.repetition_penalty = 1.1;
    let clock = CounterClock::default();
    let setting = TimingSetting::parse(timing.map(OsStr::new)).unwrap();
    let recorder = PhaseRecorder::new(setting, &clock);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut input = SourceAdapter(&mut source, case);
        let mut rules = PolicyAdapter(&mut policy, case);
        let mut commit = TargetAdapter(&mut target, case);
        if timing.is_some() {
            let mut timed_input = TimedLogits::new(&mut input, &recorder);
            let mut timed_rules = TimedPolicy::new(&mut rules, &recorder);
            let mut timed_commit = TimedCommit::new(&mut commit, &recorder);
            recorder.measure(Phase::VerifyTotal, || {
                run_verify_policy_transaction(
                    &request,
                    &mut timed_input,
                    &mut timed_rules,
                    &mut timed_commit,
                )
            })
        } else {
            run_verify_policy_transaction(&request, &mut input, &mut rules, &mut commit)
        }
    }));
    let mut host = vec![0; 4];
    let mut seq_len = 4;
    let mut kv_valid = 4;
    let outcome = match result {
        Ok(Ok(VerifyOutcome::Committed(committed))) => {
            let accepted = committed.accepted_drafts();
            let published = committed
                .publish(&mut host, &mut seq_len, &mut kv_valid)
                .unwrap();
            Outcome::Committed {
                accepted,
                emitted: published.emitted_tokens().to_vec(),
                terminal: published.terminal(),
            }
        }
        Ok(Ok(VerifyOutcome::RestoredForOrdinaryReplay(restored))) => {
            restored
                .validate_host_prefix(&host, seq_len, kv_valid)
                .unwrap();
            Outcome::Restored {
                start: restored.start(),
                anchor: restored.anchor(),
            }
        }
        Ok(Err(error)) => Outcome::Error(format!("{error:#}")),
        Err(payload) => Outcome::Panic(
            payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .expect("injected panic payload"),
        ),
    };
    let snapshot = Snapshot {
        outcome,
        events: events.borrow().clone(),
        position: target.position,
        commits: target.commits,
        aborts: target.aborts,
        poisoned: target.poisoned,
        history: policy.history,
        counters: policy.counters,
        observed_history: policy.observed_history,
        observed_rows: policy.observed_rows,
        overwritten_device: source.bytes.borrow().clone(),
        host,
        seq_len,
        kv_valid,
    };
    (snapshot, recorder.receipt(), clock.0.get() / 10)
}

#[test]
fn enabled_and_disabled_wrappers_preserve_full_partial_first_reject_and_terminal() {
    for (case, accepted, commit_rows, terminal) in [
        (Case::Full, 3, 4, false),
        (Case::Partial, 1, 2, false),
        (Case::FirstReject, 0, 1, false),
        (Case::Terminal, 2, 3, true),
    ] {
        let (baseline, _, _) = execute(case, None);
        assert_eq!(baseline.commits, vec![commit_rows]);
        assert!(
            matches!(&baseline.outcome, Outcome::Committed { accepted: a, terminal: t, .. }
            if *a == accepted && *t == terminal)
        );
        assert_eq!(baseline.history, vec![3]);
        assert_eq!(baseline.counters, [4, 5, 6]);
        for setting in ["0", "1"] {
            let (observed, receipt, clocks) = execute(case, Some(setting));
            assert_eq!(observed, baseline, "{case:?}, timing={setting}");
            if setting == "0" {
                assert!(receipt.is_none());
                assert_eq!(clocks, 0);
            } else {
                assert!(receipt.unwrap().valid);
                assert!(clocks > 0);
            }
        }
    }
}

#[test]
fn ordinary_replay_restores_without_commit_or_publication() {
    let (baseline, _, _) = execute(Case::OrdinaryReplay, None);
    let (observed, receipt, _) = execute(Case::OrdinaryReplay, Some("1"));
    assert_eq!(observed, baseline);
    assert_eq!(
        observed.outcome,
        Outcome::Restored {
            start: 4,
            anchor: 0
        }
    );
    assert_eq!(observed.aborts, 1);
    assert!(observed.commits.is_empty());
    assert_eq!(observed.host, vec![0; 4]);
    let receipt = receipt.unwrap();
    assert_eq!(receipt.phase(Phase::Abort).calls, 1);
    assert_eq!(receipt.phase(Phase::CommitTotal).calls, 0);
}

#[test]
fn every_failure_preserves_order_error_restoration_and_poison_semantics() {
    for case in [
        Case::CopyError,
        Case::ShortCopy,
        Case::CheckpointError,
        Case::PickError,
        Case::AdvanceError,
        Case::RestoreError,
        Case::AbortError,
        Case::CommitError,
        Case::WrongPosition,
    ] {
        let (baseline, _, _) = execute(case, None);
        assert!(matches!(baseline.outcome, Outcome::Error(_)));
        for setting in ["0", "1"] {
            let (observed, receipt, clocks) = execute(case, Some(setting));
            assert_eq!(observed, baseline, "{case:?}, timing={setting}");
            assert_eq!(observed.host, vec![0; 4], "error must not publish");
            if setting == "0" {
                assert_eq!(clocks, 0);
                assert!(receipt.is_none());
            } else {
                assert_eq!(receipt.unwrap().phase(Phase::VerifyTotal).errors, 1);
            }
        }
        if matches!(case, Case::CommitError | Case::WrongPosition) {
            assert!(baseline.poisoned);
            assert_eq!(
                baseline.aborts, 0,
                "persistent commit must not receive DSA-only rollback"
            );
        }
        if matches!(case, Case::RestoreError | Case::AbortError) {
            assert!(baseline.poisoned);
        }
    }
}

#[test]
fn nested_callback_counters_do_not_reclassify_commit_or_readback_as_policy_cpu() {
    let (_, receipt, _) = execute(Case::Full, Some("1"));
    let receipt = receipt.unwrap();
    for (phase, calls) in [
        (Phase::VerifyTotal, 1),
        (Phase::Readback, 1),
        (Phase::PolicyCheckpoint, 1),
        (Phase::PolicyPick, 4),
        (Phase::PolicyAdvance, 4),
        (Phase::PolicyRestore, 1),
        (Phase::CommitTotal, 1),
        (Phase::Abort, 0),
    ] {
        assert_eq!(receipt.phase(phase).calls, calls, "{phase:?}");
    }
    // Internal real-target phases are not invented by the forwarding wrapper.
    for phase in [
        Phase::WideStage,
        Phase::PartialRestore,
        Phase::FullCommitBody,
        Phase::PartialReplay,
        Phase::FinishFence,
    ] {
        assert_eq!(receipt.phase(phase).calls, 0);
    }
    assert!(receipt.inclusive);
    assert!(!receipt.performance_eligible);
}

#[test]
fn copy_policy_and_irreversible_commit_panics_are_not_caught_or_translated_by_timers() {
    for case in [Case::PanicCopy, Case::PanicPick, Case::PanicCommit] {
        let (baseline, _, _) = execute(case, None);
        assert!(matches!(baseline.outcome, Outcome::Panic(_)));
        let (observed, receipt, _) = execute(case, Some("1"));
        assert_eq!(observed, baseline, "{case:?}");
        let receipt = receipt.unwrap();
        assert!(!receipt.valid);
        assert_eq!(receipt.phase(Phase::VerifyTotal).incomplete, 1);
        let phase = match case {
            Case::PanicCopy => Phase::Readback,
            Case::PanicPick => Phase::PolicyPick,
            Case::PanicCommit => Phase::CommitTotal,
            _ => unreachable!(),
        };
        assert_eq!(receipt.phase(phase).incomplete, 1);
        assert_eq!(observed.host, vec![0; 4]);
        // Actual PolicyTarget Drop remains the GPU-owner poison authority;
        // the timing wrapper must not add a second abort/poison on unwind.
    }
}
