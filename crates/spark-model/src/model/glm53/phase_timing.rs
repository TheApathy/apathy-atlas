// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in inclusive host-wall diagnostics. No device operations or new fences.
//! A successful fenced phase includes prior work on its existing stream: in
//! particular scalar walk publishes drafter captures after its target fence.
//! Error durations do not certify completion; an unwound phase is incomplete.

use std::{
    cell::{Cell, OnceCell},
    ffi::OsStr,
    time::Instant,
};

use anyhow::{Result, bail};
use serde::Serialize;

#[derive(Clone, Copy, Debug)]
pub(crate) struct TimingSetting(bool);

impl TimingSetting {
    pub(crate) fn parse(value: Option<&OsStr>) -> Result<Self> {
        match value.and_then(OsStr::to_str) {
            None if value.is_none() => Ok(Self(false)),
            Some("0") => Ok(Self(false)),
            Some("1") => Ok(Self(true)),
            _ => bail!("ATLAS_GLM53_PHASE_TIMING must be absent or exactly UTF-8 0 or 1"),
        }
    }

    pub(crate) fn enabled(self) -> bool {
        self.0
    }
}

pub(crate) trait Clock {
    fn now_ns(&self) -> u64;
}

/// Construction is clock-free, including the permanently disabled path.
#[derive(Default)]
pub(crate) struct HostClock(OnceCell<Instant>);

impl Clock for HostClock {
    fn now_ns(&self) -> u64 {
        let now = Instant::now();
        let origin = self.0.get_or_init(|| now);
        u64::try_from(now.duration_since(*origin).as_nanos()).unwrap_or(u64::MAX)
    }
}

/// Completion labels apply only to successful return, not to a failed call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Boundary {
    HostCallback,
    SubmissionOnly,
    ExistingCompletionFence,
    EnqueueAndExistingFences,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[repr(usize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Phase {
    Proposal,
    VerifyTotal,
    StageTotal,
    SnapshotSaveEnqueue,
    WideStage,
    Readback,
    PolicyCheckpoint,
    PolicyPick,
    PolicyAdvance,
    PolicyRestore,
    CommitTotal,
    PartialRestore,
    FullCommitBody,
    PartialReplay,
    FinishFence,
    Abort,
    FailureDrain,
}

impl Phase {
    const COUNT: usize = Self::FailureDrain as usize + 1;
    const ALL: [Self; Self::COUNT] = [
        Self::Proposal,
        Self::VerifyTotal,
        Self::StageTotal,
        Self::SnapshotSaveEnqueue,
        Self::WideStage,
        Self::Readback,
        Self::PolicyCheckpoint,
        Self::PolicyPick,
        Self::PolicyAdvance,
        Self::PolicyRestore,
        Self::CommitTotal,
        Self::PartialRestore,
        Self::FullCommitBody,
        Self::PartialReplay,
        Self::FinishFence,
        Self::Abort,
        Self::FailureDrain,
    ];

    pub(crate) fn boundary(self) -> Boundary {
        match self {
            Self::SnapshotSaveEnqueue => Boundary::SubmissionOnly,
            Self::PolicyCheckpoint
            | Self::PolicyPick
            | Self::PolicyAdvance
            | Self::PolicyRestore => Boundary::HostCallback,
            Self::VerifyTotal | Self::FullCommitBody | Self::PartialReplay => {
                Boundary::EnqueueAndExistingFences
            }
            _ => Boundary::ExistingCompletionFence,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct PhaseStats {
    pub(crate) calls: u64,
    pub(crate) elapsed_ns: u64,
    pub(crate) errors: u64,
    pub(crate) incomplete: u64,
}

pub(crate) struct TimingReceipt {
    pub(crate) host_wall_only: bool,
    pub(crate) inclusive: bool,
    pub(crate) performance_eligible: bool,
    pub(crate) may_include_prior_capture: bool,
    pub(crate) valid: bool,
    phases: [PhaseStats; Phase::COUNT],
}

impl TimingReceipt {
    pub(crate) fn phase(&self, phase: Phase) -> PhaseStats {
        self.phases[phase as usize]
    }
}

/// Correlation only, never token text, logits, or publication authority.
pub(crate) struct TimingContext {
    pub(crate) kind: &'static str,
    pub(crate) generation: u64,
    pub(crate) start: usize,
    pub(crate) rows: usize,
    pub(crate) stream: u64,
    pub(crate) outcome: &'static str,
    pub(crate) accepted: Option<usize>,
    pub(crate) readback_bytes: usize,
}

pub(crate) struct PhaseRecorder<'a> {
    setting: TimingSetting,
    clock: &'a dyn Clock,
    phases: Cell<[PhaseStats; Phase::COUNT]>,
    valid: Cell<bool>,
}

impl<'a> PhaseRecorder<'a> {
    pub(crate) fn new(setting: TimingSetting, clock: &'a dyn Clock) -> Self {
        Self {
            setting,
            clock,
            phases: Cell::new([PhaseStats::default(); Phase::COUNT]),
            valid: Cell::new(true),
        }
    }

    fn add(&self, left: u64, right: u64) -> u64 {
        left.checked_add(right).unwrap_or_else(|| {
            self.valid.set(false);
            u64::MAX
        })
    }

    fn update(&self, phase: Phase, update: impl FnOnce(&mut PhaseStats)) {
        let mut phases = self.phases.get();
        update(&mut phases[phase as usize]);
        self.phases.set(phases);
    }

    pub(crate) fn measure<T>(&self, phase: Phase, action: impl FnOnce() -> Result<T>) -> Result<T> {
        if !self.setting.enabled() {
            return action();
        }
        self.update(phase, |row| row.calls = self.add(row.calls, 1));
        let mut guard = Incomplete {
            recorder: self,
            phase,
            finished: false,
        };
        let start = self.clock.now_ns();
        let result = action();
        let end = self.clock.now_ns();
        let elapsed = end.checked_sub(start).unwrap_or_else(|| {
            self.valid.set(false);
            0
        });
        // Read counters only after the callback, so nested timers never lose
        // updates and no RefCell/lock borrow is held across foreign policy.
        self.update(phase, |row| {
            row.elapsed_ns = self.add(row.elapsed_ns, elapsed);
            row.errors = self.add(row.errors, u64::from(result.is_err()));
        });
        guard.finished = true;
        result
    }

    pub(crate) fn receipt(&self) -> Option<TimingReceipt> {
        self.setting.enabled().then(|| TimingReceipt {
            host_wall_only: true,
            inclusive: true,
            performance_eligible: false,
            may_include_prior_capture: true,
            valid: self.valid.get(),
            phases: self.phases.get(),
        })
    }

    /// Called after measured callbacks return. Serialization/logging is not
    /// included in phase elapsed time. No receipt is emitted during unwind.
    pub(crate) fn emit(&self, context: TimingContext) {
        let Some(receipt) = self.receipt() else {
            return;
        };
        let phases: Vec<_> = Phase::ALL
            .into_iter()
            .filter_map(|phase| {
                let stats = receipt.phase(phase);
                (stats.calls != 0).then_some((phase, phase.boundary(), stats))
            })
            .collect();
        let attempted = receipt.phase(Phase::Readback).calls != 0;
        let payload = serde_json::json!({
            "schema": "atlas.glm53.phase_timing.v1", "kind": context.kind,
            "generation": context.generation, "start": context.start, "rows": context.rows,
            "stream": context.stream, "outcome": context.outcome, "accepted": context.accepted,
            "readback_bytes": if attempted { context.readback_bytes } else { 0 },
            "readback_bytes_basis": "full_logits_requested_on_copy_attempt",
            "host_wall_only": receipt.host_wall_only, "inclusive": receipt.inclusive,
            "performance_eligible": receipt.performance_eligible,
            "may_include_prior_capture": receipt.may_include_prior_capture,
            "valid": receipt.valid, "phase_rows": phases,
            "accounting": "inclusive_do_not_sum_nested;unattributed_is_not_cpu_policy",
            "completion": "boundary_labels_apply_only_on_success;errors_do_not_prove_completion",
            "stage_carry_in": "wide_stage_fence_may_drain_snapshot_save_and_prior_stream_work",
            "replay_capture_boundary": "last_capture_enqueue_may_complete_only_in_finish_fence",
            "proposal_scope": "runtime_propose_and_return_slice;rows_are_fixed_query_rows",
            "verify_scope": "post_binding_stage_through_transaction_return",
            "excluded": "identity_read,receipt_serialization_and_logging"
        });
        tracing::info!(schema = "atlas.glm53.phase_timing.v1", receipt = %payload,
            "GLM host-wall phase diagnostic");
    }
}

struct Incomplete<'a, 'clock> {
    recorder: &'a PhaseRecorder<'clock>,
    phase: Phase,
    finished: bool,
}

impl Drop for Incomplete<'_, '_> {
    fn drop(&mut self) {
        if !self.finished {
            self.recorder.valid.set(false);
            self.recorder.update(self.phase, |row| {
                row.incomplete = self.recorder.add(row.incomplete, 1);
            });
        }
    }
}
