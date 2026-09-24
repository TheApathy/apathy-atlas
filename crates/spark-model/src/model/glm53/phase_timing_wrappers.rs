// SPDX-License-Identifier: AGPL-3.0-only

//! Forward exactly the shipping policy callbacks; timers own no model state.

use anyhow::Result;

use super::phase_timing::{Phase, PhaseRecorder};
use super::verify_policy_transaction::{
    LogitsIo, PolicyAdvance, VerifyCommitIo, VerifyPolicy, VerifyRequest,
};

pub(crate) struct TimedLogits<'a, 'clock> {
    inner: &'a mut dyn LogitsIo,
    timing: &'a PhaseRecorder<'clock>,
}

impl<'a, 'clock> TimedLogits<'a, 'clock> {
    pub(crate) fn new(inner: &'a mut dyn LogitsIo, timing: &'a PhaseRecorder<'clock>) -> Self {
        Self { inner, timing }
    }
}

impl LogitsIo for TimedLogits<'_, '_> {
    fn copy_logits(&mut self, destination: &mut [u8]) -> Result<usize> {
        self.timing
            .measure(Phase::Readback, || self.inner.copy_logits(destination))
    }

    fn device_logits(&self) -> Option<spark_runtime::gpu::DevicePtr> {
        self.inner.device_logits()
    }
}

pub(crate) struct TimedPolicy<'a, 'clock> {
    inner: &'a mut dyn VerifyPolicy,
    timing: &'a PhaseRecorder<'clock>,
}

impl<'a, 'clock> TimedPolicy<'a, 'clock> {
    pub(crate) fn new(inner: &'a mut dyn VerifyPolicy, timing: &'a PhaseRecorder<'clock>) -> Self {
        Self { inner, timing }
    }
}

impl VerifyPolicy for TimedPolicy<'_, '_> {
    fn checkpoint(&mut self) -> Result<()> {
        self.timing
            .measure(Phase::PolicyCheckpoint, || self.inner.checkpoint())
    }

    fn pick(&mut self, row: usize, logits: &[u8]) -> Result<u32> {
        self.timing
            .measure(Phase::PolicyPick, || self.inner.pick(row, logits))
    }

    fn pick_resident(
        &mut self,
        row: usize,
        logits: &[u8],
        device_row: Option<spark_runtime::gpu::DevicePtr>,
    ) -> Result<u32> {
        self.timing.measure(Phase::PolicyPick, || {
            self.inner.pick_resident(row, logits, device_row)
        })
    }

    fn advance(&mut self, token: u32) -> Result<PolicyAdvance> {
        self.timing
            .measure(Phase::PolicyAdvance, || self.inner.advance(token))
    }

    fn restore(&mut self) -> Result<()> {
        self.timing
            .measure(Phase::PolicyRestore, || self.inner.restore())
    }
}

pub(crate) struct TimedCommit<'a, 'clock> {
    inner: &'a mut dyn VerifyCommitIo,
    timing: &'a PhaseRecorder<'clock>,
}

impl<'a, 'clock> TimedCommit<'a, 'clock> {
    pub(crate) fn new(
        inner: &'a mut dyn VerifyCommitIo,
        timing: &'a PhaseRecorder<'clock>,
    ) -> Self {
        Self { inner, timing }
    }
}

impl VerifyCommitIo for TimedCommit<'_, '_> {
    fn commit_prefix(&mut self, request: &VerifyRequest, rows: usize) -> Result<usize> {
        self.timing.measure(Phase::CommitTotal, || {
            self.inner.commit_prefix(request, rows)
        })
    }

    fn abort_staged(&mut self) -> Result<()> {
        self.timing
            .measure(Phase::Abort, || self.inner.abort_staged())
    }

    fn poison(&mut self) {
        self.inner.poison();
    }
}
