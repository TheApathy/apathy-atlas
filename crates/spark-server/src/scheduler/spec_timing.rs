// SPDX-License-Identifier: AGPL-3.0-only

//! No-sync, host-monotonic timing for the frozen SPEC_CYCLE_V2 diagnostic.
use std::time::Instant;

use super::ActiveSeq;

const PHASE_COUNT: usize = 9;

#[path = "spec_timing_records.rs"]
mod records;
use records::{CompleteRecord, TerminalRecord};

#[path = "spec_timing_phases.rs"]
mod phases;
use phases::{AwaitAccept, AwaitSecondaryWait, CompleteReady};

#[path = "spec_timing_state.rs"]
mod state;
use state::{Gate, abandon_request, begin_request, finish_request};

pub(super) fn configure(max_batch: usize) -> Result<bool, &'static str> {
    state::configure(max_batch)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    SecondaryWaitEnqueue,
    Setup,
    VerifyComplete,
    Accept,
    CommitEnqueue,
    PostCommitEnqueue,
    ProposerState,
    ProposeComplete,
    Finalize,
}

impl Phase {
    fn index(self) -> usize {
        self as usize
    }
}

struct ActiveCycle {
    req: u64,
    pre: u64,
    gamma: u32,
    output_before: usize,
    last: Instant,
    phases: [u64; PHASE_COUNT],
    next_phase: usize,
    tracked: bool,
}

pub(super) struct SpecCycle(Option<ActiveCycle>);

impl SpecCycle {
    pub(super) fn begin(active: &ActiveSeq, drafts: &[u32]) -> (Self, AwaitSecondaryWait) {
        let pending_tree_nodes = active
            .pending_tree_payload
            .as_ref()
            .map(|payload| payload.tree_token_ids.len());
        let enabled = Gate::enabled();
        Self::begin_with(
            enabled,
            active.seq.seq_len,
            drafts.len(),
            active.output_tokens.len(),
            pending_tree_nodes,
            begin_request,
            Instant::now,
        )
    }

    fn begin_with<B, F>(
        enabled: bool,
        pre: usize,
        gamma: usize,
        output_before: usize,
        pending_tree_nodes: Option<usize>,
        track: B,
        now: F,
    ) -> (Self, AwaitSecondaryWait)
    where
        B: FnOnce(u64) -> Option<u64>,
        F: FnOnce() -> Instant,
    {
        if pending_tree_nodes.is_some_and(|nodes| nodes > 0) {
            return (Self(None), AwaitSecondaryWait);
        }
        let (Ok(pre), Ok(gamma)) = (u64::try_from(pre), u32::try_from(gamma)) else {
            return (Self(None), AwaitSecondaryWait);
        };
        if gamma.checked_add(1).is_none() {
            return (Self(None), AwaitSecondaryWait);
        }
        let Some(req) = enabled.then(|| track(pre)).flatten() else {
            return (Self(None), AwaitSecondaryWait);
        };
        (
            Self(Some(ActiveCycle {
                req,
                pre,
                gamma,
                output_before,
                last: now(),
                phases: [0; PHASE_COUNT],
                next_phase: 0,
                tracked: true,
            })),
            AwaitSecondaryWait,
        )
    }

    fn mark(&mut self, phase: Phase) {
        if self.0.is_some() {
            self.mark_at(phase, Instant::now());
        }
    }

    fn mark_at(&mut self, phase: Phase, now: Instant) {
        let Some(active) = self.0.as_mut() else {
            return;
        };
        if phase.index() != active.next_phase {
            active.next_phase = PHASE_COUNT + 1;
            return;
        }
        let Some(duration) = now.checked_duration_since(active.last) else {
            active.next_phase = PHASE_COUNT + 1;
            return;
        };
        let Ok(ns) = u64::try_from(duration.as_nanos()) else {
            active.next_phase = PHASE_COUNT + 1;
            return;
        };
        active.phases[active.next_phase] = ns;
        active.next_phase += 1;
        active.last = now;
    }

    pub(super) fn complete(
        mut self,
        _: CompleteReady,
        accepted: usize,
        output_after: usize,
    ) -> Option<CompleteRecord> {
        let active = self.0.take()?;
        let Ok(accepted) = u32::try_from(accepted) else {
            return reject(active);
        };
        let Some(emitted) = accepted.checked_add(1) else {
            return reject(active);
        };
        let Some(output_delta) = output_after.checked_sub(active.output_before) else {
            return reject(active);
        };
        if active.next_phase != PHASE_COUNT
            || accepted > active.gamma
            || usize::try_from(emitted).ok() != Some(output_delta)
        {
            return reject(active);
        }
        let Some(total) = active
            .phases
            .iter()
            .try_fold(0u64, |sum, value| sum.checked_add(*value))
        else {
            return reject(active);
        };
        if total == 0 {
            return reject(active);
        }
        let Some(next_pre) = active.pre.checked_add(u64::from(emitted)) else {
            return reject(active);
        };
        if active.tracked && !finish_request(active.req, active.pre, next_pre, false) {
            return reject(active);
        }
        Some(CompleteRecord {
            active,
            accepted,
            emitted,
            total,
        })
    }

    pub(super) fn terminal(
        mut self,
        _: AwaitAccept,
        verifier_accepted: usize,
        output_after: usize,
    ) -> Option<TerminalRecord> {
        let active = self.0.take()?;
        let Ok(verifier_accepted) = u32::try_from(verifier_accepted) else {
            return reject(active);
        };
        let Some(output_delta) = output_after.checked_sub(active.output_before) else {
            return reject(active);
        };
        let Ok(emitted) = u32::try_from(output_delta) else {
            return reject(active);
        };
        let accepted_emitted = emitted.min(verifier_accepted);
        let Some(bonus_emitted) = emitted.checked_sub(accepted_emitted) else {
            return reject(active);
        };
        let branch = match (bonus_emitted, accepted_emitted == verifier_accepted) {
            (0, _) => "accepted_draft",
            (1, true) => "bonus",
            _ => return reject(active),
        };
        let Some(next_pre) = active.pre.checked_add(u64::from(emitted)) else {
            return reject(active);
        };
        if verifier_accepted > active.gamma
            || (active.tracked && !finish_request(active.req, active.pre, next_pre, true))
        {
            return reject(active);
        }
        Some(TerminalRecord {
            active,
            verifier_accepted,
            accepted_emitted,
            bonus_emitted,
            emitted,
            branch,
        })
    }
}

fn reject<T>(active: ActiveCycle) -> Option<T> {
    if active.tracked {
        abandon_request(active.req);
    }
    None
}

impl Drop for SpecCycle {
    fn drop(&mut self) {
        if let Some(active) = self.0.take().filter(|active| active.tracked) {
            abandon_request(active.req);
        }
    }
}

#[cfg(test)]
#[path = "spec_timing_tests.rs"]
mod tests;
