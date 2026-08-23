// SPDX-License-Identifier: AGPL-3.0-only

//! Compile-time phase progression for the production SPEC_CYCLE_V2 seam.

use super::{Phase, SpecCycle};

pub(in crate::scheduler) struct AwaitSecondaryWait;
pub(in crate::scheduler) struct AwaitSetup;
pub(in crate::scheduler) struct AwaitVerifyComplete;
pub(in crate::scheduler) struct AwaitAccept;
pub(in crate::scheduler) struct AwaitCommitEnqueue;
pub(in crate::scheduler) struct AwaitPostCommitEnqueue;
pub(in crate::scheduler) struct AwaitProposerState;
pub(in crate::scheduler) struct AwaitProposeComplete;
pub(in crate::scheduler) struct AwaitFinalize;
pub(in crate::scheduler) struct CompleteReady;

impl SpecCycle {
    pub(in crate::scheduler) fn secondary_wait_enqueue(
        &mut self,
        _: AwaitSecondaryWait,
    ) -> AwaitSetup {
        self.mark(Phase::SecondaryWaitEnqueue);
        AwaitSetup
    }

    pub(in crate::scheduler) fn setup(&mut self, _: AwaitSetup) -> AwaitVerifyComplete {
        self.mark(Phase::Setup);
        AwaitVerifyComplete
    }

    pub(in crate::scheduler) fn verify_complete(&mut self, _: AwaitVerifyComplete) -> AwaitAccept {
        self.mark(Phase::VerifyComplete);
        AwaitAccept
    }

    pub(in crate::scheduler) fn accept(&mut self, _: AwaitAccept) -> AwaitCommitEnqueue {
        self.mark(Phase::Accept);
        AwaitCommitEnqueue
    }

    pub(in crate::scheduler) fn commit_enqueue(
        &mut self,
        _: AwaitCommitEnqueue,
    ) -> AwaitPostCommitEnqueue {
        self.mark(Phase::CommitEnqueue);
        AwaitPostCommitEnqueue
    }

    pub(in crate::scheduler) fn post_commit_enqueue(
        &mut self,
        _: AwaitPostCommitEnqueue,
    ) -> AwaitProposerState {
        self.mark(Phase::PostCommitEnqueue);
        AwaitProposerState
    }

    pub(in crate::scheduler) fn proposer_state(
        &mut self,
        _: AwaitProposerState,
    ) -> AwaitProposeComplete {
        self.mark(Phase::ProposerState);
        AwaitProposeComplete
    }

    pub(in crate::scheduler) fn propose_complete(
        &mut self,
        _: AwaitProposeComplete,
    ) -> AwaitFinalize {
        self.mark(Phase::ProposeComplete);
        AwaitFinalize
    }

    pub(in crate::scheduler) fn finalize(&mut self, _: AwaitFinalize) -> CompleteReady {
        self.mark(Phase::Finalize);
        CompleteReady
    }
}
