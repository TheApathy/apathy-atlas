// SPDX-License-Identifier: AGPL-3.0-only

//! Paired scheduler ownership for speculative drafts and optional DDTree data.

use anyhow::Result;
use spark_model::{layers::DDTreePayload, traits::SequenceState};

use super::{ActiveSeq, Model};

#[path = "flat_batch_lifecycle.rs"]
mod flat_batch_lifecycle;
pub(crate) use flat_batch_lifecycle::{
    FixedBatchEligibility, FixedBatchWidth, FlatBatchPreflight, fixed_batch_decision,
    flat_batch_preflight_at, take_flat_batch_at,
};

/// Typed producer boundary used by the scheduler pairing authority. Tests can
/// supply a narrow source without implementing the full serving model trait.
pub(crate) trait ProposalTreeSource<State, Tree> {
    fn propose(
        &self,
        state: &mut State,
        token: u32,
        position: usize,
        num_drafts: usize,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>>;

    fn take_tree(&self, state: &mut State) -> Option<Tree>;
}

impl<M: Model + ?Sized> ProposalTreeSource<SequenceState, DDTreePayload> for M {
    fn propose(
        &self,
        state: &mut SequenceState,
        token: u32,
        position: usize,
        num_drafts: usize,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        self.run_mtp_propose_multi(token, position, num_drafts, state, 0, grammar_bitmask)
    }

    fn take_tree(&self, state: &mut SequenceState) -> Option<DDTreePayload> {
        self.take_pending_tree_payload(state)
    }
}

/// Mutable scheduler frame accepted by the production pairing authority.
pub(crate) trait SchedulerProposalFrame {
    type State;
    type Tree;

    fn position(&self) -> usize;
    fn parts(&mut self) -> (&mut Self::State, &mut Vec<u32>, &mut Option<Self::Tree>);
}

impl SchedulerProposalFrame for ActiveSeq {
    type State = SequenceState;
    type Tree = DDTreePayload;

    fn position(&self) -> usize {
        self.seq.seq_len
    }

    fn parts(&mut self) -> (&mut Self::State, &mut Vec<u32>, &mut Option<Self::Tree>) {
        (
            &mut self.seq,
            &mut self.pending_drafts,
            &mut self.pending_tree_payload,
        )
    }
}

/// Replace the current scheduler proposal with one producer result. The
/// producer-side tree slot is drained exactly once even when the result is
/// empty or failed, so topology can never cross proposal-frame boundaries.
pub(crate) fn install<T, E>(
    pending_drafts: &mut Vec<u32>,
    pending_tree: &mut Option<T>,
    result: Result<Vec<u32>, E>,
    take_tree: impl FnOnce() -> Option<T>,
) -> Result<bool, E> {
    let tree = take_tree();
    match result {
        Ok(drafts) if !drafts.is_empty() => {
            *pending_drafts = drafts;
            *pending_tree = tree;
            Ok(true)
        }
        Ok(_) => {
            clear(pending_drafts, pending_tree);
            Ok(false)
        }
        Err(error) => {
            clear(pending_drafts, pending_tree);
            Err(error)
        }
    }
}

/// Publish an already-collected result and drain its producer-side tree once.
pub(crate) fn install_collected<S, F>(
    source: &S,
    frame: &mut F,
    proposal: Result<Vec<u32>>,
) -> Result<bool>
where
    S: ProposalTreeSource<F::State, F::Tree> + ?Sized,
    F: SchedulerProposalFrame + ?Sized,
{
    let (state, pending_drafts, pending_tree) = frame.parts();
    install(pending_drafts, pending_tree, proposal, || {
        source.take_tree(state)
    })
}

/// Publish a flat draft chain that did NOT come from the model proposer.
///
/// Used by retrieval drafting (`crate::rest_store`), which pre-empts a
/// proposer call rather than following one. The producer-side tree slot is
/// still drained exactly once and then dropped, so topology left by an
/// earlier proposal can neither leak into this frame nor survive into a
/// later one; the installed frame is always flat.
///
/// Returns whether a non-empty chain was installed.
pub(crate) fn install_external_flat<S, F>(source: &S, frame: &mut F, drafts: Vec<u32>) -> bool
where
    S: ProposalTreeSource<F::State, F::Tree> + ?Sized,
    F: SchedulerProposalFrame + ?Sized,
{
    let (state, pending_drafts, pending_tree) = frame.parts();
    // Drained and dropped: an external chain publishes a flat frame.
    let _stale = source.take_tree(state);
    let installed: Result<bool, std::convert::Infallible> =
        install(pending_drafts, pending_tree, Ok(drafts), || None);
    installed.unwrap_or(false)
}

/// Run an optional proposal, preserve caller work at the proposal-complete
/// boundary, then atomically drain and publish its drafts/tree frame.
pub(crate) fn propose_and_install_with<S, F, T>(
    source: &S,
    frame: &mut F,
    token: u32,
    num_drafts: usize,
    grammar_bitmask: Option<&[i32]>,
    should_propose: bool,
    after_propose: impl FnOnce(&mut F, &mut Result<Vec<u32>>) -> T,
) -> (Result<bool>, T)
where
    S: ProposalTreeSource<F::State, F::Tree> + ?Sized,
    F: SchedulerProposalFrame + ?Sized,
{
    let position = frame.position();
    let mut proposal = if should_propose {
        let (state, _, _) = frame.parts();
        source.propose(state, token, position, num_drafts, grammar_bitmask)
    } else {
        Ok(Vec::new())
    };
    let after = after_propose(frame, &mut proposal);
    (install_collected(source, frame, proposal), after)
}

/// Run one model proposal and atomically publish its drafts/tree frame.
pub(crate) fn propose_and_install<S, F>(
    source: &S,
    frame: &mut F,
    token: u32,
    num_drafts: usize,
    grammar_bitmask: Option<&[i32]>,
) -> Result<bool>
where
    S: ProposalTreeSource<F::State, F::Tree> + ?Sized,
    F: SchedulerProposalFrame + ?Sized,
{
    propose_and_install_with(
        source,
        frame,
        token,
        num_drafts,
        grammar_bitmask,
        true,
        |_, _| (),
    )
    .0
}

/// Clear both halves of the current scheduler proposal frame.
pub(crate) fn clear<T>(drafts: &mut Vec<u32>, tree: &mut Option<T>) {
    drafts.clear();
    *tree = None;
}

/// Clear topology that has no draft frame to identify and consume it.
pub(crate) fn clear_orphan_tree<T>(drafts: &[u32], tree: &mut Option<T>) {
    if drafts.is_empty() {
        *tree = None;
    }
}

/// Consume a frame only when it is flat. Tree-bearing frames must be routed to
/// the generic verifier, which is the sole topology consumer.
pub(crate) fn take_flat<T>(drafts: &mut Vec<u32>, tree: &Option<T>) -> Option<Vec<u32>> {
    if tree.is_some() {
        return None;
    }
    Some(std::mem::take(drafts))
}

/// Apply a grammar-safe prefix to a proposal. Any truncation converts the
/// frame to a flat chain because the original tree indexes no longer describe
/// the shortened draft slice. A zero-length result clears both halves.
pub(crate) fn retain_prefix<T>(drafts: &mut Vec<u32>, tree: &mut Option<T>, keep: usize) -> bool {
    if keep > drafts.len() {
        clear(drafts, tree);
        return false;
    }
    if keep < drafts.len() {
        drafts.truncate(keep);
        *tree = None;
    }
    clear_orphan_tree(drafts, tree);
    !drafts.is_empty()
}

#[cfg(test)]
#[path = "proposal_lifecycle_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "proposal_lifecycle_model_tests.rs"]
mod model_tests;

#[cfg(test)]
#[path = "proposal_lifecycle_source_guard_tests.rs"]
mod source_guard_tests;
