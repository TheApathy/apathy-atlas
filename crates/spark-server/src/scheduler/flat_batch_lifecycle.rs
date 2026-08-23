// SPDX-License-Identifier: AGPL-3.0-only

//! Typed eligibility and atomic flat-frame gathering for fixed batched verify.

use super::ActiveSeq;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FixedBatchWidth {
    K2,
    K3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FixedBatchEligibility {
    pub draft_len: usize,
    pub grammar_active: bool,
    pub finished: bool,
    pub configured_num_drafts: usize,
    pub policy_required: bool,
    pub has_tree: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FlatBatchDecision {
    Eligible,
    Ineligible,
}

impl FlatBatchDecision {
    pub(crate) fn is_eligible(self) -> bool {
        self == Self::Eligible
    }
}

pub(crate) fn fixed_batch_decision(
    width: FixedBatchWidth,
    candidate: FixedBatchEligibility,
) -> FlatBatchDecision {
    let expected_drafts = match width {
        FixedBatchWidth::K2 => 1,
        FixedBatchWidth::K3 => 2,
    };
    let configuration_matches =
        width != FixedBatchWidth::K2 || candidate.configured_num_drafts == 1;
    if candidate.draft_len == expected_drafts
        && !candidate.grammar_active
        && !candidate.finished
        && configuration_matches
        && !candidate.policy_required
        && !candidate.has_tree
    {
        FlatBatchDecision::Eligible
    } else {
        FlatBatchDecision::Ineligible
    }
}

pub(crate) trait FlatProposalFrame {
    fn has_tree(&self) -> bool;
    fn take_drafts(&mut self) -> Vec<u32>;
}

impl FlatProposalFrame for ActiveSeq {
    fn has_tree(&self) -> bool {
        self.pending_tree_payload.is_some()
    }

    fn take_drafts(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.pending_drafts)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FlatBatchPreflight {
    Ready,
    Reject,
}

pub(crate) fn flat_batch_preflight_at<F: FlatProposalFrame>(
    frames: &[F],
    indices: &[usize],
) -> FlatBatchPreflight {
    if indices.is_empty() {
        return FlatBatchPreflight::Reject;
    }
    for (position, &index) in indices.iter().enumerate() {
        let Some(frame) = frames.get(index) else {
            return FlatBatchPreflight::Reject;
        };
        if indices[..position].contains(&index) || frame.has_tree() {
            return FlatBatchPreflight::Reject;
        }
    }
    FlatBatchPreflight::Ready
}

pub(crate) fn take_flat_batch_at<F: FlatProposalFrame>(
    frames: &mut [F],
    indices: &[usize],
) -> Option<Vec<Vec<u32>>> {
    if flat_batch_preflight_at(frames, indices) != FlatBatchPreflight::Ready {
        return None;
    }
    Some(
        indices
            .iter()
            .map(|&index| frames[index].take_drafts())
            .collect(),
    )
}

#[cfg(test)]
#[path = "flat_batch_lifecycle_tests.rs"]
mod tests;
