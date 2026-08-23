// SPDX-License-Identifier: AGPL-3.0-only

use std::cell::Cell;

use super::*;

fn candidate(width: FixedBatchWidth) -> FixedBatchEligibility {
    FixedBatchEligibility {
        draft_len: match width {
            FixedBatchWidth::K2 => 1,
            FixedBatchWidth::K3 => 2,
        },
        grammar_active: false,
        finished: false,
        configured_num_drafts: 1,
        policy_required: false,
        has_tree: false,
    }
}

#[test]
fn typed_k2_and_k3_eligibility_reject_every_mutated_guard() {
    for width in [FixedBatchWidth::K2, FixedBatchWidth::K3] {
        let valid = candidate(width);
        assert!(fixed_batch_decision(width, valid).is_eligible());

        let mutations = [
            FixedBatchEligibility {
                draft_len: valid.draft_len + 1,
                ..valid
            },
            FixedBatchEligibility {
                grammar_active: true,
                ..valid
            },
            FixedBatchEligibility {
                finished: true,
                ..valid
            },
            FixedBatchEligibility {
                policy_required: true,
                ..valid
            },
            FixedBatchEligibility {
                has_tree: true,
                ..valid
            },
        ];
        for mutation in mutations {
            assert_eq!(
                fixed_batch_decision(width, mutation),
                FlatBatchDecision::Ineligible
            );
        }
    }

    let wrong_k2_configuration = FixedBatchEligibility {
        configured_num_drafts: 2,
        ..candidate(FixedBatchWidth::K2)
    };
    assert_eq!(
        fixed_batch_decision(FixedBatchWidth::K2, wrong_k2_configuration),
        FlatBatchDecision::Ineligible
    );
}

struct Frame {
    drafts: Vec<u32>,
    has_tree: bool,
    takes: Cell<usize>,
}

impl FlatProposalFrame for Frame {
    fn has_tree(&self) -> bool {
        self.has_tree
    }

    fn take_drafts(&mut self) -> Vec<u32> {
        self.takes.set(self.takes.get() + 1);
        std::mem::take(&mut self.drafts)
    }
}

fn frame(token: u32, has_tree: bool) -> Frame {
    Frame {
        drafts: vec![token],
        has_tree,
        takes: Cell::new(0),
    }
}

#[test]
fn late_tree_rejects_before_any_frame_is_taken() {
    let mut frames = [frame(1, false), frame(2, false), frame(3, true)];
    assert_eq!(
        flat_batch_preflight_at(&frames, &[0, 1, 2]),
        FlatBatchPreflight::Reject
    );
    assert_eq!(take_flat_batch_at(&mut frames, &[0, 1, 2]), None);
    assert!(frames.iter().all(|item| item.takes.get() == 0));
    assert_eq!(frames[0].drafts, [1]);
    assert_eq!(frames[1].drafts, [2]);
}

#[test]
fn ready_batch_takes_every_distinct_frame_once() {
    let mut frames = [frame(1, false), frame(2, false), frame(3, false)];
    assert_eq!(
        take_flat_batch_at(&mut frames, &[2, 0]),
        Some(vec![vec![3], vec![1]])
    );
    assert_eq!(frames[0].takes.get(), 1);
    assert_eq!(frames[1].takes.get(), 0);
    assert_eq!(frames[2].takes.get(), 1);
}

#[test]
fn invalid_or_duplicate_indices_reject_without_a_take() {
    for indices in [&[0, 0][..], &[0, 2][..]] {
        let mut frames = [frame(1, false), frame(2, false)];
        assert_eq!(take_flat_batch_at(&mut frames, indices), None);
        assert!(frames.iter().all(|item| item.takes.get() == 0));
    }
}
