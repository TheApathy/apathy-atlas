// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::atomic::Ordering;

use super::*;

#[path = "proposal_lifecycle_model_test_support.rs"]
mod support;
use support::{Frame, TestModel, seq_state, tree};

#[test]
fn blanket_model_adapter_pairs_tree_flat_empty_and_error_with_one_take() {
    let cases = vec![
        (Ok(vec![10]), Some(tree(10)), Some(true), vec![10], Some(10)),
        (Ok(vec![20]), None, Some(true), vec![20], None),
        (
            Ok(Vec::new()),
            Some(tree(30)),
            Some(false),
            Vec::new(),
            None,
        ),
        (
            Err(anyhow::anyhow!("failed")),
            Some(tree(40)),
            None,
            Vec::new(),
            None,
        ),
    ];
    for (proposal, offered_tree, outcome, drafts, tree_token) in cases {
        let model = TestModel::new(proposal, offered_tree);
        let mut frame = Frame {
            state: seq_state(),
            drafts: vec![1],
            tree: Some(tree(1)),
        };
        assert_eq!(
            propose_and_install(&model, &mut frame, 7, 2, None).ok(),
            outcome
        );
        assert_eq!(frame.drafts, drafts);
        assert_eq!(
            frame.tree.as_ref().map(|payload| payload.tree_token_ids[0]),
            tree_token
        );
        assert_eq!(model.proposals.load(Ordering::SeqCst), 1);
        assert_eq!(model.takes.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn skipped_proposal_still_drains_once_and_clears_the_frame() {
    let model = TestModel::new(Ok(vec![10]), Some(tree(10)));
    let mut frame = Frame {
        state: seq_state(),
        drafts: vec![1],
        tree: Some(tree(1)),
    };
    let (outcome, marker) =
        propose_and_install_with(&model, &mut frame, 7, 2, None, false, |_, result| {
            assert!(result.as_ref().unwrap().is_empty());
            9
        });
    assert!(!outcome.unwrap());
    assert_eq!(marker, 9);
    assert!(frame.drafts.is_empty() && frame.tree.is_none());
    assert_eq!(model.proposals.load(Ordering::SeqCst), 0);
    assert_eq!(model.takes.load(Ordering::SeqCst), 1);
}
