// SPDX-License-Identifier: AGPL-3.0-only

//! RED gates for the actual GLM policy transaction helper, not a test-only model.

#[path = "support/glm53_verify_policy_fixture.rs"]
mod fixture;
#[path = "../src/model/glm53/verify_policy_transaction.rs"]
#[allow(dead_code)] // This target covers commits; explicit replay has its own integration target.
mod verify_policy_transaction;

use fixture::{committed, encoded, fixture};
use verify_policy_transaction::{StagedLogits, VerifyRequest, run_verify_policy_transaction};

#[test]
fn unique_max_repetition_flip_commits_only_anchor_and_publishes_matching_host_prefix() {
    // Raw A=10 > B=9. Shipping repetition penalty turns A into 5, so B wins.
    let rows = vec![vec![0., 10., 9.], vec![0., 1., 12.], vec![0., 8., 1.]];
    let request = VerifyRequest::new(4, 64, 3, &[0, 1, 2]).unwrap();
    let (mut source, mut policy, mut target, _) = fixture(&rows);
    policy.history = vec![1];
    policy.params.repetition_penalty = 2.;
    let receipt = committed(
        run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target).unwrap(),
    );
    assert_eq!(receipt.accepted_drafts(), 0);
    assert_eq!(target.commits, [1]);
    assert_eq!(target.position, 5); // The broken raw-commit path reaches 7.
    assert_eq!(policy.observed_history, [vec![1]]); // Never process rejected suffix rows.
    assert_eq!(policy.history, [1]);
    assert_eq!(policy.counters, [4, 5, 6]);
    let (mut tokens, mut len, mut valid) = (vec![0; 4], 4, 4);
    let published = receipt.publish(&mut tokens, &mut len, &mut valid).unwrap();
    assert_eq!(tokens, [0, 0, 0, 0, 0]);
    assert_eq!((len, valid), (target.position, target.position));
    assert_eq!(published.emitted_tokens(), &[2]);
    assert!(!published.terminal());
}

#[test]
fn all_supported_widths_copy_distinct_complete_rows_and_preserve_them_through_replay() {
    for count in 2..=8 {
        let rows: Vec<Vec<f32>> = (0..count)
            .map(|row| {
                let mut values: Vec<f32> = (0..8).map(|col| (row * 8 + col) as f32 / 64.).collect();
                values[(row + 1) % 8] = 20. + row as f32;
                values
            })
            .collect();
        let inputs: Vec<u32> = (0..count as u32).collect();
        let request = VerifyRequest::new(4, 64, 8, &inputs).unwrap();
        let (mut source, _, _, _) = fixture(&rows);
        let staged = StagedLogits::read(&request, &mut source).unwrap();
        let expected = encoded(&rows);
        source.bytes.borrow_mut().fill(0); // Owned capture must not alias source/replay storage.
        for row in 0..count {
            assert_eq!(
                staged.row(row).unwrap(),
                &expected[row * 16..(row + 1) * 16]
            );
        }
        assert!(staged.row(count).is_err());
        let (mut source, mut policy, mut target, _) = fixture(&rows);
        let receipt = committed(
            run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target).unwrap(),
        );
        assert_eq!(receipt.accepted_drafts(), count - 1);
        assert_eq!(policy.observed_rows.concat(), expected);
        assert!(source.bytes.borrow().iter().all(|&value| value == 0));
        assert_eq!(target.commits, [count]);
    }
}

#[test]
fn second_row_sees_accepted_prefix_history_not_stale_generation_history() {
    let rows = vec![vec![0., 10., 9.], vec![0., 10., 9.], vec![0., 8., 1.]];
    let request = VerifyRequest::new(4, 64, 3, &[0, 1, 1]).unwrap();
    let (mut source, mut policy, mut target, _) = fixture(&rows);
    policy.params.repetition_penalty = 2.;
    let receipt = committed(
        run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target).unwrap(),
    );
    assert_eq!(receipt.accepted_drafts(), 1);
    assert_eq!(policy.observed_history, [vec![], vec![1]]);
    assert!(policy.history.is_empty());
    assert_eq!(policy.counters, [4, 5, 6]);
    assert_eq!(target.commits, [2]);
}

#[test]
fn caller_bias_and_sequential_masks_reach_the_same_authoritative_decision() {
    let rows = vec![vec![0., 10., 9.], vec![0., 10., 9.], vec![0., 8., 1.]];
    for with_bias in [false, true] {
        let request = VerifyRequest::new(4, 64, 3, &[0, 1, 1]).unwrap();
        let (mut source, mut policy, mut target, _) = fixture(&rows);
        if with_bias {
            policy.params.logit_bias = vec![(2, 2.)];
        } else {
            policy.masks = vec![vec![], vec![1]];
        }
        let receipt = committed(
            run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target).unwrap(),
        );
        assert_eq!(receipt.accepted_drafts(), usize::from(!with_bias));
        assert_eq!(target.commits, [if with_bias { 1 } else { 2 }]);
        assert!(policy.history.is_empty());
    }
}

#[test]
fn commit_and_host_publication_precede_terminal_accepted_draft_or_bonus() {
    for terminal in [1, 2] {
        let rows = vec![vec![0., 10., 9.], vec![0., 1., 12.], vec![0., 8., 1.]];
        let request = VerifyRequest::new(4, 64, 3, &[0, 1, 1]).unwrap();
        let (mut source, mut policy, mut target, events) = fixture(&rows);
        policy.terminal = Some(terminal);
        let receipt = committed(
            run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target).unwrap(),
        );
        let (mut tokens, mut len, mut valid) = (vec![0; 4], 4, 4);
        let published = receipt.publish(&mut tokens, &mut len, &mut valid).unwrap();
        events.borrow_mut().push("publish");
        assert!(published.terminal());
        assert_eq!((len, valid), (target.position, target.position));
        for &token in published.emitted_tokens() {
            events.borrow_mut().push("emit");
            if token == terminal {
                break;
            }
        }
        let events = events.borrow();
        let at = |event| events.iter().position(|&v| v == event).unwrap();
        assert!(at("policy_restore") < at("commit"));
        assert!(at("commit") < at("publish"));
        assert!(at("publish") < at("emit"));
        assert_eq!(target.commits, [2]);
        assert_eq!(policy.observed_rows.len(), terminal as usize);
    }
}

#[test]
fn hostile_geometry_and_stale_host_prefix_are_rejected_without_partial_publication() {
    for (start, capacity, vocab, inputs) in [
        (4, 64, 3, vec![0]),
        (4, 64, 3, vec![0; 9]),
        (4, 5, 3, vec![0, 1]),
        (usize::MAX, usize::MAX, 3, vec![0, 1]),
        (4, 64, 0, vec![0, 1]),
        (4, 64, 154_881, vec![0, 1]),
        (4, 64, 3, vec![0, 3]),
    ] {
        assert!(VerifyRequest::new(start, capacity, vocab, &inputs).is_err());
    }
    for stale in 0..3 {
        let request = VerifyRequest::new(4, 64, 3, &[0, 1]).unwrap();
        let (mut source, mut policy, mut target, _) =
            fixture(&[vec![0., 10., 9.], vec![0., 1., 12.]]);
        let receipt = committed(
            run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target).unwrap(),
        );
        let (mut tokens, mut len, mut valid) = (vec![0; 4], 4, 4);
        match stale {
            0 => {
                tokens.pop();
            }
            1 => len -= 1,
            _ => valid -= 1,
        }
        let before = (tokens.clone(), len, valid);
        assert!(receipt.publish(&mut tokens, &mut len, &mut valid).is_err());
        assert_eq!((tokens, len, valid), before);
    }
}
