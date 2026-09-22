// SPDX-License-Identifier: AGPL-3.0-only

//! Replay is a typed, successfully restored outcome, never an error downcast.
#[path = "support/glm53_verify_policy_fixture.rs"]
mod fixture;
#[path = "../src/model/glm53/verify_policy_transaction.rs"]
#[allow(dead_code)] // This target tests restore authority rather than publication.
mod verify_policy_transaction;

use anyhow::Result;
use fixture::{Policy, fixture};
use verify_policy_transaction::{
    PolicyAdvance, VerifyOutcome, VerifyPolicy, VerifyRequest, run_verify_policy_transaction,
};

struct ReplayPolicy {
    ordinary: Policy,
    after: usize,
    advances: usize,
}

impl VerifyPolicy for ReplayPolicy {
    fn checkpoint(&mut self) -> Result<()> {
        self.ordinary.checkpoint()
    }
    fn pick(&mut self, row: usize, bytes: &[u8]) -> Result<u32> {
        self.ordinary.pick(row, bytes)
    }
    fn advance(&mut self, token: u32) -> Result<PolicyAdvance> {
        let step = self.ordinary.advance(token)?;
        self.advances += 1;
        Ok(if self.advances == self.after {
            PolicyAdvance::OrdinaryReplay
        } else {
            step
        })
    }
    fn restore(&mut self) -> Result<()> {
        self.ordinary.restore()
    }
}

#[test]
fn replay_after_any_accepted_prefix_restores_both_owners_without_commit_or_receipt() {
    for rows in 2..=8 {
        for after in 1..=rows {
            let mut inputs = vec![0];
            inputs.resize(rows, 1);
            let request = VerifyRequest::new(4, 64, 3, &inputs).unwrap();
            let (mut source, ordinary, mut target, events) =
                fixture(&vec![vec![0., 10., 9.]; rows]);
            let mut policy = ReplayPolicy {
                ordinary,
                after,
                advances: 0,
            };
            let outcome =
                run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target)
                    .unwrap();
            let VerifyOutcome::RestoredForOrdinaryReplay(restored) = outcome else {
                panic!("unsupported persistent effect must not return commit authority");
            };
            assert_eq!((restored.start(), restored.anchor()), (4, 0));
            assert_eq!(target.position, 4);
            assert_eq!(target.aborts, 1);
            assert!(target.commits.is_empty() && !target.poisoned);
            assert!(policy.ordinary.history.is_empty());
            assert_eq!(policy.ordinary.counters, [4, 5, 6]);
            assert_eq!(policy.ordinary.observed_rows.len(), after);
            let events = events.borrow();
            let restored_at = events.iter().position(|v| *v == "policy_restore").unwrap();
            let aborted_at = events.iter().position(|v| *v == "abort").unwrap();
            assert!(restored_at < aborted_at);
            assert!(!events.contains(&"commit"));
        }
    }
}

#[test]
fn policy_or_target_restore_failure_never_returns_permission_for_ordinary_replay() {
    for fail_policy in [false, true] {
        let request = VerifyRequest::new(4, 64, 3, &[0, 1, 1]).unwrap();
        let (mut source, mut ordinary, mut target, _) = fixture(&vec![vec![0., 10., 9.]; 3]);
        ordinary.fail_restore = fail_policy;
        target.fail_abort = !fail_policy;
        let mut policy = ReplayPolicy {
            ordinary,
            after: 2,
            advances: 0,
        };
        assert!(
            run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target,)
                .is_err()
        );
        assert!(target.poisoned);
        assert_eq!(target.position, 4);
        assert_eq!(target.aborts, 1);
        assert!(target.commits.is_empty());
    }
}

#[test]
fn generic_policy_failure_is_not_mistaken_for_an_ordinary_effect_request() {
    let request = VerifyRequest::new(4, 64, 3, &[0, 1]).unwrap();
    let (mut source, mut ordinary, mut target, _) = fixture(&vec![vec![0., 10., 9.]; 2]);
    ordinary.fail_advance = true;
    let mut policy = ReplayPolicy {
        ordinary,
        after: 1,
        advances: 0,
    };
    assert!(
        run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target,).is_err()
    );
    assert!(!target.poisoned);
    assert_eq!(target.aborts, 1);
    assert!(target.commits.is_empty());
    assert!(policy.ordinary.history.is_empty());
}

#[test]
fn subsequent_request_does_not_inherit_the_restored_policy_prefix() {
    let request = VerifyRequest::new(4, 64, 3, &[0, 1, 1]).unwrap();
    let (mut source, ordinary, mut target, _) = fixture(&vec![vec![0., 10., 9.]; 3]);
    let mut policy = ReplayPolicy {
        ordinary,
        after: 2,
        advances: 0,
    };
    assert!(matches!(
        run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target,).unwrap(),
        VerifyOutcome::RestoredForOrdinaryReplay(_)
    ));
    policy.after = usize::MAX;
    policy.advances = 0;
    policy.ordinary.observed_history.clear();
    let outcome =
        run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target).unwrap();
    let VerifyOutcome::Committed(receipt) = outcome else {
        panic!("spurious replay");
    };
    assert_eq!(receipt.accepted_drafts(), 2);
    assert_eq!(
        policy.ordinary.observed_history,
        [vec![], vec![1], vec![1, 1]]
    );
    assert!(policy.ordinary.history.is_empty());
    assert_eq!(target.position, 7);
}
