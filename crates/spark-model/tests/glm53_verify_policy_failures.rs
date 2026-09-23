// SPDX-License-Identifier: AGPL-3.0-only

#[path = "support/glm53_verify_policy_fixture.rs"]
mod fixture;
#[path = "../src/model/glm53/verify_policy_transaction.rs"]
#[allow(dead_code)] // This target exercises failure paths, not successful receipt publication.
mod verify_policy_transaction;

use fixture::fixture;
use verify_policy_transaction::{VerifyRequest, run_verify_policy_transaction};

#[test]
fn copy_errors_and_short_success_never_reach_policy_or_commit() {
    for short in [false, true] {
        let request = VerifyRequest::new(4, 64, 3, &[0, 1]).unwrap();
        let (mut source, mut policy, mut target, events) =
            fixture(&[vec![0., 10., 9.], vec![0., 1., 12.]]);
        source.short = short;
        source.fail = !short;
        assert!(
            run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target).is_err()
        );
        assert!(target.commits.is_empty());
        assert_eq!(target.position, 4);
        assert_eq!(target.aborts, 1);
        assert!(policy.observed_rows.is_empty());
        assert!(!events.borrow().contains(&"commit"));
    }
}

#[test]
fn failed_policy_or_grammar_advance_restores_mutations_and_aborts_staging() {
    for failure in 0..3 {
        let request = VerifyRequest::new(4, 64, 3, &[0, 1, 1]).unwrap();
        let (mut source, mut policy, mut target, _) =
            fixture(&[vec![0., 10., 9.], vec![0., 10., 9.], vec![0., 1., 12.]]);
        match failure {
            0 => policy.fail_pick = Some(1),
            1 => policy.fail_advance = true,
            _ => policy.invalid_pick = Some(3),
        }
        assert!(
            run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target).is_err()
        );
        assert!(policy.history.is_empty());
        assert_eq!(policy.counters, [4, 5, 6]);
        assert_eq!(target.position, 4);
        assert_eq!(target.aborts, 1);
        assert!(target.commits.is_empty());
    }
}

#[test]
fn rollback_failure_is_fatal_not_a_raw_pick_fallback() {
    for policy_restore in [false, true] {
        let request = VerifyRequest::new(4, 64, 3, &[0, 1]).unwrap();
        let (mut source, mut policy, mut target, _) =
            fixture(&[vec![0., 10., 9.], vec![0., 1., 12.]]);
        policy.fail_restore = policy_restore;
        if !policy_restore {
            policy.fail_pick = Some(0);
            target.fail_abort = true;
        }
        assert!(
            run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target).is_err()
        );
        assert!(target.poisoned);
        assert!(target.commits.is_empty());
    }
}

#[test]
fn irreversible_commit_failure_or_wrong_position_never_returns_a_publishable_receipt() {
    for wrong_position in [false, true] {
        let request = VerifyRequest::new(4, 64, 3, &[0, 1]).unwrap();
        let (mut source, mut policy, mut target, events) =
            fixture(&[vec![0., 10., 9.], vec![0., 1., 12.]]);
        target.wrong_position = wrong_position;
        target.fail_commit = !wrong_position;
        assert!(
            run_verify_policy_transaction(&request, &mut source, &mut policy, &mut target).is_err()
        );
        assert!(target.poisoned);
        assert_eq!(target.commits, [2]);
        assert_eq!(policy.counters, [4, 5, 6]);
        assert_eq!(
            target.aborts, 0,
            "DSA-only rollback must not claim to undo persistent KDA commit"
        );
        assert!(!events.borrow().contains(&"publish"));
    }
}
