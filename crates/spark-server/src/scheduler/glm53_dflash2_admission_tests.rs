// SPDX-License-Identifier: AGPL-3.0-only

use std::{
    collections::HashSet,
    sync::atomic::{AtomicU64, Ordering},
    thread,
};

use super::*;

fn new_policy(epoch: u64) -> Glm53Dflash2SchedulerPolicy {
    Glm53Dflash2SchedulerPolicy::new(epoch).unwrap()
}

fn request(policy: &Glm53Dflash2SchedulerPolicy, anchor: u64) -> Glm53Dflash2SchedulerRequest {
    Glm53Dflash2SchedulerRequest::greedy(policy.contract_epoch(), anchor)
}

fn admitted(policy: &mut Glm53Dflash2SchedulerPolicy, anchor: u64) -> Glm53Dflash2Decision {
    policy
        .admit(request(policy, anchor), PreAdmissionEffects::default())
        .unwrap()
}

fn receipt(accepted_drafts: u8) -> VerificationReceipt {
    VerificationReceipt {
        accepted_drafts,
        target_status: 0,
        drafter_status: 0,
        async_complete: true,
    }
}

fn started(policy: &mut Glm53Dflash2SchedulerPolicy, anchor: u64) -> Glm53Dflash2Decision {
    let mut decision = admitted(policy, anchor);
    decision
        .begin_effects(policy, decision.transaction())
        .unwrap();
    decision
}

fn v(
    d: &mut Glm53Dflash2Decision,
    p: &Glm53Dflash2SchedulerPolicy,
    t: SchedulerTransaction,
    r: VerificationReceipt,
) -> anyhow::Result<u8> {
    d.verify(p, t, Glm53VerifyCandidate::DflashGamma, 7, r)
}

#[test]
fn admission_is_exact_greedy_and_precedes_every_effect() {
    assert!(Glm53Dflash2SchedulerPolicy::new(0).is_err());
    let mut policy = new_policy(1);
    for temperature in [0.01, -0.01, f32::NAN, f32::INFINITY] {
        let mut request = request(&policy, 0);
        request.temperature = temperature;
        assert!(
            policy
                .admit(request, PreAdmissionEffects::default())
                .is_err()
        );
    }
    for effects in [
        PreAdmissionEffects {
            prefills: 1,
            ..Default::default()
        },
        PreAdmissionEffects {
            sequence_allocations: 1,
            ..Default::default()
        },
        PreAdmissionEffects {
            target_effects: 1,
            ..Default::default()
        },
        PreAdmissionEffects {
            drafter_effects: 1,
            ..Default::default()
        },
    ] {
        assert!(policy.admit(request(&policy, 0), effects).is_err());
    }
    for mutation in 0..3 {
        let mut request = request(&policy, 0);
        match mutation {
            0 => request.block_tokens = 7,
            1 => request.returned_drafts = 6,
            2 => request.window = 2_047,
            _ => unreachable!(),
        }
        assert!(
            policy
                .admit(request, PreAdmissionEffects::default())
                .is_err()
        );
    }
    let mut stale = request(&policy, 0);
    stale.contract_epoch += 1;
    assert!(policy.admit(stale, PreAdmissionEffects::default()).is_err());
    let decision = admitted(&mut policy, 0);
    assert_eq!(decision.route(), Glm53VerifyRoute::DflashGamma8);
    assert_eq!(decision.anchor_position(), 0);
    assert_eq!(decision.transaction().sequence(), 1);
    assert_eq!(
        decision.transaction().policy_owner_id(),
        policy.policy_owner_id()
    );
}

#[test]
fn full_context_boundary_is_inclusive_only_for_the_last_gamma_row() {
    let mut policy = new_policy(2);
    admitted(&mut policy, GLM53_MAX_POSITION_INCLUSIVE - 7);
    assert!(
        policy
            .admit(
                request(&policy, GLM53_MAX_POSITION_INCLUSIVE - 6),
                PreAdmissionEffects::default(),
            )
            .is_err()
    );
    assert!(
        policy
            .admit(request(&policy, u64::MAX), PreAdmissionEffects::default(),)
            .is_err()
    );
}

#[test]
fn seven_drafts_can_never_fall_through_to_generic_k4() {
    let mut policy = new_policy(3);
    let mut decision = started(&mut policy, 16);
    let transaction = decision.transaction();
    assert!(
        decision
            .verify(
                &policy,
                transaction,
                Glm53VerifyCandidate::GenericK4,
                7,
                receipt(7),
            )
            .is_err()
    );
    assert_eq!(decision.phase(), DecisionPhase::Failed);
    assert!(decision.emit(&policy, transaction, 8).is_err());

    let mut policy = new_policy(4);
    let mut wrong_length = started(&mut policy, 16);
    let transaction = wrong_length.transaction();
    assert!(
        wrong_length
            .verify(
                &policy,
                transaction,
                Glm53VerifyCandidate::DflashGamma,
                4,
                receipt(3),
            )
            .is_err()
    );
    assert_eq!(wrong_length.phase(), DecisionPhase::Failed);
}

fn run_accept_case(accepted_drafts: u8) {
    let mut policy = new_policy(10 + u64::from(accepted_drafts));
    let mut decision = started(&mut policy, 32);
    let transaction = decision.transaction();
    let rows = v(
        &mut decision,
        &policy,
        transaction,
        receipt(accepted_drafts),
    )
    .unwrap();
    assert_eq!(rows, accepted_drafts + 1);
    assert_eq!(decision.accepted_rows(), Some(rows));
    decision
        .commit_target(&policy, transaction, rows, 0, true)
        .unwrap();
    decision
        .commit_capture(&policy, transaction, rows, 0, true)
        .unwrap();
    decision.emit(&policy, transaction, rows).unwrap();
    assert_eq!(decision.phase(), DecisionPhase::Emitted);
}

#[test]
fn accept_zero_partial_and_full_publish_only_after_both_commits() {
    run_accept_case(0);
    run_accept_case(3);
    run_accept_case(7);
}

#[test]
fn emit_or_capture_before_target_commit_is_terminal() {
    for early in 0..3 {
        let mut policy = new_policy(20 + early);
        let mut decision = started(&mut policy, 64);
        let transaction = decision.transaction();
        if early != 0 {
            v(&mut decision, &policy, transaction, receipt(3)).unwrap();
        }
        if early == 2 {
            decision
                .commit_target(&policy, transaction, 4, 0, true)
                .unwrap();
        }
        let result = match early {
            0 => decision.emit(&policy, transaction, 4),
            1 => decision.commit_capture(&policy, transaction, 4, 0, true),
            2 => decision.emit(&policy, transaction, 4),
            _ => unreachable!(),
        };
        assert!(result.is_err());
        assert_eq!(decision.phase(), DecisionPhase::Failed);
    }
}

#[test]
fn status_async_and_partial_commit_failures_never_emit() {
    for failure in 0..7 {
        let mut policy = new_policy(30 + failure);
        let mut decision = started(&mut policy, 96);
        let transaction = decision.transaction();
        let result = match failure {
            0 => v(
                &mut decision,
                &policy,
                transaction,
                VerificationReceipt {
                    target_status: 1,
                    ..receipt(3)
                },
            )
            .map(|_| ()),
            1 => v(
                &mut decision,
                &policy,
                transaction,
                VerificationReceipt {
                    drafter_status: 1,
                    ..receipt(3)
                },
            )
            .map(|_| ()),
            2 => v(
                &mut decision,
                &policy,
                transaction,
                VerificationReceipt {
                    async_complete: false,
                    ..receipt(3)
                },
            )
            .map(|_| ()),
            3 => v(&mut decision, &policy, transaction, receipt(8)).map(|_| ()),
            4..=6 => {
                v(&mut decision, &policy, transaction, receipt(3)).unwrap();
                match failure {
                    4 => decision.commit_target(&policy, transaction, 3, 0, true),
                    5 => decision.commit_target(&policy, transaction, 4, 9, true),
                    6 => decision.commit_target(&policy, transaction, 4, 0, false),
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        };
        assert!(result.is_err());
        assert_eq!(decision.phase(), DecisionPhase::Failed);
        assert!(decision.emit(&policy, transaction, 4).is_err());
    }

    for failure in 0..3 {
        let mut policy = new_policy(40 + failure);
        let mut capture_failure = started(&mut policy, 96);
        let transaction = capture_failure.transaction();
        v(&mut capture_failure, &policy, transaction, receipt(3)).unwrap();
        capture_failure
            .commit_target(&policy, transaction, 4, 0, true)
            .unwrap();
        let result = match failure {
            0 => capture_failure.commit_capture(&policy, transaction, 3, 0, true),
            1 => capture_failure.commit_capture(&policy, transaction, 4, 9, true),
            2 => capture_failure.commit_capture(&policy, transaction, 4, 0, false),
            _ => unreachable!(),
        };
        assert!(result.is_err());
        assert_eq!(capture_failure.phase(), DecisionPhase::Failed);
    }

    let mut policy = new_policy(45);
    let mut wrong_emission = started(&mut policy, 96);
    let transaction = wrong_emission.transaction();
    v(&mut wrong_emission, &policy, transaction, receipt(3)).unwrap();
    wrong_emission
        .commit_target(&policy, transaction, 4, 0, true)
        .unwrap();
    wrong_emission
        .commit_capture(&policy, transaction, 4, 0, true)
        .unwrap();
    assert!(wrong_emission.emit(&policy, transaction, 3).is_err());
    assert_eq!(wrong_emission.phase(), DecisionPhase::Failed);
}

#[test]
fn stale_transaction_or_contract_rejects_without_poisoning_current_decision() {
    let mut policy = new_policy(50);
    let mut decision = admitted(&mut policy, 128);
    let transaction = decision.transaction();
    let stale = SchedulerTransaction::forged_for_test(
        transaction.contract_epoch(),
        transaction.policy_owner_id(),
        transaction.sequence() + 1,
    );
    assert!(decision.begin_effects(&policy, stale).is_err());
    assert_eq!(decision.phase(), DecisionPhase::Admitted);
    let replacement_policy = new_policy(51);
    assert!(
        decision
            .begin_effects(&replacement_policy, transaction)
            .is_err()
    );
    assert_eq!(decision.phase(), DecisionPhase::Admitted);
    decision.begin_effects(&policy, transaction).unwrap();
}

#[test]
fn minted_owner_and_sequence_reject_recreated_policy_and_cross_decision() {
    let mut first = Glm53Dflash2SchedulerPolicy::new(60).unwrap();
    let mut second = Glm53Dflash2SchedulerPolicy::new(60).unwrap();
    let mut first_decision = admitted(&mut first, 0);
    let second_decision = admitted(&mut second, 0);
    let first_transaction = first_decision.transaction();
    let second_transaction = second_decision.transaction();
    assert_eq!(first_transaction.sequence(), second_transaction.sequence());
    assert_ne!(first.policy_owner_id(), second.policy_owner_id());
    assert_ne!(first_transaction, second_transaction);
    assert!(
        first_decision
            .begin_effects(&second, first_transaction)
            .is_err()
    );
    assert!(
        first_decision
            .begin_effects(&second, second_transaction)
            .is_err()
    );
    assert_eq!(first_decision.phase(), DecisionPhase::Admitted);
    first_decision
        .begin_effects(&first, first_transaction)
        .unwrap();

    let mut policy = Glm53Dflash2SchedulerPolicy::new(61).unwrap();
    let mut earlier = admitted(&mut policy, 8);
    let later = admitted(&mut policy, 16);
    assert_eq!(
        (
            earlier.transaction().sequence(),
            later.transaction().sequence()
        ),
        (1, 2)
    );
    assert!(earlier.begin_effects(&policy, later.transaction()).is_err());
    assert_eq!(earlier.phase(), DecisionPhase::Admitted);
}

#[test]
fn atomic_owner_allocator_is_unique_under_concurrency_and_fails_closed() {
    let handles: Vec<_> = (0..32)
        .map(|_| {
            thread::spawn(|| {
                let mut policy = new_policy(70);
                let decision = admitted(&mut policy, 0);
                (policy.policy_owner_id(), decision.transaction())
            })
        })
        .collect();
    let identities: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();
    assert!(identities.iter().all(|(owner, transaction)| {
        *owner != 0 && transaction.policy_owner_id() == *owner && transaction.sequence() == 1
    }));
    assert_eq!(
        identities
            .iter()
            .map(|(owner, _)| *owner)
            .collect::<HashSet<_>>()
            .len(),
        32
    );
    let exhausted = AtomicU64::new(u64::MAX - 1);
    assert_eq!(allocate_owner_from(&exhausted).unwrap(), u64::MAX - 1);
    assert_eq!(exhausted.load(Ordering::Relaxed), u64::MAX);
    for _ in 0..2 {
        assert!(allocate_owner_from(&exhausted).is_err());
        assert_eq!(exhausted.load(Ordering::Relaxed), u64::MAX);
    }
    assert!(allocate_owner_from(&AtomicU64::new(0)).is_err());
}

fn complete_source_contract(source: &str) -> bool {
    const WRAPPER: &str = "fn allocate_policy_owner_id() -> Result<u64> {\n    allocate_owner_from(&NEXT_POLICY_OWNER_ID)\n}";
    source_contract(source) && source.matches(WRAPPER).count() == 1
}

fn mutation(source: &str, from: &str, to: &str) -> String {
    let production = source.split_once("#[cfg(test)]\nfn section").unwrap().0;
    assert_eq!(production.matches(from).count(), 1, "mutation: {from}");
    source.replacen(from, to, 1)
}

#[test]
fn owner_wrapper_rejects_unchecked_atomic_racy_and_zero_bypasses() {
    let source = include_str!("glm53_dflash2_admission.rs");
    assert!(complete_source_contract(source));
    for replacement in [
        "Ok(NEXT_POLICY_OWNER_ID.fetch_add(1, Ordering::Relaxed))",
        "{ let owner = NEXT_POLICY_OWNER_ID.load(Ordering::Relaxed); NEXT_POLICY_OWNER_ID.store(owner.wrapping_add(1), Ordering::Relaxed); Ok(owner) }",
        "Ok(0)",
    ] {
        assert!(!complete_source_contract(&mutation(
            source,
            "allocate_owner_from(&NEXT_POLICY_OWNER_ID)",
            replacement,
        )));
    }
    assert!(!complete_source_contract(&mutation(
        source,
        "counter\n        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |owner| {\n            (owner != 0).then(|| owner.checked_add(1)).flatten()\n        })\n        .map_err(|_| anyhow::anyhow!(\"GLM DFlash2 policy owner ID exhausted\"))",
        "Ok(counter.fetch_add(1, Ordering::Relaxed))",
    )));
}

#[test]
fn every_missing_or_repeated_effect_phase_is_terminal() {
    let mut policy = new_policy(62);
    let mut before_begin = admitted(&mut policy, 0);
    let transaction = before_begin.transaction();
    assert!(v(&mut before_begin, &policy, transaction, receipt(3)).is_err());
    assert_eq!(before_begin.phase(), DecisionPhase::Failed);

    let mut before_verify = started(&mut policy, 8);
    let transaction = before_verify.transaction();
    assert!(
        before_verify
            .commit_target(&policy, transaction, 4, 0, true)
            .is_err()
    );
    assert_eq!(before_verify.phase(), DecisionPhase::Failed);

    let mut repeated_begin = started(&mut policy, 16);
    let transaction = repeated_begin.transaction();
    assert!(repeated_begin.begin_effects(&policy, transaction).is_err());
    assert_eq!(repeated_begin.phase(), DecisionPhase::Failed);
}

#[test]
fn source_predicate_rejects_owner_and_effect_order_mutations() {
    let source = include_str!("glm53_dflash2_admission.rs");
    assert!(complete_source_contract(source));
    for (from, to) in [
        (
            "self.next_sequence = self\n            .next_sequence\n            .checked_add(1)",
            "self.next_sequence = self\n            .next_sequence\n            .checked_add(0)",
        ),
        ("AtomicU64::new(1)", "AtomicU64::new(0)"),
        (
            "let policy_owner_id = allocate_policy_owner_id()?;",
            "let policy_owner_id = 1;",
        ),
        ("owner.checked_add(1)", "Some(owner.wrapping_add(1))"),
        (
            "policy_owner_id: self.policy_owner_id",
            "policy_owner_id: 1",
        ),
        ("self.require_phase(DecisionPhase::Admitted)?;", ""),
        ("self.require_phase(DecisionPhase::EffectsStarted)?;", ""),
        ("self.require_phase(DecisionPhase::Verified)?;", ""),
        (
            "self.require_phase(DecisionPhase::TargetCommitted)?;",
            "self.require_phase(DecisionPhase::Verified)?;",
        ),
        (
            "self.require_phase(DecisionPhase::CaptureCommitted)?;",
            "self.require_phase(DecisionPhase::TargetCommitted)?;",
        ),
        (
            "policy.policy_owner_id == self.transaction.policy_owner_id",
            "policy.policy_owner_id != self.transaction.policy_owner_id",
        ),
    ] {
        assert!(!complete_source_contract(&mutation(source, from, to)));
    }
}
