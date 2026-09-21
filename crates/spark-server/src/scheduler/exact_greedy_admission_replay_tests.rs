// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn identity(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn facts() -> ExactGreedyFrameFacts {
    ExactGreedyFrameFacts {
        position: 101,
        fed_len: 101,
        logical_len: 102,
        prompt_len: 80,
        pending_token: 42,
        vocab_size: 256,
        canonical_fed_prefix_digest: identity(5),
        target_identity: identity(1),
        proposer_identity: identity(2),
        config_identity: identity(3),
        raw_argmax_identity: identity(4),
    }
}

fn issuer() -> ExactGreedyReceiptIssuer {
    ExactGreedyReceiptIssuer::new(7, 256, 11, 12).unwrap()
}

fn issue(
    issuer: &mut ExactGreedyReceiptIssuer,
    facts: ExactGreedyFrameFacts,
    drafts: &[u32],
) -> Result<ExactGreedyFrameReceipt> {
    ExactGreedyFrameReceipt::issue(
        issuer,
        TargetTokenAuthority::ExactRawArgmax,
        ExactDraftSource::GhostModelOnly,
        ExactServingTransforms::none(),
        ExactGreedyPolicyKey::canonical(),
        facts,
        drafts,
    )
}

fn observe<'a>(receipt: &ExactGreedyFrameReceipt, drafts: &'a [u32]) -> ExactGreedyPreverify<'a> {
    ExactGreedyPreverify {
        authority: TargetTokenAuthority::ExactRawArgmax,
        source: ExactDraftSource::GhostModelOnly,
        transforms: ExactServingTransforms::none(),
        policy: ExactGreedyPolicyKey::canonical(),
        key: receipt.frame_key(),
        drafts,
    }
}

fn mutate_bound_key(key: &mut ExactGreedyFrameKey, slot: usize) {
    match slot {
        0 => key.session_nonce += 1,
        1 => key.issuer_instance_nonce += 1,
        2 => key.receipt_nonce += 1,
        3 => key.target_commit_epoch += 1,
        4 => key.proposal_epoch += 1,
        5 => key.max_context += 1,
        6 => key.facts.canonical_fed_prefix_digest = identity(6),
        7 => key.facts.target_identity = identity(7),
        8 => key.facts.proposer_identity = identity(8),
        9 => key.facts.config_identity = identity(9),
        10 => key.facts.raw_argmax_identity = identity(10),
        _ => unreachable!(),
    }
}

#[test]
fn k31_exact_boundary_succeeds_and_k_plus_two_target_rows_reject() {
    let mut boundary = facts();
    boundary.position = 68;
    boundary.fed_len = 68;
    boundary.logical_len = 69;
    boundary.prompt_len = 64;
    let drafts = [10; MAX_EXACT_DRAFTS];
    let mut raw = vec![10; MAX_EXACT_DRAFTS + 1];
    raw[MAX_EXACT_DRAFTS] = 99;

    let mut issuer = ExactGreedyReceiptIssuer::new(7, 100, 11, 12).unwrap();
    let receipt = issue(&mut issuer, boundary, &drafts).unwrap();
    let key = receipt.frame_key();
    let observed = observe(&receipt, &drafts);
    let outcome = receipt
        .admit_preverify(&mut issuer, observed)
        .unwrap()
        .finish_raw_target(
            &mut issuer,
            ExactTargetRowSource::RawTargetArgmax,
            key,
            &raw,
        )
        .unwrap();
    assert_eq!(outcome.num_accepted, MAX_EXACT_DRAFTS);
    assert_eq!(outcome.bonus, 99);

    issuer.record_target_commit(11, 12).unwrap();
    raw.push(100);
    let receipt = issue(&mut issuer, boundary, &drafts).unwrap();
    let key = receipt.frame_key();
    let observed = observe(&receipt, &drafts);
    assert!(
        receipt
            .admit_preverify(&mut issuer, observed)
            .unwrap()
            .finish_raw_target(
                &mut issuer,
                ExactTargetRowSource::RawTargetArgmax,
                key,
                &raw,
            )
            .is_err()
    );
}

#[test]
fn k_aware_extents_bind_max_context_and_reject_near_usize_overflow() {
    let drafts = [10; MAX_EXACT_DRAFTS];
    let mut too_near = facts();
    too_near.position = 69;
    too_near.fed_len = 69;
    too_near.logical_len = 70;
    too_near.prompt_len = 64;
    let mut bounded = ExactGreedyReceiptIssuer::new(7, 100, 11, 12).unwrap();
    assert!(issue(&mut bounded, too_near, &drafts).is_err());

    let mut overflow = facts();
    overflow.position = usize::MAX - 1;
    overflow.fed_len = usize::MAX - 1;
    overflow.logical_len = usize::MAX;
    overflow.prompt_len = 0;
    let mut unbounded = ExactGreedyReceiptIssuer::new(7, usize::MAX, 11, 12).unwrap();
    assert!(issue(&mut unbounded, overflow, &[10]).is_err());

    let mut issuer = issuer();
    let receipt = issue(&mut issuer, facts(), &[10]).unwrap();
    let mut observed = observe(&receipt, &[10]);
    observed.key.max_context -= 1;
    assert!(receipt.admit_preverify(&mut issuer, observed).is_err());
}

#[test]
fn canonical_prefix_digest_is_bound_at_both_phase_seams() {
    let drafts = [10];
    let mut preverify_issuer = issuer();
    let receipt = issue(&mut preverify_issuer, facts(), &drafts).unwrap();
    let mut observed = observe(&receipt, &drafts);
    observed.key.facts.canonical_fed_prefix_digest = identity(6);
    assert!(
        receipt
            .admit_preverify(&mut preverify_issuer, observed)
            .is_err()
    );

    let mut target_issuer = issuer();
    let receipt = issue(&mut target_issuer, facts(), &drafts).unwrap();
    let mut target_key = receipt.frame_key();
    let observed = observe(&receipt, &drafts);
    let permit = receipt
        .admit_preverify(&mut target_issuer, observed)
        .unwrap();
    target_key.facts.canonical_fed_prefix_digest = identity(6);
    assert!(
        permit
            .finish_raw_target(
                &mut target_issuer,
                ExactTargetRowSource::RawTargetArgmax,
                target_key,
                &[10, 11],
            )
            .is_err()
    );
}

#[test]
fn issuer_mints_unique_monotonic_frames_and_rejects_epoch_replay() {
    let drafts = [10];
    let mut lifecycle_issuer = issuer();
    let first = issue(&mut lifecycle_issuer, facts(), &drafts).unwrap();
    let first_key = first.frame_key();
    assert!(issue(&mut lifecycle_issuer, facts(), &drafts).is_err());
    let observed = observe(&first, &drafts);
    first
        .admit_preverify(&mut lifecycle_issuer, observed)
        .unwrap()
        .finish_raw_target(
            &mut lifecycle_issuer,
            ExactTargetRowSource::RawTargetArgmax,
            first_key,
            &[10, 11],
        )
        .unwrap();

    assert!(issue(&mut lifecycle_issuer, facts(), &drafts).is_err());
    assert!(lifecycle_issuer.record_target_commit(10, 12).is_err());
    assert!(lifecycle_issuer.record_target_commit(11, 13).is_err());
    lifecycle_issuer.record_target_commit(11, 12).unwrap();

    let second = issue(&mut lifecycle_issuer, facts(), &drafts).unwrap();
    let second_key = second.frame_key();
    assert_eq!(second_key.receipt_nonce, first_key.receipt_nonce + 1);
    assert_eq!(second_key.proposal_epoch, first_key.proposal_epoch + 1);
    assert_eq!(
        second_key.target_commit_epoch,
        first_key.target_commit_epoch + 1
    );
    let observed = observe(&second, &drafts);
    second
        .admit_preverify(&mut lifecycle_issuer, observed)
        .unwrap()
        .finish_raw_target(
            &mut lifecycle_issuer,
            ExactTargetRowSource::RawTargetArgmax,
            second_key,
            &[10, 11],
        )
        .unwrap();

    lifecycle_issuer.record_target_commit(12, 13).unwrap();
    let third = issue(&mut lifecycle_issuer, facts(), &drafts).unwrap();
    assert_eq!(third.frame_key().target_commit_epoch, 13);
    let mut stale = observe(&third, &drafts);
    stale.key.target_commit_epoch = 12;
    assert!(third.admit_preverify(&mut lifecycle_issuer, stale).is_err());

    let mut replay_issuer = ExactGreedyReceiptIssuer::new(7, 256, 11, 12).unwrap();
    let replay_first = issue(&mut replay_issuer, facts(), &drafts).unwrap();
    let replay_first_key = replay_first.frame_key();
    let observed = observe(&replay_first, &drafts);
    replay_first
        .admit_preverify(&mut replay_issuer, observed)
        .unwrap()
        .finish_raw_target(
            &mut replay_issuer,
            ExactTargetRowSource::RawTargetArgmax,
            replay_first_key,
            &[10, 11],
        )
        .unwrap();
    replay_issuer.record_target_commit(11, 12).unwrap();
    let replay_second = issue(&mut replay_issuer, facts(), &drafts).unwrap();
    let mut replay = observe(&replay_second, &drafts);
    replay.key.receipt_nonce = replay_first_key.receipt_nonce;
    replay.key.proposal_epoch = replay_first_key.proposal_epoch;
    assert!(
        replay_second
            .admit_preverify(&mut replay_issuer, replay)
            .is_err()
    );

    for slot in 0..5 {
        let mut issuer = issuer();
        let receipt = issue(&mut issuer, facts(), &drafts).unwrap();
        let mut target_key = receipt.frame_key();
        let observed = observe(&receipt, &drafts);
        let permit = receipt.admit_preverify(&mut issuer, observed).unwrap();
        match slot {
            0 => target_key.session_nonce += 1,
            1 => target_key.issuer_instance_nonce += 1,
            2 => target_key.receipt_nonce += 1,
            3 => target_key.target_commit_epoch += 1,
            _ => target_key.proposal_epoch += 1,
        }
        assert!(
            permit
                .finish_raw_target(
                    &mut issuer,
                    ExactTargetRowSource::RawTargetArgmax,
                    target_key,
                    &[10, 11],
                )
                .is_err(),
            "target authority slot {slot}"
        );
    }
}

#[test]
fn every_nonce_epoch_context_prefix_and_model_identity_is_bound_at_both_seams() {
    let drafts = [10];
    for slot in 0..11 {
        let mut preverify_issuer = issuer();
        let receipt = issue(&mut preverify_issuer, facts(), &drafts).unwrap();
        let mut observed = observe(&receipt, &drafts);
        mutate_bound_key(&mut observed.key, slot);
        assert!(
            receipt
                .admit_preverify(&mut preverify_issuer, observed)
                .is_err(),
            "preverify key slot {slot}"
        );

        let mut target_issuer = issuer();
        let receipt = issue(&mut target_issuer, facts(), &drafts).unwrap();
        let mut target_key = receipt.frame_key();
        let observed = observe(&receipt, &drafts);
        let permit = receipt
            .admit_preverify(&mut target_issuer, observed)
            .unwrap();
        mutate_bound_key(&mut target_key, slot);
        assert!(
            permit
                .finish_raw_target(
                    &mut target_issuer,
                    ExactTargetRowSource::RawTargetArgmax,
                    target_key,
                    &[10, 11],
                )
                .is_err(),
            "target key slot {slot}"
        );
    }
}

#[test]
fn identical_constructor_inputs_cannot_cross_admit_between_issuers() {
    let drafts = [10];
    let mut left = ExactGreedyReceiptIssuer::new(7, 256, 11, 12).unwrap();
    let mut right = ExactGreedyReceiptIssuer::new(7, 256, 11, 12).unwrap();
    let left_receipt = issue(&mut left, facts(), &drafts).unwrap();
    let right_receipt = issue(&mut right, facts(), &drafts).unwrap();
    let left_key = left_receipt.frame_key();
    let right_key = right_receipt.frame_key();

    assert_ne!(left_key.issuer_instance_nonce, 0);
    assert_ne!(right_key.issuer_instance_nonce, 0);
    assert_ne!(
        left_key.issuer_instance_nonce,
        right_key.issuer_instance_nonce
    );
    assert_eq!(left_key.session_nonce, right_key.session_nonce);
    assert_eq!(left_key.receipt_nonce, right_key.receipt_nonce);
    assert_eq!(left_key.target_commit_epoch, right_key.target_commit_epoch);
    assert_eq!(left_key.proposal_epoch, right_key.proposal_epoch);

    let left_observed = observe(&left_receipt, &drafts);
    assert!(
        left_receipt
            .admit_preverify(&mut right, left_observed)
            .is_err()
    );
    let right_observed = observe(&right_receipt, &drafts);
    right_receipt
        .admit_preverify(&mut right, right_observed)
        .unwrap()
        .finish_raw_target(
            &mut right,
            ExactTargetRowSource::RawTargetArgmax,
            right_key,
            &[10, 11],
        )
        .unwrap();
}

#[test]
fn failed_preverify_and_target_finish_keep_issuance_closed() {
    let drafts = [10];
    let mut published = issuer();
    let receipt = issue(&mut published, facts(), &drafts).unwrap();
    let mut bad_observed = observe(&receipt, &drafts);
    bad_observed.key.facts.canonical_fed_prefix_digest = identity(6);
    assert!(
        receipt
            .admit_preverify(&mut published, bad_observed)
            .is_err()
    );
    assert!(issue(&mut published, facts(), &drafts).is_err());

    let mut verifying = issuer();
    let receipt = issue(&mut verifying, facts(), &drafts).unwrap();
    let key = receipt.frame_key();
    let observed = observe(&receipt, &drafts);
    let permit = receipt.admit_preverify(&mut verifying, observed).unwrap();
    assert!(
        permit
            .finish_raw_target(
                &mut verifying,
                ExactTargetRowSource::RawTargetArgmax,
                key,
                &[10],
            )
            .is_err()
    );
    assert!(issue(&mut verifying, facts(), &drafts).is_err());
}
