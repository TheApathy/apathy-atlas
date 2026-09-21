// SPDX-License-Identifier: AGPL-3.0-only

use std::cell::Cell;

use anyhow::Result;

use super::*;

fn identity(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn identities() -> ExactWideIdentities {
    ExactWideIdentities {
        target: identity(1),
        target_config: identity(2),
        target_state: identity(3),
        retriever: identity(4),
        retrieval_config: identity(7),
        raw_argmax: identity(5),
    }
}

fn history_and_drafts() -> (Vec<u32>, Vec<u32>) {
    let suffix = [7, 8, 9, 10];
    let drafts: Vec<u32> = (20..=51).collect();
    let mut history = suffix.to_vec();
    history.extend_from_slice(&drafts);
    history.extend_from_slice(&[60, 61, 62]);
    history.extend_from_slice(&suffix);
    (history, drafts)
}

fn frame<'a>(
    history: &'a [u32],
    drafts: &'a [u32],
    identities: ExactWideIdentities,
) -> ExactWideFrame<'a> {
    ExactWideFrame {
        source: ExactWideDraftSource::ExactSuffixHistory,
        topology: ExactWideTopology::FlatChain,
        position: history.len() - 1,
        prompt_len: 4,
        pending_token: *history.last().unwrap(),
        vocab_size: 1024,
        identities,
        match_start: 0,
        suffix_len: 4,
        canonical_tokens: history,
        drafts,
    }
}

fn issuer(ids: ExactWideIdentities, physical_k: usize) -> ExactWideReceiptIssuer {
    ExactWideReceiptIssuer::new(7, 512, physical_k, 11, 20, ids).unwrap()
}

fn publish<'a>(
    issuer: &mut ExactWideReceiptIssuer,
    frame: ExactWideFrame<'a>,
) -> Result<ExactWideFrameReceipt> {
    ExactWideFrameReceipt::issue_before_first_effect(issuer, frame, || Ok(())).map(|pair| pair.0)
}

fn finish_to_commit(issuer: &mut ExactWideReceiptIssuer, frame: ExactWideFrame<'_>, raw: &[u32]) {
    let receipt = publish(issuer, frame).unwrap();
    let permit = receipt.admit_preverify(issuer, frame).unwrap();
    let raw_receipt = permit.seal_raw_target(issuer, raw).unwrap();
    permit.finish_raw_target(issuer, raw_receipt).unwrap();
}

#[test]
fn exact_k17_and_k32_frames_commit_only_raw_target_greedy_output() {
    for draft_len in [MIN_WIDE_DRAFTS, MAX_WIDE_DRAFTS] {
        let ids = identities();
        let (history, all_drafts) = history_and_drafts();
        let drafts = &all_drafts[..draft_len];
        let frame = frame(&history, drafts, ids);
        let mut issuer = issuer(ids, draft_len + 1);
        let receipt = publish(&mut issuer, frame).unwrap();
        assert_eq!(receipt.frame_key().verify_k, draft_len + 1);

        let mut raw = drafts.to_vec();
        raw.push(99);
        raw[2] = 777;
        let permit = receipt.admit_preverify(&mut issuer, frame).unwrap();
        let raw_receipt = permit.seal_raw_target(&mut issuer, &raw).unwrap();
        let outcome = permit.finish_raw_target(&mut issuer, raw_receipt).unwrap();
        assert_eq!(outcome.num_accepted, 2);
        assert_eq!(outcome.bonus, 777);
        let commit = issuer.seal_target_state_commit(identity(9)).unwrap();
        issuer.record_target_commit(commit).unwrap();
    }
}

#[test]
fn conditional_suffix_or_tail_miss_rejects_before_effect_and_publication() {
    let ids = identities();
    let (history, all_drafts) = history_and_drafts();
    for mutation in 0..5 {
        let mut issuer = issuer(ids, 32);
        let mut mutated_history = history.clone();
        let mut mutated_drafts = all_drafts[..16].to_vec();
        if mutation == 2 {
            mutated_history[0] = 123;
        } else if mutation == 3 {
            mutated_drafts[7] = 124;
        }
        let mut candidate = frame(&mutated_history, &mutated_drafts, ids);
        match mutation {
            0 => candidate.match_start = history.len() - 3,
            1 => candidate.suffix_len = 0,
            4 => candidate.match_start = usize::MAX,
            _ => {}
        }
        let effects = Cell::new(0);
        assert!(
            ExactWideFrameReceipt::issue_before_first_effect(&mut issuer, candidate, || {
                effects.set(effects.get() + 1);
                Ok(())
            })
            .is_err(),
            "mutation {mutation}"
        );
        assert_eq!(effects.get(), 0, "mutation {mutation}");

        let valid = frame(&history, &all_drafts[..16], ids);
        let receipt = publish(&mut issuer, valid).unwrap();
        assert_eq!(receipt.frame_key().receipt_nonce, 1, "mutation {mutation}");
        assert_eq!(
            receipt.frame_key().proposal_epoch,
            21,
            "mutation {mutation}"
        );
    }
}

#[test]
fn width_physical_context_source_topology_and_vocabulary_fail_closed() {
    let ids = identities();
    let (history, drafts) = history_and_drafts();

    for width in [15, 32] {
        let mut issuer = issuer(ids, 32);
        let proposed = if width <= drafts.len() {
            drafts[..width].to_vec()
        } else {
            vec![20; width]
        };
        assert!(publish(&mut issuer, frame(&history, &proposed, ids)).is_err());
    }

    let mut too_narrow = issuer(ids, 17);
    assert!(publish(&mut too_narrow, frame(&history, &drafts[..17], ids)).is_err());
    assert!(ExactWideReceiptIssuer::new(7, 512, 16, 11, 20, ids).is_err());
    assert!(ExactWideReceiptIssuer::new(7, 512, 33, 11, 20, ids).is_err());

    let mut zero_ids = ids;
    zero_ids.retrieval_config = [0; 32];
    assert!(ExactWideReceiptIssuer::new(7, 512, 32, 11, 20, zero_ids).is_err());

    let mut context = ExactWideReceiptIssuer::new(7, history.len() + 15, 32, 11, 20, ids).unwrap();
    assert!(publish(&mut context, frame(&history, &drafts[..16], ids)).is_err());

    for mutation in 0..4 {
        let mut issuer = issuer(ids, 32);
        let mut candidate = frame(&history, &drafts[..16], ids);
        match mutation {
            0 => candidate.source = ExactWideDraftSource::NeuralOrSynthetic,
            1 => candidate.source = ExactWideDraftSource::Portfolio,
            2 => candidate.topology = ExactWideTopology::Tree,
            _ => candidate.vocab_size = 50,
        }
        assert!(
            publish(&mut issuer, candidate).is_err(),
            "mutation {mutation}"
        );
    }

    let mut overflow = issuer(ids, 32);
    let mut candidate = frame(&history, &drafts[..16], ids);
    candidate.position = usize::MAX;
    assert!(publish(&mut overflow, candidate).is_err());
}

#[test]
fn every_identity_and_frame_field_is_rechecked_before_target_effects() {
    let ids = identities();
    let (history, drafts) = history_and_drafts();
    for slot in 0..12 {
        let original = frame(&history, &drafts[..16], ids);
        let mut issuer = issuer(ids, 32);
        let receipt = publish(&mut issuer, original).unwrap();
        let mut observed = original;
        match slot {
            0 => observed.identities.target = identity(10),
            1 => observed.identities.target_config = identity(10),
            2 => observed.identities.target_state = identity(10),
            3 => observed.identities.retriever = identity(10),
            4 => observed.identities.retrieval_config = identity(10),
            5 => observed.identities.raw_argmax = identity(10),
            6 => observed.source = ExactWideDraftSource::NeuralOrSynthetic,
            7 => observed.topology = ExactWideTopology::Tree,
            8 => observed.drafts = &drafts[..17],
            9 => observed.match_start += 1,
            10 => observed.suffix_len -= 1,
            _ => observed.position += 1,
        }
        assert!(
            receipt.admit_preverify(&mut issuer, observed).is_err(),
            "slot {slot}"
        );
        assert!(publish(&mut issuer, original).is_err(), "slot {slot}");
    }
}

#[test]
fn canonical_prefix_identity_is_recomputed_from_all_token_bytes() {
    let ids = identities();
    let (history, drafts) = history_and_drafts();
    let original = frame(&history, &drafts[..16], ids);
    let mut issuer = issuer(ids, 32);
    let receipt = publish(&mut issuer, original).unwrap();
    let original_digest = receipt.frame_key().canonical_prefix_digest;
    assert_eq!(
        original_digest,
        [
            0xf0, 0xe3, 0x05, 0x66, 0xd6, 0xe0, 0x13, 0xeb, 0x8e, 0x82, 0x8e, 0x17, 0x50, 0x9e,
            0x5c, 0x60, 0x2a, 0x88, 0x98, 0xe2, 0x47, 0x0a, 0xad, 0x75, 0x20, 0xf0, 0xa2, 0xf9,
            0x51, 0x93, 0xfa, 0x0d,
        ]
    );

    let mut drifted_history = history.clone();
    drifted_history[25] = 999;
    let drifted = frame(&drifted_history, &drafts[..16], ids);
    assert_ne!(
        canonical_prefix_digest(&drifted_history).unwrap(),
        original_digest
    );
    assert!(receipt.admit_preverify(&mut issuer, drifted).is_err());
    assert!(publish(&mut issuer, original).is_err());
}

#[test]
fn only_exact_k_raw_target_rows_can_be_sealed() {
    let ids = identities();
    let (history, drafts) = history_and_drafts();
    for mutation in 0..2 {
        let frame = frame(&history, &drafts[..16], ids);
        let mut issuer = issuer(ids, 32);
        let receipt = publish(&mut issuer, frame).unwrap();
        let permit = receipt.admit_preverify(&mut issuer, frame).unwrap();
        let mut raw = drafts[..16].to_vec();
        raw.push(99);
        if mutation == 0 {
            raw.pop();
        } else {
            raw[4] = 1024;
        }
        assert!(permit.seal_raw_target(&mut issuer, &raw).is_err());
        assert!(publish(&mut issuer, frame).is_err());
    }
}

#[test]
fn one_shot_lifecycle_and_strict_state_commit_block_replay() {
    let ids = identities();
    let (history, drafts) = history_and_drafts();
    let current_frame = frame(&history, &drafts[..16], ids);
    let mut authority = issuer(ids, 32);
    let receipt = publish(&mut authority, current_frame).unwrap();
    assert!(publish(&mut authority, current_frame).is_err());
    let mut raw = drafts[..16].to_vec();
    raw.push(99);
    let permit = receipt
        .admit_preverify(&mut authority, current_frame)
        .unwrap();
    let raw_receipt = permit.seal_raw_target(&mut authority, &raw).unwrap();
    permit
        .finish_raw_target(&mut authority, raw_receipt)
        .unwrap();
    assert!(publish(&mut authority, current_frame).is_err());
    assert!(authority.seal_target_state_commit([0; 32]).is_err());
    assert!(
        authority
            .seal_target_state_commit(ids.target_state)
            .is_err()
    );
    let commit = authority.seal_target_state_commit(identity(9)).unwrap();
    authority.record_target_commit(commit).unwrap();
    let mut next_ids = ids;
    next_ids.target_state = identity(9);
    let next = publish(&mut authority, frame(&history, &drafts[..16], next_ids)).unwrap();
    assert_eq!(next.frame_key().target_commit_epoch, 12);
    assert_eq!(next.frame_key().proposal_epoch, 22);
    assert_eq!(next.frame_key().receipt_nonce, 2);

    let mut failed_effect_issuer = issuer(ids, 32);
    let effects = Cell::new(0);
    assert!(
        ExactWideFrameReceipt::issue_before_first_effect(
            &mut failed_effect_issuer,
            current_frame,
            || {
                effects.set(effects.get() + 1);
                Err::<(), _>(anyhow::anyhow!("injected external effect failure"))
            },
        )
        .is_err()
    );
    assert_eq!(effects.get(), 1);
    assert!(publish(&mut failed_effect_issuer, current_frame).is_err());
}

#[test]
fn independently_constructed_issuers_cannot_cross_admit() {
    let ids = identities();
    let (history, drafts) = history_and_drafts();
    let frame = frame(&history, &drafts[..16], ids);
    let mut left = issuer(ids, 32);
    let mut right = issuer(ids, 32);
    let receipt = publish(&mut left, frame).unwrap();
    let key = receipt.frame_key();
    assert_ne!(key.issuer_instance_nonce, right.instance_nonce_for_test());
    assert!(receipt.admit_preverify(&mut right, frame).is_err());

    let own = publish(&mut right, frame).unwrap();
    assert!(own.admit_preverify(&mut right, frame).is_ok());
}

#[test]
fn sealed_raw_target_receipts_reject_forgery_cross_frame_and_replay() {
    let ids = identities();
    let (history, drafts) = history_and_drafts();
    let current_frame = frame(&history, &drafts[..16], ids);
    let mut raw = drafts[..16].to_vec();
    raw.push(99);

    let mut forged_issuer = issuer(ids, 32);
    let permit = publish(&mut forged_issuer, current_frame)
        .unwrap()
        .admit_preverify(&mut forged_issuer, current_frame)
        .unwrap();
    let sealed = permit.seal_raw_target(&mut forged_issuer, &raw).unwrap();
    let forged = sealed.duplicate_for_test().forge_token_for_test();
    assert!(
        permit
            .finish_raw_target(&mut forged_issuer, forged)
            .is_err()
    );

    let mut left = issuer(ids, 32);
    let mut right = issuer(ids, 32);
    let left_permit = publish(&mut left, current_frame)
        .unwrap()
        .admit_preverify(&mut left, current_frame)
        .unwrap();
    let right_permit = publish(&mut right, current_frame)
        .unwrap()
        .admit_preverify(&mut right, current_frame)
        .unwrap();
    let cross = left_permit.seal_raw_target(&mut left, &raw).unwrap();
    assert!(right_permit.finish_raw_target(&mut right, cross).is_err());

    let mut replay_issuer = issuer(ids, 32);
    let permit = publish(&mut replay_issuer, current_frame)
        .unwrap()
        .admit_preverify(&mut replay_issuer, current_frame)
        .unwrap();
    let sealed = permit.seal_raw_target(&mut replay_issuer, &raw).unwrap();
    let replay = sealed.duplicate_for_test();
    permit
        .finish_raw_target(&mut replay_issuer, sealed)
        .unwrap();
    let commit = replay_issuer.seal_target_state_commit(identity(9)).unwrap();
    replay_issuer.record_target_commit(commit).unwrap();
    let mut next_ids = ids;
    next_ids.target_state = identity(9);
    let next_frame = frame(&history, &drafts[..16], next_ids);
    let next_permit = publish(&mut replay_issuer, next_frame)
        .unwrap()
        .admit_preverify(&mut replay_issuer, next_frame)
        .unwrap();
    assert!(
        next_permit
            .finish_raw_target(&mut replay_issuer, replay)
            .is_err()
    );
}

#[test]
fn sealed_state_receipts_reject_forgery_cross_frame_and_replay() {
    let ids = identities();
    let (history, drafts) = history_and_drafts();
    let frame = frame(&history, &drafts[..16], ids);
    let mut raw = drafts[..16].to_vec();
    raw.push(99);

    let mut authority = issuer(ids, 32);
    finish_to_commit(&mut authority, frame, &raw);
    let sealed = authority.seal_target_state_commit(identity(9)).unwrap();
    let replay = sealed.duplicate_for_test();
    let forged = sealed.duplicate_for_test().forge_next_state_for_test();
    assert!(authority.record_target_commit(forged).is_err());
    authority.record_target_commit(sealed).unwrap();
    assert!(authority.record_target_commit(replay).is_err());

    let mut left = issuer(ids, 32);
    let mut right = issuer(ids, 32);
    finish_to_commit(&mut left, frame, &raw);
    finish_to_commit(&mut right, frame, &raw);
    let left_commit = left.seal_target_state_commit(identity(9)).unwrap();
    let right_commit = right.seal_target_state_commit(identity(10)).unwrap();
    assert!(right.record_target_commit(left_commit).is_err());
    right.record_target_commit(right_commit).unwrap();
}

#[test]
fn receipt_and_permits_are_linear_and_module_stays_unregistered() {
    let source = include_str!("exact_wide_retrieval_admission.rs");
    let issuer = include_str!("exact_wide_retrieval_admission/issuer.rs");
    let scheduler = include_str!("mod.rs");
    assert!(!source.contains("impl Clone for ExactWideFrameReceipt"));
    assert!(!source.contains("impl Clone for ExactWideTargetPermit"));
    assert!(!source.contains("ExactWideTargetObservation"));
    assert!(!source.contains("pub canonical_prefix_digest"));
    assert_eq!(
        source
            .matches("canonical_prefix_digest(frame.canonical_tokens)?")
            .count(),
        2
    );
    assert!(!issuer.contains("impl Clone for ExactWideRawTargetReceipt"));
    assert!(!issuer.contains("impl Clone for ExactWideTargetStateReceipt"));
    for required in [
        "atlas/exact-wide/canonical-prefix/v1",
        "atlas/exact-wide/raw-target/v1",
        "IssuerPhase::RawTargetSealed",
        "IssuerPhase::CommitSealed",
        "forged, replayed, or cross-frame raw target receipt",
        "forged, replayed, or cross-frame target state receipt",
    ] {
        assert!(issuer.contains(required), "missing sealed term {required}");
    }
    assert!(!scheduler.contains("mod exact_wide_retrieval_admission;"));
}
