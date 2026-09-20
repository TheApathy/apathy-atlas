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

fn issue_with(
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

fn issue(drafts: &[u32]) -> (ExactGreedyReceiptIssuer, ExactGreedyFrameReceipt) {
    let mut issuer = issuer();
    let receipt = issue_with(&mut issuer, facts(), drafts).unwrap();
    (issuer, receipt)
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

fn verify(drafts: &[u32], raw: &[u32]) -> Result<ExactGreedyVerifyOutcome> {
    let (mut issuer, receipt) = issue(drafts);
    let key = receipt.frame_key();
    let observed = observe(&receipt, drafts);
    receipt
        .admit_preverify(&mut issuer, observed)?
        .finish_raw_target(&mut issuer, ExactTargetRowSource::RawTargetArgmax, key, raw)
}

#[test]
fn exact_receipt_returns_first_mismatch_or_full_match_bonus() {
    let drafts = [10, 11, 12];
    let mismatch = verify(&drafts, &[10, 90, 91, 92]).unwrap();
    assert_eq!(mismatch.num_accepted, 1);
    assert_eq!(mismatch.bonus, 90);

    let full = verify(&drafts, &[10, 11, 12, 99]).unwrap();
    assert_eq!(full.num_accepted, 3);
    assert_eq!(full.bonus, 99);
}

#[test]
fn authority_sources_and_target_row_transforms_fail_closed() {
    let drafts = [10];
    assert!(
        ExactGreedyFrameReceipt::issue(
            &mut issuer(),
            TargetTokenAuthority::ServingPolicy,
            ExactDraftSource::GhostModelOnly,
            ExactServingTransforms::none(),
            ExactGreedyPolicyKey::canonical(),
            facts(),
            &drafts,
        )
        .is_err()
    );

    for source in [
        ExactDraftSource::Retrieval,
        ExactDraftSource::AsyncCollection,
        ExactDraftSource::EarlyExit,
        ExactDraftSource::CfgJumpForward,
        ExactDraftSource::Portfolio,
        ExactDraftSource::Tree,
        ExactDraftSource::Ngram,
        ExactDraftSource::PolicySplice,
    ] {
        assert!(
            ExactGreedyFrameReceipt::issue(
                &mut issuer(),
                TargetTokenAuthority::ExactRawArgmax,
                source,
                ExactServingTransforms::none(),
                ExactGreedyPolicyKey::canonical(),
                facts(),
                &drafts,
            )
            .is_err(),
            "source {source:?}"
        );
    }

    for source in [
        ExactTargetRowSource::ServingPolicy,
        ExactTargetRowSource::GrammarMasked,
        ExactTargetRowSource::Resampled,
        ExactTargetRowSource::TypicalOrRelaxed,
    ] {
        let (mut issuer, receipt) = issue(&drafts);
        let key = receipt.frame_key();
        let observed = observe(&receipt, &drafts);
        let permit = receipt.admit_preverify(&mut issuer, observed).unwrap();
        assert!(
            permit
                .finish_raw_target(&mut issuer, source, key, &[10, 11])
                .is_err(),
            "target source {source:?}"
        );
    }
}

#[test]
fn every_known_serving_transform_is_rejected() {
    let drafts = [10];
    for transform in [
        ExactServingTransforms::SAMPLER_FILTER,
        ExactServingTransforms::HISTORY_PENALTY,
        ExactServingTransforms::LOGIT_BIAS,
        ExactServingTransforms::THINKING_POLICY,
        ExactServingTransforms::CONTENT_POLICY,
        ExactServingTransforms::TOOL_POLICY,
        ExactServingTransforms::GRAMMAR,
        ExactServingTransforms::ADAPTIVE_SAMPLING,
        ExactServingTransforms::LOGPROBS,
        ExactServingTransforms::TARGET_MASK_OR_SUPPRESSION,
        ExactServingTransforms::TYPICAL_OR_RELAXED_ACCEPT,
        ExactServingTransforms::WATCHDOG_OR_ROLLBACK,
    ] {
        assert!(
            ExactGreedyFrameReceipt::issue(
                &mut issuer(),
                TargetTokenAuthority::ExactRawArgmax,
                ExactDraftSource::GhostModelOnly,
                transform,
                ExactGreedyPolicyKey::canonical(),
                facts(),
                &drafts,
            )
            .is_err(),
            "transform {transform:?}"
        );
    }
}

#[test]
fn every_sampler_field_uses_canonical_scalar_bits() {
    let canonical = ExactGreedyPolicyKey::canonical();
    let mut mutations = Vec::new();
    macro_rules! mutate {
        ($field:ident, $value:expr) => {{
            let mut policy = canonical;
            policy.$field = $value;
            mutations.push(policy);
        }};
    }
    mutate!(temperature_bits, (-0.0f32).to_bits());
    mutate!(top_k, 1);
    mutate!(top_p_bits, 0.9f32.to_bits());
    mutate!(top_n_sigma_bits, 1.0f32.to_bits());
    mutate!(min_p_bits, 0.1f32.to_bits());
    mutate!(repetition_penalty_bits, 1.1f32.to_bits());
    mutate!(repetition_penalty_window, 64);
    mutate!(presence_penalty_bits, 0.1f32.to_bits());
    mutate!(frequency_penalty_bits, 0.1f32.to_bits());
    mutate!(lz_penalty_bits, 1.0f32.to_bits());
    mutate!(dry_multiplier_bits, 0.8f32.to_bits());
    mutate!(dry_base_bits, 2.0f32.to_bits());
    mutate!(dry_allowed_length, 3);
    mutate!(logit_bias_len, 1);
    mutate!(dry_sequence_breakers_len, 1);
    mutate!(seed, Some(1));
    for policy in mutations {
        assert!(policy.validate().is_err(), "policy {policy:?}");
    }
}

#[test]
fn frame_geometry_prefix_digest_and_all_identities_are_mandatory() {
    let canonical = facts();
    let mut bad = canonical;
    bad.position += 1;
    assert!(bad.validate(256).is_err());
    bad = canonical;
    bad.prompt_len = bad.fed_len + 1;
    assert!(bad.validate(256).is_err());
    bad = canonical;
    bad.logical_len += 1;
    assert!(bad.validate(256).is_err());
    bad = canonical;
    bad.fed_len = usize::MAX;
    bad.position = usize::MAX;
    bad.logical_len = 0;
    assert!(bad.validate(usize::MAX).is_err());
    bad = canonical;
    bad.vocab_size = 0;
    assert!(bad.validate(256).is_err());
    bad = canonical;
    bad.pending_token = bad.vocab_size;
    assert!(bad.validate(256).is_err());
    bad = canonical;
    bad.canonical_fed_prefix_digest = [0; 32];
    assert!(bad.validate(256).is_err());

    for slot in 0..4 {
        bad = canonical;
        match slot {
            0 => bad.target_identity = [0; 32],
            1 => bad.proposer_identity = [0; 32],
            2 => bad.config_identity = [0; 32],
            _ => bad.raw_argmax_identity = [0; 32],
        }
        assert!(bad.validate(256).is_err(), "identity slot {slot}");
    }

    assert!(ExactGreedyReceiptIssuer::new(0, 256, 11, 12).is_err());
    assert!(ExactGreedyReceiptIssuer::new(7, 0, 11, 12).is_err());
    assert!(ExactGreedyReceiptIssuer::new(7, 256, 0, 12).is_err());
    assert!(ExactGreedyReceiptIssuer::new(7, 256, 11, u64::MAX).is_err());
}

#[test]
fn empty_overwide_mutated_and_replayed_frames_are_rejected() {
    assert!(
        ExactGreedyFrameReceipt::issue(
            &mut issuer(),
            TargetTokenAuthority::ExactRawArgmax,
            ExactDraftSource::GhostModelOnly,
            ExactServingTransforms::none(),
            ExactGreedyPolicyKey::canonical(),
            facts(),
            &[],
        )
        .is_err()
    );
    assert!(
        ExactGreedyFrameReceipt::issue(
            &mut issuer(),
            TargetTokenAuthority::ExactRawArgmax,
            ExactDraftSource::GhostModelOnly,
            ExactServingTransforms::none(),
            ExactGreedyPolicyKey::canonical(),
            facts(),
            &[256],
        )
        .is_err()
    );
    assert!(
        ExactGreedyFrameReceipt::issue(
            &mut issuer(),
            TargetTokenAuthority::ExactRawArgmax,
            ExactDraftSource::GhostModelOnly,
            ExactServingTransforms::none(),
            ExactGreedyPolicyKey::canonical(),
            facts(),
            &[1; MAX_EXACT_DRAFTS + 1],
        )
        .is_err()
    );

    let drafts = [10, 11];
    let (mut issuer, receipt) = issue(&drafts);
    let observed = observe(&receipt, &[10, 12]);
    assert!(receipt.admit_preverify(&mut issuer, observed).is_err());
    let (mut issuer, receipt) = issue(&drafts);
    let key = receipt.frame_key();
    let observed = observe(&receipt, &drafts);
    let permit = receipt.admit_preverify(&mut issuer, observed).unwrap();
    assert!(
        permit
            .finish_raw_target(
                &mut issuer,
                ExactTargetRowSource::RawTargetArgmax,
                key,
                &[10, 11],
            )
            .is_err()
    );
    let (mut issuer, receipt) = issue(&drafts);
    let key = receipt.frame_key();
    let observed = observe(&receipt, &drafts);
    let permit = receipt.admit_preverify(&mut issuer, observed).unwrap();
    assert!(
        permit
            .finish_raw_target(
                &mut issuer,
                ExactTargetRowSource::RawTargetArgmax,
                key,
                &[10, 11, 256],
            )
            .is_err()
    );
    let (mut issuer, receipt) = issue(&drafts);
    let mut replay = observe(&receipt, &drafts);
    replay.key.proposal_epoch += 1;
    assert!(receipt.admit_preverify(&mut issuer, replay).is_err());
}

#[test]
fn post_publication_authority_policy_source_and_transform_drift_is_rejected() {
    let drafts = [10];

    let (mut issuer, receipt) = issue(&drafts);
    let mut observed = observe(&receipt, &drafts);
    observed.authority = TargetTokenAuthority::ServingPolicy;
    assert!(receipt.admit_preverify(&mut issuer, observed).is_err());

    let (mut issuer, receipt) = issue(&drafts);
    observed = observe(&receipt, &drafts);
    observed.source = ExactDraftSource::Retrieval;
    assert!(receipt.admit_preverify(&mut issuer, observed).is_err());

    let (mut issuer, receipt) = issue(&drafts);
    observed = observe(&receipt, &drafts);
    observed.transforms = ExactServingTransforms::GRAMMAR;
    assert!(receipt.admit_preverify(&mut issuer, observed).is_err());

    let (mut issuer, receipt) = issue(&drafts);
    observed = observe(&receipt, &drafts);
    observed.policy.top_p_bits = 0.9f32.to_bits();
    assert!(receipt.admit_preverify(&mut issuer, observed).is_err());

    let (mut issuer, mut receipt) = issue(&drafts);
    receipt.version = RECEIPT_VERSION + 1;
    let observed = observe(&receipt, &drafts);
    assert!(receipt.admit_preverify(&mut issuer, observed).is_err());

    let (mut issuer, receipt) = issue(&drafts);
    let mut target_key = receipt.frame_key();
    let observed = observe(&receipt, &drafts);
    let permit = receipt.admit_preverify(&mut issuer, observed).unwrap();
    target_key.facts.raw_argmax_identity = identity(9);
    assert!(
        permit
            .finish_raw_target(
                &mut issuer,
                ExactTargetRowSource::RawTargetArgmax,
                target_key,
                &[10, 11],
            )
            .is_err()
    );
}

#[test]
fn receipt_is_fixed_size_and_copies_the_published_drafts() {
    let mut drafts = [1, 2, 3];
    let (mut issuer, receipt) = issue(&drafts);
    drafts[1] = 9;
    let observed = observe(&receipt, &drafts);
    assert!(receipt.admit_preverify(&mut issuer, observed).is_err());
    assert!(std::mem::size_of::<ExactGreedyFrameReceipt>() < 512);
}
