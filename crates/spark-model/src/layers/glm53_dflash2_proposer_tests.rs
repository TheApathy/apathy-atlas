// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn policy() -> Glm53Dflash2Policy {
    Glm53Dflash2Policy::exact()
}

fn admit(anchor: u64) -> Glm53Dflash2Admission {
    policy().admit(Glm53Dflash2Request::greedy(anchor)).unwrap()
}

fn capture(state: &mut Glm53Dflash2SequenceState, base: u64, rows: u64) -> CaptureAppendPlan {
    let plan = state.begin_capture(base, rows).unwrap();
    state
        .commit_capture(plan.transaction(), plan.end_position())
        .unwrap();
    plan
}

fn propose(state: &mut Glm53Dflash2SequenceState, anchor: u64) -> ProposalPlan {
    let plan = state.begin_proposal(&admit(anchor)).unwrap();
    state
        .commit_proposal(plan.transaction(), &[0; 8], 0)
        .unwrap();
    plan
}

#[test]
fn exact_policy_rejects_before_effects_and_never_routes_k4() {
    assert_eq!(policy().capture_layers(), [5, 14, 24, 33, 42]);
    assert_eq!(GLM53_DFLASH2_BLOCK_TOKENS, 8);
    assert_eq!(GLM53_DFLASH2_RETURNED_DRAFTS, 7);
    assert_eq!(GLM53_DFLASH2_WINDOW, 2_048);
    assert_eq!(GLM53_DFLASH2_SEQUENCE_BYTES, 154_482_432);
    assert_eq!(
        verify_route_for_returned_drafts(7).unwrap(),
        Glm53Dflash2VerifyRoute::DflashGamma8
    );
    for invalid_len in [0, 3, 4, 6, 8] {
        assert!(verify_route_for_returned_drafts(invalid_len).is_err());
    }

    let mut effects = 0;
    for temperature in [0.01, -0.01, f32::NAN, f32::INFINITY] {
        let mut request = Glm53Dflash2Request::greedy(0);
        request.temperature = temperature;
        if policy().admit(request).is_ok() {
            effects += 1;
        }
    }
    assert_eq!(effects, 0);

    for mutate in 0..3 {
        let mut request = Glm53Dflash2Request::greedy(0);
        match mutate {
            0 => request.block_tokens = 7,
            1 => request.returned_drafts = 6,
            2 => request.window = 2_047,
            _ => unreachable!(),
        }
        assert!(policy().admit(request).is_err());
    }
    assert_eq!(admit(0).route(), Glm53Dflash2VerifyRoute::DflashGamma8);
}

#[test]
fn positions_distinguish_target_limit_from_full_gamma_start() {
    policy()
        .admit_target_position(GLM53_MAX_POSITION_INCLUSIVE)
        .unwrap();
    assert!(
        policy()
            .admit_target_position(GLM53_MAX_POSITION_EXCLUSIVE)
            .is_err()
    );
    admit(GLM53_MAX_POSITION_INCLUSIVE - 7);
    assert!(
        policy()
            .admit(Glm53Dflash2Request::greedy(
                GLM53_MAX_POSITION_INCLUSIVE - 6
            ))
            .is_err()
    );
    assert!(
        policy()
            .admit(Glm53Dflash2Request::greedy(u64::MAX))
            .is_err()
    );
}

#[test]
fn long_prompt_keeps_only_latest_capture_and_attention_tails() {
    let mut state = Glm53Dflash2SequenceState::new(1, &admit(4_097)).unwrap();
    let append = capture(&mut state, 0, 4_097);
    assert_eq!(append.source_skip_rows(), 2_049);
    assert_eq!(append.write_rows(), 2_048);
    assert_eq!(append.segments()[0].destination_row(), 0);
    assert_eq!(append.segments()[0].rows(), 2_048);
    assert_eq!(append.segments()[1].rows(), 0);
    assert_eq!(append.base_position(), 0);
    let capture_cursor = state.capture_cursor().unwrap();
    assert_eq!(capture_cursor.absolute_end(), 4_097);
    assert_eq!(capture_cursor.head(), 0);
    assert_eq!(capture_cursor.retained(), 2_048);

    let proposal = propose(&mut state, 4_097);
    assert_eq!(proposal.anchor_position(), 4_097);
    assert_eq!(proposal.logical_new_context_rows(), 4_097);
    assert_eq!(proposal.attention_new_context_rows(), 2_047);
    assert_eq!(proposal.attention_source_skip_rows(), 0);
    assert_eq!(proposal.absolute_context_end(), 4_097);
    assert_eq!(proposal.capture_oldest_position(), 2_049);
    assert_eq!(proposal.capture_retained_rows(), 2_048);
    assert_eq!(proposal.target_source_start_position(), 2_050);
    assert_eq!(proposal.capture_source_offset_rows(), 1);
    assert_eq!(proposal.capture_source_segments()[0].source_row(), 1);
    assert_eq!(proposal.capture_source_segments()[0].rows(), 2_047);
    assert_eq!(proposal.capture_source_segments()[0].source_end(), 2_048);
    assert_eq!(proposal.capture_source_segments()[1].rows(), 0);
    assert!(
        proposal.capture_source_offset_rows() + proposal.attention_new_context_rows()
            <= proposal.capture_retained_rows()
    );
    assert!(
        proposal
            .capture_source_segments()
            .iter()
            .all(|segment| segment.source_end() <= GLM53_DFLASH2_WINDOW)
    );
    assert_eq!(proposal.local_context_start_position(), 2_050);
    assert_eq!(proposal.target_tail_rows(), 2_047);
    assert_eq!(proposal.kept_past_rows(), 0);
    assert_eq!(proposal.past_drop_rows(), 0);
    assert_eq!(proposal.local_context_rows(), 2_047);
    assert_eq!(proposal.noise_rows(), 8);
    assert_eq!(proposal.returned_drafts(), 7);
    assert_eq!(proposal.route(), Glm53Dflash2VerifyRoute::DflashGamma8);
    for cursor in state.layer_cache_cursors().unwrap() {
        assert_eq!(cursor.absolute_end(), 4_097);
        assert_eq!(cursor.retained(), 2_047);
    }
    assert_eq!(state.outstanding_drafts().unwrap(), Some(7));
}

fn verify_and_capture_case(accepted: u8) {
    let mut state = Glm53Dflash2SequenceState::new(10 + u64::from(accepted), &admit(32)).unwrap();
    capture(&mut state, 0, 32);
    let proposal = propose(&mut state, 32);
    let stale = TransactionId::forged_for_test(
        proposal.transaction().generation(),
        proposal.transaction().nonce() + 1,
    );
    assert!(state.after_verify(stale, accepted).is_err());
    state
        .after_verify(proposal.transaction(), accepted)
        .unwrap();
    assert_eq!(state.expected_capture().unwrap(), Some((32, accepted + 1)));
    let retry = state.begin_capture(32, u64::from(accepted + 1)).unwrap();
    let stale = TransactionId::forged_for_test(
        retry.transaction().generation(),
        retry.transaction().nonce() + 1,
    );
    assert!(state.commit_capture(stale, retry.end_position()).is_err());
    state.rollback_capture(retry.transaction()).unwrap();
    assert_eq!(state.capture_cursor().unwrap().absolute_end(), 32);
    assert_eq!(state.expected_capture().unwrap(), Some((32, accepted + 1)));
    capture(&mut state, 32, u64::from(accepted + 1));
    assert_eq!(
        state.capture_cursor().unwrap().absolute_end(),
        33 + u64::from(accepted)
    );
    assert_eq!(state.expected_capture().unwrap(), None);
}

#[test]
fn accept_zero_partial_and_full_require_exact_accepted_plus_one_capture() {
    verify_and_capture_case(0);
    verify_and_capture_case(3);
    verify_and_capture_case(7);
}

#[test]
fn proposal_status_failure_is_invisible_until_explicit_rollback() {
    let mut state = Glm53Dflash2SequenceState::new(2, &admit(64)).unwrap();
    capture(&mut state, 0, 64);
    let before = state.layer_cache_cursors().unwrap();
    let failed = state.begin_proposal(&admit(64)).unwrap();
    let mut topk = [0; 8];
    topk[6] = 1;
    assert!(
        state
            .commit_proposal(failed.transaction(), &topk, 0)
            .is_err()
    );
    assert_eq!(state.layer_cache_cursors().unwrap(), before);
    let stale = TransactionId::forged_for_test(
        failed.transaction().generation(),
        failed.transaction().nonce() + 1,
    );
    assert!(state.rollback_proposal(stale).is_err());
    state.rollback_proposal(failed.transaction()).unwrap();
    assert_eq!(state.layer_cache_cursors().unwrap(), before);

    let failed_selector = state.begin_proposal(&admit(64)).unwrap();
    assert!(
        state
            .commit_proposal(failed_selector.transaction(), &[0; 8], 9)
            .is_err()
    );
    state
        .rollback_proposal(failed_selector.transaction())
        .unwrap();
    propose(&mut state, 64);
}

#[test]
fn capture_ring_wrap_plan_is_disjoint_and_chronological() {
    let mut state = Glm53Dflash2SequenceState::new(20, &admit(2_056)).unwrap();
    capture(&mut state, 0, 2_040);
    let wrapped = capture(&mut state, 2_040, 16);
    assert_eq!(wrapped.source_skip_rows(), 0);
    assert_eq!(wrapped.segments()[0].destination_row(), 2_040);
    assert_eq!(wrapped.segments()[0].rows(), 8);
    assert_eq!(wrapped.segments()[1].destination_row(), 0);
    assert_eq!(wrapped.segments()[1].rows(), 8);
    let cursor = state.capture_cursor().unwrap();
    assert_eq!(cursor.absolute_end(), 2_056);
    assert_eq!(cursor.head(), 8);
    assert_eq!(cursor.retained(), 2_048);
}

#[test]
fn advanced_cache_reads_wrapped_capture_tail_in_two_bounded_segments() {
    let mut state = Glm53Dflash2SequenceState::new(21, &admit(2_052)).unwrap();
    capture(&mut state, 0, 2_044);
    let first = propose(&mut state, 2_044);
    state.after_verify(first.transaction(), 7).unwrap();
    capture(&mut state, 2_044, 8);
    let proposal = state.begin_proposal(&admit(2_052)).unwrap();
    assert_eq!(proposal.logical_new_context_rows(), 8);
    assert_eq!(proposal.attention_new_context_rows(), 8);
    assert_eq!(proposal.attention_source_skip_rows(), 0);
    assert_eq!(proposal.absolute_context_end(), 2_052);
    assert_eq!(proposal.capture_oldest_position(), 4);
    assert_eq!(proposal.capture_retained_rows(), 2_048);
    assert_eq!(proposal.target_source_start_position(), 2_044);
    assert_eq!(proposal.capture_source_offset_rows(), 2_040);
    let segments = proposal.capture_source_segments();
    assert_eq!((segments[0].source_row(), segments[0].rows()), (2_044, 4));
    assert_eq!((segments[1].source_row(), segments[1].rows()), (0, 4));
    assert_eq!(segments[0].source_end(), 2_048);
    assert_eq!(segments[1].source_end(), 4);
    assert_eq!(segments[0].rows() + segments[1].rows(), 8);
    assert!(proposal.capture_source_offset_rows() + proposal.attention_new_context_rows() <= 2_048);
    assert!(segments.iter().all(|segment| segment.source_end() <= 2_048));
    assert_eq!(proposal.kept_past_rows(), 2_039);
    assert_eq!(proposal.past_drop_rows(), 5);
    assert_eq!(proposal.local_context_rows(), 2_047);
    assert_eq!(proposal.local_context_start_position(), 5);
}

#[test]
fn boundary_overflow_and_exact_verified_rows_fail_closed() {
    let boundary_anchor = GLM53_MAX_POSITION_INCLUSIVE - 7;
    let mut state = Glm53Dflash2SequenceState::new(3, &admit(boundary_anchor)).unwrap();
    capture(&mut state, 0, GLM53_MAX_POSITION_INCLUSIVE - 7);
    let anchor = state.capture_cursor().unwrap().absolute_end();
    let proposal = propose(&mut state, anchor);
    state.after_verify(proposal.transaction(), 7).unwrap();
    capture(&mut state, anchor, 8);
    assert_eq!(
        state.capture_cursor().unwrap().absolute_end(),
        GLM53_MAX_POSITION_EXCLUSIVE
    );
    assert!(
        state
            .begin_capture(GLM53_MAX_POSITION_EXCLUSIVE, 1)
            .is_err()
    );

    let mut overflow = Glm53Dflash2SequenceState::new(4, &admit(0)).unwrap();
    assert!(overflow.begin_capture(u64::MAX, 2).is_err());

    let mut exact = Glm53Dflash2SequenceState::new(5, &admit(8)).unwrap();
    capture(&mut exact, 0, 8);
    let proposal = propose(&mut exact, 8);
    exact.after_verify(proposal.transaction(), 3).unwrap();
    assert!(exact.begin_capture(8, 3).is_err());
    assert!(exact.begin_capture(9, 4).is_err());
    capture(&mut exact, 8, 4);
}

#[test]
fn release_aborts_transactions_and_generation_blocks_first_request_stale_ids() {
    let mut state = Glm53Dflash2SequenceState::new(100, &admit(16)).unwrap();
    let old = state.begin_capture(0, 16).unwrap().transaction();
    let receipt = state.release().unwrap();
    assert_eq!(receipt.lease_id(), 100);
    assert_eq!(receipt.generation(), 1);
    assert_eq!(receipt.allocation_bytes(), 154_482_432);
    assert!(state.release().is_err());
    assert!(state.capture_cursor().is_err());
    assert!(state.reset_for_reuse(100, &admit(16)).is_err());

    state.reset_for_reuse(101, &admit(16)).unwrap();
    let fresh = state.begin_capture(0, 16).unwrap();
    assert_eq!(fresh.transaction().generation(), 2);
    assert_eq!(fresh.transaction().nonce(), 1);
    assert!(state.commit_capture(old, 16).is_err());
    state
        .commit_capture(fresh.transaction(), fresh.end_position())
        .unwrap();
    propose(&mut state, 16);
}
