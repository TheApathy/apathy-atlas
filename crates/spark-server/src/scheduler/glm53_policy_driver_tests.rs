// SPDX-License-Identifier: AGPL-3.0-only

//! Actual mtp_step RED gates. Run with diagnostic sampling/seam overrides
//! absent. The model boundary owns the real binding owner/transaction; only
//! logits transport and model-state mutation are synthetic, never the policy.

#[path = "glm53_ordinary_greedy_fixture.rs"]
mod boundary;
#[path = "glm53_policy_adapter_fixture.rs"]
mod fixture;
#[path = "glm53_policy_grammar_bootstrap_tests.rs"]
mod grammar_bootstrap_tests;
#[path = "glm53_policy_driver_fixture.rs"]
mod model_boundary;

use crate::adaptive_sampler::GenerationZone;
use crate::scheduler::logit_processors::LogitsContext;
use crate::scheduler::{ActiveSeq, SsmDecodeRing};
use fixture::{fingerprint, grammar, sequence};
use model_boundary::{DriverModel, Fault};

fn context() -> LogitsContext {
    LogitsContext {
        think_end_token: None,
        think_start_token: None,
        tool_call_start_token: None,
        tool_call_end_token: None,
    }
}

fn scores(token: u32) -> [f32; 5] {
    let mut scores = [-8.0; 5];
    scores[token as usize] = 20.0;
    scores
}

fn run(
    model: &DriverModel,
    active: &mut [ActiveSeq],
    drafts: usize,
    raw: bool,
    adaptive: bool,
    fence: Option<u32>,
    ctx: &LogitsContext,
) {
    crate::scheduler::mtp_step::step_mtp(model, active, drafts, ctx, raw, adaptive, fence);
}

fn one(model: &DriverModel, mut seq: ActiveSeq) -> Vec<ActiveSeq> {
    seq.sink = model.stream();
    vec![seq]
}

fn synchronized(model: &DriverModel, seq: &ActiveSeq) {
    assert_eq!(seq.seq.tokens, model.prefix());
    assert_eq!(seq.seq.seq_len, model.position());
    assert_eq!(seq.seq.tokens.len(), seq.seq.seq_len);
    assert_eq!(seq.seq.kv_valid_tokens, seq.seq.seq_len);
    assert!(seq.terminal_error.is_none());
}

#[test]
fn bootstrap_and_steady_enter_owned_policy_for_every_width_two_through_eight() {
    for rows in 2..=8 {
        for bootstrap in [false, true] {
            let mut picks = vec![1; rows];
            picks[rows - 1] = 2;
            let draft = picks[..rows - 1].to_vec();
            let model = DriverModel::new(
                &picks.iter().copied().map(scores).collect::<Vec<_>>(),
                draft.clone(),
            );
            let mut seq = sequence();
            seq.remaining = rows;
            if !bootstrap {
                seq.pending_drafts = draft.clone();
            }
            let mut active = one(&model, seq);
            run(&model, &mut active, rows - 1, true, false, None, &context());
            let mut inputs = vec![0];
            inputs.extend_from_slice(&draft);
            assert_eq!(
                model.inputs(),
                vec![inputs],
                "rows={rows}, bootstrap={bootstrap}"
            );
            assert_eq!(model.count("verify"), 1);
            assert_eq!(model.count("commit"), 1);
            assert_eq!(model.count("decode"), 0);
            assert_eq!(model.count("propose"), usize::from(bootstrap));
            assert_eq!(active[0].output_tokens, picks);
            assert_eq!(model.emitted(), picks);
            assert!(active[0].finished);
            assert!(active[0].pending_drafts.is_empty());
            synchronized(&model, &active[0]);
        }
    }
}

#[test]
fn every_terminal_row_commits_before_streaming_and_never_reproposes() {
    for rows in 2..=8 {
        for terminal_row in 0..rows {
            for eos in [false, true] {
                let mut picks = vec![1; rows];
                if eos {
                    picks[terminal_row] = 3;
                }
                let drafts = picks[..rows - 1].to_vec();
                let model = DriverModel::new(
                    &picks.iter().copied().map(scores).collect::<Vec<_>>(),
                    drafts.clone(),
                );
                let mut seq = sequence();
                seq.pending_drafts = drafts;
                if eos {
                    seq.eos_tokens = vec![3];
                } else {
                    seq.remaining = terminal_row + 1;
                }
                let mut active = one(&model, seq);
                run(&model, &mut active, rows - 1, true, false, None, &context());
                assert!(active[0].finished);
                assert_eq!(model.count("commit"), 1);
                assert_eq!(model.count("propose"), 0);
                assert_eq!(model.position(), 2 + (terminal_row + 2).min(rows));
                assert_eq!(active[0].output_tokens, picks[..=terminal_row]);
                let streamed = if eos {
                    &picks[..terminal_row]
                } else {
                    &picks[..=terminal_row]
                };
                assert_eq!(model.emitted(), streamed);
                synchronized(&model, &active[0]);
            }
        }
    }
}

#[test]
fn unique_raw_winner_loses_under_repetition_and_next_row_sees_accepted_history() {
    let model = DriverModel::new(&[[0.0, 10.0, 9.0, -8.0, -8.0]; 2], vec![2]);
    let mut seq = sequence();
    seq.last_token = 1;
    seq.output_tokens = vec![1];
    seq.pending_drafts = vec![2];
    seq.repetition_penalty = 2.0;
    seq.remaining = 2;
    let mut active = one(&model, seq);
    run(&model, &mut active, 1, true, false, None, &context());
    assert_eq!(active[0].output_tokens, vec![1, 2, 1]);
    assert_eq!(model.emitted(), vec![2, 1]);
    assert_eq!(model.inputs(), vec![vec![1, 2]]);
    synchronized(&model, &active[0]);
}

#[test]
fn live_grammar_handles_illegal_first_draft_before_legacy_pretruncation() {
    let model = DriverModel::new(&[scores(2); 4], vec![]);
    let mut seq = sequence();
    seq.pending_drafts = vec![2, 2, 2]; // "x" is illegal at JSON start.
    seq.grammar_state = Some(grammar());
    seq.eos_tokens = vec![3];
    let mut active = one(&model, seq);
    run(&model, &mut active, 3, true, false, None, &context());
    assert_eq!(model.inputs(), vec![vec![0, 2, 2, 2]]);
    assert_eq!(active[0].output_tokens, vec![0]); // Real mask selects "{".
    assert_eq!(
        active[0]
            .grammar_state
            .as_ref()
            .unwrap()
            .num_history_steps(),
        1
    );
    assert_eq!(model.position(), 3);
    assert_eq!(model.emitted(), vec![0]);
    synchronized(&model, &active[0]);
}

#[test]
fn adaptive_sampling_and_code_fence_settings_reach_the_actual_policy() {
    for adaptive in [false, true] {
        let model = DriverModel::new(&[scores(2); 2], vec![2]);
        let mut seq = sequence();
        seq.pending_drafts = vec![2];
        seq.inside_thinking = true;
        seq.enable_thinking = true;
        seq.remaining = 1;
        let mut active = one(&model, seq);
        run(&model, &mut active, 1, true, adaptive, Some(2), &context());
        assert!(active[0].in_code_fence);
        assert_eq!(
            active[0].adaptive.zone,
            if adaptive {
                GenerationZone::Thinking
            } else {
                GenerationZone::FreeText
            }
        );
        assert_eq!(model.emitted(), vec![2]);
        synchronized(&model, &active[0]);
    }
}

#[test]
fn no_drafts_or_empty_proposal_use_genuine_ordinary_caller_bias() {
    for raw in [false, true] {
        let model = DriverModel::new(&[scores(1); 2], vec![]);
        let mut seq = sequence();
        seq.logit_bias = vec![(2, 2.0)]; // Real ordinary row: token1=10, token2=9.
        seq.remaining = 1;
        let mut active = one(&model, seq);
        run(&model, &mut active, 1, raw, false, None, &context());
        assert_eq!(model.count("verify"), 0);
        assert_eq!(model.count("decode"), 1);
        assert_eq!(model.count("propose"), usize::from(raw));
        assert_eq!(active[0].output_tokens, vec![2]);
        assert_eq!(model.emitted(), vec![2]);
        assert!(model.ordinary.full_reads() > 0);
        synchronized(&model, &active[0]);
    }
}

#[test]
fn neutral_ordinary_driver_keeps_existing_zero_copy_greedy_admission() {
    let model = DriverModel::new(&[scores(1); 2], vec![]);
    let mut seq = sequence();
    seq.remaining = 1;
    let mut active = one(&model, seq);
    run(&model, &mut active, 1, false, false, None, &context());
    assert_eq!(active[0].output_tokens, vec![1]);
    assert_eq!(model.ordinary.argmax_calls(), 1);
    assert_eq!(model.ordinary.full_reads(), 0);
    assert!(model.ordinary.scalar_offsets().is_empty());
    synchronized(&model, &active[0]);
}

#[test]
fn restored_effect_replay_decodes_once_and_never_reuses_staged_logits() {
    let model = DriverModel::new(&[scores(1); 2], vec![1]);
    let mut seq = sequence();
    seq.pending_drafts = vec![1];
    seq.ssm_rollback_ring = SsmDecodeRing::new(1); // Real RecordedEffects requests replay.
    seq.logit_bias = vec![(2, 2.0)];
    seq.remaining = 1;
    let mut active = one(&model, seq);
    run(&model, &mut active, 1, true, false, None, &context());
    let calls = model.calls();
    assert!(
        calls.iter().position(|&c| c == "abort").unwrap()
            < calls.iter().position(|&c| c == "decode").unwrap()
    );
    assert_eq!(model.count("commit"), 0);
    assert_eq!(model.count("decode"), 1);
    assert_eq!(model.count("propose"), 0);
    assert_eq!(active[0].output_tokens, vec![2]); // Wide row selects1; ordinary row selects2.
    assert_eq!(model.emitted(), vec![2]);
    assert!(active[0].finished);
    assert!(active[0].pending_drafts.is_empty());
    synchronized(&model, &active[0]);
}

#[test]
fn restored_receipt_with_wrong_exact_prefix_never_authorizes_ordinary_decode() {
    let mut model = DriverModel::new(&[scores(1); 2], vec![1]);
    model.fault = Fault::WrongPrefix;
    let mut seq = sequence();
    seq.pending_drafts = vec![1];
    seq.ssm_rollback_ring = SsmDecodeRing::new(1);
    let mut active = one(&model, seq);
    run(&model, &mut active, 1, true, false, None, &context());
    assert_eq!(model.count("abort"), 1);
    assert_eq!(model.count("commit"), 0);
    assert_eq!(model.count("decode"), 0);
    assert!(model.count("poison") > 0);
    assert!(active[0].finished && active[0].terminal_error.is_some());
    assert_eq!(active[0].seq.tokens, vec![0, 0]);
    assert_eq!(model.position(), 2);
    assert!(model.emitted().is_empty());
}

#[test]
fn bind_copy_policy_commit_and_publication_failures_never_emit_or_decode_fallback() {
    for fault in [
        Fault::Bind,
        Fault::Copy,
        Fault::ShortCopy,
        Fault::Nan,
        Fault::Abort,
        Fault::Commit,
        Fault::WrongEnd,
        Fault::WrongPrefix,
    ] {
        let mut model = DriverModel::new(&[scores(1); 2], vec![1]);
        model.fault = fault;
        let mut seq = sequence();
        seq.pending_drafts = vec![1];
        let mut active = one(&model, seq);
        let before = fingerprint(&active[0]);
        run(&model, &mut active, 1, true, false, None, &context());
        assert!(active[0].finished && active[0].terminal_error.is_some());
        assert!(active[0].pending_drafts.is_empty());
        assert!(active[0].output_tokens.is_empty());
        assert!(model.emitted().is_empty());
        assert_eq!(model.count("decode"), 0);
        assert_eq!(model.count("propose"), 0);
        if matches!(
            fault,
            Fault::Abort | Fault::Commit | Fault::WrongEnd | Fault::WrongPrefix
        ) {
            assert!(model.count("poison") > 0);
        }
        // No fabricated host rollback after GPU commit failure.
        assert_eq!(active[0].seq.tokens, vec![0, 0]);
        assert_eq!(active[0].seq.seq_len, 2);
        assert_eq!(active[0].seq.kv_valid_tokens, 2);
        assert_eq!(fingerprint(&active[0])[0], before[0]);
    }
}

#[test]
fn proposal_and_ordinary_transport_errors_are_retired_not_disguised_as_success() {
    for fault in [Fault::Propose, Fault::OrdinaryDecode, Fault::OrdinaryCopy] {
        let mut model = DriverModel::new(&[scores(1); 2], vec![]);
        model.fault = fault;
        let mut seq = sequence();
        seq.logit_bias = vec![(2, 2.0)];
        let mut active = one(&model, seq);
        run(
            &model,
            &mut active,
            1,
            fault == Fault::Propose,
            false,
            None,
            &context(),
        );
        assert!(active[0].finished && active[0].terminal_error.is_some());
        assert!(active[0].output_tokens.is_empty());
        assert!(model.emitted().is_empty());
        assert_eq!(model.count("verify"), 0);
        assert_eq!(model.count("decode"), usize::from(fault != Fault::Propose));
    }
}

#[test]
fn unexpected_concurrency_is_rejected_before_any_model_effect() {
    let model = DriverModel::new(&[scores(1); 2], vec![1]);
    let mut active = vec![sequence(), sequence()];
    run(&model, &mut active, 1, true, false, None, &context());
    assert!(
        active
            .iter()
            .all(|a| a.finished && a.terminal_error.is_some())
    );
    for effect in ["bind", "verify", "propose", "decode", "commit"] {
        assert_eq!(model.count(effect), 0);
    }
}

#[test]
#[should_panic(expected = "legacy verify entered")]
fn other_models_keep_the_legacy_route() {
    let mut model = DriverModel::new(&[scores(1); 2], vec![1]);
    model.policy_required = false;
    let mut seq = sequence();
    seq.pending_drafts = vec![1];
    run(
        &model,
        std::slice::from_mut(&mut seq),
        1,
        false,
        false,
        None,
        &context(),
    );
}
