// SPDX-License-Identifier: AGPL-3.0-only

use super::super::rollback::{
    RollbackFallback, RollbackOutcome, apply_rollback, precommit_rollback_history_is_safe,
};
use super::*;
use spark_model::traits::SequenceState;

#[test]
fn discard_branch_precedes_every_observable_commit() {
    let step = include_str!("decode_logits_step.rs");
    let handler = step.find("match handle_content_token(").unwrap();
    let discard = step[handler..]
        .find("ContentTokenDisposition::DiscardSampleAndStop")
        .unwrap()
        + handler;
    let grammar = step.find("advance_content_grammar(a, tok)").unwrap();
    let logprob = step.find("a.logprobs_data.push(lp)").unwrap();
    let output = step.find("a.output_tokens.push(tok)").unwrap();
    let discard_end = step[discard..].find("continue;").unwrap() + discard + "continue;".len();
    let discard_branch = &step[discard..discard_end];
    assert!(handler < grammar && grammar < discard && discard < logprob && discard < output);
    assert!(discard_branch.contains("continue"));
    let forbidden = [
        "output_tokens",
        "logprobs_data",
        "advance_content_grammar",
        "update_tool_body_phase",
        "snapshot_boundary_if_ssm",
        "StreamEvent",
    ];
    assert!(forbidden.iter().all(|item| !discard_branch.contains(item)));
    assert!(!step.contains("Fuzzy repetition detected; rolled back"));
}

fn seq_state(tokens: &[u32]) -> SequenceState {
    SequenceState {
        tokens: tokens.to_vec(),
        block_table: Vec::new(),
        seq_len: tokens.len(),
        qwen4_qsa_required: false,
        layer_states: Vec::new(),
        proposer_state: None,
        proposer_state_alt: None,
        slot_idx: 0,
        marconi_skip_to: 0,
        session_hash: 0,
        chunked_prefill_meta: None,
        cached_prefix_tokens: 0,
        prompt_len: 0,
        disk_block_ids: Vec::new(),
        mtp_lastk_host_buf: Vec::new(),
        mtp_lastk_host_filled: 0,
        mtp_lastk_end_abs: 0,
        disk_last_offloaded_per_layer: Vec::new(),
    }
}

pub(super) fn active_seq(
    streaming: bool,
) -> (ActiveSeq, Option<tokio::sync::mpsc::Receiver<StreamEvent>>) {
    let pattern: Vec<u32> = (20..50).collect();
    let output: Vec<u32> = pattern.iter().copied().cycle().take(89).collect();
    let logprobs = output
        .iter()
        .map(|&token_id| crate::api::TokenLogprobs {
            token_id,
            logprob: -0.5,
            top: vec![(token_id, -0.5)],
        })
        .collect();
    let (sink, rx) = if streaming {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        (ResponseSink::Streaming(tx), Some(rx))
    } else {
        (ResponseSink::Blocking(None), None)
    };
    let now = Instant::now();
    (
        ActiveSeq {
            seq: seq_state(&output),
            session_hash: 0,
            last_token: *output.last().unwrap(),
            output_tokens: output,
            max_output_tokens: 256,
            remaining: 910,
            min_tokens: 0,
            eos_tokens: vec![99],
            finished: false,
            sink,
            cancel_flag: None,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            top_n_sigma: 0.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            repetition_penalty_window: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            lz_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 3,
            dry_sequence_breakers: Vec::new(),
            logit_bias: Vec::new(),
            inside_thinking: false,
            enable_thinking: false,
            thinking_budget: None,
            spontaneous_think_budget: 512,
            thinking_tokens: 0,
            force_end_thinking: false,
            consecutive_confident: 0,
            in_code_fence: false,
            think_end_token: None,
            think_start_token: None,
            think_ended: false,
            think_just_ended: false,
            think_skip_count: 0,
            post_think_gate_steps: 0,
            tool_call_end_token: Some(98),
            require_tool_call: false,
            tool_call_start_token: Some(97),
            tool_call_opened: false,
            inside_tool_body: false,
            suppress_tool_call: false,
            disable_mtp: false,
            content_started: true,
            content_tokens: 89,
            prose_tokens_since_last_tool: 0,
            think_watchdog_fires: 0,
            rollback_count: 0,
            ssm_rollback_ring: SsmDecodeRing::new(0),
            grammar_state: None,
            pending_drafts: Vec::new(),
            draft_origin: DraftOrigin::default(),
            last_verify_accepted: 0,
            self_context: Default::default(),
            pending_tree_payload: None,
            last_token_time: now,
            request_start: now,
            decode_start: now,
            seed: None,
            top_logprobs: Some(1),
            logprobs_data: logprobs,
            timeout_at: None,
            adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(0.0),
            cached_prompt_tokens: 0,
            difficulty_probe: Default::default(),
        },
        rx,
    )
}

#[test]
fn real_handler_streaming_fuzzy_stop_never_rolls_back_or_commits() {
    let (mut active, mut rx) = active_seq(true);
    let output_before = active.output_tokens.clone();
    let previous_last_token = active.last_token;
    let previous_last_token_time = active.last_token_time;
    active.last_token = 49;
    active.last_token_time += std::time::Duration::from_secs(1);
    active.prose_tokens_since_last_tool = 7;
    let disposition = handle_content_token_with(
        &mut active,
        49,
        true,
        previous_last_token,
        previous_last_token_time,
        |_active, _min_keep| panic!("streaming history must never roll back"),
    );
    assert_eq!(
        disposition,
        ContentTokenDisposition::DiscardSampleAndStop { dropped: 0 }
    );
    assert!(active.finished && active.content_started && !active.think_just_ended);
    assert_eq!(active.output_tokens, output_before);
    assert_eq!(active.logprobs_data.len(), output_before.len());
    assert_eq!(active.remaining, 910);
    assert_eq!(active.content_tokens, 89);
    assert_eq!(active.last_token, previous_last_token);
    assert_eq!(active.last_token_time, previous_last_token_time);
    assert_eq!(active.prose_tokens_since_last_tool, 7);
    assert!(matches!(
        rx.as_mut().unwrap().try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
}

#[test]
fn catastrophic_loop_stops_even_when_configured_watchdog_is_disabled() {
    const CYCLE: [u32; 10] = [15, 20, 18, 92, 198, 59, 76367, 90, 19, 13];
    let (mut active, mut rx) = active_seq(true);
    active.output_tokens = (0u32..202).collect();
    for _ in 0..32 {
        active.output_tokens.extend(CYCLE);
    }
    active.max_output_tokens = 65_536;
    active.content_tokens = 511;
    active.remaining = 65_000;
    let previous_last_token = active.last_token;
    let previous_last_token_time = active.last_token_time;

    let disposition = handle_content_token_with(
        &mut active,
        42,
        false,
        previous_last_token,
        previous_last_token_time,
        |_active, _min_keep| panic!("a streaming history must never roll back"),
    );

    assert_eq!(
        disposition,
        ContentTokenDisposition::DiscardSampleAndStop { dropped: 0 }
    );
    assert!(active.finished);
    assert_eq!(active.content_tokens, 511);
    assert_eq!(active.remaining, 65_000);
    assert!(matches!(
        rx.as_mut().unwrap().try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
}

#[test]
fn blocking_rollback_rewinds_logprobs_with_output() {
    let (mut active, _) = active_seq(false);
    let previous_last_token = active.last_token;
    let previous_last_token_time = active.last_token_time;
    active.last_token = 49;
    let disposition = handle_content_token_with(
        &mut active,
        49,
        true,
        previous_last_token,
        previous_last_token_time,
        |active, _min_keep| {
            assert!(precommit_rollback_history_is_safe(active));
            active.logprobs_data[0].token_id = u32::MAX;
            assert!(!precommit_rollback_history_is_safe(active));
            active.logprobs_data[0].token_id = active.output_tokens[0];
            let keep = 30;
            let dropped = active.output_tokens.len() - keep;
            apply_rollback(active, keep, dropped);
            RollbackOutcome::RolledBack { dropped }
        },
    );
    assert_eq!(
        disposition,
        ContentTokenDisposition::DiscardSampleAndStop { dropped: 59 }
    );
    assert!(active.finished);
    assert_eq!(active.output_tokens.len(), 30);
    assert_eq!(active.logprobs_data.len(), 30);
    assert!(
        active
            .logprobs_data
            .iter()
            .zip(&active.output_tokens)
            .all(|(lp, &tok)| lp.token_id == tok)
    );
    assert_eq!(active.content_tokens as usize, active.output_tokens.len());
    assert_eq!(active.remaining, 910 - 1 + 59 + 1);
}

#[test]
fn declined_blocking_rollback_keeps_ordinary_commit_disposition() {
    let (mut active, _) = active_seq(false);
    let previous_last_token = active.last_token;
    let previous_last_token_time = active.last_token_time;
    active.last_token = 49;
    let disposition = handle_content_token_with(
        &mut active,
        49,
        true,
        previous_last_token,
        previous_last_token_time,
        |_active, _min_keep| RollbackOutcome::Fallback(RollbackFallback::Disabled),
    );
    assert_eq!(disposition, ContentTokenDisposition::CommitSample);
    assert!(active.finished);
    assert_eq!(active.output_tokens.len(), 89);
    assert_eq!(active.logprobs_data.len(), 89);
    assert_eq!(active.remaining, 909);
    assert_eq!(active.content_tokens, 90);
}
