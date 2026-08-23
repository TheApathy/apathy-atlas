// SPDX-License-Identifier: AGPL-3.0-only

use super::precommit_tests::active_seq;
use super::*;
use crate::scheduler::rollback::{RollbackFallback, precommit_rollback_history_is_safe};

#[test]
fn fuzzy_stop_comes_from_the_projected_sample_and_never_retains_it() {
    let (mut active, _) = active_seq(false);
    let output_before = active.output_tokens.clone();
    assert!(fuzzy_repetition_outside_tool(&active).is_none());
    assert_eq!(
        projected_fuzzy_repetition(&mut active, 49),
        Some((30, 0, 0))
    );
    assert_eq!(active.output_tokens, output_before);

    let previous_last_token = active.last_token;
    let previous_last_token_time = active.last_token_time;
    active.last_token = 49;
    let mut seen_min_keep = None;
    handle_content_token_with(
        &mut active,
        49,
        true,
        previous_last_token,
        previous_last_token_time,
        |_active, min_keep| {
            seen_min_keep = Some(min_keep);
            RollbackOutcome::Fallback(RollbackFallback::NoBoundary)
        },
    );
    assert_eq!(seen_min_keep, Some(committed_fuzzy_window(30)));
    assert_eq!(seen_min_keep, Some(89));
}

fn json_grammar() -> crate::grammar::GrammarState {
    let mut vocab: Vec<String> = (0u8..128).map(|byte| String::from(byte as char)).collect();
    vocab.push("<eos>".to_string());
    let mut engine = crate::grammar::GrammarEngine::new(&vocab, &[128]).unwrap();
    let compiled = engine.compile_json_grammar().unwrap();
    crate::grammar::GrammarState::new(&compiled, engine.vocab_size()).unwrap()
}

#[test]
fn grammar_prose_budget_declines_the_rewind_and_commits_as_before() {
    let (mut active, _) = active_seq(false);
    active.grammar_state = Some(json_grammar());
    active.prose_tokens_since_last_tool = watchdog_params().max_inter_tool_prose;
    assert!(!precommit_rollback_history_is_safe(&active));
    let previous_last_token = active.last_token;
    let previous_last_token_time = active.last_token_time;
    active.last_token = 49;
    let mut seen_min_keep = None;
    let disposition = handle_content_token_with(
        &mut active,
        49,
        true,
        previous_last_token,
        previous_last_token_time,
        |active, min_keep| {
            seen_min_keep = Some(min_keep);
            assert!(!precommit_rollback_history_is_safe(active));
            RollbackOutcome::Fallback(RollbackFallback::UnsafeObservableHistory)
        },
    );
    assert_eq!(seen_min_keep, Some(CONTENT_LOOP_PERIOD_MAX));
    assert_eq!(disposition, ContentTokenDisposition::CommitSample);
    assert!(active.finished);
    assert_eq!(active.output_tokens.len(), 89);
    assert_eq!(active.logprobs_data.len(), 89);
    assert_eq!(active.remaining, 909);
    assert_eq!(active.content_tokens, 90);
}

#[test]
fn streaming_discard_restores_the_uncommitted_prose_charge() {
    let (mut active, _) = active_seq(true);
    let last_token = active.last_token;
    let last_token_time = active.last_token_time;
    active.prose_tokens_since_last_tool = 8;
    discard_sample_after_rollback(
        &mut active,
        0,
        true,
        false,
        Some((last_token, last_token_time, 7)),
    );
    assert_eq!(active.prose_tokens_since_last_tool, 7);
}
