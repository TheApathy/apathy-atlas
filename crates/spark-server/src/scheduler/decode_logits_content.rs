// SPDX-License-Identifier: AGPL-3.0-only

//! Non-thinking token bookkeeping and degeneration watchdogs. Blocking
//! responses may rewind a proven-flat history; streaming responses discard
//! only the uncommitted sample because emitted tail tokens are irreversible.
//!
//! Every watchdog decides before sample commit and then stops; a rewind is
//! terminal tail cleanup, never a re-steer.
//!
//! [`super::rollback::precommit_rollback_history_is_safe`] admits only a flat
//! history — no grammar, no thinking, no tool phase, sidecars exactly aligned.
//! The inter-tool prose budget fires only while a grammar is live, so its
//! rewind is always declined and it keeps its legacy commit-then-stop
//! behaviour; the content-loop and fuzzy watchdogs can rewind a plain answer.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ContentTokenDisposition {
    CommitSample,
    DiscardSampleAndStop { dropped: usize },
}

fn without_current_sample(
    remaining_after_rollback: usize,
    content_tokens_after_rollback: u32,
    content_started_before: bool,
    think_just_ended_before: bool,
) -> (usize, u32, bool, bool) {
    (
        remaining_after_rollback.saturating_add(1),
        content_tokens_after_rollback.saturating_sub(1),
        content_started_before,
        think_just_ended_before,
    )
}

/// Committed length of a fuzzy window the detector matched on the projected
/// history. The window's last token is the uncommitted sample, so a rewind of
/// the committed tail must discard exactly one fewer token than the detector
/// compared — `pattern_len * 3` would demand a token that is not there.
fn committed_fuzzy_window(pattern_len: usize) -> usize {
    (pattern_len * 3).saturating_sub(1)
}

/// Run the fuzzy detector with a temporary, uncommitted tail sample.
fn projected_fuzzy_repetition(a: &mut ActiveSeq, sample: u32) -> Option<(usize, usize, usize)> {
    a.output_tokens.push(sample);
    let detection = fuzzy_repetition_outside_tool(a);
    let removed = a.output_tokens.pop();
    debug_assert_eq!(removed, Some(sample));
    detection
}

fn discard_sample_after_rollback(
    a: &mut ActiveSeq,
    dropped: usize,
    content_started_before: bool,
    think_just_ended_before: bool,
    streaming_state: Option<(u32, Instant, u32)>,
) -> ContentTokenDisposition {
    // The sampled token was charged but never committed.
    (
        a.remaining,
        a.content_tokens,
        a.content_started,
        a.think_just_ended,
    ) = without_current_sample(
        a.remaining,
        a.content_tokens,
        content_started_before,
        think_just_ended_before,
    );
    if let Some((last_token, last_token_time, prose_tokens)) = streaming_state {
        a.last_token = last_token;
        a.last_token_time = last_token_time;
        a.prose_tokens_since_last_tool = prose_tokens;
    }
    a.finished = true;
    ContentTokenDisposition::DiscardSampleAndStop { dropped }
}

/// Stop without retracting a stream; only blocking responses may rewind.
fn stop_uncommitted_loop_with<F>(
    a: &mut ActiveSeq,
    min_keep: usize,
    content_started_before: bool,
    think_just_ended_before: bool,
    previous_last_token: u32,
    previous_last_token_time: Instant,
    prose_tokens_before: u32,
    rollback: &mut F,
) -> ContentTokenDisposition
where
    F: FnMut(&mut ActiveSeq, usize) -> RollbackOutcome,
{
    if matches!(&a.sink, ResponseSink::Streaming(_)) {
        return discard_sample_after_rollback(
            a,
            0,
            content_started_before,
            think_just_ended_before,
            Some((
                previous_last_token,
                previous_last_token_time,
                prose_tokens_before,
            )),
        );
    }
    match rollback(a, min_keep) {
        RollbackOutcome::RolledBack { dropped } => discard_sample_after_rollback(
            a,
            dropped,
            content_started_before,
            think_just_ended_before,
            None,
        ),
        RollbackOutcome::Fallback(reason) => {
            tracing::debug!(
                ?reason,
                "Serial loop rollback declined; committing stop sample"
            );
            a.finished = true;
            ContentTokenDisposition::CommitSample
        }
    }
}

/// Return `CommitSample` before the caller mutates observable history.
pub fn handle_content_token(
    a: &mut ActiveSeq,
    sample: u32,
    model: &dyn Model,
    previous_last_token: u32,
    previous_last_token_time: Instant,
) -> ContentTokenDisposition {
    handle_content_token_with(
        a,
        sample,
        enable_loop_watchdog(),
        previous_last_token,
        previous_last_token_time,
        |a, min_keep| rollback_to_boundary(a, min_keep, model),
    )
}

fn handle_content_token_with<F>(
    a: &mut ActiveSeq,
    sample: u32,
    loop_watchdog_enabled: bool,
    previous_last_token: u32,
    previous_last_token_time: Instant,
    mut rollback: F,
) -> ContentTokenDisposition
where
    F: FnMut(&mut ActiveSeq, usize) -> RollbackOutcome,
{
    let content_started_before = a.content_started;
    let think_just_ended_before = a.think_just_ended;
    let prose_tokens_before = a.prose_tokens_since_last_tool;
    a.remaining -= 1;
    a.content_started = true;
    a.content_tokens = a.content_tokens.saturating_add(1);
    // `think_just_ended` is consumed by the first content sample.
    a.think_just_ended = false;

    // Repeated structured JSON inside grammar/tool bodies is legitimate.
    let catastrophic_loop = a.content_tokens >= CATASTROPHIC_LOOP_MIN_TOKENS as u32
        && a.content_tokens.is_multiple_of(CONTENT_LOOP_CHECK_STRIDE)
        && detect_catastrophic_content_loop(&a.output_tokens);
    let configured_loop = loop_watchdog_enabled
        && a.content_tokens >= CONTENT_LOOP_MIN_TOKENS
        && a.content_tokens.is_multiple_of(CONTENT_LOOP_CHECK_STRIDE)
        && (detect_content_token_loop(&a.output_tokens)
            || numeric_token_mask()
                .as_deref()
                .is_some_and(|m| detect_content_token_loop_normalized(&a.output_tokens, m)));
    if a.grammar_state.is_none() && !a.inside_tool_body && (catastrophic_loop || configured_loop) {
        tracing::warn!(
            content_tokens = a.content_tokens,
            output_len = a.output_tokens.len(),
            catastrophic = catastrophic_loop,
            "Content-loop watchdog fired (period-{}…{} repeat); ending response",
            CONTENT_LOOP_PERIOD_MIN,
            CONTENT_LOOP_PERIOD_MAX,
        );
        return stop_uncommitted_loop_with(
            a,
            CONTENT_LOOP_PERIOD_MAX,
            content_started_before,
            think_just_ended_before,
            previous_last_token,
            previous_last_token_time,
            prose_tokens_before,
            &mut rollback,
        );
    }

    // Bound free prose between grammar-constrained tool bodies.
    if !a.inside_tool_body && a.grammar_state.is_some() {
        a.prose_tokens_since_last_tool = a.prose_tokens_since_last_tool.saturating_add(1);
        let max_prose = watchdog_params().max_inter_tool_prose;
        if a.prose_tokens_since_last_tool > max_prose {
            tracing::warn!(
                prose_tokens = a.prose_tokens_since_last_tool,
                max = max_prose,
                "Inter-tool prose budget exhausted; ending response"
            );
            return stop_uncommitted_loop_with(
                a,
                CONTENT_LOOP_PERIOD_MAX,
                content_started_before,
                think_just_ended_before,
                previous_last_token,
                previous_last_token_time,
                prose_tokens_before,
                &mut rollback,
            );
        }
    }

    if loop_watchdog_enabled
        && !a.finished
        && let Some((pattern_len, mis_a, mis_b)) = projected_fuzzy_repetition(a, sample)
    {
        tracing::warn!(
            pattern_len,
            mismatches = mis_a + mis_b,
            output_len = a.output_tokens.len(),
            "Fuzzy repetition detected before serial commit; ending response"
        );
        return stop_uncommitted_loop_with(
            a,
            committed_fuzzy_window(pattern_len),
            content_started_before,
            think_just_ended_before,
            previous_last_token,
            previous_last_token_time,
            prose_tokens_before,
            &mut rollback,
        );
    }
    ContentTokenDisposition::CommitSample
}

#[cfg(test)]
#[path = "decode_logits_content_tests.rs"]
mod precommit_tests;

#[cfg(test)]
#[path = "decode_logits_content_integration_tests.rs"]
mod integration_tests;
