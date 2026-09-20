// SPDX-License-Identifier: AGPL-3.0-only

//! Ordinary post-pick policy SSOT. All persistent/model/stream effects are
//! injected; this is the existing decode transition, not the MTP emitter.

use crate::scheduler::*;
#[path = "ordinary_transition_effects.rs"]
mod effects;
use crate::scheduler::logit_processors::LogitsContext;
pub(super) use effects::{LiveEffects, OrdinaryEffects};

pub(super) struct TransitionContext {
    pub logits: LogitsContext,
    pub code_fence_token: Option<u32>,
    pub max_seq_len: usize,
    pub now: Instant,
}

pub(super) fn advance_ordinary(
    a: &mut ActiveSeq,
    tok: u32,
    logprobs: Option<crate::api::TokenLogprobs>,
    ctx: &TransitionContext,
    effects: &mut dyn OrdinaryEffects,
) -> Result<()> {
    let think_end_token = ctx.logits.think_end_token;
    let think_start_token = ctx.logits.think_start_token;
    let code_fence_token = ctx.code_fence_token;
    let tool_call_start_token = ctx.logits.tool_call_start_token;
    let tool_call_end_token = ctx.logits.tool_call_end_token;
    a.last_token = tok;
    a.last_token_time = ctx.now;

    if tool_response_stop_enabled()
        && let Some(trs) = tool_response_hard_stop()
        && tok == trs
    {
        a.output_tokens.push(tok);
        a.finished = true;
        tracing::debug!("<tool_response> hard-stop fired (id={trs}); ending turn");
        return Ok(());
    }

    if !a.inside_thinking && think_start_token == Some(tok) {
        let decay_shift = a.think_watchdog_fires.min(4);
        let decayed = a.spontaneous_think_budget >> decay_shift;
        a.inside_thinking = true;
        a.think_ended = false; // reset so </think> detection path works
        a.think_skip_count = 0;
        a.thinking_budget = Some(decayed.max(8)); // floor to keep watchdog functional
        if a.think_watchdog_fires > 0 {
            tracing::debug!(
                fires = a.think_watchdog_fires,
                decayed_budget = decayed,
                "Spontaneous <think> re-entry after watchdog; decayed budget"
            );
        } else {
            tracing::debug!("Spontaneous <think> detected, entering thinking mode");
        }
        return Ok(()); // don't emit <think> as content
    }

    if !a.inside_thinking && think_end_token == Some(tok) {
        a.think_skip_count += 1;
        if a.think_skip_count >= 50 {
            a.finished = true;
        }
        return Ok(());
    }
    if a.think_ended {
        a.think_skip_count = 0;
    }

    if !a.inside_thinking
        && let Some(ref mut gs) = a.grammar_state
    {
        effects.accept_grammar(gs, tok)?;
    }

    if a.inside_thinking {
        a.consume_generation_budget();
        if think_end_token == Some(tok) {
            a.inside_thinking = false;
            a.think_force_closed = a.force_end_thinking;
            a.force_end_thinking = false;
            a.sentence_defer_count = 0;
            a.consecutive_confident = 0;
            a.in_code_fence = false;
            a.think_ended = true;
            a.think_just_ended = true;
        } else {
            a.thinking_tokens += 1;
            a.in_code_fence = toggle_code_fence(a.in_code_fence, tok, code_fence_token);
            if let Some(budget) = a.thinking_budget
                && a.thinking_tokens >= budget
                && !a.force_end_thinking
            {
                a.force_end_thinking = true;
                a.sentence_defer_count = 0;
                tracing::info!(
                    "Thinking budget exhausted ({budget} tokens), arming </think>; \
                     deferring up to {MAX_SENTENCE_DEFER_TOKENS} tokens for sentence boundary"
                );
            }
            if !crate::scheduler::helpers::disable_watchdogs()
                && !a.force_end_thinking
                && a.thinking_tokens >= THINK_LOOP_MIN_TOKENS
                && a.thinking_tokens.is_multiple_of(THINK_LOOP_CHECK_STRIDE)
                && detect_thinking_token_loop_with(&a.output_tokens, a.repetition_detection)
            {
                a.force_end_thinking = true;
                a.sentence_defer_count = 0;
                a.think_watchdog_fires = a.think_watchdog_fires.saturating_add(1);
                tracing::warn!(
                    thinking_tokens = a.thinking_tokens,
                    watchdog_fires = a.think_watchdog_fires,
                    "Thinking-loop watchdog fired (period-{}…{} repeat in tail); forcing </think> early",
                    THINK_LOOP_PERIOD_MIN,
                    THINK_LOOP_PERIOD_MAX,
                );
            }
        }
    } else {
        handle_content_token(a, effects)?;
    }

    if a.require_tool_call && tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.require_tool_call = false;
        a.tool_call_opened = true;
    }
    if tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.prose_tokens_since_last_tool = 0;
        if a.tool_call_completed {
            a.post_completion_tool_opens = a.post_completion_tool_opens.saturating_add(1);
            const MAX_POST_COMPLETION_TOOL_OPENS: u32 = 8;
            if a.post_completion_tool_opens >= MAX_POST_COMPLETION_TOOL_OPENS {
                tracing::warn!(
                    opens = a.post_completion_tool_opens,
                    "tool-call repetition runaway: model re-opened {MAX_POST_COMPLETION_TOOL_OPENS}+ tool-call blocks after a completed call on a tool_choice=auto turn; ending response (was burning to max_tokens). Sanitizer keeps the first valid call(s)."
                );
                a.output_tokens.push(tok);
                a.tool_call_opened = true;
                // The common transition above already accepted this token.
                a.finished = true;
                return Ok(());
            }
        }
    }
    if a.require_tool_call && a.output_tokens.len() > 512 {
        tracing::warn!(
            "require_tool_call safety: no <tool_call> after 512 tokens, clearing EOS suppression"
        );
        a.require_tool_call = false;
    }

    if let Some(lp) = logprobs {
        a.logprobs_data.push(lp);
    }

    if tool_call_end_token == Some(tok) && !a.inside_thinking {
        a.output_tokens.push(tok);
        a.tool_call_completed = true;
        if matches!(a.sink, ResponseSink::Streaming(_)) {
            let event = if let Some(lp) = a.logprobs_data.last().cloned() {
                StreamEvent::TokenWithLogprobs(tok, lp)
            } else {
                StreamEvent::Token(tok)
            };
            if !effects.stream(a, event, "tool_call_end")? {
                tracing::warn!(
                    "Streaming receiver dropped during tool_call_end, finishing sequence"
                );
                a.finished = true;
                return Ok(());
            }
        }
        if a.grammar_state.is_none() && !a.tools_present {
            a.finished = true;
        }
        a.inside_tool_body = false;
        // Exactly one matcher advance per generated token. A second close
        // either corrupts auto-mode history or rejects a required-call grammar.
        a.think_ended = false;
        return Ok(());
    }

    let eos_escape = tool_eos_escape_enabled()
        && a.tool_call_completed
        && !a.inside_tool_body
        && !a.inside_thinking;
    let grammar_suppresses_eos = a.eos_tokens.contains(&tok)
        && !eos_escape
        && crate::grammar::grammar_blocks_stop(a.grammar_state.as_mut(), &a.eos_tokens);
    let legacy_suppresses_eos = a.require_tool_call;
    let min_tokens_suppresses = a.output_tokens.len() < a.min_tokens;
    let hard_ceiling = hard_ceiling_hit(a.remaining, effects.position(a), ctx.max_seq_len);
    let thinking_suppresses_eos = eos_suppressed_by_thinking(a.inside_thinking, hard_ceiling);
    const POST_THINK_MIN_CONTENT: u32 = 16;
    let post_think_content_tokens =
        (a.output_tokens.len() as u32).saturating_sub(a.thinking_tokens);
    let tools_armed = a.require_tool_call || a.tool_request;
    let post_think_suppresses_eos = tools_armed
        && a.think_ended
        && a.think_force_closed
        && post_think_content_tokens < POST_THINK_MIN_CONTENT;
    let suppress_eos = grammar_suppresses_eos
        || legacy_suppresses_eos
        || min_tokens_suppresses
        || thinking_suppresses_eos
        || post_think_suppresses_eos;

    if a.eos_tokens.contains(&tok) && !suppress_eos {
        a.output_tokens.push(tok);
        crate::scheduler::emit_step::update_tool_param_state(a, tok);
        a.finished = true;
    } else if a.eos_tokens.contains(&tok) && suppress_eos {
    } else {
        a.output_tokens.push(tok);
        crate::scheduler::emit_step::update_tool_param_state(a, tok);
        if !a.inside_thinking {
            effects.checkpoint(a)?;
        }
        let suppress_stream = a.inside_thinking && !a.enable_thinking;
        if matches!(a.sink, ResponseSink::Streaming(_)) && !suppress_stream {
            let event = if let Some(lp) = a.logprobs_data.last().cloned() {
                StreamEvent::TokenWithLogprobs(tok, lp)
            } else {
                StreamEvent::Token(tok)
            };
            if !effects.stream(a, event, "decode_logits token")? {
                tracing::debug!("Streaming receiver dropped (decode_logits), finishing seq");
                a.finished = true;
            }
        }
        if a.remaining == 0 {
            effects.grammar_close(a)?;
            tracing::info!(
                "process_decode_logits: remaining=0, output_tokens={}, thinking_tokens={}",
                a.output_tokens.len(),
                a.thinking_tokens
            );
            a.finished = true;
        }
        if !a.finished && seqlen_force_stop(effects.position(a), ctx.max_seq_len) {
            tracing::info!(
                seq_len = effects.position(a),
                max_seq_len = ctx.max_seq_len,
                output_tokens = a.output_tokens.len(),
                "process_decode_logits: max_seq_len ceiling reached; force-stop (finish=length)"
            );
            a.finished = true;
        }
        if a.grammar_state
            .as_ref()
            .is_some_and(|gs| gs.is_terminated())
        {
            a.finished = true;
        }

        let last_tc_start = a
            .tool_call_start_token
            .and_then(|t| a.output_tokens.iter().rposition(|&tok| tok == t));
        let last_tc_end = a
            .tool_call_end_token
            .and_then(|t| a.output_tokens.iter().rposition(|&tok| tok == t));
        let inside_tool_call = match (last_tc_start, last_tc_end) {
            (Some(start), Some(end)) => start > end,
            (Some(_), None) => true,
            _ => false,
        };
        if enable_loop_watchdog()
            && !a.finished
            && !a.inside_thinking
            && !inside_tool_call
            && let Some((pattern_len, mis_a, mis_b)) = detect_fuzzy_repetition(&a.output_tokens)
        {
            let min_keep = pattern_len * 3;
            match effects.rollback(a, min_keep)? {
                RollbackOutcome::RolledBack { dropped } => {
                    tracing::warn!(
                        pattern_len,
                        mismatches = mis_a + mis_b,
                        dropped,
                        rollback = a.rollback_count,
                        "Fuzzy repetition detected; rolled back to boundary, re-steering"
                    );
                }
                RollbackOutcome::Fallback(reason) => {
                    tracing::warn!(
                        "Fuzzy repetition: {pattern_len}-tok pattern x3 ({mis_a}+{mis_b} \
                         mismatches), stopping at {} tokens (rollback declined: {reason:?})",
                        a.output_tokens.len()
                    );
                    a.guard_stop = Some("fuzzy_repetition");
                    a.finished = true;
                }
            }
        }

        if !a.finished
            && let Some(deadline) = a.timeout_at
            && Instant::now() >= deadline
        {
            tracing::warn!("Request timeout after {:?}", a.request_start.elapsed());
            a.finished = true;
        }
    }

    Ok(())
}
