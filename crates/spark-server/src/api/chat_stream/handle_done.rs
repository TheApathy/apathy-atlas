// SPDX-License-Identifier: AGPL-3.0-only
//
// `StreamEvent::Done { ... }` arm of the streaming `flat_map`
// closure (originally ~396 LoC).

use axum::response::sse::Event;

use crate::openai::{ChatCompletionChunk, Usage};
use crate::tool_parser;

use super::super::failures::{bump_f12_tool_call_count, flush_content_sanitizer};
use super::super::sanitizer::sanitize_content_chunk;
use super::ctx::StreamCtx;
use super::handle_token::process_detector_content;
use super::state::StreamState;
use super::tool_handlers::{
    handle_complete_tool_call, handle_tool_call_delta, handle_tool_call_end, handle_tool_call_start,
};

type SseVec = Vec<Result<Event, std::convert::Infallible>>;

fn resolved_finish_reason(
    finish_reason: &str,
    tool_loop_capped: bool,
    has_tool_calls: bool,
    loop_watchdog_triggered: bool,
) -> &str {
    if tool_loop_capped {
        // Tool loops must remain visibly truncated so agent clients do not
        // execute another tool round and perpetuate the outer loop.
        "length"
    } else if has_tool_calls {
        "tool_calls"
    } else if loop_watchdog_triggered {
        // A content-loop guard is a deliberate server stop, not exhaustion of
        // the request's token budget. Reporting `length` makes benchmark
        // clients retry the same degenerate response with a larger allowance.
        "stop"
    } else {
        finish_reason
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_done(
    state: &mut StreamState,
    ctx: &StreamCtx,
    finish_reason: String,
    completion_tokens: usize,
    time_to_first_token_ms: f64,
    decode_time_ms: f64,
    reasoning_tokens: u32,
    cached_prompt_tokens: u32,
) -> SseVec {
    let mut sse_events: SseVec = Vec::new();

    // A natural completion can end on a proper prefix of a configured
    // stop string. Since no later delta can complete it, feed that raw
    // decoded suffix through the ordinary detector/sanitizer pipeline.
    let incomplete_prefix = if !state.stop_string_triggered {
        std::mem::take(&mut state.stop_holdback)
    } else {
        String::new()
    };

    // ── Detector flush ──────────────────────────────────────────────
    if state.detector.is_some() {
        let outputs = {
            let det = state.detector.as_mut().expect("detector is Some");
            let mut outputs = if incomplete_prefix.is_empty() {
                Vec::new()
            } else {
                det.process(&incomplete_prefix)
            };
            outputs.extend(det.flush());
            outputs
        };
        for output in outputs {
            match output {
                tool_parser::DetectorOutput::Content(text) => {
                    let sanitized = sanitize_content_chunk(
                        &text,
                        &mut state.tag_scan_buf,
                        &mut state.suppressing_param_leak,
                        &mut state.inside_envelope,
                        &ctx.leak_markers,
                    );
                    if let Some(events) = process_detector_content(state, ctx, &sanitized) {
                        sse_events.extend(events);
                    }
                }
                tool_parser::DetectorOutput::ToolCall(mut tc, tc_idx) => {
                    handle_complete_tool_call(state, ctx, &mut tc, tc_idx, &mut sse_events);
                }
                tool_parser::DetectorOutput::ToolCallStart {
                    id: tc_id,
                    name,
                    idx,
                } => {
                    handle_tool_call_start(state, ctx, tc_id, name, idx, &mut sse_events);
                }
                tool_parser::DetectorOutput::ToolCallDelta { args, idx } => {
                    handle_tool_call_delta(state, ctx, args, idx, &mut sse_events);
                }
                tool_parser::DetectorOutput::ToolCallEnd { idx } => {
                    handle_tool_call_end(state, ctx, idx);
                }
            }
        }
    } else if !incomplete_prefix.is_empty() {
        let sanitized = sanitize_content_chunk(
            &incomplete_prefix,
            &mut state.tag_scan_buf,
            &mut state.suppressing_param_leak,
            &mut state.inside_envelope,
            &ctx.leak_markers,
        );
        if let Some(events) = process_detector_content(state, ctx, &sanitized) {
            sse_events.extend(events);
        }
    }

    // ── Sanitizer tail flush ────────────────────────────────────────
    let tail = flush_content_sanitizer(
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &ctx.leak_markers,
    );
    if let Some(events) = process_detector_content(state, ctx, &tail) {
        sse_events.extend(events);
    }

    // ── Usage block ─────────────────────────────────────────────────
    let tps = if decode_time_ms > 0.0 {
        completion_tokens.saturating_sub(1) as f64 / (decode_time_ms / 1000.0)
    } else {
        0.0
    };
    let usage = Usage {
        prompt_tokens: ctx.prompt_len,
        completion_tokens,
        total_tokens: ctx.prompt_len + completion_tokens,
        prompt_tokens_details: Some(crate::openai::PromptTokensDetails {
            cached_tokens: cached_prompt_tokens as usize,
            audio_tokens: 0,
        }),
        completion_tokens_details: Some(crate::openai::CompletionTokensDetails {
            reasoning_tokens: reasoning_tokens as usize,
            audio_tokens: 0,
            accepted_prediction_tokens: 0,
            rejected_prediction_tokens: 0,
        }),
        time_to_first_token_ms,
        response_tokens_per_second: tps,
    };

    // ── Last-resort tool salvage ────────────────────────────────────
    if !state.salvaged_tool_call && !state.detector.as_ref().is_some_and(|d| d.has_tool_calls()) {
        let salvaged =
            crate::tool_salvage::salvage(&state.refusal_scan_buf, &ctx.tool_defs_for_backfill);
        for (idx, tc) in salvaged.iter().enumerate() {
            tracing::warn!(
                tool = %tc.function.name,
                block_index = idx,
                "tool_salvage: emitting synthetic tool_call from prose",
            );
            bump_f12_tool_call_count(
                &mut state.tool_calls_emitted_count,
                ctx.max_tool_calls_per_response,
                &mut state.stop_string_triggered,
            );
            let start = ChatCompletionChunk::tool_call_start_chunk(&ctx.model, &ctx.id, tc, idx);
            sse_events.push(Ok(
                Event::default().data(serde_json::to_string(&start).unwrap_or_default())
            ));
            let frag = ChatCompletionChunk::tool_call_args_fragment(
                &ctx.model,
                &ctx.id,
                idx,
                &tc.function.arguments,
            );
            sse_events.push(Ok(
                Event::default().data(serde_json::to_string(&frag).unwrap_or_default())
            ));
        }
        if !salvaged.is_empty() {
            state.salvaged_tool_call = true;
        }
    }

    let fr = resolved_finish_reason(
        finish_reason.as_str(),
        state.tool_loop_capped,
        state.detector.as_ref().is_some_and(|d| d.has_tool_calls()) || state.salvaged_tool_call,
        state.loop_watchdog_triggered,
    );

    // Refusal classification.
    let refusal_signal = if state.detector.as_ref().is_none_or(|d| !d.has_tool_calls()) {
        crate::refusal::detect(&state.refusal_scan_buf)
    } else {
        None
    };
    if let Some(ref r) = refusal_signal {
        let chunk = ChatCompletionChunk::refusal_chunk(&ctx.model, &ctx.id, r.clone());
        let json = serde_json::to_string(&chunk).unwrap_or_default();
        sse_events.push(Ok(Event::default().data(json)));
    }

    // Usage emission strategy.
    let emit_separate_usage = ctx.req_stream_include_usage;
    let usage_for_dump = usage.clone();
    if emit_separate_usage {
        let usage_chunk = ChatCompletionChunk::usage_only_chunk(&ctx.model, &ctx.id, usage.clone());
        let json = serde_json::to_string(&usage_chunk).unwrap_or_default();
        sse_events.push(Ok(Event::default().data(json)));
        // Residual: tokens whose decoded text was buffered/suppressed
        // and never rode a content chunk. Stamping them here keeps
        // Σ token_ids == completion_tokens exactly.
        let final_chunk = ChatCompletionChunk::final_chunk_no_usage(&ctx.model, &ctx.id, fr)
            .with_token_ids(state.take_ids_if(ctx.req_return_token_ids));
        let json = serde_json::to_string(&final_chunk).unwrap_or_default();
        sse_events.push(Ok(Event::default().data(json)));
    } else {
        let chunk = ChatCompletionChunk::done_chunk(&ctx.model, &ctx.id, fr, usage)
            .with_token_ids(state.take_ids_if(ctx.req_return_token_ids));
        let json = serde_json::to_string(&chunk).unwrap_or_default();
        sse_events.push(Ok(Event::default().data(json)));
    }

    // Metrics.
    crate::metrics::REQUESTS_ACTIVE.dec();
    crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(ctx.prompt_len as u64);
    crate::metrics::GENERATION_TOKENS_TOTAL.inc_by(completion_tokens as u64);
    crate::metrics::TTFT_SECONDS.observe(time_to_first_token_ms / 1000.0);

    // Rate-limit true-up.
    if let Some(ref rctx) = ctx.req_ctx {
        let actual = (ctx.prompt_len + completion_tokens) as u64;
        let refund = rctx.reserved_tokens.saturating_sub(actual);
        if refund > 0 {
            ctx.state.rate_limiter.refund_tokens(&rctx.identity, refund);
        }
    }

    // --dump synthesized response entry.
    if let (Some(seq), Some(dump)) = (ctx.dump_seq, ctx.state.dump_writer.as_ref()) {
        let has_tool_calls = state.detector.as_ref().is_some_and(|d| d.has_tool_calls());
        let body = serde_json::json!({
            "id": ctx.id,
            "model": ctx.model,
            "object": "chat.completion.synthesized",
            "finish_reason": fr,
            "content": state.refusal_scan_buf,
            "has_tool_calls": has_tool_calls,
            "usage": usage_for_dump,
            "stop_string_triggered": state.stop_string_triggered,
            "loop_watchdog_triggered": state.loop_watchdog_triggered,
            "tool_loop_capped": state.tool_loop_capped,
            "_note": "Synthesized from post-sanitizer accumulators; \
                      per-chunk capture is a follow-up.",
        });
        dump.dump_response("/v1/chat/completions", seq, &body, true);
    }

    sse_events
}

#[cfg(test)]
mod finish_reason_tests {
    use super::resolved_finish_reason;

    #[test]
    fn content_loop_is_a_stop_not_a_retryable_token_limit() {
        assert_eq!(resolved_finish_reason("length", false, false, true), "stop");
    }

    #[test]
    fn tool_protocol_reasons_keep_precedence() {
        assert_eq!(resolved_finish_reason("length", true, true, true), "length");
        assert_eq!(
            resolved_finish_reason("length", false, true, true),
            "tool_calls"
        );
    }

    #[test]
    fn ordinary_completion_preserves_scheduler_reason() {
        assert_eq!(
            resolved_finish_reason("length", false, false, false),
            "length"
        );
        assert_eq!(resolved_finish_reason("stop", false, false, false), "stop");
    }
}
