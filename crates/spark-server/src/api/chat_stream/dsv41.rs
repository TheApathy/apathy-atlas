// SPDX-License-Identifier: AGPL-3.0-only
//
// deepseek_v41: the Python server's stream shape (api/dsv41.rs) on the shared
// scheduler events, encoded as this server's chat-completion chunks.

use axum::response::sse::Event;

use crate::api::dsv41::{Delta, StreamState};
use crate::openai::{ChatCompletionChunk, Usage};

use super::ctx::StreamCtx;

type SseVec = Vec<Result<Event, std::convert::Infallible>>;

fn event(chunk: &ChatCompletionChunk) -> Result<Event, std::convert::Infallible> {
    Ok(Event::default().data(serde_json::to_string(chunk).unwrap_or_default()))
}

/// Wire chunks for one piece of the reply. `Finish` is handled by [`done`].
fn delta_events(ctx: &StreamCtx, delta: Delta, out: &mut SseVec) {
    match delta {
        Delta::Reasoning(text) => {
            let mut chunk = ChatCompletionChunk::reasoning_chunk(&ctx.model, &ctx.id, text);
            // app.py streams `reasoning_content` only.
            for c in &mut chunk.choices {
                c.delta.reasoning = None;
            }
            out.push(event(&chunk));
        }
        Delta::Content(text) => {
            out.push(event(&ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, text)));
        }
        Delta::ToolCall { index, call } => {
            let start = ChatCompletionChunk::tool_call_start_chunk(&ctx.model, &ctx.id, &call, index);
            out.push(event(&start));
            let args = ChatCompletionChunk::tool_call_args_fragment(
                &ctx.model,
                &ctx.id,
                index,
                &call.function.arguments,
            );
            out.push(event(&args));
        }
        Delta::Finish(_) => {}
    }
}

pub(super) fn on_token(st: &mut StreamState, ctx: &StreamCtx, token: u32) -> SseVec {
    let mut out = SseVec::new();
    for d in st.on_token(&ctx.state.tokenizer, token) {
        delta_events(ctx, d, &mut out);
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub(super) fn on_done(
    st: &mut StreamState,
    ctx: &StreamCtx,
    finish_reason: String,
    completion_tokens: usize,
    time_to_first_token_ms: f64,
    decode_time_ms: f64,
    reasoning_tokens: u32,
    cached_prompt_tokens: u32,
) -> SseVec {
    let mut out = SseVec::new();
    let mut reason = finish_reason;
    for d in st.on_done(&ctx.state.tokenizer, reason.clone()) {
        if let Delta::Finish(r) = &d {
            reason = r.clone();
        }
        delta_events(ctx, d, &mut out);
    }

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
    // Same framing as the generic stream (handle_done.rs): usage rides the
    // finish chunk, or its own chunk first when the client asked for it.
    if ctx.req_stream_include_usage {
        out.push(event(&ChatCompletionChunk::usage_only_chunk(
            &ctx.model,
            &ctx.id,
            usage.clone(),
        )));
        out.push(event(&ChatCompletionChunk::final_chunk_no_usage(
            &ctx.model, &ctx.id, &reason,
        )));
    } else {
        out.push(event(&ChatCompletionChunk::done_chunk(
            &ctx.model,
            &ctx.id,
            &reason,
            usage.clone(),
        )));
    }

    crate::metrics::REQUESTS_ACTIVE.dec();
    crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(ctx.prompt_len as u64);
    crate::metrics::GENERATION_TOKENS_TOTAL.inc_by(completion_tokens as u64);
    crate::metrics::TTFT_SECONDS.observe(time_to_first_token_ms / 1000.0);
    if let Some(ref rctx) = ctx.req_ctx {
        let actual = (ctx.prompt_len + completion_tokens) as u64;
        let refund = rctx.reserved_tokens.saturating_sub(actual);
        if refund > 0 {
            ctx.state.rate_limiter.refund_tokens(&rctx.identity, refund);
        }
    }
    if let (Some(seq), Some(dump)) = (ctx.dump_seq, ctx.state.dump_writer.as_ref()) {
        let body = serde_json::json!({
            "id": ctx.id,
            "model": ctx.model,
            "object": "chat.completion.synthesized",
            "finish_reason": reason,
            "usage": usage,
        });
        dump.dump_response("/v1/chat/completions", seq, &body, true);
    }
    out
}
