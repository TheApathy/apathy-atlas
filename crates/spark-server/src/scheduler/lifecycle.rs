// SPDX-License-Identifier: AGPL-3.0-only

//! Sequence lifecycle: finish, errors, swap-out, resume.

use super::*;

fn finish_reason(
    output_tokens: &[u32],
    eos_tokens: &[u32],
    tool_call_end_token: Option<u32>,
) -> &'static str {
    let last_tok = output_tokens.last().copied();
    let is_eos = last_tok.is_some_and(|t| eos_tokens.contains(&t));
    let is_chatml_role_boundary = last_tok.is_some_and(|token| im_start_hard_stop() == Some(token));
    let is_tool_call_end = last_tok.is_some_and(|token| tool_call_end_token == Some(token));
    if is_eos || is_chatml_role_boundary {
        "stop"
    } else if is_tool_call_end {
        "tool_calls"
    } else {
        "length"
    }
}

/// Send final response and free GPU resources for a completed sequence.
pub fn finish_sequence(model: &dyn Model, a: &mut ActiveSeq) {
    finish_sequence_with_cache(model, a, true);
}

/// Finish a sequence while explicitly controlling final prefix-cache admission.
///
/// Speculative verification may stop while only a prefix of an over-planned
/// verify frame has been emitted. Until that frame has an exact recurrent/KV
/// commit boundary, its `seq.tokens` must never become a reusable prefix.
pub(super) fn finish_sequence_with_cache(
    model: &dyn Model,
    a: &mut ActiveSeq,
    cache_sequence: bool,
) {
    let reason = finish_reason(&a.output_tokens, &a.eos_tokens, a.tool_call_end_token);
    match &mut a.sink {
        ResponseSink::Streaming(tx) => {
            let ttft_ms = a.decode_start.duration_since(a.request_start).as_secs_f64() * 1000.0;
            let decode_ms = a.decode_start.elapsed().as_secs_f64() * 1000.0;
            if let Err(e) = tx.blocking_send(StreamEvent::Done {
                finish_reason: reason.to_string(),
                prompt_tokens: 0, // prompt_tokens tracked by API layer
                completion_tokens: a.output_tokens.len(),
                time_to_first_token_ms: ttft_ms,
                decode_time_ms: decode_ms,
                reasoning_tokens: a.thinking_tokens,
                cached_prompt_tokens: a.cached_prompt_tokens,
            }) {
                tracing::warn!(
                    "finish_sequence: streaming Done send failed (receiver dropped): {e}"
                );
            }
        }
        ResponseSink::Blocking(tx) => {
            if let Some(tx) = tx.take() {
                let ttft_ms = a.decode_start.duration_since(a.request_start).as_secs_f64() * 1000.0;
                let decode_ms = a.decode_start.elapsed().as_secs_f64() * 1000.0;
                if tx
                    .send(Ok(InferenceResponse {
                        output_tokens: a.output_tokens.clone(),
                        finish_reason: reason.to_string(),
                        time_to_first_token_ms: ttft_ms,
                        decode_time_ms: decode_ms,
                        logprobs: std::mem::take(&mut a.logprobs_data),
                        reasoning_tokens: a.thinking_tokens,
                        cached_prompt_tokens: a.cached_prompt_tokens,
                    }))
                    .is_err()
                {
                    tracing::warn!(
                        "finish_sequence: blocking response send failed (receiver dropped)"
                    );
                }
            }
        }
    }
    let decode_s = a.decode_start.elapsed().as_secs_f64();
    let n = a.output_tokens.len();
    let tps = if decode_s > 0.0 {
        n as f64 / decode_s
    } else {
        0.0
    };
    let ttft_ms = a.decode_start.duration_since(a.request_start).as_secs_f64() * 1000.0;
    tracing::info!("Done: {n} tokens ({reason}) {tps:.1} tok/s, TTFT={ttft_ms:.1}ms");
    // ATLAS_FULL_PROFILE=1: dump per-kernel timing report after every
    // generation. Zero overhead when env var is unset.
    if spark_model::full_profile::is_enabled() {
        spark_model::full_profile::dump();
    }
    // Cache the full sequence (prompt + generated) in the prefix cache.
    // Must happen BEFORE free_sequence() so block indices are still valid.
    // Enables multi-turn sessions to reuse KV cache for prior assistant responses.
    if cache_sequence {
        model.cache_sequence(&a.seq);
    }
    if let Err(e) = model.free_sequence(&mut a.seq) {
        tracing::error!("free_sequence: {e:#}");
    }
    // EP: signal worker to free+realloc its mirrored sequence.
    if let Err(e) = model.ep_broadcast_cmd(0xFFFFFFF1) {
        tracing::error!("EP broadcast free+realloc: {e:#}");
    }
}

/// Send error to client and free GPU resources.
pub fn send_error(model: &dyn Model, a: &mut ActiveSeq, msg: &str) {
    match &mut a.sink {
        ResponseSink::Streaming(tx) => {
            if let Err(e) = tx.blocking_send(StreamEvent::Error(msg.to_string())) {
                tracing::warn!("send_error: streaming Error send failed (receiver dropped): {e}");
            }
        }
        ResponseSink::Blocking(tx) => {
            if let Some(tx) = tx.take()
                && tx.send(Err(anyhow::anyhow!("{msg}"))).is_err()
            {
                tracing::warn!("send_error: blocking Error send failed (receiver dropped)");
            }
        }
    }
    if let Err(e) = model.free_sequence(&mut a.seq) {
        tracing::error!("send_error: free_sequence: {e:#}");
    }
    if let Err(e) = model.ep_broadcast_cmd(0xFFFFFFF1) {
        tracing::error!("send_error: ep_broadcast free+realloc: {e:#}");
    }
}

/// Send an error directly to a ResponseSink that hasn't been attached
/// to an ActiveSeq yet. Used by prefill_request when it fails AFTER
/// extracting the sink from the InferenceRequest but BEFORE building
/// an ActiveSeq. Without this the sender is silently dropped, producing
/// a misleading "Inference cancelled" error on the client side.
pub fn send_error_to_sink(sink: &mut ResponseSink, msg: &str) {
    match sink {
        ResponseSink::Streaming(tx) => {
            if let Err(e) = tx.blocking_send(StreamEvent::Error(msg.to_string())) {
                tracing::warn!(
                    "send_error_to_sink: streaming Error send failed (receiver dropped): {e}"
                );
            }
        }
        ResponseSink::Blocking(tx) => {
            if let Some(tx) = tx.take()
                && tx.send(Err(anyhow::anyhow!("{msg}"))).is_err()
            {
                tracing::warn!("send_error_to_sink: blocking Error send failed (receiver dropped)");
            }
        }
    }
}

#[path = "swap_lifecycle.rs"]
mod swap_lifecycle;
pub(super) use swap_lifecycle::{resume_swapped_seq, swap_out_sequence};

#[cfg(test)]
#[path = "lifecycle_tests.rs"]
mod tests;
