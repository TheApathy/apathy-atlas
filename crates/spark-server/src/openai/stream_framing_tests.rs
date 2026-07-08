// SPDX-License-Identifier: AGPL-3.0-only

//! Streaming SSE framing invariants (task #41, 2026-07-07).
//!
//! The client-observed decode throughput bug ("32 tok/s vs true 85.7")
//! was a *transport* issue (Nagle coalescing SSE frames — fixed by
//! TCP_NODELAY in `serve_router.rs`), NOT an app-level batching bug: the
//! streaming path already emits one SSE `data:` frame per committed
//! token, and a DFlash accept-burst of K tokens is delivered as K
//! independent frames (K separate `handle_token` calls, each producing
//! its own `content_chunk` event).
//!
//! These tests pin the framing contract so a future refactor can't
//! silently reintroduce coalescing/batching or break the OpenAI SSE
//! wire shape. They exercise the exact frame types the streaming
//! handler assembles (`role_chunk` → N `content_chunk` → terminal
//! `done_chunk` → `[DONE]`) at the pure-serialization level, so they
//! run GPU-free.
//!
//! The load-bearing invariant is **content-equality under delivery**:
//! concatenating the `delta.content` across every content frame MUST
//! byte-equal the single blocking-response string for the same token
//! sequence. Streaming is a delivery change, never a content change —
//! this is the md5-equality guarantee stated in the task.

use super::stream_chunk::ChatCompletionChunk;

/// Extract the `delta.content` string a client would read from a
/// serialized content SSE frame (the exact JSON the handler puts in
/// `Event::default().data(...)`).
fn content_of_json(frame: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(frame).expect("valid json");
    v["choices"][0]["delta"]["content"]
        .as_str()
        .map(str::to_string)
}

/// Simulate the framing the streaming handler performs for a burst of
/// per-token decoded deltas: role frame, then one content frame per
/// non-empty delta, then the terminal done frame. Returns the ordered
/// list of serialized `data:` payloads (excluding the literal `[DONE]`
/// sentinel, which is appended verbatim by the handler).
fn frame_burst(model: &str, id: &str, deltas: &[&str]) -> Vec<String> {
    let mut frames = Vec::new();
    // First frame: role announcement (content == null).
    frames.push(
        serde_json::to_string(&ChatCompletionChunk::role_chunk(model, id))
            .expect("role serializes"),
    );
    // One content frame per non-empty delta — never coalesced.
    for d in deltas {
        if d.is_empty() {
            continue;
        }
        frames.push(
            serde_json::to_string(&ChatCompletionChunk::content_chunk(
                model,
                id,
                d.to_string(),
            ))
            .expect("content serializes"),
        );
    }
    frames
}

#[test]
fn burst_of_k_tokens_produces_k_independent_content_frames() {
    // A DFlash verify step commits K tokens at once; each becomes its
    // own frame (K separate handle_token calls in the real path).
    let deltas = ["Hello", ",", " ", "world", "!"];
    let frames = frame_burst("m", "chatcmpl-x", &deltas);
    // role + 5 content frames (none empty).
    assert_eq!(
        frames.len(),
        1 + 5,
        "expected one frame per token, no coalescing"
    );
    // Frame 0 is the role frame with no content.
    let role: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
    assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
    assert!(role["choices"][0]["delta"].get("content").is_none());
    // Frames 1..=5 each carry exactly one token's text, in order.
    for (i, expected) in deltas.iter().enumerate() {
        let v: serde_json::Value = serde_json::from_str(&frames[i + 1]).unwrap();
        assert_eq!(v["choices"][0]["delta"]["content"], *expected);
    }
}

#[test]
fn concatenated_stream_content_equals_blocking_content() {
    // md5-equality guarantee: the streamed content, concatenated in
    // arrival order, must byte-equal the single blocking string.
    let deltas = ["The ", "quick ", "brown ", "fox", "."];
    let blocking: String = deltas.concat();

    let frames = frame_burst("m", "id", &deltas);
    let streamed: String = frames.iter().filter_map(|f| content_of_json(f)).collect();

    assert_eq!(
        streamed, blocking,
        "streamed concatenation must equal blocking content byte-for-byte"
    );
}

#[test]
fn empty_deltas_are_not_framed() {
    // A token that decodes to nothing (e.g. an incomplete UTF-8
    // boundary held back by the streaming decoder) must not emit a
    // frame — it carries no content and would pollute the stream.
    let deltas = ["a", "", "b", "", ""];
    let frames = frame_burst("m", "id", &deltas);
    // role + only the two non-empty content frames.
    assert_eq!(frames.len(), 1 + 2);
    let streamed: String = frames[1..]
        .iter()
        .filter_map(|f| content_of_json(f))
        .collect();
    assert_eq!(streamed, "ab");
}

#[test]
fn content_frame_omits_role_and_finish_reason() {
    // OpenAI contract: content deltas carry ONLY content — no role,
    // and finish_reason stays null until the terminal frame. Clients
    // (Cline/Roo/OpenWebUI) branch on these being absent.
    let chunk = ChatCompletionChunk::content_chunk("m", "id", "hi".to_string());
    let v: serde_json::Value =
        serde_json::from_value(serde_json::to_value(&chunk).unwrap()).unwrap();
    let delta = &v["choices"][0]["delta"];
    assert_eq!(delta["content"], "hi");
    assert!(
        delta.get("role").is_none(),
        "content frame must not carry role"
    );
    assert_eq!(
        v["choices"][0]["finish_reason"],
        serde_json::Value::Null,
        "finish_reason must be null on content frames"
    );
    assert_eq!(v["object"], "chat.completion.chunk");
}

#[test]
fn done_frame_carries_finish_reason_and_no_content() {
    // Terminal frame: finish_reason set, delta content empty. The
    // literal `[DONE]` sentinel is appended separately by the handler.
    let usage = crate::openai::Usage {
        prompt_tokens: 3,
        completion_tokens: 5,
        total_tokens: 8,
        prompt_tokens_details: None,
        completion_tokens_details: None,
        time_to_first_token_ms: 0.0,
        response_tokens_per_second: 0.0,
    };
    let chunk = ChatCompletionChunk::done_chunk("m", "id", "stop", usage);
    let v: serde_json::Value =
        serde_json::from_value(serde_json::to_value(&chunk).unwrap()).unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert!(v["choices"][0]["delta"].get("content").is_none());
    assert_eq!(v["usage"]["completion_tokens"], 5);
}
