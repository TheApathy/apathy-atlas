// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1 serving seams. When the loaded model is `deepseek_v41`
// (`AppState::dsv41`), three phases of the chat pipeline switch to the
// production Python server's semantics, ported in `crate::dsv41`:
//
//   * prompt:    the raw request body goes through app.py's resolve_thinking +
//                build_chat_prompt and the checkpoint's encoding.py. The
//                generic MsgEntry/Jinja path would render ChatML, because the
//                checkpoint ships no chat template.
//   * blocking:  the output is routed and parsed exactly as app.py does it
//                (strict DSML parser, then the tolerant one, then raw text).
//   * streaming: reasoning/content deltas come out of the same OutputRouter
//                and the tool calls go out once, at the end, as in app.py.
//
// Everything that is not prompt or output shaping (sampling, scheduling,
// stop tokens, usage) stays on the shared path.

use axum::http::StatusCode;
use axum::response::Response;
use std::sync::Arc;

use crate::AppState;
use crate::dsv41::parse::{OutputRouter, ParsedCall, Phase};
use crate::dsv41::request;
use crate::ir::{self, StreamDelta};

use super::chat::prepare::PreparedChat;
use super::compact::openai_error_response_with_param;

#[allow(clippy::result_large_err)]
fn bad_request(message: String, param: Option<&str>) -> Response {
    openai_error_response_with_param(StatusCode::BAD_REQUEST, message, param, None)
}

/// `prepare_chat_prompt` for deepseek_v41: render from the raw OpenAI body.
#[allow(clippy::result_large_err)]
pub(crate) fn prepare(state: &Arc<AppState>, req: &ir::ChatRequest) -> Result<PreparedChat, Response> {
    let Some(body) = req.raw_openai_body.as_deref() else {
        // The Anthropic and Responses adapters lower their own wire format and
        // never carry an OpenAI body. The Python server has neither surface.
        return Err(bad_request(
            "deepseek_v41 is served on /v1/chat/completions and /v1/completions only".into(),
            None,
        ));
    };
    let rendered = request::render_request(body)
        .map_err(|e| bad_request(e.message.clone(), e.param.as_deref()))?;
    if !rendered.images.is_empty() {
        // The V4.1 vision tower and placeholder expansion are not ported yet
        // (DSV41_PORT/PARITY.md section 4). Refuse rather than feed the model
        // bare sentinel tokens with no image embedding behind them.
        return Err(bad_request(
            "image input for deepseek_v41 is not supported on this engine yet".into(),
            Some("messages"),
        ));
    }
    let prompt_tokens = state
        .tokenizer
        .encode(&rendered.prompt)
        .map_err(|e| bad_request(format!("Tokenization error: {e}"), None))?;
    tracing::info!(
        "deepseek_v41 prompt: {} tokens, thinking={} effort={} tools={}",
        prompt_tokens.len(),
        rendered.thinking,
        rendered.effort,
        rendered.grammar_tools.as_ref().map_or(0, Vec::len),
    );
    Ok(PreparedChat {
        // Tools reach the prompt through the renderer, so "active" only
        // means a parser is configured and the request carries tools.
        tools_active: state.tool_call_parser.is_some() && rendered.grammar_tools.is_some(),
        cwd_hint: None,
        image_pixels: Vec::new(),
        prompt_tokens,
        enable_thinking: rendered.thinking,
        // The Python engine has no thinking budget: reasoning may use the
        // whole max_tokens. `None` keeps the forced-</think> injector disarmed.
        thinking_budget: None,
    })
}

fn to_ir_calls(calls: Vec<ParsedCall>) -> Vec<ir::message::ToolCall> {
    calls
        .into_iter()
        .map(|c| {
            let arguments = serde_json::from_str(&c.arguments).unwrap_or_else(|e| {
                // The strict parser concatenates `string="false"` values
                // verbatim, so malformed JSON can reach here exactly as it
                // reaches a Python client. The IR carries a Value; keep the
                // text rather than invent an empty object.
                tracing::warn!("deepseek_v41 tool call {} has non-JSON arguments ({e})", c.name);
                serde_json::Value::String(c.arguments.clone())
            });
            ir::message::ToolCall { id: tool_call_id(), name: c.name, arguments }
        })
        .collect()
}

/// `"call_" + uuid4().hex[:24]`, as app.py mints them.
fn tool_call_id() -> String {
    let hex: String = crate::ids::uuid_v4().chars().filter(|c| *c != '-').take(24).collect();
    format!("call_{hex}")
}

/// Blocking choice for deepseek_v41 from the raw output tokens.
pub(crate) fn blocking_choice(
    state: &AppState,
    output_tokens: &[u32],
    finish_reason: &str,
    enable_thinking: bool,
    stop_strings: &[String],
    choice_idx: usize,
) -> ir::Choice {
    let tokens = if finish_reason == "stop" {
        output_tokens
            .split_last()
            .filter(|(last, _)| **last == state.tokenizer.eos_token_id())
            .map_or(output_tokens, |(_, rest)| rest)
    } else {
        output_tokens
    };
    // app.py decodes with skip_special_tokens=False: `</think>` and the DSML
    // markers are special tokens and the router splits on their text.
    let text = state.tokenizer.decode_with_special(tokens).unwrap_or_default();
    let mut router = OutputRouter::new(enable_thinking, stop_strings.to_vec(), true);
    router.feed(&text);
    router.finish();
    let calls = router.parse_tool_calls(enable_thinking);
    let mut reason = if router.stopped { "stop".to_string() } else { finish_reason.to_string() };
    let content = if !calls.is_empty() {
        reason = "tool_calls".into();
        Some(router.content.clone()).filter(|c| !c.is_empty())
    } else {
        Some(router.content.clone())
    };
    ir::Choice {
        index: choice_idx,
        content,
        reasoning: enable_thinking.then(|| router.reasoning.clone()),
        tool_calls: to_ir_calls(calls),
        refusal: None,
        finish_reason: ir::FinishReason::from(reason.as_str()),
        matched_stop: None,
        logprobs: None,
    }
}

/// Streaming state: app.py's IncrementalDetokenizer + OutputRouter.
pub(crate) struct StreamState {
    router: OutputRouter,
    thinking: bool,
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
    eos: u32,
    cancel: Arc<std::sync::atomic::AtomicBool>,
}

impl StreamState {
    pub(crate) fn new(
        state: &AppState,
        thinking: bool,
        stop_strings: Vec<String>,
        cancel: Arc<std::sync::atomic::AtomicBool>,
    ) -> Self {
        Self {
            router: OutputRouter::new(thinking, stop_strings, true),
            thinking,
            ids: Vec::new(),
            prefix: 0,
            read: 0,
            eos: state.tokenizer.eos_token_id(),
            cancel,
        }
    }

    /// Text is released only once decoding the window is stable (does not end
    /// in U+FFFD), re-decoded from `prefix` so byte-fallback pieces join.
    fn detok_push(&mut self, state: &AppState, tok: u32) -> String {
        self.ids.push(tok);
        let dec = |ids: &[u32]| state.tokenizer.decode_with_special(ids).unwrap_or_default();
        let full = dec(&self.ids[self.prefix..]);
        let prev = dec(&self.ids[self.prefix..self.read]);
        if full.len() > prev.len() && !full.ends_with('\u{FFFD}') {
            let out = full.get(prev.len()..).unwrap_or_default().to_string();
            self.prefix = self.read;
            self.read = self.ids.len();
            return out;
        }
        String::new()
    }

    fn detok_flush(&mut self, state: &AppState) -> String {
        let dec = |ids: &[u32]| state.tokenizer.decode_with_special(ids).unwrap_or_default();
        let full = dec(&self.ids[self.prefix..]);
        let prev = dec(&self.ids[self.prefix..self.read]);
        self.prefix = self.ids.len();
        self.read = self.ids.len();
        full.get(prev.len()..).unwrap_or_default().to_string()
    }

    fn deltas(events: Vec<(Phase, String)>) -> Vec<StreamDelta> {
        events
            .into_iter()
            .map(|(phase, text)| match phase {
                Phase::Reasoning => StreamDelta::Reasoning { text, token_ids: Vec::new() },
                _ => StreamDelta::Content { text, token_ids: Vec::new() },
            })
            .collect()
    }

    pub(crate) fn on_token(&mut self, state: &AppState, tok: u32) -> Vec<StreamDelta> {
        if self.router.stopped || tok == self.eos {
            return Vec::new();
        }
        let text = self.detok_push(state, tok);
        let out = Self::deltas(self.router.feed(&text));
        if self.router.stopped {
            // A stop string ended the answer: stop generating, not just emitting.
            self.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        out
    }

    pub(crate) fn on_done(
        &mut self,
        state: &AppState,
        finish_reason: String,
        usage: ir::Usage,
    ) -> Vec<StreamDelta> {
        let mut out = Vec::new();
        if !self.router.stopped {
            let tail = self.detok_flush(state);
            out.extend(Self::deltas(self.router.feed(&tail)));
        }
        out.extend(Self::deltas(self.router.finish()));
        let held = self.router.tool_text.clone();
        let calls = self.router.parse_tool_calls(self.thinking);
        let mut reason = if self.router.stopped { "stop".to_string() } else { finish_reason };
        if calls.is_empty() {
            // Unparseable tool text goes back to content. app.py does that for
            // the blocking reply but its stream never sends it (the text was
            // held back and is silently lost); send it, so stream and blocking
            // agree. A deliberate deviation, recorded in PARITY.md.
            if !held.is_empty() {
                out.push(StreamDelta::Content { text: held, token_ids: Vec::new() });
            }
        } else {
            reason = "tool_calls".into();
            for (index, call) in to_ir_calls(calls).into_iter().enumerate() {
                out.push(StreamDelta::ToolCallStart { index, id: call.id, name: call.name });
                let fragment = match call.arguments {
                    serde_json::Value::String(s) => s,
                    v => v.to_string(),
                };
                out.push(StreamDelta::ToolCallArgs { index, fragment, token_ids: Vec::new() });
            }
        }
        out.push(StreamDelta::Finish {
            reason: ir::FinishReason::from(reason.as_str()),
            usage,
            token_ids: Vec::new(),
        });
        out
    }
}
