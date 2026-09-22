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
use crate::tokenizer::ChatTokenizer;

use super::chat::prepare::PreparedChat;
use super::compact::openai_error_response_with_param;

#[allow(clippy::result_large_err)]
fn bad_request(message: String, param: Option<&str>) -> Response {
    openai_error_response_with_param(StatusCode::BAD_REQUEST, message, param, None)
}

/// `prepare_chat_prompt` for deepseek_v41: render from the raw OpenAI body.
#[allow(clippy::result_large_err)]
pub(crate) fn prepare(
    state: &Arc<AppState>,
    req: &ir::ChatRequest,
) -> Result<PreparedChat, Response> {
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

/// `POST /v1/debug/prompt` (app.py `_debug_prompt`): render a chat request
/// to the prompt text and ids without generating. deepseek_v41 only.
pub async fn debug_prompt(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    use axum::response::IntoResponse;
    if !state.dsv41 {
        return openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            "/v1/debug/prompt is only served for deepseek_v41".into(),
            None,
            None,
        );
    }
    let body: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return bad_request(format!("request body is not valid JSON: {e}"), None),
    };
    let rendered = match request::render_request(&body) {
        Ok(r) => r,
        Err(e) => return bad_request(e.message, e.param.as_deref()),
    };
    let ids = match state.tokenizer.encode(&rendered.prompt) {
        Ok(ids) => ids,
        Err(e) => return bad_request(format!("Tokenization error: {e}"), None),
    };
    let grammar = rendered.grammar_tools.as_deref().and_then(|t| {
        crate::dsv41::grammar::build_tool_grammar(t, crate::dsv41::grammar::DEFAULT_MAX_CALLS)
    });
    let mut out = serde_json::json!({
        "thinking": rendered.thinking,
        "reasoning_effort": rendered.effort,
        "prompt": rendered.prompt,
        "prompt_tokens": ids.len(),
        "prompt_ids": ids,
        "images": rendered.images.len(),
    });
    if let Some(g) = grammar {
        out["tool_grammar"] = serde_json::Value::String(g);
    }
    axum::Json(out).into_response()
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
                tracing::warn!(
                    "deepseek_v41 tool call {} has non-JSON arguments ({e})",
                    c.name
                );
                serde_json::Value::String(c.arguments.clone())
            });
            ir::message::ToolCall {
                id: tool_call_id(),
                name: c.name,
                arguments,
            }
        })
        .collect()
}

/// `"call_" + uuid4().hex[:24]`, as app.py mints them.
fn tool_call_id() -> String {
    let hex: String = crate::ids::uuid_v4()
        .chars()
        .filter(|c| *c != '-')
        .take(24)
        .collect();
    format!("call_{hex}")
}

/// Blocking choice for deepseek_v41 from the raw output tokens.
pub(crate) fn blocking_choice(
    tok: &ChatTokenizer,
    output_tokens: &[u32],
    finish_reason: &str,
    enable_thinking: bool,
    stop_strings: &[String],
    choice_idx: usize,
) -> ir::Choice {
    let tokens = if finish_reason == "stop" {
        output_tokens
            .split_last()
            .filter(|(last, _)| **last == tok.eos_token_id())
            .map_or(output_tokens, |(_, rest)| rest)
    } else {
        output_tokens
    };
    // app.py decodes with skip_special_tokens=False: `</think>` and the DSML
    // markers are special tokens and the router splits on their text.
    let text = tok.decode_with_special(tokens).unwrap_or_default();
    let mut router = OutputRouter::new(enable_thinking, stop_strings.to_vec(), true);
    router.feed(&text);
    router.finish();
    let calls = router.parse_tool_calls(enable_thinking);
    let mut reason = if router.stopped {
        "stop".to_string()
    } else {
        finish_reason.to_string()
    };
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
        tok: &ChatTokenizer,
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
            eos: tok.eos_token_id(),
            cancel,
        }
    }

    /// Text is released only once decoding the window is stable (does not end
    /// in U+FFFD), re-decoded from `prefix` so byte-fallback pieces join.
    fn detok_push(&mut self, tok: &ChatTokenizer, id: u32) -> String {
        self.ids.push(id);
        let dec = |ids: &[u32]| tok.decode_with_special(ids).unwrap_or_default();
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

    fn detok_flush(&mut self, tok: &ChatTokenizer) -> String {
        let dec = |ids: &[u32]| tok.decode_with_special(ids).unwrap_or_default();
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
                Phase::Reasoning => StreamDelta::Reasoning {
                    text,
                    token_ids: Vec::new(),
                },
                _ => StreamDelta::Content {
                    text,
                    token_ids: Vec::new(),
                },
            })
            .collect()
    }

    pub(crate) fn on_token(&mut self, tok: &ChatTokenizer, id: u32) -> Vec<StreamDelta> {
        if self.router.stopped || id == self.eos {
            return Vec::new();
        }
        let text = self.detok_push(tok, id);
        let out = Self::deltas(self.router.feed(&text));
        if self.router.stopped {
            // A stop string ended the answer: stop generating, not just emitting.
            self.cancel
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        out
    }

    pub(crate) fn on_done(
        &mut self,
        tok: &ChatTokenizer,
        finish_reason: String,
        usage: ir::Usage,
    ) -> Vec<StreamDelta> {
        let mut out = Vec::new();
        if !self.router.stopped {
            let tail = self.detok_flush(tok);
            out.extend(Self::deltas(self.router.feed(&tail)));
        }
        out.extend(Self::deltas(self.router.finish()));
        let held = self.router.tool_text.clone();
        let calls = self.router.parse_tool_calls(self.thinking);
        let mut reason = if self.router.stopped {
            "stop".to_string()
        } else {
            finish_reason
        };
        if calls.is_empty() {
            // Unparseable tool text goes back to content. app.py does that for
            // the blocking reply but its stream never sends it (the text was
            // held back and is silently lost); send it, so stream and blocking
            // agree. A deliberate deviation, recorded in PARITY.md.
            if !held.is_empty() {
                out.push(StreamDelta::Content {
                    text: held,
                    token_ids: Vec::new(),
                });
            }
        } else {
            reason = "tool_calls".into();
            for (index, call) in to_ir_calls(calls).into_iter().enumerate() {
                out.push(StreamDelta::ToolCallStart {
                    index,
                    id: call.id,
                    name: call.name,
                });
                let fragment = match call.arguments {
                    serde_json::Value::String(s) => s,
                    v => v.to_string(),
                };
                out.push(StreamDelta::ToolCallArgs {
                    index,
                    fragment,
                    token_ids: Vec::new(),
                });
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

#[cfg(test)]
mod tests {
    //! The serving seams in the TOKEN domain: completion texts from the Python
    //! fixtures are tokenized with the checkpoint tokenizer, then run through
    //! the blocking path and, token by token, through the streaming path.
    //! Both must reproduce what app.py returned for the same text.

    use super::*;

    fn fixture() -> serde_json::Value {
        let path = format!(
            "{}/tests/fixtures/dsv41/parse.json",
            env!("CARGO_MANIFEST_DIR")
        );
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
    }

    fn tokenizer(fx: &serde_json::Value) -> ChatTokenizer {
        let dir = fx["meta"]["model_dir"].as_str().unwrap();
        ChatTokenizer::from_model_dir(
            std::path::Path::new(dir),
            1,
            true,
            "deepseek_v41",
            None,
            false,
        )
        .expect("checkpoint tokenizer (a missing tokenizer is a failure, not a skip)")
    }

    fn calls(v: &[ir::message::ToolCall]) -> serde_json::Value {
        serde_json::Value::Array(
            v.iter()
                .map(|c| {
                    let args = match &c.arguments {
                        serde_json::Value::String(s) => s.clone(),
                        other => crate::dsv41::pyjson::dumps(other),
                    };
                    serde_json::json!({"name": c.name, "arguments": args})
                })
                .collect(),
        )
    }

    #[test]
    fn blocking_and_streaming_match_python_in_token_domain() {
        let fx = fixture();
        let tok = tokenizer(&fx);
        let mut checked = 0;
        for c in fx["cases"].as_array().unwrap() {
            let name = c["name"].as_str().unwrap();
            let thinking = c["mode"] == "thinking";
            let want = &c["server"];
            let mut ids = tok.encode(c["text"].as_str().unwrap()).unwrap();
            ids.push(1); // EOS, as the scheduler reports a "stop" finish
            let choice = blocking_choice(&tok, &ids, "stop", thinking, &[], 0);
            assert_eq!(
                choice.content.clone().unwrap_or_default(),
                want["content"].as_str().unwrap(),
                "{name}: content"
            );
            if thinking {
                assert_eq!(
                    choice.reasoning.as_deref(),
                    want["reasoning"].as_str(),
                    "{name}: reasoning"
                );
            }
            assert_eq!(
                calls(&choice.tool_calls),
                want["tool_calls"],
                "{name}: calls"
            );
            let want_reason = if choice.tool_calls.is_empty() {
                "stop"
            } else {
                "tool_calls"
            };
            assert_eq!(
                choice.finish_reason,
                ir::FinishReason::from(want_reason),
                "{name}: finish"
            );

            let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let mut st = StreamState::new(&tok, thinking, Vec::new(), flag);
            let mut deltas = Vec::new();
            for &id in &ids {
                deltas.extend(st.on_token(&tok, id));
            }
            deltas.extend(st.on_done(&tok, "stop".into(), ir::Usage::default()));
            let (mut reasoning, mut content, mut names, mut args) =
                (String::new(), String::new(), vec![], vec![]);
            let mut finish = None;
            for d in deltas {
                match d {
                    StreamDelta::Reasoning { text, .. } => reasoning.push_str(&text),
                    StreamDelta::Content { text, .. } => content.push_str(&text),
                    StreamDelta::ToolCallStart { name, .. } => names.push(name),
                    StreamDelta::ToolCallArgs { fragment, .. } => args.push(fragment),
                    StreamDelta::Finish { reason, .. } => finish = Some(reason),
                    other => panic!("{name}: unexpected {other:?}"),
                }
            }
            assert_eq!(
                content,
                want["content"].as_str().unwrap(),
                "{name}: streamed content"
            );
            if thinking {
                assert_eq!(
                    reasoning,
                    want["reasoning"].as_str().unwrap(),
                    "{name}: streamed reasoning"
                );
            }
            let got: Vec<serde_json::Value> = names
                .iter()
                .zip(&args)
                .map(|(n, a)| serde_json::json!({"name": n, "arguments": a}))
                .collect();
            let want_calls: Vec<serde_json::Value> = want["tool_calls"]
                .as_array()
                .unwrap()
                .iter()
                .map(|w| {
                    // the stream sends the IR Value re-serialised compactly
                    let a = w["arguments"].as_str().unwrap();
                    let a = serde_json::from_str::<serde_json::Value>(a)
                        .map_or(a.to_string(), |v| v.to_string());
                    serde_json::json!({"name": w["name"], "arguments": a})
                })
                .collect();
            assert_eq!(got, want_calls, "{name}: streamed calls");
            assert_eq!(
                finish,
                Some(ir::FinishReason::from(want_reason)),
                "{name}: streamed finish"
            );
            checked += 1;
        }
        assert!(checked >= 15);
    }

    /// NEGATIVE CONTROL: the terminal EOS must be cut before decoding. EOS is
    /// the one special token here (`</think>` and `｜DSML｜` are ordinary added
    /// tokens, so skip-special decoding would NOT be caught; checked and
    /// dropped as a control). Keeping EOS leaks its text into content.
    #[test]
    fn keeping_the_eos_is_detected() {
        let fx = fixture();
        let tok = tokenizer(&fx);
        let case = fx["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "content_only")
            .unwrap();
        let mut ids = tok.encode(case["text"].as_str().unwrap()).unwrap();
        ids.push(1);
        // "length" is the path that does not strip a trailing stop token
        let kept = blocking_choice(&tok, &ids, "length", false, &[], 0);
        assert_ne!(kept.content.as_deref(), case["server"]["content"].as_str());
        let cut = blocking_choice(&tok, &ids, "stop", false, &[], 0);
        assert_eq!(cut.content.as_deref(), case["server"]["content"].as_str());
    }
}
