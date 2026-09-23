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
use crate::openai::{ChatCompletionRequest, ChatMessage};
use crate::tokenizer::ChatTokenizer;
use crate::tool_parser::{FunctionCall, ToolCall};

use super::chat::PreparedChat;
use super::compact::openai_error_response_with_param;

#[allow(clippy::result_large_err)]
fn bad_request(message: String, param: Option<&str>) -> Response {
    openai_error_response_with_param(StatusCode::BAD_REQUEST, message, param, None)
}

/// app.py's engine headroom beyond prompt + max_tokens (`context_margin`).
const CONTEXT_MARGIN: usize = 8;
/// app.py budgets every image at this many tokens, whatever its real span.
const TOKENS_PER_IMAGE: usize = 1024;

/// app.py's context admission: `prompt_ids` is the rendered prompt BEFORE
/// image expansion. Returns the max_tokens ceiling app.py clamps to, or its
/// error message (both are 400 `context_length_exceeded`).
fn context_budget(prompt_len: usize, images: usize, max_ctx: usize) -> Result<usize, String> {
    let bound = prompt_len + TOKENS_PER_IMAGE * images;
    if images > 0 && bound + CONTEXT_MARGIN >= max_ctx {
        return Err("image-expanded prompt exceeds the engine context".into());
    }
    if prompt_len + CONTEXT_MARGIN >= max_ctx {
        return Err(format!(
            "prompt has {prompt_len} tokens, engine context is {max_ctx}"
        ));
    }
    Ok(max_ctx - bound - CONTEXT_MARGIN)
}

/// The chat prompt for deepseek_v41, rendered from the raw OpenAI body, and
/// the max_tokens ceiling app.py applies for it.
#[allow(clippy::result_large_err)]
pub(crate) fn prepare(
    state: &Arc<AppState>,
    req: &ChatCompletionRequest,
) -> Result<(PreparedChat, usize), Response> {
    let Some(body) = req.raw_body.as_deref() else {
        // The Anthropic and Responses adapters build their own request and
        // never carry an OpenAI body. The Python server has neither surface.
        return Err(bad_request(
            "deepseek_v41 is served on /v1/chat/completions and /v1/completions only".into(),
            None,
        ));
    };
    let rendered = request::render_request(body)
        .map_err(|e| bad_request(e.message.clone(), e.param.as_deref()))?;
    let text_ids = state
        .tokenizer
        .encode(&rendered.prompt)
        .map_err(|e| bad_request(format!("Tokenization error: {e}"), None))?;
    let max_tokens_cap = context_budget(text_ids.len(), rendered.images.len(), state.max_seq_len)
        .map_err(|message| {
            openai_error_response_with_param(
                StatusCode::BAD_REQUEST,
                message,
                None,
                Some("context_length_exceeded"),
            )
        })?;
    let (prompt_tokens, image_pixels) = if rendered.images.is_empty() {
        (text_ids, Vec::new())
    } else {
        expand_images(state, &rendered.images, &text_ids)?
    };
    tracing::info!(
        "deepseek_v41 prompt: {} tokens ({} image(s)), thinking={} effort={} tools={}",
        prompt_tokens.len(),
        image_pixels.len(),
        rendered.thinking,
        rendered.effort,
        rendered.grammar_tools.as_ref().map_or(0, Vec::len),
    );
    let prepared = PreparedChat {
        // Tools reach the prompt through the renderer, so "active" only
        // means a parser is configured and the request carries tools.
        tools_active: state.tool_call_parser.is_some() && rendered.grammar_tools.is_some(),
        cwd_hint: None,
        image_pixels,
        prompt_tokens,
        enable_thinking: rendered.thinking,
        // The Python engine has no thinking budget: reasoning may use the
        // whole max_tokens. `None` keeps the forced-</think> injector disarmed.
        thinking_budget: None,
    };
    Ok((prepared, max_tokens_cap))
}

/// Opt-in switch for `/v1/debug/prompt`: only an explicit `1`/`true` enables it.
fn debug_prompt_enabled(env: Option<&str>) -> bool {
    matches!(env.map(str::trim), Some("1") | Some("true"))
}

/// `POST /v1/debug/prompt` (app.py `_debug_prompt`). Behind the normal
/// `/v1/` auth gate, and disabled unless `ATLAS_DSV41_DEBUG_PROMPT=1`. Renders a chat request
/// to the prompt text and ids without generating. deepseek_v41 only.
pub async fn debug_prompt(
    crate::main_modules::model_host::CurrentModel(state): crate::main_modules::model_host::CurrentModel,
    body: axum::body::Bytes,
) -> Response {
    use axum::response::IntoResponse;
    if !debug_prompt_enabled(std::env::var("ATLAS_DSV41_DEBUG_PROMPT").ok().as_deref()) {
        // It echoes the rendered prompt, system prompt and tool schemas
        // included: off unless the operator opts in.
        return openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            "/v1/debug/prompt is disabled; set ATLAS_DSV41_DEBUG_PROMPT=1 to enable".into(),
            None,
            None,
        );
    }
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

/// Images: decode + preprocess (vision.py-exact), then expand each
/// `<｜deepseek_image｜>` into PAD + sentinel span ids. Returns the expanded ids
/// and one `(bf16 patches [vit_h*vit_w, 3,14,14], vit_h, vit_w)` per image, in
/// prompt order, for `Model::prepare_vision_embed`.
#[allow(clippy::result_large_err, clippy::type_complexity)]
fn expand_images(
    state: &Arc<AppState>,
    records: &[crate::dsv41::encoding::ImageRecord],
    text_ids: &[u32],
) -> Result<(Vec<u32>, Vec<(Vec<f32>, usize, usize)>), Response> {
    use crate::dsv41::vision;
    let Some(cfg) = state.dsv41_vision.as_ref() else {
        return Err(bad_request(
            "this checkpoint has no vision_config; image input is not available".into(),
            Some("messages"),
        ));
    };
    let mut lens = Vec::with_capacity(records.len());
    let mut pixels = Vec::with_capacity(records.len());
    for rec in records {
        let img = vision::decode_image_record(&serde_json::Value::Object(rec.clone()))
            .map_err(|e| bad_request(e.to_string(), Some("messages")))?;
        let prep = vision::preprocess_image(&img, cfg)
            .map_err(|e| bad_request(e.to_string(), Some("messages")))?;
        lens.push(prep.types.len());
        pixels.push((prep.patches, prep.plan.vit_h, prep.plan.vit_w));
    }
    let (ids, _spans) = vision::expand_image_placeholders(text_ids, &lens)
        .map_err(|e| bad_request(e.to_string(), Some("messages")))?;
    Ok((ids, pixels))
}

/// app.py's call objects: the arguments string exactly as the parser produced
/// it (the strict parser can yield non-JSON text, which a Python client
/// receives verbatim too).
fn to_tool_calls(calls: Vec<ParsedCall>) -> Vec<ToolCall> {
    calls
        .into_iter()
        .map(|c| ToolCall {
            id: tool_call_id(),
            call_type: "function".into(),
            function: FunctionCall {
                name: c.name,
                arguments: c.arguments,
            },
        })
        .collect()
}

/// `"call_" + uuid4().hex[:24]`, as app.py mints them.
fn tool_call_id() -> String {
    let hex: String = crate::openai::uuid_v4()
        .chars()
        .filter(|c| *c != '-')
        .take(24)
        .collect();
    format!("call_{hex}")
}

/// One blocking choice's message and finish reason, from the raw output tokens.
pub(crate) fn blocking_choice(
    tok: &ChatTokenizer,
    output_tokens: &[u32],
    finish_reason: &str,
    enable_thinking: bool,
    stop_strings: &[String],
) -> (ChatMessage, String) {
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
    let message = ChatMessage {
        role: "assistant".into(),
        // app.py sends `reasoning_content` only, and only in thinking mode.
        reasoning_content: enable_thinking.then(|| router.reasoning.clone()),
        reasoning: None,
        content,
        tool_calls: (!calls.is_empty()).then(|| to_tool_calls(calls)),
        annotations: None,
        refusal: None,
    };
    (message, reason)
}

/// One streamed piece of a deepseek_v41 reply, before wire encoding.
#[derive(Debug, Clone)]
pub(crate) enum Delta {
    Reasoning(String),
    Content(String),
    /// A complete call (app.py sends each call whole, once, at the end).
    ToolCall { index: usize, call: ToolCall },
    Finish(String),
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

    fn deltas(events: Vec<(Phase, String)>) -> Vec<Delta> {
        events
            .into_iter()
            .map(|(phase, text)| match phase {
                Phase::Reasoning => Delta::Reasoning(text),
                _ => Delta::Content(text),
            })
            .collect()
    }

    pub(crate) fn on_token(&mut self, tok: &ChatTokenizer, id: u32) -> Vec<Delta> {
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

    pub(crate) fn on_done(&mut self, tok: &ChatTokenizer, finish_reason: String) -> Vec<Delta> {
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
                out.push(Delta::Content(held));
            }
        } else {
            reason = "tool_calls".into();
            out.extend(
                to_tool_calls(calls)
                    .into_iter()
                    .enumerate()
                    .map(|(index, call)| Delta::ToolCall { index, call }),
            );
        }
        out.push(Delta::Finish(reason));
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
        ChatTokenizer::from_model_dir(std::path::Path::new(dir), 1, true, "deepseek_v41", None)
            .expect("checkpoint tokenizer (a missing tokenizer is a failure, not a skip)")
    }

    fn calls(v: &[ToolCall]) -> serde_json::Value {
        serde_json::Value::Array(
            v.iter()
                .map(|c| serde_json::json!({"name": c.function.name, "arguments": c.function.arguments}))
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
            let (message, reason) = blocking_choice(&tok, &ids, "stop", thinking, &[]);
            let got_calls = message.tool_calls.clone().unwrap_or_default();
            assert_eq!(
                message.content.clone().unwrap_or_default(),
                want["content"].as_str().unwrap(),
                "{name}: content"
            );
            if thinking {
                assert_eq!(
                    message.reasoning_content.as_deref(),
                    want["reasoning"].as_str(),
                    "{name}: reasoning"
                );
            } else {
                assert_eq!(message.reasoning_content, None, "{name}: no reasoning field");
            }
            assert_eq!(calls(&got_calls), want["tool_calls"], "{name}: calls");
            let want_reason = if got_calls.is_empty() {
                "stop"
            } else {
                "tool_calls"
            };
            assert_eq!(reason, want_reason, "{name}: finish");

            let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let mut st = StreamState::new(&tok, thinking, Vec::new(), flag);
            let mut deltas = Vec::new();
            for &id in &ids {
                deltas.extend(st.on_token(&tok, id));
            }
            deltas.extend(st.on_done(&tok, "stop".into()));
            let (mut reasoning, mut content, mut streamed) = (String::new(), String::new(), vec![]);
            let mut finish = None;
            for d in deltas {
                match d {
                    Delta::Reasoning(text) => reasoning.push_str(&text),
                    Delta::Content(text) => content.push_str(&text),
                    Delta::ToolCall { index, call } => {
                        assert_eq!(index, streamed.len(), "{name}: call index");
                        streamed.push(call);
                    }
                    Delta::Finish(reason) => finish = Some(reason),
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
            // The arguments go out exactly as app.py sends them.
            assert_eq!(calls(&streamed), want["tool_calls"], "{name}: streamed calls");
            assert_eq!(finish.as_deref(), Some(want_reason), "{name}: streamed finish");
            checked += 1;
        }
        assert!(checked >= 15);
    }

    /// app.py rejects when prompt + 1024/image + 8 >= max_context, and clamps
    /// max_tokens to the rest. Boundary on both sides, text and images.
    #[test]
    fn context_budget_matches_app_py_boundary() {
        let max = 10_000;
        // text only: the last admitted prompt leaves exactly 1 token
        assert_eq!(context_budget(max - 9, 0, max), Ok(1));
        assert!(context_budget(max - 8, 0, max).is_err());
        assert!(context_budget(max - 1, 0, max).is_err());
        assert!(context_budget(max, 0, max).is_err());
        assert!(context_budget(max + 1, 0, max).is_err());
        assert_eq!(context_budget(100, 0, max), Ok(max - 108));
        // one image counts 1024 whatever its size
        assert_eq!(context_budget(max - 1033, 1, max), Ok(1));
        assert_eq!(
            context_budget(max - 1032, 1, max),
            Err("image-expanded prompt exceeds the engine context".into())
        );
        assert!(context_budget(max - 1024 - 1, 1, max).is_err());
        assert!(context_budget(max - 1024, 1, max).is_err());
        assert!(context_budget(max - 1024 + 1, 1, max).is_err());
        assert_eq!(
            context_budget(max + 1, 0, max),
            Err(format!("prompt has {} tokens, engine context is {max}", max + 1))
        );
    }

    #[test]
    fn tool_call_ids_look_like_apps() {
        let id = tool_call_id();
        assert_eq!(id.len(), "call_".len() + 24, "{id}");
        assert!(id.starts_with("call_"));
        assert!(id[5..].chars().all(|c| c.is_ascii_hexdigit()), "{id}");
    }

    #[test]
    fn debug_prompt_is_disabled_by_default() {
        assert!(!debug_prompt_enabled(None));
        assert!(!debug_prompt_enabled(Some("")));
        assert!(!debug_prompt_enabled(Some("0")));
        assert!(!debug_prompt_enabled(Some("yes")));
        assert!(debug_prompt_enabled(Some("1")));
        assert!(debug_prompt_enabled(Some("true")));
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
        let (kept, _) = blocking_choice(&tok, &ids, "length", false, &[]);
        assert_ne!(kept.content.as_deref(), case["server"]["content"].as_str());
        let (cut, _) = blocking_choice(&tok, &ids, "stop", false, &[]);
        assert_eq!(cut.content.as_deref(), case["server"]["content"].as_str());
    }
}
