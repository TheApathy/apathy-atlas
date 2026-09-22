// SPDX-License-Identifier: AGPL-3.0-only

//! The production server's request -> prompt layer (`server/app.py`):
//! `resolve_thinking` and `build_chat_prompt`, over the raw request body.
//!
//! These semantics differ from Atlas's generic thinking resolution and are
//! part of the V4.1 API contract. Thinking defaults OFF, effort defaults to
//! 75. The named efforts are low=50, medium=60, high=75, xhigh=90, max=100,
//! and "low" and "none" turn thinking OFF. An explicit thinking flag beats the
//! flag an effort implies.

use serde_json::Value;

use super::encoding::{self, ImageRecord, ThinkingMode};
use super::pyjson::truthy;

pub const DEFAULT_EFFORT: u32 = 75;
pub const DEFAULT_THINKING: bool = false;

const EFFORT_ALIASES: [(&str, u32); 5] = [
    ("low", 50),
    ("medium", 60),
    ("high", 75),
    ("xhigh", 90),
    ("max", 100),
];
const NO_THINKING_EFFORTS: [&str; 2] = ["none", "low"];

/// A client error. The Python server renders these as HTTP 400.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestError {
    pub message: String,
    pub param: Option<String>,
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RequestError {}

fn bad<T>(message: impl Into<String>, param: &str) -> Result<T, RequestError> {
    Err(RequestError {
        message: message.into(),
        param: Some(param.to_string()),
    })
}

fn get<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.get(key).filter(|x| !x.is_null())
}

/// `resolve_thinking(body, default_thinking, default_effort)` -> (thinking, effort).
pub fn resolve_thinking(
    body: &Value,
    default_thinking: bool,
    default_effort: u32,
) -> Result<(bool, u32), RequestError> {
    let empty = Value::Object(Default::default());
    let ctk = match body.get("chat_template_kwargs") {
        Some(v) if truthy(Some(v)) => {
            if !v.is_object() {
                return bad(
                    "`chat_template_kwargs` must be an object",
                    "chat_template_kwargs",
                );
            }
            v
        }
        _ => &empty,
    };

    let mut thinking: Option<bool> = None;
    for (src, key) in [
        (ctk, "thinking"),
        (body, "enable_thinking"),
        (ctk, "enable_thinking"),
    ] {
        if let Some(v) = get(src, key) {
            match v.as_bool() {
                Some(b) => thinking = Some(b),
                None => return bad(format!("`{key}` must be a boolean"), key),
            }
            break;
        }
    }

    let raw = match body.get("reasoning") {
        Some(Value::Object(r)) if r.get("effort").is_some_and(|e| !e.is_null()) => r.get("effort"),
        _ => get(body, "reasoning_effort").or_else(|| get(ctk, "reasoning_effort")),
    };

    let mut effort: Option<u32> = None;
    let mut effort_thinking: Option<bool> = None;
    if let Some(raw) = raw {
        const P: &str = "reasoning_effort";
        let as_int = match raw {
            Value::Bool(_) => {
                return bad("reasoning effort must be a string or an integer 1-100", P);
            }
            Value::Number(n) => match n.as_i64() {
                Some(i) => Some(i),
                None => return bad("reasoning effort must be a string or an integer 1-100", P),
            },
            // `raw.strip().lstrip("-").isdigit()` -> int(raw.strip())
            Value::String(s) => {
                let t = s.trim();
                let digits = t.trim_start_matches('-');
                if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
                    match t.parse::<i64>() {
                        Ok(i) => Some(i),
                        Err(_) => return bad("integer reasoning effort must be within 1-100", P),
                    }
                } else {
                    None
                }
            }
            _ => return bad("reasoning effort must be a string or an integer 1-100", P),
        };
        match (as_int, raw) {
            (Some(i), _) => {
                if !(1..=100).contains(&i) {
                    return bad("integer reasoning effort must be within 1-100", P);
                }
                effort = Some(i as u32);
                effort_thinking = Some(true);
            }
            (None, Value::String(s)) => {
                let name = s.trim().to_lowercase();
                if name == "none" {
                    effort_thinking = Some(false);
                } else if let Some((_, v)) = EFFORT_ALIASES.iter().find(|(k, _)| *k == name) {
                    effort = Some(*v);
                    effort_thinking = Some(!NO_THINKING_EFFORTS.contains(&name.as_str()));
                } else {
                    return bad(
                        format!(
                            "unknown reasoning effort {s:?}; use none/low/medium/high/xhigh/max or 1-100"
                        ),
                        P,
                    );
                }
            }
            _ => unreachable!("non-string non-int efforts returned above"),
        }
    }

    Ok((
        thinking.or(effort_thinking).unwrap_or(default_thinking),
        effort.unwrap_or(default_effort),
    ))
}

/// What `build_chat_prompt` returns, minus the token ids (the caller encodes
/// `prompt` with `add_special_tokens=false`; BOS is already in the string).
#[derive(Debug, Clone)]
pub struct ChatPrompt {
    pub prompt: String,
    /// Image records in prompt order (one per `<｜deepseek_image｜>`).
    pub images: Vec<ImageRecord>,
    /// `tools_from_openai_format(tools)`: the names the prompt taught the
    /// model, which is what a tool grammar must accept. None without tools.
    pub grammar_tools: Option<Vec<Value>>,
    pub thinking: bool,
    pub effort: u32,
}

/// `build_chat_prompt(body, enc, tok, thinking, effort)`, text part.
pub fn build_chat_prompt(
    body: &Value,
    thinking: bool,
    effort: u32,
) -> Result<ChatPrompt, RequestError> {
    let Some(Value::Array(messages)) = body.get("messages").filter(|m| truthy(Some(m))) else {
        return bad("`messages` must be a non-empty list", "messages");
    };
    for (i, m) in messages.iter().enumerate() {
        if !m.get("role").is_some_and(Value::is_string) {
            return bad(
                format!("messages[{i}] must be an object with a `role`"),
                &format!("messages[{i}]"),
            );
        }
    }
    let mut messages = messages.clone();

    let mut tools = body.get("tools").cloned().filter(|t| !t.is_null());
    if body.get("tool_choice").and_then(Value::as_str) == Some("none") {
        tools = None;
    }
    if tools.as_ref().is_some_and(|t| !t.is_array()) {
        return bad("`tools` must be a list", "tools");
    }
    let schema = match body.get("response_format") {
        Some(rf) if rf.get("type").and_then(Value::as_str) == Some("json_schema") => rf
            .get("json_schema")
            .filter(|j| truthy(Some(j)))
            .and_then(|j| j.get("schema"))
            .cloned(),
        _ => None,
    };
    let has_tools = truthy(tools.as_ref());
    let has_schema = truthy(schema.as_ref());
    if has_tools || has_schema {
        if messages[0].get("role").and_then(Value::as_str) != Some("system") {
            messages.insert(0, serde_json::json!({"role": "system", "content": ""}));
        }
        let first = messages[0].as_object_mut().expect("validated above");
        if has_tools {
            first.insert("tools".into(), tools.clone().unwrap_or(Value::Null));
        }
        if has_schema {
            first.insert(
                "response_format".into(),
                schema.clone().unwrap_or(Value::Null),
            );
        }
    }

    let mode = if thinking {
        ThinkingMode::Thinking
    } else {
        ThinkingMode::Chat
    };
    let (prompt, images) =
        encoding::encode_messages(&messages, mode, effort).map_err(|e| RequestError {
            message: format!("cannot encode messages: {e}"),
            param: Some("messages".into()),
        })?;
    let grammar_tools = match &tools {
        Some(Value::Array(t)) if !t.is_empty() => encoding::tools_from_openai_format(t).ok(),
        _ => None,
    };
    Ok(ChatPrompt {
        prompt,
        images,
        grammar_tools,
        thinking,
        effort,
    })
}

/// Both steps, as `_chat` runs them.
pub fn render_request(body: &Value) -> Result<ChatPrompt, RequestError> {
    let (thinking, effort) = resolve_thinking(body, DEFAULT_THINKING, DEFAULT_EFFORT)?;
    build_chat_prompt(body, thinking, effort)
}
