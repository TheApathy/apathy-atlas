// SPDX-License-Identifier: AGPL-3.0-only

//! Port of the checkpoint's own `encoding/encoding.py` (DeepSeek-V4.1), the
//! renderer the production Python server uses. There is no Jinja template for
//! this model: the checkpoint ships none, so falling back to the generic path
//! would render ChatML.
//!
//! Every function here mirrors a named Python function and keeps its
//! behaviour, including the odd corners (`None` formatted as `"None"`, the
//! double JSON decode of tool arguments, dropping non-kept roles before the
//! last user turn). `tests::render_fixtures` compares against strings the real
//! encoding.py produced (scripts/dsv41_parity/gen_fixtures.py).

use serde_json::{Map, Value, json};

use super::pyjson::{dumps, py_str, truthy};

pub const BOS: &str = "<｜begin▁of▁sentence｜>";
pub const EOS: &str = "<｜end▁of▁sentence｜>";
pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";
pub const DSML: &str = "｜DSML｜";
pub const USER_SP: &str = "<｜User｜>";
pub const ASSISTANT_SP: &str = "<｜Assistant｜>";
pub const SYSTEM_SP: &str = "<｜System｜>";
pub const LATEST_REMINDER_SP: &str = "<｜latest_reminder｜>";
pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

/// V4.1 DSML tag names carry a LEADING SPACE (V4 had none).
pub const CALLS_BLOCK: &str = " calls";
pub const INVOKE_TAG: &str = " invoke";
pub const PARAM_TAG: &str = " parameter";

const TASKS: [(&str, &str); 6] = [
    ("action", "<｜action｜>"),
    ("query", "<｜query｜>"),
    ("authority", "<｜authority｜>"),
    ("domain", "<｜domain｜>"),
    ("title", "<｜title｜>"),
    ("read_url", "<｜read_url｜>"),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    Chat,
    Thinking,
}

/// An encoder failure. The Python server maps all of these to HTTP 400.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeError(pub String);

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EncodeError {}

type R<T> = Result<T, EncodeError>;

fn err<T>(msg: impl Into<String>) -> R<T> {
    Err(EncodeError(msg.into()))
}

fn role(m: &Value) -> Option<&str> {
    m.get("role").and_then(Value::as_str)
}

fn obj_mut(v: &mut Value) -> R<&mut Map<String, Value>> {
    v.as_object_mut()
        .ok_or_else(|| EncodeError("message is not an object".into()))
}

// ------------------------------------------------------------------ tool names

/// `_split_tool_name`: `ns::name` -> (Some(ns), name); asserts on conflicts.
pub fn split_tool_name(name: &str, namespace: Option<&str>) -> R<(Option<String>, String)> {
    let (mut ns, mut bare) = (namespace.map(str::to_string), name.to_string());
    if let Some((prefix, rest)) = name.split_once("::") {
        if let Some(n) = namespace
            && n != prefix
        {
            return err(format!("Conflicting tool namespaces: {n} != {prefix}"));
        }
        ns = Some(prefix.to_string());
        bare = rest.to_string();
    }
    if bare.contains("::") {
        return err(format!("Tool name must not contain '::': {bare}"));
    }
    if ns.as_deref().is_some_and(|n| n.contains("::")) {
        return err("Tool namespace must not contain '::'");
    }
    Ok((ns, bare))
}

fn ns_of(v: Option<&Value>) -> R<Option<String>> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(Value::Object(m)) => match m.get("name") {
            Some(Value::String(s)) => Ok(Some(s.clone())),
            _ => err("namespace object has no name"),
        },
        Some(_) => err("namespace must be a string or an object"),
    }
}

/// `_tool_name_for_encoding`.
fn tool_name_for_encoding(tool: &Map<String, Value>) -> R<String> {
    let ns = ns_of(tool.get("namespace"))?;
    let name = tool
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| EncodeError("tool has no name".into()))?;
    let (ns, name) = split_tool_name(name, ns.as_deref())?;
    Ok(match ns {
        None => name,
        Some(ns) => format!("{ns}::{name}"),
    })
}

/// `tools_from_openai_format`: the flattened function dicts, names qualified.
pub fn tools_from_openai_format(tools: &[Value]) -> R<Vec<Value>> {
    let mut out = Vec::with_capacity(tools.len());
    for tool in tools {
        let mut function = match tool.get("function") {
            Some(Value::Object(m)) => m.clone(),
            _ => return err("tool has no `function` object"),
        };
        if let Some(ns) = tool.get("namespace")
            && !ns.is_null()
        {
            function.insert("namespace".into(), ns.clone());
        }
        let name = tool_name_for_encoding(&function)?;
        function.insert("name".into(), Value::String(name));
        let namespace = function.shift_remove("namespace");
        if let Some(Value::Object(nsm)) = &namespace
            && truthy(nsm.get("description"))
        {
            let head = py_str(&nsm["description"]);
            let tail = match function.get("description") {
                Some(d) if truthy(Some(d)) => py_str(d),
                _ => String::new(),
            };
            function.insert(
                "description".into(),
                Value::String(format!("{head}\n{tail}")),
            );
        }
        out.push(Value::Object(function));
    }
    Ok(out)
}

/// `tool_calls_from_openai_format` -> internal `{name, arguments, namespace?}`.
fn tool_calls_from_openai_format(calls: &[Value]) -> R<Vec<Map<String, Value>>> {
    let mut out = Vec::with_capacity(calls.len());
    for tc in calls {
        let function = tc
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| EncodeError("tool call has no `function`".into()))?;
        let hint = match tc.get("namespace") {
            Some(v) if truthy(Some(v)) => Some(v),
            _ => function.get("namespace").filter(|v| truthy(Some(v))),
        };
        let hint = match hint {
            None => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(_) => return err("tool call namespace must be a string"),
        };
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| EncodeError("tool call has no name".into()))?;
        let (ns, name) = split_tool_name(name, hint.as_deref())?;
        let args = function
            .get("arguments")
            .cloned()
            .ok_or_else(|| EncodeError("tool call has no arguments".into()))?;
        let mut call = Map::new();
        call.insert("name".into(), Value::String(name));
        call.insert("arguments".into(), args);
        if let Some(ns) = ns {
            call.insert("namespace".into(), Value::String(ns));
        }
        out.push(call);
    }
    Ok(out)
}

// ------------------------------------------------------------------ rendering pieces

fn render_tools(tools: &[Value]) -> String {
    let schemas: Vec<String> = tools.iter().map(dumps).collect();
    let (d, cb, it, pt) = (DSML, CALLS_BLOCK, INVOKE_TAG, PARAM_TAG);
    format!(
        "## Tools\n\n\
You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<{d}{cb}>\" block like the following:\n\n\
<{d}{cb}>\n\
<{d}{it} name=\"$TOOL_NAME\">\n\
<{d}{pt} name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{d}{pt}>\n\
...\n\
</{d}{it}>\n\
<{d}{it} name=\"$TOOL_NAME2\">\n\
...\n\
</{d}{it}>\n\
</{d}{cb}>\n\n\
String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n\
If thinking_mode is enabled (triggered by {THINK_START}), you MUST output your complete reasoning inside {THINK_START}...{THINK_END} BEFORE any tool calls or final response.\n\n\
Otherwise, output directly after {THINK_END} with tool calls or final response.\n\n\
### Available Tool Schemas\n\n\
{}\n\n\
You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n",
        schemas.join("\n")
    )
}

/// `encode_arguments_to_dsml`.
fn encode_arguments_to_dsml(call: &Map<String, Value>) -> String {
    let original = call.get("arguments").cloned().unwrap_or(Value::Null);
    let mut arguments = original.clone();
    if !arguments.is_object() {
        // Tolerate JSON strings, including double-encoded ones.
        for _ in 0..2 {
            match &arguments {
                Value::String(s) => match serde_json::from_str::<Value>(s) {
                    Ok(v) => arguments = v,
                    Err(_) => break,
                },
                _ => break,
            }
        }
        if !arguments.is_object() {
            arguments = json!({ "arguments": original });
        }
    }
    let mut parts = Vec::new();
    for (k, v) in arguments.as_object().into_iter().flatten() {
        let (is_str, value) = match v {
            Value::String(s) => ("true", s.clone()),
            other => ("false", dumps(other)),
        };
        parts.push(format!(
            "<{DSML}{PARAM_TAG} name=\"{k}\" string=\"{is_str}\">{value}</{DSML}{PARAM_TAG}>"
        ));
    }
    parts.join("\n")
}

/// `find_last_user_index`: mid-conversation system messages count as users.
fn find_last_user_index(messages: &[Value]) -> isize {
    for idx in (0..messages.len()).rev() {
        match role(&messages[idx]) {
            Some("user") => return idx as isize,
            Some("system") if idx > 0 => return idx as isize,
            _ => {}
        }
    }
    -1
}

fn text_or_empty(v: Option<&Value>) -> String {
    // `x or ""`, then str.format: falsy -> "", anything else -> str(x).
    if truthy(v) {
        py_str(v.unwrap_or(&Value::Null))
    } else {
        String::new()
    }
}

fn render_message(
    index: usize,
    messages: &[Value],
    mode: ThinkingMode,
    drop_thinking: bool,
    effort: u32,
) -> R<String> {
    let msg = &messages[index];
    let last_user_idx = find_last_user_index(messages);
    let thinking = mode == ThinkingMode::Thinking;
    let r = role(msg);
    let tools = match msg.get("tools") {
        Some(Value::Array(a)) if !a.is_empty() => Some(tools_from_openai_format(a)?),
        Some(v) if truthy(Some(v)) => return err("`tools` must be a list"),
        _ => None,
    };
    let tool_calls = match msg.get("tool_calls") {
        Some(Value::Array(a)) if !a.is_empty() => Some(tool_calls_from_openai_format(a)?),
        Some(v) if truthy(Some(v)) => return err("`tool_calls` must be a list"),
        _ => None,
    };

    let effort_prompt = if index == 0 && thinking {
        format!(
            "Reasoning Effort: {effort} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
        )
    } else {
        String::new()
    };
    let mut prompt = String::new();
    if index == 0 && (!effort_prompt.is_empty() || r == Some("system")) {
        prompt.push_str(SYSTEM_SP);
    }
    prompt.push_str(&effort_prompt);

    match r {
        Some("system") => {
            if index > 0 {
                prompt.push_str(SYSTEM_SP);
            }
            prompt.push_str(&text_or_empty(msg.get("content")));
            if let Some(tools) = &tools {
                prompt.push_str("\n\n");
                prompt.push_str(&render_tools(tools));
            }
            if truthy(msg.get("response_format")) {
                prompt.push_str(
                    "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n",
                );
                prompt.push_str(&dumps(&msg["response_format"]));
            }
        }
        Some("user") => {
            prompt.push_str(USER_SP);
            match msg.get("content_blocks") {
                Some(Value::Array(blocks)) if !blocks.is_empty() => {
                    let mut parts = Vec::with_capacity(blocks.len());
                    for block in blocks {
                        parts.push(render_user_block(block)?);
                    }
                    prompt.push_str(&parts.join("\n\n"));
                }
                _ => prompt.push_str(&text_or_empty(msg.get("content"))),
            }
        }
        Some("latest_reminder") => {
            prompt.push_str(LATEST_REMINDER_SP);
            prompt.push_str(&py_str(msg.get("content").unwrap_or(&Value::Null)));
        }
        Some("tool") => {
            return err("deepseek_v41 merges tool messages into user; please preprocess");
        }
        Some("assistant") => {
            let mut tc_content = String::new();
            if let Some(calls) = &tool_calls {
                let mut list = Vec::with_capacity(calls.len());
                for tc in calls {
                    let name = tool_name_for_encoding(tc)?;
                    list.push(format!(
                        "<{DSML}{INVOKE_TAG} name=\"{name}\">\n{}\n</{DSML}{INVOKE_TAG}>",
                        encode_arguments_to_dsml(tc)
                    ));
                }
                tc_content = format!(
                    "\n\n<{DSML}{CALLS_BLOCK}>\n{}\n</{DSML}{CALLS_BLOCK}>",
                    list.join("\n")
                );
            }
            let summary = text_or_empty(msg.get("content"));
            let rc = text_or_empty(msg.get("reasoning_content"));
            let prev_has_task = index >= 1
                && messages[index - 1]
                    .get("task")
                    .is_some_and(|t| !t.is_null());
            let mut thinking_part = String::new();
            if thinking && !prev_has_task && (!drop_thinking || index as isize > last_user_idx) {
                thinking_part = format!("{rc}{THINK_END}");
            }
            prompt.push_str(&thinking_part);
            prompt.push_str(&summary);
            prompt.push_str(&tc_content);
            if !truthy(msg.get("wo_eos")) {
                prompt.push_str(EOS);
            }
        }
        Some(other) => return err(format!("Unknown role: {other}")),
        None => return err("message has no role"),
    }

    // Transition tokens depend on what follows.
    if let Some(next) = messages.get(index + 1)
        && !matches!(role(next), Some("assistant") | Some("latest_reminder"))
    {
        return Ok(prompt);
    }
    match msg.get("task") {
        Some(t) if !t.is_null() => {
            let tok = t
                .as_str()
                .and_then(|t| TASKS.iter().find(|(k, _)| *k == t))
                .map(|(_, v)| *v)
                .ok_or_else(|| EncodeError(format!("Invalid task: {t}")))?;
            if t.as_str() != Some("action") {
                prompt.push_str(tok);
            } else {
                prompt.push_str(ASSISTANT_SP);
                prompt.push_str(if thinking { THINK_START } else { THINK_END });
                prompt.push_str(tok);
            }
        }
        _ if r == Some("user") || (r == Some("system") && index > 0) => {
            prompt.push_str(ASSISTANT_SP);
            let open = thinking && (!drop_thinking || index as isize >= last_user_idx);
            prompt.push_str(if open { THINK_START } else { THINK_END });
        }
        _ => {}
    }
    Ok(prompt)
}

fn render_user_block(block: &Value) -> R<String> {
    let Some(b) = block.as_object() else {
        return err("content block is not an object");
    };
    let ty = b.get("type").cloned().unwrap_or(Value::Null);
    Ok(match ty.as_str() {
        Some("text") => py_str(b.get("text").unwrap_or(&Value::String(String::new()))),
        Some("tool_result") => {
            let content = b
                .get("content")
                .cloned()
                .unwrap_or(Value::String(String::new()));
            let text = match &content {
                Value::Array(items) => {
                    let mut parts = Vec::with_capacity(items.len());
                    for item in items {
                        let Some(it) = item.as_object() else {
                            return err("tool_result content item is not an object");
                        };
                        if it.get("type").and_then(Value::as_str) == Some("text") {
                            parts.push(py_str(
                                it.get("text").unwrap_or(&Value::String(String::new())),
                            ));
                        } else {
                            parts.push(format!(
                                "[Unsupported {}]",
                                py_str(it.get("type").unwrap_or(&Value::Null))
                            ));
                        }
                    }
                    parts.join("\n\n")
                }
                other => py_str(other),
            };
            format!("<tool_result>{text}</tool_result>")
        }
        _ => format!("[Unsupported {}]", py_str(&ty)),
    })
}

// ------------------------------------------------------------------ preprocessing

/// `merge_tool_messages`.
fn merge_tool_messages(messages: Vec<Value>) -> R<Vec<Value>> {
    let mut merged: Vec<Value> = Vec::with_capacity(messages.len());
    for mut msg in messages {
        match role(&msg) {
            Some("tool") => {
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id").cloned().unwrap_or(Value::String(String::new())),
                    "content": msg.get("content").cloned().unwrap_or(Value::String(String::new())),
                });
                if let Some(last) = merged.last_mut()
                    && role(last) == Some("user")
                    && let Some(Value::Array(blocks)) = last.get_mut("content_blocks")
                {
                    blocks.push(block);
                } else {
                    merged.push(json!({"role": "user", "content_blocks": [block]}));
                }
            }
            Some("user") => {
                let blocks = match msg.get("content_blocks") {
                    None | Some(Value::Null) => {
                        vec![
                            json!({"type": "text", "text": msg.get("content").cloned().unwrap_or(Value::String(String::new()))}),
                        ]
                    }
                    Some(Value::Array(a)) => a.clone(),
                    Some(_) => return err("content_blocks must be a list"),
                };
                if let Some(last) = merged.last_mut()
                    && role(last) == Some("user")
                    && last.get("task").is_none_or(Value::is_null)
                    && let Some(Value::Array(prev)) = last.get_mut("content_blocks")
                {
                    prev.extend(blocks);
                } else {
                    obj_mut(&mut msg)?.insert("content_blocks".into(), Value::Array(blocks));
                    merged.push(msg);
                }
            }
            _ => merged.push(msg),
        }
    }
    Ok(merged)
}

/// `sort_tool_results_by_call_order`.
fn sort_tool_results_by_call_order(messages: &mut [Value]) {
    let mut order: Vec<(String, usize)> = Vec::new();
    for msg in messages.iter_mut() {
        match role(msg) {
            Some("assistant") if truthy(msg.get("tool_calls")) => {
                order.clear();
                if let Some(Value::Array(calls)) = msg.get("tool_calls") {
                    for (idx, tc) in calls.iter().enumerate() {
                        let id = match tc.get("id") {
                            Some(v) if truthy(Some(v)) => py_str(v),
                            _ => tc
                                .get("function")
                                .and_then(|f| f.get("id"))
                                .map(py_str)
                                .unwrap_or_default(),
                        };
                        if !id.is_empty() {
                            // dict assignment: a repeated id keeps its slot, takes the new index
                            match order.iter_mut().find(|(k, _)| *k == id) {
                                Some(slot) => slot.1 = idx,
                                None => order.push((id, idx)),
                            }
                        }
                    }
                }
            }
            Some("user") if truthy(msg.get("content_blocks")) => {
                let Some(Value::Array(blocks)) = msg.get_mut("content_blocks") else {
                    continue;
                };
                let is_tr =
                    |b: &Value| b.get("type").and_then(Value::as_str) == Some("tool_result");
                let mut tool_blocks: Vec<Value> =
                    blocks.iter().filter(|b| is_tr(b)).cloned().collect();
                if tool_blocks.len() > 1 && !order.is_empty() {
                    let key = |b: &Value| {
                        let id = b.get("tool_use_id").map(py_str).unwrap_or_default();
                        order.iter().find(|(k, _)| *k == id).map_or(0, |(_, i)| *i)
                    };
                    tool_blocks.sort_by_key(key); // stable, like Python's sorted
                    let mut it = tool_blocks.into_iter();
                    for b in blocks.iter_mut() {
                        if is_tr(b)
                            && let Some(next) = it.next()
                        {
                            *b = next;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// `_drop_thinking_messages`.
fn drop_thinking_messages(messages: Vec<Value>) -> Vec<Value> {
    let last_user_idx = find_last_user_index(&messages);
    const KEEP: [&str; 5] = [
        "user",
        "system",
        "tool",
        "latest_reminder",
        "direct_search_results",
    ];
    let mut out = Vec::with_capacity(messages.len());
    for (idx, mut msg) in messages.into_iter().enumerate() {
        let r = role(&msg);
        if r.is_some_and(|r| KEEP.contains(&r)) || idx as isize >= last_user_idx {
            out.push(msg);
        } else if r == Some("assistant") {
            if let Some(m) = msg.as_object_mut() {
                m.shift_remove("reasoning_content");
            }
            out.push(msg);
        }
    }
    out
}

// ------------------------------------------------------------------ images

/// One image record in prompt order (`_extract_image`): exactly one of the
/// source keys the request carried (`url` for OpenAI `image_url`).
pub type ImageRecord = Map<String, Value>;

fn is_image_block(b: &Value) -> bool {
    matches!(
        b.get("type").and_then(Value::as_str),
        Some("image") | Some("image_url")
    )
}

fn extract_image(block: &Value) -> R<ImageRecord> {
    let mut rec = Map::new();
    rec.insert("type".into(), "image".into());
    if block.get("type").and_then(Value::as_str) == Some("image_url") {
        let url = match block.get("image_url") {
            Some(Value::String(s)) => Value::String(s.clone()),
            Some(Value::Object(m)) => m
                .get("url")
                .cloned()
                .unwrap_or(Value::String(String::new())),
            _ => Value::String(String::new()),
        };
        rec.insert("url".into(), url);
    } else {
        for key in ["source", "url", "data"] {
            if let Some(v) = block.get(key) {
                rec.insert(key.into(), v.clone());
            }
        }
    }
    if !["source", "url", "data"]
        .iter()
        .any(|k| truthy(rec.get(*k)))
    {
        return err("Image block does not contain a valid source");
    }
    Ok(rec)
}

fn process_image_blocks(blocks: &[Value], images: &mut Vec<ImageRecord>) -> R<Vec<Value>> {
    let mut out = Vec::with_capacity(blocks.len());
    for block in blocks {
        if !block.is_object() {
            out.push(block.clone());
        } else if is_image_block(block) {
            out.push(json!({"type": "text", "text": IMAGE_PLACEHOLDER}));
            images.push(extract_image(block)?);
        } else if block.get("type").and_then(Value::as_str) == Some("tool_result")
            && let Some(Value::Array(inner)) = block.get("content")
        {
            let mut b = block.clone();
            b["content"] = Value::Array(process_image_blocks(inner, images)?);
            out.push(b);
        } else if block.get("type").and_then(Value::as_str) == Some("text") {
            if let Some(Value::String(t)) = block.get("text")
                && t.contains(IMAGE_PLACEHOLDER)
            {
                return err(format!(
                    "Text block contains image placeholder '{IMAGE_PLACEHOLDER}'. Images should be separate content blocks."
                ));
            }
            out.push(block.clone());
        } else {
            out.push(block.clone());
        }
    }
    Ok(out)
}

/// `process_image_messages`.
fn process_image_messages(messages: &[Value]) -> R<(Vec<Value>, Vec<ImageRecord>)> {
    let mut processed = Vec::with_capacity(messages.len());
    let mut images = Vec::new();
    for msg in messages {
        let mut msg = msg.clone();
        for key in ["content", "reasoning_content"] {
            if let Some(Value::String(s)) = msg.get(key)
                && s.contains(IMAGE_PLACEHOLDER)
            {
                return err(format!(
                    "{key} contains image special token '{IMAGE_PLACEHOLDER}'"
                ));
            }
        }
        let m = obj_mut(&mut msg)?;
        if matches!(m.get("content"), Some(Value::Array(_))) && !m.contains_key("content_blocks") {
            let c = m.shift_remove("content").unwrap_or(Value::Null);
            m.insert("content_blocks".into(), c);
        }
        if truthy(m.get("content_blocks")) {
            let Some(Value::Array(blocks)) = m.get("content_blocks") else {
                return err("content_blocks must be a list");
            };
            let new_blocks = process_image_blocks(blocks, &mut images)?;
            if !matches!(m.get("content"), Some(Value::String(_))) {
                let texts: Vec<String> = new_blocks
                    .iter()
                    .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
                    .map(|b| py_str(b.get("text").unwrap_or(&Value::String(String::new()))))
                    .collect();
                m.insert("content".into(), Value::String(texts.join("\n\n")));
            }
            m.insert("content_blocks".into(), Value::Array(new_blocks));
        }
        processed.push(msg);
    }
    Ok((processed, images))
}

// ------------------------------------------------------------------ entry point

/// `encode_messages(messages, thinking_mode, reasoning_effort=effort,
/// return_multi_modal_data=True)` with the defaults the server uses: no
/// context, `drop_thinking=True`, BOS prepended.
pub fn encode_messages(
    messages: &[Value],
    mode: ThinkingMode,
    effort: u32,
) -> R<(String, Vec<ImageRecord>)> {
    if !(1..=100).contains(&effort) {
        return err(format!(
            "Invalid reasoning effort for deepseek_v41: {effort}"
        ));
    }
    let (processed, images) = process_image_messages(messages)?;
    let mut msgs = merge_tool_messages(processed)?;
    sort_tool_results_by_call_order(&mut msgs);
    let drop = !msgs.iter().any(|m| truthy(m.get("tools")));
    if mode == ThinkingMode::Thinking && drop {
        msgs = drop_thinking_messages(msgs);
    }
    let mut prompt = String::from(BOS);
    for idx in 0..msgs.len() {
        prompt.push_str(&render_message(idx, &msgs, mode, drop, effort)?);
    }
    Ok((prompt, images))
}
