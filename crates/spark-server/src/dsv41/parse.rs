// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 completion parsing, as the production server does it:
//!
//! 1. `OutputRouter` (app.py) splits the stream into reasoning, content and the
//!    tool text that starts at `"\n\n<｜DSML｜ calls"`, and handles stop strings.
//! 2. The checkpoint's STRICT parser (`parse_message_from_completion_text`)
//!    runs on the re-assembled text.
//! 3. If it raises, the server's TOLERANT regex pass
//!    (`_parse_tool_calls_tolerant`) runs. If that finds nothing too, the tool
//!    text goes back into content.
//!
//! The Python regexes use a lookahead that the `regex` crate cannot express,
//! so they are hand-rolled here with the same leftmost/lazy semantics.

use serde_json::{Map, Value};

use super::encoding::{
    BOS, CALLS_BLOCK, DSML, EOS, INVOKE_TAG, PARAM_TAG, THINK_END, THINK_START, split_tool_name,
};
use super::pyjson::dumps;

/// Where the server starts treating output as a tool-calls block, and where
/// the grammar engages.
pub const TOOL_CALLS_MARKER: &str = "\n\n<｜DSML｜ calls";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedCall {
    /// `ns::name` when the call was namespaced (the server re-joins it).
    pub name: String,
    /// JSON text. From the strict parser this is built by string
    /// concatenation, so a malformed `string="false"` value passes through raw.
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrictMessage {
    pub content: String,
    pub reasoning_content: String,
    pub tool_calls: Vec<StrictCall>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrictCall {
    pub name: String,
    pub namespace: Option<String>,
    pub arguments: String,
}

// ------------------------------------------------------------------ strict

/// `_read_until_stop`: earliest stop wins; on a tie the earlier-listed one.
fn read_until_stop<'a>(
    index: usize,
    text: &'a str,
    stops: &[&'a str],
) -> (usize, &'a str, Option<&'a str>) {
    let mut best: Option<(usize, &str)> = None;
    for s in stops {
        if let Some(p) = text[index..].find(s) {
            let p = index + p;
            if best.is_none_or(|(bp, _)| p < bp) {
                best = Some((p, s));
            }
        }
    }
    match best {
        Some((p, s)) => (p + s.len(), &text[index..p], Some(s)),
        None => (text.len(), &text[index..], None),
    }
}

/// `^\s*name="(.*?)">\n$` (DOTALL, and `re.findall` needs exactly one match).
fn match_tool_name(s: &str) -> Option<&str> {
    let rest = s.trim_start_matches(is_py_space).strip_prefix("name=\"")?;
    let inner = rest
        .strip_suffix("\">\n")
        .or_else(|| rest.strip_suffix("\">\n\n"))?;
    // `.*?` is lazy but anchored at both ends, so any inner text matches;
    // it is the unique match unless the inner text itself ends the pattern
    // earlier, which the `$` anchor rules out. Python `$` also matches before
    // a trailing newline, which `>\n` already consumes.
    Some(inner)
}

/// `^ name="(.*?)" string="(true|false)">(.*?)<$` (DOTALL). The name is lazy:
/// the FIRST `" string="true|false">` wins. The value runs to the final `<`
/// (Python `$` also matches before one trailing newline).
fn match_param(s: &str) -> Option<(&str, &str, &str)> {
    let rest = s.strip_prefix(" name=\"")?;
    let mut from = 0;
    while let Some(p) = rest[from..].find("\" string=\"") {
        let q = from + p;
        let after = &rest[q + "\" string=\"".len()..];
        for flag in ["true", "false"] {
            if let Some(tail) = after.strip_prefix(flag).and_then(|t| t.strip_prefix("\">"))
                && let Some(value) = tail.strip_suffix('<').or_else(|| tail.strip_suffix("<\n"))
            {
                return Some((&rest[..q], flag, value));
            }
        }
        from = q + 1;
    }
    None
}

fn parse_tool_calls_strict(
    mut index: usize,
    text: &str,
) -> Result<(usize, Option<String>, Vec<StrictCall>), String> {
    let calls_end = format!("</{DSML}{CALLS_BLOCK}>");
    let call_start = format!("<{DSML}{INVOKE_TAG}");
    let call_end = format!("</{DSML}{INVOKE_TAG}");
    let param_start = format!("<{DSML}{PARAM_TAG}");
    let param_end = format!("/{DSML}{PARAM_TAG}");
    let mut calls = Vec::new();
    let mut stop: Option<String> = None;
    while index < text.len() {
        let (i, gap, s) = read_until_stop(index, text, &[&call_start, &calls_end]);
        index = i;
        if gap != ">\n" {
            return Err(format!(
                "Tool call format error: expected '>\\n' but got '{gap}'"
            ));
        }
        stop = s.map(str::to_string);
        if s == Some(calls_end.as_str()) {
            break;
        }
        if s.is_none() {
            return Err("Missing special token in tool calls".into());
        }
        let (i, name_part, s) = read_until_stop(index, text, &[&param_start, &call_end]);
        index = i;
        let mut s = s.map(str::to_string);
        let name = match_tool_name(name_part)
            .ok_or_else(|| format!("Tool name format error: '{name_part}'"))?;
        let mut args: Vec<(String, String, String)> = Vec::new();
        while s.as_deref() == Some(param_start.as_str()) {
            let (i, pc, _) = read_until_stop(index, text, &[&param_end]);
            index = i;
            let (k, flag, v) =
                match_param(pc).ok_or_else(|| format!("Parameter format error: '{pc}'"))?;
            if args.iter().any(|(ek, _, _)| ek == k) {
                return Err(format!("Duplicate parameter name: '{k}'"));
            }
            args.push((k.to_string(), v.to_string(), flag.to_string()));
            let (i, gap, s2) = read_until_stop(index, text, &[&param_start, &call_end]);
            index = i;
            if gap != ">\n" {
                return Err(format!(
                    "Parameter format error: expected '>\\n' but got '{gap}'"
                ));
            }
            s = s2.map(str::to_string);
        }
        // decode_dsml_to_arguments: string concatenation, non-strings verbatim.
        let body: Vec<String> = args
            .iter()
            .map(|(k, v, flag)| {
                let value = if flag == "true" {
                    dumps(&Value::String(v.clone()))
                } else {
                    v.clone()
                };
                format!("{}: {value}", dumps(&Value::String(k.clone())))
            })
            .collect();
        let (namespace, bare) = split_tool_name(name, None).map_err(|e| e.0)?;
        calls.push(StrictCall {
            name: bare,
            namespace,
            arguments: format!("{{{}}}", body.join(", ")),
        });
        stop = s;
    }
    Ok((index, stop, calls))
}

/// `parse_message_from_completion_text(text, thinking_mode)`.
pub fn parse_strict(text: &str, thinking: bool) -> Result<StrictMessage, String> {
    let calls_start = format!("\n\n<{DSML}{CALLS_BLOCK}");
    let mut index = 0;
    let mut reasoning = "";
    if thinking {
        let (i, delta, stop) = read_until_stop(index, text, &[THINK_END, &calls_start]);
        index = i;
        reasoning = delta;
        if stop != Some(THINK_END) {
            return Err("Invalid thinking format: missing </think>".into());
        }
    }
    let (i, summary, stop) = read_until_stop(index, text, &[EOS, &calls_start]);
    index = i;
    let mut stop = stop.map(str::to_string);
    let mut tool_calls = Vec::new();
    if stop.as_deref() == Some(calls_start.as_str()) {
        let (i, s, calls) = parse_tool_calls_strict(index, text)?;
        index = i;
        let _ = s;
        tool_calls = calls;
        let (i, tail, s2) = read_until_stop(index, text, &[EOS]);
        index = i;
        if !tail.is_empty() {
            return Err("Unexpected content after tool calls".into());
        }
        stop = s2.map(str::to_string);
    } else if stop.as_deref() != Some(EOS) {
        return Err("Invalid format: missing EOS token".into());
    }
    if index != text.len() || !matches!(stop.as_deref(), Some(EOS) | None) {
        return Err("Unexpected content at end".into());
    }
    for sp in [BOS, EOS, THINK_START, THINK_END, DSML] {
        if summary.contains(sp) || reasoning.contains(sp) {
            return Err(format!("Unexpected special token '{sp}' in content"));
        }
    }
    Ok(StrictMessage {
        content: summary.into(),
        reasoning_content: reasoning.into(),
        tool_calls,
    })
}

// ------------------------------------------------------------------ tolerant

fn is_py_space(c: char) -> bool {
    // Python `\s` on str: Unicode whitespace plus the ASCII separators \x1c-\x1f.
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `_RE_INVOKE.finditer`: (name, body) pairs.
fn tolerant_invokes(text: &str) -> Vec<(&str, &str)> {
    let open = format!("<{DSML}{INVOKE_TAG} name=\"");
    let next_invoke = format!("<{DSML}{INVOKE_TAG} ");
    let calls_close = format!("</{DSML}{CALLS_BLOCK}>");
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(p) = text[pos..].find(&open) {
        let start = pos + p;
        let name_start = start + open.len();
        let Some(q) = text[name_start..].find('"') else {
            // no closing quote: no match can start here or later
            break;
        };
        let name = &text[name_start..name_start + q];
        let mut i = name_start + q + 1;
        let ws = text[i..].len() - text[i..].trim_start_matches(is_py_space).len();
        i += ws;
        if text[i..].starts_with('>') {
            i += 1;
        }
        if text[i..].starts_with('\n') {
            i += 1;
        }
        let end = [text[i..].find(&next_invoke), text[i..].find(&calls_close)]
            .into_iter()
            .flatten()
            .min()
            .map_or(text.len(), |e| i + e);
        out.push((name, &text[i..end]));
        // finditer resumes at the match end; an empty match is impossible here
        pos = end.max(start + 1);
    }
    out
}

/// `_RE_PARAM_SPEC.finditer(body)`: (key, "true"|"false", value).
fn tolerant_spec_params(body: &str) -> Vec<(&str, &str, &str)> {
    let open = format!("<{DSML}{PARAM_TAG} name=\"");
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(p) = body[pos..].find(&open) {
        let start = pos + p;
        let k0 = start + open.len();
        let hit = (|| {
            let q = body[k0..].find('"')?;
            let key = &body[k0..k0 + q];
            let rest = body[k0 + q..].strip_prefix("\" string=\"")?;
            let flag = if rest.starts_with("true") {
                "true"
            } else if rest.starts_with("false") {
                "false"
            } else {
                return None;
            };
            let rest = rest[flag.len()..].strip_prefix('"')?;
            let trimmed = rest.trim_start_matches(is_py_space);
            let rest = trimmed.strip_prefix('>')?;
            let lt = rest.find('<')?;
            let v_start = body.len() - rest.len();
            Some((key, flag, &rest[..lt], v_start + lt + 1))
        })();
        match hit {
            Some((k, f, v, end)) => {
                out.push((k, f, v));
                pos = end;
            }
            None => pos = start + 1,
        }
    }
    out
}

/// `_RE_PARAM_ATTR.finditer(body)`: (key, value-in-attribute).
fn tolerant_attr_params(body: &str) -> Vec<(&str, &str)> {
    let open = format!("<{DSML}{PARAM_TAG} name=\"");
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(p) = body[pos..].find(&open) {
        let start = pos + p;
        let k0 = start + open.len();
        let hit = (|| {
            let q = body[k0..].find('"')?;
            let key = &body[k0..k0 + q];
            let v0 = k0 + q + "\" string=\"".len();
            if !body[k0 + q..].starts_with("\" string=\"") {
                return None;
            }
            // lazy value: the first `"` followed by \s* then `>`
            let mut from = v0;
            while let Some(r) = body[from..].find('"') {
                let qq = from + r;
                let after = body[qq + 1..].trim_start_matches(is_py_space);
                if after.starts_with('>') {
                    let end = body.len() - after.len() + 1;
                    return Some((key, &body[v0..qq], end));
                }
                from = qq + 1;
            }
            None
        })();
        match hit {
            Some((k, v, end)) => {
                out.push((k, v));
                pos = end;
            }
            None => pos = start + 1,
        }
    }
    out
}

/// `State._parse_tool_calls_tolerant`.
pub fn parse_tolerant(text: &str) -> Vec<ParsedCall> {
    let mut out = Vec::new();
    for (name, body) in tolerant_invokes(text) {
        if name.is_empty() {
            continue;
        }
        let mut args: Map<String, Value> = Map::new();
        let spec = tolerant_spec_params(body);
        for (k, flag, v) in &spec {
            let value = if *flag == "true" {
                Value::String(v.to_string())
            } else {
                serde_json::from_str(v).unwrap_or_else(|_| Value::String(v.to_string()))
            };
            args.insert(k.to_string(), value);
        }
        for (k, v) in tolerant_attr_params(body) {
            if !k.is_empty() && !spec.iter().any(|(sk, _, _)| *sk == k) && !args.contains_key(k) {
                args.insert(k.to_string(), Value::String(v.to_string()));
            }
        }
        out.push(ParsedCall {
            name: name.to_string(),
            arguments: dumps(&Value::Object(args)),
        });
    }
    out
}

// ------------------------------------------------------------------ output router

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Reasoning,
    Content,
    Tool,
}

/// `OutputRouter`: streamed text -> reasoning / content events, stop strings,
/// and the tool text after the marker (held, never streamed as content).
#[derive(Debug, Clone)]
pub struct OutputRouter {
    pub phase: Phase,
    pending: String,
    stop_strings: Vec<String>,
    detect_tool_calls: bool,
    pub reasoning: String,
    pub content: String,
    pub tool_text: String,
    pub stopped: bool,
}

fn held_suffix_len(text: &str, markers: &[&str]) -> usize {
    let mut best = 0;
    for m in markers {
        let chars: Vec<(usize, char)> = m.char_indices().collect();
        // proper prefixes m[:k] for k = len-1 .. 1 (in characters), longest first
        for k in (1..chars.len()).rev() {
            let prefix = &m[..chars[k].0];
            if text.ends_with(prefix) {
                best = best.max(prefix.len());
                break;
            }
        }
    }
    best
}

impl OutputRouter {
    pub fn new(thinking: bool, stop_strings: Vec<String>, detect_tool_calls: bool) -> Self {
        Self {
            phase: if thinking {
                Phase::Reasoning
            } else {
                Phase::Content
            },
            pending: String::new(),
            stop_strings: stop_strings.into_iter().filter(|s| !s.is_empty()).collect(),
            detect_tool_calls,
            reasoning: String::new(),
            content: String::new(),
            tool_text: String::new(),
            stopped: false,
        }
    }

    fn markers(&self) -> Vec<&str> {
        let mut m: Vec<&str> = Vec::new();
        match self.phase {
            Phase::Reasoning => m.push(THINK_END),
            Phase::Content if self.detect_tool_calls => m.push(TOOL_CALLS_MARKER),
            Phase::Content => {}
            Phase::Tool => return m,
        }
        m.extend(self.stop_strings.iter().map(String::as_str));
        m
    }

    fn emit(&mut self, text: &str, events: &mut Vec<(Phase, String)>) {
        if text.is_empty() {
            return;
        }
        if self.phase == Phase::Reasoning {
            self.reasoning.push_str(text);
        } else {
            self.content.push_str(text);
        }
        events.push((self.phase, text.to_string()));
    }

    /// Feed decoded text; returns (Reasoning|Content, text) events.
    pub fn feed(&mut self, text: &str) -> Vec<(Phase, String)> {
        let mut events = Vec::new();
        self.pending.push_str(text);
        while !self.pending.is_empty() && !self.stopped {
            if self.phase == Phase::Tool {
                let p = std::mem::take(&mut self.pending);
                self.tool_text.push_str(&p);
                break;
            }
            let markers: Vec<String> = self.markers().into_iter().map(str::to_string).collect();
            let mut hit: Option<(usize, String)> = None;
            for m in &markers {
                if let Some(p) = self.pending.find(m.as_str())
                    && hit.as_ref().is_none_or(|(hp, _)| p < *hp)
                {
                    hit = Some((p, m.clone()));
                }
            }
            let pending = std::mem::take(&mut self.pending);
            let Some((pos, m)) = hit else {
                let refs: Vec<&str> = markers.iter().map(String::as_str).collect();
                let keep = held_suffix_len(&pending, &refs);
                let cut = pending.len() - keep;
                self.emit(&pending[..cut], &mut events);
                self.pending = pending[cut..].to_string();
                break;
            };
            self.emit(&pending[..pos], &mut events);
            if m == THINK_END && self.phase == Phase::Reasoning {
                self.phase = Phase::Content;
                self.pending = pending[pos + m.len()..].to_string();
            } else if m == TOOL_CALLS_MARKER && self.phase == Phase::Content {
                self.phase = Phase::Tool;
                self.pending = pending[pos..].to_string();
            } else {
                self.stopped = true;
            }
        }
        events
    }

    /// Release what is still held back (generation is over).
    pub fn finish(&mut self) -> Vec<(Phase, String)> {
        let mut events = Vec::new();
        let p = std::mem::take(&mut self.pending);
        if self.phase == Phase::Tool {
            self.tool_text.push_str(&p);
        } else if !self.stopped {
            self.emit(&p, &mut events);
        }
        events
    }

    /// `State._parse_tool_calls`: strict, then tolerant, then give up and move
    /// the tool text back into content. Mutates `content`/`tool_text` exactly
    /// as the server does.
    pub fn parse_tool_calls(&mut self, thinking: bool) -> Vec<ParsedCall> {
        if self.tool_text.is_empty() {
            return Vec::new();
        }
        let mut text = String::new();
        if thinking {
            text.push_str(&self.reasoning);
            text.push_str(THINK_END);
        }
        text.push_str(&self.content);
        text.push_str(&self.tool_text);
        if !text.ends_with(EOS) {
            text.push_str(EOS);
        }
        let calls = match parse_strict(&text, thinking) {
            Ok(m) => m
                .tool_calls
                .into_iter()
                .map(|c| ParsedCall {
                    name: match c.namespace {
                        Some(ns) if !ns.is_empty() => format!("{ns}::{}", c.name),
                        _ => c.name,
                    },
                    arguments: c.arguments,
                })
                .collect(),
            Err(e) => {
                let recovered = parse_tolerant(&text);
                if recovered.is_empty() {
                    tracing::warn!(
                        "dsv41 tool-call parse failed ({e}); returning raw text as content"
                    );
                    let t = std::mem::take(&mut self.tool_text);
                    self.content.push_str(&t);
                    return Vec::new();
                }
                tracing::warn!(
                    "dsv41 tool-call strict parse failed ({e}); recovered {} call(s) tolerantly",
                    recovered.len()
                );
                recovered
            }
        };
        calls
    }
}
