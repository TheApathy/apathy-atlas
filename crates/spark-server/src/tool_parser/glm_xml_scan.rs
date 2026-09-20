// SPDX-License-Identifier: AGPL-3.0-only

//! Envelope boundaries shared by GLM blocking and streaming dispatch. Values
//! are opaque bytes until </arg_value>; other models retain their old scanner.

pub(super) const OPEN: &str = "<tool_call>";
pub(super) const CLOSE: &str = "</tool_call>";
pub(super) const KEY: &str = "<arg_key>";
pub(super) const KEY_CLOSE: &str = "</arg_key>";
pub(super) const VALUE: &str = "<arg_value>";
pub(super) const VALUE_CLOSE: &str = "</arg_value>";

/// A native or still-ambiguous prefix must stay buffered. The first inner tag
/// distinguishes GLM from existing JSON/function/invoke formats. In particular
/// a foreign tag *inside* an already-open GLM value cannot change dispatch.
pub(super) fn native_prefix(body: &str) -> bool {
    let body = body.trim_start();
    if body.is_empty() {
        return true;
    }
    if body.starts_with("call:") || body.starts_with("_call:") {
        return false;
    }
    let tag = body.find('<').unwrap_or(body.len());
    let head = &body[..tag];
    if head.contains(['{', '[']) {
        return false;
    }
    if super::is_tool_name_component(head.trim()) {
        // Native owns its bare-name prefix even when the following tag is
        // malformed. Do not reinterpret a nested foreign function as a call.
        return true;
    }
    if tag == body.len() {
        return body
            .as_bytes()
            .first()
            .is_some_and(|b| b.is_ascii_alphanumeric() || matches!(*b, b'_' | b'-' | b'.'));
    }
    let tail = &body[tag..];
    [KEY, KEY_CLOSE, VALUE, VALUE_CLOSE, CLOSE]
        .iter()
        .any(|marker| tail.starts_with(*marker) || marker.starts_with(tail))
}

/// A tool close appearing inside a native value is data, not the envelope end.
/// An unclosed value owns the remaining input; do not recover an inner tool.
pub(super) fn native_close(body: &str) -> Option<usize> {
    let mut cursor = 0;
    while let Some(offset) = body[cursor..].find('<') {
        cursor += offset;
        let rest = &body[cursor..];
        if rest.starts_with(VALUE) {
            cursor += VALUE.len();
            cursor += body[cursor..].find(VALUE_CLOSE)? + VALUE_CLOSE.len();
        } else if rest.starts_with(CLOSE) {
            return Some(cursor);
        } else {
            cursor += 1; // '<' is one ASCII byte at a character boundary.
        }
    }
    None
}

/// Existing Qwen parameter-depth close scanner, moved without semantic change.
pub(super) fn legacy_close(buf: &str) -> Option<usize> {
    let bytes = buf.as_bytes();
    let mut i = 0;
    let mut depth: i32 = 0;
    while i < bytes.len() {
        if buf[i..].starts_with(CLOSE) && depth == 0 {
            return Some(i);
        }
        if buf[i..].starts_with("<parameter=") {
            depth += 1;
            i += "<parameter=".len();
            continue;
        }
        if buf[i..].starts_with("</parameter>") {
            if depth > 0 {
                depth -= 1;
            }
            i += "</parameter>".len();
            continue;
        }
        i += buf[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
    }
    None
}

/// Locate a GLM span without looking inside an earlier legacy envelope.
/// A partial native envelope is protected through the end of input.
fn next_native_span(text: &str, mut cursor: usize) -> Option<(usize, usize)> {
    while let Some(offset) = text[cursor..].find(OPEN) {
        let start = cursor + offset;
        let body_start = start + OPEN.len();
        let body = &text[body_start..];
        if native_prefix(body) {
            let end = native_close(body)
                .map(|end| body_start + end + CLOSE.len())
                .unwrap_or(text.len());
            return Some((start, end));
        }
        cursor = body_start + legacy_close(body)? + CLOSE.len();
    }
    None
}

/// Preserve the historical first-</think> split, but never split a GLM value.
pub(super) fn after_thinking(text: &str) -> &str {
    let mut cursor = 0;
    while let Some((start, end)) = next_native_span(text, cursor) {
        if let Some(offset) = text[cursor..start].find("</think>") {
            return &text[cursor + offset + "</think>".len()..];
        }
        cursor = end;
    }
    text[cursor..]
        .find("</think>")
        .map(|offset| &text[cursor + offset + "</think>".len()..])
        .unwrap_or(text)
}

fn normalize_legacy(text: &str) -> String {
    text.replace("<minimax:tool_call>", OPEN)
        .replace("</minimax:tool_call>", CLOSE)
        .replace("<minimax:_call>", OPEN)
        .replace("</minimax:_call>", CLOSE)
}

/// Existing MiniMax normalization applies only outside protected GLM spans.
pub(super) fn normalize_minimax(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    while let Some((start, end)) = next_native_span(text, cursor) {
        output.push_str(&normalize_legacy(&text[cursor..start]));
        output.push_str(&text[start..end]);
        cursor = end;
    }
    output.push_str(&normalize_legacy(&text[cursor..]));
    output
}

/// In a bulk delta, legacy Mistral scanning must not find an opener buried in
/// a GLM value before the outer <tool_call> has entered buffered mode.
pub(super) fn first_native_opener(text: &str) -> Option<usize> {
    let start = text.find(OPEN)?;
    if !native_prefix(&text[start + OPEN.len()..]) {
        return None;
    }
    for marker in [
        "[TOOL_CALLS]",
        "<|tool_call>",
        "<minimax:tool_call>",
        "<minimax:_call>",
    ] {
        if text.find(marker).is_some_and(|other| other < start) {
            return None;
        }
    }
    Some(start)
}
