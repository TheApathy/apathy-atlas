// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4 chat-template tests.
//!
//! The checkpoint has no `tool` role — `encoding_dsv4.py`'s
//! `merge_tool_messages()` folds tool results into a `<｜User｜>` turn as
//! `<tool_result>` blocks, and assistant tool calls render as a native
//! DSML envelope. Before this was modelled, turn 2 of every tool chain
//! 400'd with `Unexpected message role.`; these tests pin the fix.

use super::*;
use serde_json::json;

/// The DSML namespace token as the checkpoint emits it (U+FF5C bars).
const DSML: &str = "\u{ff5c}DSML\u{ff5c}";

fn render_dsv4(messages: &[serde_json::Value]) -> String {
    render_dsv4_thinking(messages, false)
}

fn render_dsv4_thinking(messages: &[serde_json::Value], enable_thinking: bool) -> String {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../jinja-templates/deepseek_v4.jinja"
    ))
    .expect("bundled DeepSeek-V4 template must be present");
    // Go through the same rewrite the server's override loader applies.
    let raw = super::jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env = super::jinja_helpers::build_jinja_env(&raw).expect("template compiles");
    let messages = normalize_tool_call_arguments(messages);
    env.get_template("chat")
        .unwrap()
        .render(minijinja::context! {
            messages => minijinja::Value::from_serialize(&messages),
            add_generation_prompt => true,
            enable_thinking => enable_thinking,
        })
        .expect("template renders")
}

/// The blocker: a `tool` message used to hit `raise_exception`.
#[test]
fn dsv4_tool_role_renders_as_user_tool_result() {
    let out = render_dsv4(&[
        json!({"role": "user", "content": "list the dir"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {"name": "bash", "arguments": {"command": "ls"}}
            }]
        }),
        json!({"role": "tool", "content": "total 0"}),
    ]);
    assert!(
        out.contains("<｜User｜><tool_result>total 0</tool_result><｜Assistant｜>"),
        "tool result must become a user turn: {out:?}"
    );
}

/// Assistant tool calls must round-trip through the native DSML syntax
/// the `DsmlV4Parser` decodes — string args raw, everything else JSON.
#[test]
fn dsv4_assistant_tool_calls_render_native_dsml() {
    let out = render_dsv4(&[
        json!({"role": "user", "content": "weather?"}),
        json!({
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Berlin", "days": 3}
                }
            }]
        }),
    ]);
    let expected = format!(
        "Checking.\n\n<{DSML}tool_calls>\n\
         <{DSML}invoke name=\"get_weather\">\n\
         <{DSML}parameter name=\"city\" string=\"true\">Berlin</{DSML}parameter>\n\
         <{DSML}parameter name=\"days\" string=\"false\">3</{DSML}parameter>\n\
         </{DSML}invoke>\n\
         </{DSML}tool_calls>"
    );
    assert!(out.contains(&expected), "DSML envelope drifted:\n{out}");
}

/// Parallel calls fan out to one `<invoke>` each inside a single envelope.
#[test]
fn dsv4_parallel_tool_calls_share_one_envelope() {
    let out = render_dsv4(&[
        json!({"role": "user", "content": "go"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "a", "arguments": {}}},
                {"type": "function", "function": {"name": "b", "arguments": {}}}
            ]
        }),
    ]);
    assert_eq!(
        out.matches(&format!("<{DSML}tool_calls>")).count(),
        1,
        "one envelope only: {out}"
    );
    assert!(out.contains(&format!("<{DSML}invoke name=\"a\">")), "{out}");
    assert!(out.contains(&format!("<{DSML}invoke name=\"b\">")), "{out}");
}

/// A run of parallel tool results shares ONE user turn, joined by a blank
/// line — matching `merge_tool_messages()`, which emits a single user
/// message carrying every `tool_result` block.
#[test]
fn dsv4_consecutive_tool_results_share_one_user_turn() {
    let out = render_dsv4(&[
        json!({"role": "user", "content": "go"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "a", "arguments": {}}},
                {"type": "function", "function": {"name": "b", "arguments": {}}}
            ]
        }),
        json!({"role": "tool", "content": "ra"}),
        json!({"role": "tool", "content": "rb"}),
    ]);
    assert!(
        out.contains(
            "<｜User｜><tool_result>ra</tool_result>\n\n<tool_result>rb</tool_result><｜Assistant｜>"
        ),
        "parallel results must share one user turn: {out:?}"
    );
    // Exactly two user turns total: the original question and the results.
    assert_eq!(out.matches("<｜User｜>").count(), 2, "{out:?}");
}

/// The full chain the bench exercises: user → calls → results → answer →
/// follow-up. Renders without raising, and ends primed for generation.
#[test]
fn dsv4_multi_turn_tool_chain_renders() {
    let out = render_dsv4(&[
        json!({"role": "system", "content": "be terse"}),
        json!({"role": "user", "content": "read a.txt"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {"name": "read", "arguments": {"path": "a.txt"}}
            }]
        }),
        json!({"role": "tool", "content": "hello"}),
        json!({"role": "assistant", "content": "It says hello."}),
        json!({"role": "user", "content": "and b.txt?"}),
    ]);
    assert!(out.starts_with("<｜begin▁of▁sentence｜>be terse"), "{out:?}");
    // enable_thinking=false ⇒ the generation prompt CLOSES thinking, matching
    // encoding_dsv4.py:381 (`thinking_end_token if thinking_mode != "thinking"`).
    assert!(
        out.ends_with("and b.txt?<｜Assistant｜></think>"),
        "{out:?}"
    );
    assert_eq!(
        out.matches("<｜begin▁of▁sentence｜>").count(),
        1,
        "bos must appear exactly once: {out:?}"
    );
}

/// The thinking markers are the checkpoint's own (encoding_dsv4.py:19-20).
/// Both were the empty string since the original port, so the template opened
/// no `<think>` when thinking was requested and closed none when it was not —
/// which also left the scheduler's `inside_thinking` tracking (and hence the
/// speculation think-gate) with nothing to key on.
#[test]
fn dsv4_generation_prompt_opens_or_closes_thinking() {
    let msgs = [json!({"role": "user", "content": "hi"})];
    let on = render_dsv4_thinking(&msgs, true);
    let off = render_dsv4_thinking(&msgs, false);
    assert!(on.ends_with("<｜Assistant｜><think>"), "thinking on: {on:?}");
    assert!(off.ends_with("<｜Assistant｜></think>"), "thinking off: {off:?}");
}

/// Historical assistant reasoning replays wrapped in the markers, not raw.
#[test]
fn dsv4_historical_reasoning_is_wrapped() {
    let out = render_dsv4(&[
        json!({"role": "user", "content": "q"}),
        json!({"role": "assistant", "content": "A.", "reasoning_content": "weigh it"}),
    ]);
    assert!(
        out.contains("<think>weigh it</think>A."),
        "reasoning must be wrapped: {out:?}"
    );
}
