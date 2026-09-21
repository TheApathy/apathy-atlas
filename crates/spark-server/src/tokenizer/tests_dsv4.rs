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

#[test]
fn dsv4_image_content_renders_official_marker_in_original_order() {
    let out = render_dsv4(&[json!({"role": "user", "content": [
        {"type": "text", "text": "before"},
        {"type": "image", "image": "fixture"},
        {"type": "text", "text": "between"},
        {"type": "image_url", "image_url": {"url": "fixture"}},
        {"type": "text", "text": "after"}
    ]})]);
    assert!(
        out.contains("before<｜deepseek_image｜>between<｜deepseek_image｜>after"),
        "{out}"
    );
    assert!(!out.contains("[Image]"));
}

fn render_dsv4(messages: &[serde_json::Value]) -> String {
    render_dsv4_thinking(messages, false)
}

fn render_dsv4_thinking(messages: &[serde_json::Value], enable_thinking: bool) -> String {
    render_dsv4_with_reminder(messages, enable_thinking, false, usize::MAX)
}

fn render_dsv4_with_reminder(
    messages: &[serde_json::Value],
    enable_thinking: bool,
    tool_call_reminder: bool,
    tool_call_reminder_min_bytes: usize,
) -> String {
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
            tool_call_reminder => tool_call_reminder,
            tool_call_reminder_min_bytes => tool_call_reminder_min_bytes,
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

#[test]
fn dsv4_deep_tool_reminder_is_per_result_and_prefix_stable() {
    const REMINDER: &str = "[Reminder: respond with exactly one tool call in the required format. Plain-text replies are not accepted by this harness.]";
    let messages = vec![
        json!({"role": "user", "content": "go"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {"name": "a", "arguments": {}}
            }]
        }),
        json!({"role": "tool", "content": "first"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {"name": "b", "arguments": {}}
            }]
        }),
        json!({"role": "tool", "content": "second"}),
    ];

    let off = render_dsv4_with_reminder(&messages, false, false, 0);
    assert!(
        !off.contains(REMINDER),
        "opt-out must preserve prompt bytes"
    );

    let shallow = render_dsv4_with_reminder(&messages, false, true, usize::MAX);
    assert!(
        !shallow.contains(REMINDER),
        "shallow history must be untouched"
    );

    let deep = render_dsv4_with_reminder(&messages, false, true, 0);
    assert_eq!(
        deep.matches(REMINDER).count(),
        2,
        "every deep result is reminded"
    );
    assert!(deep.contains(&format!("first\n\n{REMINDER}</tool_result>")));
    assert!(deep.contains(&format!("second\n\n{REMINDER}</tool_result>")));

    let head = render_dsv4_with_reminder(&messages[..3], false, true, 0);
    let generation_tail = "<｜Assistant｜></think>";
    let stable_head = head
        .strip_suffix(generation_tail)
        .expect("head render ends in generation prompt");
    assert!(
        deep.starts_with(stable_head),
        "extending a conversation must retain the prior rendered prefix"
    );
}

#[test]
fn dsv4_tool_reminder_depth_is_measured_in_rendered_utf8_bytes() {
    const REMINDER: &str = "[Reminder: respond with exactly one tool call";
    let messages = vec![
        json!({"role": "user", "content": "é".repeat(64)}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {"name": "read", "arguments": {"path": "é"}}
            }]
        }),
        json!({"role": "tool", "content": "é".repeat(64)}),
    ];
    let baseline = render_dsv4_with_reminder(&messages, false, false, 0);
    let result_end = baseline.find("</tool_result>").expect("tool result close");
    let scalar_count = baseline[..result_end].chars().count();
    assert!(
        result_end > scalar_count,
        "fixture must distinguish bytes from chars"
    );

    let threshold_between = scalar_count + (result_end - scalar_count) / 2;
    let rendered = render_dsv4_with_reminder(&messages, false, true, threshold_between);
    assert!(
        rendered.contains(REMINDER),
        "byte depth crosses the gate even though Unicode scalar count does not"
    );
}

#[test]
fn dsv4_tool_reminder_threshold_is_exact_for_each_complex_result_prefix() {
    const REMINDER: &str = "[Reminder: respond with exactly one tool call";
    let messages = vec![
        json!({"role": "system", "content": "système"}),
        json!({"role": "user", "content": "read é.txt"}),
        json!({
            "role": "assistant",
            "content": "checking",
            "reasoning_content": "réfléchir",
            "tool_calls": [{
                "type": "function",
                "function": {"name": "read", "arguments": {"path": "é.txt"}}
            }]
        }),
        json!({"role": "tool", "content": "résultat α"}),
        json!({"role": "tool", "content": "résultat β"}),
    ];
    let baseline = render_dsv4_with_reminder(&messages, false, false, 0);
    assert!(baseline.contains("<｜DSML｜tool_calls>"));
    assert!(baseline.contains("<think>réfléchir</think>"));
    let closes = baseline
        .match_indices("</tool_result>")
        .map(|(byte_offset, _)| byte_offset)
        .collect::<Vec<_>>();
    assert_eq!(closes.len(), 2);

    let at_first = render_dsv4_with_reminder(&messages, false, true, closes[0]);
    let after_first = render_dsv4_with_reminder(&messages, false, true, closes[0] + 1);
    assert_eq!(at_first.matches(REMINDER).count(), 2);
    assert_eq!(after_first.matches(REMINDER).count(), 1);

    let at_second = render_dsv4_with_reminder(&messages, false, true, closes[1]);
    let after_second = render_dsv4_with_reminder(&messages, false, true, closes[1] + 1);
    assert_eq!(at_second.matches(REMINDER).count(), 1);
    assert_eq!(after_second.matches(REMINDER).count(), 0);
}

#[test]
fn dsv4_thinking_control_changes_only_the_generation_marker() {
    let messages = [json!({"role": "user", "content": "keep my first byte"})];
    let off = render_dsv4_thinking(&messages, false);
    let high_effort = render_dsv4_thinking(&messages, true);
    let prefix = "<｜begin▁of▁sentence｜><｜User｜>keep my first byte<｜Assistant｜>";
    assert_eq!(off.strip_suffix("</think>"), Some(prefix));
    assert_eq!(high_effort.strip_suffix("<think>"), Some(prefix));
    assert!(!high_effort.contains("reasoning_effort"));
    assert!(!high_effort.contains("high"));
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
    assert!(
        out.starts_with("<｜begin▁of▁sentence｜>be terse"),
        "{out:?}"
    );
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
    assert!(
        on.ends_with("<｜Assistant｜><think>"),
        "thinking on: {on:?}"
    );
    assert!(
        off.ends_with("<｜Assistant｜></think>"),
        "thinking off: {off:?}"
    );
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
