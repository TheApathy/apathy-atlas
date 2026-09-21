// SPDX-License-Identifier: AGPL-3.0-only

//! Exercise actual detector outputs, including incremental argument assembly.
//! Tests select its existing buffered/live modes without changing process env.

use super::*;
use std::collections::BTreeMap;

#[derive(Default)]
struct CollectedCall {
    id: String,
    name: String,
    args: String,
    ended: bool,
}

fn outputs(chunks: &[&str], buffered: bool) -> Vec<DetectorOutput> {
    let mut detector = StreamingToolDetector::new();
    detector.buffer_args = buffered;
    let mut outputs = Vec::new();
    for chunk in chunks {
        outputs.extend(detector.process(chunk));
    }
    outputs.extend(detector.flush());
    assert!(detector.flush().is_empty(), "flush must not replay a call");
    outputs
}

fn character_chunks(text: &str) -> Vec<&str> {
    text.char_indices()
        .map(|(offset, ch)| &text[offset..offset + ch.len_utf8()])
        .collect()
}

fn collect(outputs: Vec<DetectorOutput>) -> (String, Vec<Value>) {
    let mut content = String::new();
    let mut calls: BTreeMap<usize, CollectedCall> = BTreeMap::new();
    for output in outputs {
        match output {
            DetectorOutput::Content(text) => content.push_str(&text),
            DetectorOutput::ToolCall(call, idx) => {
                assert!(!calls.contains_key(&idx), "complete call emitted twice");
                calls.insert(
                    idx,
                    CollectedCall {
                        id: call.id,
                        name: call.function.name,
                        args: call.function.arguments,
                        ended: true,
                    },
                );
            }
            DetectorOutput::ToolCallStart { id, name, idx } => {
                assert!(!calls.contains_key(&idx), "call header emitted twice");
                calls.insert(
                    idx,
                    CollectedCall {
                        id,
                        name,
                        ..Default::default()
                    },
                );
            }
            DetectorOutput::ToolCallDelta { args, idx } => {
                let call = calls.get_mut(&idx).expect("header precedes arguments");
                assert!(!call.ended && call.args.is_empty());
                call.args = args;
            }
            DetectorOutput::ToolCallArgsFragment { fragment, idx } => {
                let call = calls.get_mut(&idx).expect("header precedes fragment");
                assert!(!call.ended);
                call.args.push_str(&fragment);
            }
            DetectorOutput::ToolCallEnd { idx } => {
                let call = calls.get_mut(&idx).expect("header precedes completion");
                assert!(!call.ended, "call completed twice");
                call.ended = true;
            }
        }
    }
    let mut ids = std::collections::BTreeSet::new();
    let args = calls
        .into_iter()
        .enumerate()
        .map(|(expected_idx, (idx, call))| {
            assert_eq!(idx, expected_idx, "ordered independent call indices");
            assert!(call.ended, "partial invocation is not a successful call");
            assert!(!call.id.is_empty() && ids.insert(call.id));
            assert_eq!(call.name, "inspect");
            serde_json::from_str(&call.args).expect("complete valid argument JSON")
        })
        .collect();
    (content, args)
}

fn assert_stream(text: &str, expected: Value) {
    for buffered in [false, true] {
        for chunks in [vec![text], character_chunks(text)] {
            let (content, args) = collect(outputs(&chunks, buffered));
            assert!(content.trim().is_empty(), "envelope leaked: {content:?}");
            assert_eq!(args, vec![expected.clone()], "buffered={buffered}");
        }
    }
}

fn assert_rejected(text: &str) {
    for buffered in [false, true] {
        for chunks in [vec![text], character_chunks(text)] {
            // A rejected DSML invocation must not publish executable argument
            // fragments or a completion that cannot be retracted downstream.
            assert!(
                outputs(&chunks, buffered)
                    .iter()
                    .all(|output| { matches!(output, DetectorOutput::Content(_)) }),
                "invalid invocation published a tool event; buffered={buffered}"
            );
        }
    }
}

#[test]
fn streaming_explicit_strings_do_not_become_json_values() {
    for (raw, _) in typed_values() {
        assert_stream(&wire("true", raw), json!({"value": raw}));
    }
}

#[test]
fn streaming_explicit_json_preserves_each_native_type() {
    for (raw, expected) in typed_values() {
        assert_stream(&wire("false", raw), json!({"value": expected}));
    }
}

#[test]
fn streaming_string_whitespace_and_unicode_are_exact() {
    for raw in ["  hello  ", "\n\t雪 café \r\n", "", " \t\n "] {
        assert_stream(&wire("true", raw), json!({"value": raw}));
    }
}

#[test]
fn streaming_duplicate_parameters_publish_no_call() {
    for (first, second) in [("3", "3"), ("first", "false")] {
        assert_rejected(&duplicate_wire(first, second));
    }
}

#[test]
fn streaming_invalid_explicit_types_publish_no_call() {
    for (string, raw) in [("false", "not JSON"), ("false", "NaN"), ("maybe", "3")] {
        assert_rejected(&wire(string, raw));
    }
}

#[test]
fn streaming_mixed_short_and_frayed_tags_keep_working() {
    for text in compatible_spellings() {
        assert_stream(&text, json!({"value": "Berlin"}));
    }
}

#[test]
fn streaming_every_utf8_boundary_preserves_string_and_json_types() {
    let text = envelope(&invocation(&format!(
        "{}\n{}",
        parameter("text", "true", " \n雪 3\t "),
        parameter("data", "false", "[3,false]"),
    )));
    let expected = json!({"text": " \n雪 3\t ", "data": [3, false]});
    for offset in (0..=text.len()).filter(|&offset| text.is_char_boundary(offset)) {
        for buffered in [false, true] {
            let (content, args) = collect(outputs(&[&text[..offset], &text[offset..]], buffered));
            assert!(content.trim().is_empty(), "tag leak at byte {offset}");
            assert_eq!(
                args,
                vec![expected.clone()],
                "split={offset}, buffered={buffered}"
            );
        }
    }
}

#[test]
fn streaming_multiple_invokes_keep_order_types_and_surrounding_content() {
    let body = format!(
        "{}\n{}",
        invocation(&parameter("value", "true", "3")),
        invocation(&parameter("value", "false", "3")),
    );
    let text = format!("before\n{}\nafter", envelope(&body));
    for buffered in [false, true] {
        let (content, args) = collect(outputs(&character_chunks(&text), buffered));
        assert_eq!(
            content.split_whitespace().collect::<Vec<_>>(),
            ["before", "after"]
        );
        assert_eq!(args, vec![json!({"value": "3"}), json!({"value": 3})]);
    }
}
