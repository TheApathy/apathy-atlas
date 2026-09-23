// SPDX-License-Identifier: AGPL-3.0-only

//! Register beneath tool_parser::tests. Uses only the production parser,
//! native publication and detector. all-models port: this tree has no
//! schema `coerce_all` and a single detector streaming mode, so typed values
//! come from `native_publication` (the same pass blocking and streaming
//! publication use for glm_xml) and each stream case runs once. Event
//! assembly below is a transport observer, not a replacement wire parser.
//! No GPU, checkpoint or process access.

use super::super::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn tools() -> Vec<ToolDefinition> {
    serde_json::from_value(json!([{
        "type":"function", "function":{"name":"inspect", "parameters":{
            "type":"object", "properties":{
                "literal":{"type":"string"}, "empty":{"type":"string"},
                "json_text":{"type":"string"}, "count":{"type":"integer"},
                "ratio":{"type":"number"}, "enabled":{"type":"boolean"},
                "items":{"type":"array"}, "object":{"type":"object"},
                "nothing":{"type":"null"}
            }
        }}
    }]))
    .unwrap()
}

fn wire() -> &'static str {
    concat!(
        "<tool_call>inspect",
        "<arg_key>literal</arg_key><arg_value> \tfalse\n雪🙂 </arg_value>",
        "<arg_key>empty</arg_key><arg_value></arg_value>",
        "<arg_key>json_text</arg_key><arg_value>{\"x\":false}</arg_value>",
        "<arg_key>count</arg_key><arg_value>32</arg_value>",
        "<arg_key>ratio</arg_key><arg_value>2.5</arg_value>",
        "<arg_key>enabled</arg_key><arg_value>false</arg_value>",
        "<arg_key>items</arg_key><arg_value>[1,\"雪\"]</arg_value>",
        "<arg_key>object</arg_key><arg_value>{\"x\":false}</arg_value>",
        "<arg_key>nothing</arg_key><arg_value>null</arg_value></tool_call>"
    )
}

fn expected() -> Value {
    json!({"literal":" \tfalse\n雪🙂 ", "empty":"", "json_text":"{\"x\":false}",
        "count":32, "ratio":2.5, "enabled":false, "items":[1,"雪"],
        "object":{"x":false}, "nothing":null})
}

fn typed(calls: Vec<ToolCall>) -> Vec<ToolCall> {
    // Same production schema pass used by the blocking and streaming
    // publication handlers for the native parser.
    calls
        .iter()
        .map(|call| native_publication::validate_call(call, &tools()).unwrap())
        .collect()
}

fn assert_call(call: &ToolCall, name: &str, args: &Value) {
    assert!(!call.id.is_empty());
    assert_eq!(call.call_type, "function");
    assert_eq!(call.function.name, name);
    assert_eq!(
        serde_json::from_str::<Value>(&call.function.arguments).unwrap(),
        *args
    );
}

fn collect(events: Vec<DetectorOutput>) -> (String, Vec<ToolCall>) {
    let mut content = String::new();
    let mut pending: BTreeMap<usize, ToolCall> = BTreeMap::new();
    let mut complete: BTreeMap<usize, ToolCall> = BTreeMap::new();
    for event in events {
        match event {
            DetectorOutput::Content(text) => content.push_str(&text),
            DetectorOutput::ToolCall(call, idx) => {
                assert!(
                    !pending.contains_key(&idx),
                    "complete call duplicates header"
                );
                assert!(complete.insert(idx, call).is_none(), "call repeated");
            }
            DetectorOutput::ToolCallStart { id, name, idx } => {
                assert!(!complete.contains_key(&idx));
                let call = ToolCall {
                    id,
                    call_type: "function".into(),
                    function: FunctionCall {
                        name,
                        arguments: String::new(),
                    },
                };
                assert!(pending.insert(idx, call).is_none(), "header repeated");
            }
            DetectorOutput::ToolCallDelta { args, idx } => {
                pending
                    .get_mut(&idx)
                    .expect("delta without header")
                    .function
                    .arguments
                    .push_str(&args);
            }
            DetectorOutput::ToolCallEnd { idx } => {
                let call = pending.remove(&idx).expect("end without header");
                assert!(complete.insert(idx, call).is_none(), "end repeated");
            }
        }
    }
    assert!(pending.is_empty(), "orphan callable header at stream end");
    for (expected_idx, actual_idx) in complete.keys().enumerate() {
        assert_eq!(*actual_idx, expected_idx, "noncontiguous call order");
    }
    let calls: Vec<_> = complete.into_values().collect();
    for (i, call) in calls.iter().enumerate() {
        assert!(
            calls[..i].iter().all(|prior| prior.id != call.id),
            "reused call ID"
        );
    }
    (content, typed(calls))
}

fn run(chunks: &[&str]) -> (String, Vec<ToolCall>) {
    let mut detector = StreamingToolDetector::new();
    let mut events = Vec::new();
    for chunk in chunks {
        events.extend(detector.process(chunk));
    }
    events.extend(detector.flush());
    assert!(
        detector.flush().is_empty(),
        "second flush must not replay events"
    );
    collect(events)
}

#[test]
fn glm_tool_contract_blocking_preserves_native_raw_and_schema_types() {
    let (content, calls) = parse_tool_calls(wire());
    assert!(content.is_none());
    let calls = typed(calls);
    assert_eq!(calls.len(), 1, "native GLM tags must produce one call");
    assert_call(&calls[0], "inspect", &expected());
}

#[test]
fn glm_tool_contract_without_schema_never_guesses_raw_string_types() {
    let (_, calls) = parse_tool_calls(wire());
    assert_eq!(calls.len(), 1);
    let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["literal"], " \tfalse\n雪🙂 ");
    assert_eq!(args["enabled"], "false");
    assert_eq!(args["count"], "32");
    assert_eq!(args["object"], "{\"x\":false}");
}

#[test]
fn glm_tool_contract_all_utf8_splits_and_character_bursts_match_blocking() {
    let input = wire();
    for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
        let (content, calls) = run(&[&input[..split], &input[split..]]);
        assert!(content.is_empty(), "GLM markup leaked at split {split}");
        assert_eq!(calls.len(), 1, "split {split}");
        assert_call(&calls[0], "inspect", &expected());
    }
    let boundaries: Vec<_> = input
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(input.len()))
        .collect();
    let chunks: Vec<_> = boundaries.windows(2).map(|w| &input[w[0]..w[1]]).collect();
    let (_, calls) = run(&chunks);
    assert_eq!(calls.len(), 1);
    assert_call(&calls[0], "inspect", &expected());
}

#[test]
fn glm_tool_contract_zero_args_and_repeated_name_calls_keep_order_and_ids() {
    let input = concat!(
        "before<tool_call>inspect</tool_call>",
        "<tool_call>inspect<arg_key>literal</arg_key><arg_value>second</arg_value>",
        "</tool_call>after"
    );
    let (content, calls) = parse_tool_calls(input);
    assert_eq!(content.as_deref(), Some("before\nafter"));
    assert_eq!(calls.len(), 2);
    assert_call(&calls[0], "inspect", &json!({}));
    assert_call(&calls[1], "inspect", &json!({"literal":"second"}));
    assert_ne!(calls[0].id, calls[1].id);
    let (content, calls) = run(&[input]);
    assert_eq!(content, "beforeafter");
    assert_eq!(calls.len(), 2);
    assert_call(&calls[0], "inspect", &json!({}));
    assert_call(&calls[1], "inspect", &json!({"literal":"second"}));
}

#[test]
fn glm_tool_contract_value_region_owns_unrelated_xml_and_tool_close_text() {
    let raw = "<div>雪</div> </tool_call> <function=not_a_tool>";
    let input = format!(
        "<tool_call>inspect<arg_key>literal</arg_key><arg_value>{raw}</arg_value></tool_call>"
    );
    let (_, calls) = parse_tool_calls(&input);
    assert_eq!(calls.len(), 1);
    assert_call(&calls[0], "inspect", &json!({"literal":raw}));
    for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
        let (_, calls) = run(&[&input[..split], &input[split..]]);
        assert_eq!(calls.len(), 1, "split {split}");
        assert_call(&calls[0], "inspect", &json!({"literal":raw}));
    }
}

fn assert_no_callable(events: &[DetectorOutput], label: &str) {
    assert!(
        events
            .iter()
            .all(|event| matches!(event, DetectorOutput::Content(_))),
        "invalid/incomplete GLM emission published a callable event: {label}"
    );
}

#[test]
fn glm_tool_contract_duplicate_and_malformed_pairs_fail_before_publication() {
    let invalid = [
        "<tool_call>inspect<arg_key>literal</arg_key><arg_value>a</arg_value><arg_key>literal</arg_key><arg_value>b</arg_value></tool_call>",
        "<tool_call>inspect<arg_value>orphan</arg_value></tool_call>",
        "<tool_call>inspect<arg_key>literal</arg_key></tool_call>",
        "<tool_call>inspect<arg_key></arg_key><arg_value>x</arg_value></tool_call>",
        "<tool_call>inspect<arg_key>literal</arg_key><arg_value>x</tool_call>",
    ];
    for input in invalid {
        assert!(
            parse_tool_calls(input).1.is_empty(),
            "blocking accepted {input}"
        );
        for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
            let mut detector = StreamingToolDetector::new();
            assert_no_callable(&detector.process(&input[..split]), "first chunk");
            assert_no_callable(&detector.process(&input[split..]), "second chunk");
            assert_no_callable(&detector.flush(), "invalid flush");
            assert!(detector.flush().is_empty());
        }
    }
}

#[test]
fn glm_tool_contract_every_truncated_prefix_is_noncallable_including_flush() {
    let input =
        "<tool_call>inspect<arg_key>literal</arg_key><arg_value>雪🙂</arg_value></tool_call>";
    for end in (0..input.len()).filter(|&i| input.is_char_boundary(i)) {
        let prefix = &input[..end];
        assert!(
            parse_tool_calls(prefix).1.is_empty(),
            "blocking salvaged {prefix:?}"
        );
        let mut detector = StreamingToolDetector::new();
        assert_no_callable(&detector.process(prefix), prefix);
        assert_no_callable(&detector.flush(), prefix);
        assert!(detector.flush().is_empty());
    }
}
