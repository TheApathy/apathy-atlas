// SPDX-License-Identifier: AGPL-3.0-only

//! Checkpoint-native DSML argument contracts through the actual public parser.
//! No schema coercion: explicit string=true/false is already the wire type.

use super::super::*;
use serde_json::{Value, json};

#[path = "dsml_typed_contract_stream.rs"]
mod streaming;

fn parameter(name: &str, string: &str, raw: &str) -> String {
    format!("<｜DSML｜parameter name=\"{name}\" string=\"{string}\">{raw}</｜DSML｜parameter>")
}

fn invocation(body: &str) -> String {
    format!("<｜DSML｜invoke name=\"inspect\">\n{body}\n</｜DSML｜invoke>")
}

fn envelope(body: &str) -> String {
    format!("<｜DSML｜tool_calls>\n{body}\n</｜DSML｜tool_calls>")
}

fn wire(string: &str, raw: &str) -> String {
    envelope(&invocation(&parameter("value", string, raw)))
}

fn typed_values() -> Vec<(&'static str, Value)> {
    vec![
        ("3", json!(3)),
        ("-0.25", json!(-0.25)),
        ("false", json!(false)),
        ("null", Value::Null),
        ("[1,true,null]", json!([1, true, null])),
        (r#"{"nested":[2,"x"]}"#, json!({"nested": [2, "x"]})),
        (r#""quoted""#, json!("quoted")),
    ]
}

fn blocking_args(text: &str) -> Value {
    let (content, calls) = parse_tool_calls(text);
    assert!(content.as_deref().unwrap_or("").trim().is_empty());
    assert_eq!(calls.len(), 1, "one complete invocation: {text:?}");
    assert_eq!(calls[0].function.name, "inspect");
    serde_json::from_str(&calls[0].function.arguments).expect("valid argument JSON")
}

fn compatible_spellings() -> Vec<String> {
    let canonical = wire("true", "Berlin");
    vec![
        canonical.clone(),
        canonical.replace("｜DSML｜", "DSML｜"),
        canonical.replace("｜DSML｜tool_calls", "｜DSML｜_calls"),
        canonical
            .replace("<｜DSML｜invoke ", "<invoke ")
            .replace("</｜DSML｜invoke>", "</invoke>"),
        canonical
            .replace("<｜DSML｜parameter ", "<parameter ")
            .replace("</｜DSML｜parameter>", "</parameter>"),
    ]
}

fn duplicate_wire(first: &str, second: &str) -> String {
    envelope(&invocation(&format!(
        "{}\n{}",
        parameter("value", "true", first),
        parameter("value", "false", second),
    )))
}

#[test]
fn blocking_explicit_strings_do_not_become_json_values() {
    for (raw, _) in typed_values() {
        assert_eq!(blocking_args(&wire("true", raw)), json!({"value": raw}));
    }
}

#[test]
fn blocking_explicit_json_preserves_each_native_type() {
    for (raw, expected) in typed_values() {
        assert_eq!(
            blocking_args(&wire("false", raw)),
            json!({"value": expected}),
            "string=false value {raw:?}",
        );
    }
}

#[test]
fn blocking_string_whitespace_and_unicode_are_exact() {
    for raw in ["  hello  ", "\n\t雪 café \r\n", "", " \t\n "] {
        assert_eq!(blocking_args(&wire("true", raw)), json!({"value": raw}));
    }
}

#[test]
fn blocking_json_allows_json_whitespace_without_changing_type() {
    assert_eq!(
        blocking_args(&wire("false", " \r\n [1, false] \t")),
        json!({"value": [1, false]}),
    );
}

#[test]
fn blocking_duplicate_parameters_reject_the_invocation() {
    for (first, second) in [("3", "3"), ("first", "false")] {
        let (_, calls) = parse_tool_calls(&duplicate_wire(first, second));
        assert!(
            calls.is_empty(),
            "duplicate key must not silently overwrite"
        );
    }
}

#[test]
fn blocking_invalid_explicit_types_do_not_fall_back_to_strings() {
    for (string, raw) in [("false", "not JSON"), ("false", "NaN"), ("maybe", "3")] {
        let (_, calls) = parse_tool_calls(&wire(string, raw));
        assert!(calls.is_empty(), "invalid typed value {string:?}/{raw:?}");
    }
}

#[test]
fn blocking_mixed_short_and_frayed_tags_keep_working() {
    for text in compatible_spellings() {
        assert_eq!(blocking_args(&text), json!({"value": "Berlin"}));
    }
}

#[test]
fn blocking_multiple_invokes_keep_order_types_and_surrounding_content() {
    let body = format!(
        "{}\n{}",
        invocation(&parameter("value", "true", "3")),
        invocation(&parameter("value", "false", "3")),
    );
    let (content, calls) = parse_tool_calls(&format!("before\n{}\nafter", envelope(&body)));
    assert_eq!(content.as_deref(), Some("before\nafter"));
    assert_eq!(calls.len(), 2);
    assert_ne!(calls[0].id, calls[1].id);
    let values: Vec<Value> = calls
        .iter()
        .map(|call| {
            assert_eq!(call.function.name, "inspect");
            serde_json::from_str(&call.function.arguments).unwrap()
        })
        .collect();
    assert_eq!(values, vec![json!({"value": "3"}), json!({"value": 3})]);
}

#[test]
fn blocking_rejected_duplicate_does_not_hide_a_later_valid_call() {
    let text = format!("{}\n{}", duplicate_wire("first", "3"), wire("false", "4"));
    let (_, calls) = parse_tool_calls(&text);
    assert_eq!(
        calls.len(),
        1,
        "only the later well-formed invocation survives"
    );
    assert_eq!(calls[0].function.name, "inspect");
    let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args, json!({"value": 4}));
}
