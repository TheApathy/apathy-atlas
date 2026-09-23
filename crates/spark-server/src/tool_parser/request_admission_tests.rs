// SPDX-License-Identifier: AGPL-3.0-only
//! Adapted from the GLM branch's `api/chat_phases_tool_tests.rs` and
//! `api/tool_capability_tests.rs`. This tree has no IR request type or
//! AppState test fixture, so the same cases drive `capability` directly with
//! the native parser (and prove every other parser is left untouched).

use super::capability;
use crate::tool_parser::{ToolCallFormat, ToolCallParser, ToolChoice, ToolDefinition};
use serde_json::{Value, json};

fn glm() -> Box<dyn ToolCallParser> {
    "glm_xml".parse::<ToolCallFormat>().unwrap().into_parser()
}

fn tools(value: Value) -> Vec<ToolDefinition> {
    serde_json::from_value(value).unwrap()
}

fn choice(value: Value) -> ToolChoice {
    serde_json::from_value(value).unwrap()
}

fn tool(name: &str, parameters: Value) -> Value {
    json!({"type":"function", "function":{"name":name,"parameters":parameters}})
}

fn admit(tool_list: Value, tool_choice: Value) -> Result<(), String> {
    let parser = glm();
    capability(
        &tools(tool_list),
        Some(&choice(tool_choice)),
        Some(parser.as_ref()),
        false,
    )
}

#[test]
fn tool_admission_specific_requires_exact_declared_nonempty_name() {
    for name in ["", " ", "not_declared", "Record_Note"] {
        assert!(
            admit(
                json!([tool("record_note", json!({"type":"object"}))]),
                json!({"type":"function","function":{"name":name}}),
            )
            .is_err(),
            "{name:?}"
        );
    }
    assert!(
        admit(
            json!([]),
            json!({"type":"function","function":{"name":"record_note"}})
        )
        .is_err()
    );
}

#[test]
fn tool_admission_schema_root_and_properties_are_objects() {
    // The GLM branch also rejects an explicit `"parameters": null`; this
    // tree's FunctionDefinition deserializes null as omitted (no change).
    for schema in [
        json!([]),
        json!(false),
        json!({"type":"array"}),
        json!({"type":"object","properties":[]}),
    ] {
        assert!(
            admit(
                json!([tool("record_note", schema.clone())]),
                json!("required")
            )
            .is_err(),
            "{schema}"
        );
    }
}

#[test]
fn tool_admission_names_and_types_must_be_unambiguous() {
    let t = tool("record_note", json!({"type":"object"}));
    assert!(admit(json!([t, t]), json!("required")).is_err());
    for name in ["", " ", "bad<name"] {
        assert!(admit(json!([tool(name, json!({"type":"object"}))]), json!("auto")).is_err());
    }
    let mut t = tool("record_note", json!({"type":"object"}));
    t["type"] = json!("not_function");
    assert!(admit(json!([t]), json!("auto")).is_err());
}

#[test]
fn tool_admission_required_and_additional_properties_shape_is_checked() {
    for schema in [
        json!({"type":"object","required":"x"}),
        json!({"type":"object","required":[7]}),
        json!({"type":"object","required":["x","x"]}),
        json!({"type":"object","additionalProperties":7}),
    ] {
        assert!(
            admit(
                json!([tool("record_note", schema.clone())]),
                json!("required")
            )
            .is_err(),
            "{schema}"
        );
    }
}

#[test]
fn tool_admission_valid_modes_and_nested_schema_remain_accepted() {
    let t = tool(
        "record_note",
        json!({"type":"object","properties":{"note":{"type":"string"},"meta":{"type":"object","additionalProperties":true}},"required":["note"],"additionalProperties":false}),
    );
    for tool_choice in [
        json!("auto"),
        json!("none"),
        json!("required"),
        json!({"type":"function","function":{"name":"record_note"}}),
    ] {
        assert!(admit(json!([t.clone()]), tool_choice).is_ok());
    }
    assert!(admit(json!([]), json!("none")).is_ok());
    assert!(
        admit(
            json!([{"type":"function","function":{"name":"no_args"}}]),
            json!("required")
        )
        .is_ok()
    );
}

#[test]
fn tool_capability_native_disabled_grammar_and_unsupported_schema_fail_at_admission() {
    let parser = glm();
    let schema = json!({"type":"object","properties":{"value":{"type":"string"}},
        "required":["value"],"additionalProperties":false});
    let defs = tools(json!([tool("inspect", schema.clone())]));
    let required = choice(json!("required"));
    assert!(capability(&defs, Some(&required), Some(parser.as_ref()), false).is_ok());
    assert!(capability(&defs, Some(&required), Some(parser.as_ref()), true).is_err());
    let mut unsupported = schema;
    unsupported["properties"]["value"]["pattern"] = json!("^ok$");
    let defs = tools(json!([tool("inspect", unsupported)]));
    assert!(capability(&defs, Some(&required), Some(parser.as_ref()), false).is_err());
    // tool_choice=none never needs the grammar.
    let none = choice(json!("none"));
    assert!(capability(&defs, Some(&none), Some(parser.as_ref()), true).is_ok());
}

#[test]
fn tool_capability_native_rejects_angle_bracket_argument_names() {
    let defs = tools(json!([tool(
        "inspect",
        json!({"type":"object","properties":{"a<b":{"type":"string"}}})
    )]));
    let parser = glm();
    assert!(capability(&defs, None, Some(parser.as_ref()), false).is_err());
}

#[test]
fn tool_capability_leaves_every_other_parser_and_no_parser_untouched() {
    // Inputs the native contract rejects must pass through unchanged for the
    // existing parsers: their validation lives in chat_phases::validate_input.
    let defs = tools(json!([
        tool(
            "mcp:search",
            json!({"type":"object","properties":{"q":{"type":"string","pattern":"x"}}})
        ),
        tool("mcp:search", json!({"type":"object"})),
    ]));
    let required = choice(json!("required"));
    for name in [
        "hermes",
        "qwen3_coder",
        "gemma4",
        "mistral",
        "minimax_xml",
        "bare_json",
    ] {
        let parser = name.parse::<ToolCallFormat>().unwrap().into_parser();
        assert!(capability(&defs, Some(&required), Some(parser.as_ref()), true).is_ok());
    }
    assert!(capability(&defs, Some(&required), None, true).is_ok());
}
