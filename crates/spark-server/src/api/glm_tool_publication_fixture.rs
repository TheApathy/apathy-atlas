// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only fixtures for the real blocking and complete-stream publication
//! boundaries. No scheduler/model mock, schema repair, or replacement parser.
//! all-models port: this tree has no AppState test fixture or IR request
//! type, so the blocking and streaming tests drive the publication helpers
//! that `build_choice_message` / `handle_complete_tool_call` call for glm_xml.

use serde_json::{Value, json};

use crate::tool_parser::{ToolChoice, ToolDefinition};

pub(super) struct Case {
    pub tools: Vec<ToolDefinition>,
    pub choice: ToolChoice,
    pub wire: String,
    pub expected: Option<Value>,
}

fn tool(name: &str, schema: Value) -> ToolDefinition {
    serde_json::from_value(json!({
        "type": "function", "function": {"name": name, "parameters": schema}
    }))
    .unwrap()
}

fn schema(kind: &str) -> Value {
    json!({"type": "object", "properties": {"value": {"type": kind}},
        "required": ["value"], "additionalProperties": false})
}

pub(super) fn wire(name: &str, args: &[(&str, &str)]) -> String {
    let mut body = format!("<tool_call>{name}");
    for (key, value) in args {
        body.push_str(&format!(
            "<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
        ));
    }
    body.push_str("</tool_call>");
    body
}

pub(super) fn case(name: &str) -> Case {
    let mut case = Case {
        tools: vec![tool("inspect", schema("string"))],
        choice: ToolChoice::Mode("required".into()),
        wire: wire("inspect", &[("value", "ok")]),
        expected: None,
    };
    match name {
        "fractional_integer" => {
            case.tools = vec![tool("inspect", schema("integer"))];
            case.wire = wire("inspect", &[("value", "1.5")]);
        }
        "wrong_object" => {
            case.tools = vec![tool("inspect", schema("object"))];
            case.wire = wire("inspect", &[("value", "[]")]);
        }
        "missing_required_integer" => {
            case.tools = vec![tool("inspect", schema("integer"))];
            case.wire = wire("inspect", &[]);
        }
        "missing_required_string" => case.wire = wire("inspect", &[]),
        "unknown_name" => case.wire = wire("unrelated_unknown_tool", &[("value", "ok")]),
        "fuzzy_name" => {
            case.tools = vec![tool("get_weather", schema("string"))];
            case.wire = wire("weather", &[("value", "ok")]);
        }
        "specific_wrong_name" => {
            case.tools.push(tool("inspect_more", schema("string")));
            case.choice = serde_json::from_value(json!({"function": {"name": "inspect"}})).unwrap();
            case.wire = wire("inspect_more", &[("value", "ok")]);
        }
        "closed_extra_key" => {
            case.wire = wire("inspect", &[("value", "ok"), ("extra", "must not publish")]);
        }
        "renamed_key" => case.wire = wire("inspect", &[("VALUE", "ok")]),
        "raw_whitespace" | "raw_empty" => {
            let value = if name == "raw_empty" { "" } else { " \t\n  " };
            // This schema expressly permits empty strings. The legacy
            // description backfill must not invent replacement bytes.
            case.tools = vec![tool(
                "inspect",
                json!({"type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"], "additionalProperties": false}),
            )];
            case.wire = wire("inspect", &[("description", value)]);
            case.expected = Some(json!({"description": value}));
        }
        "raw_true_string" => {
            case.wire = wire("inspect", &[("value", "true")]);
            case.expected = Some(json!({"value": "true"}));
        }
        "valid_typed_arguments" => {
            case.tools = vec![tool(
                "inspect",
                json!({"type": "object", "properties": {
                "count": {"type": "integer"}, "enabled": {"type": "boolean"},
                "ratio": {"type": "number"}, "object": {"type": "object"},
                "items": {"type": "array", "items": {"type": "integer"}},
                "label": {"type": "string"}},
                "required": ["count", "enabled", "ratio", "object", "items", "label"],
                "additionalProperties": false}),
            )];
            case.wire = wire(
                "inspect",
                &[
                    ("count", "7"),
                    ("enabled", "true"),
                    ("ratio", "1.5"),
                    ("object", "{\"nested\":2}"),
                    ("items", "[1,2]"),
                    ("label", "true"),
                ],
            );
            case.expected = Some(json!({"count": 7, "enabled": true, "ratio": 1.5,
                "object": {"nested": 2}, "items": [1, 2], "label": "true"}));
        }
        _ => panic!("unknown publication fixture {name}"),
    }
    case
}
