// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::super::*;

fn tool(name: &str, required: &str) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: name.into(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {required: {"type": "string"}},
                "required": [required]
            })),
        },
    }
}

fn call(name: &str, arguments: &str) -> ToolCall {
    ToolCall {
        id: "call_0".into(),
        call_type: "function".into(),
        function: FunctionCall {
            name: name.into(),
            arguments: arguments.into(),
        },
    }
}

#[test]
fn missing_required_value_is_not_fabricated() {
    let weather = tool("get_weather", "city");
    let mut calls = vec![call("get_weather", "{}")];
    backfill_required_params(&mut calls, std::slice::from_ref(&weather));

    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert!(
        args.get("city").is_none(),
        "Atlas must not author `city`: {args}"
    );
    let issue = assess_tool_call(&calls[0], std::slice::from_ref(&weather)).unwrap_err();
    assert!(matches!(issue, ToolCallIssue::MissingParam(_)));
}

#[test]
fn model_authored_empty_string_is_preserved() {
    let weather = tool("get_weather", "city");
    let mut calls = vec![call("get_weather", r#"{"city":""}"#)];
    backfill_required_params(&mut calls, std::slice::from_ref(&weather));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "");
}

#[test]
fn derivable_description_is_still_filled() {
    let bash = ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "bash".into(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["command", "description"]
            })),
        },
    };
    let mut calls = vec![call("bash", r#"{"command":"ls /tmp"}"#)];
    backfill_required_params(&mut calls, std::slice::from_ref(&bash));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["description"], "Run: ls /tmp");
}

#[test]
fn ordinary_missing_param_is_attached_with_feedback() {
    let weather = tool("get_weather", "city");
    let result = validate_tool_calls(
        vec![call("get_weather", "{}")],
        std::slice::from_ref(&weather),
    );
    assert_eq!(
        result.valid.len(),
        1,
        "ordinary call remains client-visible"
    );
    assert_eq!(result.valid[0].function.arguments, "{}");
    assert_eq!(result.errors.len(), 1);
    assert!(result.errors[0].contains("city"));
}

#[test]
fn omitted_write_path_is_withheld() {
    let write = tool("write", "filePath");
    let result = validate_tool_calls(vec![call("write", "{}")], std::slice::from_ref(&write));
    assert!(result.valid.is_empty());
    assert_eq!(result.errors.len(), 1);
    assert!(result.errors[0].contains("filePath"));
}

#[test]
fn omitted_shell_command_is_withheld() {
    let exec = tool("exec", "command");
    let result = validate_tool_calls(vec![call("exec", "{}")], std::slice::from_ref(&exec));
    assert!(result.valid.is_empty());
    assert_eq!(result.errors.len(), 1);
    assert!(result.errors[0].contains("command"));
}

#[test]
fn critical_omission_wins_regardless_of_required_array_order() {
    for (name, first, critical) in [
        ("write", "content", "filePath"),
        ("exec", "description", "command"),
    ] {
        let definition = ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: name.into(),
                description: None,
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        first: {"type": "string"},
                        critical: {"type": "string"}
                    },
                    "required": [first, critical]
                })),
            },
        };
        let result = validate_tool_calls(vec![call(name, "{}")], std::slice::from_ref(&definition));
        assert!(result.valid.is_empty(), "{name} must be withheld");
        assert!(result.errors[0].contains(critical), "{:?}", result.errors);
    }
}

fn parsed_args(input: &str) -> serde_json::Value {
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1, "expected one call from {input:?}");
    serde_json::from_str(&calls[0].function.arguments).unwrap()
}

#[test]
fn qwen_attribute_drift_is_recovered_narrowly() {
    let args = parsed_args(
        "<function=get_weather>\n<parameter city=\"Santiago\">Chile</parameter>\n</function>",
    );
    assert_eq!(args["city"], "Santiago");

    let args = parsed_args(
        "<function=get_weather>\n<parameter name=\"city\">Santiago</parameter>\n</function>",
    );
    assert_eq!(args["city"], "Santiago");
}

#[test]
fn strict_parameter_wins_over_attribute_salvage() {
    let args = parsed_args(
        "<function=get_weather>\n<parameter=city>Kyoto</parameter>\n\
         <parameter city=\"Osaka\">x</parameter>\n</function>",
    );
    assert_eq!(args["city"], "Kyoto");
}

#[test]
fn attribute_salvage_rejects_ambiguous_shapes() {
    for body in [
        "<parameter city=Santiago>x</parameter>",
        "<parameter city=\"Santiago\" units=\"c\">x</parameter>",
        "<parameter 9city=\"Santiago\">x</parameter>",
    ] {
        let args = parsed_args(&format!("<function=get_weather>\n{body}\n</function>"));
        assert!(
            args.as_object().unwrap().is_empty(),
            "must not guess: {args}"
        );
    }
}

#[test]
fn attribute_salvage_does_not_cross_function_boundary() {
    let first = parse_qwen3_coder_call(
        "<function=get_weather>\n<parameter=city>Kyoto</parameter>\n</function>\n\
         <function=get_time>\n<parameter zone=\"JST\">x</parameter>\n</function>",
        0,
    )
    .unwrap();
    let args: serde_json::Value = serde_json::from_str(&first.function.arguments).unwrap();
    assert_eq!(args["city"], "Kyoto");
    assert!(args.get("zone").is_none());
}
