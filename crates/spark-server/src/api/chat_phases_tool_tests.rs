// SPDX-License-Identifier: AGPL-3.0-only
use super::validate_input;
use serde_json::{Value, json};

fn request(tools: Value, choice: Value) -> crate::ir::ChatRequest {
    serde_json::from_value::<crate::openai::ChatCompletionRequest>(json!({
        "model":"glm53-flash", "messages":[{"role":"user","content":"OK"}],
        "max_tokens":16, "tools":tools, "tool_choice":choice
    }))
    .unwrap()
    .into()
}
fn tool(name: &str, parameters: Value) -> Value {
    json!({"type":"function", "function":{"name":name,"parameters":parameters}})
}
fn reject(req: &crate::ir::ChatRequest) {
    assert_eq!(
        validate_input(req)
            .err()
            .expect("invalid tool request must reject before scheduling")
            .status(),
        axum::http::StatusCode::BAD_REQUEST
    );
}

#[test]
fn tool_admission_specific_requires_exact_declared_nonempty_name() {
    for name in ["", " ", "not_declared", "Record_Note"] {
        reject(&request(
            json!([tool("record_note", json!({"type":"object"}))]),
            json!({"type":"function","function":{"name":name}}),
        ));
    }
    reject(&request(
        json!([]),
        json!({"type":"function","function":{"name":"record_note"}}),
    ));
}

#[test]
fn tool_admission_schema_root_and_properties_are_objects() {
    for schema in [
        json!([]),
        json!(false),
        json!(null),
        json!({"type":"array"}),
        json!({"type":"object","properties":[]}),
    ] {
        reject(&request(
            json!([tool("record_note", schema)]),
            json!("required"),
        ));
    }
}

#[test]
fn tool_admission_names_and_types_must_be_unambiguous() {
    let t = tool("record_note", json!({"type":"object"}));
    reject(&request(json!([t, t]), json!("required")));
    for name in ["", " ", "bad<name"] {
        reject(&request(
            json!([tool(name, json!({"type":"object"}))]),
            json!("auto"),
        ));
    }
    let mut t = tool("record_note", json!({"type":"object"}));
    t["type"] = json!("not_function");
    reject(&request(json!([t]), json!("auto")));
}

#[test]
fn tool_admission_required_and_additional_properties_shape_is_checked() {
    for schema in [
        json!({"type":"object","required":"x"}),
        json!({"type":"object","required":[7]}),
        json!({"type":"object","required":["x","x"]}),
        json!({"type":"object","additionalProperties":7}),
    ] {
        reject(&request(
            json!([tool("record_note", schema)]),
            json!("required"),
        ));
    }
}

#[test]
fn tool_admission_valid_modes_and_nested_schema_remain_accepted() {
    let t = tool(
        "record_note",
        json!({"type":"object","properties":{"note":{"type":"string"},"meta":{"type":"object","additionalProperties":true}},"required":["note"],"additionalProperties":false}),
    );
    for choice in [
        json!("auto"),
        json!("none"),
        json!("required"),
        json!({"type":"function","function":{"name":"record_note"}}),
    ] {
        assert!(validate_input(&request(json!([t.clone()]), choice)).is_ok());
    }
    assert!(validate_input(&request(json!([]), json!("none"))).is_ok());
    assert!(
        validate_input(&request(
            json!([{"type":"function","function":{"name":"no_args"}}]),
            json!("required")
        ))
        .is_ok()
    );
}
