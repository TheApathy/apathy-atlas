// SPDX-License-Identifier: AGPL-3.0-only

use serde_json::{Value, json};

use super::{validate_call, validate_schema};
use crate::tool_parser::{ToolCall, ToolDefinition};

fn checked(schema: Value, args: Value) -> Result<ToolCall, String> {
    let tools: Vec<ToolDefinition> = serde_json::from_value(json!([{
        "type": "function", "function": {"name": "inspect", "parameters": schema}
    }]))
    .unwrap();
    let call = serde_json::from_value(json!({"id": "call_cpu", "type": "function",
        "function": {"name": "inspect", "arguments": args.to_string()}}))
    .unwrap();
    validate_call(&call, &tools)
}

fn object(value: Value) -> Value {
    json!({"type": "object", "properties": {"value": value},
        "required": ["value"], "additionalProperties": false})
}

#[test]
fn nested_required_type_and_closed_keys_are_enforced_without_repairs() {
    let schema = object(
        json!({"type": "object", "properties": {"count": {"type": "integer"}},
        "required": ["count"], "additionalProperties": false}),
    );
    for raw in [
        "{}",
        "{\"count\":1.5}",
        "{\"count\":\"7\"}",
        "{\"count\":7,\"extra\":0}",
    ] {
        assert!(
            checked(schema.clone(), json!({"value": raw})).is_err(),
            "{raw}"
        );
    }
    assert!(checked(schema, json!({"value": "{\"count\":7}"})).is_ok());
}

#[test]
fn array_items_are_recursively_validated() {
    let schema = object(json!({"type": "array", "items": {"type": "integer"}}));
    for raw in ["{}", "[1,1.5]", "[1,\"2\"]"] {
        assert!(
            checked(schema.clone(), json!({"value": raw})).is_err(),
            "{raw}"
        );
    }
    assert!(checked(schema, json!({"value": "[1,2]"})).is_ok());
}

#[test]
fn schema_valued_additional_properties_are_checked() {
    let schema = object(json!({"type": "object", "additionalProperties": {"type": "boolean"}}));
    assert!(checked(schema.clone(), json!({"value": "{\"extra\":true}"})).is_ok());
    assert!(checked(schema, json!({"value": "{\"extra\":\"true\"}"})).is_err());
}

#[test]
fn enum_and_const_constrain_converted_values_and_raw_strings() {
    assert!(
        checked(
            object(json!({"type": "integer", "enum": [1, 2]})),
            json!({"value": "3"})
        )
        .is_err()
    );
    assert!(
        checked(
            object(json!({"type": "integer", "const": 1})),
            json!({"value": "1.0"})
        )
        .is_ok()
    );
    assert!(
        checked(
            object(json!({"type": "string", "const": "true"})),
            json!({"value": "true"})
        )
        .is_ok()
    );
    assert!(
        checked(
            object(json!({"type": "string", "enum": [" x "]})),
            json!({"value": "x"})
        )
        .is_err()
    );
}

#[test]
fn numeric_enum_does_not_round_distinct_large_integers_together() {
    let schema = object(json!({"type": "number", "enum": [9007199254740993u64]}));
    assert!(checked(schema.clone(), json!({"value": "9007199254740992.0"})).is_err());
    assert!(checked(schema, json!({"value": "9007199254740993"})).is_ok());
}

#[test]
fn lossy_fractional_literals_cannot_become_integers_even_inside_objects() {
    let schema = object(json!({"type": "object", "properties": {"count": {"type": "integer"}}}));
    assert!(checked(schema, json!({"value": "{\"count\":9007199254740993.5}"})).is_err());
    assert!(
        checked(
            object(json!({"type": "integer"})),
            json!({"value": "1.0000000000000000000001"})
        )
        .is_err()
    );
    assert!(
        checked(
            object(json!({"type": "object"})),
            json!({"value": "{\"text\":\"9007199254740993.5\"}"})
        )
        .is_ok()
    );
}

#[test]
fn non_json_boolean_and_nonfinite_number_are_not_repaired() {
    for raw in ["yes", "1", "True"] {
        assert!(checked(object(json!({"type": "boolean"})), json!({"value": raw})).is_err());
    }
    for raw in ["NaN", "Infinity", "1e999"] {
        assert!(checked(object(json!({"type": "number"})), json!({"value": raw})).is_err());
    }
}

#[test]
fn defaults_are_annotations_not_missing_argument_fillers() {
    assert!(
        checked(
            object(json!({"type": "string", "default": "invented"})),
            json!({})
        )
        .is_err()
    );
}

#[test]
fn unsupported_material_keywords_are_rejected_even_in_unused_properties() {
    for (key, value) in [
        ("minimum", json!(0)),
        ("pattern", json!("x")),
        ("$ref", json!("#/$defs/value")),
        ("oneOf", json!([{"type": "string"}])),
        ("minLength", json!(1)),
        ("format", json!("email")),
    ] {
        let mut schema = json!({"type": "object", "properties": {"unused": {"type": "string"}}});
        schema["properties"]["unused"][key] = value;
        assert!(
            validate_schema(&schema).is_err(),
            "{key} was silently ignored"
        );
    }
    assert!(validate_schema(&json!({"type": ["string", "null"]})).is_err());
}

#[test]
fn malformed_schema_shapes_fail_closed() {
    for schema in [
        json!({"properties": []}),
        json!({"required": [1]}),
        json!({"required": ["value", "value"]}),
        json!({"enum": []}),
        json!({"items": []}),
        json!({"additionalProperties": 7}),
    ] {
        assert!(validate_schema(&schema).is_err(), "{schema}");
    }
}

#[test]
fn boolean_schemas_and_null_are_explicit() {
    assert!(checked(object(json!(false)), json!({"value": "anything"})).is_err());
    assert!(checked(object(json!({"type": "null"})), json!({"value": "null"})).is_ok());
    assert!(checked(object(json!({"type": "null"})), json!({"value": "false"})).is_err());
}
