// SPDX-License-Identifier: AGPL-3.0-only

//! Deliberately bounded JSON Schema subset. Unsupported assertions are errors,
//! including in unused properties; annotations never supply argument values.

use serde_json::{Map, Number, Value};

const MAX_DEPTH: usize = 64;

pub(super) fn validate_schema(schema: &Value) -> Result<(), String> {
    inspect(schema, 0)
}

fn inspect(schema: &Value, depth: usize) -> Result<(), String> {
    if depth > MAX_DEPTH {
        return Err("native tool schema exceeds depth 64".into());
    }
    if schema.is_boolean() {
        return Ok(());
    }
    let object = schema
        .as_object()
        .ok_or("native tool schema must be an object or boolean")?;
    for (key, value) in object {
        match key.as_str() {
            "type" => match value.as_str() {
                Some("string" | "integer" | "number" | "boolean" | "null" | "object" | "array") => {
                }
                _ => {
                    return Err(
                        "unsupported native tool schema type (type unions are not supported)"
                            .into(),
                    );
                }
            },
            "properties" => {
                for child in value
                    .as_object()
                    .ok_or("schema properties must be an object")?
                    .values()
                {
                    inspect(child, depth + 1)?;
                }
            }
            "required" => {
                let mut seen = std::collections::HashSet::new();
                for name in value.as_array().ok_or("schema required must be an array")? {
                    let name = name
                        .as_str()
                        .ok_or("schema required entries must be strings")?;
                    if !seen.insert(name) {
                        return Err("schema required contains a duplicate entry".into());
                    }
                }
            }
            "additionalProperties" | "items" => inspect(value, depth + 1)?,
            "enum" => {
                if value.as_array().is_none_or(Vec::is_empty) {
                    return Err("schema enum must be a nonempty array".into());
                }
            }
            "const" | "default" | "examples" => {}
            "title" | "description" | "$comment" => {
                if !value.is_string() {
                    return Err(format!("schema annotation {key} must be a string"));
                }
            }
            _ => return Err(format!("unsupported native tool schema keyword: {key}")),
        }
    }
    Ok(())
}

/// Native XML has exactly one ambiguity boundary: each argument's raw bytes.
/// Parse JSON only for an explicit non-string type; never recursively convert
/// strings inside an already-parsed JSON object/array.
pub(super) fn convert_arguments(raw: &Map<String, Value>, schema: &Value) -> Result<Value, String> {
    let mut out = Map::new();
    for (key, value) in raw {
        let text = value
            .as_str()
            .ok_or("native tool argument did not originate as a raw XML string")?;
        let property = schema
            .get("properties")
            .and_then(|p| p.get(key))
            .or_else(|| schema.get("additionalProperties"));
        let value = match property.and_then(|p| p.get("type")).and_then(Value::as_str) {
            Some("integer" | "number" | "boolean" | "null" | "object" | "array") => {
                super::numbers::parse(text)
                    .map_err(|error| format!("native argument {key}: {error}"))?
            }
            _ => value.clone(),
        };
        out.insert(key.clone(), value);
    }
    let out = Value::Object(out);
    check(&out, schema, 0)?;
    Ok(out)
}

fn check(value: &Value, schema: &Value, depth: usize) -> Result<(), String> {
    if depth > MAX_DEPTH {
        return Err("native tool argument exceeds schema depth 64".into());
    }
    if let Some(allowed) = schema.as_bool() {
        return if allowed {
            Ok(())
        } else {
            Err("native tool argument rejected by false schema".into())
        };
    }
    if let Some(kind) = schema.get("type").and_then(Value::as_str) {
        let matches = match kind {
            "string" => value.is_string(),
            "integer" => {
                value.is_i64()
                    || value.is_u64()
                    || value
                        .as_f64()
                        .is_some_and(|v| v.is_finite() && v.fract() == 0.0)
            }
            "number" => value.is_number(),
            "boolean" => value.is_boolean(),
            "null" => value.is_null(),
            "object" => value.is_object(),
            "array" => value.is_array(),
            _ => false,
        };
        if !matches {
            return Err(format!(
                "native tool argument does not match declared {kind} type"
            ));
        }
    }
    if let Some(choices) = schema.get("enum").and_then(Value::as_array)
        && !choices.iter().any(|choice| equal(value, choice))
    {
        return Err("native tool argument is outside schema enum".into());
    }
    if let Some(constant) = schema.get("const")
        && !equal(value, constant)
    {
        return Err("native tool argument does not equal schema const".into());
    }
    if let Some(object) = value.as_object() {
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for name in required {
                let name = name.as_str().ok_or("invalid required schema entry")?;
                if !object.contains_key(name) {
                    return Err(format!(
                        "native tool call is missing required argument: {name}"
                    ));
                }
            }
        }
        for (name, child) in object {
            if let Some(child_schema) = schema.get("properties").and_then(|p| p.get(name)) {
                check(child, child_schema, depth + 1)?;
            } else if let Some(additional) = schema.get("additionalProperties") {
                check(child, additional, depth + 1)?;
            }
        }
    }
    if let Some(array) = value.as_array()
        && let Some(items) = schema.get("items")
    {
        for child in array {
            check(child, items, depth + 1)?;
        }
    }
    Ok(())
}

// JSON Schema numeric equality treats 1 and 1.0 alike, but must not merge
// adjacent u64 values through a lossy f64 conversion.
fn number_equal(a: &Number, b: &Number) -> bool {
    if a.is_f64() && b.is_f64() {
        return a.as_f64() == b.as_f64();
    }
    if !a.is_f64() && !b.is_f64() {
        return a == b;
    }
    let (float, integer) = if a.is_f64() { (a, b) } else { (b, a) };
    let Some(float) = float.as_f64().filter(|f| f.is_finite() && f.fract() == 0.0) else {
        return false;
    };
    if let Some(integer) = integer.as_u64() {
        return (0.0..18446744073709551616.0).contains(&float) && float as u64 == integer;
    }
    integer.as_i64().is_some_and(|integer| {
        (-9223372036854775808.0..9223372036854775808.0).contains(&float) && float as i64 == integer
    })
}

fn equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => number_equal(a, b),
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| equal(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, a)| b.get(key).is_some_and(|b| equal(a, b)))
        }
        _ => a == b,
    }
}
