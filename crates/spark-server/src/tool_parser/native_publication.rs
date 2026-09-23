// SPDX-License-Identifier: AGPL-3.0-only

//! Native GLM publication is validation, not repair. A call is returned only
//! after exact-name admission, lossless schema-directed conversion and full
//! supported-schema validation. Legacy parser paths do not use this module.

mod numbers;
mod schema;
#[cfg(test)]
mod tests;

use serde_json::Value;

use super::{ToolCall, ToolChoice, ToolDefinition, ValidatedToolCalls};

/// The same capability check is used before scheduling and at publication.
/// This is a bounded subset, not certification of arbitrary JSON Schema.
pub(crate) fn validate_schema(schema: &Value) -> Result<(), String> {
    schema::validate_schema(schema)
}

pub(crate) fn validate_call(call: &ToolCall, tools: &[ToolDefinition]) -> Result<ToolCall, String> {
    let mut matches = tools
        .iter()
        .filter(|tool| tool.function.name == call.function.name);
    let tool = matches.next().ok_or_else(|| {
        format!(
            "native GLM called an unallowed tool: {}",
            call.function.name
        )
    })?;
    if matches.next().is_some() || tool.tool_type != "function" || call.call_type != "function" {
        return Err("native GLM tool identity is ambiguous or not a function".into());
    }
    let args: serde_json::Map<String, Value> = serde_json::from_str(&call.function.arguments)
        .map_err(|_| "native GLM arguments must be a JSON object of raw XML strings")?;
    let args = if let Some(schema) = tool.function.parameters.as_ref() {
        validate_schema(schema)?;
        schema::convert_arguments(&args, schema)?
    } else {
        if !args.values().all(Value::is_string) {
            return Err("native GLM arguments must originate as raw XML strings".into());
        }
        Value::Object(args)
    };
    let mut validated = call.clone();
    validated.function.arguments = serde_json::to_string(&args)
        .map_err(|e| format!("cannot serialize validated native GLM arguments: {e}"))?;
    Ok(validated)
}

pub(crate) fn validate_calls(
    calls: Vec<ToolCall>,
    tools: &[ToolDefinition],
    choice: Option<&ToolChoice>,
) -> ValidatedToolCalls {
    let mut result = ValidatedToolCalls {
        valid: Vec::new(),
        errors: Vec::new(),
    };
    for call in calls {
        let allowed = match choice {
            Some(ToolChoice::Specific { function }) => function.name == call.function.name,
            Some(ToolChoice::Mode(mode)) => matches!(mode.as_str(), "auto" | "required"),
            None => true,
        };
        let validated = if allowed {
            validate_call(&call, tools)
        } else {
            Err("native GLM call violates the requested tool choice".into())
        };
        match validated {
            Ok(call) => result.valid.push(call),
            Err(error) => result.errors.push(error),
        }
    }
    result
}
