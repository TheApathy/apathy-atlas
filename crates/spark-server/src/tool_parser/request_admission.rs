// SPDX-License-Identifier: AGPL-3.0-only
//! Request-only tool contract checks before prompt mutation or inference.
use super::{ToolCallParser, ToolChoice, ToolDefinition};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use std::collections::HashSet;

// Preserve explicit JSON null so invalid schemas are not erased as omission.
pub(super) fn parameters<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Value>, D::Error> {
    Value::deserialize(d).map(Some)
}

pub(crate) fn validate(
    tools: &[ToolDefinition],
    choice: Option<&ToolChoice>,
) -> Result<(), String> {
    let mut names = HashSet::new();
    for tool in tools {
        let name = &tool.function.name;
        if tool.tool_type != "function" || !super::is_tool_name_component(name) {
            return Err("tools must contain functions with nonempty ASCII names".into());
        }
        if !names.insert(name.as_str()) {
            return Err(format!("duplicate tool name: {name}"));
        }
        if let Some(schema) = &tool.function.parameters {
            validate_root(schema)?;
        }
    }
    match choice {
        Some(ToolChoice::Mode(mode)) if !["auto", "none", "required"].contains(&mode.as_str()) => {
            Err("invalid tool_choice mode".into())
        }
        Some(ToolChoice::Mode(mode)) if mode == "required" && tools.is_empty() => {
            Err("tool_choice required needs at least one tool".into())
        }
        Some(ToolChoice::Specific { function }) if !names.contains(function.name.as_str()) => {
            Err("specific tool_choice must name an exactly declared tool".into())
        }
        _ => Ok(()),
    }
}

fn validate_root(schema: &Value) -> Result<(), String> {
    let root = schema
        .as_object()
        .ok_or("tool parameters must be an object schema")?;
    if root
        .get("type")
        .is_some_and(|v| v.as_str() != Some("object"))
    {
        return Err("tool parameter root type must be object".into());
    }
    if root.get("properties").is_some_and(|v| !v.is_object()) {
        return Err("tool schema properties must be an object".into());
    }
    if let Some(value) = root.get("required") {
        let required = value
            .as_array()
            .ok_or("tool schema required must be a string array")?;
        let mut unique = HashSet::new();
        for key in required {
            let key = key
                .as_str()
                .ok_or("tool schema required must contain strings")?;
            if !unique.insert(key) {
                return Err("tool schema required contains a duplicate key".into());
            }
        }
    }
    if root
        .get("additionalProperties")
        .is_some_and(|v| !v.is_boolean() && !v.is_object())
    {
        return Err("tool schema additionalProperties must be boolean or an object schema".into());
    }
    Ok(())
}

pub(crate) fn capability(
    tools: &[ToolDefinition],
    choice: Option<&ToolChoice>,
    parser: Option<&dyn ToolCallParser>,
    grammar_disabled: bool,
) -> Result<(), String> {
    validate(tools, choice)?;
    if tools.is_empty() || choice.is_some_and(ToolChoice::is_none) {
        return Ok(());
    }
    let parser = parser.ok_or("this runtime has no tool-call parser")?;
    let required = matches!(choice, Some(ToolChoice::Specific { .. }))
        || matches!(choice,Some(ToolChoice::Mode(m)) if m=="required");
    if required && (grammar_disabled || !parser.has_tool_grammar()) {
        return Err(
            "this runtime cannot enforce the requested required or specific tool choice".into(),
        );
    }
    if parser.name() == "glm_xml" {
        if grammar_disabled {
            return Err("native GLM tools require their registered framing grammar".into());
        }
        for tool in tools {
            if let Some(schema) = &tool.function.parameters {
                super::native_publication::validate_schema(schema)?;
                if schema
                    .get("properties")
                    .and_then(Value::as_object)
                    .is_some_and(|properties| {
                        properties
                            .keys()
                            .any(|key| key.trim().is_empty() || key.contains(['<', '>']))
                    })
                {
                    return Err(
                        "native GLM argument names must be nonempty and contain no angle brackets"
                            .into(),
                    );
                }
            }
        }
    }
    Ok(())
}
