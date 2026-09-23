// SPDX-License-Identifier: AGPL-3.0-only
//! Request-only native GLM tool contract checks before prompt mutation or
//! inference.
//!
//! all-models port: these checks run only when the active parser is
//! `glm_xml`. Every other parser keeps this tree's existing request
//! validation (`chat_phases::validate_input`) byte-for-byte; the GLM branch's
//! global tightening (explicit-null parameters, duplicate names, required
//! without grammar for every parser) is intentionally not applied here.
use super::{ToolCallParser, ToolChoice, ToolDefinition};
use serde_json::Value;
use std::collections::HashSet;

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

/// Admission for the native GLM tool contract. `Ok(())` for any other parser
/// (or no parser): their requests keep the existing validation path.
pub(crate) fn capability(
    tools: &[ToolDefinition],
    choice: Option<&ToolChoice>,
    parser: Option<&dyn ToolCallParser>,
    grammar_disabled: bool,
) -> Result<(), String> {
    if !parser.is_some_and(|parser| parser.name() == "glm_xml") {
        return Ok(());
    }
    validate(tools, choice)?;
    if tools.is_empty() || choice.is_some_and(ToolChoice::is_none) {
        return Ok(());
    }
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
    Ok(())
}

#[cfg(test)]
#[path = "request_admission_tests.rs"]
mod tests;
