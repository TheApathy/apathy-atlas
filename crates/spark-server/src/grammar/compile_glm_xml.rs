// SPDX-License-Identifier: AGPL-3.0-only

//! Native GLM framing and name/key constraints. Argument values stay RAW:
//! this compiler does not enforce their schema types, required properties or
//! uniqueness. Parser coercion and request validation own those checks.
//! In particular, a declared string may be empty, whitespace or JSON-looking.

use std::collections::HashSet;

use serde_json::{Value, json};
use xgrammar::CompiledGrammar;

use super::{GrammarEngine, GrammarError};
use crate::tool_parser::ToolDefinition;

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
const VALUE_CLOSE: &str = "</arg_value>";

impl GrammarEngine {
    /// Compile GLM's `<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE`
    /// wire, never Qwen's function/parameter envelope. `use_triggers=false`
    /// requires a call; specific choice is represented by the caller's filtered
    /// tool set. Invalid definitions fail the whole compile, never skip a tool.
    pub fn compile_glm_xml_tools(
        &mut self,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        if tools.is_empty() {
            return Err(GrammarError::NoTools);
        }
        let mut names = HashSet::new();
        let mut tags = Vec::with_capacity(tools.len());
        for tool in tools {
            let name = &tool.function.name;
            if tool.tool_type != "function" || !native_name(name) {
                return Err(invalid(
                    "GLM tool must be a function with a native ASCII name",
                ));
            }
            if !names.insert(name.as_str()) {
                return Err(invalid("GLM tool names must be unique"));
            }
            let body = native_body(tool.function.parameters.as_ref())?;
            tags.push(json!({
                "type": "tag", "begin": format!("{OPEN}{name}"),
                "content": {"type": "grammar", "grammar": body}, "end": CLOSE,
            }));
        }
        // One shared trigger arms masks immediately at the native opener,
        // including with prefix-sharing names. Required/specific must emit one
        // complete call before stopping; auto may emit prose or multiple calls.
        self.compile_structural_tag_raw(&[OPEN.to_owned()], &tags, !use_triggers, !use_triggers)
    }
}

fn invalid(message: &str) -> GrammarError {
    GrammarError::InvalidSchema(message.to_owned())
}

fn native_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
}

/// Return an exact key alternation only for an explicitly closed schema.
/// Otherwise the wire permits nonempty literal keys without structural angle
/// brackets. JSON Schema defaults additionalProperties to open; no implicit
/// closed-schema rewrite or global schema sanitizer is used here.
fn key_rule(parameters: Option<&Value>) -> Result<Option<String>, GrammarError> {
    let Some(parameters) = parameters else {
        return Ok(Some("[^<>]+".to_owned()));
    };
    let object = parameters
        .as_object()
        .ok_or_else(|| invalid("GLM tool parameters must be an object schema"))?;
    if object
        .get("type")
        .is_some_and(|kind| kind.as_str() != Some("object"))
    {
        return Err(invalid("GLM tool parameter root must have object type"));
    }
    let properties = match object.get("properties") {
        None => None,
        Some(value) => Some(
            value
                .as_object()
                .ok_or_else(|| invalid("GLM tool properties must be an object"))?,
        ),
    };
    if properties.is_some_and(|props| {
        props
            .keys()
            .any(|key| key.is_empty() || key.contains(['<', '>']))
    }) {
        return Err(invalid(
            "GLM argument keys must be nonempty and contain no angle brackets",
        ));
    }
    match object.get("additionalProperties") {
        Some(Value::Bool(false)) => {
            let mut literals = Vec::new();
            if let Some(properties) = properties {
                for key in properties.keys() {
                    literals.push(
                        serde_json::to_string(key)
                            .map_err(|error| GrammarError::InvalidSchema(error.to_string()))?,
                    );
                }
            }
            if literals.is_empty() {
                Ok(None)
            } else {
                Ok(Some(literals.join(" | ")))
            }
        }
        None | Some(Value::Bool(true)) | Some(Value::Object(_)) => Ok(Some("[^<>]+".to_owned())),
        Some(_) => Err(invalid(
            "GLM additionalProperties must be boolean or an object schema",
        )),
    }
}

fn native_body(parameters: Option<&Value>) -> Result<String, GrammarError> {
    let Some(keys) = key_rule(parameters)? else {
        return Ok("root ::= \"\"\n".to_owned());
    };
    Ok(format!(
        "root ::= param*\n\
         param ::= \"<arg_key>\" key \"</arg_key><arg_value>\" value0 \"{VALUE_CLOSE}\"\n\
         key ::= {keys}\n{}",
        raw_value_rules(),
    ))
}

/// Prefix-state automaton for this fixed delimiter (its only '<' is first).
/// Every proper prefix can end a value, or restart when another '<' appears.
/// Completing the literal close is the one excluded transition. This avoids
/// swallowing an overlapping opener and permits strings ending in '<' or
/// '</arg_val'. It deliberately does not alter the existing Qwen ladder.
fn raw_value_rules() -> String {
    let close = VALUE_CLOSE.as_bytes();
    let mut rules = String::new();
    for (index, &next) in close.iter().enumerate() {
        let mut alternatives = vec!["\"\"".to_owned()];
        let mismatch = if index == 0 {
            "[^<]".to_owned()
        } else {
            alternatives.push("\"<\" value1".to_owned());
            format!("[^<{}]", char::from(next))
        };
        alternatives.push(format!("{mismatch} value0"));
        if index + 1 < close.len() {
            alternatives.push(format!("\"{}\" value{}", char::from(next), index + 1));
        }
        rules.push_str(&format!("value{index} ::= {}\n", alternatives.join(" | ")));
    }
    rules
}
