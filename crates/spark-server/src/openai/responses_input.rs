// SPDX-License-Identifier: AGPL-3.0-only

use super::{IncomingMessage, ParsedContent};
use serde_json::Value;

impl IncomingMessage {
    /// Compatibility helper for callers that explicitly represent unsupported items as None.
    /// Request/replay lowering MUST use the checked form to propagate malformed content.
    pub fn from_responses_input_item(v: &Value) -> Option<Self> {
        Self::try_from_responses_input_item(v).ok().flatten()
    }

    pub fn try_from_responses_input_item(v: &Value) -> Result<Option<Self>, String> {
        let obj = v
            .as_object()
            .ok_or("Responses input item must be an object")?;
        let string = |key: &str| -> Result<&str, String> {
            obj.get(key)
                .and_then(Value::as_str)
                .ok_or_else(|| format!("Responses item requires string '{key}'"))
        };
        let kind = match obj.get("type") {
            None => "message",
            Some(v) => v.as_str().ok_or("Responses item type must be a string")?,
        };
        let mut message = Self::synthetic_user_text(String::new());
        match kind {
            "message" => {
                message.role = if obj.contains_key("role") {
                    string("role")?
                } else {
                    "user"
                }
                .to_string();
                let content = obj
                    .get("content")
                    .ok_or("Responses message requires content")?;
                if message.role == "developer" {
                    message.role = "user".into();
                }
                message.content = ParsedContent::parse(content.clone(), true)?;
            }
            "function_call" => {
                let name = string("name")?.to_string();
                let arguments = if obj.contains_key("arguments") {
                    string("arguments")?
                } else {
                    "{}"
                }
                .to_string();
                let call_id = if obj.contains_key("call_id") {
                    string("call_id")?
                } else if obj.contains_key("id") {
                    string("id")?
                } else {
                    ""
                }
                .to_string();
                message.role = "assistant".into();
                message.tool_calls = Some(vec![crate::tool_parser::IncomingToolCall {
                    id: Some(call_id),
                    function: crate::tool_parser::IncomingFunction { name, arguments },
                }]);
            }
            "function_call_output" => {
                message.role = "tool".into();
                message.tool_call_id = Some(string("call_id")?.to_string());
                let output = obj
                    .get("output")
                    .ok_or("function_call_output requires output")?;
                // Structured outputs may carry images; arbitrary legacy JSON remains text.
                message.content = match output {
                    Value::Array(_) | Value::String(_) => {
                        ParsedContent::parse(output.clone(), true)?
                    }
                    other => ParsedContent {
                        text: other.to_string(),
                        ..ParsedContent::default()
                    },
                };
                if obj.contains_key("name") {
                    let name = string("name")?;
                    if !name.is_empty() {
                        message.name = Some(name.to_string());
                    }
                }
            }
            // Opaque reasoning is intentionally not fed back to the model.
            "reasoning" => return Ok(None),
            _ => return Err(format!("unsupported Responses input item type '{kind}'")),
        }
        Ok(Some(message))
    }
}
