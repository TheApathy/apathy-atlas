// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    image_content::{content_value, text_part, tool_result_value},
    types::ContentBlock,
};
use serde_json::{Value, json};

pub(super) fn translate_blocks(role: &str, blocks: &[ContentBlock]) -> Vec<Value> {
    let mut messages = Vec::new();
    let mut parts = Vec::new();
    let mut tool_calls = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text } => parts.push(text_part(text)),
            ContentBlock::Image { source } => parts.push(source.chat_part()),
            ContentBlock::ToolUse { id, name, input } => tool_calls.push(json!({
                "id":id,"type":"function",
                "function":{"name":name,"arguments":input.to_string()}
            })),
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                // Flush preceding user content before the result rather than
                // moving all user text (and screenshots) ahead of every tool.
                if role != "assistant" {
                    if !parts.is_empty() {
                        messages.push(json!({"role":role,
                            "content":content_value(std::mem::take(&mut parts), "")}));
                    }
                    messages.push(json!({"role":"tool", "tool_call_id":tool_use_id,
                        "content":tool_result_value(content.as_ref(), is_error.unwrap_or(false))}));
                }
            }
            ContentBlock::Thinking { .. } | ContentBlock::Unknown => {}
        }
    }
    if role == "assistant" || !parts.is_empty() || messages.is_empty() {
        let mut message = json!({"role":role, "content":content_value(parts, "")});
        if role == "assistant" && !tool_calls.is_empty() {
            message["tool_calls"] = Value::Array(tool_calls);
        }
        messages.push(message);
    }
    messages
}
