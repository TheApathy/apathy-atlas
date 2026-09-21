// SPDX-License-Identifier: AGPL-3.0-only

use super::types::{ContentBlock, ToolResultContent};
use crate::ir::message::{ImageSource, Reasoning, ToolCall};
use crate::ir::{ContentPart, ImageData, Message, Role};

fn message(role: Role, content: Vec<ContentPart>) -> Message {
    Message {
        role,
        content,
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }
}

fn text(parts: &mut Vec<ContentPart>, value: String) {
    if value.is_empty() {
        return;
    }
    if let Some(ContentPart::Text(prior)) = parts.last_mut() {
        prior.push_str(&value);
    } else {
        parts.push(ContentPart::Text(value));
    }
}

fn image(source: super::types::ImageSourceBlock) -> ContentPart {
    // Both live adapters validate before consuming the request; never drop bad images.
    let uri = source
        .maybe_get_image_uri()
        .expect("validated Anthropic image source");
    ContentPart::Image(ImageSource {
        data: ImageData::from_uri(uri),
    })
}

fn tool_parts(content: Option<ToolResultContent>) -> Vec<ContentPart> {
    match content {
        None => Vec::new(),
        Some(content) if !content.contains_image() => {
            let value = content.to_text();
            if value.is_empty() {
                Vec::new()
            } else {
                vec![ContentPart::Text(value)]
            }
        }
        Some(ToolResultContent::Blocks(blocks)) => {
            let mut parts = Vec::new();
            for block in blocks {
                match block {
                    ContentBlock::Text { text: value } => text(&mut parts, value),
                    ContentBlock::Image { source } => parts.push(image(source)),
                    _ => {} // Nested image tool results are rejected at the boundary.
                }
            }
            parts
        }
        Some(ToolResultContent::Text(_)) => unreachable!("text has no images"),
    }
}

pub(super) fn lower(role: Role, blocks: Vec<ContentBlock>) -> Vec<Message> {
    let mut messages = Vec::new();
    let mut parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut reasoning = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text: value } => text(&mut parts, value),
            ContentBlock::Image { source } => parts.push(image(source)),
            ContentBlock::ToolUse { id, name, input } => tool_calls.push(ToolCall {
                id,
                name,
                arguments: input,
            }),
            ContentBlock::Thinking { thinking } => {
                if let Some(value) = thinking
                    && !value.is_empty()
                {
                    reasoning.push(value);
                }
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                if role != Role::Assistant {
                    if !parts.is_empty() {
                        messages.push(message(Role::User, std::mem::take(&mut parts)));
                    }
                    let mut result = message(Role::Tool, tool_parts(content));
                    result.tool_call_id = Some(tool_use_id);
                    result.tool_error = is_error.unwrap_or(false);
                    messages.push(result);
                }
            }
            ContentBlock::Unknown => {}
        }
    }
    if role == Role::Assistant {
        let mut result = message(role, parts);
        result.tool_calls = tool_calls;
        if !reasoning.is_empty() {
            result.reasoning = Some(Reasoning {
                text: reasoning.join("\n"),
            });
        }
        messages.push(result);
    } else if !parts.is_empty() {
        messages.push(message(role, parts));
    }
    messages
}
