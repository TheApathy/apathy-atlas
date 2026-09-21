// SPDX-License-Identifier: AGPL-3.0-only

use super::types::{AnthropicContent, ContentBlock, MessagesRequest, SystemContent, ToolResultContent};
use serde::Deserialize;
use serde_json::{Value, json};

/// Preserve the actual source; missing fields and unsupported source kinds fail
/// deserialization. URLs are passed to the common image admission path, which
/// accepts inline base64 only and never fetches a remote resource.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

impl ImageSource {
    fn validate(&self) -> Result<(), &'static str> {
        match self {
            Self::Base64 { media_type, data }
                if matches!(
                    media_type.as_str(),
                    "image/png" | "image/jpeg" | "image/gif" | "image/webp"
                ) && !data.trim().is_empty() =>
            {
                Ok(())
            }
            Self::Url { url } if url.starts_with("data:image/") && url.contains(";base64,") => {
                Ok(())
            }
            _ => Err(
                "Images require nonempty inline base64 PNG/JPEG/GIF/WebP; remote URLs are not supported",
            ),
        }
    }

    pub(super) fn chat_part(&self) -> Value {
        let url = match self {
            Self::Base64 { media_type, data } => format!("data:{media_type};base64,{data}"),
            Self::Url { url } => url.clone(),
        };
        json!({"type":"image_url","image_url":{"url":url}})
    }
}

impl MessagesRequest {
    pub(super) fn validate_images(&self) -> Result<(), &'static str> {
        if matches!(&self.system, Some(SystemContent::Blocks(blocks))
            if blocks.iter().any(|block| block.block_type == "image")) {
            return Err("Anthropic system images are not supported");
        }
        for message in &self.messages {
            let AnthropicContent::Blocks(blocks) = &message.content else {
                continue;
            };
            for block in blocks {
                if block.contains_image() && message.role != "user" {
                    return Err("Anthropic image content is supported only in user messages");
                }
                match block {
                    ContentBlock::Image { source } => source.validate()?,
                    ContentBlock::ToolResult {
                        content: Some(ToolResultContent::Blocks(parts)),
                        ..
                    } => {
                        for part in parts {
                            match part {
                                ContentBlock::Image { source } => source.validate()?,
                                other if other.contains_image() => {
                                    return Err(
                                        "Nested tool results containing images are not supported",
                                    );
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
}

pub(super) fn text_part(text: &str) -> Value {
    json!({"type":"text","text":text})
}

/// Keep the previous text-only wire shape, while image content retains every
/// text/image boundary. Tool-result text-only blocks historically join by LF.
pub(super) fn content_value(parts: Vec<Value>, separator: &str) -> Value {
    if parts.iter().any(|p| p["type"] == "image_url") {
        Value::Array(parts)
    } else {
        Value::String(
            parts
                .iter()
                .filter_map(|p| p["text"].as_str())
                .collect::<Vec<_>>()
                .join(separator),
        )
    }
}

pub(super) fn tool_result_value(content: Option<&ToolResultContent>, error: bool) -> Value {
    let mut parts = match content {
        None => Vec::new(),
        Some(ToolResultContent::Text(text)) => vec![text_part(text)],
        Some(ToolResultContent::Blocks(blocks)) => blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text_part(text)),
                ContentBlock::Image { source } => Some(source.chat_part()),
                _ => None,
            })
            .collect(),
    };
    let has_images = parts.iter().any(|p| p["type"] == "image_url");
    if error && has_images {
        parts.insert(0, text_part("[tool error]\n"));
    }
    let mut value = content_value(parts, "\n");
    if error && !has_images {
        value = Value::String(format!(
            "[tool error]\n{}",
            value.as_str().unwrap_or_default()
        ));
    }
    value
}
