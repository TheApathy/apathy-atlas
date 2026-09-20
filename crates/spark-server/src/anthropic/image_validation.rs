// SPDX-License-Identifier: AGPL-3.0-only

use super::types::{AnthropicContent, ContentBlock, ImageSourceBlock, MessagesRequest, SystemContent, ToolResultContent};

impl ImageSourceBlock {
    fn validate(&self) -> Result<(), &'static str> {
        match self.source_type.as_str() {
            "base64" if self.data.as_ref().is_some_and(|s| !s.trim().is_empty())
                && matches!(self.media_type.as_deref(), Some("image/png" | "image/jpeg" | "image/gif" | "image/webp")) => Ok(()),
            "url" if self.url.as_ref().is_some_and(|s| s.starts_with("data:image/") && s.contains(";base64,")) => Ok(()),
            _ => Err("Images require nonempty inline base64 PNG/JPEG/GIF/WebP; remote URLs are not supported"),
        }
    }
}

impl ContentBlock {
    pub(super) fn contains_image(&self) -> bool {
        match self {
            Self::Image { .. } => true,
            Self::ToolResult { content: Some(content), .. } => content.contains_image(),
            _ => false,
        }
    }
}

impl ToolResultContent {
    pub(super) fn contains_image(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Blocks(blocks) => blocks.iter().any(ContentBlock::contains_image),
        }
    }
}

impl MessagesRequest {
    pub(super) fn contains_image(&self) -> bool {
        matches!(&self.system, Some(SystemContent::Blocks(blocks)) if blocks.iter().any(|b| b.block_type == "image"))
        || self.messages.iter().any(|m| match &m.content {
            AnthropicContent::Text(_) => false,
            AnthropicContent::Blocks(blocks) => blocks.iter().any(ContentBlock::contains_image),
        })
    }

    pub(super) fn validate_images(&self) -> Result<(), &'static str> {
        if matches!(&self.system, Some(SystemContent::Blocks(blocks)) if blocks.iter().any(|b| b.block_type == "image")) {
            return Err("Anthropic system images are not supported");
        }
        for message in &self.messages {
            let AnthropicContent::Blocks(blocks) = &message.content else { continue; };
            for block in blocks {
                if block.contains_image() && message.role != "user" {
                    return Err("Anthropic image content is supported only in user messages");
                }
                match block {
                    ContentBlock::Image { source } => source.validate()?,
                    ContentBlock::ToolResult { content: Some(ToolResultContent::Blocks(parts)), .. } => {
                        for part in parts {
                            match part {
                                ContentBlock::Image { source } => source.validate()?,
                                other if other.contains_image() => return Err("Nested tool results containing images are not supported"),
                                _ => {},
                            }
                        }
                    }
                    _ => {},
                }
            }
        }
        Ok(())
    }
}
