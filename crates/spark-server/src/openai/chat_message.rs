// SPDX-License-Identifier: AGPL-3.0-only

use serde::Deserialize;

#[path = "parsed_content.rs"]
mod parsed_content;
#[path = "responses_input.rs"]
mod responses_input;

#[derive(Debug, Deserialize, Clone)]
pub struct IncomingMessage {
    pub role: String,
    #[serde(default, deserialize_with = "deserialize_message_content")]
    pub content: ParsedContent,
    /// Tool calls from a previous assistant message (multi-turn tool conversations).
    #[serde(default)]
    pub tool_calls: Option<Vec<crate::tool_parser::IncomingToolCall>>,
    /// ID of the tool call this message is responding to (role="tool").
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// Function name for tool response messages.
    #[serde(default)]
    pub name: Option<String>,
    /// Historical reasoning trace from a prior assistant turn (Qwen3
    /// `<think>...</think>` body). Clients (vLLM/SGLang/opencode) round-trip
    /// this field so the chat template can rehydrate the historical
    /// `<think>` block. Without it the template emits empty
    /// `<think>\n\n</think>\n\n` wrappers for every historical assistant
    /// turn → empty-think poisoning → premature `<|im_end|>` abort.
    /// Accepts both `reasoning_content` (DeepSeek/vLLM/LiteLLM standard)
    /// and the shorter `reasoning` alias used by some OpenAI SDK versions.
    #[serde(default, alias = "reasoning")]
    pub reasoning_content: Option<String>,
}

/// Content extracted from a message — text and any base64-encoded images.
#[derive(Debug, Clone, Default)]
pub struct ParsedContent {
    pub text: String,
    /// Base64 data URIs: `"data:image/jpeg;base64,..."` or raw base64 strings.
    pub images: Vec<String>,
    /// Ordered UTF-8 byte offset for each image in text.
    pub image_text_offsets: Vec<usize>,
}

impl IncomingMessage {
    /// Build a synthetic system message (used by the Responses adapter to
    /// carry `instructions` into the chat-completions pipeline).
    pub fn synthetic_system(text: String) -> Self {
        Self {
            role: "system".to_string(),
            content: ParsedContent {
                text,
                images: Vec::new(),
                image_text_offsets: Vec::new(),
            },
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        }
    }

    /// Build a synthetic user message (used by the Responses adapter when
    /// `input` is a plain string).
    pub fn synthetic_user_text(text: String) -> Self {
        Self {
            role: "user".to_string(),
            content: ParsedContent {
                text,
                images: Vec::new(),
                image_text_offsets: Vec::new(),
            },
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        }
    }

}

fn deserialize_message_content<'de, D>(d: D) -> Result<ParsedContent, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(d)?;
    ParsedContent::parse(value, false).map_err(serde::de::Error::custom)
}
