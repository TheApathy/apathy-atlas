// SPDX-License-Identifier: AGPL-3.0-only
//! Reject malformed Anthropic choices at the wire boundary, before IR lowering.
use super::types::AnthropicToolChoice;
use serde::Deserialize;

#[derive(Deserialize)]
pub(super) struct WireChoice {
    #[serde(rename = "type")]
    choice_type: String,
    #[serde(default)]
    name: Option<String>,
}

impl TryFrom<WireChoice> for AnthropicToolChoice {
    type Error = String;
    fn try_from(wire: WireChoice) -> Result<Self, Self::Error> {
        match wire.choice_type.as_str() {
            "auto" | "any" | "none" if wire.name.is_none() => {}
            "tool" if wire.name.as_deref().is_some_and(|s| !s.trim().is_empty()) => {}
            _ => return Err(
                "invalid Anthropic tool_choice: use auto, any, none, or tool with a nonempty name"
                    .into(),
            ),
        }
        Ok(Self {
            choice_type: wire.choice_type,
            name: wire.name,
        })
    }
}
