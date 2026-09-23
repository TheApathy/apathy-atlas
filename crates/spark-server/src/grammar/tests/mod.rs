// SPDX-License-Identifier: AGPL-3.0-only

//! Test-only helpers and per-area test sub-modules.

use super::*;
use crate::tool_parser::ToolDefinition;

mod engine_state;
mod glm_native_tokenizer;
mod glm_tool_contract_grammar;
mod glm_tool_prefix_contract;
mod minimax;
mod misc;
mod qwen3_coder_required;
// TODO: stale tests — reference `enforce_min_length_on_required_strings`
// and `sanitize_schema_for_grammar` which have been refactored. File
// left on disk; un-comment once updated to the current schema-cleaner API.
// mod sanitize;
mod tools_basic;

/// all-models port: this tree's GrammarState has no `stop_legal` /
/// `with_stop_tokens` (upstream's EOS-exempt matcher), which the GLM grammar
/// tests use. Stop legality is read from the ordinary next-token mask.
pub(super) trait StopLegal {
    fn stop_legal(&mut self, stop: &[u32]) -> bool;
}

impl StopLegal for GrammarState {
    fn stop_legal(&mut self, stop: &[u32]) -> bool {
        self.fill_bitmask();
        stop.iter().any(|&token| self.is_token_allowed(token))
    }
}

/// Build a minimal vocabulary for testing.
/// Contains basic ASCII + JSON structural tokens.
pub(super) fn test_vocab() -> Vec<String> {
    let mut vocab = Vec::new();
    // Token 0..127: single ASCII characters
    for i in 0u8..128 {
        vocab.push(String::from(i as char));
    }
    // Token 128: <tool_call>
    vocab.push("<tool_call>".to_string());
    // Token 129: </tool_call>
    vocab.push("</tool_call>".to_string());
    // Token 130: <eos>
    vocab.push("<eos>".to_string());
    vocab
}

pub(super) fn test_tool_defs() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "get_weather".to_string(),
            description: Some("Get weather for a location".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            })),
        },
    }]
}
