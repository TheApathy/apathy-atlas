// SPDX-License-Identifier: AGPL-3.0-only

use super::super::types::MessagesRequest;
use serde_json::json;

fn request(messages: serde_json::Value) -> MessagesRequest {
    serde_json::from_value(json!({
        "model": "qwen3.8-27b",
        "max_tokens": 1,
        "messages": messages,
    }))
    .unwrap()
}

#[test]
fn detects_direct_and_nested_tool_result_images() {
    let direct = request(json!([{
        "role": "user",
        "content": [{
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "AA=="},
        }],
    }]));
    assert!(direct.contains_image());

    let nested = request(json!([{
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": [{"type": "image", "source": {"type": "url", "url": "x"}}],
        }],
    }]));
    assert!(nested.contains_image());
}

#[test]
fn does_not_scan_opaque_tool_input() {
    let tool = request(json!([{
        "role": "assistant",
        "content": [{
            "type": "tool_use",
            "id": "call_1",
            "name": "inspect",
            "input": {"type": "image", "source": "opaque tool argument"},
        }],
    }]));
    assert!(!tool.contains_image());
}

#[test]
fn yarn_guards_precede_translation_and_tokenization() {
    let source = include_str!("../handlers.rs");
    let count_start = source.find("pub async fn count_tokens(").unwrap();
    let messages = &source[..count_start];
    let count_tokens = &source[count_start..];
    let messages_guards = messages
        .match_indices("if state.yarn_context && req.contains_image()")
        .collect::<Vec<_>>();
    let count_guards = count_tokens
        .match_indices("if state.yarn_context && req.contains_image()")
        .collect::<Vec<_>>();
    assert_eq!(messages_guards.len(), 1);
    assert_eq!(count_guards.len(), 1);
    let messages_guard = messages_guards[0].0;
    assert!(messages_guard < messages.find("state.dump_writer").unwrap());
    assert!(
        messages_guard
            < messages
                .find("let chat_json = anthropic_to_chat_request_json")
                .unwrap()
    );
    assert!(count_guards[0].0 < count_tokens.find("apply_chat_template_jinja").unwrap());
}
