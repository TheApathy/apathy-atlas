// SPDX-License-Identifier: AGPL-3.0-only

//! Raw Responses-input guard for the static-YaRN text-only boundary.

use serde_json::Value;

pub(super) fn response_input_contains_image(input: &Value) -> bool {
    match input {
        Value::Array(items) => response_items_contain_image(items),
        Value::Object(_) => response_item_contains_image(input),
        _ => false,
    }
}

pub(super) fn response_items_contain_image(items: &[Value]) -> bool {
    items.iter().any(response_item_contains_image)
}

fn response_item_contains_image(item: &Value) -> bool {
    let Some(item) = item.as_object() else {
        return false;
    };
    let item_type = item
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("message");
    if is_image_type(item_type) {
        return true;
    }
    if item_type != "message" {
        return false;
    }
    match item.get("content") {
        Some(Value::Array(parts)) => parts.iter().any(content_part_contains_image),
        Some(part @ Value::Object(_)) => content_part_contains_image(part),
        _ => false,
    }
}

fn content_part_contains_image(part: &Value) -> bool {
    part.as_object()
        .and_then(|part| part.get("type"))
        .and_then(Value::as_str)
        .is_some_and(is_image_type)
}

fn is_image_type(part_type: &str) -> bool {
    matches!(
        part_type,
        "input_image" | "image_url" | "image" | "computer_screenshot"
    )
}

#[cfg(test)]
mod tests {
    use super::{response_input_contains_image, response_items_contain_image};
    use serde_json::json;

    #[test]
    fn rejects_direct_responses_image_parts() {
        for image_type in ["input_image", "image_url", "image", "computer_screenshot"] {
            let input = json!([{
                "type": "message",
                "role": "user",
                "content": [{"type": image_type, "image_url": "data:image/png;base64,AA=="}],
            }]);
            assert!(response_input_contains_image(&input), "{image_type}");
        }
        assert!(response_input_contains_image(&json!({
            "type": "input_image",
            "image_url": "data:image/png;base64,AA==",
        })));
    }

    #[test]
    fn rejects_images_replayed_from_a_conversation() {
        let items = vec![json!({
            "id": "item_1",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_image", "file_id": "file_1"}],
        })];
        assert!(response_items_contain_image(&items));
    }

    #[test]
    fn does_not_scan_opaque_tool_output_for_image_shaped_data() {
        let input = json!([{
            "type": "function_call_output",
            "call_id": "call_1",
            "output": {"type": "input_image", "image_url": "opaque tool data"},
        }]);
        assert!(!response_input_contains_image(&input));
        assert!(!response_input_contains_image(&json!("plain text")));
    }

    #[test]
    fn guard_precedes_responses_lowering_and_dispatch() {
        let source = include_str!("responses.rs");
        let direct_guard = source
            .find("if state.yarn_context && response_input_contains_image(&r.input)")
            .unwrap();
        let conversation_guard = source
            .find("if state.yarn_context && response_items_contain_image(&snap.items)")
            .unwrap();
        let lowering = source.find("lower_responses_to_chat").unwrap();
        let streaming = source.find("responses_endpoint_stream(").unwrap();
        let blocking = source.find("chat_completions_inner(").unwrap();
        for guard in [direct_guard, conversation_guard] {
            assert!(guard < lowering && guard < streaming && guard < blocking);
        }
    }
}
