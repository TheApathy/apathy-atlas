// SPDX-License-Identifier: AGPL-3.0-only

use super::super::{translate::anthropic_to_chat_request_json, types::MessagesRequest};
use serde_json::{Value, json};

fn translated(content: Value) -> Value {
    let req: MessagesRequest = serde_json::from_value(json!({
        "model":"test", "max_tokens":8,
        "messages":[{"role":"user", "content":content}]
    }))
    .unwrap();
    anthropic_to_chat_request_json(&req)
}

fn image(data: &str) -> Value {
    json!({"type":"image","source":{
        "type":"base64","media_type":"image/png","data":data
    }})
}

#[test]
fn direct_images_retain_their_interleaved_order_and_payload() {
    let result = translated(json!([
        {"type":"text","text":"First:"}, image("RED"),
        {"type":"text","text":"Second:"}, image("BLUE")
    ]));
    assert_eq!(
        result["messages"][0]["content"],
        json!([
            {"type":"text","text":"First:"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,RED"}},
            {"type":"text","text":"Second:"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,BLUE"}}
        ])
    );
}

#[test]
fn image_only_message_cannot_disappear() {
    let result = translated(json!([image("RED"), image("BLUE")]));
    assert_eq!(result["messages"].as_array().unwrap().len(), 1);
    assert_eq!(
        result["messages"][0]["content"].as_array().unwrap().len(),
        2
    );
}

#[test]
fn tool_screenshot_keeps_content_order_error_marker_and_turn_order() {
    let result = translated(json!([
        {"type":"tool_result","tool_use_id":"call_1","is_error":true,
         "content":[{"type":"text","text":"Screenshot:"},image("RED"),
                    {"type":"text","text":"Inspect it."}]},
        {"type":"text","text":"What failed?"}
    ]));
    let messages = result["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "tool");
    assert_eq!(messages[0]["tool_call_id"], "call_1");
    assert_eq!(
        messages[0]["content"],
        json!([
            {"type":"text","text":"[tool error]\n"},
            {"type":"text","text":"Screenshot:"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,RED"}},
            {"type":"text","text":"Inspect it."}
        ])
    );
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[1]["content"], "What failed?");
}

#[test]
fn missing_or_unknown_image_source_is_rejected() {
    for block in [
        json!({"type":"image"}),
        json!({"type":"image","source":{"type":"file","path":"x"}}),
        json!({"type":"image","source":{"type":"base64","media_type":"image/png"}}),
    ] {
        assert!(
            serde_json::from_value::<MessagesRequest>(json!({
                "model":"test", "max_tokens":1,
                "messages":[{"role":"user","content":[block]}]
            }))
            .is_err()
        );
    }
}

#[test]
fn text_and_tool_error_compatibility_is_unchanged() {
    let result = translated(json!([
        {"type":"tool_result","tool_use_id":"x","is_error":true,
         "content":[{"type":"text","text":"one"},{"type":"text","text":"two"}]}
    ]));
    assert_eq!(result["messages"][0]["content"], "[tool error]\none\ntwo");
}

#[test]
fn invalid_image_placement_and_sources_fail_before_translation() {
    for (role, content) in [
        ("assistant", json!([image("RED")])),
        (
            "user",
            json!([{"type":"image","source":{"type":"url","url":"https://example.invalid/x"}}]),
        ),
        ("user", json!([image("")])),
        (
            "user",
            json!([{"type":"image","source":{"type":"base64","media_type":"text/plain","data":"AA=="}}]),
        ),
        (
            "user",
            json!([{"type":"tool_result","tool_use_id":"x","content":[
                {"type":"tool_result","tool_use_id":"y","content":[image("RED")]}
            ]}]),
        ),
    ] {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model":"test", "max_tokens":1, "messages":[{"role":role,"content":content}]
        }))
        .unwrap();
        assert!(req.validate_images().is_err());
    }
    let req: MessagesRequest = serde_json::from_value(json!({
        "model":"test", "max_tokens":1, "messages":[{"role":"user","content":[image("RED")]}]
    }))
    .unwrap();
    assert!(req.validate_images().is_ok());
}

#[test]
fn system_images_are_rejected_not_silently_flattened() {
    let req: MessagesRequest = serde_json::from_value(json!({
        "model":"test", "max_tokens":1, "system":[image("RED")],
        "messages":[{"role":"user","content":"Inspect"}]
    })).unwrap();
    assert!(req.contains_image());
    assert!(req.validate_images().is_err());
}
