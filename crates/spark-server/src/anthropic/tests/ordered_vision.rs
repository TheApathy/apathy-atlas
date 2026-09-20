// SPDX-License-Identifier: AGPL-3.0-only

use super::super::types::MessagesRequest;
use crate::ir::{ChatRequest, ContentPart, Role};
use serde_json::{Value, json};

fn image(data: &str) -> Value {
    json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":data}})
}
fn request(role: &str, content: Value) -> MessagesRequest {
    serde_json::from_value(json!({"model":"test","max_tokens":1,
        "messages":[{"role":role,"content":content}]})).unwrap()
}
fn markers(parts: &[ContentPart]) -> Vec<String> {
    parts.iter().map(|p| match p {
        ContentPart::Text(t) => t.clone(), ContentPart::Image(_) => "<image>".into(),
    }).collect()
}

#[test]
fn direct_and_adjacent_images_keep_typed_ir_order() {
    let req = request("user", json!([{"type":"text","text":"🟥:"},image("RED"),
        image("BLUE"),{"type":"text","text":"end"}]));
    assert!(req.validate_images().is_ok());
    let ir: ChatRequest = req.into();
    assert_eq!(markers(&ir.messages[0].content), ["🟥:", "<image>", "<image>", "end"]);
}

#[test]
fn tool_images_keep_error_metadata_and_turn_order() {
    let req = request("user", json!([{"type":"tool_result","tool_use_id":"t1","is_error":true,
        "content":[{"type":"text","text":"before"},image("RED"),{"type":"text","text":"after"}]},
        {"type":"text","text":"Next?"}]));
    assert!(req.validate_images().is_ok());
    let ir: ChatRequest = req.into();
    assert_eq!(ir.messages.len(), 2);
    assert_eq!(ir.messages[0].role, Role::Tool);
    assert_eq!(ir.messages[0].tool_call_id.as_deref(), Some("t1"));
    assert!(ir.messages[0].tool_error);
    assert_eq!(markers(&ir.messages[0].content), ["before", "<image>", "after"]);
    assert_eq!(ir.messages[1].text(), "Next?");
}

#[test]
fn text_only_tool_results_keep_lf_join_and_reasoning_stays_typed() {
    let req = request("user", json!([{"type":"tool_result","tool_use_id":"t1",
        "content":[{"type":"text","text":"one"},{"type":"text","text":"two"}]}]));
    let ir: ChatRequest = req.into();
    assert_eq!(ir.messages[0].text(), "one\ntwo");
    let req = request("assistant", json!([{"type":"thinking","thinking":"reason"},
        {"type":"tool_use","id":"t","name":"read","input":{"path":"x"}}]));
    let ir: ChatRequest = req.into();
    assert_eq!(ir.messages[0].reasoning.as_ref().unwrap().text, "reason");
    assert_eq!(ir.messages[0].tool_calls[0].arguments, json!({"path":"x"}));
}

#[test]
fn malformed_sources_and_invalid_placement_are_rejected() {
    for content in [
        json!([{"type":"image","source":{"type":"base64","data":"AA=="}}]),
        json!([{"type":"image","source":{"type":"base64","media_type":"text/plain","data":"AA=="}}]),
        json!([{"type":"image","source":{"type":"file","url":"x"}}]),
        json!([{"type":"image","source":{"type":"url","url":"https://example.invalid/x"}}]),
        json!([image("")]),
        json!([{"type":"tool_result","tool_use_id":"a","content":[
            {"type":"tool_result","tool_use_id":"b","content":[image("RED")]}]}]),
    ] {
        assert!(request("user", content).validate_images().is_err());
    }
    assert!(request("assistant", json!([image("RED")])).validate_images().is_err());
}

#[test]
fn image_only_survives_and_tool_arguments_are_opaque() {
    let req = request("user", json!([image("RED")]));
    assert!(req.contains_image());
    let ir: ChatRequest = req.into();
    assert_eq!(markers(&ir.messages[0].content), ["<image>"]);
    let req = request("assistant", json!([{"type":"tool_use","id":"x","name":"read",
        "input":{"type":"image","source":"opaque"}}]));
    assert!(!req.contains_image());
    assert!(req.validate_images().is_ok());
}

#[test]
fn system_images_are_visible_to_admission_and_rejected() {
    let req: MessagesRequest = serde_json::from_value(json!({"model":"test","max_tokens":1,
        "system":[image("RED")], "messages":[{"role":"user","content":"hello"}]})).unwrap();
    assert!(req.contains_image());
    assert!(req.validate_images().is_err());
}

#[test]
fn live_boundaries_validate_before_lowering_and_counts_never_strip_images() {
    let source = include_str!("../handlers.rs");
    let count = source.find("pub async fn count_tokens(").unwrap();
    assert!(source[..count].find("req.validate_images()").unwrap()
        < source[..count].find("let stream = req.stream").unwrap());
    let tail = &source[count..];
    assert!(tail.find("unsupported_multimodal_token_count").unwrap()
        < tail.find("crate::ir::ChatRequest::from(req)").unwrap());
    assert!(!tail.contains(".retain(|p| !matches!(p, crate::ir::ContentPart::Image(_)))"));
}
