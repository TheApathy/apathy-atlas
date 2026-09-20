// SPDX-License-Identifier: AGPL-3.0-only

use super::{IncomingMessage, ParsedContent};
use serde_json::json;

fn ordered() -> ParsedContent {
    serde_json::from_value::<IncomingMessage>(json!({"role":"tool","content":[
        {"type":"text","text":"🟥"}, {"type":"image_url","image_url":{"url":"red"}},
        {"type":"image_url","image_url":{"url":"blue"}}, {"type":"text","text":"end"}
    ]})).unwrap().content
}

#[test]
fn markers_preserve_adjacent_unicode_order_without_payloads() {
    let c = ordered();
    let markers = ParsedContent::marker_json(&c.text, c.images.len(), &c.image_text_offsets).unwrap();
    assert_eq!(markers, json!([
        {"type":"text","text":"🟥"}, {"type":"image"}, {"type":"image"},
        {"type":"text","text":"end"}
    ]));
    assert!(!markers.to_string().contains("red"));
}

#[test]
fn chat_and_responses_roundtrip_all_positions() {
    let c = ordered();
    let chat: IncomingMessage = serde_json::from_value(json!({"role":"tool","content":c.chat_json().unwrap()})).unwrap();
    let response = IncomingMessage::try_from_responses_input_item(&json!({"role":"tool","content":c.responses_json().unwrap()})).unwrap().unwrap();
    for got in [chat.content, response.content] {
        assert_eq!(got.text, c.text);
        assert_eq!(got.images, c.images);
        assert_eq!(got.image_text_offsets, c.image_text_offsets);
    }
}

#[test]
fn prepend_moves_offsets_and_append_keeps_them() {
    let mut c = ordered();
    c.prepend_text("α:").unwrap();
    c.text.push_str("suffix");
    assert_eq!(c.image_text_offsets, ["α:🟥".len(); 2]);
    assert!(c.validate_order().is_ok());
}

#[test]
fn invalid_internal_offsets_fail_closed_without_mutation() {
    for offsets in [vec![], vec![4], vec![4, 3], vec![1, 4], vec![4, 99]] {
        let mut c = ordered();
        c.image_text_offsets = offsets;
        let old = c.text.clone();
        assert!(c.validate_order().is_err());
        assert!(c.chat_json().is_err());
        assert!(c.responses_json().is_err());
        assert!(c.prepend_text("prefix").is_err());
        assert_eq!(c.text, old);
    }
}

#[test]
fn checked_responses_reject_malformed_instead_of_dropping() {
    for item in [json!(7), json!({"content":[{"type":"input_image"}]}),
        json!({"content":[{"type":"input_text"}]}),
        json!({"content":[{"type":"input_image","image_url":""}]}),
        json!({"content":[{"type":"input_audio"}]}),
        json!({"type":"unknown"}), json!({"content":{}})] {
        assert!(IncomingMessage::try_from_responses_input_item(&item).is_err(), "{item}");
    }
    assert!(IncomingMessage::try_from_responses_input_item(&json!({"type":"reasoning"})).unwrap().is_none());
}

#[test]
fn responses_tool_output_retains_images() {
    let m = IncomingMessage::try_from_responses_input_item(&json!({
        "type":"function_call_output", "call_id":"call_1", "output":[
            {"type":"input_text","text":"before"},
            {"type":"input_image","image_url":"red"},
            {"type":"input_text","text":"after"}]
    })).unwrap().unwrap();
    assert_eq!(m.role, "tool");
    assert_eq!(m.content.text, "beforeafter");
    assert_eq!(m.content.image_text_offsets, [6]);
    assert_eq!(m.content.images, ["red"]);
}

#[test]
fn image_only_and_text_only_serialization_are_lossless() {
    let image: IncomingMessage = serde_json::from_value(json!({"role":"system","content":[
        {"type":"image_url","image_url":{"url":"red"}}]
    })).unwrap();
    assert_eq!(image.content.image_text_offsets, [0]);
    assert_eq!(ParsedContent::marker_json("", 1, &[0]).unwrap(), json!([{"type":"image"}]));
    assert_eq!(ParsedContent::default().chat_json().unwrap(), json!(""));
    assert_eq!(ParsedContent::default().responses_json().unwrap(), json!([{"type":"input_text","text":""}]));
}
