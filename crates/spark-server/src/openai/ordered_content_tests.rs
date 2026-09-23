// SPDX-License-Identifier: AGPL-3.0-only

use super::IncomingMessage;
use serde_json::json;

#[test]
fn interleaved_images_keep_unicode_text_offsets() {
    let m: IncomingMessage = serde_json::from_value(json!({"role":"user","content":[
        {"type":"text","text":"First 🟥:"},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,RED"}},
        {"type":"text","text":"Second:"},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,BLUE"}},
        {"type":"text","text":"What order?"}
    ]}))
    .unwrap();
    assert_eq!(m.content.text, "First 🟥:Second:What order?");
    assert_eq!(
        m.content.image_text_offsets,
        ["First 🟥:".len(), "First 🟥:Second:".len()]
    );
    assert_eq!(m.content.images.len(), 2);
}

#[test]
fn adjacent_images_and_trailing_text_are_ordered() {
    let m: IncomingMessage = serde_json::from_value(json!({"role":"user","content":[
        {"type":"image_url","image_url":{"url":"red"}},
        {"type":"image_url","image_url":{"url":"blue"}},
        {"type":"text","text":"Describe both."}
    ]}))
    .unwrap();
    assert_eq!(m.content.image_text_offsets, [0, 0]);
    assert_eq!(m.content.images, ["red", "blue"]);
}

#[test]
fn responses_input_images_are_not_silently_dropped() {
    let m = IncomingMessage::from_responses_input_item(&json!({"role":"user","content":[
        {"type":"input_text","text":"Before"},
        {"type":"input_image","image_url":"red"},
        {"type":"input_text","text":"After"}
    ]}))
    .unwrap();
    assert_eq!(m.content.images, ["red"]);
    assert_eq!(m.content.image_text_offsets, [6]);
}

#[test]
fn malformed_image_or_text_parts_are_rejected() {
    for part in [
        json!({"type":"image_url"}),
        json!({"type":"text"}),
        json!({"type":"input_audio","input_audio":{}}),
    ] {
        assert!(
            serde_json::from_value::<IncomingMessage>(json!({"role":"user","content":[part]}))
                .is_err()
        );
    }
}

#[test]
fn plain_and_null_content_preserve_existing_semantics() {
    for (content, expected) in [(json!("Hello"), "Hello"), (json!(null), "")] {
        let m: IncomingMessage =
            serde_json::from_value(json!({"role":"assistant","content":content})).unwrap();
        assert_eq!(m.content.text, expected);
        assert!(m.content.images.is_empty());
        assert!(m.content.image_text_offsets.is_empty());
    }
}
