// SPDX-License-Identifier: AGPL-3.0-only

use super::{IncomingMessage, messages_to_disk_json};
use serde_json::json;

#[test]
fn disk_wire_roundtrip_preserves_interleaved_and_adjacent_images() {
    let messages: Vec<IncomingMessage> = serde_json::from_value(json!([
        {"role":"system","content":[{"type":"image_url","image_url":{"url":"system"}}]},
        {"role":"tool","tool_call_id":"call_1","content":[
            {"type":"text","text":"🟥first:"},
            {"type":"image_url","image_url":{"url":"red"}},
            {"type":"text","text":"second:"},
            {"type":"image_url","image_url":{"url":"blue"}},
            {"type":"image_url","image_url":{"url":"green"}},
            {"type":"text","text":"end"}]}
    ]))
    .unwrap();
    let disk = messages_to_disk_json(&messages).unwrap();
    let replay: Vec<IncomingMessage> = serde_json::from_value(disk).unwrap();
    for (before, after) in messages.iter().zip(&replay) {
        assert_eq!(before.role, after.role);
        assert_eq!(before.tool_call_id, after.tool_call_id);
        assert_eq!(before.content.text, after.content.text);
        assert_eq!(before.content.images, after.content.images);
        assert_eq!(
            before.content.image_text_offsets,
            after.content.image_text_offsets
        );
    }
}

#[test]
fn persistence_rejects_invalid_internal_image_order() {
    let mut message = IncomingMessage::synthetic_user_text("🟥".into());
    message.content.images.push("red".into());
    assert!(messages_to_disk_json(&[message.clone()]).is_err());
    message.content.image_text_offsets.push(1);
    assert!(messages_to_disk_json(&[message]).is_err());
}
