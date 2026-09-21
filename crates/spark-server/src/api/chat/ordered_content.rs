// SPDX-License-Identifier: AGPL-3.0-only

use crate::ir::{ContentPart, Message};

/// Derive text and image offsets from the authoritative IR. A native placeholder
/// stays literal text (DeepSeek); generic images become ordered Jinja markers.
pub(super) fn flatten(message: &Message, placeholder: Option<&str>) -> (String, Vec<usize>) {
    let mut text = if message.tool_error {
        "[tool error]\n".into()
    } else {
        String::new()
    };
    let mut offsets = Vec::new();
    for part in &message.content {
        match part {
            ContentPart::Text(value) => text.push_str(value),
            ContentPart::Image(_) => match placeholder {
                Some(value) => text.push_str(value),
                None => offsets.push(text.len()),
            },
        }
    }
    (text, offsets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::message::ImageSource;
    use crate::ir::{ImageData, Role};

    #[test]
    fn prefix_and_unicode_offsets_follow_ir_order() {
        let mut message = Message::synthetic_system("α".into());
        message.role = Role::Tool;
        message.tool_error = true;
        message.content.push(ContentPart::Image(ImageSource {
            data: ImageData::Base64("red".into()),
        }));
        message.content.push(ContentPart::Text("end".into()));
        let (text, offsets) = flatten(&message, None);
        assert_eq!(text, "[tool error]\nαend");
        assert_eq!(offsets, ["[tool error]\nα".len()]);
        let (text, offsets) = flatten(&message, Some("<image>"));
        assert_eq!(text, "[tool error]\nα<image>end");
        assert!(offsets.is_empty());
    }
    #[test]
    fn actual_json_builder_keeps_interleaved_ir_order() {
        let dto: crate::openai::IncomingMessage = serde_json::from_value(serde_json::json!({
            "role":"user", "content":[{"type":"text","text":"🟥:"},
                {"type":"image_url","image_url":{"url":"red"}},
                {"type":"text","text":"blue:"},{"type":"image_url","image_url":{"url":"blue"}}]
        }))
        .unwrap();
        let message: Message = (&dto).into();
        let (content, image_text_offsets) = flatten(&message, None);
        let entry = super::super::msg_entry::MsgEntry {
            role: "user".into(),
            content,
            image_count: 2,
            image_text_offsets,
            tool_calls: None,
            reasoning_content: None,
        };
        let json = super::super::template::build_json_messages(&[entry]).unwrap();
        assert_eq!(
            json[0]["content"],
            serde_json::json!([
                {"type":"text","text":"🟥:"}, {"type":"image"},
                {"type":"text","text":"blue:"}, {"type":"image"}
            ])
        );
    }
}
