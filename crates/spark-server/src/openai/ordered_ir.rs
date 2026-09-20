// SPDX-License-Identifier: AGPL-3.0-only

use super::{ChatCompletionRequest, ParsedContent};
use crate::ir::{ContentPart, ImageData};
use crate::ir::message::ImageSource;

impl ChatCompletionRequest {
    /// Check stored/programmatic DTOs before the infallible wire-to-IR conversion.
    pub(crate) fn validate_content_order(&self) -> Result<(), String> {
        for message in &self.messages { message.content.validate_order()?; }
        Ok(())
    }
}

impl ParsedContent {
    pub(crate) fn ordered_ir_parts(&self) -> Result<Vec<ContentPart>, String> {
        self.validate_order()?;
        let mut parts = Vec::new();
        let mut start = 0;
        for (&end, image) in self.image_text_offsets.iter().zip(&self.images) {
            if end > start { parts.push(ContentPart::Text(self.text[start..end].to_string())); }
            parts.push(ContentPart::Image(ImageSource { data: ImageData::from_uri(image.clone()) }));
            start = end;
        }
        if start < self.text.len() { parts.push(ContentPart::Text(self.text[start..].to_string())); }
        Ok(parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::IncomingMessage;

    #[test]
    fn wire_to_ir_keeps_unicode_and_adjacent_image_positions() {
        let wire = serde_json::json!({"role":"user","reasoning_content":"trace","content":[
            {"type":"text","text":"🟥:"}, {"type":"image_url","image_url":{"url":"red"}},
            {"type":"image_url","image_url":{"url":"blue"}}, {"type":"text","text":"end"}]});
        let m: IncomingMessage = serde_json::from_value(wire).unwrap();
        let ir: crate::ir::Message = (&m).into();
        assert_eq!(ir.content, vec![ContentPart::Text("🟥:".into()),
            ContentPart::Image(ImageSource { data: ImageData::from_uri("red".into()) }),
            ContentPart::Image(ImageSource { data: ImageData::from_uri("blue".into()) }),
            ContentPart::Text("end".into())]);
        assert_eq!(ir.reasoning.unwrap().text, "trace");
    }

    #[test]
    fn invalid_programmatic_content_is_rejected_before_conversion() {
        let mut content = ParsedContent { text: "🟥".into(), images: vec!["red".into()], image_text_offsets: vec![1] };
        assert!(content.ordered_ir_parts().is_err());
        content.image_text_offsets = vec![4];
        assert!(content.ordered_ir_parts().is_ok());
        let mut req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({"model":"test","messages":[{"role":"user","content":"ok"}]})).unwrap();
        req.messages[0].content = content;
        req.messages[0].content.image_text_offsets.clear();
        assert!(req.validate_content_order().is_err());
    }

    #[test]
    fn prepend_inserts_before_leading_image_and_leaves_later_text_in_place() {
        for tail in ["", "later"] {
            let m: IncomingMessage = serde_json::from_value(serde_json::json!({
                "role":"system", "content":[{"type":"image_url","image_url":{"url":"red"}},
                    {"type":"text","text":tail}]
            })).unwrap();
            let mut ir: crate::ir::Message = (&m).into();
            ir.prepend_text("prefix");
            assert_eq!(ir.content[0], ContentPart::Text("prefix".into()));
            assert!(matches!(ir.content[1], ContentPart::Image(_)));
            assert_eq!(ir.content.len(), if tail.is_empty() { 2 } else { 3 });
            if !tail.is_empty() { assert_eq!(ir.content[2], ContentPart::Text(tail.into())); }
            ir.prepend_text("again");
            assert_eq!(ir.content[0], ContentPart::Text("againprefix".into()));
        }
    }
}
