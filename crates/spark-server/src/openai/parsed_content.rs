// SPDX-License-Identifier: AGPL-3.0-only

use super::ParsedContent;
use serde_json::{Value, json};

impl ParsedContent {
    /// Parse chat or Responses content without discarding unsupported/malformed parts.
    pub(super) fn parse(value: Value, responses: bool) -> Result<Self, String> {
        let mut out = Self::default();
        let parts = match value {
            Value::Null => return Ok(out),
            Value::String(text) => return Ok(Self { text, ..out }),
            Value::Array(parts) => parts,
            _ => return Err("message content must be text, null, or an array of parts".into()),
        };
        for part in parts {
            let obj = part.as_object().ok_or("content part must be an object")?;
            let kind = obj.get("type").and_then(Value::as_str).ok_or("content part requires a type")?;
            match kind {
                "text" | "input_text" | "output_text" if kind == "text" || responses => {
                    let text = obj.get("text").and_then(Value::as_str).ok_or("text part requires string text")?;
                    out.text.push_str(text);
                }
                "image_url" | "input_image" | "image" if kind == "image_url" || responses => {
                    let image = obj.get("image_url").ok_or("image part requires image_url")?;
                    let url = if responses {
                        image.as_str().or_else(|| image.get("url").and_then(Value::as_str))
                    } else {
                        image.get("url").and_then(Value::as_str)
                    }.filter(|s| !s.is_empty()).ok_or("image part requires a nonempty image URL")?;
                    out.image_text_offsets.push(out.text.len());
                    out.images.push(url.to_string());
                }
                _ => return Err(format!("unsupported content part type '{kind}'")),
            }
        }
        out.validate_order()?;
        Ok(out)
    }

    pub fn validate_order(&self) -> Result<(), String> {
        Self::validate_offsets(&self.text, self.images.len(), &self.image_text_offsets)?;
        if self.images.iter().any(String::is_empty) {
            return Err("image URLs must be nonempty".into());
        }
        Ok(())
    }

    fn validate_offsets(text: &str, count: usize, offsets: &[usize]) -> Result<(), String> {
        if count != offsets.len() {
            return Err("image count does not match ordered text offsets".into());
        }
        let mut prior = 0;
        for &offset in offsets {
            if offset < prior || offset > text.len() || !text.is_char_boundary(offset) {
                return Err("image offsets must be ordered UTF-8 byte boundaries within text".into());
            }
            prior = offset;
        }
        Ok(())
    }

    /// Prepend while preserving ownership/order. Appending text needs no offset adjustment.
    pub fn prepend_text(&mut self, prefix: &str) -> Result<(), String> {
        self.validate_order()?;
        self.text.len().checked_add(prefix.len()).ok_or("content length overflow")?;
        for offset in &mut self.image_text_offsets {
            *offset += prefix.len();
        }
        self.text.insert_str(0, prefix);
        Ok(())
    }

    fn parts(text: &str, offsets: &[usize], text_kind: &str, image: impl Fn(usize) -> Value) -> Vec<Value> {
        let mut parts = Vec::new();
        let mut start = 0;
        for (i, &end) in offsets.iter().enumerate() {
            if end > start {
                parts.push(json!({"type": text_kind, "text": &text[start..end]}));
            }
            parts.push(image(i));
            start = end;
        }
        if start < text.len() || offsets.is_empty() {
            parts.push(json!({"type": text_kind, "text": &text[start..]}));
        }
        parts
    }

    /// Template-only marker content: never exposes image URLs or pixel payloads to Jinja.
    pub fn marker_json(text: &str, count: usize, offsets: &[usize]) -> Result<Value, String> {
        Self::validate_offsets(text, count, offsets)?;
        if count == 0 {
            return Ok(Value::String(text.to_string()));
        }
        Ok(Value::Array(Self::parts(text, offsets, "text", |_| json!({"type":"image"}))))
    }

    pub fn chat_json(&self) -> Result<Value, String> {
        self.validate_order()?;
        if self.images.is_empty() {
            return Ok(Value::String(self.text.clone()));
        }
        Ok(Value::Array(Self::parts(&self.text, &self.image_text_offsets, "text", |i|
            json!({"type":"image_url", "image_url":{"url":self.images[i]}}))))
    }

    pub fn responses_json(&self) -> Result<Value, String> {
        self.validate_order()?;
        Ok(Value::Array(Self::parts(&self.text, &self.image_text_offsets, "input_text", |i|
            json!({"type":"input_image", "image_url":self.images[i]}))))
    }
}
