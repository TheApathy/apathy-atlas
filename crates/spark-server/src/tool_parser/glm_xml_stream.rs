// SPDX-License-Identifier: AGPL-3.0-only

//! Complete native-envelope publication, shared by both streaming modes.

use super::{DetectorOutput, StreamingToolDetector, glm_xml, glm_xml_scan};

pub(super) enum NativeStep {
    NotNative,
    Consumed,
    Pending,
}

impl StreamingToolDetector {
    pub(super) fn process_glm_native(&mut self, outputs: &mut Vec<DetectorOutput>) -> NativeStep {
        if !self.inside_tag {
            let Some(start) = glm_xml_scan::first_native_opener(&self.buffer) else {
                return NativeStep::NotNative;
            };
            let before = self.buffer[..start].to_string();
            self.buffer = self.buffer[start + glm_xml_scan::OPEN.len()..].to_string();
            self.inside_tag = true;
            if !before.is_empty() {
                outputs.push(DetectorOutput::Content(before));
            }
            return NativeStep::Consumed;
        }
        if !glm_xml_scan::native_prefix(&self.buffer) {
            return NativeStep::NotNative;
        }
        let Some(end) = glm_xml_scan::native_close(&self.buffer) else {
            return NativeStep::Pending;
        };
        let body = &self.buffer[..end];
        if let Some(call) = glm_xml::parse_body(body) {
            outputs.push(DetectorOutput::ToolCall(call, self.call_counter as usize));
            self.call_counter += 1;
            self.emitted_tool_calls = true;
        } else {
            outputs.push(DetectorOutput::Content(format!(
                "<tool_call>{body}</tool_call>"
            )));
        }
        self.buffer = self.buffer[end + glm_xml_scan::CLOSE.len()..].to_string();
        self.inside_tag = false;
        self.reset_call_state();
        NativeStep::Consumed
    }
}
