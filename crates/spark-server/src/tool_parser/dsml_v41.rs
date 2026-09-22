// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// DeepSeek-V4.1 DSML (`<｜DSML｜ calls>` / ` invoke` / ` parameter`, with a
/// LEADING SPACE in every tag name, unlike V4).
///
/// This parser is a registration and a name, not the parser: for
/// `deepseek_v41` the prompt is rendered and the output parsed by
/// `crate::dsv41` (the production Python server's semantics, see
/// `api/dsv41.rs`), and neither goes through the generic tool pipeline. The
/// tool schemas reach the model through the renderer, so there is no
/// behavioural system prompt to inject.
pub struct DsmlV41Parser;

impl ToolCallParser for DsmlV41Parser {
    fn name(&self) -> &str {
        "dsml_v41"
    }

    fn system_prompt(&self, _tools: &[ToolDefinition], _tool_choice: &ToolChoice) -> String {
        String::new()
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        use crate::dsv41::encoding::{CALLS_BLOCK, DSML, INVOKE_TAG, PARAM_TAG};
        use crate::dsv41::pyjson::dumps;
        let mut invokes = Vec::with_capacity(calls.len());
        for tc in calls {
            let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            let params: Vec<String> = args
                .as_object()
                .into_iter()
                .flatten()
                .map(|(k, v)| {
                    let (is_str, val) = match v {
                        serde_json::Value::String(s) => ("true", s.clone()),
                        other => ("false", dumps(other)),
                    };
                    format!("<{DSML}{PARAM_TAG} name=\"{k}\" string=\"{is_str}\">{val}</{DSML}{PARAM_TAG}>")
                })
                .collect();
            invokes.push(format!(
                "<{DSML}{INVOKE_TAG} name=\"{}\">\n{}\n</{DSML}{INVOKE_TAG}>",
                tc.function.name,
                params.join("\n")
            ));
        }
        format!(
            "\n\n<{DSML}{CALLS_BLOCK}>\n{}\n</{DSML}{CALLS_BLOCK}>",
            invokes.join("\n")
        )
    }

    fn format_tool_response(&self, content: &str) -> String {
        format!("<tool_result>{content}</tool_result>")
    }
}
