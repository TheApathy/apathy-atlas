// SPDX-License-Identifier: AGPL-3.0-only

//! GLM native name + arg_key/arg_value wire format. No heuristic repairs,
//! inferred JSON types, or publication before the complete envelope validates.

use super::glm_xml_scan::{CLOSE, KEY, KEY_CLOSE, OPEN, VALUE, VALUE_CLOSE};
use super::*;

pub struct GlmXmlParser;

impl ToolCallParser for GlmXmlParser {
    fn name(&self) -> &str {
        "glm_xml"
    }

    fn compile_tool_grammar(
        &self,
        engine: &mut GrammarEngine,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        Some(engine.compile_glm_xml_tools(tools, use_triggers))
    }

    fn has_tool_grammar(&self) -> bool {
        true
    }

    fn system_prompt(&self, tools: &[ToolDefinition], tool_choice: &ToolChoice) -> String {
        let body = tool_list_body(tools, || {
            tools
                .iter()
                .map(|tool| {
                    serde_json::to_string(&tool.function)
                        .expect("tool definition is JSON serializable")
                })
                .collect::<Vec<_>>()
                .join("\n")
        });
        let mut prompt = format!(
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
            You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{body}\n</tools>\n\n\
            For each function call, output the function name and arguments within the following XML format:\n\
            <tool_call>{{function-name}}<arg_key>{{arg-key}}</arg_key><arg_value>{{arg-value}}</arg_value>...</tool_call>"
        );
        append_tool_choice_instruction(&mut prompt, tool_choice);
        prompt
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        // The legacy trait cannot return a formatting error. Never manufacture
        // empty arguments or publish a prefix of an unrepresentable history.
        match calls.iter().map(format_call).collect::<Option<Vec<_>>>() {
            Some(calls) => calls.concat(),
            None => {
                tracing::warn!("GLM tool history is not representable in native XML");
                String::new()
            }
        }
    }
}

fn valid_key(key: &str) -> bool {
    !key.is_empty() && !key.trim().is_empty() && !key.contains(['<', '>'])
}

/// Called only after the native dispatcher has established a real outer close.
/// Without schemas the native wire is ambiguous, so all values stay strings.
pub(super) fn parse_body(body: &str) -> Option<ToolCall> {
    let body = body.trim();
    let name_end = body.find('<').unwrap_or(body.len());
    let name = body[..name_end].trim();
    if !is_tool_name_component(name) {
        return None;
    }
    let mut rest = &body[name_end..];
    let mut args = serde_json::Map::new();
    while !rest.trim_start().is_empty() {
        rest = rest.trim_start().strip_prefix(KEY)?;
        let key_end = rest.find(KEY_CLOSE)?;
        let key = &rest[..key_end];
        if !valid_key(key) || args.contains_key(key) {
            return None;
        }
        rest = rest[key_end + KEY_CLOSE.len()..]
            .trim_start()
            .strip_prefix(VALUE)?;
        let value_end = rest.find(VALUE_CLOSE)?;
        // Do not trim, decode entities, normalize markup or parse JSON here.
        args.insert(
            key.to_string(),
            serde_json::Value::String(rest[..value_end].to_string()),
        );
        rest = &rest[value_end + VALUE_CLOSE.len()..];
    }
    Some(ToolCall {
        id: next_tool_call_id(),
        call_type: "function".into(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: serde_json::to_string(&args).ok()?,
        },
    })
}

fn format_call(call: &IncomingToolCall) -> Option<String> {
    if !is_tool_name_component(&call.function.name) {
        return None;
    }
    let args: serde_json::Value = serde_json::from_str(&call.function.arguments).ok()?;
    let mut output = format!("{OPEN}{}", call.function.name);
    for (key, value) in args.as_object()? {
        if !valid_key(key) {
            return None;
        }
        let value = match value {
            serde_json::Value::String(text) => text.clone(),
            value => serde_json::to_string(value).ok()?,
        };
        if value.contains(VALUE_CLOSE) {
            return None; // Native template provides no escape for this delimiter.
        }
        output.push_str(KEY);
        output.push_str(key);
        output.push_str(KEY_CLOSE);
        output.push_str(VALUE);
        output.push_str(&value);
        output.push_str(VALUE_CLOSE);
    }
    output.push_str(CLOSE);
    Some(output)
}
