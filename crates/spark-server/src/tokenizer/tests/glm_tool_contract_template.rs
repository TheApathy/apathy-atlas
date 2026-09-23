// SPDX-License-Identifier: AGPL-3.0-only

//! JSON-only GLM template tests and the shared rendering fixture.
//! API/IR integration cases live beneath api::chat.

use super::super::{jinja_helpers, normalize_tool_call_arguments};
use serde_json::{Value, json};

const TEMPLATE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../jinja-templates/glm5_next.jinja"
));

pub(crate) fn render(messages: &[Value], thinking: bool, tools: Option<&[Value]>) -> String {
    let source = jinja_helpers::convert_python_jinja_to_minijinja(TEMPLATE);
    let env = jinja_helpers::build_jinja_env(&source).expect("real GLM template compiles");
    let normalized = normalize_tool_call_arguments(messages);
    let tools = tools
        .map(minijinja::Value::from_serialize)
        .unwrap_or(minijinja::Value::UNDEFINED);
    env.get_template("chat")
        .unwrap()
        .render(minijinja::context! {
            messages => normalized, tools => tools, add_generation_prompt => true,
            enable_thinking => thinking, reasoning_effort => if thinking { "high" } else { "none" },
            disable_tool_steering => false, add_vision_id => false,
        })
        .expect("real GLM template renders")
}

#[test]
fn glm_tool_contract_native_template_history_and_argument_normalization_are_preserved() {
    let messages = [
        json!({"role":"assistant", "content":"", "tool_calls":[{
            "id":"a", "type":"function", "function":{"name":"inspect",
                "arguments":"{\"literal\":\" \\tfalse\\n雪 \",\"enabled\":false}"}
        }]}),
        json!({"role":"tool", "tool_call_id":"a", "content":"done"}),
    ];
    let original = messages.clone();
    let rendered = render(&messages, false, None);
    assert_eq!(messages, original);
    assert!(rendered.contains("<tool_call>inspect"));
    assert!(rendered.contains("<arg_key>literal</arg_key><arg_value> \tfalse\n雪 </arg_value>"));
    assert!(rendered.contains("<arg_key>enabled</arg_key><arg_value>false</arg_value>"));
    assert!(rendered.contains("<|observation|><tool_response>done</tool_response>"));
    assert!(!rendered.contains("<function="));
}

#[test]
fn glm_tool_contract_template_keeps_native_tools_images_and_thinking_contract() {
    // This is template-only marker input: no image decoding or fake GPU encoder.
    let messages = [json!({"role":"user", "content":[
        {"type":"text","text":"RED:"}, {"type":"image"},
        {"type":"text","text":"BLUE:"}, {"type":"image"}
    ]})];
    let tools = [json!({"type":"function", "function":{"name":"inspect",
        "parameters":{"type":"object", "properties":{"literal":{"type":"string"}}}}})];
    let rendered = render(&messages, false, Some(&tools));
    assert!(rendered.contains("<tool_call>{function-name}<arg_key>"));
    assert!(rendered.contains(concat!(
        "<|user|>RED:<|begin_of_image|><|image|><|end_of_image|>",
        "BLUE:<|begin_of_image|><|image|><|end_of_image|>"
    )));
    assert!(!rendered.contains("unable to process"));
    assert!(rendered.ends_with("<|assistant|><think></think>"));
    assert!(render(&messages, true, Some(&tools)).ends_with("<|assistant|><think>"));
}

#[test]
fn glm_tool_contract_json_thinking_disabled_omits_reasoning_effort_instruction() {
    let messages = [json!({"role":"user", "content":"hello"})];
    let rendered = render(&messages, false, None);
    assert!(
        !rendered.contains("Reasoning Effort:"),
        "thinking=false must not inject a maximum-reasoning instruction: {rendered}"
    );
    assert!(rendered.starts_with("[gMASK]<sop><|user|>hello"));
    assert!(rendered.ends_with("<|assistant|><think></think>"));
}

#[test]
fn glm_tool_contract_json_thinking_enabled_keeps_requested_reasoning_effort() {
    let messages = [json!({"role":"user", "content":"hello"})];
    let rendered = render(&messages, true, None);
    assert!(rendered.starts_with("[gMASK]<sop><|system|>Reasoning Effort: High"));
    assert_eq!(rendered.matches("Reasoning Effort:").count(), 1);
    assert!(rendered.ends_with("<|assistant|><think>"));
}
