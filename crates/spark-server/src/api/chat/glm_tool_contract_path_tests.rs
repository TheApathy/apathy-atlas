// SPDX-License-Identifier: AGPL-3.0-only

//! Real IR -> MsgEntry -> JSON -> GLM template integration tests.
//! Binary-owned API dependencies stay out of the tokenizer library tests.

use crate::ir::{ContentPart, Message, Role};
use crate::tokenizer::tests::glm_tool_contract_template::render;
use serde_json::{Value, json};

fn msg(role: Role, text: &str) -> Message {
    Message {
        role,
        content: vec![ContentPart::Text(text.into())],
        tool_calls: vec![],
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }
}

fn history() -> Vec<Message> {
    let mut assistant = msg(Role::Assistant, "");
    assistant.tool_calls = vec![
        crate::ir::message::ToolCall {
            id: "a".into(),
            name: "inspect".into(),
            arguments: json!({"literal":"first"}),
        },
        crate::ir::message::ToolCall {
            id: "b".into(),
            name: "inspect".into(),
            arguments: json!({"literal":"second"}),
        },
    ];
    let mut result_b = msg(Role::Tool, "RESULT_B");
    result_b.tool_call_id = Some("b".into());
    let mut result_a = msg(Role::Tool, "RESULT_A");
    result_a.tool_call_id = Some("a".into());
    vec![
        msg(Role::User, "inspect both"),
        assistant,
        result_b,
        result_a,
    ]
}

fn actual_json(input: &[Message], tools_active: bool) -> Vec<Value> {
    let out = super::msg_entry::build_msg_entries(None, None, input, tools_active)
        .unwrap_or_else(|_| panic!("actual message builder rejected fixture"));
    super::template::build_json_messages(&out.messages).expect("valid test entries")
}

#[test]
fn glm_tool_contract_actual_message_builder_retains_result_association_ids() {
    for tools_active in [false, true] {
        let input = history();
        let untouched = input.clone();
        let messages = actual_json(&input, tools_active);
        assert_eq!(input, untouched, "input IR mutated");
        assert_eq!(messages[2]["tool_call_id"], "b");
        assert_eq!(messages[3]["tool_call_id"], "a");
        assert_eq!(messages[1]["tool_calls"][0]["id"], "a");
        assert_eq!(messages[1]["tool_calls"][1]["id"], "b");
    }
}

#[test]
fn glm_tool_contract_actual_message_path_renders_parallel_results_in_call_order() {
    let messages = actual_json(&history(), true);
    let rendered = render(&messages, false, None);
    let first = rendered
        .find("<tool_response>RESULT_A</tool_response>")
        .unwrap();
    let second = rendered
        .find("<tool_response>RESULT_B</tool_response>")
        .unwrap();
    assert!(
        first < second,
        "result arrival order must not override call-ID association"
    );
    assert_eq!(rendered.matches("<|observation|>").count(), 1);
    assert!(rendered.ends_with("<|assistant|><think></think>"));
}

#[test]
fn glm_tool_contract_unassociated_results_keep_input_order_without_guessing() {
    let mut input = history();
    input[2].tool_call_id = None;
    input[3].tool_call_id = None;
    let rendered = render(&actual_json(&input, true), false, None);
    assert!(rendered.find("RESULT_B").unwrap() < rendered.find("RESULT_A").unwrap());
}

#[test]
fn glm_tool_contract_thinking_disabled_omits_reasoning_effort_instruction() {
    let messages = actual_json(&[msg(Role::User, "hello")], false);
    let rendered = render(&messages, false, None);
    assert!(
        !rendered.contains("Reasoning Effort:"),
        "thinking=false must not inject a maximum-reasoning instruction: {rendered}"
    );
    assert!(rendered.starts_with("[gMASK]<sop><|user|>hello"));
    assert!(rendered.ends_with("<|assistant|><think></think>"));
}

#[test]
fn glm_tool_contract_thinking_enabled_keeps_requested_reasoning_effort() {
    let messages = actual_json(&[msg(Role::User, "hello")], false);
    let rendered = render(&messages, true, None);
    assert!(rendered.starts_with("[gMASK]<sop><|system|>Reasoning Effort: High"));
    assert_eq!(rendered.matches("Reasoning Effort:").count(), 1);
    assert!(rendered.ends_with("<|assistant|><think>"));
}
