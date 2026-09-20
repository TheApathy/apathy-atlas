// SPDX-License-Identifier: AGPL-3.0-only
//! Actual ordinary-policy + native XGrammar regression for the P7 GPU failure.
use super::*;
use crate::grammar::{GrammarEngine, GrammarState};
use crate::tool_parser::{ToolCallFormat, ToolDefinition};
use serde_json::json;

fn native_grammar(prefix: &[u32], auto: bool) -> GrammarState {
    let vocabulary = [
        "<tool_call>",
        "inspect<arg_key>value</arg_key><arg_value>x</arg_value>",
        "</tool_call>",
        "<eos>",
        " ",
    ]
    .map(str::to_owned);
    let definition:ToolDefinition=serde_json::from_value(json!({"type":"function","function":{"name":"inspect","parameters":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":false}}})).unwrap();
    let mut engine = GrammarEngine::new(&vocabulary, &[3]).unwrap();
    let compiled = "glm_xml"
        .parse::<ToolCallFormat>()
        .unwrap()
        .into_parser()
        .compile_tool_grammar(&mut engine, &[definition], auto)
        .unwrap()
        .unwrap();
    let mut state = GrammarState::new(&compiled, 5)
        .unwrap()
        .with_stop_tokens(&[3]);
    for &id in prefix {
        assert!(state.accept_token(id));
    }
    state
}

fn tool_context() -> LogitsContext {
    LogitsContext {
        tool_call_start_token: Some(0),
        tool_call_end_token: Some(2),
        ..context()
    }
}

#[test]
fn native_tool_boundary_close_is_accepted_once_and_survives_policy_restore() {
    let mut live = sequence();
    live.grammar_state = Some(native_grammar(&[0, 1], false));
    live.output_tokens = vec![0, 1];
    live.tools_present = true;
    live.inside_tool_body = true;
    live.tool_call_start_token = Some(0);
    let before = fingerprint(&live);
    let ctx = tool_context();
    {
        let mut policy =
            OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, options(false)).unwrap();
        policy.checkpoint().unwrap();
        assert_eq!(
            policy
                .pick(0, &row(&[-20.0, -20.0, 20.0, -20.0, -20.0]))
                .unwrap(),
            2
        );
        policy
            .advance(2)
            .expect("one close token must not be accepted twice");
        assert_eq!(
            policy
                .sequence()
                .grammar_state
                .as_ref()
                .unwrap()
                .num_history_steps(),
            3
        );
        assert!(policy.sequence().tool_call_completed);
        policy.restore().unwrap();
        assert!(policy.take_prepared().is_ok());
    }
    assert_eq!(fingerprint(&live), before);
}

#[test]
fn native_tool_boundary_runaway_open_is_not_accepted_twice() {
    let mut live = sequence();
    live.grammar_state = Some(native_grammar(&[0, 1, 2], true));
    live.output_tokens = vec![0, 1, 2];
    live.tools_present = true;
    live.tool_call_completed = true;
    live.post_completion_tool_opens = 7;
    live.tool_call_start_token = Some(0);
    let ctx = tool_context();
    let mut policy = OrdinaryVerifyPolicy::new(&MODEL, &mut live, &ctx, options(false)).unwrap();
    policy.checkpoint().unwrap();
    assert_eq!(
        policy
            .pick(0, &row(&[20.0, -20.0, -20.0, -20.0, -20.0]))
            .unwrap(),
        0
    );
    assert!(
        !policy
            .advance(0)
            .expect("runaway cap must not duplicate grammar advancement")
    );
    assert_eq!(
        policy
            .sequence()
            .grammar_state
            .as_ref()
            .unwrap()
            .num_history_steps(),
        4
    );
    policy.restore().unwrap();
}
