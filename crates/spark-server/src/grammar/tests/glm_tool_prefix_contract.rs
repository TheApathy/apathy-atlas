// SPDX-License-Identifier: AGPL-3.0-only

//! Native-only literal-close boundaries through the production compiler.

use super::{StopLegal, test_vocab};
use crate::grammar::{GrammarEngine, GrammarState};
use crate::tool_parser::ToolDefinition;
use serde_json::json;
use xgrammar::{CompiledGrammar, GrammarMatcher};

fn compile(auto: bool) -> (CompiledGrammar, usize) {
    let mut vocab = test_vocab();
    vocab.push("</arg_value>".into());
    let mut engine = GrammarEngine::new(&vocab, &[130]).unwrap();
    let tools: Vec<ToolDefinition> = serde_json::from_value(json!([{
        "type":"function", "function":{"name":"inspect", "parameters":{
            "type":"object", "properties":{"value":{"type":"string"}},
            "required":["value"], "additionalProperties":false
        }}
    }]))
    .unwrap();
    (
        engine.compile_glm_xml_tools(&tools, auto).unwrap(),
        vocab.len(),
    )
}

fn accepts(compiled: &CompiledGrammar, text: &str) -> bool {
    let mut matcher = GrammarMatcher::new(compiled, None, true, -1).unwrap();
    matcher.accept_string(text, false) && matcher.is_terminated()
}

fn wire(value: &str) -> String {
    format!("<tool_call>inspect<arg_key>value</arg_key><arg_value>{value}</arg_value></tool_call>")
}

#[test]
fn glm_tool_contract_prefix_every_proper_close_prefix_can_end_a_value() {
    let close = "</arg_value>";
    for auto in [false, true] {
        let (compiled, _) = compile(auto);
        for end in 0..close.len() {
            let value = &close[..end];
            assert!(
                accepts(&compiled, &wire(value)),
                "auto={auto}, value={value:?}"
            );
        }
    }
}

#[test]
fn glm_tool_contract_prefix_repeated_openers_and_overlap_mismatches_preserved() {
    let (compiled, _) = compile(false);
    for value in [
        "<<",
        "<<<",
        "x<<</arg_val",
        "</arg_val</arg_val",
        "</arg_valX>",
        "</arg_value<",
        "<</arg_value",
        "</arg_value><",
    ] {
        let valid = !value.contains("</arg_value>");
        assert_eq!(accepts(&compiled, &wire(value)), valid, "value={value:?}");
    }
}

#[test]
fn glm_tool_contract_prefix_complete_delimiter_cannot_be_swallowed_by_overlap() {
    let (compiled, _) = compile(false);
    for value in [
        "</arg_value>tail",
        "<</arg_value>tail",
        "<<<</arg_value>tail",
        "</arg_val</arg_value>tail",
    ] {
        assert!(
            !accepts(&compiled, &wire(value)),
            "swallowed native close: {value:?}"
        );
    }
}

#[test]
fn glm_tool_contract_prefix_token_mask_accepts_atomic_close_after_partial_prefix() {
    let (compiled, size) = compile(false);
    for value in ["", "<", "</arg_val", "<<"] {
        let mut state = GrammarState::new(&compiled, size).unwrap();
        let prefix = format!("<tool_call>inspect<arg_key>value</arg_key><arg_value>{value}");
        for token in prefix.bytes().map(u32::from) {
            state.fill_bitmask();
            assert!(
                state.is_token_allowed(token),
                "value={value:?}, token={token}"
            );
            assert!(state.accept_token(token));
        }
        assert!(!state.stop_legal(&[130]));
        state.fill_bitmask();
        assert!(state.is_token_allowed(131), "atomic close after {value:?}");
        assert!(state.accept_token(131));
        state.fill_bitmask();
        assert!(!state.is_token_allowed(b'x' as u32), "value already closed");
        assert!(state.is_token_allowed(129));
        assert!(state.accept_token(129));
        assert!(state.stop_legal(&[130]));
    }
}
