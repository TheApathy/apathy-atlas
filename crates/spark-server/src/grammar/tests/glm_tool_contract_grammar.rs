// SPDX-License-Identifier: AGPL-3.0-only

//! Register beneath grammar::tests. Real parser-selected GrammarEngine and
//! GrammarState; no test-generated grammar, model, tokenizer file or GPU.
//! The RAW fixture tests byte/special-token splits, not the actual GLM BPE.

use super::StopLegal;
use crate::grammar::{GrammarEngine, GrammarState};
use crate::tool_parser::{ToolCallFormat, ToolDefinition};
use serde_json::{Value, json};
use xgrammar::{CompiledGrammar, GrammarMatcher};

const OPEN: u32 = 128;
const CLOSE: u32 = 129;
const EOS: u32 = 130;
const KEY_OPEN: u32 = 131;

fn vocab() -> Vec<String> {
    let mut tokens = super::test_vocab();
    tokens.extend(
        [
            "<arg_key>",
            "</arg_key>",
            "<arg_value>",
            "</arg_value>",
            "<tool_call>inspect<arg_key>",
            "</arg_key><arg_value>",
            "\n</arg_value>",
            "</arg_value></tool_call>",
            "雪🙂",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    tokens
}

fn tool(name: &str, value_schema: Value) -> ToolDefinition {
    serde_json::from_value(json!({
        "type": "function", "function": {
            "name": name, "parameters": {
                "type": "object", "properties": {"value": value_schema},
                "required": ["value"], "additionalProperties": false
            }
        }
    }))
    .unwrap()
}

fn tools() -> Vec<ToolDefinition> {
    ["inspect", "inspect_more"]
        .into_iter()
        .map(|name| tool(name, json!({"type": "string"})))
        .collect()
}

fn native_parser() -> Box<dyn crate::tool_parser::ToolCallParser> {
    let format = "glm_xml"
        .parse::<ToolCallFormat>()
        .expect("native GLM parser must be registered, not aliased to Qwen/Hermes");
    assert_eq!(format.name(), "glm_xml");
    assert!(
        format.has_grammar(),
        "native grammar capability must be truthful"
    );
    format.into_parser()
}

fn compile(defs: &[ToolDefinition], auto: bool) -> (CompiledGrammar, usize) {
    let tokens = vocab();
    let mut engine = GrammarEngine::new(&tokens, &[EOS as i32]).unwrap();
    let compiled = native_parser()
        .compile_tool_grammar(&mut engine, defs, auto)
        .expect("native parser must provide a grammar, never an unconstrained None")
        .expect("native tool schema must compile through the actual GrammarEngine");
    (compiled, engine.vocab_size())
}

fn wire(name: &str, value: &str) -> String {
    format!("<tool_call>{name}<arg_key>value</arg_key><arg_value>{value}</arg_value></tool_call>")
}

fn accepts(compiled: &CompiledGrammar, input: &str) -> bool {
    let mut matcher = GrammarMatcher::new(compiled, None, true, -1).unwrap();
    matcher.accept_string(input, false) && matcher.is_terminated()
}

fn feed(state: &mut GrammarState, ids: impl IntoIterator<Item = u32>) {
    for id in ids {
        state.fill_bitmask();
        assert!(
            state.is_token_allowed(id),
            "native grammar masks token {id}"
        );
        assert!(state.accept_token(id), "native matcher refuses token {id}");
    }
}

fn ascii(state: &mut GrammarState, input: &str) {
    assert!(
        input.is_ascii(),
        "ASCII fixture helper must not split Unicode bytes"
    );
    feed(state, input.bytes().map(u32::from));
}

#[test]
fn glm_tool_contract_grammar_required_masks_prose_and_eos_from_first_token() {
    let (compiled, size) = compile(&tools(), false);
    let mut state = GrammarState::new(&compiled, size).unwrap();
    assert!(state.fill_bitmask());
    assert!(state.is_token_allowed(OPEN));
    for illegal in [b'H' as u32, b'{' as u32, CLOSE, EOS] {
        assert!(
            !state.is_token_allowed(illegal),
            "initial token {illegal} escaped"
        );
    }
    assert!(!state.stop_legal(&[EOS]));
    assert!(accepts(&compiled, &wire("inspect", "x")));
    assert!(!accepts(
        &compiled,
        &format!("Sure. {}", wire("inspect", "x"))
    ));
}

#[test]
fn glm_tool_contract_grammar_auto_prose_is_optional_but_open_arms_name_masks() {
    let (compiled, size) = compile(&tools(), true);
    let mut state = GrammarState::new(&compiled, size).unwrap();
    ascii(&mut state, "I can inspect that. ");
    assert!(state.stop_legal(&[EOS]), "auto prose alone may finish");
    feed(&mut state, [OPEN]);
    assert!(state.fill_bitmask());
    assert!(state.is_token_allowed(b'i' as u32));
    for illegal in [OPEN, CLOSE, EOS, b'<' as u32, b'g' as u32] {
        assert!(
            !state.is_token_allowed(illegal),
            "post-open token {illegal} escaped"
        );
    }
    ascii(
        &mut state,
        "inspect<arg_key>value</arg_key><arg_value>x</arg_value>",
    );
    feed(&mut state, [CLOSE]);
    assert!(state.stop_legal(&[EOS]));
}

#[test]
fn glm_tool_contract_grammar_required_supports_distinct_prefix_sharing_names() {
    let (compiled, _) = compile(&tools(), false);
    for name in ["inspect", "inspect_more"] {
        assert!(
            accepts(&compiled, &wire(name, "x")),
            "registered name {name}"
        );
    }
    for name in ["inspec", "inspect_more_more", "ghost"] {
        assert!(
            !accepts(&compiled, &wire(name, "x")),
            "unregistered name {name}"
        );
    }
}

#[test]
fn glm_tool_contract_grammar_specific_filtered_set_cannot_emit_other_tool() {
    // sampling_setup.rs passes a name-filtered tool list and auto=false for
    // Specific. This checks that engine contract, NOT the HTTP filtering code.
    let definitions = tools();
    let (compiled, size) = compile(&definitions[1..2], false);
    assert!(accepts(&compiled, &wire("inspect_more", "x")));
    assert!(!accepts(&compiled, &wire("inspect", "x")));
    let mut state = GrammarState::new(&compiled, size).unwrap();
    feed(&mut state, [OPEN]);
    ascii(&mut state, "inspect");
    assert!(state.fill_bitmask());
    assert!(state.is_token_allowed(b'_' as u32));
    assert!(!state.is_token_allowed(KEY_OPEN));
    assert!(!state.is_token_allowed(b'<' as u32));
}

#[test]
fn glm_tool_contract_grammar_native_typed_values_compile_without_json_envelope() {
    for (schema, value) in [
        (json!({"type":"integer"}), "32"),
        (json!({"type":"number"}), "-2.5"),
        (json!({"type":"boolean"}), "false"),
        (json!({"type":"array","items":{"type":"integer"}}), "[1, 2]"),
        (
            json!({"type":"object","properties":{"x":{"type":"boolean"}},
            "required":["x"],"additionalProperties":false}),
            "{\"x\":false}",
        ),
        (json!({"type":"null"}), "null"),
    ] {
        let (compiled, _) = compile(&[tool("inspect", schema)], false);
        assert!(
            accepts(&compiled, &wire("inspect", value)),
            "native value {value:?}"
        );
    }
    // Admission of valid typed values does not claim arbitrary JSON Schema
    // validation or that this grammar alone replaces parser-side coercion.
}

#[test]
fn glm_tool_contract_grammar_required_string_preserves_empty_whitespace_and_json_text() {
    let (compiled, _) = compile(&tools(), false);
    for value in [
        "",
        " \t\n  ",
        "false",
        "32",
        "[1]",
        "{\"x\":false}",
        " 雪🙂 ",
    ] {
        assert!(
            accepts(&compiled, &wire("inspect", value)),
            "raw string {value:?}"
        );
    }
}

#[test]
fn glm_tool_contract_grammar_value_closes_only_at_native_arg_value_delimiter() {
    let (compiled, _) = compile(&tools(), false);
    for value in [
        "Vec<T> <= x",
        "<div>ok</div>",
        "</arg_valX>",
        "literal </tool_call> and <arg_key> are inside the string",
    ] {
        assert!(
            accepts(&compiled, &wire("inspect", value)),
            "literal value {value:?}"
        );
    }
}

#[test]
fn glm_tool_contract_grammar_rejects_foreign_frames_and_malformed_native_pairs() {
    let (compiled, _) = compile(&tools(), false);
    for input in [
        "<tool_call>\n<function=inspect>\n<parameter=value>x</parameter>\n</function>\n</tool_call>",
        "<tool_call>{\"name\":\"inspect\",\"arguments\":{\"value\":\"x\"}}</tool_call>",
        "{\"name\":\"inspect\",\"arguments\":{\"value\":\"x\"}}",
        "<tool_call></tool_call>",
        "<tool_call>inspect<arg_value>x</arg_value></tool_call>",
        "<tool_call>inspect<arg_key>value</arg_key>x</tool_call>",
        "<tool_call>inspect<arg_key>value<arg_value>x</arg_value></tool_call>",
        "<tool_call>inspect<arg_key>value</arg_key><arg_value>x</tool_call>",
        "<tool_call>inspect<arg_key>value</arg_key><arg_value>x</arg_value>",
    ] {
        assert!(
            !accepts(&compiled, input),
            "illegal wire admitted: {input:?}"
        );
    }
}

#[test]
fn glm_tool_contract_grammar_auto_rejects_foreign_wire_after_native_trigger() {
    let (compiled, size) = compile(&tools(), true);
    let mut state = GrammarState::new(&compiled, size).unwrap();
    feed(&mut state, [OPEN]);
    assert!(
        !state.accept_token(b'<' as u32),
        "Qwen function opener must be refused"
    );
    assert!(state.fill_bitmask());
    assert!(
        state.is_token_allowed(b'i' as u32),
        "refusal must not consume state"
    );
    assert!(
        !state.is_token_allowed(b'{' as u32),
        "Hermes JSON body must be masked"
    );
}

#[test]
fn glm_tool_contract_grammar_ascii_atomic_and_cross_boundary_tokens_agree() {
    let (compiled, size) = compile(&tools(), false);
    let input = wire("inspect", " x\n");
    let mut bytes = GrammarState::new(&compiled, size).unwrap();
    ascii(&mut bytes, &input);
    assert!(bytes.stop_legal(&[EOS]));
    let mut merged = GrammarState::new(&compiled, size).unwrap();
    let tokens = vocab();
    for fragment in [
        "<tool_call>inspect<arg_key>",
        "value",
        "</arg_key><arg_value>",
        " x",
        "\n</arg_value>",
        "</tool_call>",
    ] {
        if let Some(id) = tokens.iter().position(|token| token == fragment) {
            feed(&mut merged, [id as u32]);
        } else {
            ascii(&mut merged, fragment);
        }
    }
    assert!(merged.stop_legal(&[EOS]));
    bytes.fill_bitmask();
    merged.fill_bitmask();
    assert_eq!(bytes.bitmask_data(), merged.bitmask_data());
}

#[test]
fn glm_tool_contract_grammar_rollback_restores_name_choice_masks_and_stop_legality() {
    let (compiled, size) = compile(&tools(), false);
    let mut state = GrammarState::new(&compiled, size).unwrap();
    feed(&mut state, [OPEN]);
    ascii(&mut state, "inspect");
    assert!(state.fill_bitmask());
    let mask = state.bitmask_data().to_vec();
    let checkpoint = state.num_history_steps();
    assert!(!state.stop_legal(&[EOS]));
    ascii(
        &mut state,
        "_more<arg_key>value</arg_key><arg_value>A</arg_value>",
    );
    feed(&mut state, [CLOSE]);
    assert!(state.stop_legal(&[EOS]));
    assert!(state.accept_token(EOS));
    assert!(state.is_terminated());
    // all-models port: no EOS exemption here, so EOS is an ordinary
    // rollback step and is rewound with the rest of the call.
    let accepted = state.num_history_steps();
    state.rollback(accepted - checkpoint);
    assert_eq!(state.num_history_steps(), checkpoint);
    assert!(state.fill_bitmask());
    assert_eq!(state.bitmask_data(), mask.as_slice());
    assert!(!state.stop_legal(&[EOS]));
    ascii(
        &mut state,
        "<arg_key>value</arg_key><arg_value>B</arg_value>",
    );
    feed(&mut state, [CLOSE]);
    assert!(state.stop_legal(&[EOS]));
    state.reset();
    assert_eq!(state.num_history_steps(), 0);
    assert!(!state.stop_legal(&[EOS]));
}

#[test]
fn glm_tool_contract_grammar_empty_specific_selection_fails_compilation_closed() {
    let mut engine = GrammarEngine::new(&vocab(), &[EOS as i32]).unwrap();
    let result = native_parser()
        .compile_tool_grammar(&mut engine, &[], false)
        .expect("native grammar must return an explicit compile result");
    assert!(
        result.is_err(),
        "empty filtered tool set cannot become unconstrained grammar"
    );
}
