// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in CPU gate against the deployed GLM ByteLevel tokenizer, not RAW test
//! token spellings. Root pins the supplied tokenizer before and after the run.

use crate::grammar::{GrammarEngine, GrammarState};
use crate::tool_parser::{ToolCallFormat, ToolDefinition};

#[test]
#[ignore = "requires ATLAS_GLM_TOOL_TOKENIZER pointing to the pinned checkpoint tokenizer.json"]
fn glm_tool_contract_native_bytelevel_tokenizer_masks_real_wire() {
    let path = std::env::var("ATLAS_GLM_TOOL_TOKENIZER").expect("explicit tokenizer path required");
    let tokenizer = tokenizers::Tokenizer::from_file(path).unwrap();
    for (text, id) in [
        ("<tool_call>", 154843),
        ("</tool_call>", 154844),
        ("<arg_key>", 154847),
        ("</arg_key>", 154848),
        ("<arg_value>", 154849),
        ("</arg_value>", 154850),
    ] {
        assert_eq!(
            tokenizer.token_to_id(text),
            Some(id),
            "native token identity changed"
        );
    }
    let stop = [154820, 154827, 154829];
    let mut engine = GrammarEngine::from_tokenizer(&tokenizer, Some(154880), &stop).unwrap();
    let parser = "glm_xml".parse::<ToolCallFormat>().unwrap().into_parser();
    let definitions: Vec<ToolDefinition> = serde_json::from_value(serde_json::json!([{
        "type":"function", "function":{"name":"inspect", "parameters":{
            "type":"object", "properties":{"value":{"type":"string"}},
            "required":["value"], "additionalProperties":false
        }}
    }]))
    .unwrap();
    let stop: Vec<u32> = stop.into_iter().map(|id| id as u32).collect();
    for auto in [false, true] {
        let compiled = parser
            .compile_tool_grammar(&mut engine, &definitions, auto)
            .unwrap()
            .unwrap();
        for value in [
            "",
            " \t\n  ",
            "雪🙂",
            "false",
            "{\"x\":false}",
            "</tool_call> is literal",
            "<",
            "</arg_val",
            "<<",
        ] {
            let wire = format!(
                "<tool_call>inspect<arg_key>value</arg_key><arg_value>{value}</arg_value></tool_call>"
            );
            let ids = tokenizer.encode(wire, false).unwrap();
            let mut state = GrammarState::new(&compiled, engine.vocab_size())
                .unwrap()
                .with_stop_tokens(&stop);
            assert_eq!(state.stop_legal(&stop), auto);
            for &id in ids.get_ids() {
                assert!(state.fill_bitmask());
                assert!(
                    state.is_token_allowed(id),
                    "auto={auto}, value={value:?}, token={id}"
                );
                assert!(state.accept_token(id));
            }
            assert!(state.stop_legal(&stop), "complete native call cannot stop");
            let history = state.num_history_steps();
            assert!(state.accept_token(stop[0]));
            assert_eq!(state.num_history_steps(), history);
            state.rollback(history);
            assert_eq!(state.num_history_steps(), 0);
            assert_eq!(state.stop_legal(&stop), auto);
        }
    }
}
