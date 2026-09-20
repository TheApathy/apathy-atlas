// SPDX-License-Identifier: AGPL-3.0-only

//! The GLM native detector publishes only complete envelopes. Exercise that
//! real handler with its real StreamCtx, including the later legacy repairs.

use std::sync::{Arc, atomic::AtomicBool};

use crate::api::glm_tool_publication_fixture as fixture;
use crate::ir::StreamDelta;
use crate::tool_parser::ToolChoice;

use super::ctx::StreamCtx;
use super::state::StreamState;

fn check(name: &str) {
    let case = fixture::case(name);
    let parser = fixture::parser();
    let grammar_tools = case
        .tools
        .iter()
        .filter(|tool| match &case.choice {
            ToolChoice::Specific { function } => tool.function.name == function.name,
            _ => true,
        })
        .cloned()
        .collect();
    let ctx = StreamCtx {
        _active_guard: crate::metrics::ActiveRequestGuard::new(),
        state: fixture::app_state(),
        model: "glm-publication-cpu".into(),
        id: "cpu".into(),
        prompt_len: 0,
        enable_thinking: false,
        tool_defs_for_backfill: case.tools.clone(),
        cwd_for_normalize: None,
        stop_strings: Vec::new(),
        tool_retry_enabled: false,
        prompt_tokens: Arc::new(Vec::new()),
        prompt_vocab: Arc::new(Default::default()),
        grammar_spec: Some(crate::api::GrammarSpec::ToolCall {
            tools: grammar_tools,
            parser: parser.clone(),
            use_triggers: false,
        }),
        max_tokens: 64,
        timeout_at: None,
        stop_string_buffer_len: 0,
        leak_markers: parser.leak_markers(),
        wants_typed_arguments: parser.wants_typed_arguments(),
        max_tool_calls_per_response: 16,
        req_return_token_ids: false,
        req_ctx: None,
        dump_seq: None,
    };
    let mut state = StreamState::new(true, false, Arc::new(AtomicBool::new(false)), case.tools);
    // Do not invent a typed call: enter with the real parser's raw strings,
    // exactly as DetectorOutput::ToolCall reaches the complete handler.
    let (_, mut calls) = crate::tool_parser::parse_tool_calls(&case.wire);
    assert_eq!(
        calls.len(),
        1,
        "{name}: fixture must reach publication, not fail syntax parsing"
    );
    let mut deltas = Vec::new();
    super::tool_handlers::handle_complete_tool_call(
        &mut state,
        &ctx,
        &mut calls[0],
        0,
        &mut deltas,
    );
    let starts: Vec<_> = deltas
        .iter()
        .filter_map(|d| match d {
            StreamDelta::ToolCallStart { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    let fragments: Vec<_> = deltas
        .iter()
        .filter_map(|d| match d {
            StreamDelta::ToolCallArgs { fragment, .. } => Some(fragment.as_str()),
            _ => None,
        })
        .collect();
    if let Some(expected) = case.expected {
        assert_eq!(starts, vec!["inspect"], "{name}: {deltas:?}");
        assert_eq!(fragments.len(), 1, "{name}: {deltas:?}");
        let actual: serde_json::Value = serde_json::from_str(&fragments.concat()).unwrap();
        assert_eq!(actual, expected, "{name}: raw/schema value changed");
        assert_eq!(state.tool_calls_emitted_count, 1);
        assert!(!state.stop_string_triggered);
    } else {
        assert!(
            starts.is_empty() && fragments.is_empty(),
            "{name}: callable header/arguments escaped: {deltas:?}"
        );
        assert_eq!(
            state.tool_calls_emitted_count, 0,
            "{name}: invalid call counted as published"
        );
    }
}

macro_rules! publication_cases {
    ($($name:ident),+ $(,)?) => { $(
        #[test]
        fn $name() { check(stringify!($name)); }
    )+ };
}

publication_cases!(
    fractional_integer,
    wrong_object,
    missing_required_integer,
    missing_required_string,
    unknown_name,
    fuzzy_name,
    specific_wrong_name,
    closed_extra_key,
    renamed_key,
    raw_whitespace,
    raw_empty,
    raw_true_string,
    valid_typed_arguments
);
