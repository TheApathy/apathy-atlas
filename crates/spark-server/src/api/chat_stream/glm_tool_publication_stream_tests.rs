// SPDX-License-Identifier: AGPL-3.0-only

//! The GLM native detector publishes only complete envelopes. Exercise the
//! native validation `handle_complete_tool_call` runs for glm_xml, with the
//! grammar-bound (tool_choice-filtered) tool set the stream context carries.
//! all-models port: this tree has no AppState test fixture, so the handler's
//! native step (`validate_native_glm`) is exercised directly.

use crate::api::glm_tool_publication_fixture as fixture;
use crate::tool_parser::ToolChoice;

fn check(name: &str) {
    let case = fixture::case(name);
    let grammar_tools: Vec<_> = case
        .tools
        .iter()
        .filter(|tool| match &case.choice {
            ToolChoice::Specific { function } => tool.function.name == function.name,
            _ => true,
        })
        .cloned()
        .collect();
    // Do not invent a typed call: enter with the real parser's raw strings,
    // exactly as DetectorOutput::ToolCall reaches the complete handler.
    let (_, mut calls) = crate::tool_parser::parse_tool_calls(&case.wire);
    assert_eq!(
        calls.len(),
        1,
        "{name}: fixture must reach publication, not fail syntax parsing"
    );
    let result = super::tool_handlers::validate_native_glm(&mut calls[0], Some(&grammar_tools));
    if let Some(expected) = case.expected {
        assert!(result.is_ok(), "{name}: {result:?}");
        assert_eq!(calls[0].function.name, "inspect");
        let actual: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(actual, expected, "{name}: raw/schema value changed");
    } else {
        assert!(result.is_err(), "{name}: callable escaped: {:?}", calls[0]);
    }
}

#[test]
fn missing_grammar_binding_fails_closed() {
    let case = fixture::case("raw_true_string");
    let (_, mut calls) = crate::tool_parser::parse_tool_calls(&case.wire);
    assert!(super::tool_handlers::validate_native_glm(&mut calls[0], None).is_err());
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
