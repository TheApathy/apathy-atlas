// SPDX-License-Identifier: AGPL-3.0-only

//! Runs the actual blocking publication function, including all post-parser
//! processing. A parser-only success cannot satisfy these regressions.

use crate::api::glm_tool_publication_fixture as fixture;

fn check(name: &str) {
    let case = fixture::case(name);
    let state = fixture::app_state();
    let req = fixture::request(&case);
    let choice = super::build_choice_message(
        &state,
        &req,
        &fixture::response(),
        None,
        case.wire.clone(),
        true,
        None,
        0,
    );
    if let Some(expected) = case.expected {
        assert_eq!(choice.tool_calls.len(), 1, "{name}: {choice:?}");
        assert_eq!(choice.tool_calls[0].name, "inspect");
        assert_eq!(
            choice.tool_calls[0].arguments, expected,
            "{name}: raw/schema value changed"
        );
        assert_eq!(choice.finish_reason, crate::ir::FinishReason::ToolCalls);
    } else {
        assert!(
            choice.tool_calls.is_empty(),
            "{name}: malformed callable escaped: {choice:?}"
        );
        assert_ne!(
            choice.finish_reason,
            crate::ir::FinishReason::ToolCalls,
            "{name}: rejected call still marked callable"
        );
        assert!(
            choice
                .content
                .as_deref()
                .is_some_and(|text| text.contains("[atlas] Tool call rejected:")),
            "{name}: rejected native call must not become an unexplained empty response"
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
