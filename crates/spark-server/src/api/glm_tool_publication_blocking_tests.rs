// SPDX-License-Identifier: AGPL-3.0-only

//! Runs the blocking publication helpers that `build_choice_message` uses for
//! the native GLM parser, fed by the real parser. A parser-only success cannot
//! satisfy these regressions. all-models port: `build_choice_message` needs
//! an AppState this tree has no CPU fixture for, so its two GLM steps
//! (validation, rejection feedback) are exercised directly.

use crate::api::glm_tool_publication_fixture as fixture;

fn check(name: &str) {
    let case = fixture::case(name);
    let (content, calls) = crate::tool_parser::parse_tool_calls(&case.wire);
    assert_eq!(calls.len(), 1, "{name}: fixture must reach publication");
    let validated =
        super::validate_parsed_tool_calls(true, calls, &case.tools, Some(&case.choice), None);
    let content = super::native_rejection_feedback(content, &validated.errors);
    if let Some(expected) = case.expected {
        assert_eq!(validated.valid.len(), 1, "{name}: {:?}", validated.errors);
        assert!(validated.errors.is_empty(), "{name}");
        assert_eq!(validated.valid[0].function.name, "inspect");
        let actual: serde_json::Value =
            serde_json::from_str(&validated.valid[0].function.arguments).unwrap();
        assert_eq!(actual, expected, "{name}: raw/schema value changed");
    } else {
        assert!(
            validated.valid.is_empty(),
            "{name}: malformed callable escaped: {:?}",
            validated.valid
        );
        assert!(
            content
                .as_deref()
                .is_some_and(|text| text.contains("[atlas] Tool call rejected:")),
            "{name}: rejected native call must not become an unexplained empty response"
        );
    }
}

#[test]
fn legacy_parsers_keep_backfill_and_repair_pipeline() {
    // The same wire through the non-native branch still runs the legacy
    // repairs (here: required-string backfill), proving the GLM branch is
    // the only one that skips them.
    let case = fixture::case("missing_required_string");
    let (_, calls) = crate::tool_parser::parse_tool_calls(&case.wire);
    let native = super::validate_parsed_tool_calls(
        true,
        calls.clone(),
        &case.tools,
        Some(&case.choice),
        None,
    );
    let mut legacy_calls = calls.clone();
    crate::tool_parser::backfill_required_params(&mut legacy_calls, &case.tools);
    let expected_legacy = crate::tool_parser::validate_tool_calls(legacy_calls, &case.tools);
    let legacy =
        super::validate_parsed_tool_calls(false, calls, &case.tools, Some(&case.choice), None);
    assert!(native.valid.is_empty());
    assert_eq!(legacy.errors, expected_legacy.errors);
    assert_eq!(
        legacy
            .valid
            .iter()
            .map(|call| (&call.function.name, &call.function.arguments))
            .collect::<Vec<_>>(),
        expected_legacy
            .valid
            .iter()
            .map(|call| (&call.function.name, &call.function.arguments))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        super::native_rejection_feedback(Some("kept".into()), &[]),
        Some("kept".into())
    );
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
