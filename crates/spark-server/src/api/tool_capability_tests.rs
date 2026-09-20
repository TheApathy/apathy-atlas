// SPDX-License-Identifier: AGPL-3.0-only
//! Real HTTP pipeline admission: no inference/model implementation is supplied.
use super::glm_tool_publication_fixture as fixture;
use crate::tool_parser::{ToolCallFormat, ToolChoice};
use std::sync::Arc;

#[tokio::test]
async fn tool_capability_missing_parser_rejects_before_closed_scheduler_channel() {
    for choice in ["auto", "required", "specific"] {
        let mut case = fixture::case("raw_true_string");
        if choice == "specific" {
            case.choice =
                serde_json::from_value(serde_json::json!({"function":{"name":"inspect"}})).unwrap();
        } else {
            case.choice = ToolChoice::Mode(choice.into());
        }
        let req = fixture::request(&case);
        let mut state = fixture::app_state();
        Arc::get_mut(&mut state).unwrap().tool_call_parser = None;
        let result = super::chat_completions_inner(state, None, req, None).await;
        let super::ChatOutcome::Http(response) = result else {
            panic!("unsupported tools scheduled");
        };
        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    }
}

#[tokio::test]
async fn tool_capability_native_disabled_grammar_and_unsupported_schema_fail_at_admission() {
    for disabled in [true, false] {
        let mut case = fixture::case("raw_true_string");
        if !disabled {
            case.tools[0].function.parameters.as_mut().unwrap()["properties"]["value"]["pattern"] =
                serde_json::json!("^ok$");
        }
        let req = fixture::request(&case);
        let mut state = fixture::app_state();
        Arc::get_mut(&mut state)
            .unwrap()
            .behavior
            .disable_tool_grammar = disabled;
        let result = super::chat_completions_inner(state, None, req, None).await;
        let super::ChatOutcome::Http(response) = result else {
            panic!("unsupported native constraint scheduled");
        };
        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    }
}

#[test]
fn tool_capability_none_does_not_require_parser_or_grammar() {
    let case = fixture::case("raw_true_string");
    assert!(
        crate::tool_parser::request_admission::capability(
            &case.tools,
            Some(&ToolChoice::Mode("none".into())),
            None,
            true
        )
        .is_ok()
    );
    let parser = "mistral".parse::<ToolCallFormat>().unwrap().into_parser();
    assert!(!parser.has_tool_grammar());
    assert!(
        crate::tool_parser::request_admission::capability(
            &case.tools,
            Some(&ToolChoice::Mode("auto".into())),
            Some(parser.as_ref()),
            false
        )
        .is_ok()
    );
    assert!(
        crate::tool_parser::request_admission::capability(
            &case.tools,
            Some(&ToolChoice::Mode("required".into())),
            Some(parser.as_ref()),
            false
        )
        .is_err()
    );
}

#[tokio::test]
async fn tool_capability_anthropic_handler_preserves_client_error_classification() {
    use axum::{body::Bytes, extract::State};
    use serde_json::json;

    for case in ["invalid_schema", "unknown_specific", "malformed_choice"] {
        let mut body = json!({
            "model": "glm-publication-cpu", "max_tokens": 16,
            "messages": [{"role":"user", "content":"Record a note."}],
            "tools": [{"name":"record_note", "input_schema":{
                "type":"object", "properties":{"note":{"type":"string"}},
                "required":["note"], "additionalProperties":false
            }}], "tool_choice":{"type":"any"}
        });
        match case {
            "invalid_schema" => body["tools"][0]["input_schema"]["type"] = json!("array"),
            "unknown_specific" => body["tool_choice"] = json!({"type":"tool", "name":"unknown"}),
            "malformed_choice" => body["tool_choice"] = json!({"type":"tool"}),
            _ => unreachable!(),
        }
        let response = crate::anthropic::messages(
            State(fixture::app_state()),
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await;
        assert_eq!(
            response.status(),
            axum::http::StatusCode::BAD_REQUEST,
            "{case}"
        );
        let bytes = axum::body::to_bytes(response.into_body(), 16 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["type"], "error", "{case}: {value}");
        assert_eq!(
            value["error"]["type"], "invalid_request_error",
            "{case}: {value}"
        );
        assert!(
            value["error"]["message"]
                .as_str()
                .is_some_and(|s| !s.is_empty())
        );
    }
}
