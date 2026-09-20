// SPDX-License-Identifier: AGPL-3.0-only
use super::super::types::MessagesRequest;
use serde_json::{Value, json};

fn parse(choice: Value) -> Result<MessagesRequest, serde_json::Error> {
    serde_json::from_value(
        json!({"model":"glm53-flash","messages":[{"role":"user","content":"OK"}],"max_tokens":16,"tool_choice":choice}),
    )
}

#[test]
fn tool_choice_admission_anthropic_never_rewrites_invalid_choice_to_auto() {
    for choice in [
        json!({"type":"bogus"}),
        json!({"type":"required"}),
        json!({"type":"tool"}),
        json!({"type":"tool","name":""}),
        json!({"type":"tool","name":" "}),
    ] {
        assert!(
            parse(choice.clone()).is_err(),
            "must reject original wire choice {choice}"
        );
    }
}

#[test]
fn tool_choice_admission_anthropic_valid_choices_lower_without_loss() {
    for (wire, expected) in [("auto", "auto"), ("none", "none"), ("any", "required")] {
        let ir = crate::ir::ChatRequest::from(parse(json!({"type":wire})).unwrap());
        assert!(
            matches!(ir.tool_choice,Some(crate::tool_parser::ToolChoice::Mode(mode)) if mode==expected)
        );
    }
    let ir =
        crate::ir::ChatRequest::from(parse(json!({"type":"tool","name":"record_note"})).unwrap());
    assert!(
        matches!(ir.tool_choice,Some(crate::tool_parser::ToolChoice::Specific{function}) if function.name=="record_note")
    );
}
