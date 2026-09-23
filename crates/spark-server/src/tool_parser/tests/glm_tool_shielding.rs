// SPDX-License-Identifier: AGPL-3.0-only

use super::super::*;

#[test]
fn glm_tool_contract_malformed_native_name_cannot_dispatch_a_foreign_inner_call() {
    for input in [
        "<tool_call>inspect<function=ghost><parameter=value>x</parameter></function></tool_call>",
        "<tool_call>inspect<invoke name=\"ghost\"><parameter name=\"value\">x</parameter></invoke></tool_call>",
    ] {
        assert!(
            parse_tool_calls(input).1.is_empty(),
            "foreign call recovered from native body"
        );
        for buffered in [false, true] {
            for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
                let mut detector = StreamingToolDetector::new();
                detector.buffer_args = buffered;
                for event in detector
                    .process(&input[..split])
                    .into_iter()
                    .chain(detector.process(&input[split..]))
                    .chain(detector.flush())
                {
                    assert!(
                        matches!(event, DetectorOutput::Content(_)),
                        "native malformed call published at split {split}"
                    );
                }
            }
        }
    }
}

#[test]
fn glm_tool_contract_native_values_keep_think_minimax_and_mistral_markers_verbatim() {
    for raw in [
        "x</think>y",
        "<minimax:tool_call>x</minimax:tool_call>",
        "<minimax:_call>x</minimax:_call>",
        "[TOOL_CALLS]ghost[ARGS]{\"x\":1}",
    ] {
        let input = format!(
            "<tool_call>inspect<arg_key>value</arg_key><arg_value>{raw}</arg_value></tool_call>"
        );
        let (content, calls) = parse_tool_calls(&input);
        assert!(content.is_none());
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, serde_json::json!({"value":raw}));
        for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
            let mut detector = StreamingToolDetector::new();
            let events: Vec<_> = detector
                .process(&input[..split])
                .into_iter()
                .chain(detector.process(&input[split..]))
                .chain(detector.flush())
                .collect();
            assert_eq!(events.len(), 1, "raw content leaked at split {split}");
            let DetectorOutput::ToolCall(call, 0) = &events[0] else {
                panic!("not a complete call")
            };
            assert_eq!(call.function.name, "inspect");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&call.function.arguments).unwrap(),
                args
            );
        }
    }
}
