// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// DeepSeek-V4 DSML format (the model's NATIVE tool syntax, from the
/// checkpoint's own `encoding/encoding_dsv4.py` `TOOLS_TEMPLATE`).
///
/// Outer envelope (`｜` is U+FF5C FULLWIDTH VERTICAL LINE):
/// ```text
/// <｜DSML｜tool_calls>
/// <｜DSML｜invoke name="$TOOL_NAME">
/// <｜DSML｜parameter name="$KEY" string="true|false">$VALUE</｜DSML｜parameter>
/// </｜DSML｜invoke>
/// </｜DSML｜tool_calls>
/// ```
///
/// String parameters are written raw with `string="true"`; every other type
/// (number, bool, array, object) is JSON with `string="false"`.
///
/// Parsing rides the shared scanning loop: `parse_dispatch` normalizes the
/// DSML envelope to the canonical `<tool_call>` / `<invoke name=..>` /
/// `<parameter name=..>` shapes (dropping the `string=` attribute), after
/// which the MiniMax-style inner parser and the schema-driven type coercion
/// handle the rest — `string="false"` JSON values coerce exactly like
/// qwen3_coder's raw-text values do.
pub struct DsmlV4Parser;

/// The DSML namespace token, with fullwidth bars as the checkpoint emits.
pub const DSML: &str = "\u{ff5c}DSML\u{ff5c}";

impl ToolCallParser for DsmlV4Parser {
    fn name(&self) -> &str {
        "dsml_v4"
    }

    fn system_prompt(&self, tools: &[ToolDefinition], tool_choice: &ToolChoice) -> String {
        // Mirrors the checkpoint's own `TOOLS_TEMPLATE` (encoding_dsv4.py)
        // minus the thinking-token clauses, which the serving layer owns.
        let mut prompt = format!(
            "## Tools\n\n\
             You have access to a set of tools to help answer the user's question. \
             You can invoke tools by writing a \"<{DSML}tool_calls>\" block like the following:\n\n\
             <{DSML}tool_calls>\n\
             <{DSML}invoke name=\"$TOOL_NAME\">\n\
             <{DSML}parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{DSML}parameter>\n\
             ...\n\
             </{DSML}invoke>\n\
             <{DSML}invoke name=\"$TOOL_NAME2\">\n\
             ...\n\
             </{DSML}invoke>\n\
             </{DSML}tool_calls>\n\n\
             String parameters should be specified as is and set `string=\"true\"`. \
             For all other types (numbers, booleans, arrays, objects), pass the value \
             in JSON format and set `string=\"false\"`.\n\n\
             ### Available Tool Schemas\n\n"
        );
        let body = tool_list_body(tools, || {
            let mut s = String::new();
            for tool in tools {
                let json = serde_json::to_string(tool).unwrap_or_default();
                s.push_str(&json);
                s.push('\n');
            }
            s
        });
        prompt.push_str(body.trim_end());
        prompt.push_str(
            "\n\nYou MUST strictly follow the above defined tool name and parameter \
             schemas to invoke tool calls.\n\n\
             Call a tool only when it supplies information or an action you cannot \
             produce yourself. Answer directly for facts you already know and for \
             arithmetic you can do reliably in your head. When a parameter's schema \
             does not state a format, pass the value in the tool's own notation \
             (for example, a calculator expression as `0.15*200`, not as prose).\n\n\
             This restraint does NOT apply to recovery: if a tool returns an error \
             or is unavailable, try another tool that could supply the same \
             information before telling the user you could not get it.\n",
        );
        append_tool_choice_instruction(&mut prompt, tool_choice);
        prompt
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        let mut out = String::new();
        out.push_str(&format!("\n<{DSML}tool_calls>\n"));
        for tc in calls {
            let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            out.push_str(&format!("<{DSML}invoke name=\"{}\">\n", tc.function.name));
            if let Some(obj) = args.as_object() {
                for (key, val) in obj {
                    let (is_str, val_str) = match val {
                        serde_json::Value::String(s) => ("true", s.clone()),
                        other => ("false", serde_json::to_string(other).unwrap_or_default()),
                    };
                    out.push_str(&format!(
                        "<{DSML}parameter name=\"{key}\" string=\"{is_str}\">{val_str}</{DSML}parameter>\n"
                    ));
                }
            }
            out.push_str(&format!("</{DSML}invoke>\n"));
        }
        out.push_str(&format!("</{DSML}tool_calls>"));
        out
    }

    fn format_tool_response(&self, content: &str) -> String {
        // Native template: tool_output_template = "<tool_result>{content}</tool_result>"
        format!("<tool_result>{content}</tool_result>")
    }
}

/// Normalize a DSML envelope to the canonical scanning shapes. Called from
/// `parse_dispatch` before the shared outer-tag loop; allocation only when
/// the DSML namespace token actually appears. Drops the `string="..."`
/// attribute — schema-driven coercion types the values downstream.
pub(super) fn normalize_dsml(text: &str) -> Option<String> {
    if !text.contains(DSML) {
        return None;
    }
    // DSML packs MULTIPLE `<invoke>` blocks inside one `tool_calls`
    // envelope; the shared scanning loop expects one call per
    // `<tool_call>` envelope. Drop the outer envelope (canonical AND the
    // live-observed BPE-broken `<｜DSML｜_calls>` merge) and give each
    // invoke its own.
    let s = text
        .replace(&format!("<{DSML}tool_calls>"), "")
        .replace(&format!("</{DSML}tool_calls>"), "")
        .replace(&format!("<{DSML}_calls>"), "")
        .replace(&format!("</{DSML}_calls>"), "")
        .replace(&format!("<{DSML}invoke "), "<tool_call><invoke ")
        .replace(&format!("</{DSML}invoke>"), "</invoke></tool_call>")
        .replace(&format!("<{DSML}parameter "), "<parameter ")
        .replace(&format!("</{DSML}parameter>"), "</parameter>")
        .replace(" string=\"true\">", ">")
        .replace(" string=\"false\">", ">");
    Some(s)
}

/// Streaming twin of [`normalize_dsml`]: rewrite COMPLETE DSML tags inside
/// the detector's accumulation buffer to the canonical shapes the scanning
/// loop knows. Partial (chunk-straddling) DSML tokens don't match any
/// replace and stay buffered — `safe_emit_len`'s DSML prefixes keep them
/// from leaking as content until the rest arrives. Idempotent: the rewrite
/// consumes every DSML token it matches.
pub(super) fn rewrite_dsml_in_buffer(buf: &mut String) {
    if !buf.contains(DSML) {
        return;
    }
    if let Some(s) = normalize_dsml(buf) {
        *buf = s;
    }
}

#[cfg(test)]
mod dsml_tests {
    use super::super::parse_dispatch::parse_tool_calls;
    use super::*;

    #[test]
    fn dsml_envelope_parses_to_tool_calls() {
        let text = format!(
            "Let me check the weather.\n\
             <{DSML}tool_calls>\n\
             <{DSML}invoke name=\"get_weather\">\n\
             <{DSML}parameter name=\"city\" string=\"true\">Berlin</{DSML}parameter>\n\
             <{DSML}parameter name=\"days\" string=\"false\">3</{DSML}parameter>\n\
             </{DSML}invoke>\n\
             </{DSML}tool_calls>"
        );
        let (content, calls) = parse_tool_calls(&text);
        assert_eq!(calls.len(), 1, "expected one call, got {calls:?}");
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "Berlin");
        assert!(
            args["days"] == 3 || args["days"] == "3",
            "days should be 3 (typed or string), got {:?}",
            args["days"]
        );
        assert!(content.unwrap().contains("weather"));
    }

    #[test]
    fn dsml_multi_invoke_parses_two_calls() {
        let text = format!(
            "<{DSML}tool_calls>\n\
             <{DSML}invoke name=\"a\">\n\
             <{DSML}parameter name=\"x\" string=\"true\">1</{DSML}parameter>\n\
             </{DSML}invoke>\n\
             <{DSML}invoke name=\"b\">\n\
             </{DSML}invoke>\n\
             </{DSML}tool_calls>"
        );
        let (_c, calls) = parse_tool_calls(&text);
        assert_eq!(calls.len(), 2, "expected two calls, got {calls:?}");
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.name, "b");
    }
}

#[cfg(test)]
mod dsml_bpe_tests {
    use super::super::parse_dispatch::parse_tool_calls;
    use super::*;

    #[test]
    fn dsml_bpe_broken_envelope_still_parses() {
        // Live-observed on GB10 (tool-eval-bench TC-01): the envelope token
        // BPE-merges to `<｜DSML｜_calls>`; invoke/parameter stay intact.
        let text = format!(
            "\n<{DSML}_calls>\n\
             <{DSML}invoke name=\"get_weather\">\n\
             <{DSML}parameter name=\"location\" string=\"true\">Berlin</{DSML}parameter>\n\
             </{DSML}invoke>\n\
             </{DSML}_calls>"
        );
        let (_c, calls) = parse_tool_calls(&text);
        assert_eq!(calls.len(), 1, "broken envelope should still parse: {calls:?}");
        assert_eq!(calls[0].function.name, "get_weather");
    }
}

#[cfg(test)]
mod dsml_stream_tests {
    use super::super::streaming::{DetectorOutput, StreamingToolDetector};
    use super::*;

    fn collect(det: &mut StreamingToolDetector, chunks: &[&str]) -> (String, Vec<String>) {
        let mut content = String::new();
        let mut calls = Vec::new();
        let mut handle = |o: DetectorOutput| match o {
            DetectorOutput::Content(c) => content.push_str(&c),
            DetectorOutput::ToolCall(tc, _) => calls.push(tc.function.name),
            DetectorOutput::ToolCallStart { name, .. } => calls.push(name),
            _ => {}
        };
        for ch in chunks {
            for o in det.process(ch) {
                handle(o);
            }
        }
        for o in det.flush() {
            handle(o);
        }
        (content, calls)
    }

    #[test]
    fn dsml_streams_to_tool_call_even_chunked() {
        // Split mid-DSML-token to exercise the straddle holdback.
        let text = format!(
            "Sure.\n<{DSML}_calls>\n<{DSML}invoke name=\"get_weather\">\n\
             <{DSML}parameter name=\"city\" string=\"true\">Berlin</{DSML}parameter>\n\
             </{DSML}invoke>\n</{DSML}_calls>"
        );
        let mid = text.find("DSML").unwrap() + 2; // split inside the first DSML token
        let (a, b) = text.split_at(mid);
        let mut det = StreamingToolDetector::new();
        let (content, calls) = collect(&mut det, &[a, b]);
        assert!(
            calls.iter().any(|n| n == "get_weather"),
            "expected get_weather in streamed calls, got calls={calls:?} content={content:?}"
        );
        assert!(
            !content.contains("DSML"),
            "DSML markup leaked into content: {content:?}"
        );
    }
}
