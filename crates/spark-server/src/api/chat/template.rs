// SPDX-License-Identifier: AGPL-3.0-only
//
// Chat-template rendering: build the JSON-message array, run the
// optional auto-compact, apply the Jinja template, expand image
// pad-tokens, and detect template-forced thinking.
//
// Lifted out of `chat::chat_completions_inner` (wave 4g).

use axum::http::StatusCode;
use axum::response::Response;
use std::sync::Arc;

use crate::AppState;

use super::super::compact::{compact_messages, openai_error_response};
use super::msg_entry::MsgEntry;

/// Outputs of [`render_template`]. Threaded into the streaming /
/// blocking dispatch.
pub(super) struct TemplateOut {
    pub(super) prompt_tokens: Vec<u32>,
    /// Possibly overridden by template-forced-thinking detection.
    pub(super) enable_thinking: bool,
    pub(super) thinking_budget: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::result_large_err)]
pub(super) fn render_template(
    state: &Arc<AppState>,
    tools: &[crate::tool_parser::ToolDefinition],
    messages: &[MsgEntry],
    image_pad_counts: &[usize],
    enable_thinking: bool,
    thinking_budget: Option<u32>,
    tools_active: bool,
) -> Result<TemplateOut, Response> {
    // Use closed thinking when client doesn't explicitly enable it.
    let template_thinking = enable_thinking;

    // Build JSON messages with structured tool_calls for Jinja.
    let has_images = !image_pad_counts.is_empty();
    if image_pad_counts.contains(&0) {
        return Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            "Invalid zero image pad count".into(),
        ));
    }
    let json_messages = build_json_messages(messages)
        .map_err(|e| openai_error_response(StatusCode::BAD_REQUEST, e))?;
    // When TSCG is enabled the parser's `system_prompt()` has already
    // placed the compact tool signatures into messages[0]; passing
    // `tools` to Jinja as well would re-render the full JSON schema and
    // defeat the compaction. Pass `None` so the template's `{% if tools
    // %}` branch falls through — the tool-call format instructions
    // still come from `system_prompt()`.
    let jinja_tools: Option<Vec<serde_json::Value>> =
        if tools_active && !crate::tscg::tscg_enabled() {
            Some(
                tools
                    .iter()
                    .map(|t| serde_json::to_value(t).unwrap_or_default())
                    .collect(),
            )
        } else {
            None
        };

    // Progressive auto-compact (DISABLED BY DEFAULT 2026-04-25 —
    // see project_no_auto_compaction memory feedback).
    let auto_compact_active = state
        .auto_compact_threshold
        .map(|t| t > 0.0)
        .unwrap_or(false);
    let json_messages = if !has_images && auto_compact_active && json_messages.len() > 4 {
        let trial_tokens = state
            .tokenizer
            .apply_chat_template_openai(
                &json_messages,
                jinja_tools.as_deref(),
                template_thinking,
                state.behavior.disable_tool_steering,
            )
            .map(|t| t.len())
            .unwrap_or(0);
        if trial_tokens > (state.max_seq_len as f32 * 0.70) as usize {
            compact_messages(&json_messages, trial_tokens, state.max_seq_len)
        } else {
            json_messages
        }
    } else {
        json_messages
    };

    let prompt_tokens = match state.tokenizer.apply_chat_template_openai(
        &json_messages,
        jinja_tools.as_deref(),
        template_thinking,
        state.behavior.disable_tool_steering,
    ) {
        Ok(t) => t,
        Err(e) => {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Tokenization error: {e}"),
            ));
        }
    };

    // GLM-5.3's published template is text-only and renders this exact
    // reminder for each structured image item. The model checkpoint itself is
    // multimodal, so turn only that tokenizer-exact span into the native media
    // triplet. Models whose templates already emit image-pad tokens bypass
    // this compatibility seam unchanged.
    let image_pad_id = state
        .vision_config
        .as_ref()
        .map(|vision| vision.image_pad_token_id)
        .filter(|&token| token != 0);
    let prompt_tokens = if image_pad_counts.is_empty() {
        prompt_tokens
    } else if let Some(image_pad_id) = image_pad_id {
        match materialize_glm_image_placeholders(
            &state.tokenizer,
            prompt_tokens,
            image_pad_counts.len(),
            image_pad_id,
        ) {
            Ok(tokens) => tokens,
            Err(error) => {
                return Err(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Image template error: {error:#}"),
                ));
            }
        }
    } else {
        return Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            "Image template error: model vision config has no image-pad token".to_string(),
        ));
    };

    // Expand image pads when needed.
    let prompt_tokens = if image_pad_counts.iter().any(|&c| c > 1) {
        expand_image_pads_with_id(
            prompt_tokens,
            image_pad_id.expect("nonempty image counts require a pad id"),
            image_pad_counts,
        )
    } else {
        prompt_tokens
    };

    // Template-forced thinking detection.
    let (enable_thinking, thinking_budget) = if let Some(think_start) = state.think_start_token_id {
        let tail = &prompt_tokens[prompt_tokens.len().saturating_sub(8)..];
        let last_start = tail.iter().rposition(|t| *t == think_start);
        let has_unclosed_think = match (last_start, state.think_end_token_id) {
            (Some(si), Some(end_tok)) => !tail[si + 1..].contains(&end_tok),
            (Some(_), None) => true,
            (None, _) => false,
        };
        if has_unclosed_think && !enable_thinking {
            tracing::info!(
                "Template-forced thinking detected (unclosed \\<think\\> in prompt tail) — \
                 overriding enable_thinking=true with budget={}",
                state.behavior.max_thinking_budget,
            );
            (true, Some(state.behavior.max_thinking_budget))
        } else {
            (enable_thinking, thinking_budget)
        }
    } else {
        (enable_thinking, thinking_budget)
    };

    Ok(TemplateOut {
        prompt_tokens,
        enable_thinking,
        thinking_budget,
    })
}

fn materialize_glm_image_placeholders(
    tokenizer: &crate::tokenizer::ChatTokenizer,
    tokens: Vec<u32>,
    image_count: usize,
    pad_id: u32,
) -> anyhow::Result<Vec<u32>> {
    let existing = tokens.iter().filter(|&&token| token == pad_id).count();
    if existing == image_count {
        return Ok(tokens);
    }
    anyhow::ensure!(
        existing == 0,
        "rendered prompt has {existing} image pads for {image_count} images"
    );
    let reminder = tokenizer.encode(
        "<reminder>You are unable to process this image because you don't have multi-modal input ability. Try different methods.</reminder>",
    )?;
    anyhow::ensure!(!reminder.is_empty(), "GLM image reminder encoded empty");
    let singleton = |text: &str| -> anyhow::Result<u32> {
        let encoded = tokenizer.encode(text)?;
        anyhow::ensure!(
            encoded.len() == 1,
            "GLM media marker {text} must encode to one token"
        );
        Ok(encoded[0])
    };
    let replacement = [
        singleton("<|begin_of_image|>")?,
        pad_id,
        singleton("<|end_of_image|>")?,
    ];
    replace_exact_spans(tokens, &reminder, &replacement, image_count)
}

fn expand_image_pads_with_id(tokens: Vec<u32>, pad_id: u32, counts: &[usize]) -> Vec<u32> {
    let extra: usize = counts.iter().map(|count| count.saturating_sub(1)).sum();
    let mut output = Vec::with_capacity(tokens.len() + extra);
    let mut image = 0usize;
    for token in tokens {
        if token == pad_id {
            output.extend(std::iter::repeat_n(
                pad_id,
                counts.get(image).copied().unwrap_or(1).max(1),
            ));
            image += 1;
        } else {
            output.push(token);
        }
    }
    output
}

fn replace_exact_spans(
    tokens: Vec<u32>,
    needle: &[u32],
    replacement: &[u32],
    expected: usize,
) -> anyhow::Result<Vec<u32>> {
    anyhow::ensure!(
        !needle.is_empty(),
        "image placeholder needle must not be empty"
    );
    let mut output = Vec::with_capacity(tokens.len());
    let mut cursor = 0usize;
    let mut replaced = 0usize;
    while cursor < tokens.len() {
        if tokens[cursor..].starts_with(needle) {
            output.extend_from_slice(replacement);
            cursor += needle.len();
            replaced += 1;
        } else {
            output.push(tokens[cursor]);
            cursor += 1;
        }
    }
    anyhow::ensure!(
        replaced == expected,
        "rendered prompt contained {replaced} GLM image reminders for {expected} images"
    );
    Ok(output)
}

/// Build the Jinja-facing JSON message array from the processed
/// [`MsgEntry`] vec. Pure (no tokenizer/state) so it can be
/// characterization-tested directly:
///   * `image_count == 0` → `content` is a plain string,
///   * `image_count > 0`  → `content` is an ordered text/image marker array
///     (text part omitted when empty),
///   * `tool_calls` / `reasoning_content` attached when present.
///
/// Roles arrive canonical (`developer` → `system` normalization happens
/// at MsgEntry build time).
pub(super) fn build_json_messages(messages: &[MsgEntry]) -> Result<Vec<serde_json::Value>, String> {
    messages
        .iter()
        .map(|m| {
            let content_val = crate::openai::ParsedContent::marker_json(
                &m.content,
                m.image_count,
                &m.image_text_offsets,
            )?;
            // `developer` → `system` normalization happens upstream at
            // MsgEntry build time (msg_entry.rs), so the role arrives
            // canonical here.
            let mut msg = serde_json::json!({"role": m.role, "content": content_val});
            if let Some(ref tcs) = m.tool_calls {
                msg["tool_calls"] = serde_json::Value::Array(tcs.clone());
            }
            if let Some(ref id) = m.tool_call_id {
                msg["tool_call_id"] = serde_json::Value::String(id.clone());
            }
            // F1: forward historical reasoning trace to the Jinja
            // template. The template at qwen3_5_moe.jinja:90-104
            // reads `message.reasoning_content` and rehydrates the
            // `<think>` block for the historical assistant turns.
            // Without this, every historical assistant message
            // rendered an empty `<think>\n\n</think>\n\n` wrapper
            // (empty-think poisoning, → premature `<|im_end|>`).
            if let Some(ref rc) = m.reasoning_content {
                msg["reasoning_content"] = serde_json::Value::String(rc.clone());
            }
            Ok(msg)
        })
        .collect()
}

#[cfg(test)]
mod json_message_tests {
    use super::MsgEntry;
    use super::build_json_messages;

    #[test]
    fn exact_image_reminder_replacement_is_counted_and_ordered() {
        let tokens = vec![1, 7, 8, 2, 7, 8, 3];
        let replaced = super::replace_exact_spans(tokens, &[7, 8], &[10, 11, 12], 2).unwrap();
        assert_eq!(replaced, [1, 10, 11, 12, 2, 10, 11, 12, 3]);
        assert!(super::replace_exact_spans(vec![7, 8], &[7, 8], &[9], 2).is_err());
        assert!(super::replace_exact_spans(vec![7, 8], &[], &[9], 1).is_err());
        assert_eq!(
            super::expand_image_pads_with_id(vec![1, 9, 2, 9, 3], 9, &[2, 3]),
            [1, 9, 9, 2, 9, 9, 9, 3]
        );
    }

    fn entry(role: &str, content: &str, image_count: usize) -> MsgEntry {
        MsgEntry {
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
            image_count,
            image_text_offsets: vec![0; image_count],
            reasoning_content: None,
        }
    }

    /// PROMPT-STABILITY GATE. Characterization golden for the full
    /// `Vec<ir::Message>` → `build_msg_entries` → `build_json_messages`
    /// path — the exact JSON the Jinja chat template consumes. Any
    /// change here changes rendered prompts, which breaks kv-cache
    /// prefix reuse across requests of the same conversation. Update
    /// ONLY for an intentional, documented behavior change.
    #[test]
    fn prompt_json_stability_gate() {
        use crate::ir::message::{Reasoning, ToolCall};
        use crate::ir::{ContentPart, Message, Role};

        fn text_msg(role: Role, t: &str) -> Message {
            Message {
                role,
                content: vec![ContentPart::Text(t.into())],
                tool_calls: Vec::new(),
                tool_call_id: None,
                name: None,
                reasoning: None,
                tool_error: false,
            }
        }

        let mut assistant = text_msg(Role::Assistant, "Sure.");
        assistant.reasoning = Some(Reasoning {
            text: "plan the verification".into(),
        });
        assistant.tool_calls = vec![
            ToolCall {
                id: "c1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({"command": "cargo test"}),
            },
            ToolCall {
                id: "c2".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path": "a.rs"}),
            },
        ];
        let mut tool_result = text_msg(Role::Tool, "total 0");
        tool_result.tool_call_id = Some("c1".into());
        tool_result.name = Some("bash".into());

        let msgs = vec![
            text_msg(
                Role::System,
                "You are helpful.\nworking directory: /tmp/proj",
            ),
            text_msg(Role::Other("developer".into()), "be terse"),
            text_msg(Role::User, "run the tests"),
            assistant,
            tool_result,
            text_msg(Role::User, "thanks"),
        ];

        let out = super::super::msg_entry::build_msg_entries(None, None, &msgs, true)
            .expect("fixture builds");
        assert_eq!(out.cwd_hint.as_deref(), Some("/tmp/proj"));
        let json = build_json_messages(&out.messages).unwrap();

        let expected = serde_json::json!([
            {
                "role": "system",
                "content": "You are helpful.\nworking directory: /tmp/proj\n<environment>\nworking_directory: /tmp/proj\n</environment>"
            },
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "run the tests"},
            {
                "role": "assistant",
                "content": "Sure.",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": {"command": "cargo test"}}},
                    {"id": "c2", "type": "function", "function": {"name": "read", "arguments": {"path": "a.rs"}}}
                ],
                "reasoning_content": "plan the verification"
            },
            // The original tool-call association must survive message lowering.
            {"role": "tool", "content": "total 0", "tool_call_id": "c1"},
            {"role": "user", "content": "thanks"}
        ]);
        assert_eq!(
            serde_json::Value::Array(json.clone()),
            expected,
            "prompt JSON drifted — this breaks kv-cache prefix stability:\n{}",
            serde_json::to_string_pretty(&json).unwrap()
        );
    }

    #[test]
    fn plain_text_message_serializes_to_string_content() {
        let out = build_json_messages(&[entry("user", "hi", 0)]).unwrap();
        assert_eq!(
            out,
            vec![serde_json::json!({"role": "user", "content": "hi"})]
        );
    }

    #[test]
    fn images_expand_to_structured_content_array_with_text_last() {
        let out = build_json_messages(&[entry("user", "look", 2)]).unwrap();
        assert_eq!(
            out,
            vec![serde_json::json!({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "look"}
                ]
            })]
        );
    }

    #[test]
    fn empty_text_with_images_omits_text_part() {
        let out = build_json_messages(&[entry("user", "", 1)]).unwrap();
        assert_eq!(
            out,
            vec![serde_json::json!({"role": "user", "content": [{"type": "image"}]})]
        );
    }

    #[test]
    fn tool_calls_and_reasoning_are_attached() {
        let mut e = entry("assistant", "", 0);
        e.tool_calls = Some(vec![serde_json::json!({"id": "c1"})]);
        e.reasoning_content = Some("because".to_string());
        let out = build_json_messages(&[e]).unwrap();
        assert_eq!(
            out,
            vec![serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1"}],
                "reasoning_content": "because"
            })]
        );
    }
}
