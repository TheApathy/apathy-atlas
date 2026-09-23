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
use crate::openai::{ChatCompletionRequest, ParsedContent};
use crate::tool_parser;

use super::super::compact::{compact_messages, openai_error_response, truncate_to_fit};
use super::super::failures::strip_xml_leaks_from_assistant_content;
use super::msg_entry::MsgEntry;

/// Whether the hard context-overflow truncation safety net is active
/// (Atlas task #76). Default ON; `ATLAS_CTX_OVERFLOW_TRUNCATE=0` restores the
/// strict 400-on-overflow behavior. Cached so the hot path is a relaxed load.
fn ctx_overflow_truncate_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("ATLAS_CTX_OVERFLOW_TRUNCATE").as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        )
    })
}

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
    req: &ChatCompletionRequest,
    messages: &[MsgEntry],
    image_pad_counts: &[usize],
    enable_thinking: bool,
    thinking_budget: Option<u32>,
    tools_active: bool,
) -> Result<TemplateOut, Response> {
    // Use closed thinking when client doesn't explicitly enable it.
    let template_thinking = enable_thinking;
    let image_count: usize = messages.iter().map(|m| m.image_count).sum();
    let has_images = image_count != 0;
    if image_count != image_pad_counts.len() || image_pad_counts.contains(&0) {
        return Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            "Image markers and preprocessed images must have matching nonzero counts".into(),
        ));
    }

    // Build JSON messages with structured tool_calls for Jinja.
    let stripper_tools: &[tool_parser::ToolDefinition] = req.tools.as_deref().unwrap_or(&[]);
    // GLM's template orders parallel tool results by `tool_call_id`. Only the
    // native GLM contract forwards it, so no other template's input changes.
    let forward_tool_call_id = state
        .tool_call_parser
        .as_ref()
        .is_some_and(|parser| parser.name() == "glm_xml");
    let json_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            let effective_content = if m.role == "assistant"
                && m.tool_calls.as_ref().is_some_and(|tcs| !tcs.is_empty())
                && !stripper_tools.is_empty()
                && m.image_count == 0
            {
                strip_xml_leaks_from_assistant_content(&m.content, stripper_tools)
            } else {
                m.content.clone()
            };
            let content_val = ParsedContent::marker_json(
                &effective_content,
                m.image_count,
                &m.image_text_offsets,
            )?;
            let mut msg = serde_json::json!({"role": m.role, "content": content_val});
            if let Some(ref tcs) = m.tool_calls {
                msg["tool_calls"] = serde_json::Value::Array(tcs.clone());
            }
            if forward_tool_call_id && let Some(ref id) = m.tool_call_id {
                msg["tool_call_id"] = serde_json::Value::String(id.clone());
            }
            Ok::<_, String>(msg)
        })
        .collect::<Result<_, _>>()
        .map_err(|e| openai_error_response(StatusCode::BAD_REQUEST, e))?;
    // When TSCG is enabled the parser's `system_prompt()` has already
    // placed the compact tool signatures into messages[0]; passing
    // `tools` to Jinja as well would re-render the full JSON schema and
    // defeat the compaction. Pass `None` so the template's `{% if tools
    // %}` branch falls through — the tool-call format instructions
    // still come from `system_prompt()`.
    let jinja_tools: Option<Vec<serde_json::Value>> =
        if tools_active && !crate::tscg::tscg_enabled() {
            req.tools.as_ref().map(|ts| {
                ts.iter()
                    .map(|t| serde_json::to_value(t).unwrap_or_default())
                    .collect()
            })
        } else {
            None
        };

    // Progressive auto-compact (DISABLED BY DEFAULT 2026-04-25 —
    // see project_no_auto_compaction memory feedback).
    let auto_compact_active = state
        .auto_compact_threshold
        .map(|t| t > 0.0)
        .unwrap_or(false);
    // Pixel ownership is request-wide: do not drop/rewrite image-bearing history.
    let json_messages = if !has_images && auto_compact_active && json_messages.len() > 4 {
        let trial_tokens = state
            .tokenizer
            .apply_chat_template_openai_with_options(
                &json_messages,
                jinja_tools.as_deref(),
                template_thinking,
                state.behavior.disable_tool_steering,
                req.requested_reasoning_effort(),
                req.chat_template_kwargs
                    .as_ref()
                    .and_then(|kw| kw.preserve_thinking)
                    .or(state.behavior.preserve_thinking),
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

    let tokenize = |jm: &[serde_json::Value]| {
        state.tokenizer.apply_chat_template_openai_with_options(
            jm,
            jinja_tools.as_deref(),
            template_thinking,
            state.behavior.disable_tool_steering,
            req.requested_reasoning_effort(),
            req.chat_template_kwargs
                .as_ref()
                .and_then(|kw| kw.preserve_thinking)
                .or(state.behavior.preserve_thinking),
        )
    };
    let mut prompt_tokens = match tokenize(&json_messages) {
        Ok(t) => t,
        Err(e) => {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Tokenization error: {e}"),
            ));
        }
    };

    // ── Hard context-overflow safety net (Atlas task #76) ──
    // Deep agentic multi-turn conversations (long ops briefing + accumulated
    // tool outputs) can overflow max_seq_len. Previously this either 400'd or
    // wedged the scheduler to the 900 s harness ceiling (Spark-Bench AG-11/12).
    // Reserve room for output (max_thinking_budget on think-on runs is the
    // dominant term — the champion serves 768) and, if the prompt won't fit,
    // drop the OLDEST turns and re-render — bounded, never an infinite loop.
    // Opt out with ATLAS_CTX_OVERFLOW_TRUNCATE=0 to restore strict 400-on-overflow.
    if !has_images && ctx_overflow_truncate_enabled() {
        let output_reserve = (state.behavior.max_thinking_budget as usize).saturating_add(256);
        let fit_budget = state.max_seq_len.saturating_sub(output_reserve).max(1);
        if prompt_tokens.len() >= fit_budget {
            let mut jm = json_messages.clone();
            // keep_tail shrinks each pass (6 → 4 → 2) so we always make progress.
            for keep_tail in [6usize, 4, 2] {
                match truncate_to_fit(&jm, keep_tail) {
                    Some(truncated) => {
                        let dropped = jm.len().saturating_sub(truncated.len());
                        jm = truncated;
                        match tokenize(&jm) {
                            Ok(t) => {
                                let fits = t.len() < fit_budget;
                                tracing::warn!(
                                    "ctx-overflow truncate (keep_tail={keep_tail}): dropped {dropped} \
                                     msg(s), prompt {} → {} tokens (fit_budget={fit_budget}, fits={fits})",
                                    prompt_tokens.len(),
                                    t.len(),
                                );
                                prompt_tokens = t;
                                if fits {
                                    break;
                                }
                            }
                            Err(e) => {
                                tracing::error!("re-tokenization after truncate failed: {e}");
                                break;
                            }
                        }
                    }
                    None => break, // nothing left to drop → caller fast-fails with 400
                }
            }
        }
    }

    // The image pad the model's vision path consumes: the checkpoint's
    // `image_pad_token_id` when its vision config declares one (GLM-5.3 uses
    // <|image|>, not Qwen's <|image_pad|>), else the tokenizer's Qwen marker.
    let image_pad_id = state
        .vision_config
        .as_ref()
        .map(|vision| vision.image_pad_token_id)
        .filter(|&token| token != 0)
        .or_else(|| state.tokenizer.image_pad_token_id());
    // Refuse templates that omit/duplicate images, before expansion can hide the mismatch.
    if has_images {
        let pad = image_pad_id.ok_or_else(|| {
            openai_error_response(
                StatusCode::BAD_REQUEST,
                "Tokenizer lacks an image marker token".into(),
            )
        })?;
        // GLM-5.3's published template is text-only: for each image it renders
        // a fixed "unable to process this image" reminder. The checkpoint is
        // multimodal, so that exact tokenized span is replaced by the native
        // <|begin_of_image|><|image|><|end_of_image|> triplet. A template that
        // already emits pads is left alone.
        if prompt_tokens.iter().all(|&id| id != pad) {
            prompt_tokens =
                materialize_glm_image_placeholders(&state.tokenizer, prompt_tokens, image_count, pad)
                    .map_err(|e| {
                        openai_error_response(
                            StatusCode::BAD_REQUEST,
                            format!("Image template error: {e:#}"),
                        )
                    })?;
        }
        if prompt_tokens.iter().filter(|&&id| id == pad).count() != image_count {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                "Chat template did not preserve exactly one marker per image".into(),
            ));
        }
    }
    // Expand image pads when needed.
    let prompt_tokens = match image_pad_id {
        Some(pad) if image_pad_counts.iter().any(|&c| c > 1) => {
            expand_image_pads_with_id(prompt_tokens, pad, image_pad_counts)
        }
        _ => prompt_tokens,
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

/// GLM-5.3 image placeholders (ported from perf/glm-5.3-flash-prefill-20260920).
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
