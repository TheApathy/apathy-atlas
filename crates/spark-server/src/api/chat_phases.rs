// SPDX-License-Identifier: AGPL-3.0-only

//! Helper functions extracted from `chat::chat_completions_inner`. These
//! are the F1..F60 cross-turn agentic guards — each one inspects and
//! mutates `req.messages` (and occasionally returns a flag the caller
//! uses for log-line correlation). Keeping them in a sibling file makes
//! `chat.rs` fit the 500-LoC cap and gives each guard a clear top-level
//! seam for future testing.

use axum::http::StatusCode;
use axum::response::Response;

use crate::openai::ChatCompletionRequest;

use super::compact::{openai_error_response, openai_error_response_with_param};
use super::failures::{
    F23ProgressMetrics, append_f7_reminder_to_last_user, build_f7_stall_reminder,
    collect_f7_stall_buckets, f23_build_reminder, f23_score_progress,
    f29_extract_environment_facts, f29_inject_environment_facts, f31_inject_hard_refusal,
    f32_reposition_failed_tool_result, f39_build_circuit_breaker_banner, f39_detect_recent_retries,
    f49_build_banner, f49_detect_duplicate_writes, f50_append_original_error,
    prepend_reminder_to_system, recent_message_is_tool_error,
};
use super::sanitizer::{F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD};

/// Validate the OpenAI input contract: messages length, max_tokens > 0,
/// temperature/top_p ranges, tool_choice mode/required compatibility.
/// Returns `Err(Response)` for fail-fast 400 paths so the caller can
/// `?` directly into a Response.
#[allow(clippy::result_large_err)]
pub(super) fn validate_input(req: &ChatCompletionRequest) -> Result<(), Response> {
    if req.messages.is_empty() {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "messages must contain at least one message".into(),
            Some("messages"),
            None,
        ));
    }
    if req.messages.len() > 2048 {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "messages array exceeds maximum length (2048)".into(),
            Some("messages"),
            None,
        ));
    }
    if req.stop.len() > 4 {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "stop must contain at most 4 sequences".into(),
            Some("stop"),
            None,
        ));
    }
    if req.stop.iter().any(String::is_empty) {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "stop sequences must not be empty".into(),
            Some("stop"),
            None,
        ));
    }
    if let Some(t) = req.temperature
        && (!(0.0..=2.0).contains(&t))
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "temperature must be between 0 and 2".into(),
            Some("temperature"),
            None,
        ));
    }
    if let Some(p) = req.top_p
        && !(p > 0.0 && p <= 1.0)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "top_p must be between 0 (exclusive) and 1".into(),
            Some("top_p"),
            None,
        ));
    }
    if let Some(top_n_sigma) = req.top_n_sigma
        && !(top_n_sigma.is_finite() && top_n_sigma >= 0.0)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "top_n_sigma must be finite and non-negative".into(),
            Some("top_n_sigma"),
            None,
        ));
    }
    if let Some(min_p) = req.min_p
        && !(0.0..=1.0).contains(&min_p)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "min_p must be between 0 and 1".into(),
            Some("min_p"),
            None,
        ));
    }
    if let Some(repetition_penalty) = req.repetition_penalty
        && !(repetition_penalty.is_finite() && repetition_penalty > 0.0)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "repetition_penalty must be finite and greater than 0".into(),
            Some("repetition_penalty"),
            None,
        ));
    }
    if req.max_tokens == 0 {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "max_tokens must be at least 1".into(),
            Some("max_tokens"),
            None,
        ));
    }
    if let Some(logit_bias) = &req.logit_bias {
        for (token_id, bias) in logit_bias {
            if token_id.parse::<u32>().is_err() {
                return Err(openai_error_response_with_param(
                    StatusCode::BAD_REQUEST,
                    format!("logit_bias key '{token_id}' must be a token ID"),
                    Some("logit_bias"),
                    None,
                ));
            }
            if !(-100.0..=100.0).contains(bias) {
                return Err(openai_error_response_with_param(
                    StatusCode::BAD_REQUEST,
                    "logit_bias values must be between -100 and 100".into(),
                    Some("logit_bias"),
                    None,
                ));
            }
        }
    }
    if let Some(crate::tool_parser::ToolChoice::Mode(ref s)) = req.tool_choice {
        if !["auto", "none", "required"].contains(&s.as_str()) {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "Invalid tool_choice value: '{s}'. Must be 'auto', 'none', 'required', or a function object."
                ),
            ));
        }
        if s == "required" && req.tools.as_ref().is_none_or(|t| t.is_empty()) {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                "tool_choice is 'required' but no tools were provided".into(),
            ));
        }
    }
    if let Some(crate::tool_parser::ToolChoice::Specific { ref function }) = req.tool_choice {
        let Some(tools) = req.tools.as_ref().filter(|tools| !tools.is_empty()) else {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                "specific tool_choice was provided but no tools were provided".into(),
            ));
        };
        if !tools.iter().any(|tool| tool.function.name == function.name) {
            return Err(openai_error_response_with_param(
                StatusCode::BAD_REQUEST,
                format!("tool_choice selected unknown function '{}'.", function.name),
                Some("tool_choice"),
                Some("tool_not_found"),
            ));
        }
    }
    Ok(())
}

/// Apply the cross-turn F-feature guards (F7, F23, F30, F31, F32, F39,
/// F49, F50, F35, F52, F29) to `req.messages`. Each guard inspects the
/// conversation history and mutates messages by injecting reminders /
/// circuit-breaker banners / synthesised tool_results. Returns the
/// computed `F23ProgressMetrics` so the caller can pass it through to
/// downstream phases that also key on it.
pub(super) fn apply_failure_guards(req: &mut ChatCompletionRequest) -> F23ProgressMetrics {
    // F7: cross-turn tool-arg-path stall guard.
    let stall_buckets = collect_f7_stall_buckets(&req.messages);
    if let Some(reminder) = build_f7_stall_reminder(&stall_buckets) {
        tracing::warn!(
            buckets_at_warn = stall_buckets
                .iter()
                .filter(|(_, c)| **c >= F7_STALL_WARN_THRESHOLD)
                .count(),
            buckets_at_refuse = stall_buckets
                .iter()
                .filter(|(_, c)| **c >= F7_STALL_REFUSE_THRESHOLD)
                .count(),
            "F7 stall guard: injecting reminder into last user/tool message"
        );
        append_f7_reminder_to_last_user(&mut req.messages, &reminder);
    }

    // F23: per-conversation progress tracker.
    let f23_metrics = f23_score_progress(&req.messages);
    if let Some(reminder) = f23_build_reminder(f23_metrics) {
        tracing::warn!(
            attempts = f23_metrics.attempts,
            score = f23_metrics.score,
            "F23 progress tracker: prepending reminder to system message (F30)"
        );
        prepend_reminder_to_system(&mut req.messages, &reminder);
    }

    // F31: synthesise [atlas-stall-guard] tool_result at refuse threshold.
    if f31_inject_hard_refusal(&mut req.messages, f23_metrics) {
        tracing::warn!(
            attempts = f23_metrics.attempts,
            score = f23_metrics.score,
            "F31: injected synthesised [atlas-stall-guard] tool_result"
        );
    }

    // F32: duplicate failed tool_result at conversation tail.
    if f32_reposition_failed_tool_result(&mut req.messages) {
        tracing::info!("F32: duplicated most-recent failed tool_result at conversation tail");
    }

    // F39: cross-turn permanent-failure circuit breaker.
    let f39_matches = f39_detect_recent_retries(&req.messages);
    if !f39_matches.is_empty() {
        tracing::warn!(
            n_matches = f39_matches.len(),
            sample = ?f39_matches.iter().take(3).map(|m| (&m.tool, &m.primary_arg, m.class, m.prior_failure_count)).collect::<Vec<_>>(),
            "F39: circuit-breaker — model is retrying calls with permanent prior failures"
        );
        let banner = f39_build_circuit_breaker_banner(&f39_matches);
        prepend_reminder_to_system(&mut req.messages, &banner);
    }

    // F49: duplicate-write fast detector. F50: re-surface original error.
    let f49_hits = f49_detect_duplicate_writes(&req.messages);
    if !f49_hits.is_empty() {
        tracing::warn!(
            n_hits = f49_hits.len(),
            sample = ?f49_hits.iter().take(3).map(|h| (&h.file_path, h.prior_count)).collect::<Vec<_>>(),
            "F49: duplicate-write detected — model is rewriting same file with identical content"
        );
        let banner = f49_build_banner(&f49_hits);
        prepend_reminder_to_system(&mut req.messages, &banner);
        if f50_append_original_error(&mut req.messages) {
            tracing::info!("F50: appended original [tool error] at conversation tail");
        }
    }

    // F35 / F52: turn-conditional failure-recovery clause with concrete fallbacks.
    if recent_message_is_tool_error(&req.messages) {
        let f35_clause = "<failure_recovery>\nThe previous tool call failed with an error. \
             That failure is a deterministic fact about this environment — \
             retrying with cosmetic variations (different mkdir prefix, \
             different cd path, different flag order) cannot change the \
             outcome.\n\n\
             Choose ONE of these concrete next steps:\n\
             (a) If the failed tool was Bash and the binary is missing, use the Write tool to create the files MANUALLY (write the Cargo.toml, source files, etc. directly via Write — do NOT call the missing binary again).\n\
             (b) If the failure is a path/permissions issue, try Bash with a substantively different command (different directory, different flags, different invocation).\n\
             (c) If neither (a) nor (b) is feasible, reply to the user in PLAIN TEXT (no tool call) explaining specifically what is blocking and what dependency or permission they need to provide.\n\
             Do NOT emit a generic \"please clarify your request\" question — the user already gave the request. Pick (a), (b), or (c) and execute it.\n\
             </failure_recovery>";
        prepend_reminder_to_system(&mut req.messages, f35_clause);
        tracing::debug!(
            "F35: prepended failure_recovery clause (most recent message is tool error)"
        );
    }

    // F29: scan tool-result history for repeated `command not found` failures
    // and inject an `<environment_facts>` block into the system message.
    let env_facts = f29_extract_environment_facts(&req.messages);
    if !env_facts.is_empty() {
        tracing::info!(
            facts_count = env_facts.len(),
            "F29: injecting environment_facts block into system message"
        );
        f29_inject_environment_facts(&mut req.messages, &env_facts);
    }

    f23_metrics
}

#[cfg(test)]
mod tool_choice_validation_tests {
    use super::validate_input;
    use axum::http::StatusCode;

    fn request(value: serde_json::Value) -> crate::openai::ChatCompletionRequest {
        serde_json::from_value(value).expect("valid request fixture")
    }

    #[test]
    fn specific_choice_requires_a_matching_declared_tool() {
        let absent = request(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "weather"}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        }));
        assert_eq!(
            validate_input(&absent).unwrap_err().status(),
            StatusCode::BAD_REQUEST
        );

        let unknown = request(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        }));
        assert_eq!(
            validate_input(&unknown).unwrap_err().status(),
            StatusCode::BAD_REQUEST
        );

        let matching = request(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        }));
        assert!(validate_input(&matching).is_ok());
    }

    #[test]
    fn numeric_sampling_contract_rejects_nonfinite_and_out_of_range_values() {
        let mut value = request(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}]
        }));

        for invalid in [-1.0, 2.1, f32::NAN, f32::INFINITY] {
            value.temperature = Some(invalid);
            assert_eq!(
                validate_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }

        value.temperature = None;
        for invalid in [0.0, -0.1, 1.1, f32::NAN, f32::INFINITY] {
            value.top_p = Some(invalid);
            assert_eq!(
                validate_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }
    }

    #[test]
    fn logit_bias_rejects_invalid_keys_and_values() {
        let mut value = request(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}]
        }));
        value.logit_bias = Some(std::collections::HashMap::from([
            ("0".to_string(), -100.0),
            (u32::MAX.to_string(), 100.0),
        ]));
        assert!(validate_input(&value).is_ok());

        for key in ["not-token", "-1", "4294967296"] {
            value.logit_bias = Some(std::collections::HashMap::from([(key.to_string(), 0.0)]));
            assert_eq!(
                validate_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }

        for bias in [-100.1, 100.1, f32::NAN, f32::INFINITY] {
            value.logit_bias = Some(std::collections::HashMap::from([("0".to_string(), bias)]));
            assert_eq!(
                validate_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }
    }

    #[test]
    fn extended_sampling_contract_is_fail_closed() {
        let mut value = request(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}]
        }));
        for invalid in [-0.1, f32::NAN, f32::INFINITY] {
            value.top_n_sigma = Some(invalid);
            assert_eq!(
                validate_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }

        value.top_n_sigma = None;
        for invalid in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
            value.min_p = Some(invalid);
            assert_eq!(
                validate_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }

        value.min_p = None;
        for invalid in [0.0, -0.1, f32::NAN, f32::INFINITY] {
            value.repetition_penalty = Some(invalid);
            assert_eq!(
                validate_input(&value).unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }

        value.repetition_penalty = Some(f32::MIN_POSITIVE);
        assert!(validate_input(&value).is_ok());
    }

    #[test]
    fn stop_contract_rejects_empty_or_excess_entries() {
        let mut value = request(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}]
        }));
        value.stop = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        assert!(validate_input(&value).is_ok());

        value.stop.push("e".into());
        assert_eq!(
            validate_input(&value).unwrap_err().status(),
            StatusCode::BAD_REQUEST
        );

        value.stop = vec![String::new()];
        assert_eq!(
            validate_input(&value).unwrap_err().status(),
            StatusCode::BAD_REQUEST
        );
    }
}
