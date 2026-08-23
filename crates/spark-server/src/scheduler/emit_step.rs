// SPDX-License-Identifier: AGPL-3.0-only

//! emit_token + compile_grammar_state + StartPrefillResult enum.

use super::*;

#[path = "rethink_budget.rs"]
mod rethink_budget;
pub(super) use rethink_budget::resolve_rethink_budget;

/// Whether the ordinary EOS path must defer termination for this request.
///
/// This deliberately contains no blanket post-thinking minimum.  Required
/// tool calls already have two authoritative protections (the grammar matcher
/// and the legacy `require_tool_call` fallback), while an ordinary answer must
/// be allowed to stop on its first EOS after `</think>`.  Ignoring that EOS can
/// make the model continue into the next ChatML role boundary.
pub(super) fn eos_is_suppressed(a: &ActiveSeq, output_len: usize) -> bool {
    a.grammar_state
        .as_ref()
        .is_some_and(|gs| !gs.is_terminated())
        || a.require_tool_call
        || output_len < a.min_tokens
        || a.inside_thinking
}

/// ChatML role boundaries end the assistant turn even when ordinary EOS is
/// constrained. The only exception is a boundary sampled inside `<think>`,
/// where special end tokens remain suppressed until `</think>`.
pub(super) fn is_chatml_role_boundary(a: &ActiveSeq, tok: u32, im_start: Option<u32>) -> bool {
    im_start == Some(tok) && !a.inside_thinking
}

/// Commit a ChatML role boundary before any ordinary token policy runs.
///
/// Both serial decode and speculative emission use this edge.  In particular,
/// grammar / required-tool / minimum-token EOS suppression must never discard
/// `<|im_start|>` and allow the following role text onto the assistant stream.
pub(super) fn commit_chatml_role_boundary(
    a: &mut ActiveSeq,
    tok: u32,
    im_start: Option<u32>,
) -> bool {
    if !is_chatml_role_boundary(a, tok, im_start) {
        return false;
    }
    // Keep the stop token in output_tokens so lifecycle reports `stop`.  The
    // API text path strips registered stop tokens, so it is never client text.
    a.output_tokens.push(tok);
    a.finished = true;
    tracing::debug!(
        id = im_start.unwrap_or_default(),
        "<|im_start|> hard-stop fired; ending turn before grammar/suppress_eos"
    );
    true
}

/// Advance the sampler's tool-body phase after committing one content token.
/// The content watchdog must observe the phase on entry (so a closing tag still
/// belongs to the body), while the next token's sampler must observe the new
/// phase. Both serial and speculative emission call this helper at that edge.
pub(super) fn update_tool_body_phase(a: &mut ActiveSeq, tok: u32) {
    if a.inside_thinking {
        return;
    }
    if a.tool_call_start_token == Some(tok) {
        a.inside_tool_body = true;
        a.prose_tokens_since_last_tool = 0;
    } else if a.tool_call_end_token == Some(tok) {
        a.inside_tool_body = false;
    }
}

/// Advance a content grammar exactly once for a committed non-thinking token.
/// Both serial decode and speculative emission use this edge; special token
/// handlers must not consume the same token again.
pub(super) fn advance_content_grammar(a: &mut ActiveSeq, tok: u32) {
    if !a.inside_thinking
        && let Some(ref mut gs) = a.grammar_state
    {
        gs.accept_token(tok);
    }
}

/// Commit a non-thinking tool-call close with one shared serial/spec contract.
///
/// Callers own the per-token content counters and must advance the grammar and
/// tool-body phase first.  This helper owns the one output push, the one stream
/// event, and all close-boundary termination decisions.
pub(super) fn commit_tool_call_close(a: &mut ActiveSeq, tok: u32, was_inside_thinking: bool) {
    debug_assert!(!a.inside_thinking);
    debug_assert_eq!(a.tool_call_end_token, Some(tok));

    a.output_tokens.push(tok);
    if a.think_ended && !was_inside_thinking {
        a.post_think_gate_steps = mtp_gate::advance_entry_counter(a.post_think_gate_steps);
    }

    if let ResponseSink::Streaming(ref tx) = a.sink {
        let event = if let Some(lp) = a.logprobs_data.last().cloned() {
            StreamEvent::TokenWithLogprobs(tok, lp)
        } else {
            StreamEvent::Token(tok)
        };
        match tx.try_send(event) {
            Ok(()) => {}
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                tracing::warn!(
                    "Streaming receiver dropped during tool_call_end, finishing sequence"
                );
                a.finished = true;
            }
            Err(tokio::sync::mpsc::error::TrySendError::Full(event)) => {
                if let Err(e) = tx.blocking_send(event) {
                    tracing::error!("Streaming send failed during tool_call_end backpressure: {e}");
                    a.finished = true;
                }
            }
        }
    }

    let grammar_terminal = a
        .grammar_state
        .as_ref()
        .is_some_and(|gs| gs.is_terminated());
    if a.grammar_state.is_none() || grammar_terminal {
        // Legacy mode is one-call-per-response. A stop-after-first grammar has
        // the same boundary once its close token makes the matcher terminal.
        a.finished = true;
    } else {
        // A nonterminal multi-call grammar may legitimately think again before
        // its next call; post-think masks must not remain latched across calls.
        a.think_ended = false;
    }

    if a.output_tokens.len() >= a.max_output_tokens || a.remaining == 0 {
        a.finished = true;
    }
}

/// Return a fuzzy tail-loop only when the current token stream is outside a
/// tool call. This is shared by serial decode and speculative emission so both
/// paths use identical start/end-token framing.
pub(super) fn fuzzy_repetition_outside_tool(a: &ActiveSeq) -> Option<(usize, usize, usize)> {
    if a.inside_thinking {
        return None;
    }
    let last_tc_start = a
        .tool_call_start_token
        .and_then(|t| a.output_tokens.iter().rposition(|&tok| tok == t));
    let last_tc_end = a
        .tool_call_end_token
        .and_then(|t| a.output_tokens.iter().rposition(|&tok| tok == t));
    let inside_tool_call = match (last_tc_start, last_tc_end) {
        (Some(start), Some(end)) => start > end,
        (Some(_), None) => true,
        _ => false,
    };
    if inside_tool_call {
        None
    } else {
        detect_fuzzy_repetition(&a.output_tokens)
    }
}

/// Stop speculative emission on a fuzzy tail-loop. The serial path can roll
/// back to a per-token SSM snapshot; a multi-token verify batch cannot safely
/// do that after commit, so its parity behavior is the conservative hard stop.
pub(super) fn apply_speculative_fuzzy_stop(a: &mut ActiveSeq) -> bool {
    let Some((pattern_len, mis_a, mis_b)) = fuzzy_repetition_outside_tool(a) else {
        return false;
    };
    tracing::warn!(
        pattern_len,
        mismatches = mis_a + mis_b,
        output_len = a.output_tokens.len(),
        "Fuzzy repetition detected during speculative emit; ending response early (batched rollback unavailable)"
    );
    a.finished = true;
    true
}

/// Apply the request deadline at the same post-token boundary used by serial
/// decode. Thinking and content-speculation paths share this implementation.
pub(super) fn check_request_timeout(a: &mut ActiveSeq) {
    if !a.finished
        && let Some(deadline) = a.timeout_at
        && Instant::now() >= deadline
    {
        tracing::warn!("Request timeout after {:?}", a.request_start.elapsed());
        a.finished = true;
    }
}

/// Emit a token for an active sequence (stream + bookkeeping).
///
/// Per OpenAI spec, stop/EOS tokens are NOT streamed to the client —
/// the returned text must not contain the stop sequence. The token is
/// still recorded in output_tokens for accurate token counting.
///
/// When `logprobs` is Some, the logprobs data is accumulated for blocking
/// responses and sent via `StreamEvent::TokenWithLogprobs` for streaming.
pub fn emit_token(a: &mut ActiveSeq, tok: u32, logprobs: Option<crate::api::TokenLogprobs>) {
    let output_len_before_emit = a.output_tokens.len();
    let was_inside_thinking = a.inside_thinking;
    // Cooperative cancellation from the streaming pipeline (PR #89). The
    // stream-side loop guards (Bug-2 name-run cap, F11 within-dedup, F44
    // perm-fail, loop-watchdog) flip this flag when they decide the
    // response should end. Treat it like an EOS: finalise now so
    // `handle_done` runs with the proper `tool_loop_capped` /
    // `finish_reason="length"` machinery, instead of letting the model
    // keep emitting tokens that just get suppressed.
    if let Some(ref f) = a.cancel_flag
        && f.load(std::sync::atomic::Ordering::Acquire)
    {
        a.finished = true;
        return;
    }

    // ── ATLAS_DUMP_HIDDEN: append token-emit record (catch-all spec path) ──
    // verify_k2/k3/k4/dflash all funnel through emit_token, so this hook
    // covers the speculative codepaths that bypass process_decode_logits.
    // Path lookup is cached via OnceLock in `dump_hidden_path` so the
    // common production case (var unset) is a single relaxed pointer load.
    if let Some(path) = super::helpers::dump_hidden_path() {
        const TOKEN_DUMP_MAGIC: u32 = 0xA71B5DEE;
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = f.write_all(&TOKEN_DUMP_MAGIC.to_le_bytes());
            let _ = f.write_all(&tok.to_le_bytes());
            let _ = f.write_all(&0u32.to_le_bytes());
            let _ = f.write_all(&0u32.to_le_bytes());
        }
    }

    // ChatML role-boundary HARD stop (`<|im_start|>`).
    //
    // Handled BEFORE grammar advance / EOS suppression: if the model
    // hallucinated a `<|im_start|>` mid-turn, we must end the turn regardless
    // of grammar / require_tool_call / min_tokens. The regular EOS path at
    // line ~3020 honors `suppress_eos`, which is true while a tool-call
    // grammar is active — so if we fell through to it, the tokenizer would
    // strip `<|im_start|>` (special-token) but the following role literal
    // (`user` / `assistant` — regular tokens) would stream to the client,
    // poisoning its context and causing the observed multi-turn drift /
    // "file was corrupted" hallucinations in opencode.
    if commit_chatml_role_boundary(a, tok, im_start_hard_stop()) {
        return;
    }

    // Spontaneous <think>: model generates <think> even when thinking was not
    // requested. Enter thinking mode so EOS is suppressed and thinking content
    // is stripped. This handles MTP bootstrap/verify paths.
    //
    // Spontaneous <think>: model generates <think> even when thinking was not
    // requested. Enter thinking mode so EOS is suppressed and thinking content
    // is stripped. This handles MTP bootstrap/verify paths.
    //
    // DDTree guard: bonus tokens bypass the sample-time logit mask that normally
    // blocks <think> when think_ended=true. When think_ended=true and <think>
    // arrives here, we must still enter thinking mode (the token is already
    // committed to the KV cache by the verify pass) but force an immediate exit
    // via force_end_thinking. The very next token becomes </think> (0 thinking
    // content tokens, 1 </think> overhead), and coding output continues cleanly.
    if !a.inside_thinking && a.think_start_token == Some(tok) {
        // Re-entering thinking re-arms the response-entry counter for the
        // next `</think>` boundary.
        a.post_think_gate_steps = 0;
        if !a.think_ended {
            a.inside_thinking = true;
            a.think_ended = false;
            a.think_skip_count = 0;
            a.thinking_budget = Some(resolve_rethink_budget(
                a.spontaneous_think_budget,
                a.think_watchdog_fires,
            ));
            tracing::debug!("Spontaneous <think> detected in emit_token, entering thinking mode");
        } else {
            // DDTree cliff-path <think> bonus with think_ended=true: force-exit.
            a.inside_thinking = true;
            a.think_skip_count = 0;
            a.thinking_budget = Some(0);
            a.force_end_thinking = true;
            tracing::debug!(
                "DDTree <think> bonus (think_ended=true): force-exit thinking (0 content tokens)"
            );
        }
        return; // don't emit <think> as content
    }

    // Silently skip </think> tokens outside thinking mode (same as process_decode_logits).
    if !a.inside_thinking && a.think_end_token == Some(tok) {
        a.think_skip_count += 1;
        if a.think_skip_count >= 50 {
            a.finished = true;
        }
        return;
    }
    // Match serial decode: a real post-thinking token breaks the aggregate
    // stray-</think> run. Without this reset, speculative serving stopped after
    // 50 non-consecutive close tags accumulated across otherwise valid output.
    if a.think_ended {
        a.think_skip_count = 0;
    }

    // Track <tool_call> token: once seen, legacy tool call requirement is satisfied.
    // Guard with !inside_thinking — tool calls inside thinking are spurious.
    if a.require_tool_call && a.tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.require_tool_call = false;
        a.tool_call_opened = true;
    }

    // Preserve the phase on entry for the content watchdog below.  A closing
    // </tool_call> token still belongs to the tool body; flipping the flag
    // first would incorrectly charge that token to the inter-tool prose
    // budget (the plain decoder updates this flag after content handling).
    let was_inside_tool_body = a.inside_tool_body;

    // Track CURRENT tool-body phase (P3.1, 2026-04-25). Set on the
    // open token, clear on the close. The flag drives sampler
    // scoping: when true, the main decode path zeroes
    // repetition/presence/frequency/DRY so legitimate JSON
    // micro-repetition (`":"`, `","`, key names) is not penalised.
    update_tool_body_phase(a, tok);

    // Advance grammar state with the emitted token — skip while the
    // sequence is inside `<think>`…`</think>` so the matcher only
    // sees the final-output token stream.
    advance_content_grammar(a, tok);

    // Accumulate logprobs data for blocking responses.
    if let Some(lp) = logprobs {
        a.logprobs_data.push(lp);
    }

    // Silent exit for zero-content thinking episodes (DDTree cliff or spontaneous
    // <think> with budget=0). Skip pushing </think> to output_tokens — the tag
    // would appear as a spurious delimiter and cause strip_thinking_tags to
    // truncate everything before it, treating the real output as "reasoning".
    if a.inside_thinking && a.think_end_token == Some(tok) && a.thinking_budget == Some(0) {
        a.inside_thinking = false;
        a.force_end_thinking = false;
        a.think_ended = true;
        a.think_just_ended = true;
        a.post_think_gate_steps = 0;
        tracing::info!("Thinking ended after 0 tokens (budget=Some(0)) [silent]");
        return;
    }

    // Speculative verify paths funnel through emit_token rather than
    // process_decode_logits, so they must advance the same content counters
    // and run the same degeneration detectors.  Historically they skipped
    // this entire policy layer: a plain decode could stop/rollback at 96
    // repeated tokens while DFlash/MTP emitted until max_tokens, making both
    // serving behaviour and benchmark token counts mechanism-dependent.
    //
    // A speculative verify may have committed several target/SSM positions
    // before reaching this function.  Calling rollback_to_boundary here would
    // therefore be unsafe: the per-token boundary can sit in the middle of
    // that already-committed batch.  Fail closed by ending the response.  A
    // later design can replace this with batched commit truncation plus exact
    // SSM snapshots; silent bypass is not an acceptable fallback.
    let mut speculative_watchdog_stop = false;
    if !a.inside_thinking {
        a.content_started = true;
        a.content_tokens = a.content_tokens.saturating_add(1);

        let catastrophic_loop = a.content_tokens >= CATASTROPHIC_LOOP_MIN_TOKENS as u32
            && a.content_tokens.is_multiple_of(CONTENT_LOOP_CHECK_STRIDE)
            && detect_catastrophic_content_loop(&a.output_tokens);
        let configured_loop = enable_loop_watchdog()
            && a.content_tokens >= CONTENT_LOOP_MIN_TOKENS
            && a.content_tokens.is_multiple_of(CONTENT_LOOP_CHECK_STRIDE)
            && (detect_content_token_loop(&a.output_tokens)
                || numeric_token_mask()
                    .as_deref()
                    .is_some_and(|m| detect_content_token_loop_normalized(&a.output_tokens, m)));
        if a.grammar_state.is_none()
            && !was_inside_tool_body
            && (catastrophic_loop || configured_loop)
        {
            tracing::warn!(
                content_tokens = a.content_tokens,
                output_len = a.output_tokens.len(),
                catastrophic = catastrophic_loop,
                "Content-loop watchdog fired during speculative emit; ending response early (batched rollback unavailable)"
            );
            speculative_watchdog_stop = true;
        }

        if !was_inside_tool_body && a.grammar_state.is_some() {
            a.prose_tokens_since_last_tool = a.prose_tokens_since_last_tool.saturating_add(1);
            let max_prose = watchdog_params().max_inter_tool_prose;
            if a.prose_tokens_since_last_tool > max_prose {
                tracing::warn!(
                    prose_tokens = a.prose_tokens_since_last_tool,
                    max = max_prose,
                    "Inter-tool prose budget exhausted during speculative emit; ending response (batched rollback unavailable)"
                );
                speculative_watchdog_stop = true;
            }
        }
    }

    // Tool-call close is a structural boundary, not ordinary content. The
    // serial decoder and every speculative verifier share the exact push /
    // stream / finish contract through this helper.
    if a.tool_call_end_token == Some(tok) && !a.inside_thinking {
        a.remaining -= 1;
        a.think_just_ended = false;
        commit_tool_call_close(a, tok, was_inside_thinking);
        if speculative_watchdog_stop {
            a.finished = true;
        }
        return;
    }

    a.output_tokens.push(tok);

    // Thinking tokens are "free" (don't decrement remaining).
    // Detect </think> transition. Track thinking token count for budget enforcement.
    if a.inside_thinking {
        if a.think_end_token == Some(tok) {
            a.inside_thinking = false;
            a.force_end_thinking = false;
            a.think_ended = true;
            a.post_think_gate_steps = 0;
            // One-shot for the next decode step: pin to
            // tool_call_start_token if require_tool_call (Change 3b).
            a.think_just_ended = true;
            tracing::info!(
                "Thinking ended after {} tokens (budget={:?})",
                a.thinking_tokens,
                a.thinking_budget,
            );
        } else {
            a.thinking_tokens += 1;
            if let Some(budget) = a.thinking_budget
                && a.thinking_tokens >= budget
                && !a.force_end_thinking
            {
                a.force_end_thinking = true;
                tracing::info!("Thinking budget exhausted ({budget} tokens), forcing </think>");
            }
        }
    } else {
        a.remaining -= 1;
        // Clear think_just_ended one-shot now that we've consumed the
        // token after </think>.
        a.think_just_ended = false;
    }

    // Required tools remain protected by grammar / legacy-required mode;
    // ordinary post-thinking answers are free to terminate immediately.
    let suppress_eos = eos_is_suppressed(a, output_len_before_emit);

    // Single-pass over eos_tokens: previously this Vec was linearly
    // scanned twice per emit (once for each branch of the suppress_eos
    // dispatch). On a 256-token response that's 256 extra scans of the
    // ~3-5 entry vec — small per call but mechanical to remove.
    let is_eos = a.eos_tokens.contains(&tok);
    if is_eos && !suppress_eos {
        a.finished = true;
        return;
    }
    if is_eos {
        // EOS suppressed: grammar not terminated, legacy tool call not yet seen,
        // min_tokens not reached, or thinking is still active.
        // The plain decode path discards this token rather than including it in
        // output history; undo the speculative path's earlier bookkeeping push.
        debug_assert_eq!(a.output_tokens.last(), Some(&tok));
        a.output_tokens.pop();
        return;
    }
    // Count only committed non-EOS content. Suppressed EOS rows are popped
    // above and therefore cannot age the entry pin accidentally.
    if a.think_ended && !was_inside_thinking && !a.inside_thinking {
        a.post_think_gate_steps = mtp_gate::advance_entry_counter(a.post_think_gate_steps);
    }
    // OPENCODE FIX: see process_decode_logits — same gate. Suppress streaming
    // of spontaneous-thinking content so it doesn't pollute opencode's history.
    let suppress_stream = a.inside_thinking && !a.enable_thinking;
    if let ResponseSink::Streaming(ref tx) = a.sink
        && !suppress_stream
    {
        let event = if let Some(lp) = a.logprobs_data.last().cloned() {
            StreamEvent::TokenWithLogprobs(tok, lp)
        } else {
            StreamEvent::Token(tok)
        };
        // Discriminate transient backpressure (channel full) from a real
        // consumer-drop (channel closed). The previous `try_send().is_err()`
        // collapsed the two and silently terminated the seq with
        // `finish_reason="length"` whenever the SSE consumer momentarily
        // stalled — surfaced as "request stops half-way" in Open WebUI.
        match tx.try_send(event) {
            Ok(()) => {}
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                tracing::debug!("Streaming receiver dropped, finishing seq");
                a.finished = true;
                return;
            }
            Err(tokio::sync::mpsc::error::TrySendError::Full(event)) => {
                if let Err(e) = tx.blocking_send(event) {
                    tracing::error!("Streaming send failed during backpressure: {e}");
                    a.finished = true;
                    return;
                }
            }
        }
    }
    // Grammar termination is itself a sequence boundary. Serial decode stops
    // immediately after the token that completes a stop-after-first grammar;
    // speculative emission must not continue walking already-verified rows.
    if a.grammar_state
        .as_ref()
        .is_some_and(|gs| gs.is_terminated())
    {
        a.finished = true;
        return;
    }
    // `remaining` deliberately excludes reasoning tokens, but the API's
    // max_tokens/max_completion_tokens envelope includes every generated
    // completion token.  Keep thinking free for the content budget while
    // enforcing the absolute response cap for both blocking and streaming
    // speculative paths.
    if a.output_tokens.len() >= a.max_output_tokens {
        tracing::info!(
            "emit_token: max_output_tokens={} reached, output_tokens={}, thinking_tokens={}",
            a.max_output_tokens,
            a.output_tokens.len(),
            a.thinking_tokens,
        );
        a.finished = true;
        return;
    }
    if a.remaining == 0 {
        tracing::info!(
            "emit_token: remaining=0, output_tokens={}, thinking_tokens={}",
            a.output_tokens.len(),
            a.thinking_tokens
        );
        a.finished = true;
    }
    if enable_loop_watchdog() && !a.finished {
        apply_speculative_fuzzy_stop(a);
    }
    if speculative_watchdog_stop {
        a.finished = true;
    }
    check_request_timeout(a);
}

// F72 (byte-level partial-trigger anchor) was removed in F73 / fix42.
// The sampler-side anchor hung the server in production; the broken
// envelope is now recovered at the streaming-sanitizer + parser
// layer. xgrammar's non-anchored TagDispatch limitation is pinned
// for documentation by
// `grammar.rs::tests::test_minimax_xml_grammar_masks_trigger_breaking_multibyte_token`.

/// Compile a grammar state from a grammar specification + engine.
///
/// Returns `Some(GrammarState)` if compilation succeeds, `None` otherwise
/// (logging a warning on failure so the request falls back to legacy tool_call
/// suppression). Called once per request during prefill.
pub fn compile_grammar_state(
    engine: &mut Option<GrammarEngine>,
    grammar_spec: &Option<GrammarSpec>,
) -> Option<GrammarState> {
    let spec = grammar_spec.as_ref()?;
    let engine = engine.as_mut()?;

    // F69 (2026-04-29): symmetric dispatch via the trait. The parser
    // is the single source of truth for both runtime parsing and
    // grammar compilation; no string match keyed on `parser_name`.
    // Mistral's default trait impl returns `None`, which we treat as
    // "no constraint, fall through to unconstrained decoding."
    let compiled = match spec {
        GrammarSpec::ToolCall {
            tools,
            parser,
            use_triggers,
        } => match parser.compile_tool_grammar(engine, tools, *use_triggers) {
            Some(result) => result,
            None => {
                tracing::debug!(
                    "Grammar: parser '{}' opted out of constrained decoding for this request",
                    parser.name(),
                );
                return None;
            }
        },
        GrammarSpec::JsonObject => engine.compile_json_grammar(),
        GrammarSpec::JsonSchema { schema } => engine.compile_json_schema(schema),
    };

    let label = match spec {
        GrammarSpec::ToolCall { parser, tools, .. } => {
            format!("parser={}, tools={}", parser.name(), tools.len())
        }
        GrammarSpec::JsonObject => "response_format=json_object".to_string(),
        GrammarSpec::JsonSchema { .. } => "response_format=json_schema".to_string(),
    };

    match compiled {
        Ok(grammar) => {
            let vocab_size = engine.vocab_size();
            match GrammarState::new(&grammar, vocab_size) {
                Ok(state) => {
                    tracing::info!("Grammar constrained decoding active: {label}");
                    Some(state)
                }
                Err(e) => {
                    tracing::warn!("Grammar state creation failed: {e}");
                    None
                }
            }
        }
        Err(e) => {
            tracing::warn!("Grammar compilation failed: {e}");
            None
        }
    }
}

/// Result of starting a chunked prefill.
pub enum StartPrefillResult {
    /// Prompt fit in one chunk → ready for decode.
    Active(ActiveSeq),
    /// Prompt needs more chunks → add to prefilling[].
    InProgress(PrefillInProgress),
    /// Completed during first chunk (EOS on first token).
    Finished,
}
