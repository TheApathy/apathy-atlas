// SPDX-License-Identifier: AGPL-3.0-only

//! process_decode_logits: post-decode logits processing.

use super::*;

/// Sample and process decode logits for all active sequences.
///
/// Factored out of `step_decode_only` so that `mixed_forward` can reuse
/// the same sampling + token-processing logic without duplication (SSOT).
/// `logits` must point to `[n, vocab_size]` BF16 on device where n = active.len().
pub fn process_decode_logits(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    logits: DevicePtr,
    t0: std::time::Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    reflection_suppress_ids: &[u32],
    adaptive_sampling: bool,
) {
    let n = active.len();

    // Grammar bitmask is CPU-side, so any sequence with active grammar forces
    // the host-side sampling path for its logits slice.
    let any_grammar = active.iter().any(|a| a.grammar_state.is_some());
    let any_logprobs = active.iter().any(|a| a.top_logprobs.is_some());
    // FP32 lm_head models (Gemma-4 dense) MUST use the host-side path —
    // `argmax_batch` assumes BF16 layout and would interpret 4-byte FP32
    // values as 2-byte BF16 pairs, returning garbage tokens.
    let model_logits_fp32 = model.decode_logits_fp32();
    let needs_host_logits = active
        .iter()
        .any(|a| a.inside_thinking || a.think_ended || a.grammar_state.is_some())
        || any_logprobs
        || model_logits_fp32;

    let new_tokens: Vec<(u32, Option<crate::api::TokenLogprobs>)> = if active
        .iter()
        .all(fast_path_seq_eligible)
        && !any_grammar
        && !needs_host_logits
    {
        // Fast path: all greedy, no grammar, no thinking — GPU argmax for the full batch.
        match model.argmax_batch(logits, n, 0) {
            Ok(t) => t.into_iter().map(|tok| (tok, None)).collect(),
            Err(e) => {
                tracing::error!("argmax_batch error: {e:#}");
                for mut a in active.drain(..) {
                    send_error(model, &mut a, &format!("{e:#}"));
                }
                return;
            }
        }
    } else {
        // Host-side path: copy all batch logits to host, sample per-sequence.
        // Required when any sequence has temperature > 0 or grammar constraints.
        let vocab_size = model.vocab_size();
        // FP32 lm_head dispatch (Gemma-4 dense + ATLAS_GEMMA4_FP32_LMHEAD=1).
        // When the model writes FP32 logits to its decode-logits buffer, we
        // copy 4 bytes/element and skip the BF16→FP32 expansion. Earlier
        // bisection at model.rs:1192-1201 incorrectly concluded FP32 lm_head
        // had no effect on Gemma-4 because this dispatch was never wired —
        // the scheduler always read the (stale) BF16 logits buffer.
        // FP32 lm_head dispatch (Gemma-4 dense). When `use_fp32_logits` is
        // on, the per-token decode lm_head writes 4 bytes/element. The
        // passed `logits` pointer is whatever the most-recent forward
        // returned — that's already the correct buffer (prefill or decode).
        // We just need to read it with the matching width.
        let logits_fp32 = model.decode_logits_fp32();
        let elem_bytes = if logits_fp32 { 4 } else { 2 };
        let mut buf = vec![0u8; n * vocab_size * elem_bytes];
        if let Err(e) = model.copy_logits_to_host(logits, &mut buf) {
            tracing::error!("copy_logits_to_host error: {e:#}");
            for mut a in active.drain(..) {
                send_error(model, &mut a, &format!("{e:#}"));
            }
            return;
        }
        active
            .iter_mut()
            .enumerate()
            .map(|(i, a)| {
                let sampled = process_seq_logits(
                    a,
                    &buf,
                    i,
                    vocab_size,
                    elem_bytes,
                    logits_fp32,
                    think_end_token,
                    think_start_token,
                    tool_call_start_token,
                    tool_call_end_token,
                    reflection_suppress_ids,
                    adaptive_sampling,
                );
                if n == 1
                    && std::env::var("ATLAS_SPEC_BOOTSTRAP_TRACE").ok().as_deref() == Some("1")
                {
                    let row = &buf[i * vocab_size * elem_bytes..(i + 1) * vocab_size * elem_bytes];
                    let read = |j: usize| {
                        if logits_fp32 {
                            let off = j * 4;
                            f32::from_le_bytes([row[off], row[off + 1], row[off + 2], row[off + 3]])
                        } else {
                            bf16_to_f32(row[j * 2], row[j * 2 + 1])
                        }
                    };
                    let raw_argmax = (0..vocab_size)
                        .max_by(|&x, &y| read(x).total_cmp(&read(y)))
                        .unwrap_or(0);
                    tracing::info!(
                        raw_argmax,
                        sampled = sampled.0,
                        think_ended = a.think_ended,
                        "SERIAL_POLICY_DECODE"
                    );
                }
                sampled
            })
            .collect()
    };

    // ── ATLAS_DUMP_HIDDEN: append token records (catch-all path) ──
    // Pairs decode-step hidden states (dumped per-layer) with the token that
    // was sampled at this step. Fires for both fast path (argmax_batch) and
    // host-side path (process_seq_logits). Matches the model-side hook that
    // dumps records only when ATLAS_DUMP_HIDDEN env var is set.
    // Cached via `dump_hidden_path` OnceLock so the var-unset case is a
    // single relaxed load per scheduler step.
    if let Some(path) = super::helpers::dump_hidden_path() {
        const TOKEN_DUMP_MAGIC: u32 = 0xA71B5DEE;
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            for (tok, _lp) in &new_tokens {
                let _ = f.write_all(&TOKEN_DUMP_MAGIC.to_le_bytes());
                let _ = f.write_all(&tok.to_le_bytes());
                let _ = f.write_all(&0u32.to_le_bytes());
                let _ = f.write_all(&0u32.to_le_bytes());
            }
        }
    }

    let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if tracing::enabled!(tracing::Level::DEBUG) {
        let token_ids: Vec<u32> = new_tokens.iter().map(|(t, _)| *t).collect();
        tracing::debug!(
            "DECODE: n={n} step={step_ms:.1}ms ({:.1} tok/s) tokens={:?}",
            1000.0 * n as f64 / step_ms,
            token_ids,
        );
    }

    let now = Instant::now();
    for (i, (tok, mut logprobs)) in new_tokens.into_iter().enumerate() {
        let a = &mut active[i];
        let was_inside_thinking = a.inside_thinking;
        let previous_last_token = a.last_token;
        let previous_last_token_time = a.last_token_time;
        a.last_token = tok;
        a.last_token_time = now;

        // A ChatML role boundary ends the assistant turn even when an active
        // grammar, required tool call, or min_tokens would suppress ordinary
        // EOS. Keep this before every phase/grammar side effect, exactly like
        // speculative `emit_token`.
        if commit_chatml_role_boundary(a, tok, im_start_hard_stop()) {
            continue;
        }

        // Spontaneous <think>: model generates <think> even when thinking
        // was not requested. Enter thinking mode so EOS is suppressed and
        // thinking content is stripped. Matches vLLM's behavior of always
        // parsing <think>...</think> regardless of enable_thinking setting.
        //
        // F9+F10 (2026-04-26): the sample-time logit mask at line ~1716
        // hard-blocks `<think>` when `think_ended=true`, so this branch
        // should not fire after a watchdog has force-closed thinking.
        // Defence-in-depth: if the model somehow still emits <think>
        // (e.g. the start token differs from the masked one in edge
        // cases), decay the budget by `>> watchdog_fires.min(4)` so
        // each successive re-entry has a tighter window. After 4+
        // fires, the budget is 1/16 of normal — the watchdog kills
        // re-entry within a handful of tokens.
        // DDTree guard: bonus tokens bypass the sample-time logit mask that
        // blocks <think> when think_ended=true. If think_ended=true and <think>
        // arrives, the token is already in the KV cache (committed by verify),
        // so we must enter thinking mode but force an immediate exit: next token
        // is forced to </think> (0 thinking content tokens, 1 </think> overhead).
        if !a.inside_thinking && think_start_token == Some(tok) {
            // Re-entering thinking re-arms the response-entry counter for the
            // next `</think>` boundary.
            a.post_think_gate_steps = 0;
            if !a.think_ended {
                let decayed =
                    resolve_rethink_budget(a.spontaneous_think_budget, a.think_watchdog_fires);
                a.inside_thinking = true;
                a.think_ended = false; // reset so </think> detection path works
                a.think_skip_count = 0;
                a.thinking_budget = Some(decayed);
                if a.think_watchdog_fires > 0 {
                    tracing::debug!(
                        fires = a.think_watchdog_fires,
                        decayed_budget = decayed,
                        "Spontaneous <think> re-entry after watchdog; decayed budget"
                    );
                } else {
                    tracing::debug!("Spontaneous <think> detected, entering thinking mode");
                }
            } else {
                // DDTree cliff-path <think> (think_ended=true): force-exit.
                a.inside_thinking = true;
                a.think_skip_count = 0;
                a.thinking_budget = Some(0);
                a.force_end_thinking = true;
                tracing::debug!(
                    "DDTree <think> (think_ended=true): force-exit thinking (0 content tokens)"
                );
            }
            continue; // don't emit <think> as content
        }

        // Silently skip </think> tokens outside thinking mode.
        // At long context (37k+), models degenerate into repeating </think>.
        // Skip up to 50 occurrences, then force-stop. This gives cached
        // prompts a chance to produce content while limiting degenerate loops.
        if !a.inside_thinking && think_end_token == Some(tok) {
            a.think_skip_count += 1;
            if a.think_skip_count >= 50 {
                a.finished = true;
            }
            continue;
        }
        // Reset skip counter when a real content token is generated.
        if a.think_ended {
            a.think_skip_count = 0;
        }

        // Thinking tokens don't count toward remaining (thinking is "free").
        if a.inside_thinking {
            if think_end_token == Some(tok) {
                a.inside_thinking = false;
                a.force_end_thinking = false;
                a.consecutive_confident = 0;
                a.in_code_fence = false;
                a.think_ended = true;
                a.post_think_gate_steps = 0;
                // One-shot: pin the next sampled token to the
                // tool-call-start token if the request requires a
                // tool call (Change 3b). Cleared in the `else`
                // branch below on the next emit.
                a.think_just_ended = true;
            } else {
                a.thinking_tokens += 1;
                // Track ``` code-fence parity within the thinking block:
                // each fence token flips in/out of a fenced code span.
                // The F2 confidence early-stop (process_seq_logits) is
                // suppressed while `in_code_fence` — code is near-
                // deterministic (high top-1 prob) but that is NOT a
                // "done reasoning" signal; braking here truncates the
                // model mid-statement. THINK_LOOP (below) deliberately
                // stays active even inside fences: it catches
                // *repeating* fence-narration, not one coherent block.
                a.in_code_fence = toggle_code_fence(a.in_code_fence, tok, code_fence_token);
                // Set force_end_thinking when budget exhausted (picked up next iteration)
                if let Some(budget) = a.thinking_budget
                    && a.thinking_tokens >= budget
                    && !a.force_end_thinking
                {
                    a.force_end_thinking = true;
                    tracing::info!("Thinking budget exhausted ({budget} tokens), forcing </think>");
                }
                // Token-level fence-loop detection. Catches the Qwen3.5-35B
                // phrase attractor (`Running:\`\`\`bash cmd\`\`\`Executing:…`
                // cycling) within ~24-60 tokens of the loop starting,
                // instead of waiting for the 256-token thinking budget.
                if enable_thinking_loop_watchdog()
                    && !a.force_end_thinking
                    && a.thinking_tokens >= THINK_LOOP_MIN_TOKENS
                    && a.thinking_tokens.is_multiple_of(THINK_LOOP_CHECK_STRIDE)
                    && detect_thinking_token_loop(&a.output_tokens)
                {
                    a.force_end_thinking = true;
                    a.think_watchdog_fires = a.think_watchdog_fires.saturating_add(1);
                    tracing::warn!(
                        thinking_tokens = a.thinking_tokens,
                        watchdog_fires = a.think_watchdog_fires,
                        "Thinking-loop watchdog fired (period-{}…{} repeat in tail); forcing </think> early",
                        THINK_LOOP_PERIOD_MIN,
                        THINK_LOOP_PERIOD_MAX,
                    );
                }
            }
        } else {
            // Content-phase token: budget bookkeeping + the content-loop
            // and inter-tool-prose watchdogs. Extracted to
            // `decode_logits_content.rs` to keep this file ≤500 LoC.
            // `model` is threaded through so a watchdog rollback can
            // restore SSM recurrent state on hybrid models (Phase-C).
            match handle_content_token(a, tok, model, previous_last_token, previous_last_token_time)
            {
                ContentTokenDisposition::CommitSample => {
                    advance_content_grammar(a, tok);
                }
                ContentTokenDisposition::DiscardSampleAndStop { dropped } => {
                    tracing::debug!(dropped, "Discarded sample after rollback hard-stop");
                    continue;
                }
            }
        }

        // Track <tool_call> token: once seen, legacy tool call requirement is satisfied.
        // Guard with !inside_thinking — a <tool_call> inside thinking is spurious
        // and must not clear require_tool_call (which would allow premature EOS).
        if a.require_tool_call && tool_call_start_token == Some(tok) && !a.inside_thinking {
            a.require_tool_call = false;
            a.tool_call_opened = true;
        }
        // Update the tool-body phase AFTER content handling: the opening token
        // is the boundary into the body, while the closing token still belongs
        // to the body for watchdog accounting. `process_seq_logits` observes
        // this flag on the next position and disables prose penalties inside
        // structured JSON exactly like speculative emission.
        update_tool_body_phase(a, tok);
        // Safety: if require_tool_call is still set after 512 tokens, the model
        // isn't generating a tool call (grammar may have failed to compile).
        // Clear the flag so EOS is no longer suppressed — prevents infinite gen.
        if a.require_tool_call && a.output_tokens.len() > 512 {
            tracing::warn!(
                "require_tool_call safety: no <tool_call> after 512 tokens, clearing EOS suppression"
            );
            a.require_tool_call = false;
        }

        // </tool_call> stop: in legacy mode (no grammar), stop after first tool call.
        // When grammar is active, allow the model to generate multiple tool calls —
        // the grammar controls when EOS is valid.
        if tool_call_end_token == Some(tok) && !a.inside_thinking {
            if let Some(lp) = logprobs.take() {
                a.logprobs_data.push(lp);
            }
            commit_tool_call_close(a, tok, was_inside_thinking);
            continue;
        }

        // Keep this identical to speculative emission. Required tools are
        // protected by grammar / legacy-required mode, but an ordinary answer
        // may terminate on its first EOS after `</think>`.
        let suppress_eos = eos_is_suppressed(a, a.output_tokens.len());

        // Single-pass over eos_tokens (was 2× linear scan).
        let is_eos = a.eos_tokens.contains(&tok);
        if is_eos && !suppress_eos {
            // Stop/EOS token: do NOT stream to client (OpenAI spec: returned text
            // must not contain the stop sequence). The token is still added to
            // output_tokens for correct token count; the API layer strips the
            // decoded text for blocking responses.
            if let Some(lp) = logprobs.take() {
                a.logprobs_data.push(lp);
            }
            a.output_tokens.push(tok);
            a.finished = true;
        } else if is_eos {
            // EOS suppressed: grammar not terminated or legacy tool call not yet seen.
            // Don't stop, don't stream the EOS — the model must keep generating.
            // Don't add to output_tokens (EOS is discarded).
        } else {
            if let Some(lp) = logprobs.take() {
                a.logprobs_data.push(lp);
            }
            a.output_tokens.push(tok);
            if a.think_ended && !was_inside_thinking && !a.inside_thinking {
                a.post_think_gate_steps = mtp_gate::advance_entry_counter(a.post_think_gate_steps);
            }
            // Phase-C: if this committed token is a content-phase
            // boundary token (sentence end / newline) and the model is
            // hybrid (attention + SSM), snapshot the recurrent SSM
            // state now so a later watchdog rollback to this boundary
            // can also rewind h_state/conv_state — not just the KV
            // cache. Gated to content tokens because the watchdogs that
            // roll back all fire post-`</think>`, and `apply_rollback`
            // requires every dropped token to be a content token. No-op
            // for pure-attention models / disabled rings (see
            // `rollback::snapshot_boundary_if_ssm`).
            if !a.inside_thinking {
                rollback::snapshot_boundary_if_ssm(a, model);
            }
            // OPENCODE FIX: when the model spontaneously emits `<think>` even
            // though the request didn't ask for thinking (`enable_thinking=false`),
            // the `<think>` open token itself is suppressed (line ~1356), but
            // the thinking-content tokens that follow MUST also be kept off the
            // wire — otherwise opencode persists them as `assistant.content` and
            // on the next turn the model sees its own past garbage (fake
            // `<function=…>`, fake `<tool_response>`) as a "format example" and
            // continues the pattern. Tokens stay in `output_tokens` for the
            // blocking response path's reasoning_content extraction.
            let suppress_stream = a.inside_thinking && !a.enable_thinking;
            if let ResponseSink::Streaming(ref tx) = a.sink
                && !suppress_stream
            {
                let event = if let Some(lp) = a.logprobs_data.last().cloned() {
                    StreamEvent::TokenWithLogprobs(tok, lp)
                } else {
                    StreamEvent::Token(tok)
                };
                match tx.try_send(event) {
                    Ok(()) => {}
                    Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                        tracing::debug!(
                            "Streaming receiver dropped (decode_logits), finishing seq"
                        );
                        a.finished = true;
                    }
                    Err(tokio::sync::mpsc::error::TrySendError::Full(event)) => {
                        if let Err(e) = tx.blocking_send(event) {
                            tracing::error!(
                                "Streaming send failed during backpressure (decode_logits): {e}"
                            );
                            a.finished = true;
                        }
                    }
                }
            }
            if a.output_tokens.len() >= a.max_output_tokens {
                tracing::info!(
                    "process_decode_logits: max_output_tokens={} reached, output_tokens={}, thinking_tokens={}",
                    a.max_output_tokens,
                    a.output_tokens.len(),
                    a.thinking_tokens,
                );
                a.finished = true;
            }
            if a.remaining == 0 {
                tracing::info!(
                    "process_decode_logits: remaining=0, output_tokens={}, thinking_tokens={}",
                    a.output_tokens.len(),
                    a.thinking_tokens
                );
                a.finished = true;
            }
            // Grammar termination = end of sequence. With `stop_after_first=true`
            // (tool_choice="required"), the structural-tag matcher transitions
            // to its terminal state right after the single tool call closes.
            // The model's free distribution past that point can be degenerate
            // (Nemotron-Super-120B emits a `</parameter>` loop and never
            // samples EOS naturally). Finish here instead of letting it run.
            if a.grammar_state
                .as_ref()
                .is_some_and(|gs| gs.is_terminated())
            {
                a.finished = true;
            }

            // Check request timeout.
            check_request_timeout(a);
        }
    }
}
