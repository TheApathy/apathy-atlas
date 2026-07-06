// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_THINK_SPEC=1` — speculative decode inside `<think>` spans.
//!
//! The scheduler historically forced every batch containing a thinking
//! sequence into plain `step_decode_only` (see the `!a.inside_thinking`
//! MTP gate in `scheduler::run`): the plain decode path applies per-token
//! logit interventions during thinking — F1 reflection suppression, the
//! thinking-efficiency wave, the F2 confidence early stop, the tool-call
//! hard mask, the forced `</think>` injection — that the DFlash K=γ
//! verify (raw GPU argmax) does not replicate. Thinking is the majority
//! of tokens on a reasoning model, so the whole batch collapsed to
//! ~13-17 tok/s whenever any sequence was mid-`<think>`.
//!
//! This module replicates the interventions in a post-verify CPU accept
//! filter (same shape as `dflash_masked_accept`, the grammar sibling) so
//! speculation runs through thinking spans LOSSLESSLY:
//!
//!   * SLOW PATH: a verify-logits row is lazily D2H'd (cheap on GB10
//!     unified memory) and fed through [`process_seq_logits`] — the EXACT
//!     function the plain path samples with, including F1/F2/wave state
//!     evolution, penalties, logit_bias and seeded sampling — so the two
//!     paths share one implementation and cannot drift.
//!   * FAST PATH: no D2H when the plain-path token is PROVABLY the raw
//!     verify argmax — greedy request with neutral penalties, wave off,
//!     F2 window closed, and the raw argmax outside the
//!     intervention-sensitive id set (see [`position_fast_path_ok`]).
//!     The forced `</think>` injection is distribution-independent
//!     (masks everything but `</think>`), so it stays on the fast path.
//!
//! Acceptance TRUNCATES (never rewrites): a draft is accepted only while
//! it equals the plain-path token; the first divergence becomes the
//! bonus. `</think>` and EOS are phase-boundary tokens — they always end
//! the walk as the bonus (never as an accepted draft) so `a.last_token`
//! keeps the plain-path contract of "committed but not yet fed to the
//! model" across the transition.
//!
//! Default OFF: with the env var unset, `ThinkSpecCtx::enabled` is false,
//! the scheduler gate keeps excluding thinking sequences from MTP, and no
//! code path here runs — byte-identical to the historical behavior.

use super::*;

/// `ATLAS_THINK_SPEC=1` — read once at first use.
pub(super) fn think_spec_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_THINK_SPEC").ok().as_deref() == Some("1"))
}

/// Per-run parameters threaded from `scheduler::run` into `step_mtp` /
/// `step_verify_dflash` (the tokens not carried on `ActiveSeq`).
#[derive(Clone, Copy)]
pub(super) struct ThinkSpecCtx<'a> {
    /// Master gate: `ATLAS_THINK_SPEC=1` AND the model has BF16 decode
    /// logits AND `--adaptive-sampling` is off (its per-token entropy
    /// observation is plain-path state this filter does not replicate).
    pub enabled: bool,
    /// ``` fence token — `in_code_fence` parity (gates the forced
    /// `</think>` injection deferral).
    pub code_fence_token: Option<u32>,
    /// F1 reflection-suppression id set (`-10.0` during thinking).
    pub reflection_suppress_ids: &'a [u32],
}

/// Result of a thinking-span accept walk. Tokens (accepted prefix AND
/// bonus) are already committed/streamed by the walk itself — the caller
/// must skip its own emit loop and only run the seq/SSM bookkeeping.
pub(super) struct ThinkAcceptOutcome {
    /// Drafts accepted (excludes the bonus). Same contract as the legacy
    /// accept-prefix: `seq` rollback and the SSM commit use it unchanged.
    pub num_accepted: usize,
    /// The committed bonus token. `None` only when the sequence finished
    /// (or a logits D2H failed) mid-walk — the caller must bail out
    /// exactly like the legacy emit loop's early return.
    pub bonus: Option<u32>,
}

/// Request-level fast-path eligibility: the plain path's temp-0 sample is
/// a pure post-intervention argmax only when no penalty / bias / logprob
/// machinery can re-order or observe the distribution
/// (`sample_with_params_seeded` applies these BEFORE its greedy bypass).
pub(super) fn fast_path_seq_eligible(a: &ActiveSeq) -> bool {
    a.temperature == 0.0
        && a.repetition_penalty == 1.0
        && a.presence_penalty == 0.0
        && a.frequency_penalty == 0.0
        && a.lz_penalty <= 0.0
        && a.dry_multiplier <= 0.0
        && a.logit_bias.is_empty()
        && a.top_logprobs.is_none()
}

/// Position-level fast-path gate. The interventions can only move the
/// argmax when (a) the wave shapes logits, (b) the F2 window needs the
/// distribution for its state evolution, (c) the raw argmax is itself a
/// suppressed id (a -10 on a NON-argmax id can never promote another
/// token), or (d) the raw argmax is the hard-masked `<tool_call>` start.
pub(super) fn position_fast_path_ok(
    raw_argmax: u32,
    reflection_suppress_ids: &[u32],
    tool_call_start_token: Option<u32>,
    wave_active: bool,
    f2_window_active: bool,
) -> bool {
    !wave_active
        && !f2_window_active
        && !reflection_suppress_ids.contains(&raw_argmax)
        && tool_call_start_token != Some(raw_argmax)
}

/// Whether the F2 confidence early-stop would observe THIS position on the
/// plain path (`process_seq_logits`): armed sequences skip it, and it only
/// opens past [`CONFIDENCE_EARLY_STOP_MIN_THINKING`] thinking tokens.
pub(super) fn f2_window_active(a: &ActiveSeq) -> bool {
    !a.force_end_thinking
        && a.thinking_tokens >= CONFIDENCE_EARLY_STOP_MIN_THINKING
        && watchdog_params().confidence_early_stop
}

/// Whether the plain path would force-inject `</think>` at this position
/// (mirrors the defer arithmetic in `process_seq_logits` exactly).
fn forced_injection_pending(a: &ActiveSeq) -> bool {
    let defer_hard_override = match a.thinking_budget {
        Some(b) => a.thinking_tokens >= b.saturating_mul(THINK_DEFER_BUDGET_FACTOR),
        None => a.thinking_tokens >= THINK_DEFER_ABS_CEILING,
    };
    should_inject_think_end(a.force_end_thinking, a.in_code_fence, defer_hard_override)
}

/// Thinking-span accept walk over the verify positions.
///
/// Row `i` of the verify logits is the target's prediction for the slot
/// `drafts[i]` occupies (row `drafts.len()` is the bonus slot) — the same
/// indexing as `dflash_masked_accept`. Per position: derive the
/// plain-path token (fast path = raw argmax, slow path = the full
/// [`process_seq_logits`] pipeline over the D2H'd row), COMMIT it via
/// [`commit_thinking_token`] (streaming + the plain path's per-token side
/// effects), and accept while it equals `drafts[i]`. The first
/// divergence — or a `</think>`/EOS boundary — becomes the bonus and ends
/// the walk. State (F2 run, difficulty probe, fence parity, THINK_LOOP
/// watchdog, budget arming) evolves on `a` exactly as one plain-decode
/// step per committed token.
///
/// `fetch_row(i, buf)` fills `buf` with the BF16 bytes of verify row `i`
/// and returns `true` (`false` = fatal D2H failure → the sequence is
/// finished, mirroring the other verify error paths). The fill-buffer
/// shape lets one ~vocab*2-byte allocation serve every slow-path row in
/// the walk. `on_think_end` runs after a committed `</think>`
/// (production: the plain path's SSM boundary snapshot).
pub(super) fn dflash_thinking_accept(
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
    mut fetch_row: impl FnMut(usize, &mut Vec<u8>) -> bool,
    mut on_think_end: impl FnMut(&mut ActiveSeq),
) -> ThinkAcceptOutcome {
    debug_assert!(a.inside_thinking);
    let seq_fast = fast_path_seq_eligible(a);
    let wave_active = think_efficiency_config().is_active();
    let mut row_buf: Vec<u8> = Vec::new();
    let mut num_accepted = 0usize;
    for i in 0..verified.len() {
        let raw = verified[i];
        let fast = seq_fast
            && position_fast_path_ok(
                raw,
                ctx.reflection_suppress_ids,
                a.tool_call_start_token,
                wave_active,
                f2_window_active(a),
            );
        let (target, logprobs) = if fast {
            // Forced `</think>` injection masks every logit but the end
            // token — distribution-independent, so no D2H is needed even
            // when it fires (F2 is skipped once `force_end_thinking` is
            // armed, and the wave is off on this path).
            match (forced_injection_pending(a), a.think_end_token) {
                (true, Some(end)) => (end, None),
                _ => (raw, None),
            }
        } else {
            if !fetch_row(i, &mut row_buf) {
                // Mirrors the fatal-error handling of the surrounding
                // verify step: without the row we cannot re-derive the
                // plain-path token, and committing the raw argmax would
                // be lossy — end the sequence instead.
                tracing::error!("think-spec verify: logits row {i} unavailable; finishing seq");
                a.finished = true;
                return ThinkAcceptOutcome {
                    num_accepted,
                    bonus: None,
                };
            }
            let vocab = row_buf.len() / 2;
            process_seq_logits(
                a,
                &row_buf,
                0,
                vocab,
                2,
                false,
                a.think_end_token,
                a.think_start_token,
                a.tool_call_start_token,
                a.tool_call_end_token,
                ctx.reflection_suppress_ids,
                // --adaptive-sampling disqualifies think-spec at the
                // scheduler gate, so the plain path would also run with
                // the adaptive observer inert here.
                false,
            )
        };
        // `</think>` / EOS are phase-boundary tokens: always the bonus,
        // never an accepted draft — keeps `a.last_token` on the
        // plain-path contract (committed, not yet fed to the model), so
        // the next step feeds it exactly like the plain path would.
        let boundary = a.think_end_token == Some(target) || a.eos_tokens.contains(&target);
        let draft_match = i < drafts.len() && i + 1 < verified.len() && drafts[i] == target;
        commit_thinking_token(a, target, logprobs, ctx.code_fence_token, &mut on_think_end);
        if boundary || !draft_match {
            if !a.finished {
                a.last_token = target;
            }
            return ThinkAcceptOutcome {
                num_accepted,
                bonus: Some(target),
            };
        }
        if a.finished {
            return ThinkAcceptOutcome {
                num_accepted: num_accepted + 1,
                bonus: None,
            };
        }
        num_accepted += 1;
    }
    // Unreachable with a well-formed `verified` (the `i + 1 <
    // verified.len()` guard turns the last row into the bonus); an empty
    // `verified` is rejected by the production wrapper. If a logic error
    // ever lands here, the caller's `bonus: None` early-return skips the
    // SSM commit — finish the sequence so it cannot keep decoding on an
    // uncommitted state.
    debug_assert!(
        verified.is_empty(),
        "think-spec walk fell off a non-empty verified loop"
    );
    a.finished = true;
    ThinkAcceptOutcome {
        num_accepted,
        bonus: None,
    }
}

/// Commit one thinking-span token with the plain path's per-token side
/// effects (`process_decode_logits`'s thinking branch, decode_logits_step
/// ~202-253 + the EOS suppression at ~364), on top of what `emit_token`
/// already does for the speculative paths:
///
///   * `</think>`: reset the F2 run + fence parity (the plain path does,
///     `emit_token` doesn't), then `emit_token` handles the transition
///     flags / push / stream; `on_think_end` runs the plain path's SSM
///     boundary snapshot.
///   * EOS inside thinking (`thinking_suppresses_eos`): counted as a
///     thinking token and fed back as `last_token` by the caller, but
///     never pushed to `output_tokens`, never streamed, never finishes.
///   * regular token: `emit_token` (push + stream + `thinking_tokens` +
///     budget arming), then the two plain-path effects `emit_token`
///     lacks — ``` fence parity and the THINK_LOOP watchdog.
fn commit_thinking_token(
    a: &mut ActiveSeq,
    tok: u32,
    logprobs: Option<crate::api::TokenLogprobs>,
    code_fence_token: Option<u32>,
    on_think_end: &mut impl FnMut(&mut ActiveSeq),
) {
    debug_assert!(a.inside_thinking);
    if a.think_end_token == Some(tok) {
        // Plain-path transition resets that emit_token doesn't perform.
        a.consecutive_confident = 0;
        a.in_code_fence = false;
        emit_token(a, tok, logprobs);
        clear_stale_require_tool_call(a, a.output_tokens.len().saturating_sub(1));
        if !a.finished {
            // Plain path: `</think>` is a non-thinking committed token →
            // SSM boundary snapshot opportunity (rollback ring).
            on_think_end(a);
            check_request_timeout(a);
        }
        return;
    }
    if a.eos_tokens.contains(&tok) {
        // thinking_suppresses_eos: the plain path treats an in-thinking
        // EOS as a normal thinking token (counted, fed back as input)
        // that is neither pushed, streamed, nor sequence-ending.
        if let Some(lp) = logprobs {
            a.logprobs_data.push(lp);
        }
        a.thinking_tokens += 1;
        a.in_code_fence = toggle_code_fence(a.in_code_fence, tok, code_fence_token);
        if let Some(budget) = a.thinking_budget
            && a.thinking_tokens >= budget
            && !a.force_end_thinking
        {
            a.force_end_thinking = true;
            tracing::info!("Thinking budget exhausted ({budget} tokens), forcing </think>");
        }
        run_think_loop_watchdog(a, false);
        clear_stale_require_tool_call(a, a.output_tokens.len());
        return;
    }
    emit_token(a, tok, logprobs); // push + stream + thinking_tokens += 1 + budget arming
    if a.finished {
        return;
    }
    a.in_code_fence = toggle_code_fence(a.in_code_fence, tok, code_fence_token);
    run_think_loop_watchdog(a, true);
    clear_stale_require_tool_call(a, a.output_tokens.len().saturating_sub(1));
    check_request_timeout(a);
}

/// Token-period THINK_LOOP watchdog — verbatim from the plain path
/// (decode_logits_step ~238-252). `exclude_last`: the plain path scans
/// `output_tokens` BEFORE the current token is pushed; `emit_token` has
/// already pushed it, so the emitted branches drop the tail element to
/// keep the scan frame identical. The suppressed-EOS branch never pushes,
/// so it scans the full vec.
fn run_think_loop_watchdog(a: &mut ActiveSeq, exclude_last: bool) {
    if a.force_end_thinking
        || a.thinking_tokens < THINK_LOOP_MIN_TOKENS
        || !a.thinking_tokens.is_multiple_of(THINK_LOOP_CHECK_STRIDE)
    {
        return;
    }
    let tokens: &[u32] = if exclude_last {
        &a.output_tokens[..a.output_tokens.len().saturating_sub(1)]
    } else {
        &a.output_tokens
    };
    if detect_thinking_token_loop(tokens) {
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

/// require_tool_call safety valve — verbatim from the plain path
/// (decode_logits_step ~277-285, runs for every token incl. suppressed
/// EOS). `committed_len` is `output_tokens.len()` in the plain path's
/// frame, i.e. EXCLUDING the current token (which it pushes later).
fn clear_stale_require_tool_call(a: &mut ActiveSeq, committed_len: usize) {
    if a.require_tool_call && committed_len > 512 {
        tracing::warn!(
            "require_tool_call safety: no <tool_call> after 512 tokens, clearing EOS suppression"
        );
        a.require_tool_call = false;
    }
}

/// Per-token request-timeout check — plain path runs it for every pushed
/// token (decode_logits_step ~527-534); `emit_token` never does, so the
/// commit helper restores it for the thinking span.
fn check_request_timeout(a: &mut ActiveSeq) {
    if !a.finished
        && let Some(deadline) = a.timeout_at
        && Instant::now() >= deadline
    {
        tracing::warn!("Request timeout after {:?}", a.request_start.elapsed());
        a.finished = true;
    }
}

/// Production wrapper: bind the walk to the model's resident `[K, vocab]`
/// BF16 verify-logits buffer (same lazy per-row D2H as
/// `dflash_masked_accept`) and to the SSM boundary snapshot. Returns
/// `None` (caller falls back to the legacy unmasked accept) only for the
/// defensive cases that the scheduler gate should already exclude —
/// fp32-logits models and an empty `verified`.
pub(super) fn run_dflash_thinking_accept(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
) -> Option<ThinkAcceptOutcome> {
    if verified.is_empty() {
        return None;
    }
    let logits_base = model.logits_buffer_ptr();
    if model.logits_ptr_is_fp32(logits_base) {
        // The scheduler gate clears `ctx.enabled` for fp32-logits models;
        // reaching here means the gate and the buffer disagree — fall
        // back to the legacy (lossy-in-thinking) accept and say so.
        tracing::warn!(
            "ATLAS_THINK_SPEC: fp32 verify logits despite BF16 gate; thinking filter skipped"
        );
        return None;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!(
            "ATLAS_THINK_SPEC=1: speculative decode active inside <think> spans \
             (post-verify plain-path accept filter)"
        );
    });
    let vocab = model.vocab_size();
    Some(dflash_thinking_accept(
        a,
        drafts,
        verified,
        ctx,
        |i, buf| {
            buf.resize(vocab * 2, 0);
            match model.copy_logits_to_host(logits_base.offset(i * vocab * 2), buf) {
                Ok(()) => true,
                Err(e) => {
                    tracing::error!("think-spec verify: logits D2H failed at row {i}: {e:#}");
                    false
                }
            }
        },
        |a| rollback::snapshot_boundary_if_ssm(a, model),
    ))
}

/// Bootstrap-decode sibling of the verify filter: sample ONE token from a
/// single-row decode-logits buffer through the full plain-path pipeline
/// (`process_seq_logits`) and commit it with the thinking side effects.
/// Used by `step_mtp`'s Phase A when a thinking sequence has no pending
/// drafts — byte-identical to one `step_decode_only` token for this seq.
///
/// Returns the committed token, or `None` on a fatal D2H error (the
/// sequence is finished, mirroring the bootstrap decode error handling).
pub(super) fn bootstrap_thinking_token(
    model: &dyn Model,
    a: &mut ActiveSeq,
    logits: DevicePtr,
    ctx: &ThinkSpecCtx<'_>,
) -> Option<u32> {
    let vocab = model.vocab_size();
    let mut buf = vec![0u8; vocab * 2];
    if let Err(e) = model.copy_logits_to_host(logits, &mut buf) {
        tracing::error!("think-spec bootstrap: logits D2H failed: {e:#}");
        a.finished = true;
        return None;
    }
    let (tok, lp) = process_seq_logits(
        a,
        &buf,
        0,
        vocab,
        2,
        false,
        a.think_end_token,
        a.think_start_token,
        a.tool_call_start_token,
        a.tool_call_end_token,
        ctx.reflection_suppress_ids,
        false, // see dflash_thinking_accept: adaptive-sampling is gated off
    );
    commit_thinking_token(a, tok, lp, ctx.code_fence_token, &mut |a| {
        rollback::snapshot_boundary_if_ssm(a, model)
    });
    Some(tok)
}

#[cfg(test)]
#[path = "think_spec_accept_tests.rs"]
mod think_spec_accept_tests;
