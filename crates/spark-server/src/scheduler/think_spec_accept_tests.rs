// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-free unit tests for the `ATLAS_THINK_SPEC` accept filter.
//!
//! The walk is exercised end-to-end without a device: BF16 logit rows are
//! fabricated on the host and served through the `fetch_row` closure, and
//! the `ActiveSeq` under test uses a blocking sink + disabled SSM ring so
//! `emit_token` runs its full bookkeeping with no model. Global
//! `OnceLock` state (watchdog params, think-efficiency config) is never
//! set by any test in this crate, so every test sees the inert defaults
//! (`confidence_early_stop=true`, wave off) — the same state a pre-boot
//! caller sees.

use super::super::*;
use super::{ThinkSpecCtx, dflash_thinking_accept, fast_path_seq_eligible, position_fast_path_ok};
use spark_model::traits::SequenceState;

// Tiny test vocabulary (10 ids). Content ids: 0, 1, 9.
const SUPPRESS: u32 = 2;
const FENCE: u32 = 3;
const THINK_START: u32 = 4;
const THINK_END: u32 = 5;
const TC_START: u32 = 6;
const EOS: u32 = 7;
const TC_END: u32 = 8;
const VOCAB: usize = 10;

fn seq_state() -> SequenceState {
    SequenceState {
        tokens: Vec::new(),
        block_table: Vec::new(),
        seq_len: 0,
        layer_states: Vec::new(),
        proposer_state: None,
        slot_idx: 0,
        marconi_skip_to: 0,
        session_hash: 0,
        chunked_prefill_meta: None,
        cached_prefix_tokens: 0,
        prompt_len: 0,
        disk_block_ids: Vec::new(),
        mtp_lastk_host_buf: Vec::new(),
        mtp_lastk_host_filled: 0,
        mtp_lastk_end_abs: 0,
        disk_last_offloaded_per_layer: Vec::new(),
    }
}

/// An `ActiveSeq` mid-`<think>`, greedy, neutral penalties (fast-path
/// eligible), blocking sink, disabled SSM ring.
fn think_seq() -> ActiveSeq {
    ActiveSeq {
        seq: seq_state(),
        session_hash: 0,
        last_token: 0,
        output_tokens: Vec::new(),
        remaining: 1000,
        min_tokens: 0,
        eos_tokens: vec![EOS],
        finished: false,
        sink: ResponseSink::Blocking(None),
        cancel_flag: None,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 3,
        dry_sequence_breakers: Vec::new(),
        logit_bias: Vec::new(),
        inside_thinking: true,
        enable_thinking: true,
        thinking_budget: None,
        spontaneous_think_budget: 512,
        thinking_tokens: 5,
        force_end_thinking: false,
        consecutive_confident: 0,
        in_code_fence: false,
        think_end_token: Some(THINK_END),
        think_start_token: Some(THINK_START),
        think_ended: false,
        think_just_ended: false,
        think_skip_count: 0,
        tool_call_end_token: Some(TC_END),
        require_tool_call: false,
        tool_call_start_token: Some(TC_START),
        tool_call_opened: false,
        inside_tool_body: false,
        suppress_tool_call: false,
        disable_mtp: false,
        content_started: false,
        content_tokens: 0,
        prose_tokens_since_last_tool: 0,
        think_watchdog_fires: 0,
        rollback_count: 0,
        ssm_rollback_ring: SsmDecodeRing::new(0),
        grammar_state: None,
        pending_drafts: Vec::new(),
        pending_tree_payload: None,
        last_token_time: Instant::now(),
        request_start: Instant::now(),
        decode_start: Instant::now(),
        seed: None,
        top_logprobs: None,
        logprobs_data: Vec::new(),
        timeout_at: None,
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(0.0),
        cached_prompt_tokens: 0,
        difficulty_probe: Default::default(),
    }
}

fn ctx(suppress: &[u32]) -> ThinkSpecCtx<'_> {
    ThinkSpecCtx {
        enabled: true,
        code_fence_token: Some(FENCE),
        reflection_suppress_ids: suppress,
    }
}

/// Encode f32 logits as little-endian BF16 bytes (truncation — test
/// values are chosen to be exactly representable).
fn bf16_row(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// A `[VOCAB]` row whose argmax is `top` (logit 8.0) over a -8.0 floor,
/// with optional per-id overrides.
fn row_with(top: u32, overrides: &[(u32, f32)]) -> Vec<u8> {
    let mut v = vec![-8.0f32; VOCAB];
    v[top as usize] = 8.0;
    for &(id, l) in overrides {
        v[id as usize] = l;
    }
    bf16_row(&v)
}

fn no_rows(_i: usize, _buf: &mut Vec<u8>) -> bool {
    panic!("fast path must not fetch logits rows");
}

/// Fill-buffer adapter over a slice of fabricated rows.
fn serve_rows(rows: &[Vec<u8>]) -> impl FnMut(usize, &mut Vec<u8>) -> bool + '_ {
    move |i, buf| {
        buf.clear();
        buf.extend_from_slice(&rows[i]);
        true
    }
}

fn no_snapshot(_a: &mut ActiveSeq) {}

// ── Pure gate tests ─────────────────────────────────────────────────────────

#[test]
fn seq_eligibility_requires_greedy_and_neutral_penalties() {
    let a = think_seq();
    assert!(fast_path_seq_eligible(&a));
    let mut t = think_seq();
    t.temperature = 0.7;
    assert!(!fast_path_seq_eligible(&t));
    let mut r = think_seq();
    r.repetition_penalty = 1.1;
    assert!(!fast_path_seq_eligible(&r));
    let mut b = think_seq();
    b.logit_bias = vec![(1, 2.0)];
    assert!(!fast_path_seq_eligible(&b));
    let mut l = think_seq();
    l.top_logprobs = Some(4);
    assert!(!fast_path_seq_eligible(&l));
}

#[test]
fn position_gate_slow_paths_sensitive_argmaxes() {
    let sup = [SUPPRESS];
    assert!(position_fast_path_ok(1, &sup, Some(TC_START), false, false));
    assert!(!position_fast_path_ok(
        SUPPRESS,
        &sup,
        Some(TC_START),
        false,
        false
    ));
    assert!(!position_fast_path_ok(
        TC_START,
        &sup,
        Some(TC_START),
        false,
        false
    ));
    assert!(!position_fast_path_ok(1, &sup, Some(TC_START), true, false)); // wave
    assert!(!position_fast_path_ok(1, &sup, Some(TC_START), false, true)); // F2 window
}

// ── Fast path ───────────────────────────────────────────────────────────────

#[test]
fn fast_path_accepts_matching_drafts_without_d2h() {
    let mut a = think_seq();
    let sup = [SUPPRESS];
    let drafts = [1, 0, 9];
    let verified = [1, 0, 9, 1]; // bonus row predicts 1
    let out = dflash_thinking_accept(&mut a, &drafts, &verified, &ctx(&sup), no_rows, no_snapshot);
    assert_eq!(out.num_accepted, 3);
    assert_eq!(out.bonus, Some(1));
    assert_eq!(a.output_tokens, vec![1, 0, 9, 1]);
    assert_eq!(a.last_token, 1);
    assert_eq!(a.thinking_tokens, 5 + 4);
    assert!(a.inside_thinking && !a.finished);
}

#[test]
fn fast_path_truncates_at_divergence() {
    let mut a = think_seq();
    let sup = [SUPPRESS];
    let drafts = [1, 0, 9];
    let verified = [1, 9, 0, 1]; // target at slot 1 is 9, draft said 0
    let out = dflash_thinking_accept(&mut a, &drafts, &verified, &ctx(&sup), no_rows, no_snapshot);
    assert_eq!(out.num_accepted, 1);
    assert_eq!(out.bonus, Some(9));
    assert_eq!(a.output_tokens, vec![1, 9]);
    assert_eq!(a.last_token, 9);
}

// ── F1 reflection suppression ───────────────────────────────────────────────

#[test]
fn suppressed_argmax_with_close_runner_up_truncates() {
    let mut a = think_seq();
    let sup = [SUPPRESS];
    // Raw argmax is SUPPRESS (5.0); runner-up 1 at 4.0 is within the -10
    // penalty → plain-path argmax is 1, so the draft (SUPPRESS) is wrong.
    let rows = [bf16_row(&{
        let mut v = vec![-8.0f32; VOCAB];
        v[SUPPRESS as usize] = 5.0;
        v[1] = 4.0;
        v
    })];
    let drafts = [SUPPRESS, 0];
    let verified = [SUPPRESS, 0, 1];
    let out = dflash_thinking_accept(
        &mut a,
        &drafts,
        &verified,
        &ctx(&sup),
        serve_rows(&rows),
        no_snapshot,
    );
    assert_eq!(out.num_accepted, 0);
    assert_eq!(out.bonus, Some(1));
    assert_eq!(a.output_tokens, vec![1]);
}

#[test]
fn suppressed_argmax_still_dominant_is_accepted() {
    let mut a = think_seq();
    let sup = [SUPPRESS];
    // SUPPRESS at 24.0 vs runner-up 1.0: still argmax after -10 → the
    // draft matches the plain-path token and acceptance continues.
    let mut fetched = 0usize;
    let row0 = bf16_row(&{
        let mut v = vec![-8.0f32; VOCAB];
        v[SUPPRESS as usize] = 24.0;
        v[1] = 1.0;
        v
    });
    let drafts = [SUPPRESS, 1];
    let verified = [SUPPRESS, 1, 0];
    let out = dflash_thinking_accept(
        &mut a,
        &drafts,
        &verified,
        &ctx(&sup),
        |i, buf: &mut Vec<u8>| {
            fetched += 1;
            assert_eq!(i, 0, "only the suppressed-argmax row needs D2H");
            buf.clear();
            buf.extend_from_slice(&row0);
            true
        },
        no_snapshot,
    );
    assert_eq!(fetched, 1);
    assert_eq!(out.num_accepted, 2);
    assert_eq!(out.bonus, Some(0));
    assert_eq!(a.output_tokens, vec![SUPPRESS, 1, 0]);
}

// ── Tool-call hard mask ─────────────────────────────────────────────────────

#[test]
fn tool_call_start_argmax_is_remapped_to_runner_up() {
    let mut a = think_seq();
    let sup = [SUPPRESS];
    let rows = [row_with(TC_START, &[(1, 4.0)])];
    let drafts = [TC_START];
    let verified = [TC_START, 0];
    let out = dflash_thinking_accept(
        &mut a,
        &drafts,
        &verified,
        &ctx(&sup),
        serve_rows(&rows),
        no_snapshot,
    );
    assert_eq!(out.num_accepted, 0);
    assert_eq!(out.bonus, Some(1)); // -inf mask → runner-up
    assert_eq!(a.output_tokens, vec![1]);
}

// ── Forced </think> injection ───────────────────────────────────────────────

#[test]
fn forced_injection_truncates_all_positions_to_think_end() {
    let mut a = think_seq();
    a.force_end_thinking = true;
    let sup = [SUPPRESS];
    let drafts = [1, 0];
    let verified = [1, 0, 9];
    let out = dflash_thinking_accept(&mut a, &drafts, &verified, &ctx(&sup), no_rows, no_snapshot);
    assert_eq!(out.num_accepted, 0);
    assert_eq!(out.bonus, Some(THINK_END));
    assert_eq!(a.output_tokens, vec![THINK_END]);
    assert!(!a.inside_thinking && a.think_ended && a.think_just_ended);
    assert!(!a.force_end_thinking); // cleared by the transition
    assert_eq!(a.consecutive_confident, 0);
    assert_eq!(a.last_token, THINK_END);
}

#[test]
fn forced_injection_defers_inside_code_fence() {
    let mut a = think_seq();
    a.force_end_thinking = true;
    a.in_code_fence = true;
    a.thinking_budget = Some(100); // 5 << 300 → no hard override
    let sup = [SUPPRESS];
    let drafts = [1];
    let verified = [1, 0];
    let out = dflash_thinking_accept(&mut a, &drafts, &verified, &ctx(&sup), no_rows, no_snapshot);
    assert_eq!(out.num_accepted, 1);
    assert_eq!(out.bonus, Some(0));
    assert!(
        a.inside_thinking,
        "deferred injection must not close thinking"
    );
}

#[test]
fn forced_injection_hard_override_fires_mid_fence() {
    let mut a = think_seq();
    a.force_end_thinking = true;
    a.in_code_fence = true;
    a.thinking_budget = Some(2);
    a.thinking_tokens = 6; // >= 3 * budget → hard override
    let sup = [SUPPRESS];
    let out = dflash_thinking_accept(&mut a, &[1], &[1, 0], &ctx(&sup), no_rows, no_snapshot);
    assert_eq!(out.num_accepted, 0);
    assert_eq!(out.bonus, Some(THINK_END));
    assert!(!a.inside_thinking);
}

#[test]
fn budget_exhaustion_mid_walk_arms_injection_for_next_position() {
    let mut a = think_seq();
    a.thinking_tokens = 10;
    a.thinking_budget = Some(11); // first accepted token exhausts it
    let sup = [SUPPRESS];
    let drafts = [1, 0, 9];
    let verified = [1, 0, 9, 1];
    let out = dflash_thinking_accept(&mut a, &drafts, &verified, &ctx(&sup), no_rows, no_snapshot);
    assert_eq!(out.num_accepted, 1); // token 1 accepted, then </think> injected
    assert_eq!(out.bonus, Some(THINK_END));
    assert_eq!(a.output_tokens, vec![1, THINK_END]);
}

// ── Phase boundaries ────────────────────────────────────────────────────────

#[test]
fn think_end_in_drafts_stops_as_bonus_never_as_accepted_draft() {
    let mut a = think_seq();
    let sup = [SUPPRESS];
    let drafts = [1, THINK_END, 9];
    let verified = [1, THINK_END, 9, 0];
    let out = dflash_thinking_accept(&mut a, &drafts, &verified, &ctx(&sup), no_rows, no_snapshot);
    assert_eq!(out.num_accepted, 1);
    assert_eq!(out.bonus, Some(THINK_END));
    assert_eq!(a.output_tokens, vec![1, THINK_END]);
    assert!(!a.inside_thinking && a.think_ended);
    assert_eq!(a.last_token, THINK_END);
    // Draft 9 (post-transition) was never committed.
    assert_eq!(a.thinking_tokens, 5 + 1);
}

#[test]
fn eos_inside_thinking_is_suppressed_but_fed_back() {
    let mut a = think_seq();
    let sup = [SUPPRESS];
    let drafts = [EOS, 1];
    let verified = [EOS, 1, 0];
    let out = dflash_thinking_accept(&mut a, &drafts, &verified, &ctx(&sup), no_rows, no_snapshot);
    assert_eq!(out.num_accepted, 0);
    assert_eq!(out.bonus, Some(EOS));
    assert!(
        a.output_tokens.is_empty(),
        "suppressed EOS never reaches output"
    );
    assert!(!a.finished, "EOS inside thinking must not end the sequence");
    assert!(a.inside_thinking);
    assert_eq!(
        a.thinking_tokens,
        5 + 1,
        "plain path counts it as a thinking token"
    );
    assert_eq!(a.last_token, EOS, "plain path feeds the EOS back as input");
}

// ── F2 confidence early stop (slow-path state evolution) ────────────────────

#[test]
fn f2_window_slow_paths_every_position_and_tracks_confidence() {
    let mut a = think_seq();
    a.thinking_tokens = 400; // window open (confidence_early_stop default true)
    let sup = [SUPPRESS];
    // Peaked rows: top-1 prob ≈ 1.0 ≥ 0.95 → each position increments the run.
    let mut fetched = 0usize;
    let drafts = [1, 0, 9];
    let verified = [1, 0, 9, 1];
    let out = dflash_thinking_accept(
        &mut a,
        &drafts,
        &verified,
        &ctx(&sup),
        |i, buf: &mut Vec<u8>| {
            fetched += 1;
            buf.clear();
            buf.extend_from_slice(&row_with(verified[i], &[]));
            true
        },
        no_snapshot,
    );
    assert_eq!(fetched, 4, "open F2 window forces D2H on every position");
    assert_eq!(out.num_accepted, 3);
    assert_eq!(out.bonus, Some(1));
    assert_eq!(
        a.consecutive_confident, 4,
        "one increment per processed position"
    );
}

#[test]
fn f2_confidence_parity_with_manual_plain_path_replay() {
    // Twin check: the walk's F2/consecutive_confident evolution must equal
    // a hand-rolled plain-path loop (process_seq_logits + plain emission)
    // over the same rows — they share the same implementation, so this
    // pins the walk's state threading (order + increments).
    let sup = [SUPPRESS];
    let targets = [1u32, 0, 9, 1];
    let rows: Vec<Vec<u8>> = targets.iter().map(|&t| row_with(t, &[])).collect();

    let mut spec = think_seq();
    spec.thinking_tokens = 400;
    let drafts = [1, 0, 9];
    let verified = [1, 0, 9, 1];
    let rows_c = rows.clone();
    let _ = dflash_thinking_accept(
        &mut spec,
        &drafts,
        &verified,
        &ctx(&sup),
        serve_rows(&rows_c),
        no_snapshot,
    );

    let mut plain = think_seq();
    plain.thinking_tokens = 400;
    let (te, ts, tcs, tce) = (
        plain.think_end_token,
        plain.think_start_token,
        plain.tool_call_start_token,
        plain.tool_call_end_token,
    );
    for row in &rows {
        let (tok, _) = process_seq_logits(
            &mut plain, row, 0, VOCAB, 2, false, te, ts, tcs, tce, &sup, false,
        );
        // Plain emission block for a regular thinking token.
        plain.output_tokens.push(tok);
        plain.thinking_tokens += 1;
        plain.last_token = tok;
    }
    assert_eq!(spec.consecutive_confident, plain.consecutive_confident);
    assert_eq!(spec.force_end_thinking, plain.force_end_thinking);
    assert_eq!(spec.thinking_tokens, plain.thinking_tokens);
    assert_eq!(spec.output_tokens, plain.output_tokens);
}

#[test]
fn f2_run_resets_on_unconfident_position() {
    let mut a = think_seq();
    a.thinking_tokens = 400;
    a.consecutive_confident = 7;
    let sup = [SUPPRESS];
    // Flat-ish row: two near-equal logits → top-1 prob ≈ 0.5 < 0.95.
    // `verified` has a single row, so position 0 is the bonus row.
    let rows = [row_with(1, &[(0, 8.0)])];
    let out = dflash_thinking_accept(
        &mut a,
        &[1],
        &[1],
        &ctx(&sup),
        serve_rows(&rows),
        no_snapshot,
    );
    // argmax ties resolve to the LAST max under the plain sampler
    // (`max_by` keeps later elements) — target is 1, matching the draft.
    assert_eq!(out.num_accepted, 0); // i+1 >= verified.len() → bonus row
    assert_eq!(out.bonus, Some(1));
    assert_eq!(
        a.consecutive_confident, 0,
        "unconfident token resets the run"
    );
}

// ── THINK_LOOP watchdog ─────────────────────────────────────────────────────

#[test]
fn think_loop_watchdog_fires_and_injects_think_end() {
    let mut a = think_seq();
    a.thinking_tokens = 55; // next commit → 56, a multiple of the stride (8)
    // Committed output tail carries a period-4 needle repeated 3x.
    let mut out_toks: Vec<u32> = (0..36).map(|i| 100 + i).collect();
    for _ in 0..3 {
        out_toks.extend_from_slice(&[9, 8, 1, 3]);
    }
    a.output_tokens = out_toks;
    let sup = [SUPPRESS];
    let drafts = [1, 0];
    let verified = [1, 0, 9];
    let out = dflash_thinking_accept(&mut a, &drafts, &verified, &ctx(&sup), no_rows, no_snapshot);
    assert_eq!(out.num_accepted, 1, "watchdog armed after the first accept");
    assert_eq!(out.bonus, Some(THINK_END));
    assert_eq!(a.think_watchdog_fires, 1);
}

// ── Slow-path plumbing ──────────────────────────────────────────────────────

#[test]
fn nonzero_temperature_slow_paths_every_position() {
    let mut a = think_seq();
    a.temperature = 0.7;
    a.top_k = 1; // restricts sampling to the argmax → deterministic
    let sup = [SUPPRESS];
    let mut fetched = 0usize;
    let drafts = [1, 0];
    let verified = [1, 0, 9];
    let out = dflash_thinking_accept(
        &mut a,
        &drafts,
        &verified,
        &ctx(&sup),
        |i, buf: &mut Vec<u8>| {
            fetched += 1;
            buf.clear();
            buf.extend_from_slice(&row_with(verified[i], &[]));
            true
        },
        no_snapshot,
    );
    assert_eq!(fetched, 3);
    assert_eq!(out.num_accepted, 2);
    assert_eq!(out.bonus, Some(9));
}

#[test]
fn d2h_failure_finishes_sequence_without_bonus() {
    let mut a = think_seq();
    a.temperature = 0.7; // force the slow path
    let sup = [SUPPRESS];
    let out = dflash_thinking_accept(
        &mut a,
        &[1],
        &[1, 0],
        &ctx(&sup),
        |_, _: &mut Vec<u8>| false,
        no_snapshot,
    );
    assert_eq!(out.num_accepted, 0);
    assert!(out.bonus.is_none());
    assert!(a.finished);
    assert!(a.output_tokens.is_empty());
}
