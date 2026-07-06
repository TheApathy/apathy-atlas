// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for `thinking_efficiency.rs` (hesitation penalty + soft
//! `</think>` exit bias + adaptive thinking budget). Pure functions — no
//! tokenizer, no `ActiveSeq`, no logits device buffers. Logical child of
//! `thinking_efficiency` via `#[path]`; `use super::*` resolves to that
//! module's items.

use super::*;

// ── Hesitation variant expansion + id-set builder ────────────────────────────

#[test]
fn variants_include_space_and_capitalization() {
    let v = hesitation_variants("wait");
    assert!(v.contains(&"wait".to_string()));
    assert!(v.contains(&" wait".to_string()));
    assert!(v.contains(&"Wait".to_string()));
    assert!(v.contains(&" Wait".to_string()));
    assert_eq!(v.len(), 4);
}

#[test]
fn variants_of_capitalized_first_char_only() {
    // Multi-word stems capitalize only the first char (a real tokenizer
    // rarely carries a fully-capitalized phrase as one id anyway).
    let v = hesitation_variants("let me reconsider");
    assert!(v.contains(&"Let me reconsider".to_string()));
    assert!(v.contains(&" Let me reconsider".to_string()));
}

#[test]
fn build_ids_keeps_only_single_token_variants() {
    // Fake tokenizer: "wait" and " wait" are single tokens; "however" splits
    // into two; everything else unknown → None.
    let ids = build_hesitation_ids(|s| match s {
        "wait" => Some(vec![10]),
        " wait" => Some(vec![11]),
        "Wait" => Some(vec![10]), // duplicate id — dedup should merge
        "however" => Some(vec![20, 21]), // multi-token → dropped
        _ => None,
    });
    assert_eq!(ids, vec![10, 11], "only single-token variants, deduped");
}

#[test]
fn build_ids_empty_when_tokenizer_knows_nothing() {
    let ids = build_hesitation_ids(|_| None);
    assert!(ids.is_empty());
}

// ── Env config parsing ───────────────────────────────────────────────────────

#[test]
fn parse_all_unset_is_inert() {
    let cfg = parse_config(None, None, None, None);
    assert!(cfg.hesitation_penalty.is_none());
    assert!(cfg.think_exit_bias.is_none());
    assert_eq!(cfg.think_soft_start, 0);
    assert!(!cfg.adaptive_think);
    assert!(!cfg.is_active());
}

#[test]
fn parse_penalty_zero_is_off() {
    // A 0.0 penalty is a no-op — treat as OFF so is_active() stays false.
    let cfg = parse_config(Some("0.0"), None, None, None);
    assert!(cfg.hesitation_penalty.is_none());
    assert!(!cfg.is_active());
}

#[test]
fn parse_penalty_valid() {
    let cfg = parse_config(Some("2.5"), None, None, None);
    assert_eq!(cfg.hesitation_penalty, Some(2.5));
    assert!(cfg.is_active());
}

#[test]
fn parse_exit_bias_must_be_positive() {
    assert!(
        parse_config(None, Some("-1.0"), None, None)
            .think_exit_bias
            .is_none()
    );
    assert!(
        parse_config(None, Some("0"), None, None)
            .think_exit_bias
            .is_none()
    );
    assert_eq!(
        parse_config(None, Some("5.0"), None, None).think_exit_bias,
        Some(5.0)
    );
}

#[test]
fn parse_soft_start_and_adaptive() {
    let cfg = parse_config(None, Some("4.0"), Some("128"), Some("1"));
    assert_eq!(cfg.think_soft_start, 128);
    assert!(cfg.adaptive_think);
    assert!(cfg.is_active());
    assert!(parse_config(None, None, None, Some("true")).adaptive_think);
    assert!(!parse_config(None, None, None, Some("0")).adaptive_think);
    assert!(!parse_config(None, None, None, Some("no")).adaptive_think);
}

#[test]
fn parse_junk_values_fall_back() {
    let cfg = parse_config(Some("abc"), Some("xyz"), Some("nan"), Some("maybe"));
    assert!(cfg.hesitation_penalty.is_none());
    assert!(cfg.think_exit_bias.is_none());
    assert_eq!(cfg.think_soft_start, 0);
    assert!(!cfg.adaptive_think);
}

// ── Exit-bias ramp math ──────────────────────────────────────────────────────

#[test]
fn exit_bias_zero_before_soft_start() {
    assert_eq!(exit_bias_at(0, 100, Some(500), 8.0), 0.0);
    assert_eq!(exit_bias_at(100, 100, Some(500), 8.0), 0.0);
}

#[test]
fn exit_bias_full_at_budget() {
    assert_eq!(exit_bias_at(500, 100, Some(500), 8.0), 8.0);
    assert_eq!(exit_bias_at(600, 100, Some(500), 8.0), 8.0); // past budget → clamp
}

#[test]
fn exit_bias_linear_midpoint() {
    // soft_start=100, budget=500 → span 400. At 300 tokens, progress=200 →
    // half the ramp → 4.0.
    let b = exit_bias_at(300, 100, Some(500), 8.0);
    assert!((b - 4.0).abs() < 1e-4, "got {b}");
}

#[test]
fn exit_bias_zero_when_budget_none() {
    assert_eq!(exit_bias_at(1000, 0, None, 8.0), 0.0);
}

#[test]
fn exit_bias_degenerate_window() {
    // budget <= soft_start: no room to ramp; full bias once past soft_start.
    assert_eq!(exit_bias_at(50, 100, Some(80), 8.0), 0.0); // still before soft_start
    assert_eq!(exit_bias_at(120, 100, Some(80), 8.0), 8.0); // past soft_start, degenerate
}

#[test]
fn exit_bias_monotonic_nondecreasing() {
    let mut prev = -1.0;
    for t in 0..600u32 {
        let b = exit_bias_at(t, 100, Some(500), 8.0);
        assert!(b >= prev - 1e-6, "ramp regressed at t={t}: {b} < {prev}");
        assert!((0.0..=8.0).contains(&b));
        prev = b;
    }
}

// ── Adaptive budget scaling ──────────────────────────────────────────────────

#[test]
fn adaptive_easy_shrinks() {
    // signal 1.0 (easiest) → EASY_SCALE 0.4.
    let b = adaptive_budget(1000, 1.0);
    assert_eq!(b, 400);
}

#[test]
fn adaptive_hard_grows_to_ceiling() {
    // signal 0.0 (hardest) → HARD_SCALE 1.5, but capped at base (never exceeds).
    let b = adaptive_budget(1000, 0.0);
    assert_eq!(b, 1000, "hard scale clamps to base ceiling");
}

#[test]
fn adaptive_midpoint() {
    // signal 0.5 → scale = 1.5 + (0.4-1.5)*0.5 = 0.95 → 950.
    let b = adaptive_budget(1000, 0.5);
    assert_eq!(b, 950);
}

#[test]
fn adaptive_respects_floor() {
    // Tiny base with easy signal must not drop below ADAPTIVE_MIN_BUDGET.
    let b = adaptive_budget(40, 1.0); // 40*0.4=16 < 32 floor
    assert_eq!(b, ADAPTIVE_MIN_BUDGET);
}

#[test]
fn adaptive_signal_clamped() {
    // Out-of-range signals clamp — no panic, no overshoot.
    assert_eq!(adaptive_budget(1000, 5.0), adaptive_budget(1000, 1.0));
    assert_eq!(adaptive_budget(1000, -5.0), adaptive_budget(1000, 0.0));
}

// ── Logit shaping apply-site ─────────────────────────────────────────────────

#[test]
fn shaping_noop_when_inactive() {
    let cfg = ThinkEfficiencyConfig::default();
    let mut logits = vec![0.0f32; 100];
    let touched = apply_think_logit_shaping(&mut logits, &cfg, Some(5), 10, Some(500));
    assert_eq!(touched, 0);
    assert!(logits.iter().all(|&l| l == 0.0));
}

#[test]
fn shaping_applies_hesitation_penalty() {
    let cfg = ThinkEfficiencyConfig {
        hesitation_penalty: Some(3.0),
        hesitation_ids: vec![1, 2, 99],
        ..Default::default()
    };
    let mut logits = vec![0.0f32; 100];
    let touched = apply_think_logit_shaping(&mut logits, &cfg, None, 0, None);
    assert_eq!(touched, 3);
    assert_eq!(logits[1], -3.0);
    assert_eq!(logits[2], -3.0);
    assert_eq!(logits[99], -3.0);
    assert_eq!(logits[0], 0.0);
}

#[test]
fn shaping_out_of_range_id_skipped() {
    let cfg = ThinkEfficiencyConfig {
        hesitation_penalty: Some(3.0),
        hesitation_ids: vec![1, 999], // 999 >= len → skipped, no panic
        ..Default::default()
    };
    let mut logits = vec![0.0f32; 100];
    let touched = apply_think_logit_shaping(&mut logits, &cfg, None, 0, None);
    assert_eq!(touched, 1);
    assert_eq!(logits[1], -3.0);
}

#[test]
fn shaping_applies_exit_bias_at_budget() {
    let cfg = ThinkEfficiencyConfig {
        think_exit_bias: Some(6.0),
        think_soft_start: 100,
        ..Default::default()
    };
    let mut logits = vec![0.0f32; 100];
    // At budget → full bias on think_end id 7.
    let touched = apply_think_logit_shaping(&mut logits, &cfg, Some(7), 500, Some(500));
    assert_eq!(touched, 1);
    assert_eq!(logits[7], 6.0);
}

#[test]
fn shaping_exit_bias_needs_think_end_id() {
    let cfg = ThinkEfficiencyConfig {
        think_exit_bias: Some(6.0),
        ..Default::default()
    };
    let mut logits = vec![0.0f32; 100];
    // No think_end id → nothing to bias.
    let touched = apply_think_logit_shaping(&mut logits, &cfg, None, 500, Some(500));
    assert_eq!(touched, 0);
}

// ── top1_confidence ──────────────────────────────────────────────────────────

#[test]
fn top1_uniform_is_low() {
    // Uniform over N → top-1 prob ~1/N.
    let logits = vec![0.0f32; 10];
    let c = top1_confidence(&logits);
    assert!((c - 0.1).abs() < 1e-4, "got {c}");
}

#[test]
fn top1_peaked_is_high() {
    let mut logits = vec![0.0f32; 100];
    logits[42] = 50.0; // massively dominant → prob ≈ 1.
    let c = top1_confidence(&logits);
    assert!(c > 0.999, "got {c}");
}

#[test]
fn top1_empty_and_neg_inf() {
    assert_eq!(top1_confidence(&[]), 0.0);
    assert_eq!(
        top1_confidence(&[f32::NEG_INFINITY, f32::NEG_INFINITY]),
        0.0
    );
}

// ── DifficultyProbe accumulator (Lever 3 signal) ─────────────────────────────

#[test]
fn probe_not_ready_before_window_fills() {
    let mut p = DifficultyProbe::default();
    for _ in 0..(ADAPTIVE_PROBE_TOKENS - 1) {
        p.observe(0.9);
    }
    assert!(!p.ready());
    assert!(p.commit(1000).is_none());
}

#[test]
fn probe_commits_once_when_full() {
    let mut p = DifficultyProbe::default();
    for _ in 0..ADAPTIVE_PROBE_TOKENS {
        p.observe(1.0); // maximally easy
    }
    assert!(p.ready());
    let b = p.commit(1000);
    assert_eq!(b, Some(400)); // easy → 0.4x
    // Second commit is a no-op.
    assert!(p.commit(1000).is_none());
    assert!(!p.ready());
}

#[test]
fn probe_mean_confidence_averages() {
    let mut p = DifficultyProbe::default();
    for _ in 0..(ADAPTIVE_PROBE_TOKENS / 2) {
        p.observe(1.0);
    }
    for _ in 0..(ADAPTIVE_PROBE_TOKENS / 2) {
        p.observe(0.0);
    }
    let m = p.mean_confidence().unwrap();
    assert!((m - 0.5).abs() < 1e-4, "got {m}");
}

#[test]
fn probe_ignores_observations_past_window() {
    let mut p = DifficultyProbe::default();
    for _ in 0..(ADAPTIVE_PROBE_TOKENS + 20) {
        p.observe(1.0);
    }
    assert_eq!(p.mean_confidence(), Some(1.0));
    // Only ADAPTIVE_PROBE_TOKENS were counted (no divide drift).
}

#[test]
fn probe_hard_prompt_grows_budget() {
    let mut p = DifficultyProbe::default();
    for _ in 0..ADAPTIVE_PROBE_TOKENS {
        p.observe(0.0); // maximally hard
    }
    // Hard → 1.5x but clamped to base ceiling.
    assert_eq!(p.commit(1000), Some(1000));
}
