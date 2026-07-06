// SPDX-License-Identifier: AGPL-3.0-only

//! Decoding-efficiency levers for the `<think>` span (all default-OFF,
//! env-gated, OUTPUT-SHAPING — gated by eval quality, NOT by md5 parity).
//!
//! Three independent, orthogonal knobs, each inert unless its env var is set:
//!
//! 1. **Hesitation penalty** (`ATLAS_HESITATION_PENALTY=<float>`): additive
//!    logit penalty applied — during the thinking span only — to a ~50-token
//!    "hesitation" set (`wait` / `but` / `alternatively` / `however` /
//!    `actually` / `hmm` / `let me reconsider`-class, incl. leading-space and
//!    capitalization variants). Extends the existing F1 reflection-suppress
//!    idea from a hardcoded `-10.0` mask to a tunable additive penalty over a
//!    much larger, tokenizer-derived id set. arXiv:2606.00206.
//!
//! 2. **Soft `</think>` exit bias** (`ATLAS_THINK_EXIT_BIAS=<max>`): a positive
//!    additive bias on the `</think>` token that grows LINEARLY from 0 at
//!    `ATLAS_THINK_SOFT_START` thinking tokens to `<max>` at the hard budget —
//!    a soft landing that nudges the model toward closing thought as it
//!    approaches the cap, replacing the blunt logit-truncation the budget cap
//!    performs today.
//!
//! 3. **Adaptive thinking budget** (`ATLAS_ADAPTIVE_THINK=1`): scales the
//!    effective thinking budget by a measured difficulty signal (see
//!    [`adaptive_budget`]). Easy prompts get a shorter budget; hard prompts
//!    get a longer one.
//!
//! # Why these are lossless with the flags OFF
//!
//! Every code path here is behind a `Config::is_active()` / per-field
//! `Option` guard that is `None`/false when the env var is unset. With all
//! three vars unset the module installs no id set and every apply-site is a
//! no-op, so the committed token stream — and thus the counting-eval md5 — is
//! byte-identical to before. Only when a var is set does behaviour change, and
//! then it is intentionally OUTPUT-SHAPING (fewer thinking tokens), validated
//! by eval quality, not md5.

use std::sync::OnceLock;

// ── Hesitation vocabulary ────────────────────────────────────────────────────
//
// The base "hesitation" words. At startup we expand each to its
// leading-space and capitalization variants and resolve every variant that
// the tokenizer encodes as a SINGLE token id (multi-token phrases can't be
// penalised at a single logit position, so they are skipped — the penalty is
// a per-token logit bias, not a sequence constraint). This mirrors the
// single-token filter in `tokenizer_runtime::reflection_suppress_ids` and the
// per-id table build in `cfg_jump_forward::build_delim_table`.
//
// Kept deliberately broad (~14 stems × 4 case/space variants ≈ 50 candidate
// strings) so the resolved id set lands near the ~50-token target from the
// mission even after the single-token filter drops multi-token phrases.
const HESITATION_STEMS: &[&str] = &[
    "wait",
    "but",
    "alternatively",
    "however",
    "actually",
    "hmm",
    "hold on",
    "let me reconsider",
    "on second thought",
    "maybe",
    "perhaps",
    "although",
    "though",
    "reconsider",
];

/// Expand a base stem into the surface variants a BPE tokenizer is likely to
/// carry as distinct ids: bare, leading-space, capitalized, and
/// leading-space+capitalized. Deduplicated by the caller via the resolved id
/// set (two variants that map to the same id are naturally merged).
///
/// Pure function — unit-tested without a tokenizer.
pub fn hesitation_variants(stem: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(4);
    let cap = capitalize_first(stem);
    out.push(stem.to_string());
    out.push(format!(" {stem}"));
    if cap != stem {
        out.push(cap.clone());
        out.push(format!(" {cap}"));
    }
    out
}

/// Capitalize the first character (ASCII), leaving the rest untouched.
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_ascii_uppercase().to_string() + chars.as_str(),
        None => String::new(),
    }
}

/// Build the hesitation token-id set from the tokenizer by encoding every
/// variant of every stem and keeping only those that encode to exactly ONE
/// token id. `encode` is the tokenizer's `encode` (no special tokens). Returns
/// a sorted, deduplicated id list.
///
/// Fail-open per variant: an encode error simply drops that variant.
pub fn build_hesitation_ids<F>(mut encode: F) -> Vec<u32>
where
    F: FnMut(&str) -> Option<Vec<u32>>,
{
    let mut ids: Vec<u32> = Vec::new();
    for stem in HESITATION_STEMS {
        for variant in hesitation_variants(stem) {
            if let Some(toks) = encode(&variant)
                && toks.len() == 1
            {
                ids.push(toks[0]);
            }
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

// ── Configuration (env-parsed once at startup) ───────────────────────────────

/// Resolved thinking-efficiency configuration. All fields `None`/empty ⇒ inert.
#[derive(Clone, Debug, Default)]
pub struct ThinkEfficiencyConfig {
    /// Additive logit penalty (subtracted) applied to each hesitation id during
    /// the thinking span. `None` ⇒ hesitation penalty OFF. The set is empty when
    /// the penalty is off, so the apply site skips work.
    pub hesitation_penalty: Option<f32>,
    /// Hesitation token ids (populated only when `hesitation_penalty.is_some()`).
    pub hesitation_ids: Vec<u32>,
    /// Max `</think>` exit bias at the hard budget. `None` ⇒ exit-bias OFF.
    pub think_exit_bias: Option<f32>,
    /// Thinking-token count at which the exit-bias ramp starts (bias = 0 here,
    /// growing linearly to `think_exit_bias` at the budget). Default 0.
    pub think_soft_start: u32,
    /// Adaptive thinking budget enabled (`ATLAS_ADAPTIVE_THINK=1`).
    pub adaptive_think: bool,
}

impl ThinkEfficiencyConfig {
    /// Any lever active? When false the whole module is a no-op and callers can
    /// skip even the per-seq field reads.
    pub fn is_active(&self) -> bool {
        self.hesitation_penalty.is_some() || self.think_exit_bias.is_some() || self.adaptive_think
    }
}

/// Pure parse of the raw env values into a [`ThinkEfficiencyConfig`]. Split out
/// so the parsing rules are unit-testable without touching the process-wide
/// `OnceLock` or a real tokenizer. `hesitation_ids` is populated separately by
/// the installer (needs the tokenizer); this fills only the scalar knobs.
///
/// * `penalty`: `ATLAS_HESITATION_PENALTY` — parsed as f32; unset/invalid ⇒
///   `None` (OFF). A value of `0.0` is treated as OFF (no-op penalty).
/// * `exit_bias`: `ATLAS_THINK_EXIT_BIAS` — parsed as f32; unset/invalid/≤0 ⇒
///   `None` (OFF).
/// * `soft_start`: `ATLAS_THINK_SOFT_START` — parsed as u32; unset/invalid ⇒ 0.
/// * `adaptive`: `ATLAS_ADAPTIVE_THINK` — `"1"`/`"true"` ⇒ true, else false.
pub fn parse_config(
    penalty: Option<&str>,
    exit_bias: Option<&str>,
    soft_start: Option<&str>,
    adaptive: Option<&str>,
) -> ThinkEfficiencyConfig {
    let hesitation_penalty = penalty
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|p| p.is_finite() && *p != 0.0);
    let think_exit_bias = exit_bias
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|b| b.is_finite() && *b > 0.0);
    let think_soft_start = soft_start
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(0);
    let adaptive_think = matches!(
        adaptive.map(|s| s.trim()),
        Some("1") | Some("true") | Some("TRUE") | Some("True")
    );
    ThinkEfficiencyConfig {
        hesitation_penalty,
        hesitation_ids: Vec::new(),
        think_exit_bias,
        think_soft_start,
        adaptive_think,
    }
}

static CONFIG: OnceLock<ThinkEfficiencyConfig> = OnceLock::new();

/// Install the resolved config once at startup. Idempotent (first writer wins).
pub fn set_think_efficiency_config(cfg: ThinkEfficiencyConfig) {
    let _ = CONFIG.set(cfg);
}

/// Read the installed config. Returns a reference to a process-wide default
/// (all-off) until `set_think_efficiency_config` runs — so any pre-boot caller
/// and every unit test sees the inert configuration.
pub fn think_efficiency_config() -> &'static ThinkEfficiencyConfig {
    static DEFAULT: ThinkEfficiencyConfig = ThinkEfficiencyConfig {
        hesitation_penalty: None,
        hesitation_ids: Vec::new(),
        think_exit_bias: None,
        think_soft_start: 0,
        adaptive_think: false,
    };
    CONFIG.get().unwrap_or(&DEFAULT)
}

// ── Lever 1+2: logit shaping during the thinking span ────────────────────────

/// Apply the hesitation penalty (Lever 1) and the progressive `</think>` exit
/// bias (Lever 2) to a thinking-span logit vector, in place.
///
/// * `hesitation_ids` / `penalty`: subtract `penalty` from each id's logit
///   (only when `penalty.is_some()`).
/// * `think_end_id` / `exit_bias`: add [`exit_bias_at`]`(...)` to the `</think>`
///   logit (only when `exit_bias.is_some()`).
///
/// `thinking_tokens` is the count so far this span; `budget` is the (possibly
/// adaptive-scaled) hard cap used as the ramp end. Returns the number of logit
/// slots actually modified, for the stats log. Pure aside from the in-place
/// mutation of `logits`.
#[allow(clippy::too_many_arguments)]
pub fn apply_think_logit_shaping(
    logits: &mut [f32],
    cfg: &ThinkEfficiencyConfig,
    think_end_id: Option<u32>,
    thinking_tokens: u32,
    budget: Option<u32>,
) -> u32 {
    let mut touched = 0u32;

    if let Some(penalty) = cfg.hesitation_penalty {
        for &id in &cfg.hesitation_ids {
            if let Some(slot) = logits.get_mut(id as usize) {
                *slot -= penalty;
                touched += 1;
            }
        }
    }

    if let (Some(max_bias), Some(end_id)) = (cfg.think_exit_bias, think_end_id) {
        let bias = exit_bias_at(thinking_tokens, cfg.think_soft_start, budget, max_bias);
        if bias != 0.0
            && let Some(slot) = logits.get_mut(end_id as usize)
        {
            *slot += bias;
            touched += 1;
        }
    }

    touched
}

/// Linear ramp for the `</think>` exit bias: 0 at `soft_start` thinking tokens,
/// growing to `max_bias` at `budget`. Clamped to `[0, max_bias]`.
///
/// * before `soft_start`: 0 (no nudge — let the model think freely early).
/// * `budget == None` (unbounded thinking): no ramp end, returns 0 (the exit
///   bias needs a finite endpoint; unbounded budgets fall back to the other
///   watchdogs).
/// * `budget <= soft_start`: degenerate window ⇒ full `max_bias` once at/after
///   `soft_start` (nothing to ramp over).
///
/// Pure function — unit-tested directly.
pub fn exit_bias_at(
    thinking_tokens: u32,
    soft_start: u32,
    budget: Option<u32>,
    max_bias: f32,
) -> f32 {
    let Some(budget) = budget else {
        return 0.0;
    };
    if thinking_tokens <= soft_start {
        return 0.0;
    }
    if budget <= soft_start {
        // No room to ramp — we are already past soft_start, apply full bias.
        return max_bias;
    }
    if thinking_tokens >= budget {
        return max_bias;
    }
    let span = (budget - soft_start) as f32;
    let progress = (thinking_tokens - soft_start) as f32;
    max_bias * (progress / span)
}

/// Top-1 softmax confidence of a logit vector, in `[0, 1]`. Used as the
/// per-token difficulty signal for Lever 3's [`DifficultyProbe`]. A single
/// numerically-stable pass (max-shift then sum-exp); returns 0 for an empty or
/// all-`-inf` vector. Pure — unit-tested directly.
pub fn top1_confidence(logits: &[f32]) -> f32 {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_logit.is_finite() {
        return 0.0;
    }
    let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    if sum_exp > 0.0 {
        // exp(max - max) == 1, so top-1 prob == 1 / sum_exp.
        (1.0 / sum_exp).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

// ── Lever 3: adaptive thinking budget ────────────────────────────────────────

/// Fraction of thinking tokens over which the difficulty signal is measured
/// before the adaptive budget is committed. Matches the mission's "first ~48
/// thinking tokens" window.
pub const ADAPTIVE_PROBE_TOKENS: u32 = 48;

/// Multiplier applied to the base budget when the prompt looks EASY (high
/// difficulty-signal confidence / high drafter accept-rate).
pub const ADAPTIVE_EASY_SCALE: f32 = 0.4;
/// Multiplier applied to the base budget when the prompt looks HARD.
pub const ADAPTIVE_HARD_SCALE: f32 = 1.5;
/// Absolute floor for the adaptive budget (keeps the watchdog functional).
pub const ADAPTIVE_MIN_BUDGET: u32 = 32;

/// Scale a base thinking budget by a measured difficulty signal in `[0, 1]`,
/// where HIGHER = EASIER (e.g. mean top-1 confidence over the probe window, or
/// a normalized drafter accept-rate). Linearly interpolates the multiplier
/// between [`ADAPTIVE_HARD_SCALE`] (signal 0, hardest) and
/// [`ADAPTIVE_EASY_SCALE`] (signal 1, easiest), then clamps to
/// `[ADAPTIVE_MIN_BUDGET, base]`.
///
/// Never grows the budget past `base` (the request/MODEL.toml cap is a hard
/// ceiling — adaptivity only *shortens* within it, or lengthens up to it for
/// hard prompts). Pure function — unit-tested directly.
pub fn adaptive_budget(base: u32, difficulty_signal: f32) -> u32 {
    let s = difficulty_signal.clamp(0.0, 1.0);
    // signal 0 (hard) → HARD_SCALE ; signal 1 (easy) → EASY_SCALE.
    let scale = ADAPTIVE_HARD_SCALE + (ADAPTIVE_EASY_SCALE - ADAPTIVE_HARD_SCALE) * s;
    let scaled = (base as f32 * scale).round() as i64;
    let ceiling = base as i64;
    scaled.clamp(
        ADAPTIVE_MIN_BUDGET as i64,
        ceiling.max(ADAPTIVE_MIN_BUDGET as i64),
    ) as u32
}

// ── Per-sequence difficulty accumulator (Lever 3 signal) ─────────────────────

/// Running difficulty probe over the first [`ADAPTIVE_PROBE_TOKENS`] thinking
/// tokens. Accumulates the per-token top-1 confidence (the only difficulty
/// signal available *during* the thinking span — speculative decode, and thus
/// the drafter accept-rate, is bypassed while `inside_thinking`; see
/// `verify_dflash_step::dflash_masked_accept` and the `!inside_thinking` gate
/// in `scheduler::run`'s MTP dispatch). High mean confidence ⇒ the model finds
/// the reasoning easy ⇒ shorter budget.
///
/// Default (`Default::default()`) is an empty probe; `committed` stays false
/// until the window fills, at which point [`commit`](Self::commit) returns the
/// scaled budget exactly once.
#[derive(Clone, Debug, Default)]
pub struct DifficultyProbe {
    /// Sum of per-token top-1 confidence over the observed window.
    conf_sum: f32,
    /// Count of thinking tokens observed so far (capped at the probe window).
    observed: u32,
    /// Set once the adaptive budget has been committed, so it fires only once.
    committed: bool,
}

impl DifficultyProbe {
    /// Record one thinking-token's top-1 softmax confidence in `[0, 1]`.
    /// Ignored once the probe window is full or the budget already committed.
    pub fn observe(&mut self, top1_conf: f32) {
        if self.committed || self.observed >= ADAPTIVE_PROBE_TOKENS {
            return;
        }
        self.conf_sum += top1_conf.clamp(0.0, 1.0);
        self.observed += 1;
    }

    /// True once the probe window has filled and a budget can be committed.
    pub fn ready(&self) -> bool {
        !self.committed && self.observed >= ADAPTIVE_PROBE_TOKENS
    }

    /// Mean observed confidence (the difficulty signal, higher = easier), or
    /// `None` if nothing observed yet.
    pub fn mean_confidence(&self) -> Option<f32> {
        if self.observed == 0 {
            None
        } else {
            Some(self.conf_sum / self.observed as f32)
        }
    }

    /// Commit the adaptive budget from `base` using the accumulated signal.
    /// Returns `Some(scaled_budget)` exactly once (the first time the window is
    /// full); subsequent calls return `None`. No-op signal (`observed == 0`)
    /// returns `None` and leaves the probe open.
    pub fn commit(&mut self, base: u32) -> Option<u32> {
        if self.committed {
            return None;
        }
        let signal = self.mean_confidence()?;
        if self.observed < ADAPTIVE_PROBE_TOKENS {
            return None;
        }
        self.committed = true;
        Some(adaptive_budget(base, signal))
    }
}

#[cfg(test)]
#[path = "thinking_efficiency_tests.rs"]
mod thinking_efficiency_tests;
