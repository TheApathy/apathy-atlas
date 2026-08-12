// SPDX-License-Identifier: AGPL-3.0-only

//! Adaptive speculation (`ATLAS_DFLASH_ADAPTIVE=1`).
//!
//! Speculation pays only when committed-per-step exceeds
//! verify_step / serial_step. That ratio is now MEASURED per sequence (see
//! `break_even`), not assumed: the old constant 2.0 came from a γ=16 engine
//! at 222ms verify vs 63ms serial, and at γ=5 the real ratio is 3.80
//! (173.8ms vs 45.7ms, measured 2026-08-12) — so 2.0 let speculation run at
//! a LOSS across the whole 2.0..3.80 band. Measured 2026-07-08 on
//! coherent output: MinHeap code runs τ≈5.7 (spec wins, +30% vs serial),
//! Volvo prose runs τ≈2.3 (spec LOSES ~20% vs serial). Content decides.
//!
//! Policy: per-sequence rolling window of `accepted` over the last
//! [`WINDOW`] K=γ verify steps. Window full and mean below the threshold →
//! SUSPEND speculation for that sequence (no proposing; the scheduler's
//! bootstrap path serial-decodes it at full NOSPEC pace). After
//! `reprobe_tokens()` serial tokens, UN-suspend and re-probe: the window
//! must refill before suspension can re-trigger, so a probe costs WINDOW
//! spec steps (~2.7s) once per re-probe interval — a few percent on pure
//! prose, nothing on accepting content, and mixed documents (prose→code)
//! re-engage speculation automatically.
//!
//! Net posture: never materially slower than plain decode, +30% where
//! acceptance supports it. State is transient (reset on swap/restore —
//! a resumed sequence just re-measures).
//!
//! Knobs (env, read once): `ATLAS_DFLASH_ADAPTIVE=1` master switch;
//! `ATLAS_DFLASH_ADAPTIVE_MIN` OVERRIDES the measured break-even with a
//! fixed threshold (unset = measured, which is the default); `ATLAS_DFLASH_ADAPTIVE_REPROBE` serial tokens between probes
//! (default 256).

use crate::scheduler::ActiveSeq;

/// Rolling accept window + suspend state, embedded in [`ActiveSeq`].
#[derive(Default)]
pub(crate) struct AdaptState {
    window: Vec<u32>,
    suspended: bool,
    serial_tokens: u32,
    /// Rolling low-gear draft-accept counts while suspended
    /// (see [`record_low_gear`]).
    lg_window: Vec<u32>,
    /// Current engagement was triggered by low-gear accepts (not a probe).
    lg_from: bool,
    /// EWMA of wall time between consecutive γ-verify steps (ms) and between
    /// consecutive suspended serial steps (ms). Their ratio IS the break-even
    /// acceptance — see `break_even()`.
    ewma_spec_ms: f32,
    ewma_serial_ms: f32,
    last_step_at: Option<std::time::Instant>,
    /// Consecutive low-gear re-engagements that ended in re-suspension.
    /// Each failure raises the accept bar (`record_low_gear`) so borderline
    /// content (quote-like: n-gram accepts fire but the drafter hovers just
    /// under MIN) can't thrash engage→suspend→engage, paying the WINDOW of
    /// slow spec steps every cycle. Reset by a token-count probe or by an
    /// engagement that survives a full window.
    lg_fail_streak: u32,
}

const WINDOW: usize = 12;
/// Low-gear re-engage window: 8 low-gear steps (~9-16 committed tokens) —
/// short enough to catch a prose→code transition within a line of output.
const LG_WINDOW: usize = 8;

fn enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_DFLASH_ADAPTIVE").ok().as_deref() == Some("1"))
}

fn min_mean_accepted() -> f32 {
    static CACHED: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_ADAPTIVE_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2.0)
    })
}

fn reprobe_tokens() -> u32 {
    static CACHED: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_ADAPTIVE_REPROBE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                // With low gear on, its accept feedback (`record_low_gear`)
                // re-engages on content shifts, so the token-count probe is
                // only a backstop and can be rare. Measured on the γ=5 bench
                // serve: REPROBE 256 → 1024 lifted prose 19.75 → 20.58 tok/s
                // with repeat/quote/code unchanged. Without low gear the
                // probe is the ONLY re-engagement path — keep it at 256.
                if std::env::var("ATLAS_DFLASH_LOW_GEAR").ok().as_deref() == Some("1") {
                    1024
                } else {
                    256
                }
            })
    })
}

/// Mean low-gear accepts/step (over [`LG_WINDOW`] steps) at which suspension
/// lifts early. n-gram accepts fire on verbatim-repeat content — exactly
/// where the neural drafter wins — so they are a free re-engagement signal
/// that the token-count re-probe can only discover WINDOW slow spec steps
/// later. `ATLAS_DFLASH_LG_REENGAGE=0` disables; default 0.8 — measured
/// (2026-08-10, γ=5 bench serve): at 0.6 quote-like content thrashes
/// engage↔suspend and loses ~1 tok/s; at 0.8 repeat still re-engages
/// (n-gram accept ~0.9) while quote stays in low gear (~0.6-0.75).
fn lg_reengage_mean() -> f32 {
    static CACHED: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_LG_REENGAGE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.8)
    })
}

/// Record one suspended low-gear step's accepted-draft count (0 when the
/// n-gram guess was rejected). A sustained run of accepts means the content
/// turned structured: lift suspension so the full drafter re-engages now
/// instead of after the re-probe backstop. Only meaningful while suspended;
/// the window resets on every suspend/resume transition.
pub(crate) fn record_low_gear(a: &mut ActiveSeq, n_ok: usize) {
    if !enabled() || lg_reengage_mean() <= 0.0 {
        return;
    }
    let st = &mut a.spec_adapt;
    if !st.suspended {
        return;
    }
    st.lg_window.push(n_ok as u32);
    if st.lg_window.len() > LG_WINDOW {
        st.lg_window.remove(0);
    }
    if st.lg_window.len() == LG_WINDOW {
        let mean = st.lg_window.iter().sum::<u32>() as f32 / LG_WINDOW as f32;
        // Anti-thrash: each failed low-gear re-engagement raises the bar
        // (capped at a perfect window) so borderline content converges to
        // low-gear-only instead of oscillating.
        let bar = (lg_reengage_mean() + 0.15 * st.lg_fail_streak as f32).min(1.0);
        if mean >= bar {
            st.suspended = false;
            st.serial_tokens = 0;
            st.window.clear();
            st.lg_window.clear();
            st.lg_from = true;
            tracing::info!(
                "adaptive spec: RE-ENGAGED by low-gear accepts (mean {mean:.2} >= bar {bar:.2} \
                 over {LG_WINDOW} steps, fail_streak {})",
                st.lg_fail_streak,
            );
        }
    }
}

/// EWMA smoothing for the step-time estimates. 1/8 — fast enough to track a
/// content shift within a few steps, slow enough that one descheduled step
/// cannot move the suspend decision.
const EWMA_ALPHA: f32 = 0.125;

/// The measured break-even acceptance: speculation pays only when the tokens
/// committed per verify step exceed `verify_step / serial_step`, because that
/// many serial steps would have run in the same wall time.
///
/// This was a hardcoded 2.0, derived (per the module header) from a γ=16
/// engine at 222 ms verify vs 63 ms serial. That engine is gone. Measured
/// 2026-08-12 at γ=5: spec step 173.8 ms vs plain 45.7 ms ⇒ break-even 3.80,
/// while acceptance runs 3.39 — so the whole band 2.0..3.80 was speculating
/// AT A LOSS, violating this module's own stated contract ("never materially
/// slower than plain decode"). Deriving it from the two EWMAs makes the
/// policy correct at any γ, any quantization, any hardware — including the
/// EXL3 m-row verify path, which moves the ratio toward ~2.85 and should
/// therefore RE-ENABLE speculation the threshold would otherwise forbid.
///
/// Falls back to `min_mean_accepted()` until both estimates exist.
/// `ATLAS_DFLASH_ADAPTIVE_MIN` still overrides, for A/B.
fn break_even(st: &AdaptState) -> f32 {
    if std::env::var("ATLAS_DFLASH_ADAPTIVE_MIN").is_ok() {
        return min_mean_accepted();
    }
    if st.ewma_spec_ms > 0.0 && st.ewma_serial_ms > 0.0 {
        // The bonus token is already inside `accepted`, so compare directly.
        (st.ewma_spec_ms / st.ewma_serial_ms).clamp(1.0, 16.0)
    } else {
        min_mean_accepted()
    }
}

/// Fold one observed step duration into the right EWMA. `spec` selects which.
fn note_step(st: &mut AdaptState, spec: bool) {
    let now = std::time::Instant::now();
    if let Some(prev) = st.last_step_at {
        let ms = now.duration_since(prev).as_secs_f32() * 1000.0;
        // Guard against scheduler gaps (a queued request, a swap) polluting
        // the estimate: anything over 2 s is not a step.
        if ms > 0.0 && ms < 2000.0 {
            let slot = if spec { &mut st.ewma_spec_ms } else { &mut st.ewma_serial_ms };
            *slot = if *slot == 0.0 { ms } else { *slot + EWMA_ALPHA * (ms - *slot) };
        }
    }
    st.last_step_at = Some(now);
}

/// Record one K=γ verify step's accept count; may trip suspension.
/// Call after `num_accepted` is known (verify_dflash_step).
pub(crate) fn record_verify(a: &mut ActiveSeq, num_accepted: usize) {
    if !enabled() {
        return;
    }
    let st = &mut a.spec_adapt;
    note_step(st, true);
    st.window.push(num_accepted as u32);
    if st.window.len() > WINDOW {
        st.window.remove(0);
    }
    if st.window.len() == WINDOW {
        let mean = st.window.iter().sum::<u32>() as f32 / WINDOW as f32;
        let bar = break_even(st);
        if mean >= bar && st.lg_from {
            // The low-gear re-engagement proved out (a full window at
            // healthy accept): clear the failure streak.
            st.lg_from = false;
            st.lg_fail_streak = 0;
        }
        if mean < bar {
            st.suspended = true;
            st.serial_tokens = 0;
            st.window.clear();
            st.lg_window.clear();
            if st.lg_from {
                st.lg_from = false;
                st.lg_fail_streak += 1;
            }
            tracing::info!(
                "adaptive spec: SUSPENDED (mean accepted {mean:.2} < break-even {bar:.2} \
                 [spec {:.0}ms / serial {:.0}ms] over {WINDOW} steps) — serial until re-probe",
                st.ewma_spec_ms,
                st.ewma_serial_ms,
            );
        }
    }
}

/// May this sequence propose/speculate right now? Un-suspends (re-probe)
/// once enough serial tokens have passed.
pub(crate) fn spec_allowed(a: &mut ActiveSeq) -> bool {
    if !enabled() {
        return true;
    }
    let st = &mut a.spec_adapt;
    if !st.suspended {
        return true;
    }
    if st.serial_tokens >= reprobe_tokens() {
        st.suspended = false;
        st.serial_tokens = 0;
        st.window.clear();
        st.lg_window.clear();
        st.lg_from = false;
        st.lg_fail_streak = 0;
        tracing::info!(
            "adaptive spec: RE-PROBING after {} serial tokens",
            reprobe_tokens()
        );
        return true;
    }
    false
}

/// Is this sequence currently in the adaptive-suspended (serial) regime?
/// Read-only peek — unlike `spec_allowed`, never mutates re-probe state.
pub(crate) fn is_suspended(a: &ActiveSeq) -> bool {
    enabled() && a.spec_adapt.suspended
}

/// Ctx-holes fix master switch (`ATLAS_DFLASH_SERIAL_APPEND=1`): append
/// every serially-decoded token's captured target hidden into the DFlash
/// ctx accumulator — think-gated stretches, adaptive-suspended stretches,
/// and bootstrap tokens alike. Read once.
pub(crate) fn serial_append_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_DFLASH_SERIAL_APPEND").ok().as_deref() == Some("1"))
}

/// ATLAS_DFLASH_UNIFIED_CTX=1 → route the two commit points through the
/// single `commit_ctx` (hole-immune by construction, DDD §4.1) instead of
/// the ~5 fragmented appends. Default OFF = the 24.1 path, so both paths
/// A/B on ONE binary.
pub(crate) fn unified_ctx_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_DFLASH_UNIFIED_CTX").ok().as_deref() == Some("1"))
}

/// `ATLAS_DFLASH_EAGLE_FIX=1` (cached once). Hoisted here from raw per-step
/// `std::env::var` reads in `verify_dflash_step` (2026-07-25 host-overhead
/// pass) — same read-once semantics as every other gate in this module.
pub(crate) fn eagle_fix_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_DFLASH_EAGLE_FIX").ok().as_deref() == Some("1"))
}

/// Count a serially-decoded token toward the re-probe interval.
pub(crate) fn tick_serial(a: &mut ActiveSeq) {
    if enabled() && a.spec_adapt.suspended {
        // Time the serial step too — `break_even()` is the ratio of these two
        // EWMAs, so without this half the policy has no denominator and falls
        // back to the stale constant.
        note_step(&mut a.spec_adapt, false);
        a.spec_adapt.serial_tokens = a.spec_adapt.serial_tokens.saturating_add(1);
    }
}
