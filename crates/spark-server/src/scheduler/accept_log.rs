// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_DSPARK_ACCEPT_LOG=1` — periodic speculative-accept histogram.
//!
//! Reports the DISTRIBUTION of accepted-tokens-per-verify-step, not just the
//! mean, plus the per-token draft acceptance rate that the reference engines
//! quote (`accept% = accepted / drafted`).
//!
//! Why a histogram: the reference (Entrpi/ds4-on-spark) measures 3.08
//! tokens/step suite mean — 3.38–4.00 at 80–89% acceptance on code, 2.18 at
//! 58% on adversarial prose — while our online figure sits near 1.0 even
//! though the offline engine probe reaches 3.79 tok/step on the SAME drafter.
//! A mean near 1 is consistent with two very different worlds: a drafter whose
//! every first draft is wrong, or a bimodal one that either nails a whole
//! block or whiffs immediately. Those have different root causes and different
//! fixes, so the shape has to be measured before anything is changed.
//!
//! Zero cost when the env is unset (one cached bool, then an early return).

use std::sync::atomic::{AtomicU64, Ordering};

/// Verify steps per emitted summary.
const PERIOD: u64 = 64;
/// Histogram buckets: index i counts steps that committed exactly i tokens.
/// Sized for the largest γ we run plus the bonus token.
const MAX_BUCKET: usize = 17;

static STEPS: AtomicU64 = AtomicU64::new(0);
static COMMITTED: AtomicU64 = AtomicU64::new(0);
static DRAFTED: AtomicU64 = AtomicU64::new(0);
static ACCEPTED: AtomicU64 = AtomicU64::new(0);
static ZERO_STEPS: AtomicU64 = AtomicU64::new(0);
static BUCKETS: [AtomicU64; MAX_BUCKET] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const Z: AtomicU64 = AtomicU64::new(0);
    [Z; MAX_BUCKET]
};

fn enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_DSPARK_ACCEPT_LOG").as_deref() == Ok("1"))
}

/// Record one verify step. `accepted` is how many DRAFTS the verify kept —
/// NOT including the always-correct bonus token the step also emits, so a
/// step delivers `accepted + 1` tokens. `drafted` is how many drafts the
/// proposer offered.
pub(crate) fn record(accepted: usize, drafted: usize) {
    if !enabled() {
        return;
    }
    let committed = accepted + 1; // + the bonus token every verify step emits
    COMMITTED.fetch_add(committed as u64, Ordering::Relaxed);
    DRAFTED.fetch_add(drafted as u64, Ordering::Relaxed);
    ACCEPTED.fetch_add(accepted as u64, Ordering::Relaxed);
    BUCKETS[accepted.min(MAX_BUCKET - 1)].fetch_add(1, Ordering::Relaxed);
    if accepted == 0 {
        ZERO_STEPS.fetch_add(1, Ordering::Relaxed);
    }

    let n = STEPS.fetch_add(1, Ordering::Relaxed) + 1;
    if !n.is_multiple_of(PERIOD) {
        return;
    }
    let steps = n as f64;
    let committed_tot = COMMITTED.load(Ordering::Relaxed) as f64;
    let drafted_tot = DRAFTED.load(Ordering::Relaxed).max(1) as f64;
    let accepted_tot = ACCEPTED.load(Ordering::Relaxed) as f64;
    let mut hist = String::new();
    for (i, b) in BUCKETS.iter().enumerate() {
        let c = b.load(Ordering::Relaxed);
        if c > 0 {
            hist.push_str(&format!(" {i}:{c}"));
        }
    }
    tracing::info!(
        "DSPARK accept: {:.2} tok/step over {} steps | draft accept {:.1}% \
         ({:.0}/{:.0}) | zero-accept steps {:.1}% | histogram(accepted:steps){}",
        committed_tot / steps,
        n,
        100.0 * accepted_tot / drafted_tot,
        accepted_tot,
        drafted_tot,
        100.0 * ZERO_STEPS.load(Ordering::Relaxed) as f64 / steps,
        hist,
    );
}
