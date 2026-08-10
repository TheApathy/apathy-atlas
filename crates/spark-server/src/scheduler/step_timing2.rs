// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_STEP_TIMING2=1` — high-resolution HOST-side per-step breakdown for
//! the single-stream DFlash decode path (the ~9 ms/step "outside the two
//! timed sections" hunt, 2026-07).
//!
//! The existing `ATLAS_DFLASH_STEP_TIMING` split (verify wall + propose wall)
//! accounts for ~108.6 ms of a ~117-119 ms observed step. This module times
//! everything else on the scheduler thread:
//!
//! * `loop_gap` — previous DFlash step exit → this step entry. Covers the
//!   whole scheduler-loop tail + head: retire/swap checks, pending-queue
//!   mutex, mtp_gate bookkeeping, Phase A/B dispatch, grammar draft
//!   truncation, and any `mtp_gate` MeasureDecode probe steps taken between
//!   two DFlash verify steps (those serial probes are otherwise invisible to
//!   the STEP_TIMING buckets and drag the tok/s average).
//! * `collect` — `dflash_collect_async_drafts` (event sync + pinned read) at
//!   the top of Phase A/B. A subset of `loop_gap` (recorded separately, not
//!   additive with it).
//! * `pre` — step entry → verify launch: `sync_secondary`, token staging,
//!   fork/tree payload staging.
//! * `verify` — `decode_verify_dflash` wall (host prep + graph + blocking
//!   D2H sync tail; same span as the existing STEP_TIMING verify bucket).
//! * `walk` — verify return → emit start: tree walk / pick pipeline /
//!   accept walk / rollback / spec-propose resolution / drafter-ctx commit.
//! * `emit` — the `emit_token` loop (accepted drafts + bonus + B-win tail),
//!   including the stream-channel `try_send`s. Detokenize + SSE encoding run
//!   on the tokio HTTP task, NOT here — only channel handoff is on this
//!   thread (emit_step.rs `send_stream_event`; blocks only on backpressure).
//! * `commit` — emit end → propose launch: echo/recycle stashes, metrics,
//!   logging, `commit_accepted_prefix`, `save_hidden_for_mtp`,
//!   `trim_proposer_state`, grammar mask build.
//! * `propose` — `run_mtp_propose_multi` wall (same span as STEP_TIMING).
//! * `TOTAL` — step entry → step exit; `other` in the summary is
//!   `TOTAL − (pre+verify+walk+emit+commit+propose)` (early-return paths —
//!   spec-adopt, tree-win, finished-mid-emit — leave later buckets unmarked;
//!   their time lands in `other`).
//!
//! One `info!` summary every [`SUMMARY_PERIOD`] DFlash steps, then the
//! accumulators reset. Zero behavioral effect; near-zero cost when the env
//! is unset (`enabled()` is a cached bool; every entry point early-returns).
//!
//! Single-writer by construction: only the scheduler thread calls in. The
//! `Mutex<Clock>` is uncontended (parking_lot fast path, ~20 ns).

use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// DFlash steps per summary line.
const SUMMARY_PERIOD: u64 = 64;

/// Timed buckets. `StepTotal` must stay last (it sizes the arrays).
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub(crate) enum Phase2 {
    LoopGap = 0,
    Collect,
    Pre,
    Verify,
    Walk,
    Emit,
    Commit,
    Propose,
    StepTotal,
}

const NUM_PHASES: usize = Phase2::StepTotal as usize + 1;

const NAMES: [&str; NUM_PHASES] = [
    "loop_gap", "collect", "pre", "verify", "walk", "emit", "commit", "propose", "TOTAL",
];

static SUM_US: [AtomicU64; NUM_PHASES] = [const { AtomicU64::new(0) }; NUM_PHASES];
static COUNT: [AtomicU64; NUM_PHASES] = [const { AtomicU64::new(0) }; NUM_PHASES];
static STEPS: AtomicU64 = AtomicU64::new(0);
/// Tokens committed (accepted drafts + bonus) across the current window —
/// fed by `record_committed` from the verify step so the summary line can
/// print tok/step and effective tok/s alongside the phase wall times
/// (the joined quantities the SPEC-3X arithmetic needs; docs/SPEC-3X-PLAN.md).
static COMMITTED_TOKS: AtomicU64 = AtomicU64::new(0);

/// Scheduler-thread step clock (guarded for soundness; uncontended).
struct Clock {
    /// Exit Instant of the previous DFlash step (feeds `loop_gap`).
    last_exit: Option<Instant>,
    /// Entry Instant of the current step (feeds `StepTotal`).
    step_start: Option<Instant>,
    /// Last bucket boundary inside the current step (feeds `mark`).
    last_mark: Option<Instant>,
}

static CLOCK: Mutex<Clock> = Mutex::new(Clock {
    last_exit: None,
    step_start: None,
    last_mark: None,
});

/// Whether `ATLAS_STEP_TIMING2=1` armed the accumulators (cached once).
pub(crate) fn enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_STEP_TIMING2").ok().as_deref() == Some("1"))
}

fn add(phase: Phase2, us: u64) {
    SUM_US[phase as usize].fetch_add(us, Ordering::Relaxed);
    COUNT[phase as usize].fetch_add(1, Ordering::Relaxed);
}

/// Record the tokens a verify step committed (accepted drafts + bonus).
/// No-op when disarmed. Called once per verify step, right next to
/// `accept_log::record`, so the window matches the phase timings exactly.
pub(crate) fn record_committed(toks: usize) {
    if !enabled() {
        return;
    }
    COMMITTED_TOKS.fetch_add(toks as u64, Ordering::Relaxed);
}

/// Record an out-of-step span (e.g. `Collect`) measured by the caller.
/// No-op when disarmed. Does not touch the in-step mark chain.
pub(crate) fn record(phase: Phase2, since: Instant) {
    if !enabled() {
        return;
    }
    add(phase, u64::try_from(since.elapsed().as_micros()).unwrap_or(u64::MAX));
}

/// RAII step scope: created at `step_verify_dflash` entry; `Drop` closes the
/// step on EVERY exit path (early returns included), recording `StepTotal`
/// and stamping `last_exit` for the next step's `loop_gap`.
pub(crate) struct StepGuard {
    armed: bool,
}

impl Drop for StepGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let now = Instant::now();
        {
            let mut c = CLOCK.lock();
            if let Some(ss) = c.step_start.take() {
                add(
                    Phase2::StepTotal,
                    u64::try_from((now - ss).as_micros()).unwrap_or(u64::MAX),
                );
            }
            c.last_mark = None;
            c.last_exit = Some(now);
        }
        let steps = STEPS.fetch_add(1, Ordering::Relaxed) + 1;
        if steps.is_multiple_of(SUMMARY_PERIOD) {
            summarize();
        }
    }
}

/// Open a step scope. Call once at the top of `step_verify_dflash`.
pub(crate) fn step_begin() -> StepGuard {
    if !enabled() {
        return StepGuard { armed: false };
    }
    let now = Instant::now();
    let mut c = CLOCK.lock();
    if let Some(le) = c.last_exit {
        add(
            Phase2::LoopGap,
            u64::try_from((now - le).as_micros()).unwrap_or(u64::MAX),
        );
    }
    c.step_start = Some(now);
    c.last_mark = Some(now);
    StepGuard { armed: true }
}

/// Close bucket `phase` at the current instant: the elapsed time since the
/// previous mark (or step entry) is charged to `phase`, and the mark chain
/// advances. No-op when disarmed or outside a step scope.
pub(crate) fn mark(phase: Phase2) {
    if !enabled() {
        return;
    }
    let now = Instant::now();
    let mut c = CLOCK.lock();
    if let Some(lm) = c.last_mark.replace(now) {
        add(phase, u64::try_from((now - lm).as_micros()).unwrap_or(u64::MAX));
    }
}

/// Emit the periodic summary and reset the accumulators.
fn summarize() {
    use std::fmt::Write as _;
    let mut sums = [0u64; NUM_PHASES];
    let mut line = String::with_capacity(NUM_PHASES * 32);
    for i in 0..NUM_PHASES {
        let sum = SUM_US[i].swap(0, Ordering::Relaxed);
        let cnt = COUNT[i].swap(0, Ordering::Relaxed);
        sums[i] = sum;
        if cnt == 0 {
            continue;
        }
        let per_step_ms = sum as f64 / 1000.0 / SUMMARY_PERIOD as f64;
        let fires = cnt as f64 / SUMMARY_PERIOD as f64;
        let _ = write!(line, " {}={per_step_ms:.2}ms(x{fires:.1})", NAMES[i]);
    }
    // `other` = in-step residual the buckets did not cover (early-return
    // paths, inter-bucket glue). `loop_gap`/`collect` are outside StepTotal.
    let in_step: u64 = (Phase2::Pre as usize..=Phase2::Propose as usize)
        .map(|i| sums[i])
        .sum();
    let other_ms =
        sums[Phase2::StepTotal as usize].saturating_sub(in_step) as f64 / 1000.0 / SUMMARY_PERIOD as f64;
    let wall_ms = (sums[Phase2::StepTotal as usize] + sums[Phase2::LoopGap as usize]) as f64
        / 1000.0
        / SUMMARY_PERIOD as f64;
    // Joined acceptance × step-time view: committed tokens over the SAME
    // window as the wall times, so `spec_tok_s` is the actual speculative
    // throughput this window sustained (committed / wall), not an estimate
    // stitched from two differently-windowed logs.
    let committed = COMMITTED_TOKS.swap(0, Ordering::Relaxed) as f64;
    let tok_step = committed / SUMMARY_PERIOD as f64;
    let spec_tok_s = if wall_ms > 0.0 { tok_step * 1000.0 / wall_ms } else { 0.0 };
    tracing::info!(
        "DFLASH STEP_TIMING2 [{SUMMARY_PERIOD} steps]:{line} other={other_ms:.2}ms \
         wall={wall_ms:.2}ms/step tok_step={tok_step:.2} spec_tok_s={spec_tok_s:.1}"
    );
}
