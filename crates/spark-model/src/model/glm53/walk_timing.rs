// SPDX-License-Identifier: AGPL-3.0-only

//! Env-gated decomposition of where a GLM-5.3 decode step's wall time goes.
//!
//! Three numbers, and the third is the one that matters:
//!
//! * `blocked_moe_ns`  — host time inside the MoE seam's `copy_d2h_on_stream`,
//!   which is `cuMemcpyDtoHAsync_v2` + `cuStreamSynchronize`, i.e. a full
//!   pipeline drain. 42 of them per token, each to move 32 bytes of route ids.
//! * `blocked_final_ns` — host time in the walk's closing `synchronize`. This
//!   is GPU work that had not finished when the host ran out of things to
//!   enqueue, so it is a lower bound on genuine kernel execution.
//! * residual = `walk_ns - blocked_moe_ns - blocked_final_ns` — host time spent
//!   neither waiting for a drain nor waiting for the GPU, i.e. LAUNCH OVERHEAD.
//!   A large residual means launch-bound and points at the 2,688 launches per
//!   token; a large `blocked_moe_ns` means sync-bound and points at the drains.
//!
//! # Why host timers and no CUDA events
//!
//! The thing under measurement IS synchronisation. An instrument that adds a
//! sync to time a kernel would manufacture the very cost it is looking for, and
//! a `cudaEventSynchronize` per event would do exactly that. So this only wraps
//! calls that ALREADY block the host completely — the seam's D2H and the final
//! synchronize — where a host clock is exact and cannot make things worse.
//! Everything not attributable to those two is the residual, by subtraction
//! rather than by a measurement that would perturb the subject.
//!
//! Off by default and free when off: one relaxed atomic load per call site.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);
static INIT: std::sync::Once = std::sync::Once::new();

static WALK_NS: AtomicU64 = AtomicU64::new(0);
static BLOCKED_MOE_NS: AtomicU64 = AtomicU64::new(0);
static BLOCKED_FINAL_NS: AtomicU64 = AtomicU64::new(0);
static MOE_CALLS: AtomicU64 = AtomicU64::new(0);
static WALKS: AtomicU64 = AtomicU64::new(0);

/// `ATLAS_GLM53_WALK_TIMING=1` turns it on. Read once.
pub(crate) fn enabled() -> bool {
    INIT.call_once(|| {
        let on = std::env::var_os("ATLAS_GLM53_WALK_TIMING").is_some_and(|v| v != "0");
        ENABLED.store(on, Ordering::Relaxed);
    });
    ENABLED.load(Ordering::Relaxed)
}

fn add(counter: &AtomicU64, start: Instant) {
    counter.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// Time a call that already blocks the host. Returns the inner result.
pub(crate) fn blocked_moe<T>(f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    MOE_CALLS.fetch_add(1, Ordering::Relaxed);
    let t = Instant::now();
    let out = f();
    add(&BLOCKED_MOE_NS, t);
    out
}

pub(crate) fn blocked_final<T>(f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let t = Instant::now();
    let out = f();
    add(&BLOCKED_FINAL_NS, t);
    out
}

pub(crate) fn walk<T>(f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    WALKS.fetch_add(1, Ordering::Relaxed);
    let t = Instant::now();
    let out = f();
    add(&WALK_NS, t);
    out
}

/// One line per report, printed to stderr so it cannot be mistaken for output.
pub(crate) fn report() {
    if !enabled() {
        return;
    }
    let walks = WALKS.load(Ordering::Relaxed).max(1);
    let w = WALK_NS.load(Ordering::Relaxed);
    let m = BLOCKED_MOE_NS.load(Ordering::Relaxed);
    let f = BLOCKED_FINAL_NS.load(Ordering::Relaxed);
    let residual = w.saturating_sub(m).saturating_sub(f);
    let pct = |x: u64| if w == 0 { 0.0 } else { 100.0 * x as f64 / w as f64 };
    let per = |x: u64| x as f64 / walks as f64 / 1.0e6;
    eprintln!(
        "GLM53_WALK_TIMING walks={walks} \
         wall={:.2}ms/tok \
         blocked_moe={:.2}ms/tok ({:.1}%) moe_calls={} \
         blocked_final={:.2}ms/tok ({:.1}%) \
         residual_launch={:.2}ms/tok ({:.1}%)",
        per(w),
        per(m),
        pct(m),
        MOE_CALLS.load(Ordering::Relaxed) / walks,
        per(f),
        pct(f),
        per(residual),
        pct(residual),
    );
}
