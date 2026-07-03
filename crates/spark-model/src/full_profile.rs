// SPDX-License-Identifier: AGPL-3.0-only

//! Full per-kernel profiling for the MTP K=3 verify forward path.
//!
//! Gated entirely by `ATLAS_FULL_PROFILE=1`. When off, every macro call
//! compiles down to the original `$body` expression (zero overhead).
//!
//! When on:
//!  * The K=3 verify dispatch (`verify_c.rs`) disables CUDA graph capture
//!    so per-kernel synchronization is legal.
//!  * Every kernel launch wrapped in [`kprof!`] is sandwiched by a stream
//!    sync, accumulated into a global `(calls, total_ns)` table keyed by
//!    the kernel label.
//!  * After each verify step, [`dump_step`] emits per-kernel `KPROF`
//!    log lines that can be aggregated with `awk` post-run.
//!
//! Aggregator format (one line per kernel per step):
//! ```text
//! KPROF step=<n> kernel=<name> calls=<c> total_us=<t>
//! ```

use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::Mutex;

/// Cached `ATLAS_FULL_PROFILE` env-var lookup. `OnceLock` keeps the hot path
/// branch-free after the first call.
fn enabled_cached() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_FULL_PROFILE").ok().as_deref() == Some("1"))
}

/// Public predicate. Inlined into the macro guard.
#[inline]
pub fn is_enabled() -> bool {
    enabled_cached()
}

/// Per-kernel accumulator. Keyed by static string label so we can sort the
/// final report alphabetically without re-hashing.
struct Acc {
    calls: u64,
    total_ns: u64,
}

static TABLE: OnceLock<Mutex<HashMap<&'static str, Acc>>> = OnceLock::new();

fn table() -> &'static Mutex<HashMap<&'static str, Acc>> {
    TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record one kernel invocation. `ns` is the elapsed wall time including the
/// trailing stream sync.
pub fn record(label: &'static str, ns: u64) {
    let mut t = table().lock();
    let entry = t.entry(label).or_insert(Acc {
        calls: 0,
        total_ns: 0,
    });
    entry.calls += 1;
    entry.total_ns += ns;
}

/// Step counter shared across the verify dispatch.
static STEP: AtomicU64 = AtomicU64::new(0);

/// Suppress the first N verify steps as warmup (skip CUDA driver tiered
/// loading + JIT noise). Default 5.
fn warmup_steps() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_FULL_PROFILE_WARMUP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5)
    })
}

/// Per-step active flag — flips on after warmup_steps, set by `begin_step`.
static ACTIVE: AtomicBool = AtomicBool::new(false);

/// Test whether the current step is past warmup. Used by `kprof!` to skip the
/// sync+record overhead during warmup.
#[inline]
pub fn is_active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

/// Call at the top of `decode_verify_graphed_k3_dispatch`. Increments the
/// step counter; flips ACTIVE on once warmup is over.
///
/// CRITICAL: only flip ACTIVE when profiling is actually enabled. When
/// disabled, the `kprof!` macro must compile down to the bare body
/// expression — flipping ACTIVE without checking `is_enabled()` would make
/// every `kprof!` call wrap its body in `gpu.synchronize(stream)?`, which
/// is illegal inside the K=3 verify CUDA graph capture region (status 900,
/// CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED). The first K=3 verify after the
/// previous sequence freed slot 0's cached graph would crash on the very
/// next request — only counting-shaped prompts that fit in 5 warmup steps
/// (the prior threshold) ever appeared to work.
pub fn begin_step() {
    if !enabled_cached() {
        return;
    }
    let n = STEP.fetch_add(1, Ordering::Relaxed) + 1;
    if n > warmup_steps() {
        ACTIVE.store(true, Ordering::Relaxed);
    }
}

/// Dump the accumulated stats — call at end of bench. Emits one
/// `KPROF kernel=... calls=... total_us=...` line per kernel, sorted by
/// total_us descending.
pub fn dump() {
    let t = table().lock();
    let mut rows: Vec<_> = t.iter().map(|(k, v)| (*k, v.calls, v.total_ns)).collect();
    rows.sort_by(|a, b| b.2.cmp(&a.2));
    let total_steps = STEP.load(Ordering::Relaxed).saturating_sub(warmup_steps());
    tracing::info!("KPROF SUMMARY steps={} kernels={}", total_steps, rows.len());
    for (label, calls, ns) in &rows {
        let total_us = ns / 1000;
        let per_call_us = if *calls > 0 { ns / calls / 1000 } else { 0 };
        tracing::info!(
            "KPROF kernel={} calls={} total_us={} per_call_us={}",
            label,
            calls,
            total_us,
            per_call_us
        );
    }
}

/// Wrap a kernel launch. Returns the result of `$body`. When profiling is
/// inactive (env var off, or still in warmup), expands to the bare body
/// expression — zero overhead.
///
/// Required imports at the call site:
/// ```ignore
/// use crate::full_profile;
/// ```
///
/// Usage:
/// ```ignore
/// kprof!(ctx.gpu, stream, "rms_norm", {
///     ops::rms_norm(ctx.gpu, ..., stream)?;
/// });
/// ```
#[macro_export]
macro_rules! kprof {
    ($gpu:expr, $stream:expr, $label:literal, $body:expr) => {{
        if $crate::full_profile::is_active() {
            // Drain prior queued work so the timer captures only $body.
            $gpu.synchronize($stream)?;
            let __t = std::time::Instant::now();
            let __r = { $body };
            $gpu.synchronize($stream)?;
            let __ns = __t.elapsed().as_nanos() as u64;
            $crate::full_profile::record($label, __ns);
            __r
        } else {
            $body
        }
    }};
}
