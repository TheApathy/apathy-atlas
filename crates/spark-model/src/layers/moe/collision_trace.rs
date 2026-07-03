// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-token expert collision instrumentation for K=3 verify.
//!
//! Phase 1 of cross-token expert deduplication: measure how often the 3
//! tokens in a K=3 verify slot pick overlapping experts. If avg unique
//! count / (3 * top_k) is low (i.e. collisions are common), the dedup
//! kernel work in Phase 2-3 is worth shipping. If unique ≈ 3 * top_k,
//! collisions are rare and we abandon the project.
//!
//! Enable with `ATLAS_MOE_COLLISION_TRACE=1`. Stats accumulate across
//! all layers and all calls and flush to stderr every `FLUSH_EVERY`
//! K=3 invocations.
//!
//! Cost: 1 D2H copy of 96 bytes + 1 stream sync per K=3 call when
//! enabled. Compiles to a single `is_enabled()` atomic load when not.

use std::sync::Once;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const FLUSH_EVERY: u64 = 200;

static INIT: Once = Once::new();
static ENABLED: AtomicBool = AtomicBool::new(false);

// Accumulators across all MoE layers and calls.
// unique_sum / call_count = avg unique experts per (3 tokens × top_k) slot table.
// slot_sum / call_count   = avg total slot count (== 3 * top_k always).
// pair_overlap_sum / call_count = avg pairwise overlap (#experts shared by ≥2 tokens).
// triple_overlap_sum / call_count = avg #experts shared by all 3 tokens.
static UNIQUE_SUM: AtomicU64 = AtomicU64::new(0);
static SLOT_SUM: AtomicU64 = AtomicU64::new(0);
static PAIR_OVERLAP_SUM: AtomicU64 = AtomicU64::new(0);
static TRIPLE_OVERLAP_SUM: AtomicU64 = AtomicU64::new(0);
static CALL_COUNT: AtomicU64 = AtomicU64::new(0);
// Histogram of unique-count distribution (clamped 0..=64).
static UNIQUE_HIST: [AtomicUsize; 65] = [const { AtomicUsize::new(0) }; 65];

#[inline]
pub fn is_enabled() -> bool {
    INIT.call_once(|| {
        let on = std::env::var("ATLAS_MOE_COLLISION_TRACE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        ENABLED.store(on, Ordering::Relaxed);
    });
    ENABLED.load(Ordering::Relaxed)
}

/// Sample one K=3 dispatch.
///
/// `indices_dev` points at the [3 * top_k] u32 expert-id table that the
/// fused MoE kernel will consume. We sync the stream, D2H-copy the small
/// table, and update accumulators.
pub fn sample(gpu: &dyn GpuBackend, indices_dev: DevicePtr, top_k: u32, stream: u64) -> Result<()> {
    if !is_enabled() {
        return Ok(());
    }
    let slot_count = 3 * top_k as usize;
    let bytes = slot_count * 4;
    // Stream sync required because the topk kernel is still in flight on
    // `stream` — we need its writes visible before D2H. This is purely
    // a measurement path; not on the perf-critical fast path.
    gpu.synchronize(stream)?;
    let mut buf = vec![0u8; bytes];
    gpu.copy_d2h(indices_dev, &mut buf)?;
    let indices: Vec<u32> = buf
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Compute per-call stats.
    let tk = top_k as usize;
    let tok0 = &indices[0..tk];
    let tok1 = &indices[tk..2 * tk];
    let tok2 = &indices[2 * tk..3 * tk];
    // Unique experts across all 3 tokens.
    let mut all: Vec<u32> = indices.clone();
    all.sort_unstable();
    all.dedup();
    let unique = all.len();

    // Per-expert hit count (1..=3 across the 3 tokens) — fold by membership
    // count to derive pair/triple overlap.
    let mut pair_overlap = 0u64;
    let mut triple_overlap = 0u64;
    for &e in &all {
        let in0 = tok0.contains(&e);
        let in1 = tok1.contains(&e);
        let in2 = tok2.contains(&e);
        let hits = (in0 as u32) + (in1 as u32) + (in2 as u32);
        if hits >= 2 {
            pair_overlap += 1;
        }
        if hits == 3 {
            triple_overlap += 1;
        }
    }

    UNIQUE_SUM.fetch_add(unique as u64, Ordering::Relaxed);
    SLOT_SUM.fetch_add(slot_count as u64, Ordering::Relaxed);
    PAIR_OVERLAP_SUM.fetch_add(pair_overlap, Ordering::Relaxed);
    TRIPLE_OVERLAP_SUM.fetch_add(triple_overlap, Ordering::Relaxed);
    let bucket = unique.min(64);
    UNIQUE_HIST[bucket].fetch_add(1, Ordering::Relaxed);
    let n = CALL_COUNT.fetch_add(1, Ordering::Relaxed) + 1;

    if n.is_multiple_of(FLUSH_EVERY) {
        flush(n);
    }
    Ok(())
}

fn flush(n: u64) {
    let unique = UNIQUE_SUM.load(Ordering::Relaxed) as f64 / n as f64;
    let slots = SLOT_SUM.load(Ordering::Relaxed) as f64 / n as f64;
    let pair = PAIR_OVERLAP_SUM.load(Ordering::Relaxed) as f64 / n as f64;
    let triple = TRIPLE_OVERLAP_SUM.load(Ordering::Relaxed) as f64 / n as f64;
    let dedup_ratio = if unique > 0.0 { slots / unique } else { 0.0 };
    eprintln!(
        "[moe-collision] n={n} avg_unique={unique:.2}/{slots:.0} dedup_ratio={dedup_ratio:.3}x pair_overlap={pair:.2} triple_overlap={triple:.2}"
    );
}

/// Force-print final stats. Call from server shutdown if instrumented.
#[allow(dead_code)]
pub fn finalize() {
    let n = CALL_COUNT.load(Ordering::Relaxed);
    if n > 0 {
        flush(n);
        // Dump histogram (sparse — only nonzero buckets).
        let mut buckets: Vec<(usize, usize)> = (0..65)
            .map(|i| (i, UNIQUE_HIST[i].load(Ordering::Relaxed)))
            .filter(|(_, c)| *c > 0)
            .collect();
        buckets.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        eprintln!("[moe-collision] unique-count histogram (top buckets):");
        for (k, c) in buckets.iter().take(10) {
            let pct = (*c as f64) * 100.0 / (n as f64);
            eprintln!("  unique={k:>2}: {c} calls ({pct:.1}%)");
        }
    }
}
