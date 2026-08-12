// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_MOE_ROUTE_LOG=1` — per-layer TEMPORAL expert-routing locality.
//!
//! # What this measures and why it is not `union_stats` / `dump::route_group_row`
//!
//! Two instruments already exist and both measure the WRONG axis for the
//! prefetch question:
//!
//! * [`super::union_stats`] measures the union of experts ACROSS THE ROWS of one
//!   speculative-verify batch (`m` rows of the SAME layer, same step). That
//!   bounds the dedup'd `_m` kernel's speedup.
//! * [`super::dump::route_group_row`] (`ATLAS_MOE_OVERLAP=1`) measures the same
//!   cross-row quantity from the per-row decode path.
//!
//! A PREFETCHER needs the third axis: for one fixed layer `L`, how much does the
//! top-k set at token `t` predict the top-k set at token `t+1`? That is a
//! cross-STEP quantity, and nothing in-tree measured it. This module does.
//!
//! # The statistic
//!
//! Per MoE layer, with `S_t` = the top-k expert-id set at token `t`:
//!
//! ```text
//!   P(e used at t+1 | e used at t)  ==  E[ |S_t ∩ S_{t+1}| ] / top_k
//! ```
//!
//! (The identity holds because `|S_t| = |S_{t+1}| = top_k` exactly — the top-k
//! kernel always emits `top_k` distinct ids.) This is reported as `carry`.
//!
//! It also reports the natural generalization a real prefetcher would use — the
//! sliding-window predictor "prefetch the union of the last `W` tokens' sets":
//!
//! ```text
//!   coverage(W) = E[ |S_t ∩ (S_{t-1} ∪ … ∪ S_{t-W})| ] / top_k     (hit rate)
//!   cost(W)     = E[ |S_{t-1} ∪ … ∪ S_{t-W}| ] / top_k             (bytes fetched)
//! ```
//!
//! `coverage/cost` is the prefetcher's bandwidth efficiency: a predictor is only
//! interesting if `coverage(W)` climbs faster than `cost(W)`, because on this
//! box the prefetch and the demand fetch contend for the SAME DRAM.
//!
//! Finally it reports the STATIC predictor: the fraction of routed slots that
//! land in the layer's `top_k` / `2*top_k` / `4*top_k` most-frequently-used
//! experts over the whole run (`hot6/hot12/hot24`). If routing is skewed enough
//! that a fixed hot set covers most fires, a static L2-pin beats any temporal
//! predictor and needs no prediction machinery at all.
//!
//! # Layer identity without a `layer_idx` field
//!
//! `MoeLayer` carries no layer index and threading one through would touch every
//! weight-loader constructor. Instead the layer is keyed by the ADDRESS of its
//! `MoeLayer` instance, which is unique and stable for the process lifetime; the
//! first-seen order of those addresses IS model order, because the layer loop
//! visits layers in order. So `layer=` in the output is the true MoE-layer
//! ordinal (0-based, counting only MoE layers).
//!
//! # Cost, and the two ways this instrument can silently see nothing
//!
//! Enabled, each MoE fire costs one `synchronize(stream)` + one `top_k*4`-byte
//! D2H. That perturbs STEP TIMING (do not read tok/s from a logging run) but not
//! routing, which is what we are measuring. Disabled (default), it is one cached
//! bool load.
//!
//! 1. **CUDA-graph capture**: a D2H inside capture invalidates it (CUDA 901).
//!    Guarded — the caller only invokes this when `!ctx.graph_capture`.
//! 2. **CUDA-graph REPLAY runs no host code at all.** Once decode is graphed,
//!    this tap goes blind and the counters freeze. Graphs engage at
//!    `seq_len > 266` on DeepSeek-V4 (`fp8_kv_calibration_tokens = 256`), so a
//!    long measurement run MUST also disable graphs — `ATLAS_PROFILE=1` does
//!    (by design), as does `ATLAS_CUDA_GRAPHS=0` where supported. Check the log
//!    for a rising `fires=` before trusting any number.
//!
//! # Env
//!
//! | var | default | meaning |
//! |---|---|---|
//! | `ATLAS_MOE_ROUTE_LOG` | unset | `=1` enables everything below |
//! | `ATLAS_MOE_ROUTE_LOG_EVERY` | 2048 | fires between summary lines |
//! | `ATLAS_MOE_ROUTE_LOG_FILE` | unset | append a raw CSV trace here |
//!
//! Raw CSV (for offline analysis of any statistic, not just the ones above):
//! `fire,layer,tok_index,hash_routed,id0,id1,…` — `tok_index` is that layer's
//! own fire ordinal, i.e. the token index within the layer's own sequence.

use std::collections::HashMap;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Widest sliding window the coverage/cost table reports.
const MAX_W: usize = 4;

/// Expert-id space we keep a frequency histogram over. DeepSeek-V4 has 256; a
/// larger model just loses the static-predictor column (ids ≥ this are still
/// counted in every temporal statistic).
const HIST_EXPERTS: usize = 512;

#[inline]
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ATLAS_MOE_ROUTE_LOG").ok().as_deref() == Some("1"))
}

fn log_every() -> u64 {
    static EVERY: OnceLock<u64> = OnceLock::new();
    *EVERY.get_or_init(|| {
        std::env::var("ATLAS_MOE_ROUTE_LOG_EVERY")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(2048)
    })
}

/// Per-layer accumulator. One per distinct `MoeLayer` instance address.
struct LayerAcc {
    /// First-seen ordinal == MoE-layer index in model order.
    ordinal: usize,
    /// This layer's own fire count (== token index for plain decode).
    fires: u64,
    /// Last `MAX_W` sets, newest last. Each is exactly `top_k` sorted ids.
    history: Vec<Vec<u32>>,
    /// `sum_carry / (fires_with_prev * top_k)` = P(e at t+1 | e at t).
    sum_carry: u64,
    /// Fires that had at least one predecessor (the denominator for `carry`).
    fires_with_prev: u64,
    /// `|S_t ∩ S_{t-1}|` histogram, index 0..=top_k.
    carry_hist: [u64; 33],
    /// Per-window sums; index `w-1` for window size `w`.
    sum_cov: [u64; MAX_W],
    sum_cost: [u64; MAX_W],
    fires_with_w: [u64; MAX_W],
    /// Lifetime per-expert use counts (static-predictor input).
    freq: Vec<u32>,
    /// Total routed slots (== fires * top_k), the static-predictor denominator.
    slots: u64,
    hash_routed: bool,
}

struct Acc {
    layers: HashMap<usize, LayerAcc>,
    /// Global fire counter across all layers — drives the summary cadence and
    /// the CSV `fire` column.
    fires: u64,
    trace: Option<std::io::BufWriter<std::fs::File>>,
    trace_tried: bool,
}

fn acc() -> &'static Mutex<Acc> {
    static ACC: OnceLock<Mutex<Acc>> = OnceLock::new();
    ACC.get_or_init(|| {
        Mutex::new(Acc {
            layers: HashMap::new(),
            fires: 0,
            trace: None,
            trace_tried: false,
        })
    })
}

/// Record one MoE layer's top-k expert ids for one token.
///
/// `layer_key` must be a value that is unique and stable per MoE layer for the
/// process lifetime — the caller passes `self as *const MoeLayer as usize`.
/// `indices_dev` is the `[top_k]` u32 device buffer the top-k kernel just wrote
/// on `stream`.
///
/// MUST NOT be called during CUDA-graph capture (the D2H would invalidate it).
pub(super) fn record_decode_route(
    gpu: &dyn GpuBackend,
    stream: u64,
    layer_key: usize,
    indices_dev: DevicePtr,
    top_k: u32,
    hash_routed: bool,
) -> Result<()> {
    if !enabled() || top_k == 0 {
        return Ok(());
    }
    // Order the D2H after the top-k kernel that produced the indices.
    gpu.synchronize(stream)?;
    let k = top_k as usize;
    let mut buf = vec![0u8; k * 4];
    if gpu.copy_d2h(indices_dev, &mut buf).is_err() {
        return Ok(());
    }
    let mut ids: Vec<u32> = buf
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    ids.sort_unstable();

    let mut a = acc()
        .lock()
        .map_err(|_| anyhow::anyhow!("moe route-locality mutex poisoned"))?;
    a.fires += 1;
    let fire = a.fires;
    let next_ordinal = a.layers.len();

    // ── update this layer's accumulator ──
    {
        let e = a.layers.entry(layer_key).or_insert_with(|| LayerAcc {
            ordinal: next_ordinal,
            fires: 0,
            history: Vec::with_capacity(MAX_W),
            sum_carry: 0,
            fires_with_prev: 0,
            carry_hist: [0; 33],
            sum_cov: [0; MAX_W],
            sum_cost: [0; MAX_W],
            fires_with_w: [0; MAX_W],
            freq: vec![0; HIST_EXPERTS],
            slots: 0,
            hash_routed,
        });

        // carry: |S_t ∩ S_{t-1}|
        if let Some(prev) = e.history.last() {
            let carry = intersect_len(&ids, prev);
            e.sum_carry += carry as u64;
            e.fires_with_prev += 1;
            e.carry_hist[carry.min(32)] += 1;
        }

        // sliding-window coverage/cost for W = 1..=MAX_W
        for w in 1..=MAX_W {
            if e.history.len() < w {
                continue;
            }
            let mut union: Vec<u32> = Vec::with_capacity(w * k);
            for s in e.history.iter().rev().take(w) {
                union.extend_from_slice(s);
            }
            union.sort_unstable();
            union.dedup();
            e.sum_cov[w - 1] += intersect_len(&ids, &union) as u64;
            e.sum_cost[w - 1] += union.len() as u64;
            e.fires_with_w[w - 1] += 1;
        }

        for &id in &ids {
            if (id as usize) < HIST_EXPERTS {
                e.freq[id as usize] += 1;
            }
        }
        e.slots += k as u64;
        e.fires += 1;

        if e.history.len() == MAX_W {
            e.history.remove(0);
        }
        e.history.push(ids.clone());

        let tok_index = e.fires - 1;
        let ordinal = e.ordinal;
        write_trace(&mut a, fire, ordinal, tok_index, hash_routed, &ids);
    }

    if fire.is_multiple_of(log_every()) {
        report(&a, k);
    }
    Ok(())
}

/// `|a ∩ b|` for two sorted, deduplicated id slices.
fn intersect_len(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut n) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                n += 1;
                i += 1;
                j += 1;
            }
        }
    }
    n
}

fn write_trace(
    a: &mut Acc,
    fire: u64,
    ordinal: usize,
    tok_index: u64,
    hash_routed: bool,
    ids: &[u32],
) {
    if !a.trace_tried {
        a.trace_tried = true;
        if let Ok(path) = std::env::var("ATLAS_MOE_ROUTE_LOG_FILE") {
            match std::fs::File::create(&path) {
                Ok(f) => {
                    let mut w = std::io::BufWriter::new(f);
                    let _ = writeln!(w, "fire,layer,tok_index,hash_routed,ids");
                    a.trace = Some(w);
                    tracing::info!("moe-route-log: raw trace -> {path}");
                }
                Err(e) => tracing::warn!("moe-route-log: cannot open {path}: {e}"),
            }
        }
    }
    if let Some(w) = a.trace.as_mut() {
        let ids_str = ids
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let _ = writeln!(
            w,
            "{fire},{ordinal},{tok_index},{},{ids_str}",
            u8::from(hash_routed)
        );
        let _ = w.flush();
    }
}

/// Emit the per-layer table plus the aggregate the go/no-go reads.
fn report(a: &Acc, top_k: usize) {
    let mut rows: Vec<&LayerAcc> = a.layers.values().collect();
    rows.sort_by_key(|e| e.ordinal);

    let (mut tot_carry, mut tot_prev) = (0u64, 0u64);
    let (mut tot_cov, mut tot_cost, mut tot_w) = ([0u64; MAX_W], [0u64; MAX_W], [0u64; MAX_W]);
    let (mut tot_hot, mut tot_slots) = ([0u64; 3], 0u64);

    tracing::info!(
        "moe-route-log: fires={} layers={} top_k={top_k}  \
         (carry = P(expert reused at t+1 | used at t); cov(W)/cost(W) = \
         hit-rate / bytes-multiple of prefetching the union of the last W tokens; \
         hotN = share of routed slots in the layer's N most-used experts)",
        a.fires,
        rows.len(),
    );
    for e in &rows {
        let carry = ratio(e.sum_carry, e.fires_with_prev * top_k as u64);
        let cov: Vec<f64> = (0..MAX_W)
            .map(|w| ratio(e.sum_cov[w], e.fires_with_w[w] * top_k as u64))
            .collect();
        let cost: Vec<f64> = (0..MAX_W)
            .map(|w| ratio(e.sum_cost[w], e.fires_with_w[w] * top_k as u64))
            .collect();
        let hot = hot_share(&e.freq, e.slots, top_k);

        tot_carry += e.sum_carry;
        tot_prev += e.fires_with_prev * top_k as u64;
        for w in 0..MAX_W {
            tot_cov[w] += e.sum_cov[w];
            tot_cost[w] += e.sum_cost[w];
            tot_w[w] += e.fires_with_w[w] * top_k as u64;
        }
        for i in 0..3 {
            tot_hot[i] += hot.1[i];
        }
        tot_slots += e.slots;

        tracing::info!(
            "  L{:<3} {} fires={:<7} carry={:.3}  cov(1..4)=[{:.3} {:.3} {:.3} {:.3}]  \
             cost(1..4)=[{:.2} {:.2} {:.2} {:.2}]  hot{}/{}/{} = {:.3}/{:.3}/{:.3}  \
             carry_hist={:?}",
            e.ordinal,
            if e.hash_routed { "hash" } else { "gate" },
            e.fires,
            carry,
            cov[0],
            cov[1],
            cov[2],
            cov[3],
            cost[0],
            cost[1],
            cost[2],
            cost[3],
            top_k,
            2 * top_k,
            4 * top_k,
            hot.0[0],
            hot.0[1],
            hot.0[2],
            &e.carry_hist[..=top_k.min(32)],
        );
    }

    let cov: Vec<f64> = (0..MAX_W).map(|w| ratio(tot_cov[w], tot_w[w])).collect();
    let cost: Vec<f64> = (0..MAX_W).map(|w| ratio(tot_cost[w], tot_w[w])).collect();
    tracing::info!(
        "moe-route-log ALL-LAYERS: carry={:.4}  cov(1..4)=[{:.3} {:.3} {:.3} {:.3}]  \
         cost(1..4)=[{:.2} {:.2} {:.2} {:.2}]  hot{}/{}/{} = {:.3}/{:.3}/{:.3}  \
         | random-routing baseline carry = top_k/num_experts",
        ratio(tot_carry, tot_prev),
        cov[0],
        cov[1],
        cov[2],
        cov[3],
        cost[0],
        cost[1],
        cost[2],
        cost[3],
        top_k,
        2 * top_k,
        4 * top_k,
        ratio(tot_hot[0], tot_slots),
        ratio(tot_hot[1], tot_slots),
        ratio(tot_hot[2], tot_slots),
    );
}

fn ratio(num: u64, den: u64) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

/// Share of routed slots captured by the `top_k` / `2*top_k` / `4*top_k` most
/// frequently used experts. Returns `(shares, raw_counts)`.
fn hot_share(freq: &[u32], slots: u64, top_k: usize) -> ([f64; 3], [u64; 3]) {
    let mut f: Vec<u32> = freq.to_vec();
    f.sort_unstable_by(|a, b| b.cmp(a));
    let mut shares = [0.0; 3];
    let mut counts = [0u64; 3];
    for (i, n) in [top_k, 2 * top_k, 4 * top_k].into_iter().enumerate() {
        let c: u64 = f.iter().take(n).map(|v| *v as u64).sum();
        counts[i] = c;
        shares[i] = ratio(c, slots);
    }
    (shares, counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersect_len_counts_shared_ids() {
        assert_eq!(intersect_len(&[1, 3, 5, 7], &[3, 4, 5, 9]), 2);
        assert_eq!(intersect_len(&[1, 2, 3], &[4, 5, 6]), 0);
        assert_eq!(intersect_len(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(intersect_len(&[], &[1, 2]), 0);
    }

    #[test]
    fn hot_share_is_one_when_all_slots_hit_the_hot_set() {
        // 6 experts used 10x each, everything else unused: hot6 == 1.0.
        let mut freq = vec![0u32; HIST_EXPERTS];
        for f in freq.iter_mut().take(6) {
            *f = 10;
        }
        let (shares, _) = hot_share(&freq, 60, 6);
        assert!((shares[0] - 1.0).abs() < 1e-9);
        assert!((shares[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn hot_share_matches_uniform_expectation() {
        // 256 experts each used once: hot6 = 6/256.
        let mut freq = vec![0u32; HIST_EXPERTS];
        for f in freq.iter_mut().take(256) {
            *f = 1;
        }
        let (shares, _) = hot_share(&freq, 256, 6);
        assert!((shares[0] - 6.0 / 256.0).abs() < 1e-6);
    }

    #[test]
    fn ratio_is_zero_on_empty_denominator() {
        assert_eq!(ratio(5, 0), 0.0);
        assert_eq!(ratio(3, 6), 0.5);
    }
}
