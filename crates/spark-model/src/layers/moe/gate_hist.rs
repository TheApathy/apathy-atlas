// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_MOE_GATE_HIST=1` — the gate-weight distribution logger.
//!
//! This is the MEASUREMENT that adaptive top-K
//! (`ATLAS_MOE_ADAPTIVE_TOPK`, see docs/ADAPTIVE-TOPK.md) lives or dies on:
//! how much of the routed gate mass does the smallest-weight slot actually
//! carry? Everything the feature can possibly save is bounded by that number,
//! and nothing in the code predicts it.
//!
//! ## What exactly is logged
//!
//! The values are the weights the router kernel WROTE, i.e. **post**-everything:
//!
//! ```text
//! DeepSeek-V4 (scoring_func = "sqrtsoftplus", topk_method = "noaux_tc"):
//!   score[e]     = sqrt(log(1 + exp(logit[e])))          # NOT a probability
//!   selection[e] = score[e] + correction_bias[e]          # bias steers SELECTION only
//!   idx[0..k]    = argtopk(selection)                     # descending SELECTION order
//!   w[t]         = score[idx[t]] / sum_j score[idx[j]]    # iff norm_topk_prob (V4: true)
//!   w[t]        *= routed_scaling_factor                  # V4: 1.5
//! ```
//!
//! So `w[t]` is post-normalization and post-scaling, and `sum_t w[t] ==
//! routed_scaling_factor` by construction on V4. The scale-free quantity is the
//! **gate-mass fraction** `mass[t] = w[t] / sum_j w[j]`, which is identical to
//! `score[idx[t]] / sum_j score[idx[j]]` whether or not the router normalized —
//! that is what both this logger and the prune kernel threshold on.
//!
//! ## The rank subtlety (do not skip this)
//!
//! Slots come out in descending **selection** order (`score + bias`), and the
//! correction bias is per-expert, so slot order is NOT weight order. "Slot 5"
//! (the last emitted) is not reliably the smallest weight. This logger reports
//! both orders: `slot_mass` in emitted order, and `sorted_mass` descending by
//! weight. The threshold in `moe_adaptive_topk_prune` is on the **weight**, so
//! `sorted_mass` is the column that predicts what gets dropped.
//!
//! ## Invocation
//!
//! ```text
//! ATLAS_MOE_GATE_HIST=1 \
//! ATLAS_MOE_GATE_HIST_PATH=/tmp/gate_hist.jsonl \
//! ATLAS_MOE_GATE_HIST_MAX=200000 \
//!   scripts/dsflash-serve-bench.sh <name> ...
//! ```
//!
//! Each MoE fire appends one JSON line; a rolling summary goes to `tracing`
//! every `ATLAS_MOE_GATE_HIST_EVERY` fires (default 4096). Because it
//! synchronizes and does a D2H per fire, it is a MEASUREMENT MODE, not
//! something to leave on: expect decode to slow down several-fold, and expect
//! CUDA graphs to be suppressed for the layers it touches (the caller skips it
//! under `graph_capture`, so a graphed replay logs nothing at all — run with
//! graphs disabled, e.g. the same `ATLAS_PROFILE=1` conditions the decode
//! waterfall used).

use std::io::Write;
use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Thresholds the summary reports "fraction of (layer, token) pairs whose
/// smallest kept slot falls below". These are exactly the sweep points
/// docs/ADAPTIVE-TOPK.md asks for.
const REPORT_THRESHOLDS: [f32; 4] = [0.01, 0.02, 0.05, 0.10];

#[inline]
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ATLAS_MOE_GATE_HIST").ok().as_deref() == Some("1"))
}

fn out_path() -> String {
    std::env::var("ATLAS_MOE_GATE_HIST_PATH").unwrap_or_else(|_| "moe_gate_hist.jsonl".to_string())
}

fn max_lines() -> u64 {
    std::env::var("ATLAS_MOE_GATE_HIST_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000)
}

fn summary_every() -> u64 {
    std::env::var("ATLAS_MOE_GATE_HIST_EVERY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096)
}

struct Acc {
    file: Option<std::fs::File>,
    written: u64,
    fires: u64,
    /// Sum of the gate-mass fraction at each weight-sorted rank (rank 0 = the
    /// biggest). `slots[r]` counts how many fires had a rank `r` at all, so the
    /// mean is `mass[r] / slots[r]` even with ragged `top_k` across layers.
    mass: [f64; 32],
    slots: [u64; 32],
    /// For each report threshold: how many fires had their SMALLEST slot below
    /// it, and how many total slots across all fires fell below it. The second
    /// is what converts directly to a byte saving.
    below_min: [u64; REPORT_THRESHOLDS.len()],
    below_slots: [u64; REPORT_THRESHOLDS.len()],
    total_slots: u64,
    /// Stable layer ids, keyed by the layer's own router-gate device pointer.
    /// Layers fire in model order, so first-seen order IS layer order.
    layer_ids: std::collections::HashMap<u64, u32>,
}

fn acc() -> &'static Mutex<Acc> {
    static ACC: OnceLock<Mutex<Acc>> = OnceLock::new();
    ACC.get_or_init(|| {
        let file = std::fs::File::create(out_path()).ok();
        if file.is_none() {
            tracing::warn!(
                "ATLAS_MOE_GATE_HIST: cannot open {} — JSONL disabled, summary still logged",
                out_path()
            );
        }
        Mutex::new(Acc {
            file,
            written: 0,
            fires: 0,
            mass: [0.0; 32],
            slots: [0; 32],
            below_min: [0; REPORT_THRESHOLDS.len()],
            below_slots: [0; REPORT_THRESHOLDS.len()],
            total_slots: 0,
            layer_ids: std::collections::HashMap::new(),
        })
    })
}

/// Record one MoE fire's routing decision.
///
/// `gate_key` is any per-layer-stable device pointer (the router gate weight);
/// it only ever serves as a layer identity key. `routed_scaling_factor` is
/// logged so the raw weights can be de-scaled offline.
///
/// Callers MUST skip this under CUDA-graph capture — it synchronizes and reads
/// back to host, neither of which is legal mid-capture.
#[allow(clippy::too_many_arguments)]
pub fn record(
    gpu: &dyn GpuBackend,
    stream: u64,
    indices: DevicePtr,
    weights: DevicePtr,
    top_k: u32,
    gate_key: u64,
    is_hash_routed: bool,
    routed_scaling_factor: f32,
) -> Result<()> {
    if !enabled() || top_k == 0 {
        return Ok(());
    }
    let k = top_k.min(32) as usize;

    gpu.synchronize(stream)?;
    let mut ibuf = vec![0u8; k * 4];
    let mut wbuf = vec![0u8; k * 4];
    gpu.copy_d2h(indices, &mut ibuf)?;
    gpu.copy_d2h(weights, &mut wbuf)?;
    let idx: Vec<u32> = ibuf
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let w: Vec<f32> = wbuf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // NaN-safe: a non-finite router sum means something upstream is broken, and
    // dividing by it would poison the histogram rather than report the break.
    let sum: f32 = w.iter().sum();
    if !sum.is_finite() || sum <= 1e-20 {
        return Ok(());
    }
    // Scale-free gate-mass fractions: independent of norm_topk_prob and of
    // routed_scaling_factor, so thresholds transfer across models.
    let slot_mass: Vec<f32> = w.iter().map(|x| x / sum).collect();
    let mut sorted_mass = slot_mass.clone();
    sorted_mass.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut a = acc()
        .lock()
        .map_err(|_| anyhow::anyhow!("gate-hist mutex poisoned"))?;

    let next_id = a.layer_ids.len() as u32;
    let layer = *a.layer_ids.entry(gate_key).or_insert(next_id);
    a.fires += 1;
    let fire = a.fires;

    for (r, m) in sorted_mass.iter().enumerate() {
        a.mass[r] += *m as f64;
        a.slots[r] += 1;
    }
    a.total_slots += k as u64;
    let min_mass = *sorted_mass.last().unwrap_or(&1.0);
    for (i, thr) in REPORT_THRESHOLDS.iter().enumerate() {
        if min_mass < *thr {
            a.below_min[i] += 1;
        }
        // The arg-max slot is never droppable, so it is excluded from the
        // slot count exactly as the prune kernel excludes it.
        a.below_slots[i] += sorted_mass.iter().skip(1).filter(|m| **m < *thr).count() as u64;
    }

    if a.written < max_lines() {
        let fmt = |v: &[f32]| {
            v.iter()
                .map(|x| format!("{x:.6}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let line = format!(
            "{{\"fire\":{fire},\"layer\":{layer},\"kind\":\"{}\",\"top_k\":{k},\
             \"scale\":{routed_scaling_factor},\"experts\":[{}],\"w\":[{}],\
             \"slot_mass\":[{}],\"sorted_mass\":[{}]}}\n",
            if is_hash_routed { "hash" } else { "gate" },
            idx.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(","),
            fmt(&w),
            fmt(&slot_mass),
            fmt(&sorted_mass),
        );
        if let Some(f) = a.file.as_mut()
            && f.write_all(line.as_bytes()).is_ok()
        {
            a.written += 1;
        }
    }

    if fire % summary_every() == 0 {
        let mean = |r: usize| {
            if a.slots[r] == 0 {
                0.0
            } else {
                a.mass[r] / a.slots[r] as f64
            }
        };
        let ranks: Vec<String> = (0..k).map(|r| format!("r{r}={:.4}", mean(r))).collect();
        let frac = |n: u64, d: u64| if d == 0 { 0.0 } else { n as f64 / d as f64 };
        let thr: Vec<String> = REPORT_THRESHOLDS
            .iter()
            .enumerate()
            .map(|(i, t)| {
                format!(
                    "<{t}: fires={:.3} slots={:.3}",
                    frac(a.below_min[i], a.fires),
                    frac(a.below_slots[i], a.total_slots),
                )
            })
            .collect();
        tracing::info!(
            "MOE_GATE_HIST fires={} layers={} mean gate-mass by weight-rank [{}] | \
             droppable {} | slots-fraction is the DIRECT byte saving on the routed \
             expert stream (1 slot of top_k = 1/top_k of 4.02 GB/token)",
            a.fires,
            a.layer_ids.len(),
            ranks.join(" "),
            thr.join("  "),
        );
    }
    Ok(())
}
