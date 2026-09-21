// SPDX-License-Identifier: AGPL-3.0-only

use std::time::Instant;

use anyhow::{Result, ensure};
use serde::Serialize;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::launch::{launch, median, p90};
use super::{Kernels, Plan};

#[derive(Serialize)]
pub(super) struct Timing {
    pub(super) raw_ciic_ms: Vec<[f64; 4]>,
    pub(super) parent_samples_ms: Vec<f64>,
    pub(super) candidate_samples_ms: Vec<f64>,
    pub(super) paired_deltas_ms: Vec<f64>,
    pub(super) parent_median_ms: f64,
    pub(super) candidate_median_ms: f64,
    pub(super) parent_p90_ms: f64,
    pub(super) candidate_p90_ms: f64,
    pub(super) parent_variance_ms2: f64,
    pub(super) candidate_variance_ms2: f64,
    pub(super) paired_variance_ms2: f64,
    pub(super) paired_median_ms: f64,
    pub(super) paired_mean_ms: f64,
    pub(super) paired_lower95_ms: f64,
    pub(super) positive_pair_count: usize,
    pub(super) minimum_positive_pairs: usize,
    pub(super) minimum_absolute_ms: f64,
    pub(super) relative_effect: f64,
    pub(super) minimum_relative_effect: f64,
}

#[cfg(test)]
pub(super) fn summarize(plan: Plan, parent: &[f64], candidate: &[f64]) -> Result<Timing> {
    let raw = parent
        .iter()
        .zip(candidate)
        .map(|(incumbent, direct)| [*direct, *incumbent, *incumbent, *direct])
        .collect();
    summarize_with_raw(plan, parent, candidate, raw)
}

fn summarize_with_raw(
    plan: Plan,
    parent: &[f64],
    candidate: &[f64],
    raw_ciic_ms: Vec<[f64; 4]>,
) -> Result<Timing> {
    ensure!(
        parent.len() == plan.reps && candidate.len() == plan.reps && raw_ciic_ms.len() == plan.reps,
        "timing sample count changed"
    );
    ensure!(
        parent
            .iter()
            .chain(candidate)
            .all(|sample| sample.is_finite() && *sample > 0.0),
        "timing samples must be finite and positive"
    );
    let deltas = parent
        .iter()
        .zip(candidate)
        .map(|(incumbent, direct)| incumbent - direct)
        .collect::<Vec<_>>();
    let paired_mean_ms = mean(&deltas);
    let paired_variance_ms2 = variance(&deltas, paired_mean_ms);
    let paired_lower95_ms =
        paired_mean_ms - 2.086 * (paired_variance_ms2 / plan.reps as f64).sqrt();
    let parent_median_ms = median(parent);
    let candidate_median_ms = median(candidate);
    let paired_median_ms = median(&deltas);
    let positive_pair_count = deltas.iter().filter(|delta| **delta > 0.0).count();
    let minimum_positive_pairs = (plan.reps * 4).div_ceil(5);
    let minimum_absolute_ms = if plan.m == 2_079 { 0.02 } else { 0.05 };
    let minimum_relative_effect = 0.01;
    let timing = Timing {
        raw_ciic_ms,
        parent_samples_ms: parent.to_vec(),
        candidate_samples_ms: candidate.to_vec(),
        paired_deltas_ms: deltas,
        parent_median_ms,
        candidate_median_ms,
        parent_p90_ms: p90(parent),
        candidate_p90_ms: p90(candidate),
        parent_variance_ms2: variance(parent, mean(parent)),
        candidate_variance_ms2: variance(candidate, mean(candidate)),
        paired_variance_ms2,
        paired_median_ms,
        paired_mean_ms,
        paired_lower95_ms,
        positive_pair_count,
        minimum_positive_pairs,
        minimum_absolute_ms,
        relative_effect: paired_median_ms / parent_median_ms,
        minimum_relative_effect,
    };
    ensure!(
        timing.candidate_median_ms < timing.parent_median_ms
            && timing.candidate_p90_ms < timing.parent_p90_ms
            && timing.positive_pair_count >= timing.minimum_positive_pairs
            && timing.paired_median_ms >= timing.minimum_absolute_ms
            && timing.paired_lower95_ms >= timing.minimum_absolute_ms
            && timing.relative_effect >= timing.minimum_relative_effect,
        "timing gate failed: median {:.6}->{:.6}, p90 {:.6}->{:.6}, paired median/lower95 {:.6}/{:.6}, signs {}/{}, relative {:.6}",
        timing.parent_median_ms,
        timing.candidate_median_ms,
        timing.parent_p90_ms,
        timing.candidate_p90_ms,
        timing.paired_median_ms,
        timing.paired_lower95_ms,
        timing.positive_pair_count,
        plan.reps,
        timing.relative_effect
    );
    Ok(timing)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64], mean: f64) -> f64 {
    values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64
}

#[allow(clippy::too_many_arguments)]
pub(super) fn measure_balanced(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: Kernels,
    plan: Plan,
    merged: DevicePtr,
    gate: DevicePtr,
    up: DevicePtr,
    parent_packed: DevicePtr,
    parent_scales: DevicePtr,
    candidate_packed: DevicePtr,
    candidate_scales: DevicePtr,
) -> Result<Timing> {
    let measure = |candidate| -> Result<f64> {
        gpu.synchronize(stream)?;
        let start = Instant::now();
        launch(
            candidate,
            gpu,
            stream,
            kernels,
            plan,
            merged,
            gate,
            up,
            if candidate {
                candidate_packed
            } else {
                parent_packed
            },
            if candidate {
                candidate_scales
            } else {
                parent_scales
            },
        )?;
        gpu.synchronize(stream)?;
        Ok(start.elapsed().as_secs_f64() * 1e3)
    };
    for candidate in [true, false, false, true] {
        let _ = measure(candidate)?;
    }
    let (mut parent, mut candidate, mut raw) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..plan.reps {
        let (mut incumbent, mut direct) = ([0.0; 2], [0.0; 2]);
        for (slot, is_candidate) in [true, false, false, true].into_iter().enumerate() {
            let elapsed = measure(is_candidate)?;
            match (is_candidate, slot) {
                (true, 0) => direct[0] = elapsed,
                (true, _) => direct[1] = elapsed,
                (false, 1) => incumbent[0] = elapsed,
                (false, _) => incumbent[1] = elapsed,
            }
        }
        raw.push([direct[0], incumbent[0], incumbent[1], direct[1]]);
        parent.push((incumbent[0] + incumbent[1]) / 2.0);
        candidate.push((direct[0] + direct[1]) / 2.0);
    }
    summarize_with_raw(plan, &parent, &candidate, raw)
}
