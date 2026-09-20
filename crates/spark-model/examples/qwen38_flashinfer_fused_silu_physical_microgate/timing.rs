// SPDX-License-Identifier: AGPL-3.0-only

use std::time::Instant;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::launch::{launch, median, p90};
use super::{Kernels, Plan};

pub(super) struct Timing {
    pub(super) parent_median: f64,
    pub(super) candidate_median: f64,
    pub(super) parent_p90: f64,
    pub(super) candidate_p90: f64,
    pub(super) paired_median: f64,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn measure_balanced(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: Kernels,
    plan: Plan,
    gate: DevicePtr,
    up: DevicePtr,
    temporary: DevicePtr,
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
            gate,
            up,
            temporary,
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
    for order in [[false, true], [true, false]] {
        for candidate in order {
            let _ = measure(candidate)?;
        }
    }
    let (mut parent_times, mut candidate_times, mut deltas) = (Vec::new(), Vec::new(), Vec::new());
    for index in 0..plan.reps {
        let order = if index % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        let (mut parent_time, mut candidate_time) = (0.0, 0.0);
        for candidate in order {
            let time = measure(candidate)?;
            if candidate {
                candidate_time = time
            } else {
                parent_time = time
            }
        }
        parent_times.push(parent_time);
        candidate_times.push(candidate_time);
        deltas.push(parent_time - candidate_time);
    }
    let timing = Timing {
        parent_median: median(&parent_times),
        candidate_median: median(&candidate_times),
        parent_p90: p90(&parent_times),
        candidate_p90: p90(&candidate_times),
        paired_median: median(&deltas),
    };
    ensure!(
        timing.candidate_median < timing.parent_median
            && timing.candidate_p90 < timing.parent_p90
            && timing.paired_median > 0.0,
        "timing gate failed: parent median/p90={:.6}/{:.6}, candidate={:.6}/{:.6}, paired={:.6}",
        timing.parent_median,
        timing.parent_p90,
        timing.candidate_median,
        timing.candidate_p90,
        timing.paired_median
    );
    Ok(timing)
}
