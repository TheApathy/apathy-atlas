// SPDX-License-Identifier: AGPL-3.0-only

use std::time::Instant;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{GpuBackend, KernelHandle};

use super::contract::{Case, HD, NQ, TIMING_PAIRS};
use super::guarded::Guarded;
use super::runtime::{buffers, launch};

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn p90(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() * 9).div_ceil(10) - 1]
}

#[allow(clippy::too_many_arguments)]
pub(super) fn time_case(
    gpu: &dyn GpuBackend,
    stream: u64,
    parent: KernelHandle,
    candidate: KernelHandle,
    case: Case,
) -> Result<()> {
    ensure!(
        case.q_len == 8192 && case.q_offset > 0 && case.sliding_window == 0,
        "timing is restricted to full-causal continuation cases"
    );
    let buffers = buffers(gpu, case)?;
    let output_bytes = case.q_len as usize * NQ as usize * HD as usize * 2;
    let parent_out = Guarded::output(gpu, output_bytes, 0x31)?;
    let candidate_out = Guarded::output(gpu, output_bytes, 0x73)?;
    let measure = |kernel: KernelHandle, br128: bool, output: &Guarded| -> Result<f64> {
        output.reset(gpu)?;
        gpu.synchronize(stream)?;
        let start = Instant::now();
        launch(
            gpu,
            stream,
            kernel,
            br128,
            buffers.q.payload_ptr(),
            buffers.k.payload_ptr(),
            buffers.v.payload_ptr(),
            output.payload_ptr(),
            buffers.table.payload_ptr(),
            case,
            buffers.block_stride,
            buffers.data_bytes,
        )?;
        gpu.synchronize(stream)?;
        Ok(start.elapsed().as_secs_f64() * 1_000.0)
    };
    for order in [[false, true], [true, false]] {
        for run_candidate in order {
            let _ = if run_candidate {
                measure(candidate, true, &candidate_out)?
            } else {
                measure(parent, false, &parent_out)?
            };
        }
    }
    let (mut parent_ms, mut candidate_ms, mut deltas) = (Vec::new(), Vec::new(), Vec::new());
    let (mut parent_first, mut candidate_first) = (Vec::new(), Vec::new());
    for round in 0..TIMING_PAIRS {
        let (p, c) = if round % 2 == 0 {
            (
                measure(parent, false, &parent_out)?,
                measure(candidate, true, &candidate_out)?,
            )
        } else {
            let c = measure(candidate, true, &candidate_out)?;
            (measure(parent, false, &parent_out)?, c)
        };
        let delta = p - c;
        parent_ms.push(p);
        candidate_ms.push(c);
        deltas.push(delta);
        if round % 2 == 0 {
            parent_first.push(delta)
        } else {
            candidate_first.push(delta)
        }
    }
    let metrics = (
        median(&parent_ms),
        median(&candidate_ms),
        p90(&parent_ms),
        p90(&candidate_ms),
        median(&deltas),
        median(&parent_first),
        median(&candidate_first),
    );
    ensure!(
        metrics.1 < metrics.0 && metrics.3 < metrics.2 && metrics.4 > 0.0,
        "{}: BR128 median/p90/paired timing gate failed",
        case.label()
    );
    ensure!(
        metrics.5 > 0.0 && metrics.6 > 0.0,
        "{}: order-bias gate failed parent-first={:.6} candidate-first={:.6}",
        case.label(),
        metrics.5,
        metrics.6
    );
    println!(
        "TIMING {} pairs={} parent_median_ms={:.6} candidate_median_ms={:.6} parent_p90_ms={:.6} candidate_p90_ms={:.6} paired_median_ms={:.6} parent_first_delta_ms={:.6} candidate_first_delta_ms={:.6} order_bias=PASS",
        case.label(),
        TIMING_PAIRS,
        metrics.0,
        metrics.1,
        metrics.2,
        metrics.3,
        metrics.4,
        metrics.5,
        metrics.6
    );
    buffers.verify(gpu, &case.label())?;
    parent_out.free(gpu)?;
    candidate_out.free(gpu)?;
    buffers.free(gpu)
}
