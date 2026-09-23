// SPDX-License-Identifier: AGPL-3.0-only
//! Repeated hot-buffer host-plus-GPU projection timing, never serving tok/s.
use super::{
    Operator, Projection, artifacts, contract, gemv::GemvMode, session::Session, timing_order,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use spark_runtime::{
    cublaslt::ReductionPolicy,
    gpu::{DevicePtr, GpuBackend},
};
use std::{
    fs,
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const REPEATS: usize = 128;
const ARMS: [Projection; 5] = [
    Projection::Cublas(ReductionPolicy::Baseline),
    Projection::TensorCore,
    Projection::Gemv(GemvMode::Sequential),
    Projection::Gemv(GemvMode::Gather),
    Projection::Gemv(GemvMode::Batch2),
];
fn unix_ns() -> Result<String> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_nanos()
        .to_string())
}

pub fn run(
    session: &mut Session,
    operator: &Operator,
    weight_name: &str,
    root: &Path,
) -> Result<Value> {
    let timing_root = root.join(format!("{weight_name}-timing"));
    fs::create_dir(&timing_root)?;
    let mut shapes = Vec::new();
    for rows in [1, 2] {
        let shape_root = timing_root.join(format!("m{rows}"));
        fs::create_dir(&shape_root)?;
        let bytes = rows as usize * 1024 * 2;
        let mut expected = Vec::new();
        for policy in ARMS {
            let file = if rows == 1 { "split.bin" } else { "full.bin" };
            let path = root.join(format!("{weight_name}-{}-forward/{file}", policy.name()));
            let reference = fs::read(&path)?;
            contract::validate_bf16(&reference, 4096)?;
            expected.push(reference[..bytes].to_vec());
            // Excluded warmup also proves the non-diagnostic launch matches its
            // own raw diagnostic control. This never substitutes one family.
            session.poison_output(DevicePtr(operator.output.ptr))?;
            operator.enqueue(session, policy, rows, 0, false)?;
            session.gpu.synchronize(session.stream)?;
            let output = session.read(DevicePtr(operator.output.ptr))?;
            let raw = shape_root.join(format!("warmup-{}.bin", policy.name()));
            artifacts::write(&raw, &output[..bytes])?;
            let comparison = contract::compare(expected.last().unwrap(), &output[..bytes])?;
            artifacts::json_file(
                &shape_root.join(format!("warmup-{}.json", policy.name())),
                &json!({"excluded_from_timing":true,"comparison":comparison,
                    "reference":path,"reference_sha256":artifacts::sha256(&path)?,
                    "output_sha256":artifacts::sha256(&raw)?}),
            )?;
            ensure!(
                comparison["exact"] == true,
                "normal launch differs from its own diagnostic oracle"
            );
        }
        let mut samples = Vec::new();
        let mut latencies: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::new());
        for round in 0..10 {
            for (position, arm) in timing_order::order(round).into_iter().enumerate() {
                let policy = ARMS[arm];
                session.poison_output(DevicePtr(operator.output.ptr))?;
                session.gpu.synchronize(session.stream)?;
                let unix_start_ns = unix_ns()?;
                let started = Instant::now();
                let completed = (|| -> Result<()> {
                    for _ in 0..REPEATS {
                        operator.enqueue(session, policy, rows, 0, false)?;
                    }
                    session.gpu.synchronize(session.stream)?;
                    Ok(())
                })();
                let elapsed = started.elapsed();
                let unix_end_ns = unix_ns()?;
                let stem = format!("round{round:02}-position{position}-{}", policy.name());
                let receipt_path = shape_root.join(format!("{stem}.json"));
                let average_us = elapsed.as_secs_f64() * 1e6 / REPEATS as f64;
                let mut receipt = json!({"weight":weight_name,"m":rows,"n":1024,"k":4096,
                    "round":round,"position":position,"arm":arm,"policy":policy.name(),
                    "repeats":REPEATS,"unix_start_ns":unix_start_ns,"unix_end_ns":unix_end_ns,
                    "elapsed_ns":elapsed.as_nanos().to_string(),"average_us":average_us,
                    "call_error":completed.as_ref().err().map(|e|format!("{e:#}")),
                    "raw_exact":false});
                if completed.is_err() {
                    artifacts::json_file(&receipt_path, &receipt)?;
                    completed.context(
                        "operator batch failed; receipt retained; session drain required",
                    )?;
                }
                ensure!(
                    average_us.is_finite() && average_us > 0.0,
                    "invalid operator duration"
                );
                let output = session.read(DevicePtr(operator.output.ptr))?;
                let raw = shape_root.join(format!("{stem}.bin"));
                artifacts::write(&raw, &output[..bytes])?;
                let comparison = contract::compare(&expected[arm], &output[..bytes])?;
                receipt["raw_exact"] = comparison["exact"].clone();
                receipt["comparison"] = comparison;
                receipt["output_sha256"] = json!(artifacts::sha256(&raw)?);
                artifacts::json_file(&receipt_path, &receipt)?;
                ensure!(
                    receipt["raw_exact"] == true,
                    "operator batch raw drift; receipt retained"
                );
                latencies[arm].push(average_us);
                samples.push(receipt);
            }
        }
        let mut medians = Vec::new();
        for (policy, values) in ARMS.into_iter().zip(&mut latencies) {
            ensure!(values.len() == 10, "incomplete balanced operator timing");
            values.sort_by(f64::total_cmp);
            medians.push(json!({"policy":policy.name(),"samples":values.len(),
                "median_average_us":(values[4]+values[5])/2.0}));
        }
        let report = json!({"m":rows,"repeats_per_sample":REPEATS,
            "orders":"five cyclic rotations and five reversed rotations",
            "samples":samples,"medians":medians});
        artifacts::json_file(&shape_root.join("result.json"), &report)?;
        shapes.push(report);
    }
    let report = json!({"weight":weight_name,"shapes":shapes,
        "scope":"repeated single-weight hot-buffer host-plus-GPU average projection latency",
        "includes":"normal launch builders and geometry checks, cuBLAS descriptors, amortized completion fence",
        "excludes":"metadata upload, kernel lookup, output readback, hashing, JSON and filesystem writes",
        "qualification":"8MiB weight reuse may hit L2; not cold memory, GPU-only, full-model or serving speed",
        "resource_isolation_qualified":false,"model_quality_qualified":false,"serving_speed_qualified":false});
    artifacts::json_file(&timing_root.join("result.json"), &report)?;
    Ok(report)
}
