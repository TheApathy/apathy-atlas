// SPDX-License-Identifier: AGPL-3.0-only
mod abi;
mod admission;
mod contract;
mod metrics;
mod protocol;
mod runner;
mod weight;

use anyhow::{Context, Result, ensure};
use deepseek_vision_p1::{driver::Driver, io};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::ExitCode,
};

fn options() -> Result<(String, BTreeMap<String, String>)> {
    let mut args = std::env::args_os().skip(1);
    let command = args
        .next()
        .context("command required: inspect | run")?
        .into_string()
        .map_err(|_| anyhow::anyhow!("non-UTF8 command"))?;
    let mut options = BTreeMap::new();
    while let Some(key) = args.next() {
        let key = key
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 option"))?;
        ensure!(key.starts_with("--"), "named options required");
        let value = args
            .next()
            .context("missing option value")?
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 value"))?;
        ensure!(options.insert(key, value).is_none(), "duplicate option");
        ensure!(options.len() <= 4, "option count bound");
    }
    Ok((command, options))
}
fn take(options: &mut BTreeMap<String, String>, name: &str) -> Result<String> {
    options
        .remove(name)
        .with_context(|| format!("required {name}"))
}
fn path(options: &mut BTreeMap<String, String>, name: &str) -> Result<PathBuf> {
    let path = PathBuf::from(take(options, name)?);
    ensure!(path.is_absolute(), "{name} must be absolute");
    Ok(path)
}
fn summary(out: &Path, value: &Value) -> Result<()> {
    let receipt = io::save_json(out, "result.json", value)?;
    println!("{}", json!({"status":value["status"],"result":receipt}));
    Ok(())
}
fn execute() -> Result<bool> {
    let (command, mut options) = options()?;
    if command == "inspect" {
        let model = path(&mut options, "--model")?;
        let corpus = path(&mut options, "--corpus")?;
        let out = path(&mut options, "--out")?;
        ensure!(options.is_empty(), "unknown inspect options");
        println!("{}", admission::inspect(&model, &corpus, &out)?);
        return Ok(true);
    }
    ensure!(command == "run", "unknown command");
    let admission_path = path(&mut options, "--admission")?;
    let admission_sha = take(&mut options, "--admission-sha")?;
    let out = path(&mut options, "--out")?;
    ensure!(options.is_empty(), "unknown run options");
    // Complete every file/contract/library receipt and load host input bytes
    // before opening CUDA. There is no implicit model or corpus selection.
    let admitted = admission::load(&admission_path, &admission_sha)?;
    let inputs = runner::Inputs::load(&admitted)?;
    let executable = io::library_receipt(&std::env::current_exe()?)?;
    let out = io::fresh(&out)?;
    let invocation = io::save_json(
        &out,
        "invocation.json",
        &json!({"schema":"atlas-dsv-fc1-gemmex-run-v1",
        "admission_path":io::absolute(&admission_path)?,"admission_sha256":admission_sha,
        "executable":executable,"diagnostic_only":true,"full_encoder_qualified":false,
        "performance_qualified":false,"precision_limitations":admitted["precision_limitations"]}),
    )?;
    let cuda_path = Path::new(
        admitted["libraries"]["cuda"]["path"]
            .as_str()
            .context("CUDA path")?,
    );
    let mut driver = match Driver::open(cuda_path) {
        Ok(d) => d,
        Err(error) => {
            summary(
                &out,
                &json!({"status":"ERROR","phase":"CUDA initialization","error":format!("{error:#}"),
                "invocation":invocation,"full_encoder_qualified":false}),
            )?;
            return Err(error);
        }
    };
    let result = runner::run(&mut driver, &inputs, &out);
    let identity = driver.identity.clone();
    let peak = driver.peak_bytes;
    let guards = driver.guards();
    let close = driver.close();
    match (result, guards, close) {
        (Ok(value), Ok(()), Ok(())) => {
            let exact = value["reference_exact"] == true;
            summary(
                &out,
                &json!({"status":if exact {"EXACT"} else {"REFERENCE_MISMATCH"},
                "operator":value,"gpu":identity,"peak_owned_device_bytes":peak,"owned_device_cap_bytes":33554432,
                "guards_unchanged":true,"cleanup":"SUCCESS","invocation":invocation,
                "full_encoder_qualified":false,"performance_qualified":false,
                "full_encoder_gate_unchanged":{"cosine_min":0.999,"worst_row_min":0.995,"relative_l2_max":0.05}}),
            )?;
            Ok(exact)
        }
        (result, guards, close) => {
            let error = format!("run={result:?}; guards={guards:?}; cleanup={close:?}");
            summary(
                &out,
                &json!({"status":"ERROR","error":error,"gpu":identity,"peak_owned_device_bytes":peak,
                "invocation":invocation,"full_encoder_qualified":false,"performance_qualified":false}),
            )?;
            anyhow::bail!("{error}")
        }
    }
}
fn main() -> ExitCode {
    match execute() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(2),
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::from(1)
        }
    }
}
