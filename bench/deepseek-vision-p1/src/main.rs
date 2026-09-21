// SPDX-License-Identifier: AGPL-3.0-only
use anyhow::{Context, Result, ensure};
use deepseek_vision_p1::{admission, contract, driver::Driver, fc2, io, rope};
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
        .context("command required: inspect | run-rope | run-fc2")?
        .into_string()
        .map_err(|_| anyhow::anyhow!("non-UTF8 command"))?;
    let mut opts = BTreeMap::new();
    while let Some(key) = args.next() {
        let key = key
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 option"))?;
        ensure!(key.starts_with("--"), "named options required");
        let value = args
            .next()
            .context("option value missing")?
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 value"))?;
        ensure!(opts.insert(key, value).is_none(), "duplicate option");
    }
    Ok((command, opts))
}
fn take(opts: &mut BTreeMap<String, String>, key: &str) -> Result<String> {
    opts.remove(key).with_context(|| format!("required {key}"))
}
fn path(opts: &mut BTreeMap<String, String>, key: &str) -> Result<PathBuf> {
    let p = PathBuf::from(take(opts, key)?);
    ensure!(p.is_absolute(), "{key} must be absolute");
    Ok(p)
}
fn summary(out: &Path, v: &Value) -> Result<()> {
    let receipt = io::save_json(out, "result.json", v)?;
    println!("{}", json!({"status":v["status"],"result":receipt}));
    Ok(())
}
fn execute() -> Result<bool> {
    let (command, mut opts) = options()?;
    if command == "inspect" {
        let model = path(&mut opts, "--model")?;
        let corpus = path(&mut opts, "--corpus")?;
        let out = path(&mut opts, "--out")?;
        ensure!(opts.is_empty(), "unknown options");
        admission::inspect(&model, &corpus, &out)?;
        let p = out.join("admission.json");
        println!("{}", io::receipt(&p, &io::read(&p, 4 * 1024 * 1024)?)?);
        return Ok(true);
    }
    ensure!(
        matches!(command.as_str(), "run-rope" | "run-fc2"),
        "unknown command"
    );
    let admission_path = path(&mut opts, "--admission")?;
    let admission_sha = take(&mut opts, "--admission-sha")?;
    let out = path(&mut opts, "--out")?;
    let candidate = if command == "run-rope" {
        Some((
            path(&mut opts, "--candidate-ptx")?,
            take(&mut opts, "--candidate-ptx-sha")?,
            path(&mut opts, "--build-receipt")?,
            take(&mut opts, "--build-receipt-sha")?,
        ))
    } else {
        None
    };
    ensure!(opts.is_empty(), "unknown options");
    // Every receipt is rechecked before any library load, cuInit, or context.
    let admitted = admission::load(&admission_path, &admission_sha)?;
    let mut candidate_raw = None;
    let mut build_receipt = Value::Null;
    if let Some((ptx, sha, build, build_sha)) = candidate {
        ensure!(
            contract::sha_syntax(&sha) && contract::sha_syntax(&build_sha),
            "reviewed candidate/build SHA required"
        );
        let raw = io::pinned(&ptx, &sha, 8 * 1024 * 1024)?;
        let build_raw = io::pinned(&build, &build_sha, 64 * 1024)?;
        let declared = io::json(&build_raw)?;
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("angles.cu");
        let source_sha = io::hash(&io::read(&source, 1024 * 1024)?)?;
        admission::validate_build(&declared, &source_sha, &sha)?;
        build_receipt = json!({"ptx":io::receipt(&ptx,&raw)?,"build":io::receipt(&build,&build_raw)?,"declared":declared});
        candidate_raw = Some(raw);
    }
    let binary = io::library_receipt(&std::env::current_exe()?)?;
    let out = io::fresh(&out)?;
    let invocation = json!({"command":command,"binary":binary,"admission":admission_path,
        "admission_sha256":admission_sha,"candidate_build":build_receipt,
        "diagnostic_only":true,"performance_claim":false,"source_receipts":admitted["sources"],
        "libraries":admitted["libraries"],"model_revision":admitted["checkpoint"]["model_revision"],
        "official_revision":admitted["official_revision"],"missing_reference_payloads":admitted["missing_reference_payloads"]});
    io::save_json(&out, "invocation.json", &invocation)?;
    let cuda = Path::new(
        admitted["libraries"]["cuda"]["path"]
            .as_str()
            .context("CUDA library receipt")?,
    );
    let mut driver = match Driver::open(cuda) {
        Ok(d) => d,
        Err(e) => {
            summary(
                &out,
                &json!({"status":"ERROR","phase":"CUDA open","error":format!("{e:#}")}),
            )?;
            return Err(e);
        }
    };
    let result = if command == "run-rope" {
        rope::run(
            &mut driver,
            &admitted,
            candidate_raw.as_ref().unwrap(),
            &out,
        )
    } else {
        fc2::run(&mut driver, &admitted, &out)
    };
    let identity = driver.identity.clone();
    let peak = driver.peak_bytes;
    let guards = driver.guards();
    let close = driver.close();
    match (result, guards, close) {
        (Ok(v), Ok(()), Ok(())) => {
            let exact = v["reference_exact"].as_bool() == Some(true);
            summary(
                &out,
                &json!({"status":if exact{"EXACT"}else{"REFERENCE_MISMATCH"},
                "native_and_reference":v,"gpu":identity,"peak_owned_device_bytes":peak,"device_cap_bytes":134217728,
                "guards_unchanged":true,"cleanup":"SUCCESS","invocation":invocation,
                "full_encoder_qualified":false,"original_gate_unchanged":{"cosine_min":0.999,"worst_row_min":0.995,"relative_l2_max":0.05}}),
            )?;
            Ok(exact)
        }
        (result, guards, close) => {
            let error = format!("run={result:?}; guards={guards:?}; cleanup={close:?}");
            summary(
                &out,
                &json!({"status":"ERROR","error":error,"gpu":identity,
                "peak_owned_device_bytes":peak,"invocation":invocation,"full_encoder_qualified":false}),
            )?;
            anyhow::bail!("{error}")
        }
    }
}
fn main() -> ExitCode {
    match execute() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(2),
        Err(e) => {
            eprintln!("{e:#}");
            ExitCode::from(1)
        }
    }
}
