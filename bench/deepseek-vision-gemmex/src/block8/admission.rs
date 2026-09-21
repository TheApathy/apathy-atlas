// SPDX-License-Identifier: AGPL-3.0-only
//! CPU-only selected block-8 provenance. No teacher output is substituted.
use crate::{contract, weight};
use anyhow::{Context, Result, ensure};
use deepseek_vision_p1::{
    contract::{check_bf16, sha_syntax},
    io, pins,
};
use serde_json::{Value, json};
use std::path::Path;

const SOURCES: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "BLOCK8.md",
    "src/bin/deepseek_vision_block8.rs",
    "src/contract.rs",
    "src/protocol.rs",
    "src/abi.rs",
    "src/weight.rs",
    "src/metrics.rs",
    "src/block8/contract.rs",
    "src/block8/report.rs",
    "src/block8/lt_plan.rs",
    "src/block8/lt.rs",
    "src/block8/admission.rs",
    "src/block8/runner.rs",
    "tests/block8_contract.rs",
    "tests/block8_lt.rs",
    "tests/block8_report.rs",
    "tests/block8_wiring.rs",
];
fn pinned(path: &Path, sha: &str, cap: usize, files: &mut Vec<Value>) -> Result<Vec<u8>> {
    let raw = io::pinned(path, sha, cap)?;
    files.push(io::receipt(path, &raw)?);
    Ok(raw)
}
fn collect(model: &Path, corpus: &Path) -> Result<(Value, Vec<u8>)> {
    let model = io::absolute(model)?;
    let corpus = io::absolute(corpus)?;
    let bench = io::absolute(Path::new(env!("CARGO_MANIFEST_DIR")))?;
    let repo = io::absolute(&bench.join("../.."))?;
    let mut files = Vec::new();
    pinned(
        &model.join("config.json"),
        pins::CONFIG_SHA,
        4 * 1024 * 1024,
        &mut files,
    )?;
    let index = io::json(&pinned(
        &model.join("model.safetensors.index.json"),
        pins::INDEX_SHA,
        io::MAX_FILE,
        &mut files,
    )?)?;
    let top = io::json(&pinned(
        &corpus.join("atlas-manifest.json"),
        contract::MANIFEST_SHA,
        4 * 1024 * 1024,
        &mut files,
    )?)?;
    let stage_dir = corpus.join("grid-4x5-stages");
    let stages = io::json(&pinned(
        &stage_dir.join("manifest.json"),
        contract::STAGES_SHA,
        1024 * 1024,
        &mut files,
    )?)?;
    let selected_capture = contract::validate_capture(&top, &stages)?;
    ensure!(
        selected_capture.input_sha256 == contract::INPUT_SHA
            && selected_capture.native_sha256 == contract::NATIVE_SHA,
        "selected capture pins"
    );
    let input = pinned(
        &stage_dir.join("block-08-norm2.bf16"),
        contract::INPUT_SHA,
        contract::INPUT_BYTES,
        &mut files,
    )?;
    ensure!(input.len() == contract::INPUT_BYTES, "input byte extent");
    check_bf16(&input)?;
    let input_receipt = files.last().context("input receipt")?.clone();
    let native = pinned(
        &stage_dir.join("block-08-fc1.bf16"),
        contract::NATIVE_SHA,
        contract::OUTPUT_BYTES,
        &mut files,
    )?;
    ensure!(native.len() == contract::OUTPUT_BYTES, "native byte extent");
    check_bf16(&native)?;
    let native_receipt = files.last().context("native receipt")?.clone();
    let (name, sha) = pins::PRODUCTION_FILES[3];
    pinned(&repo.join(name), sha, 1024 * 1024, &mut files)?;
    let ptx = pinned(
        Path::new(contract::PTX_PATH),
        contract::PTX_SHA,
        8 * 1024 * 1024,
        &mut files,
    )?;
    ensure!(!ptx.is_empty(), "empty native PTX");
    let ptx_receipt = files.last().context("PTX receipt")?.clone();
    let mut sources = Vec::new();
    for (root, names) in [
        (bench.clone(), SOURCES),
        (
            io::absolute(&bench.join("../deepseek-vision-p1"))?,
            pins::SOURCE_FILES,
        ),
    ] {
        for name in names {
            let path = root.join(name);
            sources.push(io::receipt(&path, &io::read(&path, 1024 * 1024)?)?);
        }
    }
    let checkpoint = &top["checkpoint"];
    let (raw, selected) = weight::selected(&model, &index, checkpoint)?;
    let libraries = json!({
        "cuda":io::library_receipt(Path::new("/usr/lib/aarch64-linux-gnu/libcuda.so.1"))?,
        "cublas":io::library_receipt(Path::new("/usr/local/cuda/lib64/libcublas.so.13"))?,
        "cublaslt":io::library_receipt(Path::new("/usr/local/cuda/lib64/libcublasLt.so.13"))?});
    let v = json!({"schema":"atlas-dsv-block8-fc1-admission-v1", "status":"CPU_ADMITTED",
        "diagnostic_only":true, "teacher_reference_available":false,
        "teacher_reference_sha256":null, "full_encoder_qualified":false, "performance_qualified":false,
        "model":model, "corpus":corpus, "checkpoint":checkpoint, "files":files, "sources":sources,
        "input":input_receipt, "native_control":native_receipt, "native_ptx":ptx_receipt,
        "weight_selection":selected, "libraries":libraries,
        "geometry":{"rows":20,"outputs":5632,"inner":1024}, "repeats":2,
        "owned_device_cap_bytes":contract::OWNED_DEVICE_CAP,
        "owned_device_bytes":contract::owned_device_bytes()?,
        "gemmex_workspace_bytes":contract::Fc1Plan::new().workspace_bytes,
        "lt_workspace_bytes":contract::LT_WORKSPACE,
        "gate":"native byte-exact twice before four diagnostic modes; finite, reset and redzone checks",
        "limitations":[
            "no authoritative block-8 FC1 teacher payload or SHA is available",
            "candidate metrics compare the captured native output, not a teacher",
            "default Lt preference is not proof of the historical Torch backend",
            "no model runtime, full encoder or performance qualification is performed",
            "only the selected BF16 tensor is read; no full shard payload rehash",
            "owned device cap excludes CUDA context and library internal allocations",
            "pinned requested libraries do not exhaustively inventory transitive dependencies",
            "legacy Driver/Lt cleanup attempts fences but still frees after fence failure; not quarantine",
            "local receipt readers require trusted immutable files; not adversarial filesystem isolation"]});
    Ok((v, raw))
}
pub fn inspect(model: &Path, corpus: &Path, out: &Path) -> Result<Value> {
    let (mut admission, weight) = collect(model, corpus)?;
    let out = io::fresh(out)?;
    admission["weight"] = io::save(&out, "block8-fc1-weight.bf16", &weight)?;
    io::save_json(&out, "admission.json", &admission)
}
pub fn load(path: &Path, expected: &str) -> Result<Value> {
    ensure!(sha_syntax(expected), "reviewed admission SHA required");
    let admission = io::json(&io::pinned(path, expected, 4 * 1024 * 1024)?)?;
    ensure!(
        admission["schema"] == "atlas-dsv-block8-fc1-admission-v1"
            && admission["status"] == "CPU_ADMITTED",
        "admission schema/status"
    );
    let model = Path::new(admission["model"].as_str().context("model path")?);
    let corpus = Path::new(admission["corpus"].as_str().context("corpus path")?);
    // Reconstruct source/library/header/payload contracts instead of trusting
    // saved status fields. The checked copy must equal the selected tensor.
    let (mut current, selected) = collect(model, corpus)?;
    let saved_path = io::absolute(path)?
        .parent()
        .context("admission parent")?
        .join("block8-fc1-weight.bf16");
    ensure!(
        admission["weight"]["path"].as_str() == saved_path.to_str(),
        "weight copy path"
    );
    let saved = io::verify_receipt(&admission["weight"])?;
    ensure!(saved == selected, "admitted selected weight copy changed");
    current["weight"] = io::receipt(&saved_path, &saved)?;
    ensure!(
        current == admission,
        "source/data/library contracts no longer match admission"
    );
    Ok(admission)
}
