// SPDX-License-Identifier: AGPL-3.0-only
//! CPU-only provenance collection. Nothing here loads CUDA or a model runtime.
use crate::{contract, weight};
use anyhow::{Context, Result, ensure};
use deepseek_vision_p1::{contract::sha_syntax, io, pins};
use serde_json::{Value, json};
use std::path::Path;

const SOURCES: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "README.md",
    "src/main.rs",
    "src/contract.rs",
    "src/protocol.rs",
    "src/abi.rs",
    "src/weight.rs",
    "src/admission.rs",
    "src/runner.rs",
    "src/metrics.rs",
    "tests/contract.rs",
    "tests/protocol.rs",
    "tests/metrics.rs",
    "tests/admission_roundtrip.rs",
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
    let mut reports = Vec::new();
    for i in [0, 2, 3] {
        let (name, sha) = pins::CORPUS_FILES[i];
        reports.push(io::json(&pinned(
            &corpus.join(name),
            sha,
            4 * 1024 * 1024,
            &mut files,
        )?)?);
    }
    let checkpoint = &reports[0]["checkpoint"];
    ensure!(
        checkpoint["model_revision"] == pins::MODEL_REV
            && checkpoint["config_sha256"] == pins::CONFIG_SHA
            && checkpoint["index_sha256"] == pins::INDEX_SHA,
        "retained checkpoint identity"
    );
    let stage_dir = corpus.join("grid-4x5-stages");
    let stages = io::json(&pinned(
        &stage_dir.join("manifest.json"),
        pins::STAGE_MANIFESTS[1],
        1024 * 1024,
        &mut files,
    )?)?;
    let references = contract::validate_references(&stages, &reports[1], &reports[2])?;
    let input = pinned(
        &stage_dir.join("block-00-norm2.bf16"),
        &references.input_sha256,
        contract::INPUT_BYTES,
        &mut files,
    )?;
    ensure!(input.len() == contract::INPUT_BYTES, "input byte extent");
    deepseek_vision_p1::contract::check_bf16(&input)?;
    let input_receipt = files.last().unwrap().clone();
    let native = pinned(
        &stage_dir.join("block-00-fc1.bf16"),
        &references.native_sha256,
        contract::OUTPUT_BYTES,
        &mut files,
    )?;
    ensure!(
        native.len() == contract::OUTPUT_BYTES,
        "native/full reference byte extent"
    );
    deepseek_vision_p1::contract::check_bf16(&native)?;
    let native_receipt = files.last().unwrap().clone();
    pinned(
        Path::new(pins::OFFICIAL_PATH),
        pins::OFFICIAL_SHA,
        1024 * 1024,
        &mut files,
    )?;
    for (name, sha) in pins::REFERENCE_SCRIPTS {
        pinned(Path::new(name), sha, 1024 * 1024, &mut files)?;
    }
    // Only this unchanged production compunit is part of this probe; do not
    // re-admit obsolete P1 host angles or blindly refresh its historical pins.
    let (name, sha) = pins::PRODUCTION_FILES[3];
    pinned(&repo.join(name), sha, 1024 * 1024, &mut files)?;
    let (name, sha) = pins::PTX_FILES[1];
    let ptx = pinned(
        &Path::new(pins::PTX_ROOT).join(name),
        sha,
        8 * 1024 * 1024,
        &mut files,
    )?;
    let ptx_receipt = files.last().unwrap().clone();
    ensure!(!ptx.is_empty(), "empty native PTX");
    let mut sources = Vec::new();
    for (root, names) in [
        (bench.clone(), SOURCES),
        (
            io::absolute(&bench.join("../deepseek-vision-p1"))?,
            pins::SOURCE_FILES,
        ),
    ] {
        for name in names {
            let p = root.join(name);
            sources.push(io::receipt(&p, &io::read(&p, 1024 * 1024)?)?);
        }
    }
    let (raw, selected) = weight::selected(&model, &index, checkpoint)?;
    let libraries = json!({"cuda":io::library_receipt(Path::new("/usr/lib/aarch64-linux-gnu/libcuda.so.1"))?,
        "cublas":io::library_receipt(Path::new("/usr/local/cuda/lib64/libcublas.so.13"))?,
        "cublaslt":io::library_receipt(Path::new("/usr/local/cuda/lib64/libcublasLt.so.13"))?});
    let v = json!({"schema":"atlas-dsv-fc1-gemmex-admission-v1","status":"CPU_ADMITTED",
        "diagnostic_only":true,"full_encoder_qualified":false,"model":model,"corpus":corpus,
        "checkpoint":checkpoint,"official_revision":pins::OFFICIAL_REV,"files":files,"sources":sources,
        "input":input_receipt,"native_control":native_receipt,"full_reference_payload":native_receipt,
        "default_reference_payload":null,"default_reference_sha256":references.default_sha256,
        "full_reference_sha256":references.full_sha256,"reference_default":reports[1],"reference_full":reports[2],
        "native_ptx":ptx_receipt,"weight_selection":selected,"libraries":libraries,
        "geometry":{"rows":20,"outputs":5632,"inner":1024},"owned_device_cap_bytes":33554432,
        "workspace_bytes":8519680,"repeats":2,"gate":"exact reference SHA per mode; no adjustable tolerance",
        "precision_limitations":["default teacher bytes unavailable: SHA gate only, no default per-element metrics",
            "full teacher bytes are the retained native control, independently matched by full-reference report SHA",
            "historical Torch BLAS preference, tunable-op and workspace environment were not captured",
            "this source-default GemmEx candidate is not a proven historical backend until exact hashes match",
            "pinned requested CUDA/cuBLAS/Lt libraries do not exhaustively inventory transitive dependencies",
            "owned device cap excludes CUDA context and library internal allocations"]});
    Ok((v, raw))
}
pub fn inspect(model: &Path, corpus: &Path, out: &Path) -> Result<Value> {
    let (mut admission, weight) = collect(model, corpus)?;
    let out = io::fresh(out)?;
    admission["weight"] = io::save(&out, "fc1-weight.bf16", &weight)?;
    io::save_json(&out, "admission.json", &admission)
}
pub fn load(path: &Path, expected: &str) -> Result<Value> {
    ensure!(sha_syntax(expected), "reviewed admission SHA required");
    let admission = io::json(&io::pinned(path, expected, 4 * 1024 * 1024)?)?;
    ensure!(
        admission["schema"] == "atlas-dsv-fc1-gemmex-admission-v1"
            && admission["status"] == "CPU_ADMITTED",
        "admission schema/status"
    );
    let model = Path::new(admission["model"].as_str().context("model path")?);
    let corpus = Path::new(admission["corpus"].as_str().context("corpus path")?);
    // Reconstruct all contracts and receipts, including selected shard bytes;
    // never trust a recorded passed/status field as admission by itself.
    let (mut current, selected) = collect(model, corpus)?;
    let saved_path = io::absolute(path)?
        .parent()
        .context("admission parent")?
        .join("fc1-weight.bf16");
    ensure!(
        admission["weight"]["path"].as_str() == saved_path.to_str(),
        "weight copy path"
    );
    let saved = io::verify_receipt(&admission["weight"])?;
    ensure!(saved == selected, "admitted selected weight copy changed");
    current["weight"] = io::receipt(&saved_path, &saved)?;
    ensure!(
        current == admission,
        "admission no longer matches reconstructed source/data/library contracts"
    );
    Ok(admission)
}
