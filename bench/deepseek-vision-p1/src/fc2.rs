// SPDX-License-Identifier: AGPL-3.0-only
use crate::{contract, driver::Driver, io, lt, pins};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::path::Path;
pub fn run(d: &mut Driver, admission: &Value, out: &Path) -> Result<Value> {
    let cases = admission["cases"].as_array().context("admitted cases")?;
    let case = cases
        .iter()
        .find(|c| c["name"] == "grid-4x5")
        .context("4x5 case")?;
    let input = io::verify_receipt(&case["stages"]["swiglu"]["file"])?;
    let weight = io::verify_receipt(&admission["weight"])?;
    ensure!(
        input.len() == 20 * 2816 * 2 && weight.len() == 1024 * 2816 * 2,
        "fc2 fixed dimensions"
    );
    let a = d.upload(&input)?;
    let w = d.upload(&weight)?;
    let native = d.allocate(20 * 1024 * 2, 0xff)?;
    let ptx = io::pinned(
        &Path::new(pins::PTX_ROOT).join(pins::PTX_FILES[1].0),
        pins::PTX_FILES[1].1,
        8 * 1024 * 1024,
    )?;
    let kernel = d.function(&ptx, "deepseek_vision_linear")?;
    d.launch(kernel, contract::fc2_launch(a.ptr, w.ptr, native.ptr)?)?;
    let raw = d.read(native)?;
    let control = io::save(out, "fc2-native-control.bf16", &raw)?;
    contract::check_bf16(&raw)?;
    ensure!(
        control["sha256"] == case["stages"]["fc2"]["file"]["sha256"],
        "fc2 native control did not replay; candidate blocked"
    );
    ensure!(
        d.read(a)? == input && d.read(w)? == weight,
        "native fc2 modified operands"
    );
    d.guards()?;
    io::save_json(
        out,
        "native-controls.json",
        &json!({"status":"EXACT","output":control}),
    )?;
    let candidate = d.allocate(20 * 1024 * 2, 0xff)?;
    let library = Path::new(
        admission["libraries"]["cublaslt"]["path"]
            .as_str()
            .context("Lt library path")?,
    );
    let dispatch = lt::fc2(d, library, a, w, candidate)?;
    let raw = d.read(candidate)?;
    let output = io::save(out, "fc2-cublaslt-candidate.bf16", &raw)?;
    contract::check_bf16(&raw)?;
    ensure!(
        d.read(a)? == input && d.read(w)? == weight,
        "candidate fc2 modified operands"
    );
    ensure!(
        io::hash(&d.read(native)?)? == control["sha256"],
        "candidate modified native control"
    );
    d.guards()?;
    Ok(
        json!({"kind":"same-input-fc2","native_controls_exact":true,"control":control,
        "candidate":output,"dispatch":dispatch,"reference_sha256":pins::FC2_REFERENCE,
        "reference_exact":output["sha256"]==pins::FC2_REFERENCE,
        "reference_modes":"retained default/full references have the same fc2 SHA",
        "reference_payload_available":false,"reference_comparison":"exact retained SHA256 only",
        "production_encoder_changed":false,"full_encoder_qualified":false}),
    )
}
