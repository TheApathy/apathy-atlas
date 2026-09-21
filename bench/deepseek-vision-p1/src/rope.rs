// SPDX-License-Identifier: AGPL-3.0-only
use crate::{
    contract::{self, Grid},
    driver::{Buffer, Driver},
    io, pins,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::path::Path;
struct Ready {
    grid: Grid,
    case: Value,
    qkv: Buffer,
    host_angles: Buffer,
    angles: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
}
fn read_outputs(d: &Driver, r: &Ready, out: &Path, label: &str) -> Result<Value> {
    let mut values = serde_json::Map::new();
    for (name, b) in [("query", r.q), ("key", r.k), ("value", r.v)] {
        let raw = d.read(b)?;
        let receipt = io::save(out, &format!("{}-{label}-{name}.bf16", r.grid.name()), &raw)?;
        contract::check_bf16(&raw)?;
        values.insert(name.to_owned(), receipt);
    }
    Ok(Value::Object(values))
}
pub fn run(d: &mut Driver, admission: &Value, ptx: &[u8], out: &Path) -> Result<Value> {
    let native = io::pinned(
        &Path::new(pins::PTX_ROOT).join(pins::PTX_FILES[0].0),
        pins::PTX_FILES[0].1,
        8 * 1024 * 1024,
    )?;
    let rope = d.function(&native, "deepseek_vision_rope")?;
    let mut ready = Vec::new();
    let mut controls = Vec::new();
    for case in admission["cases"].as_array().context("admitted cases")? {
        let g = Grid::new(
            case["grid"][0].as_u64().context("grid h")? as usize,
            case["grid"][1].as_u64().context("grid w")? as usize,
        )?;
        let raw = io::verify_receipt(&case["stages"]["qkv"]["file"])?;
        ensure!(raw.len() == g.patches() * 3072 * 2, "QKV payload shape");
        let qkv = d.upload(&raw)?;
        // Bench-owned historical host control; native hash replay catches any
        // compiler/libm drift. Current production angles are generated on CUDA.
        let angles = crate::host_angles::legacy_angles(g.h, g.w)
            .context("unsupported historical host-angle grid")?;
        let angles: Vec<u8> = angles.into_iter().flat_map(f32::to_le_bytes).collect();
        let host_receipt = io::save(out, &format!("{}-host-angles.f32", g.name()), &angles)?;
        contract::check_f32(&angles)?;
        let host_angles = d.upload(&angles)?;
        let angle_output = d.allocate(angles.len(), 0xff)?;
        let bytes = g.patches() * 1024 * 2;
        let r = Ready {
            grid: g,
            case: case.clone(),
            qkv,
            host_angles,
            angles: angle_output,
            q: d.allocate(bytes, 0xff)?,
            k: d.allocate(bytes, 0xff)?,
            v: d.allocate(bytes, 0xff)?,
        };
        d.launch(
            rope,
            contract::rope_launch(g, [r.qkv.ptr, r.q.ptr, r.k.ptr, r.v.ptr, r.host_angles.ptr])?,
        )?;
        let outputs = read_outputs(d, &r, out, "native-control")?;
        for name in ["query", "key", "value"] {
            ensure!(
                outputs[name]["sha256"] == case["stages"][name]["file"]["sha256"],
                "{} native-control {name} did not reproduce retained bytes; candidate blocked",
                g.name()
            );
        }
        ensure!(
            d.read(qkv)? == raw && d.read(host_angles)? == angles,
            "native control mutated input/angles"
        );
        d.guards()?;
        controls.push(
            json!({"case":g.name(),"native_exact":true,"angles":host_receipt,"outputs":outputs}),
        );
        ready.push(r);
    }
    io::save_json(
        out,
        "native-controls.json",
        &json!({"status":"EXACT","cases":controls}),
    )?;
    // Candidate module is only loaded after every native control is exact.
    let angle_kernel = d.function(ptx, "dsv_p1_angles")?;
    let mut candidates = Vec::new();
    let mut exact = true;
    for r in &ready {
        d.launch(angle_kernel, contract::angles_launch(r.grid, r.angles.ptr)?)?;
        let angles = d.read(r.angles)?;
        let angle_receipt = io::save(
            out,
            &format!("{}-candidate-angles.f32", r.grid.name()),
            &angles,
        )?;
        contract::check_f32(&angles)?;
        d.launch(
            rope,
            contract::rope_launch(r.grid, [r.qkv.ptr, r.q.ptr, r.k.ptr, r.v.ptr, r.angles.ptr])?,
        )?;
        let outputs = read_outputs(d, r, out, "candidate")?;
        let mut matches = serde_json::Map::new();
        for name in ["query", "key", "value"] {
            let expected = &r.case["stages"][name]["reference_sha256"];
            let pass = outputs[name]["sha256"] == *expected;
            exact &= pass;
            matches.insert(
                name.to_owned(),
                json!({"exact":pass,"expected_sha256":expected}),
            );
        }
        ensure!(
            io::hash(&d.read(r.qkv)?)? == r.case["stages"]["qkv"]["file"]["sha256"],
            "candidate mutated QKV"
        );
        ensure!(
            d.read(r.angles)? == angles,
            "RoPE mutated candidate angle input"
        );
        let host_expected = crate::host_angles::legacy_angles(r.grid.h, r.grid.w)
            .context("unsupported historical host-angle grid")?;
        let host_expected: Vec<u8> = host_expected
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        ensure!(
            d.read(r.host_angles)? == host_expected,
            "candidate mutated host angle control"
        );
        d.guards()?;
        candidates.push(json!({"case":r.grid.name(),"angles":angle_receipt,"outputs":outputs,"reference_matches":matches}));
    }
    Ok(
        json!({"kind":"same-QKV-rope","native_controls_exact":true,"reference_exact":exact,
        "controls":controls,"candidates":candidates,"angle_reference_payload_available":false,
        "reference_comparison":"retained official Q/K/V SHA256; no angle-reference payload or per-element metrics",
        "production_encoder_changed":false,"full_encoder_qualified":false}),
    )
}
