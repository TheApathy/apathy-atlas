// SPDX-License-Identifier: AGPL-3.0-only
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use spark_model::model::glm53::{ProbeCapture, ProbeLayout, ProbeStage};
use std::{fs::OpenOptions, io::Write, path::Path, process::Command};

pub fn sha256(path: &Path) -> Result<String> {
    let output = Command::new("sha256sum").arg(path).output()?;
    ensure!(
        output.status.success(),
        "sha256sum failed for {}",
        path.display()
    );
    let text = std::str::from_utf8(&output.stdout)?;
    let hash = text
        .split_whitespace()
        .next()
        .context("empty SHA receipt")?;
    ensure!(
        hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid SHA receipt"
    );
    Ok(hash.into())
}
pub fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
pub fn json_file(path: &Path, value: &Value) -> Result<()> {
    write(path, &serde_json::to_vec_pretty(value)?)
}

pub fn dump(root: &Path, capture: &ProbeCapture) -> Result<Value> {
    std::fs::create_dir(root)?;
    let mut frames = Vec::new();
    for frame in capture.retained_frames() {
        let name = format!("{}.bin", frame.stage.name());
        let path = root.join(&name);
        write(&path, &frame.bytes)?;
        frames.push(json!({"stage":frame.stage.name(), "file":name,
            "bytes":frame.bytes.len(), "sha256":sha256(&path)?}));
    }
    let value = json!({"context":capture.context(), "pending":capture.pending(),
        "complete":capture.frames().is_ok(), "frames":frames});
    json_file(&root.join("manifest.json"), &value)?;
    Ok(value)
}

fn raw_diff(a: &[u8], b: &[u8], bf16: bool) -> Value {
    let mismatches = a.iter().zip(b).filter(|(a, b)| a != b).count();
    let first = a.iter().zip(b).position(|(a, b)| a != b);
    let mut error2 = 0.0f64;
    let mut norm2 = 0.0f64;
    let mut max_abs = 0.0f64;
    if bf16 {
        for (a, b) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
            let a = f32::from_bits(u32::from(u16::from_le_bytes([a[0], a[1]])) << 16) as f64;
            let b = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16) as f64;
            error2 += (a - b) * (a - b);
            norm2 += a * a;
            max_abs = max_abs.max((a - b).abs());
        }
    }
    json!({"equal":mismatches==0,"different_bytes":mismatches,"first_byte":first,
        "first_reference":first.map(|p|a[p]),"first_candidate":first.map(|p|b[p]),
        "max_abs":bf16.then_some(max_abs),
        "relative_l2":if bf16 && norm2 > 0.0 {Some((error2/norm2).sqrt())} else {None}})
}

pub fn compare(
    layout: &ProbeLayout,
    reference: &ProbeCapture,
    candidate: &ProbeCapture,
) -> Result<Value> {
    ensure!(
        reference.context() == candidate.context(),
        "paired contexts differ"
    );
    let a = reference.frames()?;
    let b = candidate.frames()?;
    ensure!(a.len() == b.len(), "paired frame counts differ");
    let regions = layout.kv_regions(reference.context(), 8, 8 * 128 * 2)?;
    let mut equal = true;
    let mut stages = Vec::new();
    for (a, b) in a.iter().zip(b) {
        ensure!(
            a.stage == b.stage && a.bytes.len() == b.bytes.len(),
            "paired frame identity/extent differs"
        );
        let same = a.bytes == b.bytes;
        equal &= same;
        let detail = match a.stage {
            ProbeStage::KeyCache(_) | ProbeStage::ValueCache(_) => {
                let mut items = serde_json::Map::new();
                for (name, range) in [
                    ("committed", regions.committed.clone()),
                    ("noise", regions.noise.clone()),
                    ("unused", regions.unused.clone()),
                ] {
                    items.insert(
                        name.into(),
                        raw_diff(&a.bytes[range.clone()], &b.bytes[range], true),
                    );
                }
                Value::Object(items)
            }
            _ => raw_diff(&a.bytes, &b.bytes, a.stage != ProbeStage::DraftIds),
        };
        stages.push(json!({"stage":a.stage.name(),"equal":same,"detail":detail}));
    }
    Ok(
        json!({"exact":equal,"context":reference.context(),"stages":stages,
        "qualification":"raw exactness only; no throughput or natural acceptance claim"}),
    )
}

/// Keep arithmetic drift and cache invariance as independent, raw-exact gates.
pub fn compare_three(
    layout: &ProbeLayout,
    original: &ProbeCapture,
    stable_full: &ProbeCapture,
    stable_cached: &ProbeCapture,
) -> Result<Value> {
    let original_full = compare(layout, original, stable_full)?;
    let original_cached = compare(layout, original, stable_cached)?;
    let cache = compare(layout, stable_full, stable_cached)?;
    Ok(json!({
        "schema":"atlas.glm53.projection-three-way.v1",
        "stable_cache_exact":cache["exact"].as_bool().context("cache exact receipt missing")?,
        "original_baseline_exact":original_full["exact"] == true && original_cached["exact"] == true,
        "original_vs_stable_full":original_full,
        "original_vs_stable_cached":original_cached,
        "stable_full_vs_cached":cache,
        "quality_qualified":false,"speed_qualified":false,
        "qualification":"stable-family cache parity only; original arithmetic drift remains separate"
    }))
}
