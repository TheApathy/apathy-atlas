// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};
use atlas_core::config::{ModelConfig, parse_config};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

pub const REVISION: &str = "c171bea574201ff25530256fbd63626c7fd20f3c";
pub const CONFIG_SHA: &str = "28a07138554196d7de70cfb193eb63bf51c39bb42ae4cd4303ba16610b5b1bf5";
pub const INDEX_SHA: &str = "f4df075b9b9d77af5fe1482624a33466a7b5418f96c9d31f53339c34d72338d8";
pub const VISUAL_BYTES: usize = 932_786_176;
pub const VISUAL_TENSORS: usize = 267;

pub struct Manifest {
    pub config: ModelConfig,
    pub selected: BTreeSet<String>,
    pub report: Value,
}

pub fn visual_name(name: &str) -> bool {
    name.starts_with("vision.")
        || name.starts_with("aligner.")
        || matches!(
            name,
            "image_start" | "image_end" | "image_pad" | "image_newline"
        )
}

pub fn sha256_bytes(bytes: &[u8]) -> Result<String> {
    // Use the host's established SHA256 implementation; no shell interpolation.
    let mut child = Command::new("/usr/bin/sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .context("spawn sha256sum")?;
    child
        .stdin
        .take()
        .context("sha256sum stdin missing")?
        .write_all(bytes)?;
    let output = child.wait_with_output()?;
    ensure!(output.status.success(), "sha256sum failed");
    let text = std::str::from_utf8(&output.stdout)?;
    let digest = text
        .split_whitespace()
        .next()
        .context("sha256sum output missing")?;
    ensure!(
        digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
        "bad SHA256 output"
    );
    Ok(digest.into())
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let output = Command::new("/usr/bin/sha256sum")
        .arg("--")
        .arg(path)
        .output()?;
    ensure!(
        output.status.success(),
        "sha256sum failed for {}",
        path.display()
    );
    let text = std::str::from_utf8(&output.stdout)?;
    let digest = text
        .split_whitespace()
        .next()
        .context("sha256sum output missing")?;
    ensure!(
        digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
        "bad SHA256 output"
    );
    Ok(digest.into())
}

pub fn inspect(model: &Path) -> Result<Manifest> {
    ensure!(
        !model.join("extra_weights.safetensors").exists(),
        "refuse extra_weights: SafetensorsLoader's extra-file path bypasses extra_skip"
    );
    let config_path = model.join("config.json");
    let index_path = model.join("model.safetensors.index.json");
    ensure!(
        sha256_file(&config_path)? == CONFIG_SHA,
        "pinned vision config SHA256 mismatch"
    );
    ensure!(
        sha256_file(&index_path)? == INDEX_SHA,
        "pinned vision index SHA256 mismatch"
    );
    let config = parse_config(&std::fs::read_to_string(config_path)?)?;
    config
        .deepseek_vision
        .as_ref()
        .context("missing typed DeepSeek vision config")?
        .validate()?;
    let index: Value = serde_json::from_reader(std::fs::File::open(index_path)?)?;
    let map = index["weight_map"]
        .as_object()
        .context("weight map missing")?;
    ensure!(map.len() == 143_289, "unexpected checkpoint tensor count");
    let selected: BTreeSet<String> = map.keys().filter(|k| visual_name(k)).cloned().collect();
    ensure!(
        selected.len() == VISUAL_TENSORS,
        "unexpected visual tensor count"
    );
    let mut shards: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, shard) in map {
        shards
            .entry(shard.as_str().context("shard name is not a string")?.into())
            .or_default()
            .push(name.clone());
    }
    ensure!(shards.len() == 10, "expected ten checkpoint shards");
    let mut reports = Vec::new();
    let mut visual_bytes = 0usize;
    for (name, names) in shards {
        ensure!(
            !name.contains('/') && !name.contains('\\') && name.ends_with(".safetensors"),
            "unsafe shard name"
        );
        let path = model.join(&name);
        let mut file =
            std::fs::File::open(&path).with_context(|| format!("missing shard {name}"))?;
        let size = file.metadata()?.len();
        let mut first = [0u8; 8];
        file.read_exact(&mut first)?;
        let header_len = u64::from_le_bytes(first);
        ensure!(
            header_len > 0 && header_len <= 16 * 1024 * 1024 && header_len + 8 <= size,
            "invalid shard header size for {name}"
        );
        let mut raw = vec![0; header_len as usize];
        file.read_exact(&mut raw)?;
        let header: Value = serde_json::from_slice(&raw)?;
        let header = header.as_object().context("invalid safetensors header")?;
        let mut data_end = 0u64;
        for key in names {
            let tensor = header
                .get(&key)
                .with_context(|| format!("{name} missing indexed tensor {key}"))?;
            let offsets = tensor["data_offsets"]
                .as_array()
                .context("missing offsets")?;
            ensure!(offsets.len() == 2, "bad tensor offsets");
            let begin = offsets[0].as_u64().context("invalid start offset")?;
            let end = offsets[1].as_u64().context("invalid end offset")?;
            ensure!(
                begin <= end && end <= size - header_len - 8,
                "incomplete or malformed shard {name}"
            );
            data_end = data_end.max(end);
            if selected.contains(&key) {
                ensure!(tensor["dtype"] == "BF16", "{key}: expected native BF16");
                let dims = tensor["shape"].as_array().context("missing tensor shape")?;
                let elements = dims.iter().try_fold(1u64, |n, d| {
                    let d = d.as_u64().context("invalid tensor dimension")?;
                    ensure!(d > 0, "empty visual tensor");
                    n.checked_mul(d).context("visual shape overflow")
                })?;
                ensure!(
                    elements.checked_mul(2) == Some(end - begin),
                    "visual byte shape mismatch"
                );
                visual_bytes += usize::try_from(end - begin)?;
            }
        }
        ensure!(
            data_end + header_len + 8 == size,
            "truncated/trailing shard bytes: {name}"
        );
        let receipt_path = model
            .join(".cache/huggingface/download")
            .join(format!("{name}.metadata"));
        let receipt =
            std::fs::read_to_string(receipt_path).context("missing download provenance receipt")?;
        let mut lines = receipt.lines();
        ensure!(
            lines.next() == Some(REVISION),
            "download receipt revision differs"
        );
        let etag = lines.next().context("missing downloaded shard etag")?;
        ensure!(
            etag.len() == 64 && etag.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid shard etag"
        );
        reports.push(json!({"file":name,"bytes":size,"header_sha256":sha256_bytes(&raw)?,"download_etag":etag}));
    }
    ensure!(
        visual_bytes == VISUAL_BYTES,
        "unexpected visual tensor bytes {visual_bytes}"
    );
    let report = json!({"model_revision":REVISION,"config_sha256":CONFIG_SHA,"index_sha256":INDEX_SHA,
        "selected_tensors":selected.len(),"selected_bytes":visual_bytes,"complete_shards":reports,
        "payload_integrity":"download receipts; no full-shard rehash in this probe"});
    Ok(Manifest {
        config,
        selected,
        report,
    })
}

pub fn parse_grids(raw: &str) -> Result<Vec<(usize, usize)>> {
    let mut grids = Vec::new();
    for part in raw.split(',') {
        let (h, w) = part.split_once('x').context("expected HxW grid")?;
        let (h, w): (usize, usize) = (h.parse()?, w.parse()?);
        ensure!(
            h > 0 && w > 0 && h.checked_mul(w).is_some_and(|p| p <= 3456),
            "grid exceeds patch capacity"
        );
        ensure!(
            h.div_ceil(3) * w.div_ceil(3) <= 384,
            "grid exceeds aligned token capacity"
        );
        ensure!(!grids.contains(&(h, w)), "duplicate grid");
        grids.push((h, w));
    }
    ensure!(!grids.is_empty() && grids.len() <= 8, "expected 1..8 cases");
    Ok(grids)
}

pub fn make_pixels(h: usize, w: usize) -> Vec<f32> {
    (0..h * w * 588)
        .map(|i| (((i * 37 + i / 588 * 17) % 257) as i32 - 128) as f32 / 128.0)
        .collect()
}

pub fn bf16_stats(raw: &[u8]) -> Result<Value> {
    ensure!(
        !raw.is_empty() && raw.len().is_multiple_of(2),
        "invalid BF16 output byte length"
    );
    let mut max_abs = 0.0_f64;
    let mut sum = 0.0;
    let mut sumsq = 0.0;
    for pair in raw.chunks_exact(2) {
        let x = f32::from_bits((u16::from_le_bytes([pair[0], pair[1]]) as u32) << 16) as f64;
        ensure!(x.is_finite(), "nonfinite encoder output");
        max_abs = max_abs.max(x.abs());
        sum += x;
        sumsq += x * x;
    }
    Ok(
        json!({"count":raw.len()/2,"max_abs":max_abs,"mean":sum/(raw.len()/2) as f64,"sum_squares":sumsq}),
    )
}

pub fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
