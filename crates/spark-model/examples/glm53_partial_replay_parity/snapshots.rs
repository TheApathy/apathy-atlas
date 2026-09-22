// SPDX-License-Identifier: AGPL-3.0-only
//! Complete logical state, bounded reads, and explicit chronological captures.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use spark_model::model::glm53::{Glm53Exl3Model, Glm53StateProbe, StateProbeRegion};
use std::collections::BTreeMap;

const CHUNK: usize = 65_534;
const MAX_SNAPSHOT: usize = 256 * 1024 * 1024;
pub type Raw = BTreeMap<String, Vec<u8>>;
pub struct Snapshot {
    pub frame: Value,
    pub regions: Raw,
}

fn read_region(
    probe: &mut Glm53StateProbe<'_>,
    region: StateProbeRegion,
    bytes: usize,
) -> Result<Vec<u8>> {
    ensure!(
        bytes <= MAX_SNAPSHOT,
        "raw state region exceeds host budget"
    );
    let mut raw = Vec::new();
    raw.try_reserve_exact(bytes)?;
    raw.resize(bytes, 0);
    for (index, chunk) in raw.chunks_mut(CHUNK).enumerate() {
        let offset = index
            .checked_mul(CHUNK)
            .context("state read chunk offset overflow")?;
        probe.read(region, offset, chunk)?;
    }
    Ok(raw)
}

pub fn persistent(model: &mut Glm53Exl3Model, stream: u64) -> Result<Snapshot> {
    let mut probe = model.state_probe(stream)?;
    let stamp = probe.stamp();
    let frame = json!({"generation":stamp.generation(),"nonce":stamp.nonce(),
        "position":stamp.position(),"context":stamp.context(),"stream":stamp.stream()});
    let descriptors = probe.regions()?;
    ensure!(
        descriptors.len() == 135,
        "GLM committed state descriptor set changed"
    );
    let mut total = 0usize;
    for descriptor in &descriptors {
        total = total
            .checked_add(descriptor.bytes)
            .context("state snapshot aggregate overflow")?;
        ensure!(
            total <= MAX_SNAPSHOT,
            "complete state snapshot exceeds host budget"
        );
    }
    let mut regions = BTreeMap::new();
    for descriptor in descriptors {
        let name = format!("{:?}", descriptor.region);
        let bytes = read_region(&mut probe, descriptor.region, descriptor.bytes)?;
        ensure!(
            regions.insert(name, bytes).is_none(),
            "duplicate state descriptor"
        );
    }
    Ok(Snapshot { frame, regions })
}

pub fn capture_rows(model: &mut Glm53Exl3Model, stream: u64, rows: u32) -> Result<Vec<Vec<u8>>> {
    ensure!(
        (1..=8).contains(&rows),
        "fresh capture extent outside verifier width"
    );
    let mut probe = model.state_probe(stream)?;
    let mut result = Vec::new();
    for row in 0..rows {
        let mut captured = Vec::new();
        for tap in 0..5 {
            let region = StateProbeRegion::Capture { tap, row };
            let bytes = probe.region_bytes(region)?;
            ensure!(bytes == 8192, "GLM capture tap extent changed");
            let mut raw = vec![0u8; bytes];
            probe.read(region, 0, &mut raw)?;
            captured.extend_from_slice(&raw);
        }
        result.push(captured);
    }
    Ok(result)
}

pub fn capture_map(rows: Vec<Vec<u8>>) -> Raw {
    rows.into_iter()
        .enumerate()
        .map(|(row, bytes)| (format!("committed-capture-row-{row}"), bytes))
        .collect()
}

pub fn next_logits(model: &mut Glm53Exl3Model, stream: u64) -> Result<Raw> {
    let mut probe = model.state_probe(stream)?;
    let region = StateProbeRegion::Logits { row: 0 };
    let bytes = probe.region_bytes(region)?;
    Ok(BTreeMap::from([(
        "next-decode-logits".into(),
        read_region(&mut probe, region, bytes)?,
    )]))
}
