// SPDX-License-Identifier: AGPL-3.0-only
//! Diagnostic I/O boundary: retain each completed gate and a bounded failure sample.
use super::compare::compare_snapshots;
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs::OpenOptions, io::Write, path::Path};

pub fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

pub fn record_comparison(
    directory: &Path,
    label: &str,
    reference: &BTreeMap<String, Vec<u8>>,
    candidate: &BTreeMap<String, Vec<u8>>,
) -> Result<Value> {
    ensure!(
        !label.is_empty()
            && label.len() <= 32
            && label.bytes().all(|b| b.is_ascii_lowercase() || b == b'_'),
        "invalid diagnostic artifact label"
    );
    let mut report = compare_snapshots(reference, candidate)?;
    report["retained_raw"] = Value::Null;
    let first = report["regions"]
        .as_array()
        .context("missing region report")?
        .iter()
        .find(|region| region["different_bytes"].as_u64().is_some_and(|n| n > 0))
        .and_then(|region| region["name"].as_str())
        .map(str::to_owned);
    if let Some(name) = first {
        let a = &reference[&name];
        let b = &candidate[&name];
        ensure!(
            a.len() <= 32 * 1024 * 1024,
            "first mismatched region exceeds32MiB"
        );
        let reference_file = format!("{label}-reference.bin");
        let candidate_file = format!("{label}-candidate.bin");
        write_new(&directory.join(&reference_file), a)?;
        write_new(&directory.join(&candidate_file), b)?;
        report["retained_raw"] = json!({"region":name,"reference_file":reference_file,
            "candidate_file":candidate_file,"bytes_each":a.len(),
            "scope":"first mismatched region only; remaining regions retain hashes and counts"});
    }
    write_new(
        &directory.join(format!("{label}.json")),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}
