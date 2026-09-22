// SPDX-License-Identifier: AGPL-3.0-only

//! Complete raw-state comparison; changed or missing regions cannot be elided.
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::collections::BTreeMap;
#[path = "../../src/weight_loader/glm53_debug_source_sha256.rs"]
mod sha256;

pub fn digest(bytes: &[u8]) -> String {
    sha256::hex(sha256::digest(bytes))
}

pub fn compare_snapshots(
    reference: &BTreeMap<String, Vec<u8>>,
    candidate: &BTreeMap<String, Vec<u8>>,
) -> Result<Value> {
    ensure!(
        !reference.is_empty() && reference.keys().eq(candidate.keys()),
        "snapshot region set differs or is empty"
    );
    let mut exact = true;
    let mut regions = Vec::with_capacity(reference.len());
    for (name, a) in reference {
        let b = &candidate[name];
        ensure!(a.len() == b.len(), "snapshot region extent differs: {name}");
        let different_bytes = a.iter().zip(b).filter(|(x, y)| x != y).count();
        let first = a.iter().zip(b).position(|(x, y)| x != y);
        exact &= different_bytes == 0;
        regions.push(json!({"name":name,"bytes":a.len(),"different_bytes":different_bytes,
            "first_byte":first,"first_reference":first.map(|i|a[i]),"first_candidate":first.map(|i|b[i]),
            "reference_sha256":digest(a),"candidate_sha256":digest(b)}));
    }
    Ok(json!({"exact":exact,"regions":regions}))
}
