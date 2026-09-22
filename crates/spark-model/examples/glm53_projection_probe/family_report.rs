// SPDX-License-Identifier: AGPL-3.0-only
//! New arithmetic must agree with itself in every row schedule, not replace old.
use super::contract;
use anyhow::Result;
use serde_json::{Value, json};

pub fn compare(oracle: &[u8], arms: [&[u8]; 4]) -> Result<Value> {
    let names = [
        "forward-full",
        "forward-split",
        "reverse-full",
        "reverse-split",
    ];
    let mut exact = true;
    let mut reports = Vec::new();
    for (name, bytes) in names.into_iter().zip(arms) {
        let report = contract::compare(oracle, bytes)?;
        exact &= report["exact"] == true;
        reports.push(json!({"arm":name,"comparison":report}));
    }
    Ok(json!({"gemv_family_exact":exact,"arms":reports,
        "oracle":"sequential GEMV forward full output; original and TC differences retained separately",
        "model_quality_qualified":false,"speed_qualified":false}))
}
