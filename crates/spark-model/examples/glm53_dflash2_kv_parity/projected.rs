// SPDX-License-Identifier: AGPL-3.0-only
//! Exact raw input relationships only; this is not a projection math oracle.
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub struct InputPair<'a> {
    pub before: &'a [u8],
    pub after: &'a [u8],
}

pub fn compare_inputs(
    reference: InputPair<'_>,
    candidate: InputPair<'_>,
    previous: Option<&[u8]>,
    row_bytes: usize,
) -> Result<Value> {
    let bytes = reference.before.len();
    ensure!(
        row_bytes > 0 && row_bytes % 2 == 0 && bytes > 0 && bytes % row_bytes == 0,
        "invalid projected input row extent"
    );
    for raw in [
        reference.before,
        reference.after,
        candidate.before,
        candidate.after,
    ] {
        ensure!(raw.len() == bytes, "projected input pair extents differ");
        ensure!(finite(raw), "nonfinite BF16 projected input");
    }
    if let Some(previous) = previous {
        ensure!(
            !previous.is_empty() && previous.len() <= bytes && previous.len() % row_bytes == 0,
            "invalid previous projected prefix"
        );
        ensure!(finite(previous), "nonfinite previous projected input");
    }
    let pair_before_exact = reference.before == candidate.before;
    let pair_after_exact = reference.after == candidate.after;
    let reference_immutable = reference.before == reference.after;
    let candidate_immutable = candidate.before == candidate.after;
    let previous_prefix_exact = previous.map(|old| {
        [
            reference.before,
            reference.after,
            candidate.before,
            candidate.after,
        ]
        .iter()
        .all(|raw| &raw[..old.len()] == old)
    });
    Ok(json!({
        "exact": pair_before_exact && pair_after_exact && reference_immutable
            && candidate_immutable && previous_prefix_exact.unwrap_or(true),
        "pair_before_exact": pair_before_exact,
        "pair_after_exact": pair_after_exact,
        "reference_immutable": reference_immutable,
        "candidate_immutable": candidate_immutable,
        "previous_prefix_exact": previous_prefix_exact,
        "previous_rows": previous.map(|v| v.len() / row_bytes),
        "rows": bytes / row_bytes, "row_bytes": row_bytes, "bytes": bytes,
        "qualification": "raw input identity only; no projection or cache parity claim"
    }))
}

fn finite(bytes: &[u8]) -> bool {
    bytes
        .chunks_exact(2)
        .all(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 != 0x7f80)
}
