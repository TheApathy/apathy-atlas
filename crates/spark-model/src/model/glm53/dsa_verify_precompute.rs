// SPDX-License-Identifier: AGPL-3.0-only

//! Explicit admission for reordering stateless DSA inputs in exact verification.
//! This does not enable batched native arithmetic or a different causal stage.

use anyhow::{Result, bail, ensure};

pub(super) fn select(
    value: Option<&str>,
    rows: u32,
    exact_verify: bool,
    prefill: bool,
    row_exact: bool,
) -> Result<bool> {
    let requested = match value {
        None | Some("0") => false,
        Some("1") => true,
        Some(_) => bail!("ATLAS_GLM53_DSA_VERIFY_PRECOMPUTE must be exactly 0 or 1"),
    };
    // The new flag never selects ordinary decode or either prefill path.
    // M1 replay already uses the scalar causal stage and needs no precompute.
    if !requested || !exact_verify || rows == 1 {
        return Ok(false);
    }
    ensure!(
        !prefill,
        "GLM DSA verifier precompute cannot nest inside prefill"
    );
    ensure!(
        (2..=8).contains(&rows),
        "GLM DSA verifier precompute needs 2..=8 rows"
    );
    ensure!(
        row_exact,
        "GLM DSA verifier precompute requires ATLAS_GLM53_EXACT_WIDE_ROWEXACT=1"
    );
    Ok(true)
}
