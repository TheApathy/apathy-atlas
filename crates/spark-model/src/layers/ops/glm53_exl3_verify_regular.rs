// SPDX-License-Identifier: AGPL-3.0-only
//! Experimental verifier-only regular GEMM admission; no numerical claim.
use std::ffi::OsStr;

pub(super) fn selected(
    value: Option<&OsStr>,
    exact: bool,
    prefill: bool,
    rows: u32,
) -> Result<bool, &'static str> {
    let enabled = match value {
        None => false,
        Some(value) if value == OsStr::new("0") => false,
        Some(value) if value == OsStr::new("1") => true,
        Some(_) => return Err("ATLAS_GLM53_EXL3_VERIFY_REGULAR must be absent or exactly 0 or 1"),
    };
    Ok(enabled && exact && !prefill && (2..=8).contains(&rows))
}
