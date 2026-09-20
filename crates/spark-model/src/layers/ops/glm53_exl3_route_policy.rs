// SPDX-License-Identifier: AGPL-3.0-only

//! Value-only EXL3 route layout, latched once by the owning target.

use anyhow::{Context, Result, bail, ensure};
use std::ffi::OsStr;

const MAX_ROWS: u32 = 2048;
const SMALL_ROWS: u32 = 8;
const ROUTES: u64 = 8;
const ROUTE_BYTES: u64 = 4096 * size_of::<f32>() as u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53Exl3RoutePolicy {
    private: bool,
    prefill: bool,
}

impl Glm53Exl3RoutePolicy {
    /// Preserve the pre-policy allocation and route layout for legacy callers.
    pub const fn legacy() -> Self {
        Self {
            private: false,
            prefill: false,
        }
    }

    pub fn parse(private: Option<&OsStr>, prefill: Option<&OsStr>) -> Result<Self> {
        let mut policy = Self::legacy();
        policy.private = flag("ATLAS_GLM53_EXL3_ROUTE_PRIVATE", private)?;
        policy.prefill = flag("ATLAS_GLM53_EXL3_ROUTE_PRIVATE_PREFILL", prefill)?;
        ensure!(
            !policy.prefill || policy.private,
            "private prefill requires ATLAS_GLM53_EXL3_ROUTE_PRIVATE=1"
        );
        Ok(policy)
    }

    pub fn private_for(self, rows: u32) -> Result<bool> {
        validate_rows(rows)?;
        Ok(self.private && (self.prefill || rows <= SMALL_ROWS))
    }

    pub fn private_bytes(self, rows: u32) -> Result<usize> {
        validate_rows(rows)?;
        let rows = if self.prefill {
            rows
        } else {
            rows.min(SMALL_ROWS)
        };
        usize::try_from(u64::from(rows) * ROUTES * ROUTE_BYTES)
            .context("GLM EXL3 private route extent overflow")
    }

    pub const fn scratch_private_bytes(self) -> u64 {
        let rows = if self.prefill { MAX_ROWS } else { SMALL_ROWS };
        rows as u64 * ROUTES * ROUTE_BYTES
    }

    pub const fn prefill_enabled(self) -> bool {
        self.prefill
    }

    pub fn validate_moe_mode(self, mode: Option<&OsStr>) -> Result<()> {
        match mode
            .map(|v| v.to_str().context("GLM EXL3 MoE mode must be UTF-8"))
            .transpose()?
        {
            None | Some("fused") => Ok(()),
            Some("serial-reference") if !self.prefill => Ok(()),
            Some("serial-reference") => bail!("private prefill requires fused GLM EXL3 MoE"),
            Some(_) => bail!("ATLAS_GLM53_EXL3_MOE must be fused or serial-reference"),
        }
    }
}

fn validate_rows(rows: u32) -> Result<()> {
    ensure!(
        (1..=MAX_ROWS).contains(&rows),
        "GLM EXL3 routes admit 1..={MAX_ROWS} rows"
    );
    Ok(())
}

fn flag(name: &str, value: Option<&OsStr>) -> Result<bool> {
    match value
        .map(|v| v.to_str().with_context(|| format!("{name} must be UTF-8")))
        .transpose()?
    {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => bail!("{name} must be exactly 0 or 1"),
    }
}
