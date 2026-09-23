// SPDX-License-Identifier: AGPL-3.0-only

//! Value-only EXL3 route layout, latched once by the owning target.

use anyhow::{Context, Result, bail, ensure};
use std::ffi::OsStr;

const MAX_ROWS: u32 = 2048;
const SMALL_ROWS: u32 = 8;
const ROUTES: u64 = 8;
const ROUTE_BYTES: u64 = 4096 * size_of::<f32>() as u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VerifyGroup {
    Two,
    Four,
    Eight,
}

impl VerifyGroup {
    const fn width(self) -> u32 {
        match self {
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53Exl3RoutePolicy {
    private: bool,
    prefill: bool,
    verify_group: Option<VerifyGroup>,
    verify_staged_k32: bool,
}

impl Glm53Exl3RoutePolicy {
    /// Preserve the pre-policy allocation and route layout for legacy callers.
    pub const fn legacy() -> Self {
        Self {
            private: false,
            prefill: false,
            verify_group: None,
            verify_staged_k32: false,
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

    /// Diagnostic scheduling only. The model stores this value in its existing
    /// scratch policy; kernel construction never rereads a selected group.
    pub fn parse_with_verify_group(
        private: Option<&OsStr>,
        prefill: Option<&OsStr>,
        group: Option<&OsStr>,
        exact_verify: Option<&OsStr>,
    ) -> Result<Self> {
        let mut policy = Self::parse(private, prefill)?;
        policy.verify_group = match group {
            None => None,
            Some(v) if v == OsStr::new("2") => Some(VerifyGroup::Two),
            Some(v) if v == OsStr::new("4") => Some(VerifyGroup::Four),
            Some(v) if v == OsStr::new("8") => Some(VerifyGroup::Eight),
            Some(_) => bail!(
                "ATLAS_GLM53_EXL3_MOE_VERIFY_GROUP_WIDTH must be absent or exactly 2, 4, or 8"
            ),
        };
        ensure!(
            policy.verify_group.is_none()
                || (policy.private && exact_verify == Some(OsStr::new("1"))),
            "GLM EXL3 verifier scheduling requires ROUTE_PRIVATE=1 and EXACT_VERIFY=1"
        );
        Ok(policy)
    }

    /// Optional diagnostic, latched with the same model-owned route policy.
    pub fn with_verify_staged_k32(
        mut self,
        value: Option<&OsStr>,
        exact_verify: Option<&OsStr>,
    ) -> Result<Self> {
        self.verify_staged_k32 = flag("ATLAS_GLM53_EXL3_MOE_VERIFY_STAGED_K32", value)?;
        ensure!(
            !self.verify_staged_k32 || (self.private && exact_verify == Some(OsStr::new("1"))),
            "GLM private K32 verifier requires ROUTE_PRIVATE=1 and EXACT_VERIFY=1"
        );
        ensure!(
            !self.verify_staged_k32 || matches!(self.verify_group, None | Some(VerifyGroup::Eight)),
            "GLM private K32 verifier excludes group widths2/4"
        );
        Ok(self)
    }

    pub const fn verify_staged_k32_enabled(self) -> bool {
        self.verify_staged_k32
    }

    pub(crate) fn verify_staged_k32(self, rows: u32, exact: bool, prefill: bool) -> bool {
        self.verify_staged_k32 && exact && !prefill && (2..=SMALL_ROWS).contains(&rows)
    }

    pub(crate) fn verify_group(self, rows: u32, exact: bool, prefill: bool) -> Option<u32> {
        if !exact || prefill || !(2..=SMALL_ROWS).contains(&rows) {
            return None;
        }
        self.verify_group.map(VerifyGroup::width)
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
            Some("serial-reference")
                if !self.prefill && self.verify_group.is_none() && !self.verify_staged_k32 =>
            {
                Ok(())
            }
            Some("serial-reference") => {
                bail!("private prefill or verifier scheduling requires fused GLM EXL3 MoE")
            }
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
