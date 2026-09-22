// SPDX-License-Identifier: AGPL-3.0-only
//! Startup serving policy, distinct from diagnostic modes and completed KV state.
use super::projection_contract::ProjectionFamily;
use anyhow::{Context, Result, ensure};
use std::ffi::OsStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServingProjection {
    Original,
    StableGemv,
}

impl ServingProjection {
    /// Environment I/O stays in the runtime. OsStr rejects malformed UTF-8
    /// without process-global environment mutation in the CPU contract tests.
    pub fn parse(value: Option<&OsStr>, kv_prefix: bool) -> Result<Self> {
        let text = value
            .map(|value| {
                value
                    .to_str()
                    .context("ATLAS_GLM53_DFLASH2_COMMITTED_PROJECTION must be valid UTF-8")
            })
            .transpose()?;
        let choice = match text {
            None | Some("original") => Self::Original,
            Some("stable-gemv") => Self::StableGemv,
            Some(_) => anyhow::bail!(
                "ATLAS_GLM53_DFLASH2_COMMITTED_PROJECTION must be absent, original, or stable-gemv"
            ),
        };
        choice.require_prefix(kv_prefix)?;
        Ok(choice)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::StableGemv => "stable-gemv",
        }
    }

    pub fn family(self) -> ProjectionFamily {
        match self {
            Self::Original => ProjectionFamily::Original,
            Self::StableGemv => ProjectionFamily::StableGemv,
        }
    }

    /// Re-reading is admission only; it never selects or mutates a runtime
    /// family. A request reset cannot turn a startup choice into another one.
    pub fn admit_current(self, value: Option<&OsStr>, kv_prefix: bool) -> Result<()> {
        let current = Self::parse(value, kv_prefix)?;
        ensure!(
            current.family() == self.family(),
            "GLM DFlash2 serving projection changed from {} to {}; restart required",
            self.as_str(),
            current.as_str()
        );
        Ok(())
    }

    pub fn admit_proposal(self, kv_prefix: bool, capturing: bool) -> Result<()> {
        self.require_prefix(kv_prefix)?;
        ensure!(
            !kv_prefix || !capturing,
            "GLM DFlash2 committed KV-prefix reuse requires eager execution"
        );
        Ok(())
    }

    pub fn admit_graph(self, kv_prefix: bool) -> Result<()> {
        self.require_prefix(kv_prefix)?;
        ensure!(
            self == Self::Original && !kv_prefix,
            "GLM DFlash2 committed KV-prefix mode rejects graph capture"
        );
        Ok(())
    }

    fn require_prefix(self, kv_prefix: bool) -> Result<()> {
        ensure!(
            self != Self::StableGemv || kv_prefix,
            "stable-gemv serving requires ATLAS_GLM53_DFLASH2_KV_PREFIX=1"
        );
        Ok(())
    }
}
