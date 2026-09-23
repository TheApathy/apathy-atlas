// SPDX-License-Identifier: AGPL-3.0-only

//! Startup-owned selection of the experimental partial-prefix replay path.

use anyhow::{Result, bail, ensure};
use std::ffi::OsStr;

pub(super) struct PartialReplaySetting(bool);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReplayPath {
    Scalar,
    Wide,
}

impl PartialReplaySetting {
    pub(super) fn parse(value: Option<&OsStr>) -> Result<Self> {
        match value {
            None => Ok(Self(false)),
            Some(value) if value == OsStr::new("0") => Ok(Self(false)),
            Some(value) if value == OsStr::new("1") => Ok(Self(true)),
            Some(_) => bail!("ATLAS_GLM53_PARTIAL_WIDE_REPLAY must be absent or exactly 0 or 1"),
        }
    }

    pub(super) fn path(&self, total_rows: usize, committed_rows: usize) -> Result<ReplayPath> {
        ensure!(
            (2..=8).contains(&total_rows) && (1..total_rows).contains(&committed_rows),
            "GLM partial replay requires a nonempty proper prefix of 2..=8 staged rows"
        );
        Ok(if self.0 && committed_rows > 1 {
            ReplayPath::Wide
        } else {
            ReplayPath::Scalar
        })
    }
}
