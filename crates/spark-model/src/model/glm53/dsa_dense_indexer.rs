// SPDX-License-Identifier: AGPL-3.0-only

//! Checked, default-off omission of unused dense-prefix indexer query work.
//! Persistent index keys, gates, pools and tails are not part of this plan.

use anyhow::{Context, Result, bail, ensure};

use crate::layers::ops::GLM53_EXL3_MAX_WIDE_ROWS;

use super::dsa_attention::{GLM53_DSA_ABSOLUTE_POSITIONS, SELECTED};

pub(super) fn parse_dense_indexer_skip(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => bail!("ATLAS_GLM53_DSA_DENSE_INDEXER_SKIP must be exactly 0 or 1"),
    }
}

/// The same density decision must admit omitted work and select its consumer.
/// Use absolute end positions, including continuation chunks, not row count alone.
pub(super) fn dense_full_coverage(
    rows: u32,
    position: u32,
    capacity: u32,
    layer_major: bool,
) -> Result<bool> {
    let max_rows = if layer_major {
        u32::try_from(GLM53_EXL3_MAX_WIDE_ROWS)?
    } else {
        8
    };
    ensure!(
        (1..=max_rows).contains(&rows),
        "GLM DSA density rows must be 1..={max_rows}"
    );
    ensure!(
        (1..=GLM53_DSA_ABSOLUTE_POSITIONS).contains(&capacity),
        "GLM DSA density capacity is outside the absolute position space"
    );
    let end = position
        .checked_add(rows)
        .context("GLM DSA density position overflow")?;
    ensure!(end <= capacity, "GLM DSA density chunk exceeds capacity");
    Ok(rows > 1 && layer_major && end <= SELECTED)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Glm53DsaDenseIndexerPlan {
    dense: bool,
    requested: bool,
}

impl Glm53DsaDenseIndexerPlan {
    pub(super) fn new(
        rows: u32,
        position: u32,
        capacity: u32,
        layer_major: bool,
        requested: bool,
    ) -> Result<Self> {
        Ok(Self {
            dense: dense_full_coverage(rows, position, capacity, layer_major)?,
            requested,
        })
    }

    pub(super) fn dense_full_coverage(self) -> bool {
        self.dense
    }

    pub(super) fn skip_indexer_query(self) -> bool {
        self.requested && self.dense_full_coverage()
    }

    /// Compressed query projection: three launches; F32 head projection: one.
    pub(super) fn indexer_launches(self) -> u32 {
        if self.skip_indexer_query() { 0 } else { 4 }
    }
}
