// SPDX-License-Identifier: AGPL-3.0-only

//! Checked CPU admission for the existing fixed-width Qwen vision tower.

use anyhow::{Result, ensure};
use std::hash::{Hash, Hasher};

use super::MAX_AGGREGATE_BYTES;
pub(super) use super::plan::{CopyRun, PromptPlan};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub merge: usize,
    pub max_patches: usize,
    pub hidden: usize,
    pub deepstack: usize,
    pub pixel_width: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ImageLayout {
    pub grids: Vec<(usize, usize)>,
    pub rows: usize,
    pub bytes: usize,
    pub geometry: Geometry,
}

impl ImageLayout {
    pub fn validate(
        images: &[(Vec<f32>, usize, usize)],
        g: Geometry,
        max_rows: usize,
    ) -> Result<Self> {
        ensure!(g.merge == 2, "Qwen vision encoder requires spatial merge 2");
        ensure!(
            g.pixel_width == 1536,
            "Qwen vision encoder requires 1536 values per patch"
        );
        ensure!(
            g.hidden > 0 && g.max_patches > 0 && max_rows > 0,
            "invalid vision capacity"
        );
        ensure!(
            images.len() <= max_rows,
            "too many images for configured sequence capacity"
        );
        let mut grids = Vec::with_capacity(images.len());
        let mut rows = 0usize;
        for (pixels, gh, gw) in images {
            ensure!(
                *gh > 0 && *gw > 0 && gh % g.merge == 0 && gw % g.merge == 0,
                "vision grid must be positive and merge-aligned"
            );
            let p = gh
                .checked_mul(*gw)
                .ok_or_else(|| anyhow::anyhow!("vision grid overflow"))?;
            ensure!(
                p <= g.max_patches,
                "vision grid exceeds encoder scratch capacity"
            );
            let expected = p
                .checked_mul(g.pixel_width)
                .ok_or_else(|| anyhow::anyhow!("pixel size overflow"))?;
            ensure!(
                pixels.len() == expected,
                "vision pixel count does not match grid"
            );
            ensure!(
                pixels.iter().all(|v| v.is_finite()),
                "nonfinite vision pixels"
            );
            let grid = (gh / g.merge, gw / g.merge);
            let final_rows = grid.0 * grid.1;
            let outputs = final_rows
                .checked_mul(
                    g.deepstack
                        .checked_add(1)
                        .ok_or_else(|| anyhow::anyhow!("DeepStack count overflow"))?,
                )
                .ok_or_else(|| anyhow::anyhow!("vision output size overflow"))?;
            ensure!(
                outputs <= g.max_patches,
                "vision outputs exceed encoder scratch capacity"
            );
            rows = rows
                .checked_add(final_rows)
                .ok_or_else(|| anyhow::anyhow!("vision row overflow"))?;
            ensure!(
                rows <= max_rows,
                "vision rows exceed configured sequence capacity"
            );
            grids.push(grid);
        }
        let bytes = rows
            .checked_mul(g.hidden)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("vision aggregate size overflow"))?;
        ensure!(
            bytes <= MAX_AGGREGATE_BYTES,
            "vision aggregate exceeds explicit 256 MiB limit"
        );
        Ok(Self {
            grids,
            rows,
            bytes,
            geometry: g,
        })
    }

    pub fn plan(&self, tokens: &[u32], pad: u32, start: usize, len: usize) -> Result<PromptPlan> {
        super::plan::plan(tokens, pad, &self.grids, start, len)
    }
}

/// Hash every input bit, including image boundaries and effective tower toggles.
/// This process-local memoization key is not a cryptographic identity or provenance hash.
pub(super) fn cache_key(
    images: &[(Vec<f32>, usize, usize)],
    geometry: Geometry,
    variant: u8,
) -> u64 {
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    geometry.hash(&mut hash);
    variant.hash(&mut hash);
    images.len().hash(&mut hash);
    for (pixels, gh, gw) in images {
        gh.hash(&mut hash);
        gw.hash(&mut hash);
        pixels.len().hash(&mut hash);
        for value in pixels {
            value.to_bits().hash(&mut hash);
        }
    }
    hash.finish()
}
