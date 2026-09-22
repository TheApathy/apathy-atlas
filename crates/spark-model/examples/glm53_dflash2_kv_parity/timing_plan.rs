// SPDX-License-Identifier: AGPL-3.0-only
//! Diagnostic scheduling/history labels, never cache-reuse authority.
use anyhow::{Result, ensure};
use spark_model::model::glm53::Dflash2ProbeMode as Mode;

pub fn order(index: usize) -> [Mode; 3] {
    use Mode::{FullRecompute as O, StableCachedProjection as C, StableFullProjection as S};
    [
        [O, S, C],
        [O, C, S],
        [S, O, C],
        [S, C, O],
        [C, O, S],
        [C, S, O],
    ][index % 6]
}

/// Expected work derived from successful harness calls, not a device counter.
/// The real KvPrefix remains the only authority selecting the source interval.
pub fn workload(previous: u32, context: u32) -> Result<(&'static str, u32)> {
    ensure!(
        context > 0 && previous <= context,
        "timing history regressed or empty"
    );
    Ok(if previous == 0 {
        ("initial-full-context", context)
    } else if previous == context {
        ("repeat-one-row", 1)
    } else {
        ("after-advance", context - previous)
    })
}
