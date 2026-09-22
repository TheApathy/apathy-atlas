// SPDX-License-Identifier: AGPL-3.0-only
//! Preflight the whole selected capture before any target token advancement.
use super::{ProbeLayout, ProbeStage};
use anyhow::{Context, Result, ensure};

pub fn admit(
    candidate: &ProbeLayout,
    reference: &ProbeLayout,
    contexts: &[usize],
    projected: bool,
) -> Result<()> {
    ensure!(
        !contexts.is_empty() && contexts.iter().all(|c| *c > 0),
        "empty capture sweep/context"
    );
    ensure!(
        candidate.stages() == reference.stages() && candidate.vocab() == reference.vocab(),
        "probe layout identities differ"
    );
    for &stage in candidate.stages() {
        ensure!(
            candidate.bytes(stage)? == reference.bytes(stage)?,
            "probe layout extents differ"
        );
    }
    for layout in [candidate, reference] {
        let id_bytes = layout.bytes(ProbeStage::DraftIds)?;
        ensure!(
            id_bytes > 0 && id_bytes % 4 == 0,
            "invalid draft ID geometry"
        );
        let predicted = id_bytes / 4;
        let denominator = predicted
            .checked_mul(2)
            .context("hidden geometry overflow")?;
        let hidden_bytes = layout.bytes(ProbeStage::SelectedHidden)?;
        ensure!(
            hidden_bytes % denominator == 0,
            "selected hidden rows are not exact"
        );
        let hidden = hidden_bytes / denominator;
        let accepted = predicted
            .checked_add(2)
            .context("accepted context overflow")?;
        let rejected = accepted
            .checked_add(1)
            .context("rejected context overflow")?;
        for context in contexts.iter().copied().chain([1, accepted, rejected]) {
            let context = u32::try_from(context).context("capture context exceeds u32")?;
            if projected {
                layout.clone().with_projected_target(context, hidden)?;
            }
        }
    }
    Ok(())
}
