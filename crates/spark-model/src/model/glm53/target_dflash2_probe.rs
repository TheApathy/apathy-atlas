// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit observation of the installed drafter; production capture owns state.
use super::*;
use crate::model::glm53::{Dflash2ProbeMode, Glm53Dflash2ProbeObserver, ProbeLayout};

impl Glm53Exl3Model {
    pub fn installed_diagnostic_state(&self) -> Result<(u32, ProbeLayout)> {
        self.ensure_dflash2_proposal_ready()?;
        let guard = self
            .dflash2
            .lock()
            .map_err(|_| anyhow::anyhow!("installed diagnostic owner poisoned"))?;
        let runtime = guard
            .as_ref()
            .context("diagnostic requires installed DFlash2")?;
        Ok((runtime.context_tokens(), runtime.probe_layout()?))
    }

    pub fn propose_installed_diagnostic(
        &self,
        anchor: u32,
        stream: u64,
        observer: &mut dyn Glm53Dflash2ProbeObserver,
    ) -> Result<[u32; 7]> {
        self.propose_installed_diagnostic_mode(
            anchor,
            stream,
            Dflash2ProbeMode::CachedPrefix,
            observer,
        )
    }

    pub fn propose_installed_diagnostic_mode(
        &self,
        anchor: u32,
        stream: u64,
        mode: Dflash2ProbeMode,
        observer: &mut dyn Glm53Dflash2ProbeObserver,
    ) -> Result<[u32; 7]> {
        ensure!(
            matches!(
                mode,
                Dflash2ProbeMode::CachedPrefix
                    | Dflash2ProbeMode::StableCachedProjection
                    | Dflash2ProbeMode::StableGemvCachedProjection
            ),
            "installed diagnostic only admits explicit cached modes"
        );
        self.ensure_dflash2_proposal_ready()?;
        let guard = self
            .dflash2
            .lock()
            .map_err(|_| anyhow::anyhow!("installed diagnostic owner poisoned"))?;
        let runtime = guard
            .as_ref()
            .context("diagnostic requires installed DFlash2")?;
        // Proposal only borrows target embeddings/position; it performs no
        // target commit/reset callback which could reacquire this drafter lock.
        runtime.propose_diagnostic(self, anchor, stream, mode, observer)
    }
}
