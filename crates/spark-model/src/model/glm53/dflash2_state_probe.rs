// SPDX-License-Identifier: AGPL-3.0-only

//! Read-only view for the target's exclusive committed-state diagnostic.
use super::*;

impl Glm53Dflash2Runtime {
    pub(in crate::model::glm53) fn state_probe_projected(
        &self,
    ) -> Result<(GgmlIqBuffer, GgmlIqBuffer)> {
        self.ensure_capture_kv_ready()?;
        let region = self.region(self.plan.projected_target);
        let bytes = (self.context_tokens as usize)
            .checked_mul(HIDDEN as usize * 2)
            .context("projected context extent overflow")?;
        ensure!(
            bytes > 0 && bytes <= region.bytes,
            "projected context outside runtime allocation"
        );
        Ok((
            GgmlIqBuffer {
                ptr: self.arena,
                bytes: self.plan.arena_bytes,
            },
            GgmlIqBuffer {
                ptr: region.ptr,
                bytes,
            },
        ))
    }
}
