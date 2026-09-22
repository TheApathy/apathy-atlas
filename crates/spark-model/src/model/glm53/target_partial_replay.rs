// SPDX-License-Identifier: AGPL-3.0-only

//! Prefix-only restaging under the existing policy snapshot and failure owner.

use super::*;
use crate::layers::ops::{glm53_exact_wide_prefill_active, with_glm53_exact_verify};

impl Glm53Exl3Model {
    pub(super) fn replay_partial_wide(
        &self,
        start: usize,
        tokens: &[u32],
        stream: u64,
        exact_verify: bool,
    ) -> Result<DevicePtr> {
        self.ensure_verify_healthy()?;
        let start = u32::try_from(start)?;
        let rows = u32::try_from(tokens.len())?;
        let end = start
            .checked_add(rows)
            .context("GLM partial replay end overflow")?;
        ensure!(
            (2..=7).contains(&rows) && self.position() == start && end <= self.capacity,
            "GLM partial wide replay geometry/position drift"
        );
        ensure!(
            !glm53_exact_wide_prefill_active()
                && !glm53_layer_major_prefill_active()
                && !self.gpu.stream_is_capturing(stream),
            "GLM partial replay requires eager non-prefill execution"
        );
        self.dflash2
            .lock()
            .unwrap()
            .as_ref()
            .context("GLM DFlash2 runtime disappeared before partial replay")?
            .preflight_target_rows(start, rows)?;
        let (logits, at, count) = if exact_verify {
            with_glm53_exact_verify(|| self.verify_tokens_staged(tokens, stream))?
        } else {
            self.verify_tokens_staged(tokens, stream)?
        };
        ensure!(
            at == start && count == rows,
            "GLM partial wide stage extent drift"
        );
        self.commit_accepted(stream)?;
        self.gpu.synchronize(stream)?;
        self.state.lock().unwrap().position = end;
        self.dflash2
            .lock()
            .unwrap()
            .as_mut()
            .context("GLM DFlash2 runtime disappeared during partial replay")?
            .observe_target_rows_ordered(self, rows, stream)?;
        Ok(logits)
    }
}
