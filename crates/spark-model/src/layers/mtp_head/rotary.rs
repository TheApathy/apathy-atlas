// SPDX-License-Identifier: AGPL-3.0-only

use super::{MtpHead, MtpProposerState};
use crate::layer::ForwardContext;
use crate::layers::ops;
use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

pub(super) fn prompt_tail_start(base: usize, count: usize) -> Result<usize> {
    let end = base
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("MTP prompt position overflow"))?;
    ensure!(count <= end, "MTP prompt tail exceeds physical history");
    end.checked_sub(count)
        .and_then(|start| start.checked_add(1))
        .ok_or_else(|| anyhow::anyhow!("MTP predict-into position overflow"))
}

impl MtpHead {
    pub(super) fn validate_rotary_span(
        &self,
        state: &MtpProposerState,
        start: usize,
        count: usize,
        ctx: &ForwardContext,
    ) -> Result<()> {
        ensure!(count <= 1_048_576, "MTP rotary span exceeds metadata bound");
        if count == 0 {
            return Ok(());
        }
        let last = start
            .checked_add(count - 1)
            .ok_or_else(|| anyhow::anyhow!("MTP rotary span overflow"))?;
        self.rotary_axes(state, start, ctx)?;
        self.rotary_axes(state, last, ctx)?;
        Ok(())
    }
    /// Admission occurs before embeddings, cache allocation or device writes.
    pub(super) fn rotary_axes(
        &self,
        state: &MtpProposerState,
        physical: usize,
        ctx: &ForwardContext,
    ) -> Result<[u32; 3]> {
        if !state.rotary_positions.is_identity() {
            ensure!(
                ctx.config.mrope_interleaved,
                "native MTP image positions require MRoPE"
            );
            ensure!(
                self.rope_mrope_k.is_some(),
                "native MTP MRoPE kernel is unavailable"
            );
            ensure!(
                ctx.config.yarn_factor == 0.0,
                "native MTP image YaRN positions require separate numerical qualification"
            );
        }
        state.rotary_positions.position(physical)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_rotary(
        &self,
        state: &MtpProposerState,
        ctx: &ForwardContext,
        q: DevicePtr,
        k: DevicePtr,
        meta: DevicePtr,
        nq: u32,
        nkv: u32,
        hd: u32,
        stream: u64,
    ) -> Result<()> {
        if state.rotary_positions.is_identity() {
            // Keep the shipping text-only kernel and arithmetic unchanged.
            ops::rope(
                ctx.gpu,
                self.rope_k,
                q,
                k,
                meta,
                1,
                nq,
                nkv,
                hd,
                ctx.config.rotary_dim() as u32,
                ctx.config.rope_theta as f32,
                stream,
            )
        } else {
            let kernel = self
                .rope_mrope_k
                .ok_or_else(|| anyhow::anyhow!("native MTP MRoPE kernel is unavailable"))?;
            ops::rope_mrope_interleaved(
                ctx.gpu,
                kernel,
                q,
                k,
                meta,
                meta.offset(20),
                meta.offset(24),
                1,
                nq,
                nkv,
                hd,
                ctx.config.rotary_dim() as u32,
                ctx.config.rope_theta as f32,
                stream,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{ImageSpan, RotaryPositions};

    #[test]
    fn prompt_tail_predicts_into_next_physical_position_before_mapping() {
        let map = RotaryPositions::from_image_spans(
            13,
            64,
            &[
                ImageSpan {
                    start: 1,
                    height: 2,
                    width: 2,
                },
                ImageSpan {
                    start: 6,
                    height: 2,
                    width: 3,
                },
            ],
        )
        .unwrap();
        let start = prompt_tail_start(12, 4).unwrap();
        assert_eq!(start, 10);
        assert_eq!(
            map.range(start, 4).unwrap(),
            [[4, 5, 5], [4, 5, 6], [7; 3], [8; 3]]
        );
        assert!(prompt_tail_start(2, 4).is_err());
        assert!(prompt_tail_start(usize::MAX, 1).is_err());
    }
}
