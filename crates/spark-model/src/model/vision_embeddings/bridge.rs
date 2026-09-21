// SPDX-License-Identifier: AGPL-3.0-only

//! Transformer integration; all device I/O remains behind GpuBackend.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::layout::Geometry;
use crate::model::types::TransformerModel;

impl TransformerModel {
    pub(in crate::model) fn prepare_vision_aggregate(
        &self,
        images: &[(Vec<f32>, usize, usize)],
    ) -> Result<()> {
        let mut state = self.vision_embeddings.lock();
        if images.is_empty() {
            state.invalidate();
            return Ok(());
        }
        let Some(ve) = &self.vision_encoder else {
            state.invalidate();
            anyhow::bail!("images supplied without a loaded vision encoder");
        };
        let Some(config) = &self.config.vision else {
            state.invalidate();
            anyhow::bail!("loaded vision encoder lacks explicit vision configuration");
        };
        let geometry = (|| -> Result<Geometry> {
            ensure!(
                config.image_pad_token_id != 0,
                "vision image pad ID must be explicit"
            );
            ensure!(
                ve.out_hidden_size == self.config.hidden_size
                    && config.out_hidden_size == ve.out_hidden_size,
                "vision merger width does not match text hidden width"
            );
            ensure!(
                !self.config.use_fp32_residual(),
                "Qwen vision embedding splice currently requires BF16 residual storage"
            );
            ensure!(
                config.spatial_merge_size == ve.spatial_merge_size,
                "vision spatial merge configuration mismatch"
            );
            ensure!(
                config.deepstack_visual_indexes == ve.deepstack_indexes
                    && ve.deepstack.len() == ve.deepstack_indexes.len(),
                "vision DeepStack configuration mismatch"
            );
            let pixel_width = config
                .patch_size
                .checked_mul(config.patch_size)
                .and_then(|v| v.checked_mul(config.temporal_patch_size))
                .and_then(|v| v.checked_mul(3))
                .ok_or_else(|| anyhow::anyhow!("vision patch width overflow"))?;
            Ok(Geometry {
                merge: ve.spatial_merge_size,
                max_patches: ve.p_max,
                hidden: ve.out_hidden_size,
                deepstack: ve.deepstack_indexes.len(),
                pixel_width,
            })
        })();
        let geometry = match geometry {
            Ok(g) => g,
            Err(error) => {
                state.invalidate();
                return Err(error);
            }
        };
        let cache = std::env::var("ATLAS_VISION_CACHE")
            .map(|s| s != "0" && s != "false")
            .unwrap_or(true);
        let pos = std::env::var("ATLAS_VISION_POSINTERP")
            .map(|s| s != "0")
            .unwrap_or(true);
        let rope = std::env::var("ATLAS_VISION_ROPE")
            .map(|s| s != "0")
            .unwrap_or(true);
        let stream = self.gpu.default_stream();
        let hit = state.prepare(
            self.gpu.as_ref(),
            images,
            geometry,
            cache,
            u8::from(pos) | (u8::from(rope) << 1),
            stream,
            |pixels, gh, gw| {
                Ok((
                    ve.buf_out,
                    ve.forward(pixels, gh, gw, self.gpu.as_ref(), stream)?,
                ))
            },
        )?;
        let rows = state
            .published
            .as_ref()
            .map(|(layout, _)| layout.rows)
            .unwrap_or(0);
        tracing::info!(
            "Vision encoder: {rows} final-merger patches in owned aggregate (cache_hit={hit})"
        );
        Ok(())
    }

    pub(in crate::model) fn vision_prompt_present(&self, tokens: &[u32]) -> bool {
        self.vision_embeddings.lock().is_ready()
            || self
                .config
                .vision
                .as_ref()
                .is_some_and(|config| tokens.contains(&config.image_pad_token_id))
    }

    pub(in crate::model) fn validate_vision_prompt(
        &self,
        tokens: &[u32],
        start: usize,
        len: usize,
    ) -> Result<()> {
        if let Some(config) = &self.config.vision {
            self.vision_embeddings
                .lock()
                .plan(tokens, config.image_pad_token_id, start, len)?;
        }
        Ok(())
    }

    pub(in crate::model) fn splice_vision_embeddings(
        &self,
        tokens: &[u32],
        start: usize,
        len: usize,
        destination: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let Some(config) = &self.config.vision else {
            return Ok(());
        };
        self.vision_embeddings.lock().splice(
            self.gpu.as_ref(),
            tokens,
            config.image_pad_token_id,
            start,
            len,
            destination,
            self.config.hidden_size,
            if self.config.use_fp32_residual() {
                4
            } else {
                2
            },
            stream,
        )
    }

    pub(in crate::model) fn vision_positions(
        &self,
        tokens: &[u32],
        start: usize,
        len: usize,
    ) -> Result<Option<[Vec<u32>; 3]>> {
        if !self.vision_prompt_present(tokens) {
            return Ok(None);
        }
        let config = self
            .config
            .vision
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("vision configuration missing"))?;
        Ok(Some(
            self.vision_embeddings
                .lock()
                .plan(tokens, config.image_pad_token_id, start, len)?
                .positions,
        ))
    }
}
