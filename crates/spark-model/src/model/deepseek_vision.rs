// SPDX-License-Identifier: AGPL-3.0-only

//! Request-scoped DeepSeek image storage and safe embedding splice. Images
//! are encoded once, copied out of the encoder's reusable scratch, and all
//! sentinels are replaced before the first hyper-connection expansion.

use super::TransformerModel;
use crate::layers::{deepseek_vision::DeepSeekVisionEncoder, ops};
use anyhow::{Context, Result, ensure};
use atlas_core::config::{
    ImageBlock, ImageTokenType, build_deepseek_image_block, validate_deepseek_image_tokens,
};
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub(super) struct DeepSeekVisionState {
    encoder: DeepSeekVisionEncoder,
    pending: Mutex<Vec<PendingImage>>,
}

struct PendingImage {
    embeddings: DevicePtr,
    grid_h: usize,
    grid_w: usize,
}

fn validate_image_count(count: usize) -> Result<()> {
    // Common prefill preparation runs for text-only requests too. Keep the
    // zero-image path so it clears any pending request-owned image rows.
    ensure!(
        count <= 16,
        "DeepSeek supports at most 16 images per request"
    );
    Ok(())
}

fn splice_plan(
    tokens: &[u32],
    grids: &[(usize, usize)],
    vocab: u32,
    max_tokens: usize,
    chunk_len: usize,
) -> Result<Vec<ImageBlock>> {
    let spans = validate_deepseek_image_tokens(tokens, vocab, max_tokens)?;
    ensure!(
        spans.len() == grids.len(),
        "DeepSeek image/pixel count mismatch"
    );
    ensure!(
        spans.iter().all(|span| span.end < chunk_len),
        "DeepSeek image spans must be wholly inside the first prefill chunk"
    );
    let mut cursor = 0;
    let mut blocks = Vec::with_capacity(grids.len());
    for &(h, w) in grids {
        cursor += tokens[cursor..]
            .iter()
            .position(|&id| id >= vocab)
            .context("Missing DeepSeek image sentinel block")?;
        let block = build_deepseek_image_block(h, w, cursor, vocab)?;
        let end = cursor
            .checked_add(block.token_ids.len())
            .context("Image block overflow")?;
        ensure!(
            end <= chunk_len && tokens.get(cursor..end) == Some(block.token_ids.as_slice()),
            "DeepSeek image sentinel layout differs from encoded image geometry"
        );
        cursor = end;
        blocks.push(block);
    }
    Ok(blocks)
}

impl TransformerModel {
    pub(crate) fn install_deepseek_vision(&mut self, encoder: DeepSeekVisionEncoder) {
        self.deepseek_vision = Some(DeepSeekVisionState {
            encoder,
            pending: Mutex::new(Vec::new()),
        });
    }

    pub(super) fn prepare_deepseek_images(
        &self,
        images: &[(Vec<f32>, usize, usize)],
    ) -> Result<()> {
        let state = self
            .deepseek_vision
            .as_ref()
            .context("DeepSeek Vision encoder is not loaded")?;
        validate_image_count(images.len())?;
        let mut pending = state.pending.lock();
        self.clear_deepseek_pending(&mut pending)?;
        let row_bytes = self.config.hidden_size * 2;
        for (patches, gh, gw) in images {
            let result = (|| {
                let rows = state.encoder.output_rows(*gh, *gw)?;
                let output = state
                    .encoder
                    .forward(self.gpu.as_ref(), patches, *gh, *gw)?;
                let bytes = rows
                    .checked_mul(row_bytes)
                    .context("Image embedding size overflow")?;
                let saved = self.gpu.alloc(bytes)?;
                // Register ownership before enqueueing a transfer, including its error path.
                pending.push(PendingImage {
                    embeddings: saved,
                    grid_h: gh.div_ceil(3),
                    grid_w: gw.div_ceil(3),
                });
                self.gpu
                    .copy_d2d_async(output, saved, bytes, self.gpu.default_stream())?;
                self.gpu.synchronize(self.gpu.default_stream())
            })();
            if let Err(error) = result {
                // A poisoned CUDA stream must not publish partially prepared images.
                let _ = self.gpu.synchronize(self.gpu.default_stream());
                let _ = self.clear_deepseek_pending(&mut pending);
                return Err(error);
            }
        }
        tracing::info!("DeepSeek Vision: encoded {} image(s)", pending.len());
        Ok(())
    }

    fn clear_deepseek_pending(&self, pending: &mut Vec<PendingImage>) -> Result<()> {
        for image in pending.drain(..) {
            self.gpu.free(image.embeddings)?;
        }
        Ok(())
    }

    /// Returns false for other architectures. Both chunked and whole-prompt
    /// entry points must call this before their ordinary token embedding.
    pub(super) fn embed_deepseek_chunk(
        &self,
        tokens: &[u32],
        chunk_start: usize,
        chunk_len: usize,
        hidden: DevicePtr,
        stream: u64,
    ) -> Result<bool> {
        let Some(config) = &self.config.deepseek_vision else {
            return Ok(false);
        };
        let state = self
            .deepseek_vision
            .as_ref()
            .context("DeepSeek Vision encoder is not loaded")?;
        let end = chunk_start
            .checked_add(chunk_len)
            .context("DeepSeek chunk size overflow")?;
        let chunk = tokens
            .get(chunk_start..end)
            .context("DeepSeek chunk outside prompt")?;
        ensure!(
            chunk_len > 0 && chunk_len <= self.buffers.max_batch_tokens(),
            "DeepSeek chunk exceeds embedding buffer capacity"
        );
        let vocab = u32::try_from(self.config.vocab_size)?;
        let mut pending = state.pending.lock();
        let blocks = if chunk_start == 0 {
            if tokens.iter().all(|&token| token < vocab) {
                self.clear_deepseek_pending(&mut pending)?;
                Vec::new()
            } else {
                let grids: Vec<_> = pending
                    .iter()
                    .map(|image| (image.grid_h, image.grid_w))
                    .collect();
                splice_plan(tokens, &grids, vocab, config.max_tokens, chunk_len)?
            }
        } else {
            ensure!(
                chunk.iter().all(|&token| token < vocab),
                "DeepSeek images are only supported in the initial prefill chunk"
            );
            Vec::new()
        };
        let original: Vec<u8> = chunk.iter().flat_map(|token| token.to_le_bytes()).collect();
        let safe: Vec<u8> = chunk
            .iter()
            .flat_map(|&token| if token < vocab { token } else { 0 }.to_le_bytes())
            .collect();
        // Keep ORIGINAL sentinels for image-aware MoE and attention. Mask only
        // the temporary ordinary embedding IDs; no out-of-vocabulary lookup.
        self.gpu.copy_h2d(&original, self.buffers.token_ids())?;
        self.gpu.copy_h2d(&safe, self.buffers.scratch())?;
        ops::batched_embed(
            self.gpu.as_ref(),
            self.batched_embed_kernel,
            self.buffers.scratch(),
            self.embed_tokens.weight,
            hidden,
            chunk_len as u32,
            self.config.hidden_size as u32,
            stream,
        )?;
        self.scale_embeddings(hidden, chunk_len, stream)?;
        let special = state.encoder.image_special_embeddings();
        let row_bytes = self.config.hidden_size * 2;
        for (image, block) in pending.iter().zip(&blocks) {
            let mut image_row = 0;
            for (offset, kind) in block.types.iter().enumerate() {
                let source = if *kind == ImageTokenType::Image {
                    let row = block.aligner_permutation[image_row];
                    image_row += 1;
                    image.embeddings.offset(row * row_bytes)
                } else {
                    special[*kind as usize]
                };
                self.gpu.copy_d2d_async(
                    source,
                    hidden.offset((block.start_pos + offset) * row_bytes),
                    row_bytes,
                    stream,
                )?;
            }
        }
        // Model dispatch is sequential; fence before freeing request-owned
        // rows and before the next request may reuse encoder scratch.
        self.gpu.synchronize(stream)?;
        self.clear_deepseek_pending(&mut pending)?;
        Ok(true)
    }

    pub(super) fn release_deepseek_vision(&mut self) {
        if let Some(state) = self.deepseek_vision.take() {
            let _ = self.gpu.synchronize(self.gpu.default_stream());
            let mut pending = state.pending.into_inner();
            if let Err(error) = self.clear_deepseek_pending(&mut pending) {
                tracing::warn!("DeepSeek image buffer teardown: {error:#}");
            }
            if let Err(error) = state.encoder.release(self.gpu.as_ref()) {
                tracing::warn!("DeepSeek vision encoder teardown: {error:#}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deepseek_text_only_image_preparation_is_valid() {
        for count in [0, 1, 16] {
            validate_image_count(count).unwrap();
        }
        for count in [17, usize::MAX] {
            assert!(validate_image_count(count).is_err());
        }
    }

    #[test]
    fn deepseek_splice_requires_exact_geometry_and_complete_initial_chunk() {
        let block = build_deepseek_image_block(2, 2, 1, 129280).unwrap();
        let mut tokens = vec![42];
        tokens.extend_from_slice(&block.token_ids);
        tokens.push(43);
        let plan = splice_plan(&tokens, &[(2, 2)], 129280, 384, tokens.len()).unwrap();
        assert_eq!(plan[0].aligner_permutation, vec![0, 2, 1, 3]);
        assert!(splice_plan(&tokens, &[], 129280, 384, tokens.len()).is_err());
        assert!(splice_plan(&tokens, &[(1, 4)], 129280, 384, tokens.len()).is_err());
        assert!(splice_plan(&tokens, &[(2, 2)], 129280, 384, tokens.len() - 2).is_err());
    }
}
