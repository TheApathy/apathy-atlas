// SPDX-License-Identifier: AGPL-3.0-only

//! Final norm and LM-head dispatch for DFlash verification.
//!
//! The normal path keeps its wide/chunked kernels. Diagnostic controls can
//! instead run one row at a time through the exact K=1 target family, making
//! a parity-hash restoration identify the batched family at fault.

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::layers::ops;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DflashK1LmHeadProof {
    requested_rows: usize,
    engaged_rows: usize,
    vocab: usize,
}

impl DflashK1LmHeadProof {
    fn begin(
        requested_rows: usize,
        verify_vocab: usize,
        model_vocab: usize,
        fp32_logits: bool,
    ) -> Result<Self> {
        if requested_rows == 0 {
            bail!(
                "DFLASH_K1_LM_HEAD_PATH_PROOF requested=true engaged=false \
                 requirement=requested_rows must be positive"
            );
        }
        if verify_vocab != model_vocab {
            bail!(
                "DFLASH_K1_LM_HEAD_PATH_PROOF requested=true engaged=false \
                 requirement=full vocabulary required verify_vocab={verify_vocab} \
                 model_vocab={model_vocab}"
            );
        }
        if fp32_logits {
            bail!(
                "DFLASH_K1_LM_HEAD_PATH_PROOF requested=true engaged=false \
                 requirement=BF16 logits required"
            );
        }
        Ok(Self {
            requested_rows,
            engaged_rows: 0,
            vocab: model_vocab,
        })
    }

    fn engage(&mut self) -> Result<()> {
        if self.engaged_rows == self.requested_rows {
            bail!(
                "DFLASH_K1_LM_HEAD_PATH_PROOF requested=true engaged=false \
                 requirement=engaged_rows exceeded requested_rows"
            );
        }
        self.engaged_rows += 1;
        Ok(())
    }

    fn finish(self) -> Result<Self> {
        if self.engaged_rows != self.requested_rows {
            bail!(
                "DFLASH_K1_LM_HEAD_PATH_PROOF requested=true engaged=false \
                 requirement=engaged_rows does not equal requested_rows \
                 requested_rows={} engaged_rows={}",
                self.requested_rows,
                self.engaged_rows
            );
        }
        Ok(self)
    }

    pub(super) fn proof_line(
        &self,
        family: &str,
        pre_verify_len: usize,
        tokens: &[u32],
    ) -> Result<String> {
        if tokens.len() != self.requested_rows {
            bail!(
                "DFLASH_K1_LM_HEAD_PATH_PROOF requested=true engaged=false \
                 requirement=token vector length does not equal requested_rows"
            );
        }
        Ok(format!(
            "DFLASH_K1_LM_HEAD_PATH_PROOF family=\"{family}\" \
             pre_verify_len={pre_verify_len} tokens={tokens:?} requested=true engaged=true \
             requested_rows={} engaged_rows={} full_vocab=true vocab={} dtype=\"bf16\"",
            self.requested_rows, self.engaged_rows, self.vocab
        ))
    }
}

impl TransformerModel {
    pub(super) fn dflash_k1_final_norm(
        &self,
        hidden: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let residual_elem = if self.config.use_fp32_residual() {
            4usize
        } else {
            bf16
        };
        let normed = self.buffers.norm_output();
        for row in 0..rows {
            ops::rms_norm(
                self.gpu.as_ref(),
                self.rms_norm_kernel,
                hidden.offset(row * h * residual_elem),
                &self.final_norm,
                normed.offset(row * h * bf16),
                1,
                h as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;
        }
        Ok(normed)
    }

    /// Run each row through the ordinary one-token LM head and write its
    /// dtype-correct argmax to the shared scratch output `[rows]`.
    pub(super) fn dflash_k1_lm_head_argmax(
        &self,
        normed: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<(DevicePtr, DflashK1LmHeadProof)> {
        let mut proof = DflashK1LmHeadProof::begin(
            rows,
            self.verify_lmhead_vocab() as usize,
            self.config.vocab_size,
            self.use_fp32_logits,
        )?;

        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let argmax_out = self.buffers.scratch();
        for row in 0..rows {
            let logits = self.lm_head(normed.offset(row * h * bf16), stream)?;
            ops::argmax_bf16(
                self.gpu.as_ref(),
                self.argmax_kernel,
                logits,
                argmax_out.offset(row * 4),
                self.config.vocab_size as u32,
                stream,
            )?;
            proof.engage()?;
        }
        Ok((argmax_out, proof.finish()?))
    }

    pub(super) fn dflash_batched_finalize(
        &self,
        hidden: DevicePtr,
        total_rows: usize,
        serial_final_norm: bool,
        serial_lm_head: bool,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let normed = self.buffers.norm_output();

        if serial_final_norm {
            self.dflash_k1_final_norm(hidden, total_rows, stream)?;
        } else {
            ops::rms_norm(
                self.gpu.as_ref(),
                self.rms_norm_kernel,
                hidden,
                &self.final_norm,
                normed,
                total_rows as u32,
                h as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;
        }

        if serial_lm_head {
            let (argmax_out, _) = self.dflash_k1_lm_head_argmax(normed, total_rows, stream)?;
            self.gpu.synchronize(stream)?;
            return self.dflash_copy_argmax(argmax_out, total_rows);
        }

        // The shared logits buffer holds at most 32 rows. Preserve the normal
        // contiguous chunking path exactly when the diagnostic is disabled.
        let vocab = self.verify_lmhead_vocab() as usize;
        const LM_CHUNK: usize = 32;
        let mut all_argmax = vec![0u32; total_rows];
        let mut chunk_start = 0usize;
        while chunk_start < total_rows {
            let rows = (total_rows - chunk_start).min(LM_CHUNK);
            let normed_chunk = normed.offset(chunk_start * h * bf16);
            self.lm_head_batched(normed_chunk, rows as u32, stream)?;
            let argmax_out = self.buffers.scratch();
            for row in 0..rows {
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    self.buffers.logits().offset(row * vocab * bf16),
                    argmax_out.offset(row * 4),
                    vocab as u32,
                    stream,
                )?;
            }
            self.gpu.synchronize(stream)?;
            let chunk = self.dflash_copy_argmax(argmax_out, rows)?;
            all_argmax[chunk_start..chunk_start + rows].copy_from_slice(&chunk);
            chunk_start += rows;
        }
        Ok(all_argmax)
    }

    fn dflash_copy_argmax(&self, argmax_out: DevicePtr, rows: usize) -> Result<Vec<u32>> {
        let mut bytes = vec![0u8; rows * 4];
        self.gpu.copy_d2h(argmax_out, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect())
    }
}

#[cfg(test)]
#[path = "verify_dflash_output_tests.rs"]
mod tests;
