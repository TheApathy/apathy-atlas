// SPDX-License-Identifier: AGPL-3.0-only

//! Single-sequence Qwen4 decode with an eager PLE boundary and a graphed
//! layers-1..tail suffix. PLE performs host/NVMe work and therefore cannot be
//! captured, but all device addresses after that boundary are stable for a
//! fixed sequence slot.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::types::TransformerModel;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::traits::SequenceState;

impl TransformerModel {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_qwen4_ple_suffix_graph(
        &self,
        token: u32,
        hidden: DevicePtr,
        residual: DevicePtr,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        anyhow::ensure!(
            self.layers.len() > 1,
            "Qwen4 PLE suffix graph requires at least two layers"
        );
        let ple = self
            .qwen4_ple
            .as_ref()
            .context("Qwen4 PLE suffix graph requested without PLE")?;

        // Layer 0 consumes the fresh embedding before PLE chooses and injects
        // its sparse row at the layer-1 boundary.
        self.layers[0].decode(
            hidden,
            residual,
            seq.layer_states[0].as_mut(),
            kv_cache,
            seq.seq_len,
            &mut seq.block_table,
            &mut seq.disk_block_ids,
            &mut seq.disk_last_offloaded_per_layer,
            ctx,
            stream,
        )?;
        ple.forward_token(
            token,
            &seq.tokens,
            hidden,
            seq.slot_idx,
            seq.seq_len == 0,
            self.gpu.as_ref(),
            stream,
        )?;

        if let Some(graph) = self.decode_graph.lock().get(&seq.slot_idx).copied() {
            if graph.0 != 0 {
                self.gpu.launch_graph(graph, stream)?;
            }
            seq.tokens.push(token);
            seq.seq_len += 1;
            return Ok(self.decode_logits_ptr());
        }

        let graph_ctx = ForwardContext {
            graph_capture: true,
            ..*ctx
        };
        self.gpu.begin_capture(stream)?;
        for i in 1..self.layers.len() {
            self.layers[i].decode(
                hidden,
                residual,
                seq.layer_states[i].as_mut(),
                kv_cache,
                seq.seq_len,
                &mut seq.block_table,
                &mut seq.disk_block_ids,
                &mut seq.disk_last_offloaded_per_layer,
                &graph_ctx,
                stream,
            )?;
        }

        let normed = self.buffers.norm_output();
        if self.qwen4_final_hidden(hidden, residual, stream)?.is_none() {
            ops::rms_norm(
                self.gpu.as_ref(),
                self.rms_norm_kernel,
                hidden,
                &self.final_norm,
                normed,
                1,
                self.config.hidden_size as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;
        }
        self.lm_head(normed, stream)?;

        let graph = self.gpu.end_capture(stream)?;
        if graph.0 != 0 {
            tracing::info!(
                "Qwen4 PLE segmented graph captured for slot={} (layers 1..{} + head)",
                seq.slot_idx,
                self.layers.len()
            );
            self.decode_graph.lock().insert(seq.slot_idx, graph);
            self.gpu.launch_graph(graph, stream)?;
        } else {
            tracing::warn!("Qwen4 PLE segmented graph capture returned a null handle");
        }

        seq.tokens.push(token);
        seq.seq_len += 1;
        Ok(self.decode_logits_ptr())
    }
}
