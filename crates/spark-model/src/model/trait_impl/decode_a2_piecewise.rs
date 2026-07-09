// SPDX-License-Identifier: AGPL-3.0-only

#![allow(clippy::too_many_arguments)]

//! `TransformerModel::decode_batch_dispatch_piecewise` — the piecewise
//! CUDA-graph multi-seq decode path, gated by `ATLAS_MULTISEQ_GRAPHS=1`.
//!
//! # Why piecewise (and not one monolithic graph)
//!
//! A single captured graph over the whole per-step forward would have to
//! indirect EVERY per-slot device address that any captured kernel reads.
//! Two of those are tractable and already done; one is not:
//!
//! | address baked at capture | where | indirected? |
//! |--------------------------|-------|-------------|
//! | SSM `h_state`/`conv_state` ptrs | `qwen3_ssm::decode_multi_seq_inner` step 4/5 | YES — via layer-stable `ssm_multi_seq_ptr_scratch` (Fix B). Refreshed pre-replay by `multiseq_refresh_ptr_table`. |
//! | attn positions/slot/seq_len/**block table** | `upload_batch_metadata_fixed` → `scratch+32768` | YES — fixed device base, re-uploaded every step OUTSIDE any capture. Pad slots → `dummy_kv_block` (PAD_SLOT_ID). |
//! | embedding source offsets | `embed()` in Phase 1 | N/A — embed runs eager pre-loop, never captured. |
//! | **attention split-K grid geometry** | `qwen3_attention::multi_seq::mod` `num_splits` from host `max(seq_lens)+1` | **NO** — the launch grid dims are a host scalar baked into the graph node; they go stale the instant a sequence crosses a split-K threshold. This is the "one token corrupted per N=4 stream". |
//!
//! Rather than push the split-K decision onto the device (a *kernel* change,
//! out of scope here), we run FullAttention layers EAGERLY and capture only
//! the address-stable SSM/FFN runs + the norm/LM-head tail. Every captured
//! segment then reads ONLY fixed device addresses, so the segment-graph
//! cache is keyed on `(padded_n, segment_id)` alone — a graph captured for a
//! padded batch size replays for ANY active slot set of that size, and a
//! max-batch capture serves any `n <= padded_n` via the pad sentinels.
//!
//! # Per-step control flow
//!
//! 1. Eager pre-loop (never captured): embed active tokens, zero padding,
//!    ensure KV blocks, upload attn metadata to the fixed base, build dummy
//!    layer states for pad positions.
//! 2. Walk the layer list, grouping maximal runs of graph-safe layers into
//!    segments. For each segment: refresh SSM ptr tables (gather-before-
//!    replay), then replay the cached segment graph — or, on first sight of
//!    `(padded_n, seg_id)`, capture it. FullAttention layers between
//!    segments run eagerly.
//! 3. Norm + LM-head run as a final captured tail segment.
//! 4. Post-loop (eager): push tokens, bump seq_len.

use anyhow::Result;
use atlas_core::config::LayerType;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::block_mgmt::{ensure_blocks_through_decode, extract_layer_refs};
use super::super::types::TransformerModel;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::ops;
use crate::traits::SequenceState;

/// A contiguous span of layers that share a dispatch mode.
enum Segment {
    /// A run of consecutive graph-safe (SSM/FFN) layers `[start, end)`,
    /// captured into one graph and keyed by `(padded_n, seg_id)`.
    Graphed {
        start: usize,
        end: usize,
        seg_id: usize,
    },
    /// A single FullAttention layer, run eagerly (split-K grid geometry is
    /// host-derived — see module docs).
    EagerAttn(usize),
}

impl TransformerModel {
    pub(super) fn decode_batch_dispatch_piecewise(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
    ) -> Result<DevicePtr> {
        let n = tokens.len();
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        // Pad to nearest captured graph size [2, 4, 8]. One capture per
        // padded size serves any n <= padded_n via the pad sentinels.
        let padded_n = [2, 4, 8].iter().copied().find(|&s| s >= n).unwrap_or(n);

        // ── Phase 1: eager pre-loop (NOT captured) ──

        // 1a. Embed active tokens into hidden[0..n).
        for (i, &tok) in tokens.iter().enumerate() {
            self.embed(tok, hidden.offset(i * h * fp32), stream)?;
        }
        // 1b. Zero padding hidden[n..padded_n).
        for i in n..padded_n {
            self.gpu.memset(hidden.offset(i * h * fp32), 0, h * fp32)?;
        }

        // 1c. Allocate KV blocks for active sequences.
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        for seq in seqs.iter_mut() {
            let blocks_needed = (seq.seq_len / bs) + 1;
            ensure_blocks_through_decode(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
            )?;
        }

        // 1d. Upload attn metadata with fixed stride to the fixed device
        //     base. Consumed ONLY by the eager attention layers below; its
        //     device pointers are stable across steps and its contents are
        //     refreshed here every step, so nothing about it is baked into a
        //     captured segment. Pad slots point at `dummy_kv_block`.
        let metadata = self.upload_batch_metadata_fixed(seqs, padded_n, &mut kv_cache, stream)?;

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(metadata),
            profile: false,
            comm: self.comm_ref(),
            // Individual segments toggle capture themselves; the eager
            // attention layers must NOT think they're being captured.
            graph_capture: false,
            ddtree_parent_ids_dev: None,
            tree_aware_attn: None,
            ssm_multi_seq_ptr_table_override: None,
            self_spec_sparse_draft: None,
            ffn_defer: None,
        };

        // Fixed-stride seq_lens / block_tables for the padded batch (only the
        // eager attention path consumes these host vectors; the graphed SSM
        // segments read state via the indirected ptr table instead).
        let seq_lens: Vec<usize> = (0..padded_n)
            .map(|i| if i < n { seqs[i].seq_len } else { 0 })
            .collect();
        let block_tables: Vec<Vec<u32>> = (0..padded_n)
            .map(|i| {
                if i < n {
                    seqs[i].block_table.clone()
                } else {
                    vec![self.dummy_kv_block]
                }
            })
            .collect();

        // Extract real per-seq layer states; append dummy states for the pad
        // positions so `extract_layer_refs` yields `padded_n` refs per layer.
        let mut all_layer_states: Vec<Vec<Box<dyn LayerState>>> = seqs
            .iter_mut()
            .map(|s| std::mem::take(&mut s.layer_states))
            .collect();
        self.append_dummy_pad_states(&mut all_layer_states, n, padded_n)?;

        // ── Partition layers into segments ──
        let segments = self.partition_segments(padded_n);

        // ── Phase 2: run each segment (graphed) / attention layer (eager) ──
        for segment in &segments {
            match *segment {
                Segment::EagerAttn(layer_idx) => {
                    let mut refs = extract_layer_refs(&mut all_layer_states, layer_idx);
                    self.layers[layer_idx].decode_multi_seq(
                        hidden,
                        residual,
                        padded_n,
                        &mut refs,
                        &mut kv_cache,
                        &seq_lens,
                        &block_tables,
                        &ctx,
                        stream,
                    )?;
                }
                Segment::Graphed { start, end, seg_id } => {
                    self.run_graphed_segment(
                        start,
                        end,
                        seg_id,
                        padded_n,
                        hidden,
                        residual,
                        &mut all_layer_states,
                        &mut kv_cache,
                        &seq_lens,
                        &block_tables,
                        &ctx,
                        stream,
                    )?;
                }
            }
        }

        // ── Tail segment: final norm + LM head (all fixed addresses) ──
        self.run_head_tail_segment(padded_n, hidden, stream)?;

        // Restore real layer_states to sequences (dummy pad states dropped).
        for (seq, ls) in seqs.iter_mut().zip(all_layer_states.drain(..n)) {
            seq.layer_states = ls;
        }

        // ── Phase 3: post-loop (eager) — update sequence state ──
        for (i, seq) in seqs.iter_mut().enumerate() {
            seq.tokens.push(tokens[i]);
            seq.seq_len += 1;
        }

        Ok(self.decode_logits_ptr())
    }

    /// Build dummy `SsmLayerState` (pointing at the dedicated `dummy_slot()`)
    /// / empty states for pad positions `[n, padded_n)`, appended to
    /// `all_layer_states`. Mirrors the padding logic in `decode_a2.rs` so pad
    /// SSM kernel writes land in isolated dummy pool memory (PAD_SLOT_ID).
    fn append_dummy_pad_states(
        &self,
        all_layer_states: &mut Vec<Vec<Box<dyn LayerState>>>,
        n: usize,
        padded_n: usize,
    ) -> Result<()> {
        let dummy_ssm_slot = self.ssm_pool.dummy_slot();
        for _pad_pos in n..padded_n {
            let mut dummy: Vec<Box<dyn LayerState>> = Vec::with_capacity(self.layers.len());
            let mut ssm_idx = 0usize;
            for (li, layer) in self.layers.iter().enumerate() {
                if self.config.layer_type(li) == LayerType::LinearAttention {
                    dummy.push(Box::new(SsmLayerState {
                        h_state: self.ssm_pool.h_state(ssm_idx, dummy_ssm_slot),
                        conv_state: self.ssm_pool.conv_state(ssm_idx, dummy_ssm_slot),
                        h_state_checkpoint: None,
                        conv_state_checkpoint: None,
                        h_state_intermediates: Vec::new(),
                        conv_state_intermediates: Vec::new(),
                        wy17_kv_retain: None,
                        wy17_gate_retain: None,
                    }));
                    ssm_idx += 1;
                } else {
                    dummy.push(layer.alloc_state(self.gpu.as_ref())?);
                }
            }
            all_layer_states.push(dummy);
        }
        Ok(())
    }

    /// Partition the layer list into graphed SSM/FFN runs + eager attention
    /// singletons. A LinearAttention layer that is NOT multiseq-graph-safe at
    /// this `padded_n` (e.g. its multi-seq kernel handles are absent) is
    /// promoted to its own eager segment so the graphed cache never captures
    /// a baked-pointer path.
    fn partition_segments(&self, padded_n: usize) -> Vec<Segment> {
        let graph_safe: Vec<bool> = self
            .layers
            .iter()
            .enumerate()
            .map(|(i, layer)| {
                // A layer is capturable iff it is not a FullAttention layer
                // (host-derived split-K grid) AND (if SSM) it takes the
                // indirected multi-seq kernel path.
                self.config.layer_type(i) != LayerType::FullAttention
                    && layer.multiseq_graph_safe(padded_n)
            })
            .collect();
        partition_segments_from_mask(&graph_safe)
    }

    /// Replay (or first-time capture) one graphed segment `[start, end)`.
    fn run_graphed_segment(
        &self,
        start: usize,
        end: usize,
        seg_id: usize,
        padded_n: usize,
        hidden: DevicePtr,
        residual: DevicePtr,
        all_layer_states: &mut [Vec<Box<dyn LayerState>>],
        kv_cache: &mut spark_runtime::kv_cache::PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let key = (padded_n, seg_id);

        // Gather-before-replay: refresh every SSM layer's ptr table so the
        // captured H2D (and our eager H2D here) upload the CURRENT active
        // sequences' h_state/conv_state addresses. Runs OUTSIDE capture.
        for layer_idx in start..end {
            if self.config.layer_type(layer_idx) == LayerType::LinearAttention {
                let mut refs = extract_layer_refs(all_layer_states, layer_idx);
                self.layers[layer_idx].multiseq_refresh_ptr_table(
                    &mut refs,
                    padded_n,
                    self.gpu.as_ref(),
                    stream,
                )?;
            }
        }

        let cached = { self.piecewise_decode_graphs.lock().get(&key).copied() };
        if let Some(graph) = cached {
            if graph.0 != 0 {
                self.gpu.launch_graph(graph, stream)?;
            }
            return Ok(());
        }

        // First sight of this (padded_n, seg_id): capture. The segment body
        // reads only fixed device addresses (SSM ptr scratch, model buffers),
        // so the resulting graph is slot-agnostic.
        let seg_ctx = ForwardContext {
            graph_capture: true,
            ..clone_ctx(ctx)
        };
        self.gpu.begin_capture(stream)?;
        for layer_idx in start..end {
            let mut refs = extract_layer_refs(all_layer_states, layer_idx);
            self.layers[layer_idx].decode_multi_seq(
                hidden,
                residual,
                padded_n,
                &mut refs,
                kv_cache,
                seq_lens,
                block_tables,
                &seg_ctx,
                stream,
            )?;
        }
        let graph = self.gpu.end_capture(stream)?;
        if graph.0 != 0 {
            tracing::info!(
                "MULTISEQ_GRAPHS: captured segment {seg_id} (layers {start}..{end}) for padded_n={padded_n}"
            );
            self.piecewise_decode_graphs.lock().insert(key, graph);
            self.gpu.launch_graph(graph, stream)?;
        }
        Ok(())
    }

    /// Final norm + per-seq LM-head GEMVs, captured as a slot-agnostic tail
    /// segment keyed `(padded_n, usize::MAX)`.
    fn run_head_tail_segment(&self, padded_n: usize, hidden: DevicePtr, stream: u64) -> Result<()> {
        let key = (padded_n, usize::MAX);
        let cached = { self.piecewise_decode_graphs.lock().get(&key).copied() };
        if let Some(graph) = cached {
            if graph.0 != 0 {
                self.gpu.launch_graph(graph, stream)?;
            }
            return Ok(());
        }

        self.gpu.begin_capture(stream)?;
        self.emit_norm_and_head(padded_n, hidden, stream)?;
        let graph = self.gpu.end_capture(stream)?;
        if graph.0 != 0 {
            tracing::info!("MULTISEQ_GRAPHS: captured norm/head tail for padded_n={padded_n}");
            self.piecewise_decode_graphs.lock().insert(key, graph);
            self.gpu.launch_graph(graph, stream)?;
        }
        Ok(())
    }

    /// Final RMS norm [padded_n, H] + padded_n sequential LM-head GEMVs.
    /// Shared by the tail-segment capture; all addresses fixed.
    fn emit_norm_and_head(&self, padded_n: usize, hidden: DevicePtr, stream: u64) -> Result<()> {
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let normed = self.buffers.norm_output();
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            hidden,
            &self.final_norm,
            normed,
            padded_n as u32,
            h as u32,
            self.config.rms_norm_eps as f32,
            stream,
        )?;

        let logits = self.buffers.logits();
        let v = self.config.vocab_size;
        for i in 0..padded_n {
            let normed_i = normed.offset(i * h * bf16);
            let logits_i = logits.offset(i * v * bf16);
            if let Some(ref nvfp4) = self.lm_head_nvfp4 {
                ops::w4a16_gemv(
                    self.gpu.as_ref(),
                    self.w4a16_gemv_kernel,
                    normed_i,
                    nvfp4,
                    logits_i,
                    v as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    self.gpu.as_ref(),
                    self.dense_gemv_kernel,
                    normed_i,
                    &self.lm_head_weight,
                    logits_i,
                    v as u32,
                    h as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}

/// Pure segment-partitioning: given a per-layer "graph-safe" mask, produce
/// the ordered segment list. Maximal runs of `true` become one `Graphed`
/// segment (with monotonically increasing `seg_id`); every `false` layer
/// becomes its own `EagerAttn` segment and breaks the current run.
///
/// Extracted as a free function so the (host-only) control-flow logic is
/// unit-testable without constructing a full `TransformerModel`.
fn partition_segments_from_mask(graph_safe: &[bool]) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut seg_id = 0usize;
    let mut run_start: Option<usize> = None;

    for (i, &safe) in graph_safe.iter().enumerate() {
        if safe {
            run_start.get_or_insert(i);
        } else {
            if let Some(start) = run_start.take() {
                segments.push(Segment::Graphed {
                    start,
                    end: i,
                    seg_id,
                });
                seg_id += 1;
            }
            segments.push(Segment::EagerAttn(i));
        }
    }
    if let Some(start) = run_start.take() {
        segments.push(Segment::Graphed {
            start,
            end: graph_safe.len(),
            seg_id,
        });
    }
    segments
}

/// Shallow clone of a `ForwardContext` (all fields are `Copy` / shared refs).
/// Used to derive a capture-enabled context for a segment without disturbing
/// the eager attention context.
fn clone_ctx<'a>(ctx: &ForwardContext<'a>) -> ForwardContext<'a> {
    ForwardContext {
        buffers: ctx.buffers,
        gpu: ctx.gpu,
        config: ctx.config,
        attn_metadata: ctx.attn_metadata,
        profile: ctx.profile,
        comm: ctx.comm,
        graph_capture: ctx.graph_capture,
        ddtree_parent_ids_dev: None,
        tree_aware_attn: None,
        ssm_multi_seq_ptr_table_override: None,
        self_spec_sparse_draft: None,
            ffn_defer: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{Segment, partition_segments_from_mask};

    /// Compact debug encoding of a segment list for assertions:
    /// graphed run `[s,e)#id` -> "G{s}-{e}#{id}", eager layer -> "E{i}".
    fn encode(segs: &[Segment]) -> Vec<String> {
        segs.iter()
            .map(|s| match *s {
                Segment::Graphed { start, end, seg_id } => format!("G{start}-{end}#{seg_id}"),
                Segment::EagerAttn(i) => format!("E{i}"),
            })
            .collect()
    }

    #[test]
    fn all_safe_is_one_graphed_run() {
        let mask = vec![true; 5];
        assert_eq!(encode(&partition_segments_from_mask(&mask)), ["G0-5#0"]);
    }

    #[test]
    fn all_eager_no_graphed_segments() {
        let mask = vec![false; 3];
        assert_eq!(
            encode(&partition_segments_from_mask(&mask)),
            ["E0", "E1", "E2"]
        );
    }

    #[test]
    fn interleaved_attn_breaks_runs_and_bumps_seg_id() {
        // SSM SSM ATTN SSM SSM SSM ATTN SSM  (true = SSM/graph-safe)
        let mask = vec![true, true, false, true, true, true, false, true];
        assert_eq!(
            encode(&partition_segments_from_mask(&mask)),
            ["G0-2#0", "E2", "G3-6#1", "E6", "G7-8#2"],
        );
    }

    #[test]
    fn leading_attn_then_run() {
        let mask = vec![false, true, true];
        assert_eq!(
            encode(&partition_segments_from_mask(&mask)),
            ["E0", "G1-3#0"],
        );
    }

    #[test]
    fn empty_mask_yields_no_segments() {
        assert!(partition_segments_from_mask(&[]).is_empty());
    }

    #[test]
    fn seg_ids_are_contiguous_across_multiple_runs() {
        // Three separate SSM runs must get seg_id 0,1,2 so their graph-cache
        // keys (padded_n, seg_id) never collide.
        let mask = vec![true, false, true, false, true];
        let ids: Vec<usize> = partition_segments_from_mask(&mask)
            .into_iter()
            .filter_map(|s| match s {
                Segment::Graphed { seg_id, .. } => Some(seg_id),
                Segment::EagerAttn(_) => None,
            })
            .collect();
        assert_eq!(ids, [0, 1, 2]);
    }
}
