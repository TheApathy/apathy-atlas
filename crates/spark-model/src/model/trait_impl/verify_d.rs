// SPDX-License-Identifier: AGPL-3.0-only

//! K=γ (DFlash) verify path.
//!
//! ## Safety
//!
//! `unsafe { from_raw_parts(...) }` blocks reinterpret stack arrays
//! / `Vec`s of POD integers (`u32`, `i32`, `i64`, `usize`) as byte
//! slices for H2D upload. See `verify_c.rs` module docs for the full
//! safety contract — same pattern, same invariants here.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn decode_verify_graphed_kgamma_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let k = tokens.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = 2usize;

        // Item #2 (STree-style in-place K=γ verify): `h_state` IS canonical
        // — the verify kernel reads/writes it directly and the commit
        // (`commit_accepted_prefix`) rewinds it in place on reject. No
        // scratch/canonical split — dual-buffer pre-verify copy eliminated.
        // Modeled on verify_b.rs (K=2 in-place).

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 1: Pre-graph (varies per step, NOT captured) ──

        // 1a. Embed K tokens
        for t in 0..k {
            self.embed(tokens[t], hidden.offset(t * h * fp32), stream)?;
        }

        // 1b. Allocate KV blocks for all K positions
        let bs = kv_cache.block_size();
        for t in 0..k {
            let pos = seq.seq_len + t;
            let blocks_needed = (pos / bs) + 1;
            ensure_blocks_through_decode(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
            )?;
        }

        // 1c. Upload K-entry attention metadata. Layout in scratch (after
        // mtp metadata reservation): positions[K*4] | slots[K*8] | seq_lens[K*4]
        // | block_table[K*max_blocks*4]. Need K*16 + K*max_blocks*4 bytes per
        // call — at K=17 max_blocks=512 that's ~36 KB which fits comfortably
        // in the scratch arena (offset 32768).
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;

        let positions: Vec<u32> = (0..k).map(|t| (seq.seq_len + t) as u32).collect();
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, k * 4) };
        self.gpu.copy_h2d_async(pos_bytes, meta_base, stream)?;

        let mut slots = vec![0i64; k];
        for t in 0..k {
            let pos = seq.seq_len + t;
            let block_idx = pos / bs;
            let block_offset = pos % bs;
            let physical_block = seq.physical_block_for(block_idx).unwrap_or(0);
            slots[t] = (physical_block as i64) * (bs as i64) + (block_offset as i64);
        }
        // 256-byte gap mirrors K=4 layout for ABI compatibility with
        // attention kernels that index meta_base + fixed offsets.
        let slot_bytes = unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, k * 8) };
        self.gpu
            .copy_h2d_async(slot_bytes, meta_base.offset(256), stream)?;

        let seq_lens: Vec<i32> = (0..k).map(|t| (seq.seq_len + t + 1) as i32).collect();
        let sl_bytes = unsafe { std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, k * 4) };
        self.gpu
            .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

        let mb = max_blocks as usize;
        let needed = k * mb;
        let mut bt_buf = vec![0i32; needed];
        for row in 0..k {
            for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                bt_buf[row * mb + j] = block as i32;
            }
        }
        let bt_bytes =
            unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, needed * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(768), stream)?;

        // Request-scoped LoRA routing (graphed γ-verify) — see verify_b.rs. One
        // sequence → one adapter; [K]-all-equal buffer at the +128 gap, uploaded
        // pre-`begin_capture`. γ spec depth MUST stay ≤ 32 or +128+K*4 would
        // overrun slot@+256. `DevicePtr(0)` (no pool) → installed-pair path.
        debug_assert!(k <= 32, "γ verify seq_slot +128 gap holds K ≤ 32");
        let seq_slot =
            self.upload_seq_slot_uniform(seq.adapter_slot, k, meta_base.offset(128), stream)?;

        let metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: max_blocks,
            num_seqs: k as u32,
            seq_slot,
        };

        // Phase 6.2.c — HSS host I/O is illegal under CUDA graph capture.
        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        // ATLAS_DFLASH_DEBUG_NO_GRAPH=1 forces eager (no graph capture) so
        // CUDA_LAUNCH_BLOCKING=1 reports the exact failing kernel — used
        // to localize K=γ illegal-address crashes downstream of SSM.
        let force_eager = std::env::var("ATLAS_DFLASH_DEBUG_NO_GRAPH").ok().as_deref() == Some("1");
        // ATLAS_LORA_EAGER: LoRA graph-vs-eager debugging hatch (see decode_a).
        let lora_eager = self.lora.is_some() && crate::lora::lora_eager_env();
        let use_graphs = self.comm.is_none()
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !hss_engaged
            && !force_eager
            && !lora_eager;

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(metadata),
            // ATLAS_PROFILE=1 lights up the per-op `prof!` timers (MoE gate/topk/
            // exp_gate_up/exp_silu_down/wsum_blend) inside the verify forward so we
            // can split the ~682ms/step into attention vs MoE. Gated on `!use_graphs`
            // because `prof!` calls `synchronize()`, illegal mid graph-capture — so
            // it only fires in eager mode (pair with ATLAS_DFLASH_DEBUG_NO_GRAPH=1).
            profile: !use_graphs && std::env::var("ATLAS_PROFILE").is_ok(),
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            gdn_exact_replay: false,
            token_ids: None,
            routed_lora_layers: None, // #30: decode/verify never routes prefill.
            midchunk_capture: None,
        };

        // ── Phase 2: CUDA graph capture / replay ──

        let mut graph_cache = if use_graphs {
            Some(self.verify_kgamma_graph.lock())
        } else {
            None
        };

        let cache_key = (seq.slot_idx, k);
        let cached_for_slot = graph_cache
            .as_ref()
            .and_then(|c| c.get(&cache_key).copied());
        if let Some(graph) = cached_for_slot
            && graph.0 != 0
        {
            self.gpu.launch_graph(graph, stream)?;
        }
        let need_run = cached_for_slot.is_none();
        if need_run {
            let seq_lens_vec: Vec<usize> = (0..k).map(|t| seq.seq_len + t).collect();
            let block_tables_vec: Vec<Vec<u32>> = vec![seq.block_table.clone(); k];

            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let layer_type = self.config.layer_type(layer_idx);

                // forward_k16 speed probe: route SLIDING-window layers through the
                // batched decode_multi_seq path too (not just FullAttention), so all
                // 48 layers' MoE goes through forward_kn. Valid when ctx_len < window
                // (512): SWA masks nothing, so multi_seq (full-attn paged decode) is
                // bit-equivalent. ATLAS_DFLASH_ALL_MULTISEQ=1, short-context only.
                let all_multiseq =
                    std::env::var("ATLAS_DFLASH_ALL_MULTISEQ").ok().as_deref() == Some("1");
                // ATLAS_DFLASH_ALL_BATCHED=1 (debug): route ALL layers through the
                // sequential decode_batched loop (no decode_multi_seq at all) —
                // isolates whether the multiseq paged-decode path is the source of
                // the verify-argmax divergence from plain decode.
                let all_batched =
                    std::env::var("ATLAS_DFLASH_ALL_BATCHED").ok().as_deref() == Some("1");
                let use_multiseq = !hss_engaged
                    && !all_batched
                    && (layer_type == LayerType::FullAttention || all_multiseq);
                if use_multiseq {
                    let mut dummy_states: Vec<Box<dyn LayerState>> = (0..k)
                        .map(|_| layer.alloc_state(self.gpu.as_ref()))
                        .collect::<Result<_>>()?;
                    let mut refs: Vec<&mut (dyn LayerState + 'static)> =
                        dummy_states.iter_mut().map(|s| s.as_mut()).collect();
                    layer.decode_multi_seq(
                        hidden,
                        residual,
                        k,
                        &mut refs,
                        &mut kv_cache,
                        &seq_lens_vec,
                        &block_tables_vec,
                        &ctx,
                        stream,
                    )?;
                } else {
                    // HSS or sliding (default): sequential single-token decodes.
                    //
                    // ROOT-CAUSE FIX (docs/10): the single-token decode attention
                    // consumes ctx.attn_metadata (pre-uploaded device metadata),
                    // NOT its seq_len argument. Passing the K-row verify metadata
                    // unchanged made every row read ENTRY 0 — same RoPE position,
                    // same KV slot — so all K rows collapsed onto position
                    // seq_len, the last draft's K/V overwrote the committed
                    // token's slot each step, and generation degenerated
                    // progressively. Give each row its own metadata view.
                    for t in 0..k {
                        let mb = metadata.max_blocks_per_seq as usize;
                        let meta_t = AttnMetadataDev {
                            positions: metadata.positions.offset(t * 4),
                            positions_h: metadata.positions_h.offset(t * 4),
                            positions_w: metadata.positions_w.offset(t * 4),
                            slot: metadata.slot.offset(t * 8),
                            seq_len: metadata.seq_len.offset(t * 4),
                            block_table: metadata.block_table.offset(t * mb * 4),
                            max_blocks_per_seq: metadata.max_blocks_per_seq,
                            num_seqs: 1,
                            seq_slot: if metadata.seq_slot.0 != 0 {
                                metadata.seq_slot.offset(t * 4)
                            } else {
                                metadata.seq_slot
                            },
                        };
                        let ctx_t = ForwardContext {
                            buffers: ctx.buffers,
                            gpu: ctx.gpu,
                            config: ctx.config,
                            attn_metadata: Some(meta_t),
                            profile: ctx.profile,
                            comm: ctx.comm,
                            graph_capture: ctx.graph_capture,
                            gdn_exact_replay: ctx.gdn_exact_replay,
                            token_ids: ctx.token_ids,
                            routed_lora_layers: ctx.routed_lora_layers,
                            midchunk_capture: None, // decode/verify never midchunk-captures
                        };
                        layer.decode(
                            hidden.offset(t * h * bf16),
                            residual.offset(t * h * bf16),
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len + t,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            &ctx_t,
                            stream,
                        )?;
                    }
                }
                // DFlash intermediate hidden capture: snapshot each capture
                // layer's output at position k-1 (last verify token) into
                // dflash_hidden_save[slot] while hidden_states still holds
                // this layer's activation — mirrors verify_b.rs for K=2.
                // Must be inside the graph capture region so the per-layer
                // intermediate (not the final-layer-only post-loop value) is
                // recorded. Under ATLAS_DFLASH_EAGLE_FIX=1 OR
                // ATLAS_DFLASH_UNIFIED_CTX=1, capture ALL k verify rows so
                // the scheduler can append rows 0..=num_accepted to ctx
                // after the accept walk (EAGLE order). UNIFIED_CTX requires
                // the same full capture: commit_ctx copies scratch rows
                // 0..=num_accepted — with only the k-1 capture, row 0 holds
                // the WRONG token's hidden and rows 1.. are stale garbage
                // (2026-07-09 accept-collapse root cause: EAGLE_FIX=0 under
                // UNIFIED=1 starved this capture and poisoned drafter ctx).
                let capture_all = std::env::var("ATLAS_DFLASH_EAGLE_FIX").ok().as_deref()
                    == Some("1")
                    || std::env::var("ATLAS_DFLASH_UNIFIED_CTX").ok().as_deref() == Some("1");
                if capture_all {
                    self.try_dflash_capture_all(layer_idx, k, stream)?;
                } else {
                    self.try_dflash_capture(layer_idx, k - 1, stream)?;
                }
            }

            // Final norm [K, H]
            let normed = self.buffers.norm_output();
            ops::rms_norm(
                self.gpu.as_ref(),
                self.rms_norm_kernel,
                hidden,
                &self.final_norm,
                normed,
                k as u32,
                h as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;

            // LM head for K tokens
            self.lm_head_batched(normed, k as u32, self.buffers.logits(), stream)?;

            // Argmax inside graph (fixed scratch addresses — graph-safe)
            let vocab = self.config.vocab_size;
            let argmax_out = self.buffers.scratch();
            for t in 0..k {
                let logits_t = self.buffers.logits().offset(t * vocab * bf16);
                let out_t = argmax_out.offset(t * 4);
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    logits_t,
                    out_t,
                    vocab as u32,
                    stream,
                )?;
            }

            if use_graphs {
                let graph = self.gpu.end_capture(stream)?;
                if graph.0 != 0 {
                    tracing::info!(
                        "Captured CUDA graph for K=γ verify (slot={} K={})",
                        seq.slot_idx,
                        k
                    );
                    if let Some(ref mut cache) = graph_cache {
                        cache.insert(cache_key, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }
        }

        // ── Phase 3: Post-graph (D2H copy only) ──

        let out_ptr = self.buffers.scratch();
        let mut buf = vec![0u8; k * 4];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let mut out = Vec::with_capacity(k);
        for t in 0..k {
            let off = t * 4;
            out.push(u32::from_le_bytes([
                buf[off],
                buf[off + 1],
                buf[off + 2],
                buf[off + 3],
            ]));
        }

        // See decode_verify_graphed for rationale on `seq_len += k` fix.
        for &t in tokens {
            seq.tokens.push(t);
        }
        seq.seq_len += k;

        Ok(out)
    }
}
