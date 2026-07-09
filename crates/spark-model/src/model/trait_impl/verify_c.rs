// SPDX-License-Identifier: AGPL-3.0-only

//! K=3 verify path.
//!
//! ## Safety contract for the `unsafe { from_raw_parts(...) }` blocks
//!
//! This file casts small stack arrays / `Vec`s of plain integers
//! (`u32`, `i32`, `i64`) into byte slices to feed `copy_h2d_async`.
//! Invariants:
//! - All source types are POD (no padding, all-bits-valid), so the
//!   reinterpretation as `&[u8]` is sound.
//! - The byte length matches `N * size_of::<T>()` exactly.
//! - The original array/`Vec` outlives the H2D copy: copy_h2d_async on
//!   our cudarc backend completes synchronously enough for the caller
//!   to drop the host buffer after this function returns (the actual
//!   device copy is async on the stream, but the source bytes are
//!   already in the driver's pinned-memory queue).

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
    pub(super) fn decode_verify_graphed_k3_dispatch(
        &self,
        tokens: &[u32; 3],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 3]> {
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let k = 3usize;

        // F62 (2026-04-27): SpecMamba dual-buffer pre-verify copy.
        self.pre_verify_copy_async(seq)?;

        // ATLAS_FULL_PROFILE=1: bump per-step counter (warmup logic lives in
        // full_profile.rs). Cheap atomic load when env var is unset.
        crate::full_profile::begin_step();

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 1: Pre-graph (varies per step, NOT captured) ──

        // 1a. Embed 3 tokens
        crate::kprof!(self.gpu.as_ref(), stream, "embed", {
            for t in 0..k {
                self.embed(tokens[t], hidden.offset(t * h * fp32), stream)?;
            }
            anyhow::Result::<()>::Ok(())
        })?;

        // 1b. Allocate KV blocks for all 3 positions
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

        // 1c. Upload 3-entry attention metadata as a SINGLE H2D copy.
        //
        // Earlier shape issued FOUR copy_h2d_async calls (positions @ 0,
        // slots @ 256, seq_lens @ 512, block_table @ 768) each carrying
        // ~5 μs of CUDA-driver / cuMemcpyHtoDAsync API overhead. Pack
        // them into one contiguous host buffer and issue a single H2D
        // covering the [0..768 + block_table_bytes] union. The gap
        // regions (12..256, 280..512, 524..768) are zero-initialised but
        // unread by the attention kernel — it indexes positions[0..k],
        // slot[0..k], seq_len[0..k], block_table[0..k*max_blocks] at
        // fixed offsets, never touching the gaps.
        //
        // Saves 3 driver-API calls per K=3 verify step (~15 μs of CPU
        // syscall overhead at ~5 μs each on GB10). On a 256-token
        // response that's ~4 ms of saved per-token launch latency.
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;
        let mb = max_blocks as usize;
        let needed = k * mb;
        let pack_bytes = 768 + needed * 4;

        // Stack fast path for typical block-table sizes (k=3 × 1024 max
        // blocks = 3072 ints × 4 = 12 KB + 768 header = ~13 KB); heap
        // fallback retained for larger max_seq_len configs.
        let mut pack_stack = [0u8; 1024];
        let mut pack_heap: Vec<u8>;
        let pack: &mut [u8] = if pack_bytes <= pack_stack.len() {
            &mut pack_stack[..pack_bytes]
        } else {
            pack_heap = vec![0u8; pack_bytes];
            &mut pack_heap
        };

        // positions at offset 0 (3 × u32 = 12 bytes)
        let positions = [
            seq.seq_len as u32,
            (seq.seq_len + 1) as u32,
            (seq.seq_len + 2) as u32,
        ];
        pack[0..12].copy_from_slice(unsafe {
            std::slice::from_raw_parts(positions.as_ptr() as *const u8, 12)
        });

        // slots at offset 256 (3 × i64 = 24 bytes)
        let mut slots = [0i64; 3];
        for t in 0..k {
            let pos = seq.seq_len + t;
            let block_idx = pos / bs;
            let block_offset = pos % bs;
            let physical_block = seq.physical_block_for(block_idx).unwrap_or(0);
            slots[t] = (physical_block as i64) * (bs as i64) + (block_offset as i64);
        }
        pack[256..280].copy_from_slice(unsafe {
            std::slice::from_raw_parts(slots.as_ptr() as *const u8, 24)
        });

        // seq_lens at offset 512 (3 × i32 = 12 bytes)
        let seq_lens = [
            (seq.seq_len + 1) as i32,
            (seq.seq_len + 2) as i32,
            (seq.seq_len + 3) as i32,
        ];
        pack[512..524].copy_from_slice(unsafe {
            std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, 12)
        });

        // block_table at offset 768 (k × max_blocks × i32). All K rows
        // carry the same physical sequence so the inner loop writes
        // identical content to each row.
        let bt_bytes_dst = &mut pack[768..768 + needed * 4];
        for row in 0..k {
            for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                let off = (row * mb + j) * 4;
                bt_bytes_dst[off..off + 4].copy_from_slice(&(block as i32).to_le_bytes());
            }
        }

        // Single fused H2D
        self.gpu.copy_h2d_async(pack, meta_base, stream)?;

        let metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: max_blocks,
            num_seqs: k as u32,
        };

        // Phase 6.2.c — HSS host I/O is illegal under CUDA graph capture.
        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        // ATLAS_FULL_PROFILE=1 disables graph capture so per-kernel sync is
        // legal (CUDA graph capture forbids host syncs / D2H copies inside).
        let full_profile = crate::full_profile::is_enabled();
        let use_graphs = self.comm.is_none() && !hss_engaged && !full_profile;

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(metadata),
            profile: full_profile,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            ddtree_parent_ids_dev: None,
            tree_aware_attn: None,
            ssm_multi_seq_ptr_table_override: None,
            self_spec_sparse_draft: None,
            ffn_defer: None,
        };

        // ── Phase 2: CUDA graph capture / replay ──

        let mut graph_cache = if use_graphs {
            Some(self.verify3_graph.lock())
        } else {
            None
        };

        // SLOT-KEYED LOOKUP: only replay if this seq's slot has a captured graph.
        let cached_for_slot = graph_cache
            .as_ref()
            .and_then(|c| c.get(&seq.slot_idx).copied());
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

                if layer_type == LayerType::FullAttention {
                    if hss_engaged {
                        // HSS path: decode_multi_seq's paged-decode kernel
                        // reads K/V from HBM only, missing the long-context
                        // history on disk. Fall back to decode_batched
                        // (sequential single-token decodes via the HSS
                        // orchestrator). See verify_b.rs for full rationale.
                        layer.decode_batched(
                            hidden,
                            residual,
                            k,
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            &ctx,
                            stream,
                        )?;
                    } else {
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
                    }
                } else {
                    layer.decode_batched(
                        hidden,
                        residual,
                        k,
                        seq.layer_states[layer_idx].as_mut(),
                        &mut kv_cache,
                        seq.seq_len,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &ctx,
                        stream,
                    )?;
                }
                // DFlash hidden capture for ctx conditioning / training
                // data collection (ATLAS_DUMP_HIDDEN). No-op when both
                // `dflash_hidden_save` is None AND `dflash_capture_layers`
                // is empty (production hot path: zero overhead).
                for t in 0..k {
                    self.try_dflash_capture(layer_idx, t, stream)?;
                }
            }

            // Final norm [3, H]
            let normed = self.buffers.norm_output();
            crate::kprof!(self.gpu.as_ref(), stream, "final_norm", {
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
                anyhow::Result::<()>::Ok(())
            })?;

            // LM head for 3 tokens
            crate::kprof!(self.gpu.as_ref(), stream, "lm_head", {
                self.lm_head_batched(normed, k as u32, stream)?;
                anyhow::Result::<()>::Ok(())
            })?;

            // Argmax inside graph. Use the verify-side (possibly truncated)
            // vocab so the per-row logits stride AND the argmax range match
            // what `lm_head_batched` wrote — full vocab when truncation off.
            let vocab = self.verify_lmhead_vocab() as usize;
            let argmax_out = self.buffers.scratch();
            crate::kprof!(self.gpu.as_ref(), stream, "argmax", {
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
                anyhow::Result::<()>::Ok(())
            })?;

            if use_graphs {
                let graph = self.gpu.end_capture(stream)?;
                if graph.0 != 0 {
                    tracing::info!("Captured CUDA graph for K=3 verify (slot={})", seq.slot_idx);
                    if let Some(ref mut cache) = graph_cache {
                        cache.insert(seq.slot_idx, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }
        }

        // ── Phase 3: Post-graph (D2H copy only) ──

        // ATLAS_DUMP_HIDDEN: flush captured layer hiddens to file. See
        // verify_b.rs for the eager-mode safety contract.
        self.flush_hidden_dump(k)?;

        let out_ptr = self.buffers.scratch();
        let mut buf = [0u8; 12];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let tok0 = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let tok1 = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let tok2 = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);

        // See decode_verify_graphed for rationale on `seq_len += k` fix.
        for &t in tokens {
            seq.tokens.push(t);
        }
        seq.seq_len += k;

        Ok([tok0, tok1, tok2])
    }
}
