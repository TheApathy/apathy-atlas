// SPDX-License-Identifier: AGPL-3.0-only

//! K=2 verify path.
//!
//! ## Safety
//!
//! `unsafe { from_raw_parts(...) }` blocks reinterpret stack arrays
//! / `Vec`s of POD integers as byte slices for H2D upload.
//! See `verify_c.rs` module docs for the full safety contract.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, HostToDeviceCopy, KernelHandle};
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
    /// Exact token-major oracle for Qwen4's four-stream K=2 verification.
    /// Qwen4 decode-local scratch is not row-disjoint, so both tokens must
    /// traverse the complete model one at a time.
    fn decode_verify_qwen4_token_major(
        &self,
        tokens: &[u32; 2],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<[u32; 2]> {
        let row_bytes = self.config.residual_width() * 2;
        let hidden = self.buffers.hidden_states();

        let logits0 = self.decode_dispatch(tokens[0], seq, stream)?;
        let token0 = self.argmax_on_device(logits0, stream)?;
        let logits_row_bytes = self.config.vocab_size * if self.use_fp32_logits { 4 } else { 2 };
        let mut logits0_host = vec![0u8; logits_row_bytes];
        self.gpu.copy_d2h(logits0, &mut logits0_host)?;
        self.gpu
            .copy_d2d_async(hidden, self.mtp_hidden_save, row_bytes, stream)?;

        let mut ssm_layer_idx = 0usize;
        for (layer_idx, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(layer_idx) != LayerType::LinearAttention {
                continue;
            }
            let ssm = layer_state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "qwen4_exp token-major verify expected SSM state at layer {layer_idx}"
                    )
                })?;
            self.gpu.copy_d2d_async(
                ssm.h_state,
                self.ssm_pool.h_intermediate(ssm_layer_idx, seq.slot_idx, 0),
                self.ssm_pool.h_bytes,
                stream,
            )?;
            self.gpu.copy_d2d_async(
                ssm.conv_state,
                self.ssm_pool
                    .conv_intermediate(ssm_layer_idx, seq.slot_idx, 0),
                self.ssm_pool.conv_bytes,
                stream,
            )?;
            ssm_layer_idx += 1;
        }
        if let Some(ple) = &self.qwen4_ple {
            ple.save_intermediate(seq.slot_idx, 0, self.gpu.as_ref(), stream)?;
        }

        let logits1 = self.decode_dispatch(tokens[1], seq, stream)?;
        let token1 = self.argmax_on_device(logits1, stream)?;
        let mut logits1_host = vec![0u8; logits_row_bytes];
        self.gpu.copy_d2h(logits1, &mut logits1_host)?;
        // The policy verifier consumes the resident conventional [K, vocab]
        // layout after target verification. Ordinary decode writes row zero,
        // so reconstruct both rows explicitly for the exact sampler walk.
        let logits_base = self.decode_logits_ptr();
        self.gpu.copy_h2d_group_on_stream(
            &[
                HostToDeviceCopy::new(&logits0_host, logits_base),
                HostToDeviceCopy::new(&logits1_host, logits_base.offset(logits_row_bytes)),
            ],
            stream,
        )?;
        if let Some(ple) = &self.qwen4_ple {
            ple.save_intermediate(seq.slot_idx, 1, self.gpu.as_ref(), stream)?;
        }

        // Restore the conventional verify layout [row0, row1] for
        // save_hidden_for_mtp(accepted_row).
        self.gpu
            .copy_d2d_async(hidden, hidden.offset(row_bytes), row_bytes, stream)?;
        self.gpu
            .copy_d2d_async(self.mtp_hidden_save, hidden, row_bytes, stream)?;

        Ok([token0, token1])
    }

    pub(super) fn decode_verify_graphed_dispatch(
        &self,
        tokens: &[u32; 2],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 2]> {
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let persistent_row_bytes = self.config.residual_width() * fp32;
        let k = 2usize;

        // F62 (2026-04-27): SpecMamba dual-buffer pre-verify copy.
        // Copy canonical SSM state (h_state_checkpoint) → scratch (h_state)
        // BEFORE the kernel runs. The kernel mutates the scratch; the
        // canonical is preserved across verify until commit.
        self.pre_verify_copy_async(seq)?;

        let qwen4_token_major_oracle = std::env::var("ATLAS_QWEN4_VERIFY_TOKEN_MAJOR")
            .ok()
            .as_deref()
            == Some("1");
        if self.config.is_qwen4_exp() && qwen4_token_major_oracle {
            return self.decode_verify_qwen4_token_major(tokens, seq, stream);
        }

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 1: Pre-graph (varies per step, NOT captured) ──

        // 1a. Embed 2 tokens
        self.embed(tokens[0], hidden, stream)?;
        self.embed(tokens[1], hidden.offset(persistent_row_bytes), stream)?;

        // 1b. Allocate KV blocks for both positions
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

        // 1c. Upload 2-entry attention metadata as a SINGLE packed H2D.
        // See verify_c.rs (K=3 sibling) for the rationale on the 4→1 H2D
        // pack: each separate H2D carries CUDA-driver overhead;
        // packing saves 3 syscalls per K=2 verify step.
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;
        let mb = max_blocks as usize;
        let needed = k * mb;
        let pack_bytes = 768 + needed * 4;

        let mut pack_stack = [0u8; 1024];
        let mut pack_heap: Vec<u8>;
        let pack: &mut [u8] = if pack_bytes <= pack_stack.len() {
            &mut pack_stack[..pack_bytes]
        } else {
            pack_heap = vec![0u8; pack_bytes];
            &mut pack_heap
        };

        // positions at offset 0 (2 × u32)
        let positions = [seq.seq_len as u32, (seq.seq_len + 1) as u32];
        pack[0..8].copy_from_slice(unsafe {
            std::slice::from_raw_parts(positions.as_ptr() as *const u8, 8)
        });

        // slots at offset 256 (2 × i64)
        let mut slots = [0i64; 2];
        for t in 0..k {
            let pos = seq.seq_len + t;
            let block_idx = pos / bs;
            let block_offset = pos % bs;
            let physical_block = seq.physical_block_for(block_idx).unwrap_or(0);
            slots[t] = (physical_block as i64) * (bs as i64) + (block_offset as i64);
        }
        pack[256..272].copy_from_slice(unsafe {
            std::slice::from_raw_parts(slots.as_ptr() as *const u8, 16)
        });

        // seq_lens at offset 512 (2 × i32)
        let seq_lens = [(seq.seq_len + 1) as i32, (seq.seq_len + 2) as i32];
        pack[512..520].copy_from_slice(unsafe {
            std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, 8)
        });

        // block_table at offset 768 (k × max_blocks × i32). All K rows
        // carry the same physical sequence.
        let bt_bytes_dst = &mut pack[768..768 + needed * 4];
        for row in 0..k {
            for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                let off = (row * mb + j) * 4;
                bt_bytes_dst[off..off + 4].copy_from_slice(&(block as i32).to_le_bytes());
            }
        }

        // Single fused H2D
        self.gpu
            .copy_h2d_group_on_stream(&[HostToDeviceCopy::new(pack, meta_base)], stream)?;

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

        // CUDA graphs cannot capture NCCL all-reduce (disabled for EP).
        // Also disable for FP8 native: w8a16_gemv kernel's __shared__ LUT load
        // has CUDA graph capture compatibility issues.
        //
        // Honor `suppress_graphs` so FP8 KV calibration runs eagerly during
        // warmup (its observe() does host syncs that are illegal inside CUDA
        // graph capture — CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, status 900).
        // With MTP, every decode step lands here (regular `decode` is never
        // called), so we also drive the same auto-unsuppress trigger that
        // `decode` uses: once `seq_len > calibration_tokens + 10` the scales
        // are frozen and graphs become safe.
        //
        // Single atomic load below, reused for both the "freeze graphs"
        // store decision and the final `use_graphs` flag. The earlier
        // shape did one load before the if-block + a second load after
        // (to derive use_graphs); they always returned the same value
        // because we only mutate via the store on the path that took
        // the if. ATLAS_DUMP_HIDDEN check uses the cached helper now
        // instead of re-reading the env var per verify call.
        let mut suppress_graphs = self
            .suppress_graphs
            .load(std::sync::atomic::Ordering::Relaxed);
        if suppress_graphs
            && seq.seq_len > self.config.fp8_kv_calibration_tokens + 10
            && !crate::model::env_diag::dump_hidden_enabled()
        {
            self.suppress_graphs
                .store(false, std::sync::atomic::Ordering::Relaxed);
            suppress_graphs = false;
            tracing::info!("FP8 calibration frozen — re-enabling CUDA graphs (MTP verify)");
        }
        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        // ATLAS_MOE_OVERLAP=1: force eager so the per-step D2H overlap probe in
        // forward_k2 is legal (illegal under graph capture). Measurement-only.
        let moe_overlap_probe = std::env::var("ATLAS_MOE_OVERLAP").ok().as_deref() == Some("1");
        let use_graphs = self.comm.is_none()
            && !suppress_graphs
            // Qwen4 PLE performs host-backed sparse row fetches and its
            // four-stream K=2 path is intentionally serialized for exactness.
            && !self.config.is_qwen4_exp()
            // Phase 6.2.c — see decode() for rationale: HSS path's host I/O is
            // illegal under CUDA graph capture.
            && !hss_engaged
            && !moe_overlap_probe;

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(metadata),
            profile: false,
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
            Some(self.verify2_graph.lock())
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

            // Extract layer states. Attention layers use EmptyLayerState (no actual
            // state), so sharing the same alloc is safe. For SSM layers, only one
            // sequence's state exists — pass it to decode_batched directly.
            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }

            // ATLAS_MOE_OVERLAP probe: coarse per-layer-type wall accumulators
            // (eager-only — `moe_overlap_probe` forced use_graphs=false). Lets
            // us split the K=2 verify cost into attn vs SSM(+MoE) vs head.
            let mut attn_ns: u64 = 0;
            let mut ssm_ns: u64 = 0;
            let probe = moe_overlap_probe;
            let mut qwen4_ssm_layer_idx = 0usize;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let layer_type = self.config.layer_type(layer_idx);
                let __lt0 = if probe {
                    self.gpu.synchronize(stream)?;
                    Some(std::time::Instant::now())
                } else {
                    None
                };

                if self.config.is_qwen4_exp() {
                    // The four-stream kernels are decode-native. Run the two
                    // verify rows serially, but preserve the state after row
                    // zero so a rejected draft can commit the always-accepted
                    // target token exactly like the optimized K=2 path.
                    for (row, &token) in tokens.iter().enumerate() {
                        if layer_idx == 1
                            && let Some(ple) = &self.qwen4_ple
                        {
                            let mut prior = seq.tokens.clone();
                            prior.extend_from_slice(&tokens[..row]);
                            ple.forward_token(
                                token,
                                &prior,
                                hidden.offset(row * persistent_row_bytes),
                                seq.slot_idx,
                                false,
                                self.gpu.as_ref(),
                                stream,
                            )?;
                            // PLE commit uses the accepted-row index even on
                            // a full K=2 accept. Preserve both rows: saving
                            // only row zero makes a 2/2 accept restore stale
                            // slot one on the next speculative cycle.
                            ple.save_intermediate(seq.slot_idx, row, self.gpu.as_ref(), stream)?;
                        }

                        let token_metadata = AttnMetadataDev {
                            positions: metadata.positions.offset(row * 4),
                            positions_h: metadata.positions_h.offset(row * 4),
                            positions_w: metadata.positions_w.offset(row * 4),
                            slot: metadata.slot.offset(row * 8),
                            seq_len: metadata.seq_len.offset(row * 4),
                            block_table: metadata.block_table.offset(row * mb * 4),
                            num_seqs: 1,
                            ..metadata
                        };
                        let token_ctx = ForwardContext {
                            attn_metadata: Some(token_metadata),
                            ..ctx
                        };
                        layer.decode(
                            hidden.offset(row * persistent_row_bytes),
                            residual.offset(row * persistent_row_bytes),
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len + row,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            &token_ctx,
                            stream,
                        )?;

                        if std::env::var("ATLAS_QWEN4_VERIFY_DUMP").ok().as_deref() == Some("1") {
                            self.gpu.synchronize(stream)?;
                            let mut buf = vec![0u8; persistent_row_bytes];
                            self.gpu
                                .copy_d2h(hidden.offset(row * persistent_row_bytes), &mut buf)?;
                            std::fs::write(
                                format!(
                                    "/tmp/atlas_qwen4_k2_seqlen{}_row{row}_layer{layer_idx}.bin",
                                    seq.seq_len
                                ),
                                buf,
                            )?;
                        }

                        if row == 0 && layer_type == LayerType::LinearAttention {
                            let ssm = seq.layer_states[layer_idx]
                                .as_any_mut()
                                .downcast_mut::<SsmLayerState>()
                                .ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "qwen4_exp verify expected SSM state at layer {layer_idx}"
                                    )
                                })?;
                            let nv = self.config.linear_num_value_heads;
                            let vd = self.config.linear_value_head_dim;
                            let nk = self.config.linear_num_key_heads;
                            let kd = self.config.linear_key_head_dim;
                            let h_bytes = nv * vd * kd * 4;
                            let conv_dim = nk * kd * 2 + nv * vd;
                            let conv_bytes = conv_dim * self.config.linear_conv_kernel_dim * 4;
                            self.gpu.copy_d2d_async(
                                ssm.h_state,
                                self.ssm_pool
                                    .h_intermediate(qwen4_ssm_layer_idx, seq.slot_idx, 0),
                                h_bytes,
                                stream,
                            )?;
                            self.gpu.copy_d2d_async(
                                ssm.conv_state,
                                self.ssm_pool.conv_intermediate(
                                    qwen4_ssm_layer_idx,
                                    seq.slot_idx,
                                    0,
                                ),
                                conv_bytes,
                                stream,
                            )?;
                        }
                    }
                    if layer_type == LayerType::LinearAttention {
                        qwen4_ssm_layer_idx += 1;
                    }
                } else if layer_type == LayerType::FullAttention {
                    if hss_engaged {
                        // HSS path: `decode_multi_seq` calls the production
                        // paged-decode kernel which reads K/V from HBM only
                        // (`meta.block_table`). Under HSS, HBM is capped to
                        // `cache_blocks_per_seq` blocks, so older context
                        // lives only on disk and is unreachable from the
                        // multi-Q kernel — Q/V attends only over the recent
                        // ~cap×bs tokens, missing the long-context history.
                        // The single-token `decode` path routes through the
                        // HSS orchestrator (`attend_layer_on_stream`) which
                        // reads the full history from disk. Fall back to
                        // `decode_batched` (N sequential single-token
                        // decodes via the orchestrator) at the cost of
                        // ~k× attention launches per verify step. Mirrors
                        // the SSM branch below which already uses
                        // decode_batched for the same correctness reason.
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
                        // Attention: treat 2 tokens as 2 virtual sequences via
                        // decode_multi_seq. EmptyLayerState has no actual state.
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
                    // SSM: process K=2 tokens for one sequence via decode_batched.
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
                if let Some(t0) = __lt0 {
                    self.gpu.synchronize(stream)?;
                    let dt = t0.elapsed().as_nanos() as u64;
                    if layer_type == LayerType::FullAttention {
                        attn_ns += dt;
                    } else {
                        ssm_ns += dt;
                    }
                }
                // DFlash hidden capture for ctx conditioning. Save ALL k
                // tokens so the scheduler can pick the correct one
                // (num_accepted) after verify. Layout:
                // [token_idx, capture_layer, hidden] in dflash_hidden_save.
                for t in 0..k {
                    self.try_dflash_capture(layer_idx, t, stream)?;
                }
            }

            if probe && seq.seq_len.is_multiple_of(50) {
                tracing::info!(
                    "VERIFY_PHASE attn_layers_us={} ssm+moe_layers_us={} seq_len={}",
                    attn_ns / 1000,
                    ssm_ns / 1000,
                    seq.seq_len
                );
            }

            // Final norm [2, H]. Qwen4 collapses each four-stream row through
            // its learned terminal mixer. Preserve row zero in attn_output
            // while row one temporarily occupies norm_output.
            let normed = self.buffers.norm_output();
            if self.config.is_qwen4_exp() {
                let saved_row0 = self.buffers.attn_output();
                for t in 0..k {
                    let hidden_t = hidden.offset(t * persistent_row_bytes);
                    let residual_t = residual.offset(t * persistent_row_bytes);
                    self.qwen4_final_hidden(hidden_t, residual_t, stream)?
                        .ok_or_else(|| anyhow::anyhow!("qwen4_exp verify missing final mixer"))?;
                    if t == 0 {
                        self.gpu
                            .copy_d2d_async(normed, saved_row0, h * bf16, stream)?;
                    } else {
                        self.gpu.copy_d2d_async(
                            normed,
                            normed.offset(t * h * bf16),
                            h * bf16,
                            stream,
                        )?;
                    }
                }
                self.gpu
                    .copy_d2d_async(saved_row0, normed, h * bf16, stream)?;
            } else {
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
            }

            // LM head for 2 tokens (GEMM: weights loaded once)
            self.lm_head_batched(normed, k as u32, stream)?;

            // Argmax inside graph (fixed scratch addresses — graph-safe).
            // Use the verify-side (possibly truncated) vocab so the per-row
            // logits stride AND the argmax range match exactly what
            // `lm_head_batched` wrote — `verify_lmhead_vocab()` returns the
            // full vocab when `ATLAS_TARGET_LMHEAD_VOCAB` truncation is off.
            let vocab = self.verify_lmhead_vocab() as usize;
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
                    tracing::info!("Captured CUDA graph for K=2 verify (slot={})", seq.slot_idx);
                    if let Some(ref mut cache) = graph_cache {
                        cache.insert(seq.slot_idx, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }
        }

        // ── Phase 3: Post-graph (D2H copy only) ──

        // ATLAS_DUMP_HIDDEN: flush captured layer hiddens to file.
        // Cheap no-op when env var unset. Graph-safe because
        // ATLAS_DUMP_HIDDEN forces eager mode upstream (suppress_graphs
        // never lifts; see line 162 above).
        self.flush_hidden_dump(k)?;

        let out_ptr = self.buffers.scratch();
        let mut buf = [0u8; 8];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let tok0 = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let tok1 = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);

        // EXPERIMENTAL: push ALL tokens (including tokens[0]) and advance
        // seq_len by K. Prior logic (`seq_len += k-1`, push only tokens[1..])
        // assumed tokens[0] was ALREADY in seq.tokens from a prior decode,
        // but that precondition is VIOLATED in the MTP flow: scheduler's
        // step_verify_k2 calls decode_verify_graphed([a.last_token, draft])
        // where a.last_token = sampled-but-not-pushed token from prior
        // bootstrap. Off-by-one accumulates across iterations and likely
        // underlies 80B-nvfp4-mtp fib drift (positions misaligned → wrong
        // RoPE → different logits → argmax flip on edge-case tokens).
        for &t in tokens {
            seq.tokens.push(t);
        }
        seq.seq_len += k;

        Ok([tok0, tok1])
    }
}
