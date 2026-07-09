// SPDX-License-Identifier: AGPL-3.0-only

//! CROSS-SEQ BATCHED DFLASH VERIFY (#39).
//!
//! Verifies `c` sequences' K=γ+1 draft windows in ONE forward whose
//! WEIGHT-heavy FFN GEMMs read the ~14 GB of NVFP4 FFN weights ONCE for all
//! `c×K` rows, instead of the per-sequence loop (each running its own K=γ
//! verify = its own full weight sweep). Projected: c=8 aggregate 208 → 400+.
//!
//! ## Structure (v1, eager)
//!
//! Row-major layout `hidden[c*K, H]`: sequence `s` owns rows
//! `[s*K, (s+1)*K)`. Per layer:
//!   1. Run every sequence's MIXER (SSM/GDN K-window recurrence, or attention
//!      over its K rows) with `ctx.ffn_defer = Some{dst, row_offset=s*K}` so
//!      the layer writes its post-mixer FFN input into the shared collection
//!      buffer and SKIPS its own FFN.
//!   2. Run the layer's FFN ONCE over all `c*K` rows (`run_deferred_ffn` →
//!      `forward_kgamma` at M=c*K → the WIDE `w4a16_gemm_t_m128` window when
//!      32 < M ≤ 256) and residual-add into `hidden`.
//!
//! Attention and SSM stay PER-SEQUENCE (their weights are tiny relative to the
//! FFN, and both carry per-seq state — SSM recurrence, attention KV — so they
//! cannot be trivially fanned across sequences anyway). The 14 GB lives in the
//! FFN, and THAT is amortized. Per-seq attention also sidesteps the fixed
//! 32-row attention-metadata ABI (positions/slots/seq_lens packed at 256-byte
//! offsets), which cannot address c*K > 32 rows.
//!
//! ## Losslessness
//!
//! The per-row argmax this produces is bit-identical to the single-seq
//! `decode_verify` for each sequence (same kernels, same math, same state) —
//! only the FFN GEMM batches rows across sequences (a pure M-growth of a
//! deterministic GEMM). The scheduler's per-seq accept/commit/propose loops
//! run UNCHANGED, so per-sequence output is identical to c=1.
//!
//! ## v2 additions
//!   * lm_head batched: `ceil(c*K/32)` chunked passes over the contiguous
//!     `[c*K, H]` normed buffer (the logits buffer caps at 32 rows) instead of
//!     the v1 `c` per-seq passes — fewer lm_head weight-read sweeps.
//!
//! ## Not batched in v1 (see the FFN win note above)
//!   * No CUDA graph capture (eager; the weight amortization dwarfs launch
//!     overhead at these M).
//!   * No tree/DDTree payloads (flat chain only); the scheduler routes
//!     tree-payload sequences to the per-seq path.
//!   * Drafter per-layer hidden capture (`try_dflash_capture`) is skipped;
//!     the scheduler's per-seq `save_hidden_for_dflash` still runs, so output
//!     stays lossless (verify is the oracle) — only draft-conditioning
//!     quality may differ marginally.

#![allow(clippy::too_many_arguments)]

use anyhow::{Result, bail};
use atlas_core::config::LayerType;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::types::TransformerModel;
use crate::layer::{AttnMetadataDev, FfnDefer, ForwardContext, LayerState};
use crate::layers::ops;
use crate::traits::{Model, SequenceState};

impl TransformerModel {
    /// Batched DFlash verify across `c` sequences. `tokens_per_seq[s]` is
    /// sequence `s`'s `[last_token, draft0, .., draft_{γ-1}]` (all length `K`).
    /// Every sequence MUST have the same `K` (aligned by the scheduler).
    ///
    /// Returns `Vec<Vec<u32>>`: element `s` is sequence `s`'s per-position
    /// argmax (length `K`), identical layout to `decode_verify`'s return.
    ///
    /// **Side effects per seq (identical to `decode_verify`):** `seq_len += K`,
    /// all `K` tokens pushed to `seq.tokens`. Caller rolls back per-seq and
    /// calls `commit_verify_state_async` per-seq exactly as today.
    pub(super) fn decode_verify_dflash_batched_dispatch(
        &self,
        tokens_per_seq: &[Vec<u32>],
        seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<Vec<u32>>> {
        let c = seqs.len();
        if c == 0 {
            return Ok(Vec::new());
        }
        if tokens_per_seq.len() != c {
            bail!(
                "decode_verify_dflash_batched: tokens_per_seq.len()={} != seqs.len()={}",
                tokens_per_seq.len(),
                c
            );
        }
        let k = tokens_per_seq[0].len();
        if k == 0 {
            return Ok(vec![Vec::new(); c]);
        }
        for (s, t) in tokens_per_seq.iter().enumerate() {
            if t.len() != k {
                bail!(
                    "decode_verify_dflash_batched: seq {s} has K={} != K={} (unaligned windows)",
                    t.len(),
                    k
                );
            }
        }
        // c==1 has no batching to do — defer to the proven single-seq path.
        if c == 1 {
            let out = self.decode_verify_dispatch(&tokens_per_seq[0], seqs[0], 0)?;
            return Ok(vec![out]);
        }

        let total_rows = c * k;
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };

        // F62 SpecMamba dual-buffer pre-verify copy per seq (checkpoint →
        // live h_state) so the SSM verify kernels can scratch-write canonical
        // state and the post-verify commit can roll back.
        for seq in seqs.iter_mut() {
            self.pre_verify_copy_async(seq)?;
        }

        // The shared per-layer FFN-input collection buffer [c*K, H] BF16.
        // Lazily allocated (sized for the max prefill batch → always ≥ c*K).
        let ffn_input = self.dflash_batched_ffn_input(total_rows)?;

        let hidden = self.buffers.hidden_states(); // [c*K, H]
        let residual = self.buffers.residual(); // [c*K, H]

        // ── Embed every seq's K tokens into its row block ──
        for (s, toks) in tokens_per_seq.iter().enumerate() {
            for (t, &token) in toks.iter().enumerate() {
                let row = s * k + t;
                self.embed(token, hidden.offset(row * h * fp32), stream)?;
            }
        }

        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let max_blocks = self.max_blocks_per_seq;

        // Ensure KV blocks for every seq's K positions up front.
        for seq in seqs.iter_mut() {
            let last_pos = seq.seq_len + k - 1;
            let blocks_needed = (last_pos / bs) + 1;
            ensure_blocks(
                self,
                seq,
                blocks_needed.saturating_sub(1),
                &mut kv_cache,
                stream,
            )?;
        }

        // ── Per-layer forward: mixers (deferred FFN) then ONE batched FFN ──
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_type = self.config.layer_type(layer_idx);

            for (s, seq) in seqs.iter_mut().enumerate() {
                let row_base = s * k;
                let defer = FfnDefer {
                    dst_base: ffn_input,
                    row_offset: row_base,
                };
                let h_s = hidden.offset(row_base * h * fp32);
                let r_s = residual.offset(row_base * h * fp32);

                if layer_type == LayerType::FullAttention {
                    // Per-seq attention over this seq's K rows via the multi-seq
                    // path (each of the K rows is a "sequence" to the paged
                    // decode kernel: its own position + KV within THIS seq). The
                    // deferred-FFN branch writes normed FFN input into `ffn_input`
                    // at `row_base` and returns before the FFN.
                    let meta = self.upload_kwindow_attn_metadata(
                        seq,
                        k,
                        &mut kv_cache,
                        max_blocks,
                        stream,
                    )?;
                    let ctx = self.batched_verify_ctx(Some(meta), Some(defer));
                    let seq_lens_vec: Vec<usize> =
                        (0..k).map(|t| seq.seq_len + t).collect();
                    let block_tables_vec: Vec<Vec<u32>> =
                        vec![seq.block_table.clone(); k];
                    let mut dummy_states: Vec<Box<dyn LayerState>> = (0..k)
                        .map(|_| layer.alloc_state(self.gpu.as_ref()))
                        .collect::<Result<_>>()?;
                    let mut refs: Vec<&mut (dyn LayerState + 'static)> =
                        dummy_states.iter_mut().map(|s| s.as_mut()).collect();
                    layer.decode_multi_seq(
                        h_s,
                        r_s,
                        k,
                        &mut refs,
                        &mut kv_cache,
                        &seq_lens_vec,
                        &block_tables_vec,
                        &ctx,
                        stream,
                    )?;
                } else {
                    // SSM/GDN K-window recurrence for this seq, deferred FFN.
                    let ctx = self.batched_verify_ctx(None, Some(defer));
                    layer.decode_batched(
                        h_s,
                        r_s,
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
            }

            // ── ONE batched FFN over all c*K rows: reads FFN weights ONCE ──
            let ffn_ctx = self.batched_verify_ctx(None, None);
            layer.run_deferred_ffn(ffn_input, hidden, total_rows, &ffn_ctx, stream)?;
        }

        // ── Final norm over all c*K rows (pure row-parallel) ──
        let normed = self.buffers.norm_output();
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

        // ── BATCHED lm_head + argmax over ALL c*K rows (v2) ──
        // The shared logits buffer is capped at 32 rows (`logits_tokens =
        // m.min(32)` in buffers/sizes.rs), so we cannot land all `c*K` rows'
        // logits at once. Instead of the v1 per-seq loop (`c` weight-read
        // passes, one per sequence), chunk the CONTIGUOUS `[c*K, H]` normed
        // buffer into `ceil(c*K/32)` runs of ≤32 rows — each run is ONE
        // lm_head weight-read pass (the transposed m32_n64 kernel covers M≤32
        // in a single coalesced sweep). At c=8/K=17 this is
        // ceil(136/32)=5 passes vs 8. Rows may straddle sequence boundaries
        // (row `r` = seq `r/K`, position `r%K`), which is fine: the lm_head
        // projection + per-row argmax are independent per row, so a chunk that
        // mixes two sequences' rows is bit-identical to running each row alone.
        // Byte-identical to the single-seq `decode_verify` argmax: same kernel,
        // same weights, same per-row math — only the M-dimension batches.
        let vocab = self.verify_lmhead_vocab() as usize;
        const LM_CHUNK: usize = 32;
        let mut all_argmax: Vec<u32> = vec![0u32; total_rows];
        let mut chunk_start = 0usize;
        while chunk_start < total_rows {
            let rows = (total_rows - chunk_start).min(LM_CHUNK);
            let normed_chunk = normed.offset(chunk_start * h * bf16);
            self.lm_head_batched(normed_chunk, rows as u32, stream)?;
            let argmax_out = self.buffers.scratch();
            for t in 0..rows {
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
            self.gpu.synchronize(stream)?;
            let mut buf = vec![0u8; rows * 4];
            self.gpu.copy_d2h(argmax_out, &mut buf)?;
            for t in 0..rows {
                all_argmax[chunk_start + t] = u32::from_le_bytes([
                    buf[t * 4],
                    buf[t * 4 + 1],
                    buf[t * 4 + 2],
                    buf[t * 4 + 3],
                ]);
            }
            chunk_start += rows;
        }
        // Slice the flat per-row argmax back into per-seq K-length windows.
        let mut results: Vec<Vec<u32>> = Vec::with_capacity(c);
        for s in 0..c {
            results.push(all_argmax[s * k..(s + 1) * k].to_vec());
        }

        // ── Advance per-seq state exactly like decode_verify (push K, +=K) ──
        for (seq, toks) in seqs.iter_mut().zip(tokens_per_seq.iter()) {
            for &token in toks {
                seq.tokens.push(token);
            }
            seq.seq_len += k;
        }

        Ok(results)
    }

    /// Lazily allocate (and cache) the `[c*K, H]` BF16 FFN-input collection
    /// buffer. Sized to `max_batch_tokens × H × bf16` on first use so it holds
    /// any `total_rows` the batched verify can present (c×K ≤ concurrency×γ+1).
    fn dflash_batched_ffn_input(&self, total_rows: usize) -> Result<DevicePtr> {
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let mut guard = self.dflash_batched_ffn_input.lock();
        if let Some(p) = *guard {
            return Ok(p);
        }
        let cap_rows = self.buffers.max_batch_tokens().max(total_rows);
        let ptr = self.gpu.alloc(cap_rows * h * bf16)?;
        *guard = Some(ptr);
        Ok(ptr)
    }

    /// Build a `ForwardContext` for the batched verify with optional attention
    /// metadata and optional FFN-defer directive. All other fields default.
    fn batched_verify_ctx(
        &self,
        attn_metadata: Option<AttnMetadataDev>,
        ffn_defer: Option<FfnDefer>,
    ) -> ForwardContext<'_> {
        ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata,
            profile: false,
            comm: self.comm_ref(),
            graph_capture: false,
            ddtree_parent_ids_dev: None,
            tree_aware_attn: None,
            ssm_multi_seq_ptr_table_override: None,
            self_spec_sparse_draft: None,
            ffn_defer,
        }
    }

    /// Upload flat-chain K-window attention metadata for ONE sequence into the
    /// verify metadata scratch (positions/slots/seq_lens/block_table at the
    /// fixed 256-byte-spaced offsets). Row `t` = position `seq.seq_len + t`,
    /// its KV slot, seq_len `seq.seq_len + t + 1`, block_table = seq's.
    /// Identical to the single-seq `decode_verify_graphed_kgamma` flat path.
    fn upload_kwindow_attn_metadata(
        &self,
        seq: &SequenceState,
        k: usize,
        kv_cache: &mut spark_runtime::kv_cache::PagedKvCache,
        max_blocks: u32,
        stream: u64,
    ) -> Result<AttnMetadataDev> {
        let bs = kv_cache.block_size();
        let meta_base = self.buffers.scratch().offset(32768);

        let positions: Vec<u32> = (0..k).map(|t| (seq.seq_len + t) as u32).collect();
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, k * 4) };
        self.gpu.copy_h2d_async(pos_bytes, meta_base, stream)?;

        let mut slots = vec![0i64; k];
        for (t, slot) in slots.iter_mut().enumerate() {
            let pos = seq.seq_len + t;
            let block_idx = pos / bs;
            let block_offset = pos % bs;
            let physical_block = seq
                .physical_block_for(block_idx)
                .unwrap_or(self.dummy_kv_block);
            *slot = (physical_block as i64) * (bs as i64) + (block_offset as i64);
        }
        let slot_bytes = unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, k * 8) };
        self.gpu
            .copy_h2d_async(slot_bytes, meta_base.offset(256), stream)?;

        let seq_lens: Vec<i32> = (0..k).map(|t| (seq.seq_len + t + 1) as i32).collect();
        let sl_bytes = unsafe { std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, k * 4) };
        self.gpu
            .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

        let mb = max_blocks as usize;
        let mut bt_buf = vec![0i32; k * mb];
        for row in 0..k {
            for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                bt_buf[row * mb + j] = block as i32;
            }
        }
        let bt_bytes =
            unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, k * mb * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(768), stream)?;

        Ok(AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: max_blocks,
            num_seqs: k as u32,
        })
    }
}

/// Thin wrapper over `ensure_blocks_through_decode` for the batched verify.
fn ensure_blocks(
    model: &TransformerModel,
    seq: &mut SequenceState,
    extra_blocks: usize,
    kv_cache: &mut spark_runtime::kv_cache::PagedKvCache,
    stream: u64,
) -> Result<()> {
    super::super::block_mgmt::ensure_blocks_through_decode(
        seq,
        extra_blocks,
        kv_cache,
        model.prefix_cache.as_ref(),
        model.gpu.as_ref(),
        stream,
    )
}
