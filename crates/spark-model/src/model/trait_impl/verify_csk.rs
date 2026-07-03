// SPDX-License-Identifier: AGPL-3.0-only

//! K=3 verify, c-batched per-step (Path B).
//!
//! Replaces c independent single-sequence K=3 verify forwards (each at
//! M=3 = c × 3 = 12 forwards per layer per step at c=4) with K=3
//! SEQUENTIAL c-batched single-step decodes (each at M=c = 4 = 3 batched
//! forwards per layer per step at c=4, K=3). Trades 12 → 3 FFN/SSM/attn
//! forward calls per layer per step, keeping the same M=c×K total work
//! but folding c-dimensional weight reuse onto the LPDDR5X-bandwidth-
//! bound GEMMs (which dominate the per-step profile per `forward_kgamma`
//! traces).
//!
//! ## Why per-step instead of a single M=c*K forward
//!
//! A true M=c*K verify (Path A) requires a new multi-seq multi-token
//! forward that cross-cuts: the graph cache (currently single-seq slot-
//! keyed), the SSM state pool (per-seq h_state pointers, but per-step
//! intermediates needed for partial rollback), KV slot computation
//! (per-seq block tables), the MTP proposer state, EP broadcast format,
//! grammar matcher state, and `save_hidden_for_mtp` slot indices.
//! Prior agent (batch-k3-cseq task #115) declined this rewrite as
//! exceeding the stay-in-scope constraint.
//!
//! Path B uses ONLY existing primitives:
//! - `decode_batch_dispatch` for the c-batched single-step decode (it
//!   already calls `decode_multi_seq` per layer which uses the multi-seq
//!   SSM kernels when `ATLAS_SSM_MULTI_SEQ_BATCHED=1` +
//!   `ATLAS_SSM_MULTI_SEQ_KERNEL=1`).
//! - Manual D2D copy from `ssm.h_state` → `ssm_pool.h_intermediate(layer,
//!   slot, k)` after each step k, so the existing
//!   `commit_verify_state_async(seq, num_accepted, K)` partial-rollback
//!   path (which reads `h_intermediate(layer, slot, num_accepted - 1)`)
//!   works unchanged.
//!
//! ## Per-step intermediate snapshot cost
//!
//! After each of K=3 steps, we copy `h_state` + `conv_state` to a pool
//! slot for every (seq, ssm_layer). For Qwen3.6-27B with 48 SSM layers
//! and c=4 seqs, that's 4 × 48 × 2 = 384 D2D copies per K-step, each
//! ≤ ~64 KB (h_state ~= nv*vd*kd*4 = 32*128*128*4 = 2 MB; conv_state
//! similar). Total bytes/step ≈ 4 × 48 × 4 MB = 768 MB per step at
//! ~273 GB/s LPDDR5X = ~2.8 ms/step. Across K=3 = ~8 ms verify overhead.
//! Compared to FFN savings (17.1 s / 21840 calls × (12 - 3) × c × steps
//! ≈ multi-second savings), the snapshot cost is dwarfed.
//!
//! Future optimization: launch a single multi-layer-multi-seq h_state
//! snapshot kernel (one launch per step instead of 384) for ~10× lower
//! launch overhead.
//!
//! ## Status (measured 2026-05-23): scaffolding shipped, REGRESSION
//!
//! Performance numbers (AEON-Q36-27B-XS, 100-tok prompts, warm state):
//! - Baseline (per-seq graphed K=3): c=1 31.8 / c=2 24.8 / c=4 23.8 / c=8 21.4 tok/s
//! - CSK (this path):                c=1 14.2 / c=2 15.9 / c=4 17.3 / c=8 19.0 tok/s
//!
//! **Output coherence:** confirmed correct on Count/Rome/Fruits/DNS
//! prompts at c=2/4/8.
//!
//! **Blocker:** `decode_batch_dispatch` for n ≥ 2 hard-disables CUDA
//! graphs (`decode_a2.rs:99` — `let use_graphs = false`) due to SSM
//! state pointer staleness (per-seq h_state / conv_state pointers are
//! baked into per-seq kernel args at capture time; batch composition
//! changes corrupt SSM state on replay). Each ungrapehd c-batched
//! step costs 600-1300 ms at c=2-3; 3 of those per verify dwarfs the
//! single-seq graphed K=3 baseline (~85 ms/verify).
//!
//! **Handoff for next session:** the CSK plumbing is structurally
//! sound (correct argmax mapping, correct per-step intermediate snapshot
//! to match `commit_verify_state_async` semantics, correct per-seq
//! commit/emit/propose pipeline). What's missing is CUDA graph
//! capture for multi-seq decode. Two paths:
//!
//! **Path A — Indirection table (recommended).** Marshal per-seq SSM state
//! pointers into a small device-resident `[u64; max_batch_size]` array
//! (`ssm_pool.ptr_scratch` analog) that `gdn_decode_multi_seq` +
//! `conv1d_update_multi_seq` read at launch time instead of having pointers
//! baked into kernel args. Then the captured graph contains only the table
//! pointer (fixed), and per-step the host code H2Ds the actual per-seq
//! pointers. This works because the multi-seq kernels already accept
//! `h_state_ptrs` / `conv_state_ptrs` as kernel args (see
//! qwen3_ssm/trait_decode_multi_seq.rs:421-451) — they just don't survive
//! graph replay today.
//!
//! **Path B — Per-slot-tuple graph cache.** Key the multi-seq decode graph
//! cache on `(slot_tuple, padded_n)` and re-capture when the active seq's
//! slot set changes. Simpler but causes a graph-capture stall every time
//! the batch composition changes (sequence finishes / new arrival), which
//! may obviate the gain in a busy server.
//!
//! Once graphs land for c-batched decode, CSK's 12 → 3 FFN/SSM
//! forward-call reduction should hit the ~4× target.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::{Result, bail};
use atlas_core::config::LayerType;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::types::TransformerModel;
use crate::layer::{LayerState, SsmLayerState};
use crate::layers::ops;
use crate::traits::{Model, SequenceState};

impl TransformerModel {
    /// K=3 verify across c sequences via sequential c-batched K-loop.
    ///
    /// See module doc + trait method doc on `Model::decode_verify_k3_batched_csk`
    /// for the full contract and side-effect semantics.
    pub(super) fn decode_verify_k3_batched_csk_dispatch(
        &self,
        tokens_per_seq: &[[u32; 3]],
        seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<[u32; 3]>> {
        let c = seqs.len();
        let k_steps = 3usize;
        if tokens_per_seq.len() != c {
            bail!(
                "decode_verify_k3_batched_csk: tokens_per_seq.len()={} != seqs.len()={}",
                tokens_per_seq.len(),
                c
            );
        }
        if c < 2 {
            // Fall back to single-seq K=3 graphed verify (no batching to do).
            let mut out = Vec::with_capacity(c);
            for (i, seq) in seqs.iter_mut().enumerate() {
                let r = self.decode_verify_graphed_k3_dispatch(&tokens_per_seq[i], seq, 0)?;
                out.push(r);
            }
            return Ok(out);
        }

        // F62 (2026-04-27) SpecMamba dual-buffer pre-verify copy: copy
        // checkpoint → live h_state so the verify kernels can scratch-
        // write the canonical state. Must run BEFORE the first
        // intermediate snapshot.
        for seq in seqs.iter_mut() {
            self.pre_verify_copy_async(seq)?;
        }

        let stream = self.gpu.default_stream();
        let bf16 = 2usize;
        let vocab = self.config.vocab_size;

        // Out-of-band record of original seq_len so the scheduler can
        // truncate per-seq to `original_seq_len + accept_count[i] + 1`
        // after K steps. We don't return it because the scheduler holds
        // it directly; the contract on the trait method is that the
        // caller knows `original_seq_len[i] = seq.seq_len - K` after this
        // returns (since every seq was advanced exactly K positions).

        let mut results: Vec<[u32; 3]> = vec![[0u32; 3]; c];

        // ── Per-step c-batched K-loop ──
        //
        // Fix C (perf): write each step's argmax to a DISTINCT region of
        // scratch (`argmax_base + (k*c + i) * 4`) and skip the per-step
        // sync + d2h. The argmax `results[i][k]` are only consumed AFTER
        // the K-loop completes, so the per-step host sync was a pure
        // GPU pipeline stall (~80 ms per K=3 step ≈ 240 ms wasted per
        // c-batched verify). All argmax bytes are read back in ONE d2h
        // after the loop with a single sync.
        let argmax_base = self.buffers.scratch();
        for k in 0..k_steps {
            // Gather the k-th token from each seq.
            let step_tokens: Vec<u32> = (0..c).map(|i| tokens_per_seq[i][k]).collect();

            // c-batched single-step decode. This advances each
            // seq.seq_len by 1 and pushes the token. Uses
            // `decode_multi_seq` per layer (batched projections + per-seq
            // SSM state advance, possibly via multi-seq kernels when
            // ATLAS_SSM_MULTI_SEQ_BATCHED=1 + ATLAS_SSM_MULTI_SEQ_KERNEL=1).
            //
            // Returns the BF16 logits buffer pointing at
            // self.buffers.logits()[0..c, vocab]. We need argmax per row.
            let logits_ptr = self.decode_batch_dispatch(&step_tokens, seqs, stream)?;

            // Per-step argmax writes go to `argmax_base + (k*c + i) * 4`
            // — non-overlapping region per step. NO sync, NO d2h here.
            for i in 0..c {
                let logits_i = logits_ptr.offset(i * vocab * bf16);
                let out_i = argmax_base.offset((k * c + i) * 4);
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    logits_i,
                    out_i,
                    vocab as u32,
                    stream,
                )?;
            }

            // ── Snapshot per-seq SSM state into intermediate pool slot k ──
            // Required so that the post-verify `commit_verify_state_async(
            // seq, num_accepted, K)` partial-rollback can find
            // `h_intermediate(layer, slot, num_accepted - 1)` for the
            // post-step-(num_accepted - 1) state.
            //
            // Layout per single-seq decode_batched:
            //   `intermediates[t]` = state AFTER processing token t.
            // Our step `k` produces state-after-step-`k`. Save into
            // `intermediate[k]` so num_accepted=k+1 → reads intermediate[k]
            // (the post-(k+1-th)-token state) — wrong! num_accepted=1
            // should read intermediate[0] = post-token-0 state. With our
            // mapping, num_accepted=k+1 → reads intermediate[(k+1)-1] = k
            // = post-step-k state, which IS post-(k+1-th)-token state.
            // So num_accepted=1 → intermediate[0] = post-step-0 = post-
            // last_token state. ✓
            //
            // num_accepted=K (full accept) does NOT read intermediates —
            // it uses ssm.h_state (post-step-(K-1) = post-K-tokens) and
            // copies it to checkpoint via the "full accept" fast path.
            // So intermediate[K-1] is never read on full accept; skip the
            // snapshot for the last step k=K-1 as a small optimization.
            if k + 1 < k_steps {
                self.snapshot_ssm_intermediates_per_seq(seqs, k, stream)?;
            }
        }

        // Fix C: ONE sync + ONE d2h after the K-loop completes.
        // GPU pipelined all K argmax kernels with the decode forwards
        // and SSM snapshots — host wakes only when everything queued
        // since the last sync is done. argmax_buf layout: [step 0:
        // c*4 bytes, step 1: c*4, step 2: c*4] = K*c*4 = 3*c*4 bytes.
        self.gpu.synchronize(stream)?;
        let total_argmax_bytes = k_steps * c * 4;
        let mut argmax_buf = vec![0u8; total_argmax_bytes];
        self.gpu.copy_d2h(argmax_base, &mut argmax_buf)?;
        for k in 0..k_steps {
            for i in 0..c {
                let off = (k * c + i) * 4;
                let v = u32::from_le_bytes([
                    argmax_buf[off],
                    argmax_buf[off + 1],
                    argmax_buf[off + 2],
                    argmax_buf[off + 3],
                ]);
                results[i][k] = v;
            }
        }

        Ok(results)
    }

    /// K=2 verify across c sequences via sequential c-batched K-loop.
    /// K=2 sibling of `decode_verify_k3_batched_csk_dispatch`.
    ///
    /// See `Model::decode_verify_k2_batched_csk` trait doc for the contract.
    pub(super) fn decode_verify_k2_batched_csk_dispatch(
        &self,
        tokens_per_seq: &[[u32; 2]],
        seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<[u32; 2]>> {
        let c = seqs.len();
        let k_steps = 2usize;
        if tokens_per_seq.len() != c {
            bail!(
                "decode_verify_k2_batched_csk: tokens_per_seq.len()={} != seqs.len()={}",
                tokens_per_seq.len(),
                c
            );
        }
        if c < 2 {
            // Fall back to single-seq K=2 graphed verify (no batching to do).
            let mut out = Vec::with_capacity(c);
            for (i, seq) in seqs.iter_mut().enumerate() {
                let r = self.decode_verify_graphed_dispatch(&tokens_per_seq[i], seq, 0)?;
                out.push(r);
            }
            return Ok(out);
        }

        // F62 (2026-04-27) SpecMamba dual-buffer pre-verify copy: copy
        // checkpoint → live h_state so the verify kernels can scratch-
        // write the canonical state. Must run BEFORE the first
        // intermediate snapshot.
        for seq in seqs.iter_mut() {
            self.pre_verify_copy_async(seq)?;
        }

        let stream = self.gpu.default_stream();
        let bf16 = 2usize;
        let vocab = self.config.vocab_size;

        let mut results: Vec<[u32; 2]> = vec![[0u32; 2]; c];

        // ── Per-step c-batched K-loop ──
        for k in 0..k_steps {
            let step_tokens: Vec<u32> = (0..c).map(|i| tokens_per_seq[i][k]).collect();

            // c-batched single-step decode (see K=3 sibling for details).
            let logits_ptr = self.decode_batch_dispatch(&step_tokens, seqs, stream)?;

            // Argmax per seq.
            let argmax_out = self.buffers.scratch();
            for i in 0..c {
                let logits_i = logits_ptr.offset(i * vocab * bf16);
                let out_i = argmax_out.offset(i * 4);
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    logits_i,
                    out_i,
                    vocab as u32,
                    stream,
                )?;
            }
            self.gpu.synchronize(stream)?;
            let mut argmax_buf = vec![0u8; c * 4];
            self.gpu.copy_d2h(argmax_out, &mut argmax_buf)?;
            for i in 0..c {
                let off = i * 4;
                let v = u32::from_le_bytes([
                    argmax_buf[off],
                    argmax_buf[off + 1],
                    argmax_buf[off + 2],
                    argmax_buf[off + 3],
                ]);
                results[i][k] = v;
            }

            // Snapshot per-seq SSM state into intermediate pool slot k for
            // partial-rollback semantics (matches K=3 logic). Skip on last
            // step k=K-1 because full-accept (num_accepted=K) reads
            // ssm.h_state directly.
            if k + 1 < k_steps {
                self.snapshot_ssm_intermediates_per_seq(seqs, k, stream)?;
            }
        }

        Ok(results)
    }

    /// D2D copy each seq's live `ssm.h_state` and `ssm.conv_state` into
    /// the corresponding `ssm_pool.h_intermediate(layer, slot, k)` /
    /// `conv_intermediate(layer, slot, k)`. One launch per (seq, layer)
    /// × 2 (h + conv) = 2 × c × num_ssm_layers launches per step.
    ///
    /// Future opt: collapse into a single multi-seq-multi-layer kernel.
    fn snapshot_ssm_intermediates_per_seq(
        &self,
        seqs: &mut [&mut SequenceState],
        inter_slot: usize,
        stream: u64,
    ) -> Result<()> {
        let nv = self.config.linear_num_value_heads;
        let vd = self.config.linear_value_head_dim;
        let kd = self.config.linear_key_head_dim;
        let nk = self.config.linear_num_key_heads;
        let h_bytes = nv * vd * kd * 4; // FP32
        let conv_dim = nk * kd * 2 + nv * vd;
        let d_conv = self.config.linear_conv_kernel_dim;
        let conv_bytes = conv_dim * d_conv * 4; // FP32

        for seq in seqs.iter_mut() {
            let slot = seq.slot_idx;
            let mut ssm_layer_idx = 0usize;
            for (layer_i, layer_state) in seq.layer_states.iter_mut().enumerate() {
                if self.config.layer_type(layer_i) != LayerType::LinearAttention {
                    continue;
                }
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {layer_i}"))?;

                let h_inter = self
                    .ssm_pool
                    .h_intermediate(ssm_layer_idx, slot, inter_slot);
                let conv_inter = self
                    .ssm_pool
                    .conv_intermediate(ssm_layer_idx, slot, inter_slot);
                self.gpu
                    .copy_d2d_async(ssm.h_state, h_inter, h_bytes, stream)?;
                self.gpu
                    .copy_d2d_async(ssm.conv_state, conv_inter, conv_bytes, stream)?;

                ssm_layer_idx += 1;
            }
        }
        Ok(())
    }
}
