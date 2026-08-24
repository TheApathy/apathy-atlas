// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::dspark_capture::DsparkCaptureLayout;
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    /// K=2 ACCEPT-hole fix: feed the drafter the pair it never saw.
    ///
    /// On a K2 accept the sequence advances by 2 (accepted draft + bonus) but
    /// the drafter appended only 1 row (at the pre-verify propose) — the pair
    /// `(embed(accepted_draft), hidden_row0)` is skipped, punching a permanent
    /// hole in the drafter's context at ~accept-rate density. `hidden_row0`
    /// (the target's final hidden after the token that PRECEDED the draft) is
    /// still sitting in `buffers.hidden_states()` row 0 from the just-run
    /// verify, so feed it before the next propose clobbers the buffer. RoPE
    /// position = post-commit `seq.seq_len - 1` (the accepted draft's own
    /// position). Mirrors `dflash_eagle_accept_append` for the MTP proposer.
    pub(super) fn mtp_accept_feed_inner(
        &self,
        accepted_token: u32,
        hidden_row: usize,
        seq: &mut SequenceState,
    ) -> Result<()> {
        let proposer = match &self.proposer {
            Some(p) => p.as_ref(),
            None => return Ok(()),
        };
        let stream = self.gpu.default_stream();
        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
        };
        let Some(prop_state) = seq.proposer_state.as_mut() else {
            return Ok(());
        };
        let row_base = proposer.drafter_rows(prop_state.as_mut());
        if row_base == 0 {
            // Proposer without drafter-row support (or fresh state): no-op.
            return Ok(());
        }
        // feed_rows reads tokens[r+1] (r = 0): first element is a placeholder.
        let toks = [0u32, accepted_token];
        let pos = seq.seq_len.saturating_sub(1);
        let h = self.config.hidden_size;
        proposer.catchup_drafter(
            &toks,
            self.buffers.hidden_states().offset(hidden_row * h * 2),
            row_base,
            pos,
            prop_state.as_mut(),
            &ctx,
            stream,
        )?;
        Ok(())
    }

    pub(super) fn run_mtp_propose_inner(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        let proposer = match &self.proposer {
            Some(p) => p.as_ref(),
            None => return Ok(Vec::new()),
        };
        // ATLAS_DFLASH_DEBUG_DUMP_FULL=1: emit the full token sequence
        // ONCE so a Python reference can run the SAME tokens through HF
        // transformers and dump matching hidden-state captures.
        static TOKENS_DUMPED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !TOKENS_DUMPED.load(std::sync::atomic::Ordering::Relaxed)
            && std::env::var("ATLAS_DFLASH_DEBUG_DUMP_FULL")
                .ok()
                .as_deref()
                == Some("1")
        {
            let tokens_json = serde_json::json!({
                "prompt_len": position - seq.tokens.len() + seq.tokens.len(),
                "position": position,
                "last_token": token,
                "all_tokens": seq.tokens.clone(),
                "generated_tokens": seq.tokens.iter().skip(seq.prompt_len).copied().collect::<Vec<u32>>(),
            });
            if let Err(e) = std::fs::write(
                "/tmp/atlas_tokens.json",
                serde_json::to_string_pretty(&tokens_json).unwrap_or_default(),
            ) {
                tracing::warn!("DFLASH DUMP_FULL: tokens write failed: {e}");
            } else {
                tracing::info!(
                    "DFLASH DUMP_FULL: wrote /tmp/atlas_tokens.json (position={}, all_tokens.len()={}, prompt_len={})",
                    position,
                    seq.tokens.len(),
                    seq.prompt_len,
                );
            }
            TOKENS_DUMPED.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let stream = self.gpu.default_stream();
        let draft_embed_target = None;
        // MTP loads ALL experts on every rank (no EP filtering), so its MoE
        // output is already complete — no all_reduce needed. Passing comm: None
        // prevents MoeLayer::forward() from doubling the output via SUM.
        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            routed_lora_layers: None, // #30: MTP/draft decode never routes prefill.
            midchunk_capture: None,
        };
        // Accept-lift (Phase A): refresh the retrieval/PLD haystack — the
        // host mirror of the committed token sequence — when any lookup
        // source is enabled. Snapshot first to avoid a double seq borrow.
        {
            static SRC_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let on = *SRC_ON.get_or_init(|| {
                let f = |k: &str| std::env::var(k).ok().as_deref() == Some("1");
                f("ATLAS_DFLASH_PLD") || f("ATLAS_DFLASH_RETRIEVAL") || f("ATLAS_DFLASH_SAM")
            });
            if on {
                let toks_snapshot = seq.tokens.clone();
                if let Some(ps) = seq.proposer_state.as_mut()
                    && let Some(ds) = ps
                        .as_any_mut()
                        .downcast_mut::<crate::layers::DflashProposerState>()
                {
                    ds.pld_tokens.clear();
                    ds.pld_tokens.extend_from_slice(&toks_snapshot);
                }
            }
        }

        let prop_state = seq
            .proposer_state
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("No proposer state for sequence"))?;
        // ATLAS_MTP_DRAFTER_PREFILL: on the FIRST propose of a sequence,
        // batch-prefill the drafter's KV over the prompt (fresh-state check
        // and quant support live inside prefill_drafter; it fast-returns 0 on
        // every later call). Requires the capture to cover the full prompt —
        // partial capture (prefix-cache reuse / warm restore) skips cleanly.
        if !self.mtp_prefill_hidden.is_null() {
            let p = seq.prompt_len;
            let captured = self
                .mtp_prefill_capture_len
                .load(std::sync::atomic::Ordering::Relaxed);
            if p >= 2
                && captured >= p
                && seq.tokens.len() >= p
                && let Err(e) = proposer.prefill_drafter(
                    &seq.tokens[..p],
                    self.mtp_prefill_hidden,
                    prop_state.as_mut(),
                    &ctx,
                    stream,
                )
            {
                tracing::warn!("MTP drafter prefill failed (continuing without): {e:#}");
            }
        }
        // ATLAS_MTP_CATCHUP: before proposing, feed pairs the drafter missed
        // during a serial-decode stretch. Coordinates (measured 2026-07-20 on
        // the 27B rig): at propose entry `position == seq.tokens.len()` and
        // the imminent forward_one writes the pair for sequence key
        // `position - 1`; pair key k = (embed(tokens[k+1]), hidden_k), RoPE
        // k+1. The serial-decode ring stores, under label n, the hidden of
        // the step that COMMITTED token n — i.e. hidden_{n-1} — so pair key k
        // reads ring label k+1. Drafter KV slots are compacted (append-only)
        // while RoPE stays sequence-space, so RoPE gaps are already the norm:
        // partial feeds (clipped to ring coverage) are safe, and wrong feeds
        // cannot corrupt output (verify rejects bad drafts).
        if crate::speculative::mtp_catchup_enabled() && !self.mtp_catchup_ring.is_null() {
            let rows = proposer.drafter_rows(prop_state.as_mut());
            let last_key = proposer.last_pair_key(prop_state.as_mut());
            let (start, count) = *self.mtp_catchup_meta.lock();
            if let Some(last) = last_key
                && rows > 0
                && count > 0
            {
                // Missing pair keys: (last .. position-1); the propose itself
                // covers position-1. Clip to ring coverage [start, start+count)
                // in label space (label = key + 1).
                let mut k0 = (last + 1).max(start.saturating_sub(1));
                let k1 = (position.saturating_sub(2)).min((start + count).saturating_sub(2));
                let want = (position.saturating_sub(1)).saturating_sub(last + 1);
                if k0 <= k1 && want > 0 {
                    let ring_rows = super::types::MTP_CATCHUP_RING_ROWS;
                    let h = self.config.hidden_size;
                    let bf16 = 2usize;
                    let fed_from = k0;
                    while k0 <= k1 {
                        // Ring-contiguous segment: labels k0+1 .. until wrap.
                        let slot = (k0 + 1) % ring_rows;
                        let seg_last = k1.min(k0 + (ring_rows - slot) - 1);
                        let n_rows = seg_last - k0 + 1;
                        // Row r feeds pair key k0+r = embed(tokens[k0+r+1]):
                        // the impl reads prompt_tokens[r+1], so pass the
                        // window starting at index k0 (n_rows + 1 tokens).
                        let toks = &seq.tokens[k0..=seg_last + 1];
                        let hid = self.mtp_catchup_ring.offset(slot * h * bf16);
                        let row_base = proposer.drafter_rows(prop_state.as_mut());
                        match proposer.catchup_drafter(
                            toks,
                            hid,
                            row_base,
                            k0 + 1,
                            prop_state.as_mut(),
                            &ctx,
                            stream,
                        ) {
                            Ok(w) if w == n_rows => k0 = seg_last + 1,
                            Ok(w) => {
                                tracing::debug!(
                                    "MTP catch-up: short feed ({w}/{n_rows} rows) — degrading"
                                );
                                break;
                            }
                            Err(e) => {
                                tracing::debug!("MTP catch-up: feed failed ({e:#}) — degrading");
                                break;
                            }
                        }
                    }
                    if k0 > k1 {
                        tracing::debug!(
                            "MTP catch-up: fed pair keys {fed_from}..={k1} \
                             (missed {want}, position {position})"
                        );
                    }
                } else if want > 0 {
                    tracing::debug!(
                        "MTP catch-up: gap of {want} pairs outside ring coverage \
                         (last_key={last} position={position} ring=[{start},+{count}))"
                    );
                }
            }
        }
        let drafts = proposer.propose(
            token,
            self.mtp_hidden_save,
            position,
            num_drafts,
            prop_state.as_mut(),
            &ctx,
            stream,
            draft_embed_target,
            grammar_bitmask,
            self.dflash_hidden_save,
        )?;
        // Confidence clamp (ATLAS_MTP_DRAFT_CONF, staged off by default):
        // when the drafter's chain confidence is below tau, discard the
        // drafts — the next step decodes serially instead of paying a
        // verify that would most likely reject (break-even acceptance at
        // K=1 on the 35B MoE is ~0.66). The drafter KV rows written by
        // this propose MUST be trimmed exactly as a full rejection would
        // (after_verify(0)), or the drafter desyncs from the target.
        let tau = crate::speculative::draft_conf_tau();
        if tau > 0.0
            && !drafts.is_empty()
            && let Some(conf) = proposer.last_confidence()
            && conf < tau
        {
            tracing::debug!(
                "MTP draft skipped: chain confidence {conf:.3} < tau {tau:.3}                  (pos {position}, {} drafts trimmed)",
                drafts.len(),
            );
            proposer.after_verify(0, prop_state.as_mut(), stream)?;
            return Ok(Vec::new());
        }
        Ok(drafts)
    }

    /// Borrow the GPU backend for post-construction wiring (e.g. installing
    /// a DFlash proposer that needs to allocate paged KV caches against the
    /// same GPU the target uses).
    pub fn gpu_backend(&self) -> &dyn GpuBackend {
        self.gpu.as_ref()
    }

    /// Borrow the model config for post-construction wiring (e.g. building the
    /// DeepSeek-V4 MTP proposer, which needs `hidden_size` / `kv_lora_rank` /
    /// `qk_rope_head_dim` to size its private MLA KV cache).
    pub fn config_ref(&self) -> &ModelConfig {
        &self.config
    }

    /// Install a DFlash drafter as the active proposer, replacing whatever
    /// MTP proposer (if any) `TransformerModel::new` built. The target's
    /// hidden-state capture buffer is already allocated when the config's
    /// `dflash_capture_layers` is non-empty (factory.rs populates it before
    /// construction), so this method only swaps the proposer slot.
    ///
    /// Mutually exclusive with `--speculative` MTP at the CLI level
    /// (clap `conflicts_with`); this method does not enforce that — the
    /// caller is expected to have validated the flag combination already.
    pub fn set_dflash_proposer(&mut self, proposer: std::sync::Arc<dyn DraftProposer>) {
        if self.proposer.is_some() {
            tracing::info!("DFlash: replacing existing MTP proposer with BlockDiffusionDraftHead");
        }
        self.proposer = Some(proposer);
    }

    /// DFlash prefill capture: copy `proc_count` tokens × hidden_size BF16
    /// from `self.buffers.hidden_states()` (filled by the just-completed
    /// prefill layer) into the per-sequence DFlash accumulator. Called
    /// inside the prefill layer loop after each layer. No-op when:
    ///   - DFlash is disabled (capture_layers empty)
    ///   - `layer_idx` is not in `dflash_capture_layers`
    ///   - The seq has no `DflashProposerState`
    ///   - Rank > 0 under EP/TP (drafter is rank-0 only)
    ///
    /// Layout: writes `hidden[t]` BF16 into
    /// `acc[(chunk_start + t) * 5 * h + slot_idx * h]` for each t.
    /// Per-layer call performs `proc_count` strided d2d_async copies —
    /// at typical prefill of 128–4096 tokens × 5 capture layers, total
    /// 640–20480 launches per prefill. Acceptable launch overhead for
    /// first land; replace with a strided-scatter kernel if profiling
    /// shows it's a bottleneck.
    pub(super) fn try_dflash_prefill_capture_layer(
        &self,
        seq: &mut crate::traits::SequenceState,
        layer_idx: usize,
        chunk_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        let slot_idx = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let dstate = match seq.proposer_state.as_mut() {
            Some(ps) => match ps
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
            {
                Some(s) => s,
                None => return Ok(()),
            },
            None => return Ok(()),
        };
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let n_capture = self.dflash_capture_layers.len();
        let acc_base = dstate.ctx_hidden_acc;
        let max_ctx = dstate.max_ctx_len;
        let src_base = self.buffers.hidden_states();
        for t in 0..proc_count {
            let abs_pos = chunk_start + t;
            if abs_pos >= max_ctx {
                break; // accumulator full; drop later positions
            }
            let src = src_base.offset(t * h * bf16);
            let dst_offset = abs_pos * n_capture * h * bf16 + slot_idx * h * bf16;
            self.gpu
                .copy_d2d_async(src, acc_base.offset(dst_offset), h * bf16, stream)?;
        }
        Ok(())
    }

    /// After prefill completes, advance the seq's DFlash `ctx_len` to
    /// `chunk_start + proc_count` so the drafter sees all captured prompt
    /// positions on the first propose() call.
    pub(super) fn update_dflash_ctx_len_after_prefill(
        &self,
        seq: &mut crate::traits::SequenceState,
        chunk_start: usize,
        proc_count: usize,
    ) -> Result<()> {
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        if let Some(ps) = seq.proposer_state.as_mut()
            && let Some(dstate) = ps
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
        {
            let new_len = (chunk_start + proc_count).min(dstate.max_ctx_len);
            dstate.ctx_len = new_len;
            // Phase I (v2): seed per-slot fixed positions for the prompt
            // captures. Prefill slot i holds prompt position i, so the
            // fixed rope position is simply its index. Keep parallel to
            // ctx_len. Re-seed idempotently across prefill chunks.
            dstate.ctx_positions = (0..new_len).map(|i| i as i32).collect();
        }
        Ok(())
    }

    /// DFlash 5-layer hidden capture. Called inside each per-layer loop after
    /// `layer.decode(...)` returns. No-op when DFlash is disabled (the buffer
    /// is `None`) or when `layer_idx` is not in `dflash_capture_layers`.
    ///
    /// Captures only the latest-decoded-token's hidden, matching the
    /// `save_hidden_for_mtp` semantics. The `token_idx` argument selects
    /// which row of `self.buffers.hidden_states()` to read — pass 0 for the
    /// single-token decode path.
    ///
    /// Under EP/TP world > 1: only rank 0 owns the drafter (replicated, not
    /// sharded — same pattern as MTP under EP — see model.rs:7232 comment),
    /// so non-rank-0 ranks skip the capture. The captured hiddens are
    /// post-TP-allreduce so semantically correct on rank 0.
    pub(super) fn try_dflash_capture(
        &self,
        layer_idx: usize,
        token_idx: usize,
        stream: u64,
    ) -> Result<()> {
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        // Rank-0 gate (mirrors save_hidden_for_mtp's effective behavior).
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let slot = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        // The residual stream is always BF16, so DFlash hidden capture
        // copies BF16 bytes directly with no downcast.
        let src = self.buffers.hidden_states().offset(token_idx * h * bf16);
        let dst_slot = dst.offset(slot * h * bf16);
        self.gpu.copy_d2d_async(src, dst_slot, h * bf16, stream)?;
        Ok(())
    }

    /// Capture `hidden_states[token_idx]` for every DFlash capture layer into
    /// `dflash_hidden_save`. Called from `verify_dflash_step` after the Phase 3
    /// D2H sync, so `token_idx` is the confirmed bonus position. Runs outside
    /// the CUDA graph so the correct accept-prefix position can be used.
    pub(super) fn save_dflash_hidden_dispatch(&self, token_idx: usize, stream: u64) -> Result<()> {
        for &layer_idx in &self.dflash_capture_layers {
            self.try_dflash_capture(layer_idx, token_idx, stream)?;
        }
        Ok(())
    }

    /// K=gamma EAGLE capture: copy the per-layer hidden of ALL `k` verify rows into
    /// the row-major `dflash_hidden_save` ([row0 | row1 | ... ], each row =
    /// n_capture * hidden_size * bf16). Called once per capture layer inside the
    /// verify graph (k is fixed per captured graph). After verify, the scheduler
    /// appends rows 0..=num_accepted to ctx so every committed position gets its
    /// target hidden (fixes the ctx-undercount) and the bonus generator (row
    /// num_accepted) is the freshest slot (EAGLE). No-op unless DFlash is on,
    /// this layer is a capture layer, and rank 0.
    pub(super) fn try_dflash_capture_all(
        &self,
        layer_idx: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let slot = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let ctx_slot_bytes = self.dflash_capture_layers.len() * h * bf16;
        let kmax = self.dflash_hidden_save_rows;
        debug_assert!(
            k <= kmax,
            "try_dflash_capture_all: k={k} exceeds dflash_hidden_save_rows={kmax}"
        );
        let k_capped = k.min(kmax);
        for t in 0..k_capped {
            let src = self.buffers.hidden_states().offset(t * h * bf16);
            let dst_slot = dst.offset(t * ctx_slot_bytes + slot * h * bf16);
            self.gpu.copy_d2d_async(src, dst_slot, h * bf16, stream)?;
        }
        Ok(())
    }
}

impl TransformerModel {
    /// DSpark capture: hc-mean of `hc_streams` for `num_tokens` rows into
    /// this capture layer's slot of `dspark_dump_buf`. No-op unless DSpark
    /// capture is armed and `layer_idx` is a capture layer.
    ///
    /// `staged` selects the CUDA-graph-safe variant. The eager write below is
    /// indexed by `start_row` (= `seq.seq_len`), a host value that grows every
    /// step; baking it into a captured graph pins every replay to the first
    /// step's rows, so the drafter would be fed one frozen snapshot forever.
    /// Under capture the caller passes `staged = true`, sending hc_mean to the
    /// fixed row-0 address of `dspark_capture_stage`, and calls
    /// `dspark_capture_commit` after the graph launch to relocate the rows to
    /// their sequence positions eagerly.
    /// Sequence-position base the γ-verify's DSpark capture writes at.
    ///
    /// Task #45: the online captures the drafter reads were measured (probe
    /// dump vs a plain-decode `ATLAS_DSPARK_DUMP`) to hold the hidden of
    /// position `p-1` at row `p` — cos 0.98-0.99 against plain `p-1` where
    /// row `p` itself scores 0.45 — from the exact step the verify path takes
    /// over from plain decode. A one-row shift in the drafter's whole ring is
    /// enough to explain the 3.69 -> 1.02 tok/step online collapse.
    /// `ATLAS_DSPARK_CAP_WSHIFT=<int>` A/Bs the write base without a rebuild;
    /// the plain-decode and prefill captures (which agree with the offline
    /// oracle by construction) are never shifted.
    pub(super) fn dspark_verify_row_base(&self, seq_len: usize) -> usize {
        static WS: std::sync::OnceLock<i64> = std::sync::OnceLock::new();
        let shift = *WS.get_or_init(|| {
            std::env::var("ATLAS_DSPARK_CAP_WSHIFT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
        });
        (seq_len as i64 + shift).max(0) as usize
    }

    pub(super) fn try_dspark_capture(
        &self,
        layer_idx: usize,
        num_tokens: usize,
        start_row: usize,
        staged: bool,
        stream: u64,
    ) -> Result<()> {
        if self.dspark_dump_buf.is_null() || self.hc_mean_k.0 == 0 {
            return Ok(());
        }
        let Some(slot) = self
            .dspark_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        else {
            return Ok(());
        };
        let h = self.config.hidden_size;
        let layout = DsparkCaptureLayout::new(self.dspark_dump_rows, self.dspark_capture_ring);
        let (out, n) = if staged {
            let n = num_tokens.min(crate::model::DSPARK_STAGE_ROWS);
            (
                self.dspark_capture_stage
                    .offset(slot * crate::model::DSPARK_STAGE_ROWS * h * 2),
                n,
            )
        } else {
            for span in layout.spans(start_row, num_tokens) {
                let input = self
                    .buffers
                    .hc_streams()
                    .offset(span.src_row * self.config.hc_mult * h * 2);
                let out = self
                    .dspark_dump_buf
                    .offset((slot * self.dspark_dump_rows + span.dst_row) * h * 2);
                crate::layers::ops::hc_mean(
                    self.gpu.as_ref(),
                    self.hc_mean_k,
                    input,
                    out,
                    span.rows as u32,
                    h as u32,
                    self.config.hc_mult as u32,
                    stream,
                )?;
            }
            return Ok(());
        };
        crate::layers::ops::hc_mean(
            self.gpu.as_ref(),
            self.hc_mean_k,
            self.buffers.hc_streams(),
            out,
            n as u32,
            h as u32,
            self.config.hc_mult as u32,
            stream,
        )
    }

    /// Relocate a staged DSpark capture (see `try_dspark_capture(staged=true)`)
    /// from `dspark_capture_stage` row 0 to its sequence position `start_row`
    /// in `dspark_dump_buf`, for every capture layer. Enqueued on `stream`
    /// AFTER the verify graph launch, so it is ordered behind the graph's
    /// hc_mean writes and the host-computed `start_row` is fresh each step.
    pub(super) fn dspark_capture_commit(
        &self,
        num_tokens: usize,
        start_row: usize,
        stream: u64,
    ) -> Result<()> {
        if self.dspark_capture_stage.is_null() {
            return Ok(());
        }
        let h = self.config.hidden_size;
        let n = num_tokens.min(crate::model::DSPARK_STAGE_ROWS);
        let layout = DsparkCaptureLayout::new(self.dspark_dump_rows, self.dspark_capture_ring);
        for slot in 0..self.dspark_capture_layers.len() {
            let src = self
                .dspark_capture_stage
                .offset(slot * crate::model::DSPARK_STAGE_ROWS * h * 2);
            for span in layout.spans(start_row, n) {
                let dst = self
                    .dspark_dump_buf
                    .offset((slot * self.dspark_dump_rows + span.dst_row) * h * 2);
                self.gpu.copy_d2d_async(
                    src.offset(span.src_row * h * 2),
                    dst,
                    span.rows * h * 2,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// Flush one captured pass to the ATLAS_DSPARK_DUMP file. Record layout
    /// (little-endian, mirrored by the offline probe):
    ///   magic  u32 = 0x4453504B ("DSPK")
    ///   kind   u32   (0 = prefill, 1 = decode)
    ///   start  u32   (sequence position of row 0)
    ///   n      u32   (rows)
    ///   h      u32   (hidden size)
    ///   layers u32   (capture-layer count)
    ///   token  u32   (decode: the input token id; prefill: 0)
    ///   data   layers × n × h BF16
    pub(super) fn dspark_dump_flush(
        &self,
        kind: u32,
        start_pos: usize,
        num_tokens: usize,
        token: u32,
        stream: u64,
    ) -> Result<()> {
        use std::io::Write;
        let Some(ref dump) = self.dspark_dump else {
            return Ok(());
        };
        let n = num_tokens.min(self.dspark_dump_rows);
        let h = self.config.hidden_size;
        let nl = self.dspark_capture_layers.len();
        self.gpu.synchronize(stream)?;
        let mut w = dump.lock();
        for v in [
            0x4453504Bu32,
            kind,
            start_pos as u32,
            n as u32,
            h as u32,
            nl as u32,
            token,
        ] {
            w.write_all(&v.to_le_bytes())?;
        }
        let mut host = vec![0u8; n * h * 2];
        for slot in 0..nl {
            let src = self
                .dspark_dump_buf
                .offset((slot * self.dspark_dump_rows + start_pos.min(self.dspark_dump_rows)) * h * 2);
            self.gpu.copy_d2h(src, &mut host)?;
            w.write_all(&host)?;
        }
        w.flush()?;
        Ok(())
    }
}

impl TransformerModel {
    /// The DSpark hc-mean capture buffer `[layers, rows, h]` BF16 + its row
    /// capacity. NULL/0 unless ATLAS_DSPARK_CAPTURE=1 (or the dump probe)
    /// armed the capture at model build. The factory hands this to
    /// `DsparkDraftHead::set_capture` when installing the block drafter.
    pub fn dspark_capture_buf(&self) -> (DevicePtr, usize, bool) {
        (
            self.dspark_dump_buf,
            self.dspark_dump_rows,
            self.dspark_capture_ring,
        )
    }

    /// 4b inc-3 γ-verify catch-up: advance every compressor layer's compressed
    /// KV pool for the `num_committed` positions committed by the last verify
    /// (rows `0..num_committed` at absolute positions
    /// `pre_len..pre_len+num_committed`). The batched verify path
    /// (`ms_mla_decode_v4_flash`, `pos:None`) skips the decode-time compressed
    /// append, so the compressed arm would freeze during speculative decode and
    /// diverge from greedy; this replays that append from the per-layer
    /// `verify_comp_normed` capture. Eager only — the caller must NOT be under
    /// graph capture (the append re-runs host logic and reads MoE scratch, free
    /// post-forward). No-op on non-compressor layers (`verify_comp_normed` NULL).
    pub(crate) fn dspark_compress_catchup(
        &self,
        pre_len: usize,
        num_committed: usize,
        _stream: u64,
    ) -> Result<()> {
        // The verify forward runs on `default_stream()` (verify_d.rs), so the
        // append that feeds `v4_comp_pool_filled` MUST be enqueued there too —
        // otherwise the next verify step reads the advanced count but races the
        // pool-content writes (the fix would be inert). Ignore the passed value.
        let stream = self.gpu.default_stream();
        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
        };
        let eps = self.config.rms_norm_eps as f32;
        for layer in self.layers.iter() {
            let Some(attn) = layer
                .as_any()
                .and_then(|a| a.downcast_ref::<crate::layers::Qwen3AttentionLayer>())
            else {
                continue;
            };
            // Rewind anything the pre-verify speculation moved, THEN replay the
            // append for the committed rows. Order matters: the speculation ran
            // the compressor over all γ draft positions so each verify row could
            // see its own compressed blocks, and on a partial accept those
            // frontiers point past the accepted prefix. Restoring first leaves
            // the catch-up below byte-identical to the non-speculative path.
            // No-op when speculation did not run this step.
            attn.v4_compress_restore(&ctx, stream)?;
            // `pre_len` counts the last emitted-but-unforwarded token, so the
            // first committed verify row's FORWARD position is `pre_len - 1`
            // (task #45: basing the replay at `pre_len` wrote every ring slot
            // one position late and left one slot stale, so every replayed
            // pool block differed bytewise from plain decode's).
            attn.v4_compress_catchup(&ctx, pre_len.saturating_sub(1), num_committed, eps, stream)?;
        }
        Ok(())
    }
}
