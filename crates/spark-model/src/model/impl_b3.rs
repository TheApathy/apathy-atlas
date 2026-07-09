// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
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
            // ATLAS_DFLASH_DUMP_MIN_POS=N defers the one-shot dump until the
            // first propose at position >= N (mid-generation ctx capture).
            && position
                >= std::env::var("ATLAS_DFLASH_DUMP_MIN_POS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(0)
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
            ddtree_parent_ids_dev: None,
            tree_aware_attn: None,
            ssm_multi_seq_ptr_table_override: None,
            self_spec_sparse_draft: None,
            ffn_defer: None,
        };
        // ── ATLAS_DFLASH_EARLY_EXIT=1: target early-exit drafter ──
        //
        // Replace the tiny z-lab drafter's propose with the TARGET'S OWN first
        // N layers + lm_head as the draft source (see `early_exit.rs`). The
        // drafts flow into the unchanged verify path; the full 64-layer target
        // still commits only its greedy token, so output stays byte-identical
        // (LOSSLESS — verify is the oracle). This is the in-distribution
        // predictor that beats the tiny drafter on NOVEL coding tokens.
        //
        // The DFlash proposer state's ctx accumulator append (done inside
        // `propose_drafts`) is intentionally skipped here: when early-exit is
        // the draft source the neural drafter never runs, so its ctx is never
        // consumed. The verify path's SSM checkpoint/rollback is independent of
        // this and unchanged.
        if Self::early_exit_enabled() {
            return self.early_exit_propose(token, num_drafts, seq);
        }
        let prop_state = seq
            .proposer_state
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("No proposer state for sequence"))?;
        // Refresh the host token mirror used by both prompt-lookup drafting
        // (ATLAS_DFLASH_PLD) and the generalized retrieval-augmented drafter
        // (ATLAS_DFLASH_RETRIEVAL). `seq.tokens` is the FULL committed
        // sequence = prompt tokens + everything generated so far, so the
        // retrieval haystack includes any reference code in the prompt for
        // free. Only populated when one of the flags is on (default off ⇒
        // no extra copy, legacy behavior byte-for-byte).
        let want_token_mirror = std::env::var("ATLAS_DFLASH_PLD").ok().as_deref() == Some("1")
            || std::env::var("ATLAS_DFLASH_RETRIEVAL").ok().as_deref() == Some("1")
            || std::env::var("ATLAS_DFLASH_SAM").ok().as_deref() == Some("1");
        if want_token_mirror
            && let Some(ds) = prop_state
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
        {
            ds.pld_tokens.clear();
            ds.pld_tokens.extend_from_slice(&seq.tokens);
        }
        proposer.propose(
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
        )
    }

    /// Borrow the GPU backend for post-construction wiring (e.g. installing
    /// a DFlash proposer that needs to allocate paged KV caches against the
    /// same GPU the target uses).
    pub fn gpu_backend(&self) -> &dyn GpuBackend {
        self.gpu.as_ref()
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

    /// Per-chunk MTP last-K capture (called from `prefill_chunk_dispatch` after
    /// `forward_layers`). Copies the just-computed chunk's hidden states into
    /// the per-sequence host ring buffer `seq.mtp_lastk_host_buf`, shifting
    /// older rows out when the ring is full. The actual H2D into the shared
    /// `mtp_lastk_buf` device buffer + proposer `prefill_last_k` happens at
    /// `mtp_lastk_prefill_after_finalize` on the last chunk only.
    ///
    /// No-op when MTP last-K prefill is disabled (capacity==0), no proposer
    /// is wired, or rank > 0 under EP/TP.
    ///
    /// `proc_count` is the number of rows in `hidden_states` for this chunk.
    /// `chunk_start + chunk_len - 1` is the absolute position of the last
    /// captured row.
    pub(super) fn mtp_lastk_capture_chunk(
        &self,
        seq: &mut crate::traits::SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        let capacity = self.mtp_lastk_capacity;
        if capacity == 0 || self.proposer.is_none() {
            return Ok(());
        }
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        if proc_count == 0 {
            return Ok(());
        }
        let h = self.config.hidden_size;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let row_bytes = h * fp32;
        let ring_capacity_bytes = capacity * row_bytes;

        // Lazy-size the host ring buffer.
        if seq.mtp_lastk_host_buf.len() != ring_capacity_bytes {
            seq.mtp_lastk_host_buf = vec![0u8; ring_capacity_bytes];
            seq.mtp_lastk_host_filled = 0;
            seq.mtp_lastk_end_abs = 0;
        }

        // Number of rows to ingest from this chunk: at most `capacity` (the
        // last `capacity` rows of the chunk fully replace the ring), at most
        // `proc_count` rows present in `hidden_states`.
        let ingest = capacity.min(proc_count);
        if ingest == 0 {
            return Ok(());
        }

        // Source row range in `hidden_states`: the trailing `ingest` rows of
        // the chunk's proc_count outputs.
        let src_first_row = proc_count - ingest;
        let src_bytes = ingest * row_bytes;
        let src_ptr = self
            .buffers
            .hidden_states()
            .offset(src_first_row * row_bytes);

        if ingest >= capacity {
            // Chunk produced ≥ K rows — wipe ring and copy the last K rows.
            // D2H directly into the host ring buf.
            // SAFETY: copy_d2h requires &mut [u8] of exact size; the ring is
            // sized to `ring_capacity_bytes` and `src_bytes == ring_capacity_bytes`.
            self.gpu.synchronize(stream)?;
            self.gpu
                .copy_d2h(src_ptr, &mut seq.mtp_lastk_host_buf[..src_bytes])?;
            seq.mtp_lastk_host_filled = capacity;
        } else {
            // Shift the existing ring left by `ingest` rows (drop oldest),
            // then append the new rows at the tail.
            let shift_bytes = ingest * row_bytes;
            if seq.mtp_lastk_host_filled == capacity {
                // Ring is full: drop oldest `ingest` rows.
                seq.mtp_lastk_host_buf
                    .copy_within(shift_bytes..ring_capacity_bytes, 0);
                let tail_start = ring_capacity_bytes - shift_bytes;
                self.gpu.synchronize(stream)?;
                self.gpu.copy_d2h(
                    src_ptr,
                    &mut seq.mtp_lastk_host_buf[tail_start..tail_start + shift_bytes],
                )?;
            } else if seq.mtp_lastk_host_filled + ingest <= capacity {
                // Ring has room: append at `filled`.
                let tail_start = seq.mtp_lastk_host_filled * row_bytes;
                self.gpu.synchronize(stream)?;
                self.gpu.copy_d2h(
                    src_ptr,
                    &mut seq.mtp_lastk_host_buf[tail_start..tail_start + shift_bytes],
                )?;
                seq.mtp_lastk_host_filled += ingest;
            } else {
                // Partial overlap: existing rows shift out the front, new
                // rows fill the tail. New filled count is `capacity`.
                let drop = seq.mtp_lastk_host_filled + ingest - capacity;
                let drop_bytes = drop * row_bytes;
                let keep_bytes = seq.mtp_lastk_host_filled * row_bytes - drop_bytes;
                seq.mtp_lastk_host_buf
                    .copy_within(drop_bytes..drop_bytes + keep_bytes, 0);
                let tail_start = keep_bytes;
                self.gpu.synchronize(stream)?;
                self.gpu.copy_d2h(
                    src_ptr,
                    &mut seq.mtp_lastk_host_buf[tail_start..tail_start + shift_bytes],
                )?;
                seq.mtp_lastk_host_filled = capacity;
            }
        }
        seq.mtp_lastk_end_abs = chunk_start + chunk_len - 1;

        if std::env::var("ATLAS_MTP_DEBUG").ok().as_deref() == Some("1") {
            tracing::info!(
                "ATLAS_MTP_DEBUG mtp_lastk_capture_chunk: \
                 chunk=[{chunk_start},{}] proc_count={proc_count} \
                 ingest={ingest} filled={}/{capacity} end_abs={}",
                chunk_start + chunk_len,
                seq.mtp_lastk_host_filled,
                seq.mtp_lastk_end_abs,
            );
        }
        Ok(())
    }

    /// MTP last-K prefill (called from `prefill_b_finalize_last` on the last
    /// chunk). H2Ds the per-sequence cross-chunk host ring buffer
    /// (`seq.mtp_lastk_host_buf`, populated by `mtp_lastk_capture_chunk` on
    /// every chunk) into the shared `self.mtp_lastk_buf`, then asks the
    /// proposer to replay the rows so its self-attention KV cache covers the
    /// prompt tail before the first decode.
    ///
    /// No-op when:
    ///   - `mtp_lastk_capacity == 0` (env-gate disabled)
    ///   - No proposer wired (e.g. non-speculative serve)
    ///   - No `mtp_lastk_buf` allocated (defensive)
    ///   - Rank > 0 under EP/TP (MTP runs on rank 0 only)
    ///   - `seq.mtp_lastk_host_filled == 0` (capture never fired)
    ///
    /// `proc_count` and `chunk_len` are retained as args for backward
    /// compatibility with the call site but are no longer used to size the
    /// captured window — the host ring already holds the cross-chunk tail.
    pub(super) fn mtp_lastk_prefill_after_finalize(
        &self,
        tokens: &[u32],
        seq: &mut crate::traits::SequenceState,
        _chunk_start: usize,
        _chunk_len: usize,
        _proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        let capacity = self.mtp_lastk_capacity;
        if capacity == 0 {
            return Ok(());
        }
        let proposer = match &self.proposer {
            Some(p) => p.clone(),
            None => return Ok(()),
        };
        let lastk_buf = match self.mtp_lastk_buf {
            Some(p) => p,
            None => return Ok(()),
        };
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }

        // Read the per-seq cross-chunk capture state populated by
        // `mtp_lastk_capture_chunk` at every prefill chunk.
        let filled = seq.mtp_lastk_host_filled;
        if filled == 0 || seq.mtp_lastk_host_buf.is_empty() {
            // Capture never fired (e.g. all chunks early-returned via cache).
            // Nothing to prefill into MTP.
            tracing::warn!(
                "MTP last-K prefill: skipped — host ring empty (filled={filled}, \
                 buf_bytes={}). Likely all chunks were prefix-cache early-returns.",
                seq.mtp_lastk_host_buf.len(),
            );
            return Ok(());
        }
        let end_abs = seq.mtp_lastk_end_abs;
        let h = self.config.hidden_size;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let row_bytes = h * fp32;
        let used_bytes = filled * row_bytes;

        // The `filled` rows in the host ring occupy
        // `mtp_lastk_host_buf[(capacity-filled)*row_bytes ..]` when partially
        // filled OR `[0..]` when full — actually our shift logic always keeps
        // the populated rows contiguous starting at offset 0 when the ring
        // has been padded with copy_within back to slot 0 OR ending at the
        // tail. Re-read the capture invariant:
        //   - When `filled == capacity`: rows are at `[0 .. capacity*row_bytes]`
        //     (oldest at row 0, newest at row capacity-1).
        //   - When `filled < capacity`: rows are at `[0 .. filled*row_bytes]`
        //     (the append-to-tail branch fills from offset
        //     `mtp_lastk_host_filled` upward each chunk).
        // Both cases: the first `filled` rows of the buffer are the live data.
        let src_slice = &seq.mtp_lastk_host_buf[..used_bytes];
        self.gpu.copy_h2d_async(src_slice, lastk_buf, stream)?;

        // The captured tokens span `[end_abs - filled + 1 ..= end_abs]`.
        let start_abs = end_abs + 1 - filled;
        if start_abs > end_abs || end_abs >= tokens.len() {
            tracing::warn!(
                "MTP last-K prefill: invalid span [{start_abs}, {end_abs}] for tokens.len()={}",
                tokens.len(),
            );
            return Ok(());
        }
        let captured_tokens: Vec<u32> = tokens[start_abs..=end_abs].to_vec();

        // Build a ForwardContext mirroring `run_mtp_propose_inner`. MTP
        // loads ALL experts on every rank, so MoE output is already complete
        // → no all_reduce needed (comm: None).
        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            ddtree_parent_ids_dev: None,
            tree_aware_attn: None,
            ssm_multi_seq_ptr_table_override: None,
            self_spec_sparse_draft: None,
            ffn_defer: None,
        };

        let prop_state = match seq.proposer_state.as_mut() {
            Some(ps) => ps,
            None => return Ok(()),
        };

        let t0 = std::time::Instant::now();
        proposer.prefill_last_k(
            &captured_tokens,
            lastk_buf,
            end_abs,
            prop_state.as_mut(),
            &ctx,
            stream,
        )?;
        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(
            "MTP last-K prefill: filled={filled}/{capacity} rows from cross-chunk ring, \
             end_abs={end_abs} (span=[{start_abs},{end_abs}]), took {dt_ms:.1}ms",
        );
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
        }
        Ok(())
    }

    /// DFlash drafter-retrain teacher-forced capture. When
    /// `ATLAS_DUMP_CTX_HIDDEN=<path>` is set, dump the per-sequence
    /// `ctx_hidden_acc` (all-position × 5-capture-layer hiddens from the
    /// just-completed prefill, computed on the NVFP4 serving path) to the
    /// append-only file. One record per request. No-op when the env var is
    /// unset, DFlash capture is inactive, or rank > 0 under EP/TP.
    ///
    /// The `ctx_hidden_acc` device layout is `[pos, slot, hidden]` BF16
    /// (per-position stride `n_capture * hidden * 2`), which is exactly the
    /// SpecForge offline `[T, L*H]` tensor with L = capture layers in
    /// `dflash_capture_layers` order (e.g. [1,16,31,46,61]). The Python
    /// harness pads to SPECFORGE_PAD_TO and saves `{md5(padded ids)}.pt`.
    ///
    /// Record format (little-endian):
    /// ```text
    /// u32 magic        = 0xC7D5_1DEE
    /// u32 seq_len      = number of positions dumped (= dstate.ctx_len)
    /// u32 n_capture    = number of capture layers (5)
    /// u32 hidden_dim   = model hidden_size (5120)
    /// u64 tok_fnv      = FNV-1a of the request's prompt-token bytes (pairing)
    /// bf16 payload[seq_len * n_capture * hidden_dim]  (layout [pos, slot, h])
    /// ```
    /// Must run in eager mode (synchronous `copy_d2h`); prefill is already
    /// eager (no CUDA-graph capture on the prefill path).
    pub(super) fn dump_ctx_hidden_after_prefill(
        &self,
        seq: &mut crate::traits::SequenceState,
        tokens: &[u32],
    ) -> Result<()> {
        let path = match crate::model::env_diag::dump_ctx_hidden_path() {
            Some(p) => p,
            None => return Ok(()),
        };
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let n_capture = self.dflash_capture_layers.len();

        let (acc_base, n) = match seq.proposer_state.as_mut() {
            Some(ps) => match ps
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
            {
                Some(dstate) => (dstate.ctx_hidden_acc, dstate.ctx_len),
                None => return Ok(()),
            },
            None => return Ok(()),
        };
        if acc_base.0 == 0 || n == 0 {
            tracing::warn!(
                "ATLAS_DUMP_CTX_HIDDEN: no ctx_hidden_acc/ctx_len (acc={:#x}, n={}) — skipping dump",
                acc_base.0,
                n
            );
            return Ok(());
        }

        let total_bytes = n * n_capture * h * bf16;
        let mut host_buf = vec![0u8; total_bytes];
        self.gpu.copy_d2h(acc_base, &mut host_buf)?;

        // FNV-1a over the prompt token bytes — lets the Python harness assert
        // it paired the right record with the right sample.
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x100_0000_01b3;
        let mut fnv: u64 = FNV_OFFSET;
        for &t in tokens {
            for &b in t.to_le_bytes().iter() {
                fnv ^= b as u64;
                fnv = fnv.wrapping_mul(FNV_PRIME);
            }
        }

        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| anyhow::anyhow!("ATLAS_DUMP_CTX_HIDDEN open {path}: {e}"))?;
        const CTX_HIDDEN_MAGIC: u32 = 0xC7D5_1DEE;
        f.write_all(&CTX_HIDDEN_MAGIC.to_le_bytes())?;
        f.write_all(&(n as u32).to_le_bytes())?;
        f.write_all(&(n_capture as u32).to_le_bytes())?;
        f.write_all(&(h as u32).to_le_bytes())?;
        f.write_all(&fnv.to_le_bytes())?;
        f.write_all(&host_buf)?;
        tracing::info!(
            "ATLAS_DUMP_CTX_HIDDEN: wrote record seq_len={n} n_capture={n_capture} h={h} \
             ({total_bytes} bytes payload) fnv={fnv:#018x} → {path}"
        );
        Ok(())
    }

    /// THINKING-PHASE ctx capture (default-OFF, `ATLAS_DFLASH_CAPTURE_THINKING=1`).
    ///
    /// During the model's `<think>`…`</think>` span the scheduler runs the
    /// plain-decode path (`step_decode_only` → `decode_batch` → `decode`),
    /// NOT the DFlash propose/verify cycle. That plain decode still fires the
    /// per-layer `try_dflash_capture` hook, which lands the just-decoded
    /// thinking token's 5 target-layer hiddens in `dflash_hidden_save[0]` —
    /// but nothing ever appends that row into `ctx_hidden_acc` (the append
    /// lives in `propose_drafts`, which only runs in the answer phase). As a
    /// result the ctx slots spanning the thinking span are left ZERO, and
    /// when the answer phase starts the drafter attends over hundreds of
    /// zero-norm context keys.
    ///
    /// This method copies the freshly captured `dflash_hidden_save[0]` row
    /// (all `n_capture` layers, one contiguous `ctx_slot_bytes` block) into
    /// the absolute slot of the token that was just decoded, then advances
    /// `ctx_len` to keep it in lockstep with `seq.seq_len`. Must be called
    /// AFTER `decode` has incremented `seq.seq_len`, so the just-decoded
    /// token sits at absolute position `seq.seq_len - 1`.
    ///
    /// Cost: one `ctx_slot_bytes` d2d copy per thinking token. No-op when:
    ///   - the flag is unset,
    ///   - DFlash is inactive (`dflash_hidden_save` is `None` / capture layers empty),
    ///   - the seq has no `DflashProposerState`,
    ///   - rank > 0 under EP/TP (drafter is rank-0 only),
    ///   - the accumulator is already full.
    ///
    /// Drafter-conditioning ONLY: the target's verify path is untouched, so
    /// committed tokens remain byte-identical — this raises ACCEPTANCE, not
    /// output.
    pub(super) fn dflash_capture_thinking_dispatch(
        &self,
        seq: &mut crate::traits::SequenceState,
        stream: u64,
    ) -> Result<()> {
        if !crate::model::env_diag::dflash_capture_thinking_enabled() {
            return Ok(());
        }
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        let src_base = match self.dflash_hidden_save {
            Some(p) => p,
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
        // `decode` already incremented seq.seq_len; the token whose hiddens
        // are sitting in dflash_hidden_save[0] occupies absolute position
        // seq.seq_len - 1. ctx slot i == hidden of token at sequence position
        // i (matches propose_drafts' absolute-slot semantics), so target slot
        // is exactly seq.seq_len - 1.
        let abs_pos = match seq.seq_len.checked_sub(1) {
            Some(p) => p,
            None => return Ok(()),
        };
        if abs_pos >= dstate.max_ctx_len {
            return Ok(()); // accumulator full; drop later positions
        }
        // dflash_hidden_save layout is [k_max, n_capture, hidden] BF16; row 0
        // is one whole ctx slot (n_capture * hidden * bf16 == ctx_slot_bytes).
        let slot_bytes = dstate.ctx_slot_bytes;
        let dst = dstate.ctx_hidden_acc.offset(abs_pos * slot_bytes);
        self.gpu.copy_d2d_async(src_base, dst, slot_bytes, stream)?;
        // Keep ctx_len in lockstep with seq_len. Using max() guards against
        // any transient where ctx_len was already advanced past this slot.
        dstate.ctx_len = dstate.ctx_len.max((abs_pos + 1).min(dstate.max_ctx_len));
        if std::env::var("ATLAS_PROPOSE_PROBE").ok().as_deref() == Some("1") {
            tracing::info!(
                "dflash_capture_thinking: abs_pos={} ctx_len={} slot_bytes={}",
                abs_pos,
                dstate.ctx_len,
                slot_bytes,
            );
        }
        Ok(())
    }

    /// No-op. The original "bonus-decode" approach was based on the
    /// misunderstanding that the drafter needs the bonus's hidden in
    /// ctx — actually the bonus is at position seq_len (not yet in
    /// KV), and ctx should contain hiddens for positions [0..seq_len-1]
    /// which is exactly what dflash_hidden_save[0..N+1] provides
    /// after a verify (last_token at position P, drafts at P+1..P+N).
    /// The bonus appears as the FIRST noise embedding (Q-side input)
    /// at position P+N+1.
    pub(super) fn save_hidden_for_dflash_dispatch(
        &self,
        _token: u32,
        _seq: &mut crate::traits::SequenceState,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// DFlash 5-layer hidden capture. Called inside each per-layer loop after
    /// `layer.decode(...)` returns. No-op when DFlash is disabled (the buffer
    /// is `None`) or when `layer_idx` is not in `dflash_capture_layers`.
    ///
    /// Writes hidden state for input row `token_idx` into the
    /// `dflash_hidden_save` buffer at `[token_idx, slot, ..]` where `slot`
    /// is this layer's index in `dflash_capture_layers`. The buffer is laid
    /// out as `[k_max, n_capture, hidden]` BF16 so the downstream propose
    /// can read whole-token rows of stride `n_capture * hidden * bf16`.
    ///
    /// Pre-2026-05-18 bug: the per-token offset was missing — every token
    /// position wrote to slot 0 of the buffer, leaving only the last
    /// captured token's hiddens visible to the drafter and corrupting the
    /// ctx_hidden_acc on K>=2 verify paths. K=2/3/4 partly hid the bug
    /// (small γ, low ctx mismatch), but K=γ=16 collapsed accept rate to
    /// ~2% because every ctx slot got seeded with the last-position hidden.
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
        // DEBUG: bypass capture to test if dflash_hidden_save writes corrupt
        // target output. Set ATLAS_DFLASH_NO_CAPTURE=1 to skip the d2d.
        if std::env::var("ATLAS_DFLASH_NO_CAPTURE").ok().as_deref() == Some("1") {
            return Ok(());
        }
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
        debug_assert!(
            !self.config.use_fp32_residual(),
            "DFlash hidden capture currently assumes BF16 residual; FP32-residual models need a separate downcast path"
        );
        let n_capture = self.dflash_capture_layers.len();
        let src = self.buffers.hidden_states().offset(token_idx * h * bf16);
        // [token_idx, slot, hidden] BF16 layout. Per-token stride =
        // n_capture * h * bf16; per-slot stride = h * bf16.
        let dst_off = token_idx * n_capture * h * bf16 + slot * h * bf16;
        let dst_slot = dst.offset(dst_off);
        self.gpu.copy_d2d_async(src, dst_slot, h * bf16, stream)?;
        if std::env::var("ATLAS_PROPOSE_PROBE").ok().as_deref() == Some("1") {
            tracing::info!(
                "try_dflash_capture: layer_idx={} token_idx={} slot={} dst_off={} h={}",
                layer_idx,
                token_idx,
                slot,
                dst_off,
                h,
            );
        }
        Ok(())
    }

    /// Flush captured hidden states from `dflash_hidden_save` (GPU) to the
    /// path in `$ATLAS_DUMP_HIDDEN` (host file). No-op when:
    ///   - env var is unset (production hot path — zero cost)
    ///   - `dflash_hidden_save` was never allocated
    ///   - `dflash_capture_layers` is empty
    ///   - this is a non-rank-0 worker under EP/TP (drafter is replicated)
    ///
    /// Must be called from a verify path that runs in EAGER mode (no CUDA
    /// graph capture) — `copy_d2h` is a synchronous device→host transfer
    /// and would fail with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED (900)`
    /// inside a captured region. `verify_b.rs` already forces eager mode
    /// when `ATLAS_DUMP_HIDDEN` is set (see the env-var check at the
    /// `suppress_graphs` re-enable site), so unconditional calls from the
    /// post-layer-loop site are safe.
    ///
    /// Record format (little-endian, 16-byte header + bf16 payload):
    /// ```text
    /// u32 magic         = 0xA71A5DEE   // distinct from TOKEN_DUMP_MAGIC (0xA71B5DEE)
    /// u32 layer_idx                    // ABSOLUTE layer index (one of dflash_capture_layers)
    /// u32 token_idx                    // position within the K-step verify
    /// u32 hidden_dim                   // model hidden_size (5120 on AEON-27B, 2048 on 35B-A3B-abl)
    /// bf16 hidden[hidden_dim]
    /// ```
    /// Bytes per record = 16 + hidden_dim * 2. The dump is APPEND-ONLY
    /// across the run; downstream training code splits records on the
    /// magic header.
    ///
    /// Pairs with the emitted-token records written by `meta.rs::argmax*`
    /// (magic 0xA71B5DEE). Together they let an offline trainer pair
    /// each emitted token with the 5 hidden states that preceded it.
    ///
    /// `k` is the verify width (number of `token_idx` slots populated by
    /// the just-completed layer loop — K=2 for K=2 verify, K=3 for K=3
    /// MTP verify, K=γ+1 for DFlash γ verify).
    pub(super) fn flush_hidden_dump(&self, k: usize) -> Result<()> {
        // Hot-path early exit: env var check is the cheapest possible.
        let path = match std::env::var("ATLAS_DUMP_HIDDEN") {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        // Rank-0 gate (mirrors try_dflash_capture's behavior). Replicated
        // drafter under EP/TP — only rank 0 owns it.
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }

        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let n_capture = self.dflash_capture_layers.len();
        let bytes_per_slot = h * bf16;
        let bytes_per_token = n_capture * bytes_per_slot;
        let total_bytes = k * bytes_per_token;

        // copy_d2h is synchronous on the default stream — graph-capture
        // unsafe but the caller guarantees eager mode (see method doc).
        let mut host_buf = vec![0u8; total_bytes];
        self.gpu.copy_d2h(dst, &mut host_buf)?;

        // Open file in append mode. If open or any write fails we surface
        // the error rather than silently swallowing (caller is in eager
        // mode by design — we want loud failures so missing dumps are
        // visible in logs).
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| anyhow::anyhow!("ATLAS_DUMP_HIDDEN open {path}: {e}"))?;

        const HIDDEN_DUMP_MAGIC: u32 = 0xA71A_5DEE;
        for t in 0..k {
            for (slot, &layer_idx) in self.dflash_capture_layers.iter().enumerate() {
                let off = t * bytes_per_token + slot * bytes_per_slot;
                let hidden_bytes = &host_buf[off..off + bytes_per_slot];
                f.write_all(&HIDDEN_DUMP_MAGIC.to_le_bytes())?;
                f.write_all(&(layer_idx as u32).to_le_bytes())?;
                f.write_all(&(t as u32).to_le_bytes())?;
                f.write_all(&(h as u32).to_le_bytes())?;
                f.write_all(hidden_bytes)?;
            }
        }
        // No fsync — append-only writes to a single file are fine across
        // open/close cycles, and per-step fsync would tank throughput.
        Ok(())
    }
}
