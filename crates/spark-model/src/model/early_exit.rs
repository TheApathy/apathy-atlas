// SPDX-License-Identifier: AGPL-3.0-only

//! Target early-exit self-speculative drafter (`ATLAS_DFLASH_EARLY_EXIT=1`).
//!
//! The highest-ceiling no-retrain lever for NOVEL coding tokens. The tiny
//! z-lab drafter is intrinsically wrong at novel-logic cliffs (the right
//! token is not in its top-3). The fix: draft with the TARGET'S OWN first
//! `N` layers + `lm_head` — an in-distribution predictor (it IS the target,
//! at a fraction of the depth) much stronger than the tiny drafter.
//!
//! LOSSLESS by construction: the full 64-layer target still verifies greedy
//! → byte-identical output. Early-exit ONLY proposes candidate tokens; the
//! verify commits only the target's true greedy token (accepts a draft solely
//! when draft == target-greedy). A wrong draft costs one rejected speculation
//! and can never change committed output (same contract as the retrieval /
//! recycle propose sources). Token-exactness is therefore independent of `N`.
//!
//! ## How the partial forward handles state
//!
//! AEON-27B is a HYBRID model: of 64 layers, `full_attention_interval=4` makes
//! layers 3,7,11,…,63 full-attention (16 total) and the rest GDN/SSM linear
//! attention (48 total). Running layers `0..N` autoregressively therefore
//! advances BOTH:
//!   - paged KV (full-attn layers) at the draft positions, AND
//!   - the per-layer SSM `h_state` / `conv_state` recurrence.
//!
//! Both are made transient:
//!   - KV: the draft positions' KV for layers `0..N` is OVERWRITTEN when the
//!     verify recomputes the full `0..64` forward at the accepted positions
//!     (identical contract to the existing `decode_draft` self-spec path).
//!   - SSM: we `checkpoint_ssm_states` before the γ-token draft loop and
//!     `rollback_ssm_states(seq, 0)` (= restore-to-checkpoint) after, so the
//!     SSM advancement from the partial forward is fully undone. The
//!     production verify then takes its own fresh checkpoint and the
//!     existing K=γ SSM intermediate/rollback path is byte-for-byte unchanged.
//!   - `seq.seq_len` / `seq.tokens`: each draft step appends one token+pos
//!     (so layer `i`'s attention sees the running draft prefix); we rewind
//!     both to the pre-draft state before returning.

use anyhow::Result;
use atlas_core::config::LayerType;
use spark_runtime::gpu::DevicePtr;

use super::block_mgmt::ensure_blocks_through_decode;
use super::types::TransformerModel;
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;
use crate::traits::{Model, SequenceState};

#[allow(unused_imports)]
use spark_runtime::gpu::GpuBackend;

impl TransformerModel {
    /// True when target early-exit drafting is enabled.
    pub(super) fn early_exit_enabled() -> bool {
        static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *CACHE.get_or_init(|| std::env::var("ATLAS_DFLASH_EARLY_EXIT").ok().as_deref() == Some("1"))
    }

    /// Number of target layers to run before the early exit (`lm_head` is
    /// applied on the layer-`N` hidden). Default 31 (≈ half of 64). Clamped
    /// to `[1, num_layers]`.
    pub(super) fn early_exit_n_layers(&self) -> usize {
        let n = std::env::var("ATLAS_DFLASH_EARLY_EXIT_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(31);
        n.clamp(1, self.layers.len())
    }

    /// Target early-exit propose. Drafts `num_drafts` tokens autoregressively
    /// using the target's own first `N` layers + `final_norm` + `lm_head`.
    ///
    /// `token` is the bonus/last committed token (the verify's input row 0);
    /// the returned drafts are the candidate continuation `[d_0, .., d_{γ-1}]`
    /// fed to the verify exactly like the neural drafter's output. SSM state
    /// and the sequence cursor are restored before returning, so the caller's
    /// verify path is unaffected.
    pub(super) fn early_exit_propose(
        &self,
        token: u32,
        num_drafts: usize,
        seq: &mut SequenceState,
    ) -> Result<Vec<u32>> {
        let n_layers = self.early_exit_n_layers();
        let stream = self.gpu.default_stream();

        // Snapshot the live SSM state so the partial forward's recurrence
        // advancement is undone after drafting. Mirrors the verify path's
        // checkpoint/restore (verify_a.rs).
        self.checkpoint_ssm_states(seq)?;
        let seq_len_before = seq.seq_len;
        let tokens_before = seq.tokens.len();

        let profile = std::env::var("ATLAS_DFLASH_EARLY_EXIT_PROFILE")
            .ok()
            .as_deref()
            == Some("1");
        let t0 = std::time::Instant::now();

        let mut drafts = Vec::with_capacity(num_drafts);
        let mut next = token;
        for _ in 0..num_drafts {
            let logits = self.early_exit_forward(next, n_layers, seq, stream)?;
            let draft = self.argmax_on_device(logits, stream)?;
            drafts.push(draft);
            next = draft;
        }

        if profile {
            self.gpu.synchronize(stream)?;
            let us = t0.elapsed().as_micros();
            tracing::info!(
                "DFLASH EARLY_EXIT: N={n_layers} γ={num_drafts} propose={:.2}ms ({:.0}μs/draft) drafts[..min(8)]={:?}",
                us as f32 / 1000.0,
                us as f32 / num_drafts.max(1) as f32,
                &drafts[..drafts.len().min(8)],
            );
        }

        // Rewind the sequence cursor and tokens to the pre-draft state. The
        // draft positions' KV (layers 0..N) is left in place — it is
        // overwritten by the verify's full 0..64 recompute at the same slots.
        seq.seq_len = seq_len_before;
        seq.tokens.truncate(tokens_before);

        // Restore SSM state to the pre-draft checkpoint (num_accepted=0 =>
        // restore-to-checkpoint branch in rollback_ssm_states_dispatch).
        self.rollback_ssm_states(seq, 0)?;

        Ok(drafts)
    }

    /// One autoregressive step of the early-exit partial forward: embed
    /// `token`, run target layers `0..n_layers` (ALL layer types, including
    /// SSM — unlike `decode_draft` which skips SSM), then `final_norm` +
    /// `lm_head` on the layer-`N` hidden. Appends one token to `seq` (KV +
    /// SSM advance); the caller rewinds + restores after the draft loop.
    ///
    /// Metadata setup mirrors [`TransformerModel::decode_draft`].
    fn early_exit_forward(
        &self,
        token: u32,
        n_layers: usize,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<DevicePtr> {
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // 1. Embedding lookup.
        self.embed(token, hidden, stream)?;

        // 2. Ensure KV blocks for the current decode position + metadata.
        let bs = kv_cache.block_size();
        let blocks_needed = (seq.seq_len / bs) + 1;
        ensure_blocks_through_decode(
            seq,
            blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
        )?;

        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = seq.block_table.len() as u32;

        let pos_val = seq.seq_len as u32;
        self.gpu
            .copy_h2d_async(&pos_val.to_le_bytes(), meta_base, stream)?;

        let block_idx = seq
            .physical_block_for(seq.seq_len / bs)
            .unwrap_or(self.dummy_kv_block);
        let global_slot = (block_idx as i64) * (bs as i64) + ((seq.seq_len % bs) as i64);
        self.gpu
            .copy_h2d_async(&global_slot.to_le_bytes(), meta_base.offset(8), stream)?;

        let actual_seq_len = (seq.seq_len + 1) as i32;
        self.gpu
            .copy_h2d_async(&actual_seq_len.to_le_bytes(), meta_base.offset(16), stream)?;

        let bt_i32: Vec<i32> = seq.block_table.iter().map(|&b| b as i32).collect();
        let bt_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(bt_i32.as_ptr() as *const u8, bt_i32.len() * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(256), stream)?;

        let attn_metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(8),
            seq_len: meta_base.offset(16),
            block_table: meta_base.offset(256),
            max_blocks_per_seq: max_blocks,
            num_seqs: 1,
        };

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(attn_metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: false, // Eager mode — no CUDA graph.
            ddtree_parent_ids_dev: None,
            tree_aware_attn: None,
            ssm_multi_seq_ptr_table_override: None,
            self_spec_sparse_draft: None,
            ffn_defer: None,
        };

        // 3. Run the FIRST `n_layers` target layers — ALL layer types. The
        //    SSM layers advance their recurrence (restored after the draft
        //    loop); the full-attn layers append KV (overwritten by verify).
        for (i, layer) in self.layers.iter().take(n_layers).enumerate() {
            let _ = LayerType::LinearAttention; // (documented: SSM not skipped)
            layer.decode(
                hidden,
                residual,
                seq.layer_states[i].as_mut(),
                &mut kv_cache,
                seq.seq_len,
                &mut seq.block_table,
                &mut seq.disk_block_ids,
                &mut seq.disk_last_offloaded_per_layer,
                &ctx,
                stream,
            )?;
        }

        // 4. Early exit: final_norm + lm_head on the layer-N hidden. The
        //    target's `final_norm` is trained for the layer-64 hidden but is
        //    a reasonable shared head for the layer-N hidden — the verify is
        //    the oracle, so any head mismatch only costs acceptance.
        let normed = self.buffers.norm_output();
        let h = self.config.hidden_size as u32;
        let eps = self.config.rms_norm_eps as f32;
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            hidden,
            &self.final_norm,
            normed,
            1,
            h,
            eps,
            stream,
        )?;
        self.lm_head(normed, stream)?;

        // 5. Advance the cursor so the next draft step's layer attention sees
        //    this token in its prefix. Rewound by the caller after the loop.
        seq.tokens.push(token);
        seq.seq_len += 1;

        Ok(self.decode_logits_ptr())
    }
}
