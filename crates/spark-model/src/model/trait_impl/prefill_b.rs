// SPDX-License-Identifier: AGPL-3.0-only

//! `prefill_chunk_dispatch` orchestrator.
//!
//! Refactor wave-4e split a 1000-LoC monolith into Pattern-B phase fns
//! (siblings under `prefill_b/`). The MutexGuard on `kv_cache` is
//! acquired here once and threaded through each phase as `&mut`.
//!
//! Phases (by section comment in original):
//!   1+1b → embed_chunk     (token embed + vision-pad overlay)
//!   2    → prefix_lookup   (prefix-cache hit + EP-sync + Marconi)
//!   2b   → proc_range      (recompute proc_start/count after skip; may early-return)
//!   3    → upload_meta     (positions + MRoPE + slots staging upload)
//!   3b   → upload_paged    (paged-prefill block_table + seq_len upload)
//!   4    → forward_layers  (per-layer prefill/decode + diagnostics)
//!   5-8  → finalize_last   (final norm + lm_head + snapshot save) — last chunk
//!   9    → save_intermediate_checkpoint — non-last chunk

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::traits::{Model, SequenceState};

mod batch;
mod batch_kernel;
#[cfg(test)]
mod batch_kernel_tests;
mod batched_layer;
mod embed_chunk;
mod finalize_last;
mod forward_layers;
mod h_state_ptrs;
mod prefix_lookup;
mod proc_range;
mod save_checkpoint;
mod stage_batched;
mod upload_meta;
mod upload_paged;

/// Whether an SSM snapshot may seed this prefill.
///
/// An exact full-prompt snapshot contains recurrent state *after* the last
/// prompt token. Replaying that token to obtain logits would advance the SSM a
/// second time, so exact hits default to a full recurrent recompute. The
/// partial-prefix path remains safe and enabled. Keep the opt-in only for
/// controlled A/B comparisons.
pub(in crate::model) fn should_restore_ssm_snapshot(
    snapshot_tokens: usize,
    matched_tokens: usize,
    total_tokens: usize,
) -> bool {
    should_restore_ssm_snapshot_with_opt_in(
        snapshot_tokens,
        matched_tokens,
        total_tokens,
        std::env::var("ATLAS_MARCONI_EXACT").as_deref() == Ok("1"),
    )
}

fn should_restore_ssm_snapshot_with_opt_in(
    snapshot_tokens: usize,
    matched_tokens: usize,
    total_tokens: usize,
    exact_opt_in: bool,
) -> bool {
    snapshot_tokens != matched_tokens || matched_tokens != total_tokens || exact_opt_in
}

/// Prefix length whose model state is actually reusable after lookup.
///
/// KV blocks can match deeper than the surviving SSM checkpoint. Recurrent
/// execution must resume at the checkpoint, not at the KV match depth.
pub(in crate::model) fn restored_prefix_skip_tokens(
    has_ssm: bool,
    snapshot_tokens: usize,
    matched_tokens: usize,
) -> usize {
    if has_ssm {
        snapshot_tokens
    } else {
        matched_tokens
    }
}

/// F82 is an attention-only shortcut. Hybrid models may deliberately decline
/// an otherwise paired snapshot (for example the exact-full replay hazard),
/// but that must never turn into a KV-only skip with unrestored recurrent
/// state.
pub(in crate::model) fn attention_only_prefix_skip(
    matched_tokens: usize,
    snapshot_restored: bool,
    has_ssm: bool,
) -> bool {
    matched_tokens > 0 && !snapshot_restored && !has_ssm
}

/// Number of leading rows in the current processing slice whose KV entries
/// already belong to the prefix cache and must not be overwritten.
pub(in crate::model) fn cached_kv_rows_in_slice(
    cached_prefix_tokens: usize,
    process_start: usize,
    process_count: usize,
) -> usize {
    cached_prefix_tokens
        .saturating_sub(process_start)
        .min(process_count)
}

impl TransformerModel {
    pub(super) fn prefill_chunk_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        let total = tokens.len();
        assert!(
            chunk_start + chunk_len <= total,
            "chunk_start({chunk_start}) + chunk_len({chunk_len}) > total({total})"
        );

        // Guard: chunk_len must not exceed buffer arena capacity.
        // Exceeding this causes CUDA illegal memory access (error 700)
        // which permanently corrupts GPU state.
        let arena_cap = self.buffers.max_batch_tokens();
        if chunk_len > arena_cap {
            anyhow::bail!(
                "Prefill chunk ({chunk_len} tokens) exceeds buffer arena capacity ({arena_cap} tokens). \
                 Reduce --max-prefill-tokens or prompt length."
            );
        }

        // Use the caller-provided stream for compute-copy overlap,
        // unless EP is active (NCCL requires the default stream).
        let stream = if self.comm.is_some() && self.config.ep_world_size > 1 {
            self.gpu.default_stream()
        } else {
            stream
        };

        // EP=2: zero ALL buffers on every chunk (NCCL defense-in-depth).
        // EP=1, first chunk (chunk_start==0): zero essentials (stale data from prior request).
        // EP=1, subsequent chunks: skip zeroing — buffers are overwritten by embedding
        // + layer forward before read. Saves 7 memsets × (chunks-1) per prefill.
        if self.comm.is_some() {
            self.buffers.zero_all(self.gpu.as_ref(), stream)?;
        } else if chunk_start == 0 {
            self.buffers.zero_all(self.gpu.as_ref(), stream)?;
        }

        let mut kv_cache = self.kv_cache.lock();

        // Env-gated phase profiler: ATLAS_PREFILL_PHASE_PROFILE=1 prints
        // CPU-wall ms per prefill phase (host time between markers; GPU work
        // is async unless the phase syncs).
        let phase_on = crate::full_profile::phase_profile_enabled();
        let phase_t0 = std::time::Instant::now();
        let mut phase_ms: Vec<(String, f64)> = Vec::new();
        macro_rules! phase_mark {
            ($name:expr) => {
                if phase_on {
                    phase_ms.push(($name.to_string(), phase_t0.elapsed().as_secs_f64() * 1000.0));
                }
            };
        }

        // ── Phase 1+1b: embed chunk + vision pad overlay ──
        self.prefill_b_embed_chunk(tokens, chunk_start, chunk_len, stream)?;
        phase_mark!("embed");

        // ── Phase 2: prefix-cache lookup + EP sync + Marconi snapshot restore ──
        let (kv_write_start, marconi_skip) =
            self.prefill_b_prefix_lookup(tokens, seq, chunk_start, total, &mut kv_cache, stream)?;
        phase_mark!("prefix_lookup");

        // Allocate blocks needed through end of this chunk.
        let bs = kv_cache.block_size();
        let end_pos = chunk_start + chunk_len;
        let blocks_needed = (end_pos - 1) / bs + 1;
        super::super::block_mgmt::ensure_blocks_through_prefill(
            seq,
            blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
        )?;
        phase_mark!("blocks");

        // ── Phase 2b: compute effective processing range (may early-return) ──
        let (proc_start, proc_count, effective_seq_len_start) = match self.prefill_b_proc_range(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            is_last_chunk,
            kv_write_start,
            marconi_skip,
            stream,
        )? {
            proc_range::ProcRange::Compute {
                proc_start,
                proc_count,
                effective_seq_len_start,
            } => (proc_start, proc_count, effective_seq_len_start),
            proc_range::ProcRange::EarlyReturn(ptr) => return Ok(ptr),
        };
        phase_mark!("proc_range");

        // ── Phase 3: upload positions + MRoPE + slot metadata ──
        let upload_meta::MetaLayout {
            meta_base,
            slot_offset,
            pos_stream_bytes,
            use_mrope,
            needs_paged,
        } = self.prefill_b_upload_meta(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            proc_start,
            proc_count,
            effective_seq_len_start,
            &kv_cache,
            stream,
        )?;
        phase_mark!("upload_meta");

        // ── Phase 3b: paged metadata (block_table + seq_len) ──
        if needs_paged {
            self.prefill_b_upload_paged(
                seq,
                total,
                proc_start,
                proc_count,
                meta_base,
                slot_offset,
                &kv_cache,
                stream,
            )?;
        }
        phase_mark!("upload_paged");

        // Force H2D metadata copy to complete before layer forward.
        // On DGX Spark SM121, the DMA engine may not properly serialize
        // pinned H2D copy with subsequent compute on the same stream,
        // causing CUDA 700 at >9K tokens. This sync adds ~5μs overhead
        // per chunk but prevents the illegal memory access.
        self.gpu.synchronize(stream)?;
        phase_mark!("sync_meta");

        // ── Phase 4: forward through all layers ──
        self.prefill_b_forward_layers(
            seq,
            &mut kv_cache,
            chunk_start,
            chunk_len,
            is_last_chunk,
            proc_count,
            effective_seq_len_start,
            kv_write_start,
            marconi_skip,
            meta_base,
            slot_offset,
            pos_stream_bytes,
            use_mrope,
            needs_paged,
            stream,
        )?;
        phase_mark!("layers");

        // ── Phase 4b: MTP last-K cross-chunk capture ──
        // D2H the tail rows of this chunk's `hidden_states` into the per-seq
        // host ring buffer. Must run BEFORE finalize_last (which reads logits
        // off hidden_states[proc_count-1] and may clobber adjacent buffers),
        // and BEFORE the next chunk overwrites hidden_states. No-op when MTP
        // last-K prefill is disabled or no proposer is wired.
        self.mtp_lastk_capture_chunk(seq, chunk_start, chunk_len, proc_count, stream)?;

        // ── Phase 5: update sequence state incrementally ──
        // Always add chunk tokens exactly once. The early-return path for
        // fully cached non-last chunks doesn't add tokens, so this is the
        // single insertion point for all chunks that reach here.
        seq.tokens
            .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
        seq.seq_len = chunk_start + chunk_len;

        let result = if is_last_chunk {
            // ── Phase 6+7+8: final norm, lm_head, prefix-cache + snapshot save ──
            self.prefill_b_finalize_last(
                tokens,
                seq,
                &mut kv_cache,
                chunk_start,
                chunk_len,
                proc_count,
                stream,
            )
        } else {
            // ── Phase 9: intermediate Marconi checkpoint ──
            self.prefill_b_save_checkpoint(
                tokens,
                seq,
                &mut kv_cache,
                chunk_start,
                chunk_len,
                stream,
            )?;
            Ok(DevicePtr::NULL)
        };
        phase_mark!("finalize");
        if phase_on {
            let total = phase_t0.elapsed().as_secs_f64() * 1000.0;
            let mut joined: Vec<String> = Vec::with_capacity(phase_ms.len());
            let mut prev = 0.0f64;
            for (name, t) in &phase_ms {
                joined.push(format!("{name}={:.1}", t - prev));
                prev = *t;
            }
            joined.push(format!("total={total:.1}"));
            tracing::info!("PREFILL_PHASES tok={} | {}", chunk_len, joined.join(" "));
        }
        result
    }
}

#[cfg(test)]
mod snapshot_restore_tests {
    use super::{
        attention_only_prefix_skip, cached_kv_rows_in_slice, restored_prefix_skip_tokens,
        should_restore_ssm_snapshot_with_opt_in,
    };

    #[test]
    fn exact_full_prompt_restore_requires_explicit_opt_in() {
        assert!(!should_restore_ssm_snapshot_with_opt_in(64, 64, 64, false));
        assert!(should_restore_ssm_snapshot_with_opt_in(64, 64, 64, true));
        assert!(should_restore_ssm_snapshot_with_opt_in(32, 64, 64, false));
        assert!(should_restore_ssm_snapshot_with_opt_in(64, 64, 80, false));
    }

    #[test]
    fn intermediate_ssm_checkpoint_limits_full_prompt_skip() {
        assert_eq!(restored_prefix_skip_tokens(true, 32, 64), 32);
        assert_eq!(restored_prefix_skip_tokens(false, 0, 64), 64);
    }

    #[test]
    fn recurrent_replay_does_not_overwrite_deeper_cached_kv() {
        assert_eq!(cached_kv_rows_in_slice(64, 32, 48), 32);
        assert_eq!(cached_kv_rows_in_slice(64, 0, 64), 64);
        assert_eq!(cached_kv_rows_in_slice(64, 64, 16), 0);
    }

    #[test]
    fn exact_snapshot_bypass_cannot_take_attention_only_skip() {
        assert!(!attention_only_prefix_skip(64, false, true));
        assert!(attention_only_prefix_skip(64, false, false));
        assert!(!attention_only_prefix_skip(64, true, false));
    }
}
