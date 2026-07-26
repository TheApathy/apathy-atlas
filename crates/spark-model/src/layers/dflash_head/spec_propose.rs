// SPDX-License-Identifier: AGPL-3.0-only

//! ATLAS_DFLASH_SPEC_PROPOSE=1 — speculative propose ‖ verify-tail overlap.
//!
//! Bets on the FULL-ACCEPT case (38-42% of K=γ verify steps on Laguna). On a
//! full-accept step the next propose's inputs are fully determined before the
//! verify's D2H completes:
//!   * drafter ctx conditioning = the K verify rows' tap-layer hiddens,
//!     captured INSIDE the verify graph into `dflash_hidden_save`
//!     (`try_dflash_capture_all`, available after the last tap layer), and
//!   * `last_token` = the target argmax of the LAST verify row (the bonus on
//!     full accept), sitting in the verify argmax device buffer at row K-1.
//!
//! So right after the verify graph is ENQUEUED (before the blocking D2H), we
//! enqueue on the dedicated propose stream — ordered after the graph via the
//! recorded completion event —:
//!   (a) the optimistic full-accept ctx append (ALL K capture rows → the ctx
//!       accumulator; that IS exactly what `commit_ctx` / the EAGLE kgamma
//!       append would do on full accept),
//!   (b) the incremental Option-B ctx precompute of the new tail, and
//!   (c) the drafter `forward_block` with `last_token` read DEVICE-side: the
//!       drafter's first op is a batched embed of a device token-id buffer,
//!       so a 4-byte D2D from the verify argmax slot into that buffer (after
//!       the host upload, before the embed) IS the indirect embed — no new
//!       kernel needed.
//!
//! After the normal D2H + accept walk the scheduler decides:
//!   * realized full accept (raw-argmax basis, no fork/tree/grammar) → ADOPT:
//!     skip the sync ctx append (already done) and the sync propose; install
//!     a placeholder chain; the real drafts are event-collected at the top of
//!     the next step through the existing `collect_async_drafts` hook.
//!   * anything else → DISCARD: host-sync the propose stream, roll the ctx
//!     watermark back (host-side only — see below), and run the normal sync
//!     propose.
//!
//! ## Losslessness
//! Adoption happens ONLY when the realized accept == full accept, in which
//! case the speculative inputs were exactly the true inputs — the drafts are
//! bit-identical to what the sync propose would have produced. And drafts
//! only ever PROPOSE; the verify oracle commits solely the target's greedy
//! tokens. A discarded launch costs GPU time, never correctness. Default
//! OFF; byte-identical off.
//!
//! ## Ctx rollback mechanics (why it needs NO device work)
//! The optimistic append advances host-side watermarks only: `ctx_len`,
//! `ctx_positions`, `ctx_committed`, `ctx_count_drafter`,
//! `skip_next_decode_append`. The device writes land in ctx-accumulator rows
//! and drafter paged-KV slots ABOVE the saved watermark, which the real
//! commit + next propose recompute/overwrite. So rollback = restore the
//! saved host watermark and truncate `ctx_positions`. The one destructive
//! device operation — the `commit_ctx` watermark SLIDE (drop-oldest D2D
//! compaction) — is unrecoverable, so the launch REFUSES to fire when the
//! append would overflow `max_ctx_len` (slide territory).
//!
//! ## Guards
//! * Single drafter scratch set → reuses `head.async_inflight` (at most ONE
//!   in-flight, `spec: true` entries). Every other propose path resolves it
//!   first (propose entry, free_state, collect).
//! * Mutually exclusive with ATLAS_DFLASH_ASYNC (spec disables itself, warn).
//! * Fire requires: Option-B block table live, propose graphs captured (no
//!   graph capture on the spec stream — global capture mode would poison the
//!   concurrent verify D2H), capture-all envs (EAGLE_FIX / UNIFIED_CTX), all
//!   host-interactive / draft-source / tree / fork / trunc envs unset.
//! * Payload/top2/tree extraction is SKIPPED entirely (it requires a host
//!   sync inside propose; tree/fork are gated off for spec anyway).
//! * Adaptive-suspended sequences never reach the K=γ verify (no drafts →
//!   bootstrap path), and a suspension triggered by THIS step is caught by
//!   the scheduler's adopt gate (`is_suspended`) → discard.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::async_propose::{dstate_id, placeholder_drafts};
use super::{BlockDiffusionDraftHead, DflashProposerState};

/// Host-side watermark snapshot taken at launch; restored on discard.
#[derive(Debug, Clone, Copy)]
pub struct SpecWatermark {
    pub ctx_len: usize,
    pub ctx_committed: usize,
    pub ctx_count_drafter: usize,
    pub skip_next_decode_append: bool,
}

/// Master gate: `ATLAS_DFLASH_SPEC_PROPOSE=1` (default OFF). Cached.
/// Mutually exclusive with ATLAS_DFLASH_ASYNC — if both are set, spec
/// propose disables itself with a warning (async keeps its semantics).
pub fn dflash_spec_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        let on = std::env::var("ATLAS_DFLASH_SPEC_PROPOSE").ok().as_deref() == Some("1");
        if on && super::async_propose::dflash_async_enabled() {
            tracing::warn!(
                "DFLASH_SPEC: ATLAS_DFLASH_ASYNC is also set — the two share the single \
                 drafter scratch set and one in-flight slot; SPEC_PROPOSE disabled"
            );
            return false;
        }
        on
    })
}

/// Env eligibility, cached. Beyond the async list (debug dumps / host
/// post-processing), spec propose must also exclude:
///   * draft sources (echo/PLD/retrieval/SAM/recycle/redenoise) — on adopt we
///     bypass `propose_drafts`, so a source that would have pre-empted the
///     drafter would silently change WHICH proposal ships;
///   * fork/tree/trunc (payload construction does a host-synced top-2 D2H
///     inside propose — skipped for spec, so keep the paths off entirely);
///   * masked verify (adoption requires the raw-argmax bonus == the device
///     argmax the drafter embedded);
///   * CONTIG_ATTN (per-layer host sync + bounce copies — not enqueue-clean);
///   * MTP prefill/catchup/conf (extra host-side propose phases spec bypasses).
pub fn spec_env_eligible() -> bool {
    static ELIGIBLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ELIGIBLE.get_or_init(|| {
        let unset = |k: &str| std::env::var(k).is_err();
        let ok = super::async_propose::async_env_eligible()
            && unset("ATLAS_DFLASH_ECHO")
            && unset("ATLAS_DFLASH_PLD")
            && unset("ATLAS_DFLASH_RETRIEVAL")
            && unset("ATLAS_DFLASH_SAM")
            && unset("ATLAS_DFLASH_RECYCLE")
            && unset("ATLAS_DFLASH_REDENOISE")
            && unset("ATLAS_DFLASH_BLOCKFORK")
            && unset("ATLAS_DFLASH_TREE")
            && unset("ATLAS_DFLASH_TREE_M0")
            && unset("ATLAS_DFLASH_TRUNC_MARGIN")
            && unset("ATLAS_DFLASH_MASKED_VERIFY")
            && unset("ATLAS_DFLASH_CONTIG_ATTN")
            && unset("ATLAS_DFLASH_DEBUG_NO_DECODE_APPEND")
            && unset("ATLAS_DFLASH_DEBUG_FULL_PRECOMPUTE")
            && unset("ATLAS_DFLASH_OPTION_B_NO_CTX")
            && unset("ATLAS_MTP_DRAFTER_PREFILL")
            && unset("ATLAS_MTP_CATCHUP");
        if !ok {
            tracing::info!(
                "DFLASH_SPEC: an incompatible env is set (draft source / tree / fork / \
                 masked-verify / debug) — speculative propose stays off"
            );
        }
        ok
    })
}

/// Optional economics gate (`ATLAS_DFLASH_SPEC_STREAK=1`): only fire when the
/// PREVIOUS verify step full-accepted — accept streaks cluster, so this cuts
/// discard-path drain time at the cost of missing streak heads. Default off
/// (fire on every eligible verify).
fn spec_streak_gate() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_DFLASH_SPEC_STREAK").ok().as_deref() == Some("1"))
}

// ── Telemetry ───────────────────────────────────────────────────────────
static SPEC_FIRES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SPEC_ADOPTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) static SPEC_DISCARDS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(super) static SPEC_COLLECTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Cumulative host time spent draining the propose stream on discards (µs) —
/// the price of a lost bet; compare against adopts × propose_ms saved.
pub(super) static SPEC_DISCARD_WAIT_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(super) fn log_spec_telemetry() {
    use std::sync::atomic::Ordering::Relaxed;
    let f = SPEC_FIRES.load(Relaxed);
    if f > 0 && f % 64 == 0 {
        tracing::info!(
            "DFLASH_SPEC telemetry: fires={f} adopts={} discards={} collects={} \
             discard_wait={:.1}ms",
            SPEC_ADOPTS.load(Relaxed),
            SPEC_DISCARDS.load(Relaxed),
            SPEC_COLLECTS.load(Relaxed),
            SPEC_DISCARD_WAIT_US.load(Relaxed) as f64 / 1000.0,
        );
    }
}

/// Restore the host-side ctx watermark saved at launch. Device rows written
/// above the restored watermark are dead: the real commit + next propose
/// re-append / re-precompute exactly those rows (same source, same slots).
pub(super) fn spec_rollback(dstate: &mut DflashProposerState) {
    if let Some(w) = dstate.spec_watermark.take() {
        dstate.ctx_positions.truncate(w.ctx_len);
        dstate.ctx_len = w.ctx_len;
        dstate.ctx_committed = w.ctx_committed.min(w.ctx_len);
        dstate.ctx_count_drafter = w.ctx_count_drafter;
        dstate.skip_next_decode_append = w.skip_next_decode_append;
    }
    dstate.async_placeholder = false;
}

impl BlockDiffusionDraftHead {
    /// Try to enqueue the speculative (full-accept-bet) propose. Everything
    /// is enqueue-only on the dedicated propose stream, ordered after the
    /// verify graph via the recorded event; NO host sync on this path.
    ///
    /// `ctx_rows` = K (the verify row count; also the rows the full-accept
    /// commit appends). `base_pos` = pre-verify `seq_len` (RoPE stamp base).
    /// `device_last_token` = the verify argmax slot of row K-1 (u32).
    ///
    /// `Ok(true)` = launched (inflight `spec` entry installed); `Ok(false)` =
    /// ineligible / not ready (sync path unaffected). `Err` = a mid-enqueue
    /// failure — the stream was drained and the watermark rolled back before
    /// returning, so the sync path stays correct.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn spec_propose_launch_impl(
        &self,
        gpu: &dyn GpuBackend,
        default_stream: u64,
        device_last_token: DevicePtr,
        hidden_save: DevicePtr,
        ctx_rows: usize,
        base_pos: usize,
        dstate: &mut DflashProposerState,
        ctx: &crate::layer::ForwardContext,
    ) -> Result<bool> {
        if !dflash_spec_enabled() || !spec_env_eligible() || ctx_rows == 0 {
            return Ok(false);
        }
        // Option-B block table must be live (allocated by the first sync
        // propose) — its presence also proves ATLAS_DFLASH_OPTION_B=1.
        let Some(bt_dev) = dstate.block_table_dev else {
            return Ok(false);
        };
        // Never begin a graph capture on the spec stream: default capture
        // mode is GLOBAL and would make the concurrent verify D2H illegal.
        // Require the piecewise propose graphs to already be captured.
        {
            let g = self.propose_graphs.lock();
            let ready = matches!(*g, Some(ref v) if v.len() == self.layers.len() * 2 + 1);
            if !ready {
                return Ok(false);
            }
        }
        if self.ctx_window == 0 {
            return Ok(false);
        }
        // A watermark SLIDE (drop-oldest D2D compaction) is destructive and
        // unrecoverable on discard — refuse to fire near capacity.
        if dstate.ctx_len + ctx_rows > dstate.max_ctx_len {
            return Ok(false);
        }
        if dstate.max_ctx_count_drafter > 0
            && dstate.ctx_len + ctx_rows + self.gamma + 1 > dstate.max_ctx_count_drafter
        {
            return Ok(false);
        }
        // Optional streak gate: previous step must have full-accepted.
        if spec_streak_gate() && dstate.last_num_accepted < self.sync_path_draft_len() {
            return Ok(false);
        }
        let pstream = self.propose_stream_lazy(gpu);
        if pstream == 0 {
            return Ok(false);
        }
        // Single-inflight discipline: nothing should be pending here; if a
        // stale handle survived an error path, drain it (with rollback when
        // it was OUR spec launch) before touching the shared scratch.
        {
            let taken = self.async_inflight.lock().take();
            if let Some(inf) = taken {
                gpu.synchronize(inf.stream)?;
                if inf.spec && inf.owner == dstate_id(dstate) {
                    spec_rollback(dstate);
                } else {
                    tracing::warn!(
                        "DFLASH_SPEC: drained a stale foreign in-flight propose at fire \
                         (owner={:#x} spec={})",
                        inf.owner,
                        inf.spec,
                    );
                }
            }
        }
        // GPU-side ordering: the ctx-append D2Ds read `dflash_hidden_save`
        // and the indirect embed reads the argmax slot — both written by the
        // verify graph on the default stream. Event after the graph launch,
        // propose stream waits on it. Stream-ordering only; no capture.
        let ev = self
            .async_order_event
            .load(std::sync::atomic::Ordering::Acquire);
        if ev == 0 {
            return Ok(false);
        }
        gpu.record_event(ev, default_stream)?;
        gpu.stream_wait_event(pstream, ev)?;

        // ── Watermark snapshot, then the optimistic full-accept append ──
        dstate.spec_watermark = Some(SpecWatermark {
            ctx_len: dstate.ctx_len,
            ctx_committed: dstate.ctx_committed,
            ctx_count_drafter: dstate.ctx_count_drafter,
            skip_next_decode_append: dstate.skip_next_decode_append,
        });
        let mut enqueue = || -> Result<()> {
            let slot = dstate.ctx_slot_bytes;
            for t in 0..ctx_rows {
                let src = hidden_save.offset(t * slot);
                let dst = dstate.ctx_hidden_acc.offset(dstate.ctx_len * slot);
                gpu.copy_d2d_async(src, dst, slot, pstream)?;
                dstate.ctx_positions.push((base_pos + t) as i32);
                dstate.ctx_len += 1;
            }
            dstate.skip_next_decode_append = true;

            // ── Incremental Option-B precompute of the new tail (chunked to
            // ctx_window rows per pass — same discipline as propose.rs). All
            // copies inside route through the enqueue-only variants because
            // `pstream != default_stream`.
            let committed = dstate.ctx_committed.min(dstate.ctx_len);
            let slot_mapping = &self.scratch.slot_mapping_dev;
            let mut chunk_start = committed;
            while chunk_start < dstate.ctx_len {
                let chunk_count = (dstate.ctx_len - chunk_start).min(self.ctx_window);
                crate::layers::ops::fill_slots_from_block_table(
                    gpu,
                    self.kernels.fill_slots,
                    *slot_mapping,
                    bt_dev,
                    chunk_start as u32,
                    chunk_count as u32,
                    16,
                    pstream,
                )?;
                let slot_positions = &dstate.ctx_positions[chunk_start..chunk_start + chunk_count];
                self.precompute_ctx_kv(
                    dstate.ctx_hidden_acc,
                    chunk_start,
                    chunk_count,
                    slot_positions,
                    *slot_mapping,
                    ctx,
                    pstream,
                    true,
                )?;
                chunk_start += chunk_count;
            }
            dstate.ctx_committed = dstate.ctx_len;
            dstate.ctx_count_drafter = dstate.ctx_len;

            // ── Drafter forward, enqueue-only, last_token read device-side.
            // The host token value is a MASK sentinel: if the 4-byte indirect
            // copy ever failed to land, every draft would mismatch — the step
            // degrades to bonus-only, never corrupts (lossless).
            let position = base_pos + ctx_rows;
            let ctx_buffer_arg = if dstate.ctx_len > 0 {
                Some((dstate.ctx_hidden_acc, dstate.ctx_len))
            } else {
                None
            };
            let option_b_arg = Some((bt_dev, dstate.ctx_count_drafter as u32));
            self.forward_block(
                self.mask_token_id,
                position,
                ctx,
                pstream,
                ctx_buffer_arg,
                option_b_arg,
                true, // async_launch: leave the drafts D2H + event in flight
                None, // warm_tail: REDENOISE is env-excluded for spec
                Some(device_last_token),
            )?;
            Ok(())
        };
        if let Err(e) = enqueue() {
            // Drain the partially-enqueued work before the sync fallback
            // rewrites the shared scratch, then roll the watermark back.
            let _ = gpu.synchronize(pstream);
            spec_rollback(dstate);
            return Err(e);
        }

        *self.async_inflight.lock() = Some(super::async_propose::AsyncInflight {
            owner: dstate_id(dstate),
            stream: pstream,
            spec: true,
        });
        let fires = SPEC_FIRES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if fires == 1 {
            tracing::info!(
                "DFLASH_SPEC: first speculative propose launched (K={ctx_rows}, base_pos={base_pos})"
            );
        }
        log_spec_telemetry();
        Ok(true)
    }

    /// Is a speculative launch pending for THIS sequence?
    pub(super) fn spec_pending_impl(&self, dstate: &mut DflashProposerState) -> bool {
        self.async_inflight
            .lock()
            .as_ref()
            .is_some_and(|i| i.spec && i.owner == dstate_id(dstate))
    }

    /// Discard this sequence's speculative launch: host-sync the propose
    /// stream (the shared scratch must be quiescent before the real ctx
    /// append and the sync propose run), then restore the host watermark.
    pub(super) fn spec_discard_impl(
        &self,
        gpu: &dyn GpuBackend,
        dstate: &mut DflashProposerState,
    ) -> Result<()> {
        let taken = {
            let mut guard = self.async_inflight.lock();
            match guard.as_ref() {
                Some(i) if i.spec && i.owner == dstate_id(dstate) => guard.take(),
                _ => None,
            }
        };
        let Some(inf) = taken else {
            return Ok(());
        };
        let t0 = std::time::Instant::now();
        gpu.synchronize(inf.stream)?;
        SPEC_DISCARD_WAIT_US.fetch_add(
            t0.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        spec_rollback(dstate);
        SPEC_DISCARDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        log_spec_telemetry();
        Ok(())
    }

    /// Adopt this sequence's speculative launch after a realized full accept:
    /// commit the optimistic ctx state (drop the rollback snapshot), install
    /// the placeholder chain, and KEEP the in-flight handle — the real drafts
    /// are event-collected at the top of the next scheduler step through
    /// `collect_async_drafts` (overlapping the drafter tail with the step-tail
    /// host work instead of blocking here).
    pub(super) fn spec_adopt_impl(
        &self,
        dstate: &mut DflashProposerState,
    ) -> Result<Option<Vec<u32>>> {
        let matches = self
            .async_inflight
            .lock()
            .as_ref()
            .is_some_and(|i| i.spec && i.owner == dstate_id(dstate));
        if !matches {
            return Ok(None);
        }
        dstate.spec_watermark = None; // the optimistic append IS the commit
        dstate.async_placeholder = true;
        let len = self.sync_path_draft_len();
        dstate.last_num_drafted = len;
        dstate.first_propose_done = true;
        dstate.pending_block_fork = None;
        dstate.pending_m0_top2 = None;
        dstate.pending_tree_payload = None;
        let n = SPEC_ADOPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n == 1 {
            tracing::info!("DFLASH_SPEC: first ADOPT (placeholder len={len})");
        }
        log_spec_telemetry();
        Ok(Some(placeholder_drafts(len, self.mask_token_id)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dstate() -> DflashProposerState {
        DflashProposerState {
            block_table: Vec::new(),
            seq_len: 0,
            last_num_drafted: 0,
            prefill_done: false,
            ctx_hidden_acc: DevicePtr(0),
            ctx_len: 7,
            last_num_accepted: 0,
            skip_next_decode_append: false,
            max_ctx_len: 64,
            ctx_slot_bytes: 8,
            block_table_dev: None,
            ctx_count_drafter: 7,
            max_ctx_count_drafter: 0,
            ctx_committed: 7,
            ctx_positions: (0..7).collect(),
            async_placeholder: false,
            first_propose_done: false,
            pld_tokens: Vec::new(),
            retr_used_last: false,
            retr_misfire_streak: 0,
            retr_cooldown: 0,
            recycle_tail: Vec::new(),
            recycle_key: 0,
            recycle_valid: false,
            recycle_last_offered: false,
            echo_tail: Vec::new(),
            echo_key: 0,
            echo_valid: false,
            echo_streak: 0,
            echo_offered_last: false,
            pending_block_fork: None,
            pending_m0_top2: None,
            pending_tree_payload: None,
            spec_watermark: None,
        }
    }

    #[test]
    fn rollback_restores_watermark_and_truncates_positions() {
        let mut d = dstate();
        d.spec_watermark = Some(SpecWatermark {
            ctx_len: 7,
            ctx_committed: 7,
            ctx_count_drafter: 7,
            skip_next_decode_append: false,
        });
        // Simulate the optimistic K=5 append + precompute advance.
        for t in 0..5 {
            d.ctx_positions.push(100 + t);
            d.ctx_len += 1;
        }
        d.ctx_committed = d.ctx_len;
        d.ctx_count_drafter = d.ctx_len;
        d.skip_next_decode_append = true;
        d.async_placeholder = true;

        spec_rollback(&mut d);
        assert_eq!(d.ctx_len, 7);
        assert_eq!(d.ctx_positions.len(), 7);
        assert_eq!(d.ctx_positions.last(), Some(&6));
        assert_eq!(d.ctx_committed, 7);
        assert_eq!(d.ctx_count_drafter, 7);
        assert!(!d.skip_next_decode_append);
        assert!(!d.async_placeholder);
        assert!(d.spec_watermark.is_none());
    }

    #[test]
    fn rollback_without_snapshot_is_a_noop_on_watermarks() {
        let mut d = dstate();
        d.async_placeholder = true;
        spec_rollback(&mut d);
        assert_eq!(d.ctx_len, 7);
        assert_eq!(d.ctx_committed, 7);
        assert!(!d.async_placeholder);
    }

    #[test]
    fn rollback_clamps_committed_to_restored_len() {
        // A snapshot taken when committed lagged ctx_len must restore the
        // LAGGING value (never past the restored ctx_len).
        let mut d = dstate();
        d.ctx_committed = 5; // uncommitted tail of 2
        d.spec_watermark = Some(SpecWatermark {
            ctx_len: 7,
            ctx_committed: 5,
            ctx_count_drafter: 7,
            skip_next_decode_append: true,
        });
        d.ctx_len = 12;
        d.ctx_positions.extend(200..205);
        d.ctx_committed = 12;
        spec_rollback(&mut d);
        assert_eq!(d.ctx_len, 7);
        assert_eq!(d.ctx_committed, 5);
        assert!(d.skip_next_decode_append);
    }
}
