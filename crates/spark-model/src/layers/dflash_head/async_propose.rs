// SPDX-License-Identifier: AGPL-3.0-only

//! ATLAS_DFLASH_ASYNC=1 — async propose ‖ step-tail overlap.
//!
//! Port of the task-#20 async-propose module from the GB10 fixes lineage,
//! trimmed to the features this tree ships (no markov / retrieval / ddtree /
//! FUSED — none exist here).
//!
//! `propose(N+1)` is data-dependent on `verify(N)` (the drafter conditions on
//! hidden states captured during verify), so the drafter cannot overlap the
//! NEXT verify. What it can overlap is everything the scheduler thread does
//! after the drafter kernels are enqueued: today `propose_drafts` blocks in
//! `forward_block`'s final `event_synchronize` + drafts D2H, so the step-tail
//! CPU work (STEP_TIMING log, scheduler loop, HTTP stream flush, next step's
//! setup) all runs AFTER the drafter GPU work has drained. This module
//! enqueues the drafter forward on a dedicated CUDA stream (ordered after the
//! default stream via a recorded event) and returns a placeholder chain
//! immediately; the REAL drafts are collected (event sync + pinned-buffer
//! read) at the top of the NEXT scheduler step, right before the verify needs
//! the token values.
//!
//! Losslessness: byte-identical off (default). On: the drafter runs the SAME
//! kernels on the SAME inputs (the launch is ordered after all default-stream
//! writes it reads), so the drafts are bit-identical to the sync path; and
//! drafts only ever PROPOSE — the verify oracle commits solely the target's
//! greedy token. A lost/discarded async propose degrades to an empty draft
//! chain → bootstrap decode (slower, never wrong). Placeholder values are the
//! MASK token, so if a bug ever let one reach the verifier every row would
//! mismatch and the step degrades to bonus-only.
//!
//! Shared-scratch discipline: the head owns ONE drafter scratch set, so at
//! most ONE async propose may be in flight (`head.async_inflight`). Every
//! consumer resolves the handle first: `collect_async_drafts_impl` (normal
//! path), any subsequent sync `propose_drafts` (stale handle), and
//! `free_state` (use-after-free guard for the per-seq ctx buffers).

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::{BlockDiffusionDraftHead, DflashProposerState};

/// Master gate: `ATLAS_DFLASH_ASYNC=1` (default OFF). Cached.
pub fn dflash_async_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_DFLASH_ASYNC").ok().as_deref() == Some("1"))
}

/// Env eligibility, cached. Every listed env either injects host sync/D2H
/// into `forward_block` (debug dumps / traces), logs the drafts at propose
/// return (DIAG — would log the placeholder), or post-processes the returned
/// drafts on the host (confidence tau) — all need real drafts at propose()
/// return, which defeats the deferred collect.
pub fn async_env_eligible() -> bool {
    static ELIGIBLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ELIGIBLE.get_or_init(|| {
        let unset = |k: &str| std::env::var(k).is_err();
        let ok = unset("ATLAS_DFLASH_DEBUG_DUMP_FULL")
            && unset("ATLAS_DFLASH_DEBUG_DUMP")
            && unset("ATLAS_DFLASH_DEBUG_DUMP_ALL_LAYERS")
            && unset("ATLAS_DFLASH_BLOCK_DUMP")
            && unset("ATLAS_DFLASH_VERIFY_TRACE")
            && unset("ATLAS_DFLASH_LOG_DRAFTS")
            && unset("ATLAS_DFLASH_DIAG")
            && unset("ATLAS_DFLASH_OPTION_B_DIAG")
            && unset("ATLAS_DFLASH_PRECOMPUTE_DUMP")
            && unset("ATLAS_DFLASH_CTX_PARITY_DUMP")
            && unset("ATLAS_MTP_DRAFT_CONF")
            && unset("ATLAS_DFLASH_DEBUG_FORCE_PATTERN")
            && unset("ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN")
            && unset("ATLAS_DFLASH_DEBUG_CTX_OFF")
            && unset("ATLAS_DFLASH_DEBUG_CTX_USED");
        if !ok {
            tracing::info!(
                "DFLASH_ASYNC: a debug/host-interactive env is set — sync propose path"
            );
        }
        ok
    })
}

/// One in-flight async propose (at most one — single scratch buffer set).
#[derive(Debug, Clone)]
pub struct AsyncInflight {
    /// Identity of the owning `DflashProposerState` (stable Box address for
    /// the sequence's lifetime) so a different sequence never consumes
    /// another's drafts.
    pub owner: usize,
    /// Stream the drafter kernels were enqueued on.
    pub stream: u64,
}

/// Stable identity for a proposer state (Box contents don't move).
pub fn dstate_id(dstate: &DflashProposerState) -> usize {
    dstate as *const DflashProposerState as usize
}

/// Placeholder chain: same length the sync path would return (so scheduler
/// routing is identical), all MASK tokens (lossless if ever verified).
pub fn placeholder_drafts(len: usize, mask_id: u32) -> Vec<u32> {
    vec![mask_id; len]
}

static ASYNC_FIRES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ASYNC_COLLECTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ASYNC_DISCARDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn log_telemetry() {
    let f = ASYNC_FIRES.load(std::sync::atomic::Ordering::Relaxed);
    if f > 0 && f % 256 == 0 {
        tracing::info!(
            "DFLASH_ASYNC telemetry: fires={f} collects={} discards={}",
            ASYNC_COLLECTS.load(std::sync::atomic::Ordering::Relaxed),
            ASYNC_DISCARDS.load(std::sync::atomic::Ordering::Relaxed),
        );
    }
}

impl BlockDiffusionDraftHead {
    /// Lazily-created dedicated propose stream. `0` = creation failed →
    /// async permanently disabled for this process.
    fn propose_stream_lazy(&self, gpu: &dyn GpuBackend) -> u64 {
        *self.async_propose_stream.get_or_init(|| {
            match (gpu.create_stream(), gpu.create_event()) {
                (Ok(s), Ok(e)) if s != 0 && e != 0 => {
                    self.async_order_event
                        .store(e, std::sync::atomic::Ordering::Release);
                    tracing::info!("DFLASH_ASYNC: created propose stream {s:#x} + order event");
                    s
                }
                (s, e) => {
                    tracing::warn!(
                        "DFLASH_ASYNC: stream/event creation failed (stream={s:?} event={e:?}) — \
                         async propose disabled"
                    );
                    0
                }
            }
        })
    }

    /// Resolve (sync + discard) any in-flight async propose. Called before
    /// any new launch touches the shared scratch, and from `free_state`
    /// before the per-seq ctx buffers the in-flight kernels read are freed.
    /// Also clears an orphaned placeholder flag on `dstate` when provided.
    pub(crate) fn resolve_async_inflight_impl(
        &self,
        gpu: &dyn GpuBackend,
        dstate: Option<&mut DflashProposerState>,
    ) -> Result<()> {
        let taken = self.async_inflight.lock().take();
        if let Some(inflight) = taken {
            gpu.synchronize(inflight.stream)?;
            ASYNC_DISCARDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::debug!(
                "DFLASH_ASYNC: resolved stale in-flight propose (owner={:#x})",
                inflight.owner
            );
        }
        if let Some(ds) = dstate {
            ds.async_placeholder = false;
        }
        Ok(())
    }

    /// Length the sync path would return for this propose: γ minus the row-0
    /// drop (diffusion drafters), clamped by ATLAS_DFLASH_DRAFT_CAP.
    fn sync_path_draft_len(&self) -> usize {
        let raw = self.gamma;
        let dropped = if self.mask_token_id != 0 && raw > 1 { raw - 1 } else { raw };
        let cap: usize = std::env::var("ATLAS_DFLASH_DRAFT_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(self.gamma);
        dropped.min(cap)
    }

    /// Try to launch the drafter forward asynchronously. `Ok(Some(chain))` =
    /// async fired, chain is the placeholder; `Ok(None)` = caller must run
    /// the sync path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_launch_async_propose(
        &self,
        last_token: u32,
        position: usize,
        ctx: &crate::layer::ForwardContext,
        default_stream: u64,
        ctx_buffer: Option<(spark_runtime::gpu::DevicePtr, usize)>,
        option_b: Option<(spark_runtime::gpu::DevicePtr, u32)>,
        grammar_masked: bool,
        dstate: &mut DflashProposerState,
    ) -> Result<Option<Vec<u32>>> {
        if !dflash_async_enabled() || !async_env_eligible() || grammar_masked {
            return Ok(None);
        }
        let gpu = ctx.gpu;
        let pstream = self.propose_stream_lazy(gpu);
        if pstream == 0 {
            return Ok(None);
        }
        // Shared-scratch discipline: at most one in-flight propose.
        self.resolve_async_inflight_impl(gpu, None)?;

        // GPU-side ordering: everything the drafter reads (ctx-append D2Ds,
        // verify captures, the ctx precompute enqueued earlier in this very
        // propose call) is on the default stream — record an event there and
        // make the propose stream wait on it.
        let ev = self
            .async_order_event
            .load(std::sync::atomic::Ordering::Acquire);
        if ev == 0 {
            return Ok(None);
        }
        gpu.record_event(ev, default_stream)?;
        gpu.stream_wait_event(pstream, ev)?;

        // Enqueue the drafter forward with the final event-sync + host parse
        // deferred. On a mid-enqueue failure, drain the propose stream BEFORE
        // returning: no inflight handle exists yet, and the caller's sync
        // fallback would otherwise rewrite the shared scratch while the
        // partially-enqueued kernels are still running.
        let t_enqueue = std::time::Instant::now();
        if let Err(e) =
            self.forward_block(last_token, position, ctx, pstream, ctx_buffer, option_b, true)
        {
            let _ = gpu.synchronize(pstream);
            return Err(e);
        }
        // One-shot: prove the launch fires and that the enqueue is actually
        // non-blocking (a hidden sync inside forward_block would show up as
        // enqueue_ms ≈ the full drafter time and negate the overlap).
        static FIRED_DBG: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !FIRED_DBG.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!(
                "DFLASH_ASYNC: first async propose launched (enqueue={:.2}ms)",
                t_enqueue.elapsed().as_secs_f64() * 1e3,
            );
        }

        *self.async_inflight.lock() = Some(AsyncInflight {
            owner: dstate_id(dstate),
            stream: pstream,
        });
        let len = self.sync_path_draft_len();
        dstate.async_placeholder = true;
        dstate.last_num_drafted = len;
        ASYNC_FIRES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        log_telemetry();
        Ok(Some(placeholder_drafts(len, self.mask_token_id)))
    }

    /// Collect the drafts of a previously-launched async propose for this
    /// sequence: event-sync the pinned D2H, parse, apply the same row-0 drop
    /// + DRAFT_CAP trim the sync path applies.
    ///
    /// Returns `Ok(None)` (nothing pending — sync path), `Ok(Some(drafts))`
    /// (real drafts; caller replaces the placeholder), or `Ok(Some(vec![]))`
    /// (orphaned placeholder → caller bootstraps; lossless).
    pub(crate) fn collect_async_drafts_impl(
        &self,
        gpu: &dyn GpuBackend,
        dstate: &mut DflashProposerState,
    ) -> Result<Option<Vec<u32>>> {
        let mut guard = self.async_inflight.lock();
        let matches = guard
            .as_ref()
            .is_some_and(|inf| inf.owner == dstate_id(dstate));
        if !matches {
            // A foreign in-flight handle (owner died / rerouted): resolve it
            // so shared scratch is quiescent before this sequence's work.
            if let Some(stale) = guard.take() {
                drop(guard);
                gpu.synchronize(stale.stream)?;
                ASYNC_DISCARDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if dstate.async_placeholder {
                dstate.async_placeholder = false;
                tracing::warn!("DFLASH_ASYNC: orphaned placeholder — degrading to bootstrap");
                return Ok(Some(Vec::new()));
            }
            return Ok(None);
        }
        guard.take();
        drop(guard);

        // The D2H into the pinned buffer + `draft_tokens_event` record are the
        // last ops forward_block enqueued on the propose stream; waiting on
        // the event covers the whole drafter forward.
        gpu.event_synchronize(self.scratch.draft_tokens_event)?;
        let pinned_ptr = self
            .scratch
            .draft_tokens_host_pinned
            .load(std::sync::atomic::Ordering::Relaxed);
        if pinned_ptr.is_null() {
            dstate.async_placeholder = false;
            return Ok(Some(Vec::new()));
        }
        // SAFETY: pinned buffer is γ×4 bytes, allocated for the head's
        // lifetime; the event sync above orders the DMA before this read.
        let host_buf: &[u8] =
            unsafe { std::slice::from_raw_parts(pinned_ptr, self.gamma * 4) };
        let raw: Vec<u32> = host_buf
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Same post-processing as the sync tail of propose_drafts.
        let row0_dropped = self.mask_token_id != 0 && raw.len() > 1;
        let drafts: Vec<u32> = if row0_dropped { raw[1..].to_vec() } else { raw };
        let cap: usize = std::env::var("ATLAS_DFLASH_DRAFT_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(self.gamma);
        let drafts: Vec<u32> = drafts.into_iter().take(cap).collect();
        dstate.last_num_drafted = drafts.len();
        dstate.async_placeholder = false;
        ASYNC_COLLECTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        static COLLECT_DBG: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !COLLECT_DBG.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!("DFLASH_ASYNC: first collect ok ({} drafts)", drafts.len());
        }
        Ok(Some(drafts))
    }
}
