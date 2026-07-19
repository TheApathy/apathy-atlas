// SPDX-License-Identifier: AGPL-3.0-only

//! ATLAS_DFLASH_ASYNC=1 — async propose‖commit-tail overlap (task #20).
//!
//! ## What overlaps (and what cannot)
//!
//! The DFlash drafter conditions on the TARGET's hidden states captured
//! during verify at layers `[1,16,31,46,61]` of the 64-layer stack — the
//! last capture lands ~95% through the verify forward. So `propose(N+1)`
//! is **data-dependent on `verify(N)` completing**: the "propose N+1 while
//! verify N runs" full overlap the naive pipeline picture suggests is
//! architecturally impossible without severing the drafter's ctx-hidden
//! conditioning (measured elsewhere to collapse acceptance).
//!
//! What CAN overlap is everything the CPU does AFTER the drafter kernels
//! are enqueued: today `run_mtp_propose_multi` blocks the scheduler thread
//! in `forward_block`'s final `synchronize` + drafts D2H, so the step-tail
//! CPU work (STEP_TIMING log, tree-payload drain, scheduler loop, HTTP
//! stream flush, next step's phase-1 verify setup) all runs AFTER the
//! drafter GPU work has drained. This module inverts that: the drafter
//! forward is ENQUEUED on a dedicated CUDA stream (ordered after the
//! default stream via a recorded event) and `propose` returns immediately
//! with a placeholder chain; the REAL drafts are collected (stream sync +
//! γ×4-byte D2H) at the top of the NEXT scheduler step, right before the
//! verify needs the token values. The SSM commit tail (secondary stream)
//! and the step-tail CPU work now run concurrently with the drafter.
//!
//! ## Losslessness
//!
//! Byte-identical off (default). On: the drafter runs the SAME kernels on
//! the SAME inputs (the launch is ordered after all default-stream writes
//! it reads — ctx-append D2Ds, verify captures), so the proposed drafts are
//! bit-identical to the sync path; and drafts only ever PROPOSE — the
//! verify oracle commits solely the target's greedy token. A lost /
//! discarded async propose degrades to an empty draft chain → bootstrap
//! decode (slower, never wrong).
//!
//! ## Safety invariants (shared-scratch discipline)
//!
//! The drafter head owns ONE scratch buffer set, so at most ONE async
//! propose may be in flight; the handle lives on the head
//! (`async_inflight`). Every consumer of the scratch resolves the handle
//! first:
//!   * `collect_async_drafts` (scheduler, top of next step) — the normal
//!     path: sync + D2H → real drafts.
//!   * `propose_drafts` (any subsequent launch) — resolves a stale handle
//!     before touching scratch (sequence died / route changed).
//!   * `free_sequence` — resolves before the per-seq ctx buffers the
//!     in-flight kernels read are freed (use-after-free guard).

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::{BlockDiffusionDraftHead, DflashProposerState};

/// Master gate: `ATLAS_DFLASH_ASYNC=1` (default OFF). Cached.
pub fn dflash_async_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_DFLASH_ASYNC").ok().as_deref() == Some("1"))
}

/// ATLAS_DFLASH_FUSED=1: record the propose-ordering event pre-commit so the
/// drafter runs in parallel with SSM commit + KV reshape (~10ms overlap).
/// Requires ATLAS_DFLASH_ASYNC=1. Cached.
pub fn dflash_fused_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_DFLASH_FUSED").ok().as_deref() == Some("1"))
}

/// Env-derived disqualifiers, computed once. Every feature here either
/// post-processes drafts on the host inside/after `forward_block` (markov,
/// denoise>1, margin gate, adaptive gamma, cfg-jf splice, debug dumps,
/// kernel profile), builds a tree payload from host drafts (branch /
/// caterpillar / free-slots / ddtree / portfolio), or pre-empts the neural
/// drafter with host-side drafts that early-return from `propose_drafts`
/// before the async bookkeeping (retrieval/SAM, PLD, recycle) — all need the
/// drafts on the host at propose() return, which defeats the deferred
/// collect. The async path only engages on the plain flat-chain neural
/// propose (the production default config).
#[derive(Debug, Clone, Copy)]
pub struct AsyncEnvEligibility {
    pub denoise_steps: usize,
    pub margin_gate_on: bool,
    pub adaptive_gamma: bool,
    pub tree_method: bool,
    pub portfolio: bool,
    pub cfg_jf: bool,
    pub kprofile: bool,
    pub debug_dump: bool,
    /// Retrieval drafting (`ATLAS_DFLASH_RETRIEVAL=1` / `ATLAS_DFLASH_SAM=1`):
    /// pre-empts the neural drafter with host-side drafts and early-returns
    /// from `propose_drafts` BEFORE the async launch/collect bookkeeping, so a
    /// prior in-flight async propose is never resolved — violating the
    /// single-in-flight scratch invariant (measured: silent serve death at
    /// drafter init, variants.md 2026-07-18).
    pub retrieval: bool,
    /// `ATLAS_DFLASH_PLD=1`: same host-draft early-return class as retrieval.
    pub pld: bool,
    /// `ATLAS_DFLASH_RECYCLE=1`: same host-draft early-return class.
    pub recycle: bool,
}

impl AsyncEnvEligibility {
    pub fn from_env() -> Self {
        let flag = |k: &str| std::env::var(k).ok().as_deref() == Some("1");
        Self {
            denoise_steps: std::env::var("ATLAS_DFLASH_DENOISE_STEPS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1),
            margin_gate_on: std::env::var("ATLAS_DFLASH_MARGIN_GATE")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.0)
                > 0.0,
            adaptive_gamma: flag("ATLAS_DFLASH_ADAPTIVE_GAMMA"),
            tree_method: flag("ATLAS_DFLASH_BRANCH")
                || flag("ATLAS_DFLASH_CATERPILLAR")
                || std::env::var("ATLAS_DFLASH_METHOD").ok().as_deref() == Some("ddtree")
                || std::env::var("ATLAS_DFLASH_FREE_SLOTS")
                    .ok()
                    .and_then(|s| s.trim().parse::<usize>().ok())
                    .is_some_and(|n| n >= 1),
            portfolio: flag("ATLAS_DFLASH_PORTFOLIO"),
            cfg_jf: flag("ATLAS_DFLASH_CFG_JF"),
            kprofile: flag("ATLAS_DFLASH_KERNEL_PROFILE"),
            debug_dump: flag("ATLAS_DFLASH_DEBUG_DUMP")
                || flag("ATLAS_DFLASH_DEBUG_DUMP_FULL")
                || flag("ATLAS_DFLASH_DEBUG_DUMP_ALL_LAYERS"),
            retrieval: flag("ATLAS_DFLASH_RETRIEVAL") || flag("ATLAS_DFLASH_SAM"),
            pld: flag("ATLAS_DFLASH_PLD"),
            recycle: flag("ATLAS_DFLASH_RECYCLE"),
        }
    }

    /// Cached process-wide instance (all inputs are process-lifetime envs).
    pub fn cached() -> &'static Self {
        static CACHED: std::sync::OnceLock<AsyncEnvEligibility> = std::sync::OnceLock::new();
        CACHED.get_or_init(Self::from_env)
    }
}

/// Pure launch-eligibility decision (unit-tested, GPU-free).
///
/// `has_markov`: the checkpoint ships a Markov head (host-side sequential
/// re-sampling of the block after D2H — incompatible with deferred collect).
/// `grammar_masked`: caller passed a grammar bitmask (conservative: stay on
/// the sync path for constrained sequences).
pub fn async_launch_eligible(
    env: &AsyncEnvEligibility,
    has_markov: bool,
    grammar_masked: bool,
) -> bool {
    !has_markov
        && !grammar_masked
        && env.denoise_steps <= 1
        && !env.margin_gate_on
        && !env.adaptive_gamma
        && !env.tree_method
        && !env.portfolio
        && !env.cfg_jf
        && !env.kprofile
        && !env.debug_dump
        && !env.retrieval
        && !env.pld
        && !env.recycle
}

/// Placeholder chain returned by an async launch. Length == γ_eff so the
/// scheduler routes exactly as the sync path would (`len() >= 4` → K=γ+1
/// DFlash verify); values are the MASK token so that if a bug ever lets a
/// placeholder reach the verifier, every row mismatches the target argmax
/// and the step degrades to bonus-only — lossless by construction.
pub fn placeholder_drafts(gamma_eff: usize, mask_id: u32) -> Vec<u32> {
    vec![mask_id; gamma_eff]
}

/// One in-flight async propose (at most one — the head has a single scratch
/// buffer set).
#[derive(Debug, Clone, Copy)]
pub struct AsyncInflight {
    /// Identity of the owning `DflashProposerState` (stable Box address for
    /// the sequence's lifetime). Matched at collect so a different sequence
    /// never consumes another's drafts.
    pub owner: usize,
    /// Number of drafts to D2H from `scratch.draft_tokens_dev` at collect.
    pub gamma_eff: usize,
    /// Stream the drafter kernels were enqueued on.
    pub stream: u64,
}

/// Stable identity for a proposer state (Box contents don't move).
pub fn dstate_id(dstate: &DflashProposerState) -> usize {
    dstate as *const DflashProposerState as usize
}

// ── Telemetry (fire / collect / discard counters, logged periodically) ──
static ASYNC_FIRES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ASYNC_COLLECTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ASYNC_DISCARDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn log_telemetry() {
    let f = ASYNC_FIRES.load(std::sync::atomic::Ordering::Relaxed);
    if f > 0 && f.is_multiple_of(256) {
        tracing::info!(
            "DFLASH_ASYNC telemetry: fires={f} collects={} discards={}",
            ASYNC_COLLECTS.load(std::sync::atomic::Ordering::Relaxed),
            ASYNC_DISCARDS.load(std::sync::atomic::Ordering::Relaxed),
        );
    }
}

impl BlockDiffusionDraftHead {
    /// Lazily-created dedicated propose stream (non-blocking). `0` means
    /// creation failed → async permanently disabled for this process.
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
    /// ATLAS_DFLASH_FUSED=1: record the propose-ordering CUDA event immediately
    /// after verify returns, BEFORE commit kernels are enqueued on the default
    /// stream. The drafter reads only `dflash_hidden_save` (populated by verify)
    /// and `ctx_hidden_acc` (per-sequence, not written by commit). Commit writes
    /// h_state and KV cache — disjoint from everything the drafter touches.
    /// Recording the event here lets the propose stream start while commit
    /// (~10ms SSM h_state copy + KV reshape) is still running on the default
    /// stream, instead of waiting for it.
    ///
    /// `try_launch_async_propose` detects the armed flag and skips re-recording.
    /// No-op on ASYNC/FUSED flag off, or stream not yet created (falls back to
    /// record-at-launch on first step, fused on all subsequent ones).
    pub(crate) fn arm_propose_overlap(
        &self,
        gpu: &dyn GpuBackend,
        default_stream: u64,
    ) -> Result<()> {
        if !dflash_async_enabled() || !dflash_fused_enabled() {
            return Ok(());
        }
        let pstream = self.propose_stream_lazy(gpu);
        if pstream == 0 {
            return Ok(());
        }
        let ev = self
            .async_order_event
            .load(std::sync::atomic::Ordering::Acquire);
        if ev == 0 {
            return Ok(());
        }
        gpu.record_event(ev, default_stream)?;
        self.fused_event_armed
            .store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// any new launch touches the shared scratch, and from `free_sequence`
    /// before the per-seq ctx buffers are freed. Also clears an orphaned
    /// placeholder flag on `dstate` when provided.
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

    /// Try to launch the drafter forward asynchronously. Returns
    /// `Ok(Some(placeholder))` when the async path fired (drafts collected
    /// next step), `Ok(None)` when the caller must run the sync path.
    ///
    /// Ordering: records an event on `default_stream` (all prior writes the
    /// drafter reads — ctx-append D2Ds, verify captures — are enqueued
    /// there) and makes the propose stream wait on it, then enqueues the
    /// whole `forward_block` on the propose stream with the final
    /// synchronize + drafts D2H deferred.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_launch_async_propose(
        &self,
        last_token: u32,
        position: usize,
        gamma_eff: usize,
        ctx: &crate::layer::ForwardContext,
        default_stream: u64,
        grammar_masked: bool,
        dstate: &mut DflashProposerState,
    ) -> Result<Option<Vec<u32>>> {
        if !dflash_async_enabled() {
            return Ok(None);
        }
        let env = AsyncEnvEligibility::cached();
        if !async_launch_eligible(env, self.markov.is_some(), grammar_masked) {
            static INELIGIBLE_DBG: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !INELIGIBLE_DBG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::info!(
                    "DFLASH_ASYNC: launch ineligible (env={env:?} markov={} grammar_masked={}) — \
                     sync propose path",
                    self.markov.is_some(),
                    grammar_masked,
                );
            }
            return Ok(None);
        }
        let gpu = ctx.gpu;
        let pstream = self.propose_stream_lazy(gpu);
        if pstream == 0 {
            return Ok(None);
        }
        // Shared-scratch discipline: at most one in-flight propose.
        self.resolve_async_inflight_impl(gpu, None)?;

        // GPU-side ordering: propose stream waits for the ordering event.
        //
        // ATLAS_DFLASH_FUSED: `arm_propose_overlap` already recorded the event
        // immediately after verify returned (before commit was enqueued), so the
        // propose stream only waits for verify — commit and drafter run in
        // parallel. Consume the armed flag; if not set, record now (current
        // behavior: propose waits for commit too).
        let ev = self
            .async_order_event
            .load(std::sync::atomic::Ordering::Acquire);
        let already_armed = self
            .fused_event_armed
            .swap(false, std::sync::atomic::Ordering::AcqRel);
        if !already_armed {
            gpu.record_event(ev, default_stream)?;
        }
        gpu.stream_wait_event(pstream, ev)?;

        // Enqueue the drafter forward with the final sync + D2H deferred.
        // On a mid-enqueue failure, drain the propose stream BEFORE
        // returning: no inflight handle exists yet, and the caller's sync
        // fallback would otherwise rewrite the shared scratch while the
        // partially-enqueued kernels are still running.
        if let Err(e) = self.forward_block(last_token, position, ctx, pstream, dstate, true) {
            let _ = gpu.synchronize(pstream);
            return Err(e);
        }

        *self.async_inflight.lock() = Some(AsyncInflight {
            owner: dstate_id(dstate),
            gamma_eff,
            stream: pstream,
        });
        dstate.async_placeholder = true;
        dstate.last_num_drafted = gamma_eff;
        dstate.first_propose_done = true;
        dstate.pending_tree_payload = None;
        ASYNC_FIRES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        log_telemetry();
        Ok(Some(placeholder_drafts(gamma_eff, self.mask_token_id)))
    }

    /// Collect the drafts of a previously-launched async propose for this
    /// sequence: sync the propose stream, D2H the γ_eff argmax tokens.
    ///
    /// Returns:
    ///   * `Ok(None)` — nothing pending for this sequence (normal sync path).
    ///   * `Ok(Some(drafts))` — real drafts; caller replaces the placeholder.
    ///   * `Ok(Some(vec![]))` — the placeholder was orphaned (handle lost to
    ///     a stale resolve); caller falls back to bootstrap decode. Lossless.
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
            // here so shared scratch is quiescent before this sequence's
            // verify/decode work touches the GPU.
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
        let inflight = guard.take().expect("checked above");
        drop(guard);

        gpu.synchronize(inflight.stream)?;
        let mut host_buf = vec![0u8; inflight.gamma_eff * 4];
        gpu.copy_d2h(self.scratch.draft_tokens_dev, &mut host_buf)?;
        let drafts: Vec<u32> = host_buf
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        dstate.async_placeholder = false;
        ASYNC_COLLECTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Some(drafts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_clear() -> AsyncEnvEligibility {
        AsyncEnvEligibility {
            denoise_steps: 1,
            margin_gate_on: false,
            adaptive_gamma: false,
            tree_method: false,
            portfolio: false,
            cfg_jf: false,
            kprofile: false,
            debug_dump: false,
            retrieval: false,
            pld: false,
            recycle: false,
        }
    }

    #[test]
    fn eligible_on_plain_flat_config() {
        assert!(async_launch_eligible(&all_clear(), false, false));
    }

    #[test]
    fn markov_head_disqualifies() {
        assert!(!async_launch_eligible(&all_clear(), true, false));
    }

    #[test]
    fn grammar_mask_disqualifies() {
        assert!(!async_launch_eligible(&all_clear(), false, true));
    }

    #[test]
    fn denoise_multi_pass_disqualifies() {
        let env = AsyncEnvEligibility {
            denoise_steps: 2,
            ..all_clear()
        };
        assert!(!async_launch_eligible(&env, false, false));
    }

    #[test]
    fn margin_gate_disqualifies() {
        let env = AsyncEnvEligibility {
            margin_gate_on: true,
            ..all_clear()
        };
        assert!(!async_launch_eligible(&env, false, false));
    }

    #[test]
    fn host_draft_preempt_paths_disqualify() {
        // retrieval/SAM, PLD, and recycle all pre-empt the neural drafter
        // with host drafts and early-return before async collect — each must
        // force the sync path (SAM+ASYNC init death, variants.md 2026-07-18).
        for f in [
            AsyncEnvEligibility {
                retrieval: true,
                ..all_clear()
            },
            AsyncEnvEligibility {
                pld: true,
                ..all_clear()
            },
            AsyncEnvEligibility {
                recycle: true,
                ..all_clear()
            },
        ] {
            assert!(!async_launch_eligible(&f, false, false));
        }
    }

    #[test]
    fn tree_methods_disqualify() {
        for f in [
            AsyncEnvEligibility {
                tree_method: true,
                ..all_clear()
            },
            AsyncEnvEligibility {
                portfolio: true,
                ..all_clear()
            },
        ] {
            assert!(!async_launch_eligible(&f, false, false));
        }
    }

    #[test]
    fn host_postprocessing_flags_disqualify() {
        for f in [
            AsyncEnvEligibility {
                adaptive_gamma: true,
                ..all_clear()
            },
            AsyncEnvEligibility {
                cfg_jf: true,
                ..all_clear()
            },
            AsyncEnvEligibility {
                kprofile: true,
                ..all_clear()
            },
            AsyncEnvEligibility {
                debug_dump: true,
                ..all_clear()
            },
        ] {
            assert!(!async_launch_eligible(&f, false, false));
        }
    }

    #[test]
    fn placeholder_is_all_mask_and_gamma_wide() {
        let p = placeholder_drafts(16, 42);
        assert_eq!(p.len(), 16);
        assert!(p.iter().all(|&t| t == 42));
        // Degenerate γ still yields a routable chain (scheduler needs >= 4
        // for the DFlash verify path; smaller is legal and falls to K=2/3/4).
        assert_eq!(placeholder_drafts(1, 7), vec![7]);
        assert!(placeholder_drafts(0, 7).is_empty());
    }
}
