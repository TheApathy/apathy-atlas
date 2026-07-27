// SPDX-License-Identifier: AGPL-3.0-only

//! Throughput-arbitrated speculation gate — N-arm generalisation.
//!
//! Ported from upstream `scheduler/mtp_gate.rs` (2 arms: MTP vs serial) and
//! widened to an arbitrary arm set, because our fork can run more than two
//! speculation configurations against the same target:
//!
//!   - the external DFlash/DDTree drafter (γ-block, `BlockDiffusionDraftHead`)
//!   - the native in-checkpoint MTP head (K=2, `MtpHead`)
//!   - a γ-capped variant of the external drafter
//!   - plain serial decode (no speculation)
//!
//! ## Why throughput and not acceptance
//!
//! The gate compares DELIVERED throughput (emitted tokens / wall) measured
//! over whole step windows in each arm — never component step timings, and
//! never draft acceptance. Upstream's module header records that the previous
//! acceptance/step-time gate DISABLED MTP on a model where an always-on
//! control measured 18% FASTER end-to-end decode (webserver_ok A/B 2026-07-20:
//! Σ1028s/10-10 always-on vs Σ1846s/9-10 gated). Component walls miss
//! per-token costs outside the timed step and amortization effects.
//!
//! Acceptance is additionally the WRONG signal for an N-arm choice: it is
//! measured per arm in that arm's own units, and the arms have different
//! verify costs. A mean-accepted of 3 is winning for K=2 and losing badly for
//! γ=16, so no threshold in accept-space orders the arms correctly. Delivered
//! tok/s is the common currency and needs no per-arm calibration.
//!
//! ## Policy (bandit-style greedy with scheduled exploration)
//!
//! - Run the current arm; accumulate (tokens, wall) into a fixed-size step
//!   window; on window close update that arm's tok/s EWMA and a deviation EWMA.
//! - Switch only when some other arm's EWMA is faster by more than a noise
//!   margin (hysteresis) for [`SWITCH_DWELL_WINDOWS`] consecutive windows.
//! - Periodically probe ONE other arm (round-robin) for
//!   [`probe_windows`] windows, then arbitrate. Probe cadence is
//!   [`reprobe_tokens`] while off the primary arm and
//!   [`serial_refresh_tokens`] while on it — the same asymmetry upstream uses
//!   (cheap to re-check a suspected-better arm, expensive to keep re-checking
//!   from the arm you already believe in).
//! - A depth-regime change (factor [`REMEASURE_DEPTH_FACTOR`]) marks every
//!   baseline stale and schedules a refresh instead of wiping state.
//!
//! ## Deviation from upstream: 2-window probes
//!
//! Upstream probes for exactly one window. We default to TWO
//! ([`DEFAULT_PROBE_WINDOWS`]) because our fork has no drafter catch-up ring
//! (upstream's `save_hidden_for_catchup` / `mtp_catchup_ring` do not exist
//! here). A drafter re-entered after a stretch in another arm starts cold, so
//! the first probe window measures a pessimistically-conditioned drafter. The
//! second window measures it warm. `ATLAS_MTP_GATE_PROBE_WINDOWS=1` restores
//! upstream's behaviour.
//!
//! Both step types emit real, correct tokens, so arbitration never wastes
//! work — a probe is a measurement excursion, not a dry run.

use std::time::Duration;

/// Depth factor that marks baselines stale (economics are depth-dependent:
/// weight-bound at short context vs KV/SSM-bound at depth).
const REMEASURE_DEPTH_FACTOR: usize = 2;
/// Floor for the regime comparison (below this all contexts are "shallow").
const REMEASURE_DEPTH_FLOOR: usize = 512;
/// Steps per throughput window. 16 is upstream's value: long enough that one
/// window amortizes bootstrap/propose transients.
const WINDOW_STEPS: usize = 16;
/// Consecutive out-of-margin windows required before switching arm.
const SWITCH_DWELL_WINDOWS: usize = 2;
/// EWMA smoothing for per-arm tok/s (responds within ~3 windows).
const TPS_ALPHA: f64 = 0.3;
/// Relative noise floor for the switch margin. Half the summed deviation
/// EWMAs is added on top of this floor.
const MARGIN_REL_FLOOR: f64 = 0.05;
/// Windows per probe excursion. See the module header for why this is 2 and
/// not upstream's 1.
const DEFAULT_PROBE_WINDOWS: usize = 2;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Tokens between probes of another arm while OFF the primary arm.
/// Default matches upstream's `ATLAS_MTP_GATE_REPROBE` (256).
fn reprobe_tokens() -> usize {
    env_usize("ATLAS_MTP_GATE_REPROBE", 256)
}

/// Tokens between probes while ON the primary arm. One window per 1024
/// tokens bounds refresh overhead at well under 1%.
fn serial_refresh_tokens() -> usize {
    env_usize("ATLAS_MTP_GATE_REFRESH", 1024)
}

fn probe_windows() -> usize {
    env_usize("ATLAS_MTP_GATE_PROBE_WINDOWS", DEFAULT_PROBE_WINDOWS).max(1)
}

/// Default number of committed post-thinking content tokens whose dispatch is
/// pinned away from the throughput gate's serial arm.
const DEFAULT_ENTRY_PIN_TOKENS: u8 = 8;

/// Resolved `ATLAS_SPEC_ENTRY_PIN` value plus its provenance for the startup
/// record. `0` disables the pin; invalid values fail closed to the measured
/// default instead of silently disabling the correctness guard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EntryPinConfig {
    pub tokens: u8,
    pub source: &'static str,
}

pub(crate) fn parse_entry_pin_tokens(env: Option<&str>) -> EntryPinConfig {
    match env {
        None => EntryPinConfig {
            tokens: DEFAULT_ENTRY_PIN_TOKENS,
            source: "default",
        },
        Some(raw) => match raw.trim().parse::<u8>() {
            Ok(tokens) => EntryPinConfig {
                tokens,
                source: "env",
            },
            Err(_) => EntryPinConfig {
                tokens: DEFAULT_ENTRY_PIN_TOKENS,
                source: "invalid-env-default",
            },
        },
    }
}

/// Whether any sequence in an already spec-eligible batch is within the
/// post-thinking entry window. Requests that never enter thinking start with
/// `think_ended=true`, so their response opening is covered as well.
pub(crate) fn entry_pin_active<I>(tokens: u8, states: I) -> bool
where
    I: IntoIterator<Item = (bool, bool, u8)>,
{
    tokens > 0
        && states
            .into_iter()
            .any(|(think_ended, inside_thinking, emitted)| {
                think_ended && !inside_thinking && emitted < tokens
            })
}

/// Override only a serial gate verdict. `last_spec_num_drafts` belongs to the
/// proposer which remains selected while the serial arm runs: γ for DFlash,
/// but normally one draft for the native MTP alternate. Reusing the configured
/// DFlash width for the native arm would turn an entry pin into a different
/// experiment.
pub(crate) fn entry_pin_spec_width(
    arm: ArmKind,
    active: bool,
    last_spec_num_drafts: usize,
) -> Option<usize> {
    matches!(arm, ArmKind::Serial)
        .then_some(last_spec_num_drafts)
        .filter(|_| active)
}

/// An entry-pinned speculative step may leave drafts and an async live-state
/// restore behind while the gate itself remains on the same Serial arm. When
/// the counter reaches the window boundary there is no arm transition to do
/// the normal cleanup, so the scheduler needs an explicit pinned-to-serial
/// edge.
pub(crate) fn entry_pin_exits_to_serial(
    arm: ArmKind,
    was_pinned: bool,
    pinned_spec_width: Option<usize>,
) -> bool {
    was_pinned && pinned_spec_width.is_none() && matches!(arm, ArmKind::Serial)
}

/// Count one committed post-thinking content token. Saturation keeps an
/// overlong response permanently outside every valid (`u8`) pin window.
pub(crate) fn advance_entry_counter(current: u8) -> u8 {
    current.saturating_add(1)
}

/// Which proposer the model should have selected for a speculative arm.
///
/// Mirrors `spark_model::traits::Model::select_proposer_arm`. Arm 0 is
/// whatever the build installed as primary (the DFlash drafter when
/// `--dflash` was passed); arm 1 is the demoted native MTP head.
pub const PROPOSER_ARM_PRIMARY: u8 = 0;
pub const PROPOSER_ARM_ALT: u8 = 1;

/// What one arm of the gate actually runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmKind {
    /// A speculative step: select `proposer_arm` on the model, cap the
    /// drafter at `draft_cap` (0 = the drafter's own γ, no cap), then run
    /// `step_mtp` asking for `num_drafts` drafts (0 = the run's configured
    /// default).
    ///
    /// `num_drafts` is per-arm because the run has exactly one value for it
    /// and the arms disagree about the right one. A DFlash process normally
    /// carries its configured draft count (15 for a block-size-16 checkpoint),
    /// but the native MTP head is monotonically WORSE with more drafts on this model family —
    /// measured K=2 > K=4 on all six benchmark tasks, and K=8 falls below the
    /// no-speculation floor. Driving the MTP arm at the DFlash arm's 15 would
    /// make it lose every arbitration for a reason that has nothing to do with
    /// the head's quality.
    Spec {
        proposer_arm: u8,
        draft_cap: usize,
        num_drafts: usize,
    },
    /// A plain single-token decode step (no speculation).
    Serial,
}

/// Whether the gate-selected step can safely run while a sequence is inside
/// thinking. Native MTP uses the scheduler's row-by-row policy verifier by
/// default. DFlash remains explicitly opt-in because its wide raw verifier is
/// a different numerical path; FP32 verify logits cannot feed the current
/// host oracle. A Serial arm is always safe and must remain visible to the
/// throughput gate while thinking.
pub(crate) fn arm_allows_thinking(
    arm: ArmKind,
    primary_is_dflash: bool,
    dflash_spec_think: bool,
    policy_oracle_available: bool,
) -> bool {
    match arm {
        ArmKind::Serial => true,
        ArmKind::Spec { proposer_arm, .. } => {
            let is_dflash = primary_is_dflash && proposer_arm == PROPOSER_ARM_PRIMARY;
            policy_oracle_available && (!is_dflash || dflash_spec_think)
        }
    }
}

/// One arm: a name for logs plus what it runs.
#[derive(Debug, Clone, Copy)]
pub struct ArmSpec {
    pub name: &'static str,
    pub kind: ArmKind,
}

impl ArmSpec {
    pub const fn spec(
        name: &'static str,
        proposer_arm: u8,
        draft_cap: usize,
        num_drafts: usize,
    ) -> Self {
        Self {
            name,
            kind: ArmKind::Spec {
                proposer_arm,
                draft_cap,
                num_drafts,
            },
        }
    }
    pub const fn serial(name: &'static str) -> Self {
        Self {
            name,
            kind: ArmKind::Serial,
        }
    }
}

/// Handed to the scheduler when the gate switches arms, so it can do the
/// one-time transition bookkeeping (drop drafts, resync draft-head state).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArmSwitch {
    pub from: usize,
    pub to: usize,
}

#[derive(Default, Clone)]
struct ArmStats {
    /// Delivered-throughput EWMA (tokens/sec), `None` until first window.
    tps: Option<f64>,
    /// EWMA of |window tps − tps| (deviation, for the noise margin).
    dev: f64,
    /// Stale after a depth-regime change; refreshed by the next probe.
    stale: bool,
}

impl ArmStats {
    /// Fold one closed window into the estimate. `replace=true` (probe
    /// windows and post-regime-change windows) REPLACES the estimate: a
    /// sparse probe is a fresh look at a baseline that may have drifted
    /// arbitrarily since it was last run, and blending it against the stale
    /// value both lags the estimate and pollutes `dev` with the shift
    /// magnitude (inflating the hysteresis margin and delaying recovery).
    /// Continuous same-arm windows blend (EWMA) so `dev` tracks steady-state
    /// noise only.
    fn update(&mut self, window_tps: f64, replace: bool) {
        match (self.tps, replace) {
            (None, _) | (_, true) => {
                self.tps = Some(window_tps);
                self.dev *= 0.5; // decay: fresh baseline, keep a noise memory
            }
            (Some(prev), false) => {
                let next = (1.0 - TPS_ALPHA) * prev + TPS_ALPHA * window_tps;
                self.dev = (1.0 - TPS_ALPHA) * self.dev + TPS_ALPHA * (window_tps - next).abs();
                self.tps = Some(next);
            }
        }
        self.stale = false;
    }
}

/// Per-serve, single-instance gate. Lives on the scheduler thread; every
/// decode/verify step is timed and reported, so arbitration runs continuously
/// with zero dedicated measurement phases.
pub struct MtpGate {
    arms: Vec<ArmSpec>,
    stats: Vec<ArmStats>,
    /// Index into `arms` of the arm we believe is best. Arm 0 is the
    /// "primary" — the configuration the operator asked for — and the probe
    /// cadence is slower while we are on it (see [`Self::event_interval`]).
    current: usize,
    /// While probing, the arm actually being RUN (differs from `current`).
    probing: Option<usize>,
    probe_windows_left: usize,
    /// Round-robin cursor for choosing the next arm to probe.
    next_probe: usize,
    reprobe: usize,
    refresh: usize,
    probe_windows: usize,
    // Current-window accumulators (for whichever arm the steps ran in).
    win_tokens: f64,
    win_wall: f64,
    win_steps: usize,
    /// Consecutive closed windows where another arm beat the current one by
    /// more than the margin.
    losing_windows: usize,
    /// Emitted tokens since the last probe event in this arm.
    tokens_since_event: usize,
    observed_depth: usize,
    measured_at_depth: usize,
    fresh: Option<ArmSwitch>,
}

impl MtpGate {
    /// Current gate mode and delivered-throughput estimate for observability.
    pub fn observe(&self) -> (super::snapshot::MtpModeSnap, f32) {
        use super::snapshot::MtpModeSnap;

        let effective = self.effective();
        let mode = if self.probing.is_some() {
            MtpModeSnap::Probing
        } else {
            match self.arms[effective].kind {
                ArmKind::Spec { .. } => MtpModeSnap::Mtp,
                ArmKind::Serial => MtpModeSnap::Serial,
            }
        };
        (mode, self.stats[effective].tps.unwrap_or(0.0) as f32)
    }

    /// `arms` must be non-empty; index 0 is the primary arm and the arm the
    /// gate starts in, so it should be the operator's configured default
    /// (today: the external DFlash/DDTree drafter).
    pub fn new(arms: Vec<ArmSpec>) -> Self {
        assert!(!arms.is_empty(), "MtpGate needs at least one arm");
        let reprobe = reprobe_tokens();
        let refresh = serial_refresh_tokens();
        let probe_windows = probe_windows();
        let names: Vec<&str> = arms.iter().map(|a| a.name).collect();
        tracing::info!(
            "speculation gate: throughput-arbitrated over {} arm(s) {:?}; window={WINDOW_STEPS} \
             steps, dwell={SWITCH_DWELL_WINDOWS}, probe={probe_windows} window(s), \
             reprobe={reprobe} tok, refresh={refresh} tok",
            arms.len(),
            names,
        );
        let n = arms.len();
        Self {
            arms,
            stats: vec![ArmStats::default(); n],
            current: 0,
            probing: None,
            probe_windows_left: 0,
            next_probe: if n > 1 { 1 } else { 0 },
            reprobe,
            refresh,
            probe_windows,
            win_tokens: 0.0,
            win_wall: 0.0,
            win_steps: 0,
            losing_windows: 0,
            tokens_since_event: 0,
            observed_depth: 0,
            measured_at_depth: 0,
            fresh: None,
        }
    }

    /// The arm the scheduler should RUN for the next step (the probe target
    /// while probing, otherwise the current arm).
    pub fn next_arm(&self) -> ArmSpec {
        self.arms[self.effective()]
    }

    /// Index of the arm the scheduler should run next.
    pub fn next_arm_index(&self) -> usize {
        self.effective()
    }

    fn effective(&self) -> usize {
        self.probing.unwrap_or(self.current)
    }

    /// Can this gate ever change what the scheduler runs?
    ///
    /// **False for a one-arm gate** — `--mtp-gate dflash` / `--mtp-gate mtp`
    /// pin an arm, and `--mtp-gate auto` degenerates to one arm on a build
    /// that only produced one proposer. Such a gate can never probe and never
    /// switch, so every measurement it takes is dead work: the scheduler uses
    /// this to skip the depth scan, the delivered-token scans, the `Instant`
    /// pair and the window accounting entirely, leaving a pinned arm costing
    /// a branch against the disarmed path.
    ///
    /// Fixed at construction (`arms` is never resized), so callers should
    /// hoist it out of the step loop.
    pub fn arbitrates(&self) -> bool {
        self.arms.len() > 1
    }

    /// Feed the gate the batch's live context depth for this step: note it for
    /// logging and check it for a regime change.
    ///
    /// One call rather than the `note_depth` + `maybe_remeasure` pair the
    /// upstream port had, so the scheduler touches the gate once per step
    /// instead of twice.
    pub fn observe_depth(&mut self, depth: usize) {
        self.maybe_remeasure(depth);
        self.observed_depth = depth;
    }

    /// Depth-regime change: mark ALL baselines stale (economics moved) and
    /// let the normal probe cadence refresh them — no state wipe, no forced
    /// arm change.
    fn maybe_remeasure(&mut self, current_depth: usize) {
        let measured = self.measured_at_depth.max(REMEASURE_DEPTH_FLOOR);
        let live = current_depth.max(REMEASURE_DEPTH_FLOOR);
        if live >= measured * REMEASURE_DEPTH_FACTOR || measured >= live * REMEASURE_DEPTH_FACTOR {
            // DEBUG, not INFO, because this can fire on ALTERNATING STEPS.
            // Depth is the max `seq_len` over the batch, so at concurrency > 1
            // a short sequence joining a long one flips the regime one way and
            // the long one's next step flips it back. An INFO-level formatted
            // log on that cadence is real per-step cost and unbounded log
            // volume; the arm-switch line below stays at INFO because it fires
            // only on an actual decision.
            tracing::debug!(
                "speculation gate: depth regime changed ({} -> {} tokens); baselines stale, \
                 will re-probe on cadence",
                self.measured_at_depth,
                current_depth,
            );
            for s in self.stats.iter_mut() {
                s.stale = true;
            }
            self.measured_at_depth = current_depth;
            // Refresh the off-arms soon rather than waiting a full interval.
            self.tokens_since_event = self.tokens_since_event.max(self.event_interval());
        }
    }

    /// One-shot handoff of a fresh arm switch for scheduler bookkeeping.
    pub fn take_fresh_switch(&mut self) -> Option<ArmSwitch> {
        self.fresh.take()
    }

    /// Record one step. `emitted` is the number of tokens actually committed
    /// (a serial step emits 1; a bootstrap-only speculative step emits 1; a
    /// verify step emits 1 + accepted, summed over all sequences). Bootstrap
    /// and propose cost are charged to the arm that incurred them — proposing
    /// IS part of what a speculative arm costs to run.
    pub fn record_step(&mut self, wall: Duration, emitted: usize) {
        let tokens = emitted.max(1);
        self.win_tokens += tokens as f64;
        self.win_wall += wall.as_secs_f64();
        self.win_steps += 1;
        if self.probing.is_none() {
            self.tokens_since_event += tokens;
        }
        if self.win_steps >= WINDOW_STEPS {
            self.close_window();
        } else if self.arbitrates()
            && self.probing.is_none()
            && self.tokens_since_event >= self.event_interval()
        {
            // Time to look at another arm: finish the current window early so
            // the probe starts on a clean accumulator.
            //
            // The `arbitrates()` guard is load-bearing, not defensive. With a
            // single arm nothing ever resets `tokens_since_event`:
            // `close_window` only clears it when a probe opens or completes,
            // and a one-arm gate never probes. Without the guard it therefore
            // crosses `event_interval` once (after ~1024 emitted tokens) and
            // stays over it forever, closing a window on EVERY subsequent
            // step — the 16-step window silently degenerates to 1 step, which
            // turns the tok/s EWMA into per-step noise. The scheduler now
            // skips `record_step` entirely for a pinned arm, so this is
            // unreachable from the current call site; it is fixed here so the
            // invariant holds for any caller.
            self.close_window();
        }
    }

    /// Probe cadence: slow while on the primary arm (arm 0, what the operator
    /// configured), fast while off it. Mirrors upstream's Mtp=refresh /
    /// Serial=reprobe asymmetry.
    fn event_interval(&self) -> usize {
        if self.current == 0 {
            self.refresh
        } else {
            self.reprobe
        }
    }

    fn close_window(&mut self) {
        let ran = self.effective();
        if self.win_wall > 0.0 && self.win_steps > 0 {
            let window_tps = self.win_tokens / self.win_wall;
            let replace = self.probing.is_some() || self.stats[ran].stale;
            self.stats[ran].update(window_tps, replace);
        }
        self.win_tokens = 0.0;
        self.win_wall = 0.0;
        self.win_steps = 0;

        if self.probing.is_some() {
            self.probe_windows_left = self.probe_windows_left.saturating_sub(1);
            if self.probe_windows_left == 0 {
                self.probing = None;
                self.arbitrate();
                self.tokens_since_event = 0;
            }
            return;
        }

        // Scheduled exploration of another arm.
        if self.arbitrates() && self.tokens_since_event >= self.event_interval() {
            let target = self.pick_probe_target();
            self.probing = Some(target);
            self.probe_windows_left = self.probe_windows;
            return;
        }
        self.arbitrate();
    }

    /// Round-robin over the arms that are not current, so every arm's
    /// baseline gets refreshed eventually and no arm can be starved by a
    /// permanently-losing neighbour.
    fn pick_probe_target(&mut self) -> usize {
        let n = self.arms.len();
        for _ in 0..n {
            let cand = self.next_probe % n;
            self.next_probe = (self.next_probe + 1) % n;
            if cand != self.current {
                return cand;
            }
        }
        // Unreachable for n > 1; keeps the function total.
        self.current
    }

    /// Compare arm EWMAs with a hysteresis margin; switch after dwell.
    fn arbitrate(&mut self) {
        let Some(cur) = self.stats[self.current].tps else {
            return; // the arm we are in has not been measured yet
        };
        // Do not commit to the first arm that happens to beat the current one
        // before the rest of the configured portfolio has even been sampled.
        // With three arms the old policy measured arm 1, accumulated its
        // dwell on ordinary current-arm windows, and switched before the
        // round-robin cadence reached arm 2.  The same rule applies after a
        // depth-regime change: stale numbers are not comparable with the
        // refreshed current arm.  Exploration emits useful tokens, so waiting
        // for one fresh baseline per arm costs no discarded work.
        if self.stats.iter().any(|s| s.tps.is_none() || s.stale) {
            self.losing_windows = 0;
            return;
        }
        // Best OTHER arm that has a baseline. Arms never probed are simply
        // not candidates yet — guarded above, so this only handles a future
        // arm class that intentionally carries no throughput baseline.
        let mut best: Option<(usize, f64)> = None;
        for (i, s) in self.stats.iter().enumerate() {
            if i == self.current {
                continue;
            }
            if let Some(t) = s.tps
                && best.is_none_or(|(_, bt)| t > bt)
            {
                best = Some((i, t));
            }
        }
        let Some((best_i, best_tps)) = best else {
            return; // need at least one other baseline before any switch
        };
        let margin = (MARGIN_REL_FLOOR * cur)
            .max(0.5 * (self.stats[self.current].dev + self.stats[best_i].dev));
        if best_tps > cur + margin {
            self.losing_windows += 1;
            if self.losing_windows >= SWITCH_DWELL_WINDOWS {
                tracing::info!(
                    "speculation gate: switching {} -> {} (current {cur:.1} tok/s vs \
                     {best_tps:.1} tok/s, margin {margin:.1}, depth={})",
                    self.arms[self.current].name,
                    self.arms[best_i].name,
                    self.observed_depth,
                );
                let from = self.current;
                self.current = best_i;
                self.losing_windows = 0;
                self.tokens_since_event = 0;
                self.measured_at_depth = self.observed_depth;
                self.fresh = Some(ArmSwitch { from, to: best_i });
            }
        } else {
            self.losing_windows = 0;
        }
    }

    // ── Debug / test accessors ──
    pub fn arm_tps_debug(&self, i: usize) -> Option<f64> {
        self.stats.get(i).and_then(|s| s.tps)
    }
    pub fn current_arm_index(&self) -> usize {
        self.current
    }
    pub fn is_probing(&self) -> bool {
        self.probing.is_some()
    }
    /// Steps accumulated into the window that has not closed yet.
    #[cfg(test)]
    pub fn window_steps_debug(&self) -> usize {
        self.win_steps
    }
    pub fn arm_name(&self, i: usize) -> &'static str {
        self.arms[i].name
    }
}

#[cfg(test)]
mod tests;
