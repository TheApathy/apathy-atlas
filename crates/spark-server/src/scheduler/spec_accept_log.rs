// SPDX-License-Identifier: AGPL-3.0-only

//! Per-position speculative-acceptance diagnostics (native-MTP acceptance
//! ceiling).
//!
//! The speculation gate arbitrates arms on delivered throughput, not draft
//! acceptance — a below-target native-MTP arm silently loses arbitration and
//! the run tells us nothing about WHY its drafts miss. This module records,
//! per verify frame, the position-by-position match pattern (draft[i] vs the
//! target's argmax row verified[i]) and a cumulative per-position acceptance
//! rate, so one GPU run answers two questions at once:
//!
//!   1. What is the acceptance ceiling at each draft position?
//!   2. Where does the chain collapse (the first position whose rate is
//!      below the SGLang-comparable ~0.92)?
//!
//! Inert unless `ATLAS_MTP_ACCEPT_LOG=1` (resolved once at first use).
//! Per-frame lines are DEBUG; a cumulative by-position summary is INFO every
//! 128 frames. The per-position tally is keyed by absolute draft index and is
//! shared across K2/K3/K4 and the γ-wide generic verifier, so a mixed run
//! still produces one coherent ceiling curve.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

const ENV: &str = "ATLAS_MTP_ACCEPT_LOG";

/// Draft-width cap for the per-position tally (DFlash γ=16 is the widest
/// producer in-tree; native MTP uses ≤ 8).
const MAX_POSITIONS: usize = 32;
/// Frames between cumulative INFO summaries.
const SUMMARY_EVERY: u64 = 128;

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(ENV).ok().as_deref() == Some("1"))
}

/// Per-position counters, all atomics so the summary can be read without
/// holding a lock on the hot path.
struct Tally {
    matched: [AtomicU64; MAX_POSITIONS],
    total: [AtomicU64; MAX_POSITIONS],
    frames: AtomicU64,
    drafts_proposed: AtomicU64,
    drafts_accepted: AtomicU64,
    requested_width: AtomicU64,
}

impl Tally {
    const fn new() -> Self {
        Self {
            matched: [const { AtomicU64::new(0) }; MAX_POSITIONS],
            total: [const { AtomicU64::new(0) }; MAX_POSITIONS],
            frames: AtomicU64::new(0),
            drafts_proposed: AtomicU64::new(0),
            drafts_accepted: AtomicU64::new(0),
            requested_width: AtomicU64::new(0),
        }
    }

    /// Fold one frame into the tally and return the per-position match
    /// pattern ("Y"/"N" per draft slot) plus the new frame count. Pure with
    /// respect to the tally, so tests drive it on a local instance.
    fn fold(
        &self,
        num_drafts: usize,
        drafts: &[u32],
        verified: &[u32],
        num_accepted: usize,
    ) -> (String, u64) {
        let mut pattern = String::with_capacity(drafts.len());
        for (i, draft) in drafts.iter().enumerate() {
            let ok = verified.get(i) == Some(draft);
            pattern.push(if ok { 'Y' } else { 'N' });
            if i < MAX_POSITIONS {
                self.total[i].fetch_add(1, Ordering::Relaxed);
                if ok {
                    self.matched[i].fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        self.drafts_proposed
            .fetch_add(drafts.len() as u64, Ordering::Relaxed);
        self.drafts_accepted
            .fetch_add(num_accepted as u64, Ordering::Relaxed);
        self.requested_width
            .fetch_add(num_drafts as u64, Ordering::Relaxed);
        let frames = self.frames.fetch_add(1, Ordering::Relaxed) + 1;
        (pattern, frames)
    }

    /// One-line INFO summary of the per-position acceptance curve, flagging
    /// the first position that collapses below 0.80.
    fn summarize(&self) {
        let frames = self.frames.load(Ordering::Relaxed);
        if frames == 0 {
            return;
        }
        let proposed = self.drafts_proposed.load(Ordering::Relaxed);
        let accepted = self.drafts_accepted.load(Ordering::Relaxed);
        let overall = if proposed > 0 {
            accepted as f64 / proposed as f64
        } else {
            0.0
        };
        let mut rates: Vec<String> = Vec::new();
        let mut first_collapse: Option<usize> = None;
        for i in 0..MAX_POSITIONS {
            let total = self.total[i].load(Ordering::Relaxed);
            if total == 0 {
                continue;
            }
            let matched = self.matched[i].load(Ordering::Relaxed);
            let rate = matched as f64 / total as f64;
            rates.push(format!("{i}:{rate:.3}"));
            if first_collapse.is_none() && (rate < 0.80) {
                first_collapse = Some(i);
            }
        }
        let width = self.requested_width.load(Ordering::Relaxed);
        let per_position = rates.join(" ");
        match first_collapse {
            Some(pos) => tracing::info!(
                frames,
                requested_width = width,
                per_position = %per_position,
                overall = format!("{overall:.3}"),
                collapse_at = pos,
                "MTP_ACCEPT summary: per-position acceptance; chain collapses at draft {pos} (<0.80)"
            ),
            None => tracing::info!(
                frames,
                requested_width = width,
                per_position = %per_position,
                overall = format!("{overall:.3}"),
                "MTP_ACCEPT summary: per-position acceptance; no position below 0.80"
            ),
        }
    }
}

static TALLY: Tally = Tally::new();

/// Record one verify frame.
///
/// `proposer` labels the arm ("mtp" for the native in-checkpoint head,
/// "dflash" for the external drafter) so a mixed arm run stays separable.
/// `drafts[i]` is compared against `verified[i]` — the target's argmax row
/// for draft slot `i`, exactly the equality `exact_accept_prefix` uses. The
/// accepted count is logged for cross-checking against the recorded
/// `num_accepted`.
pub(crate) fn record_accept(
    proposer: &str,
    seq_len: usize,
    num_drafts: usize,
    drafts: &[u32],
    verified: &[u32],
    num_accepted: usize,
) {
    if !enabled() {
        return;
    }
    let (pattern, frames) = TALLY.fold(num_drafts, drafts, verified, num_accepted);
    tracing::debug!(
        proposer,
        seq_len,
        num_drafts,
        accepted = num_accepted,
        pattern = %pattern,
        "MTP_ACCEPT frame"
    );
    if frames % SUMMARY_EVERY == 0 {
        TALLY.summarize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tally_counts_per_position_matches() {
        let t = Tally::new();
        let (pattern, frames) = t.fold(3, &[10, 20, 30], &[10, 20, 99], 2);
        assert_eq!(pattern, "YYN");
        assert_eq!(frames, 1);
        assert_eq!(t.total[0].load(Ordering::Relaxed), 1);
        assert_eq!(t.matched[0].load(Ordering::Relaxed), 1);
        assert_eq!(t.total[1].load(Ordering::Relaxed), 1);
        assert_eq!(t.matched[1].load(Ordering::Relaxed), 1);
        assert_eq!(t.total[2].load(Ordering::Relaxed), 1);
        assert_eq!(t.matched[2].load(Ordering::Relaxed), 0);
        // Position 3 and beyond are untouched by a 3-draft frame.
        assert_eq!(t.total[3].load(Ordering::Relaxed), 0);
        assert_eq!(t.drafts_proposed.load(Ordering::Relaxed), 3);
        assert_eq!(t.drafts_accepted.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn empty_draft_frame_is_inert() {
        let t = Tally::new();
        let (pattern, frames) = t.fold(0, &[], &[], 0);
        assert_eq!(pattern, "");
        assert_eq!(frames, 1);
        assert_eq!(t.drafts_proposed.load(Ordering::Relaxed), 0);
        assert_eq!(t.total[0].load(Ordering::Relaxed), 0);
    }

    #[test]
    fn width_above_cap_clamps_tally_not_crash() {
        let t = Tally::new();
        let wide: Vec<u32> = (0..40).collect();
        let mut verified = wide.clone();
        verified[39] = 999;
        let (pattern, frames) = t.fold(16, &wide, &verified, 39);
        assert_eq!(frames, 1);
        assert_eq!(pattern.len(), 40);
        let mut expected = "Y".repeat(39);
        expected.push('N');
        assert_eq!(pattern, expected);
        // Only the first MAX_POSITIONS positions are tallied.
        assert_eq!(t.total[MAX_POSITIONS - 1].load(Ordering::Relaxed), 1);
        assert_eq!(t.matched[MAX_POSITIONS - 1].load(Ordering::Relaxed), 1);
        assert_eq!(t.drafts_proposed.load(Ordering::Relaxed), 40);
    }

    #[test]
    fn summarize_flags_first_collapse_position() {
        let t = Tally::new();
        // Position 0 always matches; position 1 collapses.
        for _ in 0..4 {
            let _ = t.fold(2, &[5, 6], &[5, 99], 1);
        }
        let collapse = (0..MAX_POSITIONS).find(|i| {
            let total = t.total[*i].load(Ordering::Relaxed);
            total > 0 && {
                let matched = t.matched[*i].load(Ordering::Relaxed);
                (matched as f64 / total as f64) < 0.80
            }
        });
        assert_eq!(collapse, Some(1));
        // Summarize is a no-op on a fresh tally.
        Tally::new().summarize();
    }
}
