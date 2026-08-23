// SPDX-License-Identifier: AGPL-3.0-only

//! Prompt-lookup speculative decoding: policy and running stats.
//!
//! The *matching* lives in [`crate::rest_store::self_context`], which
//! maintains a per-sequence suffix automaton over the full token history
//! (prompt + generated). This module owns only what is specific to the
//! `--ngram-speculative` scheduler mode: the engage threshold and the
//! accept/reject counters used for logging.
//!
//! # Why the search moved
//!
//! This file used to carry its own longest-suffix search: for every
//! candidate length, scan every position in the history, on every decode
//! step — O(n·k) per token against a history that grows without bound.
//! The self-context tier needs the same query and answers it in O(1)
//! amortized, so keeping a second implementation here would have meant
//! two things to keep correct and only one of them fast.
//!
//! Two deliberate behaviour changes came with the move, both safe because
//! verification is the oracle and a draft is only ever a proposal:
//!
//! * matches are no longer capped at 16 tokens — a longer match is a
//!   better predictor, and the automaton finds it for free;
//! * the occurrence chosen is the most recent one rather than the
//!   earliest, which is the better prior for code that repeats itself.

/// Minimum suffix-match length for a prompt-lookup draft.
///
/// Deliberately permissive: this mode proposes a single token, so a wrong
/// guess costs one rejected verify slot and nothing else.
pub const NGRAM_MIN_MATCH: usize = 2;

/// Accept/reject bookkeeping for the prompt-lookup scheduler mode.
pub struct NgramProposer {
    /// Drafts the target accepted.
    pub accepts: u64,
    /// Drafts the target rejected.
    pub rejects: u64,
}

impl NgramProposer {
    pub fn new(_order: usize) -> Self {
        Self {
            accepts: 0,
            rejects: 0,
        }
    }

    /// The engage threshold this mode drafts at.
    pub fn min_match(&self) -> usize {
        NGRAM_MIN_MATCH
    }

    /// Record an accepted draft (for stats only).
    pub fn record_accept(&mut self) {
        self.accepts += 1;
    }

    /// Record a rejected draft (for stats only).
    pub fn record_reject(&mut self) {
        self.rejects += 1;
    }

    /// Observe is a no-op: the sequence's token history IS the cache, and
    /// the self-context index is synced from it at propose time.
    pub fn observe(&mut self, _history: &[u32], _next: u32) {}

    /// Drafts proposed so far. Kept for the mode's periodic log line.
    pub fn len(&self) -> usize {
        (self.accepts + self.rejects) as usize
    }

    /// Whether anything has been drafted yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_count_both_outcomes() {
        let mut p = NgramProposer::new(4);
        assert!(p.is_empty());
        p.record_accept();
        p.record_accept();
        p.record_reject();
        assert_eq!((p.accepts, p.rejects, p.len()), (2, 1, 3));
    }

    /// The matching behaviour this mode relies on is covered where it
    /// lives, in `rest_store::self_context::tests` — including the four
    /// cases this file used to test directly (a repeated span, no match,
    /// a too-short history, and a repetitive cycle).
    #[test]
    fn engage_threshold_is_permissive_for_single_token_drafts() {
        assert_eq!(NgramProposer::new(4).min_match(), 2);
    }
}
