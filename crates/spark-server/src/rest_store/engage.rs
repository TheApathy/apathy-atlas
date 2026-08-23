// SPDX-License-Identifier: AGPL-3.0-only

//! Scheduler-facing half of the REST binding: the pre-emption gate and the
//! production counters.
//!
//! [`super::propose`] answers "does the corpus know this context". This
//! module answers the different question the scheduler actually asks:
//! *is the retrieved continuation good enough to spend a verify frame on
//! instead of the neural drafter?* DFlash runs at roughly p = 0.90
//! per-token acceptance on this target, so a REST chain that merely beats
//! nothing is still a loss. The gate below is therefore deliberately
//! stricter than the library default (see `PHASE2.md` §2).
//!
//! Three conditions must all hold before a chain pre-empts DFlash:
//!
//! 1. the store matched at least `ATLAS_REST_MIN_MATCH` context tokens
//!    (enforced inside [`super::propose`]);
//! 2. the retrieved spine is at least `num_drafts` long, so it fills the
//!    same verify width the DFlash chain would have filled — a shorter
//!    chain would narrow the frame and lose more than it gains;
//! 3. `num_drafts >= MIN_PREEMPT_WIDTH`, so the frame routes to the
//!    generic (K = γ) verifier rather than one of the fixed-width K2/K3/K4
//!    verifiers.
//!
//! None of this changes what the target emits. Verification is untouched;
//! a REST chain is the same `Vec<u32>` shape the MTP/DFlash proposer
//! produces and is accepted token-by-token against the target's argmax
//! exactly as a neural draft is. REST changes *which tokens are proposed*,
//! never which are emitted.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use rest_store::DraftTree;

/// Smallest `num_drafts` that pre-emption will engage at.
///
/// `select_verify_dispatch` (`scheduler/mtp_step.rs`) routes a frame of
/// four or more drafts to the generic verifier and anything narrower to a
/// fixed-width K2/K3/K4 verifier. Staying on the single wide path keeps
/// REST frames indistinguishable from DFlash frames at verify and keeps
/// the accepted-token accounting below on one code path.
pub const MIN_PREEMPT_WIDTH: usize = 4;

/// How many engagements between periodic summary logs.
const LOG_EVERY: u64 = 128;

/// Do not pre-empt when the drafter's last frame accepted at least this
/// many tokens.
///
/// Set to this tier's own measured yield: the held-out gate run put an
/// engaged static-store step at 13.3 accepted tokens, so a drafter frame
/// that just did better than that is not worth displacing. Higher than
/// the self-context threshold because the store retrieves longer verbatim
/// spans when it retrieves at all.
pub const DEFAULT_MAX_DRAFTER_ACCEPT: usize = 13;

/// The saturation gate, from `ATLAS_REST_MAX_DRAFTER_ACCEPT`.
pub fn max_drafter_accept() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| super::env_usize("ATLAS_REST_MAX_DRAFTER_ACCEPT", DEFAULT_MAX_DRAFTER_ACCEPT))
}

/// Process-wide REST counters.
///
/// Plain relaxed atomics: these are observability, never read back into a
/// decision, and the decode loop must not pay for ordering.
struct Counters {
    engaged: AtomicU64,
    accepted: AtomicU64,
    declined_short_spine: AtomicU64,
}

static COUNTERS: Counters = Counters {
    engaged: AtomicU64::new(0),
    accepted: AtomicU64::new(0),
    declined_short_spine: AtomicU64::new(0),
};

/// A snapshot of the REST counters, for logging and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestStats {
    /// Frames where a REST chain pre-empted the DFlash proposal. Each one
    /// is also a DFlash drafter forward pass that never ran.
    pub engaged: u64,
    /// Tokens the target accepted from REST-proposed frames.
    pub accepted: u64,
    /// Lookups that cleared the match gate but whose spine was shorter
    /// than the verify width.
    pub declined_short_spine: u64,
}

/// Read the counters. `engaged` doubles as the count of skipped DFlash
/// proposals: pre-emption is the only thing that installs a REST chain.
pub fn stats() -> RestStats {
    RestStats {
        engaged: COUNTERS.engaged.load(Ordering::Relaxed),
        accepted: COUNTERS.accepted.load(Ordering::Relaxed),
        declined_short_spine: COUNTERS.declined_short_spine.load(Ordering::Relaxed),
    }
}

fn log_stats(reason: &str) {
    let s = stats();
    tracing::info!(
        engaged = s.engaged,
        accepted_tokens = s.accepted,
        dflash_steps_skipped = s.engaged,
        declined_short_spine = s.declined_short_spine,
        accepted_per_engagement = if s.engaged == 0 {
            0.0
        } else {
            s.accepted as f64 / s.engaged as f64
        },
        "REST drafting {reason}"
    );
}

/// Count tokens the target accepted from a REST-proposed frame.
///
/// Called from the verify path *after* acceptance is decided, with the
/// count the verifier already computed. It reads nothing back — no REST
/// state can influence which tokens are accepted.
pub fn record_accepted(tokens: usize) {
    COUNTERS
        .accepted
        .fetch_add(tokens as u64, Ordering::Relaxed);
}

/// The last `max_k` context tokens, with the freshly sampled `next`
/// appended — the exact suffix whose continuation the store is asked for.
fn context_tail(committed: &[u32], next: u32, max_k: usize) -> Vec<u32> {
    let take = committed.len().min(max_k.saturating_sub(1));
    let mut ctx = Vec::with_capacity(take + 1);
    ctx.extend_from_slice(&committed[committed.len() - take..]);
    ctx.push(next);
    ctx
}

/// Flatten a retrieved tree into a chain of exactly `num_drafts` tokens,
/// or decline.
///
/// Flat, not a tree: Phase 1 measured the frequency-weighted tree beating
/// its own spine by only 3-7 %, which does not pay for the DDTree budget,
/// dispatch, thinking-mode, content-policy and EP gates a payload has to
/// clear (`PHASE2.md` §2).
fn chain_from_tree(tree: &DraftTree, num_drafts: usize) -> Option<Vec<u32>> {
    let mut chain = tree.spine();
    if chain.len() < num_drafts {
        COUNTERS
            .declined_short_spine
            .fetch_add(1, Ordering::Relaxed);
        return None;
    }
    chain.truncate(num_drafts);
    Some(chain)
}

/// Try to pre-empt this step's DFlash proposal with a retrieved chain.
///
/// `committed` is the sequence's committed token stream and `next` the
/// token just sampled but not yet appended to it; the chain continues
/// from `next`. Returns `None` — cheaply, after a single `OnceLock` load —
/// whenever REST is disabled, which is the default.
///
/// A `Some` result is counted as an engagement here, so callers must
/// install what they are given.
pub fn preempt(committed: &[u32], next: u32, num_drafts: usize) -> Option<Vec<u32>> {
    if num_drafts < MIN_PREEMPT_WIDTH || !super::enabled() {
        return None;
    }
    let ctx = context_tail(committed, next, super::config().max_k);
    let chain = chain_from_tree(&super::propose(&ctx)?, num_drafts)?;
    let engaged = COUNTERS.engaged.fetch_add(1, Ordering::Relaxed) + 1;
    if engaged == 1 {
        tracing::info!(
            match_context = ctx.len(),
            chain = num_drafts,
            "REST pre-empted the DFlash proposal for the first time"
        );
    } else if engaged.is_multiple_of(LOG_EVERY) {
        log_stats("progress");
    }
    Some(chain)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rest_store::trie::DraftNode;

    /// A straight chain of `len` tokens, as the store emits for a single
    /// unbranched occurrence.
    fn chain_tree(len: usize) -> DraftTree {
        DraftTree {
            nodes: (0..len)
                .map(|i| DraftNode {
                    token: 100 + i as u32,
                    parent: if i == 0 { -1 } else { i as i32 - 1 },
                    count: 1,
                    depth: i as u16 + 1,
                })
                .collect(),
            match_len: 12,
            occurrences: 1,
        }
    }

    #[test]
    fn context_is_the_max_k_suffix_plus_the_fresh_token() {
        let committed: Vec<u32> = (1..=100).collect();
        let ctx = context_tail(&committed, 777, 16);
        assert_eq!(
            ctx.len(),
            16,
            "max_k caps the whole context, not just the tail"
        );
        assert_eq!(ctx[0], 86);
        assert_eq!(ctx[14], 100);
        assert_eq!(
            ctx[15], 777,
            "the freshly sampled token must terminate the context"
        );
    }

    #[test]
    fn context_handles_a_short_history() {
        assert_eq!(context_tail(&[], 9, 16), vec![9]);
        assert_eq!(context_tail(&[1, 2], 9, 16), vec![1, 2, 9]);
        assert_eq!(context_tail(&[1, 2], 9, 1), vec![9]);
    }

    #[test]
    fn a_spine_shorter_than_the_verify_width_is_declined() {
        let before = stats().declined_short_spine;
        assert!(chain_from_tree(&chain_tree(3), 4).is_none());
        assert_eq!(stats().declined_short_spine, before + 1);
    }

    #[test]
    fn a_long_spine_is_truncated_to_the_verify_width() {
        let chain = chain_from_tree(&chain_tree(16), 4).expect("spine clears the width gate");
        assert_eq!(chain, vec![100, 101, 102, 103]);
    }

    #[test]
    fn an_exact_spine_is_kept_whole() {
        assert_eq!(chain_from_tree(&chain_tree(4), 4).unwrap().len(), 4);
    }

    #[test]
    fn narrow_verify_widths_never_engage() {
        // Below MIN_PREEMPT_WIDTH the frame would route to a fixed-width
        // verifier, so pre-emption declines before touching the store.
        for num_drafts in 0..MIN_PREEMPT_WIDTH {
            assert!(preempt(&[1, 2, 3], 4, num_drafts).is_none());
        }
    }

    #[test]
    fn preemption_is_inert_without_a_store() {
        // The zero-overhead claim: with ATLAS_REST_STORE unset, `preempt`
        // returns None without allocating a context or counting anything.
        if std::env::var_os("ATLAS_REST_STORE").is_none() {
            let before = stats();
            assert!(preempt(&(1..=64).collect::<Vec<u32>>(), 65, 15).is_none());
            assert_eq!(stats().engaged, before.engaged);
        }
    }
}
