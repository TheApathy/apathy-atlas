// SPDX-License-Identifier: AGPL-3.0-only

//! Self-context drafting: retrieval whose corpus is the sequence itself.
//!
//! The static `.rest` store can only draft text that resembles something
//! indexed offline. This tier indexes nothing: it matches the sequence's
//! own history — prompt plus everything generated so far — so it works on
//! prompts that were never seen, needs no file and no network, and gets
//! stronger as the generation grows. Repetitive code is exactly its case:
//! re-emitted helper bodies, recurring identifiers, a complexity section
//! that restates the method names written a thousand tokens earlier.
//!
//! # Structure
//!
//! An incremental suffix automaton ([`rest_store::sam`]) maintained per
//! sequence. Appending a token is O(1) amortized and the longest-suffix
//! match is maintained as a side effect, so the per-step cost does not
//! grow with history length — unlike the linear rescan it replaces.
//!
//! # Surviving the scheduler
//!
//! Two things happen to a sequence's history that a naive index would get
//! wrong, and both are handled in [`SelfContextIndex::sync`]:
//!
//! * **Batched appends.** An accepted γ-block adds several tokens at once.
//!   Sync replays them in order, so the automaton sees the same stream a
//!   token-at-a-time run would have.
//! * **Rollback.** Rejected speculation and the degeneration watchdog can
//!   both *shorten* the committed history. Sync detects this — by length
//!   and by a fingerprint over the tail of what was indexed, which also
//!   catches a rollback that has already regrown past its old length — and
//!   rebuilds rather than indexing a prefix that no longer exists.
//!
//! Neither case can corrupt output: a draft is a proposal and verification
//! is unchanged. What they would corrupt is the *usefulness* of the index,
//! by proposing continuations of text the sequence never emitted.
//!
//! # Memory
//!
//! Bounded per sequence by `ATLAS_SELF_CONTEXT_MAX_TOKENS`. When the
//! window fills, the automaton is rebuilt over the most recent half —
//! amortized O(1) per token, and the recent half is where the matches
//! worth having live. The index lives in `ActiveSeq`, so it is freed when
//! the sequence ends.
//!
//! # Environment
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `ATLAS_SELF_CONTEXT_DRAFT` | unset | `1` enables the tier; anything else leaves it inert |
//! | `ATLAS_SELF_CONTEXT_MIN_MATCH` | 16 | Engage gate on suffix-match length |
//! | `ATLAS_SELF_CONTEXT_MAX_TOKENS` | 16384 | Indexed window per sequence |
//! | `ATLAS_SELF_CONTEXT_MAX_DRAFTER_ACCEPT` | 10 | Skip pre-emption when the drafter's last frame accepted this many or more |

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use rest_store::sam::SuffixAutomaton;

/// Tokens of indexed tail hashed to detect a rollback that regrew.
const TAIL_FINGERPRINT: usize = 32;

/// Engage gate, in matched tokens.
///
/// This is the number the offline eval decided, and it is much higher
/// than a "is this match meaningful" threshold would suggest, because the
/// bar is not *usefulness* — it is **the drafter this chain displaces**.
/// At γ=15 and the measured per-token acceptance p≈0.90, a DFlash frame
/// is worth p(1-p^15)/(1-p) ≈ 7.15 accepted tokens. A retrieval chain
/// that accepts fewer than that is a loss, however impressive it looks
/// next to zero.
///
/// Replaying 120 real AEON generations (63,710 decode steps, the 31
/// harness-error rows excluded), accepted tokens per engaged step:
///
/// | min_match | engagement | accepted/engaged | verdict |
/// |---|---|---|---|
/// | 6 | 13.52 % | 4.64 | loses to the drafter |
/// | 8 | 8.83 % | 5.54 | loses |
/// | 10 | 6.23 % | 6.41 | loses |
/// | 12 | 4.59 % | 7.26 | break-even |
/// | 16 | 2.86 % | 8.77 | wins by 1.6 tok/engagement |
/// | 20 | 2.06 % | 9.71 | wins, at half the engagement |
///
/// 16 is where the win is clear without giving up most of the
/// engagement. A code workload (1,096 regenerated Magicoder answers,
/// 777k steps) agrees: 9.40 accepted/engaged at the same gate.
pub const DEFAULT_MIN_MATCH: usize = 16;

/// Resolved self-context configuration.
#[derive(Debug, Clone, Copy)]
pub struct SelfContextConfig {
    /// Whether the tier is enabled at all.
    pub enabled: bool,
    /// Minimum suffix-match length, counting the freshly sampled token.
    ///
    /// See [`DEFAULT_MIN_MATCH`] for why this is as high as it is.
    pub min_match: usize,
    /// Indexed window per sequence.
    pub max_tokens: usize,
}

impl Default for SelfContextConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_match: DEFAULT_MIN_MATCH,
            max_tokens: 16384,
        }
    }
}

impl SelfContextConfig {
    fn from_env() -> Self {
        let d = Self::default();
        Self {
            enabled: std::env::var("ATLAS_SELF_CONTEXT_DRAFT").ok().as_deref() == Some("1"),
            min_match: super::env_usize("ATLAS_SELF_CONTEXT_MIN_MATCH", d.min_match),
            max_tokens: super::env_usize("ATLAS_SELF_CONTEXT_MAX_TOKENS", d.max_tokens)
                .max(TAIL_FINGERPRINT * 2),
        }
    }
}

/// How many engagements between periodic summary logs.
const LOG_EVERY: u64 = 128;

/// Do not pre-empt when the drafter's last frame accepted at least this
/// many tokens.
///
/// The crossover is this tier's own measured yield: replaying the three
/// long generations, an engaged step accepts 9.69 tokens on average, so
/// displacing a drafter that just did better than that loses tokens. The
/// eval cannot measure the drafter's acceptance at those positions — it
/// has no drafter — and the sign of the whole feature depends on it:
///
/// | assumed drafter acceptance at engaged positions | net |
/// |---|---|
/// | 7.15 (its unconditional mean) | +2.54 tok/engagement, +4.5 % tok/step |
/// | 11.07 (elevated on repetitive text) | -1.39, -2.5 % |
/// | 15.00 (saturated) | -5.31, -9.4 % |
///
/// Rather than bet on which column is true, the scheduler measures it per
/// sequence and skips pre-emption when the drafter is winning.
pub const DEFAULT_MAX_DRAFTER_ACCEPT: usize = 10;

/// The saturation gate, from `ATLAS_SELF_CONTEXT_MAX_DRAFTER_ACCEPT`.
pub fn max_drafter_accept() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        super::env_usize(
            "ATLAS_SELF_CONTEXT_MAX_DRAFTER_ACCEPT",
            DEFAULT_MAX_DRAFTER_ACCEPT,
        )
    })
}

/// Cached configuration; read once, off the decode hot path.
pub fn config() -> &'static SelfContextConfig {
    static CFG: OnceLock<SelfContextConfig> = OnceLock::new();
    CFG.get_or_init(SelfContextConfig::from_env)
}

/// Whether self-context drafting is enabled.
pub fn enabled() -> bool {
    config().enabled
}

struct Counters {
    engaged: AtomicU64,
    accepted: AtomicU64,
    rebuilds: AtomicU64,
    declined_short_chain: AtomicU64,
}

static COUNTERS: Counters = Counters {
    engaged: AtomicU64::new(0),
    accepted: AtomicU64::new(0),
    rebuilds: AtomicU64::new(0),
    declined_short_chain: AtomicU64::new(0),
};

/// Count tokens the target accepted from a self-context frame.
pub fn record_accepted(tokens: usize) {
    COUNTERS
        .accepted
        .fetch_add(tokens as u64, Ordering::Relaxed);
}

/// Engagements, accepted tokens, index rebuilds, short-chain declines.
pub fn stats() -> (u64, u64, u64, u64) {
    (
        COUNTERS.engaged.load(Ordering::Relaxed),
        COUNTERS.accepted.load(Ordering::Relaxed),
        COUNTERS.rebuilds.load(Ordering::Relaxed),
        COUNTERS.declined_short_chain.load(Ordering::Relaxed),
    )
}

fn fingerprint(tokens: &[u32]) -> u64 {
    tokens.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &t| {
        (h ^ u64::from(t)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Per-sequence self-context index.
///
/// `Default` is an empty, unallocated index: a sequence that never drafts
/// from self-context carries three words and no heap.
#[derive(Default)]
pub struct SelfContextIndex {
    sam: Option<SuffixAutomaton>,
    /// Index into the sequence's token stream where the window starts.
    window_start: usize,
    /// How much of the token stream has been indexed.
    synced: usize,
    /// Fingerprint of the last [`TAIL_FINGERPRINT`] indexed tokens.
    tail_fp: u64,
}

impl SelfContextIndex {
    /// Indexed window length, in tokens.
    pub fn indexed(&self) -> usize {
        self.sam.as_ref().map_or(0, SuffixAutomaton::len)
    }

    /// Approximate heap footprint, in bytes.
    pub fn heap_bytes(&self) -> usize {
        self.sam.as_ref().map_or(0, SuffixAutomaton::heap_bytes)
    }

    /// Release the index. Called when a sequence's history is no longer
    /// worth keeping; the next sync rebuilds from scratch.
    pub fn clear(&mut self) {
        self.sam = None;
        self.window_start = 0;
        self.synced = 0;
        self.tail_fp = 0;
    }

    fn tail_of(&self, tokens: &[u32]) -> u64 {
        let end = self.synced.min(tokens.len());
        let start = end.saturating_sub(TAIL_FINGERPRINT);
        fingerprint(&tokens[start..end])
    }

    fn rebuild(&mut self, tokens: &[u32], max_tokens: usize) {
        COUNTERS.rebuilds.fetch_add(1, Ordering::Relaxed);
        let start = tokens.len().saturating_sub(max_tokens);
        let mut sam = SuffixAutomaton::new();
        sam.extend_from_slice(&tokens[start..]);
        self.sam = Some(sam);
        self.window_start = start;
        self.synced = tokens.len();
        self.tail_fp = self.tail_of(tokens);
    }

    /// Bring the index up to date with the sequence's committed tokens.
    ///
    /// Cheap in the common case (append the few tokens the last step
    /// committed) and correct in the uncommon ones: a shortened history or
    /// one that diverged from what was indexed forces a rebuild, and an
    /// overfull window is rebuilt over its recent half.
    pub fn sync(&mut self, tokens: &[u32]) {
        let cfg = config();
        let diverged = tokens.len() < self.synced || self.tail_of(tokens) != self.tail_fp;
        if self.sam.is_none() || diverged {
            self.rebuild(tokens, cfg.max_tokens);
            return;
        }
        let appended = &tokens[self.synced..];
        self.sam
            .as_mut()
            .expect("checked above")
            .extend_from_slice(appended);
        self.synced = tokens.len();
        self.tail_fp = self.tail_of(tokens);
        if self.indexed() > cfg.max_tokens {
            // Halve the window rather than dropping it: rebuilding once
            // per max_tokens/2 appended tokens keeps this amortized O(1),
            // and recent history is where the matches live anyway.
            self.rebuild(tokens, cfg.max_tokens / 2);
        }
    }

    /// The draft chain continuing `tokens ++ [next]`, or `None`.
    ///
    /// Declines unless the matched suffix reaches `min_match` (counting
    /// `next`) and the earlier occurrence has at least `num_drafts` tokens
    /// after it, so the chain fills the same verify width DFlash would.
    pub fn propose(&mut self, tokens: &[u32], next: u32, num_drafts: usize) -> Option<Vec<u32>> {
        let cfg = config();
        if num_drafts == 0 || tokens.len() < cfg.min_match {
            return None;
        }
        self.sync(tokens);
        let (match_len, end) = self.sam.as_ref()?.peek(next);
        if match_len < cfg.min_match {
            return None;
        }
        // `end` is a window offset; the chain is read from the sequence.
        let from = self.window_start + end;
        if from + num_drafts > tokens.len() {
            COUNTERS
                .declined_short_chain
                .fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let engaged = COUNTERS.engaged.fetch_add(1, Ordering::Relaxed) + 1;
        if engaged == 1 {
            tracing::info!(
                match_len,
                indexed = self.indexed(),
                "self-context drafting engaged for the first time"
            );
        } else if engaged.is_multiple_of(LOG_EVERY) {
            let (engaged, accepted, rebuilds, declined) = stats();
            tracing::info!(
                engaged,
                accepted_tokens = accepted,
                dflash_steps_skipped = engaged,
                index_rebuilds = rebuilds,
                declined_short_chain = declined,
                accepted_per_engagement = accepted as f64 / engaged as f64,
                indexed = self.indexed(),
                index_kib = self.heap_bytes() / 1024,
                "self-context drafting progress"
            );
        }
        Some(tokens[from..from + num_drafts].to_vec())
    }

    /// The single next token the history predicts — the prompt-lookup
    /// query, answered from the same index so there is one implementation
    /// of longest-suffix matching rather than two.
    pub fn propose_one(&mut self, tokens: &[u32], min_match: usize) -> Option<u32> {
        if tokens.is_empty() {
            return None;
        }
        let (&last, head) = tokens.split_last()?;
        self.sync(head);
        let (match_len, end) = self.sam.as_ref()?.peek(last);
        let from = self.window_start + end;
        (match_len >= min_match && from < head.len()).then(|| head[from])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The config is process-wide and read once, so tests assert the
    /// defaults and the structure rather than re-reading the environment.
    #[test]
    fn defaults_are_the_documented_ones() {
        let d = SelfContextConfig::default();
        assert!(!d.enabled, "the tier must be inert unless asked for");
        assert_eq!(d.min_match, DEFAULT_MIN_MATCH);
        assert_eq!(d.max_tokens, 16384);
    }

    /// The gate exists to beat the drafter, not to beat nothing.
    #[test]
    fn the_gate_clears_the_drafters_expected_acceptance() {
        // p(1-p^k)/(1-p) at p=0.90, k=15.
        let drafter = 0.9 * (1.0 - 0.9f64.powi(15)) / (1.0 - 0.9);
        assert!(
            (7.0..7.3).contains(&drafter),
            "the break-even this gate was tuned against moved: {drafter}"
        );
        assert!(
            DEFAULT_MIN_MATCH >= 12,
            "below 12 the eval measured fewer accepted tokens than the drafter it pre-empts"
        );
    }

    #[test]
    fn a_repeated_span_proposes_its_earlier_continuation() {
        let mut idx = SelfContextIndex::default();
        let gate = config().min_match;
        // A span long enough to clear the gate, then repeated up to the
        // token before the match completes.
        let base: Vec<u32> = (100..100 + (gate as u32 + 12)).collect();
        let mut tokens = base.clone();
        tokens.extend_from_slice(&base[..gate - 1]);
        let next = base[gate - 1];
        assert_eq!(
            idx.propose(&tokens, next, 4),
            Some(base[gate..gate + 4].to_vec()),
            "the chain must continue the earlier occurrence"
        );
    }

    #[test]
    fn novel_text_proposes_nothing() {
        let mut idx = SelfContextIndex::default();
        let tokens: Vec<u32> = (0..500).collect();
        assert_eq!(idx.propose(&tokens, 9999, 4), None);
    }

    #[test]
    fn a_chain_shorter_than_the_verify_width_is_declined() {
        let mut idx = SelfContextIndex::default();
        // The only occurrence sits at the very end, so nothing follows it.
        let gate = config().min_match;
        let base: Vec<u32> = (100..100 + (gate as u32 + 12)).collect();
        let mut tokens = base.clone();
        tokens.extend_from_slice(&base[..gate - 1]);
        // Ask for a wider chain than the history can supply after the match.
        assert_eq!(idx.propose(&tokens, base[gate - 1], 4096), None);
    }

    #[test]
    fn a_shortened_history_rebuilds_instead_of_indexing_a_ghost() {
        let mut idx = SelfContextIndex::default();
        let tokens: Vec<u32> = (0..200).map(|i| i % 50).collect();
        idx.sync(&tokens);
        assert_eq!(idx.indexed(), 200);

        // Watchdog rollback: history shrinks.
        idx.sync(&tokens[..120]);
        assert_eq!(
            idx.indexed(),
            120,
            "the index must follow the rollback down"
        );
    }

    #[test]
    fn a_rollback_that_regrew_is_detected_by_the_tail_fingerprint() {
        let mut idx = SelfContextIndex::default();
        let original: Vec<u32> = (0..300).map(|i| (i * 7) % 61).collect();
        idx.sync(&original);
        let synced_before = idx.synced;

        // Rolled back to 200 and regenerated DIFFERENT tokens past the old
        // length. Length alone cannot see this; the fingerprint must.
        let mut regrown = original[..200].to_vec();
        regrown.extend((0..150).map(|i| 1000 + i));
        assert!(
            regrown.len() > synced_before,
            "the regrowth passes the old length"
        );
        idx.sync(&regrown);
        assert_eq!(
            idx.indexed(),
            regrown.len(),
            "a diverged history must be reindexed, not appended to"
        );
    }

    #[test]
    fn the_window_stays_bounded_on_a_long_history() {
        let mut idx = SelfContextIndex::default();
        let cap = config().max_tokens;
        let tokens: Vec<u32> = (0..(cap + cap / 2) as u32).map(|i| i % 997).collect();
        idx.sync(&tokens);
        assert!(
            idx.indexed() <= cap,
            "indexed {} exceeds the {cap}-token window",
            idx.indexed()
        );
        // The window is anchored to the RECENT tail, which is what a
        // continuing generation matches against.
        assert_eq!(idx.window_start + idx.indexed(), tokens.len());
    }

    #[test]
    fn prompt_lookup_answers_from_the_same_index() {
        let mut idx = SelfContextIndex::default();
        // [1,2,3,4,5,1,2,3] — suffix [1,2,3] occurred, followed by 4.
        let tokens = vec![1, 2, 3, 4, 5, 1, 2, 3];
        assert_eq!(idx.propose_one(&tokens, 2), Some(4));
        assert_eq!(idx.propose_one(&[1, 2, 3, 4, 5, 6, 7, 8], 2), None);
        assert_eq!(idx.propose_one(&[1, 2], 2), None);
        assert_eq!(
            idx.propose_one(&[10, 20, 30, 10, 20, 30, 10, 20], 2),
            Some(30)
        );
    }
}
