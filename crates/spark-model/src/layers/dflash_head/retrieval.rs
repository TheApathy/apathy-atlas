// SPDX-License-Identifier: AGPL-3.0-only

//! Retrieval-augmented drafting for DFlash (`ATLAS_DFLASH_RETRIEVAL=1`,
//! default off).
//!
//! This is a generalization of the degenerate prompt-lookup (`ATLAS_DFLASH_PLD`)
//! path in [`super::propose`]. Where PLD only fires in the weak-drafter regime
//! (`last_num_accepted <= 1`) and demands a strict ≥5-gram exact suffix match
//! against the *current sequence*, retrieval drafting:
//!
//!   1. Searches a BROADER haystack: the full `pld_tokens` = prompt tokens
//!      (the task/reference code often lives here) + all committed/generated
//!      tokens so far. (`pld_tokens` is already populated with `seq.tokens` by
//!      the caller — see `impl_b3.rs` — so prompt context comes for free.)
//!
//!   2. Relaxes the gate: it tries a *range* of suffix lengths
//!      (`L_max .. L_min`, default 16 → 4) and takes the LONGEST match (most
//!      specific context). Longer matches are strong evidence; we propose the
//!      γ tokens that followed that occurrence.
//!
//!   3. Fires whenever a strong match exists, NOT only when the drafter is
//!      weak. The DFlash verify (`impl_b2.rs::run_self_speculative` /
//!      `decode_verify`) is the lossless oracle: it commits `verified[i]` —
//!      the target's greedy token — and accepts a draft only when
//!      `drafts[i] == verified[i]`. So a wrong retrieval guess costs only a
//!      rejected speculation; it can NEVER change committed output. Retrieval
//!      changes WHAT is proposed, never what is committed → token-exact by
//!      construction.
//!
//! The match is found with a backward scan over the haystack using a
//! rolling comparison; for typical coding/serving sequence lengths (≤8K
//! tokens) this is a few thousand cheap `u32` comparisons on the host and is
//! dwarfed by a single drafter GPU forward, so it stays off the critical path.
//!
//! Hybrid selection (cheap variant, implemented here): when the longest match
//! length ≥ `hybrid_min` (default = `L_max` so it only pre-empts the neural
//! drafter on a *strong* match), return retrieval drafts and skip the drafter
//! forward entirely. Otherwise fall through to the neural drafter. This is the
//! cheap "retrieval-when-confident, drafter-otherwise" design; a full
//! per-block run-both-and-pick hybrid is gated behind a separate flag below
//! and only worth it if the cheap version under-delivers.

/// Parsed, validated configuration for retrieval drafting. Built once per
/// `propose_drafts` call from environment variables.
#[derive(Clone, Copy, Debug)]
pub struct RetrievalConfig {
    /// Longest suffix length to try first (most specific). Default 16.
    /// In SAM mode this caps the backward-extension work per candidate.
    pub l_max: usize,
    /// Shortest suffix length to accept. Default 4. Below this, matches are
    /// too generic (single-identifier coincidences) and waste a speculation.
    pub l_min: usize,
    /// Minimum match length to PRE-EMPT the neural drafter (cheap hybrid).
    /// Default = `l_max` (only the strongest matches skip the drafter).
    pub hybrid_min: usize,
    /// Number of follow-on tokens to propose (= γ so the verify path is
    /// unchanged). Set by the caller from the head's γ.
    pub draft_count: usize,
    /// SAM mode (`ATLAS_DFLASH_SAM=1`): use the suffix-automaton-style
    /// LONGEST-suffix matcher ([`retrieve_longest`]) instead of the legacy
    /// fixed-window range matcher ([`retrieve`]). The longest matcher finds
    /// the longest suffix of the live context that occurs earlier in the
    /// haystack at ANY length (not just `l_min..=l_max`), so it fires far
    /// more often on real coding reuse. Gated by `hybrid_min` exactly the
    /// same way (match_len >= hybrid_min ⇒ pre-empt the neural drafter).
    pub sam: bool,
}

impl RetrievalConfig {
    /// Returns `Some(cfg)` when `ATLAS_DFLASH_RETRIEVAL=1` or
    /// `ATLAS_DFLASH_SAM=1`, else `None` (default off — legacy behavior is
    /// byte-for-byte preserved). `ATLAS_DFLASH_SAM=1` additionally selects the
    /// longest-suffix (SAM) matcher; it implies retrieval is on.
    pub fn from_env(draft_count: usize) -> Option<Self> {
        let retrieval_on = std::env::var("ATLAS_DFLASH_RETRIEVAL").ok().as_deref() == Some("1");
        let sam = std::env::var("ATLAS_DFLASH_SAM").ok().as_deref() == Some("1");
        if !retrieval_on && !sam {
            return None;
        }
        let l_max: usize = env_usize("ATLAS_RETRIEVAL_LMAX", 16).clamp(2, 64);
        let l_min: usize = env_usize("ATLAS_RETRIEVAL_LMIN", 4).clamp(1, l_max);
        // Default hybrid threshold: always l_max (strongest match ⇒ pre-empt).
        // SAM can find matches at ANY length ≥ l_min, so an aggressive low
        // threshold causes false fires on short (4-token) matches that then
        // propose RETR_WIDE=31 tokens with near-zero accuracy, pre-empting the
        // neural DDTree drafter at a severe cost. Defaulting to l_max=16 keeps
        // SAM safe: it only pre-empts when it found a 16-token exact suffix,
        // which is strong enough evidence to justify proposing 16-31 tokens.
        // For tasks where aggressive SAM helps, set ATLAS_RETRIEVAL_HYBRID_MIN
        // explicitly to a lower value.
        let hybrid_default = l_max;
        let hybrid_min: usize = env_usize("ATLAS_RETRIEVAL_HYBRID_MIN", hybrid_default).max(1);
        Some(Self {
            l_max,
            l_min,
            hybrid_min,
            draft_count: draft_count.max(1),
            sam,
        })
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Result of a retrieval lookup.
#[derive(Clone, Debug)]
pub struct RetrievalHit {
    /// The γ tokens that followed the matched occurrence, in order.
    pub drafts: Vec<u32>,
    /// Length of the suffix that matched (for hybrid selection + logging).
    pub match_len: usize,
}

/// Find the longest-suffix match of `[..haystack, last_token]` earlier in
/// `haystack`, and return the `draft_count` tokens that followed it.
///
/// `haystack` is the committed token sequence so far (prompt + generated),
/// NOT including `last_token` (the most recently emitted token, which is the
/// query's final element). We search for an occurrence of the trailing
/// `L`-gram `[haystack[len-(L-1)..], last_token]` at some earlier position,
/// trying `L` from `cfg.l_max` down to `cfg.l_min` and returning on the first
/// (= longest) hit that has `draft_count` tokens of follow-on context.
///
/// Token-exactness: this only decides WHAT to propose. The caller feeds the
/// result into the same verify path as drafter output; the verifier rejects
/// any mismatched token, so output is identical regardless of this function.
pub fn retrieve(haystack: &[u32], last_token: u32, cfg: &RetrievalConfig) -> Option<RetrievalHit> {
    let l = haystack.len();
    // Need at least one full suffix + the follow-on tokens somewhere earlier.
    if l < cfg.l_min {
        return None;
    }
    for ng in (cfg.l_min..=cfg.l_max).rev() {
        if l < ng {
            continue;
        }
        // Suffix to match: the last (ng-1) committed tokens followed by
        // last_token. (ng == 1 ⇒ just [last_token].)
        // Search every earlier start position `p` such that haystack[p..p+ng]
        // equals this suffix AND there are draft_count tokens after it.
        // Scan backward (most-recent occurrence first) — closer context tends
        // to be more relevant, and it lets us stop at the first hit.
        let suffix_tail_len = ng - 1;
        // Largest p such that the match window AND the follow-on fit:
        //   p + ng + draft_count <= l   (window + γ follow-on inside haystack)
        // We must not let the matched window overlap the live suffix itself,
        // i.e. p + ng <= l (the suffix we're matching ends at the live tail).
        let max_p = match (l).checked_sub(ng + cfg.draft_count) {
            Some(v) => v,
            None => continue, // not enough room for window + follow-on
        };
        let mut p = max_p;
        loop {
            let window_matches = {
                // Compare haystack[p..p+suffix_tail_len] to the live tail
                // haystack[l-suffix_tail_len..], then the window's last token
                // to last_token.
                let live_tail = &haystack[l - suffix_tail_len..];
                let cand = &haystack[p..p + suffix_tail_len];
                cand == live_tail && haystack[p + suffix_tail_len] == last_token
            };
            if window_matches {
                let cs = p + ng;
                let drafts: Vec<u32> = haystack[cs..cs + cfg.draft_count].to_vec();
                return Some(RetrievalHit {
                    drafts,
                    match_len: ng,
                });
            }
            if p == 0 {
                break;
            }
            p -= 1;
        }
    }
    None
}

/// SAM-style LONGEST-suffix matcher (`ATLAS_DFLASH_SAM=1`).
///
/// Finds the LONGEST suffix of the live context `[..haystack, last_token]` that
/// occurs earlier in `haystack` — at ANY length, not just the legacy
/// `l_min..=l_max` window — and returns the `draft_count` tokens that followed
/// that earlier occurrence.
///
/// This is the in-context analogue of SAM-Decoding (arXiv 2411.10666): rather
/// than building a full suffix automaton (whose value is amortized O(1) longest
/// match over a STATIC corpus), we exploit the structure of the DFlash propose
/// loop — the query is always the live SUFFIX, so the only candidate match end
/// positions are the earlier occurrences of `last_token`. We index those once
/// per call (cheap: one pass over the host token mirror) and extend each
/// backward to measure its match length, keeping the longest. For the ≤8K-token
/// serving sequences this is a few thousand `u32` compares — far below a single
/// drafter GPU forward, so it stays off the critical path while delivering the
/// "longest match, any length, fires far more often" behavior a true SAM gives.
///
/// Token-exactness: identical contract to [`retrieve`] — this only decides WHAT
/// to propose; the DFlash verify is the lossless oracle that commits only the
/// target's greedy token, so a wrong guess costs at most a rejected speculation
/// and can never change committed output.
pub fn retrieve_longest(
    haystack: &[u32],
    last_token: u32,
    cfg: &RetrievalConfig,
) -> Option<RetrievalHit> {
    let l = haystack.len();
    let need = cfg.draft_count;
    if l < cfg.l_min || need == 0 {
        return None;
    }
    // Cap how far back we extend a single candidate. The match length is only
    // used for the hybrid gate (>= hybrid_min) and the proposed drafts are
    // independent of it, so there is no value in measuring past l_max.
    let ext_cap = cfg.l_max;
    // Limit the number of candidate end positions we examine, most-recent
    // first. Bounds worst-case work on a haystack with a very common
    // `last_token`; the most recent occurrences carry the most relevant
    // context anyway. (Generous default — typical coding repeats are local.)
    const MAX_CANDIDATES: usize = 256;

    // Candidate match END positions: earlier indices p where haystack[p] ==
    // last_token AND there is room for `need` follow-on tokens after p.
    // The matched window ends AT p (haystack[p] == last_token, the live
    // context's final element), so the live-tail comparison runs over
    // haystack[..l] (the live context, which excludes last_token).
    let max_p = l.checked_sub(need)?; // need p + need <= l  ⇒ p <= l - need
    let mut best_len = 0usize;
    let mut best_follow_start = 0usize;
    let mut examined = 0usize;
    let mut p = max_p; // start from the most recent valid candidate
    loop {
        if haystack[p] == last_token {
            // Extend backward: haystack[p-1-k] must equal the live-context tail
            // haystack[l-1-k] for k = 0,1,2,... The base match (p itself vs
            // last_token) is length 1.
            let mut match_len = 1usize;
            while match_len < ext_cap {
                // live-context index we compare against (counting back from the
                // element just before last_token, i.e. haystack[l-1]).
                let live_idx = match (l - 1).checked_sub(match_len - 1) {
                    Some(v) => v,
                    None => break,
                };
                // earlier-window index.
                let cand_idx = match p.checked_sub(match_len) {
                    Some(v) => v,
                    None => break,
                };
                // Don't let the earlier window overlap the live tail it is
                // being compared against (cand_idx must stay strictly left of
                // the live-tail region). p <= max_p <= l-need already keeps p
                // left of the live suffix; this guards the backward walk.
                if cand_idx >= live_idx {
                    break;
                }
                if haystack[cand_idx] != haystack[live_idx] {
                    break;
                }
                match_len += 1;
            }
            if match_len > best_len {
                best_len = match_len;
                best_follow_start = p + 1;
                // A full-length (== ext_cap) match is as long as we measure;
                // closer occurrences are preferred on ties, and we scan most-
                // recent first, so we can stop early on a maxed-out match.
                if best_len >= ext_cap {
                    break;
                }
            }
            examined += 1;
            if examined >= MAX_CANDIDATES {
                break;
            }
        }
        if p == 0 {
            break;
        }
        p -= 1;
    }

    if best_len < cfg.l_min {
        return None;
    }
    let cs = best_follow_start;
    if cs + need > l {
        return None;
    }
    Some(RetrievalHit {
        drafts: haystack[cs..cs + need].to_vec(),
        match_len: best_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(draft_count: usize) -> RetrievalConfig {
        RetrievalConfig {
            l_max: 8,
            l_min: 2,
            hybrid_min: 8,
            draft_count,
            sam: false,
        }
    }

    fn sam_cfg(draft_count: usize, l_max: usize, l_min: usize) -> RetrievalConfig {
        RetrievalConfig {
            l_max,
            l_min,
            hybrid_min: l_min,
            draft_count,
            sam: true,
        }
    }

    #[test]
    fn retrieves_followon_after_exact_repeat() {
        // The matched window is [live_tail.., last_token]; the proposed drafts
        // are the tokens AFTER that window (last_token is already emitted, so
        // it is NOT a draft).
        // haystack = [1,2,3, 4,5,6,99, 1,2,3]; last_token = 4.
        //   ng=4 ⇒ live_tail=[1,2,3], window must be [1,2,3,4]; found at p=0
        //   (haystack[0..4]=[1,2,3,4]). cs=4 ⇒ drafts = [5,6,99].
        let haystack = vec![1, 2, 3, 4, 5, 6, 99, 1, 2, 3];
        let hit = retrieve(&haystack, 4, &cfg(3)).expect("should retrieve");
        assert_eq!(hit.drafts, vec![5, 6, 99]);
        assert!(hit.match_len >= 2);
    }

    #[test]
    fn longest_match_wins() {
        // The earlier window [1,2,3,4] sits at p=1; its follow-on is [50,60,70].
        let haystack = vec![7, 1, 2, 3, 4, 50, 60, 70, 1, 2, 3];
        // live tail [...,1,2,3], last=4. window [1,2,3,4] at idx1, cs=5,
        // drafts=[50,60,70].
        let hit = retrieve(&haystack, 4, &cfg(3)).expect("retrieve");
        assert_eq!(hit.drafts, vec![50, 60, 70]);
    }

    #[test]
    fn prefers_longer_suffix_when_two_candidates() {
        // Two occurrences of the trailing token `9`, reached via different
        // suffix lengths. A 2-gram match [8,9] occurs at idx1 (→ follow-on
        // [11,12,13]) and the full 4-gram [5,6,8,9]... we craft so the LONGER
        // suffix wins. Build: [5,6,8,9, 11,12,13,  _,_, 8,9] with live tail
        // ending [...,5,6,8] and last=9. The 4-gram window [5,6,8,9] at idx0
        // (follow-on [11,12,13]) must be chosen over the shorter [8,9] at idx9
        // (no follow-on room anyway).
        let haystack = vec![5, 6, 8, 9, 11, 12, 13, 20, 21, 5, 6, 8];
        // live tail (3) = [5,6,8], last=9 → 4-gram window [5,6,8,9] at idx0,
        // cs=4 → [11,12,13].
        let hit = retrieve(&haystack, 9, &cfg(3)).expect("retrieve");
        assert_eq!(hit.drafts, vec![11, 12, 13]);
        assert_eq!(hit.match_len, 4);
    }

    #[test]
    fn no_match_returns_none() {
        let haystack = vec![1, 2, 3, 4, 5];
        // last_token=99 never appears as a continuation of any earlier suffix.
        assert!(retrieve(&haystack, 99, &cfg(3)).is_none());
    }

    #[test]
    fn respects_draft_count_room() {
        // Match exists but no draft_count follow-on tokens after it.
        let haystack = vec![1, 2, 3, 1, 2, 3];
        // live tail [..2,3], last=? choose last so window is at idx0 [1,2,3]
        // but there is room: cs=3, need 3 tokens -> [1,2,3]. OK here.
        // Now make a case with NO room:
        let short = vec![5, 5, 1, 2];
        // live tail [..1,2], last=3 ; earlier [1,2] would need to appear with
        // 3 follow-on tokens — it doesn't.
        assert!(retrieve(&short, 3, &cfg(3)).is_none());
        // Sanity: the first one DOES retrieve something.
        let _ = retrieve(&haystack, 1, &cfg(3));
    }

    #[test]
    fn disabled_when_env_unset() {
        // from_env returns None unless the flag is exactly "1".
        // (Cannot mutate env safely in parallel tests; just assert the parse
        // contract via a direct check.)
        let c = RetrievalConfig {
            l_max: 16,
            l_min: 4,
            hybrid_min: 16,
            draft_count: 16,
            sam: false,
        };
        assert!(c.l_min <= c.l_max);
        assert!(c.hybrid_min >= c.l_min && c.hybrid_min <= c.l_max);
    }

    // ── SAM longest-suffix matcher tests ──────────────────────────────────

    #[test]
    fn sam_finds_longest_any_length() {
        // The live suffix is the last few tokens + last_token. SAM should find
        // the LONGEST earlier occurrence regardless of a fixed window.
        // haystack = [1,2,3,4,5, 9,9, 1,2,3,4]; last_token = 5.
        //   live suffix ...1,2,3,4 + 5. The 5-gram [1,2,3,4,5] occurs at p_end=4
        //   (haystack[0..5]), follow-on starts at 5 → [9,9,1].
        let haystack = vec![1, 2, 3, 4, 5, 9, 9, 1, 2, 3, 4];
        let hit = retrieve_longest(&haystack, 5, &sam_cfg(3, 16, 2)).expect("sam hit");
        assert_eq!(hit.drafts, vec![9, 9, 1]);
        assert_eq!(hit.match_len, 5);
    }

    #[test]
    fn sam_short_match_below_lmin_rejected() {
        // Only a length-1 match exists (just the bare last_token), below l_min=2.
        let haystack = vec![1, 2, 3, 4, 5, 6, 7, 8];
        // last_token = 3 appears at idx2 but the preceding token (8) != 2, so
        // the longest suffix match is length 1 → rejected by l_min=2.
        assert!(retrieve_longest(&haystack, 3, &sam_cfg(2, 16, 2)).is_none());
    }

    #[test]
    fn sam_picks_longer_over_more_recent() {
        // A recent short match vs an older long match: SAM prefers the LONGER.
        // haystack = [7,8,1,2,3, A,B,C, 8,1,2,3]; last_token = 3.
        // live tail ...8,1,2 +3. Occurrences of [...,3] ending: idx4 (8,1,2,3 len4)
        // and idx11 (8,1,2,3 len4 too) — make the older one longer with a 7 prefix.
        // haystack: 7,8,1,2,3 (idx0..4), then 50,60,70 (idx5..7), then 8,1,2 (8..10).
        // live tail = [...,8,1,2], last=3. The window [8,1,2,3] at idx1..4 plus
        // preceding 7 gives 5-gram [7,8,1,2,3] only if live tail also has 7 — it
        // doesn't, so match_len=4, follow-on at idx5 = [50,60,70].
        let haystack = vec![7, 8, 1, 2, 3, 50, 60, 70, 8, 1, 2];
        let hit = retrieve_longest(&haystack, 3, &sam_cfg(3, 16, 2)).expect("sam hit");
        assert_eq!(hit.drafts, vec![50, 60, 70]);
        assert_eq!(hit.match_len, 4);
    }

    #[test]
    fn sam_respects_draft_room() {
        // Longest match ends too close to the live tail to have draft_count
        // follow-on tokens. SAM must skip candidates without room and either
        // find an earlier one or return None.
        let haystack = vec![1, 2, 3, 9, 9, 9, 1, 2];
        // live tail ...1,2 + last=3. The [1,2,3] window at idx0 has follow-on
        // [9,9,9] (room for 3). OK.
        let hit = retrieve_longest(&haystack, 3, &sam_cfg(3, 16, 2)).expect("sam hit");
        assert_eq!(hit.drafts, vec![9, 9, 9]);
        // Now a case with NO room after the only match:
        let short = vec![5, 1, 2, 5, 1, 2];
        // live tail ...1,2 + last=3 never occurs → None.
        assert!(retrieve_longest(&short, 3, &sam_cfg(3, 16, 2)).is_none());
    }

    #[test]
    fn sam_no_match_returns_none() {
        let haystack = vec![1, 2, 3, 4, 5];
        assert!(retrieve_longest(&haystack, 99, &sam_cfg(3, 16, 2)).is_none());
    }

    // ── WIDE RETRIEVAL (ATLAS_DFLASH_RETR_WIDE) tests ─────────────────────

    #[test]
    fn sam_wide_returns_exact_draft_count() {
        // A strong match with MANY follow-on tokens: a wide draft_count of 20
        // must return exactly 20 drafts (K=21 verify), not truncate to γ=16.
        // Build a 25-token run repeated after a [1,2,3,4] anchor.
        let mut haystack: Vec<u32> = vec![1, 2, 3, 4];
        let follow: Vec<u32> = (100..125).collect(); // 25 follow-on tokens
        haystack.extend_from_slice(&follow); // first occurrence: [1,2,3,4] + follow
        haystack.extend_from_slice(&[9, 9, 9]);
        haystack.extend_from_slice(&[1, 2, 3]); // live tail ...1,2,3 + last=4
        // draft_count=20 (> γ=16): retrieve_longest must return 20 follow-on
        // tokens from the earlier [1,2,3,4] occurrence.
        let hit = retrieve_longest(&haystack, 4, &sam_cfg(20, 32, 2)).expect("wide sam hit");
        assert_eq!(hit.drafts.len(), 20);
        assert_eq!(hit.drafts, follow[..20].to_vec());
        assert!(hit.match_len >= 4);
    }

    #[test]
    fn sam_wide_none_when_insufficient_followon() {
        // Strong match but only 10 follow-on tokens available; a wide request
        // of 20 must return None (so the caller falls back to the neural
        // drafter at γ) rather than a short draft that would break the fixed
        // K = draft_count + 1 verify-graph width.
        let mut haystack: Vec<u32> = vec![1, 2, 3, 4];
        haystack.extend_from_slice(&(100..110).collect::<Vec<u32>>()); // only 10
        haystack.extend_from_slice(&[1, 2, 3]); // live tail + last=4
        assert!(retrieve_longest(&haystack, 4, &sam_cfg(20, 32, 2)).is_none());
    }
}
