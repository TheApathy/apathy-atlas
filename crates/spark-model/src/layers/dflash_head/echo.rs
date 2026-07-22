// SPDX-License-Identifier: AGPL-3.0-only

//! ECHO-DRAFTING (`ATLAS_DFLASH_ECHO=1`, default off) — pure logic.
//!
//! PROJECT-150 Phase A / BREAKTHROUGH-IDEAS Tier 1 #1: mine the discarded
//! verify logits. Every DFlash K=γ verify computes the target's OWN argmax
//! at ALL K positions (`verified[]`). On a rejection at draft index
//! `num_accepted`, `verified[num_accepted]` becomes the bonus and
//! `verified[num_accepted+1..]` are the target's own next-token choices
//! conditioned on the (near-miss) draft prefix — after a one-token
//! substitution they are usually still right. ECHO re-offers exactly those
//! tokens as the NEXT step's draft chain: a target-authored draft at zero
//! propose cost (the drafter forward — the 25-50ms slice — is skipped).
//!
//! LOSSLESS by construction: echo tokens are only PROPOSED; the verify is a
//! greedy oracle that commits solely the target's argmax, so a wrong echo
//! costs one rejected speculation and can never change committed output.
//!
//! This module holds the PURE parts (config parsing, tail extraction,
//! stash/offer gating, γ padding) so they are unit-testable GPU-free. The
//! stash side lives in `Model::dflash_stash_echo` (trait_impl/mod.rs, called
//! from `verify_dflash_step.rs` on the FLAT chain path only); the offer side
//! lives in `propose_drafts` (propose.rs), mirroring the
//! `ATLAS_DFLASH_RECYCLE` pattern — which also keeps the drafter's
//! ctx-hidden append running on echo steps (the append precedes every
//! early-return in `propose_drafts`), so the drafter is never ctx-starved.

/// Parsed ECHO gate configuration (read once at first use).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EchoConfig {
    /// Fire only when the rejected step accepted at least this many drafts
    /// (the prefix was mostly right, so the salvaged tail is plausibly
    /// good). `ATLAS_DFLASH_ECHO_MIN_ACCEPT`, default 2.
    pub min_accept: usize,
    /// Fire only when the salvaged tail has at least this many tokens
    /// (short tails aren't worth pre-empting the drafter for).
    /// `ATLAS_DFLASH_ECHO_MIN_TAIL`, default 4.
    pub min_tail: usize,
    /// Maximum CONSECUTIVE echo offers before the real drafter must run
    /// (anti-degenerate-loop: an echo step salvaging its own wreckage
    /// forever). `ATLAS_DFLASH_ECHO_MAX_STREAK`, default 2.
    pub max_streak: u32,
}

impl Default for EchoConfig {
    fn default() -> Self {
        Self {
            min_accept: 2,
            min_tail: 4,
            max_streak: 2,
        }
    }
}

impl EchoConfig {
    /// `None` when `ATLAS_DFLASH_ECHO` != "1" (feature off ⇒ byte-identical
    /// default path). Cached after the first read.
    pub fn from_env() -> Option<EchoConfig> {
        static CFG: std::sync::OnceLock<Option<EchoConfig>> = std::sync::OnceLock::new();
        *CFG.get_or_init(|| {
            if std::env::var("ATLAS_DFLASH_ECHO").ok().as_deref() != Some("1") {
                return None;
            }
            let d = EchoConfig::default();
            let cfg = EchoConfig {
                min_accept: parse_env("ATLAS_DFLASH_ECHO_MIN_ACCEPT", d.min_accept),
                min_tail: parse_env("ATLAS_DFLASH_ECHO_MIN_TAIL", d.min_tail).max(1),
                max_streak: parse_env("ATLAS_DFLASH_ECHO_MAX_STREAK", d.max_streak).max(1),
            };
            tracing::info!(
                "ATLAS_DFLASH_ECHO active: min_accept={} min_tail={} max_streak={} \
                 (target-authored salvage drafts; lossless, default-off)",
                cfg.min_accept,
                cfg.min_tail,
                cfg.max_streak,
            );
            Some(cfg)
        })
    }

    /// Boolean form for the scheduler-side stash gate.
    #[allow(dead_code)] // used by atlas-src's scheduler; ours gates via from_env
    pub fn enabled() -> bool {
        Self::from_env().is_some()
    }
}

fn parse_env<T: std::str::FromStr + Copy>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<T>().ok())
        .unwrap_or(default)
}

/// The echo candidate: the target's argmaxes at the positions AFTER the
/// bonus. `verified` is the flat-chain verify output
/// `[argmax_after_last_token, argmax_after_draft_0, ...]` (length γ+1);
/// `verified[num_accepted]` is the bonus (committed), so the salvageable
/// tail starts at `num_accepted + 1`. Empty on full accept (nothing was
/// rejected ⇒ nothing to salvage) or out-of-range `num_accepted`.
pub fn echo_tail(verified: &[u32], num_accepted: usize) -> &[u32] {
    let start = num_accepted.saturating_add(1);
    if start >= verified.len() {
        return &[];
    }
    &verified[start..]
}

/// Stash-side gate: salvage only when the prefix was mostly right
/// (`num_accepted >= min_accept`) AND the tail is long enough to be worth
/// pre-empting the drafter (`tail_len >= min_tail`).
pub fn should_stash(cfg: &EchoConfig, num_accepted: usize, tail_len: usize) -> bool {
    num_accepted >= cfg.min_accept && tail_len >= cfg.min_tail
}

/// Offer-side gate: the stash must be valid for THIS step (the committed
/// bonus equals the stash key — grammar/think interventions can replace
/// it), non-empty, and the consecutive-echo streak below the cap.
pub fn should_offer(
    cfg: &EchoConfig,
    valid: bool,
    key_match: bool,
    tail_len: usize,
    streak: u32,
) -> bool {
    valid && key_match && tail_len > 0 && streak < cfg.max_streak
}

/// Shape the echo tail to exactly `gamma_eff` drafts so the K=γ_eff+1
/// verify CUDA graph is never re-captured (same contract as the recycle
/// offer): truncate a longer tail, pad a shorter one with its last token
/// (a harmless repeat — the verify oracle rejects it for free if wrong).
/// Empty input returns empty (caller never offers an empty tail).
pub fn pad_tail_to_gamma(tail: &[u32], gamma_eff: usize) -> Vec<u32> {
    let Some(&last) = tail.last() else {
        return Vec::new();
    };
    (0..gamma_eff)
        .map(|i| tail.get(i).copied().unwrap_or(last))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> EchoConfig {
        EchoConfig::default() // min_accept=2, min_tail=4, max_streak=2
    }

    // ── echo_tail extraction ──

    #[test]
    fn tail_is_tokens_after_the_bonus() {
        // γ=6 chain, verified has γ+1 rows. Rejection at draft index 2:
        // verified[2] is the bonus; the salvage = verified[3..7].
        let verified = [10, 11, 12, 13, 14, 15, 16];
        assert_eq!(echo_tail(&verified, 2), &[13, 14, 15, 16]);
    }

    #[test]
    fn tail_empty_on_full_accept() {
        // num_accepted == γ ⇒ bonus is the last verified row ⇒ no tail.
        let verified = [10, 11, 12, 13, 14, 15, 16];
        assert_eq!(echo_tail(&verified, 6), &[] as &[u32]);
    }

    #[test]
    fn tail_empty_when_rejection_at_last_draft() {
        let verified = [10, 11, 12];
        // Reject at draft index 1 (γ=2): bonus = verified[1]... tail = [12].
        assert_eq!(echo_tail(&verified, 1), &[12]);
        // Reject at the final row index: nothing after the bonus.
        assert_eq!(echo_tail(&verified, 2), &[] as &[u32]);
    }

    #[test]
    fn tail_handles_out_of_range_and_empty() {
        assert_eq!(echo_tail(&[], 0), &[] as &[u32]);
        assert_eq!(echo_tail(&[7], 0), &[] as &[u32]);
        assert_eq!(echo_tail(&[7, 8], usize::MAX), &[] as &[u32]);
    }

    #[test]
    fn tail_immediate_rejection_salvages_rest() {
        // num_accepted = 0: bonus = verified[0], tail = verified[1..].
        let verified = [10, 11, 12, 13];
        assert_eq!(echo_tail(&verified, 0), &[11, 12, 13]);
    }

    // ── stash gating ──

    #[test]
    fn stash_requires_accept_floor() {
        let c = cfg();
        assert!(!should_stash(&c, 0, 10));
        assert!(!should_stash(&c, 1, 10));
        assert!(should_stash(&c, 2, 10));
        assert!(should_stash(&c, 15, 10));
    }

    #[test]
    fn stash_requires_min_tail() {
        let c = cfg();
        assert!(!should_stash(&c, 5, 0));
        assert!(!should_stash(&c, 5, 3));
        assert!(should_stash(&c, 5, 4));
    }

    // ── offer gating (streak cap) ──

    #[test]
    fn offer_requires_valid_stash_and_key_match() {
        let c = cfg();
        assert!(should_offer(&c, true, true, 5, 0));
        assert!(!should_offer(&c, false, true, 5, 0));
        assert!(!should_offer(&c, true, false, 5, 0));
        assert!(!should_offer(&c, true, true, 0, 0));
    }

    #[test]
    fn offer_streak_capped() {
        let c = cfg(); // max_streak = 2
        assert!(should_offer(&c, true, true, 5, 0));
        assert!(should_offer(&c, true, true, 5, 1));
        assert!(!should_offer(&c, true, true, 5, 2));
        assert!(!should_offer(&c, true, true, 5, 3));
    }

    // ── γ padding ──

    #[test]
    fn pad_extends_short_tail_with_last_token() {
        assert_eq!(pad_tail_to_gamma(&[1, 2, 3], 6), vec![1, 2, 3, 3, 3, 3]);
    }

    #[test]
    fn pad_truncates_long_tail() {
        assert_eq!(pad_tail_to_gamma(&[1, 2, 3, 4, 5], 3), vec![1, 2, 3]);
    }

    #[test]
    fn pad_exact_length_is_identity() {
        assert_eq!(pad_tail_to_gamma(&[9, 8], 2), vec![9, 8]);
    }

    #[test]
    fn pad_empty_tail_stays_empty() {
        assert_eq!(pad_tail_to_gamma(&[], 16), Vec::<u32>::new());
    }

    // ── config parsing ──

    #[test]
    fn default_config_matches_policy() {
        let c = EchoConfig::default();
        assert_eq!(c.min_accept, 2);
        assert_eq!(c.min_tail, 4);
        assert_eq!(c.max_streak, 2);
    }
}
