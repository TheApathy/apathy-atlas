// SPDX-License-Identifier: AGPL-3.0-only

//! Low-gear speculation for adaptive-suspended sequences
//! (`ATLAS_DFLASH_LOW_GEAR=1`, default off).
//!
//! When `ATLAS_DFLASH_ADAPTIVE=1` suspends full DFlash speculation (rolling
//! accept below MIN — typical on prose), the suspended sequence serial-decodes
//! at ~1 token / ~50ms step. This module recovers part of the speculation win
//! at near-zero cost: a pure HOST-side longest-suffix n-gram lookup against
//! the sequence's own committed tokens (`seq.tokens`, prompt + generated —
//! the same haystack the DFlash PLD/retrieval drafters search) proposes 1-2
//! draft tokens, and the EXISTING graphed K=2/K=3 verify runs instead of the
//! serial decode. K=2 verify costs serial + ~5-10% (one extra row through the
//! MoE union — ~11-12 experts vs ~10), so at draft accept rate p the
//! suspended regime speeds up by ~(1+p)/1.1: p=0.3 ⇒ prose 20 → ~24 tok/s.
//!
//! Losslessness: identical contract to DFlash retrieval — the lookup only
//! decides WHAT is proposed; the verify argmax is the oracle that decides
//! what commits. A wrong guess costs one rejected row, never a wrong token.
//!
//! Bookkeeping choices (deliberate, see `step_low_gear`):
//!  * Adaptive window: low-gear steps COUNT as suspended-serial progress
//!    (`tick_serial` once per committed token) so the re-probe cadence is
//!    unchanged in tokens. They are NOT fed to `record_verify` — that window
//!    measures the NEURAL drafter's accept quality to decide suspend/resume,
//!    and host n-gram accepts would pollute the re-probe verdict.
//!  * DFlash ctx accumulator: the fused verify captures per-row target
//!    hiddens into `dflash_hidden_save` (same capture the K=2 EAGLE append
//!    uses), and we commit them via `Model::commit_ctx` — the hole-immune
//!    multi-row append (explicit positions + `skip_next_decode_append`),
//!    fired iff a serial ctx-append mode is active
//!    (`ATLAS_DFLASH_UNIFIED_CTX` or `ATLAS_DFLASH_SERIAL_APPEND`).
//!    `dflash_serial_ctx_append` is NOT usable here: it appends exactly one
//!    row stamped `seq_len-1`, wrong for a 2-token commit.
//!  * Drafter stats (`after_verify` / `trim_proposer_state`) are left
//!    untouched: these accepts are not drafter accepts, and
//!    `last_num_accepted` gates the PLD weak-drafter path and the SAM
//!    misfire cooldown.
//!
//! Knobs (env, read once): `ATLAS_DFLASH_LOW_GEAR=1` master switch;
//! `ATLAS_DFLASH_LOW_GEAR_MIN` minimum n-gram match length (default 3);
//! `ATLAS_DFLASH_LOW_GEAR_K` verify width 2 or 3 (default 2 ⇒ 1 draft).

use super::*;

/// Max drafts low gear will ever submit (K=3 verify width).
pub(crate) const MAX_LOW_GEAR_DRAFTS: usize = 2;

/// How far back a candidate match is extended when measuring match length.
/// Longer measured matches only improve candidate RANKING (longest wins), so
/// a modest cap bounds host work without changing what can fire.
const EXT_CAP: usize = 32;

/// Bound on candidate end-positions examined per lookup (most-recent first).
/// Mirrors `retrieval.rs::MAX_CANDIDATES` — bounds worst-case host work on a
/// haystack with a very common `last_token`.
const MAX_CANDIDATES: usize = 256;

fn enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_DFLASH_LOW_GEAR").ok().as_deref() == Some("1"))
}

/// Minimum longest-suffix match length required to fire (default 3).
fn min_match() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_LOW_GEAR_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3)
            .max(1)
    })
}

/// Number of drafts submitted per low-gear step: verify width K-1, K ∈ {2,3}.
fn draft_need() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        let k: usize = std::env::var("ATLAS_DFLASH_LOW_GEAR_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        k.clamp(2, 1 + MAX_LOW_GEAR_DRAFTS) - 1
    })
}

/// A host-side draft found by [`lookup_drafts`]. Fixed-size, allocation-free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LowGearHit {
    pub toks: [u32; MAX_LOW_GEAR_DRAFTS],
    pub len: usize,
    pub match_len: usize,
}

/// Longest-suffix n-gram lookup, pure host, allocation-free.
///
/// Finds the LONGEST suffix of the live context `[..haystack, last_token]`
/// that occurs earlier in `haystack` with at least `need` follow-on tokens,
/// and returns those follow-on tokens as drafts. Candidates are scanned
/// most-recent first; on tie the more recent occurrence wins. Returns `None`
/// unless the best match length is >= `min_match`.
///
/// Adapted from `spark-model` `dflash_head/retrieval.rs::retrieve_longest`
/// (private module — not reachable from this crate), specialized to the
/// low-gear draft widths (`need <= MAX_LOW_GEAR_DRAFTS`).
pub(crate) fn lookup_drafts(
    haystack: &[u32],
    last_token: u32,
    need: usize,
    min_match: usize,
) -> Option<LowGearHit> {
    let l = haystack.len();
    if need == 0 || need > MAX_LOW_GEAR_DRAFTS || l < need.max(1) {
        return None;
    }
    // Candidate match END positions: p where haystack[p] == last_token AND
    // room for `need` follow-on tokens after p (p + need <= l - 1 + 1).
    let max_p = l.checked_sub(need)?;
    let ext_cap = EXT_CAP.max(min_match);
    let mut best_len = 0usize;
    let mut best_follow = 0usize;
    let mut examined = 0usize;
    let mut p = max_p;
    loop {
        if haystack[p] == last_token {
            // Base match (haystack[p] == last_token) has length 1; extend
            // backward comparing against the live-context tail.
            let mut match_len = 1usize;
            while match_len < ext_cap {
                let Some(live_idx) = (l - 1).checked_sub(match_len - 1) else {
                    break;
                };
                let Some(cand_idx) = p.checked_sub(match_len) else {
                    break;
                };
                // The earlier window must stay strictly left of the live
                // tail it is compared against.
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
                best_follow = p + 1;
                if best_len >= ext_cap {
                    break; // maxed-out measurement; most recent wins ties
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
    if best_len < min_match || best_follow + need > l {
        return None;
    }
    let mut toks = [0u32; MAX_LOW_GEAR_DRAFTS];
    toks[..need].copy_from_slice(&haystack[best_follow..best_follow + need]);
    Some(LowGearHit {
        toks,
        len: need,
        match_len: best_len,
    })
}

// ── Telemetry ────────────────────────────────────────────────────────────
// Rate-limited summary every LG_SUMMARY_PERIOD fired steps. Atomics for the
// same reason as verify_k2_step: single scheduler thread today, race-free
// under future multi-scheduler builds.
const LG_SUMMARY_PERIOD: u64 = 128;
static LG_FIRES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LG_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LG_TOK_DRAFTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LG_TOK_ACCEPTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LG_REJECT_STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn lg_record_outcome(n_drafted: usize, n_ok: usize, seq_len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    LG_TOK_DRAFTED.fetch_add(n_drafted as u64, Relaxed);
    LG_TOK_ACCEPTED.fetch_add(n_ok as u64, Relaxed);
    if n_ok == 0 {
        LG_REJECT_STEPS.fetch_add(1, Relaxed);
    }
    let fires = LG_FIRES.fetch_add(1, Relaxed) + 1;
    if fires == 1 {
        tracing::info!(
            "LOW_GEAR ACTIVE: first host-draft verify (min_match={} drafts/step={})",
            min_match(),
            draft_need(),
        );
    }
    if fires % LG_SUMMARY_PERIOD == 0 {
        let drafted = LG_TOK_DRAFTED.swap(0, Relaxed).max(1);
        let accepted = LG_TOK_ACCEPTED.swap(0, Relaxed);
        let rejects = LG_REJECT_STEPS.swap(0, Relaxed);
        let misses = LG_MISSES.swap(0, Relaxed);
        tracing::info!(
            "LOW_GEAR summary: {LG_SUMMARY_PERIOD} fires / {misses} misses, \
             {accepted}/{drafted} draft tokens accepted ({:.1}%), {rejects} full-reject steps, \
             seq_len={seq_len}",
            100.0 * accepted as f64 / drafted as f64,
        );
    }
}

/// Low-gear verify step for an adaptive-suspended sequence.
///
/// Returns `true` when the step consumed this scheduler iteration (a host
/// draft was found and the K=2/K=3 verify ran, or a fatal error finished the
/// sequence); `false` means "no draft — caller serial-decodes as today".
///
/// PRECONDITIONS (enforced by the caller gate in `mtp_step.rs` Phase A):
///  * `dflash_verify_raw_argmax && !dflash_seam_serial_enabled()` — this
///    guarantees `adaptive_spec::spec_allowed` was already evaluated this
///    iteration (the Phase-A bootstrap-propose gate) and returned false, so
///    `is_suspended` is authoritative and the re-probe cadence is untouched.
///  * `a.grammar_state.is_none()` — grammar sequences keep the serial path
///    (single-token decode uses an up-to-date grammar mask; K>1 verify with
///    grammar requires the draft-boundary machinery low gear skips).
///  * sequence is adaptive-suspended (`adaptive_spec::is_suspended`).
pub(crate) fn step_low_gear(
    model: &dyn Model,
    a: &mut ActiveSeq,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) -> bool {
    if !enabled() {
        return false;
    }
    // EP (multi-rank): the graphed verify would keep NCCL lockstep, but only
    // the FUSED verify is known to fill `dflash_hidden_save` rows for the
    // ctx commit below (same capture contract as the K=2 EAGLE append).
    // Suspended EP sequences keep the serial path.
    if model.is_ep() {
        return false;
    }
    let need = draft_need();
    let Some(hit) = lookup_drafts(&a.seq.tokens, a.last_token, need, min_match()) else {
        LG_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return false;
    };
    let n_drafts = hit.len;
    let m = n_drafts + 1; // verify width K

    if let Err(e) = model.sync_secondary() {
        tracing::error!("low_gear sync_secondary: {e:#}");
        a.finished = true;
        return true;
    }

    // EP broadcast kept for protocol parity (no-op single-rank, and correct
    // per-width cmd if the is_ep gate above is ever relaxed).
    let mut tokens = [0u32; 1 + MAX_LOW_GEAR_DRAFTS];
    tokens[0] = a.last_token;
    tokens[1..m].copy_from_slice(&hit.toks[..n_drafts]);
    let tokens = &tokens[..m];
    let cmd = if m == 2 { 0xFFFFFFF2 } else { 0xFFFFFFF3 };
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, cmd) {
        tracing::error!("EP broadcast low_gear cmd: {e:#}");
        a.finished = true;
        return true;
    }
    for &t in tokens {
        if let Err(e) = model.ep_broadcast_cmd(t) {
            tracing::error!("EP broadcast low_gear token: {e:#}");
            a.finished = true;
            return true;
        }
    }

    // Fused single-sweep verify (DFlash, single-rank — guaranteed by the
    // caller gate + is_ep check above). Same forward as step_verify_k2/k3.
    let result_vec: Vec<u32> = match model.decode_and_verify_fused(tokens, &mut a.seq, 0) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("decode_and_verify_fused (low_gear k{m}): {e:#}");
            a.finished = true;
            return true;
        }
    };
    a.last_token_time = Instant::now();

    // Verify basis: identical to K=2/K=3 — raw argmax (GOLD) unless the
    // masked-verify pipeline is enabled.
    let mut vs = [0u32; 1 + MAX_LOW_GEAR_DRAFTS];
    if dflash_verify_raw_argmax
        && !crate::scheduler::verify_pipeline_helper::dflash_masked_verify_enabled()
    {
        vs[..m].copy_from_slice(&result_vec[..m]);
    } else {
        let processed = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            model,
            &result_vec[..m],
            a,
            verify_ctx,
        );
        for i in 0..m {
            vs[i] = processed.get(i).copied().unwrap_or(result_vec[i]);
        }
    }

    // Longest accepted prefix of the host drafts.
    let mut n_ok = 0usize;
    while n_ok < n_drafts && hit.toks[n_ok] == vs[n_ok] {
        n_ok += 1;
    }
    let committed = n_ok + 1; // accepted drafts + correction/bonus token

    let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
        extract_verify_logprobs(model, &vs[..m], top_logprobs)
    } else {
        Vec::new()
    };

    // EP: result broadcast (accept count) — same wire value as k2/k3.
    if let Err(e) = model.ep_broadcast_cmd(n_ok as u32) {
        tracing::error!("EP broadcast low_gear result: {e:#}");
        a.finished = true;
        return true;
    }

    crate::metrics::SPEC_DECODE_VERIFY
        .with_label_values(&[
            if m == 2 { "lg2" } else { "lg3" },
            if n_ok == n_drafts { "accept" } else { "reject" },
        ])
        .inc();

    if n_ok == n_drafts {
        // ── FULL ACCEPT: all drafts + bonus commit (mirrors k2/k3 accept) ──
        for i in 0..n_ok {
            emit_token(a, hit.toks[i], verify_lps.get(i).cloned());
            if a.finished {
                return true;
            }
        }
        emit_token(a, vs[n_ok], verify_lps.get(n_ok).cloned());
        if a.finished {
            return true;
        }
        a.last_token = vs[n_ok];
        // Full accept ⇒ the verify kernel already wrote canonical h_state;
        // commit is a no-op but keeps the STree invariants explicit.
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, m, m) {
            tracing::error!("commit_accepted_prefix (low_gear accept): {e:#}");
            return true;
        }
    } else {
        // ── PARTIAL / REJECT: pop unaccepted rows, rewind, emit correction ──
        let pop = n_drafts - n_ok;
        a.seq.seq_len -= pop;
        for _ in 0..pop {
            a.seq.tokens.pop();
        }
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, committed, m) {
            tracing::error!("commit_accepted_prefix (low_gear partial): {e:#}");
            a.finished = true;
            return true;
        }
        for i in 0..n_ok {
            emit_token(a, hit.toks[i], verify_lps.get(i).cloned());
            if a.finished {
                return true;
            }
        }
        emit_token(a, vs[n_ok], verify_lps.get(n_ok).cloned());
        if a.finished {
            return true;
        }
        a.last_token = vs[n_ok];
    }

    // Adaptive accounting: low-gear tokens are SUSPENDED-SERIAL progress —
    // one tick per committed token keeps the re-probe interval identical in
    // tokens to plain serial decode. Deliberately NOT record_verify: that
    // window rates the neural drafter, not host n-gram luck.
    for _ in 0..committed {
        crate::scheduler::adaptive_spec::tick_serial(a);
    }

    // DFlash ctx accumulator: commit the accepted rows' captured hiddens so
    // the suspended stretch leaves no holes for the eventual re-probe.
    // Multi-row + explicit positions ⇒ commit_ctx (dflash_serial_ctx_append
    // is single-row, seq_len-1-stamped — wrong for committed==2/3). Fired
    // under either serial-append mode; when both are off, serial decode
    // leaves holes today and low gear matches that behavior.
    if crate::scheduler::adaptive_spec::unified_ctx_enabled()
        || crate::scheduler::adaptive_spec::serial_append_enabled()
    {
        let base_pos = a.seq.seq_len - committed;
        if let Err(e) = model.commit_ctx(&mut a.seq, committed, base_pos) {
            tracing::error!("commit_ctx (low_gear): {e:#}");
        }
    }

    // Freshest hidden (generator of the new last_token) for the eventual
    // re-probe propose — same row selection as k2/k3 (row n_ok).
    if let Err(e) = model.save_hidden_for_mtp(n_ok, 0) {
        tracing::error!("save_hidden_for_mtp (low_gear, {n_ok}): {e:#}");
        return true;
    }

    // Block-aligned Marconi checkpoint, same as the verify paths.
    model.decode_marconi_checkpoint(&mut a.seq);

    tracing::debug!(
        "LOW_GEAR k{m}: match_len={} drafts={:?} accepted={n_ok} seq_len={}",
        hit.match_len,
        &hit.toks[..n_drafts],
        a.seq.seq_len,
    );
    lg_record_outcome(n_drafts, n_ok, a.seq.seq_len);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_followon_after_repeat() {
        // haystack = [1,2,3,4,5, 9,9, 1,2,3,4]; last_token = 5.
        // Live suffix ...1,2,3,4 + 5 matches [1,2,3,4,5] at the start
        // (match_len 5); follow-on = [9,9] → drafts.
        let haystack = vec![1, 2, 3, 4, 5, 9, 9, 1, 2, 3, 4];
        let hit = lookup_drafts(&haystack, 5, 2, 3).expect("hit");
        assert_eq!(&hit.toks[..hit.len], &[9, 9]);
        assert_eq!(hit.match_len, 5);
    }

    #[test]
    fn single_draft_need_one() {
        let haystack = vec![1, 2, 3, 4, 5, 9, 9, 1, 2, 3, 4];
        let hit = lookup_drafts(&haystack, 5, 1, 3).expect("hit");
        assert_eq!(&hit.toks[..hit.len], &[9]);
        assert_eq!(hit.len, 1);
    }

    #[test]
    fn below_min_match_rejected() {
        // Only a length-1 match exists (bare last_token) — below min_match=3.
        let haystack = vec![1, 2, 3, 4, 5, 6, 7, 8];
        assert!(lookup_drafts(&haystack, 3, 1, 3).is_none());
    }

    #[test]
    fn longest_match_wins_over_recent_shorter() {
        // Old LONG match [7,8,1,2]+3 (len 5) → follow-on [50,60]; a more
        // recent SHORTER match [1,2]+3 (len 3) → follow-on [90,91] exists
        // too. Longest must win despite being older.
        let haystack = vec![7, 8, 1, 2, 3, 50, 60, 70, 1, 2, 3, 90, 91, 7, 8, 1, 2];
        let hit = lookup_drafts(&haystack, 3, 2, 3).expect("hit");
        assert_eq!(&hit.toks[..hit.len], &[50, 60]);
        assert_eq!(hit.match_len, 5); // [7,8,1,2,3]
    }

    #[test]
    fn respects_followon_room() {
        // The only occurrence of last_token sits at the very tail — no room
        // for follow-on drafts after it, so the lookup must return None
        // rather than a short/OOB draft.
        let haystack = vec![5, 1, 2, 3];
        assert!(lookup_drafts(&haystack, 3, 2, 1).is_none());
        assert!(lookup_drafts(&haystack, 3, 1, 1).is_none());
        // No match at all → None.
        let no_match = vec![5, 1, 2, 5, 1, 2];
        assert!(lookup_drafts(&no_match, 3, 2, 2).is_none());
    }

    #[test]
    fn empty_and_tiny_haystacks() {
        assert!(lookup_drafts(&[], 7, 1, 1).is_none());
        assert!(lookup_drafts(&[7], 7, 2, 1).is_none());
    }

    #[test]
    fn need_zero_or_oversized_rejected() {
        let haystack = vec![1, 2, 3, 1, 2];
        assert!(lookup_drafts(&haystack, 3, 0, 1).is_none());
        assert!(lookup_drafts(&haystack, 3, MAX_LOW_GEAR_DRAFTS + 1, 1).is_none());
    }

    #[test]
    fn most_recent_wins_ties() {
        // Two equal-length matches of [1,2]+3; the more recent one (follow-on
        // [40,41]) must win over the older ([20,21]).
        let haystack = vec![1, 2, 3, 20, 21, 1, 2, 3, 40, 41, 1, 2];
        let hit = lookup_drafts(&haystack, 3, 2, 3).expect("hit");
        assert_eq!(&hit.toks[..hit.len], &[40, 41]);
        assert_eq!(hit.match_len, 3);
    }
}
