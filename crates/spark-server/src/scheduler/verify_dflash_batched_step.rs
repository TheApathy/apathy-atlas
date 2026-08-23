// SPDX-License-Identifier: AGPL-3.0-only

//! CROSS-SEQ BATCHED DFLASH VERIFY (#39) — scheduler step.
//!
//! Gathers `c` active sequences' K=γ+1 draft windows into ONE batched verify
//! forward (`Model::decode_verify_dflash_batched`) whose FFN GEMMs read the
//! ~14 GB of NVFP4 FFN weights ONCE for all `c×K` rows — instead of the
//! per-sequence loop where each sequence runs its own full-weight-sweep
//! verify. The per-sequence accept / emit / commit / propose logic is
//! IDENTICAL to the flat-chain path of `step_verify_dflash`, so output is
//! lossless and byte-identical to single-stream per sequence.
//!
//! ## Eligibility (all must hold for every batched sequence)
//! - `pending_drafts.len() >= 4` (DFlash γ-block) and equal K across the batch;
//! - no active grammar (`grammar_state` None or terminated);
//! - not inside `<think>` (no thinking-accept filter in the batched path);
//! - no tree payload (`pending_tree_payload` None) and no tree/portfolio draft
//!   method active (flat chain only);
//! - not finished.
//!
//! Sequences that don't qualify fall through to the per-seq path in step_mtp.
//! Gated by `ATLAS_DFLASH_BATCHED_VERIFY=1` (default OFF).

use super::*;

/// Whether the cross-seq batched DFlash verify is enabled. Read once.
pub(super) fn dflash_batched_verify_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_DFLASH_BATCHED_VERIFY").ok().as_deref() == Some("1"))
}

/// Run the batched DFlash verify over `batched_idxs` (indices into `active`),
/// then finalize each sequence per-seq. Caller guarantees every listed
/// sequence satisfies the eligibility rules above and shares the same K.
pub(super) fn step_verify_dflash_batched(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    batched_idxs: &[usize],
    num_drafts: usize,
) {
    let sync_result = super::verify_dflash_step::dflash_capacity::preflight_batch_then(
        model.proposer_is_dflash(),
        || model.dflash_verify_capacity_k(),
        num_drafts,
        batched_idxs.iter().map(|&idx| {
            (
                active[idx].pending_drafts.len(),
                active[idx].pending_tree_payload.as_ref(),
            )
        }),
        || model.sync_secondary(),
    );
    let sync_result = match sync_result {
        Ok(result) => result,
        Err(error) => {
            tracing::error!(
                "DFlash batched pre-verify capacity guard rejected the batch: {error:#}"
            );
            for &idx in batched_idxs {
                active[idx].pending_drafts.clear();
                active[idx].pending_tree_payload = None;
            }
            return;
        }
    };
    if let Err(e) = sync_result {
        tracing::error!("sync_secondary (batched dflash): {e:#}");
        for &idx in batched_idxs {
            active[idx].finished = true;
        }
        return;
    }

    // Build per-seq token windows [last_token, ...drafts]. Record each seq's
    // drafts (owned) so the accept loop below can compare after the forward.
    let mut tokens_per_seq: Vec<Vec<u32>> = Vec::with_capacity(batched_idxs.len());
    let mut drafts_per_seq: Vec<Vec<u32>> = Vec::with_capacity(batched_idxs.len());
    for &idx in batched_idxs {
        let a = &mut active[idx];
        let drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
        let mut toks = Vec::with_capacity(drafts.len() + 1);
        toks.push(a.last_token);
        toks.extend_from_slice(&drafts);
        tokens_per_seq.push(toks);
        drafts_per_seq.push(drafts);
    }

    // ── ONE batched verify forward: FFN weights read ONCE for all c seqs ──
    let t_verify = Instant::now();
    let verified_per_seq = {
        // Collect &mut SequenceState for the batched call. Borrow each active
        // seq's inner `seq` field; the indices are distinct so no aliasing.
        let mut seq_refs: Vec<&mut spark_model::traits::SequenceState> =
            Vec::with_capacity(batched_idxs.len());
        // SAFETY: `batched_idxs` are distinct indices into `active`, so the
        // per-element `&mut active[idx].seq` borrows are disjoint. We build the
        // vec via raw pointers to satisfy the borrow checker for the split.
        let base = active.as_mut_ptr();
        for &idx in batched_idxs {
            // SAFETY: idx < active.len() and all idx distinct (checked by caller).
            let a: &mut ActiveSeq = unsafe { &mut *base.add(idx) };
            seq_refs.push(&mut a.seq);
        }
        match model.decode_verify_dflash_batched(&tokens_per_seq, &mut seq_refs, 0) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("decode_verify_dflash_batched: {e:#}");
                for &idx in batched_idxs {
                    active[idx].finished = true;
                }
                return;
            }
        }
    };
    let t_verify_us = t_verify.elapsed().as_micros();

    if std::env::var("ATLAS_DFLASH_BATCHED_VERIFY_LOG")
        .ok()
        .as_deref()
        == Some("1")
    {
        tracing::info!(
            "DFLASH BATCHED verify: c={} K={} forward={t_verify_us}μs",
            batched_idxs.len(),
            tokens_per_seq.first().map(|t| t.len()).unwrap_or(0),
        );
    }

    // ── Per-seq finalize (accept / emit / commit / propose) — UNCHANGED
    //    from the flat-chain path of step_verify_dflash ──
    for (batch_pos, &idx) in batched_idxs.iter().enumerate() {
        let a = &mut active[idx];
        let drafts = &drafts_per_seq[batch_pos];
        let verified = &verified_per_seq[batch_pos];
        finalize_flat_dflash_verify(model, a, drafts, verified, num_drafts);
    }
}

/// Flat-chain accept + emit + commit + propose for one sequence, mirroring the
/// non-tree / non-grammar / non-thinking path of `step_verify_dflash` exactly.
/// The batched verify already advanced `a.seq.seq_len` by `K = drafts.len()+1`
/// and pushed all K tokens; this rolls back to the accepted prefix.
fn finalize_flat_dflash_verify(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
    num_drafts: usize,
) {
    a.last_token_time = Instant::now();

    // Accept-prefix: drafts[i] accepted iff drafts[i] == verified[i]. Pure
    // helper (unit-tested) so the exact-match contract matches the flat path of
    // step_verify_dflash bit-for-bit.
    let num_accepted = flat_accept_prefix(drafts, verified);

    // Roll back the over-extended seq_len / seq.tokens to
    // pre_verify_len + num_accepted + 1 (accepted drafts + bonus slot).
    let k_tokens = drafts.len() + 1;
    let (target_seq_len, to_drop) = flat_rollback_plan(a.seq.seq_len, drafts.len(), num_accepted);
    if to_drop > 0 {
        a.seq.seq_len = target_seq_len;
        let pop_n = to_drop.min(a.seq.tokens.len());
        for _ in 0..pop_n {
            a.seq.tokens.pop();
        }
    }

    // Emit accepted drafts.
    let emit_take = num_accepted.min(drafts.len());
    for &tok in &drafts[..emit_take] {
        emit_token(a, tok, None);
        if a.finished {
            return;
        }
    }

    // Bonus token = verified[num_accepted].
    if let Some(&bonus) = verified.get(num_accepted) {
        emit_token(a, bonus, None);
        if a.finished {
            return;
        }
        a.last_token = bonus;
    }

    crate::metrics::SPEC_DECODE_VERIFY
        .with_label_values(&[
            "dflash",
            if num_accepted == drafts.len() {
                "accept_all"
            } else {
                "accept_partial"
            },
        ])
        .inc();

    // SSM commit / rollback (per-seq, identical to step_verify_dflash flat path).
    let total_accepted = num_accepted + 1; // bonus always accepted
    if let Err(e) = model.commit_verify_state_async(&mut a.seq, total_accepted, k_tokens) {
        tracing::error!("commit_verify_state_async (batched dflash): {e:#}");
        a.finished = true;
        return;
    }

    // Save the bonus token's hidden for the active proposer. Cross-sequence
    // batching is DFlash-only; the shared boundary keeps the pairing explicit.
    let bonus = a.last_token;
    if let Err(e) = save_hidden_for_active_proposer(model, a, bonus, num_accepted) {
        tracing::error!("save hidden for active proposer (batched verify): {e:#}");
        return;
    }
    if let Err(e) = model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
        tracing::error!("trim_proposer_state (batched dflash): {e:#}");
    }

    let token = a.last_token;
    if let Err(e) = proposal_lifecycle::propose_and_install(model, a, token, num_drafts, None) {
        tracing::error!("run_mtp_propose_multi (batched dflash): {e:#}");
    }
}

/// Pure flat-chain accept-prefix count: the length of the longest prefix where
/// `drafts[i] == verified[i]`. `verified` is `[target@last_token, target@draft0,
/// …]` (length `drafts.len()+1` in the well-formed case); a draft `i` can only
/// be accepted if its correcting/bonus slot `verified[i+1]` exists (matching the
/// `i + 1 >= verified.len()` guard in `step_verify_dflash`'s flat path).
///
/// This is the ONLY acceptance rule the batched flat path applies — grammar /
/// thinking / relax / typical / tree acceptance are all excluded by the batched
/// eligibility gate (see `mtp_step.rs`), so per-seq output stays byte-identical
/// to the single-stream flat verify.
fn flat_accept_prefix(drafts: &[u32], verified: &[u32]) -> usize {
    let mut n = 0usize;
    for i in 0..drafts.len() {
        if i + 1 >= verified.len() {
            break;
        }
        if drafts[i] == verified[i] {
            n += 1;
        } else {
            break;
        }
    }
    n
}

/// Post-verify rollback target: the batched forward advanced `seq_len` by
/// `k_tokens = drafts.len()+1`; the accepted prefix keeps
/// `pre_verify_len + num_accepted + 1` (accepted drafts + the always-accepted
/// bonus slot). Returns `(target_seq_len, num_tokens_to_pop)`.
fn flat_rollback_plan(
    current_seq_len: usize,
    drafts_len: usize,
    num_accepted: usize,
) -> (usize, usize) {
    let k_tokens = drafts_len + 1;
    let pre_verify_len = current_seq_len.saturating_sub(k_tokens);
    let target_seq_len = pre_verify_len + num_accepted + 1;
    let to_drop = current_seq_len.saturating_sub(target_seq_len);
    (target_seq_len, to_drop)
}

#[cfg(test)]
mod tests {
    use super::{flat_accept_prefix, flat_rollback_plan};

    // ── flat_accept_prefix: exact-match prefix, matches step_verify_dflash ──

    #[test]
    fn accept_prefix_full_accept() {
        // All drafts match; verified has the bonus slot at the tail.
        let drafts = [10, 11, 12, 13];
        let verified = [10, 11, 12, 13, 14];
        assert_eq!(flat_accept_prefix(&drafts, &verified), 4);
    }

    #[test]
    fn accept_prefix_first_mismatch_terminates() {
        let drafts = [10, 11, 99, 13];
        let verified = [10, 11, 12, 13, 14];
        // drafts[2] (99) != verified[2] (12) → prefix stops at 2.
        assert_eq!(flat_accept_prefix(&drafts, &verified), 2);
    }

    #[test]
    fn accept_prefix_zero_when_first_diverges() {
        let drafts = [7, 11, 12, 13];
        let verified = [8, 11, 12, 13, 14];
        assert_eq!(flat_accept_prefix(&drafts, &verified), 0);
    }

    #[test]
    fn accept_prefix_capped_by_verified_bonus_slot() {
        // verified is SHORT (no bonus slot for the last draft): the guard
        // `i + 1 >= verified.len()` caps acceptance so the bonus always exists.
        let drafts = [10, 11, 12, 13];
        let verified = [10, 11, 12]; // len 3 → i can accept only 0,1 (i+1<3)
        assert_eq!(flat_accept_prefix(&drafts, &verified), 2);
    }

    #[test]
    fn accept_prefix_empty_drafts() {
        assert_eq!(flat_accept_prefix(&[], &[42]), 0);
    }

    // ── flat_rollback_plan: seq_len over-extension unwind ──

    #[test]
    fn rollback_full_accept_keeps_all_plus_bonus() {
        // pre_verify_len=100, K=5 → seq_len advanced to 105. Full accept (4):
        // target = 100 + 4 + 1 = 105 → nothing to drop.
        let (target, drop) = flat_rollback_plan(105, 4, 4);
        assert_eq!(target, 105);
        assert_eq!(drop, 0);
    }

    #[test]
    fn rollback_partial_accept_drops_rejected_tail() {
        // pre_verify_len=100, K=5, seq_len=105, accepted 2:
        // target = 100 + 2 + 1 = 103 → drop 2 (the rejected drafts 2,3).
        let (target, drop) = flat_rollback_plan(105, 4, 2);
        assert_eq!(target, 103);
        assert_eq!(drop, 2);
    }

    #[test]
    fn rollback_zero_accept_keeps_only_bonus() {
        // seq_len=105, accepted 0: target = 100 + 0 + 1 = 101 → drop 4.
        let (target, drop) = flat_rollback_plan(105, 4, 0);
        assert_eq!(target, 101);
        assert_eq!(drop, 4);
    }

    #[test]
    fn rollback_mixed_max_tokens_shapes() {
        // Distinct K per seq is fine — the plan is per-seq (mid-batch
        // compaction case): a K=3 seq at seq_len=53 with 1 accepted.
        let (target, drop) = flat_rollback_plan(53, 2, 1);
        assert_eq!(target, 53 - 3 + 1 + 1); // pre=50, +1 accepted +1 bonus =52
        assert_eq!(drop, 1);
    }
}
