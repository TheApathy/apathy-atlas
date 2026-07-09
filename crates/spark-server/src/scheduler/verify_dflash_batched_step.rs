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
    *ON.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_BATCHED_VERIFY").ok().as_deref() == Some("1")
    })
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
    if let Err(e) = model.sync_secondary() {
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

    if std::env::var("ATLAS_DFLASH_BATCHED_VERIFY_LOG").ok().as_deref() == Some("1") {
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

    // Accept-prefix: drafts[i] accepted iff drafts[i] == verified[i].
    let mut num_accepted = 0usize;
    for i in 0..drafts.len() {
        if i + 1 >= verified.len() {
            break;
        }
        if drafts[i] == verified[i] {
            num_accepted += 1;
        } else {
            break;
        }
    }

    // Roll back the over-extended seq_len / seq.tokens to
    // pre_verify_len + num_accepted + 1 (accepted drafts + bonus slot).
    let k_tokens = drafts.len() + 1;
    let pre_verify_len = a.seq.seq_len.saturating_sub(k_tokens);
    let target_seq_len = pre_verify_len + num_accepted + 1;
    let to_drop = a.seq.seq_len.saturating_sub(target_seq_len);
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

    // Save the bonus token's hidden for the next propose.
    let bonus = a.last_token;
    if let Err(e) = model.save_hidden_for_dflash(bonus, &mut a.seq, 0) {
        tracing::error!("save_hidden_for_dflash (batched dflash): {e:#}");
    }
    if let Err(e) = model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
        tracing::error!("trim_proposer_state (batched dflash): {e:#}");
    }

    // Re-propose for next step.
    match model.run_mtp_propose_multi(a.last_token, a.seq.seq_len, num_drafts, &mut a.seq, 0, None) {
        Ok(d) if !d.is_empty() => {
            a.pending_drafts = d;
        }
        Ok(_) => {}
        Err(e) => tracing::error!("run_mtp_propose_multi (batched dflash): {e:#}"),
    }
}
