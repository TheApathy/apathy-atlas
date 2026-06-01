// SPDX-License-Identifier: AGPL-3.0-only

//! K=3 verify, c-batched per-step (Path B).
//!
//! Routes c verify-eligible sequences (each with `drafts.len() == 2`)
//! through a single c-batched K=3 verify call instead of c independent
//! single-seq `decode_verify_graphed_k3` calls. The trait method
//! `decode_verify_k3_batched_csk` runs K=3 SEQUENTIAL c-batched single-
//! step decodes (each at M=c via the existing `decode_multi_seq` path);
//! this scheduler function decodes the per-seq argmax results, applies
//! per-seq accept/reject + commit + emit + propose using the SAME helpers
//! as the per-seq `step_verify_k3` path.
//!
//! **Gated**: only invoked when `ATLAS_MTP_K3_BATCH_CSEQ=1` AND all
//! verify candidates pass the eligibility filter in `step_mtp`
//! (drafts.len() == 2, grammar inactive, not finished, c ≥ 2).

use super::*;

/// K=3 c-batched verify dispatch. `batched_idxs` indexes into `active`
/// the sequences eligible for batched verify (filtered upstream in
/// `step_mtp`). All listed sequences MUST have `drafts.len() == 2`.
pub fn step_verify_k3_batched_csk(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    batched_idxs: &[usize],
    num_drafts: usize,
) {
    let c = batched_idxs.len();
    if c < 2 {
        return;
    }

    if let Err(e) = model.sync_secondary() {
        tracing::error!("CSK sync_secondary: {e:#}");
        for &idx in batched_idxs {
            active[idx].finished = true;
        }
        return;
    }

    // ── Gather per-seq inputs ──
    // `tokens_per_seq[i] = [last_token, drafts[0], drafts[1]]`
    // `drafts[i]` saved here for accept/reject comparison below.
    let mut tokens_per_seq: Vec<[u32; 3]> = Vec::with_capacity(c);
    let mut per_seq_drafts: Vec<[u32; 2]> = Vec::with_capacity(c);
    let mut original_seq_lens: Vec<usize> = Vec::with_capacity(c);

    // Take drafts out of each ActiveSeq up-front (matches single-seq path).
    for &idx in batched_idxs {
        let a = &mut active[idx];
        let drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
        debug_assert_eq!(drafts.len(), 2, "CSK eligibility guarantees 2 drafts");
        tokens_per_seq.push([a.last_token, drafts[0], drafts[1]]);
        per_seq_drafts.push([drafts[0], drafts[1]]);
        original_seq_lens.push(a.seq.seq_len);
    }

    // EP: c-batched verify_csk is NOT EP-aware. Step_mtp's eligibility
    // filter does not check EP — gate here as a safety net. (Production
    // single-GPU deploy doesn't hit EP, so this is a defensive bail.)
    if model.is_ep() {
        tracing::warn!(
            "CSK verify path doesn't support EP; falling back to per-seq K=3 for c={c}"
        );
        // Restore drafts and fall through to per-seq path.
        for (i, &idx) in batched_idxs.iter().enumerate() {
            active[idx].pending_drafts = per_seq_drafts[i].to_vec();
            let drafts_vec = std::mem::take(&mut active[idx].pending_drafts);
            step_verify_k3(model, &mut active[idx], &drafts_vec, num_drafts);
        }
        return;
    }

    // ── Batched K-loop verify ──
    let t_verify = Instant::now();
    let results = {
        let mut refs: Vec<&mut SequenceState> = Vec::with_capacity(c);
        // Splitting a `&mut [ActiveSeq]` into multiple `&mut SequenceState`
        // at distinct indices requires unsafe-equivalent splitting via
        // indices. Use `iter_mut` collect with filtering by index.
        for (idx_pos, &idx) in batched_idxs.iter().enumerate() {
            // SAFETY: each `idx` in batched_idxs is unique (guaranteed by
            // upstream filter that iterates active.iter().enumerate()
            // without duplication). We index disjoint elements of the same
            // slice. Use split-borrow via pointer dance — Rust's lack of
            // safe "disjoint indices" forces this pattern. To keep this
            // safe-Rust-only and avoid the dance, collect mut refs by
            // sorted order then permute back.
            let _ = idx_pos; // silence unused
            let _ = idx;
        }
        // Safe construction: sort indices, partition with split_at_mut.
        // Easiest path: clone the indices, sort ascending, walk active
        // with split_at_mut to peel off each &mut ActiveSeq, push its
        // &mut seq into refs in the SORTED order, then permute back to
        // batched_idxs order after the call.
        let mut sorted: Vec<(usize, usize)> = batched_idxs.iter().enumerate().map(|(p, &i)| (i, p)).collect();
        sorted.sort_by_key(|t| t.0);
        let mut tail: &mut [ActiveSeq] = active;
        let mut cursor: usize = 0;
        let mut sorted_refs: Vec<(usize, &mut SequenceState)> = Vec::with_capacity(c);
        for (idx, orig_pos) in &sorted {
            let offset = idx - cursor;
            let (_skip, rest) = tail.split_at_mut(offset);
            let (head, rest_after) = rest.split_first_mut().expect("idx within bounds");
            sorted_refs.push((*orig_pos, &mut head.seq));
            tail = rest_after;
            cursor = idx + 1;
        }
        // Permute back to batched_idxs order.
        sorted_refs.sort_by_key(|(p, _)| *p);
        refs.extend(sorted_refs.into_iter().map(|(_, s)| s));

        // Also need tokens_per_seq in matching order — already constructed
        // in batched_idxs order. refs is now also in batched_idxs order.
        match model.decode_verify_k3_batched_csk(&tokens_per_seq, &mut refs, 0) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("decode_verify_k3_batched_csk: {e:#}");
                drop(refs);
                for &idx in batched_idxs {
                    active[idx].finished = true;
                }
                return;
            }
        }
    };
    let verify_us = t_verify.elapsed().as_micros();

    // ── Per-seq accept/reject + commit + emit + propose ──
    for (i, &idx) in batched_idxs.iter().enumerate() {
        let a = &mut active[idx];
        a.last_token_time = Instant::now();
        let [v0, v1, v2] = results[i];
        let drafts = per_seq_drafts[i];
        let orig_len = original_seq_lens[i];

        // Per-seq num_accepted (matches K=3 single-seq logic):
        //   drafts[0] != v0  → 0 drafts accepted, num_accepted=1
        //   drafts[1] != v1  → 1 draft accepted,  num_accepted=2
        //   else (full)      → 2 drafts accepted, num_accepted=3
        let num_accepted_drafts = if drafts[0] != v0 {
            0
        } else if drafts[1] != v1 {
            1
        } else {
            2
        };
        // Note: K=3 single-seq's `num_accepted` is drafts_accepted (0/1/2),
        // and commit_verify_state_async takes `k=3` + `num_accepted ∈
        // {1, 2, 3}` (full-token-count including last_token's accept).
        // Keep this naming consistent with verify_k3_step.rs comments.

        // Truncate seq state from K=3 advance to actual acceptance level.
        // decode_batch advanced seq.seq_len by K=3 and pushed 3 tokens.
        // We want `seq_len = orig_len + num_accepted_drafts + 1`.
        let target_len = orig_len + num_accepted_drafts + 1;
        debug_assert!(target_len <= a.seq.seq_len, "K-loop should have advanced past target_len");
        let pop_count = a.seq.seq_len - target_len;
        for _ in 0..pop_count {
            a.seq.tokens.pop();
        }
        a.seq.seq_len = target_len;

        // Extract logprobs (positions match the verify output buffer
        // populated by per-step argmax). Note: `extract_verify_logprobs`
        // reads from the shared lm_head logits buffer which got
        // overwritten by EACH of the K=3 steps; only the LAST step's
        // logits are still in the buffer. So logprobs are unavailable
        // for K-loop CSK path — disable to avoid emitting wrong data.
        // TODO: per-step D2H of logprobs into per-step scratch.
        let verify_lps: Vec<Option<crate::api::TokenLogprobs>> = Vec::new();

        // Per-seq commit + trim + emit + propose pipeline. Mirrors
        // verify_k3_step.rs branches.
        if num_accepted_drafts == 2 {
            // ── FULL ACCEPT (K=3 accept-3) ──
            emit_token(a, drafts[0], verify_lps.first().cloned().flatten());
            if !a.finished {
                emit_token(a, drafts[1], verify_lps.get(1).cloned().flatten());
            }
            if !a.finished {
                emit_token(a, v2, verify_lps.get(2).cloned().flatten());
            }
            if a.finished {
                continue;
            }
            a.last_token = v2;

            if let Err(e) = model.commit_verify_state_async(&mut a.seq, 3, 3) {
                tracing::error!("CSK commit_verify_state_async (K=3 accept-3): {e:#}");
                continue;
            }
            if let Err(e) = model.save_hidden_for_mtp(2, 0) {
                tracing::error!("CSK save_hidden_for_mtp(2): {e:#}");
                continue;
            }
            if let Err(e) = model.trim_proposer_state(&mut a.seq, 2, 0) {
                tracing::error!("CSK trim_proposer_state: {e:#}");
            }
            let _mtp_grammar_mask = mtp_grammar_mask_for(a);
            match model.run_mtp_propose_multi(
                v2,
                a.seq.seq_len,
                num_drafts,
                &mut a.seq,
                0,
                _mtp_grammar_mask.as_deref(),
            ) {
                Ok(d) if !d.is_empty() => a.pending_drafts = d,
                Ok(_) => {}
                Err(e) => tracing::error!("CSK run_mtp_propose_multi: {e:#}"),
            }
        } else if num_accepted_drafts == 1 {
            // ── PARTIAL ACCEPT-2 (last_token + draft[0]) ──
            if let Err(e) = model.trim_proposer_state(&mut a.seq, 1, 0) {
                tracing::error!("CSK trim_proposer_state: {e:#}");
            }
            if let Err(e) = model.commit_verify_state_async(&mut a.seq, 2, 3) {
                tracing::error!("CSK commit_verify_state_async (K=3 accept-2): {e:#}");
                a.finished = true;
                continue;
            }
            emit_token(a, drafts[0], verify_lps.first().cloned().flatten());
            if !a.finished {
                emit_token(a, v1, verify_lps.get(1).cloned().flatten());
            }
            if a.finished {
                continue;
            }
            a.last_token = v1;
            if let Err(e) = model.save_hidden_for_mtp(1, 0) {
                tracing::error!("CSK save_hidden_for_mtp(1): {e:#}");
                continue;
            }
            let _mtp_grammar_mask = mtp_grammar_mask_for(a);
            match model.run_mtp_propose_multi(
                v1,
                a.seq.seq_len,
                num_drafts,
                &mut a.seq,
                0,
                _mtp_grammar_mask.as_deref(),
            ) {
                Ok(d) if !d.is_empty() => a.pending_drafts = d,
                Ok(_) => {}
                Err(e) => tracing::error!("CSK run_mtp_propose_multi: {e:#}"),
            }
        } else {
            // ── REJECT (only last_token accepted) ──
            if let Err(e) = model.trim_proposer_state(&mut a.seq, 0, 0) {
                tracing::error!("CSK trim_proposer_state: {e:#}");
            }
            if let Err(e) = model.commit_verify_state_async(&mut a.seq, 1, 3) {
                tracing::error!("CSK commit_verify_state_async (K=3 accept-1): {e:#}");
                a.finished = true;
                continue;
            }
            emit_token(a, v0, verify_lps.first().cloned().flatten());
            if a.finished {
                continue;
            }
            a.last_token = v0;
            if let Err(e) = model.save_hidden_for_mtp(0, 0) {
                tracing::error!("CSK save_hidden_for_mtp(0): {e:#}");
                continue;
            }
            let _mtp_grammar_mask = mtp_grammar_mask_for(a);
            match model.run_mtp_propose_multi(
                v0,
                a.seq.seq_len,
                num_drafts,
                &mut a.seq,
                0,
                _mtp_grammar_mask.as_deref(),
            ) {
                Ok(d) if !d.is_empty() => a.pending_drafts = d,
                Ok(_) => {}
                Err(e) => tracing::error!("CSK run_mtp_propose_multi: {e:#}"),
            }
        }
    }

    tracing::info!("CSK K=3 verify: c={c} verify={verify_us}μs");
}
