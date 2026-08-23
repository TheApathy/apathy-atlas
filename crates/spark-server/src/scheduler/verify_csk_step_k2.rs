// SPDX-License-Identifier: AGPL-3.0-only

//! K=2 verify, c-batched per-step (Path B, K=2 sibling).
//!
//! Routes c verify-eligible sequences (each with `drafts.len() == 1`)
//! through a single c-batched K=2 verify call instead of c independent
//! single-seq `decode_verify_graphed` calls. The trait method
//! `decode_verify_k2_batched_csk` runs K=2 SEQUENTIAL c-batched single-
//! step decodes (each at M=c via the existing `decode_multi_seq` path);
//! this scheduler function decodes the per-seq argmax results, applies
//! per-seq accept/reject + commit + emit + propose using the SAME helpers
//! as the per-seq `step_verify_k2` path (verify_k2_step.rs).
//!
//! **Gated**: only invoked when `ATLAS_MTP_K2_BATCH_CSEQ=1` AND all
//! verify candidates pass the eligibility filter in `step_mtp`
//! (drafts.len() == 1, grammar inactive, not finished, c ≥ 2).
//!
//! Production rationale: Q36-35B-A3B prod runs `--num-drafts 1` which
//! never hits the K=3 CSK gate (`drafts.len() == 2`). This file enables
//! the equivalent batched verify for the production K=2 path.

use super::*;

/// K=2 c-batched verify dispatch. `batched_idxs` indexes into `active`
/// the sequences eligible for batched verify (filtered upstream in
/// `step_mtp`). All listed sequences MUST have `drafts.len() == 1`.
pub fn step_verify_k2_batched_csk(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    batched_idxs: &[usize],
    num_drafts: usize,
) {
    let c = batched_idxs.len();
    if c < 2 {
        return;
    }

    if proposal_lifecycle::flat_batch_preflight_at(active, batched_idxs)
        != proposal_lifecycle::FlatBatchPreflight::Ready
    {
        tracing::error!("CSK-K2 received a tree-bearing proposal; refusing flat batch");
        return;
    }

    if let Err(e) = model.sync_secondary() {
        tracing::error!("CSK-K2 sync_secondary: {e:#}");
        for &idx in batched_idxs {
            active[idx].finished = true;
        }
        return;
    }

    // ── Gather per-seq inputs ──
    // `tokens_per_seq[i] = [last_token, drafts[0]]`
    // `per_seq_drafts[i] = drafts[0]` saved for accept/reject comparison.
    let mut tokens_per_seq: Vec<[u32; 2]> = Vec::with_capacity(c);
    let mut per_seq_drafts: Vec<u32> = Vec::with_capacity(c);
    let mut original_seq_lens: Vec<usize> = Vec::with_capacity(c);

    let Some(batch_drafts) = proposal_lifecycle::take_flat_batch_at(active, batched_idxs) else {
        tracing::error!("CSK-K2 flat batch changed after preflight; refusing batch");
        return;
    };
    for (&idx, drafts) in batched_idxs.iter().zip(batch_drafts) {
        let a = &mut active[idx];
        debug_assert_eq!(drafts.len(), 1, "CSK-K2 eligibility guarantees 1 draft");
        tokens_per_seq.push([a.last_token, drafts[0]]);
        per_seq_drafts.push(drafts[0]);
        original_seq_lens.push(a.seq.seq_len);
    }

    // EP: c-batched verify_csk is NOT EP-aware. Defensive bail; production
    // single-GPU deploy doesn't hit EP.
    if model.is_ep() {
        tracing::warn!(
            "CSK-K2 verify path doesn't support EP; falling back to per-seq K=2 for c={c}"
        );
        for (i, &idx) in batched_idxs.iter().enumerate() {
            active[idx].pending_drafts = vec![per_seq_drafts[i]];
            let drafts_vec = std::mem::take(&mut active[idx].pending_drafts);
            step_verify_k2(model, &mut active[idx], &drafts_vec, num_drafts);
        }
        return;
    }

    // ── Batched K-loop verify ──
    let t_verify = Instant::now();
    let results = {
        // Borrow disjoint &mut SequenceState refs at the indices listed in
        // `batched_idxs` via sort + split_at_mut. Mirrors the K=3 sibling
        // pattern (verify_csk_step.rs).
        let mut refs: Vec<&mut SequenceState> = Vec::with_capacity(c);
        let mut sorted: Vec<(usize, usize)> = batched_idxs
            .iter()
            .enumerate()
            .map(|(p, &i)| (i, p))
            .collect();
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
        sorted_refs.sort_by_key(|(p, _)| *p);
        refs.extend(sorted_refs.into_iter().map(|(_, s)| s));

        match model.decode_verify_k2_batched_csk(&tokens_per_seq, &mut refs, 0) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("decode_verify_k2_batched_csk: {e:#}");
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
        let [v0, v1] = results[i];
        let draft = per_seq_drafts[i];
        let orig_len = original_seq_lens[i];

        // K=2 accept/reject:
        //   draft == v0  → ACCEPT: emit draft + v1, advance state by 2
        //   draft != v0  → REJECT: emit v0 only, advance state by 1
        let accepted = draft == v0;
        let target_len = orig_len + if accepted { 2 } else { 1 };
        debug_assert!(
            target_len <= a.seq.seq_len,
            "K-loop should have advanced past target_len"
        );
        let pop_count = a.seq.seq_len - target_len;
        for _ in 0..pop_count {
            a.seq.tokens.pop();
        }
        a.seq.seq_len = target_len;

        // Logprobs unavailable for CSK path (per K=3 sibling note: lm_head
        // logits buffer overwritten by each step; only last step's logits
        // remain). Disable to avoid emitting wrong data.
        let verify_lps: Vec<Option<crate::api::TokenLogprobs>> = Vec::new();

        if accepted {
            // ── ACCEPT (num_accepted=2) ──
            emit_token(a, draft, verify_lps.first().cloned().flatten());
            if !a.finished {
                emit_token(a, v1, verify_lps.get(1).cloned().flatten());
            }
            if a.finished {
                continue;
            }
            a.last_token = v1;

            if let Err(e) = model.commit_verify_state_async(&mut a.seq, 2, 2) {
                tracing::error!("CSK-K2 commit_verify_state_async (accept): {e:#}");
                continue;
            }
            if let Err(e) = model.save_hidden_for_mtp(1, 0) {
                tracing::error!("CSK-K2 save_hidden_for_mtp(1): {e:#}");
                continue;
            }
            if let Err(e) = model.trim_proposer_state(&mut a.seq, 1, 0) {
                tracing::error!("CSK-K2 trim_proposer_state: {e:#}");
            }
            let _mtp_grammar_mask = mtp_grammar_mask_for(a);
            match proposal_lifecycle::propose_and_install(
                model,
                a,
                v1,
                num_drafts,
                _mtp_grammar_mask.as_deref(),
            ) {
                Ok(_) => {}
                Err(e) => tracing::error!("CSK-K2 run_mtp_propose_multi: {e:#}"),
            }
        } else {
            // ── REJECT (only last_token accepted; num_accepted=1) ──
            if let Err(e) = model.trim_proposer_state(&mut a.seq, 0, 0) {
                tracing::error!("CSK-K2 trim_proposer_state: {e:#}");
            }
            if let Err(e) = model.commit_verify_state_async(&mut a.seq, 1, 2) {
                tracing::error!("CSK-K2 commit_verify_state_async (reject): {e:#}");
                a.finished = true;
                continue;
            }
            emit_token(a, v0, verify_lps.first().cloned().flatten());
            if a.finished {
                continue;
            }
            a.last_token = v0;
            if let Err(e) = model.save_hidden_for_mtp(0, 0) {
                tracing::error!("CSK-K2 save_hidden_for_mtp(0): {e:#}");
                continue;
            }
            let _mtp_grammar_mask = mtp_grammar_mask_for(a);
            match proposal_lifecycle::propose_and_install(
                model,
                a,
                v0,
                num_drafts,
                _mtp_grammar_mask.as_deref(),
            ) {
                Ok(_) => {}
                Err(e) => tracing::error!("CSK-K2 run_mtp_propose_multi: {e:#}"),
            }
        }
    }

    tracing::info!("CSK K=2 verify: c={c} verify={verify_us}μs");
}
