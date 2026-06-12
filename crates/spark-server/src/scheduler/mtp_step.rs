// SPDX-License-Identifier: AGPL-3.0-only

//! MTP speculative draft proposal step.

use super::*;

/// MTP-aware step: bootstrap sequences without drafts, then verify via CUDA graph.
/// Supports K=2 (num_drafts=1) and K=3 (num_drafts=2).
pub fn step_mtp(model: &dyn Model, active: &mut [ActiveSeq], num_drafts: usize) {
    // Stage-1 DFlash grammar gate: drop drafts proposed before the grammar
    // became constraining (e.g. the block whose emission opened a
    // `<tool_call>`). DFlash drafts and the K=γ verify argmax bypass the
    // XGrammar bitmask, so a grammar-active sequence must run the bootstrap
    // path below, where `sample_token_with_grammar` enforces it.
    for a in active.iter_mut() {
        if !a.pending_drafts.is_empty() && dflash_grammar_skip_propose(model, a) {
            a.pending_drafts.clear();
            a.pending_tree_payload = None;
        }
    }

    let mut bootstrap_idxs: Vec<usize> = Vec::new();
    let mut verify_idxs: Vec<usize> = Vec::new();
    for (i, a) in active.iter().enumerate() {
        if !a.pending_drafts.is_empty() {
            verify_idxs.push(i);
        } else {
            bootstrap_idxs.push(i);
        }
    }

    // ── Phase A: Bootstrap decode for sequences without a draft ──
    for &idx in &bootstrap_idxs {
        let a = &mut active[idx];
        // EP: broadcast token to worker before decode (worker runs decode in lockstep).
        if let Err(e) = model.ep_broadcast_cmd(a.last_token) {
            tracing::error!("EP broadcast bootstrap token: {e:#}");
            a.finished = true;
            continue;
        }
        let logits = match model.decode(a.last_token, &mut a.seq, 0) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("bootstrap decode error: {e:#}");
                a.finished = true;
                continue;
            }
        };
        let tok = match sample_token_with_grammar(
            model,
            logits,
            a.temperature,
            a.top_k,
            a.top_p,
            &[],
            a.grammar_state.as_mut(),
        ) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("bootstrap sample error: {e:#}");
                a.finished = true;
                continue;
            }
        };

        // Extract logprobs from bootstrap decode logits (single position).
        let lp = if let Some(k) = a.top_logprobs {
            extract_single_logprobs(model, logits, tok, k)
        } else {
            None
        };

        emit_token(a, tok, lp);
        if a.finished {
            continue;
        }
        a.last_token = tok;

        if let Err(e) = model.save_hidden_for_mtp(0, 0) {
            tracing::error!("save_hidden_for_mtp: {e:#}");
            continue;
        }
        // Stage-1 DFlash grammar gate: while the grammar constrains output,
        // stay non-speculative (the bootstrap decode above already sampled
        // through the grammar; drafting would bypass it at verify).
        if !dflash_grammar_skip_propose(model, a) {
            let _mtp_grammar_mask = mtp_grammar_mask_for(a);
            match model.run_mtp_propose_multi(
                tok,
                a.seq.seq_len,
                num_drafts,
                &mut a.seq,
                0,
                _mtp_grammar_mask.as_deref(),
            ) {
                Ok(drafts) if !drafts.is_empty() => {
                    tracing::debug!("MTP bootstrap: tok={tok} → drafts={drafts:?}");
                    a.pending_drafts = drafts;
                }
                Ok(_) => tracing::warn!("MTP propose returned empty"),
                Err(e) => {
                    tracing::error!("run_mtp_propose_multi: {e:#}");
                }
            }
        }

        if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
            tracing::error!("bootstrap start_checkpoint_async: {e:#}");
        }
    }

    // ── Phase B (Path B opt-in, K=2): c-batched K=2 verify when all seqs eligible ──
    // ATLAS_MTP_K2_BATCH_CSEQ=1: K=2 sibling of the K=3 CSK path below.
    // Production `--num-drafts 1` runs the K=2 path which previously
    // never aggregated at concurrency≥2 (each seq ran an independent
    // single-seq graphed K=2 verify, sequentially, on the scheduler
    // thread). This routes c eligible seqs (drafts.len()==1, no
    // grammar, not finished) through ONE c-batched K=2 verify call —
    // 2 batched forwards/layer instead of 2c.
    if spark_model::layers::mtp_k2_batch_cseq_enabled() && verify_idxs.len() >= 2 {
        let mut batched_idxs: Vec<usize> = Vec::new();
        let mut leftover_idxs: Vec<usize> = Vec::new();
        for &idx in &verify_idxs {
            let a = &active[idx];
            // Eligibility: exactly 1 draft (K=2), no grammar, not finished,
            // and num_drafts==1 (defensive — the K=2 trait path is shaped
            // for the production num_drafts=1 case).
            if a.pending_drafts.len() == 1
                && a.grammar_state.is_none()
                && !a.finished
                && num_drafts == 1
            {
                batched_idxs.push(idx);
            } else {
                leftover_idxs.push(idx);
            }
        }
        if batched_idxs.len() >= 2 {
            step_verify_k2_batched_csk(model, active, &batched_idxs, num_drafts);
        } else {
            leftover_idxs.extend(batched_idxs);
        }
        // Process anything that didn't go through the batched path.
        for &idx in &leftover_idxs {
            let a = &mut active[idx];
            let mut drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
            if drafts.is_empty() {
                continue;
            }
            if let Some(ref mut gs) = a.grammar_state {
                let kept = truncate_drafts_at_grammar_boundary(gs, &drafts);
                if kept < drafts.len() {
                    drafts.truncate(kept);
                }
                if drafts.is_empty() {
                    continue;
                }
            }
            if drafts.len() >= 4 {
                step_verify_dflash(model, a, &drafts, num_drafts);
            } else if drafts.len() >= 3 {
                step_verify_k4(model, a, &drafts, num_drafts);
            } else if drafts.len() >= 2 {
                step_verify_k3(model, a, &drafts, num_drafts);
            } else {
                step_verify_k2(model, a, &drafts, num_drafts);
            }
        }
        return;
    }

    // ── Phase B (Path B opt-in): c-batched K=3 verify when all seqs eligible ──
    // ATLAS_MTP_K3_BATCH_CSEQ=1: routes verify to the c-batched K=3 per-step
    // K-loop (see model/trait_impl/verify_csk.rs) when all verify_idxs
    // have exactly 2 drafts (K=3 path), no grammar boundary, and not
    // finished. Trades c × 3 = 12 forwards/layer (c=4) for 3 batched
    // forwards/layer + per-step intermediate D2D snapshots (~8 ms
    // overhead). Other verify_idxs (K=2/K=4/DFlash/grammar-active) fall
    // through to the per-seq loop below.
    if spark_model::layers::mtp_k3_batch_cseq_enabled() && verify_idxs.len() >= 2 {
        let mut batched_idxs: Vec<usize> = Vec::new();
        let mut leftover_idxs: Vec<usize> = Vec::new();
        for &idx in &verify_idxs {
            let a = &active[idx];
            // Eligibility: exactly 2 drafts (K=3), no grammar, not finished.
            if a.pending_drafts.len() == 2 && a.grammar_state.is_none() && !a.finished {
                batched_idxs.push(idx);
            } else {
                leftover_idxs.push(idx);
            }
        }
        if batched_idxs.len() >= 2 {
            step_verify_k3_batched_csk(model, active, &batched_idxs, num_drafts);
        } else {
            leftover_idxs.extend(batched_idxs);
        }
        // Process anything that didn't go through the batched path.
        for &idx in &leftover_idxs {
            let a = &mut active[idx];
            let mut drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
            if drafts.is_empty() {
                continue;
            }
            if let Some(ref mut gs) = a.grammar_state {
                let kept = truncate_drafts_at_grammar_boundary(gs, &drafts);
                if kept < drafts.len() {
                    drafts.truncate(kept);
                }
                if drafts.is_empty() {
                    continue;
                }
            }
            if drafts.len() >= 4 {
                step_verify_dflash(model, a, &drafts, num_drafts);
            } else if drafts.len() >= 3 {
                step_verify_k4(model, a, &drafts, num_drafts);
            } else if drafts.len() >= 2 {
                step_verify_k3(model, a, &drafts, num_drafts);
            } else {
                step_verify_k2(model, a, &drafts, num_drafts);
            }
        }
        return;
    }

    // ── Phase B (default per-seq path): Verify with pipelined checkpoint ──
    for &idx in &verify_idxs {
        let a = &mut active[idx];
        let mut drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
        if drafts.is_empty() {
            continue;
        }

        // Spec-decode boundary awareness (arXiv:2512.15834): when a
        // grammar is active, validate the draft sequence against the
        // matcher and truncate at the first token that crosses a
        // grammar transition. Without this, a draft span that crosses
        // `</function>` (or any other structural boundary) gets
        // accepted by the verifier and emitted, but the post-emit
        // `accept_token` silently fails — desync'ing the grammar
        // from the output stream. Truncating here downgrades K=4 →
        // K=3 → K=2 cleanly.
        if let Some(ref mut gs) = a.grammar_state {
            let kept = truncate_drafts_at_grammar_boundary(gs, &drafts);
            if kept < drafts.len() {
                drafts.truncate(kept);
            }
            if drafts.is_empty() {
                continue;
            }
        }

        // DFlash γ-block drafters return ≥4 drafts per step (γ=16 typical).
        // The K=2/3/4 graphed paths are MTP-shaped and don't generalize past
        // K=4 cleanly, so γ-block verify routes through `step_verify_dflash`.
        // MTP keeps using the existing graphed paths; this dispatch is purely
        // additive.
        //
        // Fix (2026-05-12): route by `drafts.len()` not `num_drafts`. DFlash
        // produces a full γ-block regardless of `num_drafts`; capping via
        // ATLAS_DFLASH_DRAFT_CAP should not force K=2 verify and discard
        // valid drafts.
        if drafts.len() >= 4 {
            step_verify_dflash(model, a, &drafts, num_drafts);
        } else if drafts.len() >= 3 {
            step_verify_k4(model, a, &drafts, num_drafts);
        } else if drafts.len() >= 2 {
            step_verify_k3(model, a, &drafts, num_drafts);
        } else {
            step_verify_k2(model, a, &drafts, num_drafts);
        }
    }
}
