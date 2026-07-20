// SPDX-License-Identifier: AGPL-3.0-only

//! MTP speculative draft proposal step.

use super::*;

/// MTP-aware step: bootstrap sequences without drafts, then verify via CUDA graph.
/// Supports K=2 (num_drafts=1) and K=3 (num_drafts=2).
pub fn step_mtp(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    num_drafts: usize,
    think: &ThinkSpecCtx<'_>,
) {
    // Stage-1 DFlash grammar gate: drop drafts proposed before the grammar
    // became constraining (e.g. the block whose emission opened a
    // `<tool_call>`). DFlash drafts and the K=γ verify argmax bypass the
    // XGrammar bitmask, so a grammar-active sequence must run the bootstrap
    // path below, where `sample_token_with_grammar` enforces it.
    for a in active.iter_mut() {
        // ATLAS_DFLASH_ASYNC (task #20): the previous step's propose was
        // ENQUEUED on a second CUDA stream and returned a placeholder chain;
        // its drafter kernels overlapped the SSM commit tail + step-tail CPU
        // work. Collect the real drafts NOW (stream sync + γ×4B D2H) before
        // any gate below inspects `pending_drafts`. `Some(vec![])` = the
        // launch was orphaned → clear and bootstrap (lossless: drafts only
        // ever propose). No-op `None` on the sync path / flag off.
        if !a.pending_drafts.is_empty() {
            match model.dflash_collect_async_drafts(&mut a.seq) {
                Ok(Some(drafts)) => {
                    a.pending_drafts = drafts;
                    // ASYNC+DDTree: collect_async_drafts_impl may have built a
                    // tree payload from the previous step's top-K kernel. Drain
                    // it NOW so this step's verify (which checks a.pending_tree_payload
                    // inside step_verify_dflash) can use the tree. The normal
                    // sync-path drain at the end of step_verify_dflash
                    // (take_pending_tree_payload, line 883) runs AFTER
                    // run_mtp_propose_multi, which on the async path clears
                    // dstate.pending_tree_payload — so line 883 returns None and
                    // does not overwrite the tree we set here.
                    let tree = model.take_pending_tree_payload(&mut a.seq);
                    if tree.is_some() {
                        a.pending_tree_payload = tree;
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::error!("dflash_collect_async_drafts: {e:#}");
                    a.pending_drafts.clear();
                    a.pending_tree_payload = None;
                }
            }
        }
        if !a.pending_drafts.is_empty() && dflash_grammar_skip_propose(model, a) {
            a.pending_drafts.clear();
            a.pending_tree_payload = None;
        }
        // ATLAS_THINK_SPEC=1: the thinking accept filter (dflash_thinking_accept)
        // works on flat token chains only — it compares drafts[i] vs target[i]
        // linearly, but DDTree compact slots are BFS siblings, not descendants,
        // so passing a tree payload through yields at most 1 accepted token (the
        // depth-0 child) before the walk diverges. That is no better than
        // bootstrap but costs a k=32 verify.
        //
        // When inside thinking with a tree payload, drop the tree metadata but
        // KEEP the flat-chain pending_drafts (the top-1 drafter spine). The
        // sequence then falls through to the flat k=17 verify → dflash_thinking_accept,
        // which correctly handles budget accounting, EOS suppression, and the
        // </think> fence. Previously both were cleared → bootstrap (1 tok/step),
        // which caused the Arm C code_long slow-run regression.
        if think.enabled && a.inside_thinking && a.pending_tree_payload.is_some() {
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
        let tok = if think.enabled && a.inside_thinking {
            // ATLAS_THINK_SPEC=1: a thinking sequence's bootstrap token
            // must carry the plain path's logit interventions + per-token
            // side effects (F1/F2/wave, EOS suppression, fence parity),
            // which `sample_token_with_grammar` + `emit_token` do not
            // replicate. `bootstrap_thinking_token` routes this single
            // row through `process_seq_logits` — byte-identical to one
            // `step_decode_only` token — and commits it. The plain
            // decode's per-layer DFlash capture hook already ran inside
            // `model.decode` above, so drafter ctx conditioning matches
            // the propose/verify steps that follow.
            match bootstrap_thinking_token(model, a, logits, think) {
                Some(t) => t,
                None => continue, // D2H failure: sequence finished
            }
        } else {
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
            tok
        };
        if a.finished {
            continue;
        }
        a.last_token = tok;

        if let Err(e) = model.save_hidden_for_mtp(0, 0) {
            tracing::error!("save_hidden_for_mtp: {e:#}");
            continue;
        }
        // DFlash ctx-drift fix: reset last_num_accepted to 0 before the
        // bootstrap propose. After a tree verify (TREE_TOKENS_VERIFY=1, k=32)
        // that accepted n>0 tokens, trim_proposer_state stores n in
        // last_num_accepted. A THINK_SPEC downgrade (or any other cause of
        // missing drafts) then routes here: the bootstrap decode advances
        // seq_len by 1 without calling trim, so the propose computes
        // num_append = n+1 → first_pos = seq_len-(n+1) ≠ ctx_len = seq_len-1,
        // causing drift = n and corrupting the drafter's RoPE conditioning.
        // Calling trim(0) resets last_num_accepted=0 and clears the stale
        // accepted-compact path, so num_append = 1 → first_pos = ctx_len (no drift).
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 0, 0) {
            tracing::warn!("trim_proposer_state (bootstrap dflash reset): {e:#}");
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
            // Thinking spans never advance the grammar (mirrors the
            // `!inside_thinking` gate in emit_token / the bitmask-skip in
            // process_seq_logits), so draft validation against the
            // matcher would spuriously truncate to zero mid-`<think>`.
            // Only reachable while thinking under ATLAS_THINK_SPEC=1.
            if !a.inside_thinking
                && let Some(ref mut gs) = a.grammar_state
            {
                let kept = truncate_drafts_at_grammar_boundary(gs, &drafts);
                if kept < drafts.len() {
                    drafts.truncate(kept);
                }
                if drafts.is_empty() {
                    continue;
                }
            }
            if drafts.len() >= 4 {
                step_verify_dflash(model, a, &drafts, num_drafts, think);
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
            // Thinking spans never advance the grammar (mirrors the
            // `!inside_thinking` gate in emit_token / the bitmask-skip in
            // process_seq_logits), so draft validation against the
            // matcher would spuriously truncate to zero mid-`<think>`.
            // Only reachable while thinking under ATLAS_THINK_SPEC=1.
            if !a.inside_thinking
                && let Some(ref mut gs) = a.grammar_state
            {
                let kept = truncate_drafts_at_grammar_boundary(gs, &drafts);
                if kept < drafts.len() {
                    drafts.truncate(kept);
                }
                if drafts.is_empty() {
                    continue;
                }
            }
            if drafts.len() >= 4 {
                step_verify_dflash(model, a, &drafts, num_drafts, think);
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

    // ── Phase B (opt-in): CROSS-SEQ BATCHED DFLASH VERIFY (#39) ──
    // ATLAS_DFLASH_BATCHED_VERIFY=1: route ELIGIBLE γ-block verifies through
    // ONE batched forward (FFN weights read once across c seqs). Eligible =
    // drafts.len()>=4, equal K, no grammar, not thinking, no tree payload /
    // tree-or-portfolio method, not finished. Others fall through per-seq.
    if dflash_batched_verify_enabled() && verify_idxs.len() >= 2 {
        let mut batched_idxs: Vec<usize> = Vec::new();
        let mut leftover_idxs: Vec<usize> = Vec::new();
        // Reference K = first eligible seq's draft count; the batch must share it.
        let mut batch_k: Option<usize> = None;
        for &idx in &verify_idxs {
            let a = &active[idx];
            let dl = a.pending_drafts.len();
            let grammar_ok = a
                .grammar_state
                .as_ref()
                .is_none_or(|gs| gs.is_terminated());
            let eligible = dl >= 4
                && grammar_ok
                && !a.finished
                && !(think.enabled && a.inside_thinking)
                && a.pending_tree_payload.is_none()
                && !dflash_tree_method_active()
                && !dflash_portfolio_active()
                && batch_k.is_none_or(|k| k == dl);
            if eligible {
                batch_k.get_or_insert(dl);
                batched_idxs.push(idx);
            } else {
                leftover_idxs.push(idx);
            }
        }
        if batched_idxs.len() >= 2 {
            step_verify_dflash_batched(model, active, &batched_idxs, num_drafts);
        } else {
            leftover_idxs.extend(batched_idxs);
        }
        // Per-seq path for anything that didn't batch.
        for &idx in &leftover_idxs {
            let a = &mut active[idx];
            let mut drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
            if drafts.is_empty() {
                continue;
            }
            if !a.inside_thinking
                && let Some(ref mut gs) = a.grammar_state
            {
                let kept = truncate_drafts_at_grammar_boundary(gs, &drafts);
                if kept < drafts.len() {
                    drafts.truncate(kept);
                }
                if drafts.is_empty() {
                    continue;
                }
            }
            if drafts.len() >= 4 {
                step_verify_dflash(model, a, &drafts, num_drafts, think);
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
        //
        // Skipped inside `<think>` (reachable only under
        // ATLAS_THINK_SPEC=1): thinking tokens never advance the grammar,
        // so matcher validation would spuriously truncate to zero.
        if !a.inside_thinking
            && let Some(ref mut gs) = a.grammar_state
        {
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
            step_verify_dflash(model, a, &drafts, num_drafts, think);
        } else if drafts.len() >= 3 {
            step_verify_k4(model, a, &drafts, num_drafts);
        } else if drafts.len() >= 2 {
            step_verify_k3(model, a, &drafts, num_drafts);
        } else {
            step_verify_k2(model, a, &drafts, num_drafts);
        }
    }
}
