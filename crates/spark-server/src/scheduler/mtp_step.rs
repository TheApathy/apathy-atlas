// SPDX-License-Identifier: AGPL-3.0-only

//! MTP speculative draft proposal step.

use super::*;
use proposal_lifecycle::{
    clear_orphan_tree, install_collected, propose_and_install, retain_prefix,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerifyDispatch {
    Bootstrap,
    K2,
    K3,
    K4,
    Generic,
}

/// Pick the verifier from draft width and whether raw argmax is policy-safe.
fn select_verify_dispatch(
    draft_len: usize,
    policy_required: bool,
    has_tree: bool,
) -> VerifyDispatch {
    if draft_len >= 4 {
        return VerifyDispatch::Generic;
    }
    if draft_len == 0 {
        return VerifyDispatch::Bootstrap;
    }
    if policy_required || has_tree {
        return VerifyDispatch::Generic;
    }
    match draft_len {
        1 => VerifyDispatch::K2,
        2 => VerifyDispatch::K3,
        3 => VerifyDispatch::K4,
        _ => VerifyDispatch::Bootstrap,
    }
}

fn verify_policy_required(a: &ActiveSeq, think: &ThinkSpecCtx<'_>) -> bool {
    if a.inside_thinking {
        think.enabled
    } else {
        content_policy_required(a, think)
    }
}

fn batched_dflash_verify_allowed(
    enabled: bool,
    sequence_count: usize,
    is_ep: bool,
    proposer_is_dflash: bool,
) -> bool {
    enabled && sequence_count >= 2 && !is_ep && proposer_is_dflash
}

/// Save the accepted bonus hidden state into the buffer consumed by the active
/// proposer. DFlash keys its captured row by the emitted token; native MTP
/// consumes the accepted row index from `mtp_hidden_save`.
pub(super) fn save_hidden_for_active_proposer(
    model: &dyn Model,
    a: &mut ActiveSeq,
    token: u32,
    row: usize,
) -> Result<()> {
    if model.proposer_is_dflash() {
        model.save_hidden_for_dflash(token, &mut a.seq, 0)
    } else {
        model.save_hidden_for_mtp(row, 0)
    }
}

fn dispatch_verify(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    num_drafts: usize,
    think: &ThinkSpecCtx<'_>,
) {
    let policy_required = verify_policy_required(a, think);
    let dispatch = select_verify_dispatch(
        drafts.len(),
        policy_required,
        a.pending_tree_payload.is_some(),
    );
    if dispatch != VerifyDispatch::Generic {
        // Only the generic verifier consumes tree topology.
        a.pending_tree_payload = None;
        // ...and only it attributes accepted tokens to a retrieval tier. A
        // frame narrowed below the generic width (grammar truncation)
        // leaves the counters alone rather than crediting the next frame's
        // acceptances to retrieval.
        a.draft_origin = DraftOrigin::Proposer;
    }
    match dispatch {
        VerifyDispatch::Generic => {
            step_verify_dflash(model, a, drafts, num_drafts, think);
        }
        VerifyDispatch::K4 => step_verify_k4(model, a, drafts, num_drafts),
        VerifyDispatch::K3 => step_verify_k3(model, a, drafts, num_drafts),
        VerifyDispatch::K2 => step_verify_k2(model, a, drafts, num_drafts),
        VerifyDispatch::Bootstrap => {
            debug_assert!(drafts.is_empty());
        }
    }
    // Fail-closed exits cannot retain an orphaned old tree.
    clear_orphan_tree(&a.pending_drafts, &mut a.pending_tree_payload);
}

fn suppress_finished_verify_cache(
    active: &[ActiveSeq],
    verify_idxs: &[usize],
    cache_on_finish: &mut [bool],
) {
    debug_assert_eq!(active.len(), cache_on_finish.len());
    for &idx in verify_idxs {
        if active[idx].finished {
            cache_on_finish[idx] = false;
        }
    }
}

/// Ask each retrieval tier, in cascade order, for a chain to pre-empt the
/// drafter with.
///
/// Self-context goes first: it is drawn from this sequence's own history,
/// so when it matches at all it matches text the model demonstrably just
/// produced. The static store is the broader, less specific fallback.
/// Both are inert unless their own environment gate is set, and both
/// return a chain of exactly `num_drafts` tokens or nothing.
///
/// `tok` is the token just sampled, or the bonus just accepted. At BOTH
/// propose sites it is the sequence's next token and is not yet in
/// `a.seq.tokens` — the verifier pushes only its INPUT tokens
/// (`verify_d.rs`) — so the committed history is `a.seq.tokens ++ [tok]`
/// and the chain continues from there.
pub(super) fn retrieval_chain(
    a: &mut ActiveSeq,
    tok: u32,
    num_drafts: usize,
) -> Option<(DraftOrigin, Vec<u32>)> {
    if num_drafts < crate::rest_store::MIN_PREEMPT_WIDTH {
        return None;
    }
    // Do not pre-empt a drafter that is currently winning. Retrieval
    // engages on repetitive text, which is exactly where the neural
    // drafter is also strongest — the engine has been observed accepting
    // 15/15 on repetitive parser code. Displacing a frame like that costs
    // the difference, and the offline eval cannot see it because it has
    // no drafter. `last_verify_accepted` can: it is what the drafter just
    // did, on this sequence, on this kind of text.
    let drafter_recent = usize::from(a.last_verify_accepted);
    if crate::rest_store::self_context::enabled()
        && drafter_recent < crate::rest_store::self_context::max_drafter_accept()
    {
        // `a.seq.tokens` is prompt + everything committed so far; `tok`
        // was sampled this step and is not in it yet.
        let chain = a.self_context.propose(&a.seq.tokens, tok, num_drafts);
        if let Some(chain) = chain {
            return Some((DraftOrigin::SelfContext, chain));
        }
    }
    if drafter_recent >= crate::rest_store::max_drafter_accept() {
        return None;
    }
    let chain = crate::rest_store::preempt(&a.seq.tokens, tok, num_drafts)?;
    Some((DraftOrigin::RestStore, chain))
}

/// Bootstrap empty sequences, then dispatch policy-safe speculative verify.
pub fn step_mtp(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    num_drafts: usize,
    think: &ThinkSpecCtx<'_>,
    cache_on_finish: &mut [bool],
) {
    assert_eq!(active.len(), cache_on_finish.len());
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
                    // ASYNC+DDTree: collect_async_drafts_impl may have built a
                    // tree payload from the previous step's top-K kernel. Drain
                    // it NOW so this step's verify (which checks a.pending_tree_payload
                    // inside step_verify_dflash) can use the tree. The normal
                    // sync-path drain at the end of step_verify_dflash
                    // (take_pending_tree_payload, line 883) runs AFTER
                    // run_mtp_propose_multi, which on the async path clears
                    // dstate.pending_tree_payload — so line 883 returns None and
                    // does not overwrite the tree we set here.
                    install_collected(model, a, Ok(drafts))
                        .expect("an infallible collected proposal cannot fail");
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
        // A tree walk is indexed by raw verify argmaxes. When the serving
        // sampler has any active policy (Qwen defaults include presence/LZ/DRY
        // penalties), or a post-think mask cannot prove every tree row safe,
        // correctness requires a linear row-by-row policy walk. Keep the
        // drafter's top-1 spine but discard the sibling topology.
        if !a.inside_thinking
            && content_policy_required(a, think)
            && a.pending_tree_payload.is_some()
        {
            if tree_content_raw_argmax_eligible(a, think) {
                static TREE_POLICY_KEEP_DBG: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let n = TREE_POLICY_KEEP_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 3 {
                    tracing::info!("DDTree policy gate: keeping neutral-greedy tree");
                }
            } else {
                static TREE_POLICY_DROP_DBG: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let n = TREE_POLICY_DROP_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 3 {
                    tracing::info!(
                        "DDTree policy gate: dropping tree; using flat sampler-policy walk"
                    );
                }
                a.pending_tree_payload = None;
            }
        }
        // EP synchronizes flat variable-width token frames, but DDTree parent
        // topology/branch compaction is not part of that protocol. Preserve
        // the proposer's top-1 spine and fail closed to flat verification.
        if model.is_ep() && a.pending_tree_payload.is_some() {
            a.pending_tree_payload = None;
        }
        clear_orphan_tree(&a.pending_drafts, &mut a.pending_tree_payload);
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
    // A previous verify may have committed its live SSM state asynchronously
    // and then intentionally left drafts empty (grammar transition, proposer
    // miss, diagnostic arm, or error recovery).  Every verify entry waits for
    // that restore; the bootstrap edge must do the same before `decode` reads
    // h_state/conv_state.  Missing this wait made the next token depend on GPU
    // timing and looked like a corrupt partial-accept intermediate.
    if !bootstrap_idxs.is_empty()
        && let Err(e) = model.sync_secondary()
    {
        tracing::error!("sync_secondary before MTP bootstrap: {e:#}");
        for &idx in &bootstrap_idxs {
            active[idx].finished = true;
        }
        return;
    }
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
                Some(t) => {
                    if std::env::var("ATLAS_SPEC_BOOTSTRAP_TRACE").ok().as_deref() == Some("1") {
                        tracing::info!(sampled = t, "SPEC_THINK_BOOTSTRAP");
                    }
                    t
                }
                None => continue, // D2H failure: sequence finished
            }
        } else if !a.inside_thinking && content_policy_required(a, think) {
            // DFlash bootstrap must use the exact same sampler as ordinary
            // decode.  Raw `sample_token_with_grammar` omits history
            // penalties, logit bias, think masks, and adaptive state.
            match bootstrap_content_token(model, a, logits, think) {
                Some(t) => t,
                None => continue,
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

            if std::env::var("ATLAS_SPEC_BOOTSTRAP_TRACE").ok().as_deref() == Some("1") {
                tracing::info!(sampled = tok, "SPEC_RAW_BOOTSTRAP");
            }

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
            // ── REST conditional pre-emption (default off) ──
            //
            // When the retrieval store holds a long enough verbatim match
            // for this context, its continuation replaces the drafter's
            // chain for THIS step only — the drafter forward pass is then
            // skipped, which is where the second half of the win comes
            // from. REST never displaces DFlash: `preempt` declines unless
            // the match and the resulting chain clear gates set well above
            // break-even, and every decline falls through to the unchanged
            // proposal below. `ATLAS_REST_STORE` unset ⇒ always `None`.
            //
            // Skipping one propose costs at most one stale ctx-accumulator
            // slot in the DFlash proposer, not a lasting shift: the append
            // writes at ABSOLUTE positions and self-realigns on the next
            // propose (`dflash_head/propose.rs`, "A skipped step now costs
            // one stale slot, not a permanent shift"). Verification is
            // untouched, so emitted tokens are unchanged either way.
            //
            // Cascade order: self-context first — it is the tier most
            // specific to THIS sequence and needs no corpus — then the
            // static store, then the drafter. Each tier declines cheaply
            // when its own env gate is unset.
            let retrieved = retrieval_chain(a, tok, num_drafts);
            if let Some((origin, chain)) = retrieved {
                let installed = proposal_lifecycle::install_external_flat(model, a, chain);
                debug_assert!(installed, "a proposed retrieval chain is never empty");
                if installed {
                    a.draft_origin = origin;
                }
                tracing::debug!(
                    "retrieval bootstrap ({origin:?}): tok={tok} → drafts={:?}",
                    a.pending_drafts
                );
            } else {
                let _mtp_grammar_mask = mtp_grammar_mask_for(a);
                match propose_and_install(model, a, tok, num_drafts, _mtp_grammar_mask.as_deref()) {
                    Ok(true) => {
                        tracing::debug!("MTP bootstrap: tok={tok} → drafts={:?}", a.pending_drafts);
                    }
                    Ok(false) => tracing::warn!("MTP propose returned empty"),
                    Err(e) => tracing::error!("run_mtp_propose_multi: {e:#}"),
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
            let eligibility = proposal_lifecycle::FixedBatchEligibility {
                draft_len: a.pending_drafts.len(),
                grammar_active: a.grammar_state.is_some(),
                finished: a.finished,
                configured_num_drafts: num_drafts,
                policy_required: verify_policy_required(a, think),
                has_tree: a.pending_tree_payload.is_some(),
            };
            if proposal_lifecycle::fixed_batch_decision(
                proposal_lifecycle::FixedBatchWidth::K2,
                eligibility,
            )
            .is_eligible()
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
                if !retain_prefix(&mut drafts, &mut a.pending_tree_payload, kept) {
                    continue;
                }
            }
            dispatch_verify(model, a, &drafts, num_drafts, think);
        }
        suppress_finished_verify_cache(active, &verify_idxs, cache_on_finish);
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
            let eligibility = proposal_lifecycle::FixedBatchEligibility {
                draft_len: a.pending_drafts.len(),
                grammar_active: a.grammar_state.is_some(),
                finished: a.finished,
                configured_num_drafts: num_drafts,
                policy_required: verify_policy_required(a, think),
                has_tree: a.pending_tree_payload.is_some(),
            };
            if proposal_lifecycle::fixed_batch_decision(
                proposal_lifecycle::FixedBatchWidth::K3,
                eligibility,
            )
            .is_eligible()
            {
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
                if !retain_prefix(&mut drafts, &mut a.pending_tree_payload, kept) {
                    continue;
                }
            }
            dispatch_verify(model, a, &drafts, num_drafts, think);
        }
        suppress_finished_verify_cache(active, &verify_idxs, cache_on_finish);
        return;
    }

    // ── Phase B (opt-in): CROSS-SEQ BATCHED DFLASH VERIFY (#39) ──
    // ATLAS_DFLASH_BATCHED_VERIFY=1: route ELIGIBLE γ-block verifies through
    // ONE batched forward (FFN weights read once across c seqs). Eligible =
    // drafts.len()>=4, equal K, no grammar, not thinking, no tree payload /
    // tree-or-portfolio method, not finished. Others fall through per-seq.
    if batched_dflash_verify_allowed(
        dflash_batched_verify_enabled(),
        verify_idxs.len(),
        model.is_ep(),
        model.proposer_is_dflash(),
    ) {
        let mut batched_idxs: Vec<usize> = Vec::new();
        let mut leftover_idxs: Vec<usize> = Vec::new();
        // Reference K = first eligible seq's draft count; the batch must share it.
        let mut batch_k: Option<usize> = None;
        for &idx in &verify_idxs {
            let a = &active[idx];
            let dl = a.pending_drafts.len();
            let grammar_ok = a.grammar_state.as_ref().is_none_or(|gs| gs.is_terminated());
            let eligible = dl >= 4
                && grammar_ok
                && !a.finished
                && !(think.enabled && a.inside_thinking)
                && !content_policy_required(a, think)
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
            // The batched verifier reports no per-sequence acceptance the
            // retrieval counters could consume, so clear provenance here: under
            // ATLAS_DFLASH_BATCHED_VERIFY=1 (default off) engagement and
            // skipped-drafter counts stay exact and accepted-from-retrieval
            // undercounts rather than crediting the wrong frame.
            for &idx in &batched_idxs {
                active[idx].draft_origin = DraftOrigin::Proposer;
            }
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
                if !retain_prefix(&mut drafts, &mut a.pending_tree_payload, kept) {
                    continue;
                }
            }
            dispatch_verify(model, a, &drafts, num_drafts, think);
        }
        suppress_finished_verify_cache(active, &verify_idxs, cache_on_finish);
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
            if !retain_prefix(&mut drafts, &mut a.pending_tree_payload, kept) {
                continue;
            }
        }

        // DFlash γ-block drafters return ≥4 drafts per step (γ=15 for a
        // block-16 checkpoint, whose first row is the known anchor), so
        // wide windows route through `step_verify_dflash`. Neutral MTP windows
        // keep the fixed K2/K3/K4 graphs; short policy-sensitive windows also
        // use the generic path because the fixed paths expose only raw argmax.
        //
        // Fix (2026-05-12): route by `drafts.len()` not `num_drafts`. DFlash
        // produces a full γ-block regardless of `num_drafts`; capping via
        // ATLAS_DFLASH_DRAFT_CAP should not force K=2 verify and discard
        // valid drafts.
        dispatch_verify(model, a, &drafts, num_drafts, think);
    }
    suppress_finished_verify_cache(active, &verify_idxs, cache_on_finish);
}

#[cfg(test)]
mod tests {
    use super::{VerifyDispatch, batched_dflash_verify_allowed, select_verify_dispatch};

    #[test]
    fn cross_sequence_dflash_batching_requires_dflash_proposer() {
        assert!(batched_dflash_verify_allowed(true, 2, false, true));
        assert!(!batched_dflash_verify_allowed(true, 2, false, false));
        assert!(!batched_dflash_verify_allowed(true, 1, false, true));
        assert!(!batched_dflash_verify_allowed(true, 2, true, true));
        assert!(!batched_dflash_verify_allowed(false, 2, false, true));
    }

    #[test]
    fn specialized_verifiers_route_hidden_save_through_proposer_pairing() {
        let cases = [
            (
                "batched DFlash",
                include_str!("verify_dflash_batched_step.rs"),
                1,
            ),
            ("K3", include_str!("verify_k3_step.rs"), 3),
            ("CSK", include_str!("verify_csk_step.rs"), 3),
        ];
        for (label, source, expected_calls) in cases {
            assert_eq!(
                source.matches("save_hidden_for_active_proposer(").count(),
                expected_calls,
                "{label} must pair every accepted outcome with the active proposer"
            );
            assert!(
                !source.contains("model.save_hidden_for_mtp("),
                "{label} must not hard-code the native-MTP hidden buffer"
            );
            assert!(
                !source.contains("model.save_hidden_for_dflash("),
                "{label} must not hard-code the DFlash hidden buffer"
            );
        }
    }

    #[test]
    fn neutral_short_windows_keep_fixed_verifiers() {
        assert_eq!(select_verify_dispatch(1, false, false), VerifyDispatch::K2);
        assert_eq!(select_verify_dispatch(2, false, false), VerifyDispatch::K3);
        assert_eq!(select_verify_dispatch(3, false, false), VerifyDispatch::K4);
    }

    #[test]
    fn policy_short_windows_use_generic_verifier() {
        for draft_len in 1..=3 {
            assert_eq!(
                select_verify_dispatch(draft_len, true, false),
                VerifyDispatch::Generic,
                "draft_len={draft_len}"
            );
        }
    }

    #[test]
    fn empty_bootstraps_and_wide_windows_use_generic_verify() {
        assert_eq!(
            select_verify_dispatch(0, false, false),
            VerifyDispatch::Bootstrap
        );
        assert_eq!(
            select_verify_dispatch(0, true, false),
            VerifyDispatch::Bootstrap
        );
        for draft_len in 4..=17 {
            assert_eq!(
                select_verify_dispatch(draft_len, false, false),
                VerifyDispatch::Generic,
                "draft_len={draft_len}"
            );
        }
    }

    #[test]
    fn tree_bearing_width_one_and_two_use_the_topology_consumer() {
        assert_eq!(
            select_verify_dispatch(1, false, true),
            VerifyDispatch::Generic
        );
        assert_eq!(
            select_verify_dispatch(2, false, true),
            VerifyDispatch::Generic
        );
    }
}

#[cfg(test)]
#[path = "rest_preempt_tests.rs"]
mod rest_preempt_tests;
