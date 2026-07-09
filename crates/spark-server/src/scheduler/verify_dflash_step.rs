// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash-based verify step (drafted token verification).

use super::*;

/// ATLAS_DFLASH_BRANCH_AUDIT=1 cross-step stash: session_hash →
/// `(expected_argmax_after_fork, fork_token)` recorded on a step whose
/// flat-safe commit ended with the fork token as bonus. The NEXT verify for
/// the same session compares its root row (`verified[0]`, the TRUE greedy
/// after the fork) against the fork ROW's argmax from the previous step —
/// a direct, per-step-deterministic measurement of branch-row conditioning
/// correctness (the 2026-07-08 deep-branch root cause). Debug-only.
fn branch_audit_stash() -> &'static std::sync::Mutex<std::collections::HashMap<u64, (u32, u32)>> {
    static STASH: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<u64, (u32, u32)>>,
    > = std::sync::OnceLock::new();
    STASH.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Running audit tally: (matches, mismatches). Logged on every comparison.
static BRANCH_AUDIT_TALLY: std::sync::Mutex<(u64, u64)> = std::sync::Mutex::new((0, 0));

/// DFlash γ-token verify with accept-prefix.
///
/// Phase 3 minimal-viable implementation: routes `[last_token, drafts...]`
/// through the eager `decode_verify_dflash` path (which today defaults to
/// `decode_verify`) and finds the first index where draft ≠ verified
/// argmax. Tokens 0..first_mismatch are accepted; the verified token at
/// the mismatch position becomes the bonus token; subsequent drafts are
/// dropped.
///
/// Deferred to Phase 6 (full integration):
///   * EP=2 broadcast of verify-cmd + drafts (drafter currently runs only
///     on rank 0; verify on a single-rank target is correct, but EP=2 needs
///     the broadcast pattern from `step_verify_k2`).
///   * Per-position logprobs extraction.
///   * SSM `commit_verify_state_async(num_accepted, k)` loop. Without it,
///     hybrid models (Qwen3.6-A3B has GDN layers) will see SSM state drift
///     after γ-verify. Single-token decode unaffected; γ-verify only
///     correct on pure-attention targets until this is wired.
///   * `save_hidden_for_mtp` / `save_hidden_for_dflash` hook on the
///     accepted bonus token (the next propose() needs the latest hidden).
///   * Sliding-window state rollback for sliding-attention layers
///     (Gemma-4-style; not used by Qwen3.6 targets).
pub fn step_verify_dflash(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    num_drafts: usize,
    think: &ThinkSpecCtx<'_>,
) {
    // ATLAS_DFLASH_STEP_TIMING=1: per-phase wall-clock breakdown of the
    // verify step, logged once per step. Companion to ATLAS_DFLASH_PROPOSE_LOG
    // (which only covers the re-propose at the tail of this function).
    let step_timing = std::env::var("ATLAS_DFLASH_STEP_TIMING").ok().as_deref() == Some("1");
    let t_step = Instant::now();

    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        a.finished = true;
        return;
    }
    let t_sync_secondary_us = t_step.elapsed().as_micros();

    // tokens = [last_verified, draft_0, draft_1, ..., draft_{γ-1}]
    //
    // ATLAS_DDTREE_TREE_TOKENS_VERIFY=1 (the final piece of the M4B v2 tree-
    // aware verify chain): when a tree payload is present, replace the linear
    // top-1 `drafts` with `payload.tree_token_ids` — the actual tokens the
    // tree topology says live at each compact slot. With this off (default),
    // verify is fed [last_token, ...linear_drafts], which puts semantically
    // wrong tokens at chain-depth slots whenever the tree builder produced
    // sibling/branch nodes (proven to corrupt counting output: "1 to-do"
    // instead of "1 to 60").
    //
    // Source-frame contract: `tokens[0]` is the bonus (kernel slot 0 = root
    // of the tree, always carries `a.last_token`); `tokens[i]` for i >= 1
    // is the token at kernel/compact slot i. `payload.tree_token_ids[i-1]`
    // is the compact-slot-i token by construction (TreePayload skips the
    // bonus). The DFS reorder layer in verify_d.rs permutes these into DFS
    // slots internally via `tokens[dfs_perm[t]]`, and un-permutes verified
    // outputs back to original-compact frame before returning, so the
    // scheduler always sees the kernel-compact frame.
    //
    // Backward compatibility: when chain_only=1 / chain_seed=true with no
    // branching, `tree_token_ids == drafts` and the two paths produce
    // identical bytes. When the tree is non-flat (M4B v2 with branches),
    // this fix puts the RIGHT tokens at the RIGHT slots so the verifier's
    // per-position argmax aligns with the tree topology.
    let tree_tokens_verify = std::env::var("ATLAS_DDTREE_TREE_TOKENS_VERIFY")
        .ok()
        .as_deref()
        == Some("1");
    let use_tree_tokens = tree_tokens_verify
        && a.pending_tree_payload
            .as_ref()
            .is_some_and(|p| !p.is_empty());

    let mut tokens = Vec::with_capacity(drafts.len() + 1);
    tokens.push(a.last_token);
    // Tracks the per-compact-slot tokens (excluding the bonus at slot 0)
    // that were ACTUALLY fed to the verifier; emit/compare paths below
    // need this to match the topology-induced acceptance pattern.
    let mut verify_input_tokens: Vec<u32> = Vec::with_capacity(drafts.len());
    if use_tree_tokens {
        // The tree builder caps at `budget` candidates while `drafts.len()
        // == γ_eff` may be larger. Kernel/parent_ids padding (verify_d.rs)
        // appends linear-chain links for indices >= tree_token_ids.len(),
        // so the only sensible tokens for those tail slots are `drafts[i]`
        // (the linear top-1 chain). For indices < tree_token_ids.len(),
        // substitute the tree topology's token. Verifier sees the right
        // token at every kernel slot.
        let payload = a.pending_tree_payload.as_ref().expect("checked above");
        let tree_len = payload.tree_token_ids.len();
        // Verify width = the FULL tree. When the DDTree budget exceeds γ the
        // tree has MORE nodes than `drafts` (wide branching: top-k siblings
        // expanded per depth), so the verifier must process all `tree_len`
        // nodes — not just `drafts.len()`. Truncating to γ would feed nodes
        // whose `parent_indices` reference siblings beyond the verified set,
        // corrupting the tree topology (observed: counting non-lossless +
        // accept collapse). When the tree is NARROWER than γ (budget < γ),
        // chain-pad the tail with the linear top-1 `drafts` so verify still
        // covers γ depths (verify_d.rs pads parent_ids with linear-chain
        // links for indices >= tree_len, matching this padding).
        let verify_len = tree_len.max(drafts.len());
        for i in 0..verify_len {
            let tok = if i < tree_len {
                payload.tree_token_ids[i]
            } else {
                drafts[i]
            };
            tokens.push(tok);
            verify_input_tokens.push(tok);
        }
        static TREE_TOK_DBG: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let n = TREE_TOK_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 3 {
            tracing::info!(
                "ATLAS_DDTREE_TREE_TOKENS_VERIFY #{n}: γ={} tree_len={} tree_tokens[..8]={:?} drafts[..8]={:?} mix[..8]={:?}",
                drafts.len(),
                tree_len,
                &payload.tree_token_ids[..tree_len.min(8)],
                &drafts[..drafts.len().min(8)],
                &verify_input_tokens[..verify_input_tokens.len().min(8)],
            );
        }
    } else {
        tokens.extend_from_slice(drafts);
        verify_input_tokens.extend_from_slice(drafts);
    }

    // M8A: if a tree payload is present from the previous propose, upload its
    // parent_indices to the model's per-step scratch so the GDN dispatch can
    // fire gdn_tree_k. Cleared after verify completes (Ok or Err).
    if let Some(payload) = a.pending_tree_payload.as_ref()
        && let Err(e) = model.set_ddtree_parent_ids(payload)
    {
        tracing::warn!("set_ddtree_parent_ids failed (falling back to flat): {e:#}");
    }

    let t_verify = Instant::now();
    let verified = match model.decode_verify_dflash(&tokens, &mut a.seq, 0) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("decode_verify_dflash: {e:#}");
            model.clear_ddtree_parent_ids();
            a.finished = true;
            return;
        }
    };
    let t_verify_us = t_verify.elapsed().as_micros();
    // Note: clear_ddtree_parent_ids is deferred until AFTER
    // commit_verify_state_async so the commit knows tree mode was active
    // and routes through the partial-accept (intermediate→h_state) path.
    a.last_token_time = Instant::now();

    // ── ATLAS_DFLASH_BRANCH_AUDIT=1: settle the previous step's stash ──
    // `verified[0]` is the target's TRUE argmax after `tokens[0]`
    // (= the committed bonus of the previous step). If the previous step
    // stashed a fork-row prediction for that exact context, compare it now.
    // Only judged when the committed token actually IS the stashed fork
    // token (grammar/think interventions could have replaced it).
    if let Some((expected, fork_tok)) = {
        let mut stash = branch_audit_stash().lock().unwrap();
        stash.remove(&a.session_hash)
    } {
        if tokens[0] == fork_tok
            && let Some(&actual) = verified.first()
        {
            let is_match = actual == expected;
            let (m, mm) = {
                let mut t = BRANCH_AUDIT_TALLY.lock().unwrap();
                if is_match {
                    t.0 += 1;
                } else {
                    t.1 += 1;
                }
                *t
            };
            tracing::info!(
                "BRANCH_AUDIT: fork_tok={fork_tok} expected_after_fork={expected} \
                 true_after_fork={actual} match={} tally={m}/{}",
                is_match as u8,
                m + mm,
            );
        }
    }

    // `decode_verify` already advanced `seq.seq_len` by `tokens.len()` and
    // pushed all γ+1 tokens into `seq.tokens`. The accept-prefix logic below
    // determines how many to keep — the rest must be rolled back so the
    // KV cache, SSM state, and emitted token sequence stay consistent.

    // Accept-prefix: drafts[i] is "accepted" iff drafts[i] == verified[i].
    // verified[i] is the target's argmax at position i (i.e. its
    // prediction for what should follow `tokens[i]`). drafts[i] was the
    // proposer's guess for the same slot. First mismatch terminates the
    // accepted prefix; verified[first_mismatch] becomes the bonus token.
    //
    // M7A (DDTree): when a.pending_tree_payload is Some, the drafts/verified
    // pair correspond to compact tree-row indices, NOT a flat chain. The
    // tree walk in spark-model::layers::ddtree::greedy_sample_ddtree would
    // return non-contiguous accepted_compact_indices. AEON-7's M11A guard
    // demands we only commit the *flat prefix* of those compact indices —
    // exactly what the existing greedy loop does for the flat path, so the
    // accept logic is the same. The DIFFERENCE for tree mode is that the
    // KV slots at non-flat compact indices need to be compacted back to
    // contiguous [0..num_accepted] in the KV cache before the next decode.
    // For Atlas's K=γ verify path this compaction is a no-op because
    // tokens were already written at compact indices [0..K-1] sequentially
    // in slot_mapping; the next propose simply reads from
    // [num_accepted..K-1] for rollback. So M7A's plumbing is in place.
    // Real branch-mode compaction lands when M8A's GDN kernel actually
    // walks non-flat tree branches.
    // M11A: when tree mode is active, walk the tree topology to find the
    // semantic accept depth. Greedy walker returns the flat-prefix-safe
    // accepted compact indices + bonus token. For chain+sibling trees this
    // correctly truncates when greedy walk diverges from compact-index-order.
    // For flat-chain payloads it degenerates to the linear loop below.
    //
    // Returns `(num_accepted, tree_last_inter_slot)`:
    //   - `num_accepted`: count of drafts accepted (excludes the prefix
    //     bonus, which is always "accepted" downstream).
    //   - `tree_last_inter_slot`: `Some(slot)` when the tree-aware GDN
    //     kernel was used → the kernel intermediate slot of the last
    //     accepted state (== max compact index, or 0 when no drafts
    //     accepted). `None` for flat-chain → commit derives slot from
    //     `num_accepted - 1` (chain-contiguous).
    // Stage-2 grammar enforcement (ATLAS_DFLASH_GRAMMAR_MODE=verify): the
    // `verified` vec came from an UNMASKED GPU argmax, so for a
    // grammar-constrained sequence each position's target token must be
    // recomputed as masked argmax over the verify logits (still resident in
    // the model's `[K, vocab]` BF16 logits buffer at this point — nothing
    // touches the model between the verify above and here). `Some((n, b))`
    // overrides the accept count and bonus token below; `None` falls
    // through to the unmasked paths (grammar inactive / tree mode / fp32
    // logits / D2H failure).
    // ATLAS_THINK_SPEC=1 thinking-span accept filter (sibling of the
    // grammar path below, same walk-and-truncate shape): re-derives the
    // PLAIN-path token per verify position — F1 reflection suppression,
    // F2 confidence early stop, efficiency wave, tool-call mask, forced
    // `</think>` injection, in-thinking EOS suppression — and truncates
    // acceptance at the first divergence or phase boundary. The walk
    // COMMITS (streams) its accepted tokens + bonus itself, because the
    // in-thinking side effects (suppressed EOS, `</think>` transition,
    // fence parity, THINK_LOOP watchdog) don't fit the plain emit loop
    // below. Mutually exclusive with the grammar path: grammar masking is
    // suspended inside `<think>` (dflash_masked_accept bails on
    // `inside_thinking`), and tree payloads are cleared for thinking
    // sequences in step_mtp before verify.
    let thinking_accept: Option<ThinkAcceptOutcome> =
        if think.enabled && a.inside_thinking && a.pending_tree_payload.is_none() {
            run_dflash_thinking_accept(model, a, &verify_input_tokens, &verified, think)
        } else {
            None
        };

    let grammar_accept: Option<(usize, u32)> = if thinking_accept.is_none()
        && dflash_grammar_mode() == DflashGrammarMode::Verify
        && a.pending_tree_payload.is_none()
    {
        dflash_masked_accept(model, a, &verify_input_tokens, &verified)
    } else {
        None
    };

    // FIX 1 — Tree-path commit (ATLAS_DFLASH_TREE_COMMIT=1, default off).
    // When active for a tree payload, the greedy walk commits the WHOLE
    // accepted path (incl. a sibling-fork tail), not just the contiguous flat
    // prefix. `tree_accepted_path` carries the (possibly non-contiguous)
    // compact-index path + the bonus row so the emit + KV-compaction below
    // can lay tokens/KV correctly. `None` on every flat-chain / non-tree path
    // → legacy contiguous behavior unchanged.
    let tree_commit_enabled =
        std::env::var("ATLAS_DFLASH_TREE_COMMIT").ok().as_deref() == Some("1");
    // 2026-07-08 deep-branch-tail root-cause guard: the FULL walker consumes
    // branch rows' argmaxes (child-acceptance + bonus at the path tip), so it
    // is only sound when the verify that just ran gave EVERY tree node
    // ancestor-exact attention (per-row KV indirection —
    // ATLAS_DDTREE_TREE_AWARE_VERIFY=1 + ATLAS_TREE_AWARE_ATTN=1). Under
    // prefix-read metadata (incl. ATLAS_DDTREE_DFS_REORDER=1) only the spine
    // rows are exact: a branch row reads its spine SIBLING and misses its own
    // key, so committing through it diverges from the greedy oracle
    // (VALIDATION-36 TEST 1 md5 corruption). Degrade to the flat-safe walker,
    // which never reads a branch row.
    let ancestor_attn_exact = model.dflash_tree_ancestor_attn_exact();
    let tree_commit_active = tree_commit_enabled && ancestor_attn_exact;
    if tree_commit_enabled && !ancestor_attn_exact && a.pending_tree_payload.is_some() {
        static DOWNGRADE_DBG: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let n = DOWNGRADE_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 3 {
            tracing::warn!(
                "ATLAS_DFLASH_TREE_COMMIT=1 but the verify was NOT ancestor-exact \
                 (prefix/DFS metadata on a non-flat tree) — deep-branch commit would \
                 corrupt; degrading to the flat-safe walker. Set \
                 ATLAS_DDTREE_TREE_AWARE_VERIFY=1 + ATLAS_TREE_AWARE_ATTN=1 (and unset \
                 ATLAS_DDTREE_DFS_REORDER) for lossless deep-branch commits."
            );
        }
    }
    let mut tree_accepted_path: Option<(Vec<usize>, usize)> = None;
    let (num_accepted, tree_last_inter_slot) = if let Some(ref t) = thinking_accept {
        (t.num_accepted, None)
    } else if let Some((n, _)) = grammar_accept {
        (n, None)
    } else if let Some(payload) = a.pending_tree_payload.as_ref() {
        use spark_model::layers::dflash_head::ddtree::{
            DDTreeRequestRuntime, greedy_sample_ddtree, greedy_sample_ddtree_full,
            last_accepted_inter_slot,
        };
        let req = DDTreeRequestRuntime {
            req_id: String::new(),
            tree_token_ids: payload.tree_token_ids.clone(),
            parent_indices: payload.parent_indices.clone(),
        };
        // greedy_sample_ddtree expects argmax for [root_row, ...compact_rows].
        // Our `verified` is exactly [verify_at_last_token, verify_at_drafts[0],
        // ..., verify_at_drafts[n-1]] — so verified[0..=req.num_nodes()] is
        // what the sampler wants.
        let expected_rows = 1 + req.num_nodes();
        let argmax = if verified.len() >= expected_rows {
            verified[..expected_rows].to_vec()
        } else {
            // Defensive: pad with last-known to avoid length mismatch.
            let mut a = verified.to_vec();
            a.resize(expected_rows, *verified.last().unwrap_or(&0));
            a
        };
        // ── ATLAS_DDTREE_CEILING_LOG=1: branch-headroom measurement ──
        // Counterfactual: on ANY run (incl. the lossless flat-safe path),
        // compute what the FULL tree walk would accept vs the flat (top-1
        // chain) prefix. tree_depth > flat_depth means a sibling branch
        // matched the target argmax where the top-1 chain diverged — i.e.
        // real branch headroom. This is pure observation (nothing committed),
        // so downstream state stays clean and the numbers are uncontaminated
        // by the fork-commit bug. The gap is the ceiling on branching gain.
        if std::env::var("ATLAS_DDTREE_CEILING_LOG").ok().as_deref() == Some("1") {
            let full = greedy_sample_ddtree_full(&req, &argmax)
                .map(|s| s.accepted_compact_indices.len())
                .unwrap_or(0);
            // Flat (top-1 chain) depth: contiguous prefix of the full walk.
            let flat = greedy_sample_ddtree(&req, &argmax)
                .map(|s| s.accepted_compact_indices.len())
                .unwrap_or(0);
            // RETRIEVAL ceiling: the target's argmax at the FIRST divergence
            // (row `flat`, i.e. where the top-1 chain stopped). Is that token
            // present in recent generated context? If so, a retrieval / n-gram
            // sibling (Graft, arXiv 2605.20104) could supply it where the
            // diffusion drafter's marginals can't. Measures the repetitive-
            // span headroom that diffusion branching alone (branch_gain) misses.
            let div_tok = argmax.get(flat).copied();
            let retr_hit = match div_tok {
                Some(t) => {
                    let toks = &a.seq.tokens;
                    let lo = toks.len().saturating_sub(1024);
                    toks[lo..].contains(&t)
                }
                None => false,
            };
            tracing::info!(
                "DDTREE_CEILING: flat_depth={flat} tree_depth={full} branch_gain={} \
                 retr_hit={} tree_nodes={}",
                full as i64 - flat as i64,
                retr_hit as u8,
                req.tree_token_ids.len(),
            );
        }
        // ── ATLAS_DFLASH_BRANCH_AUDIT=1: cross-step conditioning audit ──
        // Pure observation (commits nothing). When the flat-safe walk stops
        // at a divergence whose fork token the FULL walk would take, this
        // step commits [spine prefix] + bonus == the fork token, so the NEXT
        // verify's root row (verified[0]) is the target's TRUE argmax after
        // the fork under exact flat conditioning. The fork ROW of THIS step
        // claims that same conditioning ([ctx, spine prefix, fork]) — its
        // argmax must therefore equal next step's verified[0]. Mismatches
        // measure exactly the branch-row mis-conditioning that corrupts the
        // deep tree-commit; ~100% match certifies ancestor-exact attention.
        // Stash keyed by session hash (debug-only; batch=1 serve config).
        let audit_enabled = std::env::var("ATLAS_DFLASH_BRANCH_AUDIT").ok().as_deref() == Some("1");
        if audit_enabled
            && !tree_commit_active
            && let Ok(full) = greedy_sample_ddtree_full(&req, &argmax)
            && let Ok(flat) = greedy_sample_ddtree(&req, &argmax)
        {
            let flat_n = flat.accepted_compact_indices.len();
            let fullp = &full.accepted_compact_indices;
            if fullp.len() > flat_n {
                let fork_c = fullp[flat_n];
                let fork_tok = req.tree_token_ids.get(fork_c.saturating_sub(1)).copied();
                let expected = argmax.get(fork_c).copied();
                if let (Some(ft), Some(exp)) = (fork_tok, expected) {
                    branch_audit_stash()
                        .lock()
                        .unwrap()
                        .insert(a.session_hash, (exp, ft));
                }
            }
        }
        // FIX 1: full-path sampler when tree-commit is enabled AND the verify
        // was ancestor-exact (see the guard above), else the flat-safe
        // sampler (spine rows only — always correctly conditioned).
        let sample_res = if tree_commit_active {
            greedy_sample_ddtree_full(&req, &argmax)
        } else {
            greedy_sample_ddtree(&req, &argmax)
        };
        match sample_res {
            Ok(sample) => {
                let n = sample.accepted_compact_indices.len();
                // Derive the kernel-frame slot from the actual compact
                // indices, NOT from `n - 1`. In chain-only mode these are
                // equal; for a fork-crossing path they diverge
                // (e.g. [1, 2, 7] → slot 7, not 2).
                let slot = last_accepted_inter_slot(&sample.accepted_compact_indices);
                // Record the sparse path for the emit + KV-compaction below.
                // The bonus row is the compact tip the walk ended at
                // (== last accepted compact index, or 0 if nothing accepted).
                if tree_commit_active {
                    let bonus_row = sample.accepted_compact_indices.last().copied().unwrap_or(0);
                    tree_accepted_path = Some((sample.accepted_compact_indices.clone(), bonus_row));
                }
                (n, Some(slot))
            }
            Err(e) => {
                tracing::warn!("greedy_sample_ddtree failed ({e}); falling back to linear");
                // ATLAS_DDTREE_TREE_TOKENS_VERIFY=1: compare against the
                // tokens the verifier actually saw at each slot
                // (verify_input_tokens, which mixes tree_token_ids for
                // the first tree_len slots + drafts for the chain-padded
                // tail). When use_tree_tokens is false, verify_input_tokens
                // == drafts so this matches legacy chain-arithmetic.
                let mut n = 0usize;
                for i in 0..verify_input_tokens.len() {
                    if i + 1 >= verified.len() {
                        break;
                    }
                    if verify_input_tokens[i] == verified[i] {
                        n += 1;
                    } else {
                        break;
                    }
                }
                // Fallback path is chain-arithmetic but the GDN kernel
                // that just ran was still tree-aware — slot 0 if nothing
                // accepted, else `n` (compact-index of last linear-prefix
                // node == n under chain-equivalent payload).
                let slot = if n == 0 { 0 } else { n };
                (n, Some(slot))
            }
        }
    } else {
        // Relaxed acceptance (ATLAS_DFLASH_RELAX_ACCEPT=1, default off,
        // works at temp0 AND temp>0): accept a near-miss DRAFT token (and
        // COMMIT IT, not the argmax) when it is a high-probability token
        // under the TARGET — within the target's top-k OR with
        // p(draft)/p(argmax) >= ratio at the would-be-mismatch position.
        // Quality is preserved because we only ever commit a token the
        // target itself ranks highly; the PPL guardrail bounds the drift.
        // `None` ⇒ inactive ⇒ fall through to typical-accept, then to the
        // legacy exact-match prefix.
        //
        // Tried BEFORE typical-accept: relaxed is the more general gate
        // (subsumes typical's α·p_max test as a special case) and is the
        // one that also fires at temp0.
        let n = match dflash_relax_accept(model, a, drafts, &verified) {
            Some(n) => n,
            None => match dflash_typical_accept(model, a, drafts, &verified) {
                Some(n) => n,
                None => {
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
            },
        };
        (n, None)
    };

    // Roll back the over-extended `seq_len` and `seq.tokens`. The verify
    // advanced both by `tokens.len() = γ+1` (all γ drafts + the prefix
    // bonus slot). We keep the original prefix + `num_accepted` drafts +
    // 1 bonus position. So the post-rollback target is
    // `pre_verify_len + num_accepted + 1` — note we do NOT push the bonus
    // again via emit_token's path (emit_token only updates the user-facing
    // output buffer, not seq.tokens), so the bonus stays in seq.tokens
    // exactly where decode_verify put it.
    let pre_verify_len = a.seq.seq_len.saturating_sub(tokens.len());
    let target_seq_len = pre_verify_len + num_accepted + 1;
    let to_drop = a.seq.seq_len.saturating_sub(target_seq_len);
    if to_drop > 0 {
        a.seq.seq_len = target_seq_len;
        let pop_n = to_drop.min(a.seq.tokens.len());
        for _ in 0..pop_n {
            a.seq.tokens.pop();
        }
    }

    // ── ATLAS_DFLASH_EARLY_EXIT_PROFILE=1: per-step accept counter ──
    // Logs the accept count for THIS K=γ verify step so the early-exit
    // sweep can read mean accept/γ straight from the server log. Cheap
    // (one log line per verify); gated so it is a no-op in production.
    if std::env::var("ATLAS_DFLASH_EARLY_EXIT_PROFILE")
        .ok()
        .as_deref()
        == Some("1")
    {
        tracing::info!(
            "DFLASH_EE_VERIFY: accepted={num_accepted}/{} drafts[..min(6)]={:?}",
            drafts.len(),
            &drafts[..drafts.len().min(6)],
        );
    }

    // Emit accepted drafts.
    //
    // ATLAS_DDTREE_TREE_TOKENS_VERIFY=1: when the verify input was built
    // from `tree_token_ids` (not `drafts`), the accepted tokens at compact
    // slots [1..=num_accepted] are `tree_token_ids[0..num_accepted]`, NOT
    // `drafts[0..num_accepted]`. greedy_sample_ddtree returns
    // accepted_compact_indices in flat-prefix-safe form ([1, 2, ..., n]),
    // so compact slot i+1 corresponds to tree_token_ids[i]. With this off
    // (default), the old `drafts[i]` emission stays — backward compatible
    // when tree_token_ids == drafts (flat-chain payloads).
    // Emit the tokens that were actually accepted. With
    // ATLAS_DDTREE_TREE_TOKENS_VERIFY=1, accepted_compact_indices = [1..=n]
    // (flat-prefix-safe contract) and the token at compact slot i+1 is
    // verify_input_tokens[i] (which = tree_token_ids[i] for i < tree_len
    // else drafts[i] for the chain-padded tail). With the env off,
    // verify_input_tokens == drafts so this is identical to the legacy
    // emission. Defensive bound: take from drafts if i exceeds vector.
    // FIX 1: when the tree-commit path is active, the accepted compact
    // indices may be NON-contiguous (a sibling fork tail). Look up each
    // accepted token by its compact index (compact slot `c` carries
    // `verify_input_tokens[c-1]`) rather than slicing the contiguous prefix.
    // Falls back to the legacy contiguous slice on every flat/non-tree path.
    if let Some(ref t) = thinking_accept {
        // ATLAS_THINK_SPEC: the thinking walk already committed/streamed
        // its accepted prefix AND bonus with plain-path in-thinking
        // semantics (suppressed EOS never reaches output_tokens,
        // `</think>` runs the phase transition, `a.last_token` already
        // points at the bonus). A missing bonus means the walk finished
        // the sequence mid-emission (stream drop / cancel / D2H failure)
        // — bail out exactly like the legacy emit loop's early return.
        if a.finished || t.bonus.is_none() {
            return;
        }
    } else {
        let emit_tokens: Vec<u32> = if let Some((ref path, _)) = tree_accepted_path {
            path.iter()
                .filter_map(|&c| verify_input_tokens.get(c.saturating_sub(1)).copied())
                .collect()
        } else {
            let emit_take = num_accepted.min(verify_input_tokens.len());
            verify_input_tokens[..emit_take].to_vec()
        };
        for &tok in &emit_tokens {
            emit_token(a, tok, None);
            if a.finished {
                return;
            }
        }

        // Bonus token = verified[num_accepted] (the one that "corrected" the draft
        // at the first mismatch, or the next-prediction past the full-accept case).
        // Grammar verify mode substitutes the MASKED argmax at that position —
        // safe because the bonus has no KV/seq.tokens entry yet (it is fed as
        // verify input position 0 next step), exactly like the MTP masked path.
        //
        // FIX 1: for a tree-fork accept the bonus is the target's greedy at the
        // path TIP (the last accepted compact row), NOT verified[num_accepted]
        // (which is a contiguous-index assumption that is false once the path
        // forked). `tree_accepted_path.1` carries that compact row.
        let bonus_idx = match tree_accepted_path {
            Some((_, bonus_row)) => bonus_row,
            None => num_accepted,
        };
        let bonus_tok = match grammar_accept {
            Some((_, b)) => Some(b),
            None => verified.get(bonus_idx).copied(),
        };
        if let Some(bonus) = bonus_tok {
            emit_token(a, bonus, None);
            if a.finished {
                return;
            }
            a.last_token = bonus;
        }
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

    tracing::info!(
        "DFLASH K=γ verify: γ={} accepted={}/{} ({:.0}%) seq_len={} verify_input={:?} verified={:?}",
        drafts.len(),
        num_accepted,
        drafts.len(),
        100.0 * (num_accepted as f64) / (drafts.len() as f64),
        a.seq.seq_len,
        // verify_input_tokens reflects what was ACTUALLY embedded — equals
        // `drafts` in chain mode, equals the tree-topology tokens when
        // ATLAS_DDTREE_TREE_TOKENS_VERIFY=1 (mixed with chain-tail padding
        // for slots beyond budget).
        verify_input_tokens,
        verified,
    );

    // SSM commit / rollback. Hybrid models (Qwen3.6-A3B has 30 GDN layers)
    // advance recurrent SSM state per-position during verify; without this
    // commit, the canonical h_state stays at position+γ even if only a few
    // drafts were accepted, producing gibberish on subsequent decodes.
    //
    // Semantics (default trait impl):
    //  - num_accepted == k_verify (full accept): canonical = h_state
    //  - 0 < num_accepted < k_verify (partial): canonical = intermediate[num_accepted-1]
    //  - num_accepted == 0: canonical untouched (rollback to checkpoint)
    //
    // k_verify = total verified positions = tokens.len() (bonus slot 0 + all
    // tree/draft nodes). For a WIDE DDTree (budget > γ) this is tree_len+1,
    // NOT drafts.len()+1 — the SSM rollback range must span every verified
    // node or the canonical state lands at the wrong intermediate slot.
    let k_verify = tokens.len();
    let total_accepted = num_accepted + 1; // bonus is always "accepted"
    // Kernel slot of the LAST accepted state in `h_state_intermediates`:
    //   - Chain mode (no tree payload): `total_accepted - 1`. Slot N is the
    //     post-N-acceptances state; matches legacy arithmetic.
    //   - Tree mode: derived above from `accepted_compact_indices.last()`.
    //     For chain-shaped trees this == `total_accepted - 1`; for branch-
    //     diverged trees it's the max compact index (kernel slot directly).
    let last_inter_slot = match tree_last_inter_slot {
        Some(slot) => slot,
        None => total_accepted.saturating_sub(1),
    };
    let t_commit = Instant::now();
    let commit_res = if tree_last_inter_slot.is_some() {
        model.commit_verify_state_async_with_slot(
            &mut a.seq,
            total_accepted,
            k_verify,
            last_inter_slot,
        )
    } else {
        model.commit_verify_state_async(&mut a.seq, total_accepted, k_verify)
    };
    let t_commit_us = t_commit.elapsed().as_micros();
    if let Err(e) = commit_res {
        tracing::error!("commit_verify_state_async (dflash): {e:#}");
        model.clear_ddtree_parent_ids();
        a.finished = true;
        return;
    }
    // M8A: now safe to clear — commit has finished reading the tree-mode flag.
    model.clear_ddtree_parent_ids();

    // FIX 2 — KV compaction for a sparse (tree-fork) accept. When the accepted
    // path crossed a sibling fork, its attention KV sits at the scattered
    // compact slots `pre_verify_len + compact_idx`; gather it down to the
    // contiguous run `pre_verify_len + 1 .. pre_verify_len + num_accepted` so
    // the next decode/propose reads correct contiguous KV (mirrors the flat
    // chain layout). No-op when the path is already contiguous (every
    // flat-chain accept) or tree-commit is off. LOSSLESS: relocates committed
    // K/V bytes only. Must run AFTER the SSM commit (which reads its own
    // intermediate pool, untouched by KV) and BEFORE the re-propose decode.
    // Guard on NON-contiguity explicitly (mirrors set_dflash_accepted_compact
    // below): a contiguous accepted path `[1,2,..,n]` needs no compaction —
    // its KV already sits at the contiguous run, exactly like the flat path,
    // which never calls compact_verify_kv at all. Only a fork-crossing
    // (sparse) path requires the gather. Calling the gather on a contiguous
    // path must be a true identity; making the skip explicit here removes any
    // dependence on the kernel's internal contiguity detection being exact.
    if let Some((ref path, _)) = tree_accepted_path
        && !path.is_empty()
        && !path.iter().enumerate().all(|(i, &c)| c == i + 1)
    {
        if let Err(e) = model.compact_verify_kv(&a.seq, path, pre_verify_len) {
            tracing::error!("compact_verify_kv (dflash tree-fork): {e:#}");
            a.finished = true;
            return;
        }
    }

    // Save the bonus token's hidden state for the NEXT propose() call.
    // DFlash needs the target's hidden states for the full prefix including
    // the bonus token; the verify forward pass only processed the drafts.
    let t_save = Instant::now();
    // `a.last_token` already holds the emitted bonus (masked argmax under
    // grammar verify mode, else `verified[bonus_idx]`; unchanged when no
    // bonus row existed).
    let bonus = a.last_token;
    if let Err(e) = model.save_hidden_for_dflash(bonus, &mut a.seq, 0) {
        tracing::error!("save_hidden_for_dflash (dflash): {e:#}");
    }
    let t_save_us = t_save.elapsed().as_micros();

    let t_trim = Instant::now();
    if let Err(e) = model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
        tracing::error!("trim_proposer_state: {e:#}");
    }
    // FIX 1: stamp the sparse accepted path onto the proposer (AFTER
    // trim/after_verify, which clears it) so the next propose's ctx-hidden
    // append reads the scattered fork-capture rows, not the contiguous
    // prefix. Only when the path actually forked (non-contiguous): a
    // contiguous `[1..N]` path needs no override and stays on the fast path.
    if let Some((ref path, _)) = tree_accepted_path {
        let is_contiguous = path.iter().enumerate().all(|(i, &c)| c == i + 1);
        if !is_contiguous {
            model.set_dflash_accepted_compact(&mut a.seq, path);
        }
    }
    let t_trim_us = t_trim.elapsed().as_micros();

    // ATLAS_DFLASH_RECYCLE=1: stash the discarded draft tail BEFORE the
    // re-propose below, keyed by the corrected token the target just committed
    // (= a.last_token, which is also the `token` fed to the next propose). The
    // next propose re-offers `drafts[num_accepted+1..]` when its last_token
    // matches this key. No-op for the default (non-recycle) path. Skipped under
    // tree mode (drafts/verified are compact-tree indices, not a flat chain, so
    // the tail-after-mismatch arithmetic does not hold). Lossless: only changes
    // what is PROPOSED next step, never what is committed.
    if std::env::var("ATLAS_DFLASH_RECYCLE").ok().as_deref() == Some("1")
        && a.pending_tree_payload.is_none()
    {
        if let Err(e) = model.dflash_stash_recycle(&mut a.seq, drafts, num_accepted, a.last_token) {
            tracing::warn!("dflash_stash_recycle: {e:#}");
        }
    }

    // ── ATLAS_DFLASH_ECHO=1: stash the target's own rejected-tail argmaxes ──
    // `verified[num_accepted+1 ..]` (the tokens AFTER the bonus) are the
    // target's OWN next-token choices conditioned on a near-miss prefix —
    // usually still right after the one-token bonus substitution. Stash them
    // (keyed by the bonus = a.last_token) so the NEXT propose offers them as
    // a target-authored draft chain and skips the drafter forward (the
    // 25-50ms propose slice). Gated to the plain FLAT chain path only:
    //   - no tree payload this step and no tree/portfolio draft method active
    //     (`verified` rows are compact-tree indices there, and the accept
    //     walk is topology-aware — chain arithmetic does not hold);
    //   - not the think-spec accept walk (it commits its own tokens with
    //     forced-injection semantics; its bonus may not be verified[n]);
    //   - not the grammar-masked accept (its bonus is a MASKED argmax, and
    //     grammar-active sequences drop pending drafts in step_mtp anyway).
    // The min-accept floor / min-tail gates live in dflash_stash_echo.
    // LOSSLESS: only changes what is PROPOSED next step, never what is
    // committed — same oracle contract as recycle above.
    if spark_model::layers::dflash_head::echo::EchoConfig::enabled()
        && a.pending_tree_payload.is_none()
        && !dflash_tree_method_active()
        && !dflash_portfolio_active()
        && thinking_accept.is_none()
        && grammar_accept.is_none()
    {
        if let Err(e) = model.dflash_stash_echo(&mut a.seq, &verified, num_accepted, a.last_token) {
            tracing::warn!("dflash_stash_echo: {e:#}");
        }
    }

    // Re-propose for next step — unless the stage-1 grammar gate fires
    // (grammar now constrains output, e.g. this verify emitted the token
    // that opened a tool-call body): leave `pending_drafts` empty so the
    // next step runs the grammar-enforced bootstrap decode.
    let skip_propose = dflash_grammar_skip_propose(model, a);
    let _mtp_grammar_mask = if skip_propose {
        None
    } else {
        mtp_grammar_mask_for(a)
    };
    let t_propose = std::time::Instant::now();
    let propose_result = if skip_propose {
        Ok(Vec::new())
    } else {
        model.run_mtp_propose_multi(
            a.last_token,
            a.seq.seq_len,
            num_drafts,
            &mut a.seq,
            0,
            _mtp_grammar_mask.as_deref(),
        )
    };
    let propose_us = t_propose.elapsed().as_micros();
    if std::env::var("ATLAS_DFLASH_PROPOSE_LOG").ok().as_deref() == Some("1") {
        tracing::info!(
            "DFlash propose: {}μs num_accepted={} seq_len={}",
            propose_us,
            num_accepted,
            a.seq.seq_len
        );
    }
    match propose_result {
        Ok(d) if !d.is_empty() => {
            let mut drafts = d;
            // ── ATLAS_DFLASH_CFG_JF=1: CFG jump-forward splice (default off) ──
            //
            // Splice structurally-forced tokens (closing brackets/quotes) into
            // the freshly-proposed draft chain at positions where the neural
            // drafter disagreed with the single legal next token. LOSSLESS: the
            // verify above is a greedy oracle, so a wrong splice is rejected
            // exactly like a wrong drafter token. Skipped when a
            // NON-terminated grammar is active (xgrammar already forces those
            // positions; we must not fight it) and when the tree payload is set
            // (drafts are compact tree indices, not a flat chain). No-op unless
            // the classification table was built at startup.
            cfg_jf_splice_drafts(a, &mut drafts);
            a.pending_drafts = drafts;
        }
        Ok(_) => {}
        Err(e) => tracing::error!("run_mtp_propose_multi (dflash): {e:#}"),
    }

    if step_timing {
        let total_us = t_step.elapsed().as_micros();
        let other_us = total_us.saturating_sub(
            t_sync_secondary_us + t_verify_us + t_commit_us + t_save_us + t_trim_us + propose_us,
        );
        tracing::info!(
            "DFLASH step timing: total={total_us}μs sync_secondary={t_sync_secondary_us}μs \
             verify={t_verify_us}μs commit={t_commit_us}μs save_hidden={t_save_us}μs \
             trim={t_trim_us}μs propose={propose_us}μs other={other_us}μs accepted={num_accepted}",
        );
    }

    // DDTree M6: drain any tree payload the drafter built during the propose
    // above and stash it on ActiveSeq for the next-step verifier. Default
    // proposers return None (flat path preserved).
    a.pending_tree_payload = model.take_pending_tree_payload(&mut a.seq);
}

/// Whether any DFlash TREE draft method is active (BRANCH / CATERPILLAR /
/// DDTree). CFG jump-forward operates on a FLAT draft chain only; when a tree
/// method is on, the proposer emits compact tree-index tokens whose positions
/// don't correspond to a left-to-right token stream, so splicing would be
/// meaningless. Read once. CFG_JF is a distinct opt-in flag, so the operator is
/// not expected to combine them — this is a defensive guard.
pub(super) fn dflash_tree_method_active() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_BRANCH").ok().as_deref() == Some("1")
            || std::env::var("ATLAS_DFLASH_CATERPILLAR").ok().as_deref() == Some("1")
            || std::env::var("ATLAS_DFLASH_METHOD").ok().as_deref() == Some("ddtree")
            || std::env::var("ATLAS_DFLASH_FREE_SLOTS")
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
                .is_some_and(|n| n >= 1)
    })
}

/// Whether the 2-root PORTFOLIO forest verify is enabled
/// (ATLAS_DFLASH_PORTFOLIO=1). Not covered by `dflash_tree_method_active`
/// (portfolio emits a flat chain on steps where the retrieval sibling
/// doesn't fire), but echo-drafting is excluded whenever the mode is on:
/// its verify rows may belong to a forest topology. Read once.
pub(super) fn dflash_portfolio_active() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_DFLASH_PORTFOLIO").ok().as_deref() == Some("1"))
}

/// Apply the CFG jump-forward splice to a freshly-proposed flat draft chain
/// in place. No-op unless `ATLAS_DFLASH_CFG_JF=1`, the startup classification
/// table is present, no tree method is active, and the sequence has no active
/// (non-terminated) grammar. Emits a one-shot stats log on the first splice.
///
/// LOSSLESS: only mutates the PROPOSED tokens; the verify path commits solely
/// the target's greedy token, so a wrong splice is rejected for free and output
/// is byte-identical to the flag-off path when no splice ever helps.
fn cfg_jf_splice_drafts(a: &ActiveSeq, drafts: &mut [u32]) {
    use super::cfg_jump_forward as jf;

    if !jf::cfg_jf_enabled() || drafts.is_empty() || dflash_tree_method_active() {
        return;
    }
    // Do not fight an active grammar (xgrammar already forces those slots).
    if a.grammar_state
        .as_ref()
        .is_some_and(|gs| !gs.is_terminated())
    {
        return;
    }
    let (Some(table), Some(forced)) = (jf::delim_table(), jf::forced_ids()) else {
        return; // table not built (flag was off at startup) → inert
    };

    let stats = jf::splice_forced(drafts, &a.seq.tokens, &table, &forced);

    if stats.splices > 0 {
        static JF_DBG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = JF_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 16 {
            tracing::info!(
                "DFlash CFG_JF #{n}: spliced {} forced token(s) into draft chain \
                 (first_pos={}, seq_len={})",
                stats.splices,
                stats.first_pos,
                a.seq.seq_len,
            );
        }
    }
}

/// Grammar-masked re-derivation of the DFlash accept prefix + bonus
/// (stage 2, `ATLAS_DFLASH_GRAMMAR_MODE=verify`).
///
/// Row `i` of the model's verify logits buffer (`[K, vocab]` BF16, row
/// stride `vocab_size`) is the target's prediction for the slot
/// `drafts[i]` occupies; row `drafts.len()` is the bonus slot. Walk the
/// positions in order: fill the matcher's bitmask for the CURRENT grammar
/// state, take the MASKED argmax as the target token, and stop acceptance
/// at the first draft that diverges from it or that the matcher rejects.
/// The matcher is advanced per accepted draft (so each position's mask
/// reflects its prefix) and rolled back before returning — `emit_token`
/// re-advances it for real on emission, mirroring
/// `truncate_drafts_at_grammar_boundary`'s transient-advance pattern.
///
/// Mask-application convention mirrors the MTP masked-draft path in
/// `spark-model/src/layers/mtp_head/forward.rs`: bit `tok` set in the i32
/// bitmask ⇒ token allowed; BF16 logits are ordered by their raw bit
/// pattern reinterpreted as i16 (a total order over finite values).
///
/// Returns `(num_accepted, bonus_token)`, or `None` when masking does not
/// apply (thinking span, no/terminated grammar, fp32 logits) or the
/// logits D2H fails — the caller falls back to the unmasked accept path.
fn dflash_masked_accept(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
) -> Option<(usize, u32)> {
    if a.inside_thinking || verified.is_empty() {
        return None;
    }
    if a.grammar_state.as_ref().is_none_or(|gs| gs.is_terminated()) {
        return None;
    }
    let logits_base = model.logits_buffer_ptr();
    if model.logits_ptr_is_fp32(logits_base) {
        return None; // masked argmax below assumes BF16 rows
    }
    let vocab = model.vocab_size();
    let mut row_buf = vec![0u8; vocab * 2];

    let gs = a.grammar_state.as_mut().expect("checked above");
    let mut accepted = 0usize;
    let mut bonus: Option<u32> = None;
    for i in 0..verified.len() {
        let target_tok = if gs.fill_bitmask() {
            if let Err(e) =
                model.copy_logits_to_host(logits_base.offset(i * vocab * 2), &mut row_buf)
            {
                tracing::warn!(
                    "DFlash grammar verify: logits D2H failed ({e:#}); unmasked fallback"
                );
                if accepted > 0 {
                    gs.rollback(accepted);
                }
                return None;
            }
            match masked_argmax_bf16(&row_buf, gs.bitmask_data(), vocab) {
                Some(t) => t,
                None => {
                    // Degenerate empty allowed set (dead grammar state):
                    // keep the unmasked argmax — emit_token tolerates the
                    // failed accept exactly as the pre-fix path did.
                    tracing::warn!(
                        "DFlash grammar verify: mask allowed zero tokens at verify pos {i}"
                    );
                    verified[i]
                }
            }
        } else {
            // No constraint at this position (e.g. grammar just terminated
            // mid-walk after a stop token was accepted).
            verified[i]
        };
        // Accept drafts[i] only while a bonus row remains past it (mirrors
        // the unmasked loop's `i + 1 >= verified.len()` guard); otherwise
        // this row's target token becomes the bonus.
        if i < drafts.len()
            && i + 1 < verified.len()
            && drafts[i] == target_tok
            && gs.accept_token(drafts[i])
        {
            accepted += 1;
            continue;
        }
        bonus = Some(target_tok);
        break;
    }
    if accepted > 0 {
        gs.rollback(accepted);
    }
    Some((accepted, bonus?))
}

/// Typical acceptance for the flat-chain DFlash accept prefix
/// (`ATLAS_DFLASH_TYPICAL_ACCEPT=<epsilon>`, opt-in).
///
/// At temperature > 0 the exact-match rule (`drafts[i] == verified[i]`,
/// where `verified` is the target's UNMASKED GPU argmax) is needlessly
/// strict: a draft the target itself would plausibly sample gets rejected
/// just for not being the argmax, collapsing acceptance on creative/story
/// prompts. Instead, accept `drafts[i]` when
///
///   p_target(drafts[i]) >= max(epsilon, alpha * p_max)
///
/// where `p_target` is the temperature-scaled softmax of the target's
/// verify logits at position `i` (row `i` of the `[K, vocab]` BF16 logits
/// buffer, still resident on device — same lazy per-row D2H pattern as
/// `dflash_masked_accept`), `p_max` its max, `alpha` from
/// `ATLAS_DFLASH_TYPICAL_ALPHA` (default 0.3). Exact argmax matches are
/// accepted without the test (and without the D2H) — the typical rule
/// only ever WIDENS acceptance, so greedy-equivalent behavior is the
/// floor. The first position failing the test ends the prefix; the bonus
/// stays `verified[num_accepted]` (the existing path — note this verify
/// path emits the target's argmax as the bonus even at temp > 0).
///
/// Returns `Some(num_accepted)` when the rule is active for this request,
/// `None` to fall back to exact-match: env absent, temperature == 0
/// (greedy completely unaffected), an active (non-terminated) grammar —
/// loosened acceptance must not bypass constraint enforcement — or fp32
/// logits (row layout below assumes BF16). A failed row D2H ends the
/// prefix at that position, which equals the exact-match outcome there.
fn dflash_typical_accept(
    model: &dyn Model,
    a: &ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
) -> Option<usize> {
    let epsilon = dflash_typical_epsilon()?;
    if a.temperature <= 0.0 {
        return None;
    }
    if a.grammar_state
        .as_ref()
        .is_some_and(|gs| !gs.is_terminated())
    {
        return None;
    }
    let logits_base = model.logits_buffer_ptr();
    if model.logits_ptr_is_fp32(logits_base) {
        return None; // softmax below assumes BF16 rows
    }
    let vocab = model.vocab_size();
    let alpha = dflash_typical_alpha();
    // Lazily allocated on the first non-argmax position; pure argmax
    // prefixes never touch the device.
    let mut row_buf: Vec<u8> = Vec::new();
    let mut accepted = 0usize;
    let mut typical_hits = 0usize;
    for i in 0..drafts.len() {
        if i + 1 >= verified.len() {
            break;
        }
        if drafts[i] == verified[i] {
            accepted += 1;
            continue;
        }
        if drafts[i] as usize >= vocab {
            break;
        }
        if row_buf.is_empty() {
            row_buf = vec![0u8; vocab * 2];
        }
        if let Err(e) = model.copy_logits_to_host(logits_base.offset(i * vocab * 2), &mut row_buf) {
            tracing::warn!("DFlash typical accept: logits D2H failed ({e:#}); stopping prefix");
            break;
        }
        let (p_draft, p_max) = typical_row_probs_bf16(&row_buf, vocab, a.temperature, drafts[i]);
        if p_draft >= epsilon.max(alpha * p_max) {
            accepted += 1;
            typical_hits += 1;
        } else {
            break;
        }
    }
    if typical_hits > 0 {
        tracing::debug!(
            "DFlash typical accept: +{typical_hits} non-argmax draft(s) accepted \
             (total {accepted}/{}, ε={epsilon}, α={alpha}, T={})",
            drafts.len(),
            a.temperature,
        );
    }
    Some(accepted)
}

/// Temperature-scaled softmax probabilities over one host-copied BF16
/// logits row: `(p(tok), p_max)`. Two passes, f32 accumulation; numerically
/// stable via max-logit subtraction (so `p_max == 1/Z` exactly).
fn typical_row_probs_bf16(bytes: &[u8], vocab: usize, temperature: f32, tok: u32) -> (f32, f32) {
    let inv_t = 1.0 / temperature.max(1e-6);
    let mut max_logit = f32::NEG_INFINITY;
    for i in 0..vocab {
        let l = bf16_to_f32(bytes[2 * i], bytes[2 * i + 1]);
        if l > max_logit {
            max_logit = l;
        }
    }
    let mut z = 0f32;
    let mut e_tok = 0f32;
    for i in 0..vocab {
        let l = bf16_to_f32(bytes[2 * i], bytes[2 * i + 1]);
        let e = ((l - max_logit) * inv_t).exp();
        z += e;
        if i == tok as usize {
            e_tok = e;
        }
    }
    if z <= 0.0 || !z.is_finite() {
        return (0.0, 0.0);
    }
    (e_tok / z, 1.0 / z)
}

/// `ATLAS_DFLASH_TYPICAL_ACCEPT` epsilon, read once at first use. `None`
/// when the env is absent (feature off); a set-but-unparseable value
/// falls back to 0.05 with a warning.
fn dflash_typical_epsilon() -> Option<f32> {
    static EPS: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *EPS.get_or_init(|| {
        let raw = std::env::var("ATLAS_DFLASH_TYPICAL_ACCEPT").ok()?;
        match raw.trim().parse::<f32>() {
            Ok(v) if v.is_finite() && (0.0..=1.0).contains(&v) => Some(v),
            _ => {
                tracing::warn!(
                    "ATLAS_DFLASH_TYPICAL_ACCEPT={raw:?} unparseable (want float in [0,1]); \
                     defaulting to 0.05"
                );
                Some(0.05)
            }
        }
    })
}

/// `ATLAS_DFLASH_TYPICAL_ALPHA` (fraction of `p_max` a draft must reach),
/// read once at first use. Default 0.3.
fn dflash_typical_alpha() -> f32 {
    static ALPHA: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *ALPHA.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_TYPICAL_ALPHA")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
            .unwrap_or(0.3)
    })
}

/// Parsed `ATLAS_DFLASH_RELAX_*` configuration (read once).
#[derive(Clone, Copy)]
struct RelaxConfig {
    /// Accept a draft if it is within the target's top-`k` logits. 0 = off.
    topk: usize,
    /// Accept a draft if `p_target(draft)/p_target(argmax) >= ratio`.
    /// `<= 0.0` = off. The ratio is logit-based so it is temperature-free:
    /// `p(d)/p(amax) = exp(l_d - l_amax)` (the partition function cancels),
    /// which is exactly the relaxation we want at temp0 (no softmax temp).
    ratio: f32,
}

/// `ATLAS_DFLASH_RELAX_ACCEPT` gate config. Returns `None` (feature off)
/// unless `ATLAS_DFLASH_RELAX_ACCEPT=1` AND at least one of
/// `ATLAS_DFLASH_RELAX_TOPK` (int >= 1) / `ATLAS_DFLASH_RELAX_RATIO`
/// (float in (0,1]) is set. Read once at first use.
fn dflash_relax_config() -> Option<RelaxConfig> {
    static CFG: std::sync::OnceLock<Option<RelaxConfig>> = std::sync::OnceLock::new();
    *CFG.get_or_init(|| {
        let on = std::env::var("ATLAS_DFLASH_RELAX_ACCEPT").ok().as_deref() == Some("1");
        if !on {
            return None;
        }
        let topk = std::env::var("ATLAS_DFLASH_RELAX_TOPK")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&k| k >= 1)
            .unwrap_or(0);
        let ratio = std::env::var("ATLAS_DFLASH_RELAX_RATIO")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0 && *v <= 1.0)
            .unwrap_or(0.0);
        if topk == 0 && ratio <= 0.0 {
            tracing::warn!(
                "ATLAS_DFLASH_RELAX_ACCEPT=1 but neither RELAX_TOPK (>=1) nor \
                 RELAX_RATIO (0<r<=1) is set — relaxed accept inactive (falling \
                 back to exact / typical accept)"
            );
            return None;
        }
        tracing::info!(
            "ATLAS_DFLASH_RELAX_ACCEPT active: topk={topk} ratio={ratio} \
             (commits high-prob target near-miss drafts; PPL-bounded, default-off)"
        );
        Some(RelaxConfig { topk, ratio })
    })
}

/// PPL-bounded relaxed acceptance for the flat-chain DFlash accept prefix
/// (`ATLAS_DFLASH_RELAX_ACCEPT=1`, default off; works at temp0 and temp>0).
///
/// At each would-be-mismatch position the strict greedy rule rejects the
/// draft purely because it is not the target's argmax — even when the draft
/// is the target's 2nd/3rd choice with near-argmax probability. On novel
/// code the drafter's near-misses are usually exactly such high-probability
/// alternates, so strict greedy collapses acceptance to ~3/16.
///
/// Relaxed accept commits the DRAFT token (NOT the argmax) at a mismatch
/// when the draft is a high-probability token under the TARGET:
///
///   - draft is within the target's top-`topk` logits at this row, OR
///   - `p_target(draft) / p_target(argmax) >= ratio`, i.e.
///     `l_draft - l_argmax >= ln(ratio)` (logit-space, temperature-free).
///
/// Quality is preserved because every committed token is one the target
/// itself ranks at/near the top — the PPL guardrail bounds the drift. The
/// committed draft's KV + SSM intermediate already sit at compact slot `i`
/// (the drafter token was embedded there in `decode_verify`), so the
/// downstream chain commit (`h_state_intermediates[num_accepted-1]`) stays
/// consistent — identical plumbing to `dflash_typical_accept`.
///
/// The bonus stays `verified[num_accepted]` (the target argmax at the first
/// genuinely-rejected position). Exact argmax matches accept without any
/// D2H (the rule only ever WIDENS acceptance vs strict greedy).
///
/// Returns `Some(num_accepted)` when active, `None` to defer: env off, an
/// active grammar (relaxed accept must not bypass constraint enforcement),
/// or fp32 logits (row layout assumes BF16). A failed row D2H ends the
/// prefix at that position (== strict outcome there).
fn dflash_relax_accept(
    model: &dyn Model,
    a: &ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
) -> Option<usize> {
    let cfg = dflash_relax_config()?;
    if a.grammar_state
        .as_ref()
        .is_some_and(|gs| !gs.is_terminated())
    {
        return None;
    }
    let logits_base = model.logits_buffer_ptr();
    if model.logits_ptr_is_fp32(logits_base) {
        return None; // row scan below assumes BF16
    }
    let vocab = model.vocab_size();
    let ln_ratio = if cfg.ratio > 0.0 {
        cfg.ratio.ln()
    } else {
        f32::NEG_INFINITY
    };
    // Lazily allocated on the first non-argmax position; pure argmax
    // prefixes never touch the device.
    let mut row_buf: Vec<u8> = Vec::new();
    let mut accepted = 0usize;
    let mut relax_hits = 0usize;
    for i in 0..drafts.len() {
        if i + 1 >= verified.len() {
            break;
        }
        if drafts[i] == verified[i] {
            accepted += 1;
            continue;
        }
        if drafts[i] as usize >= vocab {
            break;
        }
        if row_buf.is_empty() {
            row_buf = vec![0u8; vocab * 2];
        }
        if let Err(e) = model.copy_logits_to_host(logits_base.offset(i * vocab * 2), &mut row_buf) {
            tracing::warn!("DFlash relax accept: logits D2H failed ({e:#}); stopping prefix");
            break;
        }
        if relax_row_accepts(&row_buf, vocab, drafts[i], cfg.topk, ln_ratio) {
            accepted += 1;
            relax_hits += 1;
        } else {
            break;
        }
    }
    if relax_hits > 0 {
        tracing::debug!(
            "DFlash relax accept: +{relax_hits} non-argmax draft(s) committed \
             (total {accepted}/{}, topk={}, ratio={}, T={})",
            drafts.len(),
            cfg.topk,
            cfg.ratio,
            a.temperature,
        );
    }
    Some(accepted)
}

/// Relaxed-accept test for one host-copied BF16 logits row. Accepts `tok`
/// when it is within the top-`topk` logits OR its logit gap to the argmax
/// satisfies `l_tok - l_max >= ln_ratio` (== `p(tok)/p(max) >= ratio`,
/// temperature-free). BF16 values compare correctly as i16 for finite
/// magnitudes (same ordering trick as `masked_argmax_bf16`); the exact
/// logit gap is computed in f32 for the ratio test.
fn relax_row_accepts(bytes: &[u8], vocab: usize, tok: u32, topk: usize, ln_ratio: f32) -> bool {
    let ti = tok as usize;
    if ti >= vocab {
        return false;
    }
    let l_tok = bf16_to_f32(bytes[2 * ti], bytes[2 * ti + 1]);
    // Single pass: find max logit and count how many logits strictly exceed
    // l_tok (the draft's rank-1 position is `n_greater`). top-k holds when
    // n_greater < topk (i.e. at most topk-1 tokens beat the draft).
    let mut max_logit = f32::NEG_INFINITY;
    let mut n_greater = 0usize;
    for i in 0..vocab {
        let l = bf16_to_f32(bytes[2 * i], bytes[2 * i + 1]);
        if l > max_logit {
            max_logit = l;
        }
        if l > l_tok {
            n_greater += 1;
            // Early exit only safe if top-k is the only criterion; with the
            // ratio test we still need the true max, so don't break here.
        }
    }
    if topk >= 1 && n_greater < topk {
        return true;
    }
    if ln_ratio.is_finite() && (l_tok - max_logit) >= ln_ratio {
        return true;
    }
    false
}

/// Masked argmax over one host-copied BF16 logits row. `None` when the
/// bitmask allows zero tokens. Same BF16-as-i16 ordering trick as the MTP
/// masked path (`mtp_head/forward.rs`) — valid for all finite values.
fn masked_argmax_bf16(bytes: &[u8], bitmask: &[i32], vocab: usize) -> Option<u32> {
    let mut best_tok: Option<u32> = None;
    let mut best_val = i16::MIN;
    for tok in 0..vocab {
        let word = tok / 32;
        let bit = tok % 32;
        if word >= bitmask.len() || (bitmask[word] & (1i32 << bit)) == 0 {
            continue;
        }
        let signed = u16::from_le_bytes([bytes[2 * tok], bytes[2 * tok + 1]]) as i16;
        if best_tok.is_none() || signed > best_val {
            best_val = signed;
            best_tok = Some(tok as u32);
        }
    }
    best_tok
}
