// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash-based verify step (drafted token verification).

use super::*;

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
pub fn step_verify_dflash(model: &dyn Model, a: &mut ActiveSeq, drafts: &[u32], num_drafts: usize) {
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
    let tree_tokens_verify =
        std::env::var("ATLAS_DDTREE_TREE_TOKENS_VERIFY").ok().as_deref() == Some("1");
    let use_tree_tokens = tree_tokens_verify
        && a.pending_tree_payload.as_ref().is_some_and(|p| !p.is_empty());

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
        for i in 0..drafts.len() {
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
    let (num_accepted, tree_last_inter_slot) = if let Some(payload) = a.pending_tree_payload.as_ref() {
        use spark_model::layers::dflash_head::ddtree::{
            greedy_sample_ddtree, last_accepted_inter_slot, DDTreeRequestRuntime,
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
        match greedy_sample_ddtree(&req, &argmax) {
            Ok(sample) => {
                let n = sample.accepted_compact_indices.len();
                // M4B-prep: derive the kernel-frame slot from the actual
                // compact indices, NOT from `n - 1`. In chain-only mode
                // these are equal; once branch-mode adapter lands they
                // may diverge (e.g. [1, 4, 7] → slot 7, not 2).
                let slot = last_accepted_inter_slot(&sample.accepted_compact_indices);
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
                    if i + 1 >= verified.len() { break; }
                    if verify_input_tokens[i] == verified[i] { n += 1; } else { break; }
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
    let emit_take = num_accepted.min(verify_input_tokens.len());
    let emit_tokens: Vec<u32> = verify_input_tokens[..emit_take].to_vec();
    for &tok in &emit_tokens {
        emit_token(a, tok, None);
        if a.finished {
            return;
        }
    }

    // Bonus token = verified[num_accepted] (the one that "corrected" the draft
    // at the first mismatch, or the next-prediction past the full-accept case).
    let bonus_idx = num_accepted;
    if bonus_idx < verified.len() {
        let bonus = verified[bonus_idx];
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
    // k_verify = drafts.len() + 1 (the prefix bonus position is also verified).
    let k_verify = drafts.len() + 1;
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

    // Save the bonus token's hidden state for the NEXT propose() call.
    // DFlash needs the target's hidden states for the full prefix including
    // the bonus token; the verify forward pass only processed the drafts.
    let t_save = Instant::now();
    let bonus = verified.get(bonus_idx).copied().unwrap_or(a.last_token);
    if let Err(e) = model.save_hidden_for_dflash(bonus, &mut a.seq, 0) {
        tracing::error!("save_hidden_for_dflash (dflash): {e:#}");
    }
    let t_save_us = t_save.elapsed().as_micros();

    let t_trim = Instant::now();
    if let Err(e) = model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
        tracing::error!("trim_proposer_state: {e:#}");
    }
    let t_trim_us = t_trim.elapsed().as_micros();

    // Re-propose for next step.
    let _mtp_grammar_mask = mtp_grammar_mask_for(a);
    let t_propose = std::time::Instant::now();
    let propose_result = model.run_mtp_propose_multi(
        a.last_token,
        a.seq.seq_len,
        num_drafts,
        &mut a.seq,
        0,
        _mtp_grammar_mask.as_deref(),
    );
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
        Ok(d) if !d.is_empty() => a.pending_drafts = d,
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
