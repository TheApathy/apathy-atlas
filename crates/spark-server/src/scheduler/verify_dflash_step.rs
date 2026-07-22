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
pub fn step_verify_dflash(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        a.finished = true;
        return;
    }

    // tokens = [last_verified, draft_0, draft_1, ..., draft_{γ-1}]
    let mut tokens = Vec::with_capacity(drafts.len() + 1);
    tokens.push(a.last_token);
    tokens.extend_from_slice(drafts);

    // STEP-TIMING (ATLAS_DFLASH_STEP_TIMING=1): split the ~0.88s/step into
    // verify (target M=1+γ forward) vs propose (drafter forward, tail below).
    // The ledger never had this split — it guessed "FFN + double sweep". This
    // measures it. Gated so the hot path pays nothing when the env is unset.
    let step_timing = std::env::var("ATLAS_DFLASH_STEP_TIMING").ok().as_deref() == Some("1");
    // Block-fork tree (doc 16): hand the fork payload to the verify (which
    // takes it and, when eligible, verifies chain B as rows k..2k). Keep a
    // host copy for the walk below.
    // S3a gate (ATLAS_DFLASH_FORK_DEGEN=1): force the fork token to the
    // draft AT the cliff so B ≡ A AND the walk's b_win condition
    // (`verified[cliff] == fork_tok` with `num_accepted == cliff`) is
    // UNREACHABLE — if verified[cliff] == drafts[cliff] then A accepted the
    // cliff, so num_accepted > cliff. Verify and walk stay consistent, so
    // the whole tree path must be byte-identical to flat. This is the true
    // transparency test for the 2K-row verify + scratch-KV machinery.
    let fork_info = a.pending_block_fork.take().map(|(c, ft)| {
        if std::env::var("ATLAS_DFLASH_FORK_DEGEN").ok().as_deref() == Some("1") {
            (c, drafts.get(c).copied().unwrap_or(ft))
        } else {
            (c, ft)
        }
    });
    a.seq.block_fork = fork_info;
    let t_verify = std::time::Instant::now();
    let verified_argmax = match model.decode_verify_dflash(&tokens, &mut a.seq, 0) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("decode_verify_dflash: {e:#}");
            a.finished = true;
            return;
        }
    };
    let verify_ms = if step_timing {
        t_verify.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };
    a.last_token_time = Instant::now();

    // DFlash drafter proposes on raw argmax; when dflash_verify_raw_argmax is set
    // (process-wide DFlash mode), skip the rep_pen/DRY pipeline so verifier and
    // drafter judge on the SAME (GOLD) basis. For non-DFlash callers (unreachable
    // today since step_verify_dflash is only dispatched at drafts.len()>=4 which
    // only DFlash produces), apply the full pre-sample pipeline as in K=2/3/4.
    let verified = if dflash_verify_raw_argmax
        && !crate::scheduler::verify_pipeline_helper::dflash_masked_verify_enabled()
    {
        verified_argmax
    } else {
        crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            model,
            &verified_argmax,
            a,
            verify_ctx,
        )
    };

    // Block-fork tree (doc 16): a tree verify returns A ++ B chains
    // (2·(γ+1) entries). Split; the A-walk below is unchanged, the B branch
    // extends it at the fork.
    let ka = tokens.len();
    let tree_mode = verified.len() == 2 * ka;
    let (verified, verified_b): (Vec<u32>, Option<Vec<u32>>) = if tree_mode {
        (verified[..ka].to_vec(), Some(verified[ka..].to_vec()))
    } else {
        (verified, None)
    };

    // `decode_verify` already advanced `seq.seq_len` by `tokens.len()` and
    // pushed all γ+1 tokens into `seq.tokens`. The accept-prefix logic below
    // determines how many to keep — the rest must be rolled back so the
    // KV cache, SSM state, and emitted token sequence stay consistent.

    // Accept-prefix: drafts[i] is "accepted" iff drafts[i] == verified[i].
    // verified[i] is the target's argmax at position i (i.e. its
    // prediction for what should follow `tokens[i]`). drafts[i] was the
    // proposer's guess for the same slot. First mismatch terminates the
    // accepted prefix; verified[first_mismatch] becomes the bonus token.
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

    // Block-fork walk: A died exactly at the hedged cliff AND the target's
    // correction IS the fork token → continue on chain B (its rows verified
    // under the forked KV). b_tail = B drafts accepted past the fork;
    // b_bonus = B's own correction after that.
    let mut b_win = false;
    let mut b_tail_len = 0usize;
    let mut b_bonus: Option<u32> = None;
    if let (Some(vb), Some((cliff, fork_tok))) = (verified_b.as_ref(), fork_info)
        && num_accepted < drafts.len()
        && num_accepted == cliff
        && verified[cliff] == fork_tok
    {
        b_win = true;
        let mut i = cliff + 1;
        while i < drafts.len() && i < vb.len() && drafts[i] == vb[i] {
            b_tail_len += 1;
            i += 1;
        }
        if i < vb.len() {
            b_bonus = Some(vb[i]);
        }
        static FORKWIN_DBG: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let n = FORKWIN_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n <= 8 || n % 64 == 0 {
            tracing::info!(
                "DFLASH_BLOCKFORK WIN #{n}: cliff={cliff} fork_tok={fork_tok}                  b_tail={b_tail_len} b_bonus={b_bonus:?} (saved a miss)"
            );
        }
    }
    let committed_extra = if b_win { 1 + b_tail_len } else { 0 };

    // Adaptive speculation (ATLAS_DFLASH_ADAPTIVE=1): feed the rolling
    // accept window; may suspend this seq's speculation (see adaptive_spec).
    crate::scheduler::adaptive_spec::record_verify(a, num_accepted + committed_extra);

    // Roll back the over-extended `seq_len` and `seq.tokens`. The verify
    // advanced both by `tokens.len() = γ+1` (all γ drafts + the prefix
    // bonus slot). We keep the original prefix + `num_accepted` drafts +
    // 1 bonus position. So the post-rollback target is
    // `pre_verify_len + num_accepted + 1` — note we do NOT push the bonus
    // again via emit_token's path (emit_token only updates the user-facing
    // output buffer, not seq.tokens), so the bonus stays in seq.tokens
    // exactly where decode_verify put it.
    let pre_verify_len = a.seq.seq_len.saturating_sub(tokens.len());
    // B-win holds extra positions: the fork token + its accepted B tail
    // (their KV lives in the adopted scratch blocks; see adopt below).
    let target_seq_len = pre_verify_len + num_accepted + 1 + committed_extra;
    let to_drop = a.seq.seq_len.saturating_sub(target_seq_len);
    if to_drop > 0 {
        a.seq.seq_len = target_seq_len;
        let pop_n = to_drop.min(a.seq.tokens.len());
        for _ in 0..pop_n {
            a.seq.tokens.pop();
        }
    }
    if b_win {
        // The pushed A-chain slot at the fork position holds the REJECTED
        // draft; patch in the fork token (the B tail beyond it is identical
        // to A's pushed drafts, so only this one entry differs).
        let fork_slot = pre_verify_len + num_accepted + 1;
        if fork_slot < a.seq.tokens.len()
            && let Some((_, ft)) = fork_info
        {
            a.seq.tokens[fork_slot] = ft;
        }
        // Adopt B's scratch KV blocks (the canonical blocks hold A's stale
        // rows at the committed fork positions).
        if let Err(e) = model.dflash_adopt_fork_blocks(&mut a.seq, true) {
            tracing::error!("dflash_adopt_fork_blocks(adopt): {e:#}");
        }
    } else if tree_mode {
        // A won (or fork missed): drop the scratch blocks.
        if let Err(e) = model.dflash_adopt_fork_blocks(&mut a.seq, false) {
            tracing::error!("dflash_adopt_fork_blocks(free): {e:#}");
        }
    }

    // EAGLE-fix (ATLAS_DFLASH_EAGLE_FIX=1): append one ctx slot per committed
    // position (rows 0..=num_accepted at N..=N+num_accepted), with the bonus
    // generator (row num_accepted) freshest. Fixes the ctx-undercount (was 1
    // slot/step regardless of num_accepted) and the EAGLE conditioning shift.
    // Sets skip_next_decode_append so the propose below does NOT re-append row 0.
    // Unified ctx commit (ATLAS_DFLASH_UNIFIED_CTX=1): ONE unconditional
    // commit at the K=gamma point — rows 0..=num_accepted at RoPE base
    // pre_verify_len. Structural replacement for dflash_eagle_kgamma_append.
    if crate::scheduler::adaptive_spec::unified_ctx_enabled() {
        if let Err(e) = model.commit_ctx(&mut a.seq, num_accepted + 1, pre_verify_len) {
            tracing::error!("commit_ctx (kgamma): {e:#}");
        }
    } else {
        let eagle_fix = std::env::var("ATLAS_DFLASH_EAGLE_FIX").ok().as_deref() == Some("1");
        // B-win: the committed stream's capture rows are B's (offset ka),
        // covering the shared prefix + fork + accepted tail + bonus generator.
        let (append_rows, append_base) = if b_win {
            (num_accepted + committed_extra, ka)
        } else {
            (num_accepted, 0)
        };
        if eagle_fix
            && let Err(e) = model.dflash_eagle_kgamma_append_at(
                &mut a.seq,
                append_rows,
                pre_verify_len,
                append_base,
            )
        {
            tracing::error!("dflash_eagle_kgamma_append: {e:#}");
        }
    }

    // Emit accepted drafts.
    for i in 0..num_accepted {
        emit_token(a, drafts[i], None);
        if a.finished {
            return;
        }
    }

    // Bonus token = verified[num_accepted] (the one that "corrected" the draft
    // at the first mismatch, or the next-prediction past the full-accept case).
    // On a fork win this IS the fork token; the accepted B tail + B's own
    // bonus follow it.
    let bonus_idx = num_accepted;
    if bonus_idx < verified.len() {
        let bonus = verified[bonus_idx];
        emit_token(a, bonus, None);
        if a.finished {
            return;
        }
        a.last_token = bonus;
    }
    if b_win {
        for i in 0..b_tail_len {
            emit_token(a, drafts[num_accepted + 1 + i], None);
            if a.finished {
                return;
            }
            a.last_token = drafts[num_accepted + 1 + i];
        }
        if let Some(bb) = b_bonus {
            emit_token(a, bb, None);
            if a.finished {
                return;
            }
            a.last_token = bb;
        }
    }

    // Accept-lift draft sources (Phase A): stash this step's salvage for the
    // next propose, keyed by the bonus (= next step's last_token). Both fns
    // early-return when their env gate is off. Flat-chain path only (this IS
    // the flat path; tree verify lands in Phase C).
    // Recycle: the drafter's discarded tail drafts[num_accepted+1..].
    if let Err(e) = model.dflash_stash_recycle(&mut a.seq, drafts, num_accepted, a.last_token) {
        tracing::warn!("dflash_stash_recycle: {e:#}");
    }
    // Echo: the target's own argmaxes verified[num_accepted+1..].
    if let Err(e) = model.dflash_stash_echo(&mut a.seq, &verified, num_accepted, a.last_token) {
        tracing::warn!("dflash_stash_echo: {e:#}");
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
        "DFLASH K=γ verify: γ={} accepted={}/{} ({:.0}%) seq_len={}",
        drafts.len(),
        num_accepted,
        drafts.len(),
        100.0 * (num_accepted as f64) / (drafts.len() as f64),
        a.seq.seq_len,
    );

    // Item #2 (STree-style in-place verify commit). h_state is canonical:
    //  - num_accepted == k_verify (full accept): no-op (h_state already correct)
    //  - 0 < num_accepted < k_verify (partial): intermediate[total_accepted-1] → h_state
    // No checkpoint write needed — the next start_checkpoint_async syncs.
    //
    // k_verify = drafts.len() + 1 (the prefix bonus position is also verified).
    let k_verify = drafts.len() + 1;
    let total_accepted = num_accepted + 1; // bonus is always "accepted"
    if let Err(e) = model.commit_accepted_prefix(&mut a.seq, total_accepted, k_verify) {
        tracing::error!("commit_accepted_prefix (dflash): {e:#}");
        a.finished = true;
        return;
    }

    // DFlash hidden is captured per-layer inside the verify graph
    // (verify_d.rs try_dflash_capture at position k-1), mirroring verify_b.rs.
    // No post-loop save needed; calling save_dflash_hidden_for_propose here
    // would overwrite the correct per-layer intermediates with a repeated
    // final-layer hidden, collapsing all 5 slots to the same value.
    let bonus_token_idx = total_accepted.saturating_sub(1);
    if let Err(e) = model.save_hidden_for_mtp(bonus_token_idx, 0) {
        tracing::error!("save_hidden_for_mtp (dflash): {e:#}");
    }

    if let Err(e) = model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
        tracing::error!("trim_proposer_state: {e:#}");
    }

    // Re-propose for next step — unless adaptive speculation just suspended
    // this seq (no drafts → the scheduler serial-decodes it via bootstrap).
    let _mtp_grammar_mask = mtp_grammar_mask_for(a);
    let t_propose = std::time::Instant::now();
    if crate::scheduler::adaptive_spec::spec_allowed(a) {
        match model.run_mtp_propose_multi(
            a.last_token,
            a.seq.seq_len,
            num_drafts,
            &mut a.seq,
            0,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(d) if !d.is_empty() => {
                a.pending_drafts = d;
                // Block-fork (doc 16): the fork payload rides with the drafts.
                a.pending_block_fork = model.dflash_take_block_fork(&mut a.seq);
            }
            Ok(_) => a.pending_block_fork = None,
            Err(e) => {
                a.pending_block_fork = None;
                tracing::error!("run_mtp_propose_multi (dflash): {e:#}");
            }
        }
    }
    if step_timing {
        let propose_ms = t_propose.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(
            "DFLASH STEP_TIMING: verify={:.1}ms propose={:.1}ms (K={}, accepted={})",
            verify_ms,
            propose_ms,
            tokens.len(),
            num_accepted,
        );
    }
}
