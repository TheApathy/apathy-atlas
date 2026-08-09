// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash-based verify step (drafted token verification).

use super::*;

/// `ATLAS_DFLASH_STEP_TIMING=1` (cached once — this gate used to be a raw
/// `std::env::var` read EVERY step; env vars never change for a live server
/// process, matching every other cached gate in this scheduler).
fn dflash_step_timing_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED
        .get_or_init(|| std::env::var("ATLAS_DFLASH_STEP_TIMING").ok().as_deref() == Some("1"))
}

/// `ATLAS_DFLASH_FORK_DEGEN=1` (cached once; see S3a gate below).
fn fork_degen_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_DFLASH_FORK_DEGEN").ok().as_deref() == Some("1"))
}

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
    // ATLAS_STEP_TIMING2: step scope. The guard's Drop records StepTotal on
    // EVERY exit path and stamps last_exit for the next step's loop_gap.
    let _t2 = crate::scheduler::step_timing2::step_begin();
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
    // (Cached: `std::env::var` takes the process env lock + linear scan —
    // this used to fire EVERY step on the hot path.)
    let step_timing = dflash_step_timing_enabled();
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
        if fork_degen_enabled() {
            (c, drafts.get(c).copied().unwrap_or(ft))
        } else {
            (c, ft)
        }
    });
    a.seq.block_fork = fork_info;
    // DDTree M2 (ATLAS_DFLASH_TREE=1): hand the tree payload to the verify
    // (which executes branch rows through per-branch scratch KV when
    // eligible, or silently degrades to flat). M3: keep a HOST copy of the
    // payload (like fork_info above) — the tree accept walk below needs the
    // tree topology to interpret the K_t tree-frame argmax rows.
    let tree_payload_host = a.pending_tree_payload.clone();
    let tree_active = a.pending_tree_payload.is_some();
    a.seq.tree_payload = a.pending_tree_payload.take();
    crate::scheduler::step_timing2::mark(crate::scheduler::step_timing2::Phase2::Pre);
    let t_verify = std::time::Instant::now();
    let mut verified_argmax = match model.decode_verify_dflash(&tokens, &mut a.seq, 0) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("decode_verify_dflash: {e:#}");
            a.finished = true;
            return;
        }
    };
    // Tree frame = [row 0 + spine rows (the flat frame)] ++ branch rows.
    // M3: when the tree executed (len > tokens.len()), keep the FULL K_t
    // tree-frame rows for the accept walk below and trim the working copy to
    // the flat frame — the pipeline / flat accept walk stay byte-identical
    // to M2 whenever the walk resolves to the spine. The winner decision
    // (adopt vs free the branch scratch) is deferred to the walk. When the
    // verify fell back to flat internally there is no scratch to resolve
    // (free is a no-op) and nothing to walk.
    let mut tree_frame_rows: Option<Vec<u32>> = None;
    if tree_active {
        if verified_argmax.len() > tokens.len() {
            TREE_STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tree_frame_rows = Some(verified_argmax.clone());
            verified_argmax.truncate(tokens.len());
        } else if let Err(e) = model.dflash_adopt_tree_branch(&mut a.seq, None) {
            tracing::error!("dflash_adopt_tree_branch(free): {e:#}");
        }
    }
    let verify_ms = if step_timing {
        t_verify.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };
    crate::scheduler::step_timing2::mark(crate::scheduler::step_timing2::Phase2::Verify);
    a.last_token_time = Instant::now();

    // ── DDTree M3: the REAL tree accept walk ────────────────────────────
    // Runs only when the verify returned tree-frame rows, and only on the
    // raw-argmax basis: the walk interprets RAW tree-frame rows, while
    // under masked verify the flat frame goes through the rep-pen/DRY
    // pipeline — the two bases could disagree, so degrade to the M2
    // trim+free there (flat walk below, branch scratch freed).
    if let Some(rows) = tree_frame_rows.as_ref() {
        let raw_basis = dflash_verify_raw_argmax
            && !crate::scheduler::verify_pipeline_helper::dflash_masked_verify_enabled();
        let walk = match (&tree_payload_host, raw_basis) {
            (Some(p), true) => resolve_tree_walk(rows, p),
            _ => TreeWalk::Spine,
        };
        match walk {
            TreeWalk::Win(w) => {
                // The path entered branch `w.winner`: swap its scratch KV
                // blocks into the block table, free the other branches'.
                if let Err(e) = model.dflash_adopt_tree_branch(&mut a.seq, Some(w.winner)) {
                    tracing::error!("dflash_adopt_tree_branch(adopt {}): {e:#}", w.winner);
                }
                let accepted = w.path_rows.len();
                commit_tree_win(model, a, drafts, tokens.len(), &w);
                if a.finished {
                    return;
                }
                dflash_propose_next(
                    model,
                    a,
                    num_drafts,
                    step_timing,
                    verify_ms,
                    tokens.len(),
                    accepted,
                );
                return;
            }
            TreeWalk::Spine => {
                // Every accepted node is spine (or the tree basis was
                // unavailable): winner=None frees every branch's scratch and
                // the flat walk below commits the identical byte stream —
                // exactly the M2 path.
                if let Err(e) = model.dflash_adopt_tree_branch(&mut a.seq, None) {
                    tracing::error!("dflash_adopt_tree_branch(free): {e:#}");
                }
            }
            TreeWalk::Malformed(why) => {
                // Safety net: never panic on the tree path. Free the scratch
                // and commit via the M2 trim — the flat frame (rows
                // 0..tokens.len()) is intact and self-consistent.
                static MALFORMED_LOGGED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !MALFORMED_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::warn!("DFLASH_TREE walk fallback to M2 trim: {why}");
                }
                if let Err(e) = model.dflash_adopt_tree_branch(&mut a.seq, None) {
                    tracing::error!("dflash_adopt_tree_branch(free): {e:#}");
                }
            }
        }
    }

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
        // In-place split (was two `to_vec` clones): `split_off(ka)` moves the
        // B-chain tail out and truncates A in place — byte-identical contents,
        // zero copies of the A prefix.
        let mut v = verified;
        let b = v.split_off(ka);
        (v, Some(b))
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
    // ATLAS_DFLASH_VSTEP_DIAG=1: one line per verify step with the full
    // draft/verified vectors. The task-#45 fork localization needs to know,
    // at the step where the committed text leaves the plain-greedy stream,
    // whether the divergent token was a BONUS (verified[acc], i.e. a row>acc
    // logits problem) or a wrongly-accepted draft (an accept-walk problem).
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ON.get_or_init(|| std::env::var("ATLAS_DFLASH_VSTEP_DIAG").as_deref() == Ok("1")) {
            tracing::info!(
                "VSTEP pre={} in={:?} verified={:?} acc={} bonus={}",
                a.seq.seq_len.saturating_sub(tokens.len()),
                tokens,
                verified,
                num_accepted,
                verified.get(num_accepted).copied().unwrap_or(u32::MAX),
            );
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

    // DDTree M0 (ATLAS_DFLASH_TREE_M0=1, measurement only): when the chain
    // dies at position d, would the drafter's SECOND choice have matched the
    // target's correction verified[d]? Per-depth hit rates bound the accept
    // gain a tree verify (top-2 branch at the death position) can deliver.
    // The stash is taken here — before this step's next-propose overwrites it.
    // M3: the scoring below indexes drafts/verified in the FLAT frame; on
    // tree-executed steps the interesting death may sit in a branch row, so
    // skip scoring (the take() still drains the stash).
    if let Some(t2) = model.dflash_take_m0_top2(&mut a.seq)
        && tree_frame_rows.is_none()
        && num_accepted < drafts.len()
        && num_accepted < verified.len()
    {
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        // depth buckets 0,1,2,3+: [deaths, top2 hits, margin_milli sum]
        static DEATHS: [AtomicU64; 4] =
            [const { AtomicU64::new(0) }; 4];
        static HITS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
        static MARGIN_MILLI: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
        let d = num_accepted;
        if let Some(&(top1, v1, top2, v2)) = t2.get(d) {
            // Only score drafter-authored deaths (draft == its own top-1;
            // retrieval/echo drafts never stash, but stay defensive).
            if drafts[d] == top1 {
                let b = d.min(3);
                DEATHS[b].fetch_add(1, Relaxed);
                let hit = verified[d] == top2;
                if hit {
                    HITS[b].fetch_add(1, Relaxed);
                }
                MARGIN_MILLI[b].fetch_add(((v1 - v2).max(0.0) * 1000.0) as u64, Relaxed);
                let total: u64 = DEATHS.iter().map(|c| c.load(Relaxed)).sum();
                if total % 64 == 0 {
                    let row = |i: usize| {
                        let n = DEATHS[i].load(Relaxed).max(1);
                        format!(
                            "d{}{}: {}/{} ({:.0}%) margin~{:.2}",
                            i,
                            if i == 3 { "+" } else { "" },
                            HITS[i].load(Relaxed),
                            DEATHS[i].load(Relaxed),
                            HITS[i].load(Relaxed) as f64 / n as f64 * 100.0,
                            MARGIN_MILLI[i].load(Relaxed) as f64 / n as f64 / 1000.0,
                        )
                    };
                    tracing::info!(
                        "TREE_M0 top2-hit-at-death: {} | {} | {} | {} (total deaths {})",
                        row(0), row(1), row(2), row(3), total
                    );
                }
            }
        }
    }

    // Adaptive speculation (ATLAS_DFLASH_ADAPTIVE=1): feed the rolling
    // accept window; may suspend this seq's speculation (see adaptive_spec).
    crate::scheduler::adaptive_spec::record_verify(a, num_accepted + committed_extra);

    // ATLAS_DSPARK_ACCEPT_LOG=1: periodic accept HISTOGRAM, not just a mean.
    // The reference reports 3.08 tok/step suite mean (4.00 on code, 2.18 on
    // adversarial prose) where our online figure sits near 1.0 despite the
    // offline engine probe reaching 3.79 on the same drafter — so the first
    // question is the SHAPE. A mean of ~1 can be a uniformly-bad drafter or a
    // bimodal one that nails whole blocks and whiffs the rest; those have
    // completely different fixes, and a peer session measured exactly such a
    // bimodal split (modes at 0 and full) on a different model.
    accept_log::record(num_accepted + committed_extra, drafts.len());

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

    // DSpark 4b inc-3: advance the DeepSeek-V4 compressed-KV pool for the
    // committed verify positions (rows 0..=num_accepted at absolute positions
    // pre_verify_len..). The batched verify path runs with pos:None and skips
    // the decode-time compressed-block append, so without this the compressed
    // attention arm freezes during speculative decode and its logits diverge
    // from plain greedy decode — the drafter's correct proposals then get
    // rejected. Replays the append from each layer's captured verify normed-x.
    // Eager only (verify runs under ATLAS_DEBUG_NO_GRAPH=1); ignores the rare
    // block-fork committed_extra. Non-fatal on error — the next step recovers.
    if let Err(e) = model.dspark_compress_catchup(pre_verify_len, num_accepted + 1, 0) {
        tracing::error!("dspark_compress_catchup: {e:#}");
    }
    // ATLAS_DSPARK_DUMP diagnostic (task #45): emit the ONLINE γ-verify-generated
    // hc-mean captures as kind=1 records so the engine probe can replay them and
    // isolate whether the acceptance collapse is the capture SOURCE (verify
    // numerics) vs the drafter. One record per committed position with its
    // committed token; no-op unless the model armed the dump file.
    {
        static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *DUMP.get_or_init(|| std::env::var("ATLAS_DSPARK_DUMP").is_ok()) {
            for i in 0..=num_accepted {
                let pos = pre_verify_len + i;
                if let Some(&tok) = a.seq.tokens.get(pos) {
                    if let Err(e) = model.dspark_dump_flush_pos(pos, tok, 0) {
                        tracing::error!("dspark_dump_flush_pos: {e:#}");
                    }
                }
            }
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

    // ── ATLAS_DFLASH_SPEC_PROPOSE resolution ────────────────────────────
    // A speculative propose (the full-accept bet) may have been enqueued
    // INSIDE the verify dispatch, on the propose stream ordered after the
    // verify graph. Decide its fate NOW — before the real ctx append and the
    // sync propose touch the drafter's shared scratch:
    //   * ADOPT iff the realized step is EXACTLY the bet: full accept, flat
    //     frame (no fork/tree win), raw-argmax pick basis (the drafter
    //     embedded the raw device argmax as its anchor), no grammar mask
    //     (the sync propose would thread the bitmask), and the seq was not
    //     just adaptively suspended. Then the optimistic ctx append IS the
    //     commit (skip the append below) and the sync propose is skipped —
    //     a placeholder chain rides pending_drafts until the next step's
    //     collect (same deferred-collect contract as ATLAS_DFLASH_ASYNC).
    //   * Otherwise DISCARD: drain the propose stream (host sync — the price
    //     of a lost bet, telemetried) and roll the drafter ctx watermark
    //     back; the flat path below then runs byte-identically.
    let mut spec_adopt = false;
    if model.dflash_spec_pending(&mut a.seq) {
        let raw_basis = dflash_verify_raw_argmax
            && !crate::scheduler::verify_pipeline_helper::dflash_masked_verify_enabled();
        spec_adopt = num_accepted == drafts.len()
            && raw_basis
            && !b_win
            && !tree_mode
            && tree_frame_rows.is_none()
            && a.grammar_state.is_none()
            && !crate::scheduler::adaptive_spec::is_suspended(a);
        if !spec_adopt
            && let Err(e) = model.dflash_spec_discard(&mut a.seq)
        {
            tracing::error!("dflash_spec_discard: {e:#}");
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
    // Spec-adopt steps skip both: the speculative launch already appended
    // exactly these rows (all K, positions pre_verify_len..) at fire time.
    if spec_adopt {
        // ctx already committed by the speculative launch.
    } else if crate::scheduler::adaptive_spec::unified_ctx_enabled() {
        if let Err(e) = model.commit_ctx(&mut a.seq, num_accepted + 1, pre_verify_len) {
            tracing::error!("commit_ctx (kgamma): {e:#}");
        }
    } else {
        let eagle_fix = crate::scheduler::adaptive_spec::eagle_fix_enabled();
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
    crate::scheduler::step_timing2::mark(crate::scheduler::step_timing2::Phase2::Walk);
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
    crate::scheduler::step_timing2::mark(crate::scheduler::step_timing2::Phase2::Emit);

    // Accept-lift draft sources (Phase A): stash this step's salvage for the
    // next propose, keyed by the bonus (= next step's last_token). Both fns
    // early-return when their env gate is off. Flat-chain path only (this IS
    // the flat path; tree verify lands in Phase C).
    // M3: SKIP both stashes on tree-executed steps (conservative). The
    // consumers assume a flat-frame verified[]; on a tree step the truncated
    // frame IS the spine slice, but branch rows past the accept may have
    // been the walk's real continuation — a stale linear tail re-offer is
    // worth less than the risk of feeding the drafter a chain the tree
    // walk already disproved. (Tree WINS return before this point anyway.)
    if tree_frame_rows.is_none() {
        // Recycle: the drafter's discarded tail drafts[num_accepted+1..].
        if let Err(e) = model.dflash_stash_recycle(&mut a.seq, drafts, num_accepted, a.last_token)
        {
            tracing::warn!("dflash_stash_recycle: {e:#}");
        }
        // Echo: the target's own argmaxes verified[num_accepted+1..].
        if let Err(e) = model.dflash_stash_echo(&mut a.seq, &verified, num_accepted, a.last_token)
        {
            tracing::warn!("dflash_stash_echo: {e:#}");
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

    // Demoted from per-step `info!` (2026-07-25 host-overhead pass): this
    // line fired EVERY DFlash step — format + synchronous subscriber write on
    // the hot scheduler thread. Full accept telemetry is still available via
    // RUST_LOG=...=debug, and the STEP_TIMING lines carry `accepted=` when
    // ATLAS_DFLASH_STEP_TIMING=1. Byte-identical output; log-level only.
    tracing::debug!(
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

    // ATLAS_DFLASH_SPEC_PROPOSE adopt: the drafter forward for the next step
    // was already enqueued during the verify with exactly the realized
    // inputs — skip the sync propose entirely. The placeholder chain routes
    // the scheduler identically (same length as the sync path); the real
    // drafts are event-collected at the top of the next step.
    if spec_adopt {
        match model.dflash_spec_adopt(&mut a.seq) {
            Ok(Some(placeholder)) if !placeholder.is_empty() => {
                a.pending_drafts = placeholder;
                a.pending_block_fork = None;
                a.pending_tree_payload = None;
                if step_timing {
                    tracing::info!(
                        "DFLASH STEP_TIMING: verify={:.1}ms propose=SPEC-ADOPTED (K={}, accepted={})",
                        verify_ms,
                        tokens.len(),
                        num_accepted,
                    );
                }
                return;
            }
            Ok(_) => {
                // Nothing in flight after all (an error path resolved it and
                // rolled the watermark back) — repair the ctx append this
                // path skipped above, then fall through to the sync propose.
                // Lossless either way (ctx is conditioning only).
                tracing::warn!("DFLASH_SPEC: adopt found no in-flight launch — sync propose");
                let repair = if crate::scheduler::adaptive_spec::unified_ctx_enabled() {
                    model.commit_ctx(&mut a.seq, num_accepted + 1, pre_verify_len)
                } else {
                    model.dflash_eagle_kgamma_append_at(&mut a.seq, num_accepted, pre_verify_len, 0)
                };
                if let Err(e) = repair {
                    tracing::error!("DFLASH_SPEC ctx-append repair: {e:#}");
                }
            }
            Err(e) => {
                tracing::error!("dflash_spec_adopt: {e:#} — sync propose fallback");
            }
        }
    }

    dflash_propose_next(
        model,
        a,
        num_drafts,
        step_timing,
        verify_ms,
        tokens.len(),
        num_accepted,
    );
}

/// Re-propose for next step — unless adaptive speculation just suspended
/// this seq (no drafts → the scheduler serial-decodes it via bootstrap).
/// Shared tail of the flat and M3 tree-win commit paths.
fn dflash_propose_next(
    model: &dyn Model,
    a: &mut ActiveSeq,
    num_drafts: usize,
    step_timing: bool,
    verify_ms: f64,
    k: usize,
    num_accepted: usize,
) {
    let _mtp_grammar_mask = mtp_grammar_mask_for(a);
    // STEP_TIMING2: everything since the Emit mark (stashes, metrics, log,
    // commit_accepted_prefix, save_hidden, trim, grammar mask) → `commit`.
    crate::scheduler::step_timing2::mark(crate::scheduler::step_timing2::Phase2::Commit);
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
                // DDTree M2: the tree payload rides with the drafts too.
                a.pending_tree_payload = model.dflash_take_tree_payload(&mut a.seq);
            }
            Ok(_) => {
                a.pending_block_fork = None;
                a.pending_tree_payload = None;
            }
            Err(e) => {
                a.pending_block_fork = None;
                a.pending_tree_payload = None;
                tracing::error!("run_mtp_propose_multi (dflash): {e:#}");
            }
        }
    }
    crate::scheduler::step_timing2::mark(crate::scheduler::step_timing2::Phase2::Propose);
    if step_timing {
        let propose_ms = t_propose.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(
            "DFLASH STEP_TIMING: verify={:.1}ms propose={:.1}ms (K={}, accepted={})",
            verify_ms,
            propose_ms,
            k,
            num_accepted,
        );
    }
}

// ════════════════════════════════════════════════════════════════════════
// DDTree M3 — tree accept walk + branch KV adoption
// ════════════════════════════════════════════════════════════════════════

/// Steps where the verify executed the tree (returned K_t tree-frame rows).
static TREE_STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Resolved M3 tree accept walk that LEFT the spine (entered a branch).
struct TreeWalkWin {
    /// Accepted path tokens, root→tip order (spine prefix ++ branch nodes).
    path_tokens: Vec<u32>,
    /// Accepted compact row indices in the K_t frame (row i = compact node
    /// i; non-contiguous past the fork, e.g. `[1, 2, 5, 6]`).
    path_rows: Vec<usize>,
    /// Target argmax at the path-tip row — the committed bonus.
    bonus: u32,
    /// Tree-frame row that generated the bonus (= last accepted compact
    /// index; the flat analog is row `num_accepted`).
    tip_row: usize,
    /// Winning branch id (index into `seq.tree_branch_scratch`).
    winner: usize,
    /// Contiguous spine-prefix length of the path — the flat-equivalent
    /// accept count (what the M2 trim would have committed as drafts).
    flat_prefix: usize,
}

/// Outcome of [`resolve_tree_walk`].
enum TreeWalk {
    /// Every accepted node is spine — commit via the flat path (winner=None
    /// frees all scratch; byte-identical to M2).
    Spine,
    /// The path entered a branch — commit the branch path (adopt winner).
    Win(TreeWalkWin),
    /// Inconsistent rows/payload — fall back to the M2 trim (log once).
    Malformed(&'static str),
}

/// Walk the K_t tree-frame argmax rows over the payload's topology.
///
/// Frame conventions (see `ddtree.rs` unit tests): argmax rows are
/// `[root_row, ...compact rows]` — row `i` is the target's greedy next token
/// AFTER compact node `i` (row 0 = after the root/last_token); compact node
/// `i` (1-based) is `payload.tree_token_ids[i-1]`. No DFS permutation in
/// this port. `greedy_sample_ddtree_full` commits the WHOLE path the greedy
/// oracle takes (possibly through a sibling fork), token contract identical
/// to the flat walk — only the recognized accept SET changes.
fn resolve_tree_walk(rows: &[u32], payload: &spark_model::layers::DDTreePayload) -> TreeWalk {
    use spark_model::layers::dflash_head::ddtree;

    let n = payload.tree_token_ids.len();
    if n == 0 || payload.parent_indices.len() != n || rows.len() != n + 1 {
        return TreeWalk::Malformed("rows/payload length mismatch");
    }
    // The plan builder re-derives spine_len + the compact-row→branch-id map
    // exactly as the verify executed them (base/block_size only shape the
    // touched-block ranges, unused here).
    let Some(plan) = ddtree::build_tree_verify_plan(payload, 0, 1) else {
        return TreeWalk::Malformed("payload not in spine+branch shape");
    };
    let req = ddtree::DDTreeRequestRuntime {
        req_id: String::new(),
        tree_token_ids: payload.tree_token_ids.clone(),
        parent_indices: payload.parent_indices.clone(),
    };
    let sample = match ddtree::greedy_sample_ddtree_full(&req, rows) {
        Ok(s) => s,
        Err(_) => return TreeWalk::Malformed("greedy_sample_ddtree_full failed"),
    };
    let path_rows = sample.accepted_compact_indices;
    let mut path_tokens = sample.output_token_ids;
    if path_tokens.len() != path_rows.len() + 1 {
        return TreeWalk::Malformed("sample path/output length mismatch");
    }
    // Winner branch: compact rows ≤ spine_len are spine (branch id None);
    // the walk can enter at most one branch (branches are linear runs) and
    // never returns to the spine — validate both instead of trusting it.
    let mut winner: Option<usize> = None;
    let mut flat_prefix = 0usize;
    for (i, &c) in path_rows.iter().enumerate() {
        if c == 0 || c >= rows.len() {
            return TreeWalk::Malformed("accepted compact index out of bounds");
        }
        match plan.row_branch.get(c).copied().flatten() {
            None => {
                if winner.is_some() || c != i + 1 {
                    return TreeWalk::Malformed("non-contiguous spine walk");
                }
                flat_prefix += 1;
            }
            Some(b) => match winner {
                None => winner = Some(b),
                Some(w) if w == b => {}
                Some(_) => return TreeWalk::Malformed("path crosses branches"),
            },
        }
    }
    let Some(winner) = winner else {
        return TreeWalk::Spine;
    };
    let Some(bonus) = path_tokens.pop() else {
        return TreeWalk::Malformed("empty sample output");
    };
    let tip_row = sample.bonus_parent_compact_index;
    if tip_row >= rows.len() {
        return TreeWalk::Malformed("bonus tip row out of bounds");
    }
    TreeWalk::Win(TreeWalkWin {
        path_tokens,
        path_rows,
        bonus,
        tip_row,
        winner,
        flat_prefix,
    })
}

/// Commit an M3 tree-walk branch WIN. Mirrors the flat path's bookkeeping
/// with the accepted PATH (spine prefix ++ winner-branch nodes) substituted
/// for the accepted draft prefix; `n_path` plays `num_accepted`:
///   * rollback: `target_seq_len = pre_verify_len + n_path + 1` (flat:
///     `pre_verify_len + num_accepted + 1`). `seq.tokens` keeps `tokens[0]`
///     at index `pre_verify_len` (where decode_verify pushed it) and the
///     path tokens replace the pushed drafts (the branch nodes' KV lives in
///     the adopted scratch blocks at exactly those positions — path depth j
///     sits at `pre_verify_len + j`). The bonus is NOT pushed (flat
///     convention: it lives only in `a.last_token` and becomes `tokens[0]`
///     of the next verify).
///   * ctx append: the path's capture rows are TREE rows — `[0] ++
///     path_rows` (capture_all captured all K_t rows) at RoPE positions
///     `pre_verify_len..=pre_verify_len+n_path`, tip appended last (EAGLE
///     order). Row-list variant; flat/spine-only steps keep commit_ctx /
///     kgamma_append.
///   * emit: path tokens, then the bonus; `a.last_token = bonus`.
fn commit_tree_win(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    k: usize,
    w: &TreeWalkWin,
) {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    let n_path = w.path_rows.len();

    // Telemetry (rate-limited, DFLASH_BLOCKFORK-WIN style). `extra` counts
    // tokens committed beyond the flat-equivalent accept: the M2 trim would
    // commit flat_prefix drafts + 1 bonus (the fork token); the tree commits
    // n_path path tokens + 1 fresh bonus.
    static TREE_WINS: AtomicU64 = AtomicU64::new(0);
    static TREE_EXTRA: AtomicU64 = AtomicU64::new(0);
    static TREE_DEPTH_WINS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
    let extra = n_path.saturating_sub(w.flat_prefix);
    let wins = TREE_WINS.fetch_add(1, Relaxed) + 1;
    TREE_EXTRA.fetch_add(extra as u64, Relaxed);
    let fork_depth = w.flat_prefix + 1; // depth of the winning fork node
    TREE_DEPTH_WINS[fork_depth.min(7)].fetch_add(1, Relaxed);
    if wins <= 8 || wins % 64 == 0 {
        let hist: Vec<String> = TREE_DEPTH_WINS
            .iter()
            .enumerate()
            .filter(|(_, c)| c.load(Relaxed) > 0)
            .map(|(d, c)| format!("d{}{}:{}", d, if d == 7 { "+" } else { "" }, c.load(Relaxed)))
            .collect();
        tracing::info!(
            "DFLASH_TREE WIN #{wins}: branch={} fork_depth={fork_depth} path={n_path} \
             (flat-equiv {}) +{extra} extra | tree_steps={} total_extra={} depth_wins=[{}]",
            w.winner,
            w.flat_prefix,
            TREE_STEPS.load(Relaxed),
            TREE_EXTRA.load(Relaxed),
            hist.join(" "),
        );
    }

    // Adaptive speculation: total accepted count (flat feeds num_accepted
    // + blockfork extra the same way).
    crate::scheduler::adaptive_spec::record_verify(a, n_path);

    // Rollback + path commit (see doc comment). decode_verify pushed the k
    // FLAT-frame tokens and advanced seq_len by k (verify_d_tree.rs tail is
    // identical to verify_d.rs here); drop the k-1 pushed drafts, keep
    // tokens[0], push the accepted path in root→tip order.
    let pre_verify_len = a.seq.seq_len.saturating_sub(k);
    let keep = a.seq.tokens.len().saturating_sub(k - 1);
    a.seq.tokens.truncate(keep);
    a.seq.tokens.extend_from_slice(&w.path_tokens);
    a.seq.seq_len = pre_verify_len + n_path + 1;

    // Drafter ctx append — TREE capture rows, same env gates as the flat
    // path (unified commit_ctx / EAGLE-fix kgamma append).
    let unified = crate::scheduler::adaptive_spec::unified_ctx_enabled();
    let eagle_fix = crate::scheduler::adaptive_spec::eagle_fix_enabled();
    if unified || eagle_fix {
        let mut rows = Vec::with_capacity(n_path + 1);
        rows.push(0usize);
        rows.extend_from_slice(&w.path_rows);
        if let Err(e) = model.dflash_ctx_append_rows(&mut a.seq, &rows, pre_verify_len) {
            tracing::error!("dflash_ctx_append_rows (tree): {e:#}");
        }
    }

    // Emit the accepted path, then the bonus (flat emit convention: drafts
    // via emit_token, bonus last + becomes a.last_token).
    for &t in &w.path_tokens {
        emit_token(a, t, None);
        if a.finished {
            return;
        }
    }
    emit_token(a, w.bonus, None);
    if a.finished {
        return;
    }
    a.last_token = w.bonus;

    // Echo/recycle stashes: SKIPPED on tree wins (conservative) — both
    // consumers assume a flat-frame verified[]; the rows past the tip here
    // are branch rows, not a linear continuation.

    crate::metrics::SPEC_DECODE_VERIFY
        .with_label_values(&[
            "dflash",
            if n_path >= drafts.len() {
                "accept_all"
            } else {
                "accept_partial"
            },
        ])
        .inc();

    // Demoted from `info!` (2026-07-25 host-overhead pass) — per-win hot-loop
    // log; the rate-limited DFLASH_TREE WIN telemetry above stays at info.
    tracing::debug!(
        "DFLASH K=γ verify (tree win): γ={} accepted={} (flat-equiv {}) seq_len={}",
        drafts.len(),
        n_path,
        w.flat_prefix,
        a.seq.seq_len,
    );

    // Same commit tail as the flat path (k_verify = k, the flat-frame row
    // count; total accepted = path + bonus). Pure-attention Laguna:
    // commit_accepted_prefix touches only LinearAttention layers — no-op
    // here, kept for parity with the flat path.
    if let Err(e) = model.commit_accepted_prefix(&mut a.seq, n_path + 1, k) {
        tracing::error!("commit_accepted_prefix (dflash tree): {e:#}");
        a.finished = true;
        return;
    }

    // The bonus generator is the path-tip TREE row (flat analog: row
    // `total_accepted - 1` = num_accepted).
    if let Err(e) = model.save_hidden_for_mtp(w.tip_row, 0) {
        tracing::error!("save_hidden_for_mtp (dflash tree): {e:#}");
    }

    if let Err(e) = model.trim_proposer_state(&mut a.seq, n_path, 0) {
        tracing::error!("trim_proposer_state: {e:#}");
    }
}

#[cfg(test)]
mod tree_walk_tests {
    use super::*;

    fn payload(tokens: &[u32], parents: &[i32]) -> spark_model::layers::DDTreePayload {
        spark_model::layers::DDTreePayload {
            tree_token_ids: tokens.to_vec(),
            parent_indices: parents.to_vec(),
        }
    }

    // Shared shape: spine [10,20,30,40] at payload idx 0..3 (compact 1..4),
    // branch 0 = fork 99 off spine node compact 1 (payload parent 0) with
    // tail 31 (payload parent 4). K_t = 7 rows.
    fn branchy() -> spark_model::layers::DDTreePayload {
        payload(&[10, 20, 30, 40, 99, 31], &[-1, 0, 1, 2, 0, 4])
    }

    #[test]
    fn spine_only_walk_resolves_to_spine() {
        // Target rides the top-1 chain 2 deep then dies: rows
        // [10, 20, X(miss), _, _, _, _] — all accepted nodes are spine.
        let rows = vec![10u32, 20, 555, 0, 0, 0, 0];
        match resolve_tree_walk(&rows, &branchy()) {
            TreeWalk::Spine => {}
            _ => panic!("expected Spine"),
        }
    }

    #[test]
    fn branch_win_commits_fork_tail_and_bonus() {
        // Row 0 → 10 (accept compact 1), row 1 (after node 1) → 99 (fork to
        // compact 5), row 5 (after fork) → 31 (accept compact 6), row 6 →
        // 777 (bonus at the tip).
        let rows = vec![10u32, 99, 0, 0, 0, 31, 777];
        match resolve_tree_walk(&rows, &branchy()) {
            TreeWalk::Win(w) => {
                assert_eq!(w.path_tokens, vec![10, 99, 31]);
                assert_eq!(w.path_rows, vec![1, 5, 6]);
                assert_eq!(w.bonus, 777);
                assert_eq!(w.tip_row, 6);
                assert_eq!(w.winner, 0);
                assert_eq!(w.flat_prefix, 1);
            }
            _ => panic!("expected Win"),
        }
    }

    #[test]
    fn root_level_fork_wins_with_zero_flat_prefix() {
        // Branch 1 = bare fork 88 attached to the ROOT (parent -1).
        let p = payload(&[10, 20, 88], &[-1, 0, -1]);
        // Row 0 → 88 (fork straight into compact 3), row 3 → 777 bonus.
        let rows = vec![88u32, 0, 0, 777];
        match resolve_tree_walk(&rows, &p) {
            TreeWalk::Win(w) => {
                assert_eq!(w.path_tokens, vec![88]);
                assert_eq!(w.path_rows, vec![3]);
                assert_eq!(w.bonus, 777);
                assert_eq!(w.tip_row, 3);
                assert_eq!(w.winner, 0);
                assert_eq!(w.flat_prefix, 0);
            }
            _ => panic!("expected Win"),
        }
    }

    #[test]
    fn two_branch_payload_walk_enters_second_branch() {
        // M4 shape: spine [10,20,30,40] (compact 1..4); branch 0 = fork 99
        // off compact 1 (payload parent 0) + tail 55; branch 1 = fork 88 off
        // compact 2 (payload parent 1) + tail 66. K_t = 9 rows.
        let p = payload(
            &[10, 20, 30, 40, 99, 55, 88, 66],
            &[-1, 0, 1, 2, 0, 4, 1, 6],
        );
        // Row 0 → 10 (accept compact 1), row 1 → 20 (accept compact 2,
        // NOT fork 99), row 2 (after node 2) → 88 (fork into branch 1,
        // compact 7), row 7 (after fork) → 66 (accept tail compact 8),
        // row 8 → 777 (bonus at the tip).
        let rows = vec![10u32, 20, 88, 0, 0, 0, 0, 66, 777];
        match resolve_tree_walk(&rows, &p) {
            TreeWalk::Win(w) => {
                assert_eq!(w.path_tokens, vec![10, 20, 88, 66]);
                assert_eq!(w.path_rows, vec![1, 2, 7, 8]);
                assert_eq!(w.bonus, 777);
                assert_eq!(w.tip_row, 8);
                assert_eq!(w.winner, 1);
                assert_eq!(w.flat_prefix, 2);
            }
            TreeWalk::Spine => panic!("expected Win, got Spine"),
            TreeWalk::Malformed(m) => panic!("expected Win, got Malformed: {m}"),
        }
    }

    #[test]
    fn wrong_row_count_is_malformed() {
        let rows = vec![10u32, 20, 30]; // needs 7
        match resolve_tree_walk(&rows, &branchy()) {
            TreeWalk::Malformed(_) => {}
            _ => panic!("expected Malformed"),
        }
    }

    #[test]
    fn spineless_payload_is_malformed() {
        // First payload node does not chain off the root contiguously in
        // the M2 spine+branch shape (parent 1 with no spine) — the plan
        // builder rejects it, so the walk must fall back.
        let p = payload(&[10, 20], &[1, -1]);
        let rows = vec![10u32, 20, 30];
        match resolve_tree_walk(&rows, &p) {
            TreeWalk::Malformed(_) => {}
            _ => panic!("expected Malformed"),
        }
    }

    #[test]
    fn degenerate_duplicate_branch_resolves_to_spine() {
        // DEGEN transparency gate (ATLAS_DFLASH_TREE_DEGEN=1): branch tokens
        // EQUAL the spine tokens at the same depths, so the cliff parent has
        // two children with identical token ids. The tie must resolve to the
        // SPINE child — otherwise the walk enters the 2-row branch and the
        // accept is capped at the branch length (live regression: DEGEN
        // accept 0.74, dist {0,1,2} capped at 2).
        //
        // Spine [10,20,30,40] (compact 1..4); branch = fork 20 off compact 1
        // (duplicate of compact 2) + tail 30 (duplicate of compact 3).
        let p = payload(&[10, 20, 30, 40, 20, 30], &[-1, 0, 1, 2, 0, 4]);
        // Target rides the full spine; bonus 777 read at the spine tip.
        let rows = vec![10u32, 20, 30, 40, 777, 30, 999];
        match resolve_tree_walk(&rows, &p) {
            TreeWalk::Spine => {}
            TreeWalk::Win(w) => panic!(
                "duplicate-token tie entered branch {} (path_rows {:?}) — must stay Spine",
                w.winner, w.path_rows
            ),
            TreeWalk::Malformed(m) => panic!("unexpected Malformed: {m}"),
        }
    }

    #[test]
    fn full_spine_accept_stays_spine() {
        // Whole spine accepted, bonus read at spine tip (compact 4): the
        // walk never enters the branch → Spine (flat path commits it).
        let rows = vec![10u32, 20, 30, 40, 777, 0, 0];
        match resolve_tree_walk(&rows, &branchy()) {
            TreeWalk::Spine => {}
            _ => panic!("expected Spine"),
        }
    }
}
