// SPDX-License-Identifier: AGPL-3.0-only

//! Self-speculative + NGram speculative decoding step + grammar helpers.

use super::*;

/// Self-speculative step: draft via layer-skipping, verify with full model.
/// Combines bootstrap + verify in one step (no pipeline).
///
/// `verify_ctx` is plumbed into the verify-time argmax replacement so
/// each verify position runs through the full 8-stage pre-sample
/// pipeline instead of falling through unmasked. See
/// `verify_pipeline_helper` for the rationale.
/// Whether a lone sequence's next step may run the model's internal speculation (DeepSeek-V4.1
/// DSpark). Greedy only (sampled speculation is not wired yet), no logprobs (spec emits none),
/// bf16 logits. Thinking and grammar ARE allowed: every draft is accepted only if it equals the
/// full per-position pipeline pick (`verify_pick_all_with_pipeline`, the MTP verify basis), and
/// the next token is picked by the normal decode path.
pub fn internal_spec_eligible(a: &ActiveSeq, model: &dyn Model) -> bool {
    // Greedy: any grammar (drafts are checked against the pipeline pick). Sampled: no grammar
    // (the p rows would need the matcher advanced along the drafts).
    (spec_greedy(a) || a.grammar_state.is_none())
        && a.top_logprobs.is_none()
        && !a.suppress_tool_call
        && !a.disable_mtp
        && !model.decode_logits_fp32()
}

/// One step of a model's internal speculation: `spec_verify` (draft + one verify pass), then
/// accept the leading drafts that equal the pipeline pick at their position and are not a
/// structural id (EOS / think / tool-call / hard stop, which always go through the normal
/// per-token handler), `spec_commit` them, emit them like MTP-accepted tokens, and pick the
/// next token from the verify row that follows them through `process_decode_logits`, exactly
/// as after a plain decode.
#[allow(clippy::too_many_arguments)]
pub fn step_internal_spec(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
) {
    let t0 = std::time::Instant::now();
    let a = &mut active[0];
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, a.last_token) {
        tracing::error!("EP broadcast internal-spec token: {e:#}");
        a.finished = true;
        return;
    }
    if !spec_greedy(a) {
        return step_internal_spec_sampled(
            model,
            active,
            verify_ctx,
            think_end_token,
            think_start_token,
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
            adaptive_sampling,
        );
    }
    let (drafts, argmax) = match model.spec_verify(a.last_token, &mut a.seq, None, 0) {
        Ok(Some(r)) => r,
        Ok(None) => {
            step_decode_only(
                model,
                active,
                think_end_token,
                think_start_token,
                code_fence_token,
                tool_call_start_token,
                tool_call_end_token,
                adaptive_sampling,
            );
            return;
        }
        Err(e) => {
            tracing::error!("internal speculation verify: {e:#}");
            let mut a = active.remove(0);
            send_error(model, &mut a, &format!("{e:#}"));
            return;
        }
    };
    let picks = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(model, &argmax, a, verify_ctx);
    let structural = |t: u32| {
        a.eos_tokens.contains(&t)
            || [think_end_token, think_start_token, tool_call_start_token, tool_call_end_token, tool_response_hard_stop()]
                .contains(&Some(t))
    };
    let accepted = drafts.iter().zip(&picks).take_while(|(d, p)| d == p && !structural(**d)).count();
    if let Err(e) = model.spec_commit(&mut a.seq, accepted, 0) {
        tracing::error!("internal speculation commit: {e:#}");
        let mut a = active.remove(0);
        send_error(model, &mut a, &format!("{e:#}"));
        return;
    }
    for &tok in &drafts[..accepted] {
        emit_token(a, tok, None);
        if a.finished {
            return;
        }
        a.last_token = tok;
    }
    let row = model.logits_buffer_ptr().offset(accepted * model.vocab_size() * 2);
    process_decode_logits(
        model,
        active,
        row,
        t0,
        think_end_token,
        think_start_token,
        code_fence_token,
        tool_call_start_token,
        tool_call_end_token,
        adaptive_sampling,
    );
}

fn spec_greedy(a: &ActiveSeq) -> bool {
    a.temperature == 0.0 || crate::scheduler::decode_logits_seq::force_temp_zero_enabled()
}

/// A uniform in [0, 1) for speculative sampling. Seeded requests derive it from the same
/// per-position seed as plain sampling (`seed + output position`), salted per use (draft /
/// accept / next), so a seeded run reproduces; unseeded requests use a process-wide stream.
fn spec_uniform(seed: Option<u64>, pos: usize, salt: u64) -> f64 {
    fn splitmix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    let bits = match seed {
        Some(s) => splitmix(splitmix(s.wrapping_add(pos as u64)) ^ salt),
        None => {
            static STATE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
            splitmix(STATE.fetch_add(0x9E37_79B9_7F4A_7C15, std::sync::atomic::Ordering::Relaxed) ^ t ^ salt)
        }
    };
    (bits >> 11) as f64 / (1u64 << 53) as f64
}

fn spec_fail(model: &dyn Model, active: &mut Vec<ActiveSeq>, what: &str, e: anyhow::Error) {
    tracing::error!("internal speculation (sampled) {what}: {e:#}");
    let mut a = active.remove(0);
    send_error(model, &mut a, &format!("{e:#}"));
}

const SALT_DRAFT: u64 = 0xD2AF_7001;
const SALT_ACCEPT: u64 = 0xACCE_7002;
const SALT_NEXT: u64 = 0x4E78_7003;

/// Sampled (temperature > 0) internal speculation: the model samples drafts from its own q, then
/// speculative rejection sampling against p = each verify row's full sampler distribution
/// (pipeline + temperature/top-k/top-p/min-p, the same distribution plain sampling draws from), so
/// emitted tokens are distributed exactly as plain sampling. Acceptance ends before any
/// structural id; the token after the accepted drafts is the rule's residual / p sample.
#[allow(clippy::too_many_arguments)]
fn step_internal_spec_sampled(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
) {
    use spark_model::weight_loader::deepseek_v41::dspark_sampling::speculative_accept;
    let a = &mut active[0];
    let base = a.output_tokens.len();
    const B: usize = 5;
    let sampling = spark_model::traits::SpecSampling {
        temperature: a.temperature,
        draft_uniforms: (0..B).map(|i| spec_uniform(a.seed, base * 8 + i, SALT_DRAFT) as f32).collect(),
    };
    let drafts = match model.spec_verify(a.last_token, &mut a.seq, Some(&sampling), 0) {
        Ok(Some((d, _))) => d,
        Ok(None) => {
            // Too close to max_seq: one plain sampled step.
            return step_decode_only(
                model,
                active,
                think_end_token,
                think_start_token,
                code_fence_token,
                tool_call_start_token,
                tool_call_end_token,
                adaptive_sampling,
            );
        }
        Err(e) => return spec_fail(model, active, "verify", e),
    };
    let vocab = model.vocab_size();
    let b = drafts.len();
    let mut rows = vec![0u8; (b + 1) * vocab * 2];
    if let Err(e) = model.copy_logits_to_host(model.logits_buffer_ptr(), &mut rows) {
        return spec_fail(model, active, "verify rows D2H", e);
    }
    let q_ptr = match model.spec_draft_probs() {
        Some(p) => p,
        None => return spec_fail(model, active, "draft probs", anyhow::anyhow!("sampled spec_verify returned no q rows")),
    };
    let mut qb = vec![0u8; b * vocab * 4];
    if let Err(e) = model.copy_logits_to_host(q_ptr, &mut qb) {
        return spec_fail(model, active, "draft probs D2H", e);
    }
    let q: Vec<Vec<f32>> = qb.chunks_exact(vocab * 4).map(|r| r.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()).collect();
    let p: Vec<Vec<f32>> = rows
        .chunks_exact(vocab * 2)
        .map(|r| crate::scheduler::verify_pipeline_helper::verify_row_distribution(r, false, vocab, a, verify_ctx))
        .collect();
    let structural = |t: u32| {
        a.eos_tokens.contains(&t)
            || [think_end_token, think_start_token, tool_call_start_token, tool_call_end_token, tool_response_hard_stop()]
                .contains(&Some(t))
    };
    // Acceptance may not pass a structural draft: test only the prefix before the first one.
    let k = drafts.iter().position(|&d| structural(d)).unwrap_or(b);
    let pr: Vec<&[f32]> = p[..=k].iter().map(|v| v.as_slice()).collect();
    let qr: Vec<&[f32]> = q[..k].iter().map(|v| v.as_slice()).collect();
    let ua: Vec<f64> = (0..k).map(|i| spec_uniform(a.seed, base * 8 + i, SALT_ACCEPT)).collect();
    let out = match speculative_accept(&pr, &qr, &drafts[..k], &ua, spec_uniform(a.seed, base * 8, SALT_NEXT)) {
        Ok(o) => o,
        Err(e) => return spec_fail(model, active, "accept", e),
    };
    if let Err(e) = model.spec_commit(&mut a.seq, out.accepted, 0) {
        return spec_fail(model, active, "commit", e);
    }
    for &tok in drafts[..out.accepted].iter().chain(std::iter::once(&out.next)) {
        emit_token(a, tok, None);
        a.last_token = tok;
        if a.finished {
            return;
        }
    }
}

pub fn step_self_spec(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
) {
    let a = &mut active[0];

    // 1. Full-model decode to get token_0
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, a.last_token) {
        tracing::error!("EP broadcast self-spec token: {e:#}");
        a.finished = true;
        return;
    }
    let logits = match model.decode(a.last_token, &mut a.seq, 0) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("self-spec decode error: {e:#}");
            a.finished = true;
            return;
        }
    };
    let token_0 = match model.argmax_on_device(logits, 0) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("self-spec argmax error: {e:#}");
            a.finished = true;
            return;
        }
    };

    // 2. Draft phase: layer-skipping for cheap predictions
    let seq_len_before_draft = a.seq.seq_len;
    let tokens_before_draft = a.seq.tokens.len();

    let mut draft_tokens = Vec::with_capacity(num_drafts);
    let mut draft_token = token_0;
    for _ in 0..num_drafts {
        let logits = match model.decode_draft(draft_token, &mut a.seq, 0) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("self-spec draft error: {e:#}");
                break;
            }
        };
        draft_token = match model.argmax_on_device(logits, 0) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("self-spec draft argmax error: {e:#}");
                break;
            }
        };
        draft_tokens.push(draft_token);
    }

    // 3. Rewind to pre-draft state (SSM unchanged since we skipped SSM layers)
    a.seq.seq_len = seq_len_before_draft;
    a.seq.tokens.truncate(tokens_before_draft);

    if draft_tokens.is_empty() {
        // No drafts: emit token_0 and continue
        emit_token(a, token_0, None);
        if !a.finished {
            a.last_token = token_0;
        }
        return;
    }

    // 4. Checkpoint SSM states before verification
    if let Err(e) = model.checkpoint_ssm_states(&mut a.seq) {
        tracing::error!("self-spec checkpoint: {e:#}");
        a.finished = true;
        return;
    }
    let seq_len_before_verify = a.seq.seq_len;

    // 5. Verify: run full model on [token_0, d1, ..., dK]
    let mut verify_tokens = vec![token_0];
    verify_tokens.extend_from_slice(&draft_tokens);

    let verified_argmax = match model.decode_verify(&verify_tokens, &mut a.seq, 0) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("self-spec verify error: {e:#}");
            a.finished = true;
            return;
        }
    };

    // Phase C-2 (2026-05-24): replay the pre-sample
    // logits-processor pipeline per verify position. `decode_verify`
    // wrote `[verify_tokens.len(), vocab]` BF16 into `logits_buffer`;
    // the helper copies it D2H and applies the same 8-stage pipeline
    // used in the non-MTP path. Falls back to the raw argmax on D2H
    // failure (see helper).
    let verified = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
        model,
        &verified_argmax,
        a,
        verify_ctx,
    );

    // 6. Compare draft vs verified, count acceptances
    let n_drafts = draft_tokens.len();
    let mut num_accepted = 0;

    emit_token(a, token_0, None);
    if a.finished {
        return;
    }

    for i in 0..n_drafts {
        if draft_tokens[i] == verified[i] {
            emit_token(a, draft_tokens[i], None);
            if a.finished {
                return;
            }
            num_accepted += 1;
        } else {
            emit_token(a, verified[i], None);
            if a.finished {
                return;
            }
            a.last_token = verified[i];
            break;
        }
    }

    if num_accepted == n_drafts && n_drafts > 0 {
        emit_token(a, verified[n_drafts], None);
        if !a.finished {
            a.last_token = verified[n_drafts];
        }
    } else if num_accepted < n_drafts {
        // Already set a.last_token above in the break
    } else {
        a.last_token = token_0;
    }

    // 7. Rollback extra verify tokens
    // tokens_added = token_0 (always kept) + accepted drafts
    let tokens_added = 1 + num_accepted;
    let expected_seq_len = seq_len_before_verify + tokens_added;

    if a.seq.seq_len > expected_seq_len {
        let extra = a.seq.seq_len - expected_seq_len;
        for _ in 0..extra {
            a.seq.seq_len -= 1;
            a.seq.tokens.pop();
        }
        // +1 because token_0 is always accepted in the verify batch
        if let Err(e) = model.rollback_ssm_states(&mut a.seq, num_accepted + 1) {
            tracing::error!("self-spec rollback: {e:#}");
        }
    }
}

/// N-gram speculative step: CPU proposer + CUDA-graphed K=2 verify.
///
/// Two-phase pipeline (same as MTP but with N-gram proposer instead):
/// 1. Bootstrap: regular decode → argmax → N-gram propose → pending_drafts
/// 2. Verify: decode_verify_graphed(K=2) → accept/reject → SSM rollback
pub fn step_ngram(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    proposer: &mut NgramProposer,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
) {
    let a = &mut active[0];

    if !a.pending_drafts.is_empty() {
        // ── Phase B: Verify pending draft ──
        let drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
        step_ngram_verify(model, a, &drafts, proposer, verify_ctx);
    } else {
        // ── Phase A: Bootstrap decode + N-gram propose ──
        if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, a.last_token) {
            tracing::error!("EP broadcast ngram bootstrap: {e:#}");
            a.finished = true;
            return;
        }
        let logits = match model.decode(a.last_token, &mut a.seq, 0) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("ngram bootstrap decode error: {e:#}");
                a.finished = true;
                return;
            }
        };
        let tok = match model.argmax_on_device(logits, 0) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("ngram bootstrap argmax error: {e:#}");
                a.finished = true;
                return;
            }
        };

        // Observe the token for future predictions
        proposer.observe(&a.seq.tokens, tok);

        emit_token(a, tok, None);
        if a.finished {
            return;
        }
        a.last_token = tok;

        // N-gram propose (CPU-only, zero GPU cost)
        if let Some(draft) = proposer.propose(&a.seq.tokens) {
            a.pending_drafts = vec![draft];

            // Checkpoint SSM for potential rollback during verify
            if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
                tracing::error!("ngram start_checkpoint_async: {e:#}");
            }
        }
        // If no proposal: next iteration will be another bootstrap (regular decode)
    }
}

/// Verify a single N-gram draft via CUDA-graphed K=2 path.
pub fn step_ngram_verify(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    proposer: &mut NgramProposer,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
) {
    let t_sync = Instant::now();
    if let Err(e) = model.sync_secondary() {
        tracing::error!("ngram sync_secondary: {e:#}");
        a.finished = true;
        return;
    }
    let sync_us = t_sync.elapsed().as_micros();

    // EP: broadcast verify K=2 command + tokens
    let tokens_k2 = [a.last_token, drafts[0]];
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, 0xFFFFFFF2) {
        tracing::error!("EP broadcast ngram verify cmd: {e:#}");
        a.finished = true;
        return;
    }
    for &t in &tokens_k2 {
        if let Err(e) = model.ep_broadcast_cmd(t) {
            tracing::error!("EP broadcast ngram verify token: {e:#}");
            a.finished = true;
            return;
        }
    }

    let t_verify = Instant::now();
    let result = match model.decode_verify_graphed(&tokens_k2, &mut a.seq, 0) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("ngram decode_verify_graphed: {e:#}");
            a.finished = true;
            return;
        }
    };
    let verify_us = t_verify.elapsed().as_micros();
    a.last_token_time = Instant::now();
    let [v0_argmax, v1_argmax] = result;

    // Phase C-2 (2026-05-24): apply the full pre-sample
    // logits-processor pipeline to each verify position before
    // computing the accept/reject argmax. Without this, ngram-verify
    // tokens escape mid-word / forced-think-end / pin-to-tool-call /
    // grammar masks — see `verify_pipeline_helper` for the root-
    // cause analysis.
    let processed = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
        model,
        &[v0_argmax, v1_argmax],
        a,
        verify_ctx,
    );
    let v0 = processed.first().copied().unwrap_or(v0_argmax);
    let v1 = processed.get(1).copied().unwrap_or(v1_argmax);
    let accepted = drafts[0] == v0;

    // EP: broadcast accept/reject to worker
    if let Err(e) = model.ep_broadcast_cmd(accepted as u32) {
        tracing::error!("EP broadcast ngram verify result: {e:#}");
        a.finished = true;
        return;
    }

    if accepted {
        // ── ACCEPTED: emit both tokens ──
        // After verify_graphed, a.seq.tokens has [.., last_token, drafts[0]] appended.
        // Observe: context ending with last_token → drafts[0] was correct
        // Observe: context ending with drafts[0] → v1 is the next prediction
        proposer.observe(&a.seq.tokens[..a.seq.tokens.len() - 1], drafts[0]);
        proposer.observe(&a.seq.tokens, v1);

        emit_token(a, drafts[0], None);
        if !a.finished {
            emit_token(a, v1, None);
        }
        if a.finished {
            return;
        }
        a.last_token = v1;

        // Checkpoint SSM for next verify
        if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
            tracing::error!("ngram accept checkpoint: {e:#}");
        }

        // Propose next draft
        if let Some(draft) = proposer.propose(&a.seq.tokens) {
            a.pending_drafts = vec![draft];
        }

        if a.seq.seq_len.is_multiple_of(50) {
            tracing::info!(
                "NGRAM K2 ACCEPT: sync={sync_us}μs verify={verify_us}μs cache={} seq_len={}",
                proposer.len(),
                a.seq.seq_len,
            );
        }
    } else {
        // ── REJECTED: rollback SSM, emit v0 only ──
        a.seq.seq_len -= 1;
        a.seq.tokens.pop();

        if let Err(e) = model.start_rollback_and_checkpoint_async(&mut a.seq, 1) {
            tracing::error!("ngram rollback: {e:#}");
            a.finished = true;
            return;
        }

        // After pop, a.seq.tokens has [.., last_token].
        // Observe: context ending with last_token → v0 is the correct next token
        proposer.observe(&a.seq.tokens, v0);

        emit_token(a, v0, None);
        if a.finished {
            return;
        }
        a.last_token = v0;

        // Propose next draft
        if let Some(draft) = proposer.propose(&a.seq.tokens) {
            a.pending_drafts = vec![draft];
        }

        tracing::info!(
            "NGRAM K2 REJECT: sync={sync_us}μs verify={verify_us}μs cache={} seq_len={}",
            proposer.len(),
            a.seq.seq_len,
        );
    }
}

/// Fill the XGrammar bitmask for the current matcher position and clone it
/// into an owned `Vec<i32>` the caller can pass into MTP draft sampling.
///
/// Returns `None` when grammar is inactive, the sequence is currently inside
/// a `<think>` span (matcher is paused), the grammar has already terminated,
/// or `fill_bitmask` reported no constraint. In all those cases MTP should
/// fall back to its unconstrained GPU-argmax path.
///
/// The owned copy is small (~ceil(vocab/32)*4 bytes, ~32KB for 100k vocab)
/// and is necessary because the matcher is borrowed mutably by the scheduler
/// between `fill_bitmask` and the subsequent `accept_token` calls inside
/// `emit_token`, while the MTP propose call borrows the model immutably —
/// cloning sidesteps the lifetime overlap.
pub fn mtp_grammar_mask_for(a: &mut ActiveSeq) -> Option<Vec<i32>> {
    if a.inside_thinking {
        return None;
    }
    let gs = a.grammar_state.as_mut()?;
    if gs.is_terminated() {
        return None;
    }
    if !gs.fill_bitmask() {
        return None;
    }
    Some(gs.bitmask_data().to_vec())
}

/// BUG#4 clamp, complete fix (2026-07-09): when a grammar is active, propose
/// only ONE draft. `run_mtp_propose_multi` masks every draft position with
/// the SAME position-0 bitmask snapshot (`mtp_head` warns "mask held fixed
/// across draft positions"), so draft\[1..\] is drafted against a stale mask —
/// grammar-illegal continuations get proposed, truncated at the boundary
/// (`truncate_drafts_at_grammar_boundary`), and acceptance collapses. The
/// original BUG#4 fix (2026-06-02) applied this clamp only in the Phase-A
/// bootstrap (`mtp_step.rs`); the five verify-path re-propose sites
/// (`verify_k2_step`, `verify_k3_step`) kept passing raw `num_drafts`, which
/// is why the warning spammed on every step after the first during grammar-
/// constrained tool calls (live opencode 42.5k session, 2026-07-09). SSOT
/// for all six propose sites — semantics identical to the bootstrap clamp
/// (`grammar_state.is_some()`). No-op when grammar is inactive: full K kept.
pub fn effective_drafts_under_grammar(a: &ActiveSeq, num_drafts: usize) -> usize {
    if a.grammar_state.is_some() {
        1
    } else {
        num_drafts
    }
}

/// Truncate a draft list at the first token the grammar would
/// reject *if it were the next emitted token at that draft position*.
///
/// Required for K=3+ MTP paths where `run_mtp_propose_multi` uses a
/// SINGLE bitmask snapshot (taken at the start of propose) for all N
/// drafts. The mask correctly constrains `drafts[0]` but does not
/// reflect the post-`drafts[0]` grammar state — so `drafts[1]` may
/// cross a structural boundary (e.g. `drafts[0] = </function>`
/// closing a tool body, then `drafts[1] = <parameter=` which is
/// invalid in the outer free-text grammar state).
///
/// Without this guard, the spec verifier accepts the cross-boundary
/// span (the model's actual sample matches whatever the in-tool
/// distribution happened to produce), `emit_token` advances the
/// grammar past `</function>`, and the next `accept_token` for
/// `drafts[1]` returns false silently — the token is already in
/// `output_tokens`, but the grammar is desync'd from the output
/// stream. Subsequent bitmasks are wrong.
///
/// Reference: arXiv:2512.15834 ("Speculative Tool Calls"). The
/// canonical fix is to re-run the grammar mask from a fresh outer
/// state for each draft; we approximate cheaply by simulating
/// `accept_token` per draft and truncating at the first rejection,
/// rolling the state back when done. The verifier then accepts at
/// most the validated prefix.
///
/// Returns the number of drafts that pass grammar validation.
/// Mutates `gs` transiently but restores it via `rollback`. K=2
/// (num_drafts=1) callers can skip this — a single draft uses its
/// own up-to-date mask.
pub fn truncate_drafts_at_grammar_boundary(gs: &mut GrammarState, drafts: &[u32]) -> usize {
    if drafts.len() < 2 || gs.is_terminated() {
        return drafts.len();
    }
    // BUG#3 (2026-06-02): roll back ACTUAL matcher advances (history delta), not
    // the `accepted` tally. `accept_token` returns true for stop/EOS tokens and
    // in the terminated state WITHOUT advancing the matcher; counting rollback
    // from `accepted` over-rewinds when such a token is in the draft span
    // (corrupt state / rollback panic). `accepted` still drives truncation.
    let steps_before = gs.num_history_steps();
    let mut accepted = 0usize;
    for &tok in drafts {
        if !gs.accept_token(tok) {
            break;
        }
        accepted += 1;
    }
    let advanced = gs.num_history_steps().saturating_sub(steps_before);
    if advanced > 0 {
        gs.rollback(advanced);
    }
    if accepted < drafts.len() {
        tracing::warn!(
            kept = accepted,
            dropped = drafts.len() - accepted,
            "spec-decode boundary: truncated drafts crossing grammar transition"
        );
    }
    accepted
}

#[cfg(test)]
mod internal_spec_tests {
    use super::*;

    #[test]
    fn spec_uniforms_reproduce_per_seed_and_differ_per_use() {
        let draw = |seed| (0..40).map(|i| spec_uniform(seed, i, SALT_DRAFT)).collect::<Vec<_>>();
        let a = draw(Some(7));
        assert_eq!(a, draw(Some(7)), "same seed, same stream");
        assert_ne!(a, draw(Some(8)), "another seed, another stream");
        assert!(a.iter().all(|&u| (0.0..1.0).contains(&u)));
        // Positions and salts give distinct uniforms (no reuse across draft / accept / next).
        let mut all: Vec<u64> = Vec::new();
        for pos in 0..40 {
            for salt in [SALT_DRAFT, SALT_ACCEPT, SALT_NEXT] {
                all.push(spec_uniform(Some(7), pos, salt).to_bits());
            }
        }
        let n = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), n, "every (position, salt) draws its own uniform");
        // Unseeded draws are not constant.
        let u: Vec<f64> = (0..8).map(|_| spec_uniform(None, 0, SALT_DRAFT)).collect();
        assert!(u.windows(2).any(|w| w[0] != w[1]));
        // Mean of 20k seeded draws ~ 0.5 (a constant stream would fail this).
        let m: f64 = (0..20_000).map(|i| spec_uniform(Some(3), i, SALT_ACCEPT)).sum::<f64>() / 20_000.0;
        assert!((m - 0.5).abs() < 0.01, "mean {m}");
    }
}
