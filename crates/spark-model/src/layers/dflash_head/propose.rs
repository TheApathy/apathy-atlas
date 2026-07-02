// SPDX-License-Identifier: AGPL-3.0-only

//! `DraftProposer::propose` body for [`super::BlockDiffusionDraftHead`].
//!
//! Split out of `dflash_head.rs` for file-size budget. Trait impl
//! delegates to [`BlockDiffusionDraftHead::propose_drafts`].

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::{BlockDiffusionDraftHead, DflashProposerState};
use crate::layer::ForwardContext;
use crate::speculative::ProposerState;

impl BlockDiffusionDraftHead {
    pub(super) fn propose_drafts(
        &self,
        last_token: u32,
        _target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        _stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        _grammar_bitmask: Option<&[i32]>,
        target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let dstate = state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;

        // ── Phase 2.5b kernel-chain scaffold (commented for next-session
        // fill-in; current path falls through to empty-Vec stub below) ──
        //
        // Reference: `dflash.py` (in the drafter's HF snapshot) lines 60-95
        // for the per-layer attention pattern. Per-layer flow (one call into
        // Atlas's existing op wrappers per bullet):
        //
        // For each layer in `self.layers`:
        //   ops::rms_norm(self.kernels.rms_norm, stream_buf, layer.input_layernorm,
        //                 norm_buf, gamma, hidden_size, eps)
        //   ops::dense_gemm(self.kernels.dense_gemm, norm_buf, layer.q_proj.weight,
        //                   q_buf, gamma, q_dim, hidden_size)        // [γ, 32*128]
        //   ops::dense_gemm(self.kernels.dense_gemm, norm_buf, layer.k_proj.weight,
        //                   k_buf, gamma, kv_dim, hidden_size)        // [γ, 4*128]
        //   ops::dense_gemm(self.kernels.dense_gemm, norm_buf, layer.v_proj.weight,
        //                   v_buf, gamma, kv_dim, hidden_size)        // [γ, 4*128]
        //   per-head q_norm: ops::rms_norm over each [γ, head_dim] slice
        //   per-head k_norm: ops::rms_norm over each [γ, head_dim] slice
        //   ops::rope_yarn(self.kernels.rope_qwen3, q_buf, k_buf, position_ids,
        //                  gamma, num_q_heads, num_kv_heads, head_dim, rotary_dim,
        //                  inv_freq, theta)
        //   ops::prefill_attention(prefill_attn_kernel, q_buf, k_buf, v_buf,
        //                          attn_out, gamma, 1, num_q_heads, num_kv_heads,
        //                          head_dim, inv_sqrt_d, /* causal = */ false,
        //                          /* sliding_window = */ 0)
        //   ops::dense_gemm(dense_gemm, attn_out, layer.o_proj.weight,
        //                   stream_buf_acc, gamma, hidden_size, q_dim)
        //   ops::residual_add(self.kernels.residual_add, stream_buf, stream_buf_acc,
        //                     stream_buf, gamma * hidden_size)
        //   ops::rms_norm(self.kernels.rms_norm, stream_buf, layer.post_attn_norm,
        //                 norm_buf, gamma, hidden_size, eps)
        //   ops::dense_gemm(dense_gemm, norm_buf, layer.gate_proj.weight,
        //                   gate_out, gamma, intermediate_size, hidden_size)
        //   ops::dense_gemm(dense_gemm, norm_buf, layer.up_proj.weight,
        //                   up_out, gamma, intermediate_size, hidden_size)
        //   ops::silu_mul(self.kernels.silu_mul, gate_out, up_out, mlp_intermediate,
        //                 gamma * intermediate_size)
        //   ops::dense_gemm(dense_gemm, mlp_intermediate, layer.down_proj.weight,
        //                   stream_buf_acc, gamma, hidden_size, intermediate_size)
        //   ops::residual_add(stream_buf, stream_buf_acc, stream_buf,
        //                     gamma * hidden_size)
        //
        // After the layer loop:
        //   ops::rms_norm(rms_norm, stream_buf, self.norm, norm_buf, gamma,
        //                 hidden_size, eps)
        //   ops::dense_gemm(dense_gemm, norm_buf, self.lm_head_shared, logits,
        //                   gamma, vocab_size, hidden_size)
        //   ops::argmax_bf16(self.kernels.argmax, logits, draft_tokens_dev,
        //                    gamma, vocab_size)
        //   gpu.copy_d2h(draft_tokens_dev, &mut host_buf, gamma * 4)
        //   parse host_buf as [u32; γ]
        //
        // Required additional state on the head (not yet allocated):
        //   - position_ids: [γ] u32 device buffer (positions = state.seq_len..+γ)
        //   - inv_freq: [head_dim/2] f32 yarn-scaled frequencies (pre-computed
        //     from drafter's rope_scaling: factor=64, beta_fast=32, beta_slow=1,
        //     original_max_position_embeddings=4096)
        //   - per-rms-norm eps from drafter config (Qwen3 default 1e-6)
        //
        // Open design questions for ctx-conditioned drafting (later iter):
        //   1. ctx_len = ? — vLLM accumulates per-token captures across all
        //      decoded positions; Atlas currently captures only the latest
        //      step's 5 hiddens (model-level single slot). Per-sequence
        //      accumulator needs to land in DflashProposerState.
        //   2. Asymmetric q_len (γ) vs k_len (γ + ctx_len) — either pad q
        //      with a dummy row or use the paged attention with a 1-block
        //      scratch cache for ctx K/V.
        //   3. RoPE position offsets — ctx K positions map to the prior
        //      decoded positions; q/noise K positions map to seq_len..+γ.

        let _ = (ctx, position, last_token);

        // Phase 2.5 stub. Real propose() implementation roadmap:
        //
        // ── Step 0: validate inputs ──
        // - target_hidden_stack must be Some(ptr) — shape [N, target_hidden]
        //   BF16 where N = self.target_layer_ids.len() (5 for Qwen3.6-DFlash).
        // - dstate.prefill_done must be true OR this is the first call after
        //   target prefill (in which case run precompute_and_store_context_kv
        //   to populate drafter KV cache from the prompt-time captures).
        //
        // ── Step 1: project current target hiddens through `fc` ──
        // - Input:  target_hidden_stack: [N * target_hidden] BF16 = [10240]
        // - Op:     dense_gemv_bf16(fc, in)         → [draft_hidden] = [2048]
        // - Op:     rms_norm(hidden_norm)           → [2048] BF16
        // - Op:     reshape_and_cache(K, V at slot dstate.seq_len) into the
        //           drafter's first layer's paged KV cache (this represents
        //           ONE token of context, written through layer 0's K/V proj
        //           → RoPE → cache slot at logical position dstate.seq_len).
        // - Note:   vLLM's `precompute_and_store_context_kv` does this for
        //           the *full* prompt prefix on the first call, and one
        //           token per step thereafter. We follow the same pattern.
        //
        // ── Step 2: build γ-token query input ──
        // - Allocate [γ, draft_hidden] scratch buffer.
        // - Embed ALL γ tokens as `mask_token_id` via shared embed_tokens_shared.
        //   The drafter is trained with mask_token_id for every noise position;
        //   context conditioning comes entirely from target_hidden (fc projection),
        //   not from embedding last_token into the noise block.
        // - Add the projected fc context to position 0 (Qwen3-DFlash
        //   `combine_hidden_states` semantics — verify against vLLM
        //   `qwen3_dflash.py:DFlashQwen3Model.forward`).
        //
        // ── Step 3: run γ tokens through 8 drafter layers ──
        // For each layer i in 0..self.num_layers:
        //   a. input_layernorm.rms_norm(input → x_norm)
        //   b. q_proj.gemm(x_norm → q [γ, num_q_heads * head_dim])
        //      k_proj.gemm(x_norm → k [γ, num_kv_heads * head_dim])
        //      v_proj.gemm(x_norm → v [γ, num_kv_heads * head_dim])
        //   c. q_norm.rms_norm per-head, k_norm.rms_norm per-head
        //   d. rope(q, k, position+0..γ-1)
        //   e. reshape_and_cache(k, v) into layer i's paged FP8 cache at
        //      slot positions [dstate.seq_len + 1 .. + γ]
        //   f. ops::prefill_attention_paged_fp8_dflash(...) — γ queries,
        //      bidirectional in-block + full prefix attention. Optional
        //      sliding window via self.window_size.
        //   g. o_proj.gemm(attn_out → o)
        //   h. residual_add(input, o)
        //   i. post_attention_layernorm.rms_norm
        //   j. gate_proj+up_proj+silu_mul+down_proj  (Qwen3 SwiGLU)
        //   k. residual_add
        //
        // ── Step 4: final RMSNorm + LM head ──
        // - self.norm.rms_norm
        // - dense_gemm(lm_head_shared) → [γ, vocab_size]
        // - argmax per row → γ candidate token IDs (DEVICE)
        // - copy_d2h γ × 4 bytes
        //
        // ── Step 5: state update ──
        // - dstate.seq_len += γ + 1   (drafter cache now holds prefix + γ + 1)
        //   note: the +1 is for the bonus-token slot we just appended in Step 1
        // - dstate.last_num_drafted = γ
        //
        // ── Required kernel handles (resolved via ctx.gpu.kernel(...)) ──
        // rms_norm, dense_gemv_bf16, dense_gemm_bf16, rope_qwen3_yarn,
        // reshape_and_cache_fp8, prefill_attention_paged_fp8_dflash,
        // silu_mul, residual_add, argmax_bf16, batched_embed
        //
        // Phase 2.5b first-iteration impl.
        //
        // Runs the γ-block forward chain (8 layers × Qwen3-decoder,
        // non-causal self-attention) through `forward_block`. Returns
        // **only the first draft** (1 token) so the scheduler routes
        // through the proven `step_verify_k2` path which already handles
        // SSM state rollback via the K=2 graphed verify kernel's
        // populated intermediates.
        //
        // Why cap at 1? Atlas's K=γ eager verify path (`decode_verify`)
        // does NOT populate `h_state_intermediates` — those are only
        // written by the K=2/3/4 specialized GDN kernels. So a γ-token
        // verify with partial accept (the typical case) produces garbage
        // SSM state on hybrid models like Qwen3.6-A3B (30 GDN layers).
        // Capping at 1 makes drafts.len()=1, scheduler picks K=2 verify
        // which DOES populate intermediates, SSM rollback works correctly.
        //
        // This loses the γ-parallel speedup (DFlash's main advantage)
        // but produces correct output with acceptance >0 — strict
        // improvement over no-spec when drafts match. The full γ-parallel
        // path needs either:
        //   (a) a K=γ specialized GDN verify kernel that populates
        //       intermediates per position (multi-week kernel work), or
        //   (b) restricting DFlash to pure-attention targets (Gemma-4,
        //       MiniMax-M2 dense) where SSM rollback isn't needed.
        //
        // Quality note: drafter runs WITHOUT context conditioning
        // (`ctx_len=0`) — it was trained with 5×target_hidden ctx, so
        // first-token acceptance will be poor (<<70%). Adding ctx is the
        // next iteration on top of `forward_block`.
        let _ = num_drafts;

        // Append the model's latest single-slot ctx capture into the
        // per-seq accumulator. Skip when `target_hidden_stack` is None
        // (e.g. EP=2 worker rank or the very first call before any
        // capture has fired). Capping at `max_ctx_len` to keep within
        // allocated bounds — drafter quality plateaus past a few hundred
        // ctx positions anyway.
        //
        // ATLAS_DFLASH_DEBUG_NO_DECODE_APPEND=1 disables the post-decode
        // append. The captured target_hidden_stack is the K-1 token of
        // the last K=2 verify (the draft, NOT the bonus). On REJECT
        // (the typical case during cold-start training-distribution
        // mismatch) the draft was never accepted, so appending its
        // hiddens to the accumulator poisons the ctx for subsequent
        // propose() calls. Setting this flag uses ONLY prefill captures
        // — clean ctx isolation for diagnosing real-traffic acceptance.
        let skip_decode_append = std::env::var("ATLAS_DFLASH_DEBUG_NO_DECODE_APPEND")
            .ok()
            .as_deref()
            == Some("1");
        // Drop the `first_propose_done` gate. The previous logic skipped
        // the append on the very first propose call, assuming
        // dflash_hidden_save was uninitialized. But after a regular
        // bootstrap decode (mtp_step's Phase A), `decode_a.rs` has
        // already populated dflash_hidden_save[0] with the
        // bootstrap-decoded token's hidden at sequence position M (=
        // seq_len after prefill, BEFORE bootstrap increment). Skipping
        // the append left ctx_len = M (prefill captures) while
        // seq.seq_len = M+1, so the drafter's RoPE positions for ctx
        // were assigned [1..M] instead of [0..M-1] — an off-by-one
        // that shifted every ctx position by 1 RoPE rotation. Every
        // subsequent step inherited the off-by-one (ctx_len fell one
        // behind seq.seq_len). Result: drafter accept rate collapsed
        // to ~1% because attention K/V were rotated to the wrong
        // positions. Verified by reading position=5, eff_ctx=4 from
        // the very first propose dump.
        //
        // Fix: always append on every propose. On first propose,
        // last_num_accepted=0 → num_append=1 → appends the bootstrap
        // hidden, ctx_len becomes M+1 = seq.seq_len, RoPE positions
        // align. On subsequent proposes, the existing logic
        // (last_num_accepted+1 slots) keeps ctx_len in lockstep with
        // seq.seq_len.
        if !skip_decode_append
            && let Some(base) = target_hidden_stack
            && dstate.ctx_len < dstate.max_ctx_len
        {
            // Append the new tokens' hidden states from the previous
            // verify step.
            //
            // dflash_hidden_save layout (set during verify of
            // [last_token, draft_0, ..., draft_{γ-1}]):
            //   [0]   = hidden of verify input position 0 = last_token
            //   [1..] = hidden of draft_0, draft_1, ...
            //
            // ctx_hidden_acc semantics: slot i = hidden of token at
            // sequence position i. ctx_start in forward_block.rs is
            // computed as `position - eff_ctx`, so position i in ctx
            // corresponds to absolute position (position - eff_ctx + i).
            //
            // After step N's verify (last_token at sequence position P,
            // num_accepted=N accepted drafts, bonus emitted next):
            //   - decode_verify wrote last_token at KV position P,
            //     draft_0 at P+1, ..., draft_{N-1} at P+N
            //   - seq.seq_len = P + N + 1 after rollback
            //   - The bonus is logically at position P+N+1 but has no
            //     KV yet (will be written when next verify processes it
            //     as input position 0)
            //
            // For step N+1 propose:
            //   - position = P + N + 1
            //   - We need ctx slots P, P+1, ..., P+N populated with
            //     hiddens of (last_token, draft_0, ..., draft_{N-1})
            //   - dflash_hidden_save[0..N+1] has EXACTLY this in order
            //
            // So src_idx 0..N+1 is correct and matches sequence order.
            // The bonus's hidden is NOT needed in ctx — bonus appears
            // as the first noise embedding (Q-side input).
            let num_append = dstate.last_num_accepted + 1;
            // FIX 1 (ATLAS_DFLASH_TREE_COMMIT): when the previous verify
            // committed a tree-fork tail, the accepted hiddens are scattered
            // across `dflash_hidden_save` at the path's COMPACT slots (verify
            // row 0 = last_token, row c = compact index c), NOT the contiguous
            // 0..num_append. Build the per-append source row list: row 0 is
            // always the last_token capture; rows 1..num_append follow the
            // accepted compact indices. Empty path → contiguous (default).
            let src_rows: Vec<usize> = if dstate.last_accepted_compact.is_empty() {
                (0..num_append).collect()
            } else {
                let mut rows = Vec::with_capacity(num_append);
                rows.push(0);
                rows.extend_from_slice(&dstate.last_accepted_compact);
                rows
            };
            // dflash_hidden_save rows hold the hiddens of the
            // tokens at absolute positions (position - num_append)..position.
            // Write each row at its ABSOLUTE slot rather than appending at
            // ctx_len: steps that commit tokens without an append (no-spec
            // fallback when propose yields <4 drafts) otherwise desync the
            // lockstep counter and shift every later slot — measured as a
            // constant d=+2 ctx misalignment on prose that collapsed accept
            // from 1.38 to 0.31 per block (probe_states_ab.py, 2026-06-12).
            // A skipped step now costs one stale slot, not a permanent shift.
            let first_pos = position.saturating_sub(num_append);
            if dstate.ctx_len != first_pos {
                tracing::warn!(
                    "DFlash ctx drift: ctx_len={} expected {} (position={}, num_append={}) — realigning by absolute slot",
                    dstate.ctx_len,
                    first_pos,
                    position,
                    num_append,
                );
            }
            tracing::info!(
                "DFlash propose append: last_num_accepted={} num_append={} first_pos={} ctx_len_before={}",
                dstate.last_num_accepted,
                num_append,
                first_pos,
                dstate.ctx_len,
            );
            for i in 0..num_append {
                let slot = first_pos + i;
                if slot >= dstate.max_ctx_len {
                    break; // accumulator full; drop later positions
                }
                // Source verify-capture row: contiguous `i` on the flat path,
                // else the sparse fork path's compact slot (src_rows[i]).
                let src_row = src_rows.get(i).copied().unwrap_or(i);
                let src = base.offset(src_row * dstate.ctx_slot_bytes);
                let dst = dstate.ctx_hidden_acc.offset(slot * dstate.ctx_slot_bytes);
                ctx.gpu.copy_d2d_async(src, dst, dstate.ctx_slot_bytes, _stream)?;
            }
            dstate.ctx_len = dstate
                .ctx_len
                .max((first_pos + num_append).min(dstate.max_ctx_len));
        }

        // ATLAS_DFLASH_PLD=1: prompt-lookup drafting. If the trailing n-gram
        // (ATLAS_PLD_NGRAM, default 3) recurs earlier in the committed
        // sequence, draft the gamma tokens that followed that occurrence and
        // skip the drafter forward entirely. Highly repetitive text (story
        // names/phrases, code idioms) accepts these at high rates; misses are
        // rejected by the verifier as usual. Draft count stays gamma so the
        // K=17 verify path is unchanged. Ctx bookkeeping above already ran.
        // Precision gates (naive 3-gram PLD regressed coding 53.7->37: short
        // matches preempt a strong drafter with wrong continuations): only
        // fire in the weak-drafter regime (previous step accepted <=1, never
        // true on coding/counting at 8-16 accepts) and require a long suffix
        // match (8 down to ATLAS_PLD_NGRAM, default 5; longest wins).
        if std::env::var("ATLAS_DFLASH_PLD").ok().as_deref() == Some("1")
            && dstate.first_propose_done
            && dstate.last_num_accepted <= 1
        {
            let ng_min: usize = std::env::var("ATLAS_PLD_NGRAM")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5);
            let need = 16usize;
            let toks = &dstate.pld_tokens;
            let l = toks.len();
            let mut hit: Option<usize> = None;
            for ng in (ng_min..=8).rev() {
                if l <= ng + need {
                    continue;
                }
                let mut suffix: Vec<u32> = toks[l - (ng - 1)..].to_vec();
                suffix.push(last_token);
                let mut p = l.saturating_sub(ng + 1);
                loop {
                    if toks[p..p + ng] == suffix[..] && p + ng + need <= l {
                        hit = Some(p + ng);
                        break;
                    }
                    if p == 0 {
                        break;
                    }
                    p -= 1;
                }
                if hit.is_some() {
                    break;
                }
            }
            if let Some(cs) = hit {
                let drafts: Vec<u32> = toks[cs..cs + need].to_vec();
                dstate.last_num_drafted = drafts.len();
                dstate.first_propose_done = true;
                return Ok(drafts);
            }
        }

        // ── ATLAS_DFLASH_RETRIEVAL=1: retrieval-augmented drafting (default off) ──
        //
        // Generalization of the PLD path above. Searches a BROADER haystack
        // (`dstate.pld_tokens` = prompt + generated, populated by the caller
        // when this flag is on) with a longest-suffix match over L_max..L_min,
        // and proposes the γ tokens that followed the longest occurrence.
        // Unlike PLD, it fires whenever a strong match exists — NOT only in
        // the weak-drafter regime — because the DFlash verify is a lossless
        // oracle: it commits only the target's greedy token and accepts a
        // draft solely when draft==greedy. A wrong retrieval guess therefore
        // costs only a rejected speculation and can never change committed
        // output (token-exact by construction; proven by greedy byte-match).
        //
        // Cheap hybrid (implemented here): pre-empt the neural drafter ONLY
        // when the match is strong (match_len >= hybrid_min, default = L_max).
        // Otherwise fall through to the drafter forward below. Keeping draft
        // count = γ leaves the K=γ verify CUDA-graph path unchanged.
        //
        // ATLAS_RETRIEVAL_LMAX (16), ATLAS_RETRIEVAL_LMIN (4),
        // ATLAS_RETRIEVAL_HYBRID_MIN (=LMAX) tune the gates.
        // Match the drafter's effective draft count exactly: forward_block
        // shrinks the noise block to γ_eff (= DRAFT_CAP clamped to [1, γ]).
        // Proposing γ_eff drafts keeps drafts.len() identical to the drafter
        // path so the K=γ_eff+1 verify dispatch is unchanged regardless of
        // which source fired. The serve script pins DRAFT_CAP=γ=16 (K=17).
        let retrieval_gamma_eff = std::env::var("ATLAS_DFLASH_DRAFT_CAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(self.gamma)
            .min(self.gamma)
            .max(1);
        if let Some(rcfg) = super::retrieval::RetrievalConfig::from_env(retrieval_gamma_eff)
            && dstate.first_propose_done
        {
            // ── Adaptive retrieval gate (ATLAS_DFLASH_SAM_ADAPTIVE, default ON) ──
            // Attribute the previous step's accept to retrieval (when it fired)
            // and back off after sustained misfires, so SAM auto-disables on
            // content where strong suffix matches mis-predict (counting: digit
            // runs match but the next number is always new → wasted drafts,
            // measured 77→65 tok/s regression) while staying fully active on
            // reuse-heavy editing (where retrieval keeps accepting). LOSSLESS:
            // only changes WHETHER we retrieve; the verify still commits the
            // target's greedy token regardless.
            let adaptive =
                std::env::var("ATLAS_DFLASH_SAM_ADAPTIVE").ok().as_deref() != Some("0");
            if adaptive {
                const MIN_ACCEPT: usize = 3; // retrieval step below this = misfire
                const MISFIRE_LIMIT: u32 = 3; // consecutive misfires → cooldown
                const COOLDOWN: u32 = 24; // steps to skip retrieval, then retry
                if dstate.retr_used_last {
                    if dstate.last_num_accepted < MIN_ACCEPT {
                        dstate.retr_misfire_streak += 1;
                    } else {
                        dstate.retr_misfire_streak = 0;
                    }
                }
                dstate.retr_used_last = false;
                if dstate.retr_cooldown == 0 && dstate.retr_misfire_streak >= MISFIRE_LIMIT {
                    dstate.retr_cooldown = COOLDOWN;
                    dstate.retr_misfire_streak = 0;
                }
            }
            let retr_suppressed = adaptive && dstate.retr_cooldown > 0;
            if dstate.retr_cooldown > 0 {
                dstate.retr_cooldown -= 1;
            }
            // SAM mode (ATLAS_DFLASH_SAM=1): longest-suffix match at ANY length
            // via retrieve_longest. Else the legacy fixed-window range matcher.
            // Suppressed ⇒ None ⇒ falls through to the neural drafter below.
            let lookup = if retr_suppressed {
                None
            } else if rcfg.sam {
                super::retrieval::retrieve_longest(&dstate.pld_tokens, last_token, &rcfg)
            } else {
                super::retrieval::retrieve(&dstate.pld_tokens, last_token, &rcfg)
            };
            if let Some(hit) = lookup
                && hit.match_len >= rcfg.hybrid_min
                && hit.drafts.len() == retrieval_gamma_eff
            {
                static RETR_DBG_DONE: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !RETR_DBG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::info!(
                        "DFlash retrieval: first hit match_len={} draft_count={} haystack_len={} (lmax={} lmin={} hybrid_min={})",
                        hit.match_len,
                        hit.drafts.len(),
                        dstate.pld_tokens.len(),
                        rcfg.l_max,
                        rcfg.l_min,
                        rcfg.hybrid_min,
                    );
                }
                tracing::debug!(
                    "DFlash retrieval hit: match_len={} proposing {} drafts (skip drafter)",
                    hit.match_len,
                    hit.drafts.len()
                );
                dstate.last_num_drafted = hit.drafts.len();
                dstate.first_propose_done = true;
                // Adaptive gate: mark that retrieval fired so the NEXT propose
                // attributes this step's accept to retrieval (misfire tracking).
                dstate.retr_used_last = true;
                // Flat chain → requires_tree_kernel false → wy17 verify path,
                // identical dispatch to drafter output. No tree payload.
                dstate.pending_tree_payload = None;
                return Ok(hit.drafts);
            }
        }

        // ── ATLAS_DFLASH_RECYCLE=1: recycle the discarded draft tail (default off) ──
        //
        // The linear γ chain discards every correct draft token DOWNSTREAM of
        // the first content miss (a measured 46.9% of correct drafts). Recover
        // them: after verify, `dflash_stash_recycle` stashed the rejected tail
        // `drafts[num_accepted+1..γ_eff]` keyed by the corrected token the
        // target committed at the miss (= this step's `last_token`). When that
        // key matches, OFFER the stashed tail as the draft instead of running
        // the neural drafter — the structural part (indentation, closing
        // brackets, `):`) is usually still correct given the corrected token,
        // so it re-accepts; content-dependent parts get rejected for free.
        //
        // LOSSLESS by construction (identical contract to the retrieval path
        // above): the recycled tokens are only PROPOSED. The DFlash verify is
        // the oracle — it commits only the target's greedy token and accepts a
        // draft solely when draft==greedy. A wrong recycle costs one rejected
        // speculation and can NEVER change committed output (proven by greedy
        // byte-match ON vs OFF).
        //
        // PRECEDENCE: secondary to PLD/RETRIEVAL. If either of those already
        // returned above, recycle never runs this step (they pre-empt the
        // drafter on a strong match; recycle is the fallback when no such match
        // exists). Recycle is consumed (cleared) whether or not it fires so a
        // stale tail is never re-offered after its key stops matching.
        //
        // PRECISION GATE (critical — mirrors the PLD gate at line ~336):
        // recycle pre-empts the NEURAL DRAFTER. On high-accept content
        // (counting, structured code: ~15/16) the drafter is excellent and a
        // recycled tail — conditioned on the OLD pre-correction context and
        // padded with a repeat — is far worse, collapsing accept to ~1/16 and
        // ballooning step count. So only fire recycle when the drafter is
        // already WEAK this step (`last_num_accepted <= ATLAS_DFLASH_RECYCLE_MAX_ACCEPT`,
        // default 1) — exactly the regime (novel prose/code) where the tail
        // recovery in the research targets the discarded 46.9%, and where
        // there is nothing to lose because the neural drafter is failing too.
        // Without this gate, ON measured ~8x SLOWER on counting (82→9.6 tok/s)
        // because recycle fired every step and replaced 15/16 drafter accepts
        // with ~1/16 stale-tail accepts. Lossless either way (verify is the
        // oracle); the gate is purely a THROUGHPUT precision filter.
        //
        // The offered tail is a FLAT CHAIN of length `recycle_gamma_eff`
        // (= γ_eff, same as the drafter / retrieval paths), so the K=γ_eff+1
        // verify dispatch is UNCHANGED (→ wy17, no tree). When the stashed
        // tail is shorter than γ_eff it is padded with the corrected token's
        // mask-equivalent neutral fill (last token repeated) so drafts.len()
        // stays at γ_eff and the verify graph is never re-captured.
        // ANTI-TRAP: never fire recycle two steps in a row. After any offer the
        // next step MUST run the real drafter so the true accept signal is
        // re-established (otherwise a poorly-re-accepting tail keeps
        // last_num_accepted low and re-opens the gate forever — measured
        // counting 82→9 tok/s). `recycle_last_offered` is set when we offer and
        // cleared on the drafter path below.
        if dstate.recycle_last_offered {
            dstate.recycle_last_offered = false;
        } else if std::env::var("ATLAS_DFLASH_RECYCLE").ok().as_deref() == Some("1")
            && dstate.first_propose_done
            && dstate.last_num_accepted
                <= std::env::var("ATLAS_DFLASH_RECYCLE_MAX_ACCEPT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1)
        {
            let recycle_gamma_eff = std::env::var("ATLAS_DFLASH_DRAFT_CAP")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(self.gamma)
                .min(self.gamma)
                .max(1);
            // Take the stash unconditionally (single-use): a tail is only valid
            // for the immediately-following step whose last_token == the key.
            let valid = dstate.recycle_valid;
            let key = dstate.recycle_key;
            let tail = std::mem::take(&mut dstate.recycle_tail);
            dstate.recycle_valid = false;
            if valid && key == last_token && !tail.is_empty() {
                // Build the offered drafts: the recycled tail, truncated or
                // padded to exactly recycle_gamma_eff so the verify dispatch
                // (K = γ_eff + 1) is identical to the drafter path. Pad with
                // the last tail token (a harmless repeat — verify rejects it
                // if wrong, at zero output cost).
                let mut drafts: Vec<u32> = Vec::with_capacity(recycle_gamma_eff);
                for i in 0..recycle_gamma_eff {
                    let t = if i < tail.len() {
                        tail[i]
                    } else {
                        *tail.last().unwrap()
                    };
                    drafts.push(t);
                }
                static RECYCLE_DBG_DONE: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !RECYCLE_DBG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::info!(
                        "DFlash recycle: first offer key={} tail_len={} γ_eff={} (offering recycled tail, skip drafter)",
                        key,
                        tail.len(),
                        recycle_gamma_eff,
                    );
                }
                tracing::debug!(
                    "DFlash recycle hit: key={} tail_len={} offering {} drafts (skip drafter)",
                    key,
                    tail.len(),
                    drafts.len()
                );
                dstate.last_num_drafted = drafts.len();
                dstate.first_propose_done = true;
                // Mark so the NEXT step skips recycle and runs the drafter
                // (anti-trap — see the gate above).
                dstate.recycle_last_offered = true;
                // Flat chain → requires_tree_kernel false → wy17 verify path,
                // identical dispatch to drafter output. No tree payload.
                dstate.pending_tree_payload = None;
                return Ok(drafts);
            }
        }

        // ── ATLAS_DFLASH_ACCEPT_FALLBACK (default off) ──
        //
        // The γ=16 draft+verify cycle is a net LOSS on low-acceptance (novel)
        // content: the expensive K=16 propose+verify emits only a few tokens,
        // so the per-step wall exceeds what plain single-token decode would
        // cost. This gate detects that regime per-sequence from the rolling
        // accept window and, when accept is low, SUPPRESSES speculation by
        // returning an empty draft vector. An empty `pending_drafts` routes
        // the sequence through the scheduler's bootstrap plain-decode path
        // (mtp_step.rs Phase A) on the next step — recovering plain-decode
        // throughput. Suppression lasts `COOLDOWN` steps, after which one
        // full-γ PROBE runs to re-measure acceptance: if the content has
        // turned predictable (counting, repeated structure) the probe accepts
        // well and we resume full speculation; if still novel, we re-suppress.
        //
        // This is intentionally a TWO-MODE switch (full-γ spec graph, or plain
        // decode) — never a variable γ — so the K=γ verify CUDA graph is never
        // re-captured. Variable-γ adaptive shrink (ATLAS_DFLASH_ADAPTIVE_GAMMA)
        // both wrecked the graph cache AND shrank γ on high-accept counting;
        // this gate avoids both failure modes.
        //
        //   ATLAS_DFLASH_ACCEPT_FALLBACK=1     enable (default off ⇒ identical
        //                                      to legacy behavior, byte-for-byte)
        //   ATLAS_DFLASH_FALLBACK_THRESH=<n>   mean-accept threshold over the
        //                                      window; below it ⇒ suppress.
        //                                      Default 6 (of γ=16).
        //   ATLAS_DFLASH_FALLBACK_COOLDOWN=<n> plain-decode steps to stay
        //                                      suppressed before the next probe.
        //                                      Default 8.
        if std::env::var("ATLAS_DFLASH_ACCEPT_FALLBACK").ok().as_deref() == Some("1") {
            let thresh: usize = std::env::var("ATLAS_DFLASH_FALLBACK_THRESH")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(6);
            let cooldown: usize = std::env::var("ATLAS_DFLASH_FALLBACK_COOLDOWN")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(8)
                .max(1);
            // Already in a suppression window: stay on plain decode, count down.
            if dstate.fallback_suppressed_remaining > 0 {
                dstate.fallback_suppressed_remaining -= 1;
                dstate.last_num_drafted = 0;
                dstate.pending_tree_payload = None;
                tracing::debug!(
                    "DFlash accept-fallback: suppressed (remaining={})",
                    dstate.fallback_suppressed_remaining
                );
                return Ok(Vec::new());
            }
            // Not suppressed. If we have a stable accept signal and it's low,
            // ENTER suppression now (this step also routes to plain decode).
            // Require a full window of >=4 verifies so warmup doesn't trip it.
            if dstate.accept_history_count >= 4 {
                let n = dstate
                    .accept_history_count
                    .min(dstate.accept_history.len());
                let sum: usize = dstate
                    .accept_history
                    .iter()
                    .take(n)
                    .map(|&v| v as usize)
                    .sum();
                let mean = sum / n;
                if mean < thresh {
                    dstate.fallback_suppressed_remaining = cooldown;
                    dstate.last_num_drafted = 0;
                    dstate.pending_tree_payload = None;
                    tracing::debug!(
                        "DFlash accept-fallback: entering suppression (mean_accept={} < thresh={}, cooldown={})",
                        mean, thresh, cooldown
                    );
                    return Ok(Vec::new());
                }
            }
            // mean >= thresh (or warming up) ⇒ fall through to full-γ propose.
        }

        let drafts = self
            .forward_block(last_token, position, ctx, _stream, dstate)
            .map_err(|e| {
                tracing::warn!("DFlash forward_block failed, falling back to no-spec: {e:#}");
                e
            })?;
        // Phase 2.5e: K=γ verify path is implemented in model.rs
        // (decode_verify_graphed_kgamma → decode_verify) and dispatched via
        // step_verify_dflash when drafts.len()>=4. The SSM intermediate
        // checkpoint/rollback path is fully wired:
        //   - SSM pool allocates 17 intermediate slots (γ+1) at model init
        //   - decode_batched → decode_batched_conv_gdn saves per-token
        //     h_state and conv_state intermediates (WY17 fused path or
        //     sequential fallback, both write to pool addresses)
        //   - commit_verify_state_async reads intermediates[num_accepted-1]
        //     from the pool for partial-accept rollback
        //
        // The earlier cap=1 default was a conservative workaround from early
        // development when SSM intermediate semantics were not yet verified.
        // Audit (2026-05-15) confirmed all paths produce correct
        // intermediates. Default raised to γ to enable full DFlash speedup.
        // Set ATLAS_DFLASH_DRAFT_CAP=1 to force K=2 verify if regression
        // is observed.
        // ATLAS_DFLASH_DRAFT_CAP is now consumed inside forward_block.rs
        // and shrinks the noise block to gamma_eff+1 rows — so `drafts`
        // already has length = gamma_eff. The legacy post-filter is removed
        // (was wasting drafter compute on tokens we'd then discard).
        dstate.last_num_drafted = drafts.len();
        dstate.first_propose_done = true;

        // ── ATLAS_DFLASH_BRANCH=1: entropy-gated top-2 cliff branch (default off) ──
        //
        // Coding acceptance dies because the linear γ=16 chain hits ONE
        // high-entropy content token (the "cliff") where the drafter's top-1 is
        // a coin-flip, the chain truncates there, and the whole predictable tail
        // after it is thrown away. Instead of committing only top-1 at that one
        // cliff, spend ONE of the 16 nodes on a SIBLING carrying the drafter's
        // top-2 token at the cliff row: a 2-way fork. The target verifies both
        // candidates and the greedy walk commits whichever it would have picked
        // — covering the cliff with two shots roughly doubles the chance of
        // clearing it. K is held at γ+1=17 (the CUDA graph shape): we shorten
        // the linear chain depth by 1 (drop the last chain node) and reuse that
        // node as the sibling, so drafts.len() (== node count) is UNCHANGED and
        // the K=17 verify graph is never re-captured.
        //
        // LOSSLESS by construction. The DFlash verify is a greedy oracle:
        // greedy_sample_ddtree walks the tree with the target's per-row argmax
        // and commits a draft token ONLY when draft == target_argmax; the bonus
        // is always the target's greedy token. The sibling merely gives the walk
        // a SECOND candidate to match at the cliff — it can never change which
        // token the target commits (proven by greedy byte-match ON vs OFF).
        //
        // GATE (the whole point): branch ONLY at a real cliff (lowest margin <
        // threshold). On confident / structural blocks (counting, prose,
        // boilerplate code) every row has a large top1−top2 margin, the gate
        // stays closed, the flat chain is emitted unchanged, and those workloads
        // do not regress. The threshold is tuned to fire on the bursty
        // high-entropy content token that code blocks usually contain exactly
        // one of per γ-block.
        // ── ATLAS_DFLASH_CATERPILLAR=1: depth-contiguous Sequoia-DP caterpillar
        //    (default off) ──
        //
        // The successor to ATLAS_DFLASH_BRANCH. The branch builder placed the
        // fork leaf at the LAST compact slot (slot ≫ depth), so committing the
        // fork tail required relocating KV whose RoPE/attention were baked at
        // the wrong slot — corrupting counting md5 (the `1..7 1..` reset).
        //
        // The caterpillar fixes this with a DEPTH-CONTIGUOUS layout
        // (`build_caterpillar_payload`): the top-1 chain is the spine at compact
        // slots 1..S, and the top-2 fork is a LEAF placed at compact slot ==
        // its tree depth (right after its parent), with the spine continuation
        // shifted one slot deeper. Then RoPE position == depth and the leaf's
        // slot-prefix == its ancestor chain, so tree-path-commit + KV-compaction
        // become byte-exact for the committed fork branch (see the builder's
        // module doc for the full invariant proof).
        //
        // Cliff gate: the margin gate below picks the SHALLOWEST low-margin row
        // (EAGLE-2: shallow accept ≫ deep). Under pure flat verify metadata only
        // the single shallowest fork is strictly lossless; deeper cliffs need
        // the depth-RoPE tree-aware verify path. So this builds a one-fork
        // depth-contiguous caterpillar per block and relies on the lossless
        // greedy oracle (verify commits only draft == target argmax) for safety.
        let caterpillar_enabled =
            std::env::var("ATLAS_DFLASH_CATERPILLAR").ok().as_deref() == Some("1");
        if caterpillar_enabled && drafts.len() >= 3 {
            let margin_thresh: f32 = std::env::var("ATLAS_DFLASH_BRANCH_MARGIN")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2.0);
            let n = drafts.len();
            match self.extract_topk_from_logits(ctx.gpu, _stream, n, 2) {
                Ok((topk_tokens, topk_logits)) if topk_logits.len() >= 2 * n => {
                    // Shallowest cliff: first row (depth) in [1, n) whose top1−top2
                    // margin drops below threshold. Row index r == spine depth of
                    // the node the fork attaches BELOW (slot r in compact frame).
                    let mut cliff = 0usize;
                    let mut cliff_margin = f32::INFINITY;
                    for r in 1..n - 1 {
                        let m = topk_logits[2 * r] - topk_logits[2 * r + 1];
                        if m < margin_thresh {
                            cliff = r;
                            cliff_margin = m;
                            break;
                        }
                    }
                    let fork_tok = if cliff >= 1 { topk_tokens[2 * cliff + 1] } else { 0 };
                    if cliff >= 1 && cliff_margin < margin_thresh && fork_tok != drafts[cliff] {
                        // build_caterpillar_payload takes the fork's tree DEPTH:
                        // the fork is a child of the spine node at depth `cliff`
                        // (compact slot `cliff`), so the leaf's depth = cliff + 1.
                        //
                        // ATLAS_DFLASH_CAT_TAIL=1: the EV variant — the top-2
                        // fork carries the post-cliff predictable tail (the
                        // linear drafts after the cliff, re-rooted on top-2) as
                        // a contiguous run; the top-1-at-cliff becomes the
                        // high-slot leaf. Gains when the target's greedy picks
                        // top-2 at the cliff and the tail re-accepts. Default
                        // (off) keeps the lossless single-leaf fork (no tail,
                        // no contamination), which proved no-gain at K=17.
                        let cat_tail =
                            std::env::var("ATLAS_DFLASH_CAT_TAIL").ok().as_deref()
                                == Some("1");
                        let payload = if cat_tail {
                            let tail: Vec<u32> = drafts[(cliff + 1).min(drafts.len())..].to_vec();
                            super::ddtree::build_caterpillar_tail_payload(
                                &drafts,
                                fork_tok,
                                cliff + 1,
                                &tail,
                            )
                        } else {
                            super::ddtree::build_caterpillar_payload(
                                &drafts,
                                fork_tok,
                                cliff + 1,
                            )
                        };
                        static CAT_DBG: std::sync::atomic::AtomicUsize =
                            std::sync::atomic::AtomicUsize::new(0);
                        let dbg = CAT_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if dbg < 8 {
                            tracing::info!(
                                "DFlash CATERPILLAR #{dbg}: n={n} cliff_depth={} margin={cliff_margin:.2} \
                                 fork_tok={fork_tok} spine_top1={} tokens[..min(8)]={:?} parents[..min(8)]={:?}",
                                cliff + 1,
                                drafts[cliff],
                                &payload.tree_token_ids[..payload.tree_token_ids.len().min(8)],
                                &payload.parent_indices[..payload.parent_indices.len().min(8)],
                            );
                        }
                        dstate.pending_tree_payload = Some(payload);
                        dstate.last_num_drafted = drafts.len();
                        return Ok(drafts);
                    }
                    // Gate closed → flat chain, identical to non-branch path.
                    dstate.pending_tree_payload = None;
                    return Ok(drafts);
                }
                Ok(_) => {
                    dstate.pending_tree_payload = None;
                    return Ok(drafts);
                }
                Err(e) => {
                    tracing::warn!(
                        "DFlash CATERPILLAR top-2 extraction failed ({e}); flat chain fallback"
                    );
                    dstate.pending_tree_payload = None;
                    return Ok(drafts);
                }
            }
        }

        let branch_enabled =
            std::env::var("ATLAS_DFLASH_BRANCH").ok().as_deref() == Some("1");
        if branch_enabled && drafts.len() >= 3 {
            // Margin threshold in raw-BF16-logit units (top tokens are O(10-30),
            // so a margin below ~2-4 means the drafter is genuinely unsure).
            let margin_thresh: f32 = std::env::var("ATLAS_DFLASH_BRANCH_MARGIN")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2.0);
            let n = drafts.len();
            // Extract per-row top-2 (token + logit), sorted descending. Row i
            // holds [top1_logit, top2_logit] at indices 2i, 2i+1 and the
            // matching tokens. Reuses the same scratch logits forward_block
            // just populated (survives until next propose).
            match self.extract_topk_from_logits(ctx.gpu, _stream, n, 2) {
                Ok((topk_tokens, topk_logits)) if topk_logits.len() >= 2 * n => {
                    // Cliff selection. Restrict to rows [1 .. n-1): row 0 is the
                    // token right after the bonus (its accept gates the whole
                    // chain anyway), and the last row is the node spent on the
                    // sibling. Two policies (ATLAS_DFLASH_BRANCH_CLIFF):
                    //   "first" (default): the FIRST row whose margin drops below
                    //     threshold. This is the row that ACTUALLY truncates the
                    //     chain — branching a later (lower-margin) row the chain
                    //     never reaches is useless. Critical: the global-min row
                    //     is typically deep (12-13), far past the ~row-3 mean
                    //     accept on coding, so it is never verified.
                    //   "min": the global lowest-margin row (diagnostic).
                    let cliff_first = std::env::var("ATLAS_DFLASH_BRANCH_CLIFF")
                        .ok()
                        .as_deref()
                        != Some("min");
                    let mut cliff = 0usize;
                    let mut min_margin = f32::INFINITY;
                    if cliff_first {
                        for r in 1..n - 1 {
                            let m = topk_logits[2 * r] - topk_logits[2 * r + 1];
                            if m < margin_thresh {
                                cliff = r;
                                min_margin = m;
                                break;
                            }
                        }
                    } else {
                        for r in 1..n - 1 {
                            let m = topk_logits[2 * r] - topk_logits[2 * r + 1];
                            if m < min_margin {
                                min_margin = m;
                                cliff = r;
                            }
                        }
                    }
                    let sib_token = topk_tokens[2 * cliff + 1]; // top-2 at cliff
                    // ATLAS_DFLASH_BRANCH_TAIL selects which fork carries the
                    // post-cliff predictable tail (only the tail laid on the
                    // CONTIGUOUS flat chain `[1,2,..]` is committed under the
                    // deployable flat-safe greedy contract — the other fork is a
                    // leaf reachable only as a single bonus):
                    //   "top1" (default): main flat chain stays on the drafter's
                    //     top-1 at the cliff (tail off top-1); the top-2 is a leaf
                    //     sibling. Lossless AND byte-stable, but cannot beat flat
                    //     for greedy (flat already emits the target's greedy at
                    //     the cliff as the free bonus) — a no-gain control.
                    //   "top2": the main flat chain takes the drafter's TOP-2 at
                    //     the cliff (tail off top-2, contiguous KV) and the top-1
                    //     becomes the leaf. Commits the post-cliff tail when the
                    //     target's greedy == top-2 at the low-margin cliff — the
                    //     EV bet that pays off at near-equiprobable cliffs.
                    let tail_top2 = std::env::var("ATLAS_DFLASH_BRANCH_TAIL")
                        .ok()
                        .as_deref()
                        == Some("top2");
                    if cliff >= 1 && min_margin < margin_thresh && sib_token != drafts[cliff] {
                        // Build the n-node branched payload (K=17 preserved).
                        // `chain_tok[cliff]` is the token the MAIN flat chain
                        // carries at the cliff; `leaf_tok` is the forked leaf.
                        let (chain_cliff_tok, leaf_tok) = if tail_top2 {
                            (sib_token, drafts[cliff]) // main chain = top-2
                        } else {
                            (drafts[cliff], sib_token) // main chain = top-1
                        };
                        // Main flat chain over compact 0..n-1: drafts for every
                        // row except the cliff (which carries chain_cliff_tok).
                        // The chain is shortened by one row (the last) to free a
                        // node for the leaf sibling.
                        let mut tree_token_ids: Vec<u32> = Vec::with_capacity(n);
                        for i in 0..n - 1 {
                            tree_token_ids.push(if i == cliff { chain_cliff_tok } else { drafts[i] });
                        }
                        // Leaf sibling at the last compact slot: forks at the
                        // cliff (parent = cliff-1, sharing the cliff's parent).
                        tree_token_ids.push(leaf_tok);
                        let mut parent_indices: Vec<i32> = Vec::with_capacity(n);
                        parent_indices.push(-1);
                        for i in 1..n - 1 {
                            parent_indices.push((i - 1) as i32);
                        }
                        parent_indices.push((cliff as i32) - 1);

                        static BRANCH_DBG: std::sync::atomic::AtomicUsize =
                            std::sync::atomic::AtomicUsize::new(0);
                        let dbg = BRANCH_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if dbg < 8 {
                            tracing::info!(
                                "DFlash BRANCH #{dbg}: n={n} cliff_row={cliff} margin={min_margin:.2} \
                                 (thresh={margin_thresh}) tail_top2={tail_top2} chain_cliff={} leaf={} \
                                 parents[..min(8)]={:?}",
                                chain_cliff_tok,
                                leaf_tok,
                                &parent_indices[..parent_indices.len().min(8)],
                            );
                        }
                        dstate.pending_tree_payload = Some(super::ddtree::TreePayload {
                            tree_token_ids,
                            parent_indices,
                        });
                        dstate.last_num_drafted = drafts.len();
                        return Ok(drafts);
                    } else {
                        static BRANCH_FLAT_DBG: std::sync::atomic::AtomicUsize =
                            std::sync::atomic::AtomicUsize::new(0);
                        let dbg =
                            BRANCH_FLAT_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if dbg < 5 {
                            tracing::info!(
                                "DFlash BRANCH gate CLOSED #{dbg}: min_margin={min_margin:.2} \
                                 >= thresh={margin_thresh} (flat chain, no tree payload)"
                            );
                        }
                        // Gate closed → flat chain, identical to non-branch path.
                        dstate.pending_tree_payload = None;
                        return Ok(drafts);
                    }
                }
                Ok(_) => {
                    dstate.pending_tree_payload = None;
                    return Ok(drafts);
                }
                Err(e) => {
                    tracing::warn!(
                        "DFlash BRANCH top-2 extraction failed ({e}); flat chain fallback"
                    );
                    dstate.pending_tree_payload = None;
                    return Ok(drafts);
                }
            }
        }

        // M4B (MVP): when DDTree mode is active, build a degenerate
        // single-chain DDTreePayload from the existing top-1 drafts and
        // stash it in dstate for the scheduler to drain. This exercises
        // the M3 payload bridge end-to-end. Real top-k extraction needs
        // a CUDA top-k kernel over 248K vocab — M4B v2.
        //
        // Activated by setting ATLAS_DFLASH_METHOD=ddtree at startup
        // (mirrors the --dflash-method=ddtree CLI flag wired in M1).
        let ddtree_active = std::env::var("ATLAS_DFLASH_METHOD")
            .ok()
            .as_deref()
            == Some("ddtree");
        // ATLAS_DDTREE_NONFLAT=1 enables the experimental non-flat root-sibling
        // topology that exercises the M8A tree kernel. Default OFF because the
        // first-pass tree kernel isn't bit-equivalent to wy17 — flat-chain
        // tokens drift numerically and drafter accept collapses. Re-enable
        // after task #45 (Python-ref bit-diff + reduction-order fix).
        let nonflat_enabled = std::env::var("ATLAS_DDTREE_NONFLAT").ok().as_deref()
            == Some("1");
        static PAYLOAD_DBG_DONE: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !PAYLOAD_DBG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!(
                "M8A propose dbg: ddtree_active={} nonflat_enabled={} drafts.len={}",
                ddtree_active, nonflat_enabled, drafts.len()
            );
        }
        // ATLAS_DDTREE_CHAIN_ONLY=1: produce a strict flat chain even in DDTree
        // mode so requires_tree_kernel returns false → wy17 path fires → no
        // numerical drift vs baseline. Validates that the drift cascade isolation
        // diagnosis from m8a_diff.py is correct.
        let chain_only =
            std::env::var("ATLAS_DDTREE_CHAIN_ONLY").ok().as_deref() == Some("1");
        if chain_only && ddtree_active && !drafts.is_empty() {
            let n = drafts.len();
            let mut parent_indices = Vec::with_capacity(n);
            parent_indices.push(-1i32);
            for i in 0..n.saturating_sub(1) {
                parent_indices.push(i as i32);
            }
            dstate.pending_tree_payload = Some(super::ddtree::TreePayload {
                tree_token_ids: drafts.clone(),
                parent_indices,
            });
            dstate.last_num_drafted = drafts.len();
            dstate.first_propose_done = true;
            return Ok(drafts);
        }
        if ddtree_active && nonflat_enabled && drafts.len() >= 2 {
            // M4B v2 lite: build a NON-FLAT tree from existing γ_eff drafts so
            // the M8A kernel actually fires (flat payloads short-circuit to
            // wy_k). Topology: drafts[0..n-1] form the main chain; drafts[n-1]
            // becomes a SIBLING of drafts[0] under root.
            //
            //                       root
            //                      /    \
            //               drafts[0]   drafts[n-1]     ← sibling branch
            //                  |
            //               drafts[1]
            //                  |
            //                 ...
            //                  |
            //               drafts[n-2]
            //
            // tree_token_ids order is drafts[0..n], parent_indices keys to
            // those compact indices. drafts[n-1] sits at compact index n-1
            // with parent -1 (root) — breaks the flat-chain assumption.
            let n = drafts.len();
            let mut parent_indices = Vec::with_capacity(n);
            parent_indices.push(-1i32); // drafts[0] → root
            for i in 1..n.saturating_sub(1) {
                parent_indices.push((i - 1) as i32); // drafts[i] → drafts[i-1]
            }
            // drafts[n-1] also attaches to root — this is the sibling.
            if n >= 2 {
                parent_indices.push(-1i32);
            }
            dstate.pending_tree_payload = Some(super::ddtree::TreePayload {
                tree_token_ids: drafts.clone(),
                parent_indices,
            });
        } else if ddtree_active && !drafts.is_empty() {
            // M4B v2: real top-K DDTree. Extract per-MASK-position top-K
            // tokens from `self.scratch.logits` (already populated by the
            // forward_block call above; survives until next propose), seed
            // the [`super::ddtree::build_ddtree`] best-first builder, then
            // serialize the result into [`TreePayload`].
            //
            // Set ATLAS_DDTREE_CHAIN_ONLY=1 (handled above as an early
            // return) to bypass this path and emit the legacy flat-chain
            // payload — needed when validating bit-equivalence of the wy17
            // GDN kernel before turning on the tree-aware verifier.
            // Match AEON-7 env var naming (DDTREE_*) with an Atlas-prefixed
            // alias (ATLAS_DDTREE_*) for consistency with the rest of the
            // codebase. AEON-7's serve scripts set the unprefixed names.
            let top_k: usize = std::env::var("ATLAS_DDTREE_TOP_K")
                .ok()
                .or_else(|| std::env::var("DDTREE_TOP_K").ok())
                .and_then(|s| s.parse().ok())
                .unwrap_or(8)
                .clamp(1, super::DDTREE_TOP_K_MAX);
            let budget: usize = std::env::var("ATLAS_DDTREE_BUDGET")
                .ok()
                .or_else(|| std::env::var("DDTREE_BUDGET").ok())
                .and_then(|s| s.parse().ok())
                .unwrap_or(15)
                .max(1);
            let min_root_branches: usize = std::env::var("ATLAS_DDTREE_MIN_ROOT_BRANCHES")
                .ok()
                .or_else(|| std::env::var("DDTREE_MIN_ROOT_BRANCHES").ok())
                .and_then(|s| s.parse().ok())
                .unwrap_or(2);
            let chain_seed = std::env::var("ATLAS_DDTREE_CHAIN_SEED")
                .ok()
                .as_deref()
                != Some("0");

            // Extract top-K logits from the just-computed γ_eff logit rows.
            let gamma_eff = drafts.len();
            match self.extract_topk_from_logits(ctx.gpu, _stream, gamma_eff, top_k) {
                Ok((topk_tokens, topk_logits)) => {
                    // Reshape into [γ_eff][top_k] DraftCandidate vectors.
                    // logprob_proxy = logit - row_max so the top-1 row gets 0
                    // and the rest are negative — preserves ranking and acts
                    // as a sensible additive path-score for the best-first
                    // tree builder. Real log-softmax would require a vocab-
                    // wide reduction we want to avoid in this hot path.
                    let mut candidates_by_depth: Vec<Vec<super::ddtree::DraftCandidate>> =
                        Vec::with_capacity(gamma_eff);
                    for row in 0..gamma_eff {
                        let base = row * top_k;
                        let row_max = topk_logits[base];
                        let row_cands: Vec<super::ddtree::DraftCandidate> = (0..top_k)
                            .map(|i| super::ddtree::DraftCandidate {
                                token_id: topk_tokens[base + i],
                                logprob: topk_logits[base + i] - row_max,
                            })
                            .collect();
                        candidates_by_depth.push(row_cands);
                    }

                    // Root token is irrelevant for the payload (tree_token_ids
                    // skips index 0); pass u32::MAX as a sentinel.
                    match super::ddtree::build_ddtree(
                        &candidates_by_depth,
                        budget,
                        top_k,
                        chain_seed,
                        min_root_branches,
                        u32::MAX,
                    ) {
                        Ok(tree) => {
                            let tree_token_ids = tree.token_ids_for_verifier();
                            let parent_indices = tree.parent_indices_for_verifier();
                            static TOPK_TREE_DBG_DONE: std::sync::atomic::AtomicBool =
                                std::sync::atomic::AtomicBool::new(false);
                            if !TOPK_TREE_DBG_DONE.swap(
                                true,
                                std::sync::atomic::Ordering::Relaxed,
                            ) {
                                tracing::info!(
                                    "M4B v2 real top-K tree: γ_eff={} k={} budget={} \
                                     min_root_branches={} chain_seed={} \
                                     → nodes={} parent_indices={:?} \
                                     tokens[..min(8)]={:?}",
                                    gamma_eff,
                                    top_k,
                                    budget,
                                    min_root_branches,
                                    chain_seed,
                                    tree_token_ids.len(),
                                    parent_indices,
                                    &tree_token_ids
                                        [..tree_token_ids.len().min(8)],
                                );
                            }
                            dstate.pending_tree_payload =
                                Some(super::ddtree::TreePayload {
                                    tree_token_ids,
                                    parent_indices,
                                });
                        }
                        Err(e) => {
                            tracing::warn!(
                                "DDTree build_ddtree failed ({e}); falling back \
                                 to flat-chain payload"
                            );
                            // Flat-chain fallback so the dispatch chain still
                            // sees a valid (degenerate) payload.
                            let n = drafts.len();
                            let mut parent_indices = Vec::with_capacity(n);
                            parent_indices.push(-1i32);
                            for i in 0..n.saturating_sub(1) {
                                parent_indices.push(i as i32);
                            }
                            dstate.pending_tree_payload =
                                Some(super::ddtree::TreePayload {
                                    tree_token_ids: drafts.clone(),
                                    parent_indices,
                                });
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "DDTree top-K extraction failed ({e}); emitting flat-chain payload"
                    );
                    let n = drafts.len();
                    let mut parent_indices = Vec::with_capacity(n);
                    parent_indices.push(-1i32);
                    for i in 0..n.saturating_sub(1) {
                        parent_indices.push(i as i32);
                    }
                    dstate.pending_tree_payload = Some(super::ddtree::TreePayload {
                        tree_token_ids: drafts.clone(),
                        parent_indices,
                    });
                }
            }
        } else {
            dstate.pending_tree_payload = None;
        }
        Ok(drafts)
    }
}
