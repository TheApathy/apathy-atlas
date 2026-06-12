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
            // dflash_hidden_save rows 0..num_append hold the hiddens of the
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
                let src = base.offset(i * dstate.ctx_slot_bytes);
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
