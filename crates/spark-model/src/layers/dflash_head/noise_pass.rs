// SPDX-License-Identifier: AGPL-3.0-only

//! One denoise pass of the DFlash γ-block forward: steps 2–5 of the
//! original `forward_block` body (noise-embedding build → 8 drafter
//! layers → final norm/lm_head → per-row argmax), extracted verbatim so
//! `forward_block` can iterate the pass for multi-step denoise drafting
//! (`ATLAS_DFLASH_DENOISE_STEPS=N`). Pass k+1 re-embeds pass k's argmax
//! at rows whose prediction was confident; the remaining masked rows are
//! then conditioned on partial predictions (DiffusionGemma-style
//! iterative refinement).
//!
//! With `committed` all-`None` (the single-pass default) this reproduces
//! the pre-extraction behavior bit-for-bit.

use anyhow::Result;

use super::{BlockDiffusionDraftHead, DflashProposerState, DraftRowLayout};
use crate::layer::ForwardContext;

/// Pass-invariant inputs to [`BlockDiffusionDraftHead::run_noise_pass`].
/// Mirrors the locals computed once per `forward_block` call.
pub(super) struct NoisePassArgs {
    pub last_token: u32,
    pub eff_ctx: usize,
    pub gamma_eff: usize,
    pub n_attn: u32,
    pub mask_id: u32,
    pub needed_start: usize,
    pub stream: u64,
    pub debug_dump: bool,
    pub kprofile: bool,
}

impl BlockDiffusionDraftHead {
    /// Run one γ-block noise pass. `committed[i]` (length `gamma_eff`),
    /// when `Some(tok)`, embeds `tok` at noise row i+1 instead of the
    /// mask token (multi-step denoise feedback).
    ///
    /// On return, `self.scratch.logits` holds the pass's `[γ_eff, vocab]`
    /// logits and `self.scratch.draft_tokens_dev[0..γ_eff]` holds the
    /// per-row argmax tokens.
    ///
    /// Cache safety for repeated calls within one propose: the per-layer
    /// ctx K/V caches are append-once with explicit range tracking — on
    /// passes ≥1 `old_ctx_count == eff_ctx` and `new_ctx_count == 0`, so
    /// the layer loop only COPIES from the caches (never re-appends) and
    /// `cache_{k,v}_start/end` stay consistent. The fc_proj cache is not
    /// touched at all here (step 0 runs once in `forward_block`).
    pub(super) fn run_noise_pass(
        &self,
        a: &NoisePassArgs,
        committed: &[Option<u32>],
        loop_pass: usize,
        ctx: &ForwardContext,
        dstate: &mut DflashProposerState,
    ) -> Result<()> {
        use crate::layers::ops;

        let NoisePassArgs {
            last_token,
            eff_ctx,
            gamma_eff,
            n_attn,
            mask_id,
            needed_start,
            stream,
            debug_dump,
            kprofile,
        } = *a;
        let h = self.hidden_size as u32;
        let q_dim = (self.num_q_heads * self.head_dim) as u32;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;
        let inter = self.intermediate_size as u32;
        let bf16 = 2usize;
        let inv_sqrt_d = 1.0f32 / (self.head_dim as f32).sqrt();
        let gpu = ctx.gpu;
        let row_layout = DraftRowLayout::for_family(self.checkpoint_family, gamma_eff);
        let noise_rows = row_layout.query_rows;
        let output_rows = gamma_eff;
        debug_assert_eq!(eff_ctx + noise_rows, n_attn as usize);

        let dump_bf16 = |label: &str, ptr: spark_runtime::gpu::DevicePtr, n: usize| -> Result<()> {
            if !debug_dump {
                return Ok(());
            }
            let mut buf = vec![0u8; n * 2];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(ptr, &mut buf)?;
            let vals: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            tracing::info!("DFLASH DUMP {label} [{n}]: {:?}", &vals);
            Ok(())
        };

        // ── Step 2: stream_buf layout ──
        // First eff_ctx rows: zero (Q-side ctx is zero; K/V-side gets
        // overwritten in step 3b' below).
        // Next γ rows: embed of [last_token (bonus), mask, mask, ..., mask].
        // The drafter is trained with query = [next_token_id, mask × (γ-1)]
        // per vLLM (qwen3_dflash.py + dflash.py:set_inputs_first_pass: "Q from
        // query embeddings (bonus + mask tokens)"). Without the bonus token at
        // position 0, the drafter has no anchor and produces a constant
        // high-frequency token (`,`, `<|im_end|>`) for every position.
        // Multi-step denoise (passes ≥1): rows with a committed prediction
        // embed the predicted token instead of the mask.
        // Total stream_buf width = n_attn rows.
        if eff_ctx > 0 {
            // Stream-ordered (was gpu.memset = default-stream + host sync):
            // under ATLAS_DFLASH_ASYNC the pass runs on the propose stream and
            // this zero must be ordered with the surrounding kernels on THAT
            // stream (the post-embed re-zero below would otherwise race
            // batched_embed). Same-stream ordering ⇒ byte-identical when off.
            gpu.memset_async(
                self.scratch.stream_buf,
                0,
                eff_ctx * self.hidden_size * bf16,
                stream,
            )?;
        }
        // [eff_ctx zeros, last_token (bonus), per-row mask-or-committed × γ_eff].
        //
        // Pass ≥1 with DeepLoop enabled: use the GPU-side commit-all path.
        // Stage pass N's argmax from draft_tokens_dev[0..gamma_eff] into
        // topk_tokens_dev (non-overlapping), then reconstruct draft_tokens_dev
        // on-stream via dflash_token_recommit — no host D2H, async-safe.
        if loop_pass > 0 && super::dflash_deeploop_enabled() {
            gpu.copy_d2d_async(
                self.scratch.draft_tokens_dev,
                self.scratch.topk_tokens_dev,
                gamma_eff * 4,
                stream,
            )?;
            ops::token_recommit(
                gpu,
                self.kernels.token_recommit,
                self.scratch.draft_tokens_dev,
                self.scratch.topk_tokens_dev,
                last_token,
                eff_ctx as u32,
                n_attn,
                stream,
            )?;
        } else {
            let token_ids_host: Vec<i32> = std::iter::repeat_n(0i32, eff_ctx)
                .chain(std::iter::once(last_token as i32))
                .chain(
                    (0..row_layout.feedback_rows())
                        .map(|i| committed[i].map(|t| t as i32).unwrap_or(mask_id as i32)),
                )
                .collect();
            if debug_dump {
                tracing::info!(
                    "DFLASH DUMP token_ids_host: mask={} eff_ctx={} ids[0..8]={:?}",
                    self.mask_token_id,
                    eff_ctx,
                    &token_ids_host[..token_ids_host.len().min(8)],
                );
            }
            let tid_bytes: Vec<u8> = token_ids_host
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect();
            gpu.copy_h2d(&tid_bytes, self.scratch.draft_tokens_dev)?;
        }
        ops::batched_embed(
            gpu,
            self.kernels.batched_embed,
            self.scratch.draft_tokens_dev,
            self.embed_tokens_shared,
            self.scratch.stream_buf,
            n_attn,
            h,
            stream,
        )?;
        // Re-zero ctx slots (batched_embed wrote token-0 embedding to them).
        if eff_ctx > 0 {
            // Stream-ordered (was gpu.memset = default-stream + host sync):
            // under ATLAS_DFLASH_ASYNC the pass runs on the propose stream and
            // this zero must be ordered with the surrounding kernels on THAT
            // stream (the post-embed re-zero below would otherwise race
            // batched_embed). Same-stream ordering ⇒ byte-identical when off.
            gpu.memset_async(
                self.scratch.stream_buf,
                0,
                eff_ctx * self.hidden_size * bf16,
                stream,
            )?;
        }
        // ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN=1: overwrite noise rows
        // [eff_ctx..n_attn) with a deterministic pattern matching the
        // PyTorch reference. Lets us compare layer-0 q/k/v post-projection
        // when both Atlas and PyTorch see identical input.
        let force_noise_pattern = std::env::var("ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN")
            .ok()
            .as_deref()
            == Some("1");
        if force_noise_pattern {
            let mut bytes = Vec::with_capacity(noise_rows * self.hidden_size * bf16);
            for t in 0..noise_rows {
                for j in 0..self.hidden_size {
                    let v =
                        0.001_f32 * ((t + 1) as f32) * ((j + 1) as f32) / (self.hidden_size as f32);
                    let bf16_bits = (v.to_bits() >> 16) as u16;
                    bytes.extend_from_slice(&bf16_bits.to_le_bytes());
                }
            }
            gpu.copy_h2d(
                &bytes,
                self.scratch
                    .stream_buf
                    .offset(eff_ctx * self.hidden_size * bf16),
            )?;
        }

        // ── Step 3: 8 drafter layers ──
        //
        // All compute runs on `n_attn = eff_ctx + γ` rows. Slots [0..eff_ctx]
        // are CTX (Q-zero / KV from fc_proj projection) and slots
        // [eff_ctx..n_attn] are NOISE (full Q/K/V from embeddings).
        // Per-layer flow follows `dflash.py:Qwen3DFlashDecoderLayer.forward`.
        // Body extracted to `forward_block_layer.rs` for the 500-LoC budget.
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let args = super::forward_block_layer::LayerArgs {
                layer_idx,
                n_attn,
                eff_ctx,
                h,
                q_dim,
                kv_dim,
                inter,
                bf16,
                inv_sqrt_d,
                stream,
                needed_start,
                window: self.ctx_window,
                loop_pass,
            };
            self.forward_block_layer(layer, &args, ctx, debug_dump, dstate, kprofile)?;
        }

        // ── Step 4: family-specific final RMSNorm + LM head rows ──
        // Generic DFlash skips the anchor and reads γ MASK rows. DSpark
        // starts at the anchor and reads all γ query rows; dropping that
        // row shifts every proposal one position into the future.
        let noise_byte_offset = (eff_ctx + row_layout.output_start) * self.hidden_size * bf16;
        let stream_noise = self.scratch.stream_buf.offset(noise_byte_offset);
        let norm_noise = self.scratch.norm_buf.offset(noise_byte_offset);
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            stream_noise,
            &self.norm,
            norm_noise,
            gamma_eff as u32,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        // Capped vocab: the shared lm_head may have fewer rows than the
        // drafter's vocab_size (e.g. target capped 248320→248077).
        let lm_vocab = self.target_vocab_size.min(self.vocab_size) as u32;
        // PERF NOTE (2026-05-19): tried switching to `dense_gemm_tc` here —
        // the shape (M=γ_eff≤16, N=vocab=248k, K=2048) looks ideal for the
        // tensor-core kernel (M_TILE=16, m16n8k16 MMA). But A/B benchmark
        // showed zero throughput change (8.70 vs 8.72 mean tok/s). The
        // lm_head GEMM is bandwidth-bound on weight read (~1GB at 273GB/s
        // ≈ 4ms) — TC compute doesn't shrink the floor. Left as scalar
        // dense_gemm to keep the dispatch simple.
        //
        // ATLAS_DFLASH_LM_HEAD_NVFP4=1 (2026-07-31, propose-to-floor): the
        // floor is the weight READ, so shrink the read — the target's
        // NVFP4-T lm_head (shared allocation, ATLAS_LM_HEAD_T) holds the
        // same rows at 1/4 the bytes; the 96k --mtp-vocab prefix is a
        // column slice of the T layout read via ldb. ~1 GB BF16 → ~0.26 GB.
        // Drafter-only (verify commits the target's own argmax) — gated on
        // accept-rate hold + md5 constitution, not bit-parity of drafts.
        // ATLAS_DFLASH_LM_HEAD_FP8 (2026-07-31, second attempt): halve the
        // read instead of quartering it — pre-scaled E4M3 slice (built at
        // load; see lm_head_shared_fp8 field doc) keeps ~3 mantissa bits of
        // logit fidelity where the NVFP4 slice's E2M1 measured accepted
        // 5.88→5.36 and lost more than the bandwidth bought. The 1/s
        // compensation lives in `self.norm`, so these logits are true-scale.
        let lm_head_t_fast = std::env::var("ATLAS_DFLASH_LM_HEAD_NVFP4").ok().as_deref()
            == Some("1")
            && self.lm_head_shared_t.is_some()
            && self.kernels.w4a16_gemm_t_m32_n64.0 != 0;
        // The NVFP4-T fast path must be checked FIRST: when `lm_head_shared_fp8`
        // is present (the default, ATLAS_DFLASH_LM_HEAD_FP8=1) the fp8 branch
        // previously shadowed it as an else-if — the NVFP4-T lm_head was built
        // (~0.26 GB vs the FP8 slice's ~1.24 GB) but never used, so
        // ATLAS_DFLASH_LM_HEAD_NVFP4=1 silently measured the fp8 path.
        if lm_head_t_fast {
            crate::kprof!(gpu, stream, "draft_lm_head_nvfp4_t", {
                ops::w4a16_gemm_n64_m32_ldb(
                    gpu,
                    self.kernels.w4a16_gemm_t_m32_n64,
                    norm_noise,
                    self.lm_head_shared_t.as_ref().unwrap(),
                    self.scratch.logits,
                    gamma_eff as u32,
                    lm_vocab,
                    h,
                    self.lm_head_shared_t_ldb,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
        } else if let Some(fp8) = self.lm_head_shared_fp8 {
            crate::kprof!(gpu, stream, "draft_lm_head_fp8", {
                ops::fp8_gemm_n128(
                    gpu,
                    self.kernels.fp8_gemm_t,
                    norm_noise,
                    fp8,
                    self.scratch.logits,
                    gamma_eff as u32,
                    lm_vocab,
                    h,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
        } else {
            ops::dense_gemm(
                gpu,
                self.kernels.dense_gemm,
                norm_noise,
                &crate::weight_map::DenseWeight {
                    weight: self.lm_head_shared,
                },
                self.scratch.logits,
                gamma_eff as u32,
                lm_vocab,
                h,
                stream,
            )?;
        }

        // Optional full-stream dump after final norm (debug; before lm_head).
        if debug_dump {
            dump_bf16("final.norm_buf[noise0]", norm_noise, 10)?;
            // Sanity-check: dump first 10 BF16 values of target's lm_head_shared.
            // If this returns zeros or garbage, the BF16 lm_head was freed by
            // factory.rs's NVFP4 quantization step.
            dump_bf16("final.lm_head_shared[0..10]", self.lm_head_shared, 10)?;
        }

        // ATLAS_DFLASH_DEBUG_DUMP_ALL_LAYERS=1: final norm/logits dumps.
        let dump_all_layers = std::env::var("ATLAS_DFLASH_DEBUG_DUMP_ALL_LAYERS")
            .ok()
            .as_deref()
            == Some("1");
        if dump_all_layers {
            let norm_bytes = output_rows * self.hidden_size * bf16;
            let mut buf = vec![0u8; norm_bytes];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(norm_noise, &mut buf)?;
            let path = "/tmp/atlas_final_norm_buf.bin";
            if !std::path::Path::new(path).exists() {
                let _ = std::fs::write(path, &buf);
                tracing::info!("DFLASH DUMP_ALL: wrote {norm_bytes}B to {path}");
            }
        }

        // ── Step 5: per-row argmax (generic) OR DFlash 2 candidate-selector
        // greedy walk (selector present) → γ_eff token ids ──
        let argmax_vocab = self.target_vocab_size.min(self.vocab_size) as u32;
        let lm_stride = self.target_vocab_size.min(self.vocab_size);
        if let Some(sel) = &self.selector {
            // CandidateSelector.select at T=0 (reference: z-lab/dflash
            // dflash/model.py): top-k per position, then a sequential greedy
            // walk scoring each candidate against the predecessor via the
            // low-rank codebooks — replaces the per-row argmax. The walk is
            // drafter-only; the target verify still commits its own greedy
            // token, so this only changes which tokens are *proposed*.
            let layout = super::forward_block::checked_topk_layout(
                gamma_eff, self.gamma, lm_stride, sel.top_k,
            )?;
            // top-k over the logits rows. sel.top_k == 16 == DDTREE_TOP_K_MAX,
            // so the [γ, DDTREE_TOP_K_MAX] scratch row strides match the
            // topk kernel's contiguous [rows, k] layout exactly.
            ops::topk_bf16(
                gpu,
                self.kernels.topk,
                self.scratch.logits,
                self.scratch.topk_tokens_dev,
                self.scratch.topk_logits_dev,
                layout.num_rows,
                layout.vocab,
                layout.k as u32,
                stream,
            )?;
            // hidden = hidden_projection(final_norm(noise)) → [γ, rank].
            // Reference: `select` applies `hidden_projection` to the
            // (already final-normed) draft hidden — `norm_noise` here.
            ops::dense_gemm(
                gpu,
                self.kernels.dense_gemm,
                norm_noise,
                &sel.hidden_projection,
                self.scratch.selector_hidden,
                gamma_eff as u32,
                sel.rank as u32,
                h,
                stream,
            )?;
            // Greedy walk seeded at the last verified token (`last_token`, the
            // bonus/root row's token id): draft_tokens_dev[i] = chosen
            // candidate at position i.
            {
                static SEL_SEEN: std::sync::Once = std::sync::Once::new();
                SEL_SEEN.call_once(|| {
                    tracing::info!("DFLASH2_EXEC: selector_walk RAN");
                });
            }
            ops::dflash2_selector_walk(
                gpu,
                self.kernels.dflash2_selector_walk,
                self.scratch.topk_logits_dev,
                self.scratch.topk_tokens_dev,
                self.scratch.selector_hidden,
                sel.predecessor_codebook.weight,
                sel.successor_codebook.weight,
                self.scratch.draft_tokens_dev,
                last_token,
                gamma_eff as i32,
                sel.rank as i32,
                stream,
            )?;
        } else {
            for i in 0..gamma_eff {
                let logits_row = self.scratch.logits.offset(i * lm_stride * bf16);
                let token_slot = self.scratch.draft_tokens_dev.offset(i * 4);
                ops::argmax_bf16(
                    gpu,
                    self.kernels.argmax,
                    logits_row,
                    token_slot,
                    argmax_vocab,
                    stream,
                )?;
            }
        }
        if debug_dump {
            dump_bf16("final.logits[noise0]", self.scratch.logits, 10)?;
        }
        if dump_all_layers {
            // The lm-head output is compact with `lm_stride`, which can be
            // smaller than the drafter checkpoint vocabulary after target
            // vocab capping. Copy only rows and columns written this pass.
            let logits_bytes = output_rows * lm_stride * bf16;
            let mut buf = vec![0u8; logits_bytes];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(self.scratch.logits, &mut buf)?;
            let path = "/tmp/atlas_final_logits.bin";
            if !std::path::Path::new(path).exists() {
                let _ = std::fs::write(path, &buf);
                tracing::info!("DFLASH DUMP_ALL: wrote {logits_bytes}B to {path}");
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod debug_row_count_tests {
    use super::DraftRowLayout;
    use crate::weight_loader::DrafterCheckpointFamily;

    #[test]
    fn generic_dflash_keeps_anchor_plus_gamma_masks_and_drops_anchor_output() {
        let layout = DraftRowLayout::for_family(DrafterCheckpointFamily::Dflash, 15);
        assert_eq!(layout.query_rows, 16);
        assert_eq!(layout.output_start, 1);
        assert_eq!(layout.feedback_rows(), 15);
    }

    #[test]
    fn dspark_uses_gamma_total_rows_and_includes_anchor_output() {
        let layout = DraftRowLayout::for_family(DrafterCheckpointFamily::Dspark, 7);
        assert_eq!(layout.query_rows, 7);
        assert_eq!(layout.output_start, 0);
        assert_eq!(layout.feedback_rows(), 6);
    }

    #[test]
    fn dspark_k14_is_one_anchor_plus_thirteen_masks_not_two_blocks() {
        let layout = DraftRowLayout::for_family(DrafterCheckpointFamily::Dspark, 14);
        assert_eq!(layout.query_rows, 14);
        assert_eq!(layout.output_start, 0);
        assert_eq!(layout.feedback_rows(), 13);
    }
}
