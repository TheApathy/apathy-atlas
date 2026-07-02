// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash γ-block forward (Phase 2 kernel chain). Split out of
//! `dflash_head.rs` for file-size budget — body still exceeds the
//! 500 LoC target because the per-step kernel chain (fc → pos →
//! 8 drafter layers → final norm/lm_head/argmax → D2H) shares
//! many locals with no clean extraction boundary.

use anyhow::Result;

use super::{BlockDiffusionDraftHead, DflashProposerState};
use crate::layer::ForwardContext;

impl BlockDiffusionDraftHead {
    pub(super) fn forward_block(
        &self,
        last_token: u32,
        position: usize,
        ctx: &ForwardContext,
        stream: u64,
        dstate: &mut DflashProposerState,
    ) -> Result<Vec<u32>> {
        use crate::layers::ops;

        let g = self.gamma as u32;
        let h = self.hidden_size as u32;
        let q_dim = (self.num_q_heads * self.head_dim) as u32;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;
        let bf16 = 2usize;
        let gpu = ctx.gpu;

        // ── Kernel profiler (ATLAS_DFLASH_KERNEL_PROFILE=1) ──
        // Lightweight per-phase timing using synchronize-and-Instant. Matches
        // `qwen3_ssm/ssm_forward.rs:prof!` pattern. Aggregates per-kernel μs
        // sums across all drafter layers via a thread-local accumulator,
        // logged at the end of forward_block. Adds ~20μs/sync × ~14 syncs/layer
        // when enabled — only use for measurement, not production.
        let kprofile = std::env::var("ATLAS_DFLASH_KERNEL_PROFILE")
            .ok()
            .as_deref()
            == Some("1");
        let t_total = std::time::Instant::now();
        if kprofile {
            gpu.synchronize(stream)?;
        }
        let t_pre_layers = std::time::Instant::now();
        super::kprof_reset_layers();

        // Determine effective ctx_len: capped by the configured ctx_window
        // and the accumulator's actual fill. Use the LAST `eff_ctx` ctx
        // positions (most recent) — drafter trained on locally recent
        // context, distant history adds noise to attention.
        // ATLAS_DFLASH_DEBUG_CTX_OFF=1 disables ctx entirely (eff_ctx=0)
        // for A/B testing whether the drafter actually responds to ctx.
        let force_no_ctx = std::env::var("ATLAS_DFLASH_DEBUG_CTX_OFF").ok().as_deref() == Some("1");
        let force_ctx_used: Option<usize> = std::env::var("ATLAS_DFLASH_DEBUG_CTX_USED")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        let (ctx_base_ptr, ctx_total, eff_ctx) = if dstate.ctx_len > 0 && !force_no_ctx {
            let n = dstate.ctx_len;
            let eff = match force_ctx_used {
                Some(forced) => forced.min(n).min(self.ctx_window),
                None => n.min(self.ctx_window),
            };
            (Some(dstate.ctx_hidden_acc), n, eff)
        } else {
            (None, 0, 0)
        };
        // Noise block layout (vLLM PR #40898 alignment): (γ_eff+1) query
        // tokens = 1 bonus (last_token) + γ_eff MASK rows. Drafts read from
        // rows [1..γ_eff+1] → γ_eff drafts.
        //
        // γ_eff defaults to self.gamma (drafter config) but is capped by
        // ATLAS_DFLASH_DRAFT_CAP when set. The cap previously only filtered
        // the returned drafts AFTER the full γ forward — wasting drafter
        // compute. Now the noise block itself shrinks to γ_eff+1 rows,
        // cutting drafter forward latency proportionally (DUET/SpecKV-
        // inspired: don't compute drafts that will be discarded).
        let cap: usize = std::env::var("ATLAS_DFLASH_DRAFT_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(self.gamma);
        let gamma_eff = cap.min(self.gamma).max(1);
        let noise_count = gamma_eff + 1;
        let n_attn = (eff_ctx + noise_count) as u32;

        // Zero all scratch buffers to eliminate non-determinism from
        // uninitialized memory (different GPU allocations may contain
        // stale data from previous operations).
        let n_attn_usize = n_attn as usize;
        gpu.memset(self.scratch.stream_buf, 0, n_attn_usize * self.hidden_size * bf16)?;
        gpu.memset(self.scratch.norm_buf, 0, n_attn_usize * self.hidden_size * bf16)?;
        gpu.memset(self.scratch.q_buf, 0, n_attn_usize * q_dim as usize * bf16)?;
        gpu.memset(self.scratch.k_buf, 0, n_attn_usize * kv_dim as usize * bf16)?;
        gpu.memset(self.scratch.v_buf, 0, n_attn_usize * kv_dim as usize * bf16)?;
        gpu.memset(self.scratch.attn_out, 0, n_attn_usize * q_dim as usize * bf16)?;
        gpu.memset(self.scratch.mlp_intermediate, 0, n_attn_usize * self.intermediate_size * bf16)?;
        gpu.memset(self.scratch.mlp_up, 0, n_attn_usize * self.intermediate_size * bf16)?;
        gpu.memset(self.scratch.stream_acc, 0, n_attn_usize * self.hidden_size * bf16)?;
        gpu.memset(self.scratch.fc_proj, 0, self.ctx_window * self.hidden_size * bf16)?;
        gpu.memset(self.scratch.logits, 0, self.gamma * self.vocab_size * bf16)?;
        gpu.memset(self.scratch.draft_tokens_dev, 0, n_attn_usize * 4)?;

        let target_hidden_dim = self.target_layer_ids.len() * self.target_hidden_size;
        let ctx_slot_bytes = target_hidden_dim * bf16;

        // Debug dump gated by env var: prints first 10 BF16 floats of key
        // intermediates so a Python reference run on the same checkpoint
        // can be compared element-wise. Use ATLAS_DFLASH_DEBUG_DUMP=1.
        let debug_dump = std::env::var("ATLAS_DFLASH_DEBUG_DUMP").ok().as_deref() == Some("1");
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

        // ── Step 0: fc projection of captured target hiddens ──
        // For each of the `eff_ctx` most-recent ctx positions, run a GEMV
        // through `self.fc` (input: 10240 BF16 → output: 2048 BF16) and
        // then per-row RMSNorm through `self.hidden_norm`. Results land
        // contiguously in `scratch.fc_proj` shaped `[eff_ctx, hidden]`.
        if let Some(base) = ctx_base_ptr {
            // Walk the LAST `eff_ctx` slots of the accumulator.
            let start_slot = ctx_total.saturating_sub(eff_ctx);
            // ATLAS_DFLASH_ZERO_LATE_LAYERS=N zeros out the LAST N capture
            // layer slots per ctx position before the fc projection. This
            // is a workaround for SSM kernel numerical drift that
            // compounds layer-by-layer in Atlas vs HF transformers
            // reference: by L61 (the 5th capture layer of [1,16,31,46,61]),
            // cosine similarity drops to 0.86 — drafter sees OOD input.
            // Zeroing zeros out the corresponding fc input rows so the
            // drafter only conditions on the cleaner early-layer captures.
            // Effective drafter input dim drops from 5*hidden to (5-N)*hidden.
            let zero_late = std::env::var("ATLAS_DFLASH_ZERO_LATE_LAYERS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            // ATLAS_DFLASH_HF_OVERRIDE=<path>            — load from file
            // (manual one-prompt test).  We use atlas_tokens.json's prompt
            // for both Atlas and the prior `hf_capture.py` run.
            //
            // ATLAS_DFLASH_FLA_SIDECAR=http://host:port  — call the FLA
            // Python sidecar that runs HF Qwen3.6-27B with FLA's
            // bit-tested SSM kernels and returns ctx hiddens for the
            // sequence tokens currently in seq.tokens.  Drop-in
            // replacement for Atlas's drift-prone SSM capture, paid for
            // by one HTTP round-trip per prefill.  This is the in-flight
            // production fix for Option A — see fla_sidecar.py.
            //
            // Both: layout MUST be [n_ctx, 5*HIDDEN] BF16 (matches
            // ctx_hidden_acc slot semantics).
            // ATLAS_DFLASH_HF_OVERRIDE=<path>  — file-based ctx override.
            //   The file is expected to contain [n_ctx, 5*HIDDEN] BF16
            //   bytes generated by tools/dflash_layer_diff/fla_sidecar.py
            //   (or hf_capture.py). The Python sidecar runs HF Qwen3.6
            //   with FLA's bit-tested SSM kernels and writes this file
            //   per prefill — Atlas reads it here instead of using its
            //   own drift-prone captures.
            //
            //   Production wire-in plan (Option A): Atlas should call
            //   the sidecar over HTTP at prefill time (and per-step
            //   ideally). This file-based hook is the proof-of-concept
            //   path — flip the bench to use it for end-to-end accept
            //   rate measurement.
            if eff_ctx > 0
                && let Ok(p) = std::env::var("ATLAS_DFLASH_HF_OVERRIDE")
            {
                let needed = eff_ctx * ctx_slot_bytes;
                match std::fs::read(&p) {
                    Ok(hf_bytes) if hf_bytes.len() >= needed => {
                        gpu.copy_h2d(
                            &hf_bytes[..needed],
                            base.offset(start_slot * ctx_slot_bytes),
                        )?;
                        if debug_dump {
                            tracing::info!(
                                "DFLASH HF_OVERRIDE: loaded {} bytes from {}",
                                needed, p,
                            );
                        }
                    }
                    Ok(hf_bytes) => tracing::warn!(
                        "DFLASH HF_OVERRIDE file too small ({} < needed {})",
                        hf_bytes.len(), needed,
                    ),
                    Err(e) => tracing::warn!("DFLASH HF_OVERRIDE read failed: {e}"),
                }
            }
            if zero_late > 0 && eff_ctx > 0 {
                let n_capture = self.target_layer_ids.len();
                let n_zero = zero_late.min(n_capture);
                let h_bytes = self.target_hidden_size * bf16;
                for slot_i in 0..eff_ctx {
                    let slot_base = base.offset((start_slot + slot_i) * ctx_slot_bytes);
                    // Zero the LAST n_zero layer slices (indices n_capture-n_zero .. n_capture)
                    for layer_i in (n_capture - n_zero)..n_capture {
                        let layer_ptr = slot_base.offset(layer_i * h_bytes);
                        gpu.memset(layer_ptr, 0, h_bytes)?;
                    }
                }
                if debug_dump {
                    tracing::info!(
                        "DFLASH ZERO_LATE: zeroed last {} of {} capture layers across {} ctx slots",
                        n_zero, n_capture, eff_ctx
                    );
                }
            }
            // ATLAS_DFLASH_DEBUG_FORCE_PATTERN=1 overwrites the captured
            // target_hidden_stack with a deterministic test pattern so a
            // PyTorch reference run on the same input produces directly
            // comparable intermediates. Pattern: row i, col j contains
            // `0.01 * (i+1) * (j+1) / target_hidden` BF16. Mirrors
            // `dflash_pytorch_reference.py:make_input_target_hidden_stack`.
            let force_pattern = std::env::var("ATLAS_DFLASH_DEBUG_FORCE_PATTERN")
                .ok()
                .as_deref()
                == Some("1");
            if force_pattern && eff_ctx > 0 {
                let n_rows = self.target_layer_ids.len();
                let n_cols = self.target_hidden_size;
                let mut bytes = Vec::with_capacity(n_rows * n_cols * 2);
                for i in 0..n_rows {
                    for j in 0..n_cols {
                        let v = 0.01_f32 * ((i + 1) as f32) * ((j + 1) as f32) / (n_cols as f32);
                        // f32 → bf16 (truncate-to-zero of low 16 bits).
                        let bits = v.to_bits();
                        let bf16_bits = (bits >> 16) as u16;
                        bytes.extend_from_slice(&bf16_bits.to_le_bytes());
                    }
                }
                gpu.copy_h2d(&bytes, base.offset(start_slot * ctx_slot_bytes))?;
            }
            // Dump the FIRST ctx slot's input target_hidden_stack (first 10 floats).
            if eff_ctx > 0 {
                dump_bf16(
                    "step0.input.target_hidden_stack[0]",
                    base.offset(start_slot * ctx_slot_bytes),
                    10,
                )?;
            }
            // ATLAS_DFLASH_DEBUG_DUMP_FULL=1: write the full 10240-element
            // target_hidden_stack (one ctx slot) to /tmp/atlas_target_hidden.bin
            // so a Python reference can run dflash.py forward on the same
            // input and compare predicted draft tokens vs Atlas drafts.
            // Also dumps last_token + drafter outputs separately for the
            // bisect script. ONE-SHOT: writes only the first propose() call.
            static FULL_DUMP_DONE: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if eff_ctx > 0
                && !FULL_DUMP_DONE.load(std::sync::atomic::Ordering::Relaxed)
                && std::env::var("ATLAS_DFLASH_DEBUG_DUMP_FULL")
                    .ok()
                    .as_deref()
                    == Some("1")
                // Mirror the tokens-dump gate: defer until position >= N.
                && position
                    >= std::env::var("ATLAS_DFLASH_DUMP_MIN_POS")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(0)
            {
                // Dump ALL eff_ctx slots — needed to reproduce the
                // multi-token ctx in PyTorch reference. Layout:
                // contiguous BF16, eff_ctx slots × 5 layers × 2048 dims.
                let n_bytes = eff_ctx * ctx_slot_bytes;
                let mut buf = vec![0u8; n_bytes];
                gpu.synchronize(stream)?;
                gpu.copy_d2h(base.offset(start_slot * ctx_slot_bytes), &mut buf)?;
                if let Err(e) = std::fs::write("/tmp/atlas_target_hidden.bin", &buf) {
                    tracing::warn!("DFLASH DUMP_FULL: target_hidden write failed: {e}");
                } else {
                    tracing::info!(
                        "DFLASH DUMP_FULL: wrote {} bytes ({} ctx slots × {} BF16 elements) to /tmp/atlas_target_hidden.bin (last_token={}, position={}, eff_ctx={})",
                        n_bytes,
                        eff_ctx,
                        ctx_slot_bytes / 2,
                        last_token,
                        position,
                        eff_ctx,
                    );
                }
                // Sibling metadata JSON so the Python reference can
                // reconstruct the exact propose() inputs without parsing
                // tracing output. Same one-shot guard as the binary.
                let meta = format!(
                    "{{\"last_token\":{},\"position\":{},\"eff_ctx\":{},\"ctx_total\":{},\"gamma\":{},\"mask_token_id\":{},\"hidden\":{},\"target_hidden_size\":{},\"n_target_layers\":{}}}",
                    last_token,
                    position,
                    eff_ctx,
                    ctx_total,
                    self.gamma,
                    self.mask_token_id,
                    self.hidden_size,
                    self.target_hidden_size,
                    self.target_layer_ids.len(),
                );
                if let Err(e) = std::fs::write("/tmp/atlas_dump_meta.json", &meta) {
                    tracing::warn!("DFLASH DUMP_FULL: meta write failed: {e}");
                }
                FULL_DUMP_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            // Persistent fc_proj cache: copy old positions, compute new ones.
            let needed_start = ctx_total.saturating_sub(eff_ctx);
            let old_fc_end = needed_start
                .saturating_add(eff_ctx)
                .min(dstate.cache_fc_end)
                .max(needed_start);
            let old_fc_count = old_fc_end.saturating_sub(needed_start);
            let new_fc_count = eff_ctx.saturating_sub(old_fc_count);
            // PERF (2026-05-19): demoted from tracing::info! — fired on every
            // propose() call (~80×/s during decode), drowning real diagnostics.
            // Re-enable via RUST_LOG=spark_model::layers::dflash_head=debug.
            tracing::debug!(
                "DFlash fc_proj cache: needed=[{}..{}), cached=[{}..{}), old={}, new={}",
                needed_start,
                needed_start + eff_ctx,
                dstate.cache_fc_start,
                dstate.cache_fc_end,
                old_fc_count,
                new_fc_count,
            );

            if old_fc_count > 0 {
                self.cache_copy_range(
                    gpu,
                    dstate.ctx_fc_cache,
                    dstate.cache_fc_start,
                    dstate.cache_fc_end,
                    self.ctx_window,
                    needed_start,
                    old_fc_end,
                    self.scratch.fc_proj,
                    self.hidden_size * bf16,
                    stream,
                )?;
            }
            // EAGLE-3.1 per-layer FC-normalization (vLLM 2026-05-26):
            // ATLAS_DFLASH_FC_LAYERNORM=1 (default OFF). When the fc layer
            // consumes a CONCATENATION of multiple target-layer hidden states,
            // "the fused input becomes increasingly imbalanced as higher-layer
            // hidden states dominate" (larger magnitude). The fix is to
            // normalize EACH captured target hidden state independently
            // BEFORE the fc layer. DFlash concatenates the 5 raw captures
            // [layers 1,16,31,46,61] into one [5*2048] BF16 stack feeding ONE
            // fc GEMV — exactly the broken thing. Here we apply a per-slice
            // unit-variance RMSNorm (zero-weight → x*rms, no learned gamma) to
            // each of the n_capture layer slices so no single late-layer
            // capture's magnitude dominates the fused fc input. Drafter-side
            // only → target verify unchanged → token-exact; raises ACCEPTANCE.
            //
            // OOD caveat: self.fc was TRAINED on UN-normalized concat, so this
            // is out-of-distribution and may help or hurt — measured A/B.
            // Variant (a): plain unit-variance (no learned norm weight).
            let fc_layernorm = std::env::var("ATLAS_DFLASH_FC_LAYERNORM")
                .ok()
                .as_deref()
                == Some("1");
            if new_fc_count > 0 {
                // Compute fc projection for new context positions.
                for i in 0..new_fc_count {
                    let abs_pos = old_fc_end + i;
                    let raw_slot = base.offset(abs_pos * ctx_slot_bytes);
                    // Per-layer FC-norm: copy each of the n_capture target-layer
                    // slices [target_hidden_size] into fc_norm_in, unit-variance
                    // RMS-normalized independently, then feed the normalized
                    // concat to the fc GEMV instead of the raw slot. Mirrors the
                    // ZERO_LATE_LAYERS per-layer slicing of `ctx_hidden_acc`.
                    let src_slot = if fc_layernorm {
                        let n_capture = self.target_layer_ids.len();
                        let h_elems = self.target_hidden_size as u32;
                        for layer_i in 0..n_capture {
                            let in_ptr =
                                raw_slot.offset(layer_i * self.target_hidden_size * bf16);
                            let out_ptr = self
                                .scratch
                                .fc_norm_in
                                .offset(layer_i * self.target_hidden_size * bf16);
                            ops::rms_norm(
                                gpu,
                                self.kernels.rms_norm,
                                in_ptr,
                                &crate::weight_map::DenseWeight {
                                    weight: self.scratch.fc_norm_zero_w,
                                },
                                out_ptr,
                                1,
                                h_elems,
                                self.rms_norm_eps,
                                stream,
                            )?;
                        }
                        self.scratch.fc_norm_in
                    } else {
                        raw_slot
                    };
                    let dst_slot = self
                        .scratch
                        .fc_proj
                        .offset((old_fc_count + i) * self.hidden_size * bf16);
                    match (self.quant, self.fc_nvfp4.as_ref()) {
                        (super::DflashQuantization::Nvfp4, Some(fc_q)) => ops::w4a16_gemv(
                            gpu,
                            self.kernels.w4a16_gemv,
                            src_slot,
                            fc_q,
                            dst_slot,
                            h,
                            target_hidden_dim as u32,
                            stream,
                        )?,
                        _ => ops::dense_gemv(
                            gpu,
                            self.kernels.dense_gemv,
                            src_slot,
                            &self.fc,
                            dst_slot,
                            h,
                            target_hidden_dim as u32,
                            stream,
                        )?,
                    }
                }
                // RMSNorm on the newly-computed slice.
                ops::rms_norm(
                    gpu,
                    self.kernels.rms_norm,
                    self.scratch.fc_proj.offset(old_fc_count * self.hidden_size * bf16),
                    &self.hidden_norm,
                    self.scratch.fc_proj.offset(old_fc_count * self.hidden_size * bf16),
                    new_fc_count as u32,
                    h,
                    self.rms_norm_eps,
                    stream,
                )?;
                // Write new fc_proj into persistent cache.
                let (new_fc_start, new_fc_end) = self.cache_write_range(
                    gpu,
                    self.scratch.fc_proj.offset(old_fc_count * self.hidden_size * bf16),
                    old_fc_end,
                    new_fc_count,
                    dstate.ctx_fc_cache,
                    dstate.cache_fc_start,
                    dstate.cache_fc_end,
                    self.ctx_window,
                    self.hidden_size * bf16,
                    stream,
                )?;
                dstate.cache_fc_start = new_fc_start;
                dstate.cache_fc_end = new_fc_end;
            }
            // RMSNorm on the cached slice (if not already normalized).
            // The cache stores post-norm fc_proj, so cached positions are
            // already normalized. Only the new slice needs norm above.
            if eff_ctx > 0 {
                dump_bf16("step0.fc_proj.pre_norm[0]", self.scratch.fc_proj, 10)?;
                dump_bf16(
                    "step0.fc_proj.post_hidden_norm[0]",
                    self.scratch.fc_proj,
                    10,
                )?;
            }
        }

        // ── Step 1: build position ids ──
        // Layout: [ctx_pos_0, ..., ctx_pos_{eff_ctx-1}, seq_pos, ..., seq_pos+γ-1].
        // ctx_pos_i = position - eff_ctx + i — the absolute target indices
        // of the captured positions in chronological order.
        let ctx_start = position.saturating_sub(eff_ctx);
        let pos_host: Vec<i32> = (0..eff_ctx)
            .map(|i| (ctx_start + i) as i32)
            .chain((0..noise_count).map(|i| (position + i) as i32))
            .collect();
        let pos_bytes: Vec<u8> = pos_host.iter().flat_map(|p| p.to_le_bytes()).collect();
        gpu.copy_h2d(&pos_bytes, self.scratch.position_ids)?;
        if debug_dump {
            tracing::info!(
                "DFLASH DUMP positions: eff_ctx={} ctx_total={} position={} pos_ids[0..min(8,n_attn)]={:?}",
                eff_ctx,
                ctx_total,
                position,
                &pos_host[..pos_host.len().min(8)]
            );
        }

        // ── Steps 2–5: noise-block forward — possibly iterated ──
        //
        // The per-pass body (noise-embedding build → 8 drafter layers →
        // final norm/lm_head → per-row argmax) lives in
        // `noise_pass.rs:run_noise_pass`, extracted verbatim from here.
        //
        // ATLAS_DFLASH_DENOISE_STEPS=N (default 1 = single pass, the
        // original behavior with zero added work) runs the γ-block noise
        // forward N times per propose. After pass k, rows whose argmax is
        // confident (top1−top2 logit margin ≥ ATLAS_DFLASH_DENOISE_MARGIN,
        // default 1.0) are "committed": pass k+1 embeds the predicted token
        // at that row instead of the mask embedding, so the still-masked
        // rows are conditioned on partial predictions (DiffusionGemma-style
        // iterative refinement). Committed rows are frozen into the final
        // drafts (ATLAS_DFLASH_DENOISE_FREEZE=0 takes the last pass's
        // argmax everywhere instead). Early exits: a pass that commits no
        // new rows would re-run on identical input (deterministic → same
        // output), and a pass with ALL rows already committed has nothing
        // masked left to refine — both break out of the loop, so uniformly
        // high pass-1 confidence (counting workloads) pays for at most one
        // extra pass.
        //
        // Cache safety: the per-layer ctx K/V caches and the fc_proj cache
        // are append-once with explicit range tracking. Step 0 (fc
        // projection) runs once above; passes ≥1 re-enter the layer loop
        // where old_ctx_count == eff_ctx and new_ctx_count == 0, so caches
        // are only COPIED from, never re-appended — cache_{k,v,fc}_start/
        // end stay consistent across passes.
        //
        // OOD caveat: the drafter is TRAINED with all-mask noise rows;
        // real-token embeddings at committed rows are out-of-distribution.
        // Measure acceptance A/B before enabling in production.
        //
        // ATLAS_DFLASH_MASK_OVERRIDE: env var override for the mask token ID.
        let mask_id = std::env::var("ATLAS_DFLASH_MASK_OVERRIDE")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(self.mask_token_id);
        let denoise_steps: usize = std::env::var("ATLAS_DFLASH_DENOISE_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
            .clamp(1, 8);
        let denoise_margin: f32 = std::env::var("ATLAS_DFLASH_DENOISE_MARGIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        let denoise_freeze =
            std::env::var("ATLAS_DFLASH_DENOISE_FREEZE").ok().as_deref() != Some("0");
        let argmax_vocab = self.target_vocab_size.min(self.vocab_size) as u32;
        let needed_start = ctx_total.saturating_sub(eff_ctx);
        let pass_args = super::noise_pass::NoisePassArgs {
            last_token,
            eff_ctx,
            gamma_eff,
            n_attn,
            mask_id,
            needed_start,
            stream,
            debug_dump,
            kprofile,
        };
        let mut committed: Vec<Option<u32>> = vec![None; gamma_eff];
        let t_layers = std::time::Instant::now();
        for pass in 0..denoise_steps {
            self.run_noise_pass(&pass_args, &committed, ctx, dstate)?;
            if pass + 1 == denoise_steps {
                break;
            }
            // Confidence feedback: top-2 over this pass's logits gives the
            // per-row top1−top2 margin; the argmax tokens are already in
            // draft_tokens_dev. Cost: one topk launch + ~γ·12 bytes D2H.
            let used_bytes = gamma_eff * 2 * 4;
            gpu.memset(self.scratch.topk_tokens_dev, 0, used_bytes)?;
            gpu.memset(self.scratch.topk_logits_dev, 0, used_bytes)?;
            ops::topk_bf16(
                gpu,
                self.kernels.topk,
                self.scratch.logits,
                self.scratch.topk_tokens_dev,
                self.scratch.topk_logits_dev,
                gamma_eff as u32,
                argmax_vocab,
                2,
                stream,
            )?;
            gpu.synchronize(stream)?;
            let mut tok_bytes = vec![0u8; gamma_eff * 4];
            gpu.copy_d2h(self.scratch.draft_tokens_dev, &mut tok_bytes)?;
            let mut top2_bytes = vec![0u8; used_bytes];
            gpu.copy_d2h(self.scratch.topk_logits_dev, &mut top2_bytes)?;
            let pass_tokens: Vec<u32> = tok_bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let top2: Vec<f32> = top2_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let mut newly_committed = 0usize;
            for i in 0..gamma_eff {
                if committed[i].is_some() {
                    continue;
                }
                let margin = top2[i * 2] - top2[i * 2 + 1];
                if margin >= denoise_margin {
                    committed[i] = Some(pass_tokens[i]);
                    newly_committed += 1;
                }
            }
            let n_committed = committed.iter().filter(|c| c.is_some()).count();
            tracing::debug!(
                "DFlash denoise pass {}: +{} committed ({}/{} rows, margin≥{})",
                pass,
                newly_committed,
                n_committed,
                gamma_eff,
                denoise_margin,
            );
            if newly_committed == 0 || n_committed == gamma_eff {
                break;
            }
        }
        let layers_us = if kprofile {
            gpu.synchronize(stream)?;
            t_layers.elapsed().as_micros()
        } else {
            0
        };
        let t_tail = std::time::Instant::now();
        let dump_all_layers = std::env::var("ATLAS_DFLASH_DEBUG_DUMP_ALL_LAYERS")
            .ok()
            .as_deref()
            == Some("1");

        // ── Step 5b: optional logit-margin gate (top-1 vs top-2) ──
        //
        // ATLAS_DFLASH_MARGIN_GATE=<float>  (default 0.0 = disabled)
        //   When > 0, runs top-K=2 on the same logits and replaces the draft
        //   token at any row whose (logit_top1 - logit_top2) margin is below
        //   the threshold with `mask_token_id`. Mask is guaranteed to mismatch
        //   the target's argmax → forces the chain to terminate at the first
        //   low-confidence draft. The drafter is essentially saying "I'm
        //   guessing here" — passing the guess to target just to be rejected
        //   wastes verify cycles. Truncating early lets the verifier emit a
        //   bonus token (its own argmax) at the truncation point.
        //
        //   Recommended values: 0.3 (mild), 0.6 (aggressive), 1.0 (very strict).
        //   BF16 logits are O(10-30) for top tokens so margins of 0.3-1.0 are
        //   the meaningful range.
        //
        // ATLAS_DFLASH_ADAPTIVE_GAMMA=1  (default 0 = disabled)
        //   Tracks a moving window of accept counts (last 8 verifies on the
        //   sequence's DflashProposerState) and shrinks the effective draft
        //   count to `clamp(mean + ADAPTIVE_SLACK, ADAPTIVE_MIN, gamma_eff)`.
        //   Excess drafter positions get replaced with `mask_token_id` so the
        //   verifier rejects them and stops the chain — same mechanism as the
        //   margin gate. Saves verifier compute when the drafter is wasted.
        //
        // ATLAS_DFLASH_ADAPTIVE_MIN=<usize>  (default 4)
        //   Lower bound for adaptive gamma. Below 4 the K=γ verify graph
        //   doesn't cache well (too many distinct shapes).
        //
        // ATLAS_DFLASH_ADAPTIVE_SLACK=<usize>  (default 2)
        //   Added to the rolling mean accept count to give the drafter a
        //   little headroom past its recent average — accept rate is bursty.
        let margin_gate: f32 = std::env::var("ATLAS_DFLASH_MARGIN_GATE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let adaptive_gamma = std::env::var("ATLAS_DFLASH_ADAPTIVE_GAMMA")
            .ok()
            .as_deref()
            == Some("1");
        let adaptive_min: usize = std::env::var("ATLAS_DFLASH_ADAPTIVE_MIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        let adaptive_slack: usize = std::env::var("ATLAS_DFLASH_ADAPTIVE_SLACK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);
        // ATLAS_DFLASH_ADAPTIVE_MAX=<usize> caps the adaptive cutoff so the
        // K=γ verify graph never fires when its wall-time exceeds the
        // throughput benefit. K=γ=16 verify on GB10 is ~2.8s/call vs
        // K=γ=4's 421ms — at 38% accept (γ=16 prose), K=16 emits only
        // ~7 tokens in 2.8s = 2.5 tok/s, dragging mean throughput far
        // below the K=4 path. ADAPTIVE_MAX=4 hard-caps cutoff so the
        // scheduler always lands on a faster verify graph. Default 0 =
        // unbounded (legacy behavior, allows up to gamma_eff).
        let adaptive_max: usize = std::env::var("ATLAS_DFLASH_ADAPTIVE_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        // ATLAS_DFLASH_ADAPTIVE_PROBE_INTERVAL=<usize> (default 0 = off)
        //   Every N adaptive-engaged steps, force cutoff = gamma_eff
        //   (= no truncation, no masking from adaptive shrink) so the
        //   verifier observes the TRUE accept ceiling for the current
        //   context. Without this, adaptive truncate is self-limiting:
        //   a low-accept burst on prose shrinks γ_eff, which fills
        //   `accept_history` with small accept counts from the truncated
        //   steps, which keeps the cutoff small forever — even after
        //   the content turns predictable (counting, lists, repeating
        //   structure). Periodic γ_max reprobes break the trap: if the
        //   content is still hard, the reprobe step contributes one
        //   small accept count and we truncate again next step; if the
        //   content has become easy, the reprobe lands a long accept
        //   chain and the moving mean climbs back. The probe step
        //   itself bypasses both the adaptive shrink AND the
        //   `ATLAS_DFLASH_ADAPTIVE_MAX` ceiling — its whole purpose is
        //   to measure ceiling, not to be capped by one. Margin-gate
        //   cuts still apply (those are per-step confidence signals
        //   independent of history).
        let adaptive_probe_interval: usize =
            std::env::var("ATLAS_DFLASH_ADAPTIVE_PROBE_INTERVAL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

        // Run top-2 if the margin gate is active, so we have the second-best
        // logit value per row. Cost: one extra kernel launch over the same
        // logits buffer plus a γ_eff×2×4 byte D2H. Tiny vs the layer chain.
        let margin_top2: Option<Vec<f32>> = if margin_gate > 0.0 {
            let k_used = 2usize;
            let used_bytes = gamma_eff * k_used * 4;
            gpu.memset(self.scratch.topk_tokens_dev, 0, used_bytes)?;
            gpu.memset(self.scratch.topk_logits_dev, 0, used_bytes)?;
            ops::topk_bf16(
                gpu,
                self.kernels.topk,
                self.scratch.logits,
                self.scratch.topk_tokens_dev,
                self.scratch.topk_logits_dev,
                gamma_eff as u32,
                argmax_vocab,
                k_used as u32,
                stream,
            )?;
            gpu.synchronize(stream)?;
            let mut logits_bytes = vec![0u8; used_bytes];
            gpu.copy_d2h(self.scratch.topk_logits_dev, &mut logits_bytes)?;
            let logits: Vec<f32> = logits_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Some(logits)
        } else {
            None
        };

        // ── Step 6: D2H γ_eff × 4 bytes ──
        let mut host_buf = vec![0u8; gamma_eff * 4];
        gpu.synchronize(stream)?;
        gpu.copy_d2h(self.scratch.draft_tokens_dev, &mut host_buf)?;
        let mut drafts: Vec<u32> = host_buf
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Multi-step denoise: committed rows freeze their commit-time
        // prediction. On the final pass those rows were embedded as REAL
        // tokens, so the row's own re-prediction is out-of-distribution
        // for the drafter (trained to predict at MASK rows only) — keep
        // the token that earned the confidence commit instead. No-op when
        // ATLAS_DFLASH_DENOISE_STEPS=1 (committed is all-None) or when
        // ATLAS_DFLASH_DENOISE_FREEZE=0 (take last-pass argmax verbatim).
        if denoise_freeze {
            for (d, c) in drafts.iter_mut().zip(committed.iter()) {
                if let Some(t) = c {
                    *d = *t;
                }
            }
        }

        // Apply adaptive gamma + margin gate. Two behaviors are available
        // via ATLAS_DFLASH_ADAPTIVE_MODE:
        //   "mask"     (default): replace excess drafts with mask_token_id
        //              and KEEP drafts.len() == gamma_eff. The verify kernel
        //              still runs K=γ+1, the CUDA graph stays hot, and the
        //              accept-prefix logic terminates at the first mask
        //              position. Use this when the K=γ verify cost is fixed
        //              and you only want to stop wasted ctx pollution.
        //   "truncate": physically shorten the drafts vector to `cutoff`
        //              entries. The dispatcher in mtp_step.rs then routes
        //              to a smaller K verify path (K=2/3/4 graphed) when
        //              drafts.len() < 4, or to eager K=cutoff+1 path. May
        //              save substantial verify cycles when accept is low,
        //              at the cost of dropping the K=γ CUDA graph cache.
        //
        // Replaced positions get `mask_token_id` which is guaranteed to
        // mismatch the target's argmax → accept-prefix terminates at the
        // first replacement (mask mode), or the truncation point (truncate
        // mode — no replacement needed, the position just disappears).
        let adaptive_mode = std::env::var("ATLAS_DFLASH_ADAPTIVE_MODE")
            .unwrap_or_else(|_| "mask".to_string());
        let truncate_mode = adaptive_mode == "truncate";
        if adaptive_gamma || margin_gate > 0.0 {
            let mask = self.mask_token_id;
            // Bump the monotonic step counter once per adaptive-engaged call.
            // Used below to decide whether this step is a γ_max reprobe.
            dstate.propose_steps = dstate.propose_steps.wrapping_add(1);
            // A "probe step" forces cutoff = gamma_eff so the verifier sees
            // the true accept ceiling, bypassing both the history-mean shrink
            // and the ATLAS_DFLASH_ADAPTIVE_MAX ceiling. Margin-gate cuts
            // still apply later (per-step confidence, not history). We trigger
            // a probe every `adaptive_probe_interval` adaptive-engaged steps;
            // a value of 0 disables reprobing (legacy behavior). We also
            // require `accept_history_count >= 4` so the warmup phase (where
            // we're already running γ_max naturally) doesn't waste probes.
            let is_probe_step = adaptive_probe_interval > 0
                && dstate.accept_history_count >= 4
                && dstate.propose_steps.is_multiple_of(adaptive_probe_interval);
            // Compute the adaptive cutoff (= number of drafts to keep as-is).
            // gamma_eff is the post-cap drafter output size. cutoff <= gamma_eff.
            let mut cutoff = gamma_eff;
            if !is_probe_step && adaptive_gamma && dstate.accept_history_count >= 4 {
                let n = dstate.accept_history_count.min(dstate.accept_history.len());
                let sum: usize = dstate
                    .accept_history
                    .iter()
                    .take(n)
                    .map(|&v| v as usize)
                    .sum();
                let mean = sum / n; // integer mean is fine — we want conservative shrink
                let target = (mean + adaptive_slack).max(adaptive_min);
                cutoff = cutoff.min(target);
            }
            // Hard ceiling: K=γ=16 verify wall-time (2.8s/call on GB10) is
            // a worse deal than K=γ=4 even at high accept rates. When
            // ATLAS_DFLASH_ADAPTIVE_MAX > 0, never allow cutoff above it.
            // Independent of accept_history — applies even on the first
            // few calls (where history is still warming up). Skipped on
            // probe steps so we can actually measure γ_max behavior.
            if !is_probe_step && adaptive_max > 0 {
                cutoff = cutoff.min(adaptive_max);
            }
            // Margin gate: walk drafts in order, replace with mask at the first
            // low-confidence position (and all subsequent positions, since the
            // chain terminates at the first reject anyway — no point computing
            // attention for tokens we know will be discarded).
            if let Some(ref logits) = margin_top2 {
                for i in 0..gamma_eff {
                    let base = i * 2;
                    let top1 = logits[base];
                    let top2 = logits[base + 1];
                    let margin = top1 - top2;
                    if margin < margin_gate {
                        cutoff = cutoff.min(i);
                        break;
                    }
                }
            }
            if truncate_mode {
                // Shrink the vector so the scheduler downgrades the verify K.
                drafts.truncate(cutoff.max(1));
            } else {
                // Mask mode (default): replace positions [cutoff..gamma_eff].
                for i in cutoff..gamma_eff {
                    drafts[i] = mask;
                }
            }
            tracing::debug!(
                "DFlash adaptive: gamma_eff={} cutoff={} accept_count={} margin_gate={} mode={} probe={} step={}",
                gamma_eff,
                cutoff,
                dstate.accept_history_count,
                margin_gate,
                adaptive_mode,
                is_probe_step,
                dstate.propose_steps,
            );
        }
        // ATLAS_DFLASH_DEBUG_DUMP_FULL=1 (one-shot): log all γ drafts so
        // we can compare against the PyTorch reference run on the same
        // captured target_hidden. Static guard mirrors the input dump.
        static DRAFTS_DUMP_DONE: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !DRAFTS_DUMP_DONE.load(std::sync::atomic::Ordering::Relaxed)
            && std::env::var("ATLAS_DFLASH_DEBUG_DUMP_FULL")
                .ok()
                .as_deref()
                == Some("1")
        {
            tracing::info!(
                "DFLASH DUMP_FULL drafts (γ={}, last_token={}, position={}, eff_ctx={}): {:?}",
                self.gamma,
                last_token,
                position,
                eff_ctx,
                drafts,
            );
            DRAFTS_DUMP_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        if dump_all_layers {
            let path = "/tmp/atlas_final_drafts.bin";
            if !std::path::Path::new(path).exists() {
                let _ = std::fs::write(path, &host_buf);
                tracing::info!("DFLASH DUMP_ALL: drafts {drafts:?} → {path}");
            }
        }
        let _ = g; // suppress unused
        if kprofile {
            gpu.synchronize(stream)?;
            let tail_us = t_tail.elapsed().as_micros();
            let pre_us = t_pre_layers.duration_since(t_total).as_micros()
                + t_layers.duration_since(t_pre_layers).as_micros();
            let total_us = t_total.elapsed().as_micros();
            let agg = super::kprof_snapshot_layers();
            tracing::info!(
                "DFLASH_KP propose: total={:.2}ms pre+steps0-2={:.0}μs layers={:.2}ms tail={:.0}μs \
                 n_attn={} eff_ctx={} γ_eff={} | per-kernel-sum-over-{}-layers (μs): \
                 input_norm={} q_proj={} kv_ctx_copy={} kv_ctx_new={} kv_noise={} \
                 qk_norm={} rope={} cache_write={} prefill_attn={} \
                 o_proj={} resid1={} post_norm={} gate_up={} silu_mul={} down_proj={} resid2={}",
                total_us as f32 / 1000.0,
                pre_us,
                layers_us as f32 / 1000.0,
                tail_us,
                n_attn,
                eff_ctx,
                gamma_eff,
                self.layers.len(),
                agg.input_norm_us,
                agg.q_proj_us,
                agg.kv_ctx_copy_us,
                agg.kv_ctx_new_us,
                agg.kv_noise_us,
                agg.qk_norm_us,
                agg.rope_us,
                agg.cache_write_us,
                agg.prefill_attn_us,
                agg.o_proj_us,
                agg.resid1_us,
                agg.post_norm_us,
                agg.gate_up_us,
                agg.silu_mul_us,
                agg.down_proj_us,
                agg.resid2_us,
            );
        }
        Ok(drafts)
    }

    /// DDTree M4B v2: extract top-K tokens + logit values per drafter
    /// MASK row from the just-computed `self.scratch.logits` buffer.
    ///
    /// Must be called immediately after [`forward_block`] (before any
    /// subsequent call overwrites the logits scratch). Returns
    /// `(tokens, logits)` host vectors where each row of length `k`
    /// is sorted by logit descending.
    ///
    /// `gamma_eff` = the number of MASK rows that produced drafts (matches
    /// `drafts.len()` from forward_block). `k` is clamped to
    /// `super::DDTREE_TOP_K_MAX`.
    #[allow(dead_code)]
    pub(super) fn extract_topk_from_logits(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        stream: u64,
        gamma_eff: usize,
        k: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        use crate::layers::ops;

        let k_used = k.clamp(1, super::DDTREE_TOP_K_MAX);
        let lm_vocab = self.target_vocab_size.min(self.vocab_size) as u32;
        // Scratch is sized [γ, DDTREE_TOP_K_MAX] but we only fill γ_eff × k
        // rows. Zero just the rows we'll read so a partial write leaves no
        // stale data from a prior step.
        let used_bytes = gamma_eff * k_used * 4;
        gpu.memset(self.scratch.topk_tokens_dev, 0, used_bytes)?;
        gpu.memset(self.scratch.topk_logits_dev, 0, used_bytes)?;

        // Logits already populated by forward_block at self.scratch.logits
        // (shape [γ_eff, lm_vocab] BF16, row-major, contiguous).
        ops::topk_bf16(
            gpu,
            self.kernels.topk,
            self.scratch.logits,
            self.scratch.topk_tokens_dev,
            self.scratch.topk_logits_dev,
            gamma_eff as u32,
            lm_vocab,
            k_used as u32,
            stream,
        )?;

        gpu.synchronize(stream)?;

        let mut tokens_bytes = vec![0u8; used_bytes];
        let mut logits_bytes = vec![0u8; used_bytes];
        gpu.copy_d2h(self.scratch.topk_tokens_dev, &mut tokens_bytes)?;
        gpu.copy_d2h(self.scratch.topk_logits_dev, &mut logits_bytes)?;

        let tokens: Vec<u32> = tokens_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let logits: Vec<f32> = logits_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((tokens, logits))
    }
}
