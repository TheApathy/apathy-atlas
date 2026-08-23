// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash γ-block forward (Phase 2 kernel chain). Split out of
//! `dflash_head.rs` for file-size budget — body still exceeds the
//! 500 LoC target because the per-step kernel chain (fc → pos →
//! 8 drafter layers → final norm/lm_head/argmax → D2H) shares
//! many locals with no clean extraction boundary.

use anyhow::{Context, Result};

use super::{BlockDiffusionDraftHead, DflashProposerState};
use crate::layer::ForwardContext;

pub(super) fn checked_topk_difference(left: f32, right: f32) -> Result<f32> {
    if left.is_nan() || right.is_nan() {
        anyhow::bail!("top-K difference contains NaN");
    }
    if left == right {
        return Ok(0.0);
    }
    let difference = left - right;
    if difference.is_nan() {
        anyhow::bail!("top-K difference produced NaN");
    }
    Ok(difference)
}

pub(super) fn validate_topk_request(k: usize, vocab: usize) -> Result<usize> {
    if vocab == 0 {
        anyhow::bail!("top-K requires vocab > 0");
    }
    if vocab > u32::MAX as usize {
        anyhow::bail!("top-K vocab exceeds u32: {vocab}");
    }
    if k == 0 {
        anyhow::bail!("top-K requires k >= 1");
    }
    if k > super::DDTREE_TOP_K_MAX {
        anyhow::bail!("top-K requires k <= {}, got {k}", super::DDTREE_TOP_K_MAX);
    }
    if k > vocab {
        anyhow::bail!("top-K requires k <= vocab ({vocab}), got {k}");
    }
    Ok(k)
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct TopkLayout {
    pub(super) num_rows: u32,
    pub(super) vocab: u32,
    pub(super) k: usize,
    pub(super) bytes: usize,
}

pub(super) fn checked_topk_layout(
    num_rows: usize,
    scratch_rows: usize,
    vocab: usize,
    k: usize,
) -> Result<TopkLayout> {
    if num_rows > scratch_rows {
        anyhow::bail!("top-K rows {num_rows} exceed scratch capacity {scratch_rows}");
    }
    let k = validate_topk_request(k, vocab)?;
    let num_rows = u32::try_from(num_rows).context("top-K row count exceeds u32")?;
    let bytes = (num_rows as usize)
        .checked_mul(k)
        .and_then(|elements| elements.checked_mul(4))
        .context("top-K result byte length overflow")?;
    Ok(TopkLayout {
        num_rows,
        vocab: vocab as u32,
        k,
        bytes,
    })
}

impl BlockDiffusionDraftHead {
    /// `async_launch` (ATLAS_DFLASH_ASYNC, task #20): when true, every op is
    /// enqueued on `stream` (the dedicated propose stream, already ordered
    /// after the default stream via an event) and the function returns
    /// IMMEDIATELY after the noise-pass loop — the final synchronize + drafts
    /// D2H (and any host-side post-processing) is deferred to
    /// `collect_async_drafts_impl` at the top of the next scheduler step.
    /// The launch-eligibility gate guarantees none of the host-interactive
    /// features (markov / denoise>1 / margin gate / topk builders / debug
    /// dumps) are active on this path. Returns an empty Vec sentinel.
    pub(super) fn forward_block(
        &self,
        last_token: u32,
        position: usize,
        gamma_eff: usize,
        ctx: &ForwardContext,
        stream: u64,
        dstate: &mut DflashProposerState,
        async_launch: bool,
    ) -> Result<Vec<u32>> {
        use crate::layers::ops;

        let h = self.hidden_size as u32;
        let bf16 = 2usize;
        let gpu = ctx.gpu;

        // ── Kernel profiler (ATLAS_DFLASH_KERNEL_PROFILE=1 or ATLAS_FULL_PROFILE=1) ──
        // Lightweight per-phase timing using synchronize-and-Instant. Matches
        // `qwen3_ssm/ssm_forward.rs:prof!` pattern. Aggregates per-kernel μs
        // sums across all drafter layers via a thread-local accumulator,
        // logged at the end of forward_block. Adds ~20μs/sync × ~14 syncs/layer
        // when enabled — only use for measurement, not production.
        let kprofile = super::kernel_profile_enabled();
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
        // `gamma_eff` is resolved exactly once by the outer proposal budget.
        // Re-reading env/runtime state here could exceed the caller's public
        // maximum or the verify buffers after an async launch.
        if gamma_eff == 0 || gamma_eff > self.gamma {
            anyhow::bail!(
                "DFlash forward width {gamma_eff} is outside trained capacity 1..={}",
                self.gamma
            )
        }
        if gamma_eff
            .checked_add(1)
            .is_none_or(|verify_k| verify_k > self.physical_verify_k)
        {
            anyhow::bail!(
                "DFlash forward width {gamma_eff} exceeds physical verify K={}",
                self.physical_verify_k
            )
        }
        let logits_layout = super::logits_layout::LogitsLayout::new(
            self.gamma,
            self.target_vocab_size,
            self.vocab_size,
        )?;
        let _active_logits_bytes = logits_layout.active_bytes(gamma_eff)?;
        let row_layout = super::DraftRowLayout::for_family(self.checkpoint_family, gamma_eff);
        let noise_count = row_layout.query_rows;
        let n_attn = (eff_ctx + noise_count) as u32;

        // Scratch buffers are producer-owned. The active ranges are fully
        // overwritten by the embed, cache-copy/GEMM, attention, LM-head, or
        // argmax producers below; context rows that must be zero are cleared
        // immediately after their producer. Clearing the full allocations
        // here only adds launches and global-memory traffic to every propose.

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
                                needed,
                                p,
                            );
                        }
                    }
                    Ok(hf_bytes) => tracing::warn!(
                        "DFLASH HF_OVERRIDE file too small ({} < needed {})",
                        hf_bytes.len(),
                        needed,
                    ),
                    Err(e) => tracing::warn!("DFLASH HF_OVERRIDE read failed: {e}"),
                }
            }
            if zero_late > 0 && eff_ctx > 0 {
                // NOT converted to memset_async, unlike every other memset in
                // this file. Two reasons:
                //
                //  1. `base` is `dstate.ctx_hidden_acc`, which is written by
                //     the TARGET model's capture hook (`try_dflash_capture`,
                //     impl_b3.rs:863) on the target's stream — not by us on
                //     `stream`. Making this stream-ordered would order it
                //     against drafter work while leaving it unordered against
                //     the producer, which is the wrong guarantee to keep. The
                //     blocking form at least serialises against everything.
                //  2. ATLAS_DFLASH_ZERO_LATE_LAYERS defaults to 0, so this
                //     never runs in production and the conversion buys nothing.
                //
                // SEPARATE HAZARD if anyone does enable it: this is
                // `eff_ctx * n_zero` BLOCKING host syncs — at the champion's
                // ctx_window=4096 with n_zero=2 that is ~8k host round-trips
                // in one propose step. Batch it (one memset per contiguous
                // run, or a strided-zero kernel) before using it for anything
                // beyond a one-off A/B.
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
                        n_zero,
                        n_capture,
                        eff_ctx
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
            let fc_layernorm =
                std::env::var("ATLAS_DFLASH_FC_LAYERNORM").ok().as_deref() == Some("1");
            if new_fc_count > 0 {
                // Batched fast path (task #64): when NOT applying per-slice
                // FC-norm, every new context position reads a contiguous
                // [target_hidden_dim] source row from `base` (stride
                // ctx_slot_bytes = target_hidden_dim*bf16 = the GEMM K) and
                // writes a contiguous [hidden_size] dst row into `fc_proj`
                // (stride h*bf16 = the GEMM N). So the whole per-position GEMV
                // loop — one ~105MB fc-weight read PER position — collapses to a
                // SINGLE w4a16_gemm over M=new_fc_count rows: one weight read.
                // Bit-exact vs the loop (same weight, same (n=h, k=K) mapping,
                // M_TILE=64 handles arbitrary M with `if (r < M)` guards); the
                // gemv's n/k args map to the gemm's n/k unchanged. The
                // fc_layernorm path (default OFF) still needs per-position
                // RMSNorm into a 1-position scratch buffer, so it keeps the
                // loop below.
                let batched_nvfp4 = !fc_layernorm
                    && matches!(self.quant, super::DflashQuantization::Nvfp4)
                    && self.fc_nvfp4.is_some();
                if batched_nvfp4 {
                    let fc_q = self.fc_nvfp4.as_ref().unwrap();
                    let src = base.offset(old_fc_end * ctx_slot_bytes);
                    let dst = self
                        .scratch
                        .fc_proj
                        .offset(old_fc_count * self.hidden_size * bf16);
                    ops::w4a16_gemm(
                        gpu,
                        self.kernels.w4a16_gemm,
                        src,
                        fc_q,
                        dst,
                        new_fc_count as u32,
                        h,
                        target_hidden_dim as u32,
                        stream,
                    )?;
                }
                // Per-position fallback: fc_layernorm path (per-slice RMSNorm)
                // and the dense (non-NVFP4) path. Skipped entirely when the
                // batched NVFP4 fast path above already computed fc_proj.
                for i in 0..new_fc_count {
                    if batched_nvfp4 {
                        break;
                    }
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
                            let in_ptr = raw_slot.offset(layer_i * self.target_hidden_size * bf16);
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
                    self.scratch
                        .fc_proj
                        .offset(old_fc_count * self.hidden_size * bf16),
                    &self.hidden_norm,
                    self.scratch
                        .fc_proj
                        .offset(old_fc_count * self.hidden_size * bf16),
                    new_fc_count as u32,
                    h,
                    self.rms_norm_eps,
                    stream,
                )?;
                // Write new fc_proj into persistent cache.
                let (new_fc_start, new_fc_end) = self.cache_write_range(
                    gpu,
                    self.scratch
                        .fc_proj
                        .offset(old_fc_count * self.hidden_size * bf16),
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
        // kprofile sub-phase marker: end of Step 0 (fc_proj) host enqueue.
        let t_pre_fc_done = std::time::Instant::now();
        // Stream-ordered pinned H2D instead of `copy_h2d`: the sync variant
        // drains `default_stream` (the target verify stream) on every propose
        // — measured ~8.7 ms of hidden serialization per cycle. The async
        // copy enqueues after earlier work on `stream`, and `pos_pinned` is
        // only rewritten by the next propose after this pass has been
        // synchronized (sync path) or drained (async path), so ordering is
        // preserved without any host-side wait.
        let pos_n_bytes = pos_bytes.len();
        dstate.pos_pinned.as_mut_slice()[..pos_n_bytes].copy_from_slice(&pos_bytes);
        let pinned = dstate.pos_pinned.pinned_slice(pos_n_bytes)?;
        // SAFETY: `pinned` borrows the page-locked `pos_pinned` buffer owned by
        // `dstate`, which outlives this copy because `dstate` is only reused by
        // the next propose after this pass has been synchronized/drained. The
        // copy is stream-ordered on `stream`, so the driver reads the pinned
        // bytes only after preceding work on that stream, never concurrently
        // with this host-side write.
        unsafe {
            gpu.copy_h2d_pinned_async(pinned, self.scratch.position_ids, stream)?;
        }
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
        if self.checkpoint_family == crate::weight_loader::DrafterCheckpointFamily::Dspark
            && denoise_steps != 1
        {
            anyhow::bail!(
                "DSpark anchor-output layout currently supports exactly one denoise pass"
            );
        }
        let denoise_margin: f32 = std::env::var("ATLAS_DFLASH_DENOISE_MARGIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        let denoise_freeze =
            std::env::var("ATLAS_DFLASH_DENOISE_FREEZE").ok().as_deref() != Some("0");
        let argmax_vocab = self.target_vocab_size.min(self.vocab_size);
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
            self.run_noise_pass(&pass_args, &committed, pass, ctx, dstate)?;
            if pass + 1 == denoise_steps {
                break;
            }
            // DeepLoop GPU-side commit-all: run_noise_pass stages argmax and
            // reconstructs draft_tokens_dev on-stream for pass N+1 — no host
            // D2H, async-eligible. Skip the margin-gated D2H path entirely.
            if super::dflash_deeploop_enabled() {
                continue;
            }
            // Confidence feedback: top-2 over this pass's logits gives the
            // per-row top1−top2 margin; the argmax tokens are already in
            // draft_tokens_dev. Cost: one topk launch + ~γ·12 bytes D2H.
            let top2 = checked_topk_layout(gamma_eff, self.gamma, argmax_vocab, 2)?;
            let used_bytes = top2.bytes;
            // Stream-ordered: only consumer is the `topk_bf16` launch on
            // `stream` immediately below, and the host D2H is gated behind
            // the `gpu.synchronize(stream)` that follows it.
            gpu.memset_async(self.scratch.topk_tokens_dev, 0, used_bytes, stream)?;
            gpu.memset_async(self.scratch.topk_logits_dev, 0, used_bytes, stream)?;
            ops::topk_bf16(
                gpu,
                self.kernels.topk,
                self.scratch.logits,
                self.scratch.topk_tokens_dev,
                self.scratch.topk_logits_dev,
                top2.num_rows,
                top2.vocab,
                top2.k as u32,
                stream,
            )?;
            gpu.synchronize(stream)?;
            let mut tok_bytes = vec![0u8; gamma_eff * 4];
            gpu.copy_d2h(self.scratch.draft_tokens_dev, &mut tok_bytes)?;
            let mut top2_token_bytes = vec![0u8; used_bytes];
            let mut top2_logit_bytes = vec![0u8; used_bytes];
            gpu.copy_d2h(self.scratch.topk_tokens_dev, &mut top2_token_bytes)?;
            gpu.copy_d2h(self.scratch.topk_logits_dev, &mut top2_logit_bytes)?;
            let pass_tokens: Vec<u32> = tok_bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let (_, top2) = match Self::decode_topk_bytes(
                top2_token_bytes,
                top2_logit_bytes,
                gamma_eff,
                self.gamma,
                top2.k,
                argmax_vocab,
            ) {
                Ok(decoded) => decoded,
                Err(error) => {
                    tracing::warn!(
                        "DFlash denoise top-2 output is invalid ({error:#}); degrading to bootstrap"
                    );
                    return Ok(Vec::new());
                }
            };
            let mut newly_committed = 0usize;
            for i in 0..gamma_eff {
                if committed[i].is_some() {
                    continue;
                }
                let margin = match checked_topk_difference(top2[i * 2], top2[i * 2 + 1]) {
                    Ok(margin) => margin,
                    Err(error) => {
                        tracing::warn!(
                            "DFlash denoise top-2 row {i} is invalid ({error:#}); degrading to bootstrap"
                        );
                        return Ok(Vec::new());
                    }
                };
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

        // ── ATLAS_DFLASH_ASYNC deferred tail ──
        // All drafter kernels (incl. the per-row argmax into
        // `scratch.draft_tokens_dev`) are enqueued on the propose stream.
        // Return now; the synchronize + D2H below runs at collect time.
        // Eligibility guarantees the skipped host-side post-processing
        // (margin gate / markov / denoise freeze / dumps) is inactive.
        if async_launch {
            return Ok(Vec::new());
        }

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
        let adaptive_gamma =
            std::env::var("ATLAS_DFLASH_ADAPTIVE_GAMMA").ok().as_deref() == Some("1");
        let tps_router = std::env::var("ATLAS_DFLASH_TPS_ROUTER").ok().as_deref() == Some("1");
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
        let adaptive_probe_interval: usize = std::env::var("ATLAS_DFLASH_ADAPTIVE_PROBE_INTERVAL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        // Run top-2 if the margin gate is active, so we have the second-best
        // logit value per row. Cost: one extra kernel launch over the same
        // logits buffer plus a γ_eff×2×4 byte D2H. Tiny vs the layer chain.
        let margin_top2: Option<Vec<f32>> = if margin_gate > 0.0 {
            let top2 = checked_topk_layout(gamma_eff, self.gamma, argmax_vocab, 2)?;
            let used_bytes = top2.bytes;
            // Stream-ordered — same argument as the denoise-pass top-2 above.
            gpu.memset_async(self.scratch.topk_tokens_dev, 0, used_bytes, stream)?;
            gpu.memset_async(self.scratch.topk_logits_dev, 0, used_bytes, stream)?;
            ops::topk_bf16(
                gpu,
                self.kernels.topk,
                self.scratch.logits,
                self.scratch.topk_tokens_dev,
                self.scratch.topk_logits_dev,
                top2.num_rows,
                top2.vocab,
                top2.k as u32,
                stream,
            )?;
            gpu.synchronize(stream)?;
            let mut token_bytes = vec![0u8; used_bytes];
            let mut logits_bytes = vec![0u8; used_bytes];
            gpu.copy_d2h(self.scratch.topk_tokens_dev, &mut token_bytes)?;
            gpu.copy_d2h(self.scratch.topk_logits_dev, &mut logits_bytes)?;
            match Self::decode_topk_bytes(
                token_bytes,
                logits_bytes,
                gamma_eff,
                self.gamma,
                top2.k,
                argmax_vocab,
            ) {
                Ok((_, logits)) => Some(logits),
                Err(error) => {
                    tracing::warn!(
                        "DFlash adaptive top-2 output is invalid ({error:#}); degrading to bootstrap"
                    );
                    return Ok(Vec::new());
                }
            }
        } else {
            None
        };

        // ── DSpark VanillaMarkov head (auto-on when checkpoint ships it) ──
        // Keep the full correction chain device-resident: each chosen u32
        // feeds the next position's W1 gather on this same stream. This must
        // happen before the only D2H below. A loaded head is fail-closed; a
        // kernel/launch failure cannot silently substitute plain DFlash.
        if self.markov.is_some() {
            self.enqueue_markov_sequential(gpu, stream, last_token, gamma_eff)
                .context("DSpark device-resident Markov correction")?;
        }

        // ── Step 6: one final D2H of gamma token IDs ──
        // ATLAS_DFLASH_ASYNC_PROBE=1: split the propose wall into CPU enqueue
        // time (fn entry → here, all kernels launched) vs GPU drain time (the
        // synchronize below). The drain is the part an async second-stream
        // launch could overlap with the step's CPU tail. Measurement-only.
        let async_probe = crate::layers::dflash_async_probe();
        let probe_enqueue_us = if async_probe {
            t_total.elapsed().as_micros()
        } else {
            0
        };
        let mut host_buf = vec![0u8; gamma_eff * 4];
        // One ordered D2H + trailing sync.  The old pair (`synchronize`
        // followed by `copy_d2h`) paid two host-blocking stream syncs and the
        // copy ran on the default stream.  This preserves the same completion
        // guarantee while coalescing it to one sync on the producer stream.
        gpu.copy_d2h_on_stream(self.scratch.draft_tokens_dev, &mut host_buf, stream)?;
        if async_probe {
            tracing::info!(
                "ASYNC_PROBE propose: enqueue={probe_enqueue_us}μs gpu_total={}μs \
                 (eff_ctx={eff_ctx} n_attn={n_attn})",
                t_total.elapsed().as_micros(),
            );
        }
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
        let adaptive_mode =
            std::env::var("ATLAS_DFLASH_ADAPTIVE_MODE").unwrap_or_else(|_| "mask".to_string());
        // The throughput router must change the physical verify shape; masking
        // would retain the wide verifier cost and defeat its objective.
        let climbdrop_mode = std::env::var("ATLAS_DFLASH_TPS_ROUTER_MODE")
            .ok()
            .as_deref()
            == Some("climbdrop");
        let truncate_mode = adaptive_mode == "truncate" || tps_router || climbdrop_mode;
        if adaptive_gamma || margin_gate > 0.0 || tps_router || climbdrop_mode {
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
            if climbdrop_mode {
                // Climb/drop controller (llama.cpp PR #27210 algorithm): the
                // depth is the controller's current state; clamp to the
                // physical max. It climbs on consecutive full accepts and
                // falls back on weighted misses, so no probing is needed.
                let floor = std::env::var("ATLAS_DFLASH_TPS_ROUTER_FLOOR")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(4);
                let cap = std::env::var("ATLAS_DFLASH_TPS_ROUTER_MAX")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(gamma_eff);
                if dstate.climbdrop_router.score() == 0 || dstate.climbdrop_router.decisions() == 0
                {
                    dstate
                        .climbdrop_router
                        .reset(cap.min(gamma_eff), floor.min(gamma_eff));
                }
                cutoff = dstate.climbdrop_router.choose().min(gamma_eff);
            } else if tps_router {
                let widths_env = std::env::var("ATLAS_DFLASH_TPS_ROUTER_WIDTHS").ok();
                let widths =
                    super::throughput_router::parse_widths(widths_env.as_deref(), gamma_eff);
                let probe_interval = std::env::var("ATLAS_DFLASH_TPS_ROUTER_PROBE_INTERVAL")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(16);
                cutoff = dstate
                    .throughput_router
                    .choose(&widths, gamma_eff, probe_interval);
            }
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
            let mut invalid_margin = false;
            if let Some(ref logits) = margin_top2 {
                for i in 0..gamma_eff {
                    let base = i * 2;
                    let top1 = logits[base];
                    let top2 = logits[base + 1];
                    let margin = match checked_topk_difference(top1, top2) {
                        Ok(margin) => margin,
                        Err(error) => {
                            tracing::warn!(
                                "DFlash adaptive top-2 row {i} is invalid ({error:#}); cutting before row"
                            );
                            cutoff = cutoff.min(i);
                            invalid_margin = true;
                            break;
                        }
                    };
                    if margin < margin_gate {
                        cutoff = cutoff.min(i);
                        break;
                    }
                }
            }
            if tps_router || climbdrop_mode {
                // Attribute the observation to the actual physical width after
                // all safety/confidence caps, not merely the router's request.
                dstate.throughput_last_width = cutoff.max(1);
            }
            if truncate_mode {
                // Shrink the vector so the scheduler downgrades the verify K.
                drafts.truncate(if invalid_margin {
                    cutoff
                } else {
                    cutoff.max(1)
                });
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
        if kprofile {
            gpu.synchronize(stream)?;
            let tail_us = t_tail.elapsed().as_micros();
            let pre_us = t_pre_layers.duration_since(t_total).as_micros()
                + t_layers.duration_since(t_pre_layers).as_micros();
            let pre_fc_us = t_pre_fc_done.duration_since(t_pre_layers).as_micros();
            let pre_pos_us = t_layers.duration_since(t_pre_fc_done).as_micros();
            let fc_is_nvfp4 = self.fc_nvfp4.is_some();
            let total_us = t_total.elapsed().as_micros();
            let agg = super::kprof_snapshot_layers();
            // Publish into the ATLAS_FULL_PROFILE table so a single
            // `ATLAS_FULL_PROFILE=1` run reports drafter kernels alongside the
            // verify path's, under `draft_*` labels.
            for (label, us) in agg.labelled() {
                crate::full_profile::record(label, (us as u64).saturating_mul(1_000));
            }
            // The residue is the diagnostic that matters: `layers_us` is the
            // whole noise pass (embed, 6 layers, final norm, lm_head, argmax)
            // while the accumulator covers only the layer-internal launches.
            // Anything large here is propose time that is still unattributed —
            // report it explicitly rather than leaving it to subtraction.
            let attributed_us = agg.attributed_us();
            let unattributed_us = layers_us.saturating_sub(attributed_us);
            crate::full_profile::record(
                "draft_unattributed",
                (unattributed_us as u64).saturating_mul(1_000),
            );
            tracing::info!(
                "DFLASH_KP residue: layers={}μs attributed={}μs unattributed={}μs \
                 (embed + final_norm + lm_head + argmax + any unlabelled launch)",
                layers_us,
                attributed_us,
                unattributed_us,
            );
            tracing::info!(
                "DFLASH_KP propose: total={:.2}ms pre+steps0-2={:.0}μs (fc={:.0}μs nvfp4_fc={}, pos={:.0}μs) layers={:.2}ms tail={:.0}μs \
                 n_attn={} eff_ctx={} γ_eff={} | per-kernel-sum-over-{}-layers (μs): \
                 input_norm={} q_proj={} kv_ctx_copy={} kv_ctx_new={} kv_noise={} \
                 qk_norm={} rope={} cache_write={} prefill_attn={} \
                 o_proj={} resid1={} post_norm={} gate_up={} silu_mul={} down_proj={} resid2={} \
                 conv_prepare={} conv_finish={}",
                total_us as f32 / 1000.0,
                pre_us,
                pre_fc_us,
                fc_is_nvfp4,
                pre_pos_us,
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
                agg.conv_prepare_us,
                agg.conv_finish_us,
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
    /// `drafts.len()` from forward_block). Invalid `k` fails without launch.
    #[allow(dead_code)]
    pub(super) fn extract_topk_from_logits(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        stream: u64,
        gamma_eff: usize,
        k: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let lm_vocab = self.target_vocab_size.min(self.vocab_size);
        let layout = checked_topk_layout(gamma_eff, self.gamma, lm_vocab, k)?;
        self.enqueue_topk_on_stream(gpu, stream, gamma_eff, layout.k)?;
        let mut tokens_bytes = vec![0u8; layout.bytes];
        let mut logits_bytes = vec![0u8; layout.bytes];
        // Drain top-K once, then copy both results through the pageable-safe
        // synchronous D2H pair boundary. This retains two copies plus one
        // producer-stream sync without requiring page-locked Vec storage.
        gpu.copy_d2h_pair_on_stream(
            self.scratch.topk_tokens_dev,
            &mut tokens_bytes,
            self.scratch.topk_logits_dev,
            &mut logits_bytes,
            stream,
        )?;
        Self::decode_topk_bytes(
            tokens_bytes,
            logits_bytes,
            gamma_eff,
            self.gamma,
            layout.k,
            lm_vocab,
        )
    }

    /// Enqueue the top-K extraction kernel on `stream` without syncing or D2H.
    /// Used by the async propose path to defer collection until `collect_topk_d2h`.
    /// Logits must already be populated at `self.scratch.logits` by `forward_block`.
    pub(super) fn enqueue_topk_on_stream(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        stream: u64,
        gamma_eff: usize,
        k: usize,
    ) -> Result<()> {
        use crate::layers::ops;
        let lm_vocab = self.target_vocab_size.min(self.vocab_size);
        let layout = checked_topk_layout(gamma_eff, self.gamma, lm_vocab, k)?;
        // Stream-ordered. This function's contract is already "enqueue on
        // `stream`, do not sync" — both callers sync before reading
        // (`extract_topk_from_logits` immediately below; the async propose
        // path via `collect_async_drafts_impl`'s stream sync), so a blocking
        // memset here was contradicting the function's own doc comment.
        gpu.memset_async(self.scratch.topk_tokens_dev, 0, layout.bytes, stream)?;
        gpu.memset_async(self.scratch.topk_logits_dev, 0, layout.bytes, stream)?;
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
        )
    }

    /// D2H the top-K results from `scratch.topk_tokens_dev`/`topk_logits_dev`.
    /// The stream holding the kernel must already be synced by the caller
    /// (e.g. via `collect_async_drafts_impl`'s stream sync).
    pub(super) fn collect_topk_d2h(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        gamma_eff: usize,
        k: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let lm_vocab = self.target_vocab_size.min(self.vocab_size);
        let layout = checked_topk_layout(gamma_eff, self.gamma, lm_vocab, k)?;
        let mut tokens_bytes = vec![0u8; layout.bytes];
        let mut logits_bytes = vec![0u8; layout.bytes];
        gpu.copy_d2h(self.scratch.topk_tokens_dev, &mut tokens_bytes)?;
        gpu.copy_d2h(self.scratch.topk_logits_dev, &mut logits_bytes)?;
        Self::decode_topk_bytes(
            tokens_bytes,
            logits_bytes,
            gamma_eff,
            self.gamma,
            layout.k,
            lm_vocab,
        )
    }

    fn decode_topk_bytes(
        tokens_bytes: Vec<u8>,
        logits_bytes: Vec<u8>,
        num_rows: usize,
        scratch_rows: usize,
        k: usize,
        vocab: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let layout = checked_topk_layout(num_rows, scratch_rows, vocab, k)?;
        let expected_bytes = layout.bytes;
        if tokens_bytes.len() != expected_bytes || logits_bytes.len() != expected_bytes {
            anyhow::bail!(
                "top-K result byte length mismatch: tokens={} logits={} expected={expected_bytes}",
                tokens_bytes.len(),
                logits_bytes.len(),
            );
        }
        let tokens: Vec<u32> = tokens_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let logits: Vec<f32> = logits_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        for row in 0..num_rows {
            let start = row * layout.k;
            let row_tokens = &tokens[start..start + layout.k];
            let row_logits = &logits[start..start + layout.k];
            let marker_tokens = row_tokens
                .iter()
                .filter(|&&token| token == u32::MAX)
                .count();
            if marker_tokens > 0 {
                let exact_markers = row_tokens
                    .iter()
                    .zip(row_logits)
                    .filter(|&(token, score)| *token == u32::MAX && score.is_nan())
                    .count();
                if marker_tokens == layout.k && exact_markers == layout.k {
                    anyhow::bail!("top-K row {row} contains invalid marker");
                }
                anyhow::bail!("top-K row {row} contains partial marker");
            }
            if row_logits.iter().any(|score| score.is_nan()) {
                anyhow::bail!("top-K row {row} contains NaN");
            }
            if !row_logits.iter().any(|score| *score > f32::NEG_INFINITY) {
                anyhow::bail!("top-K row {row} has no usable score");
            }

            let mut seen = std::collections::HashSet::with_capacity(layout.k);
            for column in 0..layout.k {
                let token = row_tokens[column];
                let score = row_logits[column];
                if token >= layout.vocab {
                    anyhow::bail!(
                        "top-K row {row} token {token} is out of range for vocab {}",
                        layout.vocab
                    );
                }
                if !seen.insert(token) {
                    anyhow::bail!("top-K row {row} contains duplicate token {token}");
                }
                if column == 0 {
                    continue;
                }
                let previous_score = row_logits[column - 1];
                let previous_token = row_tokens[column - 1];
                if previous_score < score {
                    anyhow::bail!("top-K row {row} violates score order");
                }
                if previous_score == score && previous_token > token {
                    anyhow::bail!("top-K row {row} violates token order for equal scores");
                }
            }
        }

        Ok((tokens, logits))
    }
}

#[cfg(test)]
#[path = "topk_contract_tests.rs"]
mod topk_contract_tests;
