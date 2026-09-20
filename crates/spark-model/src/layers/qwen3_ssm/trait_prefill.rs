// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::prefill.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize, // SSM layers ignore — recurrent state requires all tokens
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if crate::layers::qwen4_prefill_moe::selected()? && num_tokens > 1 {
            return self.prefill_qwen4_moe_batch(
                hidden,
                residual,
                num_tokens,
                state,
                _seq_len_start,
                ctx,
                stream,
            );
        }
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize;
        let fp32 = 4usize;

        // Qualification switch used to bisect Qwen4 recurrent prefill from
        // full-attention prefill without changing either implementation.
        // ATLAS_QWEN4_SSM_PREFILL_ROWWISE also leaves this per-token route: it
        // keeps the same exact recurrence but hoists the token-parallel work
        // (QKVZ GEMM, out-proj, MoE FFN) out of the per-token loop. See the
        // rowwise branch below.
        if self.qwen4_attn_hyper.is_some()
            && std::env::var("ATLAS_QWEN4_SSM_PREFILL_BATCH")
                .ok()
                .as_deref()
                != Some("1")
            && std::env::var("ATLAS_QWEN4_SSM_PREFILL_ROWWISE")
                .ok()
                .as_deref()
                != Some("1")
        {
            let row_bytes = ctx.config.residual_width() * 2;
            // K=3 stays on the exact hyper/QKVZ/MoE families already
            // qualified by speculative verification. K>=4 enters grouped
            // prefill math and remains explicit opt-in; set the tile to 1 for
            // the fully serialized diagnostic baseline.
            let tile = std::env::var("ATLAS_QWEN4_SSM_PREFILL_TILE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| (2..=4096).contains(&v))
                .unwrap_or(1);
            let staging_rows = ctx.buffers.sizes().gate_logits / (ctx.config.hidden_size * 2);
            anyhow::ensure!(
                staging_rows > 0,
                "Qwen4 recurrent prefill has no gate-logit staging rows"
            );
            let mut t = 0;
            while t < num_tokens {
                let width = (num_tokens - t).min(tile).min(staging_rows);
                if width == 1 {
                    self.decode_inner(
                        hidden.offset(t * row_bytes),
                        residual.offset(t * row_bytes),
                        state,
                        _kv_cache,
                        _seq_len_start + t,
                        _block_table,
                        _disk_block_ids,
                        _disk_last_offloaded_per_layer,
                        ctx,
                        stream,
                    )?;
                } else {
                    self.decode_qwen4_batched_inner(
                        hidden.offset(t * row_bytes),
                        residual.offset(t * row_bytes),
                        width,
                        state,
                        _kv_cache,
                        _seq_len_start + t,
                        _block_table,
                        _disk_block_ids,
                        _disk_last_offloaded_per_layer,
                        DevicePtr::NULL,
                        DevicePtr::NULL,
                        0,
                        0,
                        ctx,
                        stream,
                    )?;
                }
                t += width;
            }
            return Ok(());
        }

        let qwen4_hyper = match (&self.qwen4_attn_hyper, &self.qwen4_mlp_hyper) {
            (Some(attn), Some(mlp)) => Some((attn, mlp)),
            (None, None) => None,
            _ => anyhow::bail!("incomplete Qwen4 SSM hyperconnection pair"),
        };

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd; // 2048
        let value_dim = nv * vd; // 4096
        let conv_dim = key_dim * 2 + value_dim; // 8192
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size(); // 12288

        // Profiling helper: sync + timestamp when ATLAS_PROFILE=1
        macro_rules! prof {
            ($label:expr, $t0:expr) => {
                if ctx.profile {
                    if let Some(t0) = $t0 {
                        ctx.gpu.synchronize(stream)?;
                        let elapsed = t0.elapsed().as_micros();
                        tracing::info!("  SSM prefill [{}] N={}: {}µs", $label, k, elapsed);
                    }
                }
            };
        }
        let mut t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Diagnostic: sync at entry to catch prior-layer errors
        if k > 4096 {
            tracing::info!("SSM prefill ENTRY: k={k} h={h}");
            ctx.gpu
                .synchronize(stream)
                .map_err(|e| anyhow::anyhow!("SSM prefill ENTRY: stream broken (k={k}): {e}"))?;
        }

        // ── 1. Input preparation for N tokens ──
        let hyper_gemm = std::env::var("ATLAS_QWEN4_HYPER_PREFILL_GEMM")
            .ok()
            .as_deref()
            == Some("1");
        let normed = if let Some((attn_hyper, _)) = qwen4_hyper {
            if hyper_gemm {
                attn_hyper.prepare_prefill(
                    hidden,
                    residual,
                    num_tokens,
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?
            } else {
                attn_hyper.prepare_prefill_exact(
                    hidden,
                    residual,
                    num_tokens,
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?
            }
        } else {
            let normed = ctx.buffers.norm_output();
            ops::rms_norm_residual(
                ctx.gpu,
                self.rms_norm_residual_k,
                hidden,
                &self.input_norm,
                normed,
                residual,
                k,
                h as u32,
                eps,
                stream,
            )?;
            normed
        };
        if k > 4096 {
            ctx.gpu
                .synchronize(stream)
                .map_err(|e| anyhow::anyhow!("SSM prefill: SYNC after rms_norm (k={k}): {e}"))?;
        }

        prof!("rms_norm_residual", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 2+3. QKVZ GEMM (+ deinterleave if needed) ──
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let proj_dst = if self.sequential_qkvz {
            deinterleaved
        } else {
            ctx.buffers.ssm_qkvz()
        };
        if let Some(fp8) = self.qkvz_fp8 {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                normed,
                fp8,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("ssm prefill: QKVZ FP8 GEMM failed (M={k}, N={qkvz_size}): {e}")
            })?;
        } else if let Some(ref nvfp4_t) = self.qkvz_nvfp4_t {
            if k > 128 {
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "ssm prefill: QKVZ m128 GEMM failed (M={k}, N={qkvz_size}): {e}"
                    )
                })?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!("ssm prefill: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
                })?;
            }
        } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed,
                nvfp4,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("ssm prefill: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
            })?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &self.ssm.in_proj_qkvz,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        }
        if !self.sequential_qkvz {
            ops::deinterleave_qkvz(
                ctx.gpu,
                self.deinterleave_k,
                proj_dst,
                deinterleaved,
                k,
                nk as u32,
                kd as u32,
                vpg as u32,
                vd as u32,
                stream,
            )?;
        }

        prof!("qkvz_gemm", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 4+5. Fused BA GEMM + GDN gates (token-parallel) ──
        // Replaces dense_gemm([M,K]×[N,K]) + compute_gdn_gates.
        // Vectorized uint4 loads, warp shuffle reduction, inline sigmoid/exp.
        // gate_out layout: [gate(nv), beta(nv)] per token, gate_stride = 2*nv FP32.
        let ba_size = ctx.config.ssm_ba_size(); // 64
        let gates_buf = ctx.buffers.ssm_gates();
        let gate_stride = nv * 2; // FP32 elements per token
        ops::dense_gemm_ba_gates_prefill(
            ctx.gpu,
            self.ba_gates_prefill_k,
            normed,
            &self.ssm.in_proj_ba,
            self.ssm.a_log.weight,
            self.ssm.dt_bias.weight,
            gates_buf,
            k,
            ba_size as u32,
            h as u32,
            h as u32,
            gate_stride as u32,
            nv as u32,
            vpg as u32,
            stream,
        )?;
        prof!("ba+gates", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 6. Batched conv1d for all N tokens (sequential per-channel in registers) ──
        // Reuse ssm_qkvz buffer for conv output (safe: deinterleave is done)
        let conv_out_buf = ctx.buffers.ssm_qkvz();
        let gdn_out_buf = ctx.buffers.attn_output();
        let normed_out_buf = conv_out_buf;
        let z_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
        let exact_f32_prefill = qwen4_hyper.is_some()
            && std::env::var("ATLAS_QWEN4_SSM_PREFILL_FP32")
                .ok()
                .as_deref()
                == Some("1")
            && self.conv1d_l2norm_f32_sequence_k.0 != 0
            && self.gdn_f32_sequence_nosnap_k.0 != 0
            && self.gated_rms_norm_f32_multi_seq_k.0 != 0;

        // Row-wise exact recurrence inside an otherwise fully batched prefill.
        //
        // Motivation: the two batched recurrent routes below (the WY/chunked
        // `gdn_prefill_*` kernels, and the FP32 sequence kernels behind
        // ATLAS_QWEN4_SSM_PREFILL_FP32) both produce incoherent output — and
        // they do so at a 23-token prompt, so the defect is wiring, not an
        // accumulation that only shows up on long prompts. The alternative that
        // *is* correct today is the per-token route at the top of this function,
        // but it pays for correctness by running the ENTIRE layer — norm, QKVZ
        // GEMM, out-proj and the MoE FFN — once per prompt token, which is why
        // measured prefill (~40 tok/s) is indistinguishable from decode.
        //
        // Only conv1d and the GDN update are actually sequential; everything
        // else in this function is already token-parallel. So keep the batched
        // work batched and step ONLY the recurrence row by row, using the exact
        // single-token kernels — the same three calls, in the same order, as the
        // row-by-row oracle in `trait_decode_batched.rs`, which is qualified by
        // speculative verification. Prefill has no rollback, so the per-row
        // state snapshots that oracle takes are omitted here.
        //
        // conv_f32 holds ONE row (reused each step); gdn_f32 sits in the tail of
        // that row, matching the decode oracle's layout exactly.
        let exact_rowwise_prefill = qwen4_hyper.is_some()
            && std::env::var("ATLAS_QWEN4_SSM_PREFILL_ROWWISE")
                .ok()
                .as_deref()
                == Some("1")
            && self.conv1d_l2norm_f32_k.0 != 0
            && self.gdn_f32_k.0 != 0
            && self.gated_rms_norm_f32_k.0 != 0;

        let conv_seq_rowwise = exact_rowwise_prefill
            && std::env::var("ATLAS_QWEN4_SSM_PREFILL_CONVSEQ")
                .ok()
                .as_deref()
                == Some("1")
            && self.conv1d_l2norm_f32_sequence_k.0 != 0
            && self.gated_rms_norm_f32_multi_seq_k.0 != 0;

        // ── Fail closed on the two unqualified batched recurrent routes ──
        //
        // Both produce FLUENT, PLAUSIBLE, WRONG output on Qwen4 — a 22-token
        // "What is 2+2?" returns unrelated prose rather than "4" (measured;
        // route receipts `batched_f32_sequence` and `batched_bf16_wy`). That is
        // the worst failure mode available: nothing errors, nothing looks
        // malformed, and the wrong answer silently becomes the context.
        //
        // The FP32 sequence route is not merely unlucky, it is used outside its
        // qualified window. `exact_flat_ssm_route` admits the very same
        // sequence kernels ONLY for `4 <= rows <= 32` (the speculative-verify
        // width band); prefill calls them with hundreds to thousands of rows,
        // where they have never been qualified. The bf16 WY route is
        // independently broken (F31/F32).
        //
        // One real defect was found and fixed on the way here — the conv1d
        // `qk_channels` argument was passing `nk * 2` (heads) instead of
        // `key_dim * 2` (channels), 32 instead of 4,096, so ~4,064 of the Q+K
        // channels skipped their L2 normalisation. Fixing it moved the failure
        // from empty output to coherent-but-off-topic, which proves it was real
        // and also proves it was not the only defect.
        //
        // Until the remaining defect is found, refuse rather than corrupt.
        // `ATLAS_QWEN4_SSM_PREFILL_ROWWISE=1` is the qualified fast path and is
        // named in the error so the operator has somewhere to go.
        if qwen4_hyper.is_some() && !exact_rowwise_prefill {
            let route = if exact_f32_prefill {
                "batched_f32_sequence"
            } else {
                "batched_bf16_wy"
            };
            // Only cite the row-window when it actually applies: this route is
            // measured wrong even at k=25, which is inside 4..=32, so claiming
            // the window as the cause there would be a false explanation.
            let window = if exact_f32_prefill && !(4..=32).contains(&num_tokens) {
                format!(
                    " Its FP32 sequence kernels are additionally used outside \
                     their qualified 4..=32 row window (see exact_flat_ssm_route); \
                     this call has {num_tokens} rows."
                )
            } else {
                String::new()
            };
            anyhow::bail!(
                "Qwen4 batched SSM prefill is unqualified and produces incorrect \
                 output (route={route}, k={k}): measured to answer a 22-token \
                 \"what is 2+2\" with unrelated prose.{window} Use \
                 ATLAS_QWEN4_SSM_PREFILL_ROWWISE=1 — batched everything except the \
                 recurrence, exact against the per-token oracle, and 2.6x faster — \
                 or unset ATLAS_QWEN4_SSM_PREFILL_BATCH/_FP32 for the per-token path."
            );
        }

        // One-shot receipt naming the recurrent branch actually taken. Twice in
        // this investigation a conclusion was drawn about a branch that was not
        // running (a silently-unqualified flag, and a mistyped kernel symbol),
        // so the branch now identifies itself rather than being inferred.
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static SAID: AtomicBool = AtomicBool::new(false);
            if !SAID.swap(true, Ordering::Relaxed) {
                let which = if conv_seq_rowwise {
                    "convseq_rowwise"
                } else if exact_rowwise_prefill {
                    "rowwise_exact"
                } else if exact_f32_prefill {
                    "batched_f32_sequence"
                } else {
                    "batched_bf16_wy"
                };
                eprintln!(
                    "ATLAS_SSM_PREFILL_ROUTE ENGAGED route={which} k={k} \
conv_seq_k={} gdn_nosnap_k={} gnorm_multi_k={}",
                    self.conv1d_l2norm_f32_sequence_k.0 != 0,
                    self.gdn_f32_sequence_nosnap_k.0 != 0,
                    self.gated_rms_norm_f32_multi_seq_k.0 != 0,
                );
            }
        }

        // Input: deinterleaved [N, qkvz_size], output: conv_out [N, conv_dim]
        // Conv1d processes QKV channels (first conv_dim of each token's qkvz_size)
        // ATLAS_QWEN4_SSM_PREFILL_CONVSEQ=1: hoist conv1d out of the per-token
        // loop as well, keeping ONLY the GDN update sequential.
        //
        // Row-wise costs three kernel launches per token per SSM layer, which is
        // why its prefill rate collapses with length (measured 120 tok/s at 1.2k
        // falling to 29.5 at 9.3k — 36 layers x 8,193 tokens x 3 is ~885k
        // sequential launches for a single chunk). conv1d is genuinely
        // token-parallel given `conv_state`: the sequence kernel carries the
        // state internally across its own token loop. Hoisting it and the final
        // gated norm leaves 1 + k + 1 launches instead of 3k.
        //
        // It is also a bisect. The batched route is still wrong after the
        // `qk_channels` fix (F56/F57), and the defect must be in either the conv
        // sequence kernel or the GDN sequence kernel. This arm uses the conv
        // sequence kernel with the exact per-row GDN: if it is correct, conv is
        // exonerated and the defect is in the GDN sequence kernel; if it is
        // wrong, conv is implicated.
        if conv_seq_rowwise {
            // RESULT OF THIS EXPERIMENT (kept because it is the bisect that
            // localised the remaining batched-prefill defect):
            //
            //   rowwise (single-token conv + per-row GDN)  -> "2+2" = "4"   CORRECT
            //   convseq (conv SEQUENCE  + per-row GDN)     -> " nginx..."   WRONG
            //
            // Same GDN, same gates, same norm; only conv1d differs. **The defect
            // is in the conv1d sequence kernel (or this call into it), not in the
            // GDN sequence kernel.** That is the next place to look, and it
            // survives the `qk_channels` fix of F56.
            //
            // It also refuted the motivation for the experiment: removing conv
            // and the gated norm from the loop drops launches from 3k to k+2 —
            // two thirds — and buys only ~3% (120.3 -> 123.5 at 1.2k, 27.4 ->
            // 28.4 at 11.8k). Launch overhead is NOT what makes Flash-Next
            // prefill collapse with length; the serialized GDN work is.
            //
            // Fail closed for the same reason as the other batched routes: it
            // produces fluent, plausible, wrong output.
            anyhow::bail!(
                "ATLAS_QWEN4_SSM_PREFILL_CONVSEQ is a diagnostic and produces                  incorrect output (k={k}): the conv1d sequence kernel is the                  remaining defect — with it, a 22-token \"what is 2+2\" answers                  with unrelated prose, while the otherwise-identical per-token                  conv answers \"4\". It is also not worth fixing for speed alone:                  it removes two thirds of the kernel launches for ~3%. Use                  ATLAS_QWEN4_SSM_PREFILL_ROWWISE=1 without it."
            );
            #[allow(unreachable_code)]
            let conv_f32 = ctx.buffers.ssm_conv_out_f32();
            let gdn_f32 = conv_f32.offset(conv_dim * fp32);
            let gate_beta_stride = nv * 2 * fp32;
            // One launch: all k rows of conv output, rows strided by qkvz_size,
            // conv_state advanced internally. Same argument list as the
            // qualified decode-batched caller, including qk_channels = key_dim*2.
            ops::conv1d_update_l2norm_f32_sequence(
                ctx.gpu,
                self.conv1d_l2norm_f32_sequence_k,
                ssm_state.conv_state,
                deinterleaved,
                &self.ssm.conv1d,
                conv_f32,
                DevicePtr::NULL,
                k,
                conv_dim as u32,
                d_conv as u32,
                (key_dim * 2) as u32,
                kd as u32,
                1e-6,
                qkvz_size as u32,
                qkvz_size as u32,
                0,
                stream,
            )?;
            // Sequential GDN, one row at a time, reading row t of conv output and
            // writing row t of the GDN output that lives in that row's tail.
            for t in 0..num_tokens {
                let row = conv_f32.offset(t * qkvz_size * fp32);
                let gate_t = gates_buf.offset(t * gate_beta_stride);
                ops::gdn_decode(
                    ctx.gpu,
                    self.gdn_f32_k,
                    ssm_state.h_state,
                    row,
                    row.offset(key_dim * fp32),
                    row.offset(key_dim * 2 * fp32),
                    gate_t,
                    gate_t.offset(nv * fp32),
                    gdn_f32.offset(t * qkvz_size * fp32),
                    1,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    stream,
                )?;
            }
            // One launch for all rows.
            ops::gated_rms_norm_f32_multi_seq(
                ctx.gpu,
                self.gated_rms_norm_f32_multi_seq_k,
                gdn_f32,
                z_base,
                &self.ssm.norm,
                normed_out_buf,
                nv as u32,
                k,
                vd as u32,
                eps,
                qkvz_size as u32,
                qkvz_size as u32,
                value_dim as u32,
                stream,
            )?;
            prof!("ssm_convseq_rowwise", t0);
            t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
        } else if exact_rowwise_prefill {
            let conv_f32 = ctx.buffers.ssm_conv_out_f32();
            let gdn_f32 = conv_f32.offset(conv_dim * fp32);
            let qk_ch = (key_dim * 2) as u32;
            let gate_beta_stride = nv * 2 * fp32; // bytes per token
            for t in 0..num_tokens {
                let qkv_t = deinterleaved.offset(t * qkvz_size * bf16);
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_f32_k,
                    ssm_state.conv_state,
                    qkv_t,
                    &self.ssm.conv1d,
                    conv_f32,
                    conv_dim as u32,
                    d_conv as u32,
                    1,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    stream,
                )?;
                let gate_t = gates_buf.offset(t * gate_beta_stride);
                ops::gdn_decode(
                    ctx.gpu,
                    self.gdn_f32_k,
                    ssm_state.h_state,
                    conv_f32,
                    conv_f32.offset(key_dim * fp32),
                    conv_f32.offset(key_dim * 2 * fp32),
                    gate_t,
                    gate_t.offset(nv * fp32),
                    gdn_f32,
                    1,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    stream,
                )?;
                let z_t = qkv_t.offset((key_dim * 2 + value_dim) * bf16);
                ops::gated_rms_norm(
                    ctx.gpu,
                    self.gated_rms_norm_f32_k,
                    gdn_f32,
                    z_t,
                    &self.ssm.norm,
                    normed_out_buf.offset(t * value_dim * bf16),
                    nv as u32,
                    vd as u32,
                    vd as u32,
                    eps,
                    vd as u32,
                    stream,
                )?;
            }
            prof!("ssm_rowwise_exact", t0);
            t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
        } else if exact_f32_prefill {
            // Preserve the FP32 recurrent contract used by ordinary decode,
            // but scan the entire prompt in one launch per operation. Prefill
            // has no per-token rollback, so both sequence kernels skip their
            // verification snapshot stores.
            let conv_f32 = ctx.buffers.ssm_conv_out_f32();
            let gdn_f32 = conv_f32.offset(conv_dim * fp32);
            ops::conv1d_update_l2norm_f32_sequence(
                ctx.gpu,
                self.conv1d_l2norm_f32_sequence_k,
                ssm_state.conv_state,
                deinterleaved,
                &self.ssm.conv1d,
                conv_f32,
                DevicePtr::NULL,
                k,
                conv_dim as u32,
                d_conv as u32,
                // `qk_channels`: the number of Q+K CHANNELS covered by the
                // fused L2 norm, i.e. `key_dim * 2` = (nk * kd) * 2.
                //
                // This previously passed `nk * 2` — 2x the number of key HEADS,
                // not channels. On Flash-Next that is 32 instead of 4,096, so
                // the kernel normalized 32 of 4,096 Q+K channels and left the
                // rest raw, corrupting the recurrent state from the first
                // token. It is why `ATLAS_QWEN4_SSM_PREFILL_FP32=1` produced
                // empty/garbage output at a 23-token prompt (F42) rather than
                // drifting with length. The qualified decode-batched caller
                // (`trait_decode_batched.rs`, `let qk_ch = (key_dim * 2)`) has
                // always passed channels, which is why the same kernel is exact
                // under speculative verification.
                (key_dim * 2) as u32,
                kd as u32,
                1e-6,
                qkvz_size as u32,
                qkvz_size as u32,
                0,
                stream,
            )?;
            prof!("conv1d_l2_f32_sequence", t0);
            t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            ops::gdn_decode_f32_sequence(
                ctx.gpu,
                self.gdn_f32_sequence_nosnap_k,
                ssm_state.h_state,
                conv_f32,
                conv_f32.offset(key_dim * fp32),
                conv_f32.offset(key_dim * 2 * fp32),
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_f32,
                DevicePtr::NULL,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                qkvz_size as u32,
                qkvz_size as u32,
                (nv * 2) as u32,
                qkvz_size as u32,
                0,
                stream,
            )?;
            prof!("gdn_f32_sequence", t0);
            t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            ops::gated_rms_norm_f32_multi_seq(
                ctx.gpu,
                self.gated_rms_norm_f32_multi_seq_k,
                gdn_f32,
                z_base,
                &self.ssm.norm,
                normed_out_buf,
                nv as u32,
                k,
                vd as u32,
                eps,
                qkvz_size as u32,
                qkvz_size as u32,
                value_dim as u32,
                stream,
            )?;
            prof!("gated_rms_norm_f32", t0);
            t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
        } else {
            ops::conv1d_update_prefill(
                ctx.gpu,
                self.conv1d_prefill_k,
                ssm_state.conv_state,
                deinterleaved,
                &self.ssm.conv1d,
                DevicePtr::NULL,
                conv_out_buf,
                conv_dim as u32,
                d_conv as u32,
                k,
                qkvz_size as u32,
                conv_dim as u32,
                stream,
            )?;
            prof!("conv1d", t0);
            t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };

            // ── 7. Batched L2 norm on Q,K for all N tokens ──
            // Q,K are the first 2*key_dim elements of each token's conv_out.
            // Stride between tokens in conv_out = conv_dim.
            ops::l2_norm(
                ctx.gpu,
                self.l2_norm_k,
                conv_out_buf,
                (nk * 2) as u32,
                kd as u32,
                1e-6,
                k,
                conv_dim as u32,
                stream,
            )?;
            prof!("l2_norm", t0);
            t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };

            // ── 8. GDN prefill via WY4-persistent kernel ──
            // Processes 4 tokens per iteration with WY algebraic correction, keeping
            // H state in shared memory for the entire sequence. 4× fewer sequential
            // state multiplications vs single-token kernel, preventing precision
            // drift at long context (28K+). Falls back to single-token persistent,
            // then split4 for unsupported configurations.
            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gb_stride = (nv * 2) as u32;

            // GDN profile gate (env: ATLAS_GDN_PROFILE=1). Sync + Instant before kernel
            // launch, paired with sync + record after.
            let __gdn_prof = {
                use std::sync::OnceLock;
                static C: OnceLock<bool> = OnceLock::new();
                *C.get_or_init(|| std::env::var("ATLAS_GDN_PROFILE").ok().as_deref() == Some("1"))
            };
            let __gdn_t = if __gdn_prof {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            let __gdn_path: &'static str;

            // wy32 (32 tokens/WY iteration) is the default GDN prefill kernel when
            // available — matches the two-phase `prefill_gdn_full` path (see
            // trait_prefill_gdn.rs:100, which already prefers wy32 unconditionally
            // for total>32). A/B-validated on AEON-Q36-27B: ~3.0 ms/layer vs ~6.3
            // ms/layer on wy4 (2.1×), 510-tok prefill 840→674 ms, output coherent.
            // Opt out with ATLAS_GDN_PREFILL_TUNED=0 to fall back to wy4.
            if std::env::var("ATLAS_GDN_WY32_WARP").ok().as_deref() == Some("1")
                && self.gdn_prefill_wy32_warp_k.0 != 0
                && k > 32
            {
                __gdn_path = "wy32_warp";
                let smem_bytes =
                    kd * vd * 4 + 32 * kd * 2 + 32 * kd * 2 + 4 * 4 + 32 * 32 * 4 + 32 * 4 + 32 * 4;
                let smem = (smem_bytes.div_ceil(256) * 256) as u32;
                ops::gdn_prefill_persistent_smem(
                    ctx.gpu,
                    self.gdn_prefill_wy32_warp_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gates_buf,
                    gates_buf.offset(nv * fp32),
                    gdn_out_buf,
                    1,
                    k,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32,
                    conv_dim as u32,
                    gb_stride,
                    smem,
                    stream,
                )?;
            } else if std::env::var("ATLAS_GDN_PREFILL_TUNED").ok().as_deref() != Some("0")
                && self.gdn_prefill_wy32_k.0 != 0
                && k > 32
            {
                __gdn_path = "wy32_tuned";
                // SMEM layout per kernel:
                //   H[kd*vd]FP32 + smem_k[32*kd]BF16 + smem_q[32*kd]BF16
                //   + smem_warp[4]FP32 + smem_kd[32*32]FP32
                //   + smem_g[32]FP32 + smem_bt[32]FP32
                // Round up to 256 B for alignment.
                let smem_bytes =
                    kd * vd * 4 + 32 * kd * 2 + 32 * kd * 2 + 4 * 4 + 32 * 32 * 4 + 32 * 4 + 32 * 4;
                let smem = (smem_bytes.div_ceil(256) * 256) as u32;
                ops::gdn_prefill_persistent_smem(
                    ctx.gpu,
                    self.gdn_prefill_wy32_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gates_buf,
                    gates_buf.offset(nv * fp32),
                    gdn_out_buf,
                    1,
                    k,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32,
                    conv_dim as u32,
                    gb_stride,
                    smem,
                    stream,
                )?;
            } else if self.gdn_prefill_persistent_wy4_k.0 != 0 {
                __gdn_path = "wy4";
                // WY4-persistent: H in shared memory, 4 tokens per iteration
                // smem = H[K_DIM*V_DIM] + 8*k/q buffers + warp sums + WY scalars
                let smem = (kd * vd * 4 + 8 * kd * 4 + 56) as u32;
                ops::gdn_prefill_persistent_smem(
                    ctx.gpu,
                    self.gdn_prefill_persistent_wy4_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gates_buf,
                    gates_buf.offset(nv * fp32),
                    gdn_out_buf,
                    1,
                    k,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32,
                    conv_dim as u32,
                    gb_stride,
                    smem,
                    stream,
                )?;
            } else if (256..=4096).contains(&k) && self.gdn_prefill_persistent_k.0 != 0 {
                __gdn_path = "persistent";
                ops::gdn_prefill_persistent(
                    ctx.gpu,
                    self.gdn_prefill_persistent_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gates_buf,
                    gates_buf.offset(nv * fp32),
                    gdn_out_buf,
                    1,
                    k,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32,
                    conv_dim as u32,
                    gb_stride,
                    stream,
                )?;
            } else {
                __gdn_path = "split4";
                ops::gdn_prefill_split4(
                    ctx.gpu,
                    self.gdn_prefill_split4_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gates_buf,
                    gates_buf.offset(nv * fp32),
                    gdn_out_buf,
                    1,
                    k,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32,
                    conv_dim as u32,
                    gb_stride,
                    stream,
                )?;
            }

            if let Some(__t) = __gdn_t {
                ctx.gpu.synchronize(stream)?;
                let __ns = __t.elapsed().as_nanos() as u64;
                tracing::info!(
                    "GDN_PROF call total={k} us={} path={}",
                    __ns / 1000,
                    __gdn_path
                );
            }

            prof!("gdn_prefill", t0);
            t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };

            // ── 9. Gated RMS norm (batched: all tokens × heads in one launch) ──
            ops::gated_rms_norm_prefill(
                ctx.gpu,
                self.gated_rms_norm_prefill_k,
                gdn_out_buf,
                z_base,
                &self.ssm.norm,
                normed_out_buf,
                nv as u32,
                vd as u32,
                eps,
                k,
                value_dim as u32,
                qkvz_size as u32,
                stream,
            )?;
            prof!("gated_rms_norm", t0);
            t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
        }

        // ── 10. Output projection GEMM: [N, 4096] × [4096, 2048] → [N, 2048] ──
        let out_proj_buf = ctx.buffers.moe_output();
        self.prefill_out_proj_dispatch(ctx, normed_out_buf, out_proj_buf, k, h, value_dim, stream)?;

        prof!("out_proj", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 11. Batched residual/hyper injection + MoE ──
        if let Some((attn_hyper, mlp_hyper)) = qwen4_hyper {
            attn_hyper.inject_saved_batched(
                hidden,
                out_proj_buf,
                residual,
                num_tokens,
                ctx.gpu,
                stream,
            )?;
            let ffn_inputs = if hyper_gemm {
                mlp_hyper.prepare_prefill(
                    hidden,
                    residual,
                    num_tokens,
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?
            } else {
                mlp_hyper.prepare_prefill_exact(
                    hidden,
                    residual,
                    num_tokens,
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?
            };
            self.ffn
                .forward_prefill(ffn_inputs, num_tokens, ctx, stream)?;
            mlp_hyper.inject_saved_batched(
                hidden,
                ctx.buffers.moe_output(),
                residual,
                num_tokens,
                ctx.gpu,
                stream,
            )?;
        } else {
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                out_proj_buf,
                &self.post_attn_norm,
                ctx.buffers.norm_output(),
                residual,
                num_tokens as u32,
                h as u32,
                eps,
                stream,
            )?;
            self.ffn
                .forward_prefill(ctx.buffers.norm_output(), num_tokens, ctx, stream)?;
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                ctx.buffers.moe_output(),
                (num_tokens * h) as u32,
                stream,
            )?;
        }

        prof!("moe_ffn", t0);

        if qwen4_hyper.is_some() {
            crate::model::qwen4_prefill_engagement::engage(
                crate::model::qwen4_prefill_engagement::PrefillPath::Ssm,
                num_tokens,
            )?;
        }

        Ok(())
    }
}
