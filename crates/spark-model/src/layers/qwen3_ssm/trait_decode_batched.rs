// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode_batched.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_batched_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize; // bytes per BF16
        let fp32 = 4usize; // bytes per FP32

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
        let qk_ch = (key_dim * 2) as u32; // Q+K channels for fused L2 norm
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size(); // 12288

        // ── 1. RMS norm + residual for K tokens ──
        let normed = ctx.buffers.norm_output();
        crate::kprof!(ctx.gpu, stream, "ssm_rms_norm_residual", {
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
            anyhow::Result::<()>::Ok(())
        })?;

        // ── 2+3. QKVZ projection (+ deinterleave if needed) ──
        // For sequential_qkvz (Qwen3.5): write directly to deinterleaved buffer.
        // For interleaved (80B): write to qkvz_out, then deinterleave per token.
        // NOTE: K=3 fused (rms_norm_residual + w4a16_gemv_batch3) kernel exists
        // in `w4a16_gemv_fused.cu` (`rms_norm_residual_w4a16_gemv_batch3`) and
        // is exposed via `ops::rms_norm_residual_w4a16_gemv_batch3` but the
        // 3*K*2 = 30 KiB smem footprint cuts occupancy on GB10 and the
        // per-token serial RMS pass loses the parallelism the original launch
        // gets from grid.x = 3, so wall-clock is ~7% worse on the AEON-27B
        // K=3 verify path. Kept off by default; re-enable when a parallel
        // per-token RMS variant is available.
        let deinterleaved = ctx.buffers.ssm_deinterleaved(); // [K, 12288] BF16
        let proj_dst = if self.sequential_qkvz {
            deinterleaved
        } else {
            ctx.buffers.ssm_qkvz()
        };
        if num_tokens == 3 {
            if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                crate::kprof!(ctx.gpu, stream, "ssm_qkvz_w4a16_gemv_batch3", {
                    ops::w4a16_gemv_batch3(
                        ctx.gpu,
                        self.w4a16_gemv_batch3_k,
                        normed,
                        nvfp4,
                        proj_dst,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
            } else {
                for t in 0..3u32 {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed.offset(t as usize * h * bf16),
                        &self.ssm.in_proj_qkvz,
                        proj_dst.offset(t as usize * qkvz_size * bf16),
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
            }
        } else if num_tokens == 2 {
            if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                ops::w4a16_gemv_batch2(
                    ctx.gpu,
                    self.w4a16_gemv_batch2_k,
                    normed,
                    nvfp4,
                    proj_dst,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &self.ssm.in_proj_qkvz,
                    proj_dst,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed.offset(h * bf16),
                    &self.ssm.in_proj_qkvz,
                    proj_dst.offset(qkvz_size * bf16),
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else if let Some(fp8) = self.qkvz_fp8 {
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
            )?;
        } else if let Some(ref nvfp4_t) = self.qkvz_nvfp4_t {
            // m128 halves B re-reads for large M (prefill); m16 halves
            // discarded MMA work for small-M K=γ verify (MTP K=3 → M=3,
            // DFlash γ=16 → M=16-17); m64 default otherwise.
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
                )?;
            } else if k <= 32
                && self.w4a16_gemm_t_m16_k.0 != 0
                && super::super::tc_nvfp4_m16_enabled()
            {
                ops::w4a16_gemm_n128_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
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
                )?;
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
            )?;
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
            for t in 0..(num_tokens as u32) {
                let src = proj_dst.offset(t as usize * qkvz_size * bf16);
                let dst = deinterleaved.offset(t as usize * qkvz_size * bf16);
                ops::deinterleave_qkvz(
                    ctx.gpu,
                    self.deinterleave_k,
                    src,
                    dst,
                    1,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )?;
            }
        }

        // ── 4. BA projection + GDN gates per token ──
        // BA output: ssm_ba buffer; gates: ssm_gates buffer [K, nv*2] FP32
        // Layout per token: [gate(nv), beta(nv)] → stride = 2*nv FP32 elements.
        // Must match gdn_decode_chunk2's gb_stride parameter.
        let gates_buf = ctx.buffers.ssm_gates(); // [K, gate(nv) + beta(nv)] FP32
        let gate_beta_stride = nv * 2 * fp32; // bytes per token in gates buffer
        let ba_size = ctx.config.ssm_ba_size(); // 64
        crate::kprof!(ctx.gpu, stream, "ssm_ba_proj_loop", {
            // K=3 verify fast path: collapse the 3-launch per-token loop into
            // a single `dense_gemv_bf16_batch3` launch. The per-token loop is
            // entirely launch-overhead bound (24 μs/launch × 3 × 48 SSM
            // layers ≈ 3.5 ms/verify wasted), while the underlying compute
            // (N=64 × K=hidden) is trivial. Gated by
            // ATLAS_SSM_BA_BATCHED=1 and requires the optional batch3 PTX
            // module to be present (NULL handle ⇒ fall through to the
            // per-token loop). `normed` is already a contiguous [K, h] BF16
            // buffer; `ssm_ba()` is already a contiguous [K, ba_size] BF16
            // buffer — both layouts match the kernel's [3, K]→[3, N]
            // contract exactly.
            if num_tokens == 3
                && super::super::ssm_ba_batched_enabled()
                && self.dense_gemv_batch3_k.0 != 0
            {
                ops::dense_gemv_batch3(
                    ctx.gpu,
                    self.dense_gemv_batch3_k,
                    normed,
                    &self.ssm.in_proj_ba,
                    ctx.buffers.ssm_ba(),
                    ba_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                for t in 0..(num_tokens as u32) {
                    let normed_t = normed.offset(t as usize * h * bf16);
                    let ba_out = ctx.buffers.ssm_ba().offset(t as usize * ba_size * bf16);
                    // Dense GEMV for BA projection (small: 64 outputs)
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed_t,
                        &self.ssm.in_proj_ba,
                        ba_out,
                        ba_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
            }
            anyhow::Result::<()>::Ok(())
        })?;
        crate::kprof!(ctx.gpu, stream, "ssm_compute_gdn_gates_loop", {
            for t in 0..(num_tokens as u32) {
                let ba_out = ctx.buffers.ssm_ba().offset(t as usize * ba_size * bf16);
                let gate_t = gates_buf.offset(t as usize * gate_beta_stride);
                let beta_t = gates_buf.offset(t as usize * gate_beta_stride + nv * fp32);
                ops::compute_gdn_gates(
                    ctx.gpu,
                    self.compute_gdn_gates_k,
                    ba_out,
                    self.ssm.a_log.weight,
                    self.ssm.dt_bias.weight,
                    gate_t,
                    beta_t,
                    1,
                    nv as u32,
                    nk as u32,
                    vpg as u32,
                    ba_size as u32,
                    stream,
                )?;
            }
            anyhow::Result::<()>::Ok(())
        })?;

        // ── 5-7. Conv1d + L2 norm + GDN per token (with intermediate checkpoints) ──
        // Reuse ssm_qkvz buffer for conv output (safe: deinterleave is done)
        let conv_out_buf = ctx.buffers.ssm_qkvz();
        let gdn_out_buf = ctx.buffers.attn_output();
        let h_bytes = self.h_state_bytes;
        let conv_bytes = self.conv_state_bytes;

        // Intermediates are pre-allocated from the pool (fixed GPU addresses for
        // CUDA graph stability). Verify they exist BEFORE we index into them — a
        // bare `debug_assert!` is a no-op in release and produces an opaque
        // out-of-bounds panic instead of an actionable error (see #bugs
        // m0t0chan EP=2 2026-04-05). Most-common cause: EP=2 worker started
        // without `--speculative --mtp-quantization` to mirror the head.
        if ssm_state.h_state_intermediates.len() < num_tokens
            || ssm_state.conv_state_intermediates.len() < num_tokens
        {
            anyhow::bail!(
                "SSM MTP intermediate buffers not allocated (h_state_intermediates.len()={}, \
                 conv_state_intermediates.len()={}, num_tokens={}). \
                 If this is an EP=2 worker, the head node is sending MTP verify commands \
                 but the worker was started without `--speculative` (and matching \
                 `--mtp-quantization`/`--num-drafts`). Add those flags to the worker invocation.",
                ssm_state.h_state_intermediates.len(),
                ssm_state.conv_state_intermediates.len(),
                num_tokens,
            );
        }

        let args = super::trait_decode_batched_conv_gdn::ConvGdnArgs {
            num_tokens,
            deinterleaved,
            gates_buf,
            conv_out_buf,
            gdn_out_buf,
            h_bytes,
            conv_bytes,
            qkvz_size,
            conv_dim,
            key_dim,
            value_dim,
            d_conv,
            qk_ch,
            nk,
            nv,
            kd,
            vd,
            bf16,
            fp32,
            stream,
        };
        crate::kprof!(ctx.gpu, stream, "ssm_conv_gdn_combined", {
            self.decode_batched_conv_gdn(ssm_state, ctx, &args)?;
            anyhow::Result::<()>::Ok(())
        })?;

        // ── 8. Gated RMS norm per token ──
        // Z gate is in deinterleaved at offset [Q_2048 + K_2048 + V_4096]
        // Normed out: reuse conv_out_buf (freed after GDN)
        let normed_out_buf = conv_out_buf;
        crate::kprof!(ctx.gpu, stream, "ssm_gated_rms_norm_loop", {
            for t in 0..(num_tokens as u32) {
                let gdn_t = gdn_out_buf.offset(t as usize * value_dim * bf16);
                let z_t = deinterleaved
                    .offset(t as usize * qkvz_size * bf16 + (key_dim * 2 + value_dim) * bf16);
                let normed_t = normed_out_buf.offset(t as usize * value_dim * bf16);
                ops::gated_rms_norm(
                    ctx.gpu,
                    self.gated_rms_norm_k,
                    gdn_t,
                    z_t,
                    &self.ssm.norm,
                    normed_t,
                    nv as u32,
                    vd as u32,
                    vd as u32,
                    eps,
                    vd as u32,
                    stream,
                )?;
            }
            anyhow::Result::<()>::Ok(())
        })?;

        // ── 9. Output projection → [K, H] ──
        let out_proj_buf = ctx.buffers.moe_output(); // [K, H] BF16
        crate::kprof!(ctx.gpu, stream, "ssm_out_proj", {
        if let Some(ref dense_out) = self.out_proj_dense {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed_out_buf,
                dense_out,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        } else if num_tokens == 3
            && super::super::ssm_out_batch3_enabled()
            && !self.ssm.out_proj.is_null()
        {
            // ATLAS_SSM_OUT_BATCH3=1 fast path: triple-GEMV at M=3 avoids the
            // ~96% wasted MMA work of `w4a16_gemm` M_TILE=64. Requires a valid
            // NVFP4 non-transposed `ssm.out_proj` (Qwen3.5 NVFP4 + AEON-27B
            // path). Falls through to FP8 / NVFP4_T below when off or when the
            // model variant has `ssm.out_proj == null` (FP8-only loaders).
            ops::w4a16_gemv_batch3(
                ctx.gpu,
                self.w4a16_gemv_batch3_k,
                normed_out_buf,
                &self.ssm.out_proj,
                out_proj_buf,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        } else if num_tokens == 2 {
            ops::w4a16_gemv_batch2(
                ctx.gpu,
                self.w4a16_gemv_batch2_k,
                normed_out_buf,
                &self.ssm.out_proj,
                out_proj_buf,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        } else if let Some(fp8) = self.out_proj_fp8 {
            if k > 128 {
                ops::fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_gemm_t_m128_k,
                    normed_out_buf,
                    fp8,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    normed_out_buf,
                    fp8,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        } else if let Some(ref nvfp4_t) = self.out_proj_nvfp4_t {
            if k <= 32
                && self.w4a16_gemm_t_m16_k.0 != 0
                && super::super::tc_nvfp4_m16_enabled()
            {
                ops::w4a16_gemm_n128_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16_k,
                    normed_out_buf,
                    nvfp4_t,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed_out_buf,
                    nvfp4_t,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed_out_buf,
                &self.ssm.out_proj,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        }
            anyhow::Result::<()>::Ok(())
        })?;

        // ── 10. Batched residual + post-norm, then MoE + residual ──
        // residual_add_rms_norm supports multi-token (grid.x = num_tokens)
        let normed2_base = ctx.buffers.norm_output();
        crate::kprof!(ctx.gpu, stream, "ssm_post_attn_resid_norm", {
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                out_proj_buf,
                &self.post_attn_norm,
                normed2_base,
                residual,
                num_tokens as u32,
                h as u32,
                eps,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;
        if num_tokens == 3 {
            // Fused K=3 MoE: 5 kernel launches instead of 15
            crate::kprof!(ctx.gpu, stream, "ssm_ffn_forward_k3", {
                self.ffn.forward_k3(normed2_base, ctx, stream)?;
                anyhow::Result::<()>::Ok(())
            })?;
            let moe_out = ctx.buffers.moe_output();
            crate::kprof!(ctx.gpu, stream, "ssm_ffn_residual_add", {
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    hidden,
                    moe_out,
                    (3 * h) as u32,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
        } else if num_tokens == 2 {
            // Fused K=2 MoE: 5 kernel launches instead of 10
            self.ffn.forward_k2(normed2_base, ctx, stream)?;
            // Batched residual add for 2 tokens (flat element-wise, 2*h elements)
            let moe_out = ctx.buffers.moe_output();
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (2 * h) as u32,
                stream,
            )?;
        } else {
            // Per-token MoE fallback for K!=2.
            // CONCURRENT-DECODE BUG (sibling of decode_multi_seq fix at line 1102):
            // hardcoded `t * h * 4` over-strides for BF16 hidden (GB10 default).
            let residual_elem = if ctx.config.use_fp32_residual() {
                4usize
            } else {
                2usize
            };

            // Batched K=γ FFN path: replaces the per-token GEMV loop
            // (each call re-reads ~134 MB of NVFP4 FFN weights from
            // LPDDR5X) with 3 GEMMs at M=num_tokens that load each
            // weight once per layer. Gated by `ATLAS_FFN_KGAMMA_M16=1`;
            // only available for dense FFN. The `residual_add_rms_norm`
            // above this branch already wrote `num_tokens` rows into
            // `normed2_base` contiguously, so the batched FFN can read
            // it directly without the per-token offsetting.
            //
            // Threshold n > 3 (re-verified 2026-05-21): the K=2 / K=3
            // branches above own num_tokens in {2, 3} via their fused
            // batch kernels. For num_tokens >= 4 the w4a16_gemm_t_m16
            // (M_TILE=16) kernel path is the fast option and was
            // re-validated to produce coherent output across the full
            // adaptive-truncate range {4..16}. A prior defensive gate
            // `>= 16` was a workaround for a transient drafter/adaptive
            // interaction that has since been resolved upstream; keeping
            // it suppressed the fast kernel on truncated-γ verifies,
            // costing the prose path 15-20 tok/s.
            let try_kgamma = (num_tokens as u32) > 3
                && super::super::ffn_kgamma_m16_enabled();
            let used_kgamma = if try_kgamma {
                let serviced = crate::kprof!(
                    ctx.gpu,
                    stream,
                    "ssm_ffn_kgamma_dense",
                    self.ffn
                        .forward_kgamma(normed2_base, num_tokens as u32, ctx, stream)
                )?;
                if serviced {
                    let moe_out = ctx.buffers.moe_output();
                    crate::kprof!(ctx.gpu, stream, "ssm_ffn_kgamma_resid", {
                        ops::residual_add(
                            ctx.gpu,
                            self.residual_add_k,
                            hidden,
                            moe_out,
                            (num_tokens as u32) * h as u32,
                            stream,
                        )?;
                        anyhow::Result::<()>::Ok(())
                    })?;
                }
                serviced
            } else {
                false
            };

            if !used_kgamma {
                crate::kprof!(ctx.gpu, stream, "ssm_ffn_per_token_loop_n17", {
                    for t in 0..(num_tokens as u32) {
                        let normed2 = normed2_base.offset(t as usize * h * bf16);
                        let moe_out = self.ffn.forward(normed2, ctx, stream)?;
                        let hidden_t = hidden.offset(t as usize * h * residual_elem);
                        ops::residual_add(
                            ctx.gpu,
                            self.residual_add_k,
                            hidden_t,
                            moe_out,
                            h as u32,
                            stream,
                        )?;
                    }
                    anyhow::Result::<()>::Ok(())
                })?;
            }
        }

        Ok(())
    }
}
