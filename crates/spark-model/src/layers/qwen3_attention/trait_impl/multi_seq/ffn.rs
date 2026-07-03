// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 7: residual + post-norm + MoE/dense FFN.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    pub(super) fn ms_phase_ffn(&self, c: &MultiSeqCtx<'_>, o_out: DevicePtr) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            eps,
            bf16,
            hidden,
            residual,
            ..
        } = *c;

        if self.ffn.is_none() {
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                o_out,
                (n * h) as u32,
                stream,
            )?;
            return Ok(());
        }
        // MLA models (Mistral-Small-4) route the FFN through the
        // sequential per-token branch below, NOT the fused `forward_k2`
        // / `forward_k3` batched-MoE kernels. The batched-MoE K=2/K=3
        // path has a pre-existing crash for Mistral-Small-4's MoE config
        // (illegal address in `moe_expert_silu_down_shared_batch2`) — it
        // was never exercised because Mistral always ran at batch=1. The
        // sequential branch calls `FfnComponent::forward` (the proven
        // single-token MoE path used by `decode()`), processing each
        // sequence's normed input independently, so the batched MLA
        // attention path (issue #84) gets correct, isolated FFN output
        // without depending on the buggy batched-MoE kernels. Fixing the
        // batched-MoE kernel is tracked separately (out of #84 scope).
        let force_seq_ffn = self.mla.is_some();
        if n == 3 && !force_seq_ffn {
            let normed2 = fwd.buffers.norm_output();
            ops::residual_add_rms_norm(
                fwd.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_norm,
                normed2,
                residual,
                3,
                h as u32,
                eps,
                stream,
            )?;
            self.ffn.forward_k3(normed2, fwd, stream)?;
            let moe_out = fwd.buffers.moe_output();
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (3 * h) as u32,
                stream,
            )?;
        } else if n == 2 && !force_seq_ffn {
            let normed2 = fwd.buffers.norm_output();
            ops::residual_add_rms_norm(
                fwd.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_norm,
                normed2,
                residual,
                2,
                h as u32,
                eps,
                stream,
            )?;
            self.ffn.forward_k2(normed2, fwd, stream)?;
            let moe_out = fwd.buffers.moe_output();
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (2 * h) as u32,
                stream,
            )?;
        } else {
            // CONCURRENT-DECODE BUG (sibling of qwen3_ssm.rs:1102 fix):
            // the per-seq hidden/residual stride must match the actual
            // residual element size. When `use_fp32_residual()` is false
            // (BF16 hidden — GB10 default via HARDWARE.toml
            // ATLAS_HW_FP32_RESIDUAL=false), hardcoded `i * h * 4` would
            // over-stride into the wrong batch slot for i>=1.
            let residual_elem = if fwd.config.use_fp32_residual() {
                4usize
            } else {
                2usize
            };

            // Batched K=γ FFN path: replaces the n GEMV calls (each
            // re-reading ~134 MB of NVFP4 FFN weights) with 3 GEMMs at
            // M=n that load each weight once. Gated by
            // `ATLAS_FFN_KGAMMA_M16=1`; only available for dense FFN
            // (MoE has its own batched path via forward_k3 / shared
            // expert fused kernels). When the flag is off or the FFN
            // is MoE, falls through to the original per-token loop.
            //
            // Threshold n > 3 (re-verified 2026-05-21): the n == 2 / n
            // == 3 branches above own those cases via their fused
            // batch kernels. For n >= 4 the w4a16_gemm_t_m16 (M_TILE=16)
            // path is the fast option and was re-validated to produce
            // coherent output across the full adaptive-truncate range
            // {4..16}. A prior defensive gate `>= 16` was a workaround
            // for a transient drafter/adaptive interaction that has
            // since been resolved upstream; keeping it suppressed the
            // fast kernel on truncated-γ verifies, costing the prose
            // path 15-20 tok/s.
            let try_kgamma = n > 3 && crate::layers::ffn_kgamma_m16_enabled();
            let used_kgamma = if try_kgamma {
                // 1) Batched residual + norm for all n tokens. The
                // single-token slice in the fallback loop reads
                // residual_i / o_out_i / hidden_i per token; the
                // batched variant processes a contiguous [n, h] slab
                // identical to the n=2 / n=3 branches above.
                let normed2_base = fwd.buffers.norm_output();
                crate::kprof!(fwd.gpu, stream, "attn_ffn_kgamma_norm", {
                    ops::residual_add_rms_norm(
                        fwd.gpu,
                        self.residual_add_rms_norm_k,
                        hidden,
                        o_out,
                        &self.post_attn_norm,
                        normed2_base,
                        residual,
                        n as u32,
                        h as u32,
                        eps,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
                // 2) Batched FFN at M=n. Output lands in
                // `ctx.buffers.moe_output()`.
                let serviced = crate::kprof!(
                    fwd.gpu,
                    stream,
                    "attn_ffn_kgamma_dense",
                    self.ffn.forward_kgamma(normed2_base, n as u32, fwd, stream)
                )?;
                if serviced {
                    // 3) Batched residual add for all n*h elements.
                    let moe_out = fwd.buffers.moe_output();
                    crate::kprof!(fwd.gpu, stream, "attn_ffn_kgamma_resid", {
                        ops::residual_add(
                            fwd.gpu,
                            self.residual_add_k,
                            hidden,
                            moe_out,
                            (n * h) as u32,
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
                crate::kprof!(fwd.gpu, stream, "attn_ffn_per_token_loop_n17", {
                    for i in 0..n {
                        let hidden_i = hidden.offset(i * h * residual_elem);
                        let o_out_i = o_out.offset(i * h * bf16); // BF16 attn output
                        let residual_i = residual.offset(i * h * residual_elem);
                        let normed2 = fwd.buffers.norm_output().offset(i * h * bf16);
                        ops::residual_add_rms_norm(
                            fwd.gpu,
                            self.residual_add_rms_norm_k,
                            hidden_i,
                            o_out_i,
                            &self.post_attn_norm,
                            normed2,
                            residual_i,
                            1,
                            h as u32,
                            eps,
                            stream,
                        )?;
                        let moe_out = self.ffn.forward(normed2, fwd, stream)?;
                        ops::residual_add(
                            fwd.gpu,
                            self.residual_add_k,
                            hidden_i,
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
