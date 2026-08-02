// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 7: residual + post-norm + MoE/dense FFN.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

/// Kill-switch for the pairwise batched MoE decode path (`ATLAS_MOE_PAIRWISE_DECODE=0`).
fn pairwise_moe_decode_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_MOE_PAIRWISE_DECODE").as_deref() != Ok("0"))
}

/// Smallest `n` that may use the batched `forward_kn` MoE path.
///
/// The dispatch below has a hole at n == 4: n == 2 goes to `forward_k2`,
/// n == 3 to `forward_k3`, n > 4 to `forward_kn`, and **n == 4 falls through
/// to the per-token sequential loop** — which re-reads every routed expert
/// weight 4 times instead of once. That is worth ~1.5x the expert bandwidth
/// (4 x top_k = 40 expert loads vs the measured union(4) = 26.8; see
/// bench/union_curve.py and laguna-expert-union-curve) and is the mechanism
/// behind the gamma<=4 throughput cliff.
///
/// Default 5 keeps the historical behaviour exactly. `ATLAS_MOE_BATCH_MIN_N=4`
/// closes the hole so K=4 verify steps get the batched path too.
fn batch_moe_min_n() -> usize {
    use std::sync::OnceLock;
    static MIN_N: OnceLock<usize> = OnceLock::new();
    *MIN_N.get_or_init(|| {
        std::env::var("ATLAS_MOE_BATCH_MIN_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5)
    })
}

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
        } else if !force_seq_ffn
            && !self.ffn.is_dense()
            && n >= batch_moe_min_n()
            && std::env::var("ATLAS_DFLASH_BATCH_MOE").ok().as_deref() == Some("1")
        {
            // EXPERIMENT (task #30, env-gated A/B): the per-token MoE loop below
            // is the wide-γ verify bottleneck (682ms/step, ~95% MoE). The authors'
            // comment (below) says the grouped-GEMM prefill path is a NET LOSS on a
            // 256-expert MoE at small batch (per-expert M~1, sort/permute overhead).
            // This branch routes the γ=16 verify MoE through `forward_prefill`
            // (grouped GEMM) so we can MEASURE that claim on the Laguna box: run the
            // verify with ATLAS_DFLASH_BATCH_MOE=1 vs unset and compare STEP_TIMING
            // verify_ms. Reversible: off by default → zero effect on any other path.
            let normed2 = fwd.buffers.norm_output();
            ops::residual_add_rms_norm(
                fwd.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_norm,
                normed2,
                residual,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
            // ATLAS_FUSED_ELEMWISE=1: fold the FFN residual add into the MoE
            // blend launch (bit-identical). `forward_kn_residual` returns
            // false when it couldn't fuse (fallback path / missing kernel) —
            // then the separate residual_add runs exactly as before.
            let residual_arg = ops::fused_elemwise_enabled().then_some(hidden);
            let residual_fused =
                self.ffn
                    .forward_kn_residual(normed2, n, residual_arg, fwd, stream)?;
            if !residual_fused {
                let moe_out = fwd.buffers.moe_output();
                ops::residual_add(
                    fwd.gpu,
                    self.residual_add_k,
                    hidden,
                    moe_out,
                    (n * h) as u32,
                    stream,
                )?;
            }
        } else if !force_seq_ffn && self.ffn.is_dense() {
            // WIDE-VERIFY BATCHED DENSE FFN (DFlash γ=16, n=17). The dense FFN
            // (Qwen3.6-27B is dense) batches over all n rows via
            // `forward_prefill`, reading gate/up/down ONCE instead of the
            // per-token loop below that re-read the FFN weights n× — the
            // measured wide-γ verify bottleneck (~844ms → target ~150ms).
            // Direct mirror of the `forward_k3` branch above, with count=n.
            //
            // DENSE ONLY: on a 256-expert MoE the grouped-GEMM is a net loss at
            // small batch, so MoE (and MLA / force_seq) fall through to the
            // per-token loop below — no regression for 122b/35b-a3b.
            let normed2 = fwd.buffers.norm_output();
            ops::residual_add_rms_norm(
                fwd.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_norm,
                normed2,
                residual,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
            self.ffn.forward_prefill(normed2, n, fwd, stream)?;
            let moe_out = fwd.buffers.moe_output();
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (n * h) as u32,
                stream,
            )?;
        } else if !force_seq_ffn && n > 2 && n % 2 == 0 && pairwise_moe_decode_enabled() {
            // BATCHED MoE DECODE (n = 4/8 after padding). The per-token loop
            // below re-reads every routed expert weight once per token; the
            // fused batch2 kernels process a token PAIR in 5 launches. Walk the
            // batch two tokens at a time and consume moe_output before the next
            // pair overwrites it. Falls back inside forward_k2 for layouts that
            // have no fused batch2 path, which is still no worse than per-token
            // (the gate GEMM is batched there too).
            let normed2 = fwd.buffers.norm_output();
            ops::residual_add_rms_norm(
                fwd.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_norm,
                normed2,
                residual,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
            for pair in 0..(n / 2) {
                let off = pair * 2 * h;
                self.ffn
                    .forward_k2(normed2.offset(off * bf16), fwd, stream)?;
                ops::residual_add(
                    fwd.gpu,
                    self.residual_add_k,
                    hidden.offset(off * 2),
                    fwd.buffers.moe_output(),
                    (2 * h) as u32,
                    stream,
                )?;
            }
        } else {
            // force_seq_ffn (MLA / batched-MoE-unsafe): per-token sequential.
            // CONCURRENT-DECODE BUG (sibling of qwen3_ssm.rs:1102 fix):
            // the per-seq hidden/residual stride must match the residual
            // element size. The residual stream is always BF16, so the stride
            // is `i * h * 2`; a hardcoded `i * h * 4` would over-stride into
            // the wrong batch slot for i>=1.
            let residual_elem = 2usize;
            let normed_base = fwd.buffers.norm_output();
            if n > 1 {
                // One launch for all n rows — the count=n kernel is the same
                // one the K=2/K=3 branches above use, row-for-row identical to
                // the n count=1 launches it replaces.
                ops::residual_add_rms_norm(
                    fwd.gpu,
                    self.residual_add_rms_norm_k,
                    hidden,
                    o_out,
                    &self.post_attn_norm,
                    normed_base,
                    residual,
                    n as u32,
                    h as u32,
                    eps,
                    stream,
                )?;
                // WIDE VERIFY (DSpark γ=6). Two reasons this must not go
                // through the per-token loop below:
                //
                //  1. Bandwidth. `forward` reads every routed expert's ~94 MB
                //     layer once PER ROW; the dedup'd `_t` split-K kernels read
                //     it once for the whole block.
                //  2. Correctness. `forward` is a single-token entry point —
                //     its hash-MoE routing reads `token_ids[0]` unconditionally
                //     (moe/forward.rs: "decode: single token at offset 0"), so
                //     rows 1.. were routed with row 0's experts. That is a
                //     silent verify-numerics bug on DeepSeek-V4's hash layers,
                //     and it costs acceptance directly.
                //
                // Distinct kernel family from the batch2/batch3 fused path that
                // `force_seq_ffn` exists to avoid, and both entry points fall
                // back internally when a layer isn't eligible.
                if self.ffn.forward_verify_rows(normed_base, n, fwd, stream)? {
                    ops::residual_add(
                        fwd.gpu,
                        self.residual_add_k,
                        hidden,
                        fwd.buffers.moe_output(),
                        (n * h) as u32,
                        stream,
                    )?;
                    return Ok(());
                }
            } else {
                ops::residual_add_rms_norm(
                    fwd.gpu,
                    self.residual_add_rms_norm_k,
                    hidden,
                    o_out,
                    &self.post_attn_norm,
                    normed_base,
                    residual,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
            }
            // Per-token FFN + residual — dense/no-FFN layers, and any MoE
            // width `forward_verify_rows` declined. Each forward() writes
            // moe_output[0]; consume it immediately before the next iteration
            // overwrites it.
            for i in 0..n {
                let hidden_i = hidden.offset(i * h * residual_elem);
                let normed2_i = normed_base.offset(i * h * bf16);
                let moe_out = self.ffn.forward(normed2_i, fwd, stream)?;
                ops::residual_add(
                    fwd.gpu,
                    self.residual_add_k,
                    hidden_i,
                    moe_out,
                    h as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
