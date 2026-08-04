// SPDX-License-Identifier: AGPL-3.0-only

//! `MoeLayer::forward_km` — wide speculative verify MoE through the dedup'd
//! multi-row split-K `_t` kernels.
//!
//! The generalization of [`super::forward_k2`]'s unified-`_t` branch from the
//! MTP K=2 verify to the DSpark block verify (γ rows, up to
//! [`MOE_VERIFY_MAX_ROWS`]). The per-token fallback — `forward_batched` — runs
//! the whole expert dispatch once per row, so at γ=6 it streams each routed
//! expert's ~94 MB layer six times over. The `_m` kernels elect one leader
//! block per distinct expert id across ALL rows, so the bytes are read once
//! for every row routed to that expert.
//!
//! Measured union on DeepSeek-V4-Flash (`ATLAS_MOE_OVERLAP=1`): the six rows of
//! a DSpark block select far fewer than 6×top_k distinct experts — the
//! hash-routed layers pick the identical top-6 for every row, and the shared
//! expert and gate are duplicated outright.
//!
//! Everything else — routing, buffer layout, blend — is the per-token path's,
//! kernel for kernel, so verify numerics (and therefore acceptance) do not
//! move.

use super::*;

impl MoeLayer {
    /// Verify MoE for `num_tokens` rows in one dedup'd dispatch.
    ///
    /// Returns `Ok(false)` without touching any buffer when this layer or
    /// shape can't take the path, so the caller falls back to
    /// [`Self::forward_batched`]. On `Ok(true)` the output is at
    /// `moe_output()` `[num_tokens, H]`, exactly as the per-token path leaves
    /// it.
    pub fn forward_km(
        &self,
        input: DevicePtr, // [num_tokens, H] BF16 — normed MoE input
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let n = num_tokens as u32;
        if n < 2 || n > MOE_VERIFY_MAX_ROWS {
            // The width cap is a THROUGHPUT CLIFF, not a mere fallback: past
            // `MOE_VERIFY_MAX_ROWS` every row re-streams the whole routed
            // expert set. Measured on DeepSeek-V4-Flash / GB10, verify goes
            // 124.8 ms at 6 rows → 288.8 ms at 8 rows (~82 ms marginal per
            // row, vs ~16 ms/row inside the cap). Any widening of the drafter
            // (γ > 5) or of the tree (K_t > 6) silently loses far more than it
            // gains until the `_m` kernels are widened, so say so once.
            if n > MOE_VERIFY_MAX_ROWS {
                static WIDE_ONCE: std::sync::Once = std::sync::Once::new();
                WIDE_ONCE.call_once(|| {
                    tracing::warn!(
                        "MoE dedup verify declined: n={n} > MOE_VERIFY_MAX_ROWS={MOE_VERIFY_MAX_ROWS} \
                         — falling back to per-row forward_batched, which re-streams every routed \
                         expert once PER ROW (~82ms/row on GB10). Widen the `_m` kernels before \
                         raising the draft width."
                    );
                });
            }
            return Ok(false);
        }
        // The `_m` kernels read the transposed expert tables and compute the
        // shared expert in-kernel. Every regime that can't satisfy that — EP
        // (the shared half is added after the all-reduce), W3 Lloyd-Max and
        // BF16/FP8 experts (different kernel families), and the mixed
        // NVFP4-routed/BF16-shared Laguna config (the fused kernel can't do a
        // BF16 shared expert) — belongs to the per-token path.
        let is_ep = ctx.comm.is_some() && ctx.config.ep_world_size > 1;
        if is_ep
            || self.is_w3()
            || self.bf16_gate_weight_ptrs.is_some()
            || self.fp8_gate_weight_ptrs.is_some()
            || self.has_mixed_bf16_shared_expert()
            || !self.use_t_layout_for_decode()
        {
            return Ok(false);
        }
        let (Some(gate_t), Some(up_t), Some(down_t)) = (
            self.gate_ptrs_t.as_ref(),
            self.up_ptrs_t.as_ref(),
            self.down_ptrs_t.as_ref(),
        ) else {
            return Ok(false);
        };

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;

        // Bail before writing anything if the kernels/partial buffer can't
        // serve this width — `dispatch_splitk_m_t` re-checks, but the routing
        // below would already have clobbered the scratch by then.
        if !self.verify_ffn_is_batched(ctx, n) {
            return Ok(false);
        }

        // ATLAS_PROFILE: synchronized per-stage split, mirroring the per-token
        // path's `prof!` in `moe/forward.rs`. The per-token macro never covered
        // this path, so the wide verify's MoE showed up only inside the
        // VERIFY_PROFILE per-layer total — leaving the split between the
        // expert dispatch (bandwidth, irreducible at the batch's expert union)
        // and everything around it unmeasured.
        let profile = ctx.profile && !ctx.graph_capture;
        macro_rules! prof {
            ($label:expr, $body:expr) => {{
                if profile {
                    let t = std::time::Instant::now();
                    let r = $body;
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!("    MoE-km {}: {:.0}μs", $label, t.elapsed().as_micros());
                    r
                } else {
                    $body
                }
            }};
        }

        // ── Routing. Identical kernels to the per-token path, writing flat
        //    [num_tokens*top_k] indices/weights instead of one row's worth.
        let router_in = self.router_input(input, n, h, ctx, stream)?;
        let gate_logits = ctx.buffers.gate_logits(); // [n, num_experts] BF16
        prof!("gate", {
            if let Some(ref nvfp4) = self.gate_nvfp4 {
                ops::w4a16_gemm(
                    ctx.gpu,
                    self.w4a16_gemm,
                    router_in,
                    nvfp4,
                    gate_logits,
                    n,
                    num_experts,
                    h,
                    stream,
                )
            } else {
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm,
                    router_in,
                    &self.weights.gate,
                    gate_logits,
                    n,
                    num_experts,
                    h,
                    stream,
                )
            }
        })?;

        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch; // [n*top_k] u32
        let weights_dev = scratch.offset(n as usize * top_k as usize * 4); // [n*top_k] f32
        prof!(
            "route",
            self.route_rows_flat(
                ctx,
                gate_logits,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                n,
                stream,
            )
        )?;
        super::union_stats::maybe_sample_expert_union(
            ctx.gpu,
            indices_dev,
            n as usize,
            top_k as usize,
            stream,
        );

        // ── Expert dispatch. Same buffers the K=2 verify uses, widened.
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        // ⚠ logits buffer aliased — see the warning in moe/forward.rs.
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        let shared_down_out = ctx.buffers.attn_output();
        let output = ctx.buffers.moe_output();

        let null_qw = QuantizedWeight::null();
        let sh_gate_t = self.shared_gate_t.as_ref().unwrap_or(&null_qw);
        let sh_up_t = self.shared_up_t.as_ref().unwrap_or(&null_qw);
        let sh_down_t = self.shared_down_t.as_ref().unwrap_or(&null_qw);

        let dispatched = prof!(
            "exp_splitk_m_t",
            self.dispatch_splitk_m_t(
                ctx,
                input,
                expert_gate_out,
                expert_up_out,
                expert_down_out,
                shared_gate_scratch,
                shared_up_scratch,
                shared_down_out,
                indices_dev,
                gate_t,
                up_t,
                down_t,
                sh_gate_t,
                sh_up_t,
                sh_down_t,
                h,
                inter,
                top_k,
                n,
                stream,
            )
        )?;
        if !dispatched {
            // `verify_ffn_is_batched` above cleared the same predicate, so this
            // is unreachable barring an env flag flipping mid-forward. Fall
            // back rather than leave `moe_output` unwritten.
            return Ok(false);
        }

        prof!(
            "blend",
            ops::moe_weighted_sum_blend_batchn(
                ctx.gpu,
                self.moe_weighted_sum_blend_batch2,
                output,
                expert_down_out,
                weights_dev,
                shared_down_out,
                input,
                self.weights.shared_expert_gate.weight,
                h,
                top_k,
                h,
                n,
                stream,
            )
        )?;
        Ok(true)
    }

    /// Top-K (or static hash) routing for `n` rows into flat
    /// `[n*top_k]` index/weight arrays.
    ///
    /// Per-row launches of the SAME kernels `forward_batched` uses rather than
    /// the `_batched` variants: routing decides acceptance, so the wide verify
    /// must not risk diverging from the path the drafter was measured against.
    /// The launches are trivially small — `n` of them cost far less than one
    /// duplicated expert GEMV.
    #[allow(clippy::too_many_arguments)]
    fn route_rows_flat(
        &self,
        ctx: &ForwardContext,
        gate_logits: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        num_experts: u32,
        top_k: u32,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        let norm = ctx.config.norm_topk_prob;
        let scale = ctx.config.routed_scaling_factor as f32;
        for t in 0..n as usize {
            let gate_t = gate_logits.offset(t * num_experts as usize * 2);
            let idx_t = indices_dev.offset(t * top_k as usize * 4);
            let wgt_t = weights_dev.offset(t * top_k as usize * 4);
            if let Some(tid2eid) = self.tid2eid_dev {
                // DeepSeek-V4 hash routing: expert selection is static
                // `tid2eid[token_id]`; the learned gate only weights it. The
                // verify paths upload the K tokens in this same row order.
                let token_ids = ctx.token_ids.ok_or_else(|| {
                    anyhow::anyhow!(
                        "DeepSeek-V4 hash-MoE layer requires ForwardContext.token_ids (wide verify)"
                    )
                })?;
                ops::moe_hash_route(
                    ctx.gpu,
                    self.moe_hash_route_k,
                    gate_t,
                    tid2eid,
                    token_ids.offset(t * 4),
                    idx_t,
                    wgt_t,
                    num_experts,
                    top_k,
                    norm,
                    scale,
                    stream,
                )?;
            } else if let Some(bias) = self.correction_bias_dev {
                let kernel = if ctx.config.scoring_func == "sqrtsoftplus" {
                    self.moe_topk_sqrtsoftplus_k
                } else {
                    self.moe_topk_sigmoid_k
                };
                if ctx.config.scoring_func == "sqrtsoftplus" {
                    ops::moe_topk_sqrtsoftplus(
                        ctx.gpu,
                        kernel,
                        gate_t,
                        bias,
                        idx_t,
                        wgt_t,
                        num_experts,
                        top_k,
                        norm,
                        scale,
                        stream,
                    )?;
                } else {
                    ops::moe_topk_sigmoid(
                        ctx.gpu,
                        kernel,
                        gate_t,
                        bias,
                        idx_t,
                        wgt_t,
                        num_experts,
                        top_k,
                        norm,
                        scale,
                        stream,
                    )?;
                }
            } else {
                ops::moe_topk_softmax(
                    ctx.gpu,
                    self.moe_topk,
                    gate_t,
                    idx_t,
                    wgt_t,
                    num_experts,
                    top_k,
                    norm,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
