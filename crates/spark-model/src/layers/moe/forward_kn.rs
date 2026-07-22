// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::forward_kn — wide DFlash verify (K=γ, e.g. 16 tokens) batched MoE.
//!
//! `forward_k16`: processes `num_tokens` tokens through the 256-expert NVFP4 MoE
//! using the `batchN` kernels — a `num_tokens` generalization of the proven
//! `batch2` (non-transposed) decode path. Per-block math is BYTE-IDENTICAL to the
//! per-token decode MoE, so the target argmax matches the per-token path →
//! speculative acceptance is preserved. The win is parallelism: all
//! `num_tokens*(top_k+1)` expert-GEMV blocks launch in ONE grid (vs the per-token
//! loop's `num_tokens` serial launches leaving ~92% of the SMs idle). NVFP4
//! non-transposed regime (Laguna). Falls back to per-token `forward_batched`
//! otherwise (bf16/fp8 experts, EP, t-layout, or missing kernel).

use anyhow::Context as _;

use super::*;

impl MoeLayer {
    /// Batched MoE for `num_tokens` tokens (wide DFlash verify). Output at
    /// `moe_output()` [num_tokens, H]. Faithful to per-token decode numerics.
    pub fn forward_kn(
        &self,
        input: DevicePtr, // [num_tokens, H] BF16 — normed MoE input
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Fast faithful path requires: NVFP4 non-transposed experts (not bf16 /
        // not fp8), dense router gate (Laguna: gate_nvfp4=None), non-EP, and the
        // batchN kernel present. Anything else → proven per-token path.
        let is_ep = ctx.comm.is_some() && ctx.config.ep_world_size > 1;
        let can_fast = !is_ep
            && self.gate_nvfp4.is_none()
            && self.bf16_gate_weight_ptrs.is_none()
            && self.fp8_gate_weight_ptrs.is_none()
            && self.moe_expert_gate_up_shared_batchn_k.0 != 0;

        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    "FORWARD_KN: can_fast={} (gate_dense={}, not_bf16={}, not_fp8={}, handle={}) is_ep={} bias={} scoring={} num_tokens={}",
                    can_fast,
                    self.gate_nvfp4.is_none(),
                    self.bf16_gate_weight_ptrs.is_none(),
                    self.fp8_gate_weight_ptrs.is_none(),
                    self.moe_expert_gate_up_shared_batchn_k.0 != 0,
                    is_ep,
                    self.correction_bias_dev.is_some(),
                    ctx.config.scoring_func,
                    num_tokens,
                );
            }
        }
        if !can_fast {
            return self.forward_batched(input, num_tokens, ctx, stream);
        }

        let n = num_tokens as u32;
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;

        // ATLAS_KN_DIAG=1: sync after each op so the FIRST failing checkpoint's
        // context names the faulting kernel (localizes the K=γ illegal address).
        // Synchronize is illegal during CUDA graph capture (error 900), so the
        // checkpoints only arm in eager mode (pair with ATLAS_DFLASH_DEBUG_NO_GRAPH=1).
        let kn_diag = !ctx.graph_capture
            && std::env::var("ATLAS_KN_DIAG").ok().as_deref() == Some("1");
        macro_rules! kn_ck {
            ($label:expr) => {
                if kn_diag {
                    ctx.gpu.synchronize(stream).context(concat!("KN: ", $label))?;
                }
            };
        }
        kn_ck!("entry");

        // 1. Router gate: [num_tokens, num_experts] (dense, Laguna).
        let router_in = self.router_input(input, n, h, ctx, stream)?;
        let gate_logits = ctx.buffers.gate_logits();
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
        )?;
        kn_ck!("gate_dense_gemm");

        // 2. Batched top-K → [num_tokens*top_k] indices + weights. Match the
        // per-token routing exactly (sigmoid+bias / sqrtsoftplus / softmax) so
        // the verify argmax equals the decode path's.
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch; // [num_tokens*top_k] u32
        let weights_dev = scratch.offset(num_tokens * top_k as usize * 4); // f32
        if let Some(bias) = self.correction_bias_dev {
            if ctx.config.scoring_func == "sqrtsoftplus" {
                for t in 0..num_tokens {
                    ops::moe_topk_sqrtsoftplus(
                        ctx.gpu,
                        self.moe_topk_sqrtsoftplus_k,
                        gate_logits.offset(t * num_experts as usize * 2),
                        bias,
                        indices_dev.offset(t * top_k as usize * 4),
                        weights_dev.offset(t * top_k as usize * 4),
                        num_experts,
                        top_k,
                        ctx.config.norm_topk_prob,
                        ctx.config.routed_scaling_factor as f32,
                        stream,
                    )?;
                }
            } else {
                ops::moe_topk_sigmoid_batched(
                    ctx.gpu,
                    self.moe_topk_sigmoid_batched_k,
                    gate_logits,
                    bias,
                    indices_dev,
                    weights_dev,
                    num_experts,
                    top_k,
                    ctx.config.norm_topk_prob,
                    ctx.config.routed_scaling_factor as f32,
                    n,
                    stream,
                )?;
            }
        } else {
            ops::moe_topk_softmax_batched(
                ctx.gpu,
                self.moe_topk_batched,
                gate_logits,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                n,
                stream,
            )?;
        }

        kn_ck!("topk");
        // 3-5. Fused expert dispatch — one grid each, all num_tokens tokens.
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        let output = ctx.buffers.moe_output();
        // Shared-expert scratch carved from the TAIL of the (DFlash-sized) expert
        // buffers. The old m-sized logits/ssm_qkvz/attn_output scratch overflow at
        // K=γ (and ssm_qkvz is ~0 for 0-SSM Laguna). The k_max floor in sizes.rs
        // (dflash_k=20) makes room for num_tokens*top_k routed + num_tokens shared.
        let routed_inter_bytes = num_tokens * top_k as usize * inter as usize * 2;
        let routed_hidden_bytes = num_tokens * top_k as usize * h as usize * 2;
        let shared_gate_scratch = expert_gate_out.offset(routed_inter_bytes);
        let shared_up_scratch = expert_up_out.offset(routed_inter_bytes);
        let shared_down_out = expert_down_out.offset(routed_hidden_bytes);

        let batch_block = if ctx.config.hidden_size >= 3072 { 256u32 } else { 128u32 };
        ops::moe_expert_gate_up_shared_batchn(
            ctx.gpu,
            self.moe_expert_gate_up_shared_batchn_k,
            input,
            self.gate_ptrs.packed_ptrs,
            self.gate_ptrs.scale_ptrs,
            self.gate_ptrs.scale2_vals,
            expert_gate_out,
            self.up_ptrs.packed_ptrs,
            self.up_ptrs.scale_ptrs,
            self.up_ptrs.scale2_vals,
            expert_up_out,
            indices_dev,
            &self.weights.shared_expert.gate_proj,
            shared_gate_scratch,
            &self.weights.shared_expert.up_proj,
            shared_up_scratch,
            inter,
            h,
            top_k,
            n,
            batch_block,
            stream,
        )?;
        kn_ck!("gate_up_batchn");
        ops::moe_expert_silu_down_shared_batchn(
            ctx.gpu,
            self.moe_expert_silu_down_shared_batchn_k,
            expert_gate_out,
            expert_up_out,
            self.down_ptrs.packed_ptrs,
            self.down_ptrs.scale_ptrs,
            self.down_ptrs.scale2_vals,
            expert_down_out,
            indices_dev,
            shared_gate_scratch,
            shared_up_scratch,
            &self.weights.shared_expert.down_proj,
            shared_down_out,
            h,
            inter,
            top_k,
            n,
            batch_block,
            stream,
        )?;
        kn_ck!("silu_down_batchn");
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
        )?;
        kn_ck!("blend_batchn");

        // ATLAS_KN_CMP=1 (eager only): one-shot numerical compare of the batched
        // path against the proven per-token forward() for token 0 — names the
        // diverging stage (router gemm / topk / expert kernels) in one serve run.
        if !ctx.graph_capture
            && std::env::var("ATLAS_KN_CMP").ok().as_deref() == Some("1")
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static CMP_DONE: AtomicBool = AtomicBool::new(false);
            if !CMP_DONE.swap(true, Ordering::Relaxed) {
                for t in [0, num_tokens / 2, num_tokens - 1] {
                    self.kn_compare_token(
                        t,
                        input,
                        gate_logits,
                        indices_dev,
                        weights_dev,
                        output,
                        num_experts,
                        top_k,
                        h,
                        ctx,
                        stream,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Diagnostic: compare batched (already computed) vs per-token reference for
    /// token `t`. Reads back the batched results, reruns token `t` through the
    /// proven `forward()` (clobbers shared MoE buffers — safe, results saved
    /// first), and logs per-stage max-diffs.
    #[allow(clippy::too_many_arguments)]
    fn kn_compare_token(
        &self,
        t: usize,
        input: DevicePtr,
        gate_logits: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        output: DevicePtr,
        num_experts: u32,
        top_k: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let e = num_experts as usize;
        let tk = top_k as usize;
        let hh = h as usize;
        ctx.gpu.synchronize(stream)?;

        let read_bf16 = |ptr: DevicePtr, count: usize| -> Result<Vec<f32>> {
            let mut buf = vec![0u8; count * 2];
            ctx.gpu.copy_d2h(ptr, &mut buf)?;
            Ok(buf
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect())
        };
        let read_u32 = |ptr: DevicePtr, count: usize| -> Result<Vec<u32>> {
            let mut buf = vec![0u8; count * 4];
            ctx.gpu.copy_d2h(ptr, &mut buf)?;
            Ok(buf
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        let read_f32 = |ptr: DevicePtr, count: usize| -> Result<Vec<f32>> {
            let mut buf = vec![0u8; count * 4];
            ctx.gpu.copy_d2h(ptr, &mut buf)?;
            Ok(buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        let max_diff = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };

        // Save batched token-t results.
        let b_logits = read_bf16(gate_logits.offset(t * e * 2), e)?;
        let b_idx = read_u32(indices_dev.offset(t * tk * 4), tk)?;
        let b_w = read_f32(weights_dev.offset(t * tk * 4), tk)?;
        let b_out = read_bf16(output.offset(t * hh * 2), hh)?;

        // Reference: per-token forward() on token t's input row.
        let ref_out_ptr = self.forward(input.offset(t * hh * 2), ctx, stream)?;
        ctx.gpu.synchronize(stream)?;
        let r_logits = read_bf16(ctx.buffers.gate_logits(), e)?;
        let r_idx = read_u32(ctx.buffers.scratch(), tk)?;
        let r_w = read_f32(ctx.buffers.scratch().offset(tk * 4), tk)?;
        let r_out = read_bf16(ref_out_ptr, hh)?;

        let mut b_idx_sorted = b_idx.clone();
        let mut r_idx_sorted = r_idx.clone();
        b_idx_sorted.sort_unstable();
        r_idx_sorted.sort_unstable();
        let idx_match = b_idx_sorted
            .iter()
            .zip(&r_idx_sorted)
            .filter(|(a, b)| a == b)
            .count();

        tracing::info!(
            "KN_CMP tok{}: logits_maxdiff={:.6} idx_match={}/{} w_maxdiff={:.6} out_maxdiff={:.6}",
            t,
            max_diff(&b_logits, &r_logits),
            idx_match,
            tk,
            max_diff(&b_w, &r_w),
            max_diff(&b_out, &r_out),
        );
        tracing::info!(
            "KN_CMP tok{t} detail: idx_b={:?} idx_r={:?} w_b={:?} w_r={:?} out_b[0..4]={:?} out_r[0..4]={:?} logits_b[0..4]={:?} logits_r[0..4]={:?}",
            b_idx,
            r_idx,
            &b_w[..tk.min(4)],
            &r_w[..tk.min(4)],
            &b_out[..4],
            &r_out[..4],
            &b_logits[..4],
            &r_logits[..4],
        );
        Ok(())
    }
}
