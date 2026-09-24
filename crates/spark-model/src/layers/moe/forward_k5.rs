// SPDX-License-Identifier: AGPL-3.0-only

//! Exact five-row NVFP4 MoE path for Qwen4 speculative verification.

use super::*;

impl MoeLayer {
    pub fn forward_k5(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        self.forward_qwen4_exact(input, 5, ctx, stream)
    }

    pub fn forward_qwen4_exact(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            matches!(rows, 5 | 9),
            "Qwen4 exact MoE requires 5 or 9 rows"
        );
        anyhow::ensure!(
            ctx.comm.is_none_or(|comm| comm.world_size() == 1),
            "Qwen4 exact NVFP4 MoE does not support expert parallelism"
        );
        // Under ATLAS_UNIFIED_MOE_LAYOUT the untransposed expert weights are
        // freed and the gate/up/down pointer tables below go stale; only the
        // `_t` kernels may run. This path has no `_t` variant.
        anyhow::ensure!(
            !self.use_t_layout_for_decode(),
            "Qwen4 exact K5/K9 MoE (ATLAS_QWEN4_K5_NATIVE_MOE) reads the untransposed expert \
             layout, which ATLAS_UNIFIED_MOE_LAYOUT frees"
        );
        let nvfp4 = self
            .gate_nvfp4
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen4 exact MoE requires an NVFP4 router"))?;
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let router_in = self.router_input(input, rows as u32, h, ctx, stream)?;
        let gate_logits = ctx.buffers.gate_logits();

        // Preserve ordinary K=1 lane ownership and BF16 rounding exactly.
        for row in 0..rows {
            ops::w4a16_gemv(
                ctx.gpu,
                self.w4a16_gemv,
                router_in.offset(row * h as usize * 2),
                nvfp4,
                gate_logits.offset(row * num_experts as usize * 2),
                num_experts,
                h,
                stream,
            )?;
        }

        let indices_dev = ctx.buffers.scratch();
        let weights_dev = indices_dev.offset(rows * top_k as usize * 4);
        if let Some(bias) = self.correction_bias_dev {
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
                1.0,
                rows as u32,
                stream,
            )?;
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
                rows as u32,
                stream,
            )?;
        }

        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_gate_out = ctx.buffers.logits();
        let shared_up_out = ctx.buffers.ssm_qkvz();
        let shared_down_out = ctx.buffers.attn_output();
        let output = ctx.buffers.moe_output();

        // The formerly K=3 CUDA module is token-count parameterized. Its
        // per-row K16 reduction remains byte-identical to K=1/K=2/K=3.
        // ATLAS_QWEN4_MOE_DEDUP=1 swaps in the expert-deduplicated twins
        // (same per-row arithmetic, each distinct expert streamed once).
        let dedup = qwen4_moe_dedup_selected()
            .then(|| {
                let tier = usize::from(rows > 8);
                let rows_max = if tier == 0 { 8u32 } else { 16u32 };
                let gu = self.moe_expert_gate_up_shared_dedup[tier];
                let dn = self.moe_expert_silu_down_shared_dedup[tier];
                (gu.0 != 0 && dn.0 != 0 && rows <= rows_max as usize).then_some((gu, dn, rows_max))
            })
            .flatten();
        let check = dedup.is_some() && qwen4_moe_dedup_check() && !ctx.graph_capture;
        let gate_up = |kernel: KernelHandle| {
            ops::moe_expert_gate_up_shared_batch3(
                ctx.gpu,
                kernel,
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
                shared_gate_out,
                &self.weights.shared_expert.up_proj,
                shared_up_out,
                inter,
                h,
                top_k,
                rows as u32,
                stream,
            )
        };
        let down = |dedup_kernel: Option<(KernelHandle, u32)>| match dedup_kernel {
            Some((kernel, rows_max)) => ops::moe_expert_silu_down_shared_dedup(
                ctx.gpu,
                kernel,
                rows_max,
                expert_gate_out,
                expert_up_out,
                self.down_ptrs.packed_ptrs,
                self.down_ptrs.scale_ptrs,
                self.down_ptrs.scale2_vals,
                expert_down_out,
                indices_dev,
                shared_gate_out,
                shared_up_out,
                &self.weights.shared_expert.down_proj,
                shared_down_out,
                h,
                inter,
                top_k,
                rows as u32,
                stream,
            ),
            None => ops::moe_expert_silu_down_shared_batch3(
                ctx.gpu,
                self.moe_expert_silu_down_shared_batch3,
                expert_gate_out,
                expert_up_out,
                self.down_ptrs.packed_ptrs,
                self.down_ptrs.scale_ptrs,
                self.down_ptrs.scale2_vals,
                expert_down_out,
                indices_dev,
                shared_gate_out,
                shared_up_out,
                &self.weights.shared_expert.down_proj,
                shared_down_out,
                h,
                inter,
                top_k,
                rows as u32,
                stream,
            ),
        };
        let routed_inter = rows * top_k as usize * inter as usize * 2;
        let shared_inter = rows * inter as usize * 2;
        let routed_hidden = rows * top_k as usize * h as usize * 2;
        let shared_hidden = rows * h as usize * 2;
        match dedup {
            Some((gu, dn, rows_max)) if check => {
                // Self-check: reference kernels first, snapshot, then the
                // dedup twins over the same inputs; compare byte for byte.
                gate_up(self.moe_expert_gate_up_shared_batch3)?;
                let reference_up = snapshot(
                    ctx,
                    stream,
                    &[
                        (expert_gate_out, routed_inter),
                        (expert_up_out, routed_inter),
                        (shared_gate_out, shared_inter),
                        (shared_up_out, shared_inter),
                    ],
                )?;
                gate_up(gu)?;
                let mut dedup_up = snapshot(
                    ctx,
                    stream,
                    &[
                        (expert_gate_out, routed_inter),
                        (expert_up_out, routed_inter),
                        (shared_gate_out, shared_inter),
                        (shared_up_out, shared_inter),
                    ],
                )?;
                down(None)?;
                let reference_down = snapshot(
                    ctx,
                    stream,
                    &[
                        (expert_down_out, routed_hidden),
                        (shared_down_out, shared_hidden),
                    ],
                )?;
                down(Some((dn, rows_max)))?;
                let dedup_down = snapshot(
                    ctx,
                    stream,
                    &[
                        (expert_down_out, routed_hidden),
                        (shared_down_out, shared_hidden),
                    ],
                )?;
                if std::env::var("ATLAS_QWEN4_MOE_DEDUP_CHECK_CORRUPT")
                    .ok()
                    .as_deref()
                    == Some("1")
                    && let Some(byte) = dedup_up.first_mut().and_then(|b| b.first_mut())
                {
                    // Positive control: the checker must report this flip.
                    *byte ^= 0x01;
                }
                record_dedup_check(rows, &reference_up, &dedup_up, &reference_down, &dedup_down);
            }
            Some((gu, dn, rows_max)) => {
                gate_up(gu)?;
                down(Some((dn, rows_max)))?;
            }
            None => {
                gate_up(self.moe_expert_gate_up_shared_batch3)?;
                down(None)?;
            }
        }
        ops::moe_weighted_sum_blend_batch3(
            ctx.gpu,
            self.moe_weighted_sum_blend_batch3,
            output,
            expert_down_out,
            weights_dev,
            shared_down_out,
            input,
            self.weights.shared_expert_gate.weight,
            h,
            top_k,
            h,
            rows as u32,
            stream,
        )
    }
}

/// `ATLAS_QWEN4_MOE_DEDUP=1`: expert-deduplicated exact K5/K9 MoE kernels.
fn qwen4_moe_dedup_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4_MOE_DEDUP").ok().as_deref() == Some("1");
        if on {
            tracing::info!("ENGAGED ATLAS_QWEN4_MOE_DEDUP: expert-deduplicated exact-row MoE");
        }
        on
    })
}

/// `ATLAS_QWEN4_MOE_DEDUP_CHECK=1` (eager only): run the reference kernels and
/// the dedup twins on every call and compare their outputs byte for byte.
fn qwen4_moe_dedup_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4_MOE_DEDUP_CHECK").ok().as_deref() == Some("1"))
}

fn snapshot(
    ctx: &ForwardContext,
    stream: u64,
    regions: &[(DevicePtr, usize)],
) -> Result<Vec<Vec<u8>>> {
    ctx.gpu.synchronize(stream)?;
    regions
        .iter()
        .map(|&(ptr, bytes)| {
            let mut buf = vec![0u8; bytes];
            ctx.gpu.copy_d2h(ptr, &mut buf)?;
            Ok(buf)
        })
        .collect()
}

fn record_dedup_check(
    rows: usize,
    reference_up: &[Vec<u8>],
    dedup_up: &[Vec<u8>],
    reference_down: &[Vec<u8>],
    dedup_down: &[Vec<u8>],
) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static BAD_CALLS: AtomicU64 = AtomicU64::new(0);
    let differing = |a: &[Vec<u8>], b: &[Vec<u8>]| -> usize {
        a.iter()
            .zip(b)
            .map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p != q).count())
            .sum()
    };
    let up = differing(reference_up, dedup_up);
    let dn = differing(reference_down, dedup_down);
    let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if up + dn > 0 {
        let bad = BAD_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        tracing::warn!(
            rows,
            up_bytes = up,
            down_bytes = dn,
            bad,
            calls,
            "MOE_DEDUP_CHECK mismatch"
        );
    }
    if calls.is_power_of_two() || calls % 4096 == 0 {
        tracing::info!(
            calls,
            bad_calls = BAD_CALLS.load(Ordering::Relaxed),
            "MOE_DEDUP_CHECK summary"
        );
    }
}
