// SPDX-License-Identifier: AGPL-3.0-only

//! Exact five-row NVFP4 MoE path for Qwen4 speculative verification.

use super::*;

impl MoeLayer {
    pub fn forward_k5(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        anyhow::ensure!(
            ctx.comm.is_none_or(|comm| comm.world_size() == 1),
            "K=5 NVFP4 MoE does not support expert parallelism"
        );
        let nvfp4 = self
            .gate_nvfp4
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("K=5 MoE requires an NVFP4 router"))?;
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let router_in = self.router_input(input, 5, h, ctx, stream)?;
        let gate_logits = ctx.buffers.gate_logits();

        // Preserve ordinary K=1 lane ownership and BF16 rounding exactly.
        for row in 0..5usize {
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
        let weights_dev = indices_dev.offset(5 * top_k as usize * 4);
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
                5,
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
                5,
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
        ops::moe_expert_gate_up_shared_batch3(
            ctx.gpu,
            self.moe_expert_gate_up_shared_batch3,
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
            5,
            stream,
        )?;
        ops::moe_expert_silu_down_shared_batch3(
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
            5,
            stream,
        )?;
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
            5,
            stream,
        )
    }
}
