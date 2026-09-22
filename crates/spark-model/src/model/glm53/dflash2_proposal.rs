// SPDX-License-Identifier: AGPL-3.0-only
//! Shared proposal with explicit committed-projection and cache policies.

use super::*;

impl Glm53Dflash2Runtime {
    pub(super) fn enqueue_proposal(
        &self,
        target: &Glm53Exl3Model,
        anchor: u32,
        stream: u64,
        prefix: Option<&mut KvPrefix>,
        observer: Option<&mut dyn Glm53Dflash2ProbeObserver>,
    ) -> Result<(GgmlIqBuffer, GgmlIqBuffer)> {
        self.enqueue_proposal_with_projection(
            target,
            anchor,
            stream,
            prefix,
            observer,
            CommittedProjection::Original,
        )
    }

    pub(super) fn enqueue_proposal_with_projection(
        &self,
        target: &Glm53Exl3Model,
        anchor: u32,
        stream: u64,
        mut prefix: Option<&mut KvPrefix>,
        mut observer: Option<&mut dyn Glm53Dflash2ProbeObserver>,
        projection: CommittedProjection,
    ) -> Result<(GgmlIqBuffer, GgmlIqBuffer)> {
        ensure!(
            self.context_tokens > 0 && target.position() == self.context_tokens,
            "GLM DFlash2 requires every completed target capture in order"
        );
        let gpu = target.gpu();
        let p = self.plan;
        let stream_a = self.region(p.stream_a);
        let stream_b = self.region(p.stream_b);
        let norm = self.region(p.norm);
        let projected = self.region(p.projected_target);
        let query = self.region(p.query);
        let key = self.region(p.key);
        let value = self.region(p.value);
        let attention = self.region(p.attention);
        let dynamic = self.region(p.dynamic_conv);
        let mlp_gate = self.region(p.mlp_gate);
        let mlp_up = self.region(p.mlp_up);
        let mlp_intermediate = self.region(p.mlp_intermediate);

        let mut input = [self.weights.config.dflash_config.mask_token_id; QUERY_TOKENS as usize];
        input[0] = anchor;
        target.embed_dflash_tokens(&input, stream_a, stream)?;

        let attn_plan = attention_plan(self.context_tokens, target.position())?;
        let conv_plan = crate::layers::ops::Glm53Dflash2ConvPlan::new(1, 8, 4096, 16, 2)?;
        for (layer_index, layer) in self.weights.layers.iter().enumerate() {
            ops::rms_norm(
                gpu,
                self.rms_norm,
                stream_a.ptr,
                &layer.input_layernorm,
                norm.ptr,
                QUERY_TOKENS,
                HIDDEN,
                1.0e-5,
                stream,
            )?;
            dense(
                norm.ptr,
                layer.attention_conv.kernel_projection.weight,
                dynamic.ptr,
                8,
                1024,
                HIDDEN,
                stream,
            )?;
            self.conv.launch(
                gpu,
                conv_plan,
                Glm53Dflash2ConvPhase::Prepare,
                Glm53Dflash2ConvBuffers {
                    input_bf16: norm,
                    dynamic_bf16: dynamic,
                    base_bf16: exact(
                        layer.attention_conv.base_kernel.weight,
                        conv_plan.base_bytes,
                    ),
                    output_bf16: stream_b,
                },
                stream,
            )?;
            dense(
                stream_b.ptr,
                layer.q_proj.weight,
                query.ptr,
                8,
                HIDDEN,
                HIDDEN,
                stream,
            )?;
            if let Some(prefix) = prefix.as_deref_mut() {
                self.enqueue_cached_attention(prefix, layer_index, gpu, stream, projection)?;
            } else {
                projection.project_committed(
                    gpu,
                    projected,
                    &layer.k_proj,
                    key,
                    self.context_tokens,
                    stream,
                )?;
                projection.project_committed(
                    gpu,
                    projected,
                    &layer.v_proj,
                    value,
                    self.context_tokens,
                    stream,
                )?;
                let target_kv = attn_plan.target_kv_bytes;
                dense(
                    stream_b.ptr,
                    layer.k_proj.weight,
                    DevicePtr(key.ptr.0 + target_kv as u64),
                    8,
                    KV_WIDTH,
                    HIDDEN,
                    stream,
                )?;
                dense(
                    stream_b.ptr,
                    layer.v_proj.weight,
                    DevicePtr(value.ptr.0 + target_kv as u64),
                    8,
                    KV_WIDTH,
                    HIDDEN,
                    stream,
                )?;
                let (k_cache, v_cache) = self.kv[layer_index];
                self.attention.execute(
                    gpu,
                    attn_plan,
                    Glm53Dflash2AttentionBuffers {
                        q_noise_bf16: query,
                        target_tail_k_bf16: exact(key.ptr, attn_plan.target_kv_bytes),
                        target_tail_v_bf16: exact(value.ptr, attn_plan.target_kv_bytes),
                        noise_k_bf16: exact(
                            DevicePtr(key.ptr.0 + target_kv as u64),
                            attn_plan.noise_kv_bytes,
                        ),
                        noise_v_bf16: exact(
                            DevicePtr(value.ptr.0 + target_kv as u64),
                            attn_plan.noise_kv_bytes,
                        ),
                        output_bf16: attention,
                        q_norm_weight_bf16: exact(layer.q_norm.weight, attn_plan.norm_weight_bytes),
                        k_norm_weight_bf16: exact(layer.k_norm.weight, attn_plan.norm_weight_bytes),
                        target_slots_i64: exact(self.target_slots, attn_plan.target_slots_bytes),
                        noise_slots_i64: exact(self.noise_slots, attn_plan.noise_slots_bytes),
                        block_tables_u32: exact(self.block_table, attn_plan.block_tables_bytes),
                        k_cache_bf16: exact(k_cache, attn_plan.cache_pool_bytes),
                        v_cache_bf16: exact(v_cache, attn_plan.cache_pool_bytes),
                    },
                    stream,
                )?;
            }
            if let Some(observer) = observer.as_mut() {
                self.observe_layer(*observer, layer_index, gpu, stream)?;
            }
            dense(
                attention.ptr,
                layer.output.weight,
                stream_b.ptr,
                8,
                HIDDEN,
                HIDDEN,
                stream,
            )?;
            self.conv.launch(
                gpu,
                conv_plan,
                Glm53Dflash2ConvPhase::Finish,
                Glm53Dflash2ConvBuffers {
                    input_bf16: stream_b,
                    dynamic_bf16: dynamic,
                    base_bf16: exact(
                        layer.attention_conv.base_kernel.weight,
                        conv_plan.base_bytes,
                    ),
                    output_bf16: norm,
                },
                stream,
            )?;
            ops::residual_add(
                gpu,
                self.residual_add,
                stream_a.ptr,
                norm.ptr,
                QUERY_TOKENS * HIDDEN,
                stream,
            )?;

            ops::rms_norm(
                gpu,
                self.rms_norm,
                stream_a.ptr,
                &layer.post_attention_layernorm,
                norm.ptr,
                QUERY_TOKENS,
                HIDDEN,
                1.0e-5,
                stream,
            )?;
            dense(
                norm.ptr,
                layer.mlp_conv.kernel_projection.weight,
                dynamic.ptr,
                8,
                1024,
                HIDDEN,
                stream,
            )?;
            self.conv.launch(
                gpu,
                conv_plan,
                Glm53Dflash2ConvPhase::Prepare,
                Glm53Dflash2ConvBuffers {
                    input_bf16: norm,
                    dynamic_bf16: dynamic,
                    base_bf16: exact(layer.mlp_conv.base_kernel.weight, conv_plan.base_bytes),
                    output_bf16: stream_b,
                },
                stream,
            )?;
            dense(
                stream_b.ptr,
                layer.gate_proj.weight,
                mlp_gate.ptr,
                8,
                INTERMEDIATE,
                HIDDEN,
                stream,
            )?;
            dense(
                stream_b.ptr,
                layer.up_proj.weight,
                mlp_up.ptr,
                8,
                INTERMEDIATE,
                HIDDEN,
                stream,
            )?;
            ops::silu_mul(
                gpu,
                self.silu_mul,
                mlp_gate.ptr,
                mlp_up.ptr,
                mlp_intermediate.ptr,
                QUERY_TOKENS * INTERMEDIATE,
                stream,
            )?;
            dense(
                mlp_intermediate.ptr,
                layer.down_proj.weight,
                attention.ptr,
                8,
                HIDDEN,
                INTERMEDIATE,
                stream,
            )?;
            self.conv.launch(
                gpu,
                conv_plan,
                Glm53Dflash2ConvPhase::Finish,
                Glm53Dflash2ConvBuffers {
                    input_bf16: attention,
                    dynamic_bf16: dynamic,
                    base_bf16: exact(layer.mlp_conv.base_kernel.weight, conv_plan.base_bytes),
                    output_bf16: stream_b,
                },
                stream,
            )?;
            ops::residual_add(
                gpu,
                self.residual_add,
                stream_a.ptr,
                stream_b.ptr,
                QUERY_TOKENS * HIDDEN,
                stream,
            )?;
        }

        ops::rms_norm(
            gpu,
            self.rms_norm,
            stream_a.ptr,
            &self.weights.norm,
            norm.ptr,
            QUERY_TOKENS,
            HIDDEN,
            1.0e-5,
            stream,
        )?;
        let selected_hidden = exact(
            DevicePtr(norm.ptr.0 + (HIDDEN as u64 * 2)),
            PREDICTED_TOKENS as usize * HIDDEN as usize * 2,
        );
        let logits = self.prefix(p.logits, PREDICTED_TOKENS as usize * 154_880 * 2);
        if let Some(observer) = observer.as_mut() {
            observer.observe(ProbeStage::SelectedHidden, selected_hidden, gpu, stream)?;
        }
        self.head.launch(
            gpu,
            Glm53Exl3Bf16LinearBuffers {
                input_bf16: exl(selected_hidden),
                output_bf16: exl(logits),
                input_f16: exl(exact(self.head_input_f16, selected_hidden.bytes)),
                output_f16: exl(exact(self.head_output_f16, logits.bytes)),
                locks_i32: exl(exact(self.head_locks, 4 * 1024 * 1024)),
                input_hadamard_f16: exl(exact(self.head_hadamard_f16, selected_hidden.bytes)),
            },
            stream,
        )?;
        if let Some(observer) = observer.as_mut() {
            observer.observe(ProbeStage::HeadLogits, logits, gpu, stream)?;
        }
        let topk_plan = Glm53Dflash2TopkPlan::new(1, PREDICTED_TOKENS, 154_880, 16)?;
        let candidate_ids = self.prefix(p.candidate_ids, topk_plan.candidate_bytes);
        let candidate_scores = self.prefix(p.candidate_scores, topk_plan.unary_bytes);
        let topk_status = self.prefix(p.topk_status, topk_plan.status_bytes);
        self.topk.launch(
            gpu,
            topk_plan,
            Glm53Dflash2TopkBuffers {
                logits_bf16: logits,
                candidates_u32: candidate_ids,
                unary_f32: candidate_scores,
                status_u32: topk_status,
            },
            stream,
        )?;
        let selector_hidden = self.prefix(p.selector_hidden, PREDICTED_TOKENS as usize * 256 * 2);
        dense(
            selected_hidden.ptr,
            self.weights.selector.hidden_projection.weight,
            selector_hidden.ptr,
            PREDICTED_TOKENS,
            256,
            HIDDEN,
            stream,
        )?;
        let selector_plan = Glm53Dflash2SelectorPlan::new(
            1,
            PREDICTED_TOKENS,
            256,
            16,
            154_880,
            154_880,
            self.weights.config.dflash_config.mask_token_id,
        )?;
        let path = self.prefix(p.chosen_ids, selector_plan.path_bytes);
        let selector_status = self.prefix(p.selector_status, selector_plan.status_bytes);
        self.selector.launch(
            gpu,
            selector_plan,
            Glm53Dflash2SelectorBuffers {
                unary_f32: candidate_scores,
                candidates_u32: candidate_ids,
                hidden_bf16: selector_hidden,
                predecessor_bf16: exact(
                    self.weights.selector.predecessor_codebook.weight,
                    selector_plan.codebook_bytes,
                ),
                successor_bf16: exact(
                    self.weights.selector.successor_codebook.weight,
                    selector_plan.codebook_bytes,
                ),
                anchors_u32: exact(self.anchor, selector_plan.anchor_bytes),
                producer_status_u32: topk_status,
                path_u32: path,
                status_u32: selector_status,
            },
            stream,
        )?;
        if let Some(observer) = observer.as_mut() {
            observer.observe(ProbeStage::DraftIds, path, gpu, stream)?;
        }
        Ok((path, selector_status))
    }
}
