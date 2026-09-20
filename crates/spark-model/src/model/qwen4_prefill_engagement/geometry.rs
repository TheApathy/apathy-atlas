// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;

use super::Selectors;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Geometry {
    pub(super) hidden: usize,
    pub(super) layers: usize,
    pub(super) attention_layers: usize,
    pub(super) ssm_layers: usize,
    pub(super) experts: usize,
    pub(super) top_k: usize,
    pub(super) routed_intermediate: usize,
    pub(super) shared_intermediate: usize,
    pub(super) q_heads: usize,
    pub(super) kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) key_heads: usize,
    pub(super) key_dim: usize,
    pub(super) value_heads: usize,
    pub(super) value_dim: usize,
    pub(super) conv_dim: usize,
    pub(super) qkvz: usize,
}

impl Geometry {
    pub(super) fn from_config(config: &ModelConfig) -> Self {
        Self {
            hidden: config.hidden_size,
            layers: config.num_hidden_layers,
            attention_layers: config.num_attention_layers(),
            ssm_layers: config.num_ssm_layers(),
            experts: config.num_experts,
            top_k: config.num_experts_per_tok,
            routed_intermediate: config.moe_intermediate_size,
            shared_intermediate: config.shared_expert_intermediate_size,
            q_heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            key_heads: config.linear_num_key_heads,
            key_dim: config.linear_key_head_dim,
            value_heads: config.linear_num_value_heads,
            value_dim: config.linear_value_head_dim,
            conv_dim: config.linear_conv_kernel_dim,
            qkvz: config.ssm_qkvz_size(),
        }
    }

    pub(super) fn validate(self, selectors: Selectors) -> Result<()> {
        ensure!(
            self.hidden > 0 && self.layers == self.attention_layers + self.ssm_layers,
            "Qwen4 prefill receipt requires consistent nonzero model geometry"
        );
        ensure!(
            self.experts > 0
                && (1..=self.experts).contains(&self.top_k)
                && self.routed_intermediate > 0
                && self.shared_intermediate > 0,
            "Qwen4 prefill receipt requires valid MoE geometry"
        );
        if selectors.attention {
            ensure!(
                self.attention_layers > 0
                    && self.q_heads > 0
                    && self.kv_heads > 0
                    && self.kv_heads <= self.q_heads
                    && self.head_dim > 0,
                "Qwen4 attention prefill receipt requires valid attention geometry"
            );
        }
        if selectors.ssm {
            ensure!(
                self.ssm_layers > 0
                    && self.key_heads > 0
                    && self.key_dim > 0
                    && self.value_heads > 0
                    && self.value_dim > 0
                    && self.value_heads.is_multiple_of(self.key_heads)
                    && self.conv_dim > 0
                    && self.qkvz > 0,
                "Qwen4 SSM prefill receipt requires valid recurrent geometry"
            );
        }
        Ok(())
    }
}
