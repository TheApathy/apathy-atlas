// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail};

use super::{Glm53AttentionKind, Glm53GgufCatalog};
use crate::weight_loader::{Glm53GgufF32, Glm53GgufMatrix, Glm53GgufMatrixBank};

#[derive(Debug)]
pub struct Glm53KdaWeights {
    pub q: Glm53GgufMatrix,
    pub k: Glm53GgufMatrix,
    pub v: Glm53GgufMatrix,
    pub output: Glm53GgufMatrix,
    pub conv_q: Glm53GgufF32,
    pub conv_k: Glm53GgufF32,
    pub conv_v: Glm53GgufF32,
    pub a: Glm53GgufF32,
    pub beta: Glm53GgufMatrix,
    pub dt_bias: Glm53GgufF32,
    pub f_a: Glm53GgufMatrix,
    pub f_b: Glm53GgufMatrix,
    pub g_a: Glm53GgufMatrix,
    pub g_b: Glm53GgufMatrix,
    pub norm: Glm53GgufF32,
}

#[derive(Debug)]
pub struct Glm53DsaWeights {
    pub k_b: Glm53GgufMatrixBank,
    pub v_b: Glm53GgufMatrixBank,
    pub kv_a_mqa: Glm53GgufMatrix,
    pub kv_a_norm: Glm53GgufF32,
    pub q_a: Glm53GgufMatrix,
    pub q_a_norm: Glm53GgufF32,
    pub q_b: Glm53GgufMatrix,
    pub output: Glm53GgufMatrix,
    pub indexer_k: Glm53GgufMatrix,
    pub indexer_q_b: Glm53GgufMatrix,
    pub indexer_k_norm: Glm53GgufF32,
    pub indexer_k_norm_bias: Glm53GgufF32,
    pub indexer_proj: Glm53GgufF32,
    pub compressor_ape: Glm53GgufF32,
    pub compressor_gate: Glm53GgufMatrix,
}

impl Glm53GgufCatalog<'_> {
    pub(crate) fn kda(&self, layer: u32) -> Result<Glm53KdaWeights> {
        if self.descriptor(layer)?.attention != Glm53AttentionKind::Kda {
            bail!("GLM layer {layer} is not a KDA layer");
        }
        let name = |suffix: &str| format!("blk.{layer}.{suffix}");
        Ok(Glm53KdaWeights {
            q: self.matrix(&name("attn_q.weight"))?,
            k: self.matrix(&name("attn_k.weight"))?,
            v: self.matrix(&name("attn_v.weight"))?,
            output: self.matrix(&name("attn_output.weight"))?,
            conv_q: self.f32(&name("ssm_conv1d_q.weight"), &[4, 1, 8192])?,
            conv_k: self.f32(&name("ssm_conv1d_k.weight"), &[4, 1, 8192])?,
            conv_v: self.f32(&name("ssm_conv1d_v.weight"), &[4, 1, 8192])?,
            a: self.f32(&name("ssm_a"), &[64])?,
            beta: self.matrix(&name("ssm_beta.weight"))?,
            dt_bias: self.f32(&name("ssm_dt.bias"), &[8192])?,
            f_a: self.matrix(&name("ssm_f_a.weight"))?,
            f_b: self.matrix(&name("ssm_f_b.weight"))?,
            g_a: self.matrix(&name("ssm_g_a.weight"))?,
            g_b: self.matrix(&name("ssm_g_b.weight"))?,
            norm: self.f32(&name("ssm_norm.weight"), &[128])?,
        })
    }

    pub(crate) fn dsa(&self, layer: u32) -> Result<Glm53DsaWeights> {
        if self.descriptor(layer)?.attention != Glm53AttentionKind::Dsa {
            bail!("GLM layer {layer} is not a DSA layer");
        }
        let name = |suffix: &str| format!("blk.{layer}.{suffix}");
        Ok(Glm53DsaWeights {
            k_b: self.matrix_bank(&name("attn_k_b.weight"))?,
            v_b: self.matrix_bank(&name("attn_v_b.weight"))?,
            kv_a_mqa: self.matrix(&name("attn_kv_a_mqa.weight"))?,
            kv_a_norm: self.f32(&name("attn_kv_a_norm.weight"), &[512])?,
            q_a: self.matrix(&name("attn_q_a.weight"))?,
            q_a_norm: self.f32(&name("attn_q_a_norm.weight"), &[1536])?,
            q_b: self.matrix(&name("attn_q_b.weight"))?,
            output: self.matrix(&name("attn_output.weight"))?,
            indexer_k: self.matrix(&name("indexer.attn_k.weight"))?,
            indexer_q_b: self.matrix(&name("indexer.attn_q_b.weight"))?,
            indexer_k_norm: self.f32(&name("indexer.k_norm.weight"), &[128])?,
            indexer_k_norm_bias: self.f32(&name("indexer.k_norm.bias"), &[128])?,
            indexer_proj: self.f32(&name("indexer.proj.weight"), &[4096, 32])?,
            compressor_ape: self.f32(&name("indexer_compressor_ape.weight"), &[128, 4])?,
            compressor_gate: self.matrix(&name("indexer_compressor_gate.weight"))?,
        })
    }
}
