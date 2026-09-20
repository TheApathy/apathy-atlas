// SPDX-License-Identifier: AGPL-3.0-only

//! Allocation-free semantic views for the EXL3 target token walk.

use std::collections::BTreeSet;

use anyhow::{Context, Result, ensure};

use super::{
    GLM53_EXL3_TARGET_LINEAR_COUNT, GLM53_EXL3_TARGET_RAW_COUNT, Glm53Exl3Linear,
    Glm53Exl3NativeStore, Glm53Exl3NativeTensor, Glm53Exl3TargetCatalog,
};

pub const GLM53_EXL3_TARGET_LAYERS: usize = 45;
pub const GLM53_EXL3_TARGET_EXPERTS: usize = 288;

pub struct Glm53Exl3TargetWeights {
    pub embedding: Glm53Exl3NativeTensor,
    pub output_norm: Glm53Exl3NativeTensor,
    pub lm_head: Glm53Exl3Linear,
    pub layers: Vec<Glm53Exl3TargetLayerWeights>,
}

pub struct Glm53Exl3TargetLayerWeights {
    pub index: u32,
    pub norms: Glm53Exl3NormWeights,
    pub hyper: Glm53Exl3HyperWeights,
    pub attention: Glm53Exl3AttentionWeights,
    pub ffn: Glm53Exl3FfnWeights,
}

pub struct Glm53Exl3NormWeights {
    pub attention: Glm53Exl3NativeTensor,
    pub ffn: Glm53Exl3NativeTensor,
}

pub struct Glm53Exl3HyperBranchWeights {
    pub base: Glm53Exl3NativeTensor,
    pub function: Glm53Exl3NativeTensor,
    pub scale: Glm53Exl3NativeTensor,
}

pub struct Glm53Exl3HyperWeights {
    pub attention: Glm53Exl3HyperBranchWeights,
    pub ffn: Glm53Exl3HyperBranchWeights,
}

pub enum Glm53Exl3AttentionWeights {
    Kda(Glm53Exl3KdaWeights),
    Dsa(Glm53Exl3DsaWeights),
}

pub struct Glm53Exl3KdaWeights {
    pub qkv: Glm53Exl3Linear,
    pub output: Glm53Exl3Linear,
    pub a_log: Glm53Exl3NativeTensor,
    pub beta: Glm53Exl3NativeTensor,
    pub conv: Glm53Exl3NativeTensor,
    pub dt_bias: Glm53Exl3NativeTensor,
    pub f_a: Glm53Exl3NativeTensor,
    pub f_b: Glm53Exl3NativeTensor,
    pub g_a: Glm53Exl3NativeTensor,
    pub g_b: Glm53Exl3NativeTensor,
    pub norm: Glm53Exl3NativeTensor,
}

pub struct Glm53Exl3DsaWeights {
    pub q_a: Glm53Exl3Linear,
    pub q_b: Glm53Exl3Linear,
    pub kv_a: Glm53Exl3Linear,
    pub output: Glm53Exl3Linear,
    pub indexer_q_b: Glm53Exl3Linear,
    pub q_a_norm: Glm53Exl3NativeTensor,
    pub kv_a_norm: Glm53Exl3NativeTensor,
    pub k_b: Vec<Glm53Exl3NativeTensor>,
    pub v_b: Vec<Glm53Exl3NativeTensor>,
    pub indexer_proj: Glm53Exl3NativeTensor,
    pub indexer_k: Glm53Exl3NativeTensor,
    pub indexer_k_norm: Glm53Exl3NativeTensor,
    pub indexer_k_norm_bias: Glm53Exl3NativeTensor,
    pub compressor_ape: Glm53Exl3NativeTensor,
    pub compressor_gate: Glm53Exl3NativeTensor,
}

pub enum Glm53Exl3FfnWeights {
    Dense(Glm53Exl3DenseFfnWeights),
    Moe(Glm53Exl3MoeWeights),
}

pub struct Glm53Exl3DenseFfnWeights {
    pub gate: Glm53Exl3Linear,
    pub up: Glm53Exl3Linear,
    pub down: Glm53Exl3Linear,
}

pub struct Glm53Exl3ExpertWeights {
    pub gate: Glm53Exl3Linear,
    pub up: Glm53Exl3Linear,
    pub down: Glm53Exl3Linear,
}

pub struct Glm53Exl3MoeWeights {
    pub router: Glm53Exl3NativeTensor,
    pub expert_bias: Glm53Exl3NativeTensor,
    pub experts: Vec<Glm53Exl3ExpertWeights>,
    pub shared: Glm53Exl3ExpertWeights,
}

impl Glm53Exl3TargetWeights {
    pub fn new(catalog: &Glm53Exl3TargetCatalog, native: &Glm53Exl3NativeStore) -> Result<Self> {
        ensure!(
            catalog.linear_count() == GLM53_EXL3_TARGET_LINEAR_COUNT
                && catalog.raw_count() == GLM53_EXL3_TARGET_RAW_COUNT
                && native.len() == GLM53_EXL3_TARGET_RAW_COUNT,
            "GLM EXL3 typed target input census drift"
        );
        let mut used_linears = BTreeSet::new();
        let mut used_native = BTreeSet::new();
        let embedding = take_native(
            native,
            &mut used_native,
            "model.language_model.embed_tokens.weight",
        )?;
        let output_norm =
            take_native(native, &mut used_native, "model.language_model.norm.weight")?;
        let lm_head = take_linear(catalog, &mut used_linears, "lm_head")?;
        let mut layers = Vec::with_capacity(GLM53_EXL3_TARGET_LAYERS);
        for layer in 0..GLM53_EXL3_TARGET_LAYERS as u32 {
            let root = format!("model.language_model.layers.{layer}");
            let norms = Glm53Exl3NormWeights {
                attention: take_native(
                    native,
                    &mut used_native,
                    &format!("{root}.input_layernorm.weight"),
                )?,
                ffn: take_native(
                    native,
                    &mut used_native,
                    &format!("{root}.post_attention_layernorm.weight"),
                )?,
            };
            let hyper = Glm53Exl3HyperWeights {
                attention: hyper_branch(native, &mut used_native, &root, "attn")?,
                ffn: hyper_branch(native, &mut used_native, &root, "ffn")?,
            };
            let attention = if layer % 4 == 3 {
                Glm53Exl3AttentionWeights::Dsa(dsa(
                    catalog,
                    native,
                    &mut used_linears,
                    &mut used_native,
                    &root,
                )?)
            } else {
                Glm53Exl3AttentionWeights::Kda(kda(
                    catalog,
                    native,
                    &mut used_linears,
                    &mut used_native,
                    &root,
                )?)
            };
            let ffn = if layer < 3 {
                Glm53Exl3FfnWeights::Dense(dense(catalog, &mut used_linears, &root)?)
            } else {
                Glm53Exl3FfnWeights::Moe(moe(
                    catalog,
                    native,
                    &mut used_linears,
                    &mut used_native,
                    &root,
                )?)
            };
            layers.push(Glm53Exl3TargetLayerWeights {
                index: layer,
                norms,
                hyper,
                attention,
                ffn,
            });
        }
        ensure!(
            used_linears.len() == GLM53_EXL3_TARGET_LINEAR_COUNT
                && used_native.len() == GLM53_EXL3_TARGET_RAW_COUNT,
            "GLM EXL3 typed target omitted or duplicated a semantic operand"
        );
        Ok(Self {
            embedding,
            output_norm,
            lm_head,
            layers,
        })
    }
}

fn hyper_branch(
    native: &Glm53Exl3NativeStore,
    used: &mut BTreeSet<String>,
    root: &str,
    branch: &str,
) -> Result<Glm53Exl3HyperBranchWeights> {
    Ok(Glm53Exl3HyperBranchWeights {
        base: take_native(native, used, &format!("{root}.hc_{branch}_base"))?,
        function: take_native(native, used, &format!("{root}.hc_{branch}_fn"))?,
        scale: take_native(native, used, &format!("{root}.hc_{branch}_scale"))?,
    })
}

fn kda(
    catalog: &Glm53Exl3TargetCatalog,
    native: &Glm53Exl3NativeStore,
    linears: &mut BTreeSet<String>,
    raw: &mut BTreeSet<String>,
    root: &str,
) -> Result<Glm53Exl3KdaWeights> {
    let a = format!("{root}.self_attn");
    Ok(Glm53Exl3KdaWeights {
        qkv: take_linear(catalog, linears, &format!("{a}.qkv_proj"))?,
        output: take_linear(catalog, linears, &format!("{a}.o_proj"))?,
        a_log: take_native(native, raw, &format!("{a}.A_log"))?,
        beta: take_native(native, raw, &format!("{a}.b_proj.weight"))?,
        conv: take_native(native, raw, &format!("{a}.conv1d.weight"))?,
        dt_bias: take_native(native, raw, &format!("{a}.dt_bias"))?,
        f_a: take_native(native, raw, &format!("{a}.f_a_proj.weight"))?,
        f_b: take_native(native, raw, &format!("{a}.f_b_proj.weight"))?,
        g_a: take_native(native, raw, &format!("{a}.g_a_proj.weight"))?,
        g_b: take_native(native, raw, &format!("{a}.g_b_proj.weight"))?,
        norm: take_native(native, raw, &format!("{a}.o_norm.weight"))?,
    })
}

fn dsa(
    catalog: &Glm53Exl3TargetCatalog,
    native: &Glm53Exl3NativeStore,
    linears: &mut BTreeSet<String>,
    raw: &mut BTreeSet<String>,
    root: &str,
) -> Result<Glm53Exl3DsaWeights> {
    let a = format!("{root}.self_attn");
    let kv_b = take_native(native, raw, &format!("{a}.kv_b_proj.weight"))?;
    ensure!(
        kv_b.shape() == [32_768, 512],
        "GLM EXL3 DSA packed kv_b shape drift"
    );
    let mut k_b = Vec::with_capacity(64);
    let mut v_b = Vec::with_capacity(64);
    for head in 0..64u64 {
        let base = head * 512;
        k_b.push(kv_b.bf16_matrix_rows(base, 256)?);
        v_b.push(kv_b.bf16_matrix_rows(base + 256, 256)?);
    }
    Ok(Glm53Exl3DsaWeights {
        q_a: take_linear(catalog, linears, &format!("{a}.q_a_proj"))?,
        q_b: take_linear(catalog, linears, &format!("{a}.q_b_proj"))?,
        kv_a: take_linear(catalog, linears, &format!("{a}.kv_a_proj_with_mqa"))?,
        output: take_linear(catalog, linears, &format!("{a}.o_proj"))?,
        indexer_q_b: take_linear(catalog, linears, &format!("{a}.indexer.wq_b"))?,
        q_a_norm: take_native(native, raw, &format!("{a}.q_a_layernorm.weight"))?,
        kv_a_norm: take_native(native, raw, &format!("{a}.kv_a_layernorm.weight"))?,
        k_b,
        v_b,
        indexer_proj: take_native(native, raw, &format!("{a}.indexer.weights_proj.weight"))?,
        indexer_k: take_native(native, raw, &format!("{a}.indexer.wk.weight"))?,
        indexer_k_norm: take_native(native, raw, &format!("{a}.indexer.k_norm.weight"))?,
        indexer_k_norm_bias: take_native(native, raw, &format!("{a}.indexer.k_norm.bias"))?,
        compressor_ape: take_native(
            native,
            raw,
            &format!("{a}.indexer.index_kpool_compress_ape"),
        )?,
        compressor_gate: take_native(
            native,
            raw,
            &format!("{a}.indexer.index_kpool_compress_gate"),
        )?,
    })
}

fn dense(
    catalog: &Glm53Exl3TargetCatalog,
    used: &mut BTreeSet<String>,
    root: &str,
) -> Result<Glm53Exl3DenseFfnWeights> {
    let m = format!("{root}.mlp");
    Ok(Glm53Exl3DenseFfnWeights {
        gate: take_linear(catalog, used, &format!("{m}.gate_proj"))?,
        up: take_linear(catalog, used, &format!("{m}.up_proj"))?,
        down: take_linear(catalog, used, &format!("{m}.down_proj"))?,
    })
}

fn moe(
    catalog: &Glm53Exl3TargetCatalog,
    native: &Glm53Exl3NativeStore,
    linears: &mut BTreeSet<String>,
    raw: &mut BTreeSet<String>,
    root: &str,
) -> Result<Glm53Exl3MoeWeights> {
    let m = format!("{root}.mlp");
    let mut experts = Vec::with_capacity(GLM53_EXL3_TARGET_EXPERTS);
    for expert in 0..GLM53_EXL3_TARGET_EXPERTS {
        experts.push(expert_triplet(
            catalog,
            linears,
            &format!("{m}.experts.{expert}"),
        )?);
    }
    Ok(Glm53Exl3MoeWeights {
        router: take_native(native, raw, &format!("{m}.gate.weight"))?,
        expert_bias: take_native(native, raw, &format!("{m}.gate.e_score_correction_bias"))?,
        experts,
        shared: expert_triplet(catalog, linears, &format!("{m}.shared_experts"))?,
    })
}

fn expert_triplet(
    catalog: &Glm53Exl3TargetCatalog,
    used: &mut BTreeSet<String>,
    root: &str,
) -> Result<Glm53Exl3ExpertWeights> {
    Ok(Glm53Exl3ExpertWeights {
        gate: take_linear(catalog, used, &format!("{root}.gate_proj"))?,
        up: take_linear(catalog, used, &format!("{root}.up_proj"))?,
        down: take_linear(catalog, used, &format!("{root}.down_proj"))?,
    })
}

fn take_linear(
    catalog: &Glm53Exl3TargetCatalog,
    used: &mut BTreeSet<String>,
    name: &str,
) -> Result<Glm53Exl3Linear> {
    ensure!(
        used.insert(name.to_owned()),
        "duplicate typed target projection {name}"
    );
    catalog
        .linear(name)
        .cloned()
        .with_context(|| format!("typed target projection missing {name}"))
}

fn take_native(
    native: &Glm53Exl3NativeStore,
    used: &mut BTreeSet<String>,
    name: &str,
) -> Result<Glm53Exl3NativeTensor> {
    ensure!(
        used.insert(name.to_owned()),
        "duplicate typed target native operand {name}"
    );
    native
        .get(name)
        .cloned()
        .with_context(|| format!("typed target native operand missing {name}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architectural_counts_are_exact() {
        assert_eq!(GLM53_EXL3_TARGET_LAYERS, 45);
        assert_eq!(GLM53_EXL3_TARGET_EXPERTS, 288);
        assert_eq!((0..45).filter(|layer| layer % 4 == 3).count(), 11);
        assert_eq!((3..45).count() * GLM53_EXL3_TARGET_EXPERTS, 12_096);
    }
}
