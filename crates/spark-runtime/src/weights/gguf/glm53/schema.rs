// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeMap;

use super::super::GgmlType;
use super::Glm53QuantProfile;

mod layers;
mod ud_iq2_xxs;
mod ud_q2_k_xl;
use layers::{add_dsa, add_kda, add_mhc};

pub(in crate::weights::gguf) type Schema = BTreeMap<String, (Vec<u64>, GgmlType)>;

pub(super) fn add(schema: &mut Schema, name: impl Into<String>, dims: &[u64], ty: GgmlType) {
    assert!(schema.insert(name.into(), (dims.to_vec(), ty)).is_none());
}

fn add_ffn(schema: &mut Schema, layer: usize) {
    if layer < 3 {
        for projection in ["gate", "up"] {
            add(
                schema,
                format!("blk.{layer}.ffn_{projection}.weight"),
                &[4096, 12288],
                GgmlType::Q6_K,
            );
        }
        add(
            schema,
            format!("blk.{layer}.ffn_down.weight"),
            &[12288, 4096],
            GgmlType::Q6_K,
        );
        return;
    }
    add(
        schema,
        format!("blk.{layer}.ffn_gate_inp.weight"),
        &[4096, 288],
        GgmlType::F32,
    );
    add(
        schema,
        format!("blk.{layer}.exp_probs_b.bias"),
        &[288],
        GgmlType::F32,
    );
    let (up_type, down_type) = if layer == 45 {
        (GgmlType::Q2_K, GgmlType::Q3_K)
    } else if layer == 11 {
        (GgmlType::IQ3_S, GgmlType::IQ4_XS)
    } else {
        (
            GgmlType::IQ2_S,
            if matches!(layer, 12 | 44) {
                GgmlType::IQ4_XS
            } else {
                GgmlType::IQ3_S
            },
        )
    };
    for projection in ["gate", "up"] {
        add(
            schema,
            format!("blk.{layer}.ffn_{projection}_exps.weight"),
            &[4096, 2048, 288],
            up_type,
        );
    }
    add(
        schema,
        format!("blk.{layer}.ffn_down_exps.weight"),
        &[2048, 4096, 288],
        down_type,
    );
    let shared_type = if layer == 11 {
        GgmlType::Q8_0
    } else {
        GgmlType::Q6_K
    };
    for projection in ["gate", "up"] {
        add(
            schema,
            format!("blk.{layer}.ffn_{projection}_shexp.weight"),
            &[4096, 2048],
            shared_type,
        );
    }
    add(
        schema,
        format!("blk.{layer}.ffn_down_shexp.weight"),
        &[2048, 4096],
        shared_type,
    );
}

/// Vocabulary rows in the GLM-5.3-Flash GGUF payload.
///
/// The GGUF pads to 154,880 while the upstream safetensors `config.json`
/// declares 154,856. Serving from GGUF must use THIS value: it is what
/// `token_embd.weight` and `output.weight` actually hold, and it is pinned by
/// the schema below and verified against the shard SHA-256 set.
pub const GLM53_GGUF_VOCAB_SIZE: usize = 154_880;

pub(in crate::weights::gguf) fn expected_schema(profile: Glm53QuantProfile) -> Schema {
    let mut schema = Schema::new();
    add(
        &mut schema,
        "token_embd.weight",
        &[4096, 154880],
        GgmlType::Q6_K,
    );
    add(
        &mut schema,
        "output.weight",
        &[4096, 154880],
        GgmlType::Q6_K,
    );
    add(&mut schema, "output_norm.weight", &[4096], GgmlType::F32);
    for layer in 0..=45 {
        add(
            &mut schema,
            format!("blk.{layer}.attn_norm.weight"),
            &[4096],
            GgmlType::F32,
        );
        add(
            &mut schema,
            format!("blk.{layer}.ffn_norm.weight"),
            &[4096],
            GgmlType::F32,
        );
        if layer < 45 {
            add_mhc(&mut schema, layer);
        }
        if layer < 45 && layer % 4 != 3 {
            add_kda(&mut schema, layer)
        } else {
            add_dsa(&mut schema, layer)
        }
        add_ffn(&mut schema, layer);
    }
    for (name, dims) in [
        ("blk.45.nextn.eh_proj.weight", vec![8192, 4096]),
        ("blk.45.nextn.enorm.weight", vec![4096]),
        ("blk.45.nextn.hnorm.weight", vec![4096]),
        ("blk.45.nextn.shared_head_norm.weight", vec![4096]),
    ] {
        let ty = if name.ends_with("eh_proj.weight") {
            GgmlType::Q8_0
        } else {
            GgmlType::F32
        };
        add(&mut schema, name, &dims, ty);
    }
    match profile {
        Glm53QuantProfile::UdQ2KXl => ud_q2_k_xl::retarget(&mut schema),
        Glm53QuantProfile::UdIq2Xxs => ud_iq2_xxs::retarget(&mut schema),
        // UD-IQ3_XXS is the base recipe the shared builders already emit.
        Glm53QuantProfile::UdIq3Xxs => {}
    }
    schema
}
