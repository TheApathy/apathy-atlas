// SPDX-License-Identifier: AGPL-3.0-only

//! Retarget the shared GLM-5.3 names/dimensions to the pinned Unsloth
//! UD-IQ2_XXS types.
//!
//! Unlike [`super::ud_q2_k_xl`], which expresses itself as a delta from the
//! base recipe and asserts a substitution count, this profile assigns every
//! type absolutely and then asserts the resulting **type histogram**. The
//! stronger check is deliberate: UD-IQ2_XXS exists to fit one Spark once every
//! tensor is device-resident, so a silent type drift here would not surface as
//! a load failure — it would surface as a residency overrun that takes the host
//! down with it. The histogram is measured from the published shard headers and
//! covers all 1,412 tensors, so any rule that stops matching upstream fails
//! here rather than at allocation time.
//!
//! Layer topology (mirrors the base schema): DSA iff `layer % 4 == 3` or
//! `layer == 45`, else KDA; dense FFN iff `layer < 3`, else MoE; layer 45 is
//! the NextN block. Layer 11 carries an upgraded tranche throughout, and
//! layers 12/44 upgrade only their down-experts.

use super::super::super::GgmlType;
use super::Schema;

/// Measured from the four published UD-IQ2_XXS shard headers.
const EXPECTED_HISTOGRAM: &[(GgmlType, usize)] = &[
    (GgmlType::F32, 638),
    (GgmlType::Q8_0, 346),
    (GgmlType::Q5_K, 248),
    (GgmlType::IQ2_XXS, 82),
    (GgmlType::Q6_K, 49),
    (GgmlType::IQ3_XXS, 39),
    (GgmlType::IQ4_XS, 3),
    (GgmlType::Q4_K, 2),
    (GgmlType::IQ2_S, 2),
    (GgmlType::Q2_K, 2),
    (GgmlType::Q3_K, 1),
];

const EXPECTED_TENSORS: usize = 1_412;

/// The exact UD-IQ2_XXS type for one tensor name, or `None` if the name is not
/// part of the pinned schema (which the caller treats as a hard error).
fn target_type(name: &str) -> Option<GgmlType> {
    // Globals.
    match name {
        "output.weight" | "token_embd.weight" => return Some(GgmlType::Q4_K),
        "output_norm.weight" => return Some(GgmlType::F32),
        _ => {}
    }

    let (layer, suffix) = split_block(name)?;

    // Every norm, bias, gate-input, short-conv and SSM decay stays F32.
    if suffix.ends_with("_norm.weight")
        || suffix.ends_with("_norm.bias")
        || suffix == "attn_norm.weight"
        || suffix == "ffn_norm.weight"
        || suffix == "ffn_gate_inp.weight"
        || suffix == "exp_probs_b.bias"
        || suffix == "ssm_a"
        || suffix == "ssm_dt.bias"
        || suffix.starts_with("ssm_conv1d_")
        || suffix.starts_with("hc_") && (suffix.ends_with("_base.weight") || suffix.ends_with("_scale.weight"))
        || suffix == "indexer.proj.weight"
        || suffix == "indexer_compressor_ape.weight"
        // The NextN embedding/hidden norms do not carry the `_norm` suffix.
        || suffix == "nextn.enorm.weight"
        || suffix == "nextn.hnorm.weight"
    {
        return Some(GgmlType::F32);
    }

    // mHC mixing functions, SSM projections, DSA absorbed/indexer projections
    // and the NextN eh_proj are all held at Q8_0.
    if suffix.ends_with("hc_attn_fn.weight")
        || suffix.ends_with("hc_ffn_fn.weight")
        || suffix == "ssm_beta.weight"
        || suffix == "ssm_f_a.weight"
        || suffix == "ssm_f_b.weight"
        || suffix == "ssm_g_a.weight"
        || suffix == "ssm_g_b.weight"
        || suffix == "attn_k_b.weight"
        || suffix == "attn_v_b.weight"
        || suffix == "attn_kv_a_mqa.weight"
        || suffix == "attn_q_b.weight"
        || suffix == "indexer.attn_k.weight"
        || suffix == "indexer.attn_q_b.weight"
        || suffix == "indexer_compressor_gate.weight"
        || suffix == "nextn.eh_proj.weight"
    {
        return Some(GgmlType::Q8_0);
    }

    // Layer 11 is the upgraded tranche: every matrix that would otherwise be
    // Q5_K rides one step higher.
    let upgraded = layer == 11;

    // Attention edges, shared-expert gate/up, and the dense FFN gate/up.
    if suffix == "attn_q.weight"
        || suffix == "attn_k.weight"
        || suffix == "attn_v.weight"
        || suffix == "attn_output.weight"
        || suffix == "attn_q_a.weight"
        || suffix == "ffn_gate_shexp.weight"
        || suffix == "ffn_up_shexp.weight"
    {
        return Some(if upgraded {
            GgmlType::Q6_K
        } else {
            GgmlType::Q5_K
        });
    }
    if suffix == "ffn_gate.weight" || suffix == "ffn_up.weight" {
        return Some(GgmlType::Q5_K);
    }
    if suffix == "ffn_down.weight" {
        return Some(GgmlType::Q6_K);
    }
    if suffix == "ffn_down_shexp.weight" {
        return Some(if upgraded {
            GgmlType::Q8_0
        } else {
            GgmlType::Q6_K
        });
    }

    // Routed experts: the bulk of the checkpoint, and the whole reason this
    // profile exists.
    if suffix == "ffn_gate_exps.weight" || suffix == "ffn_up_exps.weight" {
        return Some(match layer {
            45 => GgmlType::Q2_K,
            11 => GgmlType::IQ2_S,
            _ => GgmlType::IQ2_XXS,
        });
    }
    if suffix == "ffn_down_exps.weight" {
        return Some(match layer {
            45 => GgmlType::Q3_K,
            11 | 12 | 44 => GgmlType::IQ4_XS,
            _ => GgmlType::IQ3_XXS,
        });
    }

    None
}

fn split_block(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (layer, suffix) = rest.split_once('.')?;
    Some((layer.parse().ok()?, suffix))
}

/// Retarget every pinned name to its UD-IQ2_XXS type.
pub(super) fn retarget(schema: &mut Schema) {
    let mut histogram: Vec<(GgmlType, usize)> = Vec::new();
    let mut total = 0usize;
    for (name, (_, ty)) in schema.iter_mut() {
        let replacement = target_type(name)
            .unwrap_or_else(|| panic!("UD-IQ2_XXS schema has no type rule for {name}"));
        *ty = replacement;
        total += 1;
        match histogram.iter_mut().find(|(t, _)| *t == replacement) {
            Some((_, count)) => *count += 1,
            None => histogram.push((replacement, 1)),
        }
    }
    assert_eq!(total, EXPECTED_TENSORS, "UD-IQ2_XXS tensor-count drift");
    histogram.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then(format!("{:?}", a.0).cmp(&format!("{:?}", b.0)))
    });
    let mut expected = EXPECTED_HISTOGRAM.to_vec();
    expected.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then(format!("{:?}", a.0).cmp(&format!("{:?}", b.0)))
    });
    assert_eq!(histogram, expected, "UD-IQ2_XXS type histogram drift");
}
