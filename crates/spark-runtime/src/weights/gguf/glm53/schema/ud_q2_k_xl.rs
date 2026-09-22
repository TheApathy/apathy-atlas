// SPDX-License-Identifier: AGPL-3.0-only

use super::super::super::GgmlType;
use super::Schema;

/// Retarget the shared names and dimensions to the pinned Unsloth UD-Q2_K_XL types.
pub(super) fn retarget(schema: &mut Schema) {
    let mut changed = 0usize;
    for (name, (_, ty)) in schema {
        let replacement = if name == "output.weight" {
            Some(GgmlType::Q4_K)
        } else if name == "token_embd.weight" {
            Some(GgmlType::Q5_K)
        } else if name.ends_with(".attn_output.weight")
            || name.ends_with(".attn_q.weight")
            || name.ends_with(".attn_q_a.weight")
        {
            match *ty {
                GgmlType::Q6_K => Some(GgmlType::Q5_K),
                GgmlType::Q8_0 => Some(GgmlType::Q6_K),
                _ => None,
            }
        } else if name.ends_with(".ffn_gate.weight") || name.ends_with(".ffn_up.weight") {
            (*ty == GgmlType::Q6_K).then_some(GgmlType::Q5_K)
        } else if name.ends_with(".ffn_gate_exps.weight") || name.ends_with(".ffn_up_exps.weight") {
            match *ty {
                GgmlType::IQ2_S => Some(GgmlType::IQ2_XS),
                GgmlType::IQ3_S => Some(GgmlType::IQ3_XXS),
                _ => None,
            }
        } else if name.ends_with(".ffn_down_exps.weight") {
            (*ty == GgmlType::IQ3_S).then_some(GgmlType::IQ3_XXS)
        } else if name.ends_with(".ffn_gate_shexp.weight") || name.ends_with(".ffn_up_shexp.weight")
        {
            match *ty {
                GgmlType::Q6_K => Some(GgmlType::Q5_K),
                GgmlType::Q8_0 => Some(GgmlType::Q6_K),
                _ => None,
            }
        } else {
            None
        };
        if let Some(replacement) = replacement {
            *ty = replacement;
            changed += 1;
        }
    }
    assert_eq!(changed, 309, "pinned UD-Q2_K_XL retarget count drift");
}
