// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;

use super::{Glm53FfnKind, Glm53GgufCatalog};
use crate::weight_loader::{Glm53GgufF32, Glm53GgufMatrix, Glm53GgufMatrixBank};

#[derive(Debug)]
pub struct Glm53DenseFfnWeights {
    pub gate: Glm53GgufMatrix,
    pub up: Glm53GgufMatrix,
    pub down: Glm53GgufMatrix,
}

#[derive(Debug)]
pub struct Glm53MoeWeights {
    pub router: Glm53GgufF32,
    pub expert_bias: Glm53GgufF32,
    pub gate_experts: Glm53GgufMatrixBank,
    pub up_experts: Glm53GgufMatrixBank,
    pub down_experts: Glm53GgufMatrixBank,
    pub shared_gate: Glm53GgufMatrix,
    pub shared_up: Glm53GgufMatrix,
    pub shared_down: Glm53GgufMatrix,
}

#[derive(Debug)]
pub enum Glm53FfnWeights {
    Dense(Glm53DenseFfnWeights),
    Moe(Glm53MoeWeights),
}

impl Glm53GgufCatalog<'_> {
    pub(crate) fn ffn(&self, layer: u32) -> Result<Glm53FfnWeights> {
        let descriptor = self.descriptor(layer)?;
        let name = |suffix: &str| format!("blk.{layer}.{suffix}");
        Ok(match descriptor.ffn {
            Glm53FfnKind::Dense => Glm53FfnWeights::Dense(Glm53DenseFfnWeights {
                gate: self.matrix(&name("ffn_gate.weight"))?,
                up: self.matrix(&name("ffn_up.weight"))?,
                down: self.matrix(&name("ffn_down.weight"))?,
            }),
            Glm53FfnKind::Moe => Glm53FfnWeights::Moe(Glm53MoeWeights {
                router: self.f32(&name("ffn_gate_inp.weight"), &[4096, 288])?,
                expert_bias: self.f32(&name("exp_probs_b.bias"), &[288])?,
                gate_experts: self.matrix_bank(&name("ffn_gate_exps.weight"))?,
                up_experts: self.matrix_bank(&name("ffn_up_exps.weight"))?,
                down_experts: self.matrix_bank(&name("ffn_down_exps.weight"))?,
                shared_gate: self.matrix(&name("ffn_gate_shexp.weight"))?,
                shared_up: self.matrix(&name("ffn_up_shexp.weight"))?,
                shared_down: self.matrix(&name("ffn_down_shexp.weight"))?,
            }),
        })
    }
}
