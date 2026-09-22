// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail};

use super::{Glm53DsaWeights, Glm53FfnWeights, Glm53GgufCatalog, Glm53LayerNorms, Glm53MoeWeights};
use crate::weight_loader::{Glm53GgufF32, Glm53GgufMatrix};

#[derive(Debug)]
pub struct Glm53NextnWeights {
    pub attention: Glm53DsaWeights,
    pub ffn: Glm53MoeWeights,
    pub norms: Glm53LayerNorms,
    pub eh_projection: Glm53GgufMatrix,
    pub embedding_norm: Glm53GgufF32,
    pub hidden_norm: Glm53GgufF32,
    pub shared_head_norm: Glm53GgufF32,
}

impl Glm53GgufCatalog<'_> {
    pub(crate) fn nextn(&self) -> Result<Glm53NextnWeights> {
        let ffn = match self.ffn(45)? {
            Glm53FfnWeights::Moe(weights) => weights,
            Glm53FfnWeights::Dense(_) => bail!("GLM layer45 NextN unexpectedly has dense FFN"),
        };
        Ok(Glm53NextnWeights {
            attention: self.dsa(45)?,
            ffn,
            norms: self.norms(45)?,
            eh_projection: self.matrix("blk.45.nextn.eh_proj.weight")?,
            embedding_norm: self.f32("blk.45.nextn.enorm.weight", &[4096])?,
            hidden_norm: self.f32("blk.45.nextn.hnorm.weight", &[4096])?,
            shared_head_norm: self.f32("blk.45.nextn.shared_head_norm.weight", &[4096])?,
        })
    }
}
