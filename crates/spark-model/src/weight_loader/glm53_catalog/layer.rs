// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail};

use super::{
    Glm53AttentionKind, Glm53DsaWeights, Glm53FfnWeights, Glm53GgufCatalog, Glm53HyperWeights,
    Glm53KdaWeights, Glm53LayerDescriptor, Glm53LayerNorms,
};

#[derive(Debug)]
pub enum Glm53AttentionWeights {
    Kda(Glm53KdaWeights),
    Dsa(Glm53DsaWeights),
}

#[derive(Debug)]
pub struct Glm53TargetLayerWeights {
    pub descriptor: Glm53LayerDescriptor,
    pub norms: Glm53LayerNorms,
    pub hyper: Glm53HyperWeights,
    pub attention: Glm53AttentionWeights,
    pub ffn: Glm53FfnWeights,
}

impl Glm53GgufCatalog<'_> {
    pub(crate) fn target_layer(&self, layer: u32) -> Result<Glm53TargetLayerWeights> {
        let descriptor = self.descriptor(layer)?;
        if descriptor.is_nextn {
            bail!("GLM layer45 is native NextN, not a target layer");
        }
        let attention = match descriptor.attention {
            Glm53AttentionKind::Kda => Glm53AttentionWeights::Kda(self.kda(layer)?),
            Glm53AttentionKind::Dsa => Glm53AttentionWeights::Dsa(self.dsa(layer)?),
        };
        Ok(Glm53TargetLayerWeights {
            descriptor,
            norms: self.norms(layer)?,
            hyper: self.hyper_connections(layer)?,
            attention,
            ffn: self.ffn(layer)?,
        })
    }
}
