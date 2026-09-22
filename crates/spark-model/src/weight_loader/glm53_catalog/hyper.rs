// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail};

use super::Glm53GgufCatalog;
use crate::weight_loader::{Glm53GgufF32, Glm53GgufMatrix};

#[derive(Debug)]
pub struct Glm53LayerNorms {
    pub attention: Glm53GgufF32,
    pub ffn: Glm53GgufF32,
}

#[derive(Debug)]
pub struct Glm53HyperBranchWeights {
    pub function: Glm53GgufMatrix,
    pub base: Glm53GgufF32,
    pub scale: Glm53GgufF32,
}

#[derive(Debug)]
pub struct Glm53HyperWeights {
    pub attention: Glm53HyperBranchWeights,
    pub ffn: Glm53HyperBranchWeights,
}

impl Glm53GgufCatalog<'_> {
    pub(crate) fn norms(&self, layer: u32) -> Result<Glm53LayerNorms> {
        self.descriptor(layer)?;
        Ok(Glm53LayerNorms {
            attention: self.f32(&format!("blk.{layer}.attn_norm.weight"), &[4096])?,
            ffn: self.f32(&format!("blk.{layer}.ffn_norm.weight"), &[4096])?,
        })
    }

    pub(crate) fn hyper_connections(&self, layer: u32) -> Result<Glm53HyperWeights> {
        if !self.descriptor(layer)?.has_hyper_connections {
            bail!("GLM NextN layer {layer} has no mHC weights");
        }
        let branch = |kind: &str| -> Result<Glm53HyperBranchWeights> {
            Ok(Glm53HyperBranchWeights {
                function: self.matrix(&format!("blk.{layer}.hc_{kind}_fn.weight"))?,
                base: self.f32(&format!("blk.{layer}.hc_{kind}_base.weight"), &[24])?,
                scale: self.f32(&format!("blk.{layer}.hc_{kind}_scale.weight"), &[3])?,
            })
        };
        Ok(Glm53HyperWeights {
            attention: branch("attn")?,
            ffn: branch("ffn")?,
        })
    }
}
