// SPDX-License-Identifier: AGPL-3.0-only

//! Typed names and topology over the exact admitted GLM-5.3 GGUF store.

mod attention;
mod ffn;
mod hyper;
mod layer;
mod nextn;
pub use attention::{Glm53DsaWeights, Glm53KdaWeights};
pub use ffn::{Glm53DenseFfnWeights, Glm53FfnWeights, Glm53MoeWeights};
pub use hyper::{Glm53HyperBranchWeights, Glm53HyperWeights, Glm53LayerNorms};
pub use layer::{Glm53AttentionWeights, Glm53TargetLayerWeights};
pub use nextn::Glm53NextnWeights;

use anyhow::{Context, Result, bail};
use spark_runtime::weights::gguf::{GgufDeviceStore, GgufDeviceTensor};

use super::{Glm53GgufF32, Glm53GgufMatrix, Glm53GgufMatrixBank};

const EXACT_TENSOR_COUNT: usize = 1412;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53AttentionKind {
    Kda,
    Dsa,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53FfnKind {
    Dense,
    Moe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53LayerDescriptor {
    pub index: u32,
    pub attention: Glm53AttentionKind,
    pub ffn: Glm53FfnKind,
    pub has_hyper_connections: bool,
    pub is_nextn: bool,
}

impl Glm53LayerDescriptor {
    pub fn new(index: u32) -> Result<Self> {
        if index > 45 {
            bail!("GLM layer index {index} exceeds target plus NextN topology");
        }
        let is_nextn = index == 45;
        let attention = if is_nextn || index % 4 == 3 {
            Glm53AttentionKind::Dsa
        } else {
            Glm53AttentionKind::Kda
        };
        Ok(Self {
            index,
            attention,
            ffn: if index < 3 {
                Glm53FfnKind::Dense
            } else {
                Glm53FfnKind::Moe
            },
            has_hyper_connections: !is_nextn,
            is_nextn,
        })
    }
}

pub struct Glm53GgufCatalog<'a> {
    store: &'a GgufDeviceStore,
}

impl<'a> Glm53GgufCatalog<'a> {
    pub(crate) fn new(store: &'a GgufDeviceStore) -> Result<Self> {
        if store.len() != EXACT_TENSOR_COUNT {
            bail!(
                "GLM device store must retain exactly {EXACT_TENSOR_COUNT} tensors, got {}",
                store.len()
            );
        }
        Ok(Self { store })
    }

    pub fn descriptor(&self, index: u32) -> Result<Glm53LayerDescriptor> {
        Glm53LayerDescriptor::new(index)
    }

    pub(crate) fn token_embedding(&self) -> Result<Glm53GgufMatrix> {
        self.matrix("token_embd.weight")
    }

    pub(crate) fn output(&self) -> Result<Glm53GgufMatrix> {
        self.matrix("output.weight")
    }

    pub(crate) fn output_norm(&self) -> Result<Glm53GgufF32> {
        self.f32("output_norm.weight", &[4096])
    }

    pub(super) fn tensor(&self, name: &str) -> Result<&GgufDeviceTensor> {
        self.store
            .get(name)
            .with_context(|| format!("missing admitted GLM tensor {name}"))
    }

    pub(super) fn matrix(&self, name: &str) -> Result<Glm53GgufMatrix> {
        Glm53GgufMatrix::new(self.tensor(name)?)
            .with_context(|| format!("invalid GLM matrix {name}"))
    }

    pub(super) fn matrix_bank(&self, name: &str) -> Result<Glm53GgufMatrixBank> {
        Glm53GgufMatrixBank::new(self.tensor(name)?)
            .with_context(|| format!("invalid GLM matrix bank {name}"))
    }

    pub(super) fn f32(&self, name: &str, dimensions: &[u64]) -> Result<Glm53GgufF32> {
        Glm53GgufF32::new(self.tensor(name)?, dimensions)
            .with_context(|| format!("invalid GLM F32 tensor {name}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_target_and_nextn_topology_is_closed() {
        let descriptors = (0..=45)
            .map(Glm53LayerDescriptor::new)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(
            descriptors[..45]
                .iter()
                .filter(|layer| layer.attention == Glm53AttentionKind::Kda)
                .count(),
            34
        );
        assert_eq!(
            descriptors[..45]
                .iter()
                .filter(|layer| layer.attention == Glm53AttentionKind::Dsa)
                .count(),
            11
        );
        assert_eq!(
            descriptors
                .iter()
                .filter(|layer| layer.ffn == Glm53FfnKind::Dense)
                .count(),
            3
        );
        assert!(descriptors[45].is_nextn);
        assert_eq!(descriptors[45].attention, Glm53AttentionKind::Dsa);
        assert!(!descriptors[45].has_hyper_connections);
        assert!(Glm53LayerDescriptor::new(46).is_err());
    }
}
