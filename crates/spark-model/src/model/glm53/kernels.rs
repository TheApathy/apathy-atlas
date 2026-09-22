// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed capability census for the dedicated GLM-5.3 executor.

use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53RuntimeCapability {
    KdaThreeStreamF32Convolution,
    DsaLatentCommit,
    DsaCurrentTokenComposition,
    DsaLayerComposer,
    CompleteT1Scratch,
    FallibleOwnedStoreTeardown,
}

pub const GLM53_REQUIRED_CAPABILITIES: [Glm53RuntimeCapability; 6] = [
    Glm53RuntimeCapability::KdaThreeStreamF32Convolution,
    Glm53RuntimeCapability::DsaLatentCommit,
    Glm53RuntimeCapability::DsaCurrentTokenComposition,
    Glm53RuntimeCapability::DsaLayerComposer,
    Glm53RuntimeCapability::CompleteT1Scratch,
    Glm53RuntimeCapability::FallibleOwnedStoreTeardown,
];

/// Current capability receipt. The constructor is intentionally private: a
/// caller cannot assert readiness with booleans. Each reviewed runtime tranche
/// must replace its corresponding `false` from inside this module while also
/// retaining the concrete kernel/owner handle that proves availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53KernelAdmission {
    ready: [bool; GLM53_REQUIRED_CAPABILITIES.len()],
}

impl Glm53KernelAdmission {
    pub(crate) const fn current() -> Self {
        Self {
            ready: [false; GLM53_REQUIRED_CAPABILITIES.len()],
        }
    }

    pub fn missing(self) -> Vec<Glm53RuntimeCapability> {
        GLM53_REQUIRED_CAPABILITIES
            .into_iter()
            .zip(self.ready)
            .filter_map(|(capability, ready)| (!ready).then_some(capability))
            .collect()
    }

    pub fn validate(self) -> Result<()> {
        let missing = self.missing();
        if !missing.is_empty() {
            bail!("GLM target executor capabilities are incomplete: {missing:?}");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_receipt_rejects_every_unimplemented_effectful_seam() {
        let admission = Glm53KernelAdmission::current();
        assert_eq!(admission.missing(), GLM53_REQUIRED_CAPABILITIES);
        assert!(admission.validate().is_err());
    }
}
