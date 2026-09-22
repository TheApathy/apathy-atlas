// SPDX-License-Identifier: AGPL-3.0-only

//! Model-boundary request validation while execution remains deliberately off.

use anyhow::{Result, bail, ensure};
use spark_runtime::weights::gguf::Glm53QuantProfile;

use crate::weight_loader::{GLM53_MAX_CONTEXT_TOKENS, Glm53DsaStorage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53TargetModelRequest {
    pub profile: Glm53QuantProfile,
    pub max_batch: u32,
    pub max_positions: u32,
    pub dsa_storage: Glm53DsaStorage,
    pub tensor_parallel: u32,
    pub expert_parallel: u32,
    pub enable_dflash2: bool,
    pub enable_native_nextn: bool,
    pub enable_lora: bool,
    pub enable_prefix_cache: bool,
    pub enable_swap: bool,
    pub enable_graphs: bool,
}

impl Glm53TargetModelRequest {
    pub const fn q2_full_1m_base() -> Self {
        Self {
            profile: Glm53QuantProfile::UdQ2KXl,
            max_batch: 1,
            max_positions: GLM53_MAX_CONTEXT_TOKENS,
            dsa_storage: Glm53DsaStorage::BF16,
            tensor_parallel: 1,
            expert_parallel: 1,
            enable_dflash2: false,
            enable_native_nextn: false,
            enable_lora: false,
            enable_prefix_cache: false,
            enable_swap: false,
            enable_graphs: false,
        }
    }

    pub fn validate(self) -> Result<()> {
        ensure!(
            self.profile == Glm53QuantProfile::UdQ2KXl,
            "full-1M GLM target admission is Q2-only until workspace headroom is measured"
        );
        ensure!(self.max_batch == 1, "GLM target starts at batch one");
        ensure!(
            self.max_positions == GLM53_MAX_CONTEXT_TOKENS,
            "GLM target admission must retain all 1,048,576 positions"
        );
        ensure!(
            self.dsa_storage == Glm53DsaStorage::BF16,
            "GLM target DSA state must remain BF16"
        );
        ensure!(
            self.tensor_parallel == 1 && self.expert_parallel == 1,
            "GLM target starts on one rank"
        );
        ensure!(
            !self.enable_dflash2,
            "DFlash2 is not wired to the target model"
        );
        ensure!(
            !self.enable_native_nextn,
            "native NextN is not wired to the target model"
        );
        ensure!(!self.enable_lora, "GLM target LoRA is unsupported");
        ensure!(
            !self.enable_prefix_cache,
            "GLM target prefix reuse is unsupported"
        );
        ensure!(
            !self.enable_swap,
            "GLM target state swapping is unsupported"
        );
        ensure!(
            !self.enable_graphs,
            "GLM target graph capture is unsupported"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53ModelPhase {
    FreshSingleTokenPrefill,
    SingleTokenDecode,
    ChunkedPrefill,
    DecodeBatch,
    MixedForward,
    SpeculativeVerify,
    Dflash2Draft,
    NativeNextnDraft,
    PrefixReuse,
    Swap,
    Compact,
}

/// All phases are rejected until the capability receipt is complete. The two
/// one-token phases have a distinct diagnostic so later wiring cannot silently
/// turn on any of the unsupported scheduler paths.
pub fn reject_unwired_phase(phase: Glm53ModelPhase) -> Result<()> {
    match phase {
        Glm53ModelPhase::FreshSingleTokenPrefill | Glm53ModelPhase::SingleTokenDecode => {
            bail!("GLM target one-token execution is blocked on runtime capabilities")
        }
        other => bail!("GLM target model phase is unsupported: {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_q2_request_passes_but_every_execution_phase_remains_closed() {
        Glm53TargetModelRequest::q2_full_1m_base()
            .validate()
            .unwrap();
        for phase in [
            Glm53ModelPhase::FreshSingleTokenPrefill,
            Glm53ModelPhase::SingleTokenDecode,
            Glm53ModelPhase::ChunkedPrefill,
            Glm53ModelPhase::DecodeBatch,
            Glm53ModelPhase::MixedForward,
            Glm53ModelPhase::SpeculativeVerify,
            Glm53ModelPhase::Dflash2Draft,
            Glm53ModelPhase::NativeNextnDraft,
            Glm53ModelPhase::PrefixReuse,
            Glm53ModelPhase::Swap,
            Glm53ModelPhase::Compact,
        ] {
            assert!(reject_unwired_phase(phase).is_err());
        }
    }

    #[test]
    fn iq3_fp8_and_optional_paths_fail_before_runtime_admission() {
        let mut request = Glm53TargetModelRequest::q2_full_1m_base();
        request.profile = Glm53QuantProfile::UdIq3Xxs;
        assert!(request.validate().is_err());
        request.profile = Glm53QuantProfile::UdQ2KXl;
        request.dsa_storage = Glm53DsaStorage::FP8;
        assert!(request.validate().is_err());
        request.dsa_storage = Glm53DsaStorage::BF16;
        request.enable_dflash2 = true;
        assert!(request.validate().is_err());
    }
}
