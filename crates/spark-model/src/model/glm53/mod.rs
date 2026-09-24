// SPDX-License-Identifier: AGPL-3.0-only

//! Dedicated GLM-5.3 target ownership and admission foundation.
//!
//! This module is intentionally not registered from `model/mod.rs` yet. It
//! owns no GPU allocation and implements no [`crate::traits::Model`] method;
//! admission stays closed until every effectful capability has a reviewed
//! concrete implementation.

use std::fmt;

use anyhow::Result;
use spark_runtime::{gpu::GpuBackend, weights::gguf::GgufDeviceStoreFreeError};

use crate::factory::{Glm53OwnedStoreCleanupError, Glm53TargetRuntimeWeights};

mod arena;
mod arena_owner;
mod b1t1_bootstrap;
mod capture_slots;
mod cublaslt_prewarm;
mod ffn_graph;
mod ffn_graph_runtime;
mod dense_ffn;
mod device_completion;
mod dflash2_probe_capture;
mod dflash2_probe_contract;
mod dflash2_runtime;
pub use dflash2_probe_capture::{ProbeCapture, ProbeFrame, ProbeIo};
pub use dflash2_probe_contract::{Dflash2ProbeMode, ProbeLayout, ProbeStage};
mod dispatch;
mod dsa_attention;
mod dsa_dense_indexer;
mod dsa_verify_plan;
mod dsa_verify_transaction;
mod executor;
mod kda_attention;
mod kda_recurrent_commit;
mod model_trait;
mod model_trait_exl3;
pub(crate) mod oracle_dump;
mod ordered_capture;
mod owned_verify_readback;
mod partial_replay;
mod phase_timing;
mod phase_timing_wrappers;
mod prefill_capture_ingest;
mod prefix_commit;
pub(crate) mod speculative_admission;
mod prefill_capture_owner;
mod prefill_capture_plan;
mod prefill_exl3;
mod prefill_input_exl3;
mod prefill_owner_exl3;
mod state_probe_frame;
mod state_read_plan;
pub use state_probe_frame::StateProbeStamp;
mod t1_state_transaction;
mod target_model;
mod target_model_exl3;
pub mod verify_policy_binding;
pub mod verify_policy_transaction;
mod workspace_binding;
pub use dflash2_runtime::{Glm53Dflash2ProbeObserver, Glm53Dflash2Runtime};
pub use dispatch::Glm53Dispatcher;
pub use executor::{Glm53Executor, Glm53ExecutorReadiness, Seam};
pub use workspace_binding::Glm53BoundWorkspace;
mod forward_one;
mod kda_state_binding;
mod kernels;
mod mhc_expansion;
mod model_impl;
mod walk_dump;
mod walk_scratch;
pub(crate) mod walk_timing;

pub use arena::{
    GLM53_FULL_1M_PERSISTENT_BYTES, GLM53_KNOWN_ARENA_BYTES, GLM53_MHC_EXPANDED_F32_BYTES,
    GLM53_T1_WORKSPACE_BYTES, Glm53ArenaPlan, Glm53ArenaRegion,
};
pub use forward_one::{
    GLM53_CAPTURE_LAYERS, GLM53_DENSE_LAYERS, GLM53_DSA_LAYERS, GLM53_KDA_LAYERS, GLM53_MOE_LAYERS,
    GLM53_TARGET_EVENTS, GLM53_TARGET_LAYERS, Glm53ForwardOnePlan,
};
pub use kda_state_binding::Glm53KdaScratchState;
pub use kernels::{GLM53_REQUIRED_CAPABILITIES, Glm53KernelAdmission, Glm53RuntimeCapability};
pub use mhc_expansion::{GLM53_MHC_FUNCTION_F32_BYTES, Glm53HyperBranch, Glm53MhcExpanded};
pub use model_impl::{Glm53ModelPhase, Glm53TargetModelRequest, reject_unwired_phase};
pub use target_model::{GLM53_BRINGUP_ENV, Glm53Model};
pub use target_model_exl3::Glm53Exl3Model;
pub use target_model_exl3::{Glm53StateProbe, StateProbeDescriptor, StateProbeRegion};

/// Fully owned but currently unreachable admitted object. Construction performs
/// only CPU validation; the capability gate fails before this value can exist.
#[must_use = "owned GLM target admission requires explicit free or consuming handoff"]
pub struct Glm53TargetModelAdmission {
    weights: Glm53TargetRuntimeWeights,
    arena: Glm53ArenaPlan,
    forward_one: Glm53ForwardOnePlan,
}

#[must_use = "model admission failure retains device weights that require cleanup"]
pub struct Glm53TargetModelAdmissionError {
    reason: anyhow::Error,
    weights: Glm53TargetRuntimeWeights,
}

impl Glm53TargetModelAdmission {
    pub fn new(
        weights: Glm53TargetRuntimeWeights,
        request: Glm53TargetModelRequest,
    ) -> std::result::Result<Self, Glm53TargetModelAdmissionError> {
        let validated = (|| -> Result<(Glm53ArenaPlan, Glm53ForwardOnePlan)> {
            request.validate()?;
            anyhow::ensure!(
                request.profile == weights.profile(),
                "GLM target request/profile does not match its owned GGUF payload"
            );
            let arena = Glm53ArenaPlan::exact_full_1m_b1()?;
            let forward_one = Glm53ForwardOnePlan::exact()?;
            Glm53KernelAdmission::current().validate()?;
            Ok((arena, forward_one))
        })();
        match validated {
            Ok((arena, forward_one)) => Ok(Self {
                weights,
                arena,
                forward_one,
            }),
            Err(reason) => Err(Glm53TargetModelAdmissionError { reason, weights }),
        }
    }

    pub fn weights(&self) -> &Glm53TargetRuntimeWeights {
        &self.weights
    }

    pub fn arena(&self) -> Glm53ArenaPlan {
        self.arena
    }

    pub fn forward_one(&self) -> &Glm53ForwardOnePlan {
        &self.forward_one
    }

    pub fn into_weights(self) -> Glm53TargetRuntimeWeights {
        self.weights
    }

    /// Explicit successful shutdown. Every GGUF tensor free is attempted and
    /// any device error is returned; no `Drop` path can hide it.
    pub fn free(self, gpu: &dyn GpuBackend) -> std::result::Result<(), GgufDeviceStoreFreeError> {
        self.into_weights().free(gpu)
    }
}

impl Glm53TargetModelAdmissionError {
    pub fn reason(&self) -> &anyhow::Error {
        &self.reason
    }

    pub fn weights(&self) -> &Glm53TargetRuntimeWeights {
        &self.weights
    }

    pub fn into_weights(self) -> Glm53TargetRuntimeWeights {
        self.weights
    }

    pub fn into_parts(self) -> (anyhow::Error, Glm53TargetRuntimeWeights) {
        (self.reason, self.weights)
    }

    /// Consume a failed model admission and attempt cleanup without replacing
    /// the construction failure if device teardown also fails.
    pub fn cleanup(self, gpu: &dyn GpuBackend) -> Glm53OwnedStoreCleanupError {
        let (failure, weights) = self.into_parts();
        let cleanup_failure = weights.free(gpu).err();
        Glm53OwnedStoreCleanupError::new("target model admission", failure, cleanup_failure)
    }
}

impl fmt::Debug for Glm53TargetModelAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Glm53TargetModelAdmissionError")
            .field("reason", &self.reason)
            .field("profile", &self.weights.profile())
            .field("tensor_count", &self.weights.tensor_count())
            .field("tensor_bytes", &self.weights.tensor_bytes())
            .finish()
    }
}

impl fmt::Display for Glm53TargetModelAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GLM target model admission failed: {:#}",
            self.reason
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MOD_SOURCE: &str = include_str!("mod.rs");

    fn production_source() -> &'static str {
        MOD_SOURCE.split("#[cfg(test)]").next().unwrap()
    }

    /// Renamed from `ownership_error_is_consuming_and_model_trait_is_not_enabled`
    /// on 2026-09-01, when `impl Model for Glm53Model` landed in
    /// `model_trait.rs`.
    ///
    /// The old assertion was `!production.contains("impl Model for Glm53")`
    /// against **`mod.rs` only**, so moving the impl into a sibling file would
    /// have satisfied it by technicality while enabling the very thing it
    /// existed to prevent. The property it actually protected is that the
    /// *runtime* gate stays closed, so that is what is asserted now: the
    /// capability admission must still refuse, and the trait must not be
    /// reachable without either a reviewed capability set or the explicitly
    /// named bring-up escape.
    #[test]
    fn ownership_error_is_consuming_and_admission_still_refuses() {
        let _: fn(Glm53TargetModelAdmissionError) -> Glm53TargetRuntimeWeights =
            Glm53TargetModelAdmissionError::into_weights;
        let _: fn(
            Glm53TargetModelAdmission,
            &dyn GpuBackend,
        ) -> std::result::Result<(), GgufDeviceStoreFreeError> = Glm53TargetModelAdmission::free;
        let _: fn(Glm53TargetModelAdmissionError, &dyn GpuBackend) -> Glm53OwnedStoreCleanupError =
            Glm53TargetModelAdmissionError::cleanup;
        let production = production_source();
        // The gate is a runtime check now, not the absence of an impl.
        assert!(
            Glm53KernelAdmission::current().validate().is_err(),
            "kernel admission must stay closed until the capabilities are reviewed"
        );
        // And the model must refuse to construct on that basis alone. This is
        // the assertion that replaces the old source-text check.
        let refusal = format!("{:#}", target_model::Glm53Model::admit().unwrap_err());
        assert!(
            refusal.contains("admission is closed"),
            "refusal must name the closed gate: {refusal}"
        );
        assert!(
            refusal.contains(target_model::GLM53_BRINGUP_ENV),
            "refusal must name the escape it is not taking: {refusal}"
        );
        // The impl exists, in its own file, and is not smuggled into mod.rs.
        assert!(!production.contains("impl Model for"));
        assert!(!production.contains("impl Drop for"));
        assert!(!production.contains("impl std::error::Error for Glm53"));
        assert_eq!(
            production.matches("self.into_weights().free(gpu)").count(),
            1
        );
        assert_eq!(production.matches("weights.free(gpu).err()").count(), 1);
        let cleanup_passes_owner = |source: &str| {
            source.contains(
                "Glm53OwnedStoreCleanupError::new(\"target model admission\", failure, cleanup_failure)",
            )
        };
        assert!(cleanup_passes_owner(production));
        let drops_owner = production.replacen(
            "failure, cleanup_failure)",
            "failure, { drop(cleanup_failure); None })",
            1,
        );
        assert!(!cleanup_passes_owner(&drops_owner));
        assert_eq!(GLM53_TARGET_EVENTS, 234);
        assert_eq!(GLM53_FULL_1M_PERSISTENT_BYTES, 12_864_209_664);
    }
}

#[cfg(test)]
mod prefill_capture_tile_tests;
#[cfg(test)]
mod prefill_mixed_exl3_wiring_tests;

#[cfg(test)]
mod dsa_dense_indexer_tests;
#[cfg(test)]
mod dsa_dense_indexer_wiring_tests;
