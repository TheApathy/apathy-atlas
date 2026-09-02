// SPDX-License-Identifier: AGPL-3.0-only

//! Owned weight/view foundation for a future GLM-5.3 target runtime.
//!
//! This module does not execute inference. It closes only the ownership seam
//! between the admitted GGUF device store and its move-only typed device views.

#![allow(dead_code)] // Public foundation consumed when the target constructor is wired.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::{
    gpu::GpuBackend,
    weights::gguf::{GgufDeviceStore, GgufDeviceStoreFreeError},
};
use std::fmt;

use super::{Glm53QuantProfile, PreparedGlm53Target};
use crate::weight_loader::{
    Glm53GgufF32, Glm53GgufMatrix, Glm53NextnWeights, Glm53TargetLayerWeights,
};

const TARGET_LAYERS: usize = 45;

#[derive(Debug)]
struct Glm53TargetRuntimeViews {
    token_embedding: Glm53GgufMatrix,
    output: Glm53GgufMatrix,
    output_norm: Glm53GgufF32,
    target_layers: [Glm53TargetLayerWeights; TARGET_LAYERS],
    native_nextn: Glm53NextnWeights,
}

impl Glm53TargetRuntimeViews {
    fn from_prepared(prepared: PreparedGlm53Target<'_>) -> (Glm53QuantProfile, Self) {
        let (profile, token_embedding, output, output_norm, target_layers, native_nextn) =
            prepared.into_runtime_parts();
        (
            profile,
            Self {
                token_embedding,
                output,
                output_norm,
                target_layers,
                native_nextn,
            },
        )
    }
}

/// Owns the admitted GGUF payload for every typed device view in this object.
///
/// This is a weight container, not a model or executor. The view bundle is
/// declared before `store`, so natural field destruction discards views before
/// the allocation owner. Cleanup errors remain visible through
/// [`GgufDeviceStore::free`] after a consuming handoff.
#[must_use = "owned GLM device weights require explicit free or consuming handoff"]
pub struct Glm53TargetRuntimeWeights {
    profile: Glm53QuantProfile,
    views: Glm53TargetRuntimeViews,
    store: GgufDeviceStore,
}

/// Admission failure that preserves the rejected allocation owner for explicit
/// cleanup. `GgufDeviceStore` cannot free itself because device frees may fail.
#[must_use = "admission failure retains device allocations that require cleanup"]
pub struct Glm53TargetRuntimeAdmissionError {
    reason: anyhow::Error,
    profile: Glm53QuantProfile,
    store: GgufDeviceStore,
}

/// Preserves a construction failure and any device allocations whose cleanup
/// failed. This deliberately is not [`std::error::Error`], so generic error
/// conversion cannot erase its retry-capable owner.
#[must_use = "cleanup failure may retain device allocations that require retry"]
pub struct Glm53OwnedStoreCleanupError {
    phase: &'static str,
    failure: anyhow::Error,
    cleanup_failure: Option<GgufDeviceStoreFreeError>,
}

impl Glm53OwnedStoreCleanupError {
    pub(crate) fn new(
        phase: &'static str,
        failure: anyhow::Error,
        cleanup_failure: Option<GgufDeviceStoreFreeError>,
    ) -> Self {
        Self {
            phase,
            failure,
            cleanup_failure,
        }
    }

    pub fn phase(&self) -> &'static str {
        self.phase
    }

    pub fn failure(&self) -> &anyhow::Error {
        &self.failure
    }

    pub fn cleanup_failure(&self) -> Option<&GgufDeviceStoreFreeError> {
        self.cleanup_failure.as_ref()
    }

    pub fn into_parts(
        self,
    ) -> (
        &'static str,
        anyhow::Error,
        Option<GgufDeviceStoreFreeError>,
    ) {
        (self.phase, self.failure, self.cleanup_failure)
    }

    /// Retry only the allocations retained by a failed cleanup. Success still
    /// returns the original construction failure for caller attribution.
    pub fn retry_cleanup(self, gpu: &dyn GpuBackend) -> std::result::Result<anyhow::Error, Self> {
        let (phase, failure, cleanup_failure) = self.into_parts();
        match retry_cleanup_owner(phase, failure, cleanup_failure, |cleanup| {
            cleanup.retry(gpu)
        }) {
            Ok(failure) => Ok(failure),
            Err((phase, failure, cleanup_failure)) => {
                Err(Self::new(phase, failure, Some(cleanup_failure)))
            }
        }
    }
}

fn retry_cleanup_owner<C>(
    phase: &'static str,
    failure: anyhow::Error,
    cleanup_failure: Option<C>,
    retry: impl FnOnce(C) -> std::result::Result<(), C>,
) -> std::result::Result<anyhow::Error, (&'static str, anyhow::Error, C)> {
    let Some(cleanup_failure) = cleanup_failure else {
        return Ok(failure);
    };
    match retry(cleanup_failure) {
        Ok(()) => Ok(failure),
        Err(cleanup_failure) => Err((phase, failure, cleanup_failure)),
    }
}

impl fmt::Debug for Glm53OwnedStoreCleanupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Glm53OwnedStoreCleanupError")
            .field("phase", &self.phase)
            .field("failure", &self.failure)
            .field("cleanup_failure", &self.cleanup_failure)
            .finish()
    }
}

impl fmt::Display for Glm53OwnedStoreCleanupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "GLM {} failed: {:#}", self.phase, self.failure)?;
        if let Some(cleanup) = &self.cleanup_failure {
            write!(formatter, "; GGUF device cleanup also failed: {cleanup:#}")?;
        }
        Ok(())
    }
}

impl Glm53TargetRuntimeAdmissionError {
    pub fn reason(&self) -> &anyhow::Error {
        &self.reason
    }

    pub fn profile(&self) -> Glm53QuantProfile {
        self.profile
    }

    pub fn tensor_count(&self) -> usize {
        self.store.len()
    }

    pub fn tensor_bytes(&self) -> usize {
        self.store.total_bytes()
    }

    pub fn into_store(self) -> GgufDeviceStore {
        self.store
    }

    pub fn into_parts(self) -> (anyhow::Error, Glm53QuantProfile, GgufDeviceStore) {
        (self.reason, self.profile, self.store)
    }

    /// Consume the rejected owner, attempt every tensor free, and preserve
    /// both the admission failure and any cleanup failure for the caller.
    pub fn cleanup(self, gpu: &dyn GpuBackend) -> Glm53OwnedStoreCleanupError {
        let (failure, _profile, store) = self.into_parts();
        let cleanup_failure = store.free(gpu).err();
        Glm53OwnedStoreCleanupError::new(
            "target runtime weight admission",
            failure,
            cleanup_failure,
        )
    }
}

impl fmt::Debug for Glm53TargetRuntimeAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Glm53TargetRuntimeAdmissionError")
            .field("reason", &self.reason)
            .field("profile", &self.profile)
            .field("tensor_count", &self.store.len())
            .field("tensor_bytes", &self.store.total_bytes())
            .finish()
    }
}

impl fmt::Display for Glm53TargetRuntimeAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GLM target runtime weight admission failed: {:#}",
            self.reason
        )
    }
}

impl Glm53TargetRuntimeWeights {
    /// Admit configuration, profile, payload shape, tensor names, and all typed
    /// views before moving the owning store into the returned container.
    pub fn new(
        profile: Glm53QuantProfile,
        config: &ModelConfig,
        store: GgufDeviceStore,
    ) -> std::result::Result<Self, Glm53TargetRuntimeAdmissionError> {
        let preparation = (|| -> Result<(Glm53QuantProfile, Glm53TargetRuntimeViews)> {
            let prepared = PreparedGlm53Target::new(profile, config, &store)?;
            Ok(Glm53TargetRuntimeViews::from_prepared(prepared))
        })();
        let (admitted_profile, views) = match preparation {
            Ok(prepared) => prepared,
            Err(reason) => {
                return Err(Glm53TargetRuntimeAdmissionError {
                    reason,
                    profile,
                    store,
                });
            }
        };
        Ok(Self {
            profile: admitted_profile,
            views,
            store,
        })
    }

    pub fn profile(&self) -> Glm53QuantProfile {
        self.profile
    }

    pub fn token_embedding(&self) -> &Glm53GgufMatrix {
        &self.views.token_embedding
    }

    pub fn output(&self) -> &Glm53GgufMatrix {
        &self.views.output
    }

    pub fn output_norm(&self) -> &Glm53GgufF32 {
        &self.views.output_norm
    }

    pub fn target_layers(&self) -> &[Glm53TargetLayerWeights] {
        &self.views.target_layers
    }

    pub fn target_layer(&self, index: usize) -> Option<&Glm53TargetLayerWeights> {
        self.views.target_layers.get(index)
    }

    /// Dormant typed layer-45 weights; this accessor does not enable NextN.
    pub fn native_nextn_weights(&self) -> &Glm53NextnWeights {
        &self.views.native_nextn
    }

    pub fn tensor_count(&self) -> usize {
        self.store.len()
    }

    pub fn tensor_bytes(&self) -> usize {
        self.store.total_bytes()
    }

    /// Explicit successful shutdown. There is deliberately no hidden `Drop`:
    /// a device free failure is returned to the owner.
    pub fn free(self, gpu: &dyn GpuBackend) -> std::result::Result<(), GgufDeviceStoreFreeError> {
        self.into_store().free(gpu)
    }

    /// Consume all typed views and return only the allocation owner.
    pub fn into_store(self) -> GgufDeviceStore {
        self.into_parts().1
    }

    /// Consume all typed views while retaining profile identity for cleanup or
    /// a later ownership-preserving constructor.
    pub fn into_parts(self) -> (Glm53QuantProfile, GgufDeviceStore) {
        let Self {
            profile,
            views: _,
            store,
        } = self;
        (profile, store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = include_str!("glm53_runtime.rs");

    fn production_source() -> &'static str {
        SOURCE.split("#[cfg(test)]").next().unwrap()
    }

    #[test]
    fn typed_accessors_are_lifetime_bound_and_handoff_is_consuming() {
        let _: for<'a> fn(
            Glm53QuantProfile,
            &'a ModelConfig,
            GgufDeviceStore,
        ) -> std::result::Result<
            Glm53TargetRuntimeWeights,
            Glm53TargetRuntimeAdmissionError,
        > = Glm53TargetRuntimeWeights::new;
        let _: for<'a> fn(&'a Glm53TargetRuntimeWeights) -> &'a Glm53GgufMatrix =
            Glm53TargetRuntimeWeights::token_embedding;
        let _: for<'a> fn(&'a Glm53TargetRuntimeWeights) -> &'a Glm53GgufMatrix =
            Glm53TargetRuntimeWeights::output;
        let _: for<'a> fn(&'a Glm53TargetRuntimeWeights) -> &'a Glm53GgufF32 =
            Glm53TargetRuntimeWeights::output_norm;
        let _: for<'a> fn(&'a Glm53TargetRuntimeWeights) -> &'a [Glm53TargetLayerWeights] =
            Glm53TargetRuntimeWeights::target_layers;
        let _: for<'a> fn(&'a Glm53TargetRuntimeWeights) -> &'a Glm53NextnWeights =
            Glm53TargetRuntimeWeights::native_nextn_weights;
        let _: fn(Glm53TargetRuntimeWeights) -> GgufDeviceStore =
            Glm53TargetRuntimeWeights::into_store;
        let _: fn(
            Glm53TargetRuntimeWeights,
            &dyn GpuBackend,
        ) -> std::result::Result<(), GgufDeviceStoreFreeError> = Glm53TargetRuntimeWeights::free;
        let _: fn(Glm53TargetRuntimeWeights) -> (Glm53QuantProfile, GgufDeviceStore) =
            Glm53TargetRuntimeWeights::into_parts;
        let _: fn(Glm53TargetRuntimeAdmissionError) -> GgufDeviceStore =
            Glm53TargetRuntimeAdmissionError::into_store;
        let _: fn(
            Glm53TargetRuntimeAdmissionError,
        ) -> (anyhow::Error, Glm53QuantProfile, GgufDeviceStore) =
            Glm53TargetRuntimeAdmissionError::into_parts;
        let _: fn(
            Glm53TargetRuntimeAdmissionError,
            &dyn GpuBackend,
        ) -> Glm53OwnedStoreCleanupError = Glm53TargetRuntimeAdmissionError::cleanup;
        assert_eq!(TARGET_LAYERS, 45);
    }

    #[test]
    fn combined_error_preserves_primary_attribution_after_clean_cleanup() {
        let combined = Glm53OwnedStoreCleanupError::new(
            "behavioral test",
            anyhow::anyhow!("primary admission failure"),
            None,
        );
        assert_eq!(combined.phase(), "behavioral test");
        assert!(
            combined
                .failure()
                .to_string()
                .contains("primary admission failure")
        );
        assert!(combined.cleanup_failure().is_none());
        assert!(combined.to_string().contains("GLM behavioral test failed"));
        let primary = combined
            .retry_cleanup(&spark_runtime::gpu::mock::MockGpuBackend::new())
            .unwrap();
        assert!(primary.to_string().contains("primary admission failure"));
    }

    #[derive(Debug)]
    struct RetryProbe(u8);

    #[test]
    fn retained_cleanup_owner_survives_failure_then_returns_primary() {
        let mut attempts = 0;
        let (phase, failure, probe) = retry_cleanup_owner(
            "behavioral retained cleanup",
            anyhow::anyhow!("primary retained failure"),
            Some(RetryProbe(1)),
            |mut probe| {
                attempts += 1;
                probe.0 -= 1;
                Err(probe)
            },
        )
        .unwrap_err();
        assert_eq!(attempts, 1);
        assert_eq!(phase, "behavioral retained cleanup");
        assert!(failure.to_string().contains("primary retained failure"));
        let primary = retry_cleanup_owner(phase, failure, Some(probe), |probe| {
            attempts += 1;
            assert_eq!(probe.0, 0);
            Ok(())
        })
        .unwrap();
        assert_eq!(attempts, 2);
        assert!(primary.to_string().contains("primary retained failure"));
    }

    #[test]
    fn production_contract_is_consuming_nonerasable_and_has_no_hidden_cleanup() {
        let production = production_source();
        let prepared_source = include_str!("glm53.rs");
        let admission = ["PreparedGlm53Target::new", "(profile, config, &store)?"].concat();
        let model_impl = ["impl ", "Model for"].concat();
        let drop_impl = ["impl ", "Drop for"].concat();
        let unsafe_block = ["unsafe", " {"].concat();
        let compact_source = production.split_whitespace().collect::<Vec<_>>().join(" ");
        let fabricated_store_tensors = ["GgufDevice", "Store { tensors:"].concat();
        let fabricated_store_bytes = ["GgufDevice", "Store { total_bytes:"].concat();
        let owner_start = production
            .find("pub struct Glm53TargetRuntimeWeights")
            .unwrap();
        let error_start = production
            .find("pub struct Glm53TargetRuntimeAdmissionError")
            .unwrap();
        let owner = &production[owner_start..error_start];
        assert_eq!(production.matches(&admission).count(), 1);
        assert!(production.contains("from_prepared(prepared: PreparedGlm53Target<'_>)"));
        assert!(production.contains("prepared.into_runtime_parts()"));
        assert!(!production.contains("copy_from(prepared"));
        assert!(prepared_source.contains("pub(crate) fn into_runtime_parts("));
        assert!(owner.find("views:").unwrap() < owner.find("store:").unwrap());
        assert!(!production.contains(&model_impl));
        assert!(!production.contains(&drop_impl));
        assert!(!production.contains(&unsafe_block));
        assert!(!compact_source.contains(&fabricated_store_tensors));
        assert!(!compact_source.contains(&fabricated_store_bytes));
        assert_eq!(production.matches("store.free(gpu)").count(), 1);
        assert_eq!(production.matches("self.into_store().free(gpu)").count(), 1);
        assert!(!production.contains("impl std::error::Error for Glm53"));
        assert!(production.contains("cleanup_failure = store.free(gpu).err()"));
        let retry_delegate = |source: &str| {
            source.contains(
                "match retry_cleanup_owner(phase, failure, cleanup_failure, |cleanup| {\n            cleanup.retry(gpu)\n        })",
            )
        };
        assert!(retry_delegate(production));
        for bypass in [
            production.replacen(
                "            cleanup.retry(gpu)",
                "            drop(cleanup);\n            Ok(())",
                1,
            ),
            production.replacen(
                "            cleanup.retry(gpu)",
                "            Err(cleanup)",
                1,
            ),
        ] {
            assert_ne!(bypass, production);
            assert!(!retry_delegate(&bypass));
        }
        let pass_through = |source: &str| {
            source.contains("Self {\n            phase,\n            failure,\n            cleanup_failure,\n        }")
                && source.contains("retry_cleanup_owner(phase, failure, cleanup_failure")
                && source.contains("Err(Self::new(phase, failure, Some(cleanup_failure)))")
                && source.contains("\"target runtime weight admission\",\n            failure,\n            cleanup_failure,")
        };
        assert!(pass_through(production));
        for mutant in [
            production.replacen(
                "            cleanup_failure,\n        }",
                "            cleanup_failure: None,\n        }",
                1,
            ),
            production.replacen(
                "            cleanup_failure,\n        )",
                "            None,\n        )",
                1,
            ),
        ] {
            assert!(!pass_through(&mutant));
        }
    }
}
