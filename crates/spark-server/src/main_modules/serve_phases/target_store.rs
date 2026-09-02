// SPDX-License-Identifier: AGPL-3.0-only

//! Typed target-checkpoint plans and loaded stores.

#![allow(dead_code)] // Consumed when the staged GLM target constructor is wired into serve.

use anyhow::{Context, Result, anyhow, bail};
use spark_model::factory::{Glm53QuantProfile, ModelSourceKind};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;
use spark_runtime::weights::gguf::{
    GgufDeviceLoadError, GgufDeviceStore, load_glm53_store, open_glm53_files,
};
use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};

const GLM53_GGUF_SHARDS: usize = 4;

/// A target-store load failure that cannot erase retained device allocations.
#[must_use = "a failed target load may retain device allocations requiring cleanup"]
pub(in crate::main_modules) enum TargetStoreLoadError {
    Admission(anyhow::Error),
    Device(GgufDeviceLoadError),
}

impl fmt::Debug for TargetStoreLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => formatter.debug_tuple("Admission").field(error).finish(),
            Self::Device(error) => formatter.debug_tuple("Device").field(error).finish(),
        }
    }
}

impl fmt::Display for TargetStoreLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => write!(formatter, "target-store admission failed: {error:#}"),
            Self::Device(error) => write!(formatter, "{error}"),
        }
    }
}

/// Target weights kept in their physical source representation.
///
/// GGUF tensors remain raw quantized device tensors; they are never projected
/// into the safetensors-only `WeightStore` / `WeightDtype` ABI.
pub(in crate::main_modules) enum LoadedTargetStore {
    Safetensors(WeightStore),
    Glm53Gguf {
        profile: Glm53QuantProfile,
        store: GgufDeviceStore,
    },
}

impl LoadedTargetStore {
    pub(in crate::main_modules) fn source_kind(&self) -> ModelSourceKind {
        match self {
            Self::Safetensors(_) => ModelSourceKind::Safetensors,
            Self::Glm53Gguf { profile, .. } => ModelSourceKind::Glm53Gguf(*profile),
        }
    }

    pub(in crate::main_modules) fn tensor_count(&self) -> usize {
        match self {
            Self::Safetensors(store) => store.len(),
            Self::Glm53Gguf { store, .. } => store.len(),
        }
    }

    pub(in crate::main_modules) fn total_bytes(&self) -> usize {
        match self {
            Self::Safetensors(store) => store.total_bytes(),
            Self::Glm53Gguf { store, .. } => store.total_bytes(),
        }
    }

    pub(in crate::main_modules) fn as_safetensors(&self) -> Result<&WeightStore> {
        match self {
            Self::Safetensors(store) => Ok(store),
            Self::Glm53Gguf { .. } => {
                bail!("GLM-5.3 GGUF store cannot be used as a safetensors WeightStore")
            }
        }
    }

    /// Consuming form of [`Self::as_glm53_gguf`].
    ///
    /// `Glm53TargetRuntimeWeights::new` takes the store by value (it retains the
    /// device allocations for the life of the model), so building a served GLM
    /// model needs ownership, not a borrow.
    pub(in crate::main_modules) fn into_glm53_gguf(
        self,
    ) -> Result<(Glm53QuantProfile, GgufDeviceStore)> {
        match self {
            Self::Glm53Gguf { profile, store } => Ok((profile, store)),
            Self::Safetensors(_) => {
                bail!("safetensors WeightStore cannot be used as a GLM-5.3 GGUF store")
            }
        }
    }

    pub(in crate::main_modules) fn as_glm53_gguf(
        &self,
    ) -> Result<(Glm53QuantProfile, &GgufDeviceStore)> {
        match self {
            Self::Glm53Gguf { profile, store } => Ok((*profile, store)),
            Self::Safetensors(_) => {
                bail!("safetensors WeightStore cannot be used as a GLM-5.3 GGUF store")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::main_modules) enum TargetStoreLoadPlan {
    Safetensors(SafetensorsLoadPlan),
    Glm53Gguf(Glm53GgufLoadPlan),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::main_modules) struct SafetensorsLoadPlan {
    model_directory: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::main_modules) struct Glm53GgufLoadPlan {
    profile: Glm53QuantProfile,
    shard_paths: [PathBuf; GLM53_GGUF_SHARDS],
}

impl TargetStoreLoadPlan {
    /// Admits an already resolved legacy model directory without opening it.
    pub(in crate::main_modules) fn safetensors(model_directory: PathBuf) -> Result<Self> {
        if model_directory.as_os_str().is_empty() {
            bail!("safetensors target directory must not be empty");
        }
        Ok(Self::Safetensors(SafetensorsLoadPlan { model_directory }))
    }

    /// Admits the ordered output of the exact-four GLM filesystem resolver.
    ///
    /// The resolver owns filename and file-type validation. This boundary
    /// additionally rejects relative, cross-directory, or aliased arrays
    /// before a future loader can open any shard.
    pub(in crate::main_modules) fn resolved_glm53(
        profile: Glm53QuantProfile,
        shard_paths: [PathBuf; GLM53_GGUF_SHARDS],
    ) -> Result<Self> {
        let first_parent = admitted_parent(&shard_paths[0])?;
        let mut unique = HashSet::with_capacity(GLM53_GGUF_SHARDS);
        for (index, path) in shard_paths.iter().enumerate() {
            if !path.is_absolute() {
                bail!("resolved GLM-5.3 GGUF shard paths must be absolute");
            }
            if path.file_name().and_then(|name| name.to_str())
                != Some(profile.canonical_file_names()[index])
            {
                bail!("resolved GLM-5.3 GGUF shard does not match its quant profile");
            }
            let parent = admitted_parent(path)?;
            if parent != first_parent {
                bail!("resolved GLM-5.3 GGUF shards must share one directory");
            }
            if !unique.insert(path) {
                bail!("resolved GLM-5.3 GGUF shard paths must be distinct");
            }
        }
        Ok(Self::Glm53Gguf(Glm53GgufLoadPlan {
            profile,
            shard_paths,
        }))
    }

    pub(in crate::main_modules) fn source_kind(&self) -> ModelSourceKind {
        match self {
            Self::Safetensors(_) => ModelSourceKind::Safetensors,
            Self::Glm53Gguf(plan) => ModelSourceKind::Glm53Gguf(plan.profile),
        }
    }

    /// Consume an admitted GLM plan, open its exact pinned profile, and load
    /// the raw quantized tensors into their typed device store.
    ///
    /// A safetensors plan is rejected by the initial match before any path is
    /// opened or any device allocation is attempted. Device-load failures keep
    /// their non-generic retry owners intact.
    pub(in crate::main_modules) fn load_glm53_gguf(
        self,
        gpu: &dyn GpuBackend,
        reserve_bytes: usize,
    ) -> std::result::Result<LoadedTargetStore, TargetStoreLoadError> {
        let plan = match self {
            Self::Glm53Gguf(plan) => plan,
            Self::Safetensors(_) => {
                return Err(TargetStoreLoadError::Admission(anyhow!(
                    "safetensors target plan cannot be loaded through the GLM-5.3 GGUF path"
                )));
            }
        };
        let profile = plan.profile;
        let mut files = open_glm53_files(profile, &plan.shard_paths)
            .map_err(TargetStoreLoadError::Admission)?;
        let store = load_glm53_store(&mut files, gpu, reserve_bytes)
            .map_err(TargetStoreLoadError::Device)?;
        Ok(LoadedTargetStore::Glm53Gguf { profile, store })
    }

    pub(in crate::main_modules) fn safetensors_directory(&self) -> Result<&Path> {
        match self {
            Self::Safetensors(plan) => Ok(&plan.model_directory),
            Self::Glm53Gguf(_) => bail!("GLM-5.3 GGUF load plan has no safetensors directory"),
        }
    }

    pub(in crate::main_modules) fn glm53_shards(
        &self,
    ) -> Result<(Glm53QuantProfile, &[PathBuf; GLM53_GGUF_SHARDS])> {
        match self {
            Self::Glm53Gguf(plan) => Ok((plan.profile, &plan.shard_paths)),
            Self::Safetensors(_) => bail!("safetensors load plan has no GLM-5.3 GGUF shards"),
        }
    }
}

fn admitted_parent(path: &Path) -> Result<&Path> {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .context("resolved GLM-5.3 GGUF shard path has no parent directory")
}

#[cfg(test)]
#[path = "target_store_test_sha256.rs"]
mod target_store_test_sha256;

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    const SOURCE: &str = include_str!("target_store.rs");
    const PRODUCTION_SHA256: &str =
        "c766f492a999fa917aebabfe89fa8faf85f1030787497f26cd4c447b2ffaf82c";
    const LOAD_METHOD_SHA256: &str =
        "ba1571ba4a9f3d79f99305e99ec03f46d7c5f3ae069fb7976545f3b28b687e2e";
    const LOAD_METHOD_START: &str = "    pub(in crate::main_modules) fn load_glm53_gguf(";
    const LOAD_METHOD_END: &str = "    pub(in crate::main_modules) fn safetensors_directory(";

    fn source_sha256(source: &str) -> String {
        target_store_test_sha256::hex(target_store_test_sha256::digest(source.as_bytes()))
    }

    fn production_source() -> &'static str {
        SOURCE.split("#[cfg(test)]").next().unwrap()
    }

    fn load_method(source: &str) -> Option<&str> {
        if source.matches(LOAD_METHOD_START).count() != 1
            || source.matches(LOAD_METHOD_END).count() != 1
        {
            return None;
        }
        let start = source.find(LOAD_METHOD_START)?;
        let end = source.find(LOAD_METHOD_END)?;
        (start < end).then_some(&source[start..end])
    }

    fn production_contract(source: &str) -> bool {
        source_sha256(source) == PRODUCTION_SHA256
            && load_method(source).is_some_and(|method| source_sha256(method) == LOAD_METHOD_SHA256)
    }

    fn resolved_paths(profile: Glm53QuantProfile) -> [PathBuf; GLM53_GGUF_SHARDS] {
        std::array::from_fn(|index| {
            PathBuf::from("/models/glm53").join(profile.canonical_file_names()[index])
        })
    }

    #[test]
    fn empty_safetensors_store_stays_typed() {
        let store = LoadedTargetStore::Safetensors(WeightStore::empty());
        assert_eq!(store.source_kind(), ModelSourceKind::Safetensors);
        assert_eq!(store.tensor_count(), 0);
        assert_eq!(store.total_bytes(), 0);
        assert!(store.as_safetensors().is_ok());
        assert!(store.as_glm53_gguf().is_err());
    }

    #[test]
    fn load_plans_expose_only_the_matching_source() {
        let legacy = TargetStoreLoadPlan::safetensors(PathBuf::from("model-cache")).unwrap();
        assert_eq!(legacy.source_kind(), ModelSourceKind::Safetensors);
        assert_eq!(
            legacy.safetensors_directory().unwrap(),
            Path::new("model-cache")
        );
        assert!(legacy.glm53_shards().is_err());

        for profile in [Glm53QuantProfile::UdQ2KXl, Glm53QuantProfile::UdIq3Xxs] {
            let paths = resolved_paths(profile);
            let glm = TargetStoreLoadPlan::resolved_glm53(profile, paths.clone()).unwrap();
            assert_eq!(glm.source_kind(), ModelSourceKind::Glm53Gguf(profile));
            assert_eq!(glm.glm53_shards().unwrap(), (profile, &paths));
            assert!(glm.safetensors_directory().is_err());
        }
    }

    #[test]
    fn glm_plan_rejects_relative_cross_directory_and_duplicate_paths() {
        let profile = Glm53QuantProfile::PRIMARY;
        let mut relative = resolved_paths(profile);
        relative[0] = PathBuf::from("part-1.gguf");
        assert!(TargetStoreLoadPlan::resolved_glm53(profile, relative).is_err());

        let mut cross_directory = resolved_paths(profile);
        cross_directory[3] = PathBuf::from("/other/part-4.gguf");
        assert!(TargetStoreLoadPlan::resolved_glm53(profile, cross_directory).is_err());

        let mut duplicate = resolved_paths(profile);
        duplicate[3] = duplicate[2].clone();
        assert!(TargetStoreLoadPlan::resolved_glm53(profile, duplicate).is_err());

        let iq3_paths = resolved_paths(Glm53QuantProfile::UdIq3Xxs);
        assert!(TargetStoreLoadPlan::resolved_glm53(profile, iq3_paths).is_err());
    }

    #[test]
    fn empty_target_paths_fail_before_io() {
        assert!(TargetStoreLoadPlan::safetensors(PathBuf::new()).is_err());
        let empty: [PathBuf; GLM53_GGUF_SHARDS] = std::array::from_fn(|_| PathBuf::new());
        assert!(TargetStoreLoadPlan::resolved_glm53(Glm53QuantProfile::PRIMARY, empty).is_err());
    }

    #[test]
    fn consuming_glm_loader_rejects_safetensors_before_io_or_allocation() {
        let gpu = MockGpuBackend::new();
        let legacy = TargetStoreLoadPlan::safetensors(PathBuf::from(
            "/definitely-absent/legacy-safetensors-model",
        ))
        .unwrap();
        let error = legacy
            .load_glm53_gguf(&gpu, usize::MAX)
            .err()
            .expect("safetensors must reject before I/O");
        assert!(
            error
                .to_string()
                .contains("cannot be loaded through the GLM-5.3 GGUF path")
        );
        assert_eq!(gpu.alloc_count(), 0);
    }

    #[test]
    fn glm_load_signature_and_rejection_order_are_pinned() {
        let _: for<'a> fn(
            TargetStoreLoadPlan,
            &'a dyn GpuBackend,
            usize,
        ) -> std::result::Result<LoadedTargetStore, TargetStoreLoadError> =
            TargetStoreLoadPlan::load_glm53_gguf;

        let production = production_source();
        assert!(production_contract(production));
        let method = load_method(production).unwrap();
        let reject = method.find("Self::Safetensors(_)").unwrap();
        let profile = method.find("let profile = plan.profile;").unwrap();
        let open = method
            .find("open_glm53_files(profile, &plan.shard_paths)")
            .unwrap();
        let load = method
            .find("load_glm53_store(&mut files, gpu, reserve_bytes)")
            .unwrap();
        let returned_profile = method
            .find("LoadedTargetStore::Glm53Gguf { profile, store }")
            .unwrap();
        assert!(reject < profile && profile < open && open < load && load < returned_profile);
        assert_eq!(
            method
                .matches(".map_err(TargetStoreLoadError::Admission)?")
                .count(),
            1
        );
        assert_eq!(
            method
                .matches(".map_err(TargetStoreLoadError::Device)?")
                .count(),
            1
        );

        let erased = method.replace(
            ".map_err(TargetStoreLoadError::Device)?",
            ".map_err(|error| TargetStoreLoadError::Admission(anyhow!(error.to_string())))?",
        );
        let erased = production.replacen(method, &erased, 1);
        assert!(!production_contract(&erased));
        let aliased_error = format!(
            "{production}\nuse std::error::Error as Erased;\nimpl Erased for TargetStoreLoadError {{}}\n"
        );
        assert!(!production_contract(&aliased_error));
        let no_must_use = production.replacen(
            "#[must_use = \"a failed target load may retain device allocations requiring cleanup\"]\n",
            "",
            1,
        );
        assert!(!production_contract(&no_must_use));
    }
}
