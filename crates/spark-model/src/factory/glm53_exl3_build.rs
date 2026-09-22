// SPDX-License-Identifier: AGPL-3.0-only

//! Construct the exact EXL3 GLM-5.3 target from an admitted device store.

use anyhow::{Context, Result, anyhow};
use spark_runtime::gpu::GpuBackend;
use std::path::Path;

use crate::model::glm53::Glm53Exl3Model;
use crate::traits::Model;
use crate::weight_loader::{
    Glm53Exl3DeviceStore, Glm53Exl3Files, Glm53Exl3TargetCatalog, Glm53Exl3TargetWeights,
    Glm53Exl3VisionCatalog, materialize_glm53_exl3_native,
};

/// Build the server-owned text+vision EXL3 model after its source store has
/// completed the exact checkpoint load.
pub fn build_glm53_exl3_model(
    files: Glm53Exl3Files,
    store: Glm53Exl3DeviceStore,
    gpu: std::sync::Arc<dyn GpuBackend>,
    max_seq_len: usize,
    dflash2_root: Option<&Path>,
) -> Result<Box<dyn Model>> {
    let positions = u32::try_from(max_seq_len)
        .map_err(|_| anyhow!("GLM-5.3 EXL3 max_seq_len {max_seq_len} exceeds u32"))?;
    let target_catalog = match Glm53Exl3TargetCatalog::new(&files, &store) {
        Ok(catalog) => catalog,
        Err(error) => {
            return Err(cleanup_source(
                error.context("bind GLM-5.3 EXL3 target catalog"),
                store,
                gpu.as_ref(),
            ));
        }
    };
    let vision_catalog = match Glm53Exl3VisionCatalog::new(&files, &store) {
        Ok(catalog) => catalog,
        Err(error) => {
            drop(target_catalog);
            return Err(cleanup_source(
                error.context("bind GLM-5.3 EXL3 vision catalog"),
                store,
                gpu.as_ref(),
            ));
        }
    };
    let native = match materialize_glm53_exl3_native(&target_catalog, gpu.as_ref()) {
        Ok(native) => native,
        Err(error) => {
            drop(vision_catalog);
            drop(target_catalog);
            let materialize = match error.retry_cleanup(gpu.as_ref()) {
                Ok(primary) => primary.context("materialize GLM-5.3 EXL3 native operands"),
                Err(retained) => anyhow!("GLM-5.3 EXL3 native cleanup failed: {retained}"),
            };
            return Err(cleanup_source(materialize, store, gpu.as_ref()));
        }
    };
    let weights = match Glm53Exl3TargetWeights::new(&target_catalog, &native) {
        Ok(weights) => weights,
        Err(error) => {
            drop(vision_catalog);
            drop(target_catalog);
            let native_cleanup = native.free(gpu.as_ref());
            let source_cleanup = store.free(gpu.as_ref());
            return Err(error.context(format!(
                "assemble GLM-5.3 EXL3 target; native cleanup={native_cleanup:?}; source cleanup={source_cleanup:?}"
            )));
        }
    };
    drop(target_catalog);
    let model =
        Glm53Exl3Model::new_multimodal(gpu, store, native, weights, vision_catalog, positions)?;
    if let Some(root) = dflash2_root {
        model
            .install_dflash2(root)
            .context("install GLM-5.3 DFlash2 runtime")?;
    }
    Ok(Box::new(model))
}

fn cleanup_source(
    error: anyhow::Error,
    store: Glm53Exl3DeviceStore,
    gpu: &dyn GpuBackend,
) -> anyhow::Error {
    match store.free(gpu) {
        Ok(()) => error,
        Err(cleanup) => error.context(format!(
            "GLM-5.3 EXL3 source cleanup also failed: {:#}",
            cleanup.failure()
        )),
    }
}
