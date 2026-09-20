// SPDX-License-Identifier: AGPL-3.0-only

//! Physical full-token diagnostic for the pinned GLM-5.3 EXL3 checkpoint.

use std::time::Instant;

use anyhow::{Context, Result, bail};
use spark_model::model::glm53::Glm53Exl3Model;
use spark_model::weight_loader::{
    Glm53Exl3TargetCatalog, Glm53Exl3TargetWeights, admit_glm53_exl3_files, load_glm53_exl3_store,
    materialize_glm53_exl3_native,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

const RESERVE_BYTES: usize = 12 * 1024 * 1024 * 1024;
const CONTEXT: u32 = 256;
const DEFAULT_FIRST_TOKEN: u32 = 151_331;

fn main() -> Result<()> {
    let root = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .context("usage: glm53_exl3_decode <exact-checkpoint-directory> [first-token]")?;
    let first_token = std::env::args()
        .nth(2)
        .map(|value| value.parse())
        .transpose()
        .context("first token must be u32")?
        .unwrap_or(DEFAULT_FIRST_TOKEN);
    let files = admit_glm53_exl3_files(&root)?;
    let backend = Box::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
    let gpu: &dyn GpuBackend = backend.as_ref();
    let store = match load_glm53_exl3_store(&files, gpu, RESERVE_BYTES) {
        Ok(store) => store,
        Err(error) => match error.retry_cleanup(gpu) {
            Ok(primary) => return Err(primary),
            Err(retained) => bail!("EXL3 load and cleanup failed: {retained}"),
        },
    };
    let catalog = Glm53Exl3TargetCatalog::new(&files, &store)?;
    let native = materialize_glm53_exl3_native(&catalog, gpu).map_err(|error| {
        let retained = error.retained_bytes();
        match error.retry_cleanup(gpu) {
            Ok(primary) => primary.context(format!(
                "materialize GLM EXL3 native operands; retained before cleanup={retained}"
            )),
            Err(error) => anyhow::anyhow!("{error}; retained={}", error.retained_bytes()),
        }
    })?;
    let weights = Glm53Exl3TargetWeights::new(&catalog, &native)?;
    drop(catalog);
    let model = Glm53Exl3Model::new(backend, store, native, weights, CONTEXT)?;
    let stream = 0;
    let mut token = first_token;
    let mut rows = Vec::new();
    for step in 0..2u32 {
        let started = Instant::now();
        let logits = model.decode_token(token, stream)?;
        let elapsed = started.elapsed();
        let next = model.argmax_host(logits)?;
        rows.push((step, token, next, elapsed.as_secs_f64()));
        token = next;
    }
    model.free()?;
    for (step, input, next, seconds) in rows {
        println!(
            "TOKEN step={step} input={input} argmax={next} seconds={seconds:.6} target_tok_s={:.6}",
            1.0 / seconds
        );
    }
    println!("RESULT: PASS full_tokens=2 context={CONTEXT}");
    Ok(())
}
