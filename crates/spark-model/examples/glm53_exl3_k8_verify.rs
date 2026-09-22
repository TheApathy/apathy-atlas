// SPDX-License-Identifier: AGPL-3.0-only

//! Fixed-token K8 target diagnostic for same-binary candidate/control timing.

use std::time::Instant;

use anyhow::{Context, Result, bail};
use spark_model::model::glm53::Glm53Exl3Model;
use spark_model::weight_loader::{
    Glm53Exl3TargetCatalog, Glm53Exl3TargetWeights, admit_glm53_exl3_files, load_glm53_exl3_store,
    materialize_glm53_exl3_native,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

const RESERVE_BYTES: usize = 8 * 1024 * 1024 * 1024;
const CAPACITY: u32 = 256;
const CONTEXT: [u32; 8] = [151_331, 963, 8_568, 14, 21, 8, 374, 372];
const BLOCKS: [[u32; 8]; 4] = [
    [963, 963, 8_568, 8_568, 14, 21, 8, 374],
    [372, 1_112, 11, 220, 80_637, 374, 279, 6_133],
    [20358, 372, 372, 220, 6133, 11, 11, 264],
    [8568, 1112, 14, 21, 8, 374, 963, 372],
];

fn main() -> Result<()> {
    let root = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .context("usage: glm53_exl3_k8_verify <exact-checkpoint-directory>")?;
    let files = admit_glm53_exl3_files(&root)?;
    let backend: std::sync::Arc<dyn spark_runtime::gpu::GpuBackend> =
        std::sync::Arc::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
    let gpu: &dyn GpuBackend = backend.as_ref();
    let store = match load_glm53_exl3_store(&files, gpu, RESERVE_BYTES) {
        Ok(store) => store,
        Err(error) => match error.retry_cleanup(gpu) {
            Ok(primary) => return Err(primary),
            Err(retained) => bail!("EXL3 load and cleanup failed: {retained}"),
        },
    };
    let catalog = Glm53Exl3TargetCatalog::new(&files, &store)?;
    let native =
        materialize_glm53_exl3_native(&catalog, gpu).map_err(|error| anyhow::anyhow!("{error}"))?;
    let weights = Glm53Exl3TargetWeights::new(&catalog, &native)?;
    drop(catalog);
    let model = Glm53Exl3Model::new(backend, store, native, weights, CAPACITY)?;
    let stream = model.gpu().default_stream();

    for &token in &CONTEXT {
        model.decode_token(token, stream)?;
    }
    for (cycle, block) in BLOCKS.iter().enumerate() {
        let started = Instant::now();
        let logits = model.verify_tokens_full(block, stream)?;
        let seconds = started.elapsed().as_secs_f64();
        let oracle = model.argmax_rows_host(logits, block.len())?;
        println!(
            "K8 cycle={cycle} position={} tokens={block:?} oracle={oracle:?} seconds={seconds:.9} target_tok_s={:.6}",
            CONTEXT.len() + cycle * block.len(),
            block.len() as f64 / seconds,
        );
    }
    model.free()?;
    println!(
        "RESULT: PASS fixed_context={} fixed_k8_cycles={}",
        CONTEXT.len(),
        BLOCKS.len()
    );
    Ok(())
}
