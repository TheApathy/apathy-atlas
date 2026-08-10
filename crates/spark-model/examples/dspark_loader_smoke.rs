// SPDX-License-Identifier: AGPL-3.0-only

//! DSpark drafter loader smoke test: opens the official 0731 drafter shards
//! and runs `load_dspark_drafter` end-to-end on the GPU, printing what
//! landed. Proves the loader against the real checkpoint before the propose
//! forward exists (docs/dspark_port.md, task chain).
//!
//! Usage:
//!   cargo run --release -p spark-model --example dspark_loader_smoke -- \
//!     [drafter_dir] [target_config_dir]
//! Defaults match this machine's layout.

use anyhow::{Context, Result};
use spark_model::weight_loader::deepseek_v4::dspark;
use spark_runtime::gpu::GpuBackend;

use spark_runtime::weights::WeightLoader;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let drafter_dir = args
        .next()
        .unwrap_or_else(|| "/home/flocka/models/DeepSeek-V4-Flash-0731-drafter".into());
    let target_dir = args
        .next()
        .unwrap_or_else(|| "/home/flocka/models/DeepSeek-V4-Flash-162B".into());

    let config_json =
        std::fs::read_to_string(std::path::Path::new(&target_dir).join("config.json"))
            .context("reading target config.json")?;
    let target_config = atlas_core::config::parse_config(&config_json)?;
    println!(
        "target: {} layers, h={}, {} experts (drafter will carry its own count)",
        target_config.num_hidden_layers, target_config.hidden_size, target_config.num_experts
    );

    let backend =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;

    // ATLAS_DSPARK_REF_DRAFT=1 exercises the compact draft (the reference
    // stack's 64-expert subset) through the exact server path.
    let subset = dspark::resolve_ref_draft_subset(std::path::Path::new(&drafter_dir))?;
    if let Some(ref s) = subset {
        println!(
            "compact draft: {} routed experts ({} structured) — {:?}…",
            s.len(),
            s.structured_count,
            &s.checkpoint_ids[..s.len().min(8)]
        );
    }

    let t0 = std::time::Instant::now();
    let mut loader = spark_runtime::weights::SafetensorsLoader::new();
    loader.peak_memory_multiplier = None;
    if let Some(ref s) = subset {
        loader.extra_skip = Some(dspark::compact_draft_skip_fn(s));
    }
    let store = loader
        .load(std::path::Path::new(&drafter_dir), gpu, 0)
        .context("loading drafter shards")?;
    println!(
        "store: {} tensors, {:.2} GB on device in {:.1}s",
        store.len(),
        store.total_bytes() as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    assert!(
        dspark::store_is_dspark(&store),
        "store not detected as DSpark"
    );

    let t1 = std::time::Instant::now();
    let module = dspark::load_dspark_drafter(
        &store,
        &target_config,
        dspark::DsparkParams::V4_FLASH_0731(),
        gpu,
        subset.as_ref(),
    )?;
    println!(
        "drafter assembled in {:.1}s: {} stages, hc_head={}, block_size={}",
        t1.elapsed().as_secs_f64(),
        module.stages.len(),
        module.hc_head.is_some(),
        module.params.block_size,
    );
    for (i, st) in module.stages.iter().enumerate() {
        println!(
            "  stage {i}: wq_a={} wkv={} wo_a={} sink={} attn_norm={}",
            !st.wq_a.weight.is_null(),
            !st.wkv.weight.is_null(),
            !st.wo_a.weight.is_null(),
            !st.attn_sink.weight.is_null(),
            !st.attn_norm.weight.is_null(),
        );
    }
    println!(
        "  heads: main_proj={} main_norm={} norm={} markov_w1={} markov_w2={} confidence={}",
        !module.main_proj.weight.is_null(),
        !module.main_norm.weight.is_null(),
        !module.norm.weight.is_null(),
        !module.markov_w1.weight.is_null(),
        !module.markov_w2.weight.is_null(),
        !module.confidence_proj.weight.is_null(),
    );
    println!("DSPARK LOADER SMOKE: PASS");
    Ok(())
}
