// SPDX-License-Identifier: AGPL-3.0-only
//! Drop gate for the served DeepSeek-V4.1 model (TUI model swap): build `Dsv41Model`, run one
//! short prefill so a sequence exists, DROP it, and require that at least 95% of the memory the
//! model took comes back in MemAvailable.
//!
//! `--control leak` `mem::forget`s the model instead. It must FAIL: that proves the gate can see
//! a leak (the model's device memory is unified memory on GB10, so it is visible to the host).
//!
//! ```text
//! ATLAS_DSV41_PACKED_KEEP=6 cargo run -p spark-model --release --example dsv41_drop_gate \
//!   --features cuda,gpu-examples -- [--control leak]
//! ```

use anyhow::{Context, Result, bail, ensure};
use std::path::Path;
use std::sync::Arc;

use atlas_core::config::parse_config;
use spark_model::model::dsv41::Dsv41Model;
use spark_model::traits::Model;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightLoader};

const MODEL_DIR: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K";

fn mem_available_kb() -> Result<u64> {
    let s = std::fs::read_to_string("/proc/meminfo")?;
    s.lines()
        .find_map(|l| l.strip_prefix("MemAvailable:").map(|r| r.trim().trim_end_matches(" kB").trim().parse::<u64>()))
        .context("no MemAvailable")?
        .map_err(anyhow::Error::from)
}

fn settle() -> Result<u64> {
    // MemAvailable moves for a moment after large frees; take the max of a few samples.
    let mut best = 0;
    for _ in 0..10 {
        best = best.max(mem_available_kb()?);
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    Ok(best)
}

fn main() -> Result<()> {
    let leak = match std::env::args().nth(1).as_deref() {
        None => false,
        Some("--control") => match std::env::args().nth(2).as_deref() {
            Some("leak") => true,
            o => bail!("--control leak, got {o:?}"),
        },
        Some(o) => bail!("unknown argument {o}"),
    };
    let config = parse_config(&std::fs::read_to_string(format!("{MODEL_DIR}/config.json"))?)?;
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let mut loader = SafetensorsLoader::new();
    loader.extra_skip = Some(Arc::new(spark_model::weight_loader::deepseek_v41::skip_tensor_for_serving));
    let store = loader.load(Path::new(MODEL_DIR), &gpu, 0)?;
    let before = settle()?;
    println!("MemAvailable before model: {:.2} GB (store resident: {:.2} GB)", before as f64 / 1048576.0, store.total_bytes() as f64 / 1e9);

    let model = Dsv41Model::new(&config, &store, Box::new(gpu), Path::new(MODEL_DIR), 8192, 512)?;
    let mut seq = model.alloc_sequence()?;
    let ids: Vec<u32> = (0..64).map(|i| 1000 + i).collect();
    model.prefill(&ids, &mut seq, 0)?;
    let loaded = settle()?;
    let took = before.saturating_sub(loaded);
    println!("MemAvailable after load + 1 prefill: {:.2} GB (model took {:.2} GB)", loaded as f64 / 1048576.0, took as f64 / 1048576.0);
    ensure!(took > 1024 * 1024, "the model took under 1 GB — the gate would be meaningless");

    if leak {
        println!("[control leak] mem::forget(model): the gate MUST FAIL");
        std::mem::forget(seq);
        std::mem::forget(model);
    } else {
        model.free_sequence(&mut seq)?;
        drop(model);
    }
    let after = settle()?;
    let back = after.saturating_sub(loaded);
    let frac = back as f64 / took as f64;
    println!("MemAvailable after drop: {:.2} GB (returned {:.2} GB = {:.1}% of what the model took)", after as f64 / 1048576.0, back as f64 / 1048576.0, frac * 100.0);
    if frac >= 0.95 {
        println!("DROP GATE PASS{}", if leak { " — BUT THIS WAS THE LEAK CONTROL: THE GATE CANNOT FAIL" } else { "" });
    } else {
        println!("DROP GATE FAIL{}", if leak { " (expected: leak control)" } else { "" });
    }
    drop(store);
    Ok(())
}
