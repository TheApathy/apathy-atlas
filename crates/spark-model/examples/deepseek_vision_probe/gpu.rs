// SPDX-License-Identifier: AGPL-3.0-only

use super::contract::*;
use anyhow::{Context, Result, ensure};
use serde_json::json;
use spark_model::layers::deepseek_vision::DeepSeekVisionEncoder;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightLoader};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

pub fn run(model: &Path, out: &Path, grid_list: &str, detail_block: Option<usize>) -> Result<()> {
    let manifest = inspect(model)?;
    let grids = parse_grids(grid_list)?;
    let config = manifest
        .config
        .deepseek_vision
        .as_ref()
        .context("vision config missing")?;
    // Reject invalid explicit selection before output, backend, or the two
    // ordinary repeat encodes. The encoder rechecks its actual loaded depth.
    if let Some(block) = detail_block {
        ensure!(
            block < config.num_hidden_layers,
            "selected detail block {block} is outside the admitted vision config"
        );
    }
    ensure!(
        !out.exists(),
        "output directory must be new: {}",
        out.display()
    );
    std::fs::create_dir(out)?;
    let executable = std::env::current_exe()?;
    let binary_sha = sha256_file(&executable)?;
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.default_stream();
    let mut loader = SafetensorsLoader::new();
    let selected = manifest.selected.clone();
    loader.extra_skip = Some(Arc::new(move |name| !selected.contains(name)));
    // The visual tower is native BF16; no transpose/dequant copies are built.
    loader.peak_memory_multiplier = Some(1.3);
    let store = loader.load(model, gpu, 2 * 1024 * 1024 * 1024)?;
    ensure!(
        store.len() == VISUAL_TENSORS && store.total_bytes() == VISUAL_BYTES,
        "selective visual loader admitted unexpected tensors/bytes"
    );
    let encoder = DeepSeekVisionEncoder::load(&store, config, manifest.config.hidden_size, gpu)?;
    let scratch_bytes = encoder.scratch_bytes()?;
    let results = (|| -> Result<Vec<serde_json::Value>> {
        let mut results = Vec::new();
        for (gh, gw) in grids {
            let name = format!("grid-{gh}x{gw}");
            let pixels = make_pixels(gh, gw);
            let input: Vec<u8> = pixels.iter().flat_map(|v| v.to_le_bytes()).collect();
            write_new(&out.join(format!("{name}.input.f32")), &input)?;
            let input_sha = sha256_bytes(&input)?;
            let rows = encoder.output_rows(gh, gw)?;
            let mut expected = None;
            let mut timings = Vec::new();
            let mut final_stats = serde_json::Value::Null;
            for repeat in 0..2 {
                gpu.synchronize(stream)?;
                let start = Instant::now();
                let output = encoder.forward(gpu, &pixels, gh, gw)?;
                gpu.synchronize(stream)?;
                let elapsed = start.elapsed().as_secs_f64();
                ensure!(
                    elapsed.is_finite() && elapsed > 0.0,
                    "invalid encoder timing"
                );
                let mut raw = vec![0u8; rows * manifest.config.hidden_size * 2];
                gpu.copy_d2h(output, &mut raw)?;
                final_stats = bf16_stats(&raw)?;
                if let Some(expected) = &expected {
                    ensure!(
                        expected == &raw,
                        "encoder output changed across identical repetitions for {name}"
                    );
                } else {
                    write_new(&out.join(format!("{name}.atlas.bf16")), &raw)?;
                    expected = Some(raw);
                }
                timings.push(json!({"repeat":repeat,"encoder_with_upload_ms":elapsed*1000.0}));
            }
            let raw = expected.context("no encoder output")?;
            if let Some(block) = detail_block {
                super::stages::capture(&encoder, gpu, &pixels, (gh, gw), out, &raw, block)?;
            }
            let row = json!({"name":name,"grid_h":gh,"grid_w":gw,"patches":gh*gw,
                "aligned_rows":rows,"hidden_size":manifest.config.hidden_size,"input_sha256":input_sha,
                "output_sha256":sha256_bytes(&raw)?,"stats":final_stats,"timings":timings,
                "repeat_byte_equal":true});
            println!("{}", serde_json::to_string(&row)?);
            results.push(row);
        }
        Ok(results)
    })();
    let release = encoder.release(gpu);
    let mut free_error = None;
    for name in store.names() {
        if let Err(err) = gpu.free(store.get(name)?.ptr) {
            free_error.get_or_insert(err);
        }
    }
    let results = results?;
    release?;
    if let Some(err) = free_error {
        return Err(err);
    }
    let report = json!({"checkpoint":manifest.report,"binary_sha256":binary_sha,
        "diagnostic_stage_capture":detail_block.is_some(),
        "selected_detail_block":detail_block,
        "binary":executable,"scratch_bytes":scratch_bytes,"cases":results,
        "timing_scope":"isolated encoder plus input upload; not LLM prompt prefill",
        "reference_parity":"pending separate pinned official torch oracle"});
    write_new(
        &out.join("atlas-manifest.json"),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}
