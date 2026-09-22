// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 vision tower on the Atlas encoder vs the production vision.py.
//!
//! Loads ONLY the visual tensors (vision.*, aligner.*, image_*; about 1 GB) from
//! the V4.1 checkpoint, runs `DeepSeekVisionEncoder::load_v41` on the oracle's
//! own input patches, and compares every tapped stage with
//! DSV41_PORT/oracle/ref/vision_tower_* (DSV41_PORT/parity/capture_vision_tower.py).
//!
//! Two checks per image:
//!   * numerics: rel_l2 / max_abs per stage vs the oracle, reported, not tuned;
//!   * layout (exact, no tolerance): the Atlas unfold output is recomputed on
//!     the host from the Atlas final-norm with torch's `F.unfold` ordering
//!     (channel-major within the 3x3 window, zero pad right/bottom), and must be
//!     identical bit for bit. Numerics cannot hide a layout bug this way.
//!
//! CONTROL (`--control`): shift the oracle's aligner rows by one; the
//! comparison must separate by orders of magnitude, or it proves nothing.
//!
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!     --example dsv41_vision_tower_microtest -- MODEL_DIR ORACLE_DIR FIXTURE_JSON

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use spark_model::layers::deepseek_vision::{DeepSeekVisionEncoder, VisionStageDtype};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightLoader};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

const STAGES: [(&str, &str); 7] = [
    ("patch", "patch_embed"),
    ("block-00-exit", "block00"),
    ("block-15-exit", "block15"),
    ("block-31-exit", "block31"),
    ("final-norm", "post_norm"),
    ("aligner-unfold", "unfold"),
    ("aligner-output", "aligner"),
];

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn read_bf16(path: &Path) -> Result<Vec<f32>> {
    let raw = std::fs::read(path).with_context(|| path.display().to_string())?;
    Ok(raw
        .chunks_exact(2)
        .map(|b| bf16(u16::from_le_bytes([b[0], b[1]])))
        .collect())
}

fn metrics(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    let (mut num, mut den, mut max_abs, mut equal) = (0f64, 0f64, 0f64, 0usize);
    for (&g, &w) in got.iter().zip(want) {
        let d = f64::from(g) - f64::from(w);
        num += d * d;
        den += f64::from(w) * f64::from(w);
        max_abs = max_abs.max(d.abs());
        equal += usize::from(g.to_bits() == w.to_bits());
    }
    (
        (num / den.max(1e-30)).sqrt(),
        max_abs,
        equal as f64 / got.len().max(1) as f64,
    )
}

/// torch `F.unfold(pad(x[C,H,W]), 3, stride=3).T`: row = (oy, ox), column = c*9 + dy*3 + dx.
fn host_unfold(norm: &[f32], gh: usize, gw: usize, hidden: usize, r: usize) -> Vec<f32> {
    let (oh, ow) = (gh.div_ceil(r), gw.div_ceil(r));
    let width = hidden * r * r;
    let mut out = vec![0f32; oh * ow * width];
    for row in 0..oh * ow {
        for c in 0..hidden {
            for dy in 0..r {
                for dx in 0..r {
                    let (y, x) = ((row / ow) * r + dy, (row % ow) * r + dx);
                    if y < gh && x < gw {
                        out[row * width + c * r * r + dy * r + dx] =
                            norm[(y * gw + x) * hidden + c];
                    }
                }
            }
        }
    }
    out
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let control = args.iter().any(|a| a == "--control");
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    ensure!(
        pos.len() == 3,
        "usage: MODEL_DIR ORACLE_DIR FIXTURE_JSON [--control]"
    );
    let (model, oracle, fixture) = (Path::new(pos[0]), Path::new(pos[1]), Path::new(pos[2]));
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(model.join("config.json"))?)?;
    let v = &cfg["vision_config"];
    let text_hidden = cfg["text_config"]["hidden_size"]
        .as_u64()
        .context("text hidden")? as usize;
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(fixture)?)?;
    let grids: HashMap<String, (usize, usize)> = fx["images"]
        .as_array()
        .context("fixture images")?
        .iter()
        .filter_map(|c| {
            Some((
                c["name"].as_str()?.to_string(),
                (c["vit_h"].as_u64()? as usize, c["vit_w"].as_u64()? as usize),
            ))
        })
        .collect();
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(oracle.join("manifest.json"))?)?;
    let images: Vec<String> = serde_json::from_value(manifest["images"].clone())?;

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let mut loader = SafetensorsLoader::new();
    loader.extra_skip = Some(Arc::new(|name: &str| {
        !(name.starts_with("vision.") || name.starts_with("aligner.") || name.starts_with("image_"))
    }));
    loader.peak_memory_multiplier = Some(1.3);
    let store = loader.load(model, gpu, 2 * 1024 * 1024 * 1024)?;
    eprintln!(
        "loaded {} visual tensors, {:.2} GB",
        store.len(),
        store.total_bytes() as f64 / 1e9
    );
    let u = |k: &str| {
        v[k].as_u64()
            .with_context(|| format!("vision_config.{k}"))
            .map(|x| x as usize)
    };
    let encoder = DeepSeekVisionEncoder::load_v41(
        &store,
        u("hidden_size")?,
        u("intermediate_size")?,
        u("num_attention_heads")?,
        u("num_hidden_layers")?,
        u("patch_size")?,
        u("downsample_ratio")?,
        v["rope_theta"].as_f64().context("rope_theta")?,
        u("max_image_tokens")?,
        text_hidden,
        gpu,
        None,
    )?;
    let r = u("downsample_ratio")?;
    // START / NEWLINE / END rows (V4.1 has no learned pad row).
    let specials: Vec<Vec<f32>> = {
        let sp = encoder.image_special_embeddings();
        [sp[0], sp[3], sp[4]]
            .into_iter()
            .map(|p| -> Result<Vec<f32>> {
                let mut raw = vec![0u8; text_hidden * 2];
                gpu.copy_d2h(p, &mut raw)?;
                Ok(raw
                    .chunks_exact(2)
                    .map(|b| bf16(u16::from_le_bytes([b[0], b[1]])))
                    .collect())
            })
            .collect::<Result<_>>()?
    };
    let hidden = u("hidden_size")?;

    let mut report = Vec::new();
    let mut layout_ok = true;
    for name in &images {
        let (gh, gw) = *grids.get(name).context("grid for image")?;
        let patches = read_bf16(&oracle.join(format!("{name}.patches.bin")))?;
        let mut taps: HashMap<String, Vec<f32>> = HashMap::new();
        let mut observer = |stage: &str,
                            ptr: spark_runtime::gpu::DevicePtr,
                            shape: [usize; 2],
                            dt: VisionStageDtype|
         -> Result<()> {
            if dt == VisionStageDtype::Bf16 && STAGES.iter().any(|(s, _)| *s == stage) {
                let mut raw = vec![0u8; shape[0] * shape[1] * 2];
                gpu.copy_d2h(ptr, &mut raw)?;
                taps.insert(
                    stage.to_string(),
                    raw.chunks_exact(2)
                        .map(|b| bf16(u16::from_le_bytes([b[0], b[1]])))
                        .collect(),
                );
            }
            Ok(())
        };
        encoder.forward_observed(gpu, &patches, gh, gw, &mut observer)?;
        let mut stages = Vec::new();
        for (stage, tap) in STAGES {
            let got = taps
                .get(stage)
                .with_context(|| format!("stage {stage} not observed"))?;
            let mut want = read_bf16(&oracle.join(format!("{name}.{tap}.bin")))?;
            ensure!(
                got.len() == want.len(),
                "{name}/{stage}: {} vs {} values",
                got.len(),
                want.len()
            );
            if control && tap == "aligner" {
                want.rotate_left(text_hidden); // shift every row by one
            }
            let (rel, max_abs, exact) = metrics(got, &want);
            stages.push(json!({"stage": stage, "oracle_tap": tap, "rel_l2": rel, "max_abs": max_abs, "bit_exact_frac": exact}));
            eprintln!(
                "{name:>24} {stage:>16}: rel_l2 {rel:.3e} max_abs {max_abs:.3e} bit-exact {:.4}",
                exact
            );
        }
        let unfold_host = host_unfold(&taps["final-norm"], gh, gw, hidden, r);
        let same = unfold_host.len() == taps["aligner-unfold"].len()
            && unfold_host
                .iter()
                .zip(&taps["aligner-unfold"])
                .all(|(a, b)| a.to_bits() == b.to_bits());
        layout_ok &= same;
        eprintln!(
            "{name:>24} unfold layout (Atlas vs host F.unfold of Atlas final-norm): {}",
            if same { "IDENTICAL" } else { "DIFFERS" }
        );
        // Span layout: V4.1 is START, then per aligner row (w IMAGE slots, NEWLINE),
        // then END. V4-Flash-Vision's N-layout interleaves row pairs. Assemble the
        // span from the Atlas aligner output both ways; V4.1 must track the
        // oracle's `span` tap and the V4 N-layout must NOT.
        let (lh, lw) = (gh.div_ceil(r), gw.div_ceil(r));
        let aligner = &taps["aligner-output"];
        let row = |i: usize| &aligner[i * text_hidden..(i + 1) * text_hidden];
        let assemble = |order: &[usize]| -> Vec<f32> {
            let mut span = Vec::with_capacity((lh * (lw + 1) + 2) * text_hidden);
            span.extend_from_slice(&specials[0]);
            let mut it = order.iter();
            for _ in 0..lh {
                for _ in 0..lw {
                    span.extend_from_slice(row(*it.next().expect("order covers the grid")));
                }
                span.extend_from_slice(&specials[1]);
            }
            span.extend_from_slice(&specials[2]);
            span
        };
        let want_span = read_bf16(&oracle.join(format!("{name}.span.bin")))?;
        let v41: Vec<usize> = (0..lh * lw).collect();
        let v4 =
            atlas_core::config::build_deepseek_image_block(lh, lw, 0, 129_280)?.aligner_permutation;
        let (rel_v41, _, _) = metrics(&assemble(&v41), &want_span);
        let (rel_v4, _, _) = metrics(&assemble(&v4), &want_span);
        eprintln!(
            "{name:>24} span vs oracle: V4.1 row-major {rel_v41:.3e} | V4 N-layout {rel_v4:.3e}"
        );
        // The tower's own bf16 noise is ~2-3.5% (production vision.py on GPU vs CPU,
        // measured); a wrong row order is uncorrelated (~0.9). 10x separates them.
        layout_ok &= rel_v4 > 10.0 * rel_v41;
        report.push(json!({"image": name, "grid": [gh, gw], "stages": stages, "unfold_layout_identical": same,
                           "span_rel_l2_v41_layout": rel_v41, "span_rel_l2_v4_n_layout": rel_v4}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"control": control, "text_hidden": text_hidden, "results": report})
        )?
    );
    ensure!(
        layout_ok,
        "unfold layout differs from torch F.unfold, or the span layout test did not separate"
    );
    println!("DONE control={control}");
    Ok(())
}
