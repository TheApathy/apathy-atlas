// SPDX-License-Identifier: AGPL-3.0-only
//! **Where the routed MoE's time goes**, one resident layer at a time, before any optimisation.
//!
//! Times `Cb3RoutedMoe` on real activations (a capture's `moe_in`, prefixes of it for smaller T)
//! and splits the per-layer cost by subtraction using the TIMING-ONLY [`ExpertWork`] modes:
//!
//! ```text
//!   scores   router GEMM + [T,384] D2H + sync      (host sees the scores)
//!   route    host top-6 over the scores
//!   skip     forward_routed with no expert work: host plan, uploads, permute, final sum
//!   recon    + CB3 reconstruct only
//!   gemm     + GEMMs + SwiGLU only (stale weights)
//!   all      the real forward
//! ```
//!
//! Wall clock around `synchronize`, warm (the arena is resident; nothing touches the disk
//! inside the timed region), median of `--iters` after `--warmup`. Also prints the roofline
//! each phase is bounded by, so a number can be read against what the hardware allows.
//!
//! ```text
//! cargo run -p spark-model --release --example dsv41_moe_bench --features cuda,gpu-examples \
//!   -- --run runC_2048 --layer 2 --tokens 2048,512,64,1
//! ```

use anyhow::{Context, Result, bail, ensure};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use atlas_core::config::{ExpertPack, SERVED_PACKED_KEEP, parse_config};
use spark_model::weight_loader::deepseek_v41::cb3_arena::Cb3ExpertArena;
use spark_model::weight_loader::deepseek_v41::moe_forward::{
    Cb3RoutedMoe, ExpertWork, ROUTER_EXPERTS, RouterF32, TOP_K,
};
use spark_model::weight_loader::deepseek_v41::ops::Dsv41Kernels;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

const MODEL_DIR: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K";
const REF_ROOT: &str = "/home/flocka/atlas/DSV41_PORT/oracle/ref";
const LAYERS: usize = 40;

fn checkpoint_tensor(name: &str) -> Result<(String, Vec<u8>)> {
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{MODEL_DIR}/model.safetensors.index.json"))?)?;
    let shard = index["weight_map"][name].as_str().with_context(|| format!("{name} not in the index"))?;
    let bytes = std::fs::read(format!("{MODEL_DIR}/{shard}"))?;
    let header_len = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len])?;
    let entry = &header[name];
    let begin = 8 + header_len + entry["data_offsets"][0].as_u64().context("offset")? as usize;
    let end = 8 + header_len + entry["data_offsets"][1].as_u64().context("offset")? as usize;
    Ok((entry["dtype"].as_str().context("dtype")?.to_string(), bytes[begin..end].to_vec()))
}

fn router(layer: usize, hidden: usize, gpu: &dyn GpuBackend) -> Result<RouterF32> {
    let p = format!("layers.{layer}.ffn.gate");
    let (dtype, w) = checkpoint_tensor(&format!("{p}.weight"))?;
    ensure!(dtype == "BF16", "{p}.weight is {dtype}");
    let bits: Vec<u16> = w.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let f32s = |n: &str| -> Result<Vec<f32>> {
        let (dtype, b) = checkpoint_tensor(n)?;
        ensure!(dtype == "F32", "{n} is {dtype}");
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    };
    RouterF32::from_host(&bits, f32s(&format!("{p}.bias"))?, f32s(&format!("{p}.bias_vl"))?, hidden, gpu)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let mut run = "runC_2048".to_string();
    let mut layer = 2usize;
    let mut token_counts = vec![2048usize, 512, 64, 1];
    let (mut warmup, mut iters) = (2usize, 7usize);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--run" => run = args.next().context("--run")?,
            "--layer" => layer = args.next().context("--layer")?.parse()?,
            "--tokens" => {
                token_counts = args.next().context("--tokens")?.split(',').map(str::parse).collect::<Result<_, _>>()?
            }
            "--warmup" => warmup = args.next().context("--warmup")?.parse()?,
            "--iters" => iters = args.next().context("--iters")?.parse()?,
            other => bail!("unknown argument {other}"),
        }
    }
    let dir = PathBuf::from(REF_ROOT).join(&run);
    ensure!(dir.join("manifest.json").is_file(), "{} incomplete", dir.display());
    let raw = std::fs::read_to_string(format!("{MODEL_DIR}/config.json"))?;
    let config = parse_config(&raw)?;
    let (hidden, inter) = (config.hidden_size, config.moe_intermediate_size);
    let moe_in = std::fs::read(dir.join(format!("L{layer:02}.moe_in.000.bin")))?;
    let captured = moe_in.len() / (hidden * 2);
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
    let ids: Vec<i64> = manifest["token_ids"].as_array().context("ids")?.iter().filter_map(|v| v.as_i64()).collect();
    let max_t = *token_counts.iter().max().context("no token counts")?;
    ensure!(max_t <= captured, "{run} L{layer:02} has {captured} captured rows, asked for {max_t}");

    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let kernels = Dsv41Kernels::load(&gpu)?;
    let pack_dir = PathBuf::from(MODEL_DIR).join("k154-cb3");
    let pack = ExpertPack::parse(&std::fs::read_to_string(pack_dir.join("manifest.json"))?, SERVED_PACKED_KEEP)?;
    let arena = Arc::new(Cb3ExpertArena::load_one_layer(Path::new(&pack_dir), &pack, layer, &gpu)?);
    let moe = Cb3RoutedMoe::new(&gpu, &kernels, &config, arena, vec![(layer, router(layer, hidden, &gpu)?)], 10.0, 1.5, max_t)?;
    let stream = gpu.default_stream();
    let d_in = gpu.alloc(max_t * hidden * 2)?;
    let d_out = gpu.alloc(max_t * hidden * 2)?;
    gpu.copy_h2d(&moe_in[..max_t * hidden * 2], d_in)?;

    // Per-expert roofline inputs.
    let packed_bytes = pack.bytes_per_expert() as f64; // 14.45 MB read by reconstruct
    let bf16_bytes = (3 * inter * hidden * 2) as f64; // 70.8 MB written by reconstruct, read by GEMM
    println!("{run} L{layer:02}, warm, median of {iters} after {warmup} warmup; per-layer ms");
    println!(
        "{:>5} {:>4} | {:>7} {:>6} {:>7} {:>7} {:>7} {:>7} | {:>7} {:>7} | {:>9} {:>9}",
        "T", "exp", "scores", "route", "skip", "recon", "gemm", "all", "recon-", "gemm-", "GEMM TF/s", "MoE tok/s"
    );
    for &t in &token_counts {
        moe.set_pass_tokens(&ids[..t]);
        let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
            let mut samples = Vec::with_capacity(iters);
            for i in 0..warmup + iters {
                gpu.synchronize(stream)?;
                let start = Instant::now();
                f()?;
                gpu.synchronize(stream)?;
                if i >= warmup {
                    samples.push(start.elapsed().as_secs_f64() * 1e3);
                }
            }
            Ok(median(samples))
        };
        let scores_ms = time(&mut || moe.scores(layer, d_in, t, stream).map(|_| ()))?;
        let scores = moe.scores(layer, d_in, t, stream)?;
        let route_ms = time(&mut || moe.route(layer, &scores, t).map(|_| ()))?;
        let routing = moe.route(layer, &scores, t)?;
        let experts = {
            let mut e: Vec<i64> = routing.indices.clone();
            e.sort_unstable();
            e.dedup();
            e.len()
        };
        let phase = |work: ExpertWork| -> Result<f64> {
            moe.set_expert_work(work);
            time(&mut || moe.forward_routed(layer, d_in, d_out, t, &routing, stream))
        };
        let skip = phase(ExpertWork::Skip)?;
        let recon = phase(ExpertWork::ReconstructOnly)?;
        let gemm = phase(ExpertWork::GemmOnly)?;
        let all = phase(ExpertWork::All)?;
        moe.set_expert_work(ExpertWork::All);

        let flops = 2.0 * (t * TOP_K) as f64 * (3 * inter * hidden) as f64;
        let gemm_tfs = flops / ((gemm - skip).max(1e-6) * 1e-3) / 1e12;
        let layer_ms = scores_ms + route_ms + all;
        let tok_s = t as f64 / (LAYERS as f64 * layer_ms * 1e-3);
        println!(
            "{t:>5} {experts:>4} | {scores_ms:>7.2} {route_ms:>6.2} {skip:>7.2} {recon:>7.2} {gemm:>7.2} {all:>7.2} | {:>7.2} {:>7.2} | {gemm_tfs:>9.1} {tok_s:>9.1}",
            recon - skip,
            gemm - skip,
        );
        // Reconstruct traffic: read the packed expert, write the bf16 copy.
        let recon_gb = experts as f64 * (packed_bytes + bf16_bytes) / 1e9;
        println!(
            "      recon moves {recon_gb:.2} GB -> {:.0} GB/s effective; GEMMs re-read {:.2} GB of bf16 weights",
            recon_gb / ((recon - skip).max(1e-6) * 1e-3),
            experts as f64 * bf16_bytes / 1e9,
        );
    }
    let _ = ROUTER_EXPERTS;
    println!("DONE");
    moe.free()
}
