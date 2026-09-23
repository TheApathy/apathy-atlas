// SPDX-License-Identifier: AGPL-3.0-only
//! Shared expert (`layers.L.ffn.shared_experts`, FP8 block-32) on real activations: the per-pass
//! dequant path vs the RESIDENT path (`SharedExpert::make_resident`: bf16 copies, w1|w3 as one
//! N = 4608 GEMM on the policy path).
//!
//! Gate: the resident path's output is byte-identical to the dequant path at every T; the
//! resident copy with w1|w3 halves swapped must differ. Then median ms of both,
//! alternated in-process, prefill-kind passes (not decode, not replay).
//!
//!   cargo run --release -p spark-model --features cuda,gpu-examples --example dsv41_shared_bench -- \
//!       [--layer 2] [--tokens 2048,512,128] [--iters 20]

use anyhow::{Context, Result, bail, ensure};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use atlas_core::config::parse_config;
use spark_model::weight_loader::deepseek_v41::fwd::{PassScratch, SharedExpert, V41Dims};
use spark_model::weight_loader::deepseek_v41::ops::{Dsv41Kernels, Ops, set_fp8_fixed_m, tiled_rows};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightLoader};

const MODEL_DIR: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K";
const REF_ROOT: &str = "/home/flocka/atlas/DSV41_PORT/oracle/ref";

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let (mut layer, mut iters) = (2usize, 20usize);
    let mut token_counts = vec![2048usize, 2047, 1000, 512, 128, 17, 16, 5];
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--layer" => layer = args.next().context("--layer")?.parse()?,
            "--iters" => iters = args.next().context("--iters")?.parse()?,
            "--tokens" => {
                token_counts = args.next().context("--tokens")?.split(',').map(str::parse).collect::<Result<_, _>>()?
            }
            other => bail!("unknown argument {other}"),
        }
    }
    // Production pins FP8-dense GEMMs at the served chunk's row count (V41Forward::load).
    set_fp8_fixed_m(tiled_rows(2048));
    let gpu = Arc::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
    let config = parse_config(&std::fs::read_to_string(format!("{MODEL_DIR}/config.json"))?)?;
    let dims = V41Dims::from_config(&config)?;
    let prefix = format!("layers.{layer}.ffn.shared_experts.");
    let mut loader = SafetensorsLoader::new();
    let keep = prefix.clone();
    loader.extra_skip = Some(Arc::new(move |name: &str| !name.starts_with(&keep)));
    let store = loader.load(Path::new(MODEL_DIR), gpu.as_ref(), 0)?;
    let kernels = Dsv41Kernels::load(gpu.as_ref())?;
    let stream = gpu.default_stream();
    let ops = Ops { gpu: gpu.as_ref(), k: &kernels, stream };
    let base = SharedExpert::load(&store, layer, &dims)?;

    let max_t = *token_counts.iter().max().context("no token counts")?;
    let largest = [base.w1, base.w2, base.w3].iter().map(|w| w.n * w.k).max().unwrap_or(0);
    let s = PassScratch::new(gpu.as_ref(), &dims, max_t, largest)?;

    // Resident copies (production's SharedExpert::make_resident): w1|w3 concatenated + w2.
    let mut resident = SharedExpert::load(&store, layer, &dims)?;
    resident.make_resident(&ops)?;
    // NEGATIVE CONTROL: the concatenated copy with its halves SWAPPED (w3|w1): real matrices,
    // the wrong ones.
    let mut control = SharedExpert::load(&store, layer, &dims)?;
    control.make_resident(&ops)?;
    let swapped = control.w13.context("resident")?;
    ops.dequant(&base.w3, swapped)?;
    ops.dequant(&base.w1, spark_runtime::gpu::DevicePtr(swapped.0 + base.w1.bf16_bytes() as u64))?;
    gpu.synchronize(stream)?;

    let run = format!("{REF_ROOT}/runC_2048/L{layer:02}.moe_in.000.bin");
    let x_all = std::fs::read(&run).with_context(|| run.clone())?;
    let captured = x_all.len() / (dims.hidden * 2);
    ensure!(max_t <= captured, "{run}: {captured} rows, asked for {max_t}");
    let d_x = gpu.alloc(max_t * dims.hidden * 2)?;
    gpu.copy_h2d(&x_all[..max_t * dims.hidden * 2], d_x)?;
    let outs = [gpu.alloc(max_t * dims.hidden * 2)?, gpu.alloc(max_t * dims.hidden * 2)?, gpu.alloc(max_t * dims.hidden * 2)?];

    println!("shared expert L{layer:02}: dequant path vs resident bf16 ({:.1} MB resident)", 3.0 * base.w1.bf16_bytes() as f64 / 1e6);
    let mut all_identical = true;
    for &t in &token_counts {
        let arms: [(&str, &SharedExpert); 3] = [("dequant", &base), ("resident", &resident), ("control", &control)];
        let mut bytes: Vec<Vec<u8>> = Vec::new();
        for ((_, e), out) in arms.iter().zip(outs) {
            e.forward(&ops, d_x, out, t, &s, &dims)?;
            gpu.synchronize(stream)?;
            let mut b = vec![0u8; t * dims.hidden * 2];
            gpu.copy_d2h(out, &mut b)?;
            bytes.push(b);
        }
        let differ = |a: &[u8], b: &[u8]| a.chunks_exact(2).zip(b.chunks_exact(2)).filter(|(x, y)| x != y).count();
        let (res, ctl) = (differ(&bytes[0], &bytes[1]), differ(&bytes[0], &bytes[2]));
        all_identical &= res == 0;
        ensure!(ctl > 0, "T={t}: the wrong-matrix control matched the dequant path: this gate proves nothing");

        let mut times = [Vec::new(), Vec::new()];
        for i in 0..iters + 3 {
            for (a, e) in [&base, &resident].into_iter().enumerate() {
                gpu.synchronize(stream)?;
                let start = Instant::now();
                e.forward(&ops, d_x, outs[a], t, &s, &dims)?;
                gpu.synchronize(stream)?;
                if i >= 3 {
                    times[a].push(start.elapsed().as_secs_f64() * 1e3);
                }
            }
        }
        let (dq, rs) = (median(times[0].clone()), median(times[1].clone()));
        println!(
            "  T={t:>5}: dequant {dq:.3} ms, resident {rs:.3} ms ({:+.1}%); resident differs in {res} values, control in {ctl}",
            100.0 * (rs / dq - 1.0)
        );
    }
    println!("{}", if all_identical { "RESIDENT: BYTE-IDENTICAL at every T" } else { "RESIDENT: DIFFERS" });

    Ok(())
}
