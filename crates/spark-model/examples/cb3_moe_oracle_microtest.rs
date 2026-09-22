// SPDX-License-Identifier: AGPL-3.0-only
//! **The routed MoE forward (`moe_forward::Cb3RoutedMoe`) against the serving engine.**
//!
//! Runs the LIBRARY code the layer forward uses — not a copy — on one resident layer and
//! compares its output with the engine's `moe_routed` tap:
//!
//! ```text
//!   moe_in [T, 5120] bf16 (captured) -> [router] -> permute -> CB3 reconstruct + cuBLASLt
//!     -> dsv41_swiglu_weighted -> cuBLASLt -> dsv41_unpermute_sum_f32  ==  moe_routed
//! ```
//!
//! Two routing sources, because they answer different questions:
//! - `--routing engine` feeds the capture's `route_idx`/`route_w`: isolates the EXPERT path.
//! - `--routing ours` runs our fp32 router GEMM, softplus, mask, per-row bias and top-k
//!   too, and additionally reports `route_idx` mismatches against the capture. This is what
//!   the layer forward actually executes.
//!
//! ## Tolerance — PRE-REGISTERED
//! Expert path (`--routing engine`): rel_l2 in [3e-5, 1e-3]. The engine's prefill MoE
//! (`tools/fp4_moe.py`) rounds to bf16 only at `h` and at the final sum, as this does; what
//! remains is fp32 accumulation order and the rounding flips it causes. Measured before the
//! move into the library: 1.1-2.8e-4, 99.1% of outputs bf16-bit-identical.
//! Full path (`--routing ours`): same band IF route_idx matches exactly; a router ulp
//! difference that flips a near-tie pick shows up as a mismatch count, reported, not hidden.
//!
//! `--control weights|expert` must FAIL at >= 10x the gate.
//!
//! `--rows S:N` keeps only rows S..S+N of the tap (the decode-size cases, N <= 8). With
//! `ATLAS_DSV41_MOE_DECODE=1` such a pass takes the decode path (GPU routing + CB3 GEMV);
//! `--routing ours` then also compares the DEVICE routing with the capture.
//!
//! ```text
//! cargo run -p spark-model --release --example cb3_moe_oracle_microtest \
//!   --features cuda,gpu-examples -- --run runA --layer 0 [--routing ours] [--dump out.bin]
//! ```

use anyhow::{Context, Result, bail, ensure};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use atlas_core::config::{ExpertPack, SERVED_PACKED_KEEP, parse_config};
use spark_model::weight_loader::deepseek_v41::cb3_arena::Cb3ExpertArena;
use spark_model::weight_loader::deepseek_v41::fwd::V41RoutedMoe;
use spark_model::weight_loader::deepseek_v41::moe_forward::{
    Cb3RoutedMoe, ExpertKernel, MoeControl, ROUTER_EXPERTS, RouterF32, TOP_K,
};
use spark_model::weight_loader::deepseek_v41::ops::{Dsv41Kernels, Ops};
use spark_model::weight_loader::deepseek_v41::routing::Routing;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const MODEL_DIR: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K";
const REF_ROOT: &str = "/home/flocka/atlas/DSV41_PORT/oracle/ref";
/// Pre-registered upper bound for the expert path.
const TOL: f64 = 1e-3;
/// A control must land at least this many times further out than the gate.
const MIN_SEPARATION: f64 = 10.0;

fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("reading {}", path.display()))
}
fn read_f32(path: &Path) -> Result<Vec<f32>> {
    Ok(read_bytes(path)?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}
fn read_i64(path: &Path) -> Result<Vec<i64>> {
    Ok(read_bytes(path)?
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

/// One tensor's raw bytes out of the main shards, by name.
fn checkpoint_tensor(name: &str) -> Result<(String, Vec<usize>, Vec<u8>)> {
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{MODEL_DIR}/model.safetensors.index.json"))?)?;
    let shard = index["weight_map"][name].as_str().with_context(|| format!("{name} not in the index"))?;
    let bytes = read_bytes(Path::new(&format!("{MODEL_DIR}/{shard}")))?;
    let header_len = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len])?;
    let entry = &header[name];
    let dtype = entry["dtype"].as_str().context("dtype")?.to_string();
    let shape = entry["shape"].as_array().context("shape")?.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect();
    let begin = 8 + header_len + entry["data_offsets"][0].as_u64().context("offset")? as usize;
    let end = 8 + header_len + entry["data_offsets"][1].as_u64().context("offset")? as usize;
    Ok((dtype, shape, bytes[begin..end].to_vec()))
}

fn router_from_checkpoint(layer: usize, hidden: usize, gpu: &dyn GpuBackend) -> Result<RouterF32> {
    let p = format!("layers.{layer}.ffn.gate");
    let (dtype, shape, w) = checkpoint_tensor(&format!("{p}.weight"))?;
    ensure!(dtype == "BF16" && shape == [ROUTER_EXPERTS, hidden], "{p}.weight is {dtype} {shape:?}");
    let bits: Vec<u16> = w.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let f32s = |name: &str| -> Result<Vec<f32>> {
        let (dtype, _, b) = checkpoint_tensor(name)?;
        ensure!(dtype == "F32", "{name} is {dtype}");
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    };
    RouterF32::from_host(&bits, f32s(&format!("{p}.bias"))?, f32s(&format!("{p}.bias_vl"))?, hidden, gpu)
}

fn main() -> Result<()> {
    let mut run = "runA".to_string();
    let mut layer = 0usize;
    let mut occurrence = 0usize;
    let mut ours = false;
    let mut control = MoeControl::None;
    let mut dump: Option<PathBuf> = None;
    let mut rows: Option<(usize, usize)> = None;
    let mut kernel = ExpertKernel::Reconstruct;
    let mut gemv_max: Option<usize> = None;
    let mut gemv_pass_t: Option<usize> = None;
    let mut invariance: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--run" => run = args.next().context("--run needs a value")?,
            "--layer" => layer = args.next().context("--layer needs a value")?.parse()?,
            "--occurrence" => occurrence = args.next().context("--occurrence needs a value")?.parse()?,
            "--rows" => {
                let v = args.next().context("--rows needs S:N")?;
                let (a, b) = v.split_once(':').context("--rows S:N")?;
                rows = Some((a.parse()?, b.parse()?));
            }
            "--dump" => dump = Some(args.next().context("--dump needs a path")?.into()),
            "--gemv-pass-t" => gemv_pass_t = Some(args.next().context("--gemv-pass-t")?.parse()?),
            "--invariance" => invariance = Some(args.next().context("--invariance needs SPLIT[;SPLIT]")?),
            "--gemv-max-rows" => gemv_max = Some(args.next().context("--gemv-max-rows")?.parse()?),
            "--kernel" => {
                kernel = match args.next().context("--kernel needs fused|reconstruct")?.as_str() {
                    "fused" => ExpertKernel::Fused,
                    "reconstruct" => ExpertKernel::Reconstruct,
                    other => bail!("unknown kernel {other}"),
                }
            }
            "--routing" => {
                ours = match args.next().context("--routing needs engine|ours")?.as_str() {
                    "engine" => false,
                    "ours" => true,
                    other => bail!("unknown routing {other}"),
                }
            }
            "--control" => {
                control = match args.next().context("--control needs weights|expert")?.as_str() {
                    "weights" => MoeControl::ReverseWeights,
                    "expert" => MoeControl::NextSlot,
                    other => bail!("unknown control {other}"),
                }
            }
            other => bail!("unknown argument {other}"),
        }
    }
    let dir = PathBuf::from(REF_ROOT).join(&run);
    ensure!(dir.join("manifest.json").is_file(), "{} has no manifest — capture incomplete", dir.display());
    let tag = format!("L{layer:02}");
    let tap = |name: &str| dir.join(format!("{tag}.{name}.{occurrence:03}.bin"));

    let raw = std::fs::read_to_string(format!("{MODEL_DIR}/config.json"))?;
    let config = parse_config(&raw)?;
    let json: serde_json::Value = serde_json::from_str(&raw)?;
    let field = |key: &str| {
        json.get(key).or_else(|| json["text_config"].get(key)).and_then(serde_json::Value::as_f64)
    };
    let limit = field("swiglu_limit").context("swiglu_limit")? as f32;
    let route_scale = field("routed_scaling_factor").context("routed_scaling_factor")? as f32;
    ensure!(limit == 10.0 && route_scale == 1.5, "config limit {limit} / scale {route_scale}");
    let hidden = config.hidden_size;

    let mut moe_in = read_bytes(&tap("moe_in"))?;
    let mut want_idx = read_i64(&tap("route_idx"))?;
    let mut want_w = read_f32(&tap("route_w"))?;
    let mut want = read_f32(&tap("moe_routed"))?;
    let mut want_scores_all = read_f32(&tap("route_scores")).unwrap_or_default();
    let full_tokens = moe_in.len() / (hidden * 2);
    ensure!(want_idx.len() == full_tokens * TOP_K && want.len() == full_tokens * hidden, "tap extents");
    let (row0, tokens) = rows.unwrap_or((0, full_tokens));
    ensure!(row0 + tokens <= full_tokens && tokens > 0, "--rows {row0}:{tokens} past {full_tokens} rows");
    if rows.is_some() {
        moe_in = moe_in[row0 * hidden * 2..(row0 + tokens) * hidden * 2].to_vec();
        want_idx = want_idx[row0 * TOP_K..(row0 + tokens) * TOP_K].to_vec();
        want_w = want_w[row0 * TOP_K..(row0 + tokens) * TOP_K].to_vec();
        want = want[row0 * hidden..(row0 + tokens) * hidden].to_vec();
        if !want_scores_all.is_empty() {
            want_scores_all = want_scores_all[row0 * ROUTER_EXPERTS..(row0 + tokens) * ROUTER_EXPERTS].to_vec();
        }
    }

    // Occurrence o of an equally-chunked capture starts at o * tokens.
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
    let all_ids: Vec<i64> = manifest["token_ids"]
        .as_array()
        .context("token_ids")?
        .iter()
        .map(|v| v.as_i64().context("token id"))
        .collect::<Result<_>>()?;
    let chunk = manifest["env"]["DSV41_PREFILL_CHUNK"].as_str().and_then(|s| s.parse::<usize>().ok());
    ensure!(occurrence == 0 || chunk == Some(full_tokens), "occurrence {occurrence} start is only derivable for equal chunks");
    let start = occurrence * full_tokens + row0;
    let pass_ids = &all_ids[start..start + tokens];
    println!(
        "{run}/{tag}.{occurrence:03}: {tokens} tokens, routing {}, {} image rows",
        if ours { "OURS (full router)" } else { "ENGINE (captured)" },
        pass_ids.iter().filter(|id| **id == 129_264 || **id == 129_265).count()
    );

    let backend = Arc::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
    let shared: spark_model::weight_loader::deepseek_v41::device_allocs::SharedGpu = backend.clone();
    let gpu: &AtlasCudaBackend = &backend;
    let kernels = Dsv41Kernels::load(gpu)?;
    let pack_dir = PathBuf::from(MODEL_DIR).join("k154-cb3");
    let pack = ExpertPack::parse(&std::fs::read_to_string(pack_dir.join("manifest.json"))?, SERVED_PACKED_KEEP)?;
    let free_before = gpu.free_memory()? as i64;
    let arena = Arc::new(Cb3ExpertArena::load_one_layer(&pack_dir, &pack, layer, &shared)?);
    let router = router_from_checkpoint(layer, hidden, gpu)?;
    let moe = Cb3RoutedMoe::new(shared.clone(), kernels, &config, arena, vec![(layer, router)], limit, route_scale, tokens)?;
    moe.set_pass_tokens(pass_ids);
    let decode_path = moe.uses_decode_path(tokens);
    println!("  expert path: {}", if decode_path { "DECODE (GPU routing + CB3 GEMV)" } else { "reconstruct + cuBLASLt" });
    moe.set_control(control);
    moe.set_expert_kernel(kernel);
    if let Some(t) = gemv_pass_t {
        moe.set_gemv_pass_t(t);
    }
    if let Some(rows) = gemv_max {
        moe.set_gemv_max_rows(rows);
    }
    println!("  expert kernel: {kernel:?}, gemv_max_rows {gemv_max:?}");
    if control != MoeControl::None {
        println!("  [control {control:?}] — MUST FAIL");
    }

    let stream = gpu.default_stream();
    let d_in = gpu.alloc(tokens * hidden * 2)?;
    let d_out = gpu.alloc(tokens * hidden * 2)?;
    gpu.copy_h2d(&moe_in, d_in)?;

    if ours {
        // The router, checked on its own before its output is used.
        let scores = moe.scores(layer, d_in, tokens, stream)?;
        let want_scores = &want_scores_all;
        ensure!(want_scores.len() == tokens * ROUTER_EXPERTS, "route_scores tap missing or wrong extent");
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (a, b) in scores.iter().zip(want_scores.iter()) {
            num += ((a - b) as f64).powi(2);
            den += (*b as f64).powi(2);
        }
        let routing = moe.route(layer, &scores, tokens)?;
        let idx_mismatch = routing.indices.iter().zip(&want_idx).filter(|(a, b)| a != b).count();
        let rows_mismatch = (0..tokens)
            .filter(|t| routing.indices[t * TOP_K..(t + 1) * TOP_K] != want_idx[t * TOP_K..(t + 1) * TOP_K])
            .count();
        let w_worst = routing.weights.iter().zip(&want_w).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!(
            "  router: scores rel_l2 {:.3e}; route_idx {idx_mismatch}/{} picks differ on {rows_mismatch}/{tokens} rows; route_w worst {w_worst:.3e}",
            (num / den).sqrt(),
            want_idx.len()
        );
        // The DEVICE router (what `forward` runs) against the engine AND the host router.
        let device = moe.route_device(layer, d_in, tokens, stream)?;
        let dev_vs_engine = device.indices.iter().zip(&want_idx).filter(|(a, b)| a != b).count();
        let dev_vs_host = device.indices.iter().zip(&routing.indices).filter(|(a, b)| a != b).count();
        let dw_engine = device.weights.iter().zip(&want_w).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let dw_host = device.weights.iter().zip(&routing.weights).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!(
            "  device router: route_idx {dev_vs_engine}/{} differ vs engine, {dev_vs_host} vs host; route_w worst {dw_engine:.3e} vs engine, {dw_host:.3e} vs host",
            want_idx.len()
        );
        moe.forward(&Ops { gpu, k: &kernels, stream }, layer, d_in, d_out, tokens)?;
        if decode_path && control == MoeControl::None {
            let dev = moe.decode_routing(tokens, stream)?;
            let idx_dev = dev.indices.iter().zip(&want_idx).filter(|(a, b)| a != b).count();
            let vs_host = dev.indices.iter().zip(&routing.indices).filter(|(a, b)| a != b).count();
            let w_dev = dev.weights.iter().zip(&want_w).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            println!(
                "  DEVICE router: route_idx {idx_dev}/{} differ from the capture, {vs_host} from the host router; route_w worst {w_dev:.3e}",
                want_idx.len()
            );
            moe.check_decode_error(stream)?;
            ensure!(idx_dev == 0, "FAIL: device routing picks differ from the capture");
        }
    } else {
        let routing = Routing { indices: want_idx.clone(), weights: want_w.clone(), k: TOP_K };
        moe.forward_routed(layer, d_in, d_out, tokens, &routing, stream)?;
    }
    gpu.synchronize(stream)?;

    let mut out_bytes = vec![0u8; tokens * hidden * 2];
    gpu.copy_d2h(d_out, &mut out_bytes)?;
    let got: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect();
    ensure!(got.iter().all(|v| v.is_finite()), "candidate has non-finite values");

    let (mut num, mut den, mut identical) = (0.0f64, 0.0f64, 0usize);
    let mut worst_token = 0.0f64;
    for (row_got, row_want) in got.chunks_exact(hidden).zip(want.chunks_exact(hidden)) {
        let (mut rn, mut rd) = (0.0f64, 0.0f64);
        for (a, b) in row_got.iter().zip(row_want) {
            let d = (*a as f64) - (*b as f64);
            rn += d * d;
            rd += (*b as f64) * (*b as f64);
            identical += (a.to_bits() == b.to_bits()) as usize;
        }
        num += rn;
        den += rd;
        worst_token = worst_token.max((rn / rd).sqrt());
    }
    let rel = (num / den).sqrt();
    println!(
        "  rel_l2 = {rel:.3e}   worst token {worst_token:.3e}   bit-identical {:.2}%   over {tokens} x {hidden}",
        100.0 * identical as f64 / got.len() as f64
    );

    if let Some(path) = &dump {
        let bytes: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))?;
        println!("  candidate dumped to {} (float32, [{tokens}, {hidden}])", path.display());
    }
    // CHUNK INVARIANCE: the same rows split into different chunks must give the SAME BYTES.
    // Each split is a list of chunk sizes summing to T, e.g. "512,512,512,512;1000,1048".
    if let Some(spec) = &invariance {
        ensure!(ours, "--invariance runs the production forward: use --routing ours");
        let d_split = gpu.alloc(tokens * hidden * 2)?;
        let ops = Ops { gpu, k: &kernels, stream };
        let mut all_identical = true;
        for split in spec.split(';') {
            let sizes: Vec<usize> = split.split(',').map(str::parse).collect::<Result<_, _>>()?;
            ensure!(sizes.iter().sum::<usize>() == tokens, "split {split} does not sum to {tokens}");
            let mut start = 0usize;
            for &n in &sizes {
                moe.set_pass_tokens(&pass_ids[start..start + n]);
                let off = (start * hidden * 2) as u64;
                moe.forward(&ops, layer, DevicePtr(d_in.0 + off), DevicePtr(d_split.0 + off), n)?;
                start += n;
            }
            gpu.synchronize(stream)?;
            let mut bytes = vec![0u8; tokens * hidden * 2];
            gpu.copy_d2h(d_split, &mut bytes)?;
            let differ = bytes.chunks_exact(2).zip(out_bytes.chunks_exact(2)).filter(|(a, b)| a != b).count();
            let rows = (0..tokens)
                .filter(|r| bytes[r * hidden * 2..(r + 1) * hidden * 2] != out_bytes[r * hidden * 2..(r + 1) * hidden * 2])
                .count();
            println!("  invariance [{split}] vs one pass of {tokens}: {differ} values differ on {rows} rows");
            all_identical &= differ == 0;
        }
        moe.set_pass_tokens(pass_ids);
        println!("  CHUNK INVARIANCE: {}", if all_identical { "BYTE-IDENTICAL" } else { "DIFFERS" });
        if control == MoeControl::None && gemv_pass_t.is_none() {
            ensure!(all_identical, "the routed MoE is not chunk-invariant");
        }
    }

    // OWNERSHIP CHECK on the real driver: the MoE and the arena free on drop. The arena is
    // 1.79 GB and the MoE scratch is sized by T, so "most of it came back" is a real check;
    // the control is the loaded state (free memory must have DROPPED by >= the arena first).
    let free_loaded = gpu.free_memory()? as i64;
    drop(moe); // frees the scratch, tiles and router weights; the arena goes with its last Arc
    let free_after = gpu.free_memory()? as i64;
    let taken = free_before - free_loaded;
    let returned = free_after - free_loaded;
    println!(
        "  ownership: load took {:.3} GB, drop returned {:.3} GB ({:.1}%)",
        taken as f64 / 1e9,
        returned as f64 / 1e9,
        100.0 * returned as f64 / taken.max(1) as f64
    );
    ensure!(taken >= 1_700_000_000, "loading one layer took only {taken} bytes — the check cannot see a leak");
    ensure!(
        returned as f64 >= 0.95 * taken as f64,
        "dropping the MoE returned only {returned} of {taken} bytes — something still owns device memory"
    );

    if control != MoeControl::None {
        let floor = TOL * MIN_SEPARATION;
        if rel > floor {
            println!("\nCONTROL FAILED AS REQUIRED: rel_l2 {rel:.3e} > {floor:.0e}.");
            return Ok(());
        }
        bail!("CONTROL DID NOT FIRE: rel_l2 {rel:.3e} inside {floor:.0e}. This gate proves nothing.");
    }
    ensure!(rel < TOL, "FAIL: rel_l2 {rel:.3e} exceeds the pre-registered {TOL:.0e}");
    println!("\nPASS: routed MoE matches the engine (rel_l2 {rel:.3e}).");
    Ok(())
}
