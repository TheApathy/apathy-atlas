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
    Cb3RoutedMoe, ExpertKernel, Fp8Act, MoeControl, ROUTER_EXPERTS, RouterF32, TOP_K,
};
use spark_model::weight_loader::deepseek_v41::moe::fused_tile_height;
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
    let mut kernel = ExpertKernel::Reconstruct;
    let mut leak_control = false;
    let mut fp8_act = false;
    let mut invariance: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--run" => run = args.next().context("--run needs a value")?,
            "--layer" => layer = args.next().context("--layer needs a value")?.parse()?,
            "--occurrence" => occurrence = args.next().context("--occurrence needs a value")?.parse()?,
            "--dump" => dump = Some(args.next().context("--dump needs a path")?.into()),
            // NEGATIVE CONTROL for the ownership check: cycles 2-6 load the arena UNOWNED (the
            // old never-free behaviour); the check must then FAIL.
            "--leak-control" => leak_control = true,
            // OPT-IN fp8 activations (NOT exact). The oracle rel_l2 is then informational; the
            // kernel gate is --dump of --kernel fused vs --kernel reconstruct (the emulation).
            "--fp8-act" => fp8_act = true,
            "--invariance" => invariance = Some(args.next().context("--invariance needs SPLIT[;SPLIT]")?),
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

    let moe_in = read_bytes(&tap("moe_in"))?;
    let want_idx = read_i64(&tap("route_idx"))?;
    let want_w = read_f32(&tap("route_w"))?;
    let want = read_f32(&tap("moe_routed"))?;
    let tokens = moe_in.len() / (hidden * 2);
    ensure!(want_idx.len() == tokens * TOP_K && want.len() == tokens * hidden, "tap extents");

    // Occurrence o of an equally-chunked capture starts at o * tokens.
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
    let all_ids: Vec<i64> = manifest["token_ids"]
        .as_array()
        .context("token_ids")?
        .iter()
        .map(|v| v.as_i64().context("token id"))
        .collect::<Result<_>>()?;
    let chunk = manifest["env"]["DSV41_PREFILL_CHUNK"].as_str().and_then(|s| s.parse::<usize>().ok());
    ensure!(occurrence == 0 || chunk == Some(tokens), "occurrence {occurrence} start is only derivable for equal chunks");
    let start = occurrence * tokens;
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
    // ONE-TIME STATE, measured by name before the MoE exists: the cuBLASLt handle + its 64 MB
    // workspace + the library's lazily loaded kernels, created by the first GEMM. The warm-up
    // uses the router's exact shape and the pinned entry point, so it only pre-pays that cost
    // (the pinned algorithm is chosen at the reference M, not at this call's M).
    let one_time_cublas = {
        let before = gpu.free_memory()? as i64;
        let a = gpu.alloc(hidden * 4)?;
        let w = gpu.alloc(ROUTER_EXPERTS * hidden * 4)?;
        let o = gpu.alloc(ROUTER_EXPERTS * 4)?;
        gpu.memset_async(a, 0, hidden * 4, gpu.default_stream())?;
        gpu.memset_async(w, 0, ROUTER_EXPERTS * hidden * 4, gpu.default_stream())?;
        spark_runtime::cublaslt::gemm_act_weight_t_typed_pinned(
            a.0, hidden as u32, w.0, o.0, ROUTER_EXPERTS as u32, 1, ROUTER_EXPERTS as u32, hidden as u32,
            spark_runtime::cublaslt::GemmDtype::F32, spark_runtime::cublaslt::GemmDtype::F32, true,
            spark_model::weight_loader::deepseek_v41::moe_forward::ROUTER_REF_M, gpu.default_stream(),
        )?;
        gpu.synchronize(gpu.default_stream())?;
        for p in [a, w, o] {
            gpu.free(p)?;
        }
        before - gpu.free_memory()? as i64
    };
    println!("  one-time state: cuBLASLt handle + workspace + library load = {:.3} GB (retained by design)", one_time_cublas as f64 / 1e9);
    // Second named one-time item: the router is read from its shard with std::fs::read, which
    // leaves the shard in page cache — and free_memory on GB10 is system-wide unified memory.
    // Pre-read it once here so the cycles below see a warm cache, and report what it cost.
    let one_time_page_cache = {
        let before = gpu.free_memory()? as i64;
        let warm = router_from_checkpoint(layer, hidden, gpu)?;
        gpu.free(warm.gate_w)?;
        before - gpu.free_memory()? as i64
    };
    println!("  one-time state: router shard read into page cache = {:.3} GB (reclaimable, not device-owned)", one_time_page_cache as f64 / 1e9);
    let free_before = gpu.free_memory()? as i64;
    let arena = Arc::new(Cb3ExpertArena::load_one_layer(&pack_dir, &pack, layer, &shared)?);
    let router = router_from_checkpoint(layer, hidden, gpu)?;
    let moe = Cb3RoutedMoe::new(shared.clone(), kernels, &config, arena, vec![(layer, router)], limit, route_scale, tokens)?;
    moe.set_pass_tokens(pass_ids);
    moe.set_control(control);
    moe.set_expert_kernel(kernel);
    moe.set_fp8_act(if fp8_act { Fp8Act::All } else { Fp8Act::Off });
    println!("  expert kernel: {kernel:?}");
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
        let want_scores = read_f32(&tap("route_scores"))?;
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (a, b) in scores.iter().zip(&want_scores) {
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
    // NECESSARY, NOT SUFFICIENT: it compares chunkings with each other, so a kernel that is
    // wrong the same way in every chunking passes (a broken 2x4 down_m32 at rel_l2 0.70 did).
    // Always pair it with the oracle rel_l2 gate above and a byte-cmp of --dump against the
    // previous kernel.
    // Each split is a list of chunk sizes summing to T, e.g. "512,512,512,512;1000,1048".
    if let Some(spec) = &invariance {
        ensure!(ours, "--invariance runs the production forward: use --routing ours");
        let d_split = gpu.alloc(tokens * hidden * 2)?;
        let ops = Ops { gpu, k: &kernels, stream };
        let mut all_identical = true;
        // Which tile shape each (row, pick) lands in: per pass, an expert's shape follows its
        // row count in THAT pass. A split only tests the shapes against each other if some
        // picks change shape between it and the single pass.
        let shapes_of = |start: usize, n: usize| -> Result<Vec<usize>> {
            let scores = moe.scores(layer, DevicePtr(d_in.0 + (start * hidden * 2) as u64), n, stream)?;
            let routing = moe.route(layer, &scores, n)?;
            let mut count = std::collections::HashMap::<i64, usize>::new();
            for &e in &routing.indices {
                *count.entry(e).or_default() += 1;
            }
            Ok(routing.indices.iter().map(|e| fused_tile_height(count[e])).collect())
        };
        moe.set_pass_tokens(pass_ids);
        let whole = shapes_of(0, tokens)?;
        let mut crossings_total = 0usize;
        for split in spec.split(';') {
            let sizes: Vec<usize> = split.split(',').map(str::parse).collect::<Result<_, _>>()?;
            ensure!(sizes.iter().sum::<usize>() == tokens, "split {split} does not sum to {tokens}");
            let mut start = 0usize;
            let mut crossings = 0usize;
            for &n in &sizes {
                moe.set_pass_tokens(&pass_ids[start..start + n]);
                let part = shapes_of(start, n)?;
                crossings += part.iter().zip(&whole[start * TOP_K..(start + n) * TOP_K]).filter(|(a, b)| a != b).count();
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
            println!(
                "  invariance [{split}] vs one pass of {tokens}: {differ} values differ on {rows} rows; \
                 {crossings} of {} (row, expert) picks change tile shape",
                tokens * TOP_K
            );
            crossings_total += crossings;
            all_identical &= differ == 0;
        }
        moe.set_pass_tokens(pass_ids);
        println!("  CHUNK INVARIANCE: {}", if all_identical { "BYTE-IDENTICAL" } else { "DIFFERS" });
        ensure!(
            crossings_total > 0,
            "no split moved any pick between tile shapes: the invariance check cannot see a shape-dependent row"
        );
        // Only the production path must be invariant; `--kernel reconstruct` (cuBLASLt per
        // expert, M = its row count) is the control that is expected to DIFFER.
        if control == MoeControl::None && kernel == ExpertKernel::Fused {
            ensure!(all_identical, "the routed MoE is not chunk-invariant");
        }
    }

    // OWNERSHIP CHECK on the real driver. Cycle 1 also pays process-global one-time costs
    // (cuBLASLt handle + 64 MB workspace + its lazily loaded kernels), which are NOT a leak, so
    // the leak test is REPEATED cycles: load 1.79 GB + scratch, run a pass, drop — cycles 2 and
    // 3 must each return >= 95% of what they took, and free memory must not drift between
    // cycles. (free_memory on GB10 is system-wide unified memory, so other processes add noise;
    // the thresholds are loose on purpose and the numbers are printed either way.)
    let free_loaded = gpu.free_memory()? as i64;
    drop(moe); // frees the scratch, tiles and router weights; the arena goes with its last Arc
    let free_after = gpu.free_memory()? as i64;
    let (taken, returned) = (free_before - free_loaded, free_after - free_loaded);
    println!(
        "  ownership cycle 1: load took {:.3} GB, drop returned {:.3} GB ({:.1}%) — includes one-time global state",
        taken as f64 / 1e9,
        returned as f64 / 1e9,
        100.0 * returned as f64 / taken.max(1) as f64
    );
    ensure!(taken >= 1_700_000_000, "loading one layer took only {taken} bytes — the check cannot see a leak");
    // Cycles 2..=6. BAND (ruled by the lead from 6 runs of noise data, before any new run):
    // over cycles 3-6, |cumulative drift| <= 0.2 GB AND MEDIAN returned >= 99%. free_memory on
    // GB10 is system-wide unified memory and moves +-0.1 GB per cycle with other processes, so
    // a per-cycle band of 0.05 GB failed non-leaking runs in BOTH directions; a real leak here
    // is -1.8 GB per cycle (-7.2 GB cumulative), > 30x outside this band.
    // FINAL REVISION (lead, 2026-09-23), from 10 runs in one window of which 4 failed the
    // mean-based band on noise, identically on two kernels: 3 had one cycle whose load
    // "took" only 1.39-1.58 GB (a concurrent release inside the cycle; the next cycle
    // returned 133-150%), 1 had mean returned 98.1% (97.6/101.5/97.0 per cycle). So: the
    // median, not the mean, and a cycle that cannot see a leak is SKIPPED and logged — once;
    // a second such cycle still aborts. The drift telescopes, so a skipped cycle's transient
    // cancels in it.
    let mut prev_after = free_after;
    let (mut cum_drift, mut returned_pct): (i64, Vec<f64>) = (0, Vec::new());
    let mut skipped: Option<usize> = None;
    for cycle in 2..=6 {
        let before = gpu.free_memory()? as i64;
        let arena = Arc::new(if leak_control {
            Cb3ExpertArena::load_one_layer_unowned(&pack_dir, &pack, layer, gpu)?
        } else {
            Cb3ExpertArena::load_one_layer(&pack_dir, &pack, layer, &shared)?
        });
        let router = router_from_checkpoint(layer, hidden, gpu)?;
        let again = Cb3RoutedMoe::new(shared.clone(), kernels, &config, arena, vec![(layer, router)], limit, route_scale, tokens)?;
        again.set_pass_tokens(pass_ids);
        again.forward(&Ops { gpu, k: &kernels, stream }, layer, d_in, d_out, tokens)?;
        gpu.synchronize(stream)?;
        let loaded = gpu.free_memory()? as i64;
        drop(again);
        let after = gpu.free_memory()? as i64;
        let (took, gave, drift) = (before - loaded, after - loaded, after - prev_after);
        println!(
            "  ownership cycle {cycle}: took {:.3} GB, returned {:.3} GB ({:.1}%), drift vs previous cycle {:+.3} GB",
            took as f64 / 1e9,
            gave as f64 / 1e9,
            100.0 * gave as f64 / took.max(1) as f64,
            drift as f64 / 1e9
        );
        let blind = took < 1_700_000_000;
        if blind {
            ensure!(
                skipped.is_none(),
                "cycle {cycle} took only {took} bytes, the second such cycle — the check cannot see a leak"
            );
            println!("  ownership cycle {cycle}: SKIPPED — took only {:.3} GB, memory moved under the cycle", took as f64 / 1e9);
            skipped = Some(cycle);
        }
        if cycle >= 3 {
            cum_drift += drift;
            if !blind {
                returned_pct.push(100.0 * gave as f64 / took.max(1) as f64);
            }
        }
        prev_after = after;
    }
    returned_pct.sort_by(|a, b| a.total_cmp(b));
    let n = returned_pct.len();
    let median_returned = if n % 2 == 1 {
        returned_pct[n / 2]
    } else {
        (returned_pct[n / 2 - 1] + returned_pct[n / 2]) / 2.0
    };
    println!(
        "  ownership band (cycles 3-6{}): cumulative drift {:+.3} GB (|.| <= 0.2), median returned {median_returned:.1}% (>= 99)",
        skipped.map(|c| format!(", cycle {c} skipped")).unwrap_or_default(),
        cum_drift as f64 / 1e9
    );
    ensure!(cum_drift.abs() <= 200_000_000, "cycles 3-6: cumulative drift {cum_drift} bytes — memory accumulates");
    ensure!(median_returned >= 99.0, "cycles 3-6: median returned {median_returned:.1}% < 99%");

    if control != MoeControl::None {
        let floor = TOL * MIN_SEPARATION;
        if rel > floor {
            println!("\nCONTROL FAILED AS REQUIRED: rel_l2 {rel:.3e} > {floor:.0e}.");
            return Ok(());
        }
        bail!("CONTROL DID NOT FIRE: rel_l2 {rel:.3e} inside {floor:.0e}. This gate proves nothing.");
    }
    if fp8_act {
        println!("\nFP8-ACT (not exact): rel_l2 {rel:.3e} vs the exact engine oracle, informational only.");
        return Ok(());
    }
    ensure!(rel < TOL, "FAIL: rel_l2 {rel:.3e} exceeds the pre-registered {TOL:.0e}");
    println!("\nPASS: routed MoE matches the engine (rel_l2 {rel:.3e}).");
    Ok(())
}
