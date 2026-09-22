// SPDX-License-Identifier: AGPL-3.0-only
//! DeepSeek-V4.1 forward driver: runs the first `--layers N` blocks over an oracle capture's
//! prompt and dumps per-layer taps in the oracle's file convention, for
//! `DSV41_PORT/oracle/compare.py`.
//!
//! `--feed attn,moe` TEACHER-FORCES a sub-layer: its output is read from the capture instead of
//! computed. That isolates the glue this driver's owner is responsible for (mHC, norms,
//! engram projection, shared expert, combine) from the lanes' attention and routed-MoE, so a
//! mismatch on `h` is a glue bug and nothing else. Feeding is only meaningful up to the first
//! layer where the capture and the port disagree on anything upstream of the fed tap.
//!
//! ```text
//! cargo run -p spark-model --release --example dsv41_forward --features cuda,gpu-examples -- \
//!   --run runA --layers 3 --feed attn,moe --tap-dir /tmp/x [--control own-pre]
//! ```
//! `--control own-pre` wires each sub-layer to its OWN `pre` (the V4-0731 wiring). It runs,
//! is finite, and must FAIL the `h` gate.

use anyhow::{Context, Result, bail, ensure};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use atlas_core::config::parse_config;
use spark_model::layers::deepseek_v41_engram::{EngramGather, EngramHashState, EngramLayout, engram_dead_heads};
use spark_model::weight_loader::deepseek_v41::fwd::{
    BlockControl, PassScratch, Tap, V41AttentionBlock, V41BlockWeights, V41Dims, V41RoutedMoe, block,
    final_logits_last_row,
};
use spark_model::weight_loader::deepseek_v41::attn_block::{
    self, AttnCore, AttnScratch, CoreArgs, HEAD_DIM, N_HEADS, RING, V41AttnWeights, compress_ratio,
};
use atlas_core::config::{ExpertPack, SERVED_PACKED_KEEP};
use spark_model::layers::deepseek_v41_attn::core::Dsv41SparseCore;
use spark_model::weight_loader::deepseek_v41::cb3_arena::Cb3ExpertArena;
use spark_model::weight_loader::deepseek_v41::moe_forward::{Cb3RoutedMoe, RouterF32};
use spark_model::weight_loader::deepseek_v41::forward::{PassHook, PassKind, PrefillMode, V41Forward, V41Seq};
use spark_model::weight_loader::deepseek_v41::ops::{Dsv41Kernels, Ops, RopeSpec, RopeTable, bf16_tensor, bytemuck_u32};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{SafetensorsLoader, WeightLoader};

const MODEL_DIR: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K";
const REF_ROOT: &str = "/home/flocka/atlas/DSV41_PORT/oracle/ref";
/// FIXTURE, not a production source: the engram lane's exported compressed-token map and the
/// multipliers from its oracle test. Serving needs `EngramHashState::for_checkpoint`.
const TOKEN_MAP: &str = "/home/flocka/atlas/dsv41-engram/bench/engram/token_map_i32.bin";
const ENGRAM_MULTIPLIERS: [[i64; 4]; 2] = [
    [76632096046245, 4839876093313, 35959672319349, 73987337458391],
    [67716810739261, 51510806800915, 30921347202721, 82619226485591],
];

fn engram_hash_state() -> Result<EngramHashState> {
    let layout = EngramLayout::new(&[1, 14], 4, 8, 256, 16_000_000)?;
    layout.validate_against_config(&[384_006_168, 384_016_682])?;
    let raw = std::fs::read(TOKEN_MAP).with_context(|| format!("engram fixture {TOKEN_MAP}"))?;
    let token_map: Vec<i32> = raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mult = ENGRAM_MULTIPLIERS.iter().map(|m| m.to_vec()).collect();
    EngramHashState::new(layout, token_map, mult, 2, 99_092)
}

/// Reads a captured tap for (layer, name) in forward order: the n-th call gets occurrence n.
struct Feeder {
    dir: PathBuf,
    counts: RefCell<HashMap<String, usize>>,
    gpu: Arc<AtlasCudaBackend>,
}

impl Feeder {
    fn next_bytes(&self, name: &str, layer: usize) -> Result<Vec<u8>> {
        let key = format!("L{layer:02}.{name}");
        let occ = {
            let mut c = self.counts.borrow_mut();
            let e = c.entry(key.clone()).or_insert(0);
            let o = *e;
            *e += 1;
            o
        };
        let p = self.dir.join(format!("{key}.{occ:03}.bin"));
        std::fs::read(&p).with_context(|| format!("feed: {} (not captured?)", p.display()))
    }

    fn feed_raw(&self, name: &str, layer: usize, out: DevicePtr, bytes: usize) -> Result<()> {
        let b = self.next_bytes(name, layer)?;
        ensure!(b.len() == bytes, "feed {name} L{layer}: {} bytes, expected {bytes}", b.len());
        self.gpu.copy_h2d(&b, out)
    }

    /// An f32 tap whose values are bf16-representable, fed as bf16. Refuses if any value
    /// is NOT exactly bf16 — then feeding would round, and the gate would measure that.
    fn feed_f32_as_bf16(&self, name: &str, layer: usize, out: DevicePtr, elems: usize) -> Result<()> {
        let b = self.next_bytes(name, layer)?;
        ensure!(b.len() == elems * 4, "feed {name} L{layer}: {} bytes, expected {}", b.len(), elems * 4);
        let mut h = Vec::with_capacity(elems * 2);
        for c in b.chunks_exact(4) {
            let bits = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            ensure!(bits & 0xffff == 0, "feed {name} L{layer}: f32 value not bf16-exact; refusing to round");
            h.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
        }
        self.gpu.copy_h2d(&h, out)
    }
}

struct FedAttention<'a>(&'a Feeder, usize);
impl V41AttentionBlock for FedAttention<'_> {
    fn forward(&self, _ops: &Ops, layer: usize, _x: DevicePtr, out: DevicePtr, t: usize, _start: usize) -> Result<()> {
        self.0.feed_raw("attn_out", layer, out, t * self.1 * 2)
    }
}

struct FedMoe<'a>(&'a Feeder, usize);
impl V41RoutedMoe for FedMoe<'_> {
    fn forward(&self, _ops: &Ops, layer: usize, _y: DevicePtr, out: DevicePtr, t: usize) -> Result<()> {
        self.0.feed_f32_as_bf16("moe_routed", layer, out, t * self.1)
    }
}

/// The attention lane's core fed from the capture (`attn_o_pre_inverse_rope`); everything
/// around it — projections, RoPE, ring, inverse RoPE, wo_a/wo_b — is computed.
struct FedCore<'a>(&'a Feeder);
impl AttnCore for FedCore<'_> {
    fn run(&self, _ops: &Ops, a: &CoreArgs) -> Result<()> {
        self.0.feed_raw("attn_o_pre_inverse_rope", a.layer, a.out, a.t * N_HEADS * HEAD_DIM * 2)
    }
}

/// Attention computed around a core: `attn_block::attention` with this driver's rings.
struct ProjAttention<'a> {
    ops: &'a Ops<'a>,
    weights: Vec<V41AttnWeights>,
    scratch: AttnScratch,
    wscratch: DevicePtr,
    freqs_c: RopeTable,
    freqs_w: RopeTable,
    rings: Vec<DevicePtr>,
    core: &'a dyn AttnCore,
    tap: &'a Tap,
    norm_eps: f32,
    rope_swap: bool,
}
impl V41AttentionBlock for ProjAttention<'_> {
    fn forward(&self, _ops: &Ops, layer: usize, x: DevicePtr, out: DevicePtr, t: usize, start: usize) -> Result<()> {
        // NEGATIVE CONTROL (--control rope-swap): each layer gets the OTHER table. Runs, is
        // finite, and must fail q / kv_new on every layer.
        let yarn = (compress_ratio(layer) != 0) != self.rope_swap;
        let rope = if yarn { &self.freqs_c } else { &self.freqs_w };
        attn_block::attention(
            self.ops, &self.weights[layer], &self.scratch, self.wscratch, rope, self.rings[layer],
            x, out, t, start, 0, self.norm_eps, self.core, self.tap,
        )
    }
}

struct Refuse(&'static str);
impl V41AttentionBlock for Refuse {
    fn forward(&self, _: &Ops, l: usize, _: DevicePtr, _: DevicePtr, _: usize, _: usize) -> Result<()> {
        bail!("layer {l}: {} is not wired into this driver yet — use --feed", self.0)
    }
}
impl V41RoutedMoe for Refuse {
    fn forward(&self, _: &Ops, l: usize, _: DevicePtr, _: DevicePtr, _: usize) -> Result<()> {
        bail!("layer {l}: {} is not wired into this driver yet — use --feed", self.0)
    }
}

fn manifest_ids(dir: &Path) -> Result<Vec<u32>> {
    let m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
    m["token_ids"]
        .as_array()
        .context("manifest has no token_ids")?
        .iter()
        .map(|v| v.as_u64().map(|x| x as u32).context("bad token id"))
        .collect()
}

/// Chunk starts the capture actually used (from the per-tensor `S`), so a fed tap and the
/// computed pass line up occurrence for occurrence.
fn manifest_chunks(dir: &Path, n: usize) -> Result<Vec<(usize, usize)>> {
    let m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
    let chunk = m["model_globals"]["MAX_CHUNK"].as_u64().context("no MAX_CHUNK")? as usize;
    Ok((0..n).step_by(chunk).map(|s| (s, chunk.min(n - s))).collect())
}

/// The engine lane's routed MoE over an arena holding layers `0..n_layers` (1.79 GB each at
/// keep=124). The full 40-layer load is 71.7 GB and needs lead approval.
fn real_moe<'a>(
    store: &spark_runtime::weights::WeightStore,
    gpu: &'a AtlasCudaBackend,
    kernels: &'a Dsv41Kernels,
    config: &atlas_core::config::ModelConfig,
    n_layers: usize,
    max_t: usize,
) -> Result<Cb3RoutedMoe<'a>> {
    let pack_dir = Path::new(MODEL_DIR).join("k154-cb3");
    let pack = ExpertPack::parse(&std::fs::read_to_string(pack_dir.join("manifest.json"))?, SERVED_PACKED_KEEP)?;
    let layers: Vec<usize> = (0..n_layers).collect();
    let arena = Arc::new(Cb3ExpertArena::load_layer_subset(&pack_dir, &pack, &layers, gpu)?);
    println!("routed MoE: {} layers resident, {:.2} GB", layers.len(), arena.resident_bytes() as f64 / 1e9);
    let stream = gpu.default_stream();
    let routers = layers
        .iter()
        .map(|&l| Ok((l, RouterF32::load(store, l, config.hidden_size, gpu, kernels, stream)?)))
        .collect::<Result<Vec<_>>>()?;
    Cb3RoutedMoe::new(gpu, kernels, config, arena, routers, 10.0, 1.5, max_t)
}

struct NoHook;
impl PassHook for NoHook {
    fn begin_pass(&self, _: PassKind, _: usize, _: usize) -> Result<()> {
        Ok(())
    }
}

/// The SERVING forward (`V41Forward`: CED encoder chunks + SWA replay + head), with the
/// attention core and the routed MoE teacher-forced from an all-layer capture (runH).
/// Validates the orchestration end to end before the lanes' halves exist.
#[allow(clippy::too_many_arguments)]
fn run_model_path(
    store: &spark_runtime::weights::WeightStore,
    ops: &Ops,
    config: &atlas_core::config::ModelConfig,
    dims: V41Dims,
    ref_dir: &Path,
    ids: &[u32],
    chunks: &[(usize, usize)],
    n_layers: usize,
    tap_dir: Option<PathBuf>,
    gpu: Arc<AtlasCudaBackend>,
    moe_real: bool,
    attn_real: bool,
    decode_n: usize,
    force_decode: bool,
    warm_prefill: bool,
    splits: Vec<Vec<usize>>,
) -> Result<()> {
    let gpu_ref: &AtlasCudaBackend = &Arc::clone(&gpu);
    let max_chunk = splits.iter().flatten().copied().chain(chunks.iter().map(|c| c.1)).max().unwrap_or(1);
    let fwd = V41Forward::load(store, ops, dims, config.vocab_size, n_layers, max_chunk, 8192, Path::new(MODEL_DIR), 128)?;
    let feeder = Feeder { dir: ref_dir.to_path_buf(), counts: RefCell::new(HashMap::new()), gpu };
    let fed_core = FedCore(&feeder);
    let real_core;
    let (core, hook): (&dyn AttnCore, &dyn PassHook) = if attn_real {
        real_core = Dsv41SparseCore::load(gpu_ref, store, config, 8192, max_chunk, fwd.freqs_c)?;
        (&real_core, &real_core)
    } else {
        (&fed_core, &NoHook)
    };
    let fed_moe = FedMoe(&feeder, dims.hidden);
    let real;
    let moe: &dyn V41RoutedMoe = if moe_real {
        real = real_moe(store, gpu_ref, ops.k, config, n_layers, max_chunk.max(128))?;
        &real
    } else {
        &fed_moe
    };
    let tap_base = tap_dir.clone();
    let tap = match tap_dir {
        Some(d) => Tap::to_dir(d, Vec::new())?,
        None => Tap::off(),
    };
    let mut seq = V41Seq::new(ops.gpu, &dims, EngramHashState::for_checkpoint(Path::new(MODEL_DIR))?)?;
    let logits = ops.gpu.alloc(spark_model::weight_loader::deepseek_v41::ops::MM_TILE * config.vocab_size * 2)?;
    // --split: the SAME prompt prefilled under several chunkings, each into <tap_dir>/split_<i>
    // with a fresh sequence, for the chunk-invariance check. Then return.
    if !splits.is_empty() {
        let base = tap_base.clone().context("--split needs --tap-dir")?;
        for (i, sp) in splits.iter().enumerate() {
            ensure!(sp.iter().sum::<usize>() == ids.len(), "split {sp:?} does not sum to {}", ids.len());
            let tap_i = Tap::to_dir(base.join(format!("split_{i}")), Vec::new())?;
            let mut s = V41Seq::new(ops.gpu, &dims, EngramHashState::for_checkpoint(Path::new(MODEL_DIR))?)?;
            let mut at = 0;
            for &n in sp {
                fwd.prefill_chunk(ops, &mut s, &ids[at..at + n], PrefillMode::Replay, hook, core, moe, &tap_i)?;
                at += n;
            }
            fwd.finish_prefill(ops, &mut s, PrefillMode::Replay, hook, core, moe, &tap_i, logits)?;
            tap_i.bf16(ops, "logits_last", 40, logits, &[config.vocab_size])?;
            ops.gpu.synchronize(ops.stream)?;
            println!("split {i} {sp:?} done");
            s.free(ops.gpu)?;
        }
        println!("DONE dsv41_forward path=model splits");
        return Ok(());
    }
    let t0 = std::time::Instant::now();
    fwd.prefill(ops, &mut seq, ids, PrefillMode::Replay, hook, core, moe, &tap, logits)?;
    ops.gpu.synchronize(ops.stream)?;
    println!("prefill {} tokens in {:.2}s", ids.len(), t0.elapsed().as_secs_f64());
    if spark_model::weight_loader::deepseek_v41::ops::profile::enabled() {
        println!("PROFILE (cold prefill, every scope synchronized):\n{}", spark_model::weight_loader::deepseek_v41::ops::profile::report());
    }
    tap.bf16(ops, "logits_last", 40, logits, &[config.vocab_size])?;
    ops.gpu.synchronize(ops.stream)?;
    let mut host = vec![0u8; config.vocab_size * 2];
    ops.gpu.copy_d2h(logits, &mut host)?;
    let v: Vec<f32> = host.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
    let (arg, max) = v.iter().enumerate().fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a });
    println!("model path (replay): prompt {} tokens, argmax {arg} (logit {max})", ids.len());
    if decode_n > 0 {
        let m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(ref_dir.join("manifest.json"))?)?;
        let want: Vec<u32> = m["greedy_continuation"].as_array().map(|a| a.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect()).unwrap_or_default();
        // Per step: our top-5, the oracle's token and its logit under OUR distribution, and the
        // margin top1 - logit(oracle token). With --force-decode the INPUT at each step is the
        // oracle's token (teacher forcing), so every step is conditioned on the oracle's prefix.
        let report = |step: usize, v: &[f32]| {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.select_nth_unstable_by(5, |&a, &b| v[b].total_cmp(&v[a]));
            let mut top: Vec<usize> = idx[..5].to_vec();
            top.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
            let o = want.get(step).copied();
            let (ol, margin) = match o {
                Some(t) => (v[t as usize], v[top[0]] - v[t as usize]),
                None => (f32::NAN, f32::NAN),
            };
            println!(
                "step {step:2}: ours {:6} oracle {:6?} | logit(oracle) {ol:7.3} margin {margin:6.3} | top5 {:?}",
                top[0], o, top.iter().map(|&i| (i, v[i])).collect::<Vec<_>>()
            );
            top[0] as u32
        };
        let to_f32 = |h: &[u8]| -> Vec<f32> { h.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect() };
        let mut got = vec![report(0, &v)];
        let t1 = std::time::Instant::now();
        for step in 1..decode_n {
            let input = if force_decode { want[step - 1] } else { *got.last().unwrap() };
            fwd.decode(ops, &mut seq, input, hook, core, moe, &Tap::off(), logits)?;
            ops.gpu.synchronize(ops.stream)?;
            ops.gpu.copy_d2h(logits, &mut host)?;
            got.push(report(step, &to_f32(&host)));
        }
        let dt = t1.elapsed().as_secs_f64();
        if spark_model::weight_loader::deepseek_v41::ops::profile::enabled() {
            println!("PROFILE (decode steps):\n{}", spark_model::weight_loader::deepseek_v41::ops::profile::report());
        }
        let agree = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
        let matches = got.iter().zip(&want).filter(|(a, b)| a == b).count();
        println!("decode ({}): {} steps in {dt:.2}s ({:.2} tok/s)", if force_decode { "teacher-forced" } else { "free" }, got.len() - 1, (got.len() - 1) as f64 / dt);
        println!("decode ours:   {got:?}");
        println!("decode oracle: {:?}", &want[..want.len().min(got.len())]);
        println!("decode: first {agree} identical; top-1 agreement {matches}/{}", got.len().min(want.len()));
    }
    if warm_prefill {
        // Second prefill of the same prompt in the same process: weights, arena, kernels and
        // cuBLASLt heuristics warm; engram rows now in page cache. Fresh sequence state.
        let mut seq2 = V41Seq::new(ops.gpu, &dims, EngramHashState::for_checkpoint(Path::new(MODEL_DIR))?)?;
        ops.gpu.synchronize(ops.stream)?;
        let t0 = std::time::Instant::now();
        fwd.prefill(ops, &mut seq2, ids, PrefillMode::Replay, hook, core, moe, &Tap::off(), logits)?;
        ops.gpu.synchronize(ops.stream)?;
        let dt = t0.elapsed().as_secs_f64();
        println!("WARM prefill: {} tokens in {dt:.3}s = {:.1} tok/s (chunk {max_chunk}, replay 128, engram page-cache warm)", ids.len(), ids.len() as f64 / dt);
        if spark_model::weight_loader::deepseek_v41::ops::profile::enabled() {
            println!("PROFILE (warm prefill, every scope synchronized; wall above is NOT a throughput number):\n{}", spark_model::weight_loader::deepseek_v41::ops::profile::report());
        }
    }
    println!("DONE dsv41_forward path=model");
    Ok(())
}

fn main() -> Result<()> {
    let mut run = "runA".to_string();
    let mut n_layers = 3usize;
    let mut feed: Vec<String> = Vec::new();
    let mut tap_dir: Option<PathBuf> = None;
    let mut control = BlockControl::None;
    let mut rope_swap = false;
    let mut model_path = false;
    let mut moe_real = false;
    let mut attn_real = false;
    let mut decode_n = 0usize;
    let mut force_decode = false;
    let mut warm_prefill = false;
    let mut splits: Vec<Vec<usize>> = Vec::new();
    let mut engram_live = false;
    let mut head_test = false;
    let mut core_real = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--run" => run = args.next().context("--run")?,
            "--layers" => n_layers = args.next().context("--layers")?.parse()?,
            "--feed" => feed = args.next().context("--feed")?.split(',').map(str::to_string).collect(),
            "--tap-dir" => tap_dir = Some(args.next().context("--tap-dir")?.into()),
            "--engram" => {
                engram_live = match args.next().context("--engram")?.as_str() {
                    "live" => true,
                    "feed" => false,
                    o => bail!("--engram live|feed, got {o}"),
                }
            }
            "--head-test" => head_test = true,
            // The attention lane's real core (compressor/indexer/sparse attention) in place of
            // the capture-fed one; only meaningful with `--feed attn-core,...`.
            "--core" => {
                core_real = match args.next().context("--core")?.as_str() {
                    "real" => true,
                    "fed" => false,
                    o => bail!("--core real|fed, got {o}"),
                }
            }
            "--moe-real" => moe_real = true,
            "--attn-real" => attn_real = true,
            "--decode" => decode_n = args.next().context("--decode")?.parse()?,
            "--force-decode" => force_decode = true,
            "--warm-prefill" => warm_prefill = true,
            // Per-run switches for the env-gated ops (read once, so set before any GPU work).
            // SAFETY: single-threaded at argument parsing; nothing has read the environment yet.
            "--fp8-rowtile" => unsafe { std::env::set_var("ATLAS_DSV41_FP8_ROWTILE", "1") },
            "--fp8-policy" => {
                let v = args.next().context("--fp8-policy")?;
                unsafe { std::env::set_var("ATLAS_DSV41_FP8_POLICY", v) }
            }
            "--prof" => unsafe { std::env::set_var("ATLAS_DSV41_PROF", "1") },
            "--tap-layers" => {
                let v = args.next().context("--tap-layers")?;
                unsafe { std::env::set_var("ATLAS_DSV41_TAP_LAYERS", v) }
            }
            "--tap-names" => {
                let v = args.next().context("--tap-names")?;
                unsafe { std::env::set_var("ATLAS_DSV41_TAP_NAMES", v) }
            }
            // --split "512,512,20;1024,20;500,544"
            "--split" => {
                splits = args
                    .next()
                    .context("--split")?
                    .split(';')
                    .map(|g| g.split(',').map(|n| n.parse::<usize>().map_err(anyhow::Error::from)).collect::<Result<Vec<_>>>())
                    .collect::<Result<Vec<_>>>()?
            }
            "--path" => {
                model_path = match args.next().context("--path")?.as_str() {
                    "model" => true,
                    "driver" => false,
                    o => bail!("--path model|driver, got {o}"),
                }
            }
            "--control" => {
                control = match args.next().context("--control")?.as_str() {
                    "own-pre" => BlockControl::OwnPre,
                    "rope-swap" => {
                        rope_swap = true;
                        BlockControl::None
                    }
                    o => bail!("unknown control {o}"),
                }
            }
            o => bail!("unknown argument {o}"),
        }
    }
    let ref_dir = PathBuf::from(REF_ROOT).join(&run);
    ensure!(ref_dir.join("manifest.json").is_file(), "{} has no manifest", ref_dir.display());
    let ids = manifest_ids(&ref_dir)?;
    let chunks = manifest_chunks(&ref_dir, ids.len())?;
    println!("{run}: {} tokens in chunks {chunks:?}; layers 0..{n_layers}; feed {feed:?}; control {control:?}", ids.len());

    let config = parse_config(&std::fs::read_to_string(format!("{MODEL_DIR}/config.json"))?)?;
    let dims = V41Dims::from_config(&config)?;
    let gpu = Arc::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);

    // Only what this run touches: no engram tables (~95 GB each), no MTP, no vision, and no
    // layer at or above n_layers. The routed experts are not in these shards at all.
    let keep_layers = n_layers;
    let mut loader = SafetensorsLoader::new();
    loader.extra_skip = Some(Arc::new(move |name: &str| {
        if name.contains(".engram.embed.") || name.starts_with("mtp.") || name.starts_with("vision") {
            return true;
        }
        if let Some(rest) = name.strip_prefix("layers.") {
            let l: usize = rest.split('.').next().and_then(|x| x.parse().ok()).unwrap_or(usize::MAX);
            return l >= keep_layers;
        }
        false
    }));
    let t0 = std::time::Instant::now();
    let store = loader.load(Path::new(MODEL_DIR), gpu.as_ref(), 0)?;
    println!("weights: {} tensors, {:.2} GB in {:.1}s", store.len(), store.total_bytes() as f64 / 1e9, t0.elapsed().as_secs_f64());

    let kernels = Dsv41Kernels::load(gpu.as_ref())?;
    let stream = gpu.default_stream();
    let ops = Ops { gpu: gpu.as_ref(), k: &kernels, stream };
    if model_path {
        return run_model_path(&store, &ops, &config, dims, &ref_dir, &ids, &chunks, n_layers, tap_dir, Arc::clone(&gpu), moe_real, attn_real, decode_n, force_decode, warm_prefill, splits);
    }
    let blocks: Vec<V41BlockWeights> = (0..n_layers).map(|l| V41BlockWeights::load(&store, l, &dims, &ops)).collect::<Result<_>>()?;
    let largest = blocks
        .iter()
        .flat_map(|b| [b.shared.w1, b.shared.w2, b.shared.w3].into_iter().chain(b.engram.as_ref().map(|e| e.wkv)))
        .map(|w| w.n * w.k)
        .max()
        .unwrap_or(0);
    let max_t = chunks.iter().map(|c| c.1).max().unwrap_or(1);
    let s = PassScratch::new(gpu.as_ref(), &dims, max_t, largest)?;
    let embed = bf16_tensor(&store, "embed.weight", &[config.vocab_size, dims.hidden])?;

    let tap = match &tap_dir {
        Some(d) => Tap::to_dir(d.clone(), Vec::new())?,
        None => Tap::off(),
    };
    let feeder = Feeder { dir: ref_dir.clone(), counts: RefCell::new(HashMap::new()), gpu: Arc::clone(&gpu) };
    let fed_attn = FedAttention(&feeder, dims.hidden);
    let fed_core = FedCore(&feeder);
    let fed_moe = FedMoe(&feeder, dims.hidden);
    let proj_attn;
    let real_core: Option<Dsv41SparseCore>;
    let attn: &dyn V41AttentionBlock = if feed.iter().any(|f| f == "attn") {
        real_core = None;
        &fed_attn
    } else if feed.iter().any(|f| f == "attn-core") {
        let weights: Vec<V41AttnWeights> = (0..n_layers).map(|l| V41AttnWeights::load(&store, l, dims.hidden)).collect::<Result<_>>()?;
        let big = weights.iter().map(|w| w.largest_weight()).max().unwrap_or(0).max(largest);
        let wscratch = gpu.alloc(big * 2)?;
        let positions = 8192 + 8;
        let spec_w = RopeSpec { dim: 64, original_seq_len: 0, base: 10000.0, factor: 16.0, beta_fast: 32.0, beta_slow: 1.0 };
        let spec_c = RopeSpec { original_seq_len: 65536, base: 160000.0, ..spec_w };
        let rings = (0..n_layers).map(|_| { let r = gpu.alloc(RING * HEAD_DIM * 2)?; gpu.memset(r, 0, RING * HEAD_DIM * 2)?; Ok(r) }).collect::<Result<Vec<_>>>()?;
        let freqs_c = spec_c.upload(gpu.as_ref(), positions)?;
        real_core = if core_real {
            Some(Dsv41SparseCore::load_prefix(gpu.as_ref(), &store, &config, 8192, max_t, freqs_c, n_layers)?)
        } else {
            None
        };
        proj_attn = ProjAttention {
            ops: &ops,
            weights,
            scratch: AttnScratch::new(gpu.as_ref(), max_t)?,
            wscratch,
            freqs_c,
            freqs_w: spec_w.upload(gpu.as_ref(), positions)?,
            rings,
            core: match &real_core {
                Some(c) => c,
                None => &fed_core,
            },
            tap: &tap,
            norm_eps: dims.norm_eps,
            rope_swap,
        };
        &proj_attn
    } else {
        real_core = None;
        &Refuse("attention")
    };
    let real;
    let moe: &dyn V41RoutedMoe = if feed.iter().any(|f| f == "moe") {
        &fed_moe
    } else if moe_real {
        real = real_moe(&store, gpu.as_ref(), &kernels, &config, n_layers, max_t)?;
        &real
    } else {
        &Refuse("routed MoE")
    };

    let mut hash = if engram_live { Some(engram_hash_state()?) } else { None };
    let gathers: HashMap<usize, EngramGather> = if engram_live {
        blocks
            .iter()
            .filter(|b| b.engram.is_some())
            .map(|b| Ok((b.layer, EngramGather::open(Path::new(MODEL_DIR), b.layer, 128)?)))
            .collect::<Result<_>>()?
    } else {
        HashMap::new()
    };
    let d_ids = gpu.alloc(max_t * 4)?;
    // Dead-head mask buffer, sized for the largest chunk and reused/overwritten per chunk,
    // matching `d_ids`'s pattern. `engram_dead_heads` is a pure function of that chunk's own
    // token ids (see deepseek_v41_engram::dead_heads's module doc: no cross-chunk carry) and
    // is layer-independent, so one upload per chunk covers both engram layers.
    let d_dead = gpu.alloc(max_t * 24)?;
    for &(start, t) in &chunks {
        moe.begin_pass(&ids[start..start + t])?;
        if let Some(c) = &real_core {
            // Every layer over the chunk: the debug (no-replay) pass, as runF was captured.
            c.begin_pass(PassKind::FullChunk, start, t)?;
        }
        let chunk_ids = &ids[start..start + t];
        let hashes = match hash.as_mut() {
            Some(h) => Some(h.forward(chunk_ids, start, None)?),
            None => None,
        };
        gpu.copy_h2d(bytemuck_u32(chunk_ids), d_ids)?;
        ops.embed(embed, d_ids, s.x, t, dims.hidden)?;
        ops.hc_expand(s.x, s.h, s.pre_mix, t, dims.hidden)?;
        if engram_live {
            let dead_host: Vec<u8> = engram_dead_heads(chunk_ids).iter().map(|&d| u8::from(d)).collect();
            gpu.copy_h2d(&dead_host, d_dead)?;
        }
        for w in &blocks {
            if let Some(e) = &w.engram {
                let dead_ptr = match (&hashes, gathers.get(&w.layer)) {
                    (Some(all), Some(g)) => {
                        // [T][layer][24] -> this layer's [T, 24].
                        let li = if w.layer == 1 { 0 } else { 1 };
                        let rows: Vec<i64> = (0..t).flat_map(|tok| all[(tok * 2 + li) * 24..(tok * 2 + li + 1) * 24].iter().copied()).collect();
                        let rb: Vec<u8> = rows.iter().flat_map(|r| r.to_le_bytes()).collect();
                        let tmp = gpu.alloc(rb.len())?;
                        gpu.copy_h2d(&rb, tmp)?;
                        tap.bytes(&ops, "engram_hashes", w.layer, tmp, rb.len())?;
                        gpu.free(tmp)?;
                        g.gather_rows_gpu(&rows, t, s.engram_rows, gpu.as_ref(), stream)?;
                        tap.f32(&ops, "engram_rows", w.layer, s.engram_rows, &[t, 24, 256])?;
                        // Live gather returns PRE-mask rows (EngramGather's own contract); the
                        // dead-head mask must be applied downstream, inside EngramProj::forward.
                        d_dead
                    }
                    _ => {
                        // Rows come from the capture (POST-mask already applied there), so no
                        // mask is applied again here -- NULL, not a double mask.
                        feeder.feed_raw("engram_rows", w.layer, s.engram_rows, t * 24 * 256 * 4)?;
                        DevicePtr::NULL
                    }
                };
                e.forward(&ops, s.h, s.engram_rows, dead_ptr, t, &s, &dims)?;
                tap.bf16(&ops, "engram_out", w.layer, s.h, &[t, dims.hc, dims.hidden])?;
            }
            block(&ops, w, &dims, &s, t, start, attn, moe, &tap, control)?;
        }
        gpu.synchronize(stream)?;
        println!("chunk S={start} T={t}: {n_layers} layers done");
    }
    if head_test {
        // Teacher-force the LAST layer's output of the LAST chunk and run the tail.
        let (_, t) = *chunks.last().context("no chunks")?;
        let last_layer = config.num_hidden_layers - 1;
        let occ = chunks.len() - 1;
        let hb = std::fs::read(ref_dir.join(format!("L{last_layer:02}.h.{occ:03}.bin")))?;
        let pb = std::fs::read(ref_dir.join(format!("L{last_layer:02}.pre_mix.{occ:03}.bin")))?;
        ensure!(hb.len() == t * dims.hc * dims.hidden * 2 && pb.len() == t * dims.hc * 4, "head-test: capture shape mismatch");
        gpu.copy_h2d(&hb, s.h)?;
        gpu.copy_h2d(&pb, s.pre_mix)?;
        let norm = bf16_tensor(&store, "norm.weight", &[dims.hidden])?;
        let head = bf16_tensor(&store, "head.weight", &[config.vocab_size, dims.hidden])?;
        let logits = gpu.alloc(spark_model::weight_loader::deepseek_v41::ops::MM_TILE * config.vocab_size * 2)?;
        final_logits_last_row(&ops, &dims, &s, t, norm, head, config.vocab_size, logits)?;
        tap.bf16(&ops, "logits_last", 40, logits, &[config.vocab_size])?;
        let mut host = vec![0u8; config.vocab_size * 2];
        gpu.synchronize(stream)?;
        gpu.copy_d2h(logits, &mut host)?;
        let v: Vec<f32> = host.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
        let (arg, max) = v.iter().enumerate().fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a });
        println!("head-test: argmax token {arg} (logit {max})");
    }
    println!("DONE dsv41_forward run={run} layers={n_layers} control={control:?}");
    Ok(())
}
