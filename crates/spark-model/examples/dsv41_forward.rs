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
use spark_model::weight_loader::deepseek_v41::ops::{Dsv41Kernels, Ops, bf16_tensor, bytemuck_u32};
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
    fn forward(&self, layer: usize, _x: DevicePtr, out: DevicePtr, t: usize, _start: usize, _stream: u64) -> Result<()> {
        self.0.feed_raw("attn_out", layer, out, t * self.1 * 2)
    }
}

struct FedMoe<'a>(&'a Feeder, usize);
impl V41RoutedMoe for FedMoe<'_> {
    fn forward(&self, layer: usize, _y: DevicePtr, out: DevicePtr, t: usize, _stream: u64) -> Result<()> {
        self.0.feed_f32_as_bf16("moe_routed", layer, out, t * self.1)
    }
}

struct Refuse(&'static str);
impl V41AttentionBlock for Refuse {
    fn forward(&self, l: usize, _: DevicePtr, _: DevicePtr, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("layer {l}: {} is not wired into this driver yet — use --feed", self.0)
    }
}
impl V41RoutedMoe for Refuse {
    fn forward(&self, l: usize, _: DevicePtr, _: DevicePtr, _: usize, _: u64) -> Result<()> {
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

fn main() -> Result<()> {
    let mut run = "runA".to_string();
    let mut n_layers = 3usize;
    let mut feed: Vec<String> = Vec::new();
    let mut tap_dir: Option<PathBuf> = None;
    let mut control = BlockControl::None;
    let mut engram_live = false;
    let mut head_test = false;
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
            "--control" => {
                control = match args.next().context("--control")?.as_str() {
                    "own-pre" => BlockControl::OwnPre,
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

    let feeder = Feeder { dir: ref_dir.clone(), counts: RefCell::new(HashMap::new()), gpu: Arc::clone(&gpu) };
    let fed_attn = FedAttention(&feeder, dims.hidden);
    let fed_moe = FedMoe(&feeder, dims.hidden);
    let attn: &dyn V41AttentionBlock = if feed.iter().any(|f| f == "attn") { &fed_attn } else { &Refuse("attention") };
    let moe: &dyn V41RoutedMoe = if feed.iter().any(|f| f == "moe") { &fed_moe } else { &Refuse("routed MoE") };
    let tap = match &tap_dir {
        Some(d) => Tap::to_dir(d.clone(), Vec::new())?,
        None => Tap::off(),
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
    for &(start, t) in &chunks {
        let chunk_ids = &ids[start..start + t];
        let hashes = match hash.as_mut() {
            Some(h) => Some(h.forward(chunk_ids, start, None)?),
            None => None,
        };
        gpu.copy_h2d(bytemuck_u32(chunk_ids), d_ids)?;
        ops.embed(embed, d_ids, s.x, t, dims.hidden)?;
        ops.hc_expand(s.x, s.h, s.pre_mix, t, dims.hidden)?;
        // `engram_dead_heads` is a pure function of this chunk's own token ids (no cross-chunk
        // carry -- see deepseek_v41_engram::dead_heads's module doc) and is layer-independent,
        // so one upload into the pre-allocated `s.engram_dead` scratch covers both engram
        // layers for this chunk.
        if engram_live {
            let dead_host: Vec<u8> = engram_dead_heads(chunk_ids).iter().map(|&d| u8::from(d)).collect();
            eprintln!("DEBUG dead_host True count = {} / {}", dead_host.iter().filter(|&&d| d != 0).count(), dead_host.len());
            gpu.copy_h2d(&dead_host, s.engram_dead)?;
            let mut readback = vec![0u8; dead_host.len()];
            gpu.copy_d2h(s.engram_dead, &mut readback)?;
            eprintln!("DEBUG readback True count = {} / {}", readback.iter().filter(|&&d| d != 0).count(), readback.len());
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
                        // dead-head mask is applied downstream, inside EngramProj::forward.
                        s.engram_dead
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
        let logits = gpu.alloc(config.vocab_size * 2)?;
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
