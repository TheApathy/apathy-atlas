// SPDX-License-Identifier: AGPL-3.0-only

//! w3-requant — offline NVFP4 → W3 Lloyd-Max (3-bit) expert requant tool.
//!
//! Reads the shipped NVFP4 routed-expert tensors straight from the HF
//! snapshot safetensors (READ ONLY), fits a symmetric 8-point Lloyd-Max
//! codebook over the scale²-weighted empirical E2M1 magnitude distribution,
//! then repacks every expert's nibbles into Turbo3-packed 3-bit codebook
//! indices. FP8 per-16 group scales and the per-tensor scale2 are carried
//! over UNCHANGED.
//!
//! Codebook scope (`--codebook`):
//!   * `per-layer` (default): one codebook fitted per MoE layer over that
//!     layer's own (sampled) histogram. The W3 v1 format already stores the
//!     LUT in each layer file's header and the runtime uploads a per-layer
//!     LUT to every `_w3` kernel, so this needs NO format or runtime change.
//!   * `global`: one codebook over all layers (the original behavior).
//!
//! Output: one `layer_{L:03}.w3x` per MoE layer (format: `spark_storage::w3`)
//! plus `summary.json` with the codebook(s) and per-layer RMSE/cosine stats.
//!
//! Usage:
//!   w3-requant --snapshot <hf-snapshot-dir> --out <w3cache-dir> \
//!       [--codebook per-layer|global] [--layers 1,2,7] \
//!       [--sample-stride 16] [--threads N] [--cosine]
//!
//! `--sample-stride N` fits on every Nth expert (per layer). Stride 1 fits
//! on the exact full histogram — recommended with `per-layer`, where the
//! extra histogram pass costs one more read of the layer (~1.3 s/layer).
//!
//! `--cosine` additionally emits a per-layer cosine-similarity summary table
//! (stats are exact, derived from the weighted code histogram — the W3
//! reconstruction differs from the NVFP4-dequant reference only in the
//! code→value map, so no float dequant of the full tensors is needed).

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;

use spark_storage::w3::{
    CodeHist, Codebook, GROUP_SIZE, W3LayerGeom, W3LayerHeader, e4m3_to_f32, repack_row,
};

// ── Minimal safetensors shard access (mmap + parsed metadata) ──────────────

struct Shard {
    mmap: Arc<memmap2::Mmap>,
    data_start: usize,
}

struct Snapshot {
    shards: Vec<Shard>,
    /// tensor name → (shard idx, byte range within data section, shape, dtype)
    tensors: HashMap<String, TensorRef>,
}

#[derive(Clone, Debug)]
struct TensorRef {
    shard: usize,
    start: usize,
    end: usize,
    shape: Vec<usize>,
}

impl Snapshot {
    fn open(dir: &Path) -> Result<Self> {
        let idx_path = dir.join("model.safetensors.index.json");
        let idx: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&idx_path).with_context(|| format!("read {idx_path:?}"))?,
        )?;
        let weight_map = idx["weight_map"]
            .as_object()
            .context("index.json missing weight_map")?;
        let mut shard_names: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        shard_names.sort();
        shard_names.dedup();

        let mut shards = Vec::new();
        let mut shard_idx = HashMap::new();
        let mut tensors = HashMap::new();
        for name in &shard_names {
            let path = dir.join(name);
            let file = std::fs::File::open(&path).with_context(|| format!("open {path:?}"))?;
            // SAFETY: file is opened read-only from an immutable snapshot.
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            ensure!(mmap.len() >= 8, "{name}: truncated safetensors");
            let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
            let data_start = 8 + header_len;
            ensure!(mmap.len() >= data_start, "{name}: truncated header");
            let header: serde_json::Value = serde_json::from_slice(&mmap[8..data_start])
                .with_context(|| format!("{name}: bad safetensors header"))?;
            let si = shards.len();
            shard_idx.insert(name.clone(), si);
            let obj = header
                .as_object()
                .context("safetensors header not an object")?;
            for (tname, info) in obj {
                if tname == "__metadata__" {
                    continue;
                }
                let offs = info["data_offsets"]
                    .as_array()
                    .with_context(|| format!("{tname}: no data_offsets"))?;
                let shape: Vec<usize> = info["shape"]
                    .as_array()
                    .with_context(|| format!("{tname}: no shape"))?
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as usize)
                    .collect();
                // dtype is validated implicitly through byte-size checks at
                // the read sites (packed = n*k/2 U8, scale = n*k/16 e4m3,
                // scalars = 4-byte f32).
                tensors.insert(
                    tname.clone(),
                    TensorRef {
                        shard: si,
                        start: offs[0].as_u64().unwrap_or(0) as usize,
                        end: offs[1].as_u64().unwrap_or(0) as usize,
                        shape,
                    },
                );
            }
            shards.push(Shard {
                mmap: Arc::new(mmap),
                data_start,
            });
        }
        Ok(Self { shards, tensors })
    }

    fn bytes(&self, name: &str) -> Result<(&[u8], &TensorRef)> {
        let t = self
            .tensors
            .get(name)
            .with_context(|| format!("tensor '{name}' not in snapshot"))?;
        let sh = &self.shards[t.shard];
        let lo = sh.data_start + t.start;
        let hi = sh.data_start + t.end;
        ensure!(hi <= sh.mmap.len(), "{name}: out of range");
        Ok((&sh.mmap[lo..hi], t))
    }

    fn scalar_f32(&self, name: &str) -> Result<f32> {
        let (b, t) = self.bytes(name)?;
        ensure!(
            b.len() == 4,
            "{name}: expected f32 scalar, {} bytes",
            b.len()
        );
        let _ = t;
        Ok(f32::from_le_bytes(b.try_into().unwrap()))
    }
}

// ── Per-tensor repack + histogram (single pass over the bytes) ─────────────

/// Remap+repack one NVFP4 tensor (`[n, k/2]` nibbles, `[n, k/16]` e4m3
/// scales) into `out` (`[n, k*3/8]`), accumulating the scale²-weighted code
/// histogram along the way.
fn repack_tensor(
    packed: &[u8],
    scales: &[u8],
    scale2: f32,
    n: usize,
    k: usize,
    cb: &Codebook,
    out: &mut [u8],
    hist: &mut CodeHist,
) {
    debug_assert_eq!(packed.len(), n * k / 2);
    debug_assert_eq!(scales.len(), n * k / GROUP_SIZE);
    debug_assert_eq!(out.len(), n * k * 3 / 8);
    let half_k = k / 2;
    let groups = k / GROUP_SIZE;
    let row3 = k * 3 / 8;
    for row in 0..n {
        let prow = &packed[row * half_k..(row + 1) * half_k];
        let srow = &scales[row * groups..(row + 1) * groups];
        repack_row(prow, &cb.map16, &mut out[row * row3..(row + 1) * row3]);
        for (g, &sb) in srow.iter().enumerate() {
            let s = e4m3_to_f32(sb) * scale2;
            let w = (s as f64) * (s as f64);
            for &byte in &prow[g * 8..(g + 1) * 8] {
                hist.mass[(byte & 0x0F) as usize] += w;
                hist.mass[(byte >> 4) as usize] += w;
            }
        }
    }
    hist.count += (n * k) as u64;
}

/// Histogram-only pass (codebook fit sampling).
fn accum_tensor_hist(packed: &[u8], scales: &[u8], scale2: f32, n: usize, k: usize) -> CodeHist {
    let mut h = CodeHist::default();
    h.accum_tensor(packed, scales, scale2, n, k);
    h
}

// ── Layer discovery + processing ───────────────────────────────────────────

struct MoeLayerInfo {
    layer: usize,
    num_experts: usize,
    hidden: usize,
    inter: usize,
}

fn discover_moe_layers(snap: &Snapshot) -> Result<Vec<MoeLayerInfo>> {
    let mut layers = Vec::new();
    for l in 0..1024usize {
        let probe = format!("model.layers.{l}.mlp.experts.0.gate_proj.weight_packed");
        let Some(t) = snap.tensors.get(&probe) else {
            continue;
        };
        ensure!(
            t.shape.len() == 2,
            "{probe}: expected 2-D, got {:?}",
            t.shape
        );
        let inter = t.shape[0];
        let hidden = t.shape[1] * 2;
        // Count experts.
        let mut e = 0usize;
        while snap.tensors.contains_key(&format!(
            "model.layers.{l}.mlp.experts.{e}.gate_proj.weight_packed"
        )) {
            e += 1;
        }
        // Down proj sanity: [hidden, inter/2].
        let dt = &snap.tensors[&format!("model.layers.{l}.mlp.experts.0.down_proj.weight_packed")];
        ensure!(
            dt.shape == [hidden, inter / 2],
            "layer {l}: down_proj shape {:?} != [{hidden}, {}]",
            dt.shape,
            inter / 2
        );
        layers.push(MoeLayerInfo {
            layer: l,
            num_experts: e,
            hidden,
            inter,
        });
    }
    ensure!(
        !layers.is_empty(),
        "no MoE expert tensors found in snapshot"
    );
    Ok(layers)
}

const PROJS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

fn expert_scale2(snap: &Snapshot, l: usize, e: usize, proj: &str) -> Result<f32> {
    let gs = snap.scalar_f32(&format!(
        "model.layers.{l}.mlp.experts.{e}.{proj}.weight_global_scale"
    ))?;
    ensure!(
        gs.is_finite() && gs.abs() >= f32::MIN_POSITIVE,
        "layer {l} expert {e} {proj}: degenerate weight_global_scale {gs}"
    );
    // compressed-tensors stores the reciprocal of the TRT-LLM/Atlas scale2
    // (see spark-model quantized_v2) — store the kernel-ready value.
    Ok(1.0 / gs)
}

fn fit_pass(
    snap: &Snapshot,
    layers: &[MoeLayerInfo],
    sample_stride: usize,
    verbose: bool,
) -> Result<Codebook> {
    let t0 = Instant::now();
    let mut jobs = Vec::new();
    for li in layers {
        for e in (0..li.num_experts).step_by(sample_stride.max(1)) {
            jobs.push((li.layer, e, li.hidden, li.inter));
        }
    }
    let hist = jobs
        .par_iter()
        .map(|&(l, e, hidden, inter)| -> Result<CodeHist> {
            let mut h = CodeHist::default();
            for proj in PROJS {
                let (n, k) = if proj == "down_proj" {
                    (hidden, inter)
                } else {
                    (inter, hidden)
                };
                let p = format!("model.layers.{l}.mlp.experts.{e}.{proj}");
                let (packed, pt) = snap.bytes(&format!("{p}.weight_packed"))?;
                ensure!(
                    pt.shape == [n, k / 2],
                    "{p}: bad packed shape {:?}",
                    pt.shape
                );
                let (scales, st) = snap.bytes(&format!("{p}.weight_scale"))?;
                ensure!(
                    st.shape == [n, k / GROUP_SIZE],
                    "{p}: bad scale shape {:?}",
                    st.shape
                );
                let s2 = expert_scale2(snap, l, e, proj)?;
                h.add(&accum_tensor_hist(packed, scales, s2, n, k));
            }
            Ok(h)
        })
        .try_reduce(CodeHist::default, |mut a, b| {
            a.add(&b);
            Ok(a)
        })?;

    let mag = hist.magnitude_mass();
    let cb = Codebook::fit(&mag);
    if verbose {
        let total: f64 = mag.iter().sum();
        println!(
            "codebook fit: {} sampled experts ({} values, {:.1}s)",
            jobs.len(),
            hist.count,
            t0.elapsed().as_secs_f64()
        );
        println!("  magnitude mass (E2M1 units, scale^2-weighted, normalized):");
        for (i, m) in mag.iter().enumerate() {
            println!(
                "    |{:>3.1}| : {:.4}",
                spark_storage::w3::E2M1_MAG[i],
                m / total
            );
        }
        println!("  LUT[8] = {:?}", cb.lut);
        println!("  map16  = {:?}", cb.map16);
    }
    Ok(cb)
}

#[derive(serde::Serialize)]
struct LayerStat {
    layer: usize,
    rmse: f64,
    ref_rms: f64,
    rel_rmse: f64,
    cosine: f64,
    secs: f64,
    bytes: u64,
    /// The codebook this layer was encoded with (== global LUT in global mode).
    lut: [f32; 8],
}

#[allow(clippy::too_many_arguments)]
fn encode_layer(
    snap: &Snapshot,
    li: &MoeLayerInfo,
    cb: &Codebook,
    out_dir: &Path,
) -> Result<LayerStat> {
    let t0 = Instant::now();
    let geom = W3LayerGeom {
        num_experts: li.num_experts,
        hidden: li.hidden,
        inter: li.inter,
    };
    let mut buf = vec![0u8; geom.file_bytes()];

    // Header.
    let header = W3LayerHeader {
        layer: li.layer as u32,
        num_experts: li.num_experts as u32,
        hidden: li.hidden as u32,
        inter: li.inter as u32,
        lut: cb.lut,
    };
    buf[..spark_storage::w3::W3_HEADER_BYTES].copy_from_slice(&header.to_bytes());

    // scale2 table (gate, up, down per expert).
    let mut scale2 = vec![0f32; li.num_experts * 3];
    for e in 0..li.num_experts {
        for (p, proj) in PROJS.iter().enumerate() {
            scale2[e * 3 + p] = expert_scale2(snap, li.layer, e, proj)?;
        }
    }
    {
        let off = geom.scale2_off();
        for (i, v) in scale2.iter().enumerate() {
            buf[off + i * 4..off + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
    }

    // Payload: parallel per-expert repack straight into the file buffer.
    let payload = &mut buf[geom.payload_off()..];
    let stride = geom.expert_stride();
    let offs = geom.expert_offsets();
    let hist = payload
        .par_chunks_mut(stride)
        .enumerate()
        .map(|(e, chunk)| -> Result<CodeHist> {
            let mut h = CodeHist::default();
            for (p, proj) in PROJS.iter().enumerate() {
                let (n, k) = if p == 2 {
                    (li.hidden, li.inter)
                } else {
                    (li.inter, li.hidden)
                };
                let prefix = format!("model.layers.{}.mlp.experts.{e}.{proj}", li.layer);
                let (packed, pt) = snap.bytes(&format!("{prefix}.weight_packed"))?;
                ensure!(
                    pt.shape == [n, k / 2],
                    "{prefix}: packed shape {:?}",
                    pt.shape
                );
                let (scales, st) = snap.bytes(&format!("{prefix}.weight_scale"))?;
                ensure!(
                    st.shape == [n, k / GROUP_SIZE],
                    "{prefix}: scale shape {:?}",
                    st.shape
                );
                let s2 = scale2[e * 3 + p];
                let (p_off, s_off) = (offs[p * 2], offs[p * 2 + 1]);
                let packed3_len = n * k * 3 / 8;
                // repack + histogram in one pass over the source bytes,
                // then copy the FP8 scale bytes verbatim (disjoint ranges).
                repack_tensor(
                    packed,
                    scales,
                    s2,
                    n,
                    k,
                    cb,
                    &mut chunk[p_off..p_off + packed3_len],
                    &mut h,
                );
                chunk[s_off..s_off + scales.len()].copy_from_slice(scales);
            }
            Ok(h)
        })
        .try_reduce(CodeHist::default, |mut a, b| {
            a.add(&b);
            Ok(a)
        })?;

    // Atomic write.
    std::fs::create_dir_all(out_dir)?;
    let final_path = out_dir.join(format!("layer_{:03}.w3x", li.layer));
    let tmp_path = out_dir.join(format!(".layer_{:03}.w3x.tmp", li.layer));
    {
        let mut f = std::fs::File::create(&tmp_path)?;
        f.write_all(&buf)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp_path, &final_path)?;

    let (rmse, cosine, ref_rms) = hist.quality(cb);
    Ok(LayerStat {
        layer: li.layer,
        rmse,
        ref_rms,
        rel_rmse: if ref_rms > 0.0 { rmse / ref_rms } else { 0.0 },
        cosine,
        secs: t0.elapsed().as_secs_f64(),
        bytes: buf.len() as u64,
        lut: cb.lut,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CodebookMode {
    PerLayer,
    Global,
}

fn main() -> Result<()> {
    let mut snapshot = String::new();
    let mut out = std::env::var("ATLAS_MOE_W3_DIR").unwrap_or_else(|_| String::from("./w3cache"));
    let mut layers_filter: Option<Vec<usize>> = None;
    let mut sample_stride = 16usize;
    let mut threads = 0usize;
    let mut emit_cosine = false;
    let mut codebook_mode = CodebookMode::PerLayer;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--snapshot" => {
                snapshot = args
                    .get(i + 1)
                    .cloned()
                    .context("--snapshot needs a value")?;
                i += 2;
            }
            "--codebook" => {
                let v = args
                    .get(i + 1)
                    .cloned()
                    .context("--codebook needs a value")?;
                codebook_mode = match v.as_str() {
                    "per-layer" => CodebookMode::PerLayer,
                    "global" => CodebookMode::Global,
                    other => bail!("--codebook must be 'per-layer' or 'global', got '{other}'"),
                };
                i += 2;
            }
            "--out" => {
                out = args.get(i + 1).cloned().context("--out needs a value")?;
                i += 2;
            }
            "--layers" => {
                let v = args.get(i + 1).cloned().context("--layers needs a value")?;
                layers_filter = Some(
                    v.split(',')
                        .map(|s| s.trim().parse::<usize>().context("bad --layers entry"))
                        .collect::<Result<Vec<_>>>()?,
                );
                i += 2;
            }
            "--sample-stride" => {
                sample_stride = args
                    .get(i + 1)
                    .context("--sample-stride needs a value")?
                    .parse()?;
                i += 2;
            }
            "--threads" => {
                threads = args
                    .get(i + 1)
                    .context("--threads needs a value")?
                    .parse()?;
                i += 2;
            }
            "--cosine" => {
                emit_cosine = true;
                i += 1;
            }
            other => bail!("unknown arg '{other}' (see bin docs)"),
        }
    }
    ensure!(
        !snapshot.is_empty(),
        "--snapshot <hf-snapshot-dir> is required"
    );
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }

    let total0 = Instant::now();
    let snap_dir = PathBuf::from(&snapshot);
    let snap = Snapshot::open(&snap_dir)?;
    let mut layers = discover_moe_layers(&snap)?;
    if let Some(filter) = &layers_filter {
        layers.retain(|l| filter.contains(&l.layer));
    }
    ensure!(!layers.is_empty(), "no MoE layers selected");
    println!(
        "w3-requant: {} MoE layers, {} experts, gate/up [{}x{}], down [{}x{}]",
        layers.len(),
        layers[0].num_experts,
        layers[0].inter,
        layers[0].hidden,
        layers[0].hidden,
        layers[0].inter,
    );

    // Pass A (global mode only): one codebook over a sampled subset of all
    // layers. In per-layer mode each layer fits its own codebook right before
    // it is encoded (the fit histogram pass re-reads that layer's bytes).
    let global_cb = match codebook_mode {
        CodebookMode::Global => Some(fit_pass(&snap, &layers, sample_stride, true)?),
        CodebookMode::PerLayer => {
            println!(
                "per-layer codebooks: fitting each layer on its own scale^2-weighted \
                 histogram (sample stride {sample_stride})"
            );
            None
        }
    };

    // Pass B: encode every layer.
    let out_dir = PathBuf::from(&out);
    let mut stats = Vec::new();
    let mut total_bytes = 0u64;
    for li in &layers {
        let cb = match &global_cb {
            Some(cb) => *cb,
            None => fit_pass(&snap, std::slice::from_ref(li), sample_stride, false)?,
        };
        let s = encode_layer(&snap, li, &cb, &out_dir)?;
        println!(
            "  layer {:>3}: rmse={:.6e} ref_rms={:.6e} rel={:.4} cos={:.6} ({:.1}s, {:.1} MiB) lut={:?}",
            s.layer,
            s.rmse,
            s.ref_rms,
            s.rel_rmse,
            s.cosine,
            s.secs,
            s.bytes as f64 / (1024.0 * 1024.0),
            &s.lut[..4],
        );
        total_bytes += s.bytes;
        stats.push(s);
    }

    // Aggregate summary.
    let n = stats.len() as f64;
    let mean_rel = stats.iter().map(|s| s.rel_rmse).sum::<f64>() / n;
    let mean_cos = stats.iter().map(|s| s.cosine).sum::<f64>() / n;
    let worst_cos = stats.iter().map(|s| s.cosine).fold(1.0f64, f64::min);
    let worst_rel = stats.iter().map(|s| s.rel_rmse).fold(0.0f64, f64::max);
    println!(
        "TOTAL: {} layers, {:.2} GiB on disk, {:.1}s wall | rel_rmse mean={:.4} worst={:.4} | cos mean={:.6} worst={:.6}",
        stats.len(),
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        total0.elapsed().as_secs_f64(),
        mean_rel,
        worst_rel,
        mean_cos,
        worst_cos,
    );
    if emit_cosine {
        println!("per-layer cosine summary:");
        for s in &stats {
            println!("  layer {:>3}: cosine={:.6}", s.layer, s.cosine);
        }
    }

    #[derive(serde::Serialize)]
    struct Summary<'a> {
        snapshot: &'a str,
        /// "per-layer" (LUT in each LayerStat + layer file header) or "global".
        codebook: &'a str,
        /// Global-mode codebook (None in per-layer mode; see layers[].lut).
        #[serde(skip_serializing_if = "Option::is_none")]
        lut: Option<[f32; 8]>,
        #[serde(skip_serializing_if = "Option::is_none")]
        map16: Option<[u8; 16]>,
        layers: &'a [LayerStat],
        total_bytes: u64,
        wall_secs: f64,
    }
    let summary = Summary {
        snapshot: &snapshot,
        codebook: match codebook_mode {
            CodebookMode::PerLayer => "per-layer",
            CodebookMode::Global => "global",
        },
        lut: global_cb.as_ref().map(|cb| cb.lut),
        map16: global_cb.as_ref().map(|cb| cb.map16),
        layers: &stats,
        total_bytes,
        wall_secs: total0.elapsed().as_secs_f64(),
    };
    std::fs::write(
        out_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(())
}
