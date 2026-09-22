// SPDX-License-Identifier: AGPL-3.0-only
//! Gate for the attention lane's seam implementation (`layers::deepseek_v41_attn::seam`)
//! against runF_faithful (checkpoint numerics, torch attention, 2 x 512 chunks).
//!
//! Teacher-forced at the seam: `attn_x` and `qr` (and for attention `q`, `ring`, `wpos`,
//! `attn_sink`) come from the capture; everything the lane owns is computed — compressor
//! (ckv/ik/pending), indexer (q/wts projections, score, candidates, top-k), the replay tail,
//! and the streaming-gather attention fed with the lane's OWN ckv and top-k.
//!
//! ```text
//! ATLAS_TARGET_MODEL=deepseek-v4.1 ATLAS_TARGET_QUANT=cb3 cargo build --release \
//!   -p spark-model --example dsv41_attn_seam --features cuda,gpu-examples
//! ```

use anyhow::{Context, Result, ensure};
use std::path::Path;
use std::sync::Arc;

use atlas_core::config::parse_config;
use spark_model::layers::deepseek_v41_attn::seam::{Dsv41SparseAttention, HEAD_DIM, REPLAY_ROWS};
use spark_model::weight_loader::deepseek_v41::ops::RopeSpec;
use spark_model::weight_loader::deepseek_v41::seams::{INDEX_TOPK, SparseShared};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{SafetensorsLoader, WeightLoader};

const MODEL_DIR: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K";
const REF: &str = "/home/flocka/atlas/DSV41_PORT/oracle/ref/runF_faithful";
const LAYERS: [usize; 4] = [2, 8, 14, 20];
const MAX_SEQ: usize = 8192;
const RING: usize = 4096;
const N_TOK: usize = 1024;
const CHUNK: usize = 512;

fn tap(layer: usize, name: &str, occ: usize) -> Result<Vec<u8>> {
    let p = format!("{REF}/L{layer:02}.{name}.{occ:03}.bin");
    std::fs::read(&p).with_context(|| p)
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}
fn i64s(b: &[u8]) -> Vec<i64> {
    b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
}
fn bf(u: u16) -> f64 {
    f32::from_bits((u as u32) << 16) as f64
}

struct Dev<'a> {
    gpu: &'a AtlasCudaBackend,
}

impl Dev<'_> {
    fn up(&self, b: &[u8]) -> Result<DevicePtr> {
        let p = self.gpu.alloc(b.len().max(8))?;
        self.gpu.copy_h2d(b, p)?;
        Ok(p)
    }
    fn down(&self, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
        self.gpu.synchronize(self.gpu.default_stream())?;
        let mut v = vec![0u8; bytes];
        self.gpu.copy_d2h(p, &mut v)?;
        Ok(v)
    }
}

/// Fraction of bf16 values bit-identical.
fn bits_equal(a: &[u16], b: &[u16]) -> f64 {
    a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len() as f64
}

fn rel_l2(a: &[u16], b: &[u16]) -> f64 {
    let (mut n, mut d) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (bf(*x), bf(*y));
        n += (x - y) * (x - y);
        d += y * y;
    }
    (n / d).sqrt()
}

/// Rows whose 512 top-k entries differ from the reference, and the worst set overlap among them.
fn topk_rows(got: &[i64], want: &[i64], t: usize) -> (usize, f64) {
    let mut bad = 0;
    let mut worst: f64 = 1.0;
    for r in 0..t {
        let g = &got[r * INDEX_TOPK..(r + 1) * INDEX_TOPK];
        let w = &want[r * INDEX_TOPK..(r + 1) * INDEX_TOPK];
        if g != w {
            bad += 1;
            let ws: std::collections::HashSet<_> = w.iter().filter(|&&v| v >= 0).collect();
            let hit = g.iter().filter(|&&v| v >= 0 && ws.contains(&v)).count();
            worst = worst.min(hit as f64 / ws.len().max(1) as f64);
        }
    }
    (bad, worst)
}

struct Inputs {
    x: Vec<u8>,  // [1024, 5120] bf16, both chunks
    qr: Vec<u8>, // [1024, 1280] bf16
}

fn inputs(layer: usize) -> Result<Inputs> {
    let mut x = tap(layer, "attn_x", 0)?;
    x.extend(tap(layer, "attn_x", 1)?);
    let mut qr = tap(layer, "qr", 0)?;
    qr.extend(tap(layer, "qr", 1)?);
    Ok(Inputs { x, qr })
}

/// Run compress + index over `splits` (chunk boundaries), all four layers per chunk in order,
/// as the encoder does. Returns the final SparseShared and the per-(chunk, layer) topk.
fn run_pass(
    dev: &Dev,
    att: &[Dsv41SparseAttention],
    ins: &[Inputs],
    splits: &[usize],
    control_drop_pending_at: Option<usize>,
) -> Result<(SparseShared, Vec<Vec<Vec<i64>>>)> {
    let gpu = dev.gpu;
    let stream = gpu.default_stream();
    let mut shared = SparseShared::default();
    let mut topks = vec![Vec::new(); att.len()];
    for w in splits.windows(2) {
        let (s, e) = (w[0], w[1]);
        let t = e - s;
        shared.begin_pass(s);
        for (i, a) in att.iter().enumerate() {
            if control_drop_pending_at == Some(s) {
                a.control_drop_pending();
            }
            let x = dev.up(&ins[i].x[s * 5120 * 2..e * 5120 * 2])?;
            let qr = dev.up(&ins[i].qr[s * 1280 * 2..e * 1280 * 2])?;
            a.compress_with(gpu, x, a.layer, s, t, &mut shared, stream)?;
            a.sparse_index_select_with(gpu, x, qr, a.layer, t, s, t, &mut shared, stream)?;
            topks[i].push(i64s(&dev.down(shared.topk.unwrap(), t * INDEX_TOPK * 8)?));
            gpu.free(x)?;
            gpu.free(qr)?;
        }
    }
    Ok((shared, topks))
}

fn main() -> Result<()> {
    let config = parse_config(&std::fs::read_to_string(format!("{MODEL_DIR}/config.json"))?)?;
    let gpu = Arc::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
    let dev = Dev { gpu: gpu.as_ref() };
    let stream = gpu.default_stream();

    let mut loader = SafetensorsLoader::new();
    loader.extra_skip = Some(Arc::new(|name: &str| {
        !LAYERS.iter().any(|l| name.starts_with(&format!("layers.{l}.attn.")))
    }));
    let store = loader.load(Path::new(MODEL_DIR), gpu.as_ref(), 0)?;
    println!("weights: {} tensors, {:.2} GB", store.len(), store.total_bytes() as f64 / 1e9);
    let spec = RopeSpec { dim: 64, original_seq_len: 65536, base: 160000.0, factor: 16.0, beta_fast: 32.0, beta_slow: 1.0 };
    let rope_c = spec.upload(gpu.as_ref(), MAX_SEQ + 8)?;
    let att: Vec<Dsv41SparseAttention> = LAYERS
        .iter()
        .map(|&l| Dsv41SparseAttention::load(gpu.as_ref(), &store, &config, l, MAX_SEQ, rope_c))
        .collect::<Result<_>>()?;
    let ins: Vec<Inputs> = LAYERS.iter().map(|&l| inputs(l)).collect::<Result<_>>()?;
    let mut fail = 0usize;

    // ---- 1. the capture's own chunking: ckv/ik, topk, candidates, attention, replay tail
    let (shared, topks) = run_pass(&dev, &att, &ins, &[0, CHUNK, N_TOK], None)?;
    for (i, a) in att.iter().enumerate() {
        let l = a.layer;
        for occ in 0..2 {
            let t = CHUNK;
            let want = i64s(&tap(l, "topk", occ)?);
            let (bad, worst) = topk_rows(&topks[i][occ], &want, t);
            println!("L{l:02} chunk {occ}: topk rows differ {bad}/{t} (worst set overlap {worst:.4})");
            fail += usize::from(bad * 50 > t); // > 2%: beyond the upstream-ulp expectation
        }
        let (ckv, ik) = a.caches().context("kv-source")?;
        let n_c = (N_TOK) / a.ratio;
        let got_ckv = u16s(&dev.down(ckv, n_c * HEAD_DIM * 2)?);
        let got_ik = u16s(&dev.down(ik, n_c * 128 * 2)?);
        let want_ckv = u16s(&tap(l, "ckv", 1)?);
        let want_ik = u16s(&tap(l, "ik", 1)?);
        let (ec, ei) = (bits_equal(&got_ckv, &want_ckv), bits_equal(&got_ik, &want_ik));
        println!(
            "L{l:02} r={}: ckv bit-exact {ec:.4} (rel {:.2e}), ik bit-exact {ei:.4} (rel {:.2e})",
            a.ratio,
            rel_l2(&got_ckv, &want_ckv),
            rel_l2(&got_ik, &want_ik)
        );
        fail += usize::from(ec < 0.97 || ei < 0.97);
    }
    // candidates (L20, chunk 1 is the one left in `shared`)
    {
        let n_pad = shared.candidates_ld;
        let got = dev.down(shared.candidates.unwrap(), CHUNK * n_pad)?;
        let want = tap(20, "cand_out", 1)?;
        let diff = got.iter().zip(&want).filter(|(g, w)| (**g != 0) != (**w != 0)).count();
        println!("L20 chunk 1: candidates differ {diff}/{} (ld {n_pad})", want.len());
        fail += usize::from(diff != 0 && want.len() == got.len());
        // replay tail = last 128 rows of the prompt
        let rt = i64s(&dev.down(shared.replay_topk.unwrap(), REPLAY_ROWS * INDEX_TOPK * 8)?);
        let want_rows = &i64s(&tap(20, "topk", 1)?)[(CHUNK - REPLAY_ROWS) * INDEX_TOPK..];
        let mine_rows = &topks[3][1][(CHUNK - REPLAY_ROWS) * INDEX_TOPK..];
        println!(
            "L20 replay tail: {} rows; equals my own chunk-1 rows 384..511: {}; vs capture: {} rows differ",
            shared.replay_rows,
            rt == mine_rows,
            topk_rows(&rt, want_rows, REPLAY_ROWS).0
        );
        fail += usize::from(shared.replay_rows != REPLAY_ROWS || rt != mine_rows);
    }

    // attention with the lane's OWN ckv + topk, chunk 1, layers 2 and 20
    for (i, a) in att.iter().enumerate().filter(|(_, a)| a.layer == 2 || a.layer == 20) {
        let l = a.layer;
        let q = dev.up(&tap(l, "q", 1)?)?;
        let ring = dev.up(&tap(l, "ring", 1)?)?;
        let wpos = dev.up(&tap(l, "wpos", 1)?)?;
        let sink = dev.up(&tap(l, "attn_sink", 1)?)?;
        let cidx = dev.up(&topks[i][1].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        let out = gpu.alloc(CHUNK * 64 * HEAD_DIM * 2)?;
        let (ckv, _) = a.caches().unwrap();
        let n_c = N_TOK / a.ratio;
        let scale = (HEAD_DIM as f32).powf(-0.5);
        a.sparse_attention_with(gpu.as_ref(), q, ring, RING, wpos, 0, Some(ckv), n_c, Some(cidx), sink, scale, CHUNK, out, stream)?;
        let got = u16s(&dev.down(out, CHUNK * 64 * HEAD_DIM * 2)?);
        let want = u16s(&tap(l, "attn_o_pre_inverse_rope", 1)?);
        let (ex, rl) = (bits_equal(&got, &want), rel_l2(&got, &want));
        // CONTROL: the "ckv cleared per chunk" bug — chunk 0's rows read as zero
        let ck0 = gpu.alloc(n_c * HEAD_DIM * 2)?;
        gpu.copy_d2d(ckv, ck0, n_c * HEAD_DIM * 2)?;
        gpu.memset(ck0, 0, (CHUNK / a.ratio) * HEAD_DIM * 2)?;
        a.sparse_attention_with(gpu.as_ref(), q, ring, RING, wpos, 0, Some(ck0), n_c, Some(cidx), sink, scale, CHUNK, out, stream)?;
        let ctl = u16s(&dev.down(out, CHUNK * 64 * HEAD_DIM * 2)?);
        let (cex, crl) = (bits_equal(&ctl, &want), rel_l2(&ctl, &want));
        println!(
            "L{l:02} chunk 1 attention (own ckv + own topk) vs fp32 reference: bf16 bit-exact {ex:.4}, rel {rl:.3e} \
             | CTRL cleared ckv: {cex:.4}, rel {crl:.3e}"
        );
        fail += usize::from(ex < 0.98 || cex > 0.5);
    }

    // ---- 2. odd chunk boundaries: exercises the ratio-2 `pending` row and a replay tail
    //      spanning two chunks. Must reproduce the SAME caches/selection as the capture.
    let splits = [0, 511, 1000, 1023, N_TOK];
    let (shared2, topks2) = run_pass(&dev, &att, &ins, &splits, None)?;
    let (ckv2, _) = att[0].caches().unwrap();
    let got2 = u16s(&dev.down(ckv2, (N_TOK / 2) * HEAD_DIM * 2)?);
    let want2 = u16s(&tap(2, "ckv", 1)?);
    let e2 = bits_equal(&got2, &want2);
    let rt2 = i64s(&dev.down(shared2.replay_topk.unwrap(), REPLAY_ROWS * INDEX_TOPK * 8)?);
    let want_tail = &i64s(&tap(20, "topk", 1)?)[(CHUNK - REPLAY_ROWS) * INDEX_TOPK..];
    let tail_bad = topk_rows(&rt2, want_tail, REPLAY_ROWS).0;
    let l20_last: Vec<i64> = topks2[3].concat();
    let tail_self = rt2[..] == l20_last[(N_TOK - REPLAY_ROWS) * INDEX_TOPK..];
    println!(
        "odd splits {splits:?}: L02 ckv bit-exact {e2:.4}; replay tail {} rows, spans chunks, == own last 128 rows: {tail_self}, vs capture {tail_bad} rows differ",
        shared2.replay_rows
    );
    fail += usize::from(e2 < 0.99 || !tail_self);
    // CHUNK INVARIANCE, directly: the same 1024 rows under two chunkings must select the same.
    for (i, a) in att.iter().enumerate() {
        let std_rows: Vec<i64> = topks[i].concat();
        let odd_rows: Vec<i64> = topks2[i].concat();
        let (bad, worst) = topk_rows(&odd_rows, &std_rows, N_TOK);
        println!("L{:02} chunk invariance (odd vs 2x512 splits): {bad}/{N_TOK} rows differ (worst overlap {worst:.4})", a.layer);
        fail += usize::from(bad != 0);
    }
    // CONTROL: drop the pending row at the 511 boundary -> the pair (510, 511) and everything
    // after it on L2 is built from the wrong positions.
    match run_pass(&dev, &att, &ins, &splits, Some(511)) {
        Err(e) => println!("CTRL dropped pending at 511: REFUSED loudly ({e:#}) -- the row count no longer adds up"),
        Ok(_) => {
            let got3 = u16s(&dev.down(ckv2, (N_TOK / 2) * HEAD_DIM * 2)?);
            let e3 = bits_equal(&got3, &want2);
            println!("CTRL dropped pending at 511: L02 ckv bit-exact {e3:.4} (must fall)");
            fail += usize::from(e3 > e2 - 0.01);
        }
    }

    ensure!(fail == 0, "SEAM GATE FAIL ({fail} checks)");
    println!("SEAM GATE PASS");
    Ok(())
}
