// SPDX-License-Identifier: AGPL-3.0-only
//! Gate for the attention lane's core (`layers::deepseek_v41_attn::core::Dsv41SparseCore`,
//! the `AttnCore` + `PassHook` the forward calls) against a faithful oracle capture.
//!
//! Teacher-forced at the core boundary: `CoreArgs` (x, qr, q, ring, wpos, sink) come from the
//! capture; everything the core owns is computed — compressor (ckv/ik/pending), indexer
//! (q/wts projections, score, candidates, top-k), L20's replay tail, sparse attention on the
//! core's OWN ckv and top-k.
//!
//! ```text
//! ATLAS_TARGET_MODEL=deepseek-v4.1 ATLAS_TARGET_QUANT=cb3 cargo build --release \
//!   -p spark-model --example dsv41_attn_seam --features cuda,gpu-examples
//! dsv41_attn_seam [runF_faithful | runG_replay]
//! ```
//! runF_faithful: 2 x 512 encoder chunks (+ odd re-chunking + controls).
//! runG_replay:   the same, then the decoder REPLAY of layer 21 over the last 128 rows.

use anyhow::{Context, Result, ensure};
use std::path::Path;
use std::sync::Arc;

use atlas_core::config::parse_config;
use spark_model::layers::deepseek_v41_attn::core::{Dsv41SparseCore, HEAD_DIM, REPLAY_ROWS};
use spark_model::layers::deepseek_v41_attn::index::INDEX_TOPK;
use spark_model::weight_loader::deepseek_v41::attn_block::{AttnCore, CoreArgs, window_positions};
use spark_model::weight_loader::deepseek_v41::forward::{PassHook, PassKind};
use spark_model::weight_loader::deepseek_v41::ops::{Dsv41Kernels, Ops, RopeSpec, bytemuck_i32};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{SafetensorsLoader, WeightLoader};

const MODEL_DIR: &str = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K";
const REF_ROOT: &str = "/home/flocka/atlas/DSV41_PORT/oracle/ref";
const LAYERS: [usize; 4] = [2, 8, 14, 20];
const MAX_SEQ: usize = 8192;
const MAX_CHUNK: usize = 512;
const N_TOK: usize = 1024;
const CHUNK: usize = 512;
const QROW: usize = 64 * HEAD_DIM * 2;

struct Cap {
    dir: String,
}

impl Cap {
    fn tap(&self, layer: usize, name: &str, occ: usize) -> Result<Vec<u8>> {
        let p = format!("{}/L{layer:02}.{name}.{occ:03}.bin", self.dir);
        std::fs::read(&p).with_context(|| p)
    }
    /// Both encoder chunks of a per-token tap, concatenated.
    fn both(&self, layer: usize, name: &str) -> Result<Vec<u8>> {
        let mut v = self.tap(layer, name, 0)?;
        v.extend(self.tap(layer, name, 1)?);
        Ok(v)
    }
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
/// Rows whose 512 entries differ, and the worst set overlap among them.
fn topk_rows(got: &[i64], want: &[i64], t: usize) -> (usize, f64) {
    let (mut bad, mut worst) = (0, 1.0f64);
    for r in 0..t {
        let (g, w) = (&got[r * INDEX_TOPK..(r + 1) * INDEX_TOPK], &want[r * INDEX_TOPK..(r + 1) * INDEX_TOPK]);
        if g != w {
            bad += 1;
            let ws: std::collections::HashSet<_> = w.iter().filter(|&&v| v >= 0).collect();
            let hit = g.iter().filter(|&&v| v >= 0 && ws.contains(&v)).count();
            worst = worst.min(hit as f64 / ws.len().max(1) as f64);
        }
    }
    (bad, worst)
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

/// Per-layer device copies of the whole prompt's core inputs.
struct LayerIn {
    layer: usize,
    x: DevicePtr,
    qr: DevicePtr,
    q: DevicePtr,
    ring: DevicePtr,
    sink: DevicePtr,
}

struct PassOut {
    /// per layer: all rows' topk and attention output, concatenated over chunks
    topk: Vec<Vec<i64>>,
    out: Vec<Vec<u16>>,
}

fn run_encoder(dev: &Dev, ops: &Ops, core: &Dsv41SparseCore, ins: &[LayerIn], splits: &[usize], drop_pending_at: Option<usize>) -> Result<PassOut> {
    let gpu = dev.gpu;
    let mut res = PassOut { topk: vec![Vec::new(); ins.len()], out: vec![Vec::new(); ins.len()] };
    let wpos_d = gpu.alloc(MAX_CHUNK * 128 * 4)?;
    let out_d = gpu.alloc(MAX_CHUNK * QROW)?;
    for w in splits.windows(2) {
        let (s, e) = (w[0], w[1]);
        let t = e - s;
        gpu.copy_h2d(bytemuck_i32(&window_positions(s, t)), wpos_d)?;
        core.begin_pass(PassKind::EncoderChunk, s, t)?;
        for (i, li) in ins.iter().enumerate() {
            if drop_pending_at == Some(s) {
                core.control_drop_pending(li.layer);
            }
            let a = CoreArgs {
                layer: li.layer,
                x: li.x.offset(s * 5120 * 2),
                qr: li.qr.offset(s * 1280 * 2),
                q: li.q.offset(s * QROW),
                ring: li.ring,
                wpos: wpos_d,
                win_lo: 0,
                sink: li.sink,
                t,
                start: s,
                out: out_d,
            };
            core.run(ops, &a)?;
            res.topk[i].extend(i64s(&dev.down(core.current_topk().unwrap(), t * INDEX_TOPK * 8)?));
            res.out[i].extend(u16s(&dev.down(out_d, t * QROW)?));
        }
    }
    gpu.free(wpos_d)?;
    gpu.free(out_d)?;
    Ok(res)
}

fn main() -> Result<()> {
    let run = std::env::args().nth(1).unwrap_or_else(|| "runF_faithful".into());
    let cap = Cap { dir: format!("{REF_ROOT}/{run}") };
    let config = parse_config(&std::fs::read_to_string(format!("{MODEL_DIR}/config.json"))?)?;
    let gpu = Arc::new(AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?);
    let dev = Dev { gpu: gpu.as_ref() };
    let kernels = Dsv41Kernels::load(gpu.as_ref())?;
    let ops = Ops { gpu: gpu.as_ref(), k: &kernels, stream: gpu.default_stream() };

    let mut loader = SafetensorsLoader::new();
    loader.extra_skip = Some(Arc::new(|name: &str| {
        let keep = LAYERS.iter().any(|l| name.starts_with(&format!("layers.{l}.attn.")))
            || [24, 28, 32, 36].iter().any(|l| name.starts_with(&format!("layers.{l}.attn.indexer.")));
        !keep
    }));
    let store = loader.load(Path::new(MODEL_DIR), gpu.as_ref(), 0)?;
    println!("run {run}; weights: {} tensors, {:.2} GB", store.len(), store.total_bytes() as f64 / 1e9);
    let spec = RopeSpec { dim: 64, original_seq_len: 65536, base: 160000.0, factor: 16.0, beta_fast: 32.0, beta_slow: 1.0 };
    let freqs_c = spec.upload(gpu.as_ref(), MAX_SEQ + 8)?;
    let core = Dsv41SparseCore::load(gpu.as_ref(), &store, &config, MAX_SEQ, MAX_CHUNK, freqs_c)?;

    let ins: Vec<LayerIn> = LAYERS
        .iter()
        .map(|&l| {
            Ok(LayerIn {
                layer: l,
                x: dev.up(&cap.both(l, "attn_x")?)?,
                qr: dev.up(&cap.both(l, "qr")?)?,
                q: dev.up(&cap.both(l, "q")?)?,
                // the ring after the LAST chunk holds every position 0..1023 (< RING, no wrap);
                // a query only reads positions <= its own, so it serves any chunking.
                ring: dev.up(&cap.tap(l, "ring", 1)?)?,
                sink: dev.up(&cap.tap(l, "attn_sink", 0)?)?,
            })
        })
        .collect::<Result<_>>()?;
    let mut fail = 0usize;

    // ---- 1. the capture's own chunking
    let std_run = run_encoder(&dev, &ops, &core, &ins, &[0, CHUNK, N_TOK], None)?;
    for (i, li) in ins.iter().enumerate() {
        let l = li.layer;
        for occ in 0..2 {
            let want = i64s(&cap.tap(l, "topk", occ)?);
            let (bad, worst) = topk_rows(&std_run.topk[i][occ * CHUNK * INDEX_TOPK..(occ + 1) * CHUNK * INDEX_TOPK], &want, CHUNK);
            let want_o = u16s(&cap.tap(l, "attn_o_pre_inverse_rope", occ)?);
            let got_o = &std_run.out[i][occ * CHUNK * QROW / 2..(occ + 1) * CHUNK * QROW / 2];
            println!(
                "L{l:02} chunk {occ}: topk rows differ {bad}/{CHUNK} (worst overlap {worst:.4}) | attention bf16 bit-exact {:.4} rel {:.2e}",
                bits_equal(got_o, &want_o),
                rel_l2(got_o, &want_o)
            );
            fail += usize::from(bad * 50 > CHUNK || bits_equal(got_o, &want_o) < 0.98);
        }
        let (ckv, ik) = core.caches(l).context("kv-source")?;
        let n_c = N_TOK / if l < 20 { 2 } else { 1 };
        let (gc, gi) = (u16s(&dev.down(ckv, n_c * HEAD_DIM * 2)?), u16s(&dev.down(ik, n_c * 128 * 2)?));
        let (wc, wi) = (u16s(&cap.tap(l, "ckv", 1)?), u16s(&cap.tap(l, "ik", 1)?));
        let (ec, ei) = (bits_equal(&gc, &wc), bits_equal(&gi, &wi));
        println!("L{l:02}: ckv bit-exact {ec:.4} (rel {:.2e}), ik bit-exact {ei:.4} (rel {:.2e})", rel_l2(&gc, &wc), rel_l2(&gi, &wi));
        fail += usize::from(ec < 0.97 || ei < 0.97);
    }
    {
        let (cand, ld) = core.current_candidates().context("L20 candidates")?;
        let got = dev.down(cand, CHUNK * ld)?;
        let want = cap.tap(20, "cand_out", 1)?;
        let diff = got.iter().zip(&want).filter(|(g, w)| (**g != 0) != (**w != 0)).count();
        println!("L20 chunk 1: candidates differ {diff}/{} (ld {ld})", want.len());
        fail += usize::from(diff != 0 || got.len() != want.len());
        let (rt, _, _, rows) = core.replay_tail();
        let rt = i64s(&dev.down(rt, REPLAY_ROWS * INDEX_TOPK * 8)?);
        let own = &std_run.topk[3][(N_TOK - REPLAY_ROWS) * INDEX_TOPK..];
        println!("L20 replay tail: {rows} rows, == own last 128 rows: {}", rt == own);
        fail += usize::from(rows != REPLAY_ROWS || rt != own);
    }
    // CONTROL: attention over a cache whose chunk-0 rows were cleared (the "ckv cleared per
    // chunk" bug). Re-run chunk 1 of L20's attention by hand with a zeroed prefix.
    {
        let (ckv, _) = core.caches(20).unwrap();
        gpu.memset(ckv, 0, CHUNK * HEAD_DIM * 2)?;
        let wpos_d = dev.up(bytemuck_i32(&window_positions(CHUNK, CHUNK)))?;
        let out_d = gpu.alloc(CHUNK * QROW)?;
        // Re-publish state for chunk 1 exactly as the encoder would, without recompressing.
        core.begin_pass(PassKind::FullChunk, CHUNK, CHUNK)?;
        let li = &ins[3];
        let a = CoreArgs { layer: 20, x: li.x.offset(CHUNK * 5120 * 2), qr: li.qr.offset(CHUNK * 1280 * 2), q: li.q.offset(CHUNK * QROW), ring: li.ring, wpos: wpos_d, win_lo: 0, sink: li.sink, t: CHUNK, start: CHUNK, out: out_d };
        // compress() rewrites rows 512..1023 only; rows 0..511 stay zero = the bug.
        core.run(&ops, &a)?;
        let got = u16s(&dev.down(out_d, CHUNK * QROW)?);
        let want = u16s(&cap.tap(20, "attn_o_pre_inverse_rope", 1)?);
        let e = bits_equal(&got, &want);
        println!("CTRL L20 chunk 1 with chunk-0 ckv rows cleared: attention bit-exact {e:.4} (must collapse)");
        fail += usize::from(e > 0.5);
    }

    // ---- 2. odd re-chunking: the ratio-2 pending row, a tail spanning chunks, chunk invariance
    let splits = [0, 511, 1000, 1023, N_TOK];
    let odd = run_encoder(&dev, &ops, &core, &ins, &splits, None)?;
    for (i, li) in ins.iter().enumerate() {
        let (bad, worst) = topk_rows(&odd.topk[i], &std_run.topk[i], N_TOK);
        let ob = bits_equal(&odd.out[i], &std_run.out[i]);
        println!("L{:02} chunk invariance {splits:?} vs 2x512: topk {bad}/{N_TOK} rows differ (worst {worst:.4}), attention bit-identical {ob:.6}", li.layer);
        fail += usize::from(bad != 0 || ob < 1.0);
    }
    let (rt, _, _, rows) = core.replay_tail();
    let rt = i64s(&dev.down(rt, REPLAY_ROWS * INDEX_TOPK * 8)?);
    let own = &odd.topk[3][(N_TOK - REPLAY_ROWS) * INDEX_TOPK..];
    println!("odd splits: replay tail {rows} rows spanning 3 chunks, == own last 128 rows: {}", rt == own);
    fail += usize::from(rows != REPLAY_ROWS || rt != own);
    match run_encoder(&dev, &ops, &core, &ins, &splits, Some(511)) {
        Err(e) => println!("CTRL dropped pending at 511: REFUSED loudly ({e:#})"),
        Ok(r) => {
            let e = bits_equal(&r.out[0], &std_run.out[0]);
            println!("CTRL dropped pending at 511: L02 attention bit-identical {e:.4} (must fall)");
            fail += usize::from(e > 0.999);
        }
    }

    // ---- 3. runG only: the decoder replay of layer 21 over the last 128 prompt rows
    if run.starts_with("runG") {
        run_encoder(&dev, &ops, &core, &ins, &[0, CHUNK, N_TOK], None)?; // restore the tail
        let start = N_TOK - REPLAY_ROWS;
        let q = dev.up(&cap.tap(21, "q", 0)?)?;
        let ring = dev.up(&cap.tap(21, "ring", 0)?)?;
        let sink = dev.up(&cap.tap(21, "attn_sink", 0)?)?;
        let wpos_d = dev.up(bytemuck_i32(&window_positions(start, REPLAY_ROWS)))?;
        let out_d = gpu.alloc(REPLAY_ROWS * QROW)?;
        let win_lo = i64s(&cap.tap(21, "win_lo", 0)?)[0] as usize;
        ensure!(win_lo == start, "runG L21 win_lo {win_lo}, expected {start}");
        core.begin_pass(PassKind::Replay, start, REPLAY_ROWS)?;
        let a = CoreArgs { layer: 21, x: DevicePtr::NULL, qr: DevicePtr::NULL, q, ring, wpos: wpos_d, win_lo, sink, t: REPLAY_ROWS, start, out: out_d };
        core.run(&ops, &a)?;
        let tk = i64s(&dev.down(core.current_topk().unwrap(), REPLAY_ROWS * INDEX_TOPK * 8)?);
        let (bad, _) = topk_rows(&tk, &i64s(&cap.tap(21, "topk", 0)?), REPLAY_ROWS);
        let got = u16s(&dev.down(out_d, REPLAY_ROWS * QROW)?);
        let want = u16s(&cap.tap(21, "attn_o_pre_inverse_rope", 0)?);
        let (e, rl) = (bits_equal(&got, &want), rel_l2(&got, &want));
        // CONTROL: the replay WITHOUT the window floor attends to ring rows the replay never wrote.
        let a0 = CoreArgs { win_lo: 0, ..a };
        core.begin_pass(PassKind::Replay, start, REPLAY_ROWS)?;
        core.run(&ops, &a0)?;
        let c = bits_equal(&u16s(&dev.down(out_d, REPLAY_ROWS * QROW)?), &want);
        println!("REPLAY L21 (inherits L20's tail): topk {bad}/{REPLAY_ROWS} rows differ; attention bit-exact {e:.4} rel {rl:.2e} | CTRL win_lo=0: {c:.4}");
        fail += usize::from(bad > 2 || e < 0.98 || c > 0.9);
    }

    ensure!(fail == 0, "CORE GATE FAIL ({fail} checks)");
    println!("CORE GATE PASS");
    Ok(())
}
