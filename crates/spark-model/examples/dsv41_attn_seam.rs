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
    let core = Dsv41SparseCore::load(&(gpu.clone() as spark_model::weight_loader::deepseek_v41::device_allocs::SharedGpu), &store, &config, MAX_SEQ, MAX_CHUNK, freqs_c)?;

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
        // Back to position 512 (a rollback, as DSpark would), then chunk 1 again over the
        // corrupted cache: compress() rewrites rows 512.., rows 0..511 stay zero = the bug.
        core.rollback(gpu.as_ref(), CHUNK, ops.stream)?;
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

    // ---- 3. the decoder REPLAY over the last 128 prompt rows (runG: L21; runH: L21..39).
    //      Layers 21..23 inherit L20's tail; 24/28/32/36 re-index INSIDE the tail's candidate
    //      pool; the rest inherit the latest selection. Every replay layer's attention and every
    //      replay indexer's top-k is compared.
    // runF also taps L21, but over whole chunks (replay OFF): only a capture whose L21 ran in
    // the replay (win_lo > 0) is a replay reference.
    let replay_capture = cap.tap(21, "win_lo", 0).map(|b| i64s(&b)[0] > 0).unwrap_or(false);
    if replay_capture {
        run_encoder(&dev, &ops, &core, &ins, &[0, CHUNK, N_TOK], None)?; // restore the tail
        let start = N_TOK - REPLAY_ROWS;
        let wpos_d = dev.up(bytemuck_i32(&window_positions(start, REPLAY_ROWS)))?;
        let out_d = gpu.alloc(REPLAY_ROWS * QROW)?;
        core.begin_pass(PassKind::Replay, start, REPLAY_ROWS)?;
        let mut worst_att: f64 = 1.0;
        // A layer's selection comes from the latest index-source layer at or below it. If that
        // layer was not captured (runG has 21, 30, 39 but not 24/28/36), the chain is broken and
        // comparing later layers would test THIS HARNESS, not the core — stop there.
        let mut chain_ok = true;
        for l in 21..40 {
            let captured = cap.tap(l, "q", 0).is_ok();
            if [24, 28, 32, 36].contains(&l) && !captured {
                chain_ok = false;
            }
            if !captured {
                continue;
            }
            if !chain_ok {
                println!("REPLAY L{l}: NOT COMPARED -- its selection comes from an index layer this capture lacks");
                continue;
            }
            let q = cap.tap(l, "q", 0)?;
            let win_lo = i64s(&cap.tap(l, "win_lo", 0)?)[0] as usize;
            ensure!(win_lo == start, "L{l} win_lo {win_lo}, expected {start}");
            let (x, qr) = if [24, 28, 32, 36].contains(&l) {
                (dev.up(&cap.tap(l, "attn_x", 0)?)?, dev.up(&cap.tap(l, "qr", 0)?)?)
            } else {
                (DevicePtr::NULL, DevicePtr::NULL)
            };
            let a = CoreArgs {
                layer: l,
                x,
                qr,
                q: dev.up(&q)?,
                ring: dev.up(&cap.tap(l, "ring", 0)?)?,
                wpos: wpos_d,
                win_lo,
                sink: dev.up(&cap.tap(l, "attn_sink", 0)?)?,
                t: REPLAY_ROWS,
                start,
                out: out_d,
            };
            core.run(&ops, &a)?;
            let tk = i64s(&dev.down(core.current_topk().unwrap(), REPLAY_ROWS * INDEX_TOPK * 8)?);
            let (bad, worst) = topk_rows(&tk, &i64s(&cap.tap(l, "topk", 0)?), REPLAY_ROWS);
            let got = u16s(&dev.down(out_d, REPLAY_ROWS * QROW)?);
            let want = u16s(&cap.tap(l, "attn_o_pre_inverse_rope", 0)?);
            let (e, rl) = (bits_equal(&got, &want), rel_l2(&got, &want));
            worst_att = worst_att.min(e);
            println!("REPLAY L{l}: topk {bad}/{REPLAY_ROWS} rows differ (worst overlap {worst:.4}); attention bit-exact {e:.4} rel {rl:.2e}");
            fail += usize::from(bad > 3 || e < 0.98);
            if l == 21 {
                // CONTROL: the replay WITHOUT the window floor attends to ring rows it never wrote.
                let a0 = CoreArgs { win_lo: 0, ..a };
                core.begin_pass(PassKind::Replay, start, REPLAY_ROWS)?;
                core.run(&ops, &a0)?;
                let c = bits_equal(&u16s(&dev.down(out_d, REPLAY_ROWS * QROW)?), &want);
                println!("CTRL replay L21 with win_lo=0: attention bit-exact {c:.4} (must collapse)");
                fail += usize::from(c > 0.9);
                core.begin_pass(PassKind::Replay, start, REPLAY_ROWS)?;
                core.run(&ops, &a)?;
            }
        }
        println!("REPLAY worst attention bit-exact over captured layers: {worst_att:.4}");
    }

    // ---- 5. DECODE: the prompt's last rows re-run as T=1 Decode passes after an encoder
    //      prefill of the rest. Decode must reproduce the prefill of the same row: compared
    //      BIT-FOR-BIT against this core's own 2x512 prefill, and against the capture.
    //      Pre-registered band: top-k identical to the prefill rows (0 extra differing rows),
    //      attention bit-identical to the core's prefill; vs the capture the prefill band.
    {
        const D0: usize = 1000; // decode positions D0..N_TOK (24 steps, both parities for r=2)
        run_encoder(&dev, &ops, &core, &ins, &[0, CHUNK, D0], None)?;
        let dec = |wpos_shift: usize| -> Result<PassOut> {
            let mut r = PassOut { topk: vec![Vec::new(); ins.len()], out: vec![Vec::new(); ins.len()] };
            let wpos_d = gpu.alloc(128 * 4)?;
            let out_d = gpu.alloc(QROW)?;
            core.rollback(gpu.as_ref(), D0, ops.stream)?; // each arm resumes at D0
            for p in D0..N_TOK {
                gpu.copy_h2d(bytemuck_i32(&window_positions(p - wpos_shift, 1)), wpos_d)?;
                core.begin_pass(PassKind::Decode, p, 1)?;
                for (i, li) in ins.iter().enumerate() {
                    let a = CoreArgs {
                        layer: li.layer,
                        x: li.x.offset(p * 5120 * 2),
                        qr: li.qr.offset(p * 1280 * 2),
                        q: li.q.offset(p * QROW),
                        ring: li.ring,
                        wpos: wpos_d,
                        win_lo: 0,
                        sink: li.sink,
                        t: 1,
                        start: p,
                        out: out_d,
                    };
                    core.run(&ops, &a)?;
                    r.topk[i].extend(i64s(&dev.down(core.current_topk().unwrap(), INDEX_TOPK * 8)?));
                    r.out[i].extend(u16s(&dev.down(out_d, QROW)?));
                }
            }
            gpu.free(wpos_d)?;
            gpu.free(out_d)?;
            Ok(r)
        };
        let d = dec(0)?;
        let n = N_TOK - D0;
        for (i, li) in ins.iter().enumerate() {
            let l = li.layer;
            let own_tk = &std_run.topk[i][D0 * INDEX_TOPK..];
            let own_o = &std_run.out[i][D0 * QROW / 2..];
            let (self_bad, _) = topk_rows(&d.topk[i], own_tk, n);
            let self_o = bits_equal(&d.out[i], own_o);
            let cap_tk = &i64s(&cap.tap(l, "topk", 1)?)[(D0 - CHUNK) * INDEX_TOPK..];
            let cap_o = &u16s(&cap.tap(l, "attn_o_pre_inverse_rope", 1)?)[(D0 - CHUNK) * QROW / 2..];
            let (cap_bad, worst) = topk_rows(&d.topk[i], cap_tk, n);
            let (ce, crl) = (bits_equal(&d.out[i], cap_o), rel_l2(&d.out[i], cap_o));
            println!(
                "DECODE L{l:02} ({n} steps, T=1): vs own prefill topk {self_bad} rows differ, attention bit-identical {self_o:.6} \
                 | vs capture topk {cap_bad}/{n} (worst {worst:.4}), attention bit-exact {ce:.4} rel {crl:.2e}"
            );
            // Decode uses the SPLIT-KV kernel (only the within-row accumulation order differs
            // from prefill's one-pass): >= 99.9% bit-identical to own prefill, rest 1-ulp.
            // PRE-REGISTERED before the prefill moved to the tensor-core kernel: decode (split,
            // fp32 P) vs own prefill (mma, P_hi+P_lo) >= 99.5% bf16-identical -- two kernels each
            // measured within 0.1% of the one-pass fp32 kernel.
            fail += usize::from(self_bad != 0 || self_o < 0.995 || cap_bad > 1 || ce < 0.98);
        }
        // CONTROL: window positions off by one (the query's own row dropped, p-128 added).
        let c = dec(1)?;
        for (i, li) in ins.iter().enumerate() {
            let e = bits_equal(&c.out[i], &d.out[i]);
            println!("CTRL DECODE L{:02} wpos shifted by one: attention bit-identical to the real decode {e:.4} (must collapse)", li.layer);
            fail += usize::from(e > 0.5);
        }
    }

    // ---- 6. DSPARK ROLLBACK. Reference A: prefill [0,1000), then 5 T=1 decode steps
    //      (1000..1004). Spec B: prefill, ONE T=6 pass at 1000 whose rows 1003..1005 are
    //      WRONG drafts (inputs of rows 900..902), rollback(1003) -- odd, so layer 2/8/14 must
    //      restore row 1002 as pending from the pass's saved inputs -- then 1003, 1004 as T=1.
    //      PRE-REGISTERED: B's accepted rows 1000..1002 and resumed rows 1003..1004 BYTE-
    //      IDENTICAL to A (top-k and attention), and the final ckv/ik identical. CONTROL: the same
    //      with the pending restore skipped must differ.
    {
        const R0: usize = 1000;
        let run_rows_kind = |start: usize, src: &[usize], kind: PassKind| -> Result<PassOut> {
            let t = src.len();
            let mut r = PassOut { topk: vec![Vec::new(); ins.len()], out: vec![Vec::new(); ins.len()] };
            let wpos_d = dev.up(bytemuck_i32(&window_positions(start, t)))?;
            let out_d = gpu.alloc(t * QROW)?;
            let (xb, qrb, qb) = (gpu.alloc(t * 5120 * 2)?, gpu.alloc(t * 1280 * 2)?, gpu.alloc(t * QROW)?);
            core.begin_pass(kind, start, t)?;
            for (i, li) in ins.iter().enumerate() {
                for (j, &sr) in src.iter().enumerate() {
                    gpu.copy_d2d(li.x.offset(sr * 5120 * 2), xb.offset(j * 5120 * 2), 5120 * 2)?;
                    gpu.copy_d2d(li.qr.offset(sr * 1280 * 2), qrb.offset(j * 1280 * 2), 1280 * 2)?;
                    gpu.copy_d2d(li.q.offset(sr * QROW), qb.offset(j * QROW), QROW)?;
                }
                let a = CoreArgs { layer: li.layer, x: xb, qr: qrb, q: qb, ring: li.ring, wpos: wpos_d, win_lo: 0, sink: li.sink, t, start, out: out_d };
                core.run(&ops, &a)?;
                r.topk[i] = i64s(&dev.down(core.current_topk().unwrap(), t * INDEX_TOPK * 8)?);
                r.out[i] = u16s(&dev.down(out_d, t * QROW)?);
            }
            for b in [wpos_d, out_d, xb, qrb, qb] {
                gpu.free(b)?;
            }
            Ok(r)
        };
        let run_rows = |start: usize, src: &[usize]| run_rows_kind(start, src, PassKind::Decode);
        // Compare only the rows BOTH runs legitimately own: those below the final length. Rows at
        // or past it hold whatever a rejected draft (or an earlier arm) wrote and are rewritten
        // before any query can see them. (A first version compared a fixed 1005 rows and "failed"
        // the odd-start case on exactly those stale rows -- the harness, not the core.)
        let caches = |core: &Dsv41SparseCore, len: usize| -> Result<Vec<Vec<u8>>> {
            let mut v = Vec::new();
            for &l in &LAYERS {
                let (ckv, ik) = core.caches(l).unwrap();
                let rows = len / if l < 20 { 2 } else { 1 };
                v.push(dev.down(ckv, rows * HEAD_DIM * 2)?);
                v.push(dev.down(ik, rows * 128 * 2)?);
            }
            Ok(v)
        };
        // A
        run_encoder(&dev, &ops, &core, &ins, &[0, CHUNK, R0], None)?;
        let mut a_steps = Vec::new();
        for p in R0..R0 + 5 {
            a_steps.push(run_rows(p, &[p])?);
        }
        let a_cache = caches(&core, 1005)?;
        // B (and the control)
        let mut verdicts = Vec::new();
        // resume as two T=1 passes, or as ONE T=2 pass: without the restore the T=1 resume trips
        // the row-count check, but the T=2 resume pairs (1003,1004) and publishes the right
        // COUNT of rows -- silently wrong values. Both must fail; both real arms must be exact.
        for (restore, joined) in [(true, false), (true, true), (false, false), (false, true)] {
            let arm = || -> Result<usize> {
            run_encoder(&dev, &ops, &core, &ins, &[0, CHUNK, R0], None)?;
            let v = run_rows_kind(R0, &[1000, 1001, 1002, 900, 901, 902], PassKind::Verify)?;
            if restore {
                PassHook::rollback(&core, &ops, 1003)?; // the forward's trait path
            } else {
                core.control_rollback_without_restore(gpu.as_ref(), 1003, ops.stream)?;
            }
            let (r3, r4) = if joined {
                let r = run_rows(1003, &[1003, 1004])?;
                let split = |k: usize| PassOut {
                    topk: r.topk.iter().map(|v| v[k * INDEX_TOPK..(k + 1) * INDEX_TOPK].to_vec()).collect(),
                    out: r.out.iter().map(|v| v[k * QROW / 2..(k + 1) * QROW / 2].to_vec()).collect(),
                };
                (split(0), split(1))
            } else {
                (run_rows(1003, &[1003])?, run_rows(1004, &[1004])?)
            };
            let mut diff = 0usize;
            for i in 0..ins.len() {
                for j in 0..3 {
                    diff += usize::from(v.topk[i][j * INDEX_TOPK..(j + 1) * INDEX_TOPK] != a_steps[j].topk[i][..]);
                    diff += v.out[i][j * QROW / 2..(j + 1) * QROW / 2].iter().zip(&a_steps[j].out[i]).filter(|(x, y)| x != y).count();
                }
                for (r, step) in [(&r3, 3usize), (&r4, 4)] {
                    diff += usize::from(r.topk[i] != a_steps[step].topk[i]);
                    diff += r.out[i].iter().zip(&a_steps[step].out[i]).filter(|(x, y)| x != y).count();
                }
            }
            let c_diff: usize = caches(&core, 1005)?.iter().zip(&a_cache).map(|(x, y)| x.iter().zip(y).filter(|(a, b)| a != b).count()).sum();
            println!(
                "ROLLBACK {} (resume {}): Verify T=6 via PassHook::rollback (3 accepted + 3 wrong drafts), rollback(1003), resume 1003..1004 -> {diff} differing top-k rows/attention values vs never-drafted decode; {c_diff} differing ckv/ik bytes",
                if restore { "real           " } else { "CTRL no restore" },
                if joined { "one T=2 pass" } else { "two T=1 passes" }
            );
            Ok(diff + c_diff)
            };
            match arm() {
                Ok(d) => verdicts.push(d),
                Err(e) if !restore => {
                    println!("ROLLBACK CTRL no restore (resume {}): REFUSED loudly ({e:#})", if joined { "one T=2 pass" } else { "two T=1 passes" });
                    verdicts.push(usize::MAX);
                }
                Err(e) => return Err(e),
            }
        }
        fail += usize::from(verdicts[0] != 0 || verdicts[1] != 0 || verdicts[2] == 0 || verdicts[3] == 0);

        // The other restore branch: a verify pass starting at an ODD position (1001, so row 1000
        // is pending BEFORE it) whose drafts are ALL rejected: rollback(1001) must restore the
        // pending row the pass started from, not one of its own inputs.
        run_encoder(&dev, &ops, &core, &ins, &[0, CHUNK, R0], None)?;
        let a2: Vec<PassOut> = (R0..R0 + 3).map(|p| run_rows(p, &[p])).collect::<Result<_>>()?;
        let a2_cache = caches(&core, R0 + 3)?;
        for restore in [true, false] {
            let arm = || -> Result<usize> {
                run_encoder(&dev, &ops, &core, &ins, &[0, CHUNK, R0], None)?;
                run_rows(R0, &[R0])?;
                run_rows_kind(R0 + 1, &[900, 901, 902, 903, 904, 905], PassKind::Verify)?; // all six drafts wrong
                if restore {
                    PassHook::rollback(&core, &ops, R0 + 1)?;
                } else {
                    core.control_rollback_without_restore(gpu.as_ref(), R0 + 1, ops.stream)?;
                }
                let r = run_rows(R0 + 1, &[R0 + 1, R0 + 2])?;
                let mut diff = 0usize;
                for i in 0..ins.len() {
                    for (k, step) in [(0usize, 1usize), (1, 2)] {
                        diff += usize::from(r.topk[i][k * INDEX_TOPK..(k + 1) * INDEX_TOPK] != a2[step].topk[i][..]);
                        diff += r.out[i][k * QROW / 2..(k + 1) * QROW / 2].iter().zip(&a2[step].out[i]).filter(|(x, y)| x != y).count();
                    }
                }
                let c_diff: usize = caches(&core, R0 + 3)?.iter().zip(&a2_cache).map(|(x, y)| x.iter().zip(y).filter(|(a, b)| a != b).count()).sum();
                Ok(diff + c_diff)
            };
            match arm() {
                Ok(d) => {
                    println!("ROLLBACK {} (verify at ODD 1001, 0 of 6 accepted, restore = the pass's own starting pending row): {d} differing values vs never-drafted decode",
                             if restore { "real           " } else { "CTRL no restore" });
                    fail += usize::from((d == 0) != restore);
                }
                Err(e) if !restore => println!("ROLLBACK CTRL no restore (verify at odd 1001): REFUSED loudly ({e:#})"),
                Err(e) => return Err(e),
            }
        }
    }

    // ---- 4. the reset rule: Decode continues the sequence; only a new prompt resets.
    run_encoder(&dev, &ops, &core, &ins, &[0, CHUNK, N_TOK], None)?;
    core.begin_pass(PassKind::Decode, N_TOK, 1)?;
    let kept = core.replay_tail().3;
    core.begin_pass(PassKind::EncoderChunk, 0, 1)?; // CONTROL: a new prompt MUST reset
    let reset = core.replay_tail().3;
    println!("RESET RULE: after Decode(pos {N_TOK}) the tail still holds {kept} rows; after a new prompt {reset}");
    fail += usize::from(kept != REPLAY_ROWS || reset != 0);

    ensure!(fail == 0, "CORE GATE FAIL ({fail} checks)");
    println!("CORE GATE PASS");
    Ok(())
}
